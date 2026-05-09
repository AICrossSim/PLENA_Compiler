"""Lift a raw (pre-pipeline) PrimFunc directly to a :class:`Graph`.

This is the eventual replacement for the chain
``annotate_group → annotate_sync → split_lane_groups → fuse_elementwise
→ scope_inference → allocate_group_memory → lower_fp_row_patterns →
lift_to_blocks → lift_to_graph``.

Why
---
All of those passes are stmt rewriters that communicate via stmt-level
attributes (``T.attr(0, plena.group, ...)`` etc) and structural mutation
(splitting fors, rewriting buffer shapes). Each one re-walks the IR.
Migrating each rewriter into the graph layer removes the stmt-walker
overhead and lets passes communicate via :class:`graph_ir` attrs
(``node.attrs[ATTR_*]`` keys, BufferNode.physical_scope, etc).

Status
------
Phase A: this module is **forward-looking infrastructure** — it exists,
is unit-tested, but is NOT yet wired into ``compile_func``. The current
pipeline still uses the stmt-walker chain + the older
``lift_to_graph`` (which lifts from a post-stmt-walker IR).

Phase B-D will:
  * write graph-layer pass equivalents for each stmt-walker pass;
  * verify each one byte-identical against the stmt-walker chain;
  * cut the pipeline over to ``lift_from_raw_primfunc`` + the new graph
    passes once parity is confirmed.

What this lift produces
-----------------------
A :class:`Graph` whose root is a chain of :class:`ForRoot` nodes (the
grid bindings — bx / by / etc) wrapping either a :class:`LaneGroup` (if
any grid axis was lane-fusion-eligible — TODO, today not detected here;
the graph_passes/annotate_group_pass will set the LaneGroup membership
later) or a :class:`NodeRoot` (everything else).

Each :class:`tir.Call` inside the kernel body becomes a
:class:`GraphNode` with reads/writes derived from the call's region
arguments (or, for already-lowered ``tir.call_extern`` calls, an empty
set — no region info available). Each user ``T.serial`` /
``T.Parallel`` for-loop becomes a :class:`NestedForGroup` whose body is
recursively lifted. ``BufferStore`` and other "non-op" stmts become
:class:`RawStmt` items.

What this lift does NOT do (yet)
--------------------------------
  * Identify lane-fusion grid axes (= future
    ``graph_passes/annotate_group_pass``).
  * Set ``ATTR_IS_SYNC`` / ``ATTR_GEMM_KIND`` on graph nodes (= future
    graph passes).
  * Resolve buffer scopes / fuse elementwise patterns / lower fp row
    patterns / split lane groups / allocate lane memory (= future graph
    passes).

After this lift runs, the Graph is "raw" — it just mirrors the source
TIR structure with each op pulled into a GraphNode. Subsequent graph
passes do the real work.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import tvm
from tvm import tir

from . import graph_ir
from .graph_ir import (
    Graph, GraphNode, NestedForGroup, LaneGroup, NodeRoot, ForRoot, RootItem,
    RawStmt, BufferNode, BufferAccess, ForNode,
    ATTR_GEMM_KIND,
)


# Stmt-level attr key the user writes via
# ``with T.attr(0, KIND_KEY, "btmm"): T.gemm(...)`` to mark a gemm site
# as BTMM. Used by lift to absorb the AttrStmt into ``ATTR_GEMM_KIND``
# on the resulting GraphNode.
KIND_KEY = "plena.gemm_kind"


_TILEOP_COPY = "tl.tileop.copy"
_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REGION = "tl.tileop.region"
_TILEOP_REDUCE = "tl.tileop.reduce"


class LiftFromRawError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Region → BufferAccess conversion (same logic as lift_to_blocks; kept
# local so this module doesn't import lift_to_blocks).
# ---------------------------------------------------------------------------

def _region_to_buffer_access(call: tir.Call) -> Optional[BufferAccess]:
    """``tl.tileop.region(BufferLoad, mode, ext_0, ext_1, ...)`` → BufferAccess.

    Pads with extent-1 ranges on the leading axes when the user gave
    fewer extents than the buffer's rank (matches the convention in
    ``lift_to_blocks``)."""
    if not isinstance(call, tir.Call):
        return None
    if call.op.name != _TILEOP_REGION:
        return None
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        return None
    starts = list(load.indices)
    extents = list(call.args[2:])
    if len(starts) != len(extents):
        diff = len(starts) - len(extents)
        if diff > 0:
            extents = [tir.IntImm("int32", 1)] * diff + extents
        else:
            return None
    return BufferAccess(
        buffer_name=load.buffer.name,
        starts=starts,
        extents=extents,
    )


def _full_buffer_access(buf: tir.Buffer) -> BufferAccess:
    """Cover the entire buffer (used for already-lowered plena.* externs
    where region info isn't directly recoverable)."""
    return BufferAccess(
        buffer_name=buf.name,
        starts=[tir.IntImm("int32", 0) for _ in buf.shape],
        extents=list(buf.shape),
    )


# ---------------------------------------------------------------------------
# Op-call → GraphNode (with reads/writes derived from the call's args)
# ---------------------------------------------------------------------------

def _reads_writes_from_call(call: tir.Call):
    """Best-effort reads/writes extraction:
      * tl.tileop.copy(src, dst)        → reads=[src], writes=[dst]
      * tl.tileop.gemm_py(A, B, C, ...) → reads=[A, B, C], writes=[C]
        (C is read-modify-write because gemm accumulates into it.)
      * tl.tileop.reduce(src, dst, ...) → reads=[src, dst], writes=[dst]
      * other tir.call_extern           → empty (region info not available)
    Returned reads/writes are :class:`BufferAccess` instances.
    """
    op_name = call.op.name
    if op_name == _TILEOP_COPY:
        src = _region_to_buffer_access(call.args[0])
        dst = _region_to_buffer_access(call.args[1])
        return ([src] if src else []), ([dst] if dst else [])
    if op_name == _TILEOP_GEMM:
        a = _region_to_buffer_access(call.args[0])
        b = _region_to_buffer_access(call.args[1])
        c = _region_to_buffer_access(call.args[2])
        reads = [r for r in (a, b, c) if r is not None]
        return reads, ([c] if c else [])
    if op_name == _TILEOP_REDUCE:
        # reduce(src_region, dst_region, dim, clear)
        src = _region_to_buffer_access(call.args[0]) if len(call.args) >= 1 else None
        dst = _region_to_buffer_access(call.args[1]) if len(call.args) >= 2 else None
        reads = [r for r in (src, dst) if r is not None]
        return reads, ([dst] if dst else [])
    return [], []


# ---------------------------------------------------------------------------
# Name generation
# ---------------------------------------------------------------------------

class _NameGen:
    def __init__(self):
        self._counts: Dict[str, int] = {}

    def fresh(self, prefix: str) -> str:
        n = self._counts.get(prefix, 0)
        self._counts[prefix] = n + 1
        return f"{prefix}_{n}"

    def name_for(self, call: tir.Call) -> str:
        op_name = call.op.name
        if op_name == _TILEOP_COPY:
            return self.fresh("copy")
        if op_name == _TILEOP_GEMM:
            return self.fresh("gemm")
        if op_name == _TILEOP_REDUCE:
            return self.fresh("reduce")
        if op_name == "tir.call_extern" and call.args:
            head = call.args[0]
            if isinstance(head, tir.StringImm):
                short = head.value.replace("plena.", "").replace(".", "_")
                return self.fresh(short)
        return self.fresh("op")


# ---------------------------------------------------------------------------
# Buffer collection (BufferNode for every alloc'd / param buffer)
# ---------------------------------------------------------------------------

def _collect_buffers(func: tir.PrimFunc) -> Dict[str, BufferNode]:
    """Walk every Block.alloc_buffers and func.buffer_map; return a
    name → BufferNode dict.

    Sets ``declared_scope`` from the buffer's tilelang scope (or
    ``"global"`` for params). ``physical_scope`` left None — graph-layer
    scope_inference fills it later.
    """
    out: Dict[str, BufferNode] = {}

    def make_node(buf: tir.Buffer, scope: str) -> BufferNode:
        return BufferNode(
            name=buf.name,
            shape=list(buf.shape),
            dtype=str(buf.dtype),
            declared_scope=scope,
            physical_scope=None,
            data_var=buf.data,
        )

    # Function parameters → HBM (scope is "global" on the tir.Buffer
    # because tilelang doesn't tag params with a tilelang scope).
    for buf in func.buffer_map.values():
        if buf.name not in out:
            out[buf.name] = make_node(buf, "global")

    # Alloc'd buffers (under any tir.Block in the body).
    def visit(s):
        if isinstance(s, tir.BlockRealize):
            for buf in s.block.alloc_buffers:
                if buf.name not in out:
                    declared = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
                    out[buf.name] = make_node(buf, declared)
            visit(s.block.body)
            if s.block.init is not None:
                visit(s.block.init)
            return
        if isinstance(s, tir.SeqStmt):
            for c in s.seq:
                visit(c)
            return
        if isinstance(s, (tir.AttrStmt, tir.For, tir.LetStmt)):
            visit(s.body)
            return
        if isinstance(s, tir.IfThenElse):
            visit(s.then_case)
            if s.else_case is not None:
                visit(s.else_case)
            return

    visit(func.body)
    return out


# ---------------------------------------------------------------------------
# Body lift — produce a flat list of items from a stmt subtree
# ---------------------------------------------------------------------------

def _items_from_stmt(stmt: tir.Stmt,
                     namegen: _NameGen,
                     pending_attrs: Optional[Dict[str, Any]] = None
                     ) -> List[Union[GraphNode, NestedForGroup, RawStmt]]:
    """Recursively lift a stmt subtree into a flat list of graph items.

    ``pending_attrs`` accumulates any plena.* AttrStmt wrappers we've
    walked past (e.g. ``T.attr(0, plena.gemm_kind, "btmm")``). When we
    finally hit the wrapped Evaluate we attach those attrs to the
    resulting GraphNode.
    """
    if pending_attrs is None:
        pending_attrs = {}

    if isinstance(stmt, tir.SeqStmt):
        out: List = []
        for c in stmt.seq:
            out.extend(_items_from_stmt(c, namegen, pending_attrs))
            # pending_attrs is consumed by whatever stmt picks them up;
            # we conservatively reset to empty here so an attr on stmt 0
            # doesn't leak to stmt 1.
            pending_attrs = {}
        return out

    if isinstance(stmt, tir.AttrStmt):
        if stmt.attr_key == KIND_KEY:
            new_pending = dict(pending_attrs)
            v = stmt.value
            kind = v.value if isinstance(v, tir.StringImm) else str(v)
            new_pending[ATTR_GEMM_KIND] = kind
            return _items_from_stmt(stmt.body, namegen, new_pending)
        # Other AttrStmts (thread_extent for grid bindings, etc) — not
        # graph-relevant at this level; skip the wrapper. (Grid bindings
        # are handled in _lift_root.)
        return _items_from_stmt(stmt.body, namegen, pending_attrs)

    if isinstance(stmt, tir.Evaluate):
        if not isinstance(stmt.value, tir.Call):
            return [RawStmt(name=namegen.fresh("raw_eval"), stmt=stmt)]
        call = stmt.value
        reads, writes = _reads_writes_from_call(call)
        return [GraphNode(
            name=namegen.name_for(call),
            op_call=call,
            attrs=dict(pending_attrs),
            reads=reads,
            writes=writes,
        )]

    if isinstance(stmt, tir.For):
        body_items = _items_from_stmt(stmt.body, namegen, {})
        return [NestedForGroup(
            loop_var=stmt.loop_var,
            min=stmt.min,
            extent=stmt.extent,
            kind=stmt.kind,
            thread_binding=stmt.thread_binding,
            annotations=dict(stmt.annotations) if stmt.annotations else None,
            items=body_items,
        )]

    if isinstance(stmt, tir.BlockRealize):
        # Inner blocks beyond the top-level tilelang_root: descend,
        # pulling the inner items out (graph IR has no general "Block
        # node" — we flatten).
        return _items_from_stmt(stmt.block.body, namegen, pending_attrs)

    if isinstance(stmt, tir.IfThenElse):
        # No graph IR for IfThenElse yet — wrap as raw.
        return [RawStmt(name=namegen.fresh("raw_if"), stmt=stmt)]

    if isinstance(stmt, tir.LetStmt):
        # Lifted by the inline_let_stmts pass before any of this; if
        # one slips through, wrap raw.
        return [RawStmt(name=namegen.fresh("raw_let"), stmt=stmt)]

    if isinstance(stmt, tir.BufferStore):
        return [RawStmt(name=namegen.fresh("raw_store"), stmt=stmt)]

    raise LiftFromRawError(
        f"unsupported stmt of type {type(stmt).__name__} during raw lift"
    )


# ---------------------------------------------------------------------------
# Root lift — peel grid bindings, find tilelang_root, lift body
# ---------------------------------------------------------------------------

def _lift_root(stmt: tir.Stmt,
               namegen: _NameGen,
               outer_allocs: Optional[List[tir.Buffer]] = None) -> RootItem:
    """Lift the top-level structure: skip the synthesised root block,
    peel grid bindings (``T.launch_thread`` AttrStmts), find
    tilelang_root, lift its body.

    ``outer_allocs`` accumulates ``alloc_buffers`` from outer
    ``BlockRealize``s (e.g. the synthesised ``with T.block("root"):``
    that wraps a top-level For). They get merged into the leaf
    NodeRoot/LaneGroup's alloc_buffers so materialize sees them too —
    same trick as ``lift_to_graph._build_root``.
    """
    if outer_allocs is None:
        outer_allocs = []

    if isinstance(stmt, tir.AttrStmt) and stmt.attr_key == "thread_extent":
        node = stmt.node
        ext = stmt.value
        is_thread = (isinstance(node, tir.IterVar)
                     and node.thread_tag is not None
                     and node.thread_tag.startswith("threadIdx"))
        is_block_extent_1 = (isinstance(node, tir.IterVar)
                             and node.thread_tag is not None
                             and node.thread_tag.startswith("blockIdx")
                             and isinstance(ext, tir.IntImm)
                             and int(ext.value) == 1)
        if is_thread or is_block_extent_1:
            return _lift_root(stmt.body, namegen, outer_allocs)
        inner = _lift_root(stmt.body, namegen, outer_allocs)
        loop_var = node.var if isinstance(node, tir.IterVar) else None
        if loop_var is None:
            return inner
        return ForRoot(
            loop_var=loop_var,
            min=tir.IntImm(loop_var.dtype, 0),
            extent=ext,
            kind=tir.ForKind.SERIAL,
            thread_binding=None,
            annotations=None,
            body=inner,
        )

    if isinstance(stmt, tir.AttrStmt):
        return _lift_root(stmt.body, namegen, outer_allocs)

    if isinstance(stmt, tir.BlockRealize):
        if stmt.block.name_hint == "tilelang_root":
            items = _items_from_stmt(stmt.block.body, namegen, {})
            return NodeRoot(
                items=items,
                alloc_buffers=list(outer_allocs) + list(stmt.block.alloc_buffers),
            )
        # Outer "root" block etc — accumulate its alloc_buffers and recurse.
        new_outer = list(outer_allocs) + list(stmt.block.alloc_buffers)
        return _lift_root(stmt.block.body, namegen, new_outer)

    if isinstance(stmt, tir.SeqStmt):
        items: List = []
        for c in stmt.seq:
            items.extend(_items_from_stmt(c, namegen, {}))
        return NodeRoot(items=items, alloc_buffers=list(outer_allocs))

    if isinstance(stmt, tir.For):
        inner = _lift_root(stmt.body, namegen, outer_allocs)
        return ForRoot(
            loop_var=stmt.loop_var,
            min=stmt.min, extent=stmt.extent,
            kind=stmt.kind, thread_binding=stmt.thread_binding,
            annotations=dict(stmt.annotations) if stmt.annotations else None,
            body=inner,
        )

    if isinstance(stmt, tir.Evaluate):
        items = _items_from_stmt(stmt, namegen, {})
        return NodeRoot(items=items, alloc_buffers=list(outer_allocs))

    raise LiftFromRawError(
        f"unsupported top-level stmt of type {type(stmt).__name__} "
        f"during raw lift"
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def lift_from_raw_primfunc(func: tir.PrimFunc) -> Graph:
    """Lift a raw (pre-pipeline) ``tir.PrimFunc`` into a :class:`Graph`.

    The returned Graph mirrors the source structure: each tile-DSL op
    is a GraphNode; user for-loops become NestedForGroups; grid-binding
    AttrStmts wrap the result in ForRoot chains.

    Subsequent graph passes (graph_passes/annotate_*, fuse_elementwise,
    scope_inference, allocate_group_memory, lower_fp_row_patterns,
    split_lane_groups) refine this base graph. None of those passes
    exist yet — this function is forward-looking infrastructure.
    """
    namegen = _NameGen()
    root = _lift_root(func.body, namegen)
    buffer_nodes = _collect_buffers(func)
    return Graph(
        root=root,
        params=list(func.params),
        buffer_map=dict(func.buffer_map),
        ret_type=func.ret_type,
        attrs=func.attrs,
        buffer_nodes=buffer_nodes,
    )


__all__ = ["lift_from_raw_primfunc", "LiftFromRawError"]
