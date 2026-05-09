"""Graph-IR back end: :class:`Graph` → final TIR PrimFunc.

This module owns the materialization step of the graph pipeline.
It consumes a :class:`graph_ir.Graph` (the output of any sequence of
graph passes) and produces a TIR PrimFunc with plena.* extern stmts
and lane-fusion segmentation applied — the form ``PlenaCodegen`` consumes.

Concerns
--------
 * Sync vs. per-lane partitioning ("the curtain horizontal-bundle
   algorithm" — see PIPELINE_ARCHITECTURE.md).
 * Per-op lowering (delegates to ``lower_to_hlir._lower_copy /
   _lower_gemm`` for the actual plena.* extern emission).
 * Wrapping per-lane runs in ``for(lane_var, range(lane_count))`` with
   the right ForKind (UNROLLED if the run contains plena.matmul, else
   SERIAL).
 * Recursive handling of :class:`NestedForGroup` (e.g. ``for kv_block``)
   inside lane groups: the partition happens INSIDE the for-loop too.

Operations on graph nodes consult ``node.attrs[ATTR_IS_SYNC]`` and
``node.attrs[ATTR_GEMM_KIND]`` instead of probing the original
plena.sync / plena.gemm_kind AttrStmts. By this point those AttrStmts
have been absorbed into graph attrs by ``lift_to_graph``.
"""

from __future__ import annotations

from typing import List, Optional, Union

import tvm
from tvm import tir

from .graph_passes.scope_inference import BufferScopeMap
from .lower_to_hlir import _lower_copy, _lower_gemm
from .graph_ir import (
    Graph, GraphNode, LaneGroup, NestedForGroup, NodeRoot, ForRoot, RootItem,
    RawStmt, ATTR_GEMM_KIND, ATTR_IS_SYNC,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TILEOP_COPY = "tl.tileop.copy"
_TILEOP_GEMM = "tl.tileop.gemm_py"

# Already-lowered plena.* extern calls that span all lanes in one HW
# instruction. Consulted by ``lift_to_graph`` to set
# ``ATTR_IS_SYNC = True`` on already-fused ops that don't carry an
# explicit ``plena.sync`` annotation.
INHERENTLY_SYNC_EXTERNS = frozenset({
    "plena.zero_v",
    "plena.v_add", "plena.v_sub", "plena.v_mul",
    "plena.dma_h2v_slice", "plena.dma_h2m_slice", "plena.dma_v2h_slice",
    "plena.btmm", "plena.btmv",
    "plena.copy_v_to_v",
    "plena.row_load_v_to_fp", "plena.row_store_fp_to_v",
})

# Already-lowered plena.* externs that, when emitted inside a per-lane
# run, signal "use UNROLLED for-by" instead of SERIAL.
PER_LANE_UNROLLED_EXTERNS = frozenset({
    "plena.matmul",
})


class GraphPipelineError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Sync / per-lane classification (pure attr lookup — no stmt probing)
# ---------------------------------------------------------------------------

def _is_sync(node: GraphNode) -> bool:
    return bool(node.attrs.get(ATTR_IS_SYNC, False))


def _is_per_lane_unrolled(node: GraphNode) -> bool:
    """A per-lane node that should drive the surrounding for-by to be
    UNROLLED rather than SERIAL.

    Two forms apply:
      * already-lowered ``plena.matmul`` (in PER_LANE_UNROLLED_EXTERNS);
      * tile-DSL ``tl.tileop.gemm_py`` (kind != "btmm"; btmm is sync
        and never reaches a per-lane run). Such a gemm will lower to
        ``plena.matmul`` or ``plena.mv``.
    """
    if node.op_call.op.name == "tir.call_extern":
        name_arg = node.op_call.args[0]
        if isinstance(name_arg, tir.StringImm):
            return name_arg.value in PER_LANE_UNROLLED_EXTERNS
    if node.op_call.op.name == _TILEOP_GEMM:
        kind = node.attrs.get(ATTR_GEMM_KIND, "overwrite")
        return kind != "btmm"
    return False


def _has_any_sync(items) -> bool:
    """Recursively: does this item-tree contain any sync node?"""
    for item in items:
        if isinstance(item, GraphNode):
            if _is_sync(item):
                return True
        elif isinstance(item, NestedForGroup):
            if _has_any_sync(item.items):
                return True
        # RawStmt is never sync — it's per-lane opaque work.
    return False


def _items_contain_unrolled_matmul(items) -> bool:
    for item in items:
        if isinstance(item, GraphNode) and _is_per_lane_unrolled(item):
            return True
        if isinstance(item, NestedForGroup) and _items_contain_unrolled_matmul(item.items):
            return True
    return False


# ---------------------------------------------------------------------------
# Op-level lowering (delegates to lower_to_hlir helpers)
# ---------------------------------------------------------------------------

def _lower_node(node: GraphNode,
                lane_var: Optional[tir.Var],
                in_sync: bool,
                scopes: BufferScopeMap,
                lane_count: int,
                target_mlen: int,
                target_hlen: int,
                target_layout: str) -> tir.Stmt:
    """Lower a single GraphNode to a stmt."""
    op_name = node.op_call.op.name
    lane_var_name = lane_var.name if lane_var is not None else None
    if op_name == _TILEOP_COPY:
        return _lower_copy(
            node.op_call, scopes,
            lane_count=lane_count,
            lane_var=lane_var_name,
            in_sync=in_sync,
            target_mlen=target_mlen,
            target_hlen=target_hlen,
            target_layout=target_layout,
        )
    if op_name == _TILEOP_GEMM:
        kind = node.attrs.get(ATTR_GEMM_KIND, "overwrite")
        return _lower_gemm(
            node.op_call, scopes,
            kind=kind,
            lane_count=lane_count,
            target_mlen=target_mlen,
            target_hlen=target_hlen,
            lane_var=lane_var_name,
        )
    if op_name == "tir.call_extern":
        # Already lowered upstream (e.g. by fuse_elementwise → plena.zero_v).
        return tir.Evaluate(node.op_call)
    # Unknown / not-yet-supported tile op (e.g. tl.tileop.reduce). Emit
    # verbatim — graph_pipeline doesn't lower it, but materialization
    # stays valid; the backend handles it (or fails later, which is the
    # same behaviour as before this pass).
    return tir.Evaluate(node.op_call)


# ---------------------------------------------------------------------------
# Per-lane materialization
# ---------------------------------------------------------------------------

def _materialize_per_lane_seq(items,
                              lane_var: tir.Var,
                              lane_count: int,
                              scopes: BufferScopeMap,
                              target_mlen: int,
                              target_hlen: int,
                              target_layout: str) -> tir.Stmt:
    """Lower a sequence of per-lane items WITHOUT introducing a new
    for-lane wrapper. Used inside NestedForGroups whose body is all
    per-lane: the surrounding for-lane (if any) was already emitted by
    the caller; this just lowers each item with ``in_sync=False``."""
    stmts: List[tir.Stmt] = []
    for item in items:
        if isinstance(item, GraphNode):
            stmts.append(_lower_node(
                item, lane_var=lane_var, in_sync=False,
                scopes=scopes,
                lane_count=lane_count,
                target_mlen=target_mlen,
                target_hlen=target_hlen,
                target_layout=target_layout,
            ))
        elif isinstance(item, NestedForGroup):
            inner_body = _materialize_per_lane_seq(
                item.items, lane_var, lane_count,
                scopes, target_mlen, target_hlen, target_layout,
            )
            stmts.append(tir.For(
                item.loop_var, item.min, item.extent, item.kind,
                inner_body, item.thread_binding, item.annotations or {},
            ))
        elif isinstance(item, RawStmt):
            stmts.append(item.stmt)
    if not stmts:
        return tir.Evaluate(tir.IntImm("int32", 0))
    return stmts[0] if len(stmts) == 1 else tir.SeqStmt(stmts)


def _materialize_per_lane_for(items_to_lower,
                              lane_var: tir.Var,
                              lane_count: int,
                              scopes: BufferScopeMap,
                              target_mlen: int,
                              target_hlen: int,
                              target_layout: str) -> tir.Stmt:
    """Wrap a list of per-lane items in `for lane_var in range(lane_count)`."""
    stmts: List[tir.Stmt] = []
    has_unrolled_matmul = False
    for item in items_to_lower:
        if isinstance(item, GraphNode):
            if _is_per_lane_unrolled(item):
                has_unrolled_matmul = True
            stmts.append(_lower_node(
                item, lane_var=lane_var, in_sync=False,
                scopes=scopes,
                lane_count=lane_count,
                target_mlen=target_mlen,
                target_hlen=target_hlen,
                target_layout=target_layout,
            ))
        elif isinstance(item, NestedForGroup):
            inner_body = _materialize_per_lane_seq(
                item.items, lane_var, lane_count,
                scopes, target_mlen, target_hlen, target_layout,
            )
            if _items_contain_unrolled_matmul(item.items):
                has_unrolled_matmul = True
            stmts.append(tir.For(
                item.loop_var, item.min, item.extent, item.kind,
                inner_body, item.thread_binding, item.annotations or {},
            ))
        elif isinstance(item, RawStmt):
            stmts.append(item.stmt)
    body = stmts[0] if len(stmts) == 1 else tir.SeqStmt(stmts)
    kind = tir.ForKind.UNROLLED if has_unrolled_matmul else tir.ForKind.SERIAL
    return tir.For(
        lane_var,
        tvm.tir.IntImm(lane_var.dtype, 0),
        tvm.tir.IntImm(lane_var.dtype, lane_count),
        kind, body, None, {},
    )


# ---------------------------------------------------------------------------
# Sync/per-lane partitioning (the "curtain" algorithm)
# ---------------------------------------------------------------------------

def _partition_and_materialize(items: List[Union[GraphNode, NestedForGroup]],
                                lane_var: tir.Var,
                                lane_count: int,
                                scopes: BufferScopeMap,
                                target_mlen: int,
                                target_hlen: int,
                                target_layout: str) -> tir.Stmt:
    """Walk items, partitioning at sync boundaries:
      * sync GraphNode: flush per-lane run, emit op once (in_sync=True);
      * non-sync GraphNode: accumulate into per-lane run;
      * NestedForGroup with no inner sync: accumulate into per-lane run;
      * NestedForGroup with inner sync: flush per-lane run, recursively
                                         partition body, wrap in original
                                         for(loop_var).
    """
    out: List[tir.Stmt] = []
    cur_run: List = []

    def flush_run() -> None:
        if not cur_run:
            return
        out.append(_materialize_per_lane_for(
            cur_run, lane_var, lane_count,
            scopes, target_mlen, target_hlen, target_layout,
        ))
        cur_run.clear()

    for item in items:
        if isinstance(item, GraphNode):
            if _is_sync(item):
                flush_run()
                out.append(_lower_node(
                    item, lane_var=lane_var, in_sync=True,
                    scopes=scopes,
                    lane_count=lane_count,
                    target_mlen=target_mlen,
                    target_hlen=target_hlen,
                    target_layout=target_layout,
                ))
            else:
                cur_run.append(item)
        elif isinstance(item, NestedForGroup):
            if not _has_any_sync(item.items):
                cur_run.append(item)
            else:
                flush_run()
                inner_body = _partition_and_materialize(
                    item.items, lane_var, lane_count,
                    scopes, target_mlen, target_hlen, target_layout,
                )
                out.append(tir.For(
                    item.loop_var, item.min, item.extent, item.kind,
                    inner_body, item.thread_binding, item.annotations or {},
                ))
        elif isinstance(item, RawStmt):
            cur_run.append(item)
    flush_run()

    if not out:
        return tir.Evaluate(tir.IntImm("int32", 0))
    return out[0] if len(out) == 1 else tir.SeqStmt(out)


def _materialize_lane_group(group: LaneGroup,
                            scopes: BufferScopeMap,
                            target_mlen: int,
                            target_hlen: int,
                            target_layout: str) -> tir.Stmt:
    return _partition_and_materialize(
        group.items, group.lane_var, group.lane_count,
        scopes, target_mlen, target_hlen, target_layout,
    )


# ---------------------------------------------------------------------------
# No-lane-fusion materialization (mm64-style)
# ---------------------------------------------------------------------------

def _materialize_no_lane_seq(items,
                              scopes: BufferScopeMap,
                              target_mlen: int,
                              target_hlen: int,
                              target_layout: str) -> tir.Stmt:
    stmts: List[tir.Stmt] = []
    for item in items:
        if isinstance(item, GraphNode):
            stmts.append(_lower_node(
                item, lane_var=None, in_sync=False,
                scopes=scopes,
                lane_count=4,  # unused when lane_var is None
                target_mlen=target_mlen,
                target_hlen=target_hlen,
                target_layout=target_layout,
            ))
        elif isinstance(item, NestedForGroup):
            inner = _materialize_no_lane_seq(
                item.items, scopes, target_mlen, target_hlen, target_layout,
            )
            stmts.append(tir.For(
                item.loop_var, item.min, item.extent, item.kind,
                inner, item.thread_binding, item.annotations or {},
            ))
        elif isinstance(item, RawStmt):
            stmts.append(item.stmt)
    if not stmts:
        return tir.Evaluate(tir.IntImm("int32", 0))
    return stmts[0] if len(stmts) == 1 else tir.SeqStmt(stmts)


# ---------------------------------------------------------------------------
# Root materialization
# ---------------------------------------------------------------------------

def _materialize_root(root: RootItem,
                      scopes: BufferScopeMap,
                      target_mlen: int,
                      target_hlen: int,
                      target_layout: str
                      ) -> tuple[tir.Stmt, List[tir.Buffer]]:
    """Return (body_stmt, alloc_buffers). The caller wraps body_stmt in
    a tilelang_root Block with these alloc_buffers."""
    if isinstance(root, LaneGroup):
        return (
            _materialize_lane_group(
                root, scopes, target_mlen, target_hlen, target_layout,
            ),
            list(root.alloc_buffers),
        )
    if isinstance(root, NodeRoot):
        return (
            _materialize_no_lane_seq(
                root.items, scopes, target_mlen, target_hlen, target_layout,
            ),
            list(root.alloc_buffers),
        )
    if isinstance(root, ForRoot):
        inner_body, allocs = _materialize_root(
            root.body, scopes, target_mlen, target_hlen, target_layout,
        )
        return (
            tir.For(
                root.loop_var, root.min, root.extent, root.kind,
                inner_body, root.thread_binding, root.annotations or {},
            ),
            allocs,
        )
    raise GraphPipelineError(f"unknown RootItem type {type(root).__name__}")


# ---------------------------------------------------------------------------
# Public entry: Graph → PrimFunc
# ---------------------------------------------------------------------------

def _layout_from_func_attrs(attrs) -> str:
    if attrs is None or "plena.layout" not in attrs:
        return "BSHD"
    val = attrs["plena.layout"]
    if isinstance(val, tir.StringImm):
        return str(val.value)
    return str(val)


def materialize_to_primfunc(graph: Graph,
                             scopes: BufferScopeMap,
                             lane_count: int = 4,
                             target_mlen: int = 64,
                             target_hlen: int = 16,
                             target_layout: Optional[str] = None,
                             expand_lane_buffers: bool = False,
                             ) -> tir.PrimFunc:
    """Final stage of the graph pipeline: emit a TIR PrimFunc for the
    backend to consume.

    When ``expand_lane_buffers=True`` the materialize step also runs the
    graph-layer ``allocate_group_memory.analyze`` + ``expand_buffers.expand``
    pair (the migration replacement for the legacy stmt-walker
    ``allocate_group_memory`` pass — see graph_passes/expand_buffers).
    Default is False so the existing backwards-compat entry (``run()``)
    keeps doing exactly what it used to: graph already comes in
    pre-expanded by the legacy pass.
    """
    if target_layout is None:
        target_layout = _layout_from_func_attrs(graph.attrs)

    if expand_lane_buffers:
        from .graph_passes import allocate_group_memory as g_alloc
        from .graph_passes import expand_buffers as g_expand
        from .graph_passes import lower_fp_row_patterns as g_lower_fp
        graph = g_alloc.analyze(graph, scopes, lane_count=lane_count)
        graph = g_expand.expand(graph, lane_count=lane_count)
        # lower_fp_row_patterns must run AFTER expand because the
        # row-parallel pattern matcher requires the 4D-expanded
        # buffer shape (matches legacy stmt-walker ordering).
        graph = g_lower_fp.run(graph, scopes)

    body_stmt, allocs = _materialize_root(
        graph.root, scopes, target_mlen, target_hlen, target_layout,
    )

    # Wrap body in a synthesised tilelang_root block so codegen finds
    # the alloc'd buffers.
    new_block = tir.Block(
        iter_vars=[], reads=[], writes=[],
        name_hint="tilelang_root",
        body=body_stmt,
        init=None,
        alloc_buffers=allocs,
        match_buffers=[],
        annotations={},
    )
    new_realize = tir.BlockRealize(
        iter_values=[],
        predicate=tvm.tir.IntImm("bool", 1),
        block=new_block,
    )

    return tir.PrimFunc(
        params=graph.params,
        body=new_realize,
        ret_type=graph.ret_type,
        buffer_map=graph.buffer_map,
        attrs=graph.attrs,
    )


# ---------------------------------------------------------------------------
# Backwards-compatible entry: PrimFunc (post-lift_to_blocks) → PrimFunc
__all__ = [
    "materialize_to_primfunc",
    "GraphPipelineError",
    "INHERENTLY_SYNC_EXTERNS", "PER_LANE_UNROLLED_EXTERNS",
]
