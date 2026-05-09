"""Graph pass: assign each buffer a PLENA storage scope based on how
it's used inside the graph.

This is the graph-IR replacement for the legacy stmt-walker
``frontend/passes/scope_inference.py``. Same rules, but operating on
GraphNodes (with op_call args reachable directly) rather than walking
tir stmts.

Scope rules (mirrors stmt-walker version)
-----------------------------------------
* A param buffer (HBM-backed)                              → ``"hbm"``
* User-declared ``global.<phys>`` scope                    → that scope
  (face-value; usage-consistency check elsewhere)
* ``shared.dyn`` buffer used as gemm RHS (arg[1] of any
  ``tl.tileop.gemm_py`` or arg[2] of a lowered
  ``plena.matmul``/``btmm``/``mv``/``btmv``)               → ``"mram"``
* All other ``shared.dyn`` buffers                         → ``"vram"``
* ``local.fragment`` buffer used at an FP-scalar / row-FP
  operand position of ``plena.fp_*_at`` /
  ``plena.row_*_at``, OR with rank-1 shape, OR appearing
  as a ``T.reduce`` destination with rank-1 shape, OR
  written via a BufferStore on a rank-1 buffer            → ``"fpram"``
* Other ``local.fragment``                                  → ``"vram"``

Output
------
Returns a ``BufferScopeMap`` (``dict[str, str]``) keyed by buffer name —
bit-for-bit compatible with the stmt-walker version's output, so
downstream passes (``graph_pipeline._lower_node`` etc) accept it as-is.

Status
------
Current pipeline still calls the stmt-walker ``scope_inference.infer``
for compatibility. This graph pass is invocable on a Graph object — a
follow-up wires the pipeline to call this instead, deletes the
stmt-walker version, and switches consumers to read
``BufferNode.physical_scope`` directly.
"""

from __future__ import annotations

from typing import Dict, List, Set

from tvm import tir

from .... import scope as _scope
from ..graph_ir import (
    Graph, GraphNode, NestedForGroup, LaneGroup, NodeRoot, ForRoot,
    RawStmt, RootItem,
)


# Public type alias and exception class — owned by this module now that
# the legacy stmt-walker scope_inference is gone.
BufferScopeMap = Dict[str, str]


class ScopeInferenceError(RuntimeError):
    pass


_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REGION = "tl.tileop.region"
_TILEOP_REDUCE = "tl.tileop.reduce"


# Same FP-extern operand-position table the stmt-walker uses. Keeps the
# two implementations in sync; if a new FP intrinsic is added it goes
# here once (future cleanup can move it to a shared module).
_FP_EXTERN_POSITIONS = {
    "plena.fp_copy_at": (0, 1),
    "plena.fp_zero_at": (0,),
    "plena.fp_add_at": (0, 1, 2),
    "plena.fp_sub_at": (0, 1, 2),
    "plena.fp_mul_at": (0, 1, 2),
    "plena.fp_max_at": (0, 1, 2),
    "plena.fp_exp_at": (0, 1),
    "plena.fp_reci_at": (0, 1),
    "plena.fp_sqrt_at": (0, 1),
    "plena.row_reduce_max_at": (1,),
    "plena.row_reduce_sum_at": (1,),
    "plena.row_sub_fp_at": (1,),
    "plena.row_mul_fp_at": (1,),
    "plena.row_add_fp_at": (1,),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _region_buffer_name(call: tir.Call):
    if not isinstance(call, tir.Call) or call.op.name != _TILEOP_REGION:
        return None
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        return None
    return load.buffer.name


def _region_buffer(call: tir.Call):
    if not isinstance(call, tir.Call) or call.op.name != _TILEOP_REGION:
        return None
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        return None
    return load.buffer


def _mark_rank1_fragment_loads(expr, out: Set[str]) -> None:
    """Walk ``expr`` and add to ``out`` the name of every BufferLoad
    whose buffer has rank-1 shape (= candidate FPRAM fragment)."""
    if isinstance(expr, tir.BufferLoad):
        if len(expr.buffer.shape) == 1:
            out.add(expr.buffer.name)
        for i in expr.indices:
            _mark_rank1_fragment_loads(i, out)
        return
    if isinstance(expr, tir.Call):
        for a in expr.args:
            _mark_rank1_fragment_loads(a, out)
        return
    if hasattr(expr, "a") and hasattr(expr, "b"):
        _mark_rank1_fragment_loads(expr.a, out)
        _mark_rank1_fragment_loads(expr.b, out)
        return
    if hasattr(expr, "value"):
        _mark_rank1_fragment_loads(expr.value, out)


# ---------------------------------------------------------------------------
# Per-op usage collector
# ---------------------------------------------------------------------------

def _collect_uses_from_node(node: GraphNode,
                             mram_names: Set[str],
                             fpram_names: Set) -> None:
    """Scan ``node.op_call`` and update mram/fpram usage sets."""
    call = node.op_call
    op_name = call.op.name

    # Tile DSL gemm: arg[1] is the RHS region → mram.
    if op_name == _TILEOP_GEMM:
        rhs_name = _region_buffer_name(call.args[1])
        if rhs_name is not None:
            mram_names.add(rhs_name)
        return

    # Tile DSL reduce: arg[1] is the dst region; if rank-1, it's an
    # FPRAM destination (stmt-walker rule).
    if op_name == _TILEOP_REDUCE:
        if len(call.args) >= 2:
            dst = _region_buffer(call.args[1])
            if dst is not None and len(dst.shape) == 1:
                fpram_names.add(dst.name)
        return

    if op_name == "tir.call_extern":
        if not call.args or not isinstance(call.args[0], tir.StringImm):
            return
        name = call.args[0].value
        # Already-lowered matmul/btmm/mv/btmv: arg[2] (after the name)
        # is the RHS data Var; the buffer it points to is mram.
        if name in ("plena.matmul", "plena.btmm", "plena.mv", "plena.btmv"):
            if len(call.args) >= 3 and isinstance(call.args[2], tir.Var):
                mram_names.add(call.args[2])
            return
        # FP / row_*_at: certain operand positions are FP-scalar / row.
        positions = _FP_EXTERN_POSITIONS.get(name, ())
        raw_args = list(call.args[1:])
        for pos in positions:
            if pos >= len(raw_args):
                continue
            arg = raw_args[pos]
            if isinstance(arg, tir.BufferLoad):
                fpram_names.add(arg.buffer.name)
        return


def _collect_uses_from_raw_stmt(stmt: tir.Stmt,
                                  mram_names: Set[str],
                                  fpram_names: Set) -> None:
    """Walk a RawStmt's underlying tir.Stmt and harvest fpram-related
    information (rank-1 buffer stores are FPRAM destinations; rank-1
    fragment loads are FPRAM sources)."""
    if isinstance(stmt, tir.SeqStmt):
        for c in stmt.seq:
            _collect_uses_from_raw_stmt(c, mram_names, fpram_names)
        return
    if isinstance(stmt, (tir.AttrStmt, tir.For, tir.LetStmt)):
        _collect_uses_from_raw_stmt(stmt.body, mram_names, fpram_names)
        return
    if isinstance(stmt, tir.IfThenElse):
        _collect_uses_from_raw_stmt(stmt.then_case, mram_names, fpram_names)
        if stmt.else_case is not None:
            _collect_uses_from_raw_stmt(stmt.else_case, mram_names, fpram_names)
        return
    if isinstance(stmt, tir.BufferStore):
        if len(stmt.buffer.shape) == 1:
            fpram_names.add(stmt.buffer.name)
        _mark_rank1_fragment_loads(stmt.value, fpram_names)
        return
    if isinstance(stmt, tir.BlockRealize):
        _collect_uses_from_raw_stmt(stmt.block.body, mram_names, fpram_names)
        return


# ---------------------------------------------------------------------------
# Walker over Graph
# ---------------------------------------------------------------------------

def _walk_items(items, mram_names: Set, fpram_names: Set) -> None:
    for item in items:
        if isinstance(item, GraphNode):
            _collect_uses_from_node(item, mram_names, fpram_names)
        elif isinstance(item, NestedForGroup):
            _walk_items(item.items, mram_names, fpram_names)
        elif isinstance(item, RawStmt):
            _collect_uses_from_raw_stmt(item.stmt, mram_names, fpram_names)


def _walk_root(root: RootItem, mram_names: Set, fpram_names: Set) -> None:
    if isinstance(root, LaneGroup):
        _walk_items(root.items, mram_names, fpram_names)
    elif isinstance(root, NodeRoot):
        _walk_items(root.items, mram_names, fpram_names)
    elif isinstance(root, ForRoot):
        _walk_root(root.body, mram_names, fpram_names)


# ---------------------------------------------------------------------------
# Buffer enumeration
# ---------------------------------------------------------------------------

def _collect_alloc_buffers(root: RootItem, out: List[tir.Buffer]) -> None:
    """All alloc_buffers reachable from the root."""
    if isinstance(root, LaneGroup):
        out.extend(root.alloc_buffers)
    elif isinstance(root, NodeRoot):
        out.extend(root.alloc_buffers)
    elif isinstance(root, ForRoot):
        _collect_alloc_buffers(root.body, out)


def _resolve_var_names(mram_set: Set, allocs: List[tir.Buffer]) -> Set[str]:
    """Map any tir.Var entries in ``mram_set`` (added by lowered matmul
    extern detection) back to buffer names by looking up the buffer
    whose ``.data`` matches."""
    var_to_name = {buf.data: buf.name for buf in allocs}
    out: Set[str] = set()
    for x in mram_set:
        if isinstance(x, str):
            out.add(x)
        elif isinstance(x, tir.Var) and x in var_to_name:
            out.add(var_to_name[x])
    return out


def _assign_scope(buf: tir.Buffer,
                  mram_names: Set[str],
                  fpram_names: Set[str]) -> str:
    declared = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
    if _scope.is_global_scope(declared):
        phys = _scope.physical_scope(declared)
        if buf.name in mram_names and phys != _scope.MRAM:
            raise ScopeInferenceError(
                f"buffer {buf.name!r} declared scope {declared!r} but is "
                f"used as gemm RHS — RHS operands must be in MRAM. "
                f"Declare scope='global.mram' instead."
            )
        if buf.name in fpram_names and phys != _scope.FPRAM:
            raise ScopeInferenceError(
                f"buffer {buf.name!r} declared scope {declared!r} but is "
                f"used as an FP-scalar operand — must be in FPRAM. "
                f"Declare scope='global.fpram' instead."
            )
        return declared
    if declared == "shared.dyn":
        return "mram" if buf.name in mram_names else "vram"
    if declared == "local.fragment":
        if buf.name in fpram_names or len(buf.shape) == 1:
            return "fpram"
        return "vram"
    raise ScopeInferenceError(
        f"buffer {buf.name!r} has unsupported declared scope {declared!r}; "
        f"slim scope_inference handles shared.dyn, local.fragment, and "
        f"global.vram / global.fpram / global.mram"
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def infer(graph: Graph,
          extra_buffers: List[tir.Buffer] = None) -> BufferScopeMap:
    """Walk the graph, return a ``buffer_name → scope`` map.

    ``extra_buffers``: additional alloc'd buffers not reachable from the
    graph root (e.g. ``__tmp_fp_*`` injected by lower_compound_fp_stores
    into outer blocks before lift; they sit in ``LaneGroup.alloc_buffers``
    after lift_to_graph merges them in, but if you call this on a Graph
    pre-merge, pass them here).
    """
    scopes: BufferScopeMap = {}

    # 1. Params → HBM.
    for buf in graph.buffer_map.values():
        scopes[buf.name] = "hbm"

    # 2. Walk the graph collecting uses.
    mram_names: Set = set()
    fpram_names: Set[str] = set()
    _walk_root(graph.root, mram_names, fpram_names)

    # 3. Resolve scopes for every alloc'd buffer.
    allocs: List[tir.Buffer] = []
    _collect_alloc_buffers(graph.root, allocs)
    if extra_buffers:
        allocs.extend(extra_buffers)
    mram_resolved = _resolve_var_names(mram_names, allocs)
    for buf in allocs:
        scopes[buf.name] = _assign_scope(buf, mram_resolved, fpram_names)

    return scopes


__all__ = ["infer"]
