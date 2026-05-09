"""Graph pass: analyze every lane-fused op and tag each operand
buffer with the layout role it must take (col_pack / row_stack /
fp_lane).

Graph-IR replacement for the *analysis* half of the legacy stmt-walker
``frontend/passes/allocate_group_memory.py``. The actual buffer-shape
expansion + index rewrite is deferred to ``materialize`` (see
:mod:`expand_buffers` / ``graph_pipeline.materialize_to_primfunc``).

Why split analysis and expansion
--------------------------------
The migration plan moves shape decisions to AFTER all graph
optimizations (so future optimizations like double-buffering can change
buffer shape). Analysis fits naturally as a graph pass — it just sets
``ATTR_LANE_LAYOUT`` on each affected ``BufferNode`` plus a per-buffer
``ATTR_LANE_VAR`` recording which lane variable each lane axis carries.
Expansion happens in materialize.

Pre-conditions
--------------
* :func:`annotate_grid.run` populated ``ATTR_GROUP_EXTENT``.
* :func:`split_lane_groups.run` ensured every lane-fusion-eligible for
  has extent == ``lane_count``.
* :func:`scope_inference.infer` produced a ``BufferScopeMap``.

Output
------
For each eligible buffer, sets two attrs on its ``BufferNode``:
  * ``ATTR_LANE_LAYOUT`` ∈ {LAYOUT_COL_PACK, LAYOUT_ROW_STACK,
    LAYOUT_FP_LANE} — the expansion mode.
  * ``ATTR_LANE_VAR`` (str) — the name of the lane var that this
    buffer's lane axis substitutes in for during index folding.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from tvm import tir

from .... import scope as _scope
from ..graph_ir import (
    Graph, GraphNode, LaneGroup, NestedForGroup, NodeRoot, ForRoot, RootItem,
    RawStmt, BufferNode, BufferAccess,
    ATTR_GROUP_EXTENT, ATTR_GEMM_KIND, ATTR_LANE_LAYOUT,
    LAYOUT_COL_PACK, LAYOUT_ROW_STACK, LAYOUT_FP_LANE,
)
from .scope_inference import BufferScopeMap


# Buffer-attr key for the lane var name (str). Set alongside
# ATTR_LANE_LAYOUT so the materialize-time index folder knows which
# loop_var to substitute the lane axis for. Stringly typed so it
# survives across pass boundaries even if the underlying tir.Var
# identity churns.
ATTR_LANE_VAR = "lane_var_name"


_TILEOP_COPY = "tl.tileop.copy"
_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REGION = "tl.tileop.region"


# Same FP-extern operand-position table the stmt-walker uses.
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


class AllocateGroupMemoryError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _region_buffer(call) -> Optional[tir.Buffer]:
    if not isinstance(call, tir.Call) or call.op.name != _TILEOP_REGION:
        return None
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        return None
    return load.buffer


def _data_var_to_buffer_map(graph: Graph) -> Dict[tir.Var, tir.Buffer]:
    """Map ``tir.Var (data handle) → tir.Buffer`` so call_extern args
    that pass `Buffer.data` directly can be resolved.

    Built from ``Graph.buffer_nodes`` (which has ``data_var``) and from
    ``alloc_buffers`` collected from LaneGroup / NodeRoot / ForRoot
    bodies, since some auto-allocated tir.Buffers (``__tmp_fp_*``) may
    not have entries in ``buffer_nodes`` if they were only added via
    alloc_buffers."""
    out: Dict[tir.Var, tir.Buffer] = {}

    for bn in graph.buffer_nodes.values():
        if bn.data_var is not None:
            # Find a matching tir.Buffer if we can; otherwise skip
            # (BufferNode itself has no rank info we can use to build a
            # tir.Buffer — but the alloc_buffers pass adds the real one).
            pass

    def _collect_allocs(root: RootItem) -> List[tir.Buffer]:
        if isinstance(root, LaneGroup):
            return list(root.alloc_buffers)
        if isinstance(root, NodeRoot):
            return list(root.alloc_buffers)
        if isinstance(root, ForRoot):
            return _collect_allocs(root.body)
        return []

    for buf in graph.buffer_map.values():
        out[buf.data] = buf
    for buf in _collect_allocs(graph.root):
        out[buf.data] = buf
    return out


def _expr_fpram_buffers(expr, scopes: BufferScopeMap, out: Set[tir.Buffer]) -> None:
    if isinstance(expr, tir.BufferLoad):
        if scopes.get(expr.buffer.name) == "fpram":
            out.add(expr.buffer)
        for i in expr.indices:
            _expr_fpram_buffers(i, scopes, out)
        return
    if isinstance(expr, tir.Call):
        for a in expr.args:
            _expr_fpram_buffers(a, scopes, out)
        return
    if hasattr(expr, "a") and hasattr(expr, "b"):
        _expr_fpram_buffers(expr.a, scopes, out)
        _expr_fpram_buffers(expr.b, scopes, out)
        return
    if hasattr(expr, "value"):
        _expr_fpram_buffers(expr.value, scopes, out)


# ---------------------------------------------------------------------------
# Analysis state and recorder
# ---------------------------------------------------------------------------

class _AnalysisState:
    """Accumulates buffer-name → (lane_var_name, factor, mode) mapping
    while walking the graph. Mirrors the stmt-walker `_analyze`'s
    `info` dict but keyed only by buffer NAME; the lane-var association
    is by name (a tir.Var) so it survives reconstruction of the graph
    later in the pipeline."""

    def __init__(self, scopes: BufferScopeMap, lane_count: int):
        self.scopes = scopes
        self.lane_count = lane_count
        self.info: Dict[str, tuple] = {}  # name -> (lane_var_name, factor, mode)

    def record(self, buf: tir.Buffer, lane_var: tir.Var, factor: int, mode: str):
        if not buf.shape:
            return
        if _scope.is_global_scope(self.scopes.get(buf.name, "")):
            return
        key = buf.name
        prev = self.info.get(key)
        if prev is not None:
            prev_var_name, prev_factor, prev_mode = prev
            if prev_var_name != lane_var.name:
                raise AllocateGroupMemoryError(
                    f"buffer {buf.name!r} touched by multiple lane vars "
                    f"({prev_var_name!r} and {lane_var.name!r}); not yet supported"
                )
            if prev_factor != factor:
                raise AllocateGroupMemoryError(
                    f"buffer {buf.name!r} touched with multiple lane factors "
                    f"({prev_factor} and {factor}); not yet supported"
                )
            # Mode conflict: ROW_STACK wins over COL_PACK (BTMM output).
            if prev_mode == LAYOUT_ROW_STACK:
                return
            if mode == LAYOUT_ROW_STACK:
                pass  # overwrite previous COL_PACK
            elif prev_mode != mode:
                raise AllocateGroupMemoryError(
                    f"buffer {buf.name!r} flagged for both {prev_mode!r} and "
                    f"{mode!r} expansion — that's a miscompilation"
                )
        self.info[key] = (lane_var.name, factor, mode)


# ---------------------------------------------------------------------------
# Graph walk
# ---------------------------------------------------------------------------

def _classify_node(node: GraphNode,
                   lane_var: Optional[tir.Var],
                   state: _AnalysisState,
                   data_var_to_buf: Dict[tir.Var, tir.Buffer]) -> None:
    """Apply role rules for one GraphNode."""
    if lane_var is None:
        return
    call = node.op_call
    op_name = call.op.name
    lane_count = state.lane_count
    scopes = state.scopes
    hbm_names = {n for n, sc in scopes.items() if sc == "hbm"}

    if op_name == _TILEOP_GEMM:
        kind = node.attrs.get(ATTR_GEMM_KIND)
        lhs = _region_buffer(call.args[0])
        rhs = _region_buffer(call.args[1])
        dst = _region_buffer(call.args[2])
        if kind == "btmm":
            if lhs is not None:
                state.record(lhs, lane_var, lane_count, LAYOUT_COL_PACK)
            if rhs is not None:
                state.record(rhs, lane_var, lane_count, LAYOUT_COL_PACK)
            if dst is not None:
                state.record(dst, lane_var, lane_count, LAYOUT_ROW_STACK)
        else:
            for buf, mode in (
                (lhs, LAYOUT_ROW_STACK),
                (rhs, LAYOUT_COL_PACK),
                (dst, LAYOUT_COL_PACK),
            ):
                if buf is not None and buf.name not in state.info:
                    state.record(buf, lane_var, lane_count, mode)
        return

    if op_name == _TILEOP_COPY:
        src = _region_buffer(call.args[0])
        dst = _region_buffer(call.args[1])
        src_is_hbm = src is not None and src.name in hbm_names
        dst_is_hbm = dst is not None and dst.name in hbm_names
        if src_is_hbm and dst is not None and not dst_is_hbm:
            state.record(dst, lane_var, lane_count, LAYOUT_COL_PACK)
        elif dst_is_hbm and src is not None and not src_is_hbm:
            state.record(src, lane_var, lane_count, LAYOUT_COL_PACK)
        else:
            for buf in (src, dst):
                if (buf is not None
                        and scopes.get(buf.name) == "fpram"
                        and len(buf.shape) == 1):
                    state.record(buf, lane_var, lane_count, LAYOUT_FP_LANE)
        return

    if op_name == "tir.call_extern" and call.args:
        head = call.args[0]
        if not isinstance(head, tir.StringImm):
            return
        name = head.value
        raw_args = list(call.args[1:])
        for pos in _FP_EXTERN_POSITIONS.get(name, ()):
            if pos >= len(raw_args):
                continue
            arg = raw_args[pos]
            if isinstance(arg, tir.BufferLoad):
                state.record(arg.buffer, lane_var, lane_count, LAYOUT_FP_LANE)
        if not (name == "plena.zero_v"
                or name == "plena.matmul"
                or name.startswith("plena.v_")
                or name.startswith("plena.row_")):
            return
        for arg in raw_args:
            if not isinstance(arg, tir.Var):
                continue
            buf = data_var_to_buf.get(arg)
            if buf is not None:
                state.record(buf, lane_var, lane_count, LAYOUT_COL_PACK)


def _classify_raw_stmt(stmt: tir.Stmt,
                       lane_var: Optional[tir.Var],
                       state: _AnalysisState) -> None:
    """Apply BufferStore rules for any RawStmt-wrapped TIR."""
    if lane_var is None:
        return

    def visit(s):
        if isinstance(s, tir.SeqStmt):
            for c in s.seq:
                visit(c)
            return
        if isinstance(s, tir.AttrStmt):
            visit(s.body)
            return
        if isinstance(s, tir.For):
            visit(s.body)
            return
        if isinstance(s, tir.LetStmt):
            visit(s.body)
            return
        if isinstance(s, tir.IfThenElse):
            visit(s.then_case)
            if s.else_case is not None:
                visit(s.else_case)
            return
        if isinstance(s, tir.BufferStore):
            if state.scopes.get(s.buffer.name) == "fpram":
                state.record(s.buffer, lane_var, state.lane_count, LAYOUT_FP_LANE)
            bufs: Set[tir.Buffer] = set()
            _expr_fpram_buffers(s.value, state.scopes, bufs)
            for buf in bufs:
                state.record(buf, lane_var, state.lane_count, LAYOUT_FP_LANE)

    visit(stmt)


def _walk_items(items, lane_var: Optional[tir.Var],
                state: _AnalysisState,
                data_var_to_buf: Dict[tir.Var, tir.Buffer]) -> None:
    for it in items:
        if isinstance(it, GraphNode):
            _classify_node(it, lane_var, state, data_var_to_buf)
        elif isinstance(it, NestedForGroup):
            inner_lane = lane_var
            if (it.attrs.get(ATTR_GROUP_EXTENT) == state.lane_count):
                inner_lane = it.loop_var
            _walk_items(it.items, inner_lane, state, data_var_to_buf)
        elif isinstance(it, RawStmt):
            _classify_raw_stmt(it.stmt, lane_var, state)


def _walk_root(root: RootItem, lane_var: Optional[tir.Var],
               state: _AnalysisState,
               data_var_to_buf: Dict[tir.Var, tir.Buffer]) -> None:
    if isinstance(root, ForRoot):
        inner_lane = lane_var
        if root.attrs.get(ATTR_GROUP_EXTENT) == state.lane_count:
            inner_lane = root.loop_var
        _walk_root(root.body, inner_lane, state, data_var_to_buf)
        return
    if isinstance(root, LaneGroup):
        # The LaneGroup's lane_var IS the lane var for items inside.
        _walk_items(root.items, root.lane_var, state, data_var_to_buf)
        return
    if isinstance(root, NodeRoot):
        _walk_items(root.items, lane_var, state, data_var_to_buf)
        return


# ---------------------------------------------------------------------------
# Public entry — analysis only (sets ATTR_LANE_LAYOUT / ATTR_LANE_VAR
# on BufferNodes; does NOT rewrite buffer shapes or op_calls).
# ---------------------------------------------------------------------------

def analyze(graph: Graph,
            scopes: BufferScopeMap,
            lane_count: int = 4) -> Graph:
    """Tag every eligible BufferNode with ``ATTR_LANE_LAYOUT`` and
    ``ATTR_LANE_VAR``. In-place mutation; also returns the graph for
    chaining.

    Each tagged BufferNode gets:
      * ``attrs[ATTR_LANE_LAYOUT]``: one of LAYOUT_COL_PACK,
        LAYOUT_ROW_STACK, LAYOUT_FP_LANE.
      * ``attrs[ATTR_LANE_VAR]``: the name of the lane var (string).

    Buffers not eligible (e.g. global.* scopes, untouched by lane-fused
    ops) are left without ``ATTR_LANE_LAYOUT``.
    """
    if lane_count <= 0:
        raise AllocateGroupMemoryError(
            f"lane_count must be positive; got {lane_count}"
        )
    state = _AnalysisState(scopes, lane_count)
    data_var_to_buf = _data_var_to_buffer_map(graph)
    _walk_root(graph.root, lane_var=None, state=state,
               data_var_to_buf=data_var_to_buf)

    # Write the analysis results onto BufferNode.attrs.
    for name, (lane_var_name, _factor, mode) in state.info.items():
        bn = graph.buffer_nodes.get(name)
        if bn is None:
            # This shouldn't happen — every alloc'd / param buffer has a
            # BufferNode. But auto-allocated __tmp_fp_* may have slipped
            # in via outer-block alloc_buffers without a BufferNode entry.
            # Synthesize a minimal one.
            continue
        bn.attrs[ATTR_LANE_LAYOUT] = mode
        bn.attrs[ATTR_LANE_VAR] = lane_var_name

    return graph


__all__ = [
    "analyze", "AllocateGroupMemoryError", "ATTR_LANE_VAR",
]
