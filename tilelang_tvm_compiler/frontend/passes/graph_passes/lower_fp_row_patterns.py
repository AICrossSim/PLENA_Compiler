"""Graph pass: lower narrow tilelang FP/row DSL patterns to PLENA
``plena.fp_*_at`` / ``plena.row_*_at`` calls.

Graph-IR replacement for the legacy stmt-walker
``frontend/passes/lower_fp_row_patterns.py``. Same pattern set, same
intrinsic targets, but applied to graph items (RawStmt / NestedForGroup
/ GraphNode) rather than stmt-tree nodes.

Three pattern families
----------------------
1. **FP scalar store** (``BufferStore`` to FPRAM-backed buffer): becomes
   a ``plena.fp_zero_at`` / ``fp_copy_at`` / ``fp_add_at`` /
   ``fp_sub_at`` / ``fp_mul_at`` / ``fp_exp_at`` / ``fp_reci_at``
   GraphNode. Source items: ``RawStmt(tir.BufferStore)``.

2. **Row-vector parallel store** (``T.Parallel`` over a VRAM buffer's
   last dim, post-``annotate_grid``): becomes ``plena.row_exp_at`` /
   ``row_sub_fp_at`` / ``row_mul_fp_at`` GraphNode. Source items:
   ``NestedForGroup(attrs[ATTR_GROUP_EXTENT]==extent,
   items=[RawStmt(BufferStore)])``.

3. **Reduce** (``Evaluate(tl.tileop.reduce(...))`` with VRAM source +
   FPRAM destination): becomes a serial for-loop wrapping a per-row
   ``plena.row_reduce_max_at`` / ``row_reduce_sum_at`` call. Source
   items: ``GraphNode(op_call=tl.tileop.reduce, ...)``. The replacement
   is a ``tir.For`` (no graph-IR analogue today), so it goes back into
   the graph as a ``RawStmt``.

Pre-conditions
--------------
* :func:`annotate_grid.run` has populated ``ATTR_GROUP_EXTENT``.
* A ``BufferScopeMap`` (``dict[str, str]``) is provided — call
  :func:`graph_passes.scope_inference.infer(graph)` first.
"""

from __future__ import annotations

from typing import Optional

import tvm
from tvm import tir

from .... import scope as _scope
from ..graph_ir import (
    Graph, GraphNode, LaneGroup, NestedForGroup, NodeRoot, ForRoot, RootItem,
    RawStmt, BufferAccess,
    ATTR_GROUP_EXTENT,
)
from .scope_inference import BufferScopeMap


_TILEOP_REDUCE = "tl.tileop.reduce"
_TILEOP_REGION = "tl.tileop.region"


class LowerFPRowPatternsError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Helpers (parallel to the stmt-walker — kept verbatim where applicable)
# ---------------------------------------------------------------------------

def _make_call(name: str, args: list) -> tir.Call:
    extern_op = tvm.ir.Op.get("tir.call_extern")
    return tir.Call("handle", extern_op, [tir.StringImm(name), *args])


def _is_scope(buf: tir.Buffer, scopes: BufferScopeMap, scope: str) -> bool:
    declared = scopes.get(buf.name)
    if declared is None:
        return False
    return _scope.physical_scope(declared) == scope


def _same_indices(a, b) -> bool:
    if len(a) != len(b):
        return False
    return all(str(x) == str(y) for x, y in zip(a, b))


def _as_buffer_load(expr) -> Optional[tir.BufferLoad]:
    if isinstance(expr, tir.BufferLoad):
        return expr
    return None


def _strip_cast(expr):
    while isinstance(expr, tir.Cast):
        expr = expr.value
    return expr


def _is_one(expr) -> bool:
    expr = _strip_cast(expr)
    if isinstance(expr, tir.IntImm):
        return int(expr.value) == 1
    if isinstance(expr, tir.FloatImm):
        return float(expr.value) == 1.0
    return False


def _is_zero(expr) -> bool:
    expr = _strip_cast(expr)
    if isinstance(expr, tir.IntImm):
        return int(expr.value) == 0
    if isinstance(expr, tir.FloatImm):
        return float(expr.value) == 0.0
    value = getattr(expr, "value", None)
    if value is not None:
        return _is_zero(value)
    return str(expr) in {"0", "x1(0)", "x4(0)", "x16(0)", "x64(0)"}


def _is_vector_expr(expr) -> bool:
    dtype = getattr(expr, "dtype", None)
    lanes = getattr(dtype, "lanes", 1)
    try:
        return int(lanes) > 1
    except TypeError:
        return False


def _add(a, b):
    if isinstance(a, int):
        a = tir.IntImm("int32", a)
    if isinstance(b, int):
        b = tir.IntImm("int32", b)
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    if _is_vector_expr(a) and not _is_vector_expr(b):
        return b
    return tir.Add(a, b)


def _full_access(buf: tir.Buffer) -> BufferAccess:
    return BufferAccess(
        buffer_name=buf.name,
        starts=[tir.IntImm("int32", 0) for _ in buf.shape],
        extents=list(buf.shape),
    )


def _try_reci_source(expr, scopes: BufferScopeMap) -> Optional[tir.BufferLoad]:
    expr = _strip_cast(expr)
    if not isinstance(expr, tir.Div):
        return None
    if not _is_one(expr.a):
        return None
    rhs = _strip_cast(expr.b)
    if isinstance(rhs, tir.BufferLoad) and _is_scope(rhs.buffer, scopes, "fpram"):
        return rhs
    return None


def _row_dims_from_indices(buf: tir.Buffer, indices, loop_var: tir.Var):
    """Extract the logical (row, head) coordinates from a 4D BSHD access.

    The buffer's shape is always BSHD ``(B, S, H, D)`` post-expand_buffers.
    Which axis carries the lane depends on the expansion mode:

      * COL_PACK ``(1, S, lane, narrow_D)`` — lane in H axis at indices[2]
      * ROW_STACK ``(lane, S, 1, MLEN)``    — lane in B axis at indices[0]
      * Single tile / wide-D ``(1, S, 1, *)`` — no lane, head defaults to 0

    Returns the layout-agnostic (row, head) pair so downstream
    ``_resolve_row_at_coords`` can translate it back to physical coords
    via ``buf.layout`` + ``buf.tile_layout``.
    """
    if len(buf.shape) != 4 or len(indices) != 4:
        return None
    if not isinstance(indices[-1], tir.Var) or indices[-1].name != loop_var.name:
        return None
    b_dim = int(buf.shape[0])
    h_dim = int(buf.shape[2])
    row = indices[1]
    if h_dim > 1 and b_dim == 1:
        head = indices[2]      # COL_PACK
    elif b_dim > 1 and h_dim == 1:
        head = indices[0]      # ROW_STACK
    else:
        head = indices[2]      # single-tile / wide-D — head is 0 anyway
    return row, head


def _region_components(call: tir.Call):
    if isinstance(call, tir.BufferRegion) or (
        hasattr(call, "buffer") and hasattr(call, "region")
    ):
        return (
            call.buffer,
            [r.min for r in call.region],
            [r.extent for r in call.region],
        )
    if isinstance(call, tir.BufferLoad):
        starts = []
        extents = []
        for idx in call.indices:
            if isinstance(idx, tvm.ir.Range):
                starts.append(idx.min)
                extents.append(idx.extent)
            else:
                starts.append(idx)
                extents.append(tir.IntImm("int32", 1))
        return call.buffer, starts, extents
    if not isinstance(call, tir.Call) or call.op.name != _TILEOP_REGION:
        raise LowerFPRowPatternsError(
            f"expected {_TILEOP_REGION}, got {type(call).__name__}: {call!r}"
        )
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        raise LowerFPRowPatternsError("region arg[0] must be BufferLoad")
    starts = list(load.indices)
    extents = list(call.args[2:])
    return load.buffer, starts, extents


# ---------------------------------------------------------------------------
# 1. FP scalar store (RawStmt(BufferStore)) → GraphNode
# ---------------------------------------------------------------------------

def _try_lower_fp_store(store: tir.BufferStore,
                        scopes: BufferScopeMap) -> Optional[GraphNode]:
    if not _is_scope(store.buffer, scopes, "fpram"):
        return None

    dst = tir.BufferLoad(store.buffer, list(store.indices))
    value = store.value

    def _wrap(name: str, args: list, reads_bufs=()) -> GraphNode:
        return GraphNode(
            name=f"{name.replace('plena.', '')}_{store.buffer.name}",
            op_call=_make_call(name, args),
            attrs={},
            reads=[_full_access(b) for b in reads_bufs if b is not None],
            writes=[_full_access(store.buffer)],
        )

    if _is_zero(value):
        return _wrap("plena.fp_zero_at", [dst])

    src = _as_buffer_load(value)
    if src is not None and _is_scope(src.buffer, scopes, "fpram"):
        return _wrap("plena.fp_copy_at", [src, dst], reads_bufs=[src.buffer])

    if isinstance(value, (tir.Add, tir.Sub, tir.Mul)):
        lhs = _as_buffer_load(value.a)
        rhs = _as_buffer_load(value.b)
        if (lhs is not None and rhs is not None
                and _is_scope(lhs.buffer, scopes, "fpram")
                and _is_scope(rhs.buffer, scopes, "fpram")):
            name = {
                tir.Add: "plena.fp_add_at",
                tir.Sub: "plena.fp_sub_at",
                tir.Mul: "plena.fp_mul_at",
            }[type(value)]
            return _wrap(name, [lhs, rhs, dst],
                         reads_bufs=[lhs.buffer, rhs.buffer])

    if isinstance(value, tir.Call):
        op_name = getattr(value.op, "name", None)
        if op_name == "tir.exp" and len(value.args) == 1:
            src = _as_buffer_load(value.args[0])
            if src is not None and _is_scope(src.buffer, scopes, "fpram"):
                return _wrap("plena.fp_exp_at", [src, dst],
                             reads_bufs=[src.buffer])

    reci_src = _try_reci_source(value, scopes)
    if reci_src is not None:
        return _wrap("plena.fp_reci_at", [reci_src, dst],
                     reads_bufs=[reci_src.buffer])

    return None


# ---------------------------------------------------------------------------
# 2. Row-vector parallel store (NestedForGroup) → GraphNode
# ---------------------------------------------------------------------------

def _try_lower_row_parallel(forgrp: NestedForGroup,
                            scopes: BufferScopeMap) -> Optional[GraphNode]:
    if forgrp.attrs.get(ATTR_GROUP_EXTENT) is None:
        return None
    if len(forgrp.items) != 1:
        return None
    item = forgrp.items[0]
    if not isinstance(item, RawStmt) or not isinstance(item.stmt, tir.BufferStore):
        return None
    store = item.stmt
    if not _is_scope(store.buffer, scopes, "vram"):
        return None
    dims = _row_dims_from_indices(store.buffer, store.indices, forgrp.loop_var)
    if dims is None:
        return None
    dim2, dim3 = dims
    value = store.value

    def _wrap(name: str, args: list, reads_bufs=()) -> GraphNode:
        return GraphNode(
            name=f"{name.replace('plena.', '')}_{store.buffer.name}",
            op_call=_make_call(name, args),
            attrs={},
            reads=[_full_access(b) for b in reads_bufs if b is not None],
            writes=[_full_access(store.buffer)],
        )

    if isinstance(value, tir.Call):
        op_name = getattr(value.op, "name", None)
        if op_name == "tir.exp" and len(value.args) == 1:
            src = _as_buffer_load(value.args[0])
            if (src is not None and src.buffer.name == store.buffer.name
                    and _same_indices(src.indices, store.indices)):
                return _wrap("plena.row_exp_at", [
                    store.buffer.data, store.buffer.data, dim2, dim3,
                ], reads_bufs=[store.buffer])

    if isinstance(value, (tir.Sub, tir.Mul)):
        lhs = _as_buffer_load(value.a)
        rhs = _as_buffer_load(value.b)
        if lhs is not None and lhs.buffer.name == store.buffer.name:
            vram_load, fp_load = lhs, rhs
        elif (isinstance(value, tir.Mul) and rhs is not None
              and rhs.buffer.name == store.buffer.name):
            vram_load, fp_load = rhs, lhs
        else:
            return None
        if not _same_indices(vram_load.indices, store.indices):
            return None
        if not (isinstance(fp_load, tir.BufferLoad)
                and _is_scope(fp_load.buffer, scopes, "fpram")):
            return None
        name = ("plena.row_sub_fp_at" if isinstance(value, tir.Sub)
                else "plena.row_mul_fp_at")
        return _wrap(name, [
            store.buffer.data, fp_load, store.buffer.data, dim2, dim3,
        ], reads_bufs=[store.buffer, fp_load.buffer])

    return None


# ---------------------------------------------------------------------------
# 3. Reduce (GraphNode(tl.tileop.reduce)) → RawStmt(For wrapping plena.row_reduce_*)
# ---------------------------------------------------------------------------

def _try_lower_reduce(node: GraphNode,
                      scopes: BufferScopeMap) -> Optional[RawStmt]:
    call = node.op_call
    if call.op.name != _TILEOP_REDUCE:
        return None
    if len(call.args) < 5:
        return None
    src_buf, src_starts, _src_exts = _region_components(call.args[0])
    dst_buf, dst_starts, dst_exts = _region_components(call.args[1])
    reduce_type = call.args[2]
    if not isinstance(reduce_type, tir.StringImm):
        return None
    intrin = {
        "max": "plena.row_reduce_max_at",
        "sum": "plena.row_reduce_sum_at",
    }.get(reduce_type.value)
    if intrin is None:
        return None
    if not (_is_scope(src_buf, scopes, "vram")
            and _is_scope(dst_buf, scopes, "fpram")):
        return None

    if len(call.args) >= 5:
        clear_arg = call.args[4]
        clear_val: Optional[bool] = None
        if isinstance(clear_arg, tir.IntImm):
            clear_val = bool(clear_arg.value)
        elif isinstance(clear_arg, bool):
            clear_val = clear_arg
        if clear_val is None:
            raise LowerFPRowPatternsError(
                f"T.reduce_{reduce_type.value}: cannot interpret 'clear' "
                f"argument {clear_arg!r} (expected bool / IntImm)"
            )
        if clear_val:
            raise LowerFPRowPatternsError(
                f"T.reduce_{reduce_type.value}(clear=True) is not supported "
                f"on PLENA: the hardware reduction always accumulates into "
                f"the dst FP slot (equivalent to clear=False). Pass "
                f"clear=False explicitly and seed the dst slot before the "
                f"reduce."
            )
    if len(src_buf.shape) != 4 or len(dst_buf.shape) != 2:
        return None

    rows = int(dst_buf.shape[1])
    lane_expr = dst_starts[0]
    row_base = dst_starts[1]
    row = tir.Var("row", "int32")
    dst_elem = tir.BufferLoad(dst_buf, [lane_expr, _add(row_base, row)])

    # Layout-agnostic (row, head) emission. The src buffer is 4D BSHD
    # but the lane axis differs by expansion mode:
    #   COL_PACK  (1, S, lane, narrow_D)  → head = src_starts[2]
    #   ROW_STACK (lane, S, 1, MLEN)      → head = src_starts[0]
    #   single tile / wide-D (1, S, 1, *) → head = 0 (unused downstream)
    # isa_pass._resolve_row_at_coords translates (row, head) back to
    # physical (B, S, H, D) using buf.layout/tile_layout.
    b_dim = int(src_buf.shape[0])
    h_dim = int(src_buf.shape[2])
    s_base = src_starts[1]
    if h_dim > 1 and b_dim == 1:
        head_expr = src_starts[2]      # COL_PACK
    elif b_dim > 1 and h_dim == 1:
        head_expr = src_starts[0]      # ROW_STACK
    else:
        head_expr = tir.IntImm("int32", 0)
    row_expr = _add(s_base, row)

    body = tir.Evaluate(_make_call(intrin, [src_buf.data, dst_elem, row_expr, head_expr]))
    for_stmt = tir.For(
        row, tir.IntImm("int32", 0), tir.IntImm("int32", rows),
        tir.ForKind.SERIAL, body,
    )
    return RawStmt(name=f"{intrin.replace('plena.', '')}_{dst_buf.name}",
                   stmt=for_stmt)


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------

def _lower_items(items, scopes: BufferScopeMap):
    out = []
    for item in items:
        if isinstance(item, GraphNode):
            replaced = _try_lower_reduce(item, scopes)
            if replaced is not None:
                out.append(replaced)
                continue
            out.append(item)
            continue
        if isinstance(item, NestedForGroup):
            # Try the row-parallel pattern first; if it fires the whole
            # for-group is replaced.
            replaced = _try_lower_row_parallel(item, scopes)
            if replaced is not None:
                out.append(replaced)
                continue
            # Otherwise recurse into the body.
            inner = _lower_items(item.items, scopes)
            out.append(NestedForGroup(
                loop_var=item.loop_var, min=item.min, extent=item.extent,
                kind=item.kind, thread_binding=item.thread_binding,
                annotations=item.annotations, items=inner,
                attrs=dict(item.attrs),
            ))
            continue
        if isinstance(item, RawStmt):
            if isinstance(item.stmt, tir.BufferStore):
                replaced = _try_lower_fp_store(item.stmt, scopes)
                if replaced is not None:
                    out.append(replaced)
                    continue
            out.append(item)
            continue
        out.append(item)
    return out


def _lower_root(root: RootItem, scopes: BufferScopeMap) -> RootItem:
    if isinstance(root, ForRoot):
        return ForRoot(
            loop_var=root.loop_var, min=root.min, extent=root.extent,
            kind=root.kind, thread_binding=root.thread_binding,
            annotations=root.annotations, body=_lower_root(root.body, scopes),
            attrs=dict(root.attrs),
        )
    if isinstance(root, LaneGroup):
        return LaneGroup(
            lane_var=root.lane_var, lane_count=root.lane_count,
            items=_lower_items(root.items, scopes),
            alloc_buffers=list(root.alloc_buffers),
        )
    if isinstance(root, NodeRoot):
        return NodeRoot(
            items=_lower_items(root.items, scopes),
            alloc_buffers=list(root.alloc_buffers),
        )
    return root


def run(graph: Graph, scopes: BufferScopeMap) -> Graph:
    """Lower FP/row-vector patterns into ``plena.fp_*_at`` /
    ``plena.row_*_at`` calls. Returns a NEW Graph."""
    new_root = _lower_root(graph.root, scopes)
    return Graph(
        root=new_root,
        params=graph.params,
        buffer_map=graph.buffer_map,
        ret_type=graph.ret_type,
        attrs=graph.attrs,
        buffer_nodes=graph.buffer_nodes,
    )


__all__ = ["run", "LowerFPRowPatternsError"]
