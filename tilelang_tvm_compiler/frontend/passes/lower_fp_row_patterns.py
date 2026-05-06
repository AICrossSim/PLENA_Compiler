"""Lower narrow tilelang FP/row DSL patterns to PLENA row/scalar ops.

This pass is intentionally pattern-based and conservative. It recognizes
only element-level FPRAM assignments and row-wise vector/reduce idioms that
map directly to existing ``plena.*_at`` intrinsics.
"""

from __future__ import annotations

from typing import Optional

import tvm
from tvm import tir

from .annotate_group import GROUP_KEY
from .scope_inference import BufferScopeMap


_TILEOP_REDUCE = "tl.tileop.reduce"
_TILEOP_REGION = "tl.tileop.region"


class LowerFPRowPatternsError(RuntimeError):
    pass


def _make_call(name: str, args: list) -> tir.Call:
    extern_op = tvm.ir.Op.get("tir.call_extern")
    return tir.Call("handle", extern_op, [tir.StringImm(name), *args])


def _evaluate(name: str, args: list) -> tir.Evaluate:
    return tir.Evaluate(_make_call(name, args))


def _is_scope(buf: tir.Buffer, scopes: BufferScopeMap, scope: str) -> bool:
    return scopes.get(buf.name) == scope


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


def _try_lower_fp_store(store: tir.BufferStore, scopes: BufferScopeMap):
    if not _is_scope(store.buffer, scopes, "fpram"):
        return None

    dst = tir.BufferLoad(store.buffer, list(store.indices))
    value = store.value

    src = _as_buffer_load(value)
    if src is not None and _is_scope(src.buffer, scopes, "fpram"):
        return _evaluate("plena.fp_copy_at", [src, dst])

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
            return _evaluate(name, [lhs, rhs, dst])

    if isinstance(value, tir.Call):
        op_name = getattr(value.op, "name", None)
        if op_name == "tir.exp" and len(value.args) == 1:
            src = _as_buffer_load(value.args[0])
            if src is not None and _is_scope(src.buffer, scopes, "fpram"):
                return _evaluate("plena.fp_exp_at", [src, dst])

    reci_src = _try_reci_source(value, scopes)
    if reci_src is not None:
        return _evaluate("plena.fp_reci_at", [reci_src, dst])

    return None


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
    if len(buf.shape) != 4 or len(indices) != 4:
        return None
    if not isinstance(indices[-1], tir.Var) or indices[-1].name != loop_var.name:
        return None
    if int(buf.shape[-1]) == 64:
        return indices[1], indices[2]
    return indices[1], indices[2]


def _try_lower_row_parallel(for_stmt: tir.For, scopes: BufferScopeMap):
    if not isinstance(for_stmt.body, tir.AttrStmt):
        return None
    attr = for_stmt.body
    if attr.attr_key != GROUP_KEY:
        return None
    if not isinstance(attr.body, tir.BufferStore):
        return None

    store = attr.body
    if not _is_scope(store.buffer, scopes, "vram"):
        return None
    dims = _row_dims_from_indices(store.buffer, store.indices, for_stmt.loop_var)
    if dims is None:
        return None
    dim2, dim3 = dims
    dst_load = tir.BufferLoad(store.buffer, list(store.indices))
    value = store.value

    if isinstance(value, tir.Call):
        op_name = getattr(value.op, "name", None)
        if op_name == "tir.exp" and len(value.args) == 1:
            src = _as_buffer_load(value.args[0])
            if (src is not None and src.buffer.name == store.buffer.name
                    and _same_indices(src.indices, store.indices)):
                return _evaluate("plena.row_exp_at", [
                    store.buffer.data, store.buffer.data, dim2, dim3,
                ])

    if isinstance(value, (tir.Sub, tir.Mul)):
        lhs = _as_buffer_load(value.a)
        rhs = _as_buffer_load(value.b)
        if lhs is not None and lhs.buffer.name == store.buffer.name:
            vram_load, fp_load = lhs, rhs
        elif isinstance(value, tir.Mul) and rhs is not None and rhs.buffer.name == store.buffer.name:
            vram_load, fp_load = rhs, lhs
        else:
            return None
        if not _same_indices(vram_load.indices, store.indices):
            return None
        if not (isinstance(fp_load, tir.BufferLoad)
                and _is_scope(fp_load.buffer, scopes, "fpram")):
            return None
        name = "plena.row_sub_fp_at" if isinstance(value, tir.Sub) else "plena.row_mul_fp_at"
        return _evaluate(name, [
            store.buffer.data, fp_load, store.buffer.data, dim2, dim3,
        ])

    return None


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


def _add(a, b):
    if isinstance(a, int):
        a = tir.IntImm("int32", a)
    if isinstance(b, int):
        b = tir.IntImm("int32", b)
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    # BufferRegion ranges created from T.Parallel can carry a vector-typed
    # zero/ramp as the range min. Row-reduce lowering reintroduces an
    # explicit scalar row loop, so the scalar loop var is the address we want.
    if _is_vector_expr(a) and not _is_vector_expr(b):
        return b
    return tir.Add(a, b)


def _try_lower_reduce(call: tir.Call, scopes: BufferScopeMap):
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
    if not (_is_scope(src_buf, scopes, "vram") and _is_scope(dst_buf, scopes, "fpram")):
        return None

    # PLENA's V_RED_MAX / V_RED_SUM always accumulate into the destination FP
    # slot (the codegen emits S_LD_FP -> V_RED_* -> S_ST_FP, so the existing
    # dst value is folded into the result). That matches T.reduce_*(clear=False)
    # semantics. T.reduce_*(clear=True) -- "clear dst then reduce" -- has no
    # hardware analogue here, and silently lowering it as if it were clear=False
    # produces wrong results when the dst slot still holds stale data.
    # Reject it explicitly and point users at the manual-seed pattern.
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
                f"T.reduce_{reduce_type.value}(clear=True) is not supported on PLENA: "
                f"the hardware reduction always accumulates into the dst FP slot "
                f"(equivalent to clear=False). Pass clear=False explicitly and seed "
                f"the dst slot before the reduce, e.g.\n"
                f"    M_CURR[row] = M_OLD[row]\n"
                f"    T.reduce_max(S_loc, M_CURR, dim=1, clear=False)\n"
                f"See kernels/flash_attention_min.py for the canonical pattern."
            )
    if len(src_buf.shape) != 4 or len(dst_buf.shape) != 2:
        return None

    # FPRAM buffers are authored as 1-D per-head fragments, then expanded to
    # (lane, rows).  The TileLang reduce destination region can still carry a
    # unit extent after lane expansion, so use the concrete buffer row extent.
    rows = int(dst_buf.shape[1])

    lane_expr = dst_starts[0]
    row_base = dst_starts[1]
    row = tir.Var("row", "int32")
    dst_elem = tir.BufferLoad(dst_buf, [lane_expr, _add(row_base, row)])

    if int(src_buf.shape[-1]) == 64:
        dim2 = src_starts[1]
        dim3 = _add(src_starts[2], row)
    else:
        dim2 = _add(src_starts[1], row)
        dim3 = src_starts[2]

    body = _evaluate(intrin, [src_buf.data, dst_elem, dim2, dim3])
    return tir.For(
        row, tir.IntImm("int32", 0), tir.IntImm("int32", rows),
        tir.ForKind.SERIAL, body,
    )


def _walk(stmt, scopes: BufferScopeMap):
    if isinstance(stmt, tir.SeqStmt):
        return tir.SeqStmt([_walk(c, scopes) for c in stmt.seq])
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values, predicate=stmt.predicate,
            block=_walk(stmt.block, scopes),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint, body=_walk(stmt.body, scopes),
            init=_walk(stmt.init, scopes) if stmt.init is not None else None,
            alloc_buffers=stmt.alloc_buffers, match_buffers=stmt.match_buffers,
            annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value, _walk(stmt.body, scopes),
        )
    if isinstance(stmt, tir.For):
        replaced = _try_lower_row_parallel(stmt, scopes)
        if replaced is not None:
            return replaced
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            _walk(stmt.body, scopes), stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.BufferStore):
        replaced = _try_lower_fp_store(stmt, scopes)
        return replaced if replaced is not None else stmt
    if isinstance(stmt, tir.Evaluate):
        v = stmt.value
        if isinstance(v, tir.Call) and getattr(v.op, "name", None) == _TILEOP_REDUCE:
            replaced = _try_lower_reduce(v, scopes)
            if replaced is not None:
                return replaced
        return stmt
    return stmt


def run(func: tir.PrimFunc, scopes: BufferScopeMap) -> tir.PrimFunc:
    return tir.PrimFunc(
        params=func.params,
        body=_walk(func.body, scopes),
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "LowerFPRowPatternsError"]
