"""Annotate every `tl.tileop.gemm_py` with its PLENA kind.

The kind comes from a user-written `T.attr(0, "plena.gemm_kind", ...)`
wrapping the gemm. If a gemm has no surrounding kind annotation, this
pass wraps it with a default of ``"overwrite"``.

Valid kinds (mirrors ``frontend.gemm_macros``):

  * ``"overwrite"`` — direct write, no accumulation. Lowers to
    ``plena.matmul``. **Default when no annotation.** Sliced operands
    are folded into the call's offset args.

  * ``"mv"`` — single-head matrix-vector. Lowers to ``plena.mv``
    (M_MV / M_MV_WO). Sliced operands fold into the three offset args.

  * ``"add"`` — additive ``C += A @ B``. Reserved for the cache-pass
    work; this pass raises ``NotImplementedError`` if it sees the kind
    so kernel authors know it's not yet wired through.

  * ``"btmm"`` — head-fused matmul. Lowers to ``plena.btmm`` under the
    surrounding group annotation.

Output: every gemm Evaluate is wrapped in an ``AttrStmt(plena.gemm_kind,
StringImm(<kind>))``. Downstream passes (``lower_to_hlir`` etc.) read
the kind directly off that AttrStmt.
"""

from __future__ import annotations

from typing import Optional

from tvm import tir


_TILEOP_GEMM = "tl.tileop.gemm_py"
KIND_KEY = "plena.gemm_kind"

VALID_KINDS = ("overwrite", "add", "btmm", "mv")
DEFAULT_KIND = "overwrite"


class GemmKindError(RuntimeError):
    pass


def _wrap_kind(stmt: tir.Stmt, kind: str) -> tir.Stmt:
    return tir.AttrStmt(
        node=tir.IntImm("int32", 0),
        attr_key=KIND_KEY,
        value=tir.StringImm(kind),
        body=stmt,
    )


def _validate(kind: str) -> None:
    if kind not in VALID_KINDS:
        raise GemmKindError(
            f"unknown {KIND_KEY}={kind!r}; expected one of {VALID_KINDS}"
        )
    if kind == "add":
        raise NotImplementedError(
            f'{KIND_KEY}="add" is not yet supported; the additive cache '
            f'pass is unimplemented. Use kind="overwrite" for now.'
        )


def _walk(stmt, active_kind: Optional[str]):
    if isinstance(stmt, tir.SeqStmt):
        return tir.SeqStmt([_walk(c, active_kind) for c in stmt.seq])
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values,
            predicate=stmt.predicate,
            block=_walk(stmt.block, active_kind),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint,
            body=_walk(stmt.body, active_kind),
            init=stmt.init, alloc_buffers=stmt.alloc_buffers,
            match_buffers=stmt.match_buffers, annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        if stmt.attr_key == KIND_KEY:
            new_kind = (
                stmt.value.value
                if isinstance(stmt.value, tir.StringImm)
                else None
            )
            if new_kind is not None:
                _validate(new_kind)
            # Drop the user-written wrapper; the gemm Evaluate downstream
            # will get its own normalised wrapper attached by this pass
            # (so the AttrStmt is produced exactly once per gemm in a
            # consistent shape, regardless of whether the user wrote the
            # annotation themselves).
            return _walk(stmt.body, active_kind=new_kind)
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value,
            _walk(stmt.body, active_kind),
        )
    if isinstance(stmt, tir.For):
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            _walk(stmt.body, active_kind),
            stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.Evaluate):
        v = stmt.value
        if isinstance(v, tir.Call) and v.op.name == _TILEOP_GEMM:
            kind = active_kind if active_kind is not None else DEFAULT_KIND
            _validate(kind)
            return _wrap_kind(stmt, kind)
        return stmt
    return stmt


def run(func: tir.PrimFunc) -> tir.PrimFunc:
    new_body = _walk(func.body, active_kind=None)
    return tir.PrimFunc(
        params=func.params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "GemmKindError", "KIND_KEY", "VALID_KINDS", "DEFAULT_KIND"]
