"""Sanity check: kernel authors must not write ``T.call_extern("plena.*")``.

Runs as the **first** frontend pass — before anything else gets a chance
to lower tile DSL into ``plena.*`` calls — so it sees only what the
kernel author actually wrote. Any direct ``T.call_extern("plena.<x>")``
in the input PrimFunc raises ``PlenaExternForbiddenError`` with the
offending op name.

Rationale: the user-facing surface is tilelang DSL only (``T.copy``,
``T.gemm``, ``T.Parallel`` patterns, etc.); ``plena.*`` extern calls are
a compiler-internal IR layer produced by lower-passes (``lower_to_hlir``,
``fuse_elementwise``). Letting authors write them directly couples
kernels to compiler internals and was the source of the
``flash_decode_min`` FPRAM-address bug — the kernel hand-rolled offset
literals (``by * MLEN``) that disagreed with the compiler's actual
buffer-allocation result.
"""

from __future__ import annotations

from tvm import tir


class PlenaExternForbiddenError(RuntimeError):
    pass


def _walk_for_plena(stmt) -> None:
    if isinstance(stmt, tir.SeqStmt):
        for c in stmt.seq:
            _walk_for_plena(c)
        return
    if isinstance(stmt, tir.BlockRealize):
        _walk_for_plena(stmt.block)
        return
    if isinstance(stmt, tir.Block):
        _walk_for_plena(stmt.body)
        if stmt.init is not None:
            _walk_for_plena(stmt.init)
        return
    if isinstance(stmt, tir.AttrStmt):
        _walk_for_plena(stmt.body)
        return
    if isinstance(stmt, tir.For):
        _walk_for_plena(stmt.body)
        return
    if isinstance(stmt, tir.LetStmt):
        _walk_for_plena(stmt.body)
        return
    if isinstance(stmt, tir.IfThenElse):
        _walk_for_plena(stmt.then_case)
        if stmt.else_case is not None:
            _walk_for_plena(stmt.else_case)
        return
    if isinstance(stmt, tir.Evaluate):
        v = stmt.value
        if (isinstance(v, tir.Call)
                and getattr(v.op, "name", None) == "tir.call_extern"
                and v.args
                and isinstance(v.args[0], tir.StringImm)
                and v.args[0].value.startswith("plena.")):
            raise PlenaExternForbiddenError(
                f"kernel may not call plena.* extern directly; "
                f"saw {v.args[0].value!r}. Use the equivalent tilelang "
                f"DSL (T.gemm + KIND, T.Parallel + binary op for v_add, "
                f"T.Parallel + 0-fill for zero_v, T.copy for DMA / row "
                f"transfers). plena.* is a compiler-internal IR layer."
            )
        return


def run(func: tir.PrimFunc) -> tir.PrimFunc:
    _walk_for_plena(func.body)
    return func


__all__ = ["run", "PlenaExternForbiddenError"]
