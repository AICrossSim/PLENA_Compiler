"""User-facing helpers for tagging a `T.gemm` with an explicit PLENA kind.

Four kinds are recognised today:

  * ``"overwrite"`` — the most common case. C is overwritten with A @ B;
    no software accumulation is needed. Lowers to the unified
    ``plena.matmul`` op. Sliced operands are supported: starts on any
    of A / B / C are folded into ``lhs_offset / rhs_offset / dst_offset``
    so per-head ``T.gemm(A[..., by, ...], B[..., by, ...], C[..., by, ...])``
    works without dropping to ``T.call_extern``.

  * ``"mv"`` — single-head matrix-vector via ``M_MV / M_MV_WO``. Same
    lowering shape as ``overwrite`` but emits ``plena.mv`` (no
    M_tiles / K_tiles / dst_row_stride; just the three offsets). Use
    this when the LHS is a single MLEN-wide row of a row-stacked
    fragment — e.g., per-head P @ V in the decode flash-attention
    kernel.

  * ``"add"`` — additive ``C += A @ B``. Requires a cache + element-wise
    add to preserve the prior C value because PLENA's matmul hardware
    overwrites its destination. **Not yet implemented** at the lowering
    level; the annotation pass raises ``GemmPathError`` if it sees this
    kind. Reserved here so kernel authors can lock in the right intent
    and the compiler will pick it up once the cache pass lands.

  * ``"btmm"`` — head-fused matmul. Lowers to ``plena.btmm`` (and uses
    the M_BTMM / M_BMM_WO hardware path). The kernel must already have
    set up a head-fused grid (``T.Kernel`` with a ``head_like`` axis at
    extent ``btmm_lane_count``); ``transpose_B=True`` is the typical
    Q@K^T case.

Usage (inline form — REQUIRED inside a ``@T.prim_func`` body, since
tilelang's eager TVMScript builder does AST analysis and cannot trace
into a helper function call)::

    from tilelang_tvm_compiler.frontend.gemm_macros import KIND

    @T.prim_func
    def k(...):
        ...
        with T.attr(0, KIND, "overwrite"):
            T.gemm(A_sh, B_sh, C_loc)

        with T.attr(0, KIND, "btmm"):
            T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)

        # Per-head P @ V: slice S_loc / V_sh / PV_loc by the head index
        # and let the lowering fold the slice starts into mv offsets.
        with T.attr(0, KIND, "mv"):
            T.gemm(S_loc[0, by, 0, 0:MLEN],
                   V_sh[0, 0:rows, by, 0:hlen],
                   PV_loc[0, 0, by, 0:hlen])

The ``T.attr`` wraps the next statement (the ``T.gemm``) in a
``tir.AttrStmt`` carrying ``attr_key="plena.gemm_kind"`` and
``value=StringImm(<kind>)``. The ``annotate_gemm_path`` pass picks it
up and overrides any shape-driven auto-detection.
"""

from __future__ import annotations


# Attribute key used by `annotate_gemm_path` to read the explicit kind.
# Kernel authors should pass this constant to `T.attr(0, KIND, "<kind>")`
# (rather than typing the literal string) so refactors of the key name
# stay consistent.
KIND = "plena.gemm_kind"


# Valid kind values (mirrors the lookup in `annotate_gemm_path`).
OVERWRITE = "overwrite"
ADD = "add"
BTMM = "btmm"
MV = "mv"


VALID_KINDS = (OVERWRITE, ADD, BTMM, MV)


__all__ = ["KIND", "OVERWRITE", "ADD", "BTMM", "MV", "VALID_KINDS"]
