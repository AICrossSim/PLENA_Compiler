"""User-facing helpers for tagging a `T.gemm` with an explicit PLENA kind.

The user-facing API has **two** kinds today (plus one reserved):

  * ``"overwrite"`` (the default — used when no ``T.attr`` wraps the
    gemm) — every gemm that is **not** head-fused. Internally the
    compiler decides between two HW lowerings based on the LHS shape:

      * LHS rows == 1 → ``plena.mv``  (M_MV / M_MV_WO, per-head 1D LHS)
      * otherwise     → ``plena.matmul`` (M_MM / M_MM_WO, per-head 2D)

    Lane-axis layout of the operands is also handled automatically:

      * If the surrounding DMA / btmm / extern call already marked an
        operand's lane axis, the gemm leaves it alone (idempotent — this
        preserves the "matmul is neutral" semantics for the whole-buffer
        DMA-driven case).
      * If an operand has no surrounding marking (typical for fragment
        outputs like PV_loc in flash-attention P @ V), the gemm marks
        LHS=ROW_STACK and RHS / DST=COL_PACK, so each lane addresses
        its own head slice.

    Sliced operands are supported: starts on any of A / B / C are folded
    into ``lhs_offset / rhs_offset / dst_offset``. Whole-buffer gemms in
    a lane group get the per-lane offset auto-injected from each
    buffer's lane-axis stride — kernel authors never have to spell out a
    ``by * stride`` literal.

  * ``"btmm"`` — head-fused matmul (Q @ K^T style: same Q broadcast
    across all lanes, K split per lane, one per-head score row out per
    lane). Lowers to ``plena.btmm`` or ``plena.btmv`` (auto-dispatched
    on LHS rows the same way ``"overwrite"`` picks matmul vs mv). The
    kernel must already have set up a head-fused grid (``T.Kernel`` with
    a ``head_like`` axis at extent ``btmm_lane_count``);
    ``transpose_B=True`` is the typical Q@K^T case.

  * ``"add"`` (**reserved, not yet implemented**) — additive
    ``C += A @ B``. The planned interface: kernel author allocates a
    scratch buffer and passes it via ``T.attr`` around the gemm:

        scratch = T.alloc_fragment((rows, hlen), "float16")
        with T.attr(scratch.data, "plena.gemm_scratch", 0):
            with T.attr(0, KIND, "add"):
                T.gemm(A, B, C)        # C += A @ B

    The lowering would emit ``plena.matmul → scratch`` then
    ``plena.v_add(C, scratch, C)``. Currently the lowering raises
    ``NotImplementedError``; for now write the two ops manually
    (``T.gemm(A, B, scratch)`` + an inline T.Parallel add that
    fuse_elementwise auto-folds to ``plena.v_add``).

Usage (inline form — REQUIRED inside a ``@T.prim_func`` body, since
tilelang's eager TVMScript builder does AST analysis and cannot trace
into a helper function call)::

    from tilelang_tvm_compiler.frontend.gemm_macros import KIND

    @T.prim_func
    def k(...):
        ...
        # Default — no T.attr needed:
        T.gemm(A_sh, B_sh, C_loc)            # whole-buffer or per-head, compiler picks
        T.gemm(S_loc, V_sh, PV_loc)          # per-head P @ V (decode if rows=1, prefill else)

        # Head-fused Q @ K^T:
        with T.attr(0, KIND, "btmm"):
            T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)

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
# OVERWRITE is the default — applied when no ``T.attr(KIND, ...)`` wraps
# the gemm. BTMM is the explicit head-fused path. ADD is reserved (the
# annotate pass accepts it; the lowering raises NotImplementedError).
OVERWRITE = "overwrite"
BTMM = "btmm"
ADD = "add"


VALID_KINDS = (OVERWRITE, BTMM, ADD)


# AttrStmt key the kernel author would use to attach a scratch buffer
# to a kind="add" gemm (since ``T.gemm`` itself has no slot for it).
# Reserved for the future kind="add" lowering — see gemm_macros
# docstring above and PIPELINE_ARCHITECTURE.md § 5.4.
GEMM_SCRATCH_KEY = "plena.gemm_scratch"


__all__ = ["KIND", "OVERWRITE", "BTMM", "ADD", "VALID_KINDS", "GEMM_SCRATCH_KEY"]
