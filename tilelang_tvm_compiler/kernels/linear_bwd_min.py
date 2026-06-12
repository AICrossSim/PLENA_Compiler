"""Linear-layer BACKWARD — numerically exact two-GEMM gradient.

Forward (linear_min):  Y = X @ W^T,  X=[M,K] activation, W=[N,K] weight
                       (nn.Linear convention), Y=[M,N].

Backward needs two GEMMs (the classic "forward 1 matmul -> backward 2"):
    dX = dY @ W            [M,N] @ [N,K] -> [M,K]    (contract N)
    dW = dY^T @ X          [N,M] @ [M,K] -> [N,K]    (contract M)

Each is its own kernel (one grid each — manager maps one node per kernel):
    make_linear_bwd_dx_min   -> dX
    make_linear_bwd_dw_min   -> dW

Both have the same MLEN^3-per-tile cost as the forward GEMM, so the linear
backward is ~2x the forward matmul — the dominant rule for the whole block.

PLENA mapping (A=VRAM, B=MRAM; B transposes via M_TMM, A via M_TMM_A):
  * dX = dY @ W : contract N. W stored [N,K] -> a PLAIN (non-transposed)
    matmul contracts B's first dim (N). Plain matmul overwrites dst, so the
    n_blocks N-partials are accumulated by hand.
  * dW = dY^T @ X : contract M. dY is stored [M,N] (natural forward layout)
    and used as the VRAM (A) operand with ``transpose_A=True``, so the
    matrix core transposes the A tile [M,N] -> [N,M] on the fly (lowers to
    M_TMM_A — the A-side counterpart of M_TMM). Numerically exact and
    correct GEMM cost; no pre-transpose proxy.
"""

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes


def make_linear_bwd_dx_min(*, m_blocks=1, n_blocks=1, k_blocks=1, **_ignore):
    """dX = dY @ W   ([M,N] @ [N,K] -> [M,K], contract N)."""
    _hw = _load_sizes()
    MLEN = _hw.mlen
    M, N, K = m_blocks * MLEN, n_blocks * MLEN, k_blocks * MLEN

    @T.prim_func
    def linear_bwd_dx_min(
        dY_hbm: T.Tensor((1, M, 1, N), "float16"),
        W_hbm:  T.Tensor((1, N, 1, K), "float16"),
        dX_hbm: T.Tensor((1, M, 1, K), "float16"),
    ):
        with T.Kernel(k_blocks, m_blocks, threads=128) as (bx, by):
            dY_sh  = T.alloc_shared((MLEN, MLEN), "float16")
            W_sh   = T.alloc_shared((MLEN, MLEN), "float16")
            dX_sh  = T.alloc_shared((MLEN, MLEN), "float16")
            SCR_loc = T.alloc_fragment((MLEN, MLEN), "float16")  # per-nb matmul out
            dX_loc  = T.alloc_fragment((MLEN, MLEN), "float16")  # N-accumulator
            # dX = dY @ W, contracting N. W stored [N,K] -> plain (non-transposed)
            # matmul contracts B's first dim (N). The plain matmul OVERWRITES
            # its dst (no hw accumulator like btmm_mm), so we accumulate the
            # n_blocks partials by hand: SCR = dY@W per block, dX += SCR.
            for row in T.serial(MLEN):
                for col in T.Parallel(MLEN):
                    dX_loc[row, col] = T.float16(0)
            for nb in T.serial(n_blocks):
                T.copy(dY_hbm[0, by*MLEN:(by+1)*MLEN, 0, nb*MLEN:(nb+1)*MLEN], dY_sh)
                T.copy(W_hbm[0, nb*MLEN:(nb+1)*MLEN, 0, bx*MLEN:(bx+1)*MLEN], W_sh)
                T.gemm(dY_sh, W_sh, SCR_loc)
                for row in T.serial(MLEN):
                    for col in T.Parallel(MLEN):
                        dX_loc[row, col] = dX_loc[row, col] + SCR_loc[row, col]
            T.copy(dX_loc, dX_sh)
            T.copy(dX_sh, dX_hbm[0, by*MLEN:(by+1)*MLEN, 0, bx*MLEN:(bx+1)*MLEN])

    return linear_bwd_dx_min, {"MLEN": MLEN, "M": M, "N": N, "K": K,
                               "M_BLOCKS": m_blocks, "N_BLOCKS": n_blocks, "K_BLOCKS": k_blocks}


def make_linear_bwd_dw_min(*, m_blocks=1, n_blocks=1, k_blocks=1, **_ignore):
    """dW = dY^T @ X   ([N,M] @ [M,K] -> [N,K], contract M).

    dY is stored [M,N] and transposed on the fly via transpose_A=True
    (M_TMM_A) — numerically exact, no pre-transpose proxy."""
    _hw = _load_sizes()
    MLEN = _hw.mlen
    M, N, K = m_blocks * MLEN, n_blocks * MLEN, k_blocks * MLEN

    @T.prim_func
    def linear_bwd_dw_min(
        dY_hbm: T.Tensor((1, M, 1, N), "float16"),
        X_hbm:  T.Tensor((1, M, 1, K), "float16"),
        dW_hbm: T.Tensor((1, N, 1, K), "float16"),
    ):
        with T.Kernel(k_blocks, n_blocks, threads=128) as (bx, by):
            dY_sh  = T.alloc_shared((MLEN, MLEN), "float16")
            X_sh   = T.alloc_shared((MLEN, MLEN), "float16")
            dW_sh  = T.alloc_shared((MLEN, MLEN), "float16")
            SCR_loc = T.alloc_fragment((MLEN, MLEN), "float16")  # per-mb matmul out
            dW_loc  = T.alloc_fragment((MLEN, MLEN), "float16")  # M-accumulator
            # dW = dY^T @ X, contracting M. dY is stored [M,N] (its natural
            # forward layout); we get the contraction over M with a real
            # transpose-A GEMM: ``transpose_A=True`` makes the matrix core
            # transpose the VRAM (A) tile [M,N] -> [N,M] on the fly as it
            # streams into the array (lowers to M_TMM_A), exactly the
            # symmetric counterpart of M_TMM's MRAM-side transpose. This is
            # numerically exact (dW = dY^T @ X) AND carries the right GEMM
            # cost. Plain output (no hw accumulator), so the m_blocks
            # M-partials are summed by hand, mirroring dX above.
            for row in T.serial(MLEN):
                for col in T.Parallel(MLEN):
                    dW_loc[row, col] = T.float16(0)
            for mb in T.serial(m_blocks):
                T.copy(dY_hbm[0, mb*MLEN:(mb+1)*MLEN, 0, by*MLEN:(by+1)*MLEN], dY_sh)
                T.copy(X_hbm[0, mb*MLEN:(mb+1)*MLEN, 0, bx*MLEN:(bx+1)*MLEN], X_sh)
                T.gemm(dY_sh, X_sh, SCR_loc, transpose_A=True)
                for row in T.serial(MLEN):
                    for col in T.Parallel(MLEN):
                        dW_loc[row, col] = dW_loc[row, col] + SCR_loc[row, col]
            T.copy(dW_loc, dW_sh)
            T.copy(dW_sh, dW_hbm[0, by*MLEN:(by+1)*MLEN, 0, bx*MLEN:(bx+1)*MLEN])

    return linear_bwd_dw_min, {"MLEN": MLEN, "M": M, "N": N, "K": K,
                               "M_BLOCKS": m_blocks, "N_BLOCKS": n_blocks, "K_BLOCKS": k_blocks}


__all__ = ["make_linear_bwd_dx_min", "make_linear_bwd_dw_min"]
