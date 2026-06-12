"""RoPE BACKWARD — exact shuffle-matmul gradient.

Forward (rope_min):  OUT = X (.) COS + shuffle(X) (.) SGN_SIN,
                     shuffle(X) = X @ P,  P = pair-swap permutation (symmetric).

Backward:  dX = dOUT (.) COS + shuffle(dOUT (.) SGN_SIN)
              = dOUT (.) COS + (dOUT (.) SGN_SIN) @ P
Because P is a permutation and symmetric (P = P^T = P^-1), the backward is
structurally identical to the forward — one shuffle-GEMM plus whole-tile
vector ops. So rope backward matmul = forward matmul (1.0x), unlike the 2x
of a general linear layer.

Same BSHD-collapsed layout as rope_min: (1, SEQ, 1, H*D). The computation is
the exact RoPE input gradient (P is the symmetric pair-swap permutation).
"""

import tilelang.language as T

from ..frontend.gemm_macros import KIND, BTMM_MM
from ..plena_settings import load_sizes as _load_sizes


def make_rope_bwd_min(*, hlen=None, head_count=8, half_dim=None,
                      num_s_blocks=2, batch=1, **_ignore):
    _hw = _load_sizes()
    MLEN = _hw.mlen
    if hlen is None:
        hlen = _hw.hlen
    seq_len = num_s_blocks * MLEN
    HD = head_count * hlen
    m_blocks = seq_len // MLEN
    n_blocks = HD // MLEN

    @T.prim_func
    def rope_bwd_min(
        dOUT_hbm:    T.Tensor((1, seq_len, 1, HD), "float16"),  # upstream grad
        COS_hbm:     T.Tensor((1, seq_len, 1, HD), "float16"),
        SGN_SIN_hbm: T.Tensor((1, seq_len, 1, HD), "float16"),
        P_hbm:       T.Tensor((1, MLEN, 1, MLEN), "float16"),
        dX_hbm:      T.Tensor((1, seq_len, 1, HD), "float16"),
    ):
        with T.Kernel(n_blocks, m_blocks, threads=128) as (bx, by):
            dG_sh  = T.alloc_shared((MLEN, MLEN), "float16")  # dOUT (.) SGN_SIN
            P_sh   = T.alloc_shared((MLEN, MLEN), "float16")  # pair-swap P -> MRAM
            COS_sh = T.alloc_shared((MLEN, MLEN), "float16")
            SGN_sh = T.alloc_shared((MLEN, MLEN), "float16")
            dO_sh  = T.alloc_shared((MLEN, MLEN), "float16")
            dX_sh  = T.alloc_shared((MLEN, MLEN), "float16")
            SH_loc = T.alloc_fragment((MLEN, MLEN), "float16")  # shuffle((dOUT.SGN))

            T.copy(dOUT_hbm[0, by*MLEN:(by+1)*MLEN, 0, bx*MLEN:(bx+1)*MLEN], dO_sh)
            T.copy(SGN_SIN_hbm[0, by*MLEN:(by+1)*MLEN, 0, bx*MLEN:(bx+1)*MLEN], SGN_sh)
            T.copy(COS_hbm[0, by*MLEN:(by+1)*MLEN, 0, bx*MLEN:(bx+1)*MLEN], COS_sh)
            T.copy(P_hbm[0, 0:MLEN, 0, 0:MLEN], P_sh)

            # dG = dOUT (.) SGN_SIN   (whole-tile vector mul).
            for row in T.serial(MLEN):
                for col in T.Parallel(MLEN):
                    dG_sh[row, col] = dO_sh[row, col] * SGN_sh[row, col]

            # SH = shuffle(dG) = dG @ P   (single shuffle-GEMM).
            with T.attr(0, KIND, BTMM_MM):
                T.gemm(dG_sh, P_sh, SH_loc, transpose_B=True)

            # dX = dOUT (.) COS + SH   (whole-tile vector).
            for row in T.serial(MLEN):
                for col in T.Parallel(MLEN):
                    dX_sh[row, col] = dO_sh[row, col] * COS_sh[row, col] + SH_loc[row, col]

            T.copy(dX_sh, dX_hbm[0, by*MLEN:(by+1)*MLEN, 0, bx*MLEN:(bx+1)*MLEN])

    return rope_bwd_min, {"MLEN": MLEN, "HLEN": hlen, "HEAD_COUNT": head_count,
                          "HD": HD, "NUM_S_BLOCKS": num_s_blocks,
                          "M_BLOCKS": m_blocks, "N_BLOCKS": n_blocks}


__all__ = ["make_rope_bwd_min"]
