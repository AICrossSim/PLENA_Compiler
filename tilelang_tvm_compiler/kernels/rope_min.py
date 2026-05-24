"""RoPE-min kernel — shuffle-matrix matmul form (no FPRAM pair-swap).

RoPE's pair-swap (OUT[d] depends on X[d] and its pair partner X[d^1]) used
to be done by mapping each VRAM row into FPRAM and doing per-element scalar
FMA — a V↔FPRAM MAP + scalar chain per row that dominated latency (~53 ms
at 1024×1024). This rewrite expresses the whole thing as whole-tile vector
+ BTMM ops:

    OUT = X ⊙ COS  +  shuffle(X) ⊙ SGN_SIN
    shuffle(X) = X @ P

where:
  * P is the (H*D × H*D) **pair-swap permutation matrix** (block-diagonal
    2×2 swaps [[0,1],[1,0]]); X @ P swaps every adjacent (even,odd) pair.
  * SGN_SIN[d] = -sin at even d, +sin at odd d (host pre-combines the old
    NEG_SIN/SIN split into one tensor).
  * ⊙ is whole-tile elementwise (V_MUL / V_ADD); X @ P runs over the BTMM
    ``btmm_mm`` path (M_BTMM + deferred bmm_wo drain + accumulate), exactly
    like linear_min.

Layout: BSHD is collapsed to ``(1, SEQ, 1, H*D)`` so head+dim form one big
linear-style axis (H*D = head_count*hlen). The shuffle is then a plain
(SEQ × H*D) @ (H*D × H*D) GEMM. Head boundaries are multiples of hlen
(even), so no 2×2 pair straddles a head — the global pair-swap equals the
per-head pair-swap.

K-side RoPE is identical with SGN_SIN's sign convention flipped.
"""

import tilelang.language as T

from ..frontend.gemm_macros import KIND, BTMM_MM
from ..plena_settings import load_sizes as _load_sizes


def make_rope_min(
    *,
    rows: int | None = None,
    hlen: int | None = None,
    head_count: int = 8,
    half_dim: int | None = None,
    num_s_blocks: int = 2,
    batch: int = 1,
):
    _hw = _load_sizes()
    MLEN = _hw.mlen
    if hlen is None:
        hlen = _hw.hlen
    if rows is None:
        rows = MLEN
    if rows != MLEN:
        raise ValueError(f"rope_min requires rows == MLEN ({MLEN}), got {rows}")
    if MLEN % hlen != 0:
        raise ValueError(f"hlen must divide MLEN ({MLEN}); got hlen={hlen}")
    if hlen % 2 != 0:
        raise ValueError(f"hlen must be even for RoPE pair-swap; got {hlen}")
    if num_s_blocks < 1:
        raise ValueError(f"num_s_blocks must be >= 1, got {num_s_blocks}")
    # half_dim kept for signature compat with the chain / testbench; the
    # vector form doesn't need it (pair-swap is the permutation matrix).
    if half_dim is not None and half_dim * 2 != hlen:
        raise ValueError(
            f"half_dim*2 ({half_dim * 2}) must equal hlen ({hlen})"
        )

    seq_len = num_s_blocks * rows
    HD = head_count * hlen          # collapsed head*dim axis
    if HD % MLEN != 0:
        raise ValueError(
            f"H*D (head_count*hlen = {HD}) must be a multiple of MLEN "
            f"({MLEN})"
        )
    m_blocks = seq_len // MLEN
    n_blocks = HD // MLEN           # output spans H*D in MLEN-wide blocks
    if seq_len % MLEN != 0:
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of MLEN")

    @T.prim_func
    def rope_min(
        XQ_hbm:      T.Tensor((1, seq_len, 1, HD), "float16"),
        COS_hbm:     T.Tensor((1, seq_len, 1, HD), "float16"),
        SGN_SIN_hbm: T.Tensor((1, seq_len, 1, HD), "float16"),
        P_hbm:       T.Tensor((1, MLEN, 1, MLEN), "float16"),
        Q_OUT_hbm:   T.Tensor((1, seq_len, 1, HD), "float16"),
    ):
        # Grid: one program per (n_block, m_block) output tile, same axis
        # order as linear_min (bx along N=H*D, by along M=SEQ).
        with T.Kernel(n_blocks, m_blocks, threads=128) as (bx, by):
            X_sh   = T.alloc_shared((MLEN, MLEN), "float16")  # X[by, bx]
            P_sh   = T.alloc_shared((MLEN, MLEN), "float16")  # shared pair-swap P (→ MRAM)
            COS_sh = T.alloc_shared((MLEN, MLEN), "float16")
            SGN_sh = T.alloc_shared((MLEN, MLEN), "float16")
            OUT_sh = T.alloc_shared((MLEN, MLEN), "float16")

            XS_loc = T.alloc_fragment((MLEN, MLEN), "float16")  # shuffle(X) tile

            # shuffle: XS = X[by, bx] @ P. P is the (MLEN×MLEN) pair-swap
            # permutation — block-diagonal, so the pair-swap never crosses a
            # MLEN boundary and EACH output column block bx only needs X's
            # OWN block bx times the single shared diagonal P block. No K
            # accumulation (single btmm_mm). transpose_B is harmless (P
            # symmetric). The deferred bmm_wo drain (before XS_loc's first
            # read below) writes the result into XS_loc.
            T.copy(
                XQ_hbm[0, by * MLEN:(by + 1) * MLEN, 0,
                       bx * MLEN:(bx + 1) * MLEN],
                X_sh,
            )
            T.copy(P_hbm[0, 0:MLEN, 0, 0:MLEN], P_sh)
            with T.attr(0, KIND, BTMM_MM):
                T.gemm(X_sh, P_sh, XS_loc, transpose_B=True)

            # Elementwise term operands at the same (by, bx) block.
            T.copy(
                COS_hbm[0, by * MLEN:(by + 1) * MLEN, 0,
                        bx * MLEN:(bx + 1) * MLEN],
                COS_sh,
            )
            T.copy(
                SGN_SIN_hbm[0, by * MLEN:(by + 1) * MLEN, 0,
                            bx * MLEN:(bx + 1) * MLEN],
                SGN_sh,
            )

            # OUT = X ⊙ COS + shuffle(X) ⊙ SGN_SIN  (whole-tile vector).
            # X_sh still holds X[by, bx] (gemm wrote XS_loc, not X_sh).
            for row in T.serial(MLEN):
                for col in T.Parallel(MLEN):
                    OUT_sh[row, col] = (
                        X_sh[row, col] * COS_sh[row, col]
                        + XS_loc[row, col] * SGN_sh[row, col]
                    )

            T.copy(
                OUT_sh,
                Q_OUT_hbm[0, by * MLEN:(by + 1) * MLEN, 0,
                          bx * MLEN:(bx + 1) * MLEN],
            )

    lowered = rope_min

    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "HD": HD,
        "BATCH": batch,
        "NUM_S_BLOCKS": num_s_blocks,
        "M_BLOCKS": m_blocks,
        "N_BLOCKS": n_blocks,
    }
    return lowered, constants


__all__ = ["make_rope_min"]
