"""flash_attention gemm-only debug kernel.

Minimal slice that exercises just the two gemms (Q@K^T BTMM + P@V
matmul) of flash_attention, dropping all softmax / row_*_at /
fpram / online-state machinery. Used to bisect the new
region+dim_roles gemm schema in isolation.

Pseudocode per (q_block, by):
    Q, K, V = load from HBM
    S = Q @ K^T                  # BTMM, packed-head
    out = S @ V                  # matmul, per-head (4 lanes)

S is a stand-in for the "attention scores" tensor; output is
written directly without applying softmax. The numerical answer
won't match real attention, but the *physical shape* of S and
``out`` matches flash_attention so the gemm code paths produce the
exact same ISA shape.
"""

import tilelang.language as T

from ..frontend.gemm_macros import KIND


def make_flash_attention_gemm_only(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int | None = None,
    num_kv_blocks: int = 1,
    num_q_blocks: int = 1,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(
            f"flash_attention_gemm_only requires rows == MLEN ({MLEN}), got {rows}"
        )
    if MLEN % hlen != 0:
        raise ValueError(
            f"hlen must divide MLEN ({MLEN}); got hlen={hlen}"
        )
    hardware_lane_count = MLEN // hlen
    if head_count is None:
        head_count = hardware_lane_count
    if head_count % hardware_lane_count != 0:
        raise ValueError(
            f"head_count must be a multiple of MLEN/hlen={hardware_lane_count}; "
            f"got {head_count}"
        )

    kv_seq = num_kv_blocks * rows
    q_seq = num_q_blocks * rows

    @T.prim_func
    def flash_attention_gemm_only(
        Q_hbm: T.Tensor((1, q_seq,  head_count, hlen), "float16"),
        K_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        V_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        O_hbm: T.Tensor((1, q_seq,  head_count, hlen), "float16"),
    ):
        with T.Kernel(num_q_blocks, head_count, threads=128) as (q_block, by):
            Q_sh = T.alloc_shared((rows, hlen), "float16")
            K_sh = T.alloc_shared((rows, hlen), "float16")   # gemm RHS → mram
            V_sh = T.alloc_shared((rows, hlen), "float16")   # matmul RHS → mram
            S_loc = T.alloc_fragment((rows, MLEN), "float16")  # BTMM output
            PV_loc = T.alloc_fragment((rows, hlen), "float16")
            O_loc = T.alloc_fragment((rows, hlen), "float16")

            T.copy(
                Q_hbm[0, q_block * rows : (q_block + 1) * rows, by, 0:hlen],
                Q_sh,
            )

            # Zero output (single kv_block → no accumulation across kv).
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    O_loc[row, col] = T.float16(0)

            for kv_block in T.serial(num_kv_blocks):
                T.copy(
                    K_hbm[0, kv_block * rows : (kv_block + 1) * rows, by, 0:hlen],
                    K_sh,
                )
                T.copy(
                    V_hbm[0, kv_block * rows : (kv_block + 1) * rows, by, 0:hlen],
                    V_sh,
                )

                # BTMM Q @ K^T → S_loc.
                with T.attr(0, KIND, "btmm"):
                    T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)

                # Per-head P @ V → PV_loc, then O += PV_loc.
                T.gemm(S_loc, V_sh, PV_loc)
                for row in T.serial(rows):
                    for col in T.Parallel(hlen):
                        O_loc[row, col] = O_loc[row, col] + PV_loc[row, col]

            T.copy(
                O_loc,
                O_hbm[0, q_block * rows : (q_block + 1) * rows, by, 0:hlen],
            )

    lowered = flash_attention_gemm_only
    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
        "NUM_KV_BLOCKS": num_kv_blocks,
        "NUM_Q_BLOCKS": num_q_blocks,
    }
    return lowered, constants


__all__ = ["make_flash_attention_gemm_only"]
