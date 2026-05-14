"""flash_decode gemm-only debug kernel.

Strips flash_decode_min down to just BTMV(Q@K^T) + MV(S@V) — no
softmax, no online state, no FPRAM scalars. Used to bisect the
new region+dim_roles gemm schema on the multi-by_number path.

Per by_o iteration:
    Q_sh ← Q_cache[by_o*lane_count, 0]   (vram→vram MLEN-wide pull)
    K_sh, V_sh ← HBM
    S_loc  = Q_sh @ K_sh^T                (BTMV, packed-head)
    PV_loc = S_loc @ V_sh                 (MV, per-head)
    O_loc  = (zero) + PV_loc accumulated over kv_blocks
    O_cache[by_o*lane_count, 0] ← O_loc   (vram→vram MLEN-wide store)
"""

import tilelang.language as T

from ..frontend.gemm_macros import KIND


def make_flash_decode_min_gemm_only(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int | None = None,
    num_kv_blocks: int = 2,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"requires rows == MLEN ({MLEN}), got {rows}")
    if MLEN % hlen != 0:
        raise ValueError(f"hlen must divide MLEN ({MLEN}); got hlen={hlen}")
    hardware_lane_count = MLEN // hlen
    if head_count is None:
        head_count = hardware_lane_count
    if head_count % hardware_lane_count != 0:
        raise ValueError(
            f"head_count must be a multiple of hardware_lane_count "
            f"({hardware_lane_count}); got head_count={head_count}"
        )
    if num_kv_blocks < 1:
        raise ValueError(f"num_kv_blocks must be >= 1, got {num_kv_blocks}")

    kv_seq = num_kv_blocks * rows

    @T.prim_func
    def flash_decode_min_gemm_only(
        K_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        V_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
    ):
        with T.Kernel(1, head_count, threads=128) as (_, by):
            Q_cache = T.alloc_shared((head_count, hlen), "float16",
                                      scope="global.vram")
            O_cache = T.alloc_shared((head_count, hlen), "float16",
                                      scope="global.vram")
            Q_sh   = T.alloc_shared((1, hlen), "float16")
            K_sh   = T.alloc_shared((rows, hlen), "float16")
            V_sh   = T.alloc_shared((rows, hlen), "float16")
            S_loc  = T.alloc_fragment((1, MLEN), "float16")
            PV_loc = T.alloc_fragment((1, hlen), "float16")
            O_loc  = T.alloc_fragment((1, hlen), "float16")

            T.copy(Q_cache[by, 0], Q_sh)

            for col in T.Parallel(hlen):
                O_loc[0, col] = T.float16(0)

            for kv_block in T.unroll(num_kv_blocks):
                T.copy(
                    K_hbm[0, kv_block * rows : (kv_block + 1) * rows, by, 0:hlen],
                    K_sh,
                )
                T.copy(
                    V_hbm[0, kv_block * rows : (kv_block + 1) * rows, by, 0:hlen],
                    V_sh,
                )

                with T.attr(0, KIND, "btmm"):
                    T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)

                T.gemm(S_loc, V_sh, PV_loc)

                for col in T.Parallel(hlen):
                    O_loc[0, col] = O_loc[0, col] + PV_loc[0, col]

            T.copy(O_loc, O_cache[by, 0])

    lowered = flash_decode_min_gemm_only
    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
        "NUM_KV_BLOCKS": num_kv_blocks,
        "CACHE_NUM_MLEN_ROWS": (head_count * hlen) // MLEN,
    }
    return lowered, constants


__all__ = ["make_flash_decode_min_gemm_only"]
