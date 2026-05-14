"""SiLU-min kernel — sigmoid linear unit on FP scalar pipeline.

    SiLU(x)    = x * sigmoid(x)
    sigmoid(x) = 1 / (1 + exp(-x))

PLENA has no sigmoid ISA; the kernel composes it from
``exp / add / reci / mul``. Only two scalar constants are needed —
``1.0`` (for the reci numerator and the denominator add) and ``-1.0``
(to express ``exp(-x)`` as ``exp(NEG_ONE * x)``, since there is no
unary negate in the FP scalar set). Both come from preloaded rank-1
``local.fragment`` slots, mirroring gelu_min and flash_attention_min.

Layout: HBM -> VRAM (shared) -> per-row FPRAM scratch -> VRAM -> HBM.
"""

import tilelang.language as T


def make_silu_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"silu_min requires rows == MLEN ({MLEN}), got {rows}")
    if MLEN % hlen != 0:
        raise ValueError(f"hlen must divide MLEN ({MLEN}); got hlen={hlen}")
    hardware_lane_count = MLEN // hlen
    if head_count % hardware_lane_count != 0:
        raise ValueError(
            f"head_count must be a multiple of MLEN/hlen={hardware_lane_count}; "
            f"got {head_count}"
        )
    if num_s_blocks < 1:
        raise ValueError(f"num_s_blocks must be >= 1, got {num_s_blocks}")

    seq_len = num_s_blocks * rows

    @T.prim_func
    def silu_min(
        X_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh = T.alloc_shared((rows, hlen), "float16")
            Y_sh = T.alloc_shared((rows, hlen), "float16")

            X_FP = T.alloc_fragment((hlen,), "float16")
            Y_FP = T.alloc_fragment((hlen,), "float16")

            ONE     = T.alloc_fragment((hlen,), "float16")   #  1.0
            NEG_ONE = T.alloc_fragment((hlen,), "float16")   # -1.0

            neg_x   = T.alloc_fragment((hlen,), "float16")   # -x
            e_negx  = T.alloc_fragment((hlen,), "float16")   # exp(-x)
            denom   = T.alloc_fragment((hlen,), "float16")   # 1 + exp(-x)
            sig     = T.alloc_fragment((hlen,), "float16")   # sigmoid(x)

            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )

            for row in T.serial(rows):
                T.copy(X_sh[row, 0], X_FP)

                for i in T.unroll(hlen):
                    neg_x[i]  = NEG_ONE[i] * X_FP[i]
                    e_negx[i] = T.exp(neg_x[i])
                    denom[i]  = ONE[i] + e_negx[i]
                    # ``1.0 / x`` literal numerator — fold lowers this
                    # to fp_reci_at. A BufferLoad/BufferLoad div would
                    # not match the reci pattern.
                    sig[i]    = T.float16(1.0) / denom[i]
                    Y_FP[i]   = X_FP[i] * sig[i]

                T.copy(Y_FP, Y_sh[row, 0])

            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
            )

    lowered = silu_min

    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "BATCH": batch,
        "NUM_S_BLOCKS": num_s_blocks,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
    }
    return lowered, constants


__all__ = ["make_silu_min"]
