"""SiLU-min kernel (sigmoid linear unit) on the FP scalar pipeline.

    SiLU(x) = x * sigmoid(x),  sigmoid(x) = 1 / (1 + exp(-x))

No native sigmoid; composed from exp/add/reci/mul. ``exp(-x)`` is written as
``exp(NEG_ONE * x)`` (no unary negate in the FP scalar set). The chain is
inline; ``lower_compound_fp_stores`` decomposes the nested RHS and
``hoist_float_constants`` preloads each ``T.float16(...)`` literal into a
1-slot global.fpram buffer. The ``1.0 / x`` reci form is required.
"""

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes


def make_silu_min(
    *,
    rows: int | None = None,
    hlen: int | None = None,
    head_count: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
):
    _hw = _load_sizes()
    MLEN = _hw.mlen
    hlen = hlen if hlen is not None else _hw.hlen
    rows = rows if rows is not None else MLEN
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

            T.copy(X_hbm[0, s_block * rows:(s_block + 1) * rows, by, 0:hlen], X_sh)
            for row in T.serial(rows):
                for i in T.unroll(hlen):
                    sig = T.float16(1.0) / (
                        T.float16(1.0) + T.exp(T.float16(-1.0) * X_sh[row, i])
                    )
                    Y_sh[row, i] = X_sh[row, i] * sig
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows:(s_block + 1) * rows, by, 0:hlen],
            )

    constants = {
        "ROWS": rows, "MLEN": MLEN, "HLEN": hlen, "HEAD_COUNT": head_count,
        "BATCH": batch, "NUM_S_BLOCKS": num_s_blocks,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
    }
    return silu_min, constants


__all__ = ["make_silu_min"]
