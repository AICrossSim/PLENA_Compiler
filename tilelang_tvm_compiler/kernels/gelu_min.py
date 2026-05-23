"""GELU-min kernel (tanh approximation) on the FP scalar pipeline.

    GELU(x) = 0.5 * x * (1 + tanh(u)),  u = sqrt(2/pi) * (x + 0.044715 * x^3)
    tanh(u) = 1 - 2 / (exp(2u) + 1)   (no native tanh; expand via exp/reci)

The whole chain is written inline; ``lower_compound_fp_stores`` decomposes
the nested RHS into single-op ``__tmp_fp_*`` stores, and
``hoist_float_constants`` turns each unique ``T.float16(...)`` literal into a
1-slot global.fpram buffer. The ``1.0 / x`` reci form is required (only div
shape fold lowers to ``fp_reci_at``).
"""

import math

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes


def make_gelu_min(
    *,
    rows: int | None = None,
    hlen: int | None = None,
    head_count: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
    o_head_count: int | None = None,
    o_head_offset: int = 0,
):
    """GELU (tanh approximation). ``o_head_count``/``o_head_offset`` write
    into a head-slice of a wider output (the single-stream-block chain drops
    GELU(mlp) into the right half of ``concat([attn, mlp])``)."""
    _hw = _load_sizes()
    MLEN = _hw.mlen
    hlen = hlen if hlen is not None else _hw.hlen
    rows = rows if rows is not None else MLEN
    if rows != MLEN:
        raise ValueError(f"gelu_min requires rows == MLEN ({MLEN}), got {rows}")
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
    if o_head_count is None:
        o_head_count = head_count
    if o_head_count < head_count:
        raise ValueError(
            f"o_head_count ({o_head_count}) must be >= head_count ({head_count})"
        )
    if not (0 <= o_head_offset <= o_head_count - head_count):
        raise ValueError(
            f"o_head_offset ({o_head_offset}) + head_count ({head_count}) "
            f"must fit within o_head_count ({o_head_count})"
        )

    seq_len = num_s_blocks * rows
    SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)

    @T.prim_func
    def gelu_min(
        X_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm: T.Tensor((batch, seq_len, o_head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh = T.alloc_shared((rows, hlen), "float16")
            Y_sh = T.alloc_shared((rows, hlen), "float16")

            T.copy(X_hbm[0, s_block * rows:(s_block + 1) * rows, by, 0:hlen], X_sh)
            for row in T.serial(rows):
                for i in T.parallel(hlen):
                    u = T.float16(SQRT_2_OVER_PI) * (
                        X_sh[row, i]
                        + T.float16(0.044715) * (X_sh[row, i] * X_sh[row, i] * X_sh[row, i])
                    )
                    tanh_u = T.float16(1.0) - T.float16(2.0) * (
                        T.float16(1.0) / (T.exp(T.float16(2.0) * u) + T.float16(1.0))
                    )
                    Y_sh[row, i] = (T.float16(0.5) * X_sh[row, i]) * (T.float16(1.0) + tanh_u)
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows:(s_block + 1) * rows, by + o_head_offset, 0:hlen],
            )

    constants = {
        "ROWS": rows, "MLEN": MLEN, "HLEN": hlen, "HEAD_COUNT": head_count,
        "BATCH": batch, "NUM_S_BLOCKS": num_s_blocks,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
    }
    return gelu_min, constants


__all__ = ["make_gelu_min"]
