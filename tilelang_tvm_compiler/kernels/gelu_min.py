"""GELU-min kernel — exercises FP scalar compound-store decomposition.

GELU (tanh approximation):

    GELU(x) = 0.5 * x * (1 + tanh(u))
    u       = sqrt(2/pi) * (x + 0.044715 * x^3)

PLENA has no native tanh. We expand it inline using only exp / reci /
add / sub / mul (all of which are PLENA FP scalar primitives):

    tanh(u) = 1 - 2 / (exp(2u) + 1)

The five scalar constants (0.5, 1.0, 2.0, sqrt(2/pi), 0.044715) are
embedded directly as ``T.float16(...)`` literals; the
``hoist_float_constants`` pre-pass synthesises one 1-slot
``global.fpram`` buffer per unique value, and ``test_helper`` auto-
preloads the values from the buffer-addrs dump.
"""

import math

import tilelang.language as T


def make_gelu_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
    o_head_count: int | None = None,
    o_head_offset: int = 0,
):
    """GELU (tanh approximation).

    ``o_head_count`` / ``o_head_offset`` let GELU write into a
    head-slice of a WIDER output tensor — the single-stream-block chain
    uses this to drop GELU(mlp) into the right half of
    ``concat([attn, mlp])``.
    """
    MLEN = 64
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
    # GELU tanh-approximation constants. Embedded directly as
    # ``T.float16(...)`` in the kernel body; the
    # ``hoist_float_constants`` pre-pass turns each unique value into a
    # 1-slot global.fpram buffer at compile time.
    sqrt_2_over_pi_val = math.sqrt(2.0 / math.pi)

    @T.prim_func
    def gelu_min(
        X_hbm:   T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm:   T.Tensor((batch, seq_len, o_head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh   = T.alloc_shared((rows, hlen), "float16")
            Y_sh   = T.alloc_shared((rows, hlen), "float16")

            # Per-row FPRAM scratch for the input and output of GELU.
            X_FP   = T.alloc_fragment((hlen,), "float16")
            Y_FP   = T.alloc_fragment((hlen,), "float16")

            # The five GELU scalar constants (0.5, 1.0, 2.0, sqrt(2/pi),
            # 0.044715) are inlined as ``T.float16(...)`` below. The
            # ``hoist_float_constants`` pre-pass auto-allocates a 1-slot
            # global.fpram buffer per unique value.

            # Intermediate FPRAM scratch fragments. Allocating them
            # explicitly (instead of letting lower_compound_fp_stores
            # auto-allocate ``__tmp_fp_*``) keeps the chain readable and
            # gives every subop a foldable ``dst[i] = a[i] op b[i]``
            # shape — no FloatImm leaves anywhere in the RHS.
            x3        = T.alloc_fragment((hlen,), "float16")   # x*x*x
            cx3       = T.alloc_fragment((hlen,), "float16")   # 0.044715 * x^3
            inner_raw = T.alloc_fragment((hlen,), "float16")   # x + cx3
            u         = T.alloc_fragment((hlen,), "float16")   # sqrt(2/pi) * inner_raw
            two_u     = T.alloc_fragment((hlen,), "float16")   # 2 * u
            e2u       = T.alloc_fragment((hlen,), "float16")   # exp(2u)
            denom     = T.alloc_fragment((hlen,), "float16")   # exp(2u) + 1
            reci_d    = T.alloc_fragment((hlen,), "float16")   # 1 / denom
            two_recid = T.alloc_fragment((hlen,), "float16")   # 2 * reci_d
            tanh_u    = T.alloc_fragment((hlen,), "float16")   # 1 - two_recid
            one_p     = T.alloc_fragment((hlen,), "float16")   # 1 + tanh_u
            hx        = T.alloc_fragment((hlen,), "float16")   # 0.5 * x

            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )

            for row in T.serial(rows):
                T.copy(X_sh[row, 0], X_FP)

                for i in T.unroll(hlen):
                    # u = sqrt(2/pi) * (x + 0.044715 * x^3)
                    x3[i]        = X_FP[i] * X_FP[i] * X_FP[i]
                    cx3[i]       = T.float16(0.044715) * x3[i]
                    inner_raw[i] = X_FP[i] + cx3[i]
                    u[i]         = T.float16(sqrt_2_over_pi_val) * inner_raw[i]

                    # tanh(u) = 1 - 2 * (1 / (exp(2u) + 1))
                    two_u[i]     = T.float16(2.0) * u[i]
                    e2u[i]       = T.exp(two_u[i])
                    denom[i]     = e2u[i] + T.float16(1.0)
                    # ``1.0 / x`` is the only div form fold recognises
                    # (it picks the FloatImm-1 literal numerator and
                    # lowers to ``fp_reci_at``). A ``BufferLoad / BufferLoad``
                    # would fall through to fold's binop arm and fail.
                    reci_d[i]    = T.float16(1.0) / denom[i]
                    two_recid[i] = T.float16(2.0) * reci_d[i]
                    tanh_u[i]    = T.float16(1.0) - two_recid[i]

                    # GELU(x) = 0.5 * x * (1 + tanh(u))
                    one_p[i]     = T.float16(1.0) + tanh_u[i]
                    hx[i]        = T.float16(0.5) * X_FP[i]
                    Y_FP[i]      = hx[i] * one_p[i]

                T.copy(Y_FP, Y_sh[row, 0])

            # Destination head shifted by o_head_offset so GELU's output
            # can land in a head-slice of a wider tensor (concat).
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows,
                      by + o_head_offset, 0:hlen],
            )

    lowered = gelu_min

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


__all__ = ["make_gelu_min"]
