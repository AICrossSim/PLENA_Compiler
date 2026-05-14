"""GELU-min kernel — exercises FP scalar compound-store decomposition.

GELU (tanh approximation):

    GELU(x) = 0.5 * x * (1 + tanh(u))
    u       = sqrt(2/pi) * (x + 0.044715 * x^3)

PLENA has no native tanh. We expand it inline using only exp / reci /
add / sub / mul (all of which are PLENA FP scalar primitives):

    tanh(u) = 1 - 2 / (exp(2u) + 1)

The five scalar constants (0.5, 1.0, 2.0, sqrt(2/pi), 0.044715) cannot
appear as literals in the FP scalar pipeline — there is no FP load-imm
ISA, and ``lower_fp_row_patterns`` rejects any BufferStore RHS that
contains a non-zero ``FloatImm``. So each is declared as a rank-1
``local.fragment`` that PLENA auto-routes to FPRAM. The testbench
preloads every slot with the constant value before the kernel runs,
mirroring how flash_attention_min preloads ``SCALE`` / ``M_INIT`` /
``L_INIT``.

Layout: HBM -> VRAM (shared) -> per-row FPRAM scratch -> VRAM -> HBM.
``hlen`` (== FPRAM fragment length) is intentionally small so the
fragments fit in FPRAM and rank-1 fragments stay on the FP scalar path.
"""

import tilelang.language as T


def make_gelu_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
):
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

    seq_len = num_s_blocks * rows

    @T.prim_func
    def gelu_min(
        X_hbm:   T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm:   T.Tensor((batch, seq_len, head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh   = T.alloc_shared((rows, hlen), "float16")
            Y_sh   = T.alloc_shared((rows, hlen), "float16")

            # Per-row FPRAM scratch for the input and output of GELU.
            X_FP   = T.alloc_fragment((hlen,), "float16")
            Y_FP   = T.alloc_fragment((hlen,), "float16")

            # FP scalar constants (testbench preloads each slot with the
            # named value). Each is a rank-1 fragment → FPRAM scalar slot.
            HALF      = T.alloc_fragment((hlen,), "float16")   # 0.5
            ONE       = T.alloc_fragment((hlen,), "float16")   # 1.0
            TWO       = T.alloc_fragment((hlen,), "float16")   # 2.0
            SQRT_2_PI = T.alloc_fragment((hlen,), "float16")   # sqrt(2/pi)
            COEFF     = T.alloc_fragment((hlen,), "float16")   # 0.044715

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
                    cx3[i]       = COEFF[i] * x3[i]
                    inner_raw[i] = X_FP[i] + cx3[i]
                    u[i]         = SQRT_2_PI[i] * inner_raw[i]

                    # tanh(u) = 1 - 2 * (1 / (exp(2u) + 1))
                    two_u[i]     = TWO[i] * u[i]
                    e2u[i]       = T.exp(two_u[i])
                    denom[i]     = e2u[i] + ONE[i]
                    # ``1.0 / x`` is the only div form fold recognises
                    # (it picks the FloatImm-1 literal numerator and
                    # lowers to ``fp_reci_at``). A ``BufferLoad / BufferLoad``
                    # would fall through to fold's binop arm and fail.
                    reci_d[i]    = T.float16(1.0) / denom[i]
                    two_recid[i] = TWO[i] * reci_d[i]
                    tanh_u[i]    = ONE[i] - two_recid[i]

                    # GELU(x) = 0.5 * x * (1 + tanh(u))
                    one_p[i]     = ONE[i] + tanh_u[i]
                    hx[i]        = HALF[i] * X_FP[i]
                    Y_FP[i]      = hx[i] * one_p[i]

                T.copy(Y_FP, Y_sh[row, 0])

            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
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
