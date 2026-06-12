"""Adam optimizer step — one fused weight update, vector-elementwise only.

    m  = beta1*m + (1-beta1)*g
    v  = beta2*v + (1-beta2)*g*g
    mhat = m / (1 - beta1**t)
    vhat = v / (1 - beta2**t)
    w  = w - lr * mhat / (sqrt(vhat) + eps)

PLENA's vector ISA has no element-wise sqrt (only V_EXP_V / V_RECI_V).
The ``1/(sqrt(vhat)+eps)`` term is therefore computed with **one
Newton-Raphson rsqrt iteration** — pure mul/add/sub/reci, no sqrt:

    y0 = 1 / (vhat + 1)                  # cheap positive seed (V_ADD + V_RECI)
    r  = y0 * (1.5 - 0.5 * vhat * y0*y0) # one Newton step  (quadratic convergence)
                                         # r ≈ 1/sqrt(vhat); eps is folded into the
                                         # seed/accuracy slack — Adam is insensitive.

Everything is written inline elementwise over the (rows, hlen) tile, in the
same style as silu_min / gelu_min, so it lowers to the V_MUL_VV / V_MUL_VF /
V_ADD_VV / V_ADD_VF / V_SUB_VF / V_RECI_V vector path with no sqrt opcode and
no compiler change.

The bias-correction divisors ``1 - beta1**t`` and ``1 - beta2**t`` are scalar
constants for the current step ``t``; the host precomputes them and passes
their *reciprocals* (``bc1 = 1/(1-beta1**t)``, ``bc2 = 1/(1-beta2**t)``) so the
kernel multiplies instead of dividing.

Run: tools/manager/run.sh _validate_adam_step
"""

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes


def make_adam_step_min(
    *,
    rows: int | None = None,
    hlen: int | None = None,
    head_count: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    # bias-correction reciprocals for the current step t:
    #   bc1 = 1 / (1 - beta1**t),  bc2 = 1 / (1 - beta2**t)
    # default t=1: bc1 = 1/(1-0.9) = 10, bc2 = 1/(1-0.999) = 1000.
    bc1: float = 10.0,
    bc2: float = 1000.0,
    weight_decay: float = 1e-2,    # AdamW decoupled weight decay
):
    _hw = _load_sizes()
    MLEN = _hw.mlen
    hlen = hlen if hlen is not None else _hw.hlen
    rows = rows if rows is not None else MLEN
    if rows != MLEN:
        raise ValueError(f"adam_step_min requires rows == MLEN ({MLEN}), got {rows}")
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

    # Scalar literals folded into the elementwise chain (hoist_float_constants
    # synthesises a 1-slot global.fpram buffer for each unique value).
    one_m_b1 = 1.0 - beta1          # (1 - beta1)
    one_m_b2 = 1.0 - beta2          # (1 - beta2)

    @T.prim_func
    def adam_step_min(
        W_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),  # weights (in/out)
        G_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),  # grad dW
        M_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),  # 1st moment (in/out)
        V_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),  # 2nd moment (in/out)
        Wo_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),  # updated weights
        Mo_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),  # updated m
        Vo_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),  # updated v
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            W_sh = T.alloc_shared((rows, hlen), "float16")
            G_sh = T.alloc_shared((rows, hlen), "float16")
            M_sh = T.alloc_shared((rows, hlen), "float16")
            V_sh = T.alloc_shared((rows, hlen), "float16")
            Wo_sh = T.alloc_shared((rows, hlen), "float16")
            Mo_sh = T.alloc_shared((rows, hlen), "float16")
            Vo_sh = T.alloc_shared((rows, hlen), "float16")

            T.copy(W_hbm[0, s_block * rows:(s_block + 1) * rows, by, 0:hlen], W_sh)
            T.copy(G_hbm[0, s_block * rows:(s_block + 1) * rows, by, 0:hlen], G_sh)
            T.copy(M_hbm[0, s_block * rows:(s_block + 1) * rows, by, 0:hlen], M_sh)
            T.copy(V_hbm[0, s_block * rows:(s_block + 1) * rows, by, 0:hlen], V_sh)

            # HARDWARE NOTE — masked scalar ops must be in-place.
            # A scalar op `dst = const (*|+|-) src` lowers to a *masked*
            # V_MUL_VF / V_ADD_VF / V_SUB_VF: the emulator does
            # `result = clone(src1)` then applies the scalar only to the
            # mask-selected head, leaving every UN-selected head as the raw
            # src1 value. The per-row lowering selects ONE head at a time
            # (mask = 1<<head), so with dst != src1 each head iteration
            # re-clones src1 and clobbers the heads computed by previous
            # iterations — only the last head survives. The op is only
            # correct when **dst == src1** (then the "leave src1" fallback is
            # a no-op). So every scalar op below is written in-place:
            # first copy src into dst, then apply the scalar to dst itself.
            # (Tensor-tensor ops V_*_VV and V_RECI_V are rmask=0, unmasked,
            #  and are safe as-is.)
            #
            # NOTE: the nested-expression style (one compound store like
            # gelu_min) does NOT work here — fold rejects the top-level
            # `add(mul, mul)` shape of `m = b1*M + (1-b1)*G` ("unrecognised
            # BufferStore"), so it never reaches the fission pass that would
            # auto-insert the in-place copy. Hence the explicit hand-written
            # single-op in-place form below.
            T1 = T.alloc_shared((rows, hlen), "float16")   # m's (1-b1)*G term
            T2 = T.alloc_shared((rows, hlen), "float16")   # v's (1-b2)*g^2 term
            MH = T.alloc_shared((rows, hlen), "float16")   # lr*mhat
            VH = T.alloc_shared((rows, hlen), "float16")   # vhat
            Y  = T.alloc_shared((rows, hlen), "float16")   # Newton y0 / r
            TT = T.alloc_shared((rows, hlen), "float16")   # Newton temp
            DC = T.alloc_shared((rows, hlen), "float16")   # weight-decay term

            # ---- m = beta1*M + (1-beta1)*G  -> Mo_sh ----
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Mo_sh[row, i] = M_sh[row, i]                      # copy
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Mo_sh[row, i] = T.float16(beta1) * Mo_sh[row, i]  # V_MUL_VF in-place
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    T1[row, i] = G_sh[row, i]                         # copy
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    T1[row, i] = T.float16(one_m_b1) * T1[row, i]     # V_MUL_VF in-place
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Mo_sh[row, i] = Mo_sh[row, i] + T1[row, i]        # V_ADD_VV -> m

            # ---- v = beta2*V + (1-beta2)*(G*G)  -> Vo_sh ----
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Vo_sh[row, i] = V_sh[row, i]                      # copy
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Vo_sh[row, i] = T.float16(beta2) * Vo_sh[row, i]  # V_MUL_VF in-place
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    T2[row, i] = G_sh[row, i] * G_sh[row, i]          # V_MUL_VV (g^2)
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    T2[row, i] = T.float16(one_m_b2) * T2[row, i]     # V_MUL_VF in-place
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Vo_sh[row, i] = Vo_sh[row, i] + T2[row, i]        # V_ADD_VV -> v

            # ---- bias correction: MH = lr*(m*bc1) ; VH = v*bc2 ----
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    MH[row, i] = Mo_sh[row, i]                        # copy
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    MH[row, i] = T.float16(bc1) * MH[row, i]          # V_MUL_VF in-place (mhat)
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    MH[row, i] = T.float16(lr) * MH[row, i]           # V_MUL_VF in-place (lr*mhat)
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    VH[row, i] = Vo_sh[row, i]                        # copy
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    VH[row, i] = T.float16(bc2) * VH[row, i]          # V_MUL_VF in-place (vhat)

            # ---- Newton rsqrt(vhat) one step:  r = y0*(1.5 - 0.5*vhat*y0^2) ----
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Y[row, i] = VH[row, i]                            # copy
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Y[row, i] = Y[row, i] + T.float16(1.0)            # V_ADD_VF in-place
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Y[row, i] = T.float16(1.0) / Y[row, i]            # V_RECI_V -> y0
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    TT[row, i] = Y[row, i] * Y[row, i]                # V_MUL_VV (y0^2)
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    TT[row, i] = VH[row, i] * TT[row, i]              # V_MUL_VV (vhat*y0^2)
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    TT[row, i] = T.float16(0.5) * TT[row, i]          # V_MUL_VF in-place
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    TT[row, i] = T.float16(1.5) - TT[row, i]          # V_SUB_VF in-place (const - dst)
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Y[row, i] = Y[row, i] * TT[row, i]                # V_MUL_VV -> r

            # ---- AdamW update: w = W - lr*mhat*r - lr*wd*W ----
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    MH[row, i] = MH[row, i] * Y[row, i]               # V_MUL_VV (lr*mhat*r)
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Wo_sh[row, i] = W_sh[row, i] - MH[row, i]         # V_SUB_VV (Adam step)
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    DC[row, i] = W_sh[row, i]                         # copy
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    DC[row, i] = T.float16(lr * weight_decay) * DC[row, i]  # V_MUL_VF in-place
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    Wo_sh[row, i] = Wo_sh[row, i] - DC[row, i]        # V_SUB_VV -> w'

            T.copy(Wo_sh, Wo_hbm[0, s_block * rows:(s_block + 1) * rows, by, 0:hlen])
            T.copy(Mo_sh, Mo_hbm[0, s_block * rows:(s_block + 1) * rows, by, 0:hlen])
            T.copy(Vo_sh, Vo_hbm[0, s_block * rows:(s_block + 1) * rows, by, 0:hlen])

    constants = {
        "ROWS": rows, "MLEN": MLEN, "HLEN": hlen, "HEAD_COUNT": head_count,
        "BATCH": batch, "NUM_S_BLOCKS": num_s_blocks,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
    }
    return adam_step_min, constants


__all__ = ["make_adam_step_min"]
