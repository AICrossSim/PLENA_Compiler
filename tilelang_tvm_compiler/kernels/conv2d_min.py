"""Conv2D-min — 2D convolution with optional multi-channel support.

Shapes (NCHW):

    Input  :  (1, C_IN,  H_PAD, W_PAD)   pre-padded right/bottom for the kernel
    Weight :  (C_OUT, C_IN, KH, KW)      flattened to FPRAM in (oc, ic, kh*KW+kw) order
    Output :  (1, C_OUT, H, W)           same spatial as logical input

For the simplest case (C_IN=C_OUT=1), this reduces to the original
single-channel kernel.

Why no GEMM here:

  The natural per-output-row work is a (MLEN_w, C_IN*KH*KW) gather ×
  (C_IN*KH*KW,) weight = (MLEN_w,) row — a GEMV (matrix × vector).
  PLENA has no GEMV instruction; the smallest matmul tile is
  (MLEN, MLEN) × (MLEN, MLEN), which would be very sparse for typical
  conv shapes (especially when C_IN*KH*KW < MLEN). Instead we lower
  the whole thing to vector-scalar FMAs:

      for c_in: for kh: for kw:                # C_IN*KH*KW unrolled
          for m in T.Parallel(MLEN):           # one HW vector op
              C_loc[m] += in_shifted[m] * weight[oc, c_in, kh, kw]

  Each iter is one ``plena.v_mul + plena.v_add`` (or fused FMA) on a
  64-wide vector.

Construction (per (oc, oh) output row):

  1. **kw-shift via FPRAM padded fragment**: per (ic, kh), copy one
     MLEN-wide input row into ``in_FP_padded`` (size MLEN+KW-1, last
     KW-1 slots zero-init). For each kw, read shifted slice
     ``in_FP_padded[m + kw_idx]`` into shift_FP, then back to VRAM
     A_sh.

  2. **Vector-scalar FMA**: ``A_sh *= B_FP[oc, ic, kh, kw]``, then
     ``A_sh_acc += A_sh``.

  3. **Writeback**: one ``T.copy`` writes A_sh_acc into
     ``C_loc[0, oc, oh, :]``. After all (oc, oh) iterations, one
     ``T.copy(C_loc, Output[0, 0, 0, 0])`` dumps everything to HBM.

Constraints:
  * KH * KW == HLEN (= 16)
  * W == MLEN (single output W tile per row)
  * C_OUT * C_IN * MLEN must fit in FPRAM (B_FP holds the entire
    weight tensor MLEN-padded). For C_OUT=4, C_IN=4: 1024 elements.
  * Output staging in __main__._emit_output_staging currently only
    handles C_OUT == 1 cleanly (it walks logical 2D as rows × cols
    with constant stride; for C_OUT > 1 NCHW the cross-channel stride
    differs from the cross-row stride). Multi-C_OUT staging is
    queued for a follow-up.

Layout contract for ``B_cache`` (testbench-side preload):
    B_cache[oc * C_IN + ic, k_tap] = Weight[oc, ic, kh, kw]
    where k_tap = kh*KW + kw (same row-major weight ordering the
    original single-channel kernel used; just extended by the
    (oc, ic) outer pair).
"""

import tilelang.language as T


def make_conv2d_min(
    *,
    h_in: int = 64,
    w_in: int = 64,
    kh: int = 4,
    kw: int = 4,
    c_in: int = 1,
    c_out: int = 1,
):
    MLEN = 64
    HLEN = 16

    if kh * kw != HLEN:
        raise ValueError(
            f"first-cut conv2d_min requires kh*kw == HLEN ({HLEN}); "
            f"got kh={kh}, kw={kw}, kh*kw={kh*kw}"
        )
    if w_in != MLEN:
        raise ValueError(
            f"first-cut conv2d_min requires w_in == MLEN ({MLEN}); got w_in={w_in}"
        )
    if h_in <= 0:
        raise ValueError(f"h_in must be positive; got h_in={h_in}")
    if c_in <= 0 or c_out <= 0:
        raise ValueError(f"c_in and c_out must be positive; got c_in={c_in}, c_out={c_out}")

    H = h_in
    W = w_in
    KH = kh
    KW = kw
    K_FLAT = KH * KW          # = HLEN, the unrolled-1D tap count
    C_IN = c_in
    C_OUT = c_out
    OC_IC = C_OUT * C_IN       # number of (oc, ic) weight rows in B_cache

    def _round_up_to_mlen(x: int) -> int:
        return (x + MLEN - 1) // MLEN * MLEN
    H_PAD = _round_up_to_mlen(H + KH - 1)
    W_PAD = _round_up_to_mlen(W + KW - 1)

    @T.prim_func
    def conv2d_min(
        Input:  T.Tensor((1, C_IN,  H_PAD, W_PAD), "float16"),
        Output: T.Tensor((1, C_OUT, H,     W),     "float16"),
    ):
        T.func_attr({"plena.layout": "NCHW"})
        if False:
            _ = (H_PAD, W_PAD, H, W, C_IN, C_OUT, OC_IC)

        with T.Kernel(1, threads=1) as _bx:
            # Single-channel padded input tile, re-staged per ic.
            in_stage = T.alloc_shared((H_PAD, W_PAD), "float16")

            # ``A_sh`` and ``A_sh_acc`` live in VRAM so the per-tap
            # multiply + accumulate lower to vector instructions
            # (``V_MUL_VF`` / ``V_ADD_VV``) instead of the per-element
            # FPRAM scalar loop. The kw-shift chain stays in FPRAM —
            # only the multiply and the accumulate move to vram.
            #
            # Shape ``(1, MLEN)`` (not ``(MLEN,)``) on purpose: fold's
            # broadcast detection only kicks in when ``len(src.indices)
            # < len(dst.indices)``, so a 1D shared dst with a scalar
            # fp src ``w_aux[0]`` fails to fold (same rank, no
            # broadcast path). 2D dst + 1D scalar src matches the
            # path flash_attention already uses.
            A_sh     = T.alloc_shared((1, MLEN), "float16")
            A_sh_acc = T.alloc_shared((1, MLEN), "float16")

            # ``B_FP`` holds the full weight tensor after MLEN-padding:
            # OC_IC rows of MLEN slots each. The testbench's
            # ``fp_preload`` writes weights into FPRAM at this buffer's
            # allocated address before the kernel runs.
            B_FP         = T.alloc_fragment((OC_IC * MLEN,), "float16",
                                             scope="global.fpram")
            # Per-tap weight scalar — 1D so the FPRAM-scalar fold path
            # accepts the multiply (`A_sh[m] = A_sh[m] * w_aux[0]`).
            w_aux        = T.alloc_fragment((1,), "float16")
            in_FP_aux    = T.alloc_fragment((MLEN,), "float16")
            in_FP_padded = T.alloc_fragment((MLEN + KW - 1,), "float16")
            shift_FP     = T.alloc_fragment((MLEN,), "float16")

            # Single-channel output tile, drained to HBM per oc.
            C_loc = T.alloc_shared((MLEN, MLEN), "float16")

            for k in T.serial(KW - 1):
                in_FP_padded[MLEN + k] = T.float16(0)

            for oc in T.serial(C_OUT):
                for oh in T.serial(H):
                    for m in T.Parallel(MLEN):
                        A_sh_acc[0, m] = T.float16(0)

                    for ic in T.serial(C_IN):
                        # (Re-)stage this input channel's full padded
                        # tile into VRAM. Inner kh_idx reads from it
                        # row-wise.
                        T.copy(
                            Input[0, ic, 0:H_PAD, 0:W_PAD],
                            in_stage[0:H_PAD, 0:W_PAD],
                        )
                        for kh_idx in T.unroll(KH):
                            T.copy(
                                in_stage[oh + kh_idx, 0:MLEN],
                                in_FP_aux[0:MLEN],
                            )

                            for i in T.serial(MLEN):
                                in_FP_padded[i] = in_FP_aux[i]

                            for kw_idx in T.unroll(KW):
                                k_tap = kh_idx * KW + kw_idx

                                for m in T.serial(MLEN):
                                    shift_FP[m] = in_FP_padded[m + kw_idx]

                                # fpram → vram row copy: lowers to a
                                # single ``S_MAP_V_FP`` (whole MLEN row
                                # in one issue) instead of MLEN scalar
                                # loads + stores.
                                T.copy(shift_FP[0:MLEN], A_sh[0, 0:MLEN])

                                w_aux[0] = B_FP[(oc * C_IN + ic) * MLEN + k_tap]

                                # vram × fp_scalar broadcast → one
                                # ``V_MUL_VF`` per row; with A_sh being
                                # 1×MLEN that's a single instruction.
                                for m in T.Parallel(MLEN):
                                    A_sh[0, m] = A_sh[0, m] * w_aux[0]
                                # vram + vram → one ``V_ADD_VV``.
                                for m in T.Parallel(MLEN):
                                    A_sh_acc[0, m] = A_sh_acc[0, m] + A_sh[0, m]

                    T.copy(
                        A_sh_acc[0, 0:MLEN],
                        C_loc[oh, 0:MLEN],
                    )

                # Drain this oc's full (MLEN, MLEN) tile to HBM.
                T.copy(
                    C_loc[0:MLEN, 0:MLEN],
                    Output[0, oc, 0:MLEN, 0:MLEN],
                )

    # Return the raw PrimFunc. ``compile_kernel`` runs stmt prep + the
    # mid_ir pipeline itself, so factories no longer need to call into
    # the legacy compile_func.
    lowered = conv2d_min

    constants = {
        "H": H, "W": W,
        "H_PAD": H_PAD, "W_PAD": W_PAD,
        "KH": KH, "KW": KW,
        "K_FLAT": K_FLAT,
        "MLEN": MLEN, "HLEN": HLEN,
        "C_IN": C_IN, "C_OUT": C_OUT,
    }
    return lowered, constants


__all__ = ["make_conv2d_min"]
