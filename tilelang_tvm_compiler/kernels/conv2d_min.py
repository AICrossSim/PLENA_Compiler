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

from ..frontend.pipeline import compile_func


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
        # NCHW. ``T.func_attr({"plena.layout": "NCHW"})`` below tells
        # the compiler axes[2] is the row dim (s-tiled) and axes[1] is
        # the channel dim (lane-grouped).
        Input:  T.Tensor((1, C_IN,  H_PAD, W_PAD), "float16"),
        Output: T.Tensor((1, C_OUT, H,     W),     "float16"),
    ):
        T.func_attr({"plena.layout": "NCHW"})
        if False:
            _ = (H_PAD, W_PAD, H, W, C_IN, C_OUT, OC_IC)

        with T.Kernel(1, threads=128) as _bx:
            # ---- VRAM buffers ----
            # No B_cache: weights are pre-loaded *directly* into
            # ``B_FP`` at FPRAM startup (the testbench's ``fp_preload``
            # writes to B_FP's FPRAM address, derived from
            # ``--dump-buffer-addrs``). This avoids the awkward
            # ``T.copy(B_cache[r, 0], B_FP[r * MLEN])`` indirection,
            # which silently drops its body during lowering — tilelang
            # treats ``B_FP[r * MLEN]`` as a scalar access (not a
            # region slice) and produces an empty for-loop, so B_FP
            # never gets populated and every FMA multiplies by zero.

            # Whole padded input staged in VRAM. Multi-tile h2v emitter
            # walks the (C_IN, S_TILES, D_TILES) inner-tile grid and
            # fires one H_LOAD_V per tile. NCHW layout — axis 2 is the
            # row dim (s-tiled), axis 3 is the col dim (d-tiled), and
            # axis 1 (C_IN) becomes the lane-group dim under canonical
            # BSHD ordering.
            in_stage = T.alloc_shared((1, C_IN, H_PAD, W_PAD), "float16")

            # VRAM scratch — per-tap intermediate. Holds the kw-shifted
            # input row * weight scalar for one (ic, kh, kw) tap.
            A_sh = T.alloc_shared((1, 1, 1, MLEN), "float16")

            # VRAM scratch — per-(oc, oh) accumulator. Reset to zero at
            # the start of each output row, then receives all
            # C_IN * KH * KW vector-scalar contributions before
            # being copied into ``C_loc``.
            A_sh_acc = T.alloc_shared((1, 1, 1, MLEN), "float16")

            # ---- FPRAM fragments (1D so scope_inference keeps them in fpram) ----
            # ``B_FP`` holds the full weight tensor after MLEN-padding:
            # OC_IC rows of MLEN slots each, indexed as
            # ``B_FP[(oc * C_IN + ic) * MLEN + k_tap]``. Only the first
            # K_FLAT slots in each row are real weights — the rest are
            # zero-padded by the testbench so the row-wise S_MAP_FP_V
            # transfer can move whole MLEN-wide chunks. Marked global.fpram
            # because the testbench's fp_preload writes the weights into
            # FPRAM at this buffer's allocated address before the kernel
            # runs — its layout is the user's contract with the testbench
            # and must not be reshaped by lane-fusion expansion.
            B_FP         = T.alloc_fragment((OC_IC * MLEN,), "float16",
                                             scope="global.fpram")
            in_FP_aux    = T.alloc_fragment((MLEN,), "float16")
            in_FP_padded = T.alloc_fragment((MLEN + KW - 1,), "float16")
            shift_FP     = T.alloc_fragment((MLEN,), "float16")

            # Final output (1, C_OUT, MLEN, MLEN). With NCHW layout
            # the channel dim becomes the lane-group axis (canonical H)
            # — for C_OUT > 1 the buffer needs multi-tile placement.
            # Stage_output's writeback path works for C_OUT == 1; the
            # multi-C_OUT case is gated until __main__._emit_output_staging
            # learns the per-channel stride.
            C_loc = T.alloc_shared((1, C_OUT, MLEN, MLEN), "float16")

            # ---- Stage whole padded input HBM->VRAM (multi-tile DMA) ----
            T.copy(Input[0, 0, 0, 0], in_stage)

            # ---- Weights live in FPRAM from the start ----
            # ``B_FP`` is preloaded by the testbench (fp_preload writes
            # the weight tensor into FPRAM at B_FP's allocated address).
            # No kernel-side staging needed.

            # ---- One-time init of in_FP_padded's zero tail ----
            for k in T.serial(KW - 1):
                in_FP_padded[MLEN + k] = T.float16(0)

            for oc in T.serial(C_OUT):
                for oh in T.serial(H):
                    # ---- Zero per-row accumulator ----
                    for m in T.Parallel(MLEN):
                        A_sh_acc[0, 0, 0, m] = T.float16(0)

                    # ---- C_IN × KH × KW vector-scalar FMA chain ----
                    for ic in T.serial(C_IN):
                        for kh_idx in T.unroll(KH):
                            # Load input row from input channel ic.
                            # NCHW indexing: row at axis 2.
                            T.copy(in_stage[0, ic, oh + kh_idx, 0], in_FP_aux)

                            for i in T.serial(MLEN):
                                in_FP_padded[i] = in_FP_aux[i]

                            for kw_idx in T.unroll(KW):
                                k_tap = kh_idx * KW + kw_idx

                                for m in T.serial(MLEN):
                                    shift_FP[m] = in_FP_padded[m + kw_idx]

                                T.copy(shift_FP, A_sh[0, 0, 0, 0])

                                # B_FP layout: row r = oc*C_IN + ic,
                                # tap k_tap = kh*KW + kw.
                                # Flat index = r * MLEN + k_tap.
                                for m in T.Parallel(MLEN):
                                    A_sh[0, 0, 0, m] = (
                                        A_sh[0, 0, 0, m]
                                        * B_FP[(oc * C_IN + ic) * MLEN + k_tap]
                                    )
                                for m in T.Parallel(MLEN):
                                    A_sh_acc[0, 0, 0, m] = (
                                        A_sh_acc[0, 0, 0, m] + A_sh[0, 0, 0, m]
                                    )

                    # ---- Per-(oc, oh) writeback into C_loc ----
                    # NCHW indexing: oc at axis 1, oh at axis 2.
                    T.copy(A_sh_acc, C_loc[0, oc, oh, 0])

            # ---- Writeback ALL output rows in one full-tile DMA ----
            T.copy(C_loc, Output[0, 0, 0, 0])

    lowered = compile_func(conv2d_min)

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
