"""Tiled NHWC Conv2D — written in tilelang style.

Standard 2D convolution (stride=1, padding=0, dilation=1):

    Output[n, oh, ow, oc] = sum_{kh, kw, ic}
        Input[n, oh+kh, ow+kw, ic] * Weight[kh, kw, ic, oc]

There is no im2col intrinsic in PLENA's ISA, so we don't flatten the
spatial sum into a single big GEMM. Instead, for each (kh, kw) we treat
the contribution as a 1x1 "conv" whose lhs is a shifted view of the
input feature map. That makes each (kh, kw) contribution a clean
``T.copy`` of an MLEN-wide W slice — same trailing-dim contract that
``mm64`` uses for its 64x64 lhs.

GEMM dimensions (per micro-step):
    M = OW   (one output row, tiled into MLEN chunks along the W axis)
    K = C_in
    N = C_out

Tilelang-DSL parts:
  * ``T.Kernel(OH, NUM_OC, threads=128) as (oh, oc_block)`` — grid axes.
    Each grid block produces one (M=ow_block * MLEN, N=MLEN) output tile;
    the K reduction over (kh, kw, ic_block) lives inside.
  * ``T.copy`` for HBM<->VRAM/MRAM transfers (4D source point + 2D shared
    shape; trailing dims auto-extent to MLEN x MLEN).
  * ``T.gemm`` with the default kind (overwrite) -> ``plena.matmul``.
  * Inline ``T.serial(MLEN) + T.Parallel(MLEN)`` zero-init / add-into
    pairs that ``fuse_elementwise`` folds to ``plena.zero_v`` /
    ``plena.v_add``. This is the same pattern flash_attention uses for
    its O accumulator (see flash_attention_min.py: zero O_loc; O_loc +=
    PV_loc) — until the reserved ``KIND="add"`` gemm path lands, this
    is the documented way to express ``C += A @ B`` (see
    frontend/gemm_macros.py docstring).

Constraints:
    * stride = 1, padding = 0, dilation = 1   (extend later)
    * OW    % MLEN == 0
    * C_in  % MLEN == 0
    * C_out % MLEN == 0

Shapes:
    Input:  (N, H_in, W_in, C_in)        NHWC
    Weight: (KH, KW, C_in, C_out)        HWIO
    Output: (N, OH,   OW,   C_out)
where OH = H_in - KH + 1,  OW = W_in - KW + 1.
"""

from __future__ import annotations

import tilelang.language as T

from ..frontend import compile_func


def make_tiled_conv2d(
    *,
    batch: int = 1,
    h_in: int = 6,
    w_in: int = 66,        # OW = w_in - kw + 1 = 64 = MLEN
    c_in: int = 64,
    c_out: int = 64,
    kh: int = 3,
    kw: int = 3,
):
    MLEN = 64
    if batch != 1:
        raise ValueError(f"tiled_conv2d currently requires batch == 1, got {batch}")
    if kh < 1 or kw < 1:
        raise ValueError(f"kernel size must be positive, got kh={kh}, kw={kw}")

    OH = h_in - kh + 1
    OW = w_in - kw + 1
    if OH <= 0 or OW <= 0:
        raise ValueError(
            f"invalid output spatial size: OH={OH}, OW={OW} "
            f"(h_in={h_in}, w_in={w_in}, kh={kh}, kw={kw})"
        )
    if OW % MLEN:
        raise ValueError(f"OW ({OW}) must be a multiple of MLEN ({MLEN})")
    if c_in % MLEN:
        raise ValueError(f"c_in ({c_in}) must be a multiple of MLEN ({MLEN})")
    if c_out % MLEN:
        raise ValueError(f"c_out ({c_out}) must be a multiple of MLEN ({MLEN})")

    BATCH = batch
    H_IN = h_in
    W_IN = w_in
    C_IN = c_in
    C_OUT = c_out
    KH = kh
    KW = kw
    NUM_OW = OW // MLEN
    NUM_IC = C_IN // MLEN
    NUM_OC = C_OUT // MLEN

    @T.prim_func
    def tiled_conv2d(
        Input:  T.Tensor((BATCH, H_IN, W_IN, C_IN),  "float16"),
        Weight: T.Tensor((KH, KW, C_IN, C_OUT),      "float16"),
        Output: T.Tensor((BATCH, OH, OW, C_OUT),     "float16"),
    ):
        # Force Python to allocate closure cells for the shape-only
        # constants. tilelang's eager builder (builder.py:854) reads
        # `func.__closure__` to populate the type-annotation eval scope,
        # but CPython only creates a cell for a free variable that is
        # actually *referenced* in the function body. Names like BATCH /
        # H_IN / OW that appear only inside `T.Tensor(...)` annotations
        # would NameError at parse time without this dead-code touch.
        # `if False` is constant-folded out of the bytecode but the
        # symbol-table pass still records the reads.
        if False:
            _ = (BATCH, H_IN, W_IN, C_IN, C_OUT, OW)

        # Grid: one block per (oh, oc_block). The remaining spatial-W
        # tiles (NUM_OW), the K reduction (kh, kw, ic_block), and the
        # batch axis are serialized inside.
        with T.Kernel(OH, NUM_OC, threads=128) as (oh, oc_block):
            A_sh      = T.alloc_shared((MLEN, MLEN), "float16")    # M=W, K=C_in
            B_sh      = T.alloc_shared((MLEN, MLEN), "float16")    # K=C_in, N=C_out
            C_partial = T.alloc_fragment((MLEN, MLEN), "float16")  # one micro-GEMM result
            C_loc     = T.alloc_fragment((MLEN, MLEN), "float16")  # running accumulator

            for ow_block in T.serial(NUM_OW):
                # Zero the running accumulator. fuse_elementwise folds
                # this nested (serial, Parallel) zero-store into a single
                # plena.zero_v over the whole C_loc fragment.
                for row in T.serial(MLEN):
                    for col in T.Parallel(MLEN):
                        C_loc[row, col] = T.float16(0)

                # K-reduction across the conv window and input channels.
                # IMPORTANT: khi / kwi are unrolled at Python parse time
                # via plain `range()` (NOT `T.unroll`). lower_to_hlir's
                # _derive_per_dim_extents requires at most one loop var
                # per tensor axis: with khi as a TIR loop var, the H-axis
                # start `oh + khi` would carry two free vars (oh from
                # the grid + khi) and the var-stride check fails. Python-
                # range unrolls produce literal khi values per copy of
                # the body, leaving the H axis as `oh + <const>` and the
                # W axis as `ow_block * MLEN + <const>` — one var each.
                for khi in range(KH):
                    for kwi in range(KW):
                        for ic_block in T.serial(NUM_IC):
                            # Input slice: NHWC point at
                            #   (0, oh + khi, ow_block*MLEN + kwi, ic_block*MLEN)
                            # with trailing extents (MLEN, MLEN) -> A_sh.
                            # Last two dims map to (M=W, K=C_in).
                            T.copy(
                                Input[
                                    0,
                                    oh + khi,
                                    ow_block * MLEN + kwi,
                                    ic_block * MLEN,
                                ],
                                A_sh,
                            )
                            # Weight slice: HWIO point at
                            #   (khi, kwi, ic_block*MLEN, oc_block*MLEN)
                            # with trailing extents (MLEN, MLEN) -> B_sh.
                            # Last two dims map to (K=C_in, N=C_out).
                            T.copy(
                                Weight[
                                    khi,
                                    kwi,
                                    ic_block * MLEN,
                                    oc_block * MLEN,
                                ],
                                B_sh,
                            )
                            # C_partial = A_sh @ B_sh   (overwrite -> plena.matmul)
                            T.gemm(A_sh, B_sh, C_partial)
                            # C_loc += C_partial. fuse_elementwise folds
                            # this into a single plena.v_add over the
                            # whole tile (same idiom as flash_attention's
                            # O += PV, see flash_attention_min.py).
                            for row in T.serial(MLEN):
                                for col in T.Parallel(MLEN):
                                    C_loc[row, col] = (
                                        C_loc[row, col] + C_partial[row, col]
                                    )

                # Writeback: NHWC slice at (0, oh, ow_block*MLEN, oc_block*MLEN).
                T.copy(
                    C_loc,
                    Output[0, oh, ow_block * MLEN, oc_block * MLEN],
                )

    lowered = compile_func(tiled_conv2d)

    constants = {
        "BATCH": BATCH, "H_IN": H_IN, "W_IN": W_IN,
        "C_IN": C_IN, "C_OUT": C_OUT, "KH": KH, "KW": KW,
        "OH": OH, "OW": OW, "MLEN": MLEN,
        "NUM_OW": NUM_OW, "NUM_IC": NUM_IC, "NUM_OC": NUM_OC,
    }
    return lowered, constants


__all__ = ["make_tiled_conv2d"]
