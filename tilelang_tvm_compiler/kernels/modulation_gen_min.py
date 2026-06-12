"""Modulation-generation kernel — adaLN-Zero shift/scale/gate from a vec.

Mirrors Open-Sora / Flux ``Modulation`` (opensora/models/mmdit/layers.py:179):

    out = Linear(SiLU(vec))            # then chunk into shift/scale/gate

This kernel covers ONE vec at a time. ``vec`` lives in a wider input tensor
``VEC_hbm (1, pipelined, 1, HD)`` that holds ``pipelined`` distinct vecs (e.g.
the per-block vecs of a denoise step, loaded once); ``pipe_idx`` selects which
one — a **compile-time constant** so the row offset ``pipe_idx*HD`` is a clean
constant (no run-time lvar address arithmetic, which is spec-forbidden / broken
for index-in-address — see project memory).

Structure (NO head axis — vec is one flat (1, HD) vector, not per-head):

  1. Read VEC_hbm[0, pipe_idx, 0, :] -> V_sh (1, HD), one MLEN-wide chunk per
     K block.
  2. inline SiLU on the (1, HD) row (rows=1 FP scalar chain; sigmoid =
     1/(1+exp(-x)), same expansion as silu_min but single-row).
  3. Three independent Linears (shift / scale / gate), each W is (HD, HD)
     row-major (nn.Linear convention). We split the qkv-style fused Linear
     into THREE square H->H gemms so every tile stays MLEN-square.

     Each output column block n accumulates over K blocks k:
        OUT[n] = sum_k  silu_vec[k] @ W[n, k]^T
     Done as **multiple rows=1 gemms + manual elementwise add** (no reliance
     on an mv K-accumulator / mv_wo drain): each (n, k) is one single-block
     ``T.gemm(rows=1)`` -> auto kind="mv" (M_MV / M_TMV via transpose_B);
     the K-add is an explicit ``T.Parallel`` add that fuse_elementwise folds
     to ``plena.v_add`` (same pattern as flash_decode's ``O += PV``).

``transpose_B=True`` -> the (N,K) weight is transposed in-array, so the host
feeds plain nn.Linear weights and the golden is ``silu(vec) @ W.T``.
"""

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes


def make_modulation_gen_min(
    *,
    hd: int | None = None,
    pipelined: int = 1,
    pipe_idx: int = 0,
    seq: int | None = None,
):
    _hw = _load_sizes()
    MLEN = _hw.mlen
    # HD follows geometry (head_count*hlen); do NOT hardcode. Default to one
    # MLEN tile if not given, so a minimal single-K-block test can run.
    if hd is None:
        hd = MLEN
    HD = hd
    if HD % MLEN != 0:
        raise ValueError(f"hd ({HD}) must be a multiple of MLEN ({MLEN})")
    if pipelined < 1:
        raise ValueError(f"pipelined must be >= 1, got {pipelined}")
    if not (0 <= pipe_idx < pipelined):
        raise ValueError(
            f"pipe_idx ({pipe_idx}) must be in [0, pipelined={pipelined})"
        )
    # Output sequence length: the kernel broadcasts each (1, HD) modulation
    # vector across SEQ rows so the result is directly usable by the
    # downstream (1+scale)*x + shift / x + gate*y — no separate broadcast
    # kernel. Defaults to one MLEN tile (single s_block).
    if seq is None:
        seq = MLEN
    SEQ = seq
    if SEQ % MLEN != 0:
        raise ValueError(f"seq ({SEQ}) must be a multiple of MLEN ({MLEN})")

    n_blocks = HD // MLEN     # output column blocks (N)
    k_blocks = HD // MLEN     # contraction blocks (K)
    s_blocks = SEQ // MLEN    # output row blocks (broadcast target)
    NEG_ONE = -1.0

    @T.prim_func
    def modulation_gen_min(
        VEC_hbm:   T.Tensor((1, MLEN, 1, HD), "float16"),  # pipelined rows in [0:pipelined]
        W_SHIFT:   T.Tensor((1, HD, 1, HD), "float16"),  # (N, K) row-major

        SHIFT_hbm: T.Tensor((1, SEQ, 1, HD), "float16"),  # broadcast over SEQ
        
    ):
        # Grid axis = N output blocks. Each program produces one MLEN-wide
        # column slice of all three (shift/scale/gate), broadcast across all
        # SEQ rows. (No head axis; vec is one flat (1, HD) vector.)
        with T.Kernel(n_blocks, threads=128) as (nb,):
            # All staging buffers are (MLEN, MLEN) and all DMAs move whole
            # MLEN x MLEN tiles — EXACTLY like linear_min's A/B/C copies. VEC
            # is staged as a full tile (its pipelined rows live in the first
            # `pipelined` rows; the rest is unused padding); the pipe_idx row
            # is then picked via a VRAM->VRAM copy into SV_sh.
            VEC_tile = T.alloc_shared((MLEN, MLEN), "float16")
            # SV_sh is (1, MLEN): the picked vec row. Keeping it 1-row makes
            # the silu loop's `for i` index the COLUMN (SIMD inner dim). If
            # SV_sh were (MLEN, MLEN), `SV_sh[0, i]` would be mis-lowered so
            # `i` indexes the ROW — silu then runs on the wrong axis.
            SV_sh = T.alloc_shared((1, MLEN), "float16")
            SG_sh = T.alloc_shared((1, MLEN), "float16")     # silu sigmoid scratch
            W_sh  = T.alloc_shared((MLEN, MLEN), "float16")  # weight (N,K) tile
            SCR   = T.alloc_fragment((1, MLEN), "float16")   # per-(n,k) gemm out
            OSH   = T.alloc_fragment((1, MLEN), "float16")   # shift accumulator
            OSC   = T.alloc_fragment((1, MLEN), "float16")   # scale accumulator
            OGT   = T.alloc_fragment((1, MLEN), "float16")   # gate accumulator
            # broadcast tiles: each (1, MLEN) row replicated down MLEN rows,
            # written to HBM as full tiles.
            BSH = T.alloc_shared((MLEN, MLEN), "float16")
            BSC = T.alloc_shared((MLEN, MLEN), "float16")
            BGT = T.alloc_shared((MLEN, MLEN), "float16")

            # === SHIFT-ONLY (isolate the single h2m+mv path) ===
            for col in T.serial(MLEN):
                OSH[0, col] = T.float16(0)

            for kb in T.serial(k_blocks):
                # DMA a whole MLEN x MLEN tile of VEC — same form as
                # linear_min's A copy.
                T.copy(
                    VEC_hbm[0, 0:MLEN, 0, kb * MLEN:(kb + 1) * MLEN],
                    VEC_tile,
                )
                # pick the pipe_idx row into SV_sh (VRAM->VRAM).
                T.copy(VEC_tile[pipe_idx:pipe_idx + 1, :], SV_sh)

                # SiLU on the (1, MLEN) K-slice (elementwise, so per-slice ==
                # silu of the whole vec restricted to this slice). sigmoid =
                # 1/(1+exp(-x)); then x*sigmoid. SG_sh starts as a COPY of
                # SV_sh so the first op (negate) is in-place on SG_sh and never
                # masks unselected lane-heads with a different src (see memory:
                # masked VRAM ops overwrite unselected heads with src1, safe
                # only when dst==src1).
                for i in T.serial(MLEN):
                    SG_sh[0, i] = SV_sh[0, i]
                for i in T.serial(MLEN):
                    SG_sh[0, i] = T.float16(NEG_ONE) * SG_sh[0, i]   # -x
                for i in T.serial(MLEN):
                    SG_sh[0, i] = T.exp(SG_sh[0, i])                 # exp(-x)
                for i in T.serial(MLEN):
                    SG_sh[0, i] = T.float16(1.0) + SG_sh[0, i]       # 1+exp(-x)
                for i in T.serial(MLEN):
                    SG_sh[0, i] = T.float16(1.0) / SG_sh[0, i]       # sigmoid
                for i in T.serial(MLEN):
                    SV_sh[0, i] = SV_sh[0, i] * SG_sh[0, i]          # x*sigmoid

                # ---- shift: vec @ W (no transpose) ----
                # mv computes SV @ W_tile, so for OUT[n] = sum_k m[k] @ W[k, n]
                # the tile rows index K (kb) and cols index N (nb). NOT W[nb, kb]
                # — that transposes the block grid and only coincides when there
                # is a single block (HD == MLEN).
                T.copy(W_SHIFT[0, kb * MLEN:(kb + 1) * MLEN, 0,
                               nb * MLEN:(nb + 1) * MLEN], W_sh)
                T.gemm(SV_sh, W_sh, SCR)   # rows=1 -> M_MV (no transpose)
                for col in T.Parallel(MLEN):
                    OSH[0, col] = OSH[0, col] + SCR[0, col]

            # broadcast OSH down a full (MLEN, MLEN) tile via per-row copy,
            # then write each s_block as a whole MLEN x MLEN tile to HBM.
            for row in T.serial(MLEN):
                T.copy(OSH, BSH[row:row + 1, :])

            for sb in T.serial(s_blocks):
                T.copy(BSH, SHIFT_hbm[0, sb * MLEN:(sb + 1) * MLEN, 0,
                                      nb * MLEN:(nb + 1) * MLEN])

    constants = {
        "MLEN": MLEN,
        "HD": HD,
        "SEQ": SEQ,
        "PIPELINED": pipelined,
        "PIPE_IDX": pipe_idx,
        "N_BLOCKS": n_blocks,
        "K_BLOCKS": k_blocks,
        "S_BLOCKS": s_blocks,
    }
    return modulation_gen_min, constants


__all__ = ["make_modulation_gen_min"]
