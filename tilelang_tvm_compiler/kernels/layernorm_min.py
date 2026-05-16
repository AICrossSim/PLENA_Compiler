"""LayerNorm-min kernel — ``out = (x - mean(x)) * rsqrt(var(x) + eps) * scale + bias``.

Differs from ``rmsnorm_min`` in two key ways:

  1. **Reduction span**: LayerNorm normalises over the entire
     ``hidden_size = H*D`` dimension of a ``(B, S, H*D)`` tensor (no
     per-head split). HBM layout is ``(1, S, 1, H*D)`` and D > MLEN
     is the common case (``hidden_size`` is typically 128, 256, …).
     The reduce / row_*_fp / tile_* emitters handle the multi-d_tile
     unroll themselves via the 7D ``tile_layout``.

  2. **Extra subtraction + bias**: LayerNorm subtracts the mean before
     squaring (RMSNorm doesn't), and adds a learnable bias at the end
     (RMSNorm's affine has scale only).

Decomposed into PLENA single-op stores:

    mean_sum[i]  = sum_j(X[i,j])              row_reduce_sum_at
    mu[i]        = mean_sum[i] * INV_N[i]     fp_mul
    XC[i,j]      = X[i,j] - mu[i]             row_sub_fp_at
    SQ[i,j]      = XC[i,j] * XC[i,j]          tile_mul
    var_sum[i]   = sum_j(SQ[i,j])             row_reduce_sum_at
    var[i]       = var_sum[i] * INV_N[i]      fp_mul
    var_eps[i]   = var[i] + EPS[i]            fp_add
    norm[i]      = sqrt(var_eps[i])           fp_sqrt
    inv[i]       = 1 / norm[i]                fp_reci
    Y[i,j]       = XC[i,j] * SCALE[i,j]       tile_mul   (host broadcasts scale)
    Y[i,j]      *= inv[i]                     row_mul_fp_at (in-place)
    Y[i,j]      += BIAS[i,j]                  tile_add   (host broadcasts bias)

Like ``rmsnorm_min``:
  * ``INV_N = 1/hidden_size`` and ``EPS`` are FPRAM-preloaded scalars.
  * The accumulating ``V_RED_SUM`` is seeded from a preloaded zero
    fragment (``SS_INIT``) before each reduce — same pattern flash
    attention uses for ``L_INIT``.
"""

import tilelang.language as T


def make_layernorm_min(
    *,
    rows: int = 64,
    hidden_size: int = 128,
    num_s_blocks: int = 2,
    batch: int = 1,
    eps: float = 1e-6,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(
            f"layernorm_min requires rows == MLEN ({MLEN}), got {rows}"
        )
    if hidden_size % MLEN != 0:
        raise ValueError(
            f"hidden_size must be a multiple of MLEN ({MLEN}); "
            f"got hidden_size={hidden_size}"
        )
    if num_s_blocks < 1:
        raise ValueError(f"num_s_blocks must be >= 1, got {num_s_blocks}")

    seq_len = num_s_blocks * rows
    H = hidden_size
    # INV_N (= 1/hidden_size) and eps are inlined as T.float16(...)
    # literals; auto-hoisted into 1-slot global.fpram buffers.
    inv_n_val = 1.0 / hidden_size

    @T.prim_func
    def layernorm_min(
        X_hbm:     T.Tensor((batch, seq_len, 1, H), "float16"),
        SCALE_hbm: T.Tensor((batch, seq_len, 1, H), "float16"),
        BIAS_hbm:  T.Tensor((batch, seq_len, 1, H), "float16"),
        Y_hbm:     T.Tensor((batch, seq_len, 1, H), "float16"),
    ):
        with T.Kernel(num_s_blocks, threads=128) as s_block:
            # HBM <-> on-chip staging (shared). No head packing (H=1 in
            # the BSHD layout); ``hidden_size > MLEN`` triggers the 7D
            # tile_layout with d_tiles = hidden_size/MLEN.
            X_sh     = T.alloc_shared((rows, H), "float16")
            SCALE_sh = T.alloc_shared((rows, H), "float16")
            BIAS_sh  = T.alloc_shared((rows, H), "float16")
            Y_sh     = T.alloc_shared((rows, H), "float16")

            X_loc    = T.alloc_fragment((rows, H), "float16")
            SC_loc   = T.alloc_fragment((rows, H), "float16")
            BI_loc   = T.alloc_fragment((rows, H), "float16")
            SQ_loc   = T.alloc_fragment((rows, H), "float16")
            Y_loc    = T.alloc_fragment((rows, H), "float16")

            # Per-row FP scratch (rank-1 -> FPRAM scalar slots).
            MEAN_SUM = T.alloc_fragment((rows,), "float16")
            MU       = T.alloc_fragment((rows,), "float16")
            VAR_SUM  = T.alloc_fragment((rows,), "float16")
            VAR      = T.alloc_fragment((rows,), "float16")
            VAR_EPS  = T.alloc_fragment((rows,), "float16")
            NORM     = T.alloc_fragment((rows,), "float16")
            INV      = T.alloc_fragment((rows,), "float16")

            # INV_N (1/hidden_size) and eps are inlined as
            # T.float16(...) literals below; the zero seed is
            # T.float16(0) which takes fold's zero-fill path.

            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, 0, 0:H],
                X_sh,
            )
            T.copy(
                SCALE_hbm[0, s_block * rows : (s_block + 1) * rows, 0, 0:H],
                SCALE_sh,
            )
            T.copy(
                BIAS_hbm[0, s_block * rows : (s_block + 1) * rows, 0, 0:H],
                BIAS_sh,
            )
            T.copy(X_sh, X_loc)
            T.copy(SCALE_sh, SC_loc)
            T.copy(BIAS_sh, BI_loc)

            # Seed mean accumulator from zero before reduce —
            # V_RED_SUM accumulates into its FPRAM slot.
            for row in T.serial(rows):
                MEAN_SUM[row] = T.float16(0)

            # mean_sum[i] = sum_j(X[i, j])
            T.reduce_sum(X_loc, MEAN_SUM, dim=1)

            # mu[i] = mean_sum[i] * INV_N[i]
            for row in T.serial(rows):
                MU[row] = MEAN_SUM[row] * T.float16(inv_n_val)

            # XC = X - mu  (in-place on X_loc to avoid extra fragment).
            for row in T.serial(rows):
                for col in T.Parallel(H):
                    X_loc[row, col] = X_loc[row, col] - MU[row]

            # SQ = XC * XC
            for row in T.serial(rows):
                for col in T.Parallel(H):
                    SQ_loc[row, col] = X_loc[row, col] * X_loc[row, col]
                VAR_SUM[row] = T.float16(0)

            T.reduce_sum(SQ_loc, VAR_SUM, dim=1)

            for row in T.serial(rows):
                VAR[row]     = VAR_SUM[row] * T.float16(inv_n_val)
                VAR_EPS[row] = VAR[row] + T.float16(eps)
                NORM[row]    = T.sqrt(VAR_EPS[row])
                INV[row]     = T.float16(1.0) / NORM[row]

            # Y = XC * scale (host-broadcast SCALE into (rows, H)).
            # Write directly into Y_loc so the in-place row_mul_fp_at
            # below has dst == src (no cross-lane pollution; not strictly
            # needed here because there is no packed-head mask in this
            # path, but it keeps the kernel parallel to rmsnorm_min).
            for row in T.serial(rows):
                for col in T.Parallel(H):
                    Y_loc[row, col] = X_loc[row, col] * SC_loc[row, col]

            # Y *= inv  (row_mul_fp_at; D > MLEN unrolls inside the emitter).
            for row in T.serial(rows):
                for col in T.Parallel(H):
                    Y_loc[row, col] = Y_loc[row, col] * INV[row]

            # Y += bias  (host-broadcast BIAS into (rows, H)).
            for row in T.serial(rows):
                for col in T.Parallel(H):
                    Y_loc[row, col] = Y_loc[row, col] + BI_loc[row, col]

            T.copy(Y_loc, Y_sh)
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows, 0, 0:H],
            )

    lowered = layernorm_min
    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HIDDEN_SIZE": hidden_size,
        "BATCH": batch,
        "NUM_S_BLOCKS": num_s_blocks,
    }
    return lowered, constants


__all__ = ["make_layernorm_min"]
