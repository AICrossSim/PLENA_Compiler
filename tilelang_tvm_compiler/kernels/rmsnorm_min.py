"""RMSNorm-min kernel — ``out = x * scale * rsqrt(mean(x^2) + eps)``.

Decomposed into PLENA's single-op stores:

    sq[i,j]    = x[i,j] * x[i,j]              tile_mul
    ss[i]      = sum_j sq[i,j]                row_reduce_sum_at
    ss_n[i]    = ss[i] * INV_N[i]             fp_mul       (INV_N preloaded = 1/N)
    ss_eps[i]  = ss_n[i] + EPS[i]             fp_add       (EPS preloaded)
    norm[i]    = sqrt(ss_eps[i])              fp_sqrt
    inv[i]     = 1 / norm[i]                  fp_reci      (literal-1 numerator)
    xs[i,j]    = x[i,j] * SCALE[i,j]          tile_mul     (host broadcasts scale)
    out[i,j]   = xs[i,j] * inv[i]             row_mul_fp_at

The ``1/N`` and ``eps`` scalars come from preloaded FPRAM fragments
(mirroring gelu_min). The learnable ``scale`` weight ``(hlen,)`` is
pre-broadcast on the host into a ``(rows, hlen)`` tile so we can use
``tile_mul`` instead of a row-broadcast op (none exists for VRAM x VRAM
broadcast yet).
"""

import tilelang.language as T


def make_rmsnorm_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
    eps: float = 1e-6,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"rmsnorm_min requires rows == MLEN ({MLEN}), got {rows}")
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
    # Constants inlined as T.float16(...) literals below; the
    # hoist_float_constants pre-pass synthesises 1-slot global.fpram
    # buffers for each unique value.
    inv_n_val = 1.0 / hlen

    @T.prim_func
    def rmsnorm_min(
        X_hbm:     T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        SCALE_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm:     T.Tensor((batch, seq_len, head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            # HBM ↔ on-chip staging (shared).
            X_sh     = T.alloc_shared((rows, hlen), "float16")
            SCALE_sh = T.alloc_shared((rows, hlen), "float16")
            Y_sh     = T.alloc_shared((rows, hlen), "float16")

            # Rank-2 VRAM work fragments — row_*_fp_at / tile_mul take
            # their dst/src from these. (rank-1 fragments would land on
            # FPRAM scalars, rank-2 stays on VRAM.)
            X_loc   = T.alloc_fragment((rows, hlen), "float16")
            SC_loc  = T.alloc_fragment((rows, hlen), "float16")
            SQ_loc  = T.alloc_fragment((rows, hlen), "float16")
            Y_loc   = T.alloc_fragment((rows, hlen), "float16")

            # Per-row FP scratch (rank-1 → FPRAM scalar slots).
            SS     = T.alloc_fragment((rows,), "float16")
            SS_N   = T.alloc_fragment((rows,), "float16")
            SS_EPS = T.alloc_fragment((rows,), "float16")
            NORM   = T.alloc_fragment((rows,), "float16")
            INV    = T.alloc_fragment((rows,), "float16")

            # 1/hlen and eps are inlined as T.float16(...) literals
            # below; auto-hoisted into 1-slot global.fpram buffers.
            # The zero seed for SS (``SS = SS_INIT[row]``) is also
            # inlined; T.float16(0) takes fold's zero-fill path and
            # doesn't go through FPRAM at all.

            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )
            T.copy(
                SCALE_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                SCALE_sh,
            )
            T.copy(X_sh, X_loc)
            T.copy(SCALE_sh, SC_loc)

            # sq = x * x, and seed SS from preloaded zero in the same
            # row loop. V_RED_SUM accumulates into the FPRAM slot, so
            # SS must be pre-zeroed before the reduce. Mirrors
            # flash_attention_min's pattern of folding
            # ``P_SUM[row] = L_INIT[row]`` into the row loop that
            # precedes ``T.reduce_sum(..., clear=False)``.
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    SQ_loc[row, col] = X_loc[row, col] * X_loc[row, col]
                SS[row] = T.float16(0)

            # ss[i] = sum_j sq[i,j]
            T.reduce_sum(SQ_loc, SS, dim=1)

            for row in T.serial(rows):
                SS_N[row]   = SS[row] * T.float16(inv_n_val)
                SS_EPS[row] = SS_N[row] + T.float16(eps)
                NORM[row]   = T.sqrt(SS_EPS[row])
                # literal-1 numerator so fold picks the reci pattern
                INV[row]    = T.float16(1.0) / NORM[row]

            # y = x * scale  (host has broadcast SCALE into (rows, hlen)).
            # Write directly into Y_loc — packed-head row_mul_fp_at below
            # requires dst == src (its unmasked heads otherwise overwrite
            # dst with src verbatim, destroying cross-by_phase writes).
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    Y_loc[row, col] = X_loc[row, col] * SC_loc[row, col]

            # out[i,j] = y[i,j] * inv[i]  (in-place on Y_loc)
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    Y_loc[row, col] = Y_loc[row, col] * INV[row]

            T.copy(Y_loc, Y_sh)
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
            )

    lowered = rmsnorm_min

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


__all__ = ["make_rmsnorm_min"]
