"""Modulate-min kernel — adaLN ``out = (1 + scale) * x + shift``.

Designed for the tile_mul / tile_add VRAM elementwise path (no FPRAM
involvement). The literal ``1 + scale`` term is hoisted out of the
kernel: the testbench passes ``scale_plus_one = scale + 1`` directly,
so the kernel reduces to one tile_mul + one tile_add:

    tmp = scale_plus_one * x
    out = tmp + shift

Both stores are single-op binops over same-shape VRAM tiles — exactly
what fold + plena.tile_* expects.

Layout: HBM -> VRAM tiles -> tile_mul -> tile_add -> VRAM -> HBM.
"""

import tilelang.language as T


def make_modulate_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"modulate_min requires rows == MLEN ({MLEN}), got {rows}")
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
    def modulate_min(
        X_hbm:        T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        SCALE1P_hbm:  T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        SHIFT_hbm:    T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm:        T.Tensor((batch, seq_len, head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh       = T.alloc_shared((rows, hlen), "float16")
            SCALE1P_sh = T.alloc_shared((rows, hlen), "float16")
            SHIFT_sh   = T.alloc_shared((rows, hlen), "float16")
            TMP_sh     = T.alloc_shared((rows, hlen), "float16")
            Y_sh       = T.alloc_shared((rows, hlen), "float16")

            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )
            T.copy(
                SCALE1P_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                SCALE1P_sh,
            )
            T.copy(
                SHIFT_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                SHIFT_sh,
            )

            # tmp = (1 + scale) * x
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    TMP_sh[row, col] = SCALE1P_sh[row, col] * X_sh[row, col]

            # out = tmp + shift
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    Y_sh[row, col] = TMP_sh[row, col] + SHIFT_sh[row, col]

            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
            )

    lowered = modulate_min

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


__all__ = ["make_modulate_min"]
