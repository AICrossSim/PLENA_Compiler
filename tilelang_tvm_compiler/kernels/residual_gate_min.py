"""Residual-gate-min kernel — ``out = x + gate * y`` on VRAM tiles.

Two single-op stores: one tile_mul (``tmp = gate * y``) then one
tile_add (``out = x + tmp``). All operands are same-shape VRAM
tiles — exactly the pattern fold + plena.tile_* expects.

Layout: HBM -> VRAM tiles -> tile_mul -> tile_add -> VRAM -> HBM.
"""

import tilelang.language as T


def make_residual_gate_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"residual_gate_min requires rows == MLEN ({MLEN}), got {rows}")
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
    def residual_gate_min(
        X_hbm:    T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        GATE_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm:    T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        OUT_hbm:  T.Tensor((batch, seq_len, head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh    = T.alloc_shared((rows, hlen), "float16")
            GATE_sh = T.alloc_shared((rows, hlen), "float16")
            Y_sh    = T.alloc_shared((rows, hlen), "float16")
            TMP_sh  = T.alloc_shared((rows, hlen), "float16")
            OUT_sh  = T.alloc_shared((rows, hlen), "float16")

            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )
            T.copy(
                GATE_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                GATE_sh,
            )
            T.copy(
                Y_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                Y_sh,
            )

            # tmp = gate * y
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    TMP_sh[row, col] = GATE_sh[row, col] * Y_sh[row, col]

            # out = x + tmp
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    OUT_sh[row, col] = X_sh[row, col] + TMP_sh[row, col]

            T.copy(
                OUT_sh,
                OUT_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
            )

    lowered = residual_gate_min

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


__all__ = ["make_residual_gate_min"]
