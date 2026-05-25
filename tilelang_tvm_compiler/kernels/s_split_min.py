"""S-split-min kernel — sequence-axis split of one head-packed tensor into
two.

The inverse of s_concat_min, and the double-stream counterpart to the
``attn1[:, :txt_len], attn1[:, txt_len:]`` split in the MMDiT
DoubleStreamBlock: after the joint attention runs over the concatenated
``[S_a + S_b, HD]`` sequence, its output is split back into the txt segment
(first S_a rows) and the img segment (last S_b rows), each landing at its own
HBM address so the two streams' post-attention proj/mlp run independently.

In the head-packed view ``[B, S, 1, HD]`` every (b, s) row is HD contiguous
fp16 elements. Sequence split just peels whole rows:

    A_hbm[:, :, 0, :] = X[:, 0:S_a,        0, :]    (first S_a rows)
    B_hbm[:, :, 0, :] = X[:, S_a:S_a+S_b,  0, :]    (last  S_b rows)

Why a real kernel (not an HBM alias): the same MX-E4M3 ``[elem][scale]``
two-region packing reason as s_concat_min — peeling rows by address-offset
would slice through the wrong region, so each segment is re-packed by a real
VRAM->VRAM copy into its own correctly-packed tensor. Plain copy, no compute.
"""

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes


def make_s_split_min(
    *,
    rows: int | None = None,
    hd: int = 128,
    num_s_blocks_a: int = 2,
    num_s_blocks_b: int = 2,
    batch: int = 1,
):
    """Sequence-axis split of one head-packed tensor into two.

    X_hbm : [batch, S_a + S_b, 1, hd]    S_a = num_s_blocks_a * rows
    A_hbm : [batch, S_a, 1, hd]          = X[:, 0:S_a]
    B_hbm : [batch, S_b, 1, hd]          = X[:, S_a:S_a+S_b]

    ``hd`` must be a multiple of MLEN (the copy walks MLEN-wide feature
    blocks). Mirrors s_concat_min's geometry exactly.
    """
    _hw = _load_sizes()
    MLEN = _hw.mlen
    if rows is None:
        rows = MLEN
    if rows != MLEN:
        raise ValueError(f"s_split_min requires rows == MLEN ({MLEN}), got {rows}")
    if hd % MLEN != 0:
        raise ValueError(f"hd must be a multiple of MLEN ({MLEN}); got {hd}")
    if num_s_blocks_a < 1 or num_s_blocks_b < 1:
        raise ValueError("num_s_blocks_a / num_s_blocks_b must be >= 1")

    s_a = num_s_blocks_a * rows
    s_b = num_s_blocks_b * rows
    s_total = s_a + s_b
    feat_blocks = hd // MLEN

    @T.prim_func
    def s_split_min(
        X_hbm: T.Tensor((batch, s_total, 1, hd), "float16"),
        A_hbm: T.Tensor((batch, s_a, 1, hd), "float16"),
        B_hbm: T.Tensor((batch, s_b, 1, hd), "float16"),
    ):
        # Grid iterates over ALL input seq blocks. The first num_s_blocks_a
        # tiles (X rows [0:s_a]) go to A; the rest (X rows [s_a:]) go to B.
        # All offsets are compile-time constants.
        with T.Kernel(num_s_blocks_a + num_s_blocks_b, threads=128) as blk:
            sh = T.alloc_shared((rows, MLEN), "float16")

            # X[0:s_a] -> A
            for sb in T.serial(num_s_blocks_a):
                for fb in T.serial(feat_blocks):
                    T.copy(
                        X_hbm[0, sb * rows : (sb + 1) * rows,
                              0, fb * MLEN : (fb + 1) * MLEN],
                        sh,
                    )
                    T.copy(
                        sh,
                        A_hbm[0, sb * rows : (sb + 1) * rows,
                              0, fb * MLEN : (fb + 1) * MLEN],
                    )

            # X[s_a : s_a + s_b] -> B
            for sb in T.serial(num_s_blocks_b):
                for fb in T.serial(feat_blocks):
                    T.copy(
                        X_hbm[0, s_a + sb * rows : s_a + (sb + 1) * rows,
                              0, fb * MLEN : (fb + 1) * MLEN],
                        sh,
                    )
                    T.copy(
                        sh,
                        B_hbm[0, sb * rows : (sb + 1) * rows,
                              0, fb * MLEN : (fb + 1) * MLEN],
                    )

    lowered = s_split_min

    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HD": hd,
        "S_A": s_a,
        "S_B": s_b,
        "S_TOTAL": s_total,
        "FEAT_BLOCKS": feat_blocks,
        "NUM_S_BLOCKS_A": num_s_blocks_a,
        "NUM_S_BLOCKS_B": num_s_blocks_b,
        "BATCH": batch,
    }
    return lowered, constants


__all__ = ["make_s_split_min"]
