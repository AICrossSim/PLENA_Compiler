"""S-concat-min kernel — sequence-axis concatenation of two head-packed
tensors.

This is the **double-stream** counterpart to concat_min. concat_min joins
two tensors along the FEATURE (D) axis (A's columns then B's columns within
each row). This kernel joins along the SEQUENCE (S) axis (A's rows then B's
rows), which is what the MMDiT DoubleStreamBlock needs before its joint
attention:

    q = torch.cat((txt_q, img_q), dim=2)   # dim=2 is the sequence axis

In the head-packed view ``[B, S, 1, HD]`` every (b, s) row is HD contiguous
fp16 elements. Sequence concat just stacks whole rows:

    Y[:, 0:S_a,        0, :] = A          (A's S_a rows -> Y's first rows)
    Y[:, S_a:S_a+S_b,  0, :] = B          (B's S_b rows -> Y's next rows)

Why a real kernel (not an HBM alias): in MX-E4M3 each tensor is packed as
``[all elem bytes][all scale bytes]`` (two separate regions, see binio.py).
Laying A and B back-to-back gives ``[A.elem][A.scale][B.elem][B.scale]`` but a
genuine fused tensor needs ``[A.elem][B.elem][A.scale][B.scale]`` — A's scale
region sits where B's elements should be, so a shape-only reinterpret reads
garbage. This kernel does a real VRAM->VRAM copy so the producer writes a
correctly-packed fused tensor. Plain copy — no FPRAM, no compute.

Both inputs share the same feature width HD; only the sequence length differs.
The copy walks MLEN-wide feature blocks for each row-tile (the hardware tile
granularity), identical machinery to concat_min but stacking on S not D.
"""

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes


def make_s_concat_min(
    *,
    rows: int | None = None,
    hd: int = 128,
    num_s_blocks_a: int = 2,
    num_s_blocks_b: int = 2,
    batch: int = 1,
):
    """Sequence-axis concat of two head-packed tensors of equal feature width.

    A_hbm : [batch, S_a, 1, hd]          S_a = num_s_blocks_a * rows
    B_hbm : [batch, S_b, 1, hd]          S_b = num_s_blocks_b * rows
    Y_hbm : [batch, S_a + S_b, 1, hd]
            Y[:, 0:S_a,        0, :] = A
            Y[:, S_a:S_a+S_b,  0, :] = B

    ``hd`` must be a multiple of MLEN (the copy walks MLEN-wide feature
    blocks). Both inputs are the head-packed [B,S,1,HD] view; a BSHD producer
    output aliases this byte-for-byte.
    """
    _hw = _load_sizes()
    MLEN = _hw.mlen
    if rows is None:
        rows = MLEN
    if rows != MLEN:
        raise ValueError(f"s_concat_min requires rows == MLEN ({MLEN}), got {rows}")
    if hd % MLEN != 0:
        raise ValueError(f"hd must be a multiple of MLEN ({MLEN}); got {hd}")
    if num_s_blocks_a < 1 or num_s_blocks_b < 1:
        raise ValueError("num_s_blocks_a / num_s_blocks_b must be >= 1")

    s_a = num_s_blocks_a * rows
    s_b = num_s_blocks_b * rows
    s_total = s_a + s_b
    feat_blocks = hd // MLEN

    @T.prim_func
    def s_concat_min(
        A_hbm: T.Tensor((batch, s_a, 1, hd), "float16"),
        B_hbm: T.Tensor((batch, s_b, 1, hd), "float16"),
        Y_hbm: T.Tensor((batch, s_total, 1, hd), "float16"),
    ):
        # Grid iterates over ALL output seq blocks (A's then B's). Each program
        # copies one (rows x hd) row-tile, MLEN-wide feature block at a time.
        # A's num_s_blocks_a tiles land at Y row-offset 0; B's tiles land at
        # Y row-offset s_a. All offsets are compile-time constants.
        with T.Kernel(num_s_blocks_a + num_s_blocks_b, threads=128) as blk:
            sh = T.alloc_shared((rows, MLEN), "float16")

            # A's row-tiles -> Y[0:s_a]
            for sb in T.serial(num_s_blocks_a):
                for fb in T.serial(feat_blocks):
                    T.copy(
                        A_hbm[0, sb * rows : (sb + 1) * rows,
                              0, fb * MLEN : (fb + 1) * MLEN],
                        sh,
                    )
                    T.copy(
                        sh,
                        Y_hbm[0, sb * rows : (sb + 1) * rows,
                              0, fb * MLEN : (fb + 1) * MLEN],
                    )

            # B's row-tiles -> Y[s_a : s_a + s_b]
            for sb in T.serial(num_s_blocks_b):
                for fb in T.serial(feat_blocks):
                    T.copy(
                        B_hbm[0, sb * rows : (sb + 1) * rows,
                              0, fb * MLEN : (fb + 1) * MLEN],
                        sh,
                    )
                    T.copy(
                        sh,
                        Y_hbm[0, s_a + sb * rows : s_a + (sb + 1) * rows,
                              0, fb * MLEN : (fb + 1) * MLEN],
                    )

    lowered = s_concat_min

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


__all__ = ["make_s_concat_min"]
