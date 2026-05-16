"""Concat-min kernel — feature-axis concatenation of two head-packed
tensors.

Both inputs are taken in the HEAD-PACKED view ``[B, S, 1, dim]`` (see
_head_layout.py: BSHD ``[B,S,H,D]`` and packed ``[B,S,1,H*D]`` share
the same row-major fp16 bytes — heads folded into the feature axis).
Concatenating along that feature axis:

    Y[:, :, 0, 0:Adim]            = A
    Y[:, :, 0, Adim:Adim+Bdim]    = B

The copy is done in MLEN-wide blocks (the hardware tile granularity),
NOT per-head: each (B,S) row is ``dim`` contiguous fp16 elements =
``dim / MLEN`` blocks, and the kernel walks those blocks. A's blocks
land in Y's first ``Adim`` columns, B's in the next ``Bdim``.

Plain VRAM->VRAM copy — no FPRAM, no compute.

Why a dedicated kernel: the single-stream-block chain needs
``concat([attn_out, mlp_out])`` as linear2's input. Letting each
producer (flash_attention, gelu) write its OWN compact tensor and
joining them here keeps attention/gelu on their plain compact-output
path (no o_head_offset writeback) and keeps every step independently
verifiable.
"""

import tilelang.language as T


def make_concat_min(
    *,
    rows: int = 64,
    a_dim: int = 128,
    b_dim: int = 128,
    num_s_blocks: int = 2,
    batch: int = 1,
):
    """Feature-axis concat of two head-packed tensors.

    A_hbm : [batch, seq, 1, a_dim]
    B_hbm : [batch, seq, 1, b_dim]
    Y_hbm : [batch, seq, 1, a_dim + b_dim]
            Y[..., 0, 0:a_dim]            = A
            Y[..., 0, a_dim:a_dim+b_dim]  = B

    ``a_dim`` and ``b_dim`` must each be a multiple of MLEN (the copy
    walks MLEN-wide blocks). Inputs are the head-packed [B,S,1,dim]
    view; a BSHD producer output aliases this byte-for-byte.
    """
    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"concat_min requires rows == MLEN ({MLEN}), got {rows}")
    if a_dim % MLEN != 0:
        raise ValueError(f"a_dim must be a multiple of MLEN ({MLEN}); got {a_dim}")
    if b_dim % MLEN != 0:
        raise ValueError(f"b_dim must be a multiple of MLEN ({MLEN}); got {b_dim}")
    if num_s_blocks < 1:
        raise ValueError(f"num_s_blocks must be >= 1, got {num_s_blocks}")

    seq_len = num_s_blocks * rows
    total_dim = a_dim + b_dim
    a_blocks = a_dim // MLEN
    b_blocks = b_dim // MLEN

    @T.prim_func
    def concat_min(
        A_hbm: T.Tensor((batch, seq_len, 1, a_dim), "float16"),
        B_hbm: T.Tensor((batch, seq_len, 1, b_dim), "float16"),
        Y_hbm: T.Tensor((batch, seq_len, 1, total_dim), "float16"),
    ):
        # Grid iterates seq blocks. Each program copies one (rows x dim)
        # tile per source, MLEN-wide block at a time, in a single
        # uniform body. A's blocks fill Y[..., 0:a_dim]; B's blocks fill
        # Y[..., a_dim:total_dim]. All block offsets are compile-time
        # constants — no grid-variable branch.
        with T.Kernel(num_s_blocks, threads=128) as s_block:
            A_sh = T.alloc_shared((rows, MLEN), "float16")
            B_sh = T.alloc_shared((rows, MLEN), "float16")

            # A -> first a_dim columns of Y.
            for blk in T.serial(a_blocks):
                T.copy(
                    A_hbm[0, s_block * rows : (s_block + 1) * rows,
                          0, blk * MLEN : (blk + 1) * MLEN],
                    A_sh,
                )
                T.copy(
                    A_sh,
                    Y_hbm[0, s_block * rows : (s_block + 1) * rows,
                          0, blk * MLEN : (blk + 1) * MLEN],
                )

            # B -> next b_dim columns of Y (shifted by a_dim).
            for blk in T.serial(b_blocks):
                T.copy(
                    B_hbm[0, s_block * rows : (s_block + 1) * rows,
                          0, blk * MLEN : (blk + 1) * MLEN],
                    B_sh,
                )
                T.copy(
                    B_sh,
                    Y_hbm[0, s_block * rows : (s_block + 1) * rows,
                          0, a_dim + blk * MLEN : a_dim + (blk + 1) * MLEN],
                )

    lowered = concat_min

    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "A_DIM": a_dim,
        "B_DIM": b_dim,
        "TOTAL_DIM": total_dim,
        "A_BLOCKS": a_blocks,
        "B_BLOCKS": b_blocks,
        "BATCH": batch,
        "NUM_S_BLOCKS": num_s_blocks,
    }
    return lowered, constants


__all__ = ["make_concat_min"]
