"""Lowering for the final hidden-to-vocabulary projection (LM head)."""

from __future__ import annotations

from .projection_asm import projection_T_asm

# The LM head reuses the checkpoint's native ``lm_head.weight`` layout: row-major
# ``(vocab_size, hidden_size)``, so row ``v`` holds the full hidden-dimension
# vector for vocabulary entry ``v`` and the HBM row stride is ``hidden_size``.
# ``logits = hidden @ lm_head.weight.T`` therefore lowers to the transposed
# projection, which streams one weight-row group at a time and needs no
# transpose pass over the 151k-row weight matrix.
HBM_WEIGHT_LAYOUT = "row_major_vocab_by_hidden"


def lm_head_vocab_padding(vocab_size: int, blen: int) -> int:
    """Return the vocabulary size rounded up to a whole number of BLEN tiles.

    The matrix unit emits ``blen`` output features per tile, so a vocabulary that
    is not a multiple of ``blen`` is padded. Padded logits are masked out by the
    sampler and never change the selected token.
    """

    if vocab_size <= 0 or blen <= 0:
        raise ValueError("vocab_size and blen must be positive")
    return ((vocab_size + blen - 1) // blen) * blen


def lm_head_asm(
    mlen: int,
    blen: int,
    batch: int,
    hidden_size: int,
    vocab_size: int,
    alive_registers: list[int],
    lm_head_weight_hbm_offset_reg: int,
    activation_base_address: int,
    result_base_address: int,
) -> str:
    """Generate assembly for the hidden-to-vocabulary projection.

    Computes ``logits = hidden_states @ lm_head.weight.T``, mapping
    ``(batch, hidden_size) @ (vocab_size, hidden_size).T -> (batch, vocab_size)``.

    Args:
        mlen: Matrix tile length; must divide ``hidden_size``.
        blen: Batch/block tile length.
        batch: Decode batch size.
        hidden_size: Model hidden dimension, the reduction dimension K.
        vocab_size: Vocabulary size N, padded up to a multiple of ``blen``.
        alive_registers: At least six free general-purpose register indices.
        lm_head_weight_hbm_offset_reg: Address register holding the HBM base of
            ``lm_head.weight``.
        activation_base_address: Vector SRAM base of the final hidden states.
        result_base_address: Vector SRAM base for the emitted logits.

    Returns:
        Assembly text for the projection.
    """
    if len(alive_registers) < 6:
        raise ValueError(
            "lm_head_asm requires at least 6 alive registers "
            f"(got {len(alive_registers)})"
        )
    if hidden_size % mlen:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be a multiple of MLEN ({mlen})"
        )

    padded_vocab = lm_head_vocab_padding(vocab_size, blen)

    header = [
        "; === LM head: hidden -> vocab projection ===",
        f"; layout: {HBM_WEIGHT_LAYOUT} (vocab_size={vocab_size}, hidden_size={hidden_size})",
        f"; padded vocab: {padded_vocab} ({padded_vocab - vocab_size} masked entries)",
    ]
    body = projection_T_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=hidden_size,
        alive_registers=alive_registers,
        w_base_hbm_offset_reg=lm_head_weight_hbm_offset_reg,
        activation_base_address=activation_base_address,
        result_base_address=result_base_address,
        out_features=padded_vocab,
    )
    return "\n".join(header) + "\n" + body
