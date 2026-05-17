# TODO: full hardware implementation pending.
# This stub emits a structured comment block so downstream tooling (grep, ASM
# line-count checks) can detect the lm_head section.  Replace the body with
# real M_MM / M_MO sequences once the HBM weight-layout for the vocab
# projection is finalised.
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
    """
    Generate assembly stub for the final hidden→vocab_size projection (LM head).

    The LM head is a linear projection:
        logits = hidden_states @ lm_head.weight.T
        shape : (batch, seq, hidden_size) @ (hidden_size, vocab_size)
              → (batch, seq, vocab_size)

    Args:
        mlen: Matrix lane width (hardware MLEN).
        blen: Batch lane width (hardware BLEN).
        batch: Batch size.
        hidden_size: Model hidden dimension (K of the matmul).
        vocab_size: Vocabulary size (N of the matmul).
        alive_registers: List of available GP register indices (need ≥ 4).
        lm_head_weight_hbm_offset_reg: Address-register index pointing to the
            lm_head weight matrix in HBM.
        activation_base_address: VRAM base address of the input hidden states.
        result_base_address: VRAM base address for storing output logits.

    Returns:
        str: Assembly code string for the LM head projection.
    """
    assert len(alive_registers) >= 4, "lm_head_asm requires at least 4 alive registers"

    code = "; === LM head: hidden→vocab projection ===\n"
    code += f"; lm_head_asm: batch={batch}, hidden_size={hidden_size}, vocab_size={vocab_size}\n"
    code += f"; weight HBM offset reg: a{lm_head_weight_hbm_offset_reg}\n"
    code += f"; activation VRAM base: {activation_base_address}, result VRAM base: {result_base_address}\n"
    code += "; TODO: replace stub with M_MM/M_MO tiled matmul sequence\n"
    code += "; lm_head stub end\n"
    return code
