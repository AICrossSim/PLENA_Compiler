def embedding_asm(
    vlen: int,
    batch: int,
    hidden_size: int,
    alive_registers: list[int],
    activation_base_address: int,
    voc_table_base_addr_reg_index: int,
    input_ids: list[int],
    voc_table_row_bytes: int,
) -> str:
    """Generate assembly for token-embedding lookup as a direct HBM->VRAM DMA.

    For each token id in ``input_ids`` we copy one embedding row (of length
    ``hidden_size`` elements) out of HBM into VRAM at ``activation_base_address +
    token_idx * hidden_size``. The row is transferred in chunks of ``vlen``
    elements via ``H_PREFETCH_V``.

    The HBM vocab table is MXFP8-packed: each row occupies
    ``voc_table_row_bytes`` bytes (= ``(hidden_size // block_dim) *
    (act_block_width // 8 + scale_width // 8)``). The caller supplies that
    stride so this template stays quantization-format-agnostic.

    Cost: ``hidden_size // vlen`` H_PREFETCH_V + a handful of S_ADDI_INT per
    token. No MRAM usage, no M_MM.

    Args:
        vlen: Vector lane width in elements (matches VLEN).
        batch: Number of tokens in this batch; must equal ``len(input_ids)``.
        hidden_size: Embedding dimension. Must be a multiple of ``vlen``.
        alive_registers: Free general-purpose registers. We consume the first
            two (vram-dest pointer, hbm-byte-offset pointer).
        activation_base_address: VRAM element offset where token 0's embedding
            row lands. Subsequent tokens stack at +hidden_size elements each.
        voc_table_base_addr_reg_index: Index of the ``a<N>`` HBM address
            register holding the vocab-table base pointer.
        input_ids: Token ids emitted at codegen time (runtime token ids would
            need a scalar-indexed HBM load primitive we don't have).
        voc_table_row_bytes: Bytes per vocab row in HBM (MXFP8-packed).

    Returns:
        str: Generated assembly.
    """
    assert len(input_ids) == batch, f"Input IDs length {len(input_ids)} must match batch {batch}"
    assert hidden_size % vlen == 0, f"hidden_size {hidden_size} must be multiple of VLEN {vlen}"
    assert len(alive_registers) >= 2, "embedding_asm needs at least 2 alive registers"

    rows_per_token = hidden_size // vlen
    assert voc_table_row_bytes % rows_per_token == 0, (
        f"voc_table_row_bytes {voc_table_row_bytes} must be divisible by rows_per_token "
        f"{rows_per_token} (hidden_size // vlen)"
    )
    hbm_bytes_per_vlen_chunk = voc_table_row_bytes // rows_per_token

    vram_dest_reg = alive_registers[0]
    hbm_offset_reg = alive_registers[1]

    generated_code = "; Embedding_asm generation (DMA row-copy)\n"
    generated_code += (
        f"; vlen={vlen} hidden_size={hidden_size} batch={batch} "
        f"voc_table_row_bytes={voc_table_row_bytes} "
        f"hbm_bytes_per_vlen_chunk={hbm_bytes_per_vlen_chunk}\n"
    )

    for token_idx, token_id in enumerate(input_ids):
        vram_start = activation_base_address + token_idx * hidden_size
        hbm_byte_offset_start = token_id * voc_table_row_bytes
        generated_code += f"; token {token_idx} (id={token_id})\n"
        generated_code += f"S_ADDI_INT gp{vram_dest_reg}, gp0, {vram_start} \n"
        generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp0, {hbm_byte_offset_start} \n"
        for _ in range(rows_per_token):
            generated_code += (
                f"H_PREFETCH_V gp{vram_dest_reg}, gp{hbm_offset_reg}, "
                f"a{voc_table_base_addr_reg_index}, 0, 0 \n"
            )
            generated_code += f"S_ADDI_INT gp{vram_dest_reg}, gp{vram_dest_reg}, {vlen} \n"
            generated_code += (
                f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_offset_reg}, {hbm_bytes_per_vlen_chunk} \n"
            )

    return generated_code
