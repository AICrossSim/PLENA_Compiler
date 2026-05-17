from ._imm import load_large_int_str as _load_large_int


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

    Stride convention (IMPORTANT): the ``offset`` GP register that advances
    between VLEN chunks is counted in *data bytes only*, not data+scale.  The
    Rust emulator's H_PREFETCH_V implementation
    (``transactional_emulator/src/main.rs:2031-2037``) auto-derives the scale
    byte offset from the data byte offset via
    ``scale_offset = data_offset / (elem_bits * block / scale_bits)``,
    which means the scale pointer is internal to the emulator.  Advancing
    ``offset`` by data+scale bytes would cause the derived scale offset to
    double-count.  Caller must therefore pass ``voc_table_row_bytes =
    hidden_size * elem_bytes`` (the on-disk *data*-only stride, irrespective of
    how much scale metadata is interleaved in HBM).

    Cost: ``hidden_size // vlen`` H_PREFETCH_V + a handful of S_ADDI_INT per
    token. No MRAM usage, no M_MM.

    Args:
        vlen: Vector lane width in elements (matches VLEN).
        batch: Number of tokens to embed; must equal ``len(input_ids)``.  For a
            decoder LLM this is ``batch_size * seq_len`` so the VRAM output
            spans every token in the sequence.
        hidden_size: Embedding dimension. Must be a multiple of ``vlen``.
        alive_registers: Free general-purpose registers. We consume the first
            two (vram-dest pointer, hbm-byte-offset pointer).
        activation_base_address: VRAM element offset where token 0's embedding
            row lands. Subsequent tokens stack at +hidden_size elements each.
        voc_table_base_addr_reg_index: Index of the ``a<N>`` HBM address
            register holding the vocab-table base pointer.
        input_ids: Token ids emitted at codegen time (runtime token ids would
            need a scalar-indexed HBM load primitive we don't have).
        voc_table_row_bytes: Data bytes per vocab row in HBM.  Must be
            ``hidden_size * elem_bytes`` (NOT include scale bytes — the
            emulator auto-derives the scale offset from the data offset).

    Returns:
        str: Generated assembly.
    """
    assert len(input_ids) == batch, f"Input IDs length {len(input_ids)} must match batch {batch}"
    assert hidden_size % vlen == 0, f"hidden_size {hidden_size} must be multiple of VLEN {vlen}"
    assert len(alive_registers) >= 2, "embedding_asm needs at least 2 alive registers"

    rows_per_token = hidden_size // vlen
    hbm_bytes_per_vlen_chunk = voc_table_row_bytes // rows_per_token
    assert hbm_bytes_per_vlen_chunk * rows_per_token == voc_table_row_bytes, (
        f"stride mismatch: {hbm_bytes_per_vlen_chunk} * {rows_per_token} "
        f"!= {voc_table_row_bytes}"
    )
    assert hbm_bytes_per_vlen_chunk < (1 << 18), (
        f"hbm_bytes_per_vlen_chunk ({hbm_bytes_per_vlen_chunk}) exceeds S_ADDI_INT 18-bit immediate. "
        f"Use addi_large_int_str for chunk advance."
    )
    assert vlen < (1 << 18), f"vlen ({vlen}) exceeds S_ADDI_INT 18-bit immediate"

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
        generated_code += _load_large_int(vram_dest_reg, vram_start)
        generated_code += _load_large_int(hbm_offset_reg, hbm_byte_offset_start)
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
