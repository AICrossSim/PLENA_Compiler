from ._imm import load_large_int_str as _load_large_int


def elementwise_add_vram_asm(
    vlen: int,
    num_vectors: int,
    alive_registers: list[int],
    dst_base_address: int,
    src_base_address: int,
    vector_stride: int | None = None,
) -> str:
    """Generate VRAM-to-VRAM elementwise add assembly.

    This emits a reusable pattern for cases where both inputs already reside in
    VRAM and the result is written back in place to the destination region.
    """

    if len(alive_registers) < 2:
        raise ValueError("elementwise_add_vram_asm requires at least 2 GP registers")
    if vlen <= 0:
        raise ValueError("vlen must be positive")
    if num_vectors < 0:
        raise ValueError("num_vectors must be non-negative")

    step = vlen if vector_stride is None else vector_stride
    if step <= 0:
        raise ValueError("vector_stride must be positive")
    dst_reg = alive_registers[0]
    src_reg = alive_registers[1]

    generated_code = "; Elementwise add VRAM-to-VRAM generation\n"
    generated_code += _load_large_int(dst_reg, dst_base_address)
    generated_code += _load_large_int(src_reg, src_base_address)

    if len(alive_registers) >= 3:
        loop_reg = alive_registers[2]
        generated_code += f"C_LOOP_START gp{loop_reg}, {num_vectors}\n"
        generated_code += f"V_ADD_VV gp{dst_reg}, gp{dst_reg}, gp{src_reg}, 0\n"
        generated_code += f"S_ADDI_INT gp{dst_reg}, gp{dst_reg}, {step}\n"
        generated_code += f"S_ADDI_INT gp{src_reg}, gp{src_reg}, {step}\n"
        generated_code += f"C_LOOP_END gp{loop_reg}\n"
    else:
        for _ in range(num_vectors):
            generated_code += f"V_ADD_VV gp{dst_reg}, gp{dst_reg}, gp{src_reg}, 0\n"
            generated_code += f"S_ADDI_INT gp{dst_reg}, gp{dst_reg}, {step}\n"
            generated_code += f"S_ADDI_INT gp{src_reg}, gp{src_reg}, {step}\n"

    return generated_code


def elementwise_add_bias_vram_asm(
    vlen: int,
    num_hidden_vectors: int,
    seq_len: int,
    alive_registers: list[int],
    dst_base_address: int,
    bias_base_address: int,
    dst_vector_stride: int | None = None,
    bias_vector_stride: int | None = None,
) -> str:
    """Generate VRAM bias-broadcast add assembly.

    Intended for chunk-major [hidden_chunk, seq, vlen] style regions where a
    single bias vector is reused across all sequence positions for each hidden
    chunk. This avoids materializing a fully tiled bias buffer in VRAM.

    Loop structure:
      for h in range(num_hidden_vectors):
          for s in range(seq_len):
              dst[h, s, :] += bias[h, :]
    """

    if len(alive_registers) < 4:
        raise ValueError("elementwise_add_bias_vram_asm requires at least 4 GP registers")
    if vlen <= 0:
        raise ValueError("vlen must be positive")
    if num_hidden_vectors < 0:
        raise ValueError("num_hidden_vectors must be non-negative")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")

    dst_step = vlen if dst_vector_stride is None else dst_vector_stride
    bias_step = vlen if bias_vector_stride is None else bias_vector_stride
    if dst_step <= 0:
        raise ValueError("dst_vector_stride must be positive")
    if bias_step <= 0:
        raise ValueError("bias_vector_stride must be positive")

    dst_reg = alive_registers[0]
    bias_reg = alive_registers[1]
    outer_loop_reg = alive_registers[2]
    inner_loop_reg = alive_registers[3]

    generated_code = "; Elementwise add bias broadcast VRAM generation\n"
    generated_code += _load_large_int(dst_reg, dst_base_address)
    generated_code += _load_large_int(bias_reg, bias_base_address)

    generated_code += f"C_LOOP_START gp{outer_loop_reg}, {num_hidden_vectors}\n"
    generated_code += f"C_LOOP_START gp{inner_loop_reg}, {seq_len}\n"
    generated_code += f"V_ADD_VV gp{dst_reg}, gp{dst_reg}, gp{bias_reg}, 0\n"
    generated_code += f"S_ADDI_INT gp{dst_reg}, gp{dst_reg}, {dst_step}\n"
    generated_code += f"C_LOOP_END gp{inner_loop_reg}\n"
    generated_code += f"S_ADDI_INT gp{bias_reg}, gp{bias_reg}, {bias_step}\n"
    generated_code += f"C_LOOP_END gp{outer_loop_reg}\n"

    return generated_code


def elementwise_mul_bias_vram_asm(
    vlen: int,
    num_hidden_vectors: int,
    seq_len: int,
    alive_registers: list[int],
    dst_base_address: int,
    bias_base_address: int,
    dst_vector_stride: int | None = None,
    bias_vector_stride: int | None = None,
) -> str:
    """Generate VRAM bias-broadcast multiply assembly.

    Intended for chunk-major [hidden_chunk, seq, vlen] style regions where a
    single multiplicative vector is reused across all sequence positions for
    each hidden chunk.

    Loop structure:
      for h in range(num_hidden_vectors):
          for s in range(seq_len):
              dst[h, s, :] *= bias[h, :]
    """

    if len(alive_registers) < 4:
        raise ValueError("elementwise_mul_bias_vram_asm requires at least 4 GP registers")
    if vlen <= 0:
        raise ValueError("vlen must be positive")
    if num_hidden_vectors < 0:
        raise ValueError("num_hidden_vectors must be non-negative")
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")

    dst_step = vlen if dst_vector_stride is None else dst_vector_stride
    bias_step = vlen if bias_vector_stride is None else bias_vector_stride
    if dst_step <= 0:
        raise ValueError("dst_vector_stride must be positive")
    if bias_step <= 0:
        raise ValueError("bias_vector_stride must be positive")

    dst_reg = alive_registers[0]
    bias_reg = alive_registers[1]
    outer_loop_reg = alive_registers[2]
    inner_loop_reg = alive_registers[3]

    generated_code = "; Elementwise mul bias broadcast VRAM generation\n"
    generated_code += _load_large_int(dst_reg, dst_base_address)
    generated_code += _load_large_int(bias_reg, bias_base_address)

    generated_code += f"C_LOOP_START gp{outer_loop_reg}, {num_hidden_vectors}\n"
    generated_code += f"C_LOOP_START gp{inner_loop_reg}, {seq_len}\n"
    generated_code += f"V_MUL_VV gp{dst_reg}, gp{dst_reg}, gp{bias_reg}, 0\n"
    generated_code += f"S_ADDI_INT gp{dst_reg}, gp{dst_reg}, {dst_step}\n"
    generated_code += f"C_LOOP_END gp{inner_loop_reg}\n"
    generated_code += f"S_ADDI_INT gp{bias_reg}, gp{bias_reg}, {bias_step}\n"
    generated_code += f"C_LOOP_END gp{outer_loop_reg}\n"

    return generated_code