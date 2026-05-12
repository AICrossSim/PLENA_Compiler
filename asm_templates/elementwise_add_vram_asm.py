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
    if num_vectors < 0:
        raise ValueError("num_vectors must be non-negative")

    step = vlen if vector_stride is None else vector_stride
    dst_reg = alive_registers[0]
    src_reg = alive_registers[1]

    generated_code = "; Elementwise add VRAM-to-VRAM generation\n"
    generated_code += _load_large_int(dst_reg, dst_base_address)
    generated_code += _load_large_int(src_reg, src_base_address)

    for _ in range(num_vectors):
        generated_code += f"V_ADD_VV gp{dst_reg}, gp{dst_reg}, gp{src_reg}, 0\n"
        generated_code += f"S_ADDI_INT gp{dst_reg}, gp{dst_reg}, {step}\n"
        generated_code += f"S_ADDI_INT gp{src_reg}, gp{src_reg}, {step}\n"

    return generated_code