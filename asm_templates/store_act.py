import math


def store_act_asm(
    vlen: int,
    batch: int,
    hidden_size: int,
    act_vram_offset: int,
    alive_registers: list[int],
    hbm_addr_reg: int,
    stride_size=None,
    store_amount: int = 4,
) -> str:
    """
    Generate assembly for storing an activation matrix from VRAM back to HBM.

    Assumed layouts:
    - VRAM: column-block major, consistent with `preload_act_asm`
    - HBM: row-major contiguous

    This mirrors the current preload path closely enough for compiler-side ISA generation.
    """
    generated_code = "; Store Activation Generation\n"

    hbm_offset_register = alive_registers[0]
    set_stride_register = alive_registers[1]
    vram_register = alive_registers[2]
    outer_loop_register = alive_registers[3]
    inner_loop_register = alive_registers[4]

    stride_len = vlen if stride_size is None else stride_size
    store_width = vlen * store_amount
    hidden_blocks = math.ceil(hidden_size / vlen)

    generated_code += f"S_ADDI_INT gp{hbm_offset_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{vram_register}, gp0, {act_vram_offset}\n"

    if batch == 1:
        for _ in range(math.ceil(hidden_size / store_width)):
            generated_code += f"H_STORE_V gp{vram_register}, gp{hbm_offset_register}, a{hbm_addr_reg}, 0\n"
            generated_code += f"S_ADDI_INT gp{vram_register}, gp{vram_register}, {store_width}\n"
            generated_code += f"S_ADDI_INT gp{hbm_offset_register}, gp{hbm_offset_register}, {store_width}\n"
        return generated_code

    generated_code += f"S_ADDI_INT gp{set_stride_register}, gp0, {stride_len}\n"
    generated_code += f"C_SET_STRIDE_REG gp{set_stride_register}\n"
    generated_code += f"C_LOOP_START gp{outer_loop_register}, {hidden_blocks}\n"
    generated_code += f"S_ADDI_INT gp{set_stride_register}, gp{hbm_offset_register}, 0\n"
    if batch > store_amount:
        generated_code += f"C_LOOP_START gp{inner_loop_register}, {math.ceil(batch / store_amount)}\n"
    generated_code += f"H_STORE_V gp{vram_register}, gp{set_stride_register}, a{hbm_addr_reg}, 0\n"
    generated_code += f"S_ADDI_INT gp{vram_register}, gp{vram_register}, {store_width}\n"
    if batch > store_amount:
        generated_code += (
            f"S_ADDI_INT gp{set_stride_register}, gp{set_stride_register}, {hidden_size * store_amount}\n"
        )
        generated_code += f"C_LOOP_END gp{inner_loop_register}\n"
    generated_code += f"S_ADDI_INT gp{hbm_offset_register}, gp{hbm_offset_register}, {vlen}\n"
    generated_code += f"C_LOOP_END gp{outer_loop_register}\n"
    return generated_code
