import math


def store_act_asm(
    vlen: int,
    batch: int,
    hidden_size: int,
    alive_registers: list[int],
    act_vram_offset: int,
    hbm_addr_reg: int,
    stride_size: int | None = None,
    store_amount: int = 4,
) -> str:
    """Store activation from VRAM back to HBM (reverse of preload_act_asm).

    VRAM layout: [batch, mlen, hidden/mlen] -> HBM: [batch, hidden_size] row-major.
    Uses H_STORE_V with stride mode for format conversion.
    """
    generated_code = "; Store Activation Generation\n"

    hbm_offset_reg = alive_registers[0]
    set_stride_register = alive_registers[1]
    vram_reg = alive_registers[2]
    outer_loop_register = alive_registers[3]
    inner_loop_register = alive_registers[4]

    stride_len = hidden_size if stride_size is None else stride_size
    store_amount_per_hidden = math.ceil(hidden_size / vlen)

    # Initialize VRAM source address
    generated_code += f"S_ADDI_INT gp{vram_reg}, gp0, {act_vram_offset}\n"
    # Initialize HBM offset to 0
    generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp0, 0\n"

    if batch == 1:
        # Simple case: no stride needed, store sequentially
        elements_per_store = vlen * store_amount
        for i in range(math.ceil(hidden_size / elements_per_store)):
            generated_code += f"H_STORE_V gp{vram_reg}, gp{hbm_offset_reg}, a{hbm_addr_reg}, 0, 0\n"
            generated_code += f"S_ADDI_INT gp{vram_reg}, gp{vram_reg}, {elements_per_store}\n"
            generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_offset_reg}, {elements_per_store}\n"
    else:
        # Set stride register (HBM row stride = hidden_size)
        generated_code += f"S_ADDI_INT gp{set_stride_register}, gp0, {stride_len}\n"
        generated_code += f"C_SET_STRIDE_REG gp{set_stride_register}\n"
        hbm_base_reg = set_stride_register  # reuse after stride is set

        # Outer loop: iterate over column blocks (hidden_size / vlen)
        generated_code += f"C_LOOP_START gp{outer_loop_register}, {store_amount_per_hidden}\n"
        generated_code += f"S_ADDI_INT gp{hbm_base_reg}, gp{hbm_offset_reg}, 0\n"

        if batch > store_amount:
            # Inner loop: iterate over batch blocks
            generated_code += f"C_LOOP_START gp{inner_loop_register}, {math.ceil(batch / store_amount)}\n"

        generated_code += f"H_STORE_V gp{vram_reg}, gp{hbm_base_reg}, a{hbm_addr_reg}, 1, 0\n"
        generated_code += f"S_ADDI_INT gp{vram_reg}, gp{vram_reg}, {vlen * store_amount}\n"

        if batch > store_amount:
            generated_code += f"S_ADDI_INT gp{hbm_base_reg}, gp{hbm_base_reg}, {hidden_size * store_amount}\n"
            generated_code += f"C_LOOP_END gp{inner_loop_register}\n"

        # Move to next column block in HBM
        generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_offset_reg}, {vlen}\n"
        generated_code += f"C_LOOP_END gp{outer_loop_register}\n"

    return generated_code
