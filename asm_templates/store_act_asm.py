import math
from typing import List, Optional

IMM2_BOUND = 2**18

def store_act_asm(
    vlen: int,
    batch: int,
    hidden_size: int,
    alive_registers: List[int],
    act_vram_offset: int,
    hbm_addr_reg: int,
    stride_size: Optional[int] = None,
    store_amount: int = 4,
) -> str:
    """
    Generates assembly code for storing activation from VRAM back to HBM.
    This is the reverse operation of preload_act_asm.

    Format:
        VRAM: [batch, mlen, hidden/mlen] - hardware block format
        HBM:  [batch, hidden_size] - row-major contiguous

    The hardware H_STORE_V instruction handles format conversion automatically
    when using stride mode (rstride=1), mirroring H_PREFETCH_V behavior.

    VRAM address increments linearly. HBM uses stride to skip between rows.

    H_STORE_V rd, rs1, rs2, rstride, precision
        rd:        register containing VRAM source address
        rs1:       register containing HBM offset
        rs2:       HBM address register index (a0-a7)
        rstride:   stride register selector (0 = no stride, 1 = use STRIDE_REG)
        precision: 0 = Activation, 1 = KeyValue

    Args:
        vlen:             vector length (default 64)
        batch:            batch size
        hidden_size:      hidden dimension size
        alive_registers:  list of 5 available GP registers
        act_vram_offset:  source base address in VRAM
        hbm_addr_reg:     HBM address register index (a0-a7)
        stride_size:      HBM row stride (defaults to hidden_size)
        store_amount:     rows per H_STORE_V (HBM_V_Writeback_Amount, default 4)

    Returns:
        Generated ISA code string.
    """
    generated_code = "; Store Activation Generation\n"

    hbm_offset_reg      = alive_registers[0]
    set_stride_register  = alive_registers[1]
    vram_reg             = alive_registers[2]
    outer_loop_register  = alive_registers[3]
    inner_loop_register  = alive_registers[4]

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

        assert batch * hidden_size <= IMM2_BOUND, f"batch * hidden_size must be less than {IMM2_BOUND}"

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
