from __future__ import annotations

import math

from ._imm import load_large_int_str as _load_large_int


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
    generated_code += _load_large_int(vram_reg, act_vram_offset)
    # Initialize HBM offset to 0
    generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp0, 0\n"
    # HBM MX formats store scale bytes after the element payload.  H_STORE_V
    # uses C_SET_SCALE_REG as the scale-section base offset; do not inherit a
    # stale value from a previous HBM load/store.
    generated_code += _load_large_int(set_stride_register, batch * hidden_size)
    generated_code += f"C_SET_SCALE_REG gp{set_stride_register}\n"

    if batch == 1:
        # Simple case: no stride needed, store sequentially
        elements_per_store = vlen * store_amount
        for i in range(math.ceil(hidden_size / elements_per_store)):
            generated_code += f"H_STORE_V gp{vram_reg}, gp{hbm_offset_reg}, a{hbm_addr_reg}, 0, 0\n"
            generated_code += f"S_ADDI_INT gp{vram_reg}, gp{vram_reg}, {elements_per_store}\n"
            generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_offset_reg}, {elements_per_store}\n"
    else:
        # Set stride register (HBM row stride = hidden_size)
        generated_code += _load_large_int(set_stride_register, stride_len)
        generated_code += f"C_SET_STRIDE_REG gp{set_stride_register}\n"
        hbm_base_reg = set_stride_register  # reuse after stride is set

        # Delay for V controller to be ready for writes after preload reads
        for _ in range(10):
            generated_code += "S_ADDI_INT gp0, gp0, 0\n"

        # Unrolled: C_LOOP_START/C_LOOP_END not implemented in RTL.
        # Correct operand order: H_STORE_V gp(vram), gp(hbm_offset), a(addr_reg), stride, flags.
        # 20 NOPs between H_STORE_V for V controller timing.
        if batch > store_amount:
            inner_iters = math.ceil(batch / store_amount)
            for _outer_i in range(store_amount_per_hidden):
                generated_code += f"S_ADDI_INT gp{hbm_base_reg}, gp{hbm_offset_reg}, 0\n"
                for _ in range(10):
                    generated_code += "S_ADDI_INT gp0, gp0, 0\n"
                for _inner_i in range(inner_iters):
                    generated_code += f"H_STORE_V a{hbm_addr_reg}, gp{vram_reg}, gp{hbm_base_reg}, 1, 0\n"
                    # 2 NOPs: delay gp3 increment so late_rdata1_d1 captures pre-increment value
                    # when H_STORE_V reaches mem_stage (regfile write at exe overlaps late read).
                    generated_code += "S_ADDI_INT gp0, gp0, 0\n"
                    generated_code += "S_ADDI_INT gp0, gp0, 0\n"
                    generated_code += f"S_ADDI_INT gp{vram_reg}, gp{vram_reg}, {vlen * store_amount}\n"
                    # 3 NOPs: delay S_ADDI on hbm_base_reg so H_STORE_V passes mem_stage before
                    # gp2 changes.
                    for _ in range(3):
                        generated_code += "S_ADDI_INT gp0, gp0, 0\n"
                    generated_code += f"S_ADDI_INT gp{hbm_base_reg}, gp{hbm_base_reg}, {hidden_size * store_amount}\n"
                    for _ in range(15):
                        generated_code += "S_ADDI_INT gp0, gp0, 0\n"
                generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_offset_reg}, {vlen}\n"
        else:
            for _outer_i in range(store_amount_per_hidden):
                generated_code += f"S_ADDI_INT gp{hbm_base_reg}, gp{hbm_offset_reg}, 0\n"
                generated_code += f"H_STORE_V a{hbm_addr_reg}, gp{vram_reg}, gp{hbm_base_reg}, 1, 0\n"
                generated_code += f"S_ADDI_INT gp{vram_reg}, gp{vram_reg}, {vlen * store_amount}\n"
                generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_offset_reg}, {vlen}\n"
                for _ in range(20):
                    generated_code += "S_ADDI_INT gp0, gp0, 0\n"

    return generated_code
