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
    precision: int = 0,
    element_bits: int = 8,
    element_plane_bytes: int | None = None,
    initial_hbm_offset_bytes: int = 0,
) -> str:
    """Store activation from VRAM back to HBM (reverse of preload_act_asm).

    VRAM layout: [batch, mlen, hidden/mlen] -> HBM: [batch, hidden_size] row-major.
    Uses H_STORE_V with stride mode for format conversion.
    """
    if precision not in (0, 1):
        raise ValueError("precision must be 0 (activation) or 1 (key/value)")
    generated_code = "; Store Vector Data Generation\n"

    hbm_offset_reg = alive_registers[0]
    set_stride_register = alive_registers[1]
    vram_reg = alive_registers[2]
    outer_loop_register = alive_registers[3]
    inner_loop_register = alive_registers[4]

    if element_bits <= 0:
        raise ValueError("element_bits must be positive")
    if initial_hbm_offset_bytes < 0:
        raise ValueError("initial_hbm_offset_bytes must be non-negative")

    def physical_bytes(elements: int) -> int:
        bits = elements * element_bits
        if bits % 8:
            raise ValueError("HBM element offsets must be byte aligned")
        return bits // 8

    stride_elements = hidden_size if stride_size is None else stride_size
    stride_bytes = physical_bytes(stride_elements)
    store_amount_per_hidden = math.ceil(hidden_size / vlen)
    if element_plane_bytes is None:
        element_plane_bytes = physical_bytes(batch * hidden_size)
    if element_plane_bytes <= 0:
        raise ValueError("element_plane_bytes must be positive")

    # Initialize VRAM source address
    generated_code += _load_large_int(vram_reg, act_vram_offset)
    generated_code += _load_large_int(
        hbm_offset_reg,
        initial_hbm_offset_bytes,
    )
    # HBM MX formats store scale bytes after the element payload.  H_STORE_V
    # uses C_SET_SCALE_REG as the scale-section base offset; do not inherit a
    # stale value from a previous HBM load/store.
    generated_code += _load_large_int(set_stride_register, element_plane_bytes)
    generated_code += f"C_SET_SCALE_REG gp{set_stride_register}\n"

    if batch == 1:
        # Simple case: no stride needed, store sequentially
        elements_per_store = vlen * store_amount
        bytes_per_store = physical_bytes(elements_per_store)
        for i in range(math.ceil(hidden_size / elements_per_store)):
            generated_code += (
                f"H_STORE_V gp{vram_reg}, gp{hbm_offset_reg}, "
                f"a{hbm_addr_reg}, 0, {precision}\n"
            )
            generated_code += f"S_ADDI_INT gp{vram_reg}, gp{vram_reg}, {elements_per_store}\n"
            generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_offset_reg}, {bytes_per_store}\n"
    else:
        # Set stride register (HBM row stride = hidden_size)
        generated_code += _load_large_int(set_stride_register, stride_bytes)
        generated_code += f"C_SET_STRIDE_REG gp{set_stride_register}\n"
        hbm_base_reg = set_stride_register  # reuse after stride is set

        # Outer loop: iterate over column blocks (hidden_size / vlen)
        generated_code += f"C_LOOP_START gp{outer_loop_register}, {store_amount_per_hidden}\n"
        generated_code += f"S_ADDI_INT gp{hbm_base_reg}, gp{hbm_offset_reg}, 0\n"

        if batch > store_amount:
            # Inner loop: iterate over batch blocks
            generated_code += f"C_LOOP_START gp{inner_loop_register}, {math.ceil(batch / store_amount)}\n"

        generated_code += (
            f"H_STORE_V gp{vram_reg}, gp{hbm_base_reg}, "
            f"a{hbm_addr_reg}, 1, {precision}\n"
        )
        generated_code += f"S_ADDI_INT gp{vram_reg}, gp{vram_reg}, {vlen * store_amount}\n"

        if batch > store_amount:
            batch_step_bytes = physical_bytes(hidden_size * store_amount)
            generated_code += f"S_ADDI_INT gp{hbm_base_reg}, gp{hbm_base_reg}, {batch_step_bytes}\n"
            generated_code += f"C_LOOP_END gp{inner_loop_register}\n"

        # Move to next column block in HBM
        generated_code += f"S_ADDI_INT gp{hbm_offset_reg}, gp{hbm_offset_reg}, {physical_bytes(vlen)}\n"
        generated_code += f"C_LOOP_END gp{outer_loop_register}\n"

    return generated_code
