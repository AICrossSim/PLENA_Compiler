from __future__ import annotations

import math

from ._imm import load_large_int_str as _load_large_int


def preload_act_asm(
    vlen: int,
    preload_len: int,
    batch: int,
    hidden_size: int,
    act_vram_offset: int,
    alive_registers: list[int],
    activation_offset_reg: int,
    stride_size=None,
    vram_stride_mult: int = 1,
    storage_precision: int = 1,
    element_bits: int | None = None,
    element_plane_bytes: int | None = None,
    precision: int = 0,
) -> str:
    """Preload activation from HBM to VRAM. Layout: (hidden//mlen, batch, mlen)."""
    generated_code = "; Preload Activation Generation \n"
    a_actual_register = alive_registers[0]
    set_stride_register = alive_registers[1]
    result_register = alive_registers[2]
    outer_loop_register = alive_registers[3]
    inner_loop_register = alive_registers[4]

    if element_bits is None:
        element_bits = storage_precision * 8
    if precision not in (0, 1):
        raise ValueError("precision must be 0 (activation) or 1 (key/value)")
    if element_bits <= 0:
        raise ValueError("element_bits must be positive")

    def physical_bytes(elements: int) -> int:
        bits = elements * element_bits
        if bits % 8:
            raise ValueError("HBM element offsets must be byte aligned")
        return bits // 8

    stride_elements = vlen if stride_size is None else stride_size
    stride_bytes = physical_bytes(stride_elements)
    if element_plane_bytes is None:
        element_plane_bytes = physical_bytes(hidden_size * batch)

    # Set scale offset
    generated_code += _load_large_int(a_actual_register, element_plane_bytes)
    generated_code += f"C_SET_SCALE_REG gp{a_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, 0 \n"
    generated_code += _load_large_int(result_register, act_vram_offset)
    load_amount_per_hidden = math.ceil(hidden_size / vlen)

    if batch == 1:
        # Each H_PREFETCH_V loads preload_len rows (vlen * preload_len elements)
        # HBM offset should increment by the same amount as VSRAM offset
        elements_per_prefetch = vlen * preload_len
        bytes_per_prefetch = physical_bytes(elements_per_prefetch)
        vram_step = elements_per_prefetch * vram_stride_mult
        for i in range(math.ceil(hidden_size / elements_per_prefetch)):
            generated_code += (
                f"H_PREFETCH_V gp{result_register}, gp{a_actual_register}, "
                f"a{activation_offset_reg}, 0, {precision} \n"
            )
            generated_code += f"S_ADDI_INT gp{result_register}, gp{result_register}, {vram_step} \n"
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {bytes_per_prefetch} \n"
    else:
        generated_code += f"S_ADDI_INT gp{set_stride_register}, gp0, {stride_bytes} \n"
        generated_code += f"C_SET_STRIDE_REG gp{set_stride_register} \n"
        a_offset_register = set_stride_register
        generated_code += f"C_LOOP_START gp{outer_loop_register}, {load_amount_per_hidden} \n"
        generated_code += f"S_ADDI_INT gp{a_offset_register}, gp{a_actual_register}, 0 \n"
        if batch > preload_len:
            generated_code += f"C_LOOP_START gp{inner_loop_register}, {math.ceil(batch / preload_len)} \n"
        generated_code += f"H_PREFETCH_V gp{result_register}, gp{a_offset_register}, a{activation_offset_reg}, 1, {precision} \n"
        generated_code += (
            f"S_ADDI_INT gp{result_register}, gp{result_register}, {vlen * preload_len * vram_stride_mult} \n"
        )
        if batch > preload_len:
            batch_step_bytes = physical_bytes(hidden_size * preload_len)
            generated_code += f"S_ADDI_INT gp{a_offset_register}, gp{a_offset_register}, {batch_step_bytes} \n"
            generated_code += f"C_LOOP_END gp{inner_loop_register} \n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {physical_bytes(vlen)} \n"
        generated_code += f"C_LOOP_END gp{outer_loop_register} \n"
    return generated_code
