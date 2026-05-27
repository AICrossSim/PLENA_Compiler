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
) -> str:
    """Preload activation from HBM to VRAM. Layout: (hidden//mlen, batch, mlen)."""
    generated_code = "; Preload Activation Generation \n"
    a_actual_register = alive_registers[0]
    set_stride_register = alive_registers[1]
    result_register = alive_registers[2]
    outer_loop_register = alive_registers[3]
    inner_loop_register = alive_registers[4]

    stride_len = vlen if stride_size is None else stride_size

    # Set scale offset
    generated_code += _load_large_int(a_actual_register, hidden_size * batch)
    generated_code += f"C_SET_SCALE_REG gp{a_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, 0 \n"
    generated_code += _load_large_int(result_register, act_vram_offset)
    load_amount_per_hidden = math.ceil(hidden_size / vlen)

    if batch == 1:
        # Each H_PREFETCH_V loads preload_len rows (vlen * preload_len elements)
        # HBM offset should increment by the same amount as VSRAM offset
        elements_per_prefetch = vlen * preload_len
        vram_step = elements_per_prefetch * vram_stride_mult
        for i in range(math.ceil(hidden_size / elements_per_prefetch)):
            generated_code += (
                f"H_PREFETCH_V gp{result_register}, gp{a_actual_register}, a{activation_offset_reg}, 0, 0, 0 \n"
            )
            generated_code += f"S_ADDI_INT gp{result_register}, gp{result_register}, {vram_step} \n"
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {elements_per_prefetch} \n"
    else:
        generated_code += _load_large_int(set_stride_register, stride_len)
        generated_code += f"C_SET_STRIDE_REG gp{set_stride_register} \n"
        a_offset_register = set_stride_register
        # Unrolled: C_LOOP_START/C_LOOP_END not implemented in RTL.
        # Correct operand order: H_PREFETCH_V gp(dest), gp(offset), a(base), stride, flags.
        # 20 NOPs between H_PREFETCH_V for V controller timing.
        inner_iters = math.ceil(batch / preload_len) if batch > preload_len else 1
        for _outer_i in range(load_amount_per_hidden):
            generated_code += f"S_ADDI_INT gp{a_offset_register}, gp{a_actual_register}, 0 \n"
            # NOPs: let S_ADDI writeback + V controller be ready for first request
            for _ in range(10):
                generated_code += "S_ADDI_INT gp0, gp0, 0 \n"
            for _inner_i in range(inner_iters):
                generated_code += f"H_PREFETCH_V a{activation_offset_reg}, gp{result_register}, gp{a_offset_register}, 1, 0 \n"
                generated_code += (
                    f"S_ADDI_INT gp{result_register}, gp{result_register}, {vlen * preload_len * vram_stride_mult} \n"
                )
                if batch > preload_len:
                    generated_code += f"S_ADDI_INT gp{a_offset_register}, gp{a_offset_register}, {hidden_size * preload_len} \n"
                for _ in range(20):
                    generated_code += "S_ADDI_INT gp0, gp0, 0 \n"
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {vlen} \n"
    return generated_code
