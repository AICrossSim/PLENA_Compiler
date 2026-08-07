"""Shared emitter for the x*sigmoid(k*x) activation family.

SiLU and GELU differ only in whether the input is pre-scaled before the
sigmoid: SiLU(x) = x*sigmoid(x), GELU(x) ~= x*sigmoid(1.702*x). Both then run
the identical negate / exp / +1 / reciprocal / multiply sequence over the same
register assignment, so they share one emitter parameterised by the optional
pre-scale constant.
"""

from __future__ import annotations

from ._imm import load_large_int_str as _load_large_int


def emit_sigmoid_activation(
    *,
    banner: str,
    const_one_fp_address: int,
    pre_scale_fp_address: int | None,
    alive_registers: list[int],
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int,
) -> str:
    """Emit ``x * sigmoid(k*x)`` in place over ``batch_size*hidden_dim`` elements.

    ``pre_scale_fp_address`` is the FP SRAM address holding ``k``. Pass None for
    k == 1 (SiLU), which skips both the constant load and the scaling multiply.
    """
    act_addr = alive_registers[0]
    scratchpad_addr = alive_registers[1]
    loop_reg = alive_registers[2]

    num_vectors = (batch_size * hidden_dim) // vlen

    generated_code = banner
    generated_code += _load_large_int(act_addr, activation_base_address)
    generated_code += _load_large_int(scratchpad_addr, scratchpad_base_address)

    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"
    if pre_scale_fp_address is not None:
        generated_code += f"S_LD_FP f2, gp0, {pre_scale_fp_address}\n"

    generated_code += f"C_LOOP_START gp{loop_reg}, {num_vectors}\n"

    # k*x, when k != 1. The negation below then reads the scaled value out of
    # the scratchpad instead of reading the activation directly.
    if pre_scale_fp_address is not None:
        generated_code += f"V_MUL_VF gp{scratchpad_addr}, gp{act_addr}, f2, 0\n"
        negate_src = scratchpad_addr
    else:
        negate_src = act_addr

    # -k*x (negate against f0=0 with the reverse-order flag)
    generated_code += f"V_SUB_VF gp{scratchpad_addr}, gp{negate_src}, f0, 0, 1\n"
    # exp(-k*x)
    generated_code += f"V_EXP_V gp{scratchpad_addr}, gp{scratchpad_addr}, 0\n"
    # 1 + exp(-k*x)
    generated_code += f"V_ADD_VF gp{scratchpad_addr}, gp{scratchpad_addr}, f1, 0\n"
    # 1 / (1 + exp(-k*x)) = sigmoid(k*x)
    generated_code += f"V_RECI_V gp{scratchpad_addr}, gp{scratchpad_addr}, 0\n"
    # x * sigmoid(k*x), stored in place
    generated_code += f"V_MUL_VV gp{act_addr}, gp{scratchpad_addr}, gp{act_addr}, 0\n"

    # Move to next vector
    generated_code += f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen}\n"

    generated_code += f"C_LOOP_END gp{loop_reg}\n"

    return generated_code
