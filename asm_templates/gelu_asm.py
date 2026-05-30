from ._imm import load_large_int_str as _load_large_int


def gelu_asm(
    const_one_fp_address: int,
    const_1702_fp_address: int,
    alive_registers: list[int],
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int,
) -> str:
    """
    Generate assembly code for GELU activation using sigmoid approximation.

    Sigmoid approximation: GELU(x) = x * sigmoid(1.702 * x)

    The approximation simplifies to:
    GELU(x) ≈ x * (1 / (1 + exp(-1.702 * x)))

    Args:
        const_one_fp_address: FP SRAM address containing constant 1.0
        const_1702_fp_address: FP SRAM address containing constant 1.702
        alive_registers: List of available integer registers
        activation_base_address: VRAM base address for input activations
        scratchpad_base_address: VRAM base address for intermediate results
        vlen: Vector length (number of elements per vector)
        batch_size: Batch size dimension
        hidden_dim: Hidden dimension size

    Returns:
        Generated assembly code string
    """
    act_addr = alive_registers[0]
    scratchpad_addr = alive_registers[1]
    loop_reg = alive_registers[2]

    num_vectors = (batch_size * hidden_dim) // vlen

    generated_code = "; GELU Activation Generation\n"
    generated_code += _load_large_int(act_addr, activation_base_address)
    generated_code += _load_large_int(scratchpad_addr, scratchpad_base_address)

    # Load constants into FP registers
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"
    generated_code += f"S_LD_FP f2, gp0, {const_1702_fp_address}\n"

    generated_code += f"C_LOOP_START gp{loop_reg}, {num_vectors}\n"

    # GELU computation: x * sigmoid(1.702 * x) = x * (1 / (1 + exp(-1.702 * x)))
    # Step 1: 1.702 * x
    generated_code += f"V_MUL_VF gp{scratchpad_addr}, gp{act_addr}, f2, 0\n"
    # Step 2: -1.702 * x (negate using f0=0, with reverse order flag)
    generated_code += f"V_SUB_VF gp{scratchpad_addr}, gp{scratchpad_addr}, f0, 0, 1\n"
    # Step 3: exp(-1.702 * x)
    generated_code += f"V_EXP_V gp{scratchpad_addr}, gp{scratchpad_addr}, 0\n"
    # Step 4: 1 + exp(-1.702 * x)
    generated_code += f"V_ADD_VF gp{scratchpad_addr}, gp{scratchpad_addr}, f1, 0\n"
    # Step 5: 1 / (1 + exp(-1.702 * x)) = sigmoid(1.702 * x)
    generated_code += f"V_RECI_V gp{scratchpad_addr}, gp{scratchpad_addr}, 0\n"
    # Step 6: x * sigmoid(1.702 * x) = gelu(x), store in-place
    generated_code += f"V_MUL_VV gp{act_addr}, gp{scratchpad_addr}, gp{act_addr}, 0\n"

    # Move to next vector
    generated_code += f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen}\n"

    generated_code += f"C_LOOP_END gp{loop_reg}\n"

    return generated_code


def gelu_tanh_asm(
        const_one_fp_address: int,
        const_half_fp_address: int,
        const_cubic_fp_address: int,
        const_sqrt_2_over_pi_fp_address: int,
        alive_registers: list[int],
        activation_base_address: int,
        scratchpad0_base_address: int,
        scratchpad1_base_address: int,
        vlen: int,
        batch_size: int,
        hidden_dim: int,
) -> str:
        """Generate assembly code for GELU tanh approximation.

        Uses the common approximation:
            GELU(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

        The tanh term is implemented via exponentials to match available ISA ops:
            tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)

        Args:
                const_one_fp_address: FP SRAM slot containing 1.0
                const_half_fp_address: FP SRAM slot containing 0.5
                const_cubic_fp_address: FP SRAM slot containing 0.044715
                const_sqrt_2_over_pi_fp_address: FP SRAM slot containing sqrt(2/pi)
                alive_registers: Available GP registers (needs at least 4)
                activation_base_address: VRAM base address for input/output activations
                scratchpad0_base_address: VRAM base for first scratch vector region
                scratchpad1_base_address: VRAM base for second scratch vector region
                vlen: Vector length in elements
                batch_size: Batch dimension
                hidden_dim: Hidden dimension size

        Returns:
                Generated assembly code string
        """
        act_addr = alive_registers[0]
        scratch0_addr = alive_registers[1]
        scratch1_addr = alive_registers[2]
        loop_reg = alive_registers[3]

        num_vectors = (batch_size * hidden_dim) // vlen

        generated_code = "; GELU (tanh approximation) Activation Generation\n"
        generated_code += _load_large_int(act_addr, activation_base_address)
        generated_code += _load_large_int(scratch0_addr, scratchpad0_base_address)
        generated_code += _load_large_int(scratch1_addr, scratchpad1_base_address)

        # Constants used by the approximation.
        generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"
        generated_code += f"S_LD_FP f2, gp0, {const_half_fp_address}\n"
        generated_code += f"S_LD_FP f3, gp0, {const_cubic_fp_address}\n"
        generated_code += f"S_LD_FP f4, gp0, {const_sqrt_2_over_pi_fp_address}\n"

        generated_code += f"C_LOOP_START gp{loop_reg}, {num_vectors}\n"

        # scratch1 = x^2
        generated_code += f"V_MUL_VV gp{scratch1_addr}, gp{act_addr}, gp{act_addr}, 0\n"
        # scratch1 = x^3
        generated_code += f"V_MUL_VV gp{scratch1_addr}, gp{scratch1_addr}, gp{act_addr}, 0\n"
        # scratch1 = 0.044715 * x^3
        generated_code += f"V_MUL_VF gp{scratch1_addr}, gp{scratch1_addr}, f3, 0\n"
        # scratch1 = x + 0.044715 * x^3
        generated_code += f"V_ADD_VV gp{scratch1_addr}, gp{scratch1_addr}, gp{act_addr}, 0\n"
        # scratch0 = z = sqrt(2/pi) * (x + 0.044715*x^3)
        generated_code += f"V_MUL_VF gp{scratch0_addr}, gp{scratch1_addr}, f4, 0\n"

        # scratch1 = 2z (avoid extra constant by z + z)
        generated_code += f"V_ADD_VV gp{scratch1_addr}, gp{scratch0_addr}, gp{scratch0_addr}, 0\n"
        # scratch1 = exp(2z)
        generated_code += f"V_EXP_V gp{scratch1_addr}, gp{scratch1_addr}, 0\n"
        # scratch1 = num = exp(2z) - 1
        generated_code += f"V_SUB_VF gp{scratch1_addr}, gp{scratch1_addr}, f1, 0, 0\n"

        # scratch0 = den = exp(2z) + 1 = (exp(2z)-1) + 2
        generated_code += f"V_ADD_VF gp{scratch0_addr}, gp{scratch1_addr}, f1, 0\n"
        generated_code += f"V_ADD_VF gp{scratch0_addr}, gp{scratch0_addr}, f1, 0\n"
        # scratch0 = 1/den
        generated_code += f"V_RECI_V gp{scratch0_addr}, gp{scratch0_addr}, 0\n"
        # scratch0 = tanh(z) = num * (1/den)
        generated_code += f"V_MUL_VV gp{scratch0_addr}, gp{scratch1_addr}, gp{scratch0_addr}, 0\n"

        # scratch0 = 1 + tanh(z)
        generated_code += f"V_ADD_VF gp{scratch0_addr}, gp{scratch0_addr}, f1, 0\n"
        # scratch1 = 0.5 * x
        generated_code += f"V_MUL_VF gp{scratch1_addr}, gp{act_addr}, f2, 0\n"
        # act = 0.5*x*(1+tanh(z))
        generated_code += f"V_MUL_VV gp{act_addr}, gp{scratch1_addr}, gp{scratch0_addr}, 0\n"

        generated_code += f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen}\n"
        generated_code += f"S_ADDI_INT gp{scratch0_addr}, gp{scratch0_addr}, {vlen}\n"
        generated_code += f"S_ADDI_INT gp{scratch1_addr}, gp{scratch1_addr}, {vlen}\n"
        generated_code += f"C_LOOP_END gp{loop_reg}\n"

        return generated_code
