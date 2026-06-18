from ._imm import load_large_int_str as _load_large_int


def rms_norm_asm(
    _eps_offset: int,
    reci_hid_offset: int,
    alive_registers: list[int],
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int,
    unroll: bool = True,
) -> str:
    """
    Generate assembly code for RMS normalization.
    """
    act_addr = alive_registers[0]
    scratchpad_addr = alive_registers[1]
    stats_addr = alive_registers[2]
    # Rolled path uses the spare 4th register (already allocated by normalize()) as the
    # C_LOOP counter. Only accessed when rolled, so unrolled callers may pass 3 registers.
    loop_addr = alive_registers[3] if not unroll else None

    generated_code = "; RMS Norm generation \n"
    generated_code += _load_large_int(scratchpad_addr, scratchpad_base_address)

    # Load eps into f1
    generated_code += f"S_LD_FP f1, gp0, {_eps_offset} \n"
    # Reset f2 as accumulator for reduction
    generated_code += "S_ADD_FP f2, f0, f0 \n"
    # Load the 1/hidden_dim into f3
    generated_code += f"S_LD_FP f3, gp0, {reci_hid_offset} \n"

    for batch in range(batch_size):
        # Set act_addr to start of current batch
        generated_code += _load_large_int(act_addr, activation_base_address + vlen * batch)
        # Set stats_addr to same position for iteration
        generated_code += _load_large_int(stats_addr, activation_base_address + vlen * batch)

        # First loop: compute sum of squares using stats_addr
        if unroll:
            for i in range(hidden_dim // vlen):
                # Compute square of the activation vector and summation
                generated_code += f"V_MUL_VV gp{scratchpad_addr}, gp{stats_addr}, gp{stats_addr}, 0 \n"
                generated_code += f"V_RED_SUM f2, gp{scratchpad_addr} \n"

                # Move stats pointer to next vector
                generated_code += f"S_ADDI_INT gp{stats_addr}, gp{stats_addr}, {vlen * batch_size} \n"
        else:
            generated_code += f"C_LOOP_START gp{loop_addr}, {hidden_dim // vlen} \n"
            generated_code += f"V_MUL_VV gp{scratchpad_addr}, gp{stats_addr}, gp{stats_addr}, 0 \n"
            generated_code += f"V_RED_SUM f2, gp{scratchpad_addr} \n"
            generated_code += f"S_ADDI_INT gp{stats_addr}, gp{stats_addr}, {vlen * batch_size} \n"
            generated_code += f"C_LOOP_END gp{loop_addr} \n"

        # Taking the avg
        generated_code += "S_MUL_FP f2, f2, f3 \n"

        # Plus epsilon
        generated_code += "S_ADD_FP f2, f2, f1 \n"

        # Compute square root
        generated_code += "S_SQRT_FP f2, f2 \n"

        # Compute reciprocal
        generated_code += "S_RECI_FP f2, f2 \n"

        # Second loop: normalize using act_addr
        if unroll:
            for i in range(hidden_dim // vlen):
                # Normalize the activation vector
                generated_code += f"V_MUL_VF gp{act_addr}, gp{act_addr}, f2, 0 \n"

                # Move to next vector
                generated_code += f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen * batch_size} \n"
        else:
            generated_code += f"C_LOOP_START gp{loop_addr}, {hidden_dim // vlen} \n"
            generated_code += f"V_MUL_VF gp{act_addr}, gp{act_addr}, f2, 0 \n"
            generated_code += f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen * batch_size} \n"
            generated_code += f"C_LOOP_END gp{loop_addr} \n"

        # Reset accumulator for next batch
        generated_code += "S_ADD_FP f2, f0, f0 \n"

    return generated_code


def layer_norm_asm(
    _eps_offset: int,
    reci_hid_offset: int,
    alive_registers: list[int],
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int,
    unroll: bool = True,
    fused_reduce: bool = False,
) -> str:
    """
    Generate assembly code for layer normalization.

    When ``fused_reduce`` is set, the statistics loop replaces the per-col-block
    pair of ``V_RED_SUM`` (one for sum(x), one for sum(x^2)) with lane-wise
    ``V_ADD_VV`` into two VRAM accumulators, then a SINGLE ``V_RED_SUM`` each at
    the end.  This collapses ``2 * (hidden_dim/vlen)`` 30-cyc reduces per row down
    to 2, trading them for cheaper 9-cyc vector adds.  It is numerically
    equivalent (only reorders an associative summation; ``V_RED_SUM`` cost is
    content-independent).  The caller (``normalize()``) must then provide 6 alive
    registers and a scratchpad of ``3 * vlen`` (scratch + acc_x + acc_x2).
    """
    act_addr = alive_registers[0]
    scratchpad_addr = alive_registers[1]
    stats_addr = alive_registers[2]
    # Rolled path uses the spare 4th register (already allocated by normalize()) as the
    # C_LOOP counter. Only accessed when rolled, so unrolled callers may pass 3 registers.
    loop_addr = alive_registers[3] if not unroll else None
    n_blocks = hidden_dim // vlen

    generated_code = "; Layer Norm generation \n"
    generated_code += _load_large_int(scratchpad_addr, scratchpad_base_address)

    # Load constants
    generated_code += f"S_LD_FP f1, gp0, {_eps_offset} \n"  # epsilon
    generated_code += "S_ADD_FP f2, f0, f0 \n"  # sum(x) accumulator
    generated_code += "S_ADD_FP f3, f0, f0 \n"  # sum(x^2) accumulator
    generated_code += f"S_LD_FP f4, gp0, {reci_hid_offset} \n"  # 1/hidden_dim

    if fused_reduce:
        # Two lane-wise VRAM accumulators live just past the scratchpad vector.
        accx_addr = alive_registers[4]
        accx2_addr = alive_registers[5]
        generated_code += _load_large_int(accx_addr, scratchpad_base_address + vlen)
        generated_code += _load_large_int(accx2_addr, scratchpad_base_address + 2 * vlen)

    for batch in range(batch_size):
        # Set act_addr to start of current batch
        generated_code += _load_large_int(act_addr, activation_base_address + vlen * batch)
        # Set stats_addr to same position for iteration
        generated_code += _load_large_int(stats_addr, activation_base_address + vlen * batch)

        # First loop: compute sum(x) and sum(x^2) using stats_addr
        if fused_reduce:
            # Zero the lane accumulators safely: (real block-0 activation) * 0 = 0.
            # Multiplying a loaded activation by f0 avoids reading uninitialised VRAM.
            generated_code += f"V_MUL_VF gp{accx_addr}, gp{stats_addr}, f0, 0 \n"
            generated_code += f"V_MUL_VF gp{accx2_addr}, gp{stats_addr}, f0, 0 \n"

            # Accumulate x and x^2 lane-wise across all col-blocks (one reduce at the end).
            if unroll:
                for i in range(n_blocks):
                    generated_code += f"V_ADD_VV gp{accx_addr}, gp{accx_addr}, gp{stats_addr}, 0 \n"
                    generated_code += f"V_MUL_VV gp{scratchpad_addr}, gp{stats_addr}, gp{stats_addr}, 0 \n"
                    generated_code += f"V_ADD_VV gp{accx2_addr}, gp{accx2_addr}, gp{scratchpad_addr}, 0 \n"
                    generated_code += f"S_ADDI_INT gp{stats_addr}, gp{stats_addr}, {vlen * batch_size} \n"
            else:
                generated_code += f"C_LOOP_START gp{loop_addr}, {n_blocks} \n"
                generated_code += f"V_ADD_VV gp{accx_addr}, gp{accx_addr}, gp{stats_addr}, 0 \n"
                generated_code += f"V_MUL_VV gp{scratchpad_addr}, gp{stats_addr}, gp{stats_addr}, 0 \n"
                generated_code += f"V_ADD_VV gp{accx2_addr}, gp{accx2_addr}, gp{scratchpad_addr}, 0 \n"
                generated_code += f"S_ADDI_INT gp{stats_addr}, gp{stats_addr}, {vlen * batch_size} \n"
                generated_code += f"C_LOOP_END gp{loop_addr} \n"

            # f2 = sum(x), f3 = sum(x^2)   (f2,f3 are 0 here -> a single reduce each)
            generated_code += f"V_RED_SUM f2, gp{accx_addr} \n"
            generated_code += f"V_RED_SUM f3, gp{accx2_addr} \n"
        elif unroll:
            for i in range(n_blocks):
                # sum(x)
                generated_code += f"V_RED_SUM f2, gp{stats_addr} \n"

                # sum(x^2)
                generated_code += f"V_MUL_VV gp{scratchpad_addr}, gp{stats_addr}, gp{stats_addr}, 0 \n"
                generated_code += f"V_RED_SUM f3, gp{scratchpad_addr} \n"

                # Move stats pointer to next vector
                generated_code += f"S_ADDI_INT gp{stats_addr}, gp{stats_addr}, {vlen * batch_size} \n"
        else:
            generated_code += f"C_LOOP_START gp{loop_addr}, {n_blocks} \n"
            generated_code += f"V_RED_SUM f2, gp{stats_addr} \n"
            generated_code += f"V_MUL_VV gp{scratchpad_addr}, gp{stats_addr}, gp{stats_addr}, 0 \n"
            generated_code += f"V_RED_SUM f3, gp{scratchpad_addr} \n"
            generated_code += f"S_ADDI_INT gp{stats_addr}, gp{stats_addr}, {vlen * batch_size} \n"
            generated_code += f"C_LOOP_END gp{loop_addr} \n"

        # f2 = sum(x) * (1/hidden_dim) = mean(x)
        generated_code += "S_MUL_FP f2, f2, f4 \n"

        # f3 = sum(x^2) * (1/hidden_dim) = mean(x^2)
        generated_code += "S_MUL_FP f3, f3, f4 \n"

        # f5 = mean(x)^2
        generated_code += "S_MUL_FP f5, f2, f2 \n"

        # f5 = mean(x^2) - mean(x)^2 = variance
        generated_code += "S_SUB_FP f5, f3, f5 \n"

        # f5 = variance + epsilon
        generated_code += "S_ADD_FP f5, f5, f1 \n"

        # f5 = sqrt(variance + epsilon) = std
        generated_code += "S_SQRT_FP f5, f5 \n"

        # f5 = 1/std
        generated_code += "S_RECI_FP f5, f5 \n"

        # Second loop: normalize using act_addr (still at batch start)
        if unroll:
            for i in range(hidden_dim // vlen):
                # normalized = (x - mean) * (1/std)
                # Store (x - mean) in scratchpad first
                generated_code += f"V_SUB_VF gp{scratchpad_addr}, gp{act_addr}, f2, 0, 0 \n"
                # Then multiply by 1/std and write back to activation
                generated_code += f"V_MUL_VF gp{act_addr}, gp{scratchpad_addr}, f5, 0 \n"

                # Move to next vector
                generated_code += f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen * batch_size} \n"
        else:
            generated_code += f"C_LOOP_START gp{loop_addr}, {hidden_dim // vlen} \n"
            generated_code += f"V_SUB_VF gp{scratchpad_addr}, gp{act_addr}, f2, 0, 0 \n"
            generated_code += f"V_MUL_VF gp{act_addr}, gp{scratchpad_addr}, f5, 0 \n"
            generated_code += f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen * batch_size} \n"
            generated_code += f"C_LOOP_END gp{loop_addr} \n"

        # Reset accumulators for next batch
        generated_code += "S_ADD_FP f2, f0, f0 \n"
        generated_code += "S_ADD_FP f3, f0, f0 \n"

    return generated_code
