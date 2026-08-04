from ._imm import load_large_int_str as _load_large_int


def segmented_rms_norm_asm(
    eps_offset: int,
    reci_segment_offset: int,
    alive_registers: list[int],
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    physical_rows: int,
    logical_rows: int,
    logical_hidden_dim: int,
    physical_hidden_dim: int,
    segment_width: int,
) -> str:
    """Normalize independent segments in-place with the vector mask."""

    if len(alive_registers) < 3:
        raise ValueError("segmented RMSNorm requires three GP registers")
    dimensions = (
        vlen,
        physical_rows,
        logical_rows,
        logical_hidden_dim,
        physical_hidden_dim,
        segment_width,
    )
    if min(dimensions) <= 0:
        raise ValueError("segmented RMSNorm dimensions must be positive")
    if vlen % segment_width != 0:
        raise ValueError("segment_width must divide vlen")
    if logical_hidden_dim % segment_width != 0:
        raise ValueError("logical_hidden_dim must contain complete segments")
    if physical_hidden_dim % vlen != 0:
        raise ValueError("physical_hidden_dim must be VLEN-aligned")
    if logical_hidden_dim > physical_hidden_dim:
        raise ValueError("logical_hidden_dim exceeds physical storage")

    segments_per_vector = vlen // segment_width
    if segments_per_vector > 32:
        raise ValueError("the vector mask supports at most 32 segments")

    act_addr, scratchpad_addr, mask_reg = alive_registers[:3]
    segment_count = logical_hidden_dim // segment_width
    generated_code = "; Segmented RMSNorm\n"
    generated_code += _load_large_int(scratchpad_addr, scratchpad_base_address)
    generated_code += f"S_LD_FP f1, gp0, {eps_offset}\n"
    generated_code += f"S_LD_FP f3, gp0, {reci_segment_offset}\n"

    for row in range(logical_rows):
        for segment in range(segment_count):
            col_block = segment // segments_per_vector
            lane_segment = segment % segments_per_vector
            vector_addr = (
                activation_base_address
                + col_block * physical_rows * vlen
                + row * vlen
            )
            generated_code += _load_large_int(mask_reg, 1 << lane_segment)
            generated_code += f"C_SET_V_MASK_REG gp{mask_reg}\n"
            generated_code += _load_large_int(act_addr, vector_addr)
            generated_code += "S_ADD_FP f2, f0, f0\n"
            generated_code += (
                f"V_MUL_VV gp{scratchpad_addr}, gp{act_addr}, gp{act_addr}, 1\n"
            )
            generated_code += f"V_RED_SUM f2, gp{scratchpad_addr}, 1\n"
            generated_code += "S_MUL_FP f2, f2, f3\n"
            generated_code += "S_ADD_FP f2, f2, f1\n"
            generated_code += "S_SQRT_FP f2, f2\n"
            generated_code += "S_RECI_FP f2, f2\n"
            generated_code += f"V_MUL_VF gp{act_addr}, gp{act_addr}, f2, 1\n"

    return generated_code


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
    destination_base_address: int | None = None,
) -> str:
    """Generate assembly code for RMS normalization.

    With *destination_base_address* the scaling pass writes to that tensor and
    leaves the input intact, which lets a caller keep the pre-norm activation as
    its residual instead of copying it first. `V_MUL_VF` takes independent
    destination and source address registers, so the out-of-place form issues
    the same instructions as the in-place one; only the write cursor differs.
    """
    act_addr = alive_registers[0]
    scratchpad_addr = alive_registers[1]
    stats_addr = alive_registers[2]
    # Rolled path uses the spare 4th register (already allocated by normalize()) as the
    # C_LOOP counter. Only accessed when rolled, so unrolled callers may pass 3 registers.
    loop_addr = alive_registers[3] if not unroll else None
    out_of_place = destination_base_address is not None
    if out_of_place:
        if len(alive_registers) < 5:
            raise ValueError("out-of-place RMSNorm needs a fifth GP register")
        dst_addr = alive_registers[4]
    else:
        dst_addr = act_addr

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
        if out_of_place:
            generated_code += _load_large_int(
                dst_addr, destination_base_address + vlen * batch
            )

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

        # Second loop: scale from act_addr into dst_addr (the same tensor unless
        # a destination was given).
        advance = f"S_ADDI_INT gp{act_addr}, gp{act_addr}, {vlen * batch_size} \n"
        if out_of_place:
            advance += f"S_ADDI_INT gp{dst_addr}, gp{dst_addr}, {vlen * batch_size} \n"
        if unroll:
            for i in range(hidden_dim // vlen):
                generated_code += f"V_MUL_VF gp{dst_addr}, gp{act_addr}, f2, 0 \n"
                generated_code += advance
        else:
            generated_code += f"C_LOOP_START gp{loop_addr}, {hidden_dim // vlen} \n"
            generated_code += f"V_MUL_VF gp{dst_addr}, gp{act_addr}, f2, 0 \n"
            generated_code += advance
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
) -> str:
    """
    Generate assembly code for layer normalization.
    """
    act_addr = alive_registers[0]
    scratchpad_addr = alive_registers[1]
    stats_addr = alive_registers[2]
    # Rolled path uses the spare 4th register (already allocated by normalize()) as the
    # C_LOOP counter. Only accessed when rolled, so unrolled callers may pass 3 registers.
    loop_addr = alive_registers[3] if not unroll else None

    generated_code = "; Layer Norm generation \n"
    generated_code += _load_large_int(scratchpad_addr, scratchpad_base_address)

    # Load constants
    generated_code += f"S_LD_FP f1, gp0, {_eps_offset} \n"  # epsilon
    generated_code += "S_ADD_FP f2, f0, f0 \n"  # sum(x) accumulator
    generated_code += "S_ADD_FP f3, f0, f0 \n"  # sum(x^2) accumulator
    generated_code += f"S_LD_FP f4, gp0, {reci_hid_offset} \n"  # 1/hidden_dim

    for batch in range(batch_size):
        # Set act_addr to start of current batch
        generated_code += _load_large_int(act_addr, activation_base_address + vlen * batch)
        # Set stats_addr to same position for iteration
        generated_code += _load_large_int(stats_addr, activation_base_address + vlen * batch)

        # First loop: compute sum(x) and sum(x^2) using stats_addr
        if unroll:
            for i in range(hidden_dim // vlen):
                # sum(x)
                generated_code += f"V_RED_SUM f2, gp{stats_addr} \n"

                # sum(x^2)
                generated_code += f"V_MUL_VV gp{scratchpad_addr}, gp{stats_addr}, gp{stats_addr}, 0 \n"
                generated_code += f"V_RED_SUM f3, gp{scratchpad_addr} \n"

                # Move stats pointer to next vector
                generated_code += f"S_ADDI_INT gp{stats_addr}, gp{stats_addr}, {vlen * batch_size} \n"
        else:
            generated_code += f"C_LOOP_START gp{loop_addr}, {hidden_dim // vlen} \n"
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
