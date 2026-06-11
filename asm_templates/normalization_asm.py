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

        # Second loop: normalize the activation in-place. Ping-pong the per-chunk
        # address across two registers with a settle gap so the in-place V_MUL_VF
        # never reads a just-written (stale) write-address register. chunk 0 uses
        # act_addr (settled since the batch top); chunk 1 is pre-loaded before the FP
        # tail; chunk i>=2 is loaded after chunk i-2 frees its register.
        base = activation_base_address + vlen * batch
        stride = vlen * batch_size
        n_chunks = hidden_dim // vlen
        addr_regs = (act_addr, stats_addr)

        if n_chunks > 1:
            generated_code += _load_large_int(addr_regs[1], base + stride)

        # Taking the avg
        generated_code += "S_MUL_FP f2, f2, f3 \n"

        # Plus epsilon
        generated_code += "S_ADD_FP f2, f2, f1 \n"

        # Compute square root
        generated_code += "S_SQRT_FP f2, f2 \n"

        # Compute reciprocal
        generated_code += "S_RECI_FP f2, f2 \n"

        # Spacer so the multi-cycle S_RECI_FP retires before the first V_MUL_VF reads f2.
        for _ in range(4):
            generated_code += "S_ADDI_INT gp0, gp0, 0 \n"

        for i in range(n_chunks):
            cur = addr_regs[i % 2]
            generated_code += f"V_MUL_VF gp{cur}, gp{cur}, f2, 0 \n"
            if i + 2 < n_chunks:
                # Loading the next-next chunk address also spaces consecutive V_MUL_VF.
                generated_code += _load_large_int(cur, base + stride * (i + 2))
            elif i + 1 < n_chunks:
                # No load to emit but another V_MUL_VF follows: insert a one-instruction
                # spacer so the two in-place writebacks land on different port-A cycles
                # (adjacent V_MUL_VF collide on the shared write port and the first write
                # is dropped, leaving that chunk un-normalized). Writes gp0 (zero reg) = NOP.
                generated_code += "S_ADDI_INT gp0, gp0, 0 \n"

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
