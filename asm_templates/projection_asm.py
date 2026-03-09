from __future__ import annotations


def _load_large_int(reg: int, value: int) -> list[str]:
    """Generate instructions to load any non-negative integer into a GP register.

    S_ADDI_INT supports 18-bit immediates (values 0..2^18-1).
    For values < 2^18, emit a single S_ADDI_INT from gp0.
    For values >= 2^18, use S_LUI_INT (rd = imm << 12) + optional S_ADDI_INT.
    """
    IMM2_BOUND = 1 << 18
    if value < IMM2_BOUND:
        return [f"S_ADDI_INT gp{reg}, gp0, {value}"]
    upper = value >> 12
    lower = value & 0xFFF
    lines = [f"S_LUI_INT gp{reg}, {upper}"]
    if lower:
        lines.append(f"S_ADDI_INT gp{reg}, gp{reg}, {lower}")
    return lines


def _addi_large_int(dest_reg: int, src_reg: int, value: int, temp_reg: int) -> list[str]:
    """Generate rd = rs1 + value, handling values >= 2^18 by loading into temp_reg first."""
    IMM2_BOUND = 1 << 18
    if value < IMM2_BOUND:
        return [f"S_ADDI_INT gp{dest_reg}, gp{src_reg}, {value}"]
    else:
        lines = _load_large_int(temp_reg, value)
        lines.append(f"S_ADD_INT gp{dest_reg}, gp{src_reg}, gp{temp_reg}")
        return lines


def projection_asm(
    mlen: int,
    blen: int,
    batch: int,
    hidden_size: int,
    alive_registers: list[int],
    w_base_hbm_offset_reg: int,
    activation_base_address: int,
    result_base_address: int,
    rope_enabled: bool = False,
    rope_hbm_offset_reg: int = 0,
    rope_on_chip_address: int = 0,
    out_features: int | None = None,
) -> str:
    """
    Generates optimized assembly code for matrix multiplication (linear layer).
    (Batch, in_features) @ (in_features, out_features) -> (Batch, out_features)

    Supports both square matrices (hidden_size x hidden_size) and rectangular
    matrices when out_features is specified.

    Args:
        mlen: Matrix tile size (rows)
        blen: Vector tile size (batch dimension)
        batch: Batch size (unused, assumed = blen)
        hidden_size: Input dimension (in_features)
        alive_registers: Available GP registers [result, w_actual, w_hbm_offset, a_actual]
        w_base_hbm_offset_reg: HBM address register index for weights
        activation_base_address: Vector SRAM address for activations
        result_base_address: Vector SRAM address for output
        rope_enabled: Whether RoPE is enabled (unused)
        rope_hbm_offset_reg: RoPE HBM address register (unused)
        rope_on_chip_address: RoPE on-chip address (unused)
        out_features: Output dimension. If None, defaults to hidden_size (square matrix)

    Returns:
        Generated assembly code string
    """
    # Suppress unused parameter warnings (API compatibility)
    _ = batch, rope_enabled, rope_hbm_offset_reg, rope_on_chip_address

    # Support rectangular matrices: in_features x out_features
    in_features = hidden_size
    if out_features is None:
        out_features = hidden_size  # Backward compatible: square matrix

    # Unpack registers
    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    act_reg = alive_registers[2]
    intermediate_register = alive_registers[3]
    w_hbm_offset_register = alive_registers[4]
    result_reg = alive_registers[5]

    # Build assembly as list of lines
    lines = ["; Projection Generation (Optimized)"]
    lines.append(f"; Linear: (batch, {in_features}) @ ({in_features}, {out_features}) -> (batch, {out_features})")

    # Setup scale and stride registers (use act_reg as temp)
    # Scale = total weight matrix size, Stride = output dimension
    lines.extend(_load_large_int(act_reg, in_features * out_features))
    lines.append(f"C_SET_SCALE_REG gp{act_reg}")
    lines.append(f"S_ADDI_INT gp{act_reg}, gp0, {out_features}")
    lines.append(f"C_SET_STRIDE_REG gp{act_reg}")

    # Initialize activation register
    lines.append(f"S_ADDI_INT gp{act_reg}, gp0, {activation_base_address}")
    lines.append(f"S_ADDI_INT gp{result_reg}, gp0, {result_base_address}")

    for weight_row in range(out_features // blen):
        if weight_row % (mlen // blen) == 0:
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 ")
            lines.append(f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, {weight_row * blen} ")
            lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{result_reg}, 0 ")
            for weight_col in range(hidden_size // mlen):
                lines.append(
                    f"H_PREFETCH_M gp{w_actual_register}, gp{w_hbm_offset_register}, a{w_base_hbm_offset_reg}, 1, 0 "
                )
                lines.append(f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} ")
                lines.extend(_addi_large_int(w_hbm_offset_register, w_hbm_offset_register, mlen * out_features, w_temp_register))
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 ")
        else:
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, {(weight_row % (mlen // blen)) * blen} ")
            lines.append(
                f"S_ADDI_INT gp{intermediate_register}, gp{result_reg}, {(weight_row % (mlen // blen)) * blen} "
            )
        for act_col in range(batch // blen):
            lines.append(f"S_ADDI_INT gp{act_reg}, gp0, {activation_base_address + act_col * mlen * blen} ")
            lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0 ")
            for inner_loop_index in range(hidden_size // mlen):
                lines.append(f"M_MM 0, gp{w_temp_register}, gp{act_reg} ")
                lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen} ")
                lines.append(f"S_ADDI_INT gp{act_reg}, gp{act_reg}, {mlen * batch} ")
            lines.append(f"M_MM_WO gp{intermediate_register}, gp0, 0 ")
            lines.append(
                f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen} "
            )
        if (weight_row + 1) % (mlen // blen) == 0 and weight_row != out_features // blen - 1:
            lines.append(f"S_ADDI_INT gp{result_reg}, gp{result_reg}, {mlen * batch} ")

    return "\n".join(lines) + "\n"


def projection_T_asm(
    mlen: int,
    blen: int,
    batch: int,
    hidden_size: int,
    alive_registers: list[int],
    w_base_hbm_offset_reg: int,
    activation_base_address: int,
    result_base_address: int,
    rope_enabled: bool = False,
    rope_hbm_offset_reg: int = 0,
    rope_on_chip_address: int = 0,
    out_features: int | None = None,
) -> str:
    """
    Generates assembly for transposed projection: act @ weight.T
    weight stored in HBM as (out_features, in_features).

    Args:
        mlen: Matrix tile size (rows)
        blen: Vector tile size (batch dimension)
        batch: Batch size
        hidden_size: Input dimension (in_features)
        alive_registers: Available GP registers [w_actual, w_temp, act_reg, intermediate, w_hbm_offset, result_reg]
        w_base_hbm_offset_reg: Address register pointing to weight HBM base
        activation_base_address: VRAM address of activations
        result_base_address: VRAM address for output
        out_features: Output dimension (defaults to hidden_size)

    Returns:
        Generated assembly code string
    """
    _ = rope_enabled, rope_hbm_offset_reg, rope_on_chip_address

    in_features = hidden_size
    if out_features is None:
        out_features = hidden_size

    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    act_reg = alive_registers[2]
    intermediate_register = alive_registers[3]
    w_hbm_offset_register = alive_registers[4]
    result_reg = alive_registers[5]

    tiles_per_mlen = mlen // blen

    lines = ["; Projection_T Generation (act @ weight.T)"]
    lines.append(f"; Linear T: (batch, {in_features}) @ ({out_features}, {in_features})^T -> (batch, {out_features})")

    # Scale = total weight size, Stride = in_features (row stride of weight in HBM)
    lines.extend(_load_large_int(act_reg, in_features * out_features))
    lines.append(f"C_SET_SCALE_REG gp{act_reg}")
    lines.append(f"S_ADDI_INT gp{act_reg}, gp0, {in_features}")
    lines.append(f"C_SET_STRIDE_REG gp{act_reg}")

    lines.append(f"S_ADDI_INT gp{act_reg}, gp0, {activation_base_address}")
    lines.append(f"S_ADDI_INT gp{result_reg}, gp0, {result_base_address}")

    for weight_row in range(out_features // blen):
        if weight_row % tiles_per_mlen == 0:
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 ")
            # HBM offset: weight_row*blen rows into (out_features, in_features) layout
            lines.extend(_load_large_int(w_hbm_offset_register, weight_row * blen * in_features))
            lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{result_reg}, 0 ")
            for weight_col in range(hidden_size // mlen):
                lines.append(f"H_PREFETCH_M gp{w_actual_register}, gp{w_hbm_offset_register}, a{w_base_hbm_offset_reg}, 1, 0 ")
                lines.append(f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} ")
                # Move to next mlen-wide column block within this row group
                lines.append(f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen} ")
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 ")
        else:
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, {(weight_row % tiles_per_mlen) * blen} ")
            lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{result_reg}, {(weight_row % tiles_per_mlen) * blen} ")
        for act_col in range(batch // blen):
            lines.append(f"S_ADDI_INT gp{act_reg}, gp0, {activation_base_address + act_col * mlen * blen} ")
            lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0 ")
            for inner_loop_index in range(hidden_size // mlen):
                lines.append(f"M_MM 0, gp{w_temp_register}, gp{act_reg} ")
                lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen} ")
                lines.append(f"S_ADDI_INT gp{act_reg}, gp{act_reg}, {mlen * batch} ")
            lines.append(f"M_MM_WO gp{intermediate_register}, gp0, 0 ")
            lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen} ")
        if (weight_row + 1) % tiles_per_mlen == 0 and weight_row != out_features // blen - 1:
            lines.append(f"S_ADDI_INT gp{result_reg}, gp{result_reg}, {mlen * batch} ")

    return "\n".join(lines) + "\n"
