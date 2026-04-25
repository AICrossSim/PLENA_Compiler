from __future__ import annotations

import math

from ._imm import addi_large_int as _addi_large_int
from ._imm import load_large_int as _load_large_int
from ._k_split import k_chunks as _proj_k_chunks


def _emit_projection_chunk(
    *,
    mlen: int,
    blen: int,
    batch: int,
    in_features: int,
    out_features: int,
    w_base_hbm_offset_reg: int,
    activation_base_address: int,
    k_start_tile: int,
    k_tile_count: int,
    target_base_address: int,
    w_actual_register: int,
    w_temp_register: int,
    act_reg: int,
    intermediate_register: int,
    w_hbm_offset_register: int,
    result_reg: int,
) -> list[str]:
    """Emit one K-chunk of a projection as unrolled ASM lines.

    Computes ``(batch, in_features[k_start*mlen .. (k_start+k_count)*mlen]) @
    (out_features, in_features[...]).T`` writing to ``target_base_address``.

    HBM weight layout: row-major (out_features, in_features), so the first
    K-tile of weight row ``weight_row`` starts at HBM offset:
        weight_row * blen * in_features  +  k_start_tile * mlen
    (for the non-transposed ``projection_asm`` layout where stride = out_features
    and the prefetch advances by ``mlen * out_features`` per K column).

    Wait — the original layout is (in_features, out_features) stored
    column-major from the weight's perspective: HBM row stride = out_features,
    prefetch advance = mlen * out_features. So:
        chunk_hbm_base = weight_row * blen  +  k_start_tile * mlen * out_features
    And activation chunk offset = k_start_tile * mlen * batch.
    """
    chunk_hbm_base = k_start_tile * mlen * out_features
    chunk_act_base = k_start_tile * mlen * batch

    lines: list[str] = []
    # Load target base into result_reg
    lines.extend(_load_large_int(result_reg, target_base_address))

    for weight_row in range(out_features // blen):
        if weight_row % (mlen // blen) == 0:
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 ")
            hbm_off = chunk_hbm_base + weight_row * blen
            if hbm_off >= (1 << 18):
                lines.extend(_addi_large_int(w_hbm_offset_register, 0, hbm_off, w_temp_register))
            else:
                lines.append(f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, {hbm_off} ")
            lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{result_reg}, 0 ")
            for _ in range(k_tile_count):
                lines.append(
                    f"H_PREFETCH_M gp{w_actual_register}, gp{w_hbm_offset_register}, a{w_base_hbm_offset_reg}, 1, 0 "
                )
                lines.append(f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} ")
                lines.extend(
                    _addi_large_int(w_hbm_offset_register, w_hbm_offset_register, mlen * out_features, w_temp_register)
                )
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 ")
        else:
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, {(weight_row % (mlen // blen)) * blen} ")
            lines.append(
                f"S_ADDI_INT gp{intermediate_register}, gp{result_reg}, {(weight_row % (mlen // blen)) * blen} "
            )
        for act_col in range(batch // blen):
            addr = activation_base_address + act_col * mlen * blen + chunk_act_base
            if addr >= (1 << 18):
                lines.extend(_addi_large_int(act_reg, 0, addr, w_temp_register))
            else:
                lines.append(f"S_ADDI_INT gp{act_reg}, gp0, {addr} ")
            lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0 ")
            for _ in range(k_tile_count):
                lines.append(f"M_MM 0, gp{w_temp_register}, gp{act_reg} ")
                lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen} ")
                lines.append(f"S_ADDI_INT gp{act_reg}, gp{act_reg}, {mlen * batch} ")
            lines.append(f"M_MM_WO gp{intermediate_register}, gp0, 0 ")
            lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen} ")
        if (weight_row + 1) % (mlen // blen) == 0 and weight_row != out_features // blen - 1:
            lines.append(f"S_ADDI_INT gp{result_reg}, gp{result_reg}, {mlen * batch} ")

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
    matrix_sram_size: int = 1024,
    scratch_base_address: int = 0,
    vlen: int = 64,
) -> str:
    """
    Generates optimized assembly code for matrix multiplication (linear layer).
    (Batch, in_features) @ (in_features, out_features) -> (Batch, out_features)

    Supports both square matrices (hidden_size x hidden_size) and rectangular
    matrices when out_features is specified.

    When ``in_features / mlen > matrix_sram_size // mlen`` (i.e. K tiles exceed
    MRAM capacity), the projection is split into K-chunks.  First chunk writes to
    ``result_base_address``; subsequent chunks write to ``scratch_base_address``
    and are accumulated via ``V_ADD_VV``.

    Args:
        mlen: Matrix tile size (rows)
        blen: Vector tile size (batch dimension)
        batch: Batch size (unused, assumed = blen)
        hidden_size: Input dimension (in_features)
        alive_registers: Available GP registers [w_actual, w_temp, act, intermediate, w_hbm_offset, result]
        w_base_hbm_offset_reg: HBM address register index for weights
        activation_base_address: Vector SRAM address for activations
        result_base_address: Vector SRAM address for output
        rope_enabled: Whether RoPE is enabled (unused)
        rope_hbm_offset_reg: RoPE HBM address register (unused)
        rope_on_chip_address: RoPE on-chip address (unused)
        out_features: Output dimension. If None, defaults to hidden_size (square matrix)
        matrix_sram_size: MRAM capacity in elements (tiles * mlen * mlen). Used to
            derive MAX_K_TILES = matrix_sram_size // mlen.  Default 1024 = 16 × 64.
        scratch_base_address: VRAM address for K-split partial-sum scratch region.
            Only used when K-split is active.  Must not overlap with
            result_base_address or activation_base_address.
        vlen: Vector register width in elements (used for V_ADD_VV accumulation).

    Returns:
        Generated assembly code string
    """
    # Suppress unused parameter warnings (API compatibility)
    _ = rope_enabled, rope_hbm_offset_reg, rope_on_chip_address

    # Support rectangular matrices: in_features x out_features
    in_features = hidden_size
    if out_features is None:
        out_features = hidden_size  # Backward compatible: square matrix

    assert in_features % mlen == 0, f"K ({in_features}) must be a multiple of MLEN ({mlen})"
    MAX_K_TILES = max(1, matrix_sram_size // mlen)
    num_k_tiles = in_features // mlen

    # Unpack registers (same layout as before)
    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    act_reg = alive_registers[2]
    intermediate_register = alive_registers[3]
    w_hbm_offset_register = alive_registers[4]
    result_reg = alive_registers[5]

    # Build assembly as list of lines
    lines: list[str] = ["; Projection Generation (Optimized)"]
    lines.append(f"; Linear: (batch, {in_features}) @ ({in_features}, {out_features}) -> (batch, {out_features})")

    # Setup scale and stride registers (use act_reg as temp)
    # Scale = total weight matrix size, Stride = output dimension
    lines.extend(_load_large_int(act_reg, in_features * out_features))
    lines.append(f"C_SET_SCALE_REG gp{act_reg}")
    lines.extend(_load_large_int(act_reg, out_features))
    lines.append(f"C_SET_STRIDE_REG gp{act_reg}")

    if num_k_tiles <= MAX_K_TILES:
        lines.append(f" ; K-split inactive: num_k_tiles={num_k_tiles} <= MAX_K_TILES={MAX_K_TILES}")
        # Original single-pass path (unchanged behaviour)
        lines.extend(_load_large_int(act_reg, activation_base_address))
        lines.extend(_load_large_int(result_reg, result_base_address))

        for weight_row in range(out_features // blen):
            if weight_row % (mlen // blen) == 0:
                lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 ")
                lines.extend(_load_large_int(w_hbm_offset_register, weight_row * blen))
                lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{result_reg}, 0 ")
                for weight_col in range(hidden_size // mlen):
                    lines.append(
                        f"H_PREFETCH_M gp{w_actual_register}, gp{w_hbm_offset_register}, a{w_base_hbm_offset_reg}, 1, 0 "
                    )
                    lines.append(f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} ")
                    lines.extend(
                        _addi_large_int(w_hbm_offset_register, w_hbm_offset_register, mlen * out_features, w_temp_register)
                    )
                lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 ")
            else:
                lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, {(weight_row % (mlen // blen)) * blen} ")
                lines.append(
                    f"S_ADDI_INT gp{intermediate_register}, gp{result_reg}, {(weight_row % (mlen // blen)) * blen} "
                )
            for act_col in range(batch // blen):
                lines.extend(_load_large_int(act_reg, activation_base_address + act_col * mlen * blen))
                lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0 ")
                for inner_loop_index in range(hidden_size // mlen):
                    lines.append(f"M_MM 0, gp{w_temp_register}, gp{act_reg} ")
                    lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen} ")
                    lines.append(f"S_ADDI_INT gp{act_reg}, gp{act_reg}, {mlen * batch} ")
                lines.append(f"M_MM_WO gp{intermediate_register}, gp0, 0 ")
                lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen} ")
            if (weight_row + 1) % (mlen // blen) == 0 and weight_row != out_features // blen - 1:
                lines.append(f"S_ADDI_INT gp{result_reg}, gp{result_reg}, {mlen * batch} ")
    else:
        # K-split path
        chunks = _proj_k_chunks(num_k_tiles, MAX_K_TILES)
        lines.append(
            f" ; K-split active: num_k_tiles={num_k_tiles}, MAX_K_TILES={MAX_K_TILES}, "
            f"chunks={len(chunks)} (partial sums accumulated via V_ADD_VV)"
        )
        output_elements = out_features * batch
        per_vlen_adds = math.ceil(output_elements / vlen)

        for chunk_idx, (k_start, k_count) in enumerate(chunks):
            is_first = chunk_idx == 0
            target_addr = result_base_address if is_first else scratch_base_address
            lines.append(f" ; K-chunk {chunk_idx}/{len(chunks)}: k_start_tile={k_start}, k_count={k_count}")
            lines.extend(
                _emit_projection_chunk(
                    mlen=mlen,
                    blen=blen,
                    batch=batch,
                    in_features=in_features,
                    out_features=out_features,
                    w_base_hbm_offset_reg=w_base_hbm_offset_reg,
                    activation_base_address=activation_base_address,
                    k_start_tile=k_start,
                    k_tile_count=k_count,
                    target_base_address=target_addr,
                    w_actual_register=w_actual_register,
                    w_temp_register=w_temp_register,
                    act_reg=act_reg,
                    intermediate_register=intermediate_register,
                    w_hbm_offset_register=w_hbm_offset_register,
                    result_reg=result_reg,
                )
            )

            if not is_first:
                lines.append(
                    f" ; K-split accumulate: output[0..{output_elements}] += scratch[0..{output_elements}]"
                )
                lines.extend(_load_large_int(w_actual_register, result_base_address))
                lines.extend(_load_large_int(w_temp_register, scratch_base_address))
                for _ in range(per_vlen_adds):
                    lines.append(
                        f"V_ADD_VV gp{w_actual_register}, gp{w_actual_register}, gp{w_temp_register}, 0 "
                    )
                    lines.append(f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {vlen} ")
                    lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {vlen} ")

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
    lines.extend(_load_large_int(act_reg, in_features))
    lines.append(f"C_SET_STRIDE_REG gp{act_reg}")

    lines.extend(_load_large_int(act_reg, activation_base_address))
    lines.extend(_load_large_int(result_reg, result_base_address))

    for weight_row in range(out_features // blen):
        if weight_row % tiles_per_mlen == 0:
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 ")
            # HBM offset: weight_row*blen rows into (out_features, in_features) layout
            lines.extend(_load_large_int(w_hbm_offset_register, weight_row * blen * in_features))
            lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{result_reg}, 0 ")
            for weight_col in range(hidden_size // mlen):
                lines.append(
                    f"H_PREFETCH_M gp{w_actual_register}, gp{w_hbm_offset_register}, a{w_base_hbm_offset_reg}, 1, 0 "
                )
                lines.append(f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} ")
                # Move to next mlen-wide column block within this row group
                lines.append(f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen} ")
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 ")
        else:
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, {(weight_row % tiles_per_mlen) * blen} ")
            lines.append(
                f"S_ADDI_INT gp{intermediate_register}, gp{result_reg}, {(weight_row % tiles_per_mlen) * blen} "
            )
        for act_col in range(batch // blen):
            lines.extend(_load_large_int(act_reg, activation_base_address + act_col * mlen * blen))
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
