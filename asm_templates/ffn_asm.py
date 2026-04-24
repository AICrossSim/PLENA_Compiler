from ._imm import addi_large_int_str as _addi_large_int
from ._imm import load_large_int_str as _load_large_int


def ffn_asm(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    alive_registers: list[int],
    gate_weight_hbm_offset_reg: int,
    up_weight_hbm_offset_reg: int,
    down_weight_hbm_offset_reg: int,
    const_one_fp_address: int,
    activation_base_address: int,
    use_loop_instructions: bool = False,
    use_fused_up_gate: bool = False,
    matrix_sram_size: int = 1024,
) -> str:
    """
    Generates assembly code for a FFN operation.

    Set use_loop_instructions=True to use C_LOOP_START/END for compact code.
    Set use_fused_up_gate=True to fuse upsize and gate projections (requires 12 registers).

    ``matrix_sram_size`` is the MRAM capacity (element-units-per-tile * tiles). When
    a projection's K dimension exceeds ``matrix_sram_size // mlen`` tiles, the template
    emits a K-split partial-sum accumulation loop (mirroring
    ``aten/ops/plena/linear_ops.py::linear_plena``). This prevents OOB
    ``H_PREFETCH_M`` addresses for models whose intermediate/hidden dims exceed the
    MRAM tile count.
    """
    if use_fused_up_gate:
        return _ffn_asm_fused_up_gate(
            mlen,
            vlen,
            blen,
            batch,
            seq_len,
            hidden_size,
            intermediate_size,
            alive_registers,
            gate_weight_hbm_offset_reg,
            up_weight_hbm_offset_reg,
            down_weight_hbm_offset_reg,
            const_one_fp_address,
            activation_base_address,
        )
    elif use_loop_instructions:
        return _ffn_asm_with_loops(
            mlen,
            vlen,
            blen,
            batch,
            seq_len,
            hidden_size,
            intermediate_size,
            alive_registers,
            gate_weight_hbm_offset_reg,
            up_weight_hbm_offset_reg,
            down_weight_hbm_offset_reg,
            const_one_fp_address,
            activation_base_address,
        )
    else:
        return _ffn_asm_unrolled(
            mlen,
            vlen,
            blen,
            batch,
            seq_len,
            hidden_size,
            intermediate_size,
            alive_registers,
            gate_weight_hbm_offset_reg,
            up_weight_hbm_offset_reg,
            down_weight_hbm_offset_reg,
            const_one_fp_address,
            activation_base_address,
            matrix_sram_size=matrix_sram_size,
        )


def _k_chunks(num_k_tiles: int, max_k_tiles: int) -> list[tuple[int, int]]:
    """Split ``num_k_tiles`` K dimension tiles into chunks of at most
    ``max_k_tiles``. Returns list of (k_start_tile, k_count) pairs."""
    assert max_k_tiles >= 1, f"MAX_K_TILES must be >= 1, got {max_k_tiles}"
    chunks: list[tuple[int, int]] = []
    k_pos = 0
    while k_pos < num_k_tiles:
        count = min(max_k_tiles, num_k_tiles - k_pos)
        chunks.append((k_pos, count))
        k_pos += count
    return chunks


def _ffn_asm_unrolled(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    alive_registers: list[int],
    gate_weight_hbm_offset_reg: int,
    up_weight_hbm_offset_reg: int,
    down_weight_hbm_offset_reg: int,
    const_one_fp_address: int,
    activation_base_address: int,
    matrix_sram_size: int = 1024,
) -> str:
    """Unrolled FFN: up + gate + SiLU + down projections.

    When a projection's K dimension tile-count exceeds
    ``matrix_sram_size // mlen``, we split K into chunks of at most
    ``MAX_K_TILES = matrix_sram_size // mlen`` and accumulate partial sums in
    VRAM via V_ADD_VV. This mirrors the ATen-path K-split in
    ``aten/ops/plena/linear_ops.py::linear_plena``.
    """

    # memory assignment
    # 0 -> activation
    # b * s * hidden_size -> upsize intermediate results
    # b * s * (hidden_size + intermediate_size) -> gate projection results

    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    a_actual_register = alive_registers[2]
    up_result_register = alive_registers[3]
    intermediate_register = alive_registers[4]
    gate_result_register = alive_registers[5]
    w_hbm_offset_register = alive_registers[6]
    m_stride_register = alive_registers[7]

    # reset the registers
    generated_code = "; FFN Generation \n"

    # Settings for up and gate weight matrices prefetching
    generated_code += _load_large_int(w_actual_register, hidden_size * intermediate_size)
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {intermediate_size} \n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
    # Set the address for on-chip sram
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += _load_large_int(gate_result_register, batch * seq_len * (hidden_size + intermediate_size))

    # K-split config: when K tile count > MRAM tile capacity, we split K and
    # accumulate partial sums. `activation` region (used as input) starts at
    # `activation_base_address`; the K-split scratch region for up/gate lives
    # *after* the up+gate output regions at
    # `batch*seq_len*(hidden_size+2*intermediate_size)` (hidden_size chunk for
    # activation, intermediate_size each for up and gate results).
    MAX_K_TILES = max(1, matrix_sram_size // mlen)

    # --- FFN Upsize Linear (K = hidden_size) ---
    up_num_k_tiles = hidden_size // mlen
    up_scratch_base = batch * seq_len * (hidden_size + 2 * intermediate_size)
    generated_code += _emit_ffn_projection_unrolled(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=batch,
        seq_len=seq_len,
        k_size=hidden_size,
        out_size=intermediate_size,
        weight_stride=intermediate_size,
        weight_hbm_offset_reg=up_weight_hbm_offset_reg,
        result_base_register=up_result_register,
        result_base_value=batch * seq_len * hidden_size,
        activation_base_address=activation_base_address,
        activation_base_register=None,
        max_k_tiles=MAX_K_TILES,
        w_actual_register=w_actual_register,
        w_temp_register=w_temp_register,
        a_actual_register=a_actual_register,
        intermediate_register=intermediate_register,
        w_hbm_offset_register=w_hbm_offset_register,
        scratch_base_value=up_scratch_base,
        section_comment="FFN Upsize Linear Generation",
    )

    generated_code += " ; FFN Gate Projection Generation \n"
    gate_scratch_base = batch * seq_len * (hidden_size + 2 * intermediate_size)
    generated_code += _emit_ffn_projection_unrolled(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=batch,
        seq_len=seq_len,
        k_size=hidden_size,
        out_size=intermediate_size,
        weight_stride=intermediate_size,
        weight_hbm_offset_reg=gate_weight_hbm_offset_reg,
        result_base_register=gate_result_register,
        result_base_value=batch * seq_len * (hidden_size + intermediate_size),
        activation_base_address=activation_base_address,
        activation_base_register=None,
        max_k_tiles=MAX_K_TILES,
        w_actual_register=w_actual_register,
        w_temp_register=w_temp_register,
        a_actual_register=a_actual_register,
        intermediate_register=intermediate_register,
        w_hbm_offset_register=w_hbm_offset_register,
        scratch_base_value=gate_scratch_base,
        section_comment="FFN Gate Projection (inlined)",
    )

    generated_code += "; SILU Generation \n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address} \n"
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += _load_large_int(gate_result_register, batch * seq_len * (hidden_size + intermediate_size))
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp0, {activation_base_address} \n"

    # SiLU: sigmoid(x) * x * gate, using activation region as scratchpad
    for b in range(batch * seq_len):
        for i in range(intermediate_size // vlen):
            generated_code += f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1 \n"
            generated_code += f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0 \n"
            generated_code += f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0 \n"
            generated_code += f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0 \n"
            generated_code += (
                f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0 \n"
            )
            generated_code += (
                f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0 \n"
            )
            generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen} \n"
            generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen} \n"

    generated_code += "; FFN Downsize Linear Generation \n"
    generated_code += _load_large_int(w_actual_register, hidden_size * intermediate_size)
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size} \n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
    generated_code += f"S_ADDI_INT gp{m_stride_register}, gp0, {((batch * seq_len) // blen)} \n"
    # Storing the results to the activation base region
    act_result_register = gate_result_register
    # Down projection: K = intermediate_size. Activation input is at
    # VRAM address batch*seq_len*hidden_size (up_result region, post-SiLU).
    # Scratch for K-split lives past the gate region so it never collides
    # with input (up_result) or output (activation_base_address).
    down_scratch_base = batch * seq_len * (hidden_size + 2 * intermediate_size)
    generated_code += _emit_ffn_projection_unrolled(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=batch,
        seq_len=seq_len,
        k_size=intermediate_size,
        out_size=hidden_size,
        weight_stride=hidden_size,
        weight_hbm_offset_reg=down_weight_hbm_offset_reg,
        result_base_register=act_result_register,
        result_base_value=activation_base_address,
        activation_base_address=None,
        activation_base_register=up_result_register,
        activation_base_register_value=batch * seq_len * hidden_size,
        max_k_tiles=max(1, matrix_sram_size // mlen),
        w_actual_register=w_actual_register,
        w_temp_register=w_temp_register,
        a_actual_register=a_actual_register,
        intermediate_register=intermediate_register,
        w_hbm_offset_register=w_hbm_offset_register,
        scratch_base_value=down_scratch_base,
        section_comment="FFN Downsize Linear (inlined)",
    )
    return generated_code


def _emit_ffn_projection_unrolled(
    *,
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    k_size: int,
    out_size: int,
    weight_stride: int,
    weight_hbm_offset_reg: int,
    result_base_register: int,
    result_base_value: int,
    activation_base_address: int | None,
    activation_base_register: int | None,
    activation_base_register_value: int | None = None,
    max_k_tiles: int = 16,
    w_actual_register: int,
    w_temp_register: int,
    a_actual_register: int,
    intermediate_register: int,
    w_hbm_offset_register: int,
    scratch_base_value: int,
    section_comment: str,
) -> str:
    """Emit a single FFN-style projection (one of up/gate/down) as unrolled ASM.

    The projection computes ``out[r][c] = sum_k act[r][k] * weight[k][c]`` for
    ``k_size`` K-dimension columns. The weight matrix has HBM row-stride
    ``weight_stride`` (intermediate_size for up/gate, hidden_size for down).

    When ``k_size // mlen > max_k_tiles``, emits a K-split partial-sum
    accumulation loop. First chunk writes to ``result_base_register`` VRAM
    region. Subsequent chunks write to a scratch region at
    ``scratch_base_value`` and a bulk V_ADD_VV pass accumulates scratch into
    output at the end of each chunk.

    Either ``activation_base_address`` (an absolute VRAM address, e.g. block1)
    or ``activation_base_register`` (a register holding a VRAM address, e.g.
    up_result_register) must be provided. The activation for K-tile ``k`` is
    read from ``act_base + k*mlen*batch*seq_len`` + per-tile column offsets.
    """

    num_k_tiles = k_size // mlen
    num_m_blocks = out_size // mlen  # output row-blocks (MLEN wide)
    tiles_per_mlen = mlen // blen
    num_act_cols = (batch * seq_len) // blen

    lines: list[str] = [f" ; {section_comment} (k_size={k_size}, out_size={out_size})\n"]

    if num_k_tiles <= max_k_tiles:
        lines.append(f" ; K-split inactive: num_k_tiles={num_k_tiles} <= MAX_K_TILES={max_k_tiles}\n")
        lines.append(
            _emit_ffn_projection_chunk(
                mlen=mlen,
                blen=blen,
                batch=batch,
                seq_len=seq_len,
                k_size=k_size,
                out_size=out_size,
                weight_stride=weight_stride,
                weight_hbm_offset_reg=weight_hbm_offset_reg,
                result_base_register=result_base_register,
                result_base_value=result_base_value,
                activation_base_address=activation_base_address,
                activation_base_register=activation_base_register,
                activation_base_register_value=activation_base_register_value,
                k_start_tile=0,
                k_tile_count=num_k_tiles,
                w_actual_register=w_actual_register,
                w_temp_register=w_temp_register,
                a_actual_register=a_actual_register,
                intermediate_register=intermediate_register,
                w_hbm_offset_register=w_hbm_offset_register,
                target_base_value_override=None,
                reset_act_base_register=True,
            )
        )
        return "".join(lines)

    # K-split active: split K into chunks of at most max_k_tiles.
    lines.append(
        f" ; K-split active: num_k_tiles={num_k_tiles}, MAX_K_TILES={max_k_tiles} "
        f"(partial sums accumulated via V_ADD_VV into VRAM)\n"
    )
    chunks = _k_chunks(num_k_tiles, max_k_tiles)
    # Total output region size (elements) for the VRAM accumulator pass.
    output_elements = out_size * batch * seq_len
    per_vlen_adds = output_elements // vlen

    for chunk_idx, (k_start, k_count) in enumerate(chunks):
        lines.append(f" ; K-chunk {chunk_idx}/{len(chunks)}: k_start_tile={k_start}, k_count={k_count}\n")
        is_first = chunk_idx == 0
        target_base_value = None if is_first else scratch_base_value
        lines.append(
            _emit_ffn_projection_chunk(
                mlen=mlen,
                blen=blen,
                batch=batch,
                seq_len=seq_len,
                k_size=k_size,
                out_size=out_size,
                weight_stride=weight_stride,
                weight_hbm_offset_reg=weight_hbm_offset_reg,
                result_base_register=result_base_register,
                result_base_value=result_base_value,
                activation_base_address=activation_base_address,
                activation_base_register=activation_base_register,
                activation_base_register_value=activation_base_register_value,
                k_start_tile=k_start,
                k_tile_count=k_count,
                w_actual_register=w_actual_register,
                w_temp_register=w_temp_register,
                a_actual_register=a_actual_register,
                intermediate_register=intermediate_register,
                w_hbm_offset_register=w_hbm_offset_register,
                target_base_value_override=target_base_value,
                reset_act_base_register=True,
            )
        )

        if not is_first:
            # V_ADD_VV output += scratch  for the entire output region.
            # Use w_actual_register as output pointer, w_temp_register as scratch ptr.
            lines.append(
                f" ; K-split accumulate: output[0..{output_elements}] += scratch[0..{output_elements}]\n"
            )
            lines.append(_load_large_int(w_actual_register, result_base_value))
            lines.append(_load_large_int(w_temp_register, scratch_base_value))
            for _ in range(per_vlen_adds):
                lines.append(
                    f"V_ADD_VV gp{w_actual_register}, gp{w_actual_register}, gp{w_temp_register}, 0 \n"
                )
                lines.append(f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {vlen} \n")
                lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {vlen} \n")

    # After the K-split loop the result_base_register value has been advanced
    # by the chunk helper (per MLEN-block, inside the chunk). Restore it for
    # whatever comes next by re-loading its base value — some callers (the
    # SiLU/etc. stages) reset these registers explicitly, but the following
    # code in `_ffn_asm_unrolled` reloads up_result_register / gate_result_register
    # itself before use.
    return "".join(lines)


def _emit_ffn_projection_chunk(
    *,
    mlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    k_size: int,
    out_size: int,
    weight_stride: int,
    weight_hbm_offset_reg: int,
    result_base_register: int,
    result_base_value: int,
    activation_base_address: int | None,
    activation_base_register: int | None,
    activation_base_register_value: int | None,
    k_start_tile: int,
    k_tile_count: int,
    w_actual_register: int,
    w_temp_register: int,
    a_actual_register: int,
    intermediate_register: int,
    w_hbm_offset_register: int,
    target_base_value_override: int | None,
    reset_act_base_register: bool,
) -> str:
    """Emit one K-chunk of an FFN projection.

    Mirrors the existing un-rolled projection structure but restricted to K
    tiles ``[k_start_tile, k_start_tile + k_tile_count)`` and capable of
    redirecting the output store to a scratch region.

    - HBM offset for a chunk starts at ``weight_row*blen + k_start_tile * mlen * weight_stride``
    - Activation offset for a chunk is advanced by ``k_start_tile * mlen * batch*seq_len``
    - MRAM prefetch destination always resets to 0 per MLEN block (we only
      prefetch this chunk's ``k_tile_count`` tiles)
    """

    num_m_blocks = out_size // mlen
    tiles_per_mlen = mlen // blen
    num_act_cols = (batch * seq_len) // blen
    chunk_hbm_base_offset = k_start_tile * mlen * weight_stride
    chunk_act_base_offset = k_start_tile * mlen * batch * seq_len

    # Target base (either real output or scratch). If scratch, reset the
    # result_base_register on entry so MLEN-block advancement steps can reuse it.
    if target_base_value_override is None:
        target_base_value = result_base_value
        need_reset_target = False
    else:
        target_base_value = target_base_value_override
        need_reset_target = True

    lines: list[str] = []
    if need_reset_target:
        lines.append(_load_large_int(result_base_register, target_base_value))
    else:
        lines.append(_load_large_int(result_base_register, target_base_value))

    # If the activation base is a register-held address (e.g. up_result_register
    # for down proj), ensure it holds its canonical value at the start of each
    # chunk so `+ chunk_act_base_offset` lands in the right spot.
    if reset_act_base_register and activation_base_register is not None:
        assert activation_base_register_value is not None
        lines.append(_load_large_int(activation_base_register, activation_base_register_value))

    for weight_row in range(out_size // blen):
        if weight_row % (mlen // blen) == 0:
            # Reset MRAM pointer for this MLEN block
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n")
            # HBM offset = chunk_hbm_base_offset + weight_row*blen
            lines.append(
                _addi_large_int(w_hbm_offset_register, 0, chunk_hbm_base_offset + weight_row * blen, w_temp_register)
                if chunk_hbm_base_offset + weight_row * blen >= (1 << 18)
                else f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, {chunk_hbm_base_offset + weight_row * blen} \n"
            )
            lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{result_base_register}, 0 \n")
            for _ in range(k_tile_count):
                lines.append(
                    f"H_PREFETCH_M gp{w_actual_register}, gp{w_hbm_offset_register}, a{weight_hbm_offset_reg}, 1, 0 \n"
                )
                lines.append(f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} \n")
                lines.append(
                    _addi_large_int(
                        w_hbm_offset_register, w_hbm_offset_register, mlen * weight_stride, w_temp_register
                    )
                )
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n")
        else:
            lines.append(
                f"S_ADDI_INT gp{w_actual_register}, gp0, {(weight_row % (mlen // blen)) * blen} \n"
            )
            lines.append(
                f"S_ADDI_INT gp{intermediate_register}, gp{result_base_register}, {(weight_row % (mlen // blen)) * blen} \n"
            )

        for act_col in range(num_act_cols):
            # Set activation pointer for this act_col + chunk.
            if activation_base_address is not None:
                addr = activation_base_address + act_col * mlen * blen + chunk_act_base_offset
                lines.append(
                    _addi_large_int(a_actual_register, 0, addr, w_temp_register)
                    if addr >= (1 << 18)
                    else f"S_ADDI_INT gp{a_actual_register}, gp0, {addr} \n"
                )
            else:
                # Activation base comes from a register (e.g. up_result_register).
                # a_actual = activation_base_register + act_col*mlen*blen + chunk_act_base_offset
                offset = act_col * mlen * blen + chunk_act_base_offset
                lines.append(
                    _addi_large_int(a_actual_register, activation_base_register, offset, w_temp_register)
                    if offset >= (1 << 18)
                    else f"S_ADDI_INT gp{a_actual_register}, gp{activation_base_register}, {offset} \n"
                )

            lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0 \n")
            for _ in range(k_tile_count):
                lines.append(f"M_MM 0, gp{w_temp_register}, gp{a_actual_register} \n")
                lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen} \n")
                lines.append(
                    f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len} \n"
                )
            lines.append(f"M_MM_WO gp{intermediate_register}, gp0, 0 \n")
            lines.append(f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen} \n")

        if (weight_row + 1) % (mlen // blen) == 0 and weight_row != out_size // blen - 1:
            lines.append(
                f"S_ADDI_INT gp{result_base_register}, gp{result_base_register}, {mlen * batch * seq_len} \n"
            )

    return "".join(lines)


def ffn_up_silu_asm(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    alive_registers: list[int],
    up_weight_hbm_offset_reg: int,
    const_one_fp_address: int,
    activation_base_address: int,
) -> str:
    """Up projection + SiLU only (no gate/down). Uses C_LOOP instructions."""
    # Register allocation
    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    a_actual_register = alive_registers[2]
    up_result_register = alive_registers[3]
    intermediate_register = alive_registers[4]
    w_hbm_offset_register = alive_registers[5]

    assert len(alive_registers) >= 10, "Loop version requires 10 registers (9 minimum + 1 temp)"
    loop_outer_reg = alive_registers[6]
    loop_inner_reg = alive_registers[7]
    loop_inner2_reg = alive_registers[8]
    temp_save_reg = alive_registers[9]  # Use this as temp save for a_actual_register

    generated_code = "; FFN Up Projection + SILU Generation\n"

    # Setup: scale/stride registers
    generated_code += _load_large_int(w_actual_register, hidden_size * intermediate_size)
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {intermediate_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base address for up result
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)

    # Upsize linear (loop)
    generated_code += "; FFN Upsize Linear Generation (Loop)\n"

    # Outer loop: weight_row from 0 to intermediate_size // mlen (MLEN blocks)
    num_mlen_blocks = intermediate_size // mlen
    tiles_per_mlen = mlen // blen
    num_weight_tiles = hidden_size // mlen
    num_act_cols = (batch * seq_len) // blen

    # w_hbm_offset_register tracks the START offset for each MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_weight_tiles):
        generated_code += (
            f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        )
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += _addi_large_int(
            a_actual_register, a_actual_register, mlen * intermediate_size, w_temp_register
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Inner loop: act_col iterations
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    # Copy weight pointer
    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual_register value before inner accumulation modifies it
    generated_code += f"S_ADDI_INT gp{temp_save_reg}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    # Write output
    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"

    # Restore a_actual_register and advance to next activation column
    act_col_advance = mlen * blen
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{temp_save_reg}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance up_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    generated_code += "; Result (up projection only) stored at up_result_register location\n"

    return generated_code


def ffn_intermediate_asm(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    alive_registers: list[int],
    gate_weight_hbm_offset_reg: int,
    up_weight_hbm_offset_reg: int,
    const_one_fp_address: int,
    activation_base_address: int,
) -> str:
    """Up + gate + SiLU (no down projection). Uses C_LOOP instructions."""
    # Register allocation
    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    a_actual_register = alive_registers[2]
    up_result_register = alive_registers[3]
    intermediate_register = alive_registers[4]
    gate_result_register = alive_registers[5]
    w_hbm_offset_register = alive_registers[6]

    assert len(alive_registers) >= 10, "Loop version requires 10 registers"
    loop_outer_reg = alive_registers[7]
    loop_inner_reg = alive_registers[8]
    loop_inner2_reg = alive_registers[9]

    generated_code = "; FFN Intermediate Generation (Up + Gate + SILU only)\n"

    # Setup: scale/stride registers
    generated_code += _load_large_int(w_actual_register, hidden_size * intermediate_size)
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {intermediate_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base addresses for results
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += _load_large_int(gate_result_register, batch * seq_len * (hidden_size + intermediate_size))

    # Upsize linear (loop)
    generated_code += "; FFN Upsize Linear Generation (Loop)\n"

    # Outer loop: weight_row from 0 to intermediate_size // mlen (MLEN blocks)
    num_mlen_blocks = intermediate_size // mlen
    tiles_per_mlen = mlen // blen
    num_weight_tiles = hidden_size // mlen
    num_act_cols = (batch * seq_len) // blen

    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_weight_tiles):
        generated_code += (
            f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        )
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += _addi_large_int(
            a_actual_register, a_actual_register, mlen * intermediate_size, w_temp_register
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Reset activation base for each middle loop iteration
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual before accumulation loop (gate_result_register as temp)
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    act_col_advance = mlen * blen
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{gate_result_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance up_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # Gate projection (loop)
    generated_code += "; FFN Gate Projection Generation (Loop)\n"

    # Reset base addresses
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += _load_large_int(gate_result_register, batch * seq_len * (hidden_size + intermediate_size))
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"

    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_weight_tiles):
        generated_code += (
            f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{gate_weight_hbm_offset_reg}, 1, 0\n"
        )
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += _addi_large_int(
            a_actual_register, a_actual_register, mlen * intermediate_size, w_temp_register
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Reset activation base for each middle loop iteration
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual before accumulation loop (up_result_register as temp)
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance gate_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # SiLU activation (loop)
    generated_code += "; SILU Generation (Loop)\n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"

    # Reset addresses
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += _load_large_int(gate_result_register, batch * seq_len * (hidden_size + intermediate_size))
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp0, {activation_base_address}\n"

    # Loop over batch * seq_len * (intermediate_size // vlen)
    num_silu_iters = batch * seq_len * (intermediate_size // vlen)
    generated_code += f"; SILU loop: {num_silu_iters} iterations\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_silu_iters}\n"

    # SILU computation: sigmoid(x) * x * gate
    generated_code += f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1\n"
    generated_code += f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    generated_code += f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0\n"
    generated_code += f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0\n"
    generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen}\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # Note: Result is stored in up_result_register at base address batch * seq_len * hidden_size
    generated_code += "; Intermediate result (up + gate + SILU) stored at up_result_register location\n"

    return generated_code


def _ffn_asm_with_loops(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    alive_registers: list[int],
    gate_weight_hbm_offset_reg: int,
    up_weight_hbm_offset_reg: int,
    down_weight_hbm_offset_reg: int,
    const_one_fp_address: int,
    activation_base_address: int,
) -> str:
    """Full FFN (up + gate + SiLU + down) using C_LOOP instructions."""

    # Register allocation
    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    a_actual_register = alive_registers[2]
    up_result_register = alive_registers[3]
    intermediate_register = alive_registers[4]
    gate_result_register = alive_registers[5]
    w_hbm_offset_register = alive_registers[6]

    assert len(alive_registers) >= 10, "Loop version requires 10 registers"
    loop_outer_reg = alive_registers[7]
    loop_inner_reg = alive_registers[8]
    loop_inner2_reg = alive_registers[9]

    generated_code = "; FFN Generation (Loop-Optimized)\n"

    # Setup: scale/stride registers
    generated_code += _load_large_int(w_actual_register, hidden_size * intermediate_size)
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {intermediate_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base addresses for results
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += _load_large_int(gate_result_register, batch * seq_len * (hidden_size + intermediate_size))

    # Upsize linear (loop)
    generated_code += "; FFN Upsize Linear Generation (Loop)\n"

    # Outer loop: weight_row from 0 to intermediate_size // mlen (MLEN blocks)
    num_mlen_blocks = intermediate_size // mlen
    tiles_per_mlen = mlen // blen
    num_weight_tiles = hidden_size // mlen
    num_act_cols = (batch * seq_len) // blen

    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_weight_tiles):
        generated_code += (
            f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        )
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += _addi_large_int(
            a_actual_register, a_actual_register, mlen * intermediate_size, w_temp_register
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Reset activation base for each middle loop iteration
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual before accumulation loop (gate_result_register as temp)
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    act_col_advance = mlen * blen
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{gate_result_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance up_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # Gate projection (loop)
    generated_code += "; FFN Gate Projection Generation (Loop)\n"

    # Reset base addresses
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += _load_large_int(gate_result_register, batch * seq_len * (hidden_size + intermediate_size))
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"

    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_weight_tiles):
        generated_code += (
            f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{gate_weight_hbm_offset_reg}, 1, 0\n"
        )
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += _addi_large_int(
            a_actual_register, a_actual_register, mlen * intermediate_size, w_temp_register
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Reset activation base for each middle loop iteration
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual before accumulation loop (up_result_register as temp)
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance gate_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # SiLU activation (loop)
    generated_code += "; SILU Generation (Loop)\n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"

    # Reset addresses
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += _load_large_int(gate_result_register, batch * seq_len * (hidden_size + intermediate_size))
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp0, {activation_base_address}\n"

    # Loop over batch * seq_len * (intermediate_size // vlen)
    num_silu_iters = batch * seq_len * (intermediate_size // vlen)
    generated_code += f"; SILU loop: {num_silu_iters} iterations\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_silu_iters}\n"

    # SILU computation: sigmoid(x) * x * gate
    generated_code += f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1\n"
    generated_code += f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    generated_code += f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0\n"
    generated_code += f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0\n"
    generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen}\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # Downsize linear (loop)
    generated_code += "; FFN Downsize Linear Generation (Loop)\n"

    # Setup scale and stride for downsize
    generated_code += _load_large_int(w_actual_register, hidden_size * intermediate_size)
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Result goes to activation base region
    act_result_register = gate_result_register
    generated_code += f"S_ADDI_INT gp{act_result_register}, gp0, {activation_base_address}\n"
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)

    # Downsize: (b*s, intermediate_size) @ (intermediate_size, hidden_size) -> (b*s, hidden_size)
    num_down_mlen_blocks = hidden_size // mlen
    num_down_weight_tiles = intermediate_size // mlen
    down_act_col_advance = mlen * blen

    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_down_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_down_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_down_weight_tiles):
        generated_code += (
            f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{down_weight_hbm_offset_reg}, 1, 0\n"
        )
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * hidden_size}\n"

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Reset activation base; up_result_register recomputed here (used as temp below)
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, 0\n"
    num_down_act_cols = (batch * seq_len) // blen
    generated_code += f"; Inner loop: {num_down_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_down_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual before accumulation (up_result_register recomputed each middle iter)
    down_act_save_reg = up_result_register
    generated_code += f"S_ADDI_INT gp{down_act_save_reg}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_down_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_down_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{down_act_save_reg}, {down_act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance act_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{act_result_register}, gp{act_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    return generated_code


def _ffn_asm_fused_up_gate(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    alive_registers: list[int],
    gate_weight_hbm_offset_reg: int,
    up_weight_hbm_offset_reg: int,
    down_weight_hbm_offset_reg: int,
    const_one_fp_address: int,
    activation_base_address: int,
) -> str:
    """Fused FFN: overlaps up/gate prefetch to reduce HBM traffic. Requires 12 registers."""

    # Register allocation for fused version
    assert len(alive_registers) >= 12, "Fused version requires 12 registers"

    w_actual_register = alive_registers[0]  # Weight MRAM offset (shared)
    w_temp_register = alive_registers[1]  # Weight temp pointer
    a_actual_register = alive_registers[2]  # Activation VRAM pointer
    up_result_register = alive_registers[3]  # Upsize result base
    intermediate_register = alive_registers[4]  # Output write pointer
    gate_result_register = alive_registers[5]  # Gate result base
    w_hbm_offset_register = alive_registers[6]  # HBM block offset for prefetch
    loop_outer_reg = alive_registers[7]  # Outer loop counter
    loop_inner_reg = alive_registers[8]  # Middle loop counter
    loop_inner2_reg = alive_registers[9]  # Inner loop counter
    # Extra registers for fused version
    a_save_register = alive_registers[10]  # Activation save
    w_gate_base_register = alive_registers[11]  # Gate weight base in MRAM

    generated_code = "; FFN Generation (Fused Up+Gate Optimized)\n"

    # Setup: scale/stride registers
    generated_code += _load_large_int(w_actual_register, hidden_size * intermediate_size)
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {intermediate_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base addresses for results
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += _load_large_int(gate_result_register, batch * seq_len * (hidden_size + intermediate_size))

    # Fused up + gate linear with overlapped prefetch
    generated_code += "; Fused Up+Gate Linear (overlapped prefetch optimization)\n"

    num_mlen_blocks = intermediate_size // mlen
    tiles_per_mlen = mlen // blen
    num_weight_tiles = hidden_size // mlen
    num_act_cols = (batch * seq_len) // blen
    act_col_advance = mlen * blen
    gate_mram_offset = num_weight_tiles * mlen * mlen  # Gate weights start after up weights in MRAM

    # Calculate how to spread GATE prefetches across UP computation
    # UP projection has tiles_per_mlen * num_act_cols iterations of inner work
    total_up_inner_iters = tiles_per_mlen * num_act_cols
    gate_prefetch_interval = max(1, total_up_inner_iters // num_weight_tiles)

    # HBM offset tracking
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch UP weights only (GATE will be prefetched during UP compute)
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"

    # Prefetch up weights (to MRAM at offset 0)
    for weight_col in range(num_weight_tiles):
        generated_code += (
            f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        )
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += _addi_large_int(
            a_actual_register, a_actual_register, mlen * intermediate_size, w_temp_register
        )

    # Setup for UP compute and GATE prefetch overlap
    generated_code += f"S_ADDI_INT gp{w_gate_base_register}, gp0, {gate_mram_offset}\n"

    # Reset for UP compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"

    # Up projection with interleaved gate prefetch
    generated_code += f"; Up projection for MLEN block (with GATE prefetch every {gate_prefetch_interval} iters)\n"

    # Unroll to interleave GATE prefetches during UP computation
    # NOTE: We compute GATE HBM offset directly from w_hbm_offset_register + offset
    # instead of tracking it in a_save_register (which is reused for weight offset in inner loop)
    gate_prefetch_count = 0
    gate_mram_ptr = gate_mram_offset

    for tile_idx in range(tiles_per_mlen):
        # Reset activation base for this tile
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"

        for act_col in range(num_act_cols):
            iter_num = tile_idx * num_act_cols + act_col

            # Check if we should insert a GATE prefetch
            if iter_num % gate_prefetch_interval == 0 and gate_prefetch_count < num_weight_tiles:
                generated_code += f"; Prefetch GATE weight tile {gate_prefetch_count} during UP compute\n"
                # Save current a_actual_register (activation pointer) to w_temp_register
                generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{a_actual_register}, 0\n"
                # Compute GATE HBM offset directly: base_offset + prefetch_count * stride
                gate_hbm_offset = gate_prefetch_count * mlen * intermediate_size
                # Set MRAM destination
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {gate_mram_ptr}\n"
                # Set HBM source: w_hbm_offset_register + gate_hbm_offset
                generated_code += f"S_ADDI_INT gp{a_save_register}, gp{w_hbm_offset_register}, {gate_hbm_offset}\n"
                generated_code += (
                    f"H_PREFETCH_M gp{a_actual_register}, gp{a_save_register}, a{gate_weight_hbm_offset_reg}, 1, 0\n"
                )
                gate_mram_ptr += mlen * mlen
                gate_prefetch_count += 1
                # Restore activation pointer
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_temp_register}, 0\n"

            # Save activation column base before weight tile loop modifies a_actual_register
            generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{a_actual_register}, 0\n"

            # UP weight accumulation
            generated_code += f"S_ADDI_INT gp{a_save_register}, gp{w_actual_register}, 0\n"  # save weight offset

            for inner_idx in range(num_weight_tiles):
                generated_code += f"M_MM 0, gp{a_save_register}, gp{a_actual_register}\n"
                generated_code += f"S_ADDI_INT gp{a_save_register}, gp{a_save_register}, {mlen * mlen}\n"
                if inner_idx < num_weight_tiles - 1:
                    generated_code += (
                        f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"
                    )

            generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"

            # Restore activation and advance to next column
            if act_col < num_act_cols - 1:
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_temp_register}, {act_col_advance}\n"

        # After all act_cols for this tile, advance weight offset
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
        generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
        generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    # Gate projection (weights already prefetched)
    generated_code += "; Gate projection for MLEN block (weights pre-fetched during UP)\n"
    generated_code += f"S_ADDI_INT gp{a_save_register}, gp0, 0\n"  # tile offset tracker for output
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_gate_base_register}, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"

    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save activation pointer - use w_gate_base_register
    generated_code += f"S_ADDI_INT gp{w_gate_base_register}, gp{a_actual_register}, 0\n"

    for inner_idx in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_idx < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore activation from saved base + advance to next column
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_gate_base_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    generated_code += f"S_ADDI_INT gp{a_save_register}, gp{a_save_register}, {blen}\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{a_save_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # Advance for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # SiLU activation with overlapped down-weight prefetch
    num_down_mlen_blocks = hidden_size // mlen
    num_down_weight_tiles = intermediate_size // mlen
    num_silu_iters = batch * seq_len * (intermediate_size // vlen)

    generated_code += "; SILU Generation (with overlapped DOWN prefetch)\n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"

    # Set up DOWN weight prefetch parameters
    generated_code += _load_large_int(w_actual_register, hidden_size * intermediate_size)
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"

    # Initialize SILU pointers
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += _load_large_int(gate_result_register, batch * seq_len * (hidden_size + intermediate_size))
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp0, {activation_base_address}\n"

    # Initialize DOWN prefetch pointers (w_actual_register=MRAM offset, a_actual_register=HBM offset)
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, 0\n"

    # Compute how many SILU iters per prefetch op to spread prefetches across SILU loop
    # We have num_down_weight_tiles prefetches to do for the first block
    # Spread them evenly across the SILU loop
    prefetch_interval = max(1, num_silu_iters // num_down_weight_tiles)

    generated_code += f"; SILU loop: {num_silu_iters} iterations (prefetch every {prefetch_interval} iters)\n"
    generated_code += f"; Prefetching {num_down_weight_tiles} DOWN weight tiles during SILU\n"

    # Unroll SILU loop to interleave prefetch operations
    for silu_iter in range(num_silu_iters):
        # SILU computation
        generated_code += f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1\n"
        generated_code += f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
        generated_code += f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0\n"
        generated_code += f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
        generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0\n"
        generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0\n"
        generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen}\n"
        generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen}\n"

        # Insert prefetch at appropriate intervals
        prefetch_idx = silu_iter // prefetch_interval
        if silu_iter % prefetch_interval == 0 and prefetch_idx < num_down_weight_tiles:
            generated_code += f"; Prefetch DOWN weight tile {prefetch_idx}\n"
            generated_code += (
                f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{down_weight_hbm_offset_reg}, 1, 0\n"
            )
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * hidden_size}\n"

    # Downsize linear (first block already prefetched)
    generated_code += "; FFN Downsize Linear Generation (first block pre-fetched during SILU)\n"

    act_result_register = gate_result_register
    generated_code += f"S_ADDI_INT gp{act_result_register}, gp0, {activation_base_address}\n"
    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)

    down_act_col_advance = mlen * blen

    # First block: weights already prefetched, just do computation
    generated_code += "; First DOWN block (weights pre-fetched during SILU)\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, {mlen}\n"  # Next block HBM offset

    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
    tiles_per_mlen_down = mlen // blen

    # First block computation
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen_down}\n"

    generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, 0\n"
    num_down_act_cols = (batch * seq_len) // blen

    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_down_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

    for inner_idx in range(num_down_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_idx < num_down_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {down_act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # Advance to second block base
    generated_code += f"S_ADDI_INT gp{act_result_register}, gp{act_result_register}, {mlen * batch * seq_len}\n"

    # Remaining blocks (if any) - standard prefetch then compute
    if num_down_mlen_blocks > 1:
        generated_code += f"; Remaining {num_down_mlen_blocks - 1} DOWN blocks\n"
        generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_down_mlen_blocks - 1}\n"

        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
        for weight_col in range(num_down_weight_tiles):
            generated_code += (
                f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{down_weight_hbm_offset_reg}, 1, 0\n"
            )
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * hidden_size}\n"

        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
        generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"

        generated_code += f"; Middle loop: {tiles_per_mlen_down} tiles per MLEN block\n"
        generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen_down}\n"

        generated_code += _load_large_int(up_result_register, batch * seq_len * hidden_size)
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, 0\n"

        generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_down_act_cols}\n"

        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
        generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

        for inner_idx in range(num_down_weight_tiles):
            generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
            generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
            if inner_idx < num_down_weight_tiles - 1:
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

        generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
        generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {down_act_col_advance}\n"

        generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
        generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
        generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

        generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

        generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
        generated_code += f"S_ADDI_INT gp{act_result_register}, gp{act_result_register}, {mlen * batch * seq_len}\n"

        generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    return generated_code
