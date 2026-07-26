from __future__ import annotations

import math

from ._imm import addi_large_int_str as _addi_large_int
from ._imm import IMM2_BOUND
from ._imm import load_large_int_str as _load_large_int
from ._k_split import k_chunks as _k_chunks
from .ffn_address_plan import (
    FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1,
    build_ffn_address_plan,
    uses_invariant_stride,
)
from .ffn_projection_plan import (
    FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2,
    FFN_PROJECTION_SCHEDULE_LEGACY_AUTO_V1,
    build_ffn_projection_plan,
)


def _ffn_relative_update(
    register: int,
    value: int,
    *,
    mode: str,
    stride_register: int,
) -> str:
    if uses_invariant_stride(value, update_count=1, mode=mode):
        return (
            f"S_ADD_INT gp{register}, gp{register}, "
            f"gp{stride_register}\n"
        )
    return f"S_ADDI_INT gp{register}, gp{register}, {value}\n"


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
    workspace_base_address: int = 0,
    ffn_address_schedule: str = FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1,
    ffn_projection_schedule: str = FFN_PROJECTION_SCHEDULE_LEGACY_AUTO_V1,
) -> str:
    """
    Generates assembly code for a FFN operation.

    Set use_loop_instructions=True to use C_LOOP_START/END for compact code.
    Set use_fused_up_gate=True to fuse upsize and gate projections (requires 12 registers).

    ``workspace_base_address`` is the VRAM base for FFN temporaries. The default
    preserves the historical direct_emit layout at low VRAM addresses. Compiler
    lowering should pass an allocator-managed base so FFN temporaries cannot
    clobber persistent tensors such as RoPE tables.

    ``matrix_sram_size`` is the MRAM capacity (element-units-per-tile * tiles). When
    a projection's K dimension exceeds ``matrix_sram_size // mlen`` tiles, the template
    emits a K-split partial-sum accumulation loop (mirroring
    ``aten/ops/plena/linear_ops.py::linear_plena``). This prevents OOB
    ``H_PREFETCH_M`` addresses for models whose intermediate/hidden dims exceed the
    MRAM tile count.
    """
    if ffn_projection_schedule == FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2:
        if use_fused_up_gate:
            raise ValueError(
                "affine-loop-v2 does not support the experimental fused up/gate path"
            )
        return _ffn_asm_affine_loop_v2(
            mlen=mlen,
            vlen=vlen,
            blen=blen,
            batch=batch,
            seq_len=seq_len,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            alive_registers=alive_registers,
            gate_weight_hbm_offset_reg=gate_weight_hbm_offset_reg,
            up_weight_hbm_offset_reg=up_weight_hbm_offset_reg,
            down_weight_hbm_offset_reg=down_weight_hbm_offset_reg,
            const_one_fp_address=const_one_fp_address,
            activation_base_address=activation_base_address,
            matrix_sram_size=matrix_sram_size,
            workspace_base_address=workspace_base_address,
            ffn_address_schedule=ffn_address_schedule,
        )
    if ffn_projection_schedule != FFN_PROJECTION_SCHEDULE_LEGACY_AUTO_V1:
        raise ValueError(
            f"unsupported ffn_projection_schedule={ffn_projection_schedule!r}"
        )
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
            workspace_base_address=workspace_base_address,
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
            workspace_base_address=workspace_base_address,
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
            workspace_base_address=workspace_base_address,
            ffn_address_schedule=ffn_address_schedule,
        )


def _ffn_workspace_layout(
    batch: int,
    seq_len: int,
    intermediate_size: int,
    workspace_base_address: int,
) -> tuple[int, int, int]:
    rows = batch * seq_len
    up_result_base = workspace_base_address
    gate_result_base = up_result_base + rows * intermediate_size
    scratch_base = gate_result_base + rows * intermediate_size
    return up_result_base, gate_result_base, scratch_base


def _loop_start(register: int, count: int) -> str:
    return f"C_LOOP_START gp{register}, {count}\n" if count > 1 else ""


def _loop_end(register: int, count: int) -> str:
    return f"C_LOOP_END gp{register}\n" if count > 1 else ""


def _emit_ffn_projection_affine_chunk(
    *,
    mlen: int,
    blen: int,
    batch_rows: int,
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
    target_base_value: int,
    w_actual_register: int,
    w_temp_register: int,
    a_actual_register: int,
    intermediate_register: int,
    hbm_block_base_register: int,
    activation_chunk_base_register: int,
    output_block_loop_register: int,
    output_tile_loop_register: int,
    activation_column_loop_register: int,
    k_loop_register: int,
    working_base_register: int,
    section_comment: str,
) -> str:
    """Render one projection chunk as explicit affine hardware loops.

    The loop order and all Matrix/HBM addresses match
    ``_emit_ffn_projection_chunk``.  Only address generation changes: the
    existing loop AGU sees the affine updates directly instead of recovering
    loops from an expanded instruction stream.
    """

    output_blocks = out_size // mlen
    output_tiles = mlen // blen
    activation_columns = batch_rows // blen
    chunk_hbm_base = k_start_tile * mlen * weight_stride
    chunk_activation_offset = k_start_tile * mlen * batch_rows

    lines = [
        f" ; {section_comment}: affine chunk k_start={k_start_tile}, "
        f"k_count={k_tile_count}\n",
        _load_large_int(result_base_register, target_base_value),
        _load_large_int(hbm_block_base_register, chunk_hbm_base),
    ]
    if activation_base_address is not None:
        lines.append(
            _load_large_int(
                activation_chunk_base_register,
                activation_base_address + chunk_activation_offset,
            )
        )
    else:
        if activation_base_register is None or activation_base_register_value is None:
            raise ValueError(
                "register-based affine FFN input requires a canonical base value"
            )
        lines.append(
            _load_large_int(
                activation_base_register, activation_base_register_value
            )
        )
        lines.append(
            _addi_large_int(
                activation_chunk_base_register,
                activation_base_register,
                chunk_activation_offset,
                w_temp_register,
            )
        )

    lines.append(_loop_start(output_block_loop_register, output_blocks))

    # Prefetch this output block's K chunk into MRAM.
    lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n")
    lines.append(
        f"S_ADDI_INT gp{working_base_register}, "
        f"gp{hbm_block_base_register}, 0\n"
    )
    lines.append(_loop_start(k_loop_register, k_tile_count))
    lines.append(
        f"H_PREFETCH_M gp{w_actual_register}, gp{working_base_register}, "
        f"a{weight_hbm_offset_reg}, 1, 0\n"
    )
    if k_tile_count > 1:
        lines.append(
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, "
            f"{mlen * mlen}\n"
        )
        lines.append(
            f"S_ADDI_INT gp{working_base_register}, "
            f"gp{working_base_register}, {mlen * weight_stride}\n"
        )
    lines.append(_loop_end(k_loop_register, k_tile_count))

    # Compute every BLEN output tile for the current MLEN output block.
    lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n")
    lines.append(_loop_start(output_tile_loop_register, output_tiles))
    lines.append(
        f"S_ADDI_INT gp{working_base_register}, "
        f"gp{activation_chunk_base_register}, 0\n"
    )
    lines.append(
        f"S_ADD_INT gp{intermediate_register}, "
        f"gp{result_base_register}, gp{w_actual_register}\n"
    )
    lines.append(_loop_start(activation_column_loop_register, activation_columns))
    lines.append(
        f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    )
    lines.append(
        f"S_ADDI_INT gp{a_actual_register}, gp{working_base_register}, 0\n"
    )
    lines.append(_loop_start(k_loop_register, k_tile_count))
    lines.append(f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n")
    if k_tile_count > 1:
        lines.append(
            f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, "
            f"{mlen * mlen}\n"
        )
        lines.append(
            f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, "
            f"{mlen * batch_rows}\n"
        )
    lines.append(_loop_end(k_loop_register, k_tile_count))
    lines.append(f"M_MM_WO gp{intermediate_register}, gp0, 0\n")
    if activation_columns > 1:
        lines.append(
            f"S_ADDI_INT gp{working_base_register}, "
            f"gp{working_base_register}, {mlen * blen}\n"
        )
        lines.append(
            f"S_ADDI_INT gp{intermediate_register}, "
            f"gp{intermediate_register}, {blen * mlen}\n"
        )
    lines.append(
        _loop_end(activation_column_loop_register, activation_columns)
    )
    if output_tiles > 1:
        lines.append(
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
        )
    lines.append(_loop_end(output_tile_loop_register, output_tiles))

    if output_blocks > 1:
        lines.append(
            f"S_ADDI_INT gp{hbm_block_base_register}, "
            f"gp{hbm_block_base_register}, {mlen}\n"
        )
        lines.append(
            f"S_ADDI_INT gp{result_base_register}, "
            f"gp{result_base_register}, {mlen * batch_rows}\n"
        )
    lines.append(_loop_end(output_block_loop_register, output_blocks))
    return "".join(lines)


def _ffn_asm_affine_loop_v2(
    *,
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
    matrix_sram_size: int,
    workspace_base_address: int,
    ffn_address_schedule: str,
) -> str:
    """Render the dense FFN through the unified affine-loop projection plan."""

    if vlen != mlen:
        raise ValueError("affine-loop-v2 requires VLEN == MLEN")
    if len(alive_registers) < 15:
        raise ValueError("affine-loop-v2 requires gp1-gp15")
    if ffn_address_schedule != FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1:
        raise ValueError(
            "affine-loop-v2 requires ffn_address_schedule='live-stride-v1'"
        )

    up_base, gate_base, scratch_base = _ffn_workspace_layout(
        batch, seq_len, intermediate_size, workspace_base_address
    )
    batch_rows = batch * seq_len
    max_k_tiles = max(1, matrix_sram_size // mlen)

    (
        w_actual,
        w_temp,
        a_actual,
        up_result,
        intermediate,
        gate_result,
        hbm_block_base,
        activation_chunk_base,
        _unused_gp9,
        _unused_gp10,
        output_block_loop,
        output_tile_loop,
        activation_column_loop,
        k_loop,
        working_base,
    ) = alive_registers[:15]

    lines = ["; FFN Generation (affine-loop-v2)\n"]

    def projection(
        *,
        name: str,
        k_size: int,
        out_size: int,
        weight_stride: int,
        weight_addr_reg: int,
        result_register: int,
        result_base: int,
        input_address: int | None,
        input_register: int | None,
        input_register_value: int | None,
        configure_matrix: bool,
    ) -> None:
        if configure_matrix:
            lines.append(
                _load_large_int(w_actual, hidden_size * intermediate_size)
            )
            lines.append(f"C_SET_SCALE_REG gp{w_actual}\n")
            lines.append(_load_large_int(w_actual, weight_stride))
            lines.append(f"C_SET_STRIDE_REG gp{w_actual}\n")
        plan = build_ffn_projection_plan(
            schedule=FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2,
            mlen=mlen,
            blen=blen,
            batch_rows=batch_rows,
            k_size=k_size,
            out_size=out_size,
            max_k_tiles=max_k_tiles,
        )
        for chunk in plan.chunks:
            target = result_base if chunk.chunk_index == 0 else scratch_base
            lines.append(
                _emit_ffn_projection_affine_chunk(
                    mlen=mlen,
                    blen=blen,
                    batch_rows=batch_rows,
                    k_size=k_size,
                    out_size=out_size,
                    weight_stride=weight_stride,
                    weight_hbm_offset_reg=weight_addr_reg,
                    result_base_register=result_register,
                    result_base_value=result_base,
                    activation_base_address=input_address,
                    activation_base_register=input_register,
                    activation_base_register_value=input_register_value,
                    k_start_tile=chunk.k_start_tile,
                    k_tile_count=chunk.k_tile_count,
                    target_base_value=target,
                    w_actual_register=w_actual,
                    w_temp_register=w_temp,
                    a_actual_register=a_actual,
                    intermediate_register=intermediate,
                    hbm_block_base_register=hbm_block_base,
                    activation_chunk_base_register=activation_chunk_base,
                    output_block_loop_register=output_block_loop,
                    output_tile_loop_register=output_tile_loop,
                    activation_column_loop_register=activation_column_loop,
                    k_loop_register=k_loop,
                    working_base_register=working_base,
                    section_comment=name,
                )
            )
            if chunk.chunk_index:
                vector_adds = math.ceil(out_size * batch_rows / vlen)
                lines.append(_load_large_int(w_actual, result_base))
                lines.append(_load_large_int(w_temp, scratch_base))
                lines.append(_loop_start(output_block_loop, vector_adds))
                lines.append(
                    f"V_ADD_VV gp{w_actual}, gp{w_actual}, gp{w_temp}, 0\n"
                )
                if vector_adds > 1:
                    lines.append(
                        f"S_ADDI_INT gp{w_actual}, gp{w_actual}, {vlen}\n"
                    )
                    lines.append(
                        f"S_ADDI_INT gp{w_temp}, gp{w_temp}, {vlen}\n"
                    )
                lines.append(_loop_end(output_block_loop, vector_adds))

    projection(
        name="FFN up projection",
        k_size=hidden_size,
        out_size=intermediate_size,
        weight_stride=intermediate_size,
        weight_addr_reg=up_weight_hbm_offset_reg,
        result_register=up_result,
        result_base=up_base,
        input_address=activation_base_address,
        input_register=None,
        input_register_value=None,
        configure_matrix=True,
    )
    projection(
        name="FFN gate projection",
        k_size=hidden_size,
        out_size=intermediate_size,
        weight_stride=intermediate_size,
        weight_addr_reg=gate_weight_hbm_offset_reg,
        result_register=gate_result,
        result_base=gate_base,
        input_address=activation_base_address,
        input_register=None,
        input_register_value=None,
        configure_matrix=False,
    )

    lines.extend(
        (
            "; FFN SiLU (affine loop)\n",
            f"S_LD_FP f1, gp0, {const_one_fp_address}\n",
            _load_large_int(up_result, up_base),
            _load_large_int(gate_result, gate_base),
            _load_large_int(intermediate, activation_base_address),
        )
    )
    silu_iterations = batch_rows * (intermediate_size // vlen)
    lines.append(_loop_start(output_block_loop, silu_iterations))
    lines.extend(
        (
            f"V_SUB_VF gp{intermediate}, gp{up_result}, f0, 0, 1\n",
            f"V_EXP_V gp{intermediate}, gp{intermediate}, 0\n",
            f"V_ADD_VF gp{intermediate}, gp{intermediate}, f1, 0\n",
            f"V_RECI_V gp{intermediate}, gp{intermediate}, 0\n",
            f"V_MUL_VV gp{intermediate}, gp{intermediate}, gp{up_result}, 0\n",
            f"V_MUL_VV gp{up_result}, gp{intermediate}, gp{gate_result}, 0\n",
        )
    )
    if silu_iterations > 1:
        lines.extend(
            (
                f"S_ADDI_INT gp{gate_result}, gp{gate_result}, {vlen}\n",
                f"S_ADDI_INT gp{up_result}, gp{up_result}, {vlen}\n",
            )
        )
    lines.append(_loop_end(output_block_loop, silu_iterations))

    projection(
        name="FFN down projection",
        k_size=intermediate_size,
        out_size=hidden_size,
        weight_stride=hidden_size,
        weight_addr_reg=down_weight_hbm_offset_reg,
        result_register=gate_result,
        result_base=activation_base_address,
        input_address=None,
        input_register=up_result,
        input_register_value=up_base,
        configure_matrix=True,
    )
    return "".join(lines)


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
    workspace_base_address: int = 0,
    ffn_address_schedule: str = FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1,
) -> str:
    """Unrolled FFN: up + gate + SiLU + down projections.

    When a projection's K dimension tile-count exceeds
    ``matrix_sram_size // mlen``, we split K into chunks of at most
    ``MAX_K_TILES = matrix_sram_size // mlen`` and accumulate partial sums in
    VRAM via V_ADD_VV. This mirrors the ATen-path K-split in
    ``aten/ops/plena/linear_ops.py::linear_plena``.
    """

    up_result_base, gate_result_base, scratch_base = _ffn_workspace_layout(
        batch, seq_len, intermediate_size, workspace_base_address
    )

    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    a_actual_register = alive_registers[2]
    up_result_register = alive_registers[3]
    intermediate_register = alive_registers[4]
    gate_result_register = alive_registers[5]
    w_hbm_offset_register = alive_registers[6]
    matrix_stride_register = alive_registers[7]
    result_stride_register = alive_registers[8]
    output_stride_register = alive_registers[9]

    # reset the registers
    generated_code = "; FFN Generation \n"

    # Settings for up and gate weight matrices prefetching
    generated_code += _load_large_int(
        w_actual_register, hidden_size * intermediate_size
    )
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register} \n"
    generated_code += _load_large_int(w_actual_register, intermediate_size)
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
    # Set the address for on-chip sram
    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += _load_large_int(gate_result_register, gate_result_base)
    if ffn_address_schedule == FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1:
        if mlen * mlen >= IMM2_BOUND:
            generated_code += _load_large_int(
                matrix_stride_register, mlen * mlen
            )
        if mlen * batch * seq_len >= IMM2_BOUND:
            generated_code += _load_large_int(
                result_stride_register, mlen * batch * seq_len
            )
        if blen * mlen >= IMM2_BOUND:
            generated_code += _load_large_int(
                output_stride_register, blen * mlen
            )

    # K-split config: when K tile count > MRAM tile capacity, we split K and
    # accumulate partial sums. `activation` region (used as input) starts at
    # `activation_base_address`; the K-split scratch region lives after the
    # allocator-managed up/gate output regions.
    MAX_K_TILES = max(1, matrix_sram_size // mlen)

    # --- FFN Upsize Linear (K = hidden_size) ---
    up_num_k_tiles = hidden_size // mlen
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
        result_base_value=up_result_base,
        activation_base_address=activation_base_address,
        activation_base_register=None,
        max_k_tiles=MAX_K_TILES,
        w_actual_register=w_actual_register,
        w_temp_register=w_temp_register,
        a_actual_register=a_actual_register,
        intermediate_register=intermediate_register,
        w_hbm_offset_register=w_hbm_offset_register,
        scratch_base_value=scratch_base,
        section_comment="FFN Upsize Linear Generation",
        ffn_address_schedule=ffn_address_schedule,
        matrix_stride_register=matrix_stride_register,
        result_stride_register=result_stride_register,
        output_stride_register=output_stride_register,
    )

    generated_code += " ; FFN Gate Projection Generation \n"
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
        result_base_value=gate_result_base,
        activation_base_address=activation_base_address,
        activation_base_register=None,
        max_k_tiles=MAX_K_TILES,
        w_actual_register=w_actual_register,
        w_temp_register=w_temp_register,
        a_actual_register=a_actual_register,
        intermediate_register=intermediate_register,
        w_hbm_offset_register=w_hbm_offset_register,
        scratch_base_value=scratch_base,
        section_comment="FFN Gate Projection (inlined)",
        ffn_address_schedule=ffn_address_schedule,
        matrix_stride_register=matrix_stride_register,
        result_stride_register=result_stride_register,
        output_stride_register=output_stride_register,
    )

    generated_code += "; SILU Generation \n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address} \n"
    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += _load_large_int(gate_result_register, gate_result_base)
    generated_code += _load_large_int(intermediate_register, activation_base_address)

    # SiLU: sigmoid(x) * x * gate, using activation region as scratchpad
    for b in range(batch * seq_len):
        for i in range(intermediate_size // vlen):
            generated_code += f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1 \n"
            generated_code += (
                f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0 \n"
            )
            generated_code += f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0 \n"
            generated_code += (
                f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0 \n"
            )
            generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0 \n"
            generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0 \n"
            generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen} \n"
            generated_code += (
                f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen} \n"
            )

    generated_code += "; FFN Downsize Linear Generation \n"
    generated_code += _load_large_int(
        w_actual_register, hidden_size * intermediate_size
    )
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register} \n"
    generated_code += _load_large_int(w_actual_register, hidden_size)
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
    if ffn_address_schedule != FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1:
        generated_code += (
            f"S_ADDI_INT gp{matrix_stride_register}, gp0, "
            f"{((batch * seq_len) // blen)} \n"
        )
    # Storing the results to the activation base region
    act_result_register = gate_result_register
    # Down projection: K = intermediate_size. Activation input is at
    # VRAM address up_result_base (post-SiLU). Scratch for K-split lives past
    # the gate region so it never collides with input or output.
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
        activation_base_register_value=up_result_base,
        max_k_tiles=max(1, matrix_sram_size // mlen),
        w_actual_register=w_actual_register,
        w_temp_register=w_temp_register,
        a_actual_register=a_actual_register,
        intermediate_register=intermediate_register,
        w_hbm_offset_register=w_hbm_offset_register,
        scratch_base_value=scratch_base,
        section_comment="FFN Downsize Linear (inlined)",
        ffn_address_schedule=ffn_address_schedule,
        matrix_stride_register=matrix_stride_register,
        result_stride_register=result_stride_register,
        output_stride_register=output_stride_register,
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
    ffn_address_schedule: str,
    matrix_stride_register: int,
    result_stride_register: int,
    output_stride_register: int,
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

    assert k_size % mlen == 0, f"K ({k_size}) must be a multiple of MLEN ({mlen})"
    assert (
        out_size % mlen == 0
    ), f"out_size ({out_size}) must be a multiple of MLEN ({mlen})"
    num_k_tiles = k_size // mlen
    num_act_cols = (batch * seq_len) // blen

    lines: list[str] = [
        f" ; {section_comment} (k_size={k_size}, out_size={out_size})\n"
    ]

    if num_k_tiles <= max_k_tiles:
        lines.append(
            f" ; K-split inactive: num_k_tiles={num_k_tiles} <= MAX_K_TILES={max_k_tiles}\n"
        )
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
                ffn_address_schedule=ffn_address_schedule,
                matrix_stride_register=matrix_stride_register,
                result_stride_register=result_stride_register,
                output_stride_register=output_stride_register,
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
    per_vlen_adds = math.ceil(output_elements / vlen)

    for chunk_idx, (k_start, k_count) in enumerate(chunks):
        lines.append(
            f" ; K-chunk {chunk_idx}/{len(chunks)}: k_start_tile={k_start}, k_count={k_count}\n"
        )
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
                ffn_address_schedule=ffn_address_schedule,
                matrix_stride_register=matrix_stride_register,
                result_stride_register=result_stride_register,
                output_stride_register=output_stride_register,
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
                lines.append(
                    f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {vlen} \n"
                )
                lines.append(
                    f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {vlen} \n"
                )

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
    ffn_address_schedule: str,
    matrix_stride_register: int,
    result_stride_register: int,
    output_stride_register: int,
) -> str:
    """Emit one K-chunk of an FFN projection.

    Mirrors the existing un-rolled projection structure but restricted to K
    tiles ``[k_start_tile, k_start_tile + k_tile_count)`` and capable of
    redirecting the output store to a scratch region.

    - HBM offset for a chunk starts at ``weight_row*blen + k_start_tile * mlen * weight_stride``
    - Activation offset for a chunk is advanced by ``k_start_tile * mlen * batch*seq_len``
    - MRAM prefetch destination always resets to 0 per MLEN block (we only
      prefetch this chunk's ``k_tile_count`` tiles)

    M_MM_WO semantics (confirmed from transactional_emulator/src/main.rs):
    ``mm_wo`` OVERWRITEs the VRAM slice with the current m_accum, then zeros
    m_accum.  It does NOT accumulate into existing VRAM content.  Therefore
    the first-chunk direct write to the real output region is safe, and the
    K-split partial-sum accumulation via V_ADD_VV is the correct mechanism.

    Note: previous single-pass code emitted M_MM against w_actual_register
    (a latent bug when hidden_size > mlen — the MRAM pointer did not advance
    between inner-loop iterations). This unified path uses w_temp_register
    throughout, matching K-split semantics, so hidden_size > mlen now
    correctly advances the MRAM pointer.
    """

    assert k_size % mlen == 0, f"K ({k_size}) must be a multiple of MLEN ({mlen})"
    num_act_cols = (batch * seq_len) // blen
    address_plan = build_ffn_address_plan(
        mode=ffn_address_schedule,
        k_tile_count=k_tile_count,
        num_activation_columns=num_act_cols,
    )
    chunk_hbm_base_offset = k_start_tile * mlen * weight_stride
    chunk_act_base_offset = k_start_tile * mlen * batch * seq_len

    # Target base: first chunk writes to real output, subsequent chunks write to scratch.
    # Both cases load result_base_register with the appropriate base value so the
    # MLEN-block advancement steps (S_ADDI_INT result_base_register, ...) work uniformly.
    target_base_value = (
        result_base_value
        if target_base_value_override is None
        else target_base_value_override
    )

    lines: list[str] = []
    lines.append(_load_large_int(result_base_register, target_base_value))

    # If the activation base is a register-held address (e.g. up_result_register
    # for down proj), ensure it holds its canonical value at the start of each
    # chunk so `+ chunk_act_base_offset` lands in the right spot.
    if reset_act_base_register and activation_base_register is not None:
        assert activation_base_register_value is not None
        lines.append(
            _load_large_int(activation_base_register, activation_base_register_value)
        )

    for weight_row in range(out_size // blen):
        if weight_row % (mlen // blen) == 0:
            # Reset MRAM pointer for this MLEN block
            lines.append(f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n")
            # HBM offset = chunk_hbm_base_offset + weight_row*blen
            lines.append(
                _addi_large_int(
                    w_hbm_offset_register,
                    0,
                    chunk_hbm_base_offset + weight_row * blen,
                    w_temp_register,
                )
                if chunk_hbm_base_offset + weight_row * blen >= (1 << 18)
                else f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, {chunk_hbm_base_offset + weight_row * blen} \n"
            )
            lines.append(
                f"S_ADDI_INT gp{intermediate_register}, gp{result_base_register}, 0 \n"
            )
            for tile_index in range(k_tile_count):
                lines.append(
                    f"H_PREFETCH_M gp{w_actual_register}, gp{w_hbm_offset_register}, a{weight_hbm_offset_reg}, 1, 0 \n"
                )
                if tile_index < address_plan.prefetch_pointer_updates:
                    lines.append(
                        _ffn_relative_update(
                            w_actual_register,
                            mlen * mlen,
                            mode=ffn_address_schedule,
                            stride_register=matrix_stride_register,
                        )
                    )
                    lines.append(
                        _addi_large_int(
                            w_hbm_offset_register,
                            w_hbm_offset_register,
                            mlen * weight_stride,
                            w_temp_register,
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
                addr = (
                    activation_base_address
                    + act_col * mlen * blen
                    + chunk_act_base_offset
                )
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
                    _addi_large_int(
                        a_actual_register,
                        activation_base_register,
                        offset,
                        w_temp_register,
                    )
                    if offset >= (1 << 18)
                    else f"S_ADDI_INT gp{a_actual_register}, gp{activation_base_register}, {offset} \n"
                )

            lines.append(f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0 \n")
            # Inner K-accumulation rolled into a hardware C_LOOP. The body is
            # already loop-carried (w_temp += MLEN^2, a_actual += MLEN*B*S), and
            # the weight HBM-offset register is dead here (only the per-MLEN-block
            # prefetch above uses it, and it is reloaded each block), so it doubles
            # as the trip counter. The systolic accumulator sums across all
            # k_tile_count M_MMs before the M_MM_WO flush, identical to the
            # unrolled form; this collapses 3*k_tile_count emitted lines to ~5.
            if k_tile_count > 1:
                lines.append(f"C_LOOP_START gp{w_hbm_offset_register}, {k_tile_count} \n")
            lines.append(f"M_MM 0, gp{w_temp_register}, gp{a_actual_register} \n")
            if address_plan.matrix_pointer_updates:
                # Keep loop-carried updates as immediates so loop-AGU-v1 can
                # recognize and eliminate them from hardware-loop bodies.
                lines.append(
                    f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen} \n"
                )
                lines.append(
                    f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len} \n"
                )
            if k_tile_count > 1:
                lines.append(f"C_LOOP_END gp{w_hbm_offset_register} \n")
            lines.append(f"M_MM_WO gp{intermediate_register}, gp0, 0 \n")
            if act_col < address_plan.output_pointer_updates:
                lines.append(
                    _ffn_relative_update(
                        intermediate_register,
                        blen * mlen,
                        mode=ffn_address_schedule,
                        stride_register=output_stride_register,
                    )
                )

        if (weight_row + 1) % (
            mlen // blen
        ) == 0 and weight_row != out_size // blen - 1:
            lines.append(
                _ffn_relative_update(
                    result_base_register,
                    mlen * batch * seq_len,
                    mode=ffn_address_schedule,
                    stride_register=result_stride_register,
                )
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
    workspace_base_address: int = 0,
) -> str:
    """Up projection + SiLU only (no gate/down). Uses C_LOOP instructions."""
    # Register allocation
    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    a_actual_register = alive_registers[2]
    up_result_register = alive_registers[3]
    intermediate_register = alive_registers[4]
    w_hbm_offset_register = alive_registers[5]

    assert (
        len(alive_registers) >= 10
    ), "Loop version requires 10 registers (9 minimum + 1 temp)"
    loop_outer_reg = alive_registers[6]
    loop_inner_reg = alive_registers[7]
    loop_inner2_reg = alive_registers[8]
    temp_save_reg = alive_registers[9]  # Use this as temp save for a_actual_register

    generated_code = "; FFN Up Projection + SILU Generation\n"

    # Setup: scale/stride registers
    generated_code += _load_large_int(
        w_actual_register, hidden_size * intermediate_size
    )
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += _load_large_int(w_actual_register, intermediate_size)
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base address for up result
    up_result_base, _, _ = _ffn_workspace_layout(
        batch, seq_len, intermediate_size, workspace_base_address
    )
    generated_code += _load_large_int(up_result_register, up_result_base)

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
    generated_code += (
        f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    )
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        )
        generated_code += _addi_large_int(
            a_actual_register,
            a_actual_register,
            mlen * intermediate_size,
            w_temp_register,
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    )

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Inner loop: act_col iterations
    generated_code += _load_large_int(a_actual_register, activation_base_address)
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    # Copy weight pointer
    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual_register value before inner accumulation modifies it
    generated_code += f"S_ADDI_INT gp{temp_save_reg}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += (
            f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        )
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    # Write output
    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"

    # Restore a_actual_register and advance to next activation column
    act_col_advance = mlen * blen
    generated_code += (
        f"S_ADDI_INT gp{a_actual_register}, gp{temp_save_reg}, {act_col_advance}\n"
    )

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    )
    # Reset intermediate back for next tile row
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    )
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    )
    # Advance up_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    generated_code += (
        "; Result (up projection only) stored at up_result_register location\n"
    )

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
    workspace_base_address: int = 0,
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
    up_result_base, gate_result_base, _ = _ffn_workspace_layout(
        batch, seq_len, intermediate_size, workspace_base_address
    )

    # Setup: scale/stride registers
    generated_code += _load_large_int(
        w_actual_register, hidden_size * intermediate_size
    )
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += _load_large_int(w_actual_register, intermediate_size)
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base addresses for results
    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += _load_large_int(gate_result_register, gate_result_base)

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
    generated_code += (
        f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    )
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        )
        generated_code += _addi_large_int(
            a_actual_register,
            a_actual_register,
            mlen * intermediate_size,
            w_temp_register,
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    )

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Reset activation base for each middle loop iteration
    generated_code += _load_large_int(a_actual_register, activation_base_address)
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual before accumulation loop (gate_result_register as temp)
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += (
            f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        )
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    act_col_advance = mlen * blen
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{gate_result_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    )
    # Reset intermediate back for next tile row
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    )
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    )
    # Advance up_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # Gate projection (loop)
    generated_code += "; FFN Gate Projection Generation (Loop)\n"

    # Reset base addresses
    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += _load_large_int(gate_result_register, gate_result_base)
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"

    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    )
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{gate_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        )
        generated_code += _addi_large_int(
            a_actual_register,
            a_actual_register,
            mlen * intermediate_size,
            w_temp_register,
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    )

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Reset activation base for each middle loop iteration
    generated_code += _load_large_int(a_actual_register, activation_base_address)
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual before accumulation loop (up_result_register as temp)
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += (
            f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        )
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    generated_code += (
        f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {act_col_advance}\n"
    )

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    )
    # Reset intermediate back for next tile row
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    )
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    )
    # Advance gate_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # SiLU activation (loop)
    generated_code += "; SILU Generation (Loop)\n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"

    # Reset addresses
    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += _load_large_int(gate_result_register, gate_result_base)
    generated_code += _load_large_int(intermediate_register, activation_base_address)

    # Loop over batch * seq_len * (intermediate_size // vlen)
    num_silu_iters = batch * seq_len * (intermediate_size // vlen)
    generated_code += f"; SILU loop: {num_silu_iters} iterations\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_silu_iters}\n"

    # SILU computation: sigmoid(x) * x * gate
    generated_code += (
        f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1\n"
    )
    generated_code += (
        f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    )
    generated_code += (
        f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0\n"
    )
    generated_code += (
        f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    )
    generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0\n"
    generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen}\n"
    )
    generated_code += (
        f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen}\n"
    )

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # Note: Result is stored in up_result_register at up_result_base
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
    workspace_base_address: int = 0,
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
    up_result_base, gate_result_base, _ = _ffn_workspace_layout(
        batch, seq_len, intermediate_size, workspace_base_address
    )

    # Setup: scale/stride registers
    generated_code += _load_large_int(
        w_actual_register, hidden_size * intermediate_size
    )
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += _load_large_int(w_actual_register, intermediate_size)
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base addresses for results
    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += _load_large_int(gate_result_register, gate_result_base)

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
    generated_code += (
        f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    )
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        )
        generated_code += _addi_large_int(
            a_actual_register,
            a_actual_register,
            mlen * intermediate_size,
            w_temp_register,
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    )

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Reset activation base for each middle loop iteration
    generated_code += _load_large_int(a_actual_register, activation_base_address)
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual before accumulation loop (gate_result_register as temp)
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += (
            f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        )
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    act_col_advance = mlen * blen
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{gate_result_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    )
    # Reset intermediate back for next tile row
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    )
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    )
    # Advance up_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # Gate projection (loop)
    generated_code += "; FFN Gate Projection Generation (Loop)\n"

    # Reset base addresses
    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += _load_large_int(gate_result_register, gate_result_base)
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"

    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    )
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{gate_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        )
        generated_code += _addi_large_int(
            a_actual_register,
            a_actual_register,
            mlen * intermediate_size,
            w_temp_register,
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    )

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Reset activation base for each middle loop iteration
    generated_code += _load_large_int(a_actual_register, activation_base_address)
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual before accumulation loop (up_result_register as temp)
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += (
            f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        )
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    generated_code += (
        f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {act_col_advance}\n"
    )

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    )
    # Reset intermediate back for next tile row
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    )
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    )
    # Advance gate_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # SiLU activation (loop)
    generated_code += "; SILU Generation (Loop)\n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"

    # Reset addresses
    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += _load_large_int(gate_result_register, gate_result_base)
    generated_code += _load_large_int(intermediate_register, activation_base_address)

    # Loop over batch * seq_len * (intermediate_size // vlen)
    num_silu_iters = batch * seq_len * (intermediate_size // vlen)
    generated_code += f"; SILU loop: {num_silu_iters} iterations\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_silu_iters}\n"

    # SILU computation: sigmoid(x) * x * gate
    generated_code += (
        f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1\n"
    )
    generated_code += (
        f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    )
    generated_code += (
        f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0\n"
    )
    generated_code += (
        f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    )
    generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0\n"
    generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen}\n"
    )
    generated_code += (
        f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen}\n"
    )

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # Downsize linear (loop)
    generated_code += "; FFN Downsize Linear Generation (Loop)\n"

    # Setup scale and stride for downsize
    generated_code += _load_large_int(
        w_actual_register, hidden_size * intermediate_size
    )
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += _load_large_int(w_actual_register, hidden_size)
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Result goes to activation base region
    act_result_register = gate_result_register
    generated_code += _load_large_int(act_result_register, activation_base_address)
    generated_code += _load_large_int(up_result_register, up_result_base)

    # Downsize: (b*s, intermediate_size) @ (intermediate_size, hidden_size) -> (b*s, hidden_size)
    num_down_mlen_blocks = hidden_size // mlen
    num_down_weight_tiles = intermediate_size // mlen
    down_act_col_advance = mlen * blen

    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_down_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_down_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    )
    for weight_col in range(num_down_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{down_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        )
        generated_code += _addi_large_int(
            a_actual_register,
            a_actual_register,
            mlen * hidden_size,
            w_temp_register,
        )

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
    )

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Reset activation base; up_result_register recomputed here (used as temp below)
    generated_code += _load_large_int(up_result_register, up_result_base)
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
        generated_code += (
            f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        )
        if inner_loop_index < num_down_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{down_act_save_reg}, {down_act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    )
    # Reset intermediate back for next tile row
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
    )
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    )
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
    workspace_base_address: int = 0,
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
    up_result_base, gate_result_base, _ = _ffn_workspace_layout(
        batch, seq_len, intermediate_size, workspace_base_address
    )

    # Setup: scale/stride registers
    generated_code += _load_large_int(
        w_actual_register, hidden_size * intermediate_size
    )
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += _load_large_int(w_actual_register, intermediate_size)
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base addresses for results
    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += _load_large_int(gate_result_register, gate_result_base)

    # Fused up + gate linear with overlapped prefetch
    generated_code += "; Fused Up+Gate Linear (overlapped prefetch optimization)\n"

    num_mlen_blocks = intermediate_size // mlen
    tiles_per_mlen = mlen // blen
    num_weight_tiles = hidden_size // mlen
    num_act_cols = (batch * seq_len) // blen
    act_col_advance = mlen * blen
    gate_mram_offset = (
        num_weight_tiles * mlen * mlen
    )  # Gate weights start after up weights in MRAM

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
    generated_code += (
        f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    )

    # Prefetch up weights (to MRAM at offset 0)
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        )
        generated_code += _addi_large_int(
            a_actual_register,
            a_actual_register,
            mlen * intermediate_size,
            w_temp_register,
        )

    # Setup for UP compute and GATE prefetch overlap
    generated_code += _load_large_int(w_gate_base_register, gate_mram_offset)

    # Reset for UP compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    )

    # Up projection with interleaved gate prefetch
    generated_code += f"; Up projection for MLEN block (with GATE prefetch every {gate_prefetch_interval} iters)\n"

    # Unroll to interleave GATE prefetches during UP computation
    # NOTE: We compute GATE HBM offset directly from w_hbm_offset_register + offset
    # instead of tracking it in a_save_register (which is reused for weight offset in inner loop)
    gate_prefetch_count = 0
    gate_mram_ptr = gate_mram_offset

    for tile_idx in range(tiles_per_mlen):
        # Reset activation base for this tile
        generated_code += _load_large_int(a_actual_register, activation_base_address)

        for act_col in range(num_act_cols):
            iter_num = tile_idx * num_act_cols + act_col

            # Check if we should insert a GATE prefetch
            if (
                iter_num % gate_prefetch_interval == 0
                and gate_prefetch_count < num_weight_tiles
            ):
                generated_code += f"; Prefetch GATE weight tile {gate_prefetch_count} during UP compute\n"
                # Save current a_actual_register (activation pointer) to w_temp_register
                generated_code += (
                    f"S_ADDI_INT gp{w_temp_register}, gp{a_actual_register}, 0\n"
                )
                # Compute GATE HBM offset directly: base_offset + prefetch_count * stride
                gate_hbm_offset = gate_prefetch_count * mlen * intermediate_size
                # Set MRAM destination
                generated_code += _load_large_int(a_actual_register, gate_mram_ptr)
                # Set HBM source: w_hbm_offset_register + gate_hbm_offset
                generated_code += f"S_ADDI_INT gp{a_save_register}, gp{w_hbm_offset_register}, {gate_hbm_offset}\n"
                generated_code += f"H_PREFETCH_M gp{a_actual_register}, gp{a_save_register}, a{gate_weight_hbm_offset_reg}, 1, 0\n"
                gate_mram_ptr += mlen * mlen
                gate_prefetch_count += 1
                # Restore activation pointer
                generated_code += (
                    f"S_ADDI_INT gp{a_actual_register}, gp{w_temp_register}, 0\n"
                )

            # Save activation column base before weight tile loop modifies a_actual_register
            generated_code += (
                f"S_ADDI_INT gp{w_temp_register}, gp{a_actual_register}, 0\n"
            )

            # UP weight accumulation
            generated_code += f"S_ADDI_INT gp{a_save_register}, gp{w_actual_register}, 0\n"  # save weight offset

            for inner_idx in range(num_weight_tiles):
                generated_code += (
                    f"M_MM 0, gp{a_save_register}, gp{a_actual_register}\n"
                )
                generated_code += f"S_ADDI_INT gp{a_save_register}, gp{a_save_register}, {mlen * mlen}\n"
                if inner_idx < num_weight_tiles - 1:
                    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

            generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"

            # Restore activation and advance to next column
            if act_col < num_act_cols - 1:
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_temp_register}, {act_col_advance}\n"

        # After all act_cols for this tile, advance weight offset
        generated_code += (
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
        )
        generated_code += (
            f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
        )
        generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    # Gate projection (weights already prefetched)
    generated_code += (
        "; Gate projection for MLEN block (weights pre-fetched during UP)\n"
    )
    generated_code += (
        f"S_ADDI_INT gp{a_save_register}, gp0, 0\n"  # tile offset tracker for output
    )
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_gate_base_register}, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    )

    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    generated_code += _load_large_int(a_actual_register, activation_base_address)
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save activation pointer - use w_gate_base_register
    generated_code += f"S_ADDI_INT gp{w_gate_base_register}, gp{a_actual_register}, 0\n"

    for inner_idx in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += (
            f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        )
        if inner_idx < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore activation from saved base + advance to next column
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_gate_base_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    generated_code += (
        f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    )
    generated_code += f"S_ADDI_INT gp{a_save_register}, gp{a_save_register}, {blen}\n"
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    )
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{a_save_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # Advance for next MLEN block
    generated_code += (
        f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    )
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
    generated_code += _load_large_int(
        w_actual_register, hidden_size * intermediate_size
    )
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += _load_large_int(w_actual_register, hidden_size)
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"

    # Initialize SILU pointers
    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += _load_large_int(gate_result_register, gate_result_base)
    generated_code += _load_large_int(intermediate_register, activation_base_address)

    # Initialize DOWN prefetch pointers (w_actual_register=MRAM offset, a_actual_register=HBM offset)
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, 0\n"

    # Compute how many SILU iters per prefetch op to spread prefetches across SILU loop
    # We have num_down_weight_tiles prefetches to do for the first block
    # Spread them evenly across the SILU loop
    prefetch_interval = max(1, num_silu_iters // num_down_weight_tiles)

    generated_code += f"; SILU loop: {num_silu_iters} iterations (prefetch every {prefetch_interval} iters)\n"
    generated_code += (
        f"; Prefetching {num_down_weight_tiles} DOWN weight tiles during SILU\n"
    )

    # Unroll SILU loop to interleave prefetch operations
    for silu_iter in range(num_silu_iters):
        # SILU computation
        generated_code += (
            f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1\n"
        )
        generated_code += (
            f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
        )
        generated_code += (
            f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0\n"
        )
        generated_code += (
            f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
        )
        generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0\n"
        generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen}\n"
        )
        generated_code += (
            f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen}\n"
        )

        # Insert prefetch at appropriate intervals
        prefetch_idx = silu_iter // prefetch_interval
        if silu_iter % prefetch_interval == 0 and prefetch_idx < num_down_weight_tiles:
            generated_code += f"; Prefetch DOWN weight tile {prefetch_idx}\n"
            generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{down_weight_hbm_offset_reg}, 1, 0\n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
            generated_code += _addi_large_int(
                a_actual_register,
                a_actual_register,
                mlen * hidden_size,
                w_temp_register,
            )

    # Downsize linear (first block already prefetched)
    generated_code += (
        "; FFN Downsize Linear Generation (first block pre-fetched during SILU)\n"
    )

    act_result_register = gate_result_register
    generated_code += _load_large_int(act_result_register, activation_base_address)
    generated_code += _load_large_int(up_result_register, up_result_base)

    down_act_col_advance = mlen * blen

    # First block: weights already prefetched, just do computation
    generated_code += "; First DOWN block (weights pre-fetched during SILU)\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += (
        f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, {mlen}\n"  # Next block HBM offset
    )

    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
    )
    tiles_per_mlen_down = mlen // blen

    # First block computation
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen_down}\n"

    generated_code += _load_large_int(up_result_register, up_result_base)
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, 0\n"
    num_down_act_cols = (batch * seq_len) // blen

    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_down_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

    for inner_idx in range(num_down_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += (
            f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        )
        if inner_idx < num_down_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {down_act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    generated_code += (
        f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    )
    generated_code += (
        f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
    )
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # Advance to second block base
    generated_code += f"S_ADDI_INT gp{act_result_register}, gp{act_result_register}, {mlen * batch * seq_len}\n"

    # Remaining blocks (if any) - standard prefetch then compute
    if num_down_mlen_blocks > 1:
        generated_code += f"; Remaining {num_down_mlen_blocks - 1} DOWN blocks\n"
        generated_code += (
            f"C_LOOP_START gp{loop_outer_reg}, {num_down_mlen_blocks - 1}\n"
        )

        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
        )
        for weight_col in range(num_down_weight_tiles):
            generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{down_weight_hbm_offset_reg}, 1, 0\n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
            generated_code += _addi_large_int(
                a_actual_register,
                a_actual_register,
                mlen * hidden_size,
                w_temp_register,
            )

        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
        )

        generated_code += f"; Middle loop: {tiles_per_mlen_down} tiles per MLEN block\n"
        generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen_down}\n"

        generated_code += _load_large_int(up_result_register, up_result_base)
        generated_code += (
            f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, 0\n"
        )

        generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_down_act_cols}\n"

        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
        generated_code += (
            f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"
        )

        for inner_idx in range(num_down_weight_tiles):
            generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
            generated_code += (
                f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
            )
            if inner_idx < num_down_weight_tiles - 1:
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

        generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
        generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {down_act_col_advance}\n"

        generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

        generated_code += (
            f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
        )
        generated_code += (
            f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
        )
        generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

        generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

        generated_code += (
            f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
        )
        generated_code += f"S_ADDI_INT gp{act_result_register}, gp{act_result_register}, {mlen * batch * seq_len}\n"

        generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    return generated_code
