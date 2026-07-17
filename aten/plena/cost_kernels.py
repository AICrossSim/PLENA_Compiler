"""Algebraic cost lowering for legacy templates with large Python unrolls."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from compiler.asm_templates._k_split import k_chunks
from compiler.asm_templates._imm import IMM2_BOUND, add_large_int, load_large_int
from compiler.aten.cost_emitter import (
    ScheduleAffineAdd,
    ScheduleAffineLoad,
    ScheduleInstruction,
    ScheduleRepeat,
    ScheduleSequence,
)
from compiler.aten.isa_builder import DmaTransfer, RepeatAxis


@dataclass
class KernelDmaStream:
    transfer: DmaTransfer
    multiplicity: int
    axes: tuple[RepeatAxis, ...] = ()


@dataclass
class KernelCostSummary:
    static: Counter[str] = field(default_factory=Counter)
    dynamic: Counter[str] = field(default_factory=Counter)
    memory_streams: list[KernelDmaStream] = field(default_factory=list)

    def add(self, opcode: str, static: int = 1, dynamic: int | None = None) -> None:
        if dynamic is None:
            dynamic = static
        self.static[opcode] += static
        self.dynamic[opcode] += dynamic

    def merge(self, other: KernelCostSummary, multiplier: int = 1) -> None:
        for opcode, count in other.static.items():
            self.static[opcode] += count * multiplier
        for opcode, count in other.dynamic.items():
            self.dynamic[opcode] += count * multiplier
        if multiplier != 1 and other.memory_streams:
            raise ValueError("merging DMA streams with a scalar multiplier requires explicit axes")
        self.memory_streams.extend(other.memory_streams)

    def add_dma(
        self,
        transfer: DmaTransfer,
        *,
        multiplicity: int = 1,
        axes: tuple[RepeatAxis, ...] = (),
    ) -> None:
        self.memory_streams.append(KernelDmaStream(transfer, multiplicity, axes))


KernelCounts = KernelCostSummary


def _schedule_instruction(opcode: str, *args, memory_stream_index=None):
    return ScheduleInstruction(
        opcode,
        tuple(str(arg) for arg in args),
        memory_stream_index=memory_stream_index,
    )


def _load_large_schedule(register: int, value: int) -> tuple[ScheduleInstruction, ...]:
    instructions = []
    for line in load_large_int(register, value):
        opcode, raw_args = line.split(maxsplit=1)
        instructions.append(
            _schedule_instruction(
                opcode, *(part.strip() for part in raw_args.split(","))
            )
        )
    return tuple(instructions)


def _add_large_schedule(
    destination: int, source: int, value: int
) -> tuple[ScheduleInstruction, ...]:
    instructions = []
    for line in add_large_int(destination, source, value):
        opcode, raw_args = line.split(maxsplit=1)
        instructions.append(
            _schedule_instruction(
                opcode, *(part.strip() for part in raw_args.split(","))
            )
        )
    return tuple(instructions)


def _add_with_temp_schedule(
    destination: int, source: int, value: int, temp: int
) -> tuple[ScheduleInstruction, ...]:
    instructions = []
    for line in add_large_int(destination, source, value, temp_reg=temp):
        opcode, raw_args = line.split(maxsplit=1)
        instructions.append(
            _schedule_instruction(
                opcode, *(part.strip() for part in raw_args.split(","))
            )
        )
    return tuple(instructions)


def _load_large(result: KernelCounts, value: int, *, static: int = 1, dynamic: int | None = None) -> None:
    if dynamic is None:
        dynamic = static
    if value < IMM2_BOUND:
        result.add("S_ADDI_INT", static, dynamic)
        return
    result.add("S_LUI_INT", static, dynamic)
    if value & 0xFFF:
        result.add("S_ADDI_INT", static, dynamic)


def _load_large_sequence(
    result: KernelCounts,
    *,
    start: int,
    step: int,
    count: int,
    multiplier: int = 1,
    dynamic_multiplier: int | None = None,
) -> None:
    """Count load_large_int over an arithmetic address sequence in O(1)."""
    if count <= 0:
        return
    if dynamic_multiplier is None:
        dynamic_multiplier = multiplier
    if step < 0:
        raise ValueError(f"negative address step is unsupported: {step}")
    if step == 0:
        _load_large(result, start, static=count * multiplier, dynamic=count * dynamic_multiplier)
        return
    below = max(0, min(count, (IMM2_BOUND - start + step - 1) // step)) if start < IMM2_BOUND else 0
    high = count - below
    zero_low = 0
    if high:
        modulus = 1 << 12
        divisor = math.gcd(step, modulus)
        rhs = -start
        if rhs % divisor == 0:
            reduced_modulus = modulus // divisor
            first = ((rhs // divisor) * pow(step // divisor, -1, reduced_modulus)) % reduced_modulus
            first_high = below
            if first < first_high:
                first += ((first_high - first + reduced_modulus - 1) // reduced_modulus) * reduced_modulus
            if first < count:
                zero_low = 1 + (count - 1 - first) // reduced_modulus
    result.add("S_LUI_INT", static=high * multiplier, dynamic=high * dynamic_multiplier)
    addi_base = below + high - zero_low
    if addi_base:
        result.add(
            "S_ADDI_INT",
            static=addi_base * multiplier,
            dynamic=addi_base * dynamic_multiplier,
        )


def _direct_addi(
    result: KernelCounts,
    value: int,
    *,
    source_is_zero: bool,
    static: int = 1,
    dynamic: int | None = None,
) -> None:
    """Count one raw S_ADDI_INT after compiler-wide legalization."""
    if dynamic is None:
        dynamic = static
    if value < IMM2_BOUND:
        result.add("S_ADDI_INT", static, dynamic)
    elif source_is_zero:
        _load_large(result, value, static=static, dynamic=dynamic)
    else:
        chunks = (value + IMM2_BOUND - 2) // (IMM2_BOUND - 1)
        result.add("S_ADDI_INT", static * chunks, dynamic * chunks)


def _addi_with_temp(
    result: KernelCounts,
    value: int,
    *,
    static: int = 1,
    dynamic: int | None = None,
) -> None:
    """Count asm_templates._imm.addi_large_int with a non-aliasing temp."""
    if dynamic is None:
        dynamic = static
    if value < IMM2_BOUND:
        result.add("S_ADDI_INT", static, dynamic)
        return
    _load_large(result, value, static=static, dynamic=dynamic)
    result.add("S_ADD_INT", static, dynamic)


def _addi_with_temp_sequence(
    result: KernelCounts,
    *,
    start: int,
    step: int,
    count: int,
    multiplier: int = 1,
) -> None:
    """Count addi_large_int(temp=non-aliasing) over an arithmetic sequence."""
    if count <= 0:
        return
    below = max(0, min(count, (IMM2_BOUND - start + step - 1) // step)) if start < IMM2_BOUND else 0
    high = count - below
    if below:
        result.add("S_ADDI_INT", below * multiplier)
    if not high:
        return
    modulus = 1 << 12
    divisor = math.gcd(step, modulus)
    zero_low = 0
    rhs = -start
    if rhs % divisor == 0:
        reduced_modulus = modulus // divisor
        first = ((rhs // divisor) * pow(step // divisor, -1, reduced_modulus)) % reduced_modulus
        if first < below:
            first += ((below - first + reduced_modulus - 1) // reduced_modulus) * reduced_modulus
        if first < count:
            zero_low = 1 + (count - 1 - first) // reduced_modulus
    result.add("S_LUI_INT", high * multiplier)
    result.add("S_ADD_INT", high * multiplier)
    if high != zero_low:
        result.add("S_ADDI_INT", (high - zero_low) * multiplier)


def _ffn_projection_chunk_counts(
    *,
    mlen: int,
    blen: int,
    batch_rows: int,
    k_size: int,
    out_size: int,
    weight_stride: int,
    result_base_value: int,
    activation_base_address: int | None,
    activation_base_register_value: int | None,
    k_start_tile: int,
    k_tile_count: int,
    target_base_value: int,
    weight_hbm_base: int,
    hbm_prefetch_amount: int,
    source: str,
) -> KernelCounts:
    result = KernelCounts()
    num_act_cols = batch_rows // blen
    chunk_hbm_base_offset = k_start_tile * mlen * weight_stride
    chunk_act_base_offset = k_start_tile * mlen * batch_rows
    tiles_per_mlen = mlen // blen
    weight_rows = out_size // blen
    block_rows = weight_rows // tiles_per_mlen

    result.add_dma(
        DmaTransfer(
            opcode="H_PREFETCH_M",
            direction="read",
            precision="weight",
            element_base=weight_hbm_base + chunk_hbm_base_offset,
            scale_base=(
                weight_hbm_base
                + k_size * out_size
                + chunk_hbm_base_offset // 8
            ),
            dim=mlen,
            amount=hbm_prefetch_amount,
            stride=weight_stride,
            rstride=1,
            write_amount=mlen,
            geometry_fidelity="exact",
            source=source,
            memory_object=f"hbm:{weight_hbm_base}:{k_size * out_size}",
            logical_object_elements=k_size * out_size,
            precision_role="weight",
            logical_element_offset=chunk_hbm_base_offset,
            logical_scale_offset=chunk_hbm_base_offset // 8,
            logical_stride=weight_stride,
        ),
        multiplicity=block_rows * k_tile_count,
        axes=(
            RepeatAxis(
                "output_mlen_block",
                block_rows,
                element_base_delta=mlen,
                scale_base_delta=mlen // 8,
            ),
            RepeatAxis(
                "k_tile",
                k_tile_count,
                element_base_delta=mlen * weight_stride,
                scale_base_delta=mlen * weight_stride // 8,
            ),
        ),
    )

    _load_large(result, target_base_value)
    if activation_base_address is None:
        if activation_base_register_value is None:
            raise ValueError("activation_base_register_value is required for register-based FFN input")
        _load_large(result, activation_base_register_value)

    result.add("S_ADDI_INT", block_rows)  # MRAM pointer reset
    _addi_with_temp_sequence(
        result,
        start=chunk_hbm_base_offset,
        step=mlen,
        count=block_rows,
    )
    result.add("S_ADDI_INT", block_rows)  # intermediate pointer
    result.add("H_PREFETCH_M", block_rows * k_tile_count)
    _direct_addi(
        result,
        mlen * mlen,
        source_is_zero=False,
        static=block_rows * k_tile_count,
    )
    _addi_with_temp(
        result,
        mlen * weight_stride,
        static=block_rows * k_tile_count,
    )
    result.add("S_ADDI_INT", block_rows)
    result.add("S_ADDI_INT", 2 * (weight_rows - block_rows))

    pointer_start = chunk_act_base_offset
    if activation_base_address is not None:
        pointer_start += activation_base_address
    _addi_with_temp_sequence(
        result,
        start=pointer_start,
        step=mlen * blen,
        count=num_act_cols,
        multiplier=weight_rows,
    )

    cells = weight_rows * num_act_cols
    result.add("S_ADDI_INT", cells)
    if k_tile_count > 1:
        result.add("C_LOOP_START", cells)
    result.add("M_MM", static=cells, dynamic=cells * k_tile_count)
    _direct_addi(
        result,
        mlen * mlen,
        source_is_zero=False,
        static=cells,
        dynamic=cells * k_tile_count,
    )
    _direct_addi(
        result,
        mlen * batch_rows,
        source_is_zero=False,
        static=cells,
        dynamic=cells * k_tile_count,
    )
    if k_tile_count > 1:
        result.add("C_LOOP_END", static=cells, dynamic=cells * k_tile_count)
    result.add("M_MM_WO", cells)
    _direct_addi(result, blen * mlen, source_is_zero=False, static=cells)
    if block_rows > 1:
        _direct_addi(
            result,
            mlen * batch_rows,
            source_is_zero=False,
            static=block_rows - 1,
        )
    return result


def _ffn_projection_counts(
    *,
    mlen: int,
    vlen: int,
    blen: int,
    batch_rows: int,
    k_size: int,
    out_size: int,
    weight_stride: int,
    result_base_value: int,
    activation_base_address: int | None,
    activation_base_register_value: int | None,
    max_k_tiles: int,
    scratch_base_value: int,
    weight_hbm_base: int,
    hbm_prefetch_amount: int,
    source: str,
) -> KernelCounts:
    result = KernelCounts()
    num_k_tiles = k_size // mlen
    chunks = k_chunks(num_k_tiles, max_k_tiles)
    for chunk_idx, (k_start, k_count) in enumerate(chunks):
        target_base = result_base_value if chunk_idx == 0 else scratch_base_value
        result.merge(
            _ffn_projection_chunk_counts(
                mlen=mlen,
                blen=blen,
                batch_rows=batch_rows,
                k_size=k_size,
                out_size=out_size,
                weight_stride=weight_stride,
                result_base_value=result_base_value,
                activation_base_address=activation_base_address,
                activation_base_register_value=activation_base_register_value,
                k_start_tile=k_start,
                k_tile_count=k_count,
                target_base_value=target_base,
                weight_hbm_base=weight_hbm_base,
                hbm_prefetch_amount=hbm_prefetch_amount,
                source=source,
            )
        )
        if chunk_idx:
            _load_large(result, result_base_value)
            _load_large(result, scratch_base_value)
            vector_adds = math.ceil(out_size * batch_rows / vlen)
            result.add("V_ADD_VV", vector_adds)
            result.add("S_ADDI_INT", 2 * vector_adds)
    return result


def ffn_unrolled_cost_counts(
    *,
    mlen: int,
    vlen: int,
    blen: int,
    batch_rows: int,
    hidden_size: int,
    intermediate_size: int,
    activation_base_address: int,
    workspace_base_address: int,
    matrix_sram_size: int,
    gate_weight_hbm_base: int,
    up_weight_hbm_base: int,
    down_weight_hbm_base: int,
    hbm_prefetch_amount: int,
) -> KernelCounts:
    """Exact opcode counts for asm_templates.ffn_asm._ffn_asm_unrolled."""
    if batch_rows % blen:
        raise ValueError(f"batch_rows={batch_rows} must be divisible by BLEN={blen}")
    up_result_base = workspace_base_address
    gate_result_base = up_result_base + batch_rows * intermediate_size
    scratch_base = gate_result_base + batch_rows * intermediate_size
    max_k_tiles = max(1, matrix_sram_size // mlen)
    result = KernelCounts()

    _load_large(result, hidden_size * intermediate_size)
    result.add("C_SET_SCALE_REG")
    _load_large(result, intermediate_size)
    result.add("C_SET_STRIDE_REG")
    result.add("S_ADDI_INT")
    _load_large(result, up_result_base)
    _load_large(result, gate_result_base)

    for result_base, weight_hbm_base, source in (
        (up_result_base, up_weight_hbm_base, "ffn:up_weight"),
        (gate_result_base, gate_weight_hbm_base, "ffn:gate_weight"),
    ):
        result.merge(
            _ffn_projection_counts(
                mlen=mlen,
                vlen=vlen,
                blen=blen,
                batch_rows=batch_rows,
                k_size=hidden_size,
                out_size=intermediate_size,
                weight_stride=intermediate_size,
                result_base_value=result_base,
                activation_base_address=activation_base_address,
                activation_base_register_value=None,
                max_k_tiles=max_k_tiles,
                scratch_base_value=scratch_base,
                weight_hbm_base=weight_hbm_base,
                hbm_prefetch_amount=hbm_prefetch_amount,
                source=source,
            )
        )

    result.add("S_LD_FP")
    _load_large(result, up_result_base)
    _load_large(result, gate_result_base)
    _load_large(result, activation_base_address)
    silu_iters = batch_rows * (intermediate_size // vlen)
    for opcode in ("V_SUB_VF", "V_EXP_V", "V_ADD_VF", "V_RECI_V"):
        result.add(opcode, silu_iters)
    result.add("V_MUL_VV", 2 * silu_iters)
    result.add("S_ADDI_INT", 2 * silu_iters)

    _load_large(result, hidden_size * intermediate_size)
    result.add("C_SET_SCALE_REG")
    _load_large(result, hidden_size)
    result.add("C_SET_STRIDE_REG")
    result.add("S_ADDI_INT")
    _direct_addi(result, batch_rows // blen, source_is_zero=True)
    result.merge(
        _ffn_projection_counts(
            mlen=mlen,
            vlen=vlen,
            blen=blen,
            batch_rows=batch_rows,
            k_size=intermediate_size,
            out_size=hidden_size,
            weight_stride=hidden_size,
            result_base_value=activation_base_address,
            activation_base_address=None,
            activation_base_register_value=up_result_base,
            max_k_tiles=max_k_tiles,
            scratch_base_value=scratch_base,
            weight_hbm_base=down_weight_hbm_base,
            hbm_prefetch_amount=hbm_prefetch_amount,
            source="ffn:down_weight",
        )
    )
    return result


def _ffn_projection_chunk_schedule(
    *,
    mlen: int,
    blen: int,
    batch_rows: int,
    k_size: int,
    out_size: int,
    weight_stride: int,
    result_base_register: int,
    result_base_value: int,
    activation_base_address: int | None,
    activation_base_register: int | None,
    activation_base_register_value: int | None,
    k_start_tile: int,
    k_tile_count: int,
    target_base_value: int,
    weight_addr_reg: int,
    memory_stream_index: int,
    key_prefix: str,
) -> ScheduleSequence:
    """Compressed exact order for one unrolled FFN projection K chunk."""
    w_actual = 1
    w_temp = 2
    a_actual = 3
    intermediate = 5
    w_hbm_offset = 7
    tiles_per_mlen = mlen // blen
    block_rows = out_size // mlen
    num_act_cols = batch_rows // blen
    chunk_hbm_offset = k_start_tile * mlen * weight_stride
    chunk_act_offset = k_start_tile * mlen * batch_rows

    children = [*_load_large_schedule(result_base_register, target_base_value)]
    if activation_base_address is None:
        if activation_base_register is None or activation_base_register_value is None:
            raise ValueError(
                "register-based FFN input requires register and canonical value"
            )
        children.extend(
            _load_large_schedule(
                activation_base_register, activation_base_register_value
            )
        )

    weight_offset = ScheduleAffineAdd(
        key=f"{key_prefix}:weight_block_offset",
        destination=f"gp{w_hbm_offset}",
        source="gp0",
        temp=f"gp{w_temp}",
        start=chunk_hbm_offset,
        step=mlen,
        period=block_rows,
    )
    weight_tile = ScheduleAffineLoad(
        key=f"{key_prefix}:weight_tile",
        register=f"gp{w_actual}",
        start=blen,
        step=blen,
        period=max(1, tiles_per_mlen - 1),
    )
    output_tile = ScheduleAffineAdd(
        key=f"{key_prefix}:output_tile",
        destination=f"gp{intermediate}",
        source=f"gp{result_base_register}",
        temp=f"gp{w_temp}",
        start=blen,
        step=blen,
        period=max(1, tiles_per_mlen - 1),
    )
    activation_pointer = ScheduleAffineAdd(
        key=f"{key_prefix}:activation_column",
        destination=f"gp{a_actual}",
        source=(
            "gp0"
            if activation_base_address is not None
            else f"gp{activation_base_register}"
        ),
        temp=f"gp{w_temp}",
        start=(
            int(activation_base_address) + chunk_act_offset
            if activation_base_address is not None
            else chunk_act_offset
        ),
        step=mlen * blen,
        period=num_act_cols,
    )

    prefetch_body = ScheduleSequence(
        (
            _schedule_instruction(
                "H_PREFETCH_M",
                f"gp{w_actual}",
                f"gp{w_hbm_offset}",
                f"a{weight_addr_reg}",
                1,
                0,
                memory_stream_index=memory_stream_index,
            ),
            *_add_large_schedule(w_actual, w_actual, mlen * mlen),
            *_add_with_temp_schedule(
                w_hbm_offset,
                w_hbm_offset,
                mlen * weight_stride,
                w_temp,
            ),
        )
    )
    matrix_body = [
        _schedule_instruction("M_MM", 0, f"gp{w_temp}", f"gp{a_actual}"),
        *_add_large_schedule(w_temp, w_temp, mlen * mlen),
        *_add_large_schedule(a_actual, a_actual, mlen * batch_rows),
    ]
    if k_tile_count > 1:
        matrix_body.append(
            _schedule_instruction("C_LOOP_END", f"gp{w_hbm_offset}")
        )
        matrix_schedule = (
            _schedule_instruction(
                "C_LOOP_START", f"gp{w_hbm_offset}", k_tile_count
            ),
            ScheduleRepeat(
                k_tile_count,
                ScheduleSequence(tuple(matrix_body)),
                name="ffn_projection_k_compute",
                repeat_kind="hardware_loop",
            ),
        )
    else:
        matrix_schedule = tuple(matrix_body)

    act_column_body = ScheduleSequence(
        (
            activation_pointer,
            _schedule_instruction(
                "S_ADDI_INT", f"gp{w_temp}", f"gp{w_actual}", 0
            ),
            *matrix_schedule,
            _schedule_instruction(
                "M_MM_WO", f"gp{intermediate}", "gp0", 0
            ),
            *_add_large_schedule(
                intermediate, intermediate, blen * mlen
            ),
        )
    )
    later_tile_body = ScheduleSequence(
        (
            weight_tile,
            output_tile,
            ScheduleRepeat(
                num_act_cols,
                act_column_body,
                name="ffn_projection_activation_columns",
                repeat_kind="compile_time",
            ),
        )
    )

    def block_body(*, advance_result: bool) -> ScheduleSequence:
        body = [
            weight_offset,
            _schedule_instruction(
                "S_ADDI_INT", f"gp{intermediate}", f"gp{result_base_register}", 0
            ),
            ScheduleRepeat(
                k_tile_count,
                prefetch_body,
                name="ffn_projection_k_prefetch",
                repeat_kind="compile_time",
            ),
            _schedule_instruction(
                "S_ADDI_INT", f"gp{w_actual}", "gp0", 0
            ),
            ScheduleRepeat(
                num_act_cols,
                act_column_body,
                name="ffn_projection_first_tile_columns",
                repeat_kind="compile_time",
            ),
        ]
        if tiles_per_mlen > 1:
            body.append(
                ScheduleRepeat(
                    tiles_per_mlen - 1,
                    later_tile_body,
                    name="ffn_projection_remaining_output_tiles",
                    repeat_kind="compile_time",
                )
            )
        # The template resets MRAM before the block prefetch and again before
        # compute.  ``weight_tile`` supplies the second reset for tile zero.
        body.insert(
            0,
            _schedule_instruction(
                "S_ADDI_INT", f"gp{w_actual}", "gp0", 0
            ),
        )
        if advance_result:
            body.extend(
                _add_large_schedule(
                    result_base_register,
                    result_base_register,
                    mlen * batch_rows,
                )
            )
        return ScheduleSequence(tuple(body))

    if block_rows > 1:
        children.append(
            ScheduleRepeat(
                block_rows - 1,
                block_body(advance_result=True),
                name="ffn_projection_output_blocks",
                repeat_kind="compile_time",
            )
        )
    children.extend(block_body(advance_result=False).children)
    return ScheduleSequence(tuple(children))


def _ffn_projection_cost_schedule(
    *,
    mlen: int,
    vlen: int,
    blen: int,
    batch_rows: int,
    k_size: int,
    out_size: int,
    weight_stride: int,
    result_base_register: int,
    result_base_value: int,
    activation_base_address: int | None,
    activation_base_register: int | None,
    activation_base_register_value: int | None,
    max_k_tiles: int,
    scratch_base_value: int,
    weight_addr_reg: int,
    stream_index_start: int,
    key_prefix: str,
) -> tuple[ScheduleSequence, int]:
    children = []
    stream_index = stream_index_start
    for chunk_index, (k_start, k_count) in enumerate(
        k_chunks(k_size // mlen, max_k_tiles)
    ):
        target = result_base_value if chunk_index == 0 else scratch_base_value
        chunk_key = f"{key_prefix}:chunk{chunk_index}"
        children.extend(
            _ffn_projection_chunk_schedule(
                mlen=mlen,
                blen=blen,
                batch_rows=batch_rows,
                k_size=k_size,
                out_size=out_size,
                weight_stride=weight_stride,
                result_base_register=result_base_register,
                result_base_value=result_base_value,
                activation_base_address=activation_base_address,
                activation_base_register=activation_base_register,
                activation_base_register_value=activation_base_register_value,
                k_start_tile=k_start,
                k_tile_count=k_count,
                target_base_value=target,
                weight_addr_reg=weight_addr_reg,
                memory_stream_index=stream_index,
                key_prefix=chunk_key,
            ).children
        )
        stream_index += 1
        if chunk_index:
            vector_adds = math.ceil(out_size * batch_rows / vlen)
            children.extend(_load_large_schedule(1, result_base_value))
            children.extend(_load_large_schedule(2, scratch_base_value))
            children.append(
                ScheduleRepeat(
                    vector_adds,
                    ScheduleSequence(
                        (
                            _schedule_instruction(
                                "V_ADD_VV", "gp1", "gp1", "gp2", 0
                            ),
                            _schedule_instruction(
                                "S_ADDI_INT", "gp1", "gp1", vlen
                            ),
                            _schedule_instruction(
                                "S_ADDI_INT", "gp2", "gp2", vlen
                            ),
                        )
                    ),
                    name="ffn_k_split_accumulate",
                    repeat_kind="compile_time",
                )
            )
    return ScheduleSequence(tuple(children)), stream_index


def ffn_unrolled_cost_schedule(
    *,
    mlen: int,
    vlen: int,
    blen: int,
    batch_rows: int,
    hidden_size: int,
    intermediate_size: int,
    activation_base_address: int,
    workspace_base_address: int,
    matrix_sram_size: int,
) -> ScheduleSequence:
    """Ordered compressed counterpart of ``_ffn_asm_unrolled``."""
    up_base = workspace_base_address
    gate_base = up_base + batch_rows * intermediate_size
    scratch_base = gate_base + batch_rows * intermediate_size
    max_k_tiles = max(1, matrix_sram_size // mlen)
    children = [
        *_load_large_schedule(1, hidden_size * intermediate_size),
        _schedule_instruction("C_SET_SCALE_REG", "gp1"),
        *_load_large_schedule(1, intermediate_size),
        _schedule_instruction("C_SET_STRIDE_REG", "gp1"),
        _schedule_instruction("S_ADDI_INT", "gp1", "gp0", 0),
        *_load_large_schedule(4, up_base),
        *_load_large_schedule(6, gate_base),
    ]
    stream_index = 0
    for name, result_register, result_base, addr_reg in (
        ("up", 4, up_base, 2),
        ("gate", 6, gate_base, 1),
    ):
        schedule, stream_index = _ffn_projection_cost_schedule(
            mlen=mlen,
            vlen=vlen,
            blen=blen,
            batch_rows=batch_rows,
            k_size=hidden_size,
            out_size=intermediate_size,
            weight_stride=intermediate_size,
            result_base_register=result_register,
            result_base_value=result_base,
            activation_base_address=activation_base_address,
            activation_base_register=None,
            activation_base_register_value=None,
            max_k_tiles=max_k_tiles,
            scratch_base_value=scratch_base,
            weight_addr_reg=addr_reg,
            stream_index_start=stream_index,
            key_prefix=f"ffn:{workspace_base_address}:{name}",
        )
        children.extend(schedule.children)

    children.extend(
        (
            _schedule_instruction("S_LD_FP", "f1", "gp0", 5),
            *_load_large_schedule(4, up_base),
            *_load_large_schedule(6, gate_base),
            *_load_large_schedule(5, activation_base_address),
            ScheduleRepeat(
                batch_rows * (intermediate_size // vlen),
                ScheduleSequence(
                    (
                        _schedule_instruction(
                            "V_SUB_VF", "gp5", "gp4", "f0", 0, 1
                        ),
                        _schedule_instruction("V_EXP_V", "gp5", "gp5", 0),
                        _schedule_instruction(
                            "V_ADD_VF", "gp5", "gp5", "f1", 0
                        ),
                        _schedule_instruction("V_RECI_V", "gp5", "gp5", 0),
                        _schedule_instruction(
                            "V_MUL_VV", "gp5", "gp5", "gp4", 0
                        ),
                        _schedule_instruction(
                            "V_MUL_VV", "gp4", "gp5", "gp6", 0
                        ),
                        _schedule_instruction("S_ADDI_INT", "gp6", "gp6", vlen),
                        _schedule_instruction("S_ADDI_INT", "gp4", "gp4", vlen),
                    )
                ),
                name="ffn_silu_elements",
                repeat_kind="compile_time",
            ),
            *_load_large_schedule(1, hidden_size * intermediate_size),
            _schedule_instruction("C_SET_SCALE_REG", "gp1"),
            *_load_large_schedule(1, hidden_size),
            _schedule_instruction("C_SET_STRIDE_REG", "gp1"),
            _schedule_instruction("S_ADDI_INT", "gp1", "gp0", 0),
            *_load_large_schedule(8, batch_rows // blen),
        )
    )
    down, stream_index = _ffn_projection_cost_schedule(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch_rows=batch_rows,
        k_size=intermediate_size,
        out_size=hidden_size,
        weight_stride=hidden_size,
        result_base_register=6,
        result_base_value=activation_base_address,
        activation_base_address=None,
        activation_base_register=4,
        activation_base_register_value=up_base,
        max_k_tiles=max_k_tiles,
        scratch_base_value=scratch_base,
        weight_addr_reg=3,
        stream_index_start=stream_index,
        key_prefix=f"ffn:{workspace_base_address}:down",
    )
    children.extend(down.children)
    return ScheduleSequence(tuple(children))


def rms_norm_cost_counts(
    *,
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int,
    unroll: bool,
) -> KernelCounts:
    """Exact opcode counts for asm_templates.normalization_asm.rms_norm_asm."""
    if hidden_dim % vlen:
        raise ValueError(f"hidden_dim={hidden_dim} must be divisible by vlen={vlen}")
    result = KernelCounts()
    chunks = hidden_dim // vlen
    stride = vlen * batch_size

    _load_large(result, scratchpad_base_address)
    result.add("S_LD_FP", 2)
    result.add("S_ADD_FP")

    # Per-batch activation/stat pointer setup.
    _load_large_sequence(
        result,
        start=activation_base_address,
        step=vlen,
        count=batch_size,
        multiplier=2,
    )
    if unroll:
        result.add("V_MUL_VV", batch_size * chunks)
        result.add("V_RED_SUM", batch_size * chunks)
        addi_chunks = 1 if stride < IMM2_BOUND else (stride + IMM2_BOUND - 2) // (IMM2_BOUND - 1)
        result.add("S_ADDI_INT", batch_size * chunks * addi_chunks)
    else:
        result.add("C_LOOP_START", batch_size)
        result.add("V_MUL_VV", static=batch_size, dynamic=batch_size * chunks)
        result.add("V_RED_SUM", static=batch_size, dynamic=batch_size * chunks)
        addi_chunks = 1 if stride < IMM2_BOUND else (stride + IMM2_BOUND - 2) // (IMM2_BOUND - 1)
        result.add(
            "S_ADDI_INT",
            static=batch_size * addi_chunks,
            dynamic=batch_size * chunks * addi_chunks,
        )
        result.add("C_LOOP_END", static=batch_size, dynamic=batch_size * chunks)

    # Ping-pong addresses loaded for chunks 1..N-1 form one contiguous
    # activation-address sequence after flattening (chunk-major, then batch).
    if chunks > 1:
        _load_large_sequence(
            result,
            start=activation_base_address + stride,
            step=vlen,
            count=(chunks - 1) * batch_size,
        )
    result.add("S_MUL_FP", batch_size)
    result.add("S_ADD_FP", 2 * batch_size)  # epsilon + accumulator reset
    result.add("S_SQRT_FP", batch_size)
    result.add("S_RECI_FP", batch_size)
    result.add("S_ADDI_INT", 4 * batch_size)
    result.add("V_MUL_VF", batch_size * chunks)
    if chunks > 1:
        result.add("S_ADDI_INT", batch_size)  # final write-port spacer
    return result


def rms_norm_cost_schedule(
    *,
    activation_base_address: int,
    scratchpad_base_address: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int,
    unroll: bool,
    alive_registers: list[int],
    eps_offset: int,
    reci_hid_offset: int,
    key_prefix: str,
) -> ScheduleSequence:
    """Ordered compressed counterpart of ``rms_norm_asm``.

    Batch-specific pointer materialization remains explicit because large
    immediate legalization can differ by address. The hidden-vector loop is
    represented as a repeat, avoiding Python expansion of its dynamic body.
    """
    if hidden_dim % vlen:
        raise ValueError(f"hidden_dim={hidden_dim} must be divisible by vlen={vlen}")
    if len(alive_registers) < (3 if unroll else 4):
        raise ValueError("RMS schedule requires the same GP registers as rms_norm_asm")
    act_addr, scratchpad_addr, stats_addr = alive_registers[:3]
    loop_addr = None if unroll else alive_registers[3]
    chunks = hidden_dim // vlen
    stride = vlen * batch_size
    prefix = [*_load_large_schedule(scratchpad_addr, scratchpad_base_address)]
    prefix.extend(
        (
            _schedule_instruction("S_LD_FP", "f1", "gp0", eps_offset),
            _schedule_instruction("S_ADD_FP", "f2", "f0", "f0"),
            _schedule_instruction("S_LD_FP", "f3", "gp0", reci_hid_offset),
        )
    )

    def affine_load(key: str, register: int, offset: int) -> ScheduleAffineLoad:
        return ScheduleAffineLoad(
            key=f"{key_prefix}:{key}",
            register=f"gp{register}",
            start=activation_base_address + offset,
            step=vlen,
            period=batch_size,
        )

    batch_body = [
        affine_load("rms_act_base", act_addr, 0),
        affine_load("rms_stats_base", stats_addr, 0),
    ]
    reduction_body = ScheduleSequence(
        (
            _schedule_instruction(
                "V_MUL_VV",
                f"gp{scratchpad_addr}",
                f"gp{stats_addr}",
                f"gp{stats_addr}",
                0,
            ),
            _schedule_instruction("V_RED_SUM", "f2", f"gp{scratchpad_addr}"),
            *_add_large_schedule(stats_addr, stats_addr, stride),
        )
    )
    if unroll:
        batch_body.append(
            ScheduleRepeat(
                chunks,
                reduction_body,
                name="rms_reduce_chunks",
                repeat_kind="compile_time",
            )
        )
    else:
        assert loop_addr is not None
        batch_body.append(
            _schedule_instruction("C_LOOP_START", f"gp{loop_addr}", chunks)
        )
        batch_body.append(
            ScheduleRepeat(
                chunks,
                ScheduleSequence(
                    (
                        *reduction_body.children,
                        _schedule_instruction("C_LOOP_END", f"gp{loop_addr}"),
                    )
                ),
                name="rms_reduce_chunks",
                repeat_kind="hardware_loop",
            )
        )

    address_registers = (act_addr, stats_addr)
    if chunks > 1:
        batch_body.append(
            affine_load("rms_normalize_chunk_1", address_registers[1], stride)
        )
    batch_body.extend(
        (
            _schedule_instruction("S_MUL_FP", "f2", "f2", "f3"),
            _schedule_instruction("S_ADD_FP", "f2", "f2", "f1"),
            _schedule_instruction("S_SQRT_FP", "f2", "f2"),
            _schedule_instruction("S_RECI_FP", "f2", "f2"),
        )
    )
    batch_body.extend(
        _schedule_instruction("S_ADDI_INT", "gp0", "gp0", 0)
        for _ in range(4)
    )
    for chunk in range(chunks):
        current = address_registers[chunk % 2]
        batch_body.append(
            _schedule_instruction(
                "V_MUL_VF", f"gp{current}", f"gp{current}", "f2", 0
            )
        )
        if chunk + 2 < chunks:
            batch_body.append(
                affine_load(
                    f"rms_normalize_chunk_{chunk + 2}",
                    current,
                    stride * (chunk + 2),
                )
            )
        elif chunk + 1 < chunks:
            batch_body.append(
                _schedule_instruction("S_ADDI_INT", "gp0", "gp0", 0)
            )
    batch_body.append(_schedule_instruction("S_ADD_FP", "f2", "f0", "f0"))
    prefix.append(
        ScheduleRepeat(
            batch_size,
            ScheduleSequence(tuple(batch_body)),
            name="rms_batches",
            repeat_kind="compile_time",
        )
    )
    return ScheduleSequence(tuple(prefix))


__all__ = [
    "KernelCostSummary",
    "KernelCounts",
    "KernelDmaStream",
    "ffn_unrolled_cost_counts",
    "ffn_unrolled_cost_schedule",
]


def projection_call_cost_counts(
    *,
    mlen: int,
    blen: int,
    full_batch: int,
    row_loop_count: int,
    num_hidden_blocks: int,
    hbm_base_addr: int,
    hbm_rows: int,
    hbm_cols: int,
    hbm_offsets: list[int],
    vram_row_start_addr: int,
    result_vram_addr: int,
) -> KernelCounts:
    """Count one load_sub_matrix_col + rolled vram_sub_projection_to call."""
    result = KernelCounts()

    # _emit_hbm_matrix_load / preload_addr_reg_asm.
    _load_large(result, hbm_base_addr)
    result.add("C_SET_ADDR_REG")
    _direct_addi(result, hbm_rows * hbm_cols, source_is_zero=True)
    result.add("C_SET_SCALE_REG")
    _direct_addi(result, hbm_cols, source_is_zero=True)
    result.add("C_SET_STRIDE_REG")
    for block_idx, hbm_offset in enumerate(hbm_offsets):
        _direct_addi(result, block_idx * mlen * mlen, source_is_zero=True)
        _direct_addi(result, hbm_offset, source_is_zero=True)
        result.add("H_PREFETCH_M")

    # vram_sub_projection_asm_impl, rolled form.
    tiles_per_mlen = mlen // blen
    _load_large(result, 0)  # MRAM column base resets for every projection.
    _load_large(result, result_vram_addr)
    result.add("C_LOOP_START")
    _load_large(result, vram_row_start_addr, dynamic=tiles_per_mlen)
    result.add("S_ADDI_INT", static=1, dynamic=tiles_per_mlen)
    result.add("C_LOOP_START", static=1, dynamic=tiles_per_mlen)

    outer_middle = tiles_per_mlen * row_loop_count
    result.add("S_ADDI_INT", static=2, dynamic=2 * outer_middle)
    result.add("C_LOOP_START", static=1, dynamic=outer_middle)
    result.add("M_MM", static=1, dynamic=outer_middle * num_hidden_blocks)
    _direct_addi(
        result,
        full_batch * mlen,
        source_is_zero=False,
        static=1,
        dynamic=outer_middle * num_hidden_blocks,
    )
    _direct_addi(
        result,
        mlen * mlen,
        source_is_zero=False,
        static=1,
        dynamic=outer_middle * num_hidden_blocks,
    )
    result.add("C_LOOP_END", static=1, dynamic=outer_middle * num_hidden_blocks)
    result.add("M_MM_WO", static=1, dynamic=outer_middle)
    _direct_addi(result, blen * mlen, source_is_zero=False, static=2, dynamic=2 * outer_middle)
    result.add("C_LOOP_END", static=1, dynamic=outer_middle)
    _direct_addi(result, blen, source_is_zero=False, static=2, dynamic=2 * tiles_per_mlen)
    result.add("C_LOOP_END", static=1, dynamic=tiles_per_mlen)
    return result


def linear_projection_cost_counts(
    *,
    mlen: int,
    blen: int,
    full_batch: int,
    hbm_base_addr: int,
    hbm_rows: int,
    hbm_cols: int,
    input_base_addr: int,
    input_physical_rows: int,
    output_base_addr: int,
    output_physical_rows: int,
    temp_base_addr: int | None,
    num_row_blocks: int,
    num_col_blocks: int,
    chunks: list[tuple[int, int]],
    row_loop_counts: list[int],
    hbm_offsets: list[list[int]],
    hbm_prefetch_amount: int,
    source: str,
) -> KernelCounts:
    """Aggregate the complete ProgramMatrixOps.linear_projection cost in O(K*C)."""
    result = KernelCounts()
    tiles_per_mlen = mlen // blen
    calls_per_chunk = num_row_blocks * num_col_blocks

    for chunk_idx, (k_start, k_count) in enumerate(chunks):
        # HBM address register + scale/stride setup are redundantly emitted for
        # every output tile by the current compiler.
        _load_large(result, hbm_base_addr, static=calls_per_chunk, dynamic=calls_per_chunk)
        result.add("C_SET_ADDR_REG", calls_per_chunk)
        _direct_addi(
            result,
            hbm_rows * hbm_cols,
            source_is_zero=True,
            static=calls_per_chunk,
            dynamic=calls_per_chunk,
        )
        result.add("C_SET_SCALE_REG", calls_per_chunk)
        _direct_addi(
            result,
            hbm_cols,
            source_is_zero=True,
            static=calls_per_chunk,
            dynamic=calls_per_chunk,
        )
        result.add("C_SET_STRIDE_REG", calls_per_chunk)

        for col in range(num_col_blocks):
            offsets = hbm_offsets[col][k_start : k_start + k_count]
            first_offset = offsets[0]
            result.add_dma(
                DmaTransfer(
                    opcode="H_PREFETCH_M",
                    direction="read",
                    precision="weight",
                    element_base=hbm_base_addr + first_offset,
                    scale_base=hbm_base_addr + hbm_rows * hbm_cols + first_offset // 8,
                    dim=mlen,
                    amount=hbm_prefetch_amount,
                    stride=hbm_cols,
                    rstride=1,
                    write_amount=mlen,
                    geometry_fidelity="exact",
                    source=source,
                    memory_object=f"hbm:{hbm_base_addr}:{hbm_rows * hbm_cols}",
                    logical_object_elements=hbm_rows * hbm_cols,
                    precision_role="weight",
                    logical_element_offset=first_offset,
                    logical_scale_offset=first_offset // 8,
                    logical_stride=hbm_cols,
                ),
                multiplicity=num_row_blocks * k_count,
                axes=(
                    RepeatAxis("output_row_tile", num_row_blocks),
                    RepeatAxis(
                        "k_tile",
                        k_count,
                        element_base_delta=mlen * hbm_cols,
                        scale_base_delta=mlen * hbm_cols // 8,
                    ),
                ),
            )
            for local_k, offset in enumerate(offsets):
                _direct_addi(
                    result,
                    local_k * mlen * mlen,
                    source_is_zero=True,
                    static=num_row_blocks,
                    dynamic=num_row_blocks,
                )
                _direct_addi(
                    result,
                    offset,
                    source_is_zero=True,
                    static=num_row_blocks,
                    dynamic=num_row_blocks,
                )
            result.add("H_PREFETCH_M", static=k_count * num_row_blocks, dynamic=k_count * num_row_blocks)

        # Projection address loads outside hardware loops.
        result.add("S_ADDI_INT", calls_per_chunk)  # MRAM base = 0
        if chunk_idx == 0:
            for col in range(num_col_blocks):
                result_start = output_base_addr + col * output_physical_rows * mlen
                _load_large_sequence(
                    result,
                    start=result_start,
                    step=mlen * mlen,
                    count=num_row_blocks,
                )
        else:
            if temp_base_addr is None:
                raise ValueError("temp_base_addr is required for K-split projection")
            _load_large(result, temp_base_addr, static=calls_per_chunk, dynamic=calls_per_chunk)

        input_start = input_base_addr + k_start * input_physical_rows * mlen
        _load_large_sequence(
            result,
            start=input_start,
            step=mlen * mlen,
            count=num_row_blocks,
            multiplier=num_col_blocks,
            dynamic_multiplier=num_col_blocks * tiles_per_mlen,
        )

        for row_loop_count in set(row_loop_counts):
            rows_with_count = row_loop_counts.count(row_loop_count)
            calls = rows_with_count * num_col_blocks
            outer_middle = calls * tiles_per_mlen * row_loop_count
            result.add("C_LOOP_START", static=calls, dynamic=calls)
            result.add("S_ADDI_INT", static=calls, dynamic=calls * tiles_per_mlen)
            result.add("C_LOOP_START", static=calls, dynamic=calls * tiles_per_mlen)
            result.add("S_ADDI_INT", static=2 * calls, dynamic=2 * outer_middle)
            result.add("C_LOOP_START", static=calls, dynamic=outer_middle)
            result.add("M_MM", static=calls, dynamic=outer_middle * k_count)
            _direct_addi(
                result,
                full_batch * mlen,
                source_is_zero=False,
                static=calls,
                dynamic=outer_middle * k_count,
            )
            _direct_addi(
                result,
                mlen * mlen,
                source_is_zero=False,
                static=calls,
                dynamic=outer_middle * k_count,
            )
            result.add("C_LOOP_END", static=calls, dynamic=outer_middle * k_count)
            result.add("M_MM_WO", static=calls, dynamic=outer_middle)
            _direct_addi(
                result,
                blen * mlen,
                source_is_zero=False,
                static=2 * calls,
                dynamic=2 * outer_middle,
            )
            result.add("C_LOOP_END", static=calls, dynamic=outer_middle)
            _direct_addi(
                result,
                blen,
                source_is_zero=False,
                static=2 * calls,
                dynamic=2 * calls * tiles_per_mlen,
            )
            result.add("C_LOOP_END", static=calls, dynamic=calls * tiles_per_mlen)

        if chunk_idx:
            # One compact block-add kernel per output tile.
            for col in range(num_col_blocks):
                output_start = output_base_addr + col * output_physical_rows * mlen
                _load_large_sequence(
                    result,
                    start=output_start,
                    step=mlen * mlen,
                    count=num_row_blocks,
                    multiplier=2,
                )
            _load_large(
                result,
                temp_base_addr,
                static=calls_per_chunk,
                dynamic=calls_per_chunk,
            )
            result.add("C_LOOP_START", calls_per_chunk)
            result.add("V_ADD_VV", static=calls_per_chunk, dynamic=calls_per_chunk * mlen)
            result.add("S_ADDI_INT", static=3 * calls_per_chunk, dynamic=3 * calls_per_chunk * mlen)
            result.add("C_LOOP_END", static=calls_per_chunk, dynamic=calls_per_chunk * mlen)
    return result


def _rolled_projection_schedule(
    *,
    mlen: int,
    blen: int,
    full_batch: int,
    vram_row_load: ScheduleAffineLoad,
    result_vram_load: ScheduleAffineLoad | tuple[ScheduleInstruction, ...],
    num_hidden_blocks: int,
    row_loop_count: int,
) -> ScheduleSequence:
    """Compressed order of rolled ``vram_sub_projection_asm_impl``."""
    gp_act, gp_mat, gp_result = 1, 2, 3
    gp_loop_outer, gp_loop_middle, gp_loop_inner = 4, 5, 6
    gp_act_row_base, gp_mat_col_base, gp_result_col_base = 7, 8, 9
    tiles_per_mlen = mlen // blen
    children = [
        *_load_large_schedule(gp_mat_col_base, 0),
        *(
            (result_vram_load,)
            if isinstance(result_vram_load, ScheduleAffineLoad)
            else result_vram_load
        ),
        _schedule_instruction(
            "C_LOOP_START", f"gp{gp_loop_outer}", tiles_per_mlen
        ),
    ]
    outer_body = [
        vram_row_load,
        _schedule_instruction(
            "S_ADDI_INT",
            f"gp{gp_result}",
            f"gp{gp_result_col_base}",
            0,
        ),
        _schedule_instruction(
            "C_LOOP_START", f"gp{gp_loop_middle}", row_loop_count
        ),
    ]
    middle_body = [
        _schedule_instruction(
            "S_ADDI_INT", f"gp{gp_act}", f"gp{gp_act_row_base}", 0
        ),
        _schedule_instruction(
            "S_ADDI_INT", f"gp{gp_mat}", f"gp{gp_mat_col_base}", 0
        ),
        _schedule_instruction(
            "C_LOOP_START", f"gp{gp_loop_inner}", num_hidden_blocks
        ),
    ]
    inner_body = [
        _schedule_instruction("M_MM", f"gp{gp_mat}", f"gp{gp_act}"),
        *_add_large_schedule(
            gp_act, gp_act, full_batch * mlen
        ),
        *_add_large_schedule(gp_mat, gp_mat, mlen * mlen),
        _schedule_instruction("C_LOOP_END", f"gp{gp_loop_inner}"),
    ]
    middle_body.append(
        ScheduleRepeat(
            num_hidden_blocks,
            ScheduleSequence(tuple(inner_body)),
            name="projection_k_tiles",
            repeat_kind="hardware_loop",
        )
    )
    middle_body.extend(
        (
            _schedule_instruction(
                "M_MM_WO", f"gp{gp_result}", "gp0", 0
            ),
            *_add_large_schedule(
                gp_act_row_base, gp_act_row_base, blen * mlen
            ),
            *_add_large_schedule(gp_result, gp_result, blen * mlen),
            _schedule_instruction("C_LOOP_END", f"gp{gp_loop_middle}"),
        )
    )
    outer_body.append(
        ScheduleRepeat(
            row_loop_count,
            ScheduleSequence(tuple(middle_body)),
            name="projection_batch_rows",
            repeat_kind="hardware_loop",
        )
    )
    outer_body.extend(
        (
            *_add_large_schedule(
                gp_mat_col_base, gp_mat_col_base, blen
            ),
            *_add_large_schedule(
                gp_result_col_base, gp_result_col_base, blen
            ),
            _schedule_instruction("C_LOOP_END", f"gp{gp_loop_outer}"),
        )
    )
    children.append(
        ScheduleRepeat(
            tiles_per_mlen,
            ScheduleSequence(tuple(outer_body)),
            name="projection_output_columns",
            repeat_kind="hardware_loop",
        )
    )
    return ScheduleSequence(tuple(children))


def _projection_hbm_load_schedule(
    *,
    hbm_base_addr: int,
    hbm_rows: int,
    hbm_cols: int,
    offset_start: int,
    offset_step: int,
    k_count: int,
    mlen: int,
    memory_stream_index: int,
    key_prefix: str,
) -> ScheduleSequence:
    children = [
        *_load_large_schedule(4, hbm_base_addr),
        _schedule_instruction("C_SET_ADDR_REG", "a1", "gp0", "gp4"),
        *_load_large_schedule(1, hbm_rows * hbm_cols),
        _schedule_instruction("C_SET_SCALE_REG", "gp1"),
        *_load_large_schedule(2, hbm_cols),
        _schedule_instruction("C_SET_STRIDE_REG", "gp2"),
    ]
    children.append(
        ScheduleRepeat(
            k_count,
            ScheduleSequence(
                (
                    ScheduleAffineLoad(
                        key=f"{key_prefix}:mram_k",
                        register="gp3",
                        start=0,
                        step=mlen * mlen,
                        period=k_count,
                    ),
                    ScheduleAffineLoad(
                        key=f"{key_prefix}:hbm_k",
                        register="gp1",
                        start=offset_start,
                        step=offset_step,
                        period=k_count,
                    ),
                    _schedule_instruction(
                        "H_PREFETCH_M",
                        "gp3",
                        "gp1",
                        "a1",
                        1,
                        0,
                        memory_stream_index=memory_stream_index,
                    ),
                )
            ),
            name="projection_hbm_k_tiles",
            repeat_kind="compile_time",
        )
    )
    return ScheduleSequence(tuple(children))


def _projection_partial_add_schedule(
    *,
    mlen: int,
    output_load_1: ScheduleAffineLoad,
    output_load_2: ScheduleAffineLoad,
    temp_addr: int,
) -> ScheduleSequence:
    children = [
        output_load_1,
        output_load_2,
        *_load_large_schedule(3, temp_addr),
        _schedule_instruction("C_LOOP_START", "gp4", mlen),
        ScheduleRepeat(
            mlen,
            ScheduleSequence(
                (
                    _schedule_instruction(
                        "V_ADD_VV", "gp1", "gp2", "gp3", 0
                    ),
                    _schedule_instruction("S_ADDI_INT", "gp1", "gp1", mlen),
                    _schedule_instruction("S_ADDI_INT", "gp2", "gp2", mlen),
                    _schedule_instruction("S_ADDI_INT", "gp3", "gp3", mlen),
                    _schedule_instruction("C_LOOP_END", "gp4"),
                )
            ),
            name="projection_partial_add_rows",
            repeat_kind="hardware_loop",
        ),
    ]
    return ScheduleSequence(tuple(children))


def linear_projection_cost_schedule(
    *,
    mlen: int,
    blen: int,
    full_batch: int,
    hbm_base_addr: int,
    hbm_rows: int,
    hbm_cols: int,
    input_base_addr: int,
    input_physical_rows: int,
    output_base_addr: int,
    output_physical_rows: int,
    temp_base_addr: int | None,
    num_row_blocks: int,
    num_col_blocks: int,
    chunks: list[tuple[int, int]],
    row_loop_counts: list[int],
    hbm_offsets: list[list[int]],
) -> ScheduleSequence:
    """Program-order schedule matching ``ProgramMatrixOps.linear_projection``."""
    children = []
    stream_index = 0
    for chunk_index, (k_start, k_count) in enumerate(chunks):
        for col in range(num_col_blocks):
            offsets = hbm_offsets[col][k_start : k_start + k_count]
            if not offsets:
                raise ValueError("projection chunk has no HBM tile offsets")
            offset_step = (
                offsets[1] - offsets[0]
                if len(offsets) > 1
                else mlen * hbm_cols
            )
            if any(
                right - left != offset_step
                for left, right in zip(offsets, offsets[1:])
            ):
                raise ValueError(
                    "projection HBM K-tile offsets are not affine: "
                    f"{offsets!r}"
                )
            key_prefix = (
                f"projection:{hbm_base_addr}:{input_base_addr}:"
                f"{output_base_addr}:c{chunk_index}:o{col}"
            )
            input_load = ScheduleAffineLoad(
                key=f"{key_prefix}:input_rows",
                register="gp7",
                start=(
                    input_base_addr
                    + k_start * input_physical_rows * mlen
                ),
                step=mlen * mlen,
                period=num_row_blocks * (mlen // blen),
                advance_every=mlen // blen,
            )
            if chunk_index == 0:
                result_load: ScheduleAffineLoad | tuple[
                    ScheduleInstruction, ...
                ] = ScheduleAffineLoad(
                    key=f"{key_prefix}:result_rows",
                    register="gp9",
                    start=(
                        output_base_addr
                        + col * output_physical_rows * mlen
                    ),
                    step=mlen * mlen,
                    period=num_row_blocks,
                )
            else:
                if temp_base_addr is None:
                    raise ValueError("temp_base_addr is required for K-split projection")
                result_load = _load_large_schedule(9, temp_base_addr)

            def row_schedule(row_loop_count: int) -> ScheduleSequence:
                row_children = list(
                    _projection_hbm_load_schedule(
                        hbm_base_addr=hbm_base_addr,
                        hbm_rows=hbm_rows,
                        hbm_cols=hbm_cols,
                        offset_start=offsets[0],
                        offset_step=offset_step,
                        k_count=k_count,
                        mlen=mlen,
                        memory_stream_index=stream_index,
                        key_prefix=key_prefix,
                    ).children
                )
                row_children.extend(
                    _rolled_projection_schedule(
                        mlen=mlen,
                        blen=blen,
                        full_batch=full_batch,
                        vram_row_load=input_load,
                        result_vram_load=result_load,
                        num_hidden_blocks=k_count,
                        row_loop_count=row_loop_count,
                    ).children
                )
                if chunk_index:
                    output_start = (
                        output_base_addr + col * output_physical_rows * mlen
                    )
                    assert temp_base_addr is not None
                    row_children.extend(
                        _projection_partial_add_schedule(
                            mlen=mlen,
                            output_load_1=ScheduleAffineLoad(
                                key=f"{key_prefix}:partial_out_1",
                                register="gp1",
                                start=output_start,
                                step=mlen * mlen,
                                period=num_row_blocks,
                            ),
                            output_load_2=ScheduleAffineLoad(
                                key=f"{key_prefix}:partial_out_2",
                                register="gp2",
                                start=output_start,
                                step=mlen * mlen,
                                period=num_row_blocks,
                            ),
                            temp_addr=temp_base_addr,
                        ).children
                    )
                return ScheduleSequence(tuple(row_children))

            run_start = 0
            while run_start < num_row_blocks:
                row_loop_count = row_loop_counts[run_start]
                run_end = run_start + 1
                while (
                    run_end < num_row_blocks
                    and row_loop_counts[run_end] == row_loop_count
                ):
                    run_end += 1
                children.append(
                    ScheduleRepeat(
                        run_end - run_start,
                        row_schedule(row_loop_count),
                        name="projection_output_rows",
                        repeat_kind="compile_time",
                    )
                )
                run_start = run_end
            stream_index += 1
    return ScheduleSequence(tuple(children))


def vram_block_add_cost_counts(
    *,
    mlen: int,
    dst_addr: int,
    src1_addr: int,
    src2_addr: int,
) -> KernelCounts:
    result = KernelCounts()
    for address in (dst_addr, src1_addr, src2_addr):
        _load_large(result, address)
    result.add("C_LOOP_START")
    result.add("V_ADD_VV", static=1, dynamic=mlen)
    result.add("S_ADDI_INT", static=3, dynamic=3 * mlen)
    result.add("C_LOOP_END", static=1, dynamic=mlen)
    return result


def vram_matrix_binary_cost_counts(
    *,
    opcode: str,
    mlen: int,
    dst_base: int,
    src_base: int,
    dst_physical_rows: int,
    src_physical_rows: int,
    physical_cols: int,
    dst_row_offset: int,
    src_row_offset: int,
    num_rows: int,
    block_aligned: bool,
) -> KernelCounts:
    result = KernelCounts()
    num_col_blocks = physical_cols // mlen
    if block_aligned and opcode == "V_ADD_VV":
        row_blocks = num_rows // mlen
        block_count = row_blocks * num_col_blocks
        for col in range(num_col_blocks):
            dst_start = (
                dst_base
                + col * dst_physical_rows * mlen
                + (dst_row_offset // mlen) * mlen * mlen
            )
            src_start = (
                src_base
                + col * src_physical_rows * mlen
                + (src_row_offset // mlen) * mlen * mlen
            )
            _load_large_sequence(result, start=dst_start, step=mlen * mlen, count=row_blocks, multiplier=2)
            _load_large_sequence(result, start=src_start, step=mlen * mlen, count=row_blocks)
        result.add("C_LOOP_START", block_count)
        result.add("V_ADD_VV", static=block_count, dynamic=block_count * mlen)
        result.add("S_ADDI_INT", static=3 * block_count, dynamic=3 * block_count * mlen)
        result.add("C_LOOP_END", static=block_count, dynamic=block_count * mlen)
        return result

    for col in range(num_col_blocks):
        dst_start = dst_base + col * dst_physical_rows * mlen + dst_row_offset * mlen
        src_start = src_base + col * src_physical_rows * mlen + src_row_offset * mlen
        _load_large_sequence(result, start=dst_start, step=mlen, count=num_rows)
        _load_large_sequence(result, start=src_start, step=mlen, count=num_rows)
    result.add(opcode, num_rows * num_col_blocks)
    return result


def vram_matrix_binary_cost_schedule(
    *,
    opcode: str,
    mlen: int,
    dst_base: int,
    src_base: int,
    dst_physical_rows: int,
    src_physical_rows: int,
    physical_cols: int,
    dst_row_offset: int,
    src_row_offset: int,
    num_rows: int,
    block_aligned: bool,
    gp_regs: list[int],
) -> ScheduleSequence:
    """Ordered schedule for ``vram_matrix_add/mul`` without row expansion."""
    if len(gp_regs) < 4:
        raise ValueError("VRAM matrix binary schedule requires four GP registers")
    gp_dst, gp_src1, gp_src2, gp_loop = gp_regs[:4]
    num_col_blocks = physical_cols // mlen
    children = []

    if block_aligned and opcode == "V_ADD_VV":
        row_blocks = num_rows // mlen
        dst_row_block_base = dst_row_offset // mlen
        src_row_block_base = src_row_offset // mlen
        row_body = []
        key_base = (
            f"vram:{opcode}:{dst_base}:{src_base}:{dst_row_offset}:"
            f"{src_row_offset}:{num_rows}:{physical_cols}:"
            f"{dst_physical_rows}:{src_physical_rows}:{mlen}:block:"
            f"{gp_dst}:{gp_src1}:{gp_src2}"
        )
        for col_block in range(num_col_blocks):
            dst_start = (
                dst_base
                + col_block * dst_physical_rows * mlen
                + dst_row_block_base * mlen * mlen
            )
            src_start = (
                src_base
                + col_block * src_physical_rows * mlen
                + src_row_block_base * mlen * mlen
            )
            for role, register, start in (
                ("dst", gp_dst, dst_start),
                ("src1", gp_src1, dst_start),
                ("src2", gp_src2, src_start),
            ):
                row_body.append(
                    ScheduleAffineLoad(
                        key=f"{key_base}:c{col_block}:{role}",
                        register=f"gp{register}",
                        start=start,
                        step=mlen * mlen,
                        period=row_blocks,
                    )
                )
            row_body.append(
                _schedule_instruction(
                    "C_LOOP_START", f"gp{gp_loop}", mlen
                )
            )
            row_body.append(
                ScheduleRepeat(
                    mlen,
                    ScheduleSequence(
                        (
                            _schedule_instruction(
                                "V_ADD_VV",
                                f"gp{gp_dst}",
                                f"gp{gp_src1}",
                                f"gp{gp_src2}",
                                0,
                            ),
                            _schedule_instruction(
                                "S_ADDI_INT",
                                f"gp{gp_dst}",
                                f"gp{gp_dst}",
                                mlen,
                            ),
                            _schedule_instruction(
                                "S_ADDI_INT",
                                f"gp{gp_src1}",
                                f"gp{gp_src1}",
                                mlen,
                            ),
                            _schedule_instruction(
                                "S_ADDI_INT",
                                f"gp{gp_src2}",
                                f"gp{gp_src2}",
                                mlen,
                            ),
                            _schedule_instruction(
                                "C_LOOP_END", f"gp{gp_loop}"
                            ),
                        )
                    ),
                    name="vram_block_rows",
                    repeat_kind="hardware_loop",
                )
            )
        children.append(
            ScheduleRepeat(
                row_blocks,
                ScheduleSequence(tuple(row_body)),
                name="vram_matrix_row_blocks",
                repeat_kind="compile_time",
            )
        )
        return ScheduleSequence(tuple(children))

    # The fallback is compiler-unrolled by logical row/column block. Keep the
    # row-major order, but make each column's absolute address an affine stream
    # across the outer row repeat.
    key_base = (
        f"vram:{opcode}:{dst_base}:{src_base}:{dst_row_offset}:"
        f"{src_row_offset}:{num_rows}:{physical_cols}:"
        f"{dst_physical_rows}:{src_physical_rows}:{mlen}:row:"
        f"{gp_dst}:{gp_src1}"
    )
    row_body = []
    for col_block in range(num_col_blocks):
        dst_start = (
            dst_base
            + col_block * dst_physical_rows * mlen
            + dst_row_offset * mlen
        )
        src_start = (
            src_base
            + col_block * src_physical_rows * mlen
            + src_row_offset * mlen
        )
        row_body.extend(
            (
                ScheduleAffineLoad(
                    key=f"{key_base}:c{col_block}:dst",
                    register=f"gp{gp_dst}",
                    start=dst_start,
                    step=mlen,
                    period=num_rows,
                ),
                ScheduleAffineLoad(
                    key=f"{key_base}:c{col_block}:src",
                    register=f"gp{gp_src1}",
                    start=src_start,
                    step=mlen,
                    period=num_rows,
                ),
                _schedule_instruction(
                    opcode,
                    f"gp{gp_dst}",
                    f"gp{gp_dst}",
                    f"gp{gp_src1}",
                    0,
                ),
            )
        )
    children.append(
        ScheduleRepeat(
            num_rows,
            ScheduleSequence(tuple(row_body)),
            name="vram_matrix_rows",
            repeat_kind="compile_time",
        )
    )
    return ScheduleSequence(tuple(children))


__all__ += [
    "linear_projection_cost_counts",
    "linear_projection_cost_schedule",
    "projection_call_cost_counts",
    "rms_norm_cost_counts",
    "rms_norm_cost_schedule",
    "vram_block_add_cost_counts",
    "vram_matrix_binary_cost_counts",
    "vram_matrix_binary_cost_schedule",
]
