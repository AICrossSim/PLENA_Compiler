"""Algebraic cost lowering for legacy templates with large Python unrolls."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from compiler.asm_templates._k_split import k_chunks
from compiler.asm_templates._imm import IMM2_BOUND
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


__all__ = [
    "KernelCostSummary",
    "KernelCounts",
    "KernelDmaStream",
    "ffn_unrolled_cost_counts",
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


__all__ += [
    "linear_projection_cost_counts",
    "projection_call_cost_counts",
    "rms_norm_cost_counts",
    "vram_block_add_cost_counts",
    "vram_matrix_binary_cost_counts",
]
