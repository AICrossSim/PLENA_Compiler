"""Algebraic cost adapters for legacy compiler templates.

Main still renders the FFN kernel through a Python-unrolled assembly template.
That representation is useful for executable output but scales poorly for
shape-only analytical compilation.  The adapter below is compiler-owned and
mirrors the template after large-immediate legalization.  Tests compare it
opcode-for-opcode with the rendered final schedule on representative shapes.
Timing and memory models consume only the resulting CostTrace.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math

from compiler.asm_templates._imm import IMM2_BOUND
from compiler.asm_templates._k_split import k_chunks
from compiler.aten.isa_builder import DmaTransfer, RepeatAxis


@dataclass(frozen=True)
class KernelDmaStream:
    transfer: DmaTransfer
    multiplicity: int
    axes: tuple[RepeatAxis, ...] = ()


@dataclass
class KernelCostSummary:
    opcodes: Counter[str] = field(default_factory=Counter)
    dma_streams: list[KernelDmaStream] = field(default_factory=list)

    def add(self, opcode: str, count: int = 1) -> None:
        if count < 0:
            raise ValueError(f"negative {opcode} count")
        self.opcodes[opcode] += count

    def merge(self, other: "KernelCostSummary") -> None:
        self.opcodes.update(other.opcodes)
        self.dma_streams.extend(other.dma_streams)


def _load_large(result: KernelCostSummary, value: int, count: int = 1) -> None:
    if value < IMM2_BOUND:
        result.add("S_ADDI_INT", count)
        return
    result.add("S_LUI_INT", count)
    if value & 0xFFF:
        result.add("S_ADDI_INT", count)


def _load_large_sequence(
    result: KernelCostSummary,
    *,
    start: int,
    step: int,
    count: int,
    multiplier: int = 1,
) -> None:
    """Count ``load_large_int`` over an affine sequence in O(1)."""
    if count <= 0:
        return
    if step < 0:
        raise ValueError("negative affine address step is unsupported")
    if step == 0:
        _load_large(result, start, count * multiplier)
        return
    below = (
        max(0, min(count, (IMM2_BOUND - start + step - 1) // step))
        if start < IMM2_BOUND
        else 0
    )
    high = count - below
    zero_low = 0
    if high:
        modulus = 1 << 12
        divisor = math.gcd(step, modulus)
        rhs = -start
        if rhs % divisor == 0:
            reduced_modulus = modulus // divisor
            first = (
                (rhs // divisor) * pow(step // divisor, -1, reduced_modulus)
            ) % reduced_modulus
            if first < below:
                first += (
                    (below - first + reduced_modulus - 1) // reduced_modulus
                ) * reduced_modulus
            if first < count:
                zero_low = 1 + (count - 1 - first) // reduced_modulus
    result.add("S_LUI_INT", high * multiplier)
    result.add("S_ADDI_INT", (below + high - zero_low) * multiplier)


def _direct_addi(
    result: KernelCostSummary,
    value: int,
    *,
    source_is_zero: bool,
    count: int = 1,
) -> None:
    """Count a raw S_ADDI_INT after final-schedule legalization."""
    if value < IMM2_BOUND:
        result.add("S_ADDI_INT", count)
    elif source_is_zero:
        _load_large(result, value, count)
    else:
        chunks = (value + IMM2_BOUND - 2) // (IMM2_BOUND - 1)
        result.add("S_ADDI_INT", count * chunks)


def _addi_with_temp(result: KernelCostSummary, value: int, count: int = 1) -> None:
    if value < IMM2_BOUND:
        result.add("S_ADDI_INT", count)
        return
    _load_large(result, value, count)
    result.add("S_ADD_INT", count)


def _addi_with_temp_sequence(
    result: KernelCostSummary,
    *,
    start: int,
    step: int,
    count: int,
    multiplier: int = 1,
) -> None:
    """Count ``addi_large_int(..., temp)`` over an affine sequence."""
    if count <= 0:
        return
    below = (
        max(0, min(count, (IMM2_BOUND - start + step - 1) // step))
        if start < IMM2_BOUND
        else 0
    )
    high = count - below
    result.add("S_ADDI_INT", below * multiplier)
    if not high:
        return
    modulus = 1 << 12
    divisor = math.gcd(step, modulus)
    zero_low = 0
    rhs = -start
    if rhs % divisor == 0:
        reduced_modulus = modulus // divisor
        first = (
            (rhs // divisor) * pow(step // divisor, -1, reduced_modulus)
        ) % reduced_modulus
        if first < below:
            first += (
                (below - first + reduced_modulus - 1) // reduced_modulus
            ) * reduced_modulus
        if first < count:
            zero_low = 1 + (count - 1 - first) // reduced_modulus
    result.add("S_LUI_INT", high * multiplier)
    result.add("S_ADD_INT", high * multiplier)
    result.add("S_ADDI_INT", (high - zero_low) * multiplier)


def _projection_chunk(
    *,
    mlen: int,
    blen: int,
    batch_rows: int,
    k_size: int,
    out_size: int,
    weight_stride: int,
    result_base: int,
    activation_base: int | None,
    activation_register_base: int | None,
    k_start_tile: int,
    k_tile_count: int,
    target_base: int,
    weight_dma: DmaTransfer,
) -> KernelCostSummary:
    result = KernelCostSummary()
    num_act_cols = batch_rows // blen
    tiles_per_mlen = mlen // blen
    weight_rows = out_size // blen
    block_rows = out_size // mlen
    chunk_hbm_offset = k_start_tile * mlen * weight_stride
    chunk_act_offset = k_start_tile * mlen * batch_rows

    _load_large(result, target_base)
    if activation_base is None:
        if activation_register_base is None:
            raise ValueError("register activation requires a canonical base")
        _load_large(result, activation_register_base)

    # Once per output MLEN block: MRAM reset, HBM offset and output pointer.
    result.add("S_ADDI_INT", block_rows)
    _addi_with_temp_sequence(
        result,
        start=chunk_hbm_offset,
        step=mlen,
        count=block_rows,
    )
    result.add("S_ADDI_INT", block_rows)

    prefetches = block_rows * k_tile_count
    result.add("H_PREFETCH_M", prefetches)
    _direct_addi(result, mlen * mlen, source_is_zero=False, count=prefetches)
    _addi_with_temp(result, mlen * weight_stride, prefetches)
    result.add("S_ADDI_INT", block_rows)

    # The remaining BLEN tiles in each MLEN block select MRAM/output columns.
    subtile_rows = weight_rows - block_rows
    for subtile in range(1, tiles_per_mlen):
        _direct_addi(
            result,
            subtile * blen * mlen,
            source_is_zero=True,
            count=block_rows,
        )
        result.add("S_ADDI_INT", block_rows)
    if subtile_rows != block_rows * (tiles_per_mlen - 1):
        raise AssertionError("FFN output tiling is inconsistent")

    pointer_start = chunk_act_offset + (activation_base or 0)
    _addi_with_temp_sequence(
        result,
        start=pointer_start,
        step=mlen * blen,
        count=num_act_cols,
        multiplier=weight_rows,
    )
    cells = weight_rows * num_act_cols
    result.add("S_ADDI_INT", cells)  # w_temp = w_actual
    if k_tile_count > 1:
        result.add("C_LOOP_START", cells)
    result.add("M_MM", cells * k_tile_count)
    _direct_addi(
        result,
        mlen * mlen,
        source_is_zero=False,
        count=cells * k_tile_count,
    )
    _direct_addi(
        result,
        mlen * batch_rows,
        source_is_zero=False,
        count=cells * k_tile_count,
    )
    if k_tile_count > 1:
        result.add("C_LOOP_END", cells * k_tile_count)
    result.add("M_MM_WO", cells)
    _direct_addi(
        result,
        blen * mlen,
        source_is_zero=False,
        count=cells,
    )
    if block_rows > 1:
        _direct_addi(
            result,
            mlen * batch_rows,
            source_is_zero=False,
            count=block_rows - 1,
        )

    result.dma_streams.append(
        KernelDmaStream(
            transfer=DmaTransfer(
                **{
                    **weight_dma.__dict__,
                    "element_base_bytes": (
                        weight_dma.element_base_bytes + chunk_hbm_offset
                    ),
                    "scale_base_bytes": (
                        None
                        if weight_dma.scale_base_bytes is None
                        else weight_dma.scale_base_bytes + chunk_hbm_offset // 8
                    ),
                }
            ),
            multiplicity=prefetches,
            axes=(
                RepeatAxis.from_mapping(
                    "output_mlen_block",
                    block_rows,
                    {
                        "element_base_bytes": mlen,
                        "scale_base_bytes": mlen // 8,
                    },
                ),
                RepeatAxis.from_mapping(
                    "k_tile",
                    k_tile_count,
                    {
                        "element_base_bytes": mlen * weight_stride,
                        "scale_base_bytes": mlen * weight_stride // 8,
                    },
                ),
            ),
        )
    )
    return result


def _projection(
    *,
    mlen: int,
    vlen: int,
    blen: int,
    batch_rows: int,
    k_size: int,
    out_size: int,
    weight_stride: int,
    result_base: int,
    activation_base: int | None,
    activation_register_base: int | None,
    max_k_tiles: int,
    scratch_base: int,
    weight_dma: DmaTransfer,
) -> KernelCostSummary:
    result = KernelCostSummary()
    chunks = k_chunks(k_size // mlen, max_k_tiles)
    for chunk_index, (k_start, k_count) in enumerate(chunks):
        result.merge(
            _projection_chunk(
                mlen=mlen,
                blen=blen,
                batch_rows=batch_rows,
                k_size=k_size,
                out_size=out_size,
                weight_stride=weight_stride,
                result_base=result_base,
                activation_base=activation_base,
                activation_register_base=activation_register_base,
                k_start_tile=k_start,
                k_tile_count=k_count,
                target_base=result_base if chunk_index == 0 else scratch_base,
                weight_dma=weight_dma,
            )
        )
        if chunk_index:
            _load_large(result, result_base)
            _load_large(result, scratch_base)
            adds = math.ceil(out_size * batch_rows / vlen)
            result.add("V_ADD_VV", adds)
            result.add("S_ADDI_INT", 2 * adds)
    return result


def ffn_unrolled_cost_summary(
    *,
    mlen: int,
    vlen: int,
    blen: int,
    batch_rows: int,
    hidden_size: int,
    intermediate_size: int,
    activation_base: int,
    workspace_base: int,
    matrix_sram_size: int,
    gate_weight_dma: DmaTransfer,
    up_weight_dma: DmaTransfer,
    down_weight_dma: DmaTransfer,
) -> KernelCostSummary:
    """Exact dynamic census for main's ``_ffn_asm_unrolled`` schedule."""
    if batch_rows <= 0 or batch_rows % blen:
        raise ValueError("FFN rows must be a positive BLEN multiple")
    up_base = workspace_base
    gate_base = up_base + batch_rows * intermediate_size
    scratch_base = gate_base + batch_rows * intermediate_size
    max_k_tiles = max(1, matrix_sram_size // mlen)
    result = KernelCostSummary()

    _load_large(result, hidden_size * intermediate_size)
    result.add("C_SET_SCALE_REG")
    _load_large(result, intermediate_size)
    result.add("C_SET_STRIDE_REG")
    result.add("S_ADDI_INT")
    _load_large(result, up_base)
    _load_large(result, gate_base)

    for result_base, dma in ((up_base, up_weight_dma), (gate_base, gate_weight_dma)):
        result.merge(
            _projection(
                mlen=mlen,
                vlen=vlen,
                blen=blen,
                batch_rows=batch_rows,
                k_size=hidden_size,
                out_size=intermediate_size,
                weight_stride=intermediate_size,
                result_base=result_base,
                activation_base=activation_base,
                activation_register_base=None,
                max_k_tiles=max_k_tiles,
                scratch_base=scratch_base,
                weight_dma=dma,
            )
        )

    result.add("S_LD_FP")
    _load_large(result, up_base)
    _load_large(result, gate_base)
    _load_large(result, activation_base)
    silu = batch_rows * (intermediate_size // vlen)
    for opcode in ("V_SUB_VF", "V_EXP_V", "V_ADD_VF", "V_RECI_V"):
        result.add(opcode, silu)
    result.add("V_MUL_VV", 2 * silu)
    result.add("S_ADDI_INT", 2 * silu)

    _load_large(result, hidden_size * intermediate_size)
    result.add("C_SET_SCALE_REG")
    _load_large(result, hidden_size)
    result.add("C_SET_STRIDE_REG")
    result.add("S_ADDI_INT", 2)  # w_actual reset and historical m_stride load
    result.merge(
        _projection(
            mlen=mlen,
            vlen=vlen,
            blen=blen,
            batch_rows=batch_rows,
            k_size=intermediate_size,
            out_size=hidden_size,
            weight_stride=hidden_size,
            result_base=activation_base,
            activation_base=None,
            activation_register_base=up_base,
            max_k_tiles=max_k_tiles,
            scratch_base=scratch_base,
            weight_dma=down_weight_dma,
        )
    )
    return result


def router_topk_cost_summary(
    *,
    token_count: int,
    top_k: int,
    weights_fp_base: int,
    indices_int_base: int,
    logits_base: int,
    logits_token_stride: int,
    weights_stride: int | None = None,
    indices_stride: int | None = None,
) -> KernelCostSummary:
    """Census repeated V_TOPK rows without materializing per-token objects."""
    if token_count <= 0 or top_k <= 0 or logits_token_stride <= 0:
        raise ValueError("router dimensions must be positive")
    weights_stride = top_k if weights_stride is None else weights_stride
    indices_stride = top_k if indices_stride is None else indices_stride
    if weights_stride < 0 or indices_stride < 0:
        raise ValueError("router output strides must be nonnegative")
    result = KernelCostSummary()
    _load_large_sequence(
        result,
        start=weights_fp_base,
        step=weights_stride,
        count=token_count,
    )
    _load_large_sequence(
        result,
        start=logits_base,
        step=logits_token_stride,
        count=token_count,
    )
    _load_large_sequence(
        result,
        start=indices_int_base,
        step=indices_stride,
        count=token_count,
    )
    result.add("V_TOPK", token_count)
    return result


def vram_matrix_binary_cost_summary(
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
    block_add: bool,
) -> KernelCostSummary:
    """Census main's VRAM add/multiply lowering without row expansion."""
    result = KernelCostSummary()
    col_blocks = physical_cols // mlen
    if block_add:
        if num_rows % mlen:
            raise ValueError("block-add rows must be MLEN aligned")
        row_blocks = num_rows // mlen
        blocks = row_blocks * col_blocks
        for col in range(col_blocks):
            dst_start = (
                dst_base
                + col * dst_physical_rows * mlen
                + dst_row_offset * mlen
            )
            src_start = (
                src_base
                + col * src_physical_rows * mlen
                + src_row_offset * mlen
            )
            _load_large_sequence(
                result,
                start=dst_start,
                step=mlen * mlen,
                count=row_blocks,
                multiplier=2,
            )
            _load_large_sequence(
                result,
                start=src_start,
                step=mlen * mlen,
                count=row_blocks,
            )
        result.add("C_LOOP_START", blocks)
        result.add(opcode, blocks * mlen)
        result.add("S_ADDI_INT", 3 * blocks * mlen)
        result.add("C_LOOP_END", blocks * mlen)
        return result

    for col in range(col_blocks):
        _load_large_sequence(
            result,
            start=(
                dst_base
                + col * dst_physical_rows * mlen
                + dst_row_offset * mlen
            ),
            step=mlen,
            count=num_rows,
        )
        _load_large_sequence(
            result,
            start=(
                src_base
                + col * src_physical_rows * mlen
                + src_row_offset * mlen
            ),
            step=mlen,
            count=num_rows,
        )
    result.add(opcode, num_rows * col_blocks)
    return result


def normalization_cost_summary(
    *,
    mode: str,
    activation_base: int,
    scratchpad_base: int,
    vlen: int,
    batch_size: int,
    hidden_dim: int,
    unroll: bool,
) -> KernelCostSummary:
    """Exact census for main's RMSNorm/LayerNorm assembly templates."""
    mode = mode.lower()
    if mode not in {"rms", "layer"}:
        raise ValueError(f"unsupported normalization mode {mode!r}")
    chunks = hidden_dim // vlen
    stride = vlen * batch_size
    result = KernelCostSummary()
    _load_large(result, scratchpad_base)

    if mode == "rms":
        result.add("S_LD_FP", 2)
        result.add("S_ADD_FP")
        _load_large_sequence(
            result,
            start=activation_base,
            step=vlen,
            count=batch_size,
            multiplier=2,
        )
        if unroll:
            result.add("V_MUL_VV", batch_size * chunks)
            result.add("V_RED_SUM", batch_size * chunks)
            _direct_addi(
                result,
                stride,
                source_is_zero=False,
                count=batch_size * chunks,
            )
        else:
            result.add("C_LOOP_START", batch_size)
            result.add("V_MUL_VV", batch_size * chunks)
            result.add("V_RED_SUM", batch_size * chunks)
            _direct_addi(
                result,
                stride,
                source_is_zero=False,
                count=batch_size * chunks,
            )
            result.add("C_LOOP_END", batch_size * chunks)
        if chunks > 1:
            _load_large_sequence(
                result,
                start=activation_base + stride,
                step=vlen,
                count=batch_size,
            )
        result.add("S_MUL_FP", batch_size)
        result.add("S_ADD_FP", 2 * batch_size)  # epsilon plus reset
        result.add("S_SQRT_FP", batch_size)
        result.add("S_RECI_FP", batch_size)
        result.add("S_ADDI_INT", 4 * batch_size)
        result.add("V_MUL_VF", batch_size * chunks)
        for chunk in range(2, chunks):
            _load_large_sequence(
                result,
                start=activation_base + stride * chunk,
                step=vlen,
                count=batch_size,
            )
        if chunks > 1:
            result.add("S_ADDI_INT", batch_size)
        return result

    result.add("S_LD_FP", 2)
    result.add("S_ADD_FP", 2)
    _load_large_sequence(
        result,
        start=activation_base,
        step=vlen,
        count=batch_size,
        multiplier=2,
    )
    if not unroll:
        result.add("C_LOOP_START", 2 * batch_size)
    result.add("V_RED_SUM", 2 * batch_size * chunks)
    result.add("V_MUL_VV", batch_size * chunks)
    _direct_addi(
        result,
        stride,
        source_is_zero=False,
        count=2 * batch_size * chunks,
    )
    if not unroll:
        result.add("C_LOOP_END", 2 * batch_size * chunks)
    result.add("S_MUL_FP", 3 * batch_size)
    result.add("S_SUB_FP", batch_size)
    result.add("S_ADD_FP", 3 * batch_size)  # epsilon plus two resets
    result.add("S_SQRT_FP", batch_size)
    result.add("S_RECI_FP", batch_size)
    result.add("V_SUB_VF", batch_size * chunks)
    result.add("V_MUL_VF", batch_size * chunks)
    return result


def linear_projection_cost_summary(
    *,
    mlen: int,
    blen: int,
    full_batch: int,
    hbm_base: int,
    hbm_rows: int,
    hbm_cols: int,
    input_base: int,
    input_physical_rows: int,
    output_base: int,
    output_physical_rows: int,
    temp_base: int | None,
    num_row_blocks: int,
    num_col_blocks: int,
    chunks: list[tuple[int, int]],
    row_loop_counts: list[int],
    hbm_offsets: list[list[int]],
    dma_prototypes: list[DmaTransfer],
    include_hbm_base_setup: bool = True,
) -> KernelCostSummary:
    """Aggregate main's tiled ``linear_projection`` in O(K+C) space."""
    result = KernelCostSummary()
    tiles_per_mlen = mlen // blen
    calls_per_chunk = num_row_blocks * num_col_blocks
    total_k_tiles = sum(k_count for _k_start, k_count in chunks)

    for col, prototype in enumerate(dma_prototypes):
        result.dma_streams.append(
            KernelDmaStream(
                transfer=prototype,
                multiplicity=num_row_blocks * total_k_tiles,
                axes=(
                    RepeatAxis.from_mapping(
                        "output_row_tile",
                        num_row_blocks,
                        {},
                    ),
                    RepeatAxis.from_mapping(
                        "k_tile",
                        total_k_tiles,
                        {
                            "element_base_bytes": mlen * hbm_cols,
                            "scale_base_bytes": mlen * hbm_cols // 8,
                        },
                    ),
                ),
            )
        )

    for chunk_index, (k_start, k_count) in enumerate(chunks):
        if include_hbm_base_setup:
            _load_large(result, hbm_base, calls_per_chunk)
        result.add("C_SET_ADDR_REG", calls_per_chunk)
        _direct_addi(
            result,
            hbm_rows * hbm_cols,
            source_is_zero=True,
            count=calls_per_chunk,
        )
        result.add("C_SET_SCALE_REG", calls_per_chunk)
        _direct_addi(
            result,
            hbm_cols,
            source_is_zero=True,
            count=calls_per_chunk,
        )
        result.add("C_SET_STRIDE_REG", calls_per_chunk)

        for col in range(num_col_blocks):
            offsets = hbm_offsets[col][k_start : k_start + k_count]
            for local_k, offset in enumerate(offsets):
                _direct_addi(
                    result,
                    local_k * mlen * mlen,
                    source_is_zero=True,
                    count=num_row_blocks,
                )
                _direct_addi(
                    result,
                    offset,
                    source_is_zero=True,
                    count=num_row_blocks,
                )
            result.add("H_PREFETCH_M", num_row_blocks * k_count)

        result.add("S_ADDI_INT", calls_per_chunk)
        if chunk_index == 0:
            for col in range(num_col_blocks):
                _load_large_sequence(
                    result,
                    start=output_base + col * output_physical_rows * mlen,
                    step=mlen * mlen,
                    count=num_row_blocks,
                )
        else:
            if temp_base is None:
                raise ValueError("K-split projection requires temp_base")
            _load_large(result, temp_base, calls_per_chunk)

        _load_large_sequence(
            result,
            start=input_base + k_start * input_physical_rows * mlen,
            step=mlen * mlen,
            count=num_row_blocks,
            multiplier=num_col_blocks * tiles_per_mlen,
        )

        for row_loop_count in set(row_loop_counts):
            calls = row_loop_counts.count(row_loop_count) * num_col_blocks
            outer_middle = calls * tiles_per_mlen * row_loop_count
            result.add("C_LOOP_START", calls)
            result.add("S_ADDI_INT", calls * tiles_per_mlen)
            result.add("C_LOOP_START", calls * tiles_per_mlen)
            result.add("S_ADDI_INT", 2 * outer_middle)
            result.add("C_LOOP_START", outer_middle)
            result.add("M_MM", outer_middle * k_count)
            _direct_addi(
                result,
                full_batch * mlen,
                source_is_zero=False,
                count=outer_middle * k_count,
            )
            _direct_addi(
                result,
                mlen * mlen,
                source_is_zero=False,
                count=outer_middle * k_count,
            )
            result.add("C_LOOP_END", outer_middle * k_count)
            result.add("M_MM_WO", outer_middle)
            _direct_addi(
                result,
                blen * mlen,
                source_is_zero=False,
                count=2 * outer_middle,
            )
            result.add("C_LOOP_END", outer_middle)
            _direct_addi(
                result,
                blen,
                source_is_zero=False,
                count=2 * calls * tiles_per_mlen,
            )
            result.add("C_LOOP_END", calls * tiles_per_mlen)

        if chunk_index:
            assert temp_base is not None
            for col in range(num_col_blocks):
                _load_large_sequence(
                    result,
                    start=output_base + col * output_physical_rows * mlen,
                    step=mlen * mlen,
                    count=num_row_blocks,
                    multiplier=2,
                )
            _load_large(result, temp_base, calls_per_chunk)
            result.add("C_LOOP_START", calls_per_chunk)
            result.add("V_ADD_VV", calls_per_chunk * mlen)
            result.add("S_ADDI_INT", 3 * calls_per_chunk * mlen)
            result.add("C_LOOP_END", calls_per_chunk * mlen)
    return result


def projection_hbm_base_cost_summary(
    base_loads: list[tuple[int, int]],
) -> KernelCostSummary:
    """Count absolute HBM-base materialization for projection calls."""
    result = KernelCostSummary()
    for base, count in base_loads:
        if count < 0:
            raise ValueError("projection HBM-base load count must be nonnegative")
        _load_large(result, base, count)
    return result


def true_zero_rows_cost_summary(
    *,
    mlen: int,
    matrix_base: int,
    matrix_physical_rows: int,
    hidden: int,
    fp_zero_base: int,
    row_start: int,
    row_step: int,
    row_count: int,
) -> KernelCostSummary:
    """Census routed-MoE true-zero row initialization."""
    result = KernelCostSummary()
    _load_large(result, fp_zero_base)
    result.add("C_LOOP_START")
    result.add("S_ST_FP", mlen)
    result.add("S_ADDI_INT", mlen)
    result.add("C_LOOP_END", mlen)
    _load_large(result, fp_zero_base)
    for col in range(hidden // mlen):
        _load_large_sequence(
            result,
            start=(
                matrix_base
                + col * matrix_physical_rows * mlen
                + row_start * mlen
            ),
            step=row_step * mlen,
            count=row_count,
        )
    result.add("S_MAP_V_FP", row_count * (hidden // mlen))
    return result


def _load_modulo_sequence(
    result: KernelCostSummary,
    *,
    base: int,
    step: int,
    start_index: int,
    count: int,
    modulus: int,
) -> None:
    remaining = count
    index = start_index % modulus
    first = min(remaining, modulus - index)
    _load_large_sequence(
        result,
        start=base + index * step,
        step=step,
        count=first,
    )
    remaining -= first
    if not remaining:
        return
    full, tail = divmod(remaining, modulus)
    if full:
        _load_large_sequence(
            result,
            start=base,
            step=step,
            count=modulus,
            multiplier=full,
        )
    if tail:
        _load_large_sequence(
            result,
            start=base,
            step=step,
            count=tail,
        )


def routed_row_copy_cost_summary(
    *,
    opcode: str,
    mlen: int,
    hidden: int,
    route_start: int,
    route_count: int,
    token_count: int,
    slot_rows: int,
    dst_base: int,
    dst_physical_rows: int,
    src_base: int,
    src_physical_rows: int,
    token_is_destination: bool,
) -> KernelCostSummary:
    """Census affine gather/scatter row copies without route objects."""
    result = KernelCostSummary()
    blocks = hidden // mlen
    for col in range(blocks):
        dst_col = dst_base + col * dst_physical_rows * mlen
        src_col = src_base + col * src_physical_rows * mlen
        if token_is_destination:
            _load_modulo_sequence(
                result,
                base=dst_col,
                step=mlen,
                start_index=route_start,
                count=route_count,
                modulus=token_count,
            )
            _load_large_sequence(
                result,
                start=src_col,
                step=slot_rows * mlen,
                count=route_count,
            )
        else:
            _load_large_sequence(
                result,
                start=dst_col,
                step=slot_rows * mlen,
                count=route_count,
            )
            _load_modulo_sequence(
                result,
                base=src_col,
                step=mlen,
                start_index=route_start,
                count=route_count,
                modulus=token_count,
            )
    result.add(opcode, route_count * blocks)
    return result


def bf16_stream_k_accum_cost_summary(
    *,
    mlen: int,
    blen: int,
    hbm_base: int,
    hbm_cols: int,
    full_batch: int,
    output_physical_rows: int,
    input_vram_bases: list[list[int]],
    output_tile_bases: list[list[int]],
    row_loop_counts: list[int],
    chunks: list[tuple[int, int]],
    hbm_offsets: list[list[int]],
    dma_prototypes: list[DmaTransfer],
    hbm_element_bytes: int = 2,
) -> KernelCostSummary:
    """Census main's BF16 microtile cross-K accumulator projection.

    The production lowering reloads each K chunk for every output microtile,
    keeps the Matrix accumulator live across chunks, and writes once.  This
    adapter follows those exact addresses without mutating MRAM allocator state.
    """
    if mlen <= 0 or blen <= 0 or mlen % blen:
        raise ValueError("MLEN and BLEN must be positive with BLEN dividing MLEN")
    if len(input_vram_bases) != len(row_loop_counts):
        raise ValueError("input row blocks and row-loop counts differ")
    if len(output_tile_bases) != len(row_loop_counts):
        raise ValueError("output row blocks and row-loop counts differ")
    if len(hbm_offsets) != len(dma_prototypes):
        raise ValueError("HBM offsets and DMA prototypes differ")

    result = KernelCostSummary()
    tiles_per_mlen = mlen // blen
    matrix_block_size = mlen * mlen
    output_row_stride = blen * mlen
    matrix_col_stride = blen * mlen
    microtiles_per_col = tiles_per_mlen * sum(row_loop_counts)

    micro_rows = sum(row_loop_counts)
    all_microtiles = len(hbm_offsets) * tiles_per_mlen * micro_rows
    setup_calls = all_microtiles * len(chunks)
    total_k_tiles = len(hbm_offsets[0]) if hbm_offsets else 0
    if any(len(offsets) != total_k_tiles for offsets in hbm_offsets):
        raise ValueError("ragged HBM K offsets")
    input_base = input_vram_bases[0][0]
    output_base = output_tile_bases[0][0]

    # Every microtile/chunk reload repeats the same address-register and stride
    # setup. Count those calls directly instead of walking the microtile grid.
    _load_large(result, hbm_base, setup_calls)
    result.add("C_SET_ADDR_REG", setup_calls)
    _load_large(result, hbm_cols * hbm_element_bytes, setup_calls)
    result.add("C_SET_STRIDE_REG", setup_calls)

    for col_idx, (col_offsets, prototype) in enumerate(
        zip(hbm_offsets, dma_prototypes, strict=True)
    ):
        result.dma_streams.append(
            KernelDmaStream(
                transfer=prototype,
                multiplicity=microtiles_per_col * len(col_offsets),
                axes=(
                    RepeatAxis.from_mapping(
                        "router_microtile_reload",
                        microtiles_per_col,
                        {},
                    ),
                    RepeatAxis.from_mapping(
                        "k_tile",
                        len(col_offsets),
                        {"element_base_bytes": mlen * hbm_cols * hbm_element_bytes},
                    ),
                ),
            )
        )
        chunk_microtiles = tiles_per_mlen * micro_rows
        for k_start, k_count in chunks:
            if k_count <= 0 or k_start < 0 or k_start + k_count > len(col_offsets):
                raise ValueError(f"invalid K chunk {(k_start, k_count)}")
            for local_k, k_idx in enumerate(range(k_start, k_start + k_count)):
                _load_large(result, local_k * matrix_block_size, chunk_microtiles)
                _load_large(
                    result,
                    col_offsets[k_idx] * hbm_element_bytes,
                    chunk_microtiles,
                )

    result.add("H_PREFETCH_M", all_microtiles * total_k_tiles)

    # Activation addresses form one contiguous BLEN-row progression for each K
    # tile. The same progression is reused by every output column/micro-column.
    for k_idx in range(total_k_tiles):
        _load_large_sequence(
            result,
            start=input_base + k_idx * full_batch * mlen,
            step=output_row_stride,
            count=micro_rows,
            multiplier=len(hbm_offsets) * tiles_per_mlen,
        )

    # MRAM addresses repeat for every row and output column but depend only on
    # the local K position inside a chunk and the micro-column.
    for _k_start, k_count in chunks:
        for micro_col_idx in range(tiles_per_mlen):
            for local_k in range(k_count):
                _load_large(
                    result,
                    micro_col_idx * matrix_col_stride + local_k * matrix_block_size,
                    micro_rows * len(hbm_offsets),
                )

    result.add("M_MM", all_microtiles * total_k_tiles)

    # Each output microtile is written once after the final K chunk.
    for col_idx in range(len(hbm_offsets)):
        for micro_col_idx in range(tiles_per_mlen):
            _load_large_sequence(
                result,
                start=(
                    output_base
                    + col_idx * output_physical_rows * mlen
                    + micro_col_idx * blen
                ),
                step=output_row_stride,
                count=micro_rows,
            )
    result.add("M_MM_WO", all_microtiles)
    return result


def standard_swiglu_cost_summary(
    *,
    mlen: int,
    sigmoid_base: int,
    sigmoid_physical_rows: int,
    intermediate: int,
    rows: int,
    one_fp_base: int,
    neg_one_fp_base: int,
) -> KernelCostSummary:
    """Census main's contiguous-row standard SwiGLU tile operations."""
    if rows <= 0 or intermediate <= 0 or intermediate % mlen:
        raise ValueError("SwiGLU rows must be positive and width MLEN-aligned")
    result = KernelCostSummary()
    for col in range(intermediate // mlen):
        base = sigmoid_base + col * sigmoid_physical_rows * mlen

        # vram_fill_zero(sigmoid)
        _load_large(result, base)
        result.add("C_LOOP_START")
        result.add("V_MUL_VF", rows)
        _direct_addi(result, mlen, source_is_zero=False, count=rows)
        result.add("C_LOOP_END", rows)

        # Multiply by -1 and add 1 use matching contiguous FPRAM rows.
        for opcode, fp_base in (("V_MUL_VF", neg_one_fp_base), ("V_ADD_VF", one_fp_base)):
            _load_large(result, base)
            _load_large(result, fp_base)
            result.add("C_LOOP_START")
            result.add("S_LD_FP", rows)
            result.add(opcode, rows)
            _direct_addi(result, mlen, source_is_zero=False, count=rows)
            _direct_addi(result, 1, source_is_zero=False, count=rows)
            result.add("C_LOOP_END", rows)

        for opcode in ("V_EXP_V", "V_RECI_V"):
            _load_large(result, base)
            result.add("C_LOOP_START")
            result.add(opcode, rows)
            _direct_addi(result, mlen, source_is_zero=False, count=rows)
            result.add("C_LOOP_END", rows)
    return result


__all__ = [
    "KernelCostSummary",
    "KernelDmaStream",
    "ffn_unrolled_cost_summary",
    "router_topk_cost_summary",
    "vram_matrix_binary_cost_summary",
    "normalization_cost_summary",
    "linear_projection_cost_summary",
    "projection_hbm_base_cost_summary",
    "true_zero_rows_cost_summary",
    "routed_row_copy_cost_summary",
    "bf16_stream_k_accum_cost_summary",
    "standard_swiglu_cost_summary",
]
