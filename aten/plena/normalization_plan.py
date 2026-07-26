"""Structured native normalization lowerings shared by ASM and CostEmitter.

The legacy normalization templates render assembly and derive cost counts in a
separate module.  Native decoder optimization needs exact parity between the
two paths, so this module builds one compact instruction plan and derives the
rendered program and both opcode histograms from that plan.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import math
from typing import Iterable

from compiler.asm_templates._imm import load_large_int
from compiler.aten.cost_emitter import (
    ScheduleInstruction,
    ScheduleRepeat,
    ScheduleSequence,
)


@dataclass(frozen=True)
class StructuredNormalizationLowering:
    """One normalization program represented in all required output forms."""

    rendered_asm: str
    static_opcodes: Counter[str]
    dynamic_opcodes: Counter[str]
    schedule: ScheduleSequence
    metadata: dict[str, int]


class _PlanBuilder:
    """Build rendered instructions and a compressed ordered schedule together."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.children: list[ScheduleInstruction | ScheduleRepeat] = []

    def comment(self, text: str) -> None:
        self.lines.append(f"; {text}")

    def instruction(self, opcode: str, *args: object) -> None:
        rendered_args = tuple(str(arg) for arg in args)
        suffix = "" if not rendered_args else " " + ", ".join(rendered_args)
        self.lines.append(f"{opcode}{suffix}")
        self.children.append(ScheduleInstruction(opcode, rendered_args))

    def raw_instructions(self, lines: Iterable[str]) -> None:
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            opcode, _, operands = stripped.partition(" ")
            args = tuple(
                operand.strip() for operand in operands.split(",") if operand.strip()
            )
            self.lines.append(stripped)
            self.children.append(ScheduleInstruction(opcode, args))

    def repeat(
        self,
        count: int,
        body: "_PlanBuilder",
        *,
        name: str,
        repeat_kind: str = "hardware_loop",
    ) -> None:
        if count <= 0:
            raise ValueError(f"repeat count must be positive, got {count}")
        self.lines.extend(body.lines)
        self.children.append(
            ScheduleRepeat(
                count=count,
                body=ScheduleSequence(tuple(body.children)),
                name=name,
                repeat_kind=repeat_kind,
            )
        )

    def finish(self, *, metadata: dict[str, int]) -> StructuredNormalizationLowering:
        static = Counter(
            line.split(None, 1)[0]
            for line in self.lines
            if line and not line.lstrip().startswith(";")
        )

        def dynamic_counts(
            node: ScheduleInstruction | ScheduleRepeat | ScheduleSequence,
            multiplier: int = 1,
        ) -> Counter[str]:
            if isinstance(node, ScheduleInstruction):
                return Counter({node.opcode: multiplier})
            if isinstance(node, ScheduleRepeat):
                return dynamic_counts(node.body, multiplier * node.count)
            result: Counter[str] = Counter()
            for child in node.children:
                result.update(dynamic_counts(child, multiplier))
            return result

        schedule = ScheduleSequence(tuple(self.children))
        return StructuredNormalizationLowering(
            rendered_asm="\n".join(self.lines) + "\n",
            static_opcodes=+static,
            dynamic_opcodes=+dynamic_counts(schedule),
            schedule=schedule,
            metadata=dict(metadata),
        )


def _validate_active_row_ranges(
    physical_rows: int,
    active_row_ranges: Iterable[tuple[int, int]] | None,
) -> tuple[tuple[int, int], ...]:
    if physical_rows <= 0:
        raise ValueError(f"physical_rows must be positive, got {physical_rows}")
    ranges = (
        ((0, physical_rows),)
        if active_row_ranges is None
        else tuple((int(start), int(end)) for start, end in active_row_ranges)
    )
    previous_end = 0
    for start, end in ranges:
        if start < previous_end or start < 0 or end <= start or end > physical_rows:
            raise ValueError(
                "active row ranges must be sorted, disjoint, non-empty, and fit "
                f"[0, {physical_rows}); got {ranges}"
            )
        previous_end = end
    return ranges


def build_grouped_segmented_rms_norm(
    *,
    name: str,
    tensor_base_address: int,
    scratch_base_address: int,
    physical_rows: int,
    physical_cols: int,
    mlen: int,
    hlen: int,
    segments: Iterable[tuple[int, int]],
    active_row_ranges: Iterable[tuple[int, int]] | None,
    gp_src: int,
    gp_scratch: int,
    gp_mask: int,
    gp_loop: int,
    gp_stats: int | None = None,
    inverse_head_dim_slot: int = 6,
    epsilon_slot: int = 3,
    rtl_v2: bool = False,
    rtl_v3: bool = False,
    rtl_v4: bool = False,
) -> StructuredNormalizationLowering:
    """Normalize packed head lanes while sharing vector preparation work.

    The scratch clear/source copy are performed once per ``(column block,
    row)``.  Each selected lane executes the same masked reduction and scalar
    normalization sequence as the legacy lowering.  A direct square refresh
    after each lane preserves the current masked-reduction semantics without
    repeating the legacy clear and copy.
    """

    if mlen <= 0 or hlen <= 0 or mlen % hlen:
        raise ValueError(f"HLEN must divide MLEN, got HLEN={hlen}, MLEN={mlen}")
    if (rtl_v2 or rtl_v3 or rtl_v4) and hlen & (hlen - 1):
        raise ValueError(f"RTL segment reduction requires power-of-two HLEN, got {hlen}")
    segment_width_log2 = int(math.log2(hlen))
    segments_per_block = mlen // hlen
    requested_rtl_v3 = bool(rtl_v3)
    requested_rtl_v4 = bool(rtl_v4)
    segment_parallel_fallback = bool(
        (rtl_v3 or rtl_v4) and segments_per_block > 16
    )
    if segment_parallel_fallback:
        # The compact-stat datapath has exactly 16 output lanes.  Wider vector
        # words remain legal by using the existing per-segment reduction path;
        # silently treating all 32+ segments as compact would fabricate RTL
        # capability and corrupt the upper lanes.
        rtl_v2 = True
        rtl_v3 = False
        rtl_v4 = False
    if rtl_v3 or rtl_v4:
        if gp_stats is None:
            raise ValueError(
                "rtl-v3/rtl-v4 grouped RMSNorm requires a compact-stats GP register"
            )
    if physical_cols <= 0 or physical_cols % mlen:
        raise ValueError(
            f"physical_cols must be a positive MLEN multiple, got {physical_cols}"
        )
    ranges = _validate_active_row_ranges(physical_rows, active_row_ranges)
    grouped: OrderedDict[int, list[int]] = OrderedDict()
    for block, lane in segments:
        block = int(block)
        lane = int(lane)
        if block < 0 or block >= physical_cols // mlen:
            raise ValueError(f"Q/K norm column block {block} out of range")
        if lane < 0 or lane >= mlen // hlen:
            raise ValueError(f"Q/K norm lane {lane} out of range")
        lanes = grouped.setdefault(block, [])
        if lane not in lanes:
            lanes.append(lane)
    if not grouped:
        return _PlanBuilder().finish(metadata={})

    builder = _PlanBuilder()
    builder.comment(f"=== Grouped segmented Q/K RMSNorm: {name} ===")
    builder.raw_instructions(load_large_int(gp_scratch, scratch_base_address))
    if rtl_v3 or rtl_v4:
        builder.raw_instructions(load_large_int(gp_stats, scratch_base_address + mlen))
    builder.instruction("S_LD_FP", "f2", "gp0", inverse_head_dim_slot)
    builder.instruction("S_LD_FP", "f3", "gp0", epsilon_slot)

    active_rows = sum(end - start for start, end in ranges)
    for block, lanes in grouped.items():
        block_base = tensor_base_address + block * physical_rows * mlen
        if rtl_v3 or rtl_v4:
            block_mask = sum(1 << lane for lane in lanes)
            builder.instruction("S_ADDI_INT", f"gp{gp_mask}", "gp0", block_mask)
            builder.instruction("C_SET_V_MASK_REG", f"gp{gp_mask}")
        for range_idx, (start, end) in enumerate(ranges):
            builder.raw_instructions(
                load_large_int(gp_src, block_base + start * mlen)
            )
            builder.instruction("C_LOOP_START", f"gp{gp_loop}", end - start)
            body = _PlanBuilder()
            body.instruction(
                "V_MUL_VF", f"gp{gp_scratch}", f"gp{gp_scratch}", "f0", 0
            )
            body.instruction(
                "V_ADD_VV",
                f"gp{gp_scratch}",
                f"gp{gp_scratch}",
                f"gp{gp_src}",
                0,
            )
            body.instruction(
                "V_MUL_VV",
                f"gp{gp_scratch}",
                f"gp{gp_scratch}",
                f"gp{gp_scratch}",
                0,
            )
            if rtl_v3 or rtl_v4:
                body.instruction(
                    "V_RED_SUM_SEGS",
                    f"gp{gp_stats}",
                    f"gp{gp_scratch}",
                    segment_width_log2,
                )
                if rtl_v4:
                    if lanes != list(range(len(lanes))):
                        raise ValueError(
                            "rtl-v4 compact statistics require contiguous lanes "
                            f"starting at zero, got {lanes}"
                        )
                    body.instruction(
                        "V_STAT_MUL_F",
                        f"gp{gp_stats}",
                        f"gp{gp_stats}",
                        "f2",
                        len(lanes),
                    )
                    body.instruction(
                        "V_STAT_ADD_F",
                        f"gp{gp_stats}",
                        f"gp{gp_stats}",
                        "f3",
                        len(lanes),
                    )
                    body.instruction(
                        "V_STAT_RSQRT",
                        f"gp{gp_stats}",
                        f"gp{gp_stats}",
                        "f0",
                        len(lanes),
                    )
                else:
                    # Eight rotating FP registers expose independent scalar
                    # chains to the in-order ROB.
                    for lane_batch_start in range(0, len(lanes), 8):
                        lane_batch = lanes[lane_batch_start : lane_batch_start + 8]
                        fp_regs = [4 + idx for idx in range(len(lane_batch))]
                        for lane, fp_reg in zip(lane_batch, fp_regs, strict=True):
                            body.instruction("S_ADDI_INT", f"gp{gp_mask}", "gp0", lane)
                            body.instruction(
                                "S_LD_VLANE_FP",
                                f"f{fp_reg}",
                                f"gp{gp_stats}",
                                f"gp{gp_mask}",
                            )
                        for fp_reg in fp_regs:
                            body.instruction("S_MUL_FP", f"f{fp_reg}", f"f{fp_reg}", "f2")
                        for fp_reg in fp_regs:
                            body.instruction("S_ADD_FP", f"f{fp_reg}", f"f{fp_reg}", "f3")
                        for fp_reg in fp_regs:
                            body.instruction("S_RSQRT_FP", f"f{fp_reg}", f"f{fp_reg}")
                        for lane, fp_reg in zip(lane_batch, fp_regs, strict=True):
                            body.instruction("S_ADDI_INT", f"gp{gp_mask}", "gp0", lane)
                            body.instruction(
                                "S_ST_VLANE_FP",
                                f"f{fp_reg}",
                                f"gp{gp_stats}",
                                f"gp{gp_mask}",
                            )
                body.instruction(
                    "V_MUL_VSEG",
                    f"gp{gp_src}",
                    f"gp{gp_src}",
                    f"gp{gp_stats}",
                    segment_width_log2,
                    1,
                )
            for lane_idx, lane in enumerate(() if (rtl_v3 or rtl_v4) else lanes):
                mask = 1 << lane
                if rtl_v2:
                    body.instruction("S_ADDI_INT", f"gp{gp_mask}", "gp0", lane)
                    body.instruction("S_MV_FP", "f1", "f0")
                    body.instruction(
                        "V_RED_SUM_SEG",
                        "f1",
                        f"gp{gp_scratch}",
                        f"gp{gp_mask}",
                        int(math.log2(hlen)),
                    )
                else:
                    body.instruction("S_ADD_FP", "f1", "f0", "f0")
                    body.instruction(
                        "V_RED_SUM", "f1", f"gp{gp_scratch}", 1, 0
                    )
                body.instruction("S_MUL_FP", "f1", "f1", "f2")
                body.instruction("S_ADD_FP", "f1", "f1", "f3")
                if rtl_v2:
                    body.instruction("S_RSQRT_FP", "f1", "f1")
                else:
                    body.instruction("S_SQRT_FP", "f1", "f1", 0)
                    body.instruction("S_RECI_FP", "f1", "f1", 0)
                body.instruction("S_ADDI_INT", f"gp{gp_mask}", "gp0", mask)
                body.instruction("C_SET_V_MASK_REG", f"gp{gp_mask}")
                body.instruction(
                    "V_MUL_VF", f"gp{gp_src}", f"gp{gp_src}", "f1", 1
                )
                if not rtl_v2 and lane_idx + 1 < len(lanes):
                    # The current emulator/RTL-facing masked reduction carries
                    # the unselected lanes into its accumulator.  Legacy code
                    # therefore observes squares recomputed after every prior
                    # lane normalization.  Refreshing the square directly is
                    # bitwise-equivalent to legacy zero+copy+square for normal
                    # finite SRAM values, while eliminating two vector ops.
                    body.instruction(
                        "V_MUL_VV",
                        f"gp{gp_scratch}",
                        f"gp{gp_src}",
                        f"gp{gp_src}",
                        0,
                    )
            body.instruction("S_ADDI_INT", f"gp{gp_src}", f"gp{gp_src}", mlen)
            body.instruction("C_LOOP_END", f"gp{gp_loop}")
            builder.repeat(
                end - start,
                body,
                name=f"{name}_block{block}_range{range_idx}_rows",
            )

    segment_count = sum(len(lanes) for lanes in grouped.values())
    legacy_preparation_ops = physical_rows * segment_count
    grouped_copy_ops = active_rows * len(grouped)
    grouped_square_ops = active_rows * segment_count
    legacy_constant_loads = 2 * physical_rows * segment_count
    metadata = {
        "segmented_norm_square_ops_elided": max(
            0, legacy_preparation_ops - grouped_square_ops
        ),
        "segmented_norm_copy_ops_elided": max(
            0, legacy_preparation_ops - grouped_copy_ops
        ),
        "segmented_norm_constant_loads_elided": max(
            0, legacy_constant_loads - 2
        ),
        "inactive_norm_rows_elided": max(
            0, (physical_rows - active_rows) * segment_count
        ),
    }
    if rtl_v2:
        metadata.update(
            {
                "segment_reductions_emitted": active_rows * segment_count,
                "segment_reduction_levels_elided": active_rows
                * segment_count
                * max(
                    0,
                    math.ceil(math.log2(mlen + 1))
                    - (int(math.log2(hlen)) + 1),
                ),
                "scalar_moves_emitted": active_rows * segment_count,
                "scalar_rsqrt_emitted": active_rows * segment_count,
            }
        )
    if rtl_v3 or rtl_v4:
        metadata.update(
            {
                "multi_segment_reductions_emitted": active_rows * len(grouped),
                "single_segment_reductions_elided": active_rows
                * (segment_count - len(grouped)),
                "compact_stats_lane_loads": (
                    0 if rtl_v4 else active_rows * segment_count
                ),
                "compact_stats_lane_stores": (
                    0 if rtl_v4 else active_rows * segment_count
                ),
                "compact_stats_lane_loads_before": active_rows * segment_count,
                "compact_stats_lane_stores_before": active_rows * segment_count,
                "compact_lane_selectors_before": 2 * active_rows * segment_count,
                "compact_lane_selectors_remaining": (
                    0 if rtl_v4 else 2 * active_rows * segment_count
                ),
                "segment_broadcast_ops": active_rows * len(grouped),
                "scalar_modulo_schedule_width": 0 if rtl_v4 else 8,
                "compact_stat_simd_ops": (
                    3 * active_rows * len(grouped) if rtl_v4 else 0
                ),
                "compact_scalar_chain_ops_elided": (
                    5 * active_rows * segment_count if rtl_v4 else 0
                ),
                "compact_lane_selectors_elided": (
                    2 * active_rows * segment_count if rtl_v4 else 0
                ),
            }
        )
    if segment_parallel_fallback:
        metadata.update(
            {
                "segment_parallel_fallback_blocks": len(grouped),
                "segment_parallel_fallback_segments": segment_count,
                "segment_parallel_requested_rtl_v3": int(requested_rtl_v3),
                "segment_parallel_requested_rtl_v4": int(requested_rtl_v4),
            }
        )
    return builder.finish(metadata=metadata)


def build_split_head_segmented_rms_norm(
    *,
    name: str,
    tensor_base_addresses: Iterable[int],
    scratch_base_address: int,
    physical_rows: int,
    mlen: int,
    hlen: int,
    active_row_ranges: Iterable[tuple[int, int]] | None,
    gp_heads: Iterable[int],
    gp_packed: int,
    gp_shifted: int,
    gp_stats: int,
    gp_index: int,
    gp_loop: int,
    inverse_head_dim_slot: int = 6,
    epsilon_slot: int = 3,
    rtl_v4: bool = False,
) -> StructuredNormalizationLowering:
    """Normalize separate head tensors with one segmented reduction per row.

    K heads are projected into independent MLEN-wide tensors because the
    existing attention/HBM path consumes one tensor per KV head.  Re-running a
    full reduction for every such tensor defeats the segment-parallel RTL.
    This lowering temporarily packs the active HLEN lanes into one vector word,
    squares that packed word once, and obtains every head statistic with one
    ``V_RED_SUM_SEGS``.  The resulting scalar factors are then applied to the
    original tensors, preserving their established storage and DMA layout.

    The scratch allocation contains three consecutive vector words:

    ``packed values | shifted-head temporary | compact statistics``.
    """

    addresses = tuple(int(address) for address in tensor_base_addresses)
    head_regs = tuple(int(reg) for reg in gp_heads)
    if not addresses or len(addresses) != len(head_regs):
        raise ValueError(
            "split-head RMSNorm requires one GP register per non-empty tensor list"
        )
    if mlen <= 0 or hlen <= 0 or mlen % hlen:
        raise ValueError(f"HLEN must divide MLEN, got HLEN={hlen}, MLEN={mlen}")
    if hlen & (hlen - 1):
        raise ValueError(f"RTL segment reduction requires power-of-two HLEN, got {hlen}")
    if len(addresses) > min(16, mlen // hlen):
        raise ValueError(
            "split-head RMSNorm cannot pack all heads into one vector word: "
            f"heads={len(addresses)}, capacity={min(16, mlen // hlen)}"
        )

    ranges = _validate_active_row_ranges(physical_rows, active_row_ranges)
    packed_address = int(scratch_base_address)
    shifted_address = packed_address + mlen
    stats_address = shifted_address + mlen
    segment_width_log2 = int(math.log2(hlen))

    builder = _PlanBuilder()
    builder.comment(f"=== Split-head segmented K RMSNorm: {name} ===")
    builder.raw_instructions(load_large_int(gp_packed, packed_address))
    builder.raw_instructions(load_large_int(gp_shifted, shifted_address))
    builder.raw_instructions(load_large_int(gp_stats, stats_address))
    builder.instruction("S_LD_FP", "f2", "gp0", inverse_head_dim_slot)
    builder.instruction("S_LD_FP", "f3", "gp0", epsilon_slot)

    for range_idx, (start, end) in enumerate(ranges):
        for address, gp_head in zip(addresses, head_regs, strict=True):
            builder.raw_instructions(load_large_int(gp_head, address + start * mlen))
        builder.instruction("C_LOOP_START", f"gp{gp_loop}", end - start)
        body = _PlanBuilder()
        body.instruction(
            "V_MUL_VF", f"gp{gp_packed}", f"gp{gp_packed}", "f0", 0
        )
        for lane, gp_head in enumerate(head_regs):
            body.instruction("S_ADDI_INT", f"gp{gp_index}", "gp0", lane * hlen)
            body.instruction(
                "V_SHIFT_V", f"gp{gp_shifted}", f"gp{gp_head}", f"gp{gp_index}"
            )
            body.instruction(
                "V_ADD_VV",
                f"gp{gp_packed}",
                f"gp{gp_packed}",
                f"gp{gp_shifted}",
                0,
            )
        body.instruction(
            "V_MUL_VV",
            f"gp{gp_shifted}",
            f"gp{gp_packed}",
            f"gp{gp_packed}",
            0,
        )
        body.instruction(
            "V_RED_SUM_SEGS",
            f"gp{gp_stats}",
            f"gp{gp_shifted}",
            segment_width_log2,
        )

        if rtl_v4:
            body.instruction(
                "V_STAT_MUL_F",
                f"gp{gp_stats}",
                f"gp{gp_stats}",
                "f2",
                len(head_regs),
            )
            body.instruction(
                "V_STAT_ADD_F",
                f"gp{gp_stats}",
                f"gp{gp_stats}",
                "f3",
                len(head_regs),
            )
            body.instruction(
                "V_STAT_RSQRT",
                f"gp{gp_stats}",
                f"gp{gp_stats}",
                "f0",
                len(head_regs),
            )
            for lane, gp_head in enumerate(head_regs):
                body.instruction("S_ADDI_INT", f"gp{gp_index}", "gp0", lane)
                body.instruction(
                    "S_LD_VLANE_FP", "f4", f"gp{gp_stats}", f"gp{gp_index}"
                )
                body.instruction("V_MUL_VF", f"gp{gp_head}", f"gp{gp_head}", "f4", 0)
        else:
            # At most eight independent scalar chains are live at once.
            for lane_batch_start in range(0, len(head_regs), 8):
                batch_regs = head_regs[lane_batch_start : lane_batch_start + 8]
                fp_regs = tuple(4 + offset for offset in range(len(batch_regs)))
                for offset, fp_reg in enumerate(fp_regs):
                    lane = lane_batch_start + offset
                    body.instruction("S_ADDI_INT", f"gp{gp_index}", "gp0", lane)
                    body.instruction(
                        "S_LD_VLANE_FP",
                        f"f{fp_reg}",
                        f"gp{gp_stats}",
                        f"gp{gp_index}",
                    )
                for fp_reg in fp_regs:
                    body.instruction("S_MUL_FP", f"f{fp_reg}", f"f{fp_reg}", "f2")
                for fp_reg in fp_regs:
                    body.instruction("S_ADD_FP", f"f{fp_reg}", f"f{fp_reg}", "f3")
                for fp_reg in fp_regs:
                    body.instruction("S_RSQRT_FP", f"f{fp_reg}", f"f{fp_reg}")
                for offset, fp_reg in enumerate(fp_regs):
                    lane = lane_batch_start + offset
                    body.instruction("S_ADDI_INT", f"gp{gp_index}", "gp0", lane)
                    body.instruction(
                        "S_ST_VLANE_FP",
                        f"f{fp_reg}",
                        f"gp{gp_stats}",
                        f"gp{gp_index}",
                    )
                for gp_head, fp_reg in zip(batch_regs, fp_regs, strict=True):
                    body.instruction(
                        "V_MUL_VF", f"gp{gp_head}", f"gp{gp_head}", f"f{fp_reg}", 0
                    )

        for gp_head in head_regs:
            body.instruction("S_ADDI_INT", f"gp{gp_head}", f"gp{gp_head}", mlen)
        body.instruction("C_LOOP_END", f"gp{gp_loop}")
        builder.repeat(
            end - start,
            body,
            name=f"{name}_range{range_idx}_rows",
        )

    active_rows = sum(end - start for start, end in ranges)
    head_count = len(addresses)
    return builder.finish(
        metadata={
            "multi_segment_reductions_emitted": active_rows,
            "single_segment_reductions_elided": active_rows * (head_count - 1),
            "compact_stats_lane_loads": active_rows * head_count,
            "compact_stats_lane_stores": 0 if rtl_v4 else active_rows * head_count,
            "compact_stats_lane_loads_before": active_rows * head_count,
            "compact_stats_lane_stores_before": active_rows * head_count,
            "compact_lane_selectors_before": 2 * active_rows * head_count,
            "compact_lane_selectors_remaining": (
                active_rows * head_count if rtl_v4 else 2 * active_rows * head_count
            ),
            "compact_stat_simd_ops": 3 * active_rows if rtl_v4 else 0,
            "compact_scalar_chain_ops_elided": (
                4 * active_rows * head_count if rtl_v4 else 0
            ),
            "compact_lane_selectors_elided": (
                active_rows * head_count if rtl_v4 else 0
            ),
            "segment_broadcast_ops": 0,
            "split_head_vector_scale_ops": active_rows * head_count,
            "split_head_pack_ops": active_rows * head_count * 2,
            "segmented_norm_square_ops_elided": active_rows * (head_count - 1),
            "segmented_norm_copy_ops_elided": 0,
            "segmented_norm_constant_loads_elided": max(
                0, 2 * physical_rows * head_count - 2
            ),
            "inactive_norm_rows_elided": max(
                0, (physical_rows - active_rows) * head_count
            ),
            "scalar_modulo_schedule_width": min(8, head_count),
        }
    )


def build_active_row_rms_norm(
    *,
    name: str,
    activation_base_address: int,
    scratch_base_address: int,
    physical_rows: int,
    hidden_dim: int,
    vlen: int,
    active_row_ranges: Iterable[tuple[int, int]] | None,
    gp_row: int,
    gp_scratch: int,
    gp_stats: int,
    gp_act: int,
    gp_loop: int,
    gp_stride: int,
    epsilon_slot: int,
    reciprocal_hidden_slot: int,
    rtl_v2: bool = False,
) -> StructuredNormalizationLowering:
    """Build bitwise-preserving RMSNorm over only active physical rows."""

    if hidden_dim <= 0 or vlen <= 0 or hidden_dim % vlen:
        raise ValueError(
            f"hidden_dim={hidden_dim} must be a positive multiple of VLEN={vlen}"
        )
    ranges = _validate_active_row_ranges(physical_rows, active_row_ranges)
    chunks = hidden_dim // vlen
    column_stride = physical_rows * vlen
    active_rows = sum(end - start for start, end in ranges)

    builder = _PlanBuilder()
    builder.comment(f"=== Active-row RMSNorm: {name} ===")
    builder.raw_instructions(load_large_int(gp_scratch, scratch_base_address))
    builder.raw_instructions(load_large_int(gp_stride, column_stride))
    builder.instruction("S_LD_FP", "f1", "gp0", epsilon_slot)
    if rtl_v2:
        builder.instruction("S_MV_FP", "f2", "f0")
    else:
        builder.instruction("S_ADD_FP", "f2", "f0", "f0")
    builder.instruction("S_LD_FP", "f3", "gp0", reciprocal_hidden_slot)

    for range_idx, (start, end) in enumerate(ranges):
        builder.raw_instructions(
            load_large_int(gp_row, activation_base_address + start * vlen)
        )
        builder.instruction("C_LOOP_START", f"gp{gp_loop}", end - start)
        body = _PlanBuilder()
        body.instruction("S_ADDI_INT", f"gp{gp_stats}", f"gp{gp_row}", 0)
        for _ in range(chunks):
            body.instruction(
                "V_MUL_VV",
                f"gp{gp_scratch}",
                f"gp{gp_stats}",
                f"gp{gp_stats}",
                0,
            )
            body.instruction("V_RED_SUM", "f2", f"gp{gp_scratch}")
            body.instruction(
                "S_ADD_INT", f"gp{gp_stats}", f"gp{gp_stats}", f"gp{gp_stride}"
            )

        body.instruction("S_MUL_FP", "f2", "f2", "f3")
        body.instruction("S_ADD_FP", "f2", "f2", "f1")
        if rtl_v2:
            body.instruction("S_RSQRT_FP", "f2", "f2")
        else:
            body.instruction("S_SQRT_FP", "f2", "f2")
            body.instruction("S_RECI_FP", "f2", "f2")

        # These independent address operations occupy the four-cycle reciprocal
        # settle window used by the legacy template.  They prepare both the
        # current row's normalization pointers and the next active row.
        body.instruction("S_ADDI_INT", f"gp{gp_act}", f"gp{gp_row}", 0)
        if chunks > 1:
            body.instruction(
                "S_ADD_INT", f"gp{gp_stats}", f"gp{gp_row}", f"gp{gp_stride}"
            )
        else:
            body.instruction("S_ADDI_INT", "gp0", "gp0", 0)
        body.instruction("S_ADDI_INT", f"gp{gp_row}", f"gp{gp_row}", vlen)
        body.instruction("S_ADDI_INT", "gp0", "gp0", 0)

        address_registers = (gp_act, gp_stats)
        for chunk in range(chunks):
            current = address_registers[chunk % 2]
            body.instruction(
                "V_MUL_VF", f"gp{current}", f"gp{current}", "f2", 0
            )
            if chunk + 2 < chunks:
                body.instruction(
                    "S_ADD_INT",
                    f"gp{current}",
                    f"gp{current}",
                    f"gp{gp_stride}",
                )
                body.instruction(
                    "S_ADD_INT",
                    f"gp{current}",
                    f"gp{current}",
                    f"gp{gp_stride}",
                )
            elif chunk + 1 < chunks:
                body.instruction("S_ADDI_INT", "gp0", "gp0", 0)
        if rtl_v2:
            body.instruction("S_MV_FP", "f2", "f0")
        else:
            body.instruction("S_ADD_FP", "f2", "f0", "f0")
        body.instruction("C_LOOP_END", f"gp{gp_loop}")
        builder.repeat(
            end - start,
            body,
            name=f"{name}_range{range_idx}_rows",
        )

    # Legacy emits two absolute row-base loads and up to one absolute load per
    # additional hidden chunk for every physical row.  Compiler-v1 carries row
    # and stride addresses through loops and only materializes each range base.
    legacy_row_address_loads = physical_rows * (2 + max(0, chunks - 1))
    new_range_address_loads = len(ranges) + 2
    return builder.finish(
        metadata={
            "inactive_norm_rows_elided": physical_rows - active_rows,
            "rms_norm_address_loads_elided": max(
                0, legacy_row_address_loads - new_range_address_loads
            ),
            "rms_norm_nops_elided": 3 * active_rows,
            "scalar_moves_emitted": (active_rows + 1) if rtl_v2 else 0,
            "scalar_rsqrt_emitted": active_rows if rtl_v2 else 0,
        }
    )


__all__ = [
    "StructuredNormalizationLowering",
    "build_active_row_rms_norm",
    "build_grouped_segmented_rms_norm",
    "build_split_head_segmented_rms_norm",
]
