"""Symbolic instruction and DMA cost collection for the ATen compiler.

The cost path consumes the same nodes as the assembly renderer.  It never
materializes compile-time repeats or hardware-loop iterations, so a program
whose physical ISA contains tens of millions of instructions stays compact.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from collections.abc import Callable, Iterable, Mapping
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field, replace
from typing import Any, ClassVar

from compiler.aten.isa_builder import (
    AsmItem,
    Comment,
    CompileTimeRepeat,
    DmaTransfer,
    HardwareLoop,
    Instr,
    IsaBuilder,
    RepeatAxis,
    Sequence,
    Stage,
    as_sequence,
    legalize_large_immediates,
    render_arg,
    render_asm,
)
from compiler.asm_templates._imm import IMM2_BOUND


MATRIX_COMPUTE_OPS = {
    "M_MM",
    "M_TMM",
    "M_BMM",
    "M_BTMM",
    "M_MV",
    "M_TMV",
    "M_BMV",
    "M_BTMV",
    "M_BMM_WO",
    "M_MM_WO",
    "M_MM_WO_PACKED_ACC",
    "M_MV_WO",
    "M_BMV_WO",
}
VECTOR_COMPUTE_OPS = {
    "V_ADD_VV",
    "V_ADD_VF",
    "V_SUB_VV",
    "V_SUB_VF",
    "V_MUL_VV",
    "V_MUL_VF",
    "V_EXP_V",
    "V_RECI_V",
    "V_RED_SUM",
    "V_RED_MAX",
    "V_RED_SUM_SEG",
    "V_RED_MAX_SEG",
    "V_RED_SUM_SEGS",
    "V_RED_MAX_SEGS",
    "V_RED_SUM_ROWS",
    "V_RED_MAX_ROWS",
    "V_ADD_VSEG",
    "V_SUB_VSEG",
    "V_MUL_VSEG",
    "V_STAT_MUL_F",
    "V_STAT_ADD_F",
    "V_STAT_RSQRT",
    "V_SUB_ROWS",
    "V_EXP_ROWS",
    "V_MUL_ROWS_STATS",
    "V_MUL_ROWS_F",
    "V_SFM_MAX_ROWS",
    "V_SFM_SUM_ROWS",
    "V_SFM_FINAL_ROWS",
    "V_RED_SUM_OVR",
    "V_RED_MAX_OVR",
    "V_RED_SUM_SEG_OVR",
    "V_RED_MAX_SEG_OVR",
    "V_SHIFT_V",
}
SCALAR_COMPUTE_OPS = {
    "S_ADD_FP",
    "S_SUB_FP",
    "S_MAX_FP",
    "S_MUL_FP",
    "S_EXP_FP",
    "S_RECI_FP",
    "S_SQRT_FP",
    "S_MV_FP",
    "S_RSQRT_FP",
    "S_LD_FP",
    "S_ST_FP",
    "S_MAP_V_FP",
    "S_LD_VLANE_FP",
    "S_ST_VLANE_FP",
    "S_ADD_INT",
    "S_ADDI_INT",
    "S_SUB_INT",
    "S_MUL_INT",
    "S_LUI_INT",
    "S_LD_INT",
    "S_ST_INT",
}
CONTROL_OPS = {
    "C_SET_ADDR_REG",
    "C_SET_SCALE_REG",
    "C_SET_STRIDE_REG",
    "C_SET_V_MASK_REG",
    "C_LOOP_START",
    "C_LOOP_END",
    "C_AGU_BIND",
    "C_AGU_LOOP_LEN",
    "C_LOOP_START_AGU",
    "C_BREAK",
}
MEMORY_OPS = {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}
LOOP_AGU_STREAM_COUNT = 6
COST_TRACE_GRANULARITY_DETAILED = "detailed"
COST_TRACE_GRANULARITY_AFFINE_BLOCK_SUMMARY_V1 = "affine-block-summary-v1"
COST_TRACE_GRANULARITIES = {
    COST_TRACE_GRANULARITY_DETAILED,
    COST_TRACE_GRANULARITY_AFFINE_BLOCK_SUMMARY_V1,
}


def opcode_category(opcode: str) -> str:
    """Mirror transactional_emulator/src/profiler.rs category_for."""
    if opcode in MEMORY_OPS:
        return "memory"
    if opcode in MATRIX_COMPUTE_OPS:
        return "matrix_compute"
    if opcode in VECTOR_COMPUTE_OPS:
        return "vector_compute"
    if opcode in SCALAR_COMPUTE_OPS:
        return "scalar_compute"
    if opcode in CONTROL_OPS:
        return "control"
    return "other"


class RawAsmCostError(ValueError):
    """Raised when cost-only lowering reaches unstructured assembly text."""


@dataclass(frozen=True)
class MemoryEvent:
    stage: str
    transfer: DmaTransfer
    multiplicity: int
    enclosing_axes: tuple[RepeatAxis, ...] = ()
    stream_index: int = -1
    parallel_kernel: ParallelKernelTag | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self.transfer)
        repeat_axes = [asdict(axis) for axis in self.enclosing_axes]
        result.update(
            {
                "stage": self.stage,
                "multiplicity": self.multiplicity,
                "stream_instruction_count": self.multiplicity,
                "memory_stream_index": self.stream_index,
                "geometry_fidelity": self.transfer.geometry_fidelity,
                "repeat_axes": repeat_axes,
                "enclosing_axes": repeat_axes,
                "parallel_kernel": (
                    None
                    if self.parallel_kernel is None
                    else self.parallel_kernel.to_dict()
                ),
            }
        )
        return result


@dataclass(frozen=True)
class EnergyAction:
    """Compressed hardware activity emitted from the compiler lowering.

    An action describes *what toggles* rather than assigning power to an ISA
    mnemonic.  Geometry-dependent fields are intentionally allowed to be zero
    here: the power model resolves them from the trial's hardware and
    precision configuration.  Keeping the count in CostTrace guarantees that
    loops, algebraic kernel summaries, and generated assembly share one source
    of truth.
    """

    stage: str
    component: str
    action: str
    count: int
    precision: str = "runtime"
    active_lanes: int = 0
    total_lanes: int = 0
    active_bits: int = 0
    busy_cycles: int = 0
    bytes: int = 0
    variant: str = ""
    segment_log2: int = -1
    segment_count: int = 0
    activity_fidelity: str = "unannotated"
    parallel_kernel: str = "__unclassified__"

    def __post_init__(self) -> None:
        for name in (
            "count",
            "active_lanes",
            "total_lanes",
            "active_bits",
            "busy_cycles",
            "bytes",
            "segment_count",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"EnergyAction.{name} must be nonnegative")
        if self.segment_log2 < -1:
            raise ValueError("EnergyAction.segment_log2 must be -1 or nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClockWork:
    """Compressed clocked-area occupancy for one hardware subcomponent.

    ``equivalent_full_area_cycles`` folds partial lane/slice activity into the
    number of cycles for which the complete subcomponent would consume the
    same clock energy.  It is intentionally area-independent: CostEmitter
    supplies occupancy while the power model combines it with the selected
    area proxy and calibrated clock-energy density.
    """

    stage: str
    component: str
    subcomponent: str
    equivalent_full_area_cycles: float
    component_active_cycles: int
    source_opcode: str
    active_instances: int
    total_instances: int
    fidelity: str

    def __post_init__(self) -> None:
        if self.equivalent_full_area_cycles < 0:
            raise ValueError("ClockWork.equivalent_full_area_cycles must be nonnegative")
        for name in ("component_active_cycles", "active_instances", "total_instances"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"ClockWork.{name} must be nonnegative")
        if self.active_instances > self.total_instances:
            raise ValueError(
                "ClockWork.active_instances cannot exceed total_instances: "
                f"{self.active_instances} > {self.total_instances}"
            )


@dataclass(frozen=True)
class ParallelKernelCensusEntry:
    """Compressed semantic work record for analytical multi-chip partitioning.

    The single-chip compiler still emits the authoritative opcode stream.  The
    census only labels that stream with the logical dimensions along which a
    distributed analytical model may repartition it.
    """

    stage: str
    kernel: str
    opcode: str
    count: int
    tp_semantics: str
    cp_semantics: str
    ep_semantics: str
    logical_rows: int = 0
    logical_m: int = 0
    logical_n: int = 0
    logical_k: int = 0
    matrix_mlen: int = 0
    matrix_blen: int = 0
    multiplicity: int = 1
    fidelity: str = "compiler_semantic_classification"

    def __post_init__(self) -> None:
        for name in (
            "count",
            "logical_rows",
            "logical_m",
            "logical_n",
            "logical_k",
            "matrix_mlen",
            "matrix_blen",
            "multiplicity",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(
                    f"ParallelKernelCensusEntry.{name} must be nonnegative"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParallelKernelTag:
    """Semantic ownership carried by every compressed schedule leaf.

    Unlike a stage-level opcode classification, this tag survives algebraic
    repeats and schedule rewrites.  The final multi-chip census is rebuilt
    from the transformed schedule, so a deferred repeat cannot lose whether
    an instruction belongs to softmax, a projection, or token-wise setup.
    """

    kernel: str
    tp_semantics: str
    cp_semantics: str
    ep_semantics: str
    logical_rows: int = 0
    logical_m: int = 0
    logical_n: int = 0
    logical_k: int = 0
    matrix_mlen: int = 0
    matrix_blen: int = 0
    fidelity: str = "compiler_kernel_lineage_exact_v3"

    def __post_init__(self) -> None:
        for name in (
            "logical_rows",
            "logical_m",
            "logical_n",
            "logical_k",
            "matrix_mlen",
            "matrix_blen",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"ParallelKernelTag.{name} must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parallel_kernel_lineage_id(
    tag: ParallelKernelTag | Mapping[str, Any] | None,
) -> str:
    """Return the stable identity used to join schedule, power, and DSE work."""

    if tag is None:
        return "__unclassified__"
    value = tag.to_dict() if isinstance(tag, ParallelKernelTag) else dict(tag)
    if value.get("kernel") == "__unclassified__":
        return "__unclassified__"
    fields = (
        "kernel",
        "tp_semantics",
        "cp_semantics",
        "ep_semantics",
        "logical_rows",
        "logical_m",
        "logical_n",
        "logical_k",
        "matrix_mlen",
        "matrix_blen",
        "fidelity",
    )
    return "|".join(str(value.get(name, "")) for name in fields)


def _logic_energy_actions(
    opcode: str,
    args: tuple[str, ...] = (),
) -> tuple[tuple[str, str, str, int], ...]:
    """Map an instruction to a structural logic action family.

    The mapping deliberately has fewer action families than opcodes.  Power
    calibration is performed on shared hardware (array, vector ALU, scalar
    SFU, and so on), which avoids an underdetermined coefficient per mnemonic.
    """

    if opcode in MATRIX_COMPUTE_OPS:
        if opcode == "M_MM_WO_PACKED_ACC":
            packed_action = (
                "accumulate"
                if len(args) >= 4 and str(args[3]).strip() == "1"
                else "overwrite"
            )
            return (
                ("matrix", "output_conversion", opcode, -1),
                ("packed_pv_accumulator", packed_action, opcode, -1),
            )
        if opcode.endswith("_WO"):
            return (("matrix", "output_conversion", opcode, -1),)
        if opcode in {"M_MV", "M_TMV", "M_BMV", "M_BTMV"}:
            return (
                ("matrix", "matrix_vector_compute", opcode, -1),
                ("matrix", "cross_k_reduce", opcode, -1),
            )
        return (
            ("matrix", "array_compute", opcode, -1),
            ("matrix", "cross_k_reduce", opcode, -1),
        )
    if opcode in {
        "V_RED_SUM",
        "V_RED_SUM_OVR",
        "V_RED_SUM_SEG",
        "V_RED_SUM_SEG_OVR",
        "V_RED_SUM_SEGS",
        "V_RED_SUM_ROWS",
    }:
        segment_log2 = _segment_log2(opcode, args)
        base_opcode = opcode.removesuffix("_OVR")
        suffix = (
            "rows"
            if base_opcode.endswith("ROWS")
            else "segments"
            if base_opcode.endswith("SEGS")
            else "segment"
            if base_opcode.endswith("SEG")
            else "full"
        )
        return (("vector", f"reduction_sum_{suffix}", opcode, segment_log2),)
    if opcode in {
        "V_RED_MAX",
        "V_RED_MAX_OVR",
        "V_RED_MAX_SEG",
        "V_RED_MAX_SEG_OVR",
        "V_RED_MAX_SEGS",
        "V_RED_MAX_ROWS",
    }:
        segment_log2 = _segment_log2(opcode, args)
        base_opcode = opcode.removesuffix("_OVR")
        suffix = (
            "rows"
            if base_opcode.endswith("ROWS")
            else "segments"
            if base_opcode.endswith("SEGS")
            else "segment"
            if base_opcode.endswith("SEG")
            else "full"
        )
        return (("vector", f"reduction_max_{suffix}", opcode, segment_log2),)
    if opcode in {"V_ADD_VV", "V_SUB_VV"}:
        return (("vector", "lane_add_sub_vv", opcode, -1),)
    if opcode in {"V_ADD_VF", "V_SUB_VF"}:
        return (("vector", "lane_add_sub_vf", opcode, -1),)
    if opcode == "V_SUB_ROWS":
        return (("vector", "softmax_row_subtract", opcode, _segment_log2(opcode, args)),)
    if opcode in {"V_ADD_VSEG", "V_SUB_VSEG"}:
        return (("vector", "lane_add_sub_vseg", opcode, _segment_log2(opcode, args)),)
    if opcode == "V_MUL_VV":
        return (("vector", "lane_multiply_vv", opcode, -1),)
    if opcode == "V_MUL_VF":
        return (("vector", "lane_multiply_vf", opcode, -1),)
    if opcode in {"V_MUL_ROWS_STATS", "V_MUL_ROWS_F"}:
        return (("vector", "softmax_row_multiply", opcode, _segment_log2(opcode, args)),)
    if opcode == "V_MUL_VSEG":
        return (("vector", "lane_multiply_vseg", opcode, _segment_log2(opcode, args)),)
    if opcode in {"V_STAT_MUL_F", "V_STAT_ADD_F", "V_STAT_RSQRT"}:
        family = {
            "V_STAT_MUL_F": "compact_stats_mul",
            "V_STAT_ADD_F": "compact_stats_add",
            "V_STAT_RSQRT": "compact_stats_rsqrt",
        }[opcode]
        return (("vector", family, opcode, -1),)
    if opcode == "V_EXP_V":
        return (("vector", "lane_sfu_exp", opcode, -1),)
    if opcode == "V_EXP_ROWS":
        return (("vector", "softmax_row_exp", opcode, _segment_log2(opcode, args)),)
    if opcode in {"V_SFM_MAX_ROWS", "V_SFM_SUM_ROWS", "V_SFM_FINAL_ROWS"}:
        phase = {
            "V_SFM_MAX_ROWS": "max_update",
            "V_SFM_SUM_ROWS": "sum_update",
            "V_SFM_FINAL_ROWS": "final_reciprocal",
        }[opcode]
        return (("softmax_state", phase, opcode, _segment_log2(opcode, args)),)
    if opcode == "V_RECI_V":
        return (("vector", "lane_sfu_reciprocal", opcode, -1),)
    if opcode == "V_SHIFT_V":
        return (("vector", "lane_movement_shift", opcode, -1),)
    if opcode in SCALAR_COMPUTE_OPS:
        if opcode in {"S_ADD_FP", "S_SUB_FP", "S_MAX_FP", "S_MV_FP"}:
            return (("scalar", "fp_add_sub_move", opcode, -1),)
        if opcode == "S_MUL_FP":
            return (("scalar", "fp_multiply", opcode, -1),)
        scalar_sfu = {
            "S_EXP_FP": "fp_sfu_exp",
            "S_RECI_FP": "fp_sfu_reciprocal",
            "S_SQRT_FP": "fp_sfu_sqrt",
            "S_RSQRT_FP": "fp_sfu_rsqrt",
        }
        if opcode in scalar_sfu:
            return (("scalar", scalar_sfu[opcode], opcode, -1),)
        if opcode == "S_MUL_INT":
            return (("scalar", "integer_multiply", opcode, -1),)
        if opcode in {"S_ADD_INT", "S_ADDI_INT", "S_SUB_INT", "S_LUI_INT"}:
            return (("scalar", "integer_alu", opcode, -1),)
        if opcode == "S_LD_VLANE_FP":
            return (("scalar", "vector_lane_load", opcode, -1),)
        if opcode == "S_ST_VLANE_FP":
            return (("scalar", "vector_lane_store", opcode, -1),)
        return (("scalar", "register_or_sram_access", opcode, -1),)
    if opcode == "C_AGU_BIND":
        return (("agu", "agu_config", opcode, -1),)
    if opcode in {"C_AGU_LOOP_LEN", "C_LOOP_START_AGU"}:
        return (("agu", "agu_loop_setup", opcode, -1),)
    if opcode in CONTROL_OPS:
        return (("control", "frontend_issue", opcode, -1),)
    return ()


def _segment_log2(opcode: str, args: tuple[str, ...]) -> int:
    """Extract the encoded segment width without expanding schedule repeats."""

    if not args:
        return -1
    index = -2 if opcode.endswith("VSEG") and len(args) >= 2 else -1
    try:
        return int(args[index], 0)
    except (ValueError, IndexError):
        return -1


def _row_action_count(opcode: str, args: tuple[str, ...]) -> int:
    if opcode in {
        "V_RED_SUM_ROWS",
        "V_RED_MAX_ROWS",
        "V_SFM_MAX_ROWS",
        "V_SFM_SUM_ROWS",
        "V_SFM_FINAL_ROWS",
    }:
        try:
            return int(args[2], 0)
        except (IndexError, ValueError):
            return 0
    if opcode in {"V_SUB_ROWS", "V_EXP_ROWS", "V_MUL_ROWS_STATS", "V_MUL_ROWS_F"}:
        try:
            return int(args[-2], 0)
        except (IndexError, ValueError):
            return 0
    return 0


def _row_lane_tier(opcode: str, args: tuple[str, ...], fallback: int) -> int:
    """Decode the configured row tier carried by rtl-v6 row instructions."""

    if _row_action_count(opcode, args) <= 0:
        return 0
    try:
        tier_log2 = int(args[-1], 0)
    except (IndexError, ValueError):
        return fallback
    tier = 1 << tier_log2
    return tier if tier in {1, 2, 4, 8, 16} else fallback


def _sram_actions(opcode: str, args: tuple[str, ...] = ()) -> tuple[tuple[str, str, int], ...]:
    """Return logical macro accesses implied by one dynamic instruction."""

    rows = _row_action_count(opcode, args)
    if opcode in {"V_RED_SUM_ROWS", "V_RED_MAX_ROWS"}:
        return (
            ("vector_sram", "read", rows),
            ("softmax_state_sram", "write", rows),
        )
    if opcode in {"V_SFM_MAX_ROWS", "V_SFM_SUM_ROWS", "V_SFM_FINAL_ROWS"}:
        return (
            ("softmax_state_sram", "read", rows),
            ("softmax_state_sram", "write", rows),
        )
    if opcode == "V_SUB_ROWS":
        return (
            ("vector_sram", "read", rows),
            ("vector_sram", "write", rows),
            ("softmax_state_sram", "read", rows),
        )
    if opcode in {"V_EXP_ROWS", "V_MUL_ROWS_F"}:
        return (("vector_sram", "read", rows), ("vector_sram", "write", rows))
    if opcode == "V_MUL_ROWS_STATS":
        return (
            ("vector_sram", "read", rows),
            ("vector_sram", "write", rows),
            ("softmax_state_sram", "read", rows),
        )
    if opcode == "M_MM_WO_PACKED_ACC":
        accumulate = len(args) >= 4 and str(args[3]).strip() == "1"
        vector_read = (("vector_sram", "read", 1),) if accumulate else ()
        return (
            ("matrix_sram", "read", 1),
            *vector_read,
            ("vector_sram", "write", 1),
        )
    if opcode in MATRIX_COMPUTE_OPS:
        return (("matrix_sram", "read", 2), ("matrix_sram", "write", 1))
    if opcode in VECTOR_COMPUTE_OPS:
        reads = 2 if opcode.endswith("VV") or "VSEG" in opcode else 1
        return (("vector_sram", "read", reads), ("vector_sram", "write", 1))
    if opcode in {"S_LD_FP", "S_LD_VLANE_FP"}:
        return (("scalar_fp_sram", "read", 1),)
    if opcode in {"S_ST_FP", "S_ST_VLANE_FP", "S_MAP_V_FP"}:
        return (("scalar_fp_sram", "write", 1),)
    if opcode == "S_LD_INT":
        return (("scalar_int_sram", "read", 1),)
    if opcode == "S_ST_INT":
        return (("scalar_int_sram", "write", 1),)
    return ()


def _build_energy_actions(trace: CostTrace) -> list[EnergyAction]:
    """Materialize a compact, complete action inventory from CostTrace."""

    actions: list[EnergyAction] = []
    if trace.schedule_unavailable_reasons:
        for stage_name, stage in sorted(trace.stages.items()):
            for opcode, raw_count in sorted(stage.dynamic_opcodes.items()):
                count = int(raw_count)
                if count <= 0 or opcode in MEMORY_OPS:
                    continue
                for component, action, source_opcode, segment_log2 in (
                    _logic_energy_actions(opcode)
                ):
                    actions.append(
                        EnergyAction(
                            stage=stage_name,
                            component=component,
                            action=action,
                            count=count,
                            precision=source_opcode,
                            variant="aggregate",
                            segment_log2=segment_log2,
                            activity_fidelity=(
                                "clock_work_unavailable"
                                if opcode in VECTOR_COMPUTE_OPS
                                else "aggregate"
                            ),
                        )
                    )
                for component, action, accesses in _sram_actions(opcode):
                    actions.append(
                        EnergyAction(
                            stage=stage_name,
                            component=component,
                            action=action,
                            count=count * accesses,
                            precision=opcode,
                        )
                    )
        lineage_items: Iterable[tuple[str, ParallelKernelTag | None]] = ()
    else:
        lineage_items = sorted(
            _schedule_parallel_lineage_keys(trace.schedule),
            key=lambda item: (item[0], parallel_kernel_lineage_id(item[1])),
        )
    for stage_name, tag in lineage_items:
        filtered = _filter_schedule_parallel_lineage(
            trace.schedule,
            stage=stage_name,
            tag=tag,
        )
        lineage = parallel_kernel_lineage_id(tag)
        counts = _schedule_opcode_counts(filtered)
        selected_opcodes = {
            opcode
            for opcode, count in counts.items()
            if count > 0 and opcode not in MEMORY_OPS
        }
        variants = schedule_instruction_activity_variants(
            filtered,
            opcodes=selected_opcodes,
            stage=stage_name,
        )
        variants_by_opcode: dict[str, list[tuple[tuple[str, ...], int, str, int]]] = defaultdict(list)
        for (opcode, args, activity_fidelity, active_segments), count in variants.items():
            variants_by_opcode[opcode].append((args, int(count), activity_fidelity, active_segments))
        for opcode, raw_count in sorted(counts.items()):
            count = int(raw_count)
            if count <= 0 or opcode in MEMORY_OPS:
                continue
            opcode_variants = variants_by_opcode.get(opcode, [])
            represented = sum(variant_count for _, variant_count, _, _ in opcode_variants)
            if represented > count:
                raise ValueError(
                    "energy variants over-count "
                    f"{stage_name}/{lineage}/{opcode}: {represented} > {count}"
                )
            if represented < count:
                # Counts-only summaries cannot preserve operands. Keep their
                # structural family visible with an explicit aggregate
                # variant instead of silently dropping energy coverage.
                opcode_variants.append(
                    (
                        (),
                        count - represented,
                        ("clock_work_unavailable" if opcode in VECTOR_COMPUTE_OPS else "aggregate"),
                        0,
                    )
                )
            for args, variant_count, activity_fidelity, active_segments in opcode_variants:
                variant_text = ",".join(args) if args else "aggregate"
                for component, action, source_opcode, segment_log2 in _logic_energy_actions(opcode, args):
                    row_operation = _row_action_count(opcode, args) > 0
                    configured_row_tier = int(
                        trace.metadata.get("packed_attention", {}).get(
                            "softmax_row_lanes", 1
                        )
                    )
                    row_tier = _row_lane_tier(
                        opcode, args, configured_row_tier
                    )
                    actions.append(
                        EnergyAction(
                            stage=stage_name,
                            component=component,
                            action=action,
                            count=variant_count,
                            precision=source_opcode,
                            variant=variant_text,
                            segment_log2=segment_log2,
                            segment_count=active_segments,
                            active_lanes=(active_segments if row_operation else 0),
                            total_lanes=(row_tier if row_operation else 0),
                            activity_fidelity=activity_fidelity,
                            parallel_kernel=lineage,
                        )
                    )
                for component, action, accesses in _sram_actions(opcode, args):
                    actions.append(
                        EnergyAction(
                            stage=stage_name,
                            component=component,
                            action=action,
                            count=variant_count * accesses,
                            precision=opcode,
                            parallel_kernel=lineage,
                        )
                    )
    for event in trace.memory_events:
        transfer = event.transfer
        lineage = parallel_kernel_lineage_id(event.parallel_kernel)
        actions.append(
            EnergyAction(
                stage=event.stage,
                component="hbm_controller",
                action={
                    "H_PREFETCH_M": "matrix_prefetch",
                    "H_PREFETCH_V": "vector_prefetch",
                    "H_STORE_V": "vector_writeback",
                }[transfer.opcode],
                count=int(event.multiplicity),
                precision=str(transfer.precision_role or transfer.precision),
                active_lanes=int(transfer.amount),
                total_lanes=int(transfer.dim),
                parallel_kernel=lineage,
            )
        )
        sram_component = "matrix_sram" if transfer.opcode == "H_PREFETCH_M" else "vector_sram"
        sram_action = "read" if transfer.opcode == "H_STORE_V" else "write"
        actions.append(
            EnergyAction(
                stage=event.stage,
                component=sram_component,
                action=sram_action,
                count=int(event.multiplicity),
                precision=str(transfer.precision_role or transfer.precision),
                active_lanes=int(transfer.amount),
                total_lanes=int(transfer.dim),
                parallel_kernel=lineage,
            )
        )
    actions.extend(_agu_runtime_energy_actions(trace.schedule))
    # DMA streams retain address geometry separately in MemoryEvent. Energy
    # only needs aggregate hardware activity, so merge identical action shapes
    # here instead of carrying tens of thousands of per-stream records into
    # every DSE trial.
    return _merge_energy_actions(actions)


def _merge_energy_actions(actions: Iterable[EnergyAction]) -> list[EnergyAction]:
    """Merge an iterable of exact action records without losing variants."""

    grouped: dict[tuple[Any, ...], list[int]] = {}
    for item in actions:
        key = (
            item.stage,
            item.component,
            item.action,
            item.precision,
            item.active_lanes,
            item.total_lanes,
            item.active_bits,
            item.variant,
            item.segment_log2,
            item.segment_count,
            item.activity_fidelity,
            item.parallel_kernel,
        )
        totals = grouped.setdefault(key, [0, 0, 0])
        totals[0] += item.count
        totals[1] += item.busy_cycles
        totals[2] += item.bytes
    return [
        EnergyAction(
            stage=key[0],
            component=key[1],
            action=key[2],
            precision=key[3],
            active_lanes=key[4],
            total_lanes=key[5],
            active_bits=key[6],
            variant=key[7],
            segment_log2=key[8],
            segment_count=key[9],
            activity_fidelity=key[10],
            parallel_kernel=key[11],
            count=totals[0],
            busy_cycles=totals[1],
            bytes=totals[2],
        )
        for key, totals in sorted(grouped.items())
    ]


def _build_summary_energy_actions(
    trace: CostTrace,
    variants_by_lineage: Mapping[
        tuple[str, ParallelKernelTag | None],
        Mapping[tuple[str, tuple[str, ...], str, int], int],
    ],
) -> list[EnergyAction]:
    """Build logic/SRAM actions from incrementally collected exact variants."""

    actions: list[EnergyAction] = []
    counts_by_lineage: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for entry in trace.parallel_kernel_census:
        lineage = parallel_kernel_lineage_id(
            {
                "kernel": entry.kernel,
                "tp_semantics": entry.tp_semantics,
                "cp_semantics": entry.cp_semantics,
                "ep_semantics": entry.ep_semantics,
                "logical_rows": entry.logical_rows,
                "logical_m": entry.logical_m,
                "logical_n": entry.logical_n,
                "logical_k": entry.logical_k,
                "matrix_mlen": entry.matrix_mlen,
                "matrix_blen": entry.matrix_blen,
                "fidelity": entry.fidelity,
            }
        )
        counts_by_lineage[(entry.stage, lineage)][entry.opcode] += entry.count

    tags_by_identity = {
        (stage, parallel_kernel_lineage_id(tag)): tag
        for stage, tag in variants_by_lineage
    }
    for stage_name, lineage in sorted(counts_by_lineage):
        grouped: dict[str, list[tuple[tuple[str, ...], int, str, int]]] = defaultdict(list)
        tag = tags_by_identity.get((stage_name, lineage))
        for (
            opcode,
            args,
            fidelity,
            active_segments,
        ), count in variants_by_lineage.get((stage_name, tag), {}).items():
            grouped[opcode].append((args, int(count), fidelity, int(active_segments)))
        for opcode, raw_count in sorted(counts_by_lineage[(stage_name, lineage)].items()):
            count = int(raw_count)
            if count <= 0 or opcode in MEMORY_OPS:
                continue
            opcode_variants = grouped.get(opcode, [])
            represented = sum(item[1] for item in opcode_variants)
            if represented > count:
                raise ValueError(
                    "summary energy variants over-count "
                    f"{stage_name}/{lineage}/{opcode}: {represented} > {count}"
                )
            if represented < count:
                opcode_variants.append(
                    (
                        (),
                        count - represented,
                        ("clock_work_unavailable" if opcode in VECTOR_COMPUTE_OPS else "aggregate"),
                        0,
                    )
                )
            for args, variant_count, fidelity, active_segments in opcode_variants:
                variant_text = ",".join(args) if args else "aggregate"
                row_operation = _row_action_count(opcode, args) > 0
                configured_row_tier = int(
                    trace.metadata.get("packed_attention", {}).get(
                        "softmax_row_lanes", 1
                    )
                )
                row_tier = _row_lane_tier(opcode, args, configured_row_tier)
                for component, action, source_opcode, segment_log2 in _logic_energy_actions(opcode, args):
                    actions.append(
                        EnergyAction(
                            stage=stage_name,
                            component=component,
                            action=action,
                            count=variant_count,
                            precision=source_opcode,
                            variant=variant_text,
                            segment_log2=segment_log2,
                            segment_count=active_segments,
                            active_lanes=(active_segments if row_operation else 0),
                            total_lanes=(row_tier if row_operation else 0),
                            activity_fidelity=fidelity,
                            parallel_kernel=lineage,
                        )
                    )
                for component, action, accesses in _sram_actions(opcode, args):
                    actions.append(
                        EnergyAction(
                            stage=stage_name,
                            component=component,
                            action=action,
                            count=variant_count * accesses,
                            precision=opcode,
                            parallel_kernel=lineage,
                        )
                    )
    return _merge_energy_actions(actions)


def _encode_schedule_variants(
    variants: Mapping[tuple[str, tuple[str, ...]], int],
) -> list[dict[str, Any]]:
    """Serialize operand-qualified counts for unscheduled timing consumers."""

    return [
        {"opcode": opcode, "args": list(args), "count": int(count)}
        for (opcode, args), count in sorted(variants.items())
        if int(count)
    ]


def _agu_runtime_energy_actions(schedule: ScheduleNode) -> list[EnergyAction]:
    """Count implicit AGU boundary work without inventing dynamic opcodes.

    The transformed schedule keeps bindings immediately before an
    ``agu_hardware_loop`` repeat.  Walking that structure preserves outer
    model-layer multiplicity while keeping the inventory compressed.
    """

    grouped: Counter[tuple[str, str, str, int, int]] = Counter()

    def register_mentions(node: ScheduleNode, registers: set[str]) -> int:
        if isinstance(node, ScheduleInstruction):
            if node.opcode.startswith("C_AGU") or node.opcode == "C_LOOP_END":
                return 0
            return sum(argument in registers for argument in node.args)
        if isinstance(node, ScheduleSequence):
            return sum(register_mentions(child, registers) for child in node.children)
        if isinstance(node, ScheduleRepeat):
            return node.count * register_mentions(node.body, registers)
        return 0

    def visit(node: ScheduleNode, multiplier: int) -> None:
        if isinstance(node, ScheduleSequence):
            pending_bindings: list[str] = []
            for child in node.children:
                if isinstance(child, ScheduleInstruction):
                    if child.opcode == "C_AGU_BIND" and child.args:
                        pending_bindings.append(child.args[0])
                    elif child.opcode not in {
                        "C_AGU_LOOP_LEN",
                        "C_LOOP_START_AGU",
                    }:
                        pending_bindings.clear()
                    continue
                if isinstance(child, ScheduleRepeat) and child.repeat_kind == "agu_hardware_loop":
                    owners = _schedule_parallel_lineage_keys(child.body)
                    if len(owners) == 1:
                        stage, tag = next(iter(owners))
                    else:
                        stage, tag = "global", None
                    lineage = parallel_kernel_lineage_id(tag)
                    stream_count = len(pending_bindings)
                    boundaries = multiplier * child.count
                    grouped[
                        (stage, lineage, "agu_loop_boundary", 1, 1)
                    ] += boundaries
                    if stream_count:
                        grouped[
                            (
                                stage,
                                lineage,
                                "agu_stream_step",
                                stream_count,
                                LOOP_AGU_STREAM_COUNT,
                            )
                        ] += boundaries
                        reads_per_iteration = register_mentions(child.body, set(pending_bindings))
                        if reads_per_iteration:
                            grouped[
                                (stage, lineage, "agu_offset_read", 1, 2)
                            ] += multiplier * child.count * reads_per_iteration
                    visit(child.body, multiplier * child.count)
                    pending_bindings.clear()
                    continue
                visit(child, multiplier)
                pending_bindings.clear()
            return
        if isinstance(node, ScheduleRepeat):
            visit(node.body, multiplier * node.count)

    visit(schedule, 1)
    return [
        EnergyAction(
            stage=stage,
            component="agu",
            action=action,
            count=count,
            precision="implicit_loop_agu_v1",
            active_lanes=active_instances,
            total_lanes=total_instances,
            activity_fidelity="structural_exact_count",
            parallel_kernel=lineage,
        )
        for (
            stage,
            lineage,
            action,
            active_instances,
            total_instances,
        ), count in sorted(grouped.items())
    ]


@dataclass(frozen=True)
class ScheduleInstruction:
    """One ordered dynamic instruction in the compressed schedule IR.

    Arguments retain architectural register names so the Python shadow
    scheduler can either resolve dependencies exactly or reject the schedule
    explicitly. ``memory_stream_index`` links a DMA instruction to the same
    compressed geometry used by the HBM cost model.
    """

    opcode: str
    args: tuple[str, ...] = ()
    stage: str = "global"
    memory_stream_index: int | None = None
    parallel_kernel: ParallelKernelTag | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "instruction",
            "opcode": self.opcode,
            "args": list(self.args),
            "stage": self.stage,
        }
        if self.memory_stream_index is not None:
            result["memory_stream_index"] = self.memory_stream_index
        if self.parallel_kernel is not None:
            result["parallel_kernel"] = self.parallel_kernel.to_dict()
        return result


@dataclass(frozen=True)
class ScheduleAffineLoad:
    """Load an affine address without materializing every compiler iteration.

    The node expands to the exact legalized ``S_ADDI_INT``/``S_LUI_INT``
    sequence for ``start + step * position``. ``period`` resets the position
    when the surrounding kernel is replayed, for example for every decoder
    layer.
    """

    key: str
    register: str
    start: int
    step: int
    period: int | None = None
    advance_every: int = 1
    stage: str = "global"
    parallel_kernel: ParallelKernelTag | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "affine_load",
            "key": self.key,
            "register": self.register,
            "start": self.start,
            "step": self.step,
            "period": self.period,
            "advance_every": self.advance_every,
            "stage": self.stage,
            "parallel_kernel": (
                None
                if self.parallel_kernel is None
                else self.parallel_kernel.to_dict()
            ),
        }


@dataclass(frozen=True)
class ScheduleAffineAdd:
    """Add an affine immediate, preserving large-immediate temp semantics."""

    key: str
    destination: str
    source: str
    temp: str
    start: int
    step: int
    period: int | None = None
    advance_every: int = 1
    stage: str = "global"
    parallel_kernel: ParallelKernelTag | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "affine_add",
            "key": self.key,
            "destination": self.destination,
            "source": self.source,
            "temp": self.temp,
            "start": self.start,
            "step": self.step,
            "period": self.period,
            "advance_every": self.advance_every,
            "stage": self.stage,
            "parallel_kernel": (
                None
                if self.parallel_kernel is None
                else self.parallel_kernel.to_dict()
            ),
        }


@dataclass(frozen=True)
class ScheduleSequence:
    children: tuple[ScheduleNode, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sequence",
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class ScheduleRepeat:
    count: int
    body: ScheduleSequence
    name: str = "repeat"
    repeat_kind: str = "compile_time"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "repeat",
            "count": self.count,
            "name": self.name,
            "repeat_kind": self.repeat_kind,
            "body": self.body.to_dict(),
        }


@dataclass(frozen=True)
class ScheduleUnavailable:
    """A counts-only region whose instruction order was not preserved."""

    reason: str
    stage: str
    dynamic_instruction_count: int
    dynamic_opcodes: tuple[tuple[str, int], ...] = ()
    parallel_kernel: ParallelKernelTag | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "unavailable",
            "reason": self.reason,
            "stage": self.stage,
            "dynamic_instruction_count": self.dynamic_instruction_count,
            "dynamic_opcodes": dict(self.dynamic_opcodes),
            "parallel_kernel": (
                None
                if self.parallel_kernel is None
                else self.parallel_kernel.to_dict()
            ),
        }


type ScheduleNode = (
    ScheduleInstruction
    | ScheduleAffineLoad
    | ScheduleAffineAdd
    | ScheduleSequence
    | ScheduleRepeat
    | ScheduleUnavailable
)


def _schedule_opcode_counts(node: ScheduleNode) -> Counter[str]:
    regular: Counter[str] = Counter()
    affine_specs: dict[str, ScheduleAffineLoad] = {}
    affine_visits: Counter[str] = Counter()
    affine_add_specs: dict[str, ScheduleAffineAdd] = {}
    affine_add_visits: Counter[str] = Counter()

    def visit(current: ScheduleNode, multiplier: int) -> None:
        if isinstance(current, ScheduleInstruction):
            regular[current.opcode] += multiplier
            return
        if isinstance(current, ScheduleAffineLoad):
            previous = affine_specs.setdefault(current.key, current)
            if (
                replace(
                    previous,
                    stage=current.stage,
                    parallel_kernel=current.parallel_kernel,
                )
                != current
            ):
                raise ValueError(
                    f"affine schedule key {current.key!r} has inconsistent "
                    f"definitions: previous={previous!r}, current={current!r}"
                )
            affine_visits[current.key] += multiplier
            return
        if isinstance(current, ScheduleAffineAdd):
            previous = affine_add_specs.setdefault(current.key, current)
            if (
                replace(
                    previous,
                    stage=current.stage,
                    parallel_kernel=current.parallel_kernel,
                )
                != current
            ):
                raise ValueError(
                    f"affine-add schedule key {current.key!r} has "
                    f"inconsistent definitions: previous={previous!r}, "
                    f"current={current!r}"
                )
            affine_add_visits[current.key] += multiplier
            return
        if isinstance(current, ScheduleUnavailable):
            if not current.dynamic_opcodes:
                raise ValueError(
                    "cannot derive opcode histogram from unavailable schedule "
                    "node without exact opcode counts"
                )
            encoded = Counter(
                {
                    str(opcode): int(count)
                    for opcode, count in current.dynamic_opcodes
                    if int(count)
                }
            )
            if sum(encoded.values()) != current.dynamic_instruction_count:
                raise ValueError(
                    "unavailable schedule opcode counts do not sum to its "
                    f"dynamic count: node={current!r}"
                )
            regular.update(
                {
                    opcode: count * multiplier
                    for opcode, count in encoded.items()
                }
            )
            return
        if isinstance(current, ScheduleSequence):
            for child in current.children:
                visit(child, multiplier)
            return
        if isinstance(current, ScheduleRepeat):
            visit(current.body, multiplier * current.count)
            return
        raise TypeError(type(current).__name__)

    def affine_unique_value_counts(affine: ScheduleAffineLoad, count: int) -> Counter[str]:
        if count <= 0:
            return Counter()
        if affine.step < 0 or affine.start < 0:
            raise ValueError(f"negative affine load is unsupported: {affine!r}")
        below = (
            max(
                0,
                min(
                    count,
                    (IMM2_BOUND - affine.start + affine.step - 1) // affine.step,
                ),
            )
            if affine.start < IMM2_BOUND and affine.step
            else count
            if affine.start < IMM2_BOUND
            else 0
        )
        high = count - below
        zero_low = 0
        if high:
            modulus = 1 << 12
            if affine.step == 0:
                zero_low = high if affine.start % modulus == 0 else 0
            else:
                divisor = math.gcd(affine.step, modulus)
                rhs = -affine.start
                if rhs % divisor == 0:
                    reduced_modulus = modulus // divisor
                    first = ((rhs // divisor) * pow(affine.step // divisor, -1, reduced_modulus)) % reduced_modulus
                    if first < below:
                        first += ((below - first + reduced_modulus - 1) // reduced_modulus) * reduced_modulus
                    if first < count:
                        zero_low = 1 + (count - 1 - first) // reduced_modulus
        return Counter(
            {
                "S_LUI_INT": high,
                "S_ADDI_INT": below + high - zero_low,
            }
        )

    def affine_sequence_counts(affine: ScheduleAffineLoad, count: int) -> Counter[str]:
        if affine.advance_every <= 0:
            raise ValueError(f"affine advance_every must be positive: {affine!r}")
        full_values, remainder = divmod(count, affine.advance_every)
        unique = affine_unique_value_counts(affine, full_values)
        result = Counter({opcode: opcode_count * affine.advance_every for opcode, opcode_count in unique.items()})
        if remainder:
            value = affine.start + affine.step * full_values
            tail = replace(
                affine,
                start=value,
                step=0,
                advance_every=1,
                period=None,
            )
            result.update(
                {
                    opcode: opcode_count * remainder
                    for opcode, opcode_count in affine_unique_value_counts(tail, 1).items()
                }
            )
        return +result

    def affine_add_unique_value_counts(affine: ScheduleAffineAdd, count: int) -> Counter[str]:
        if count <= 0:
            return Counter()
        if affine.step < 0 or affine.start < 0:
            raise ValueError(f"negative affine add is unsupported: {affine!r}")
        below = (
            max(
                0,
                min(
                    count,
                    (IMM2_BOUND - affine.start + affine.step - 1) // affine.step,
                ),
            )
            if affine.start < IMM2_BOUND and affine.step
            else count
            if affine.start < IMM2_BOUND
            else 0
        )
        high = count - below
        zero_low = 0
        if high:
            modulus = 1 << 12
            if affine.step == 0:
                zero_low = high if affine.start % modulus == 0 else 0
            else:
                divisor = math.gcd(affine.step, modulus)
                rhs = -affine.start
                if rhs % divisor == 0:
                    reduced_modulus = modulus // divisor
                    first = ((rhs // divisor) * pow(affine.step // divisor, -1, reduced_modulus)) % reduced_modulus
                    if first < below:
                        first += ((below - first + reduced_modulus - 1) // reduced_modulus) * reduced_modulus
                    if first < count:
                        zero_low = 1 + (count - 1 - first) // reduced_modulus
        return +Counter(
            {
                "S_ADDI_INT": below + high - zero_low,
                "S_LUI_INT": high,
                "S_ADD_INT": high,
            }
        )

    def affine_add_sequence_counts(affine: ScheduleAffineAdd, count: int) -> Counter[str]:
        if affine.advance_every <= 0:
            raise ValueError(f"affine-add advance_every must be positive: {affine!r}")
        full_values, remainder = divmod(count, affine.advance_every)
        unique = affine_add_unique_value_counts(affine, full_values)
        result = Counter({opcode: opcode_count * affine.advance_every for opcode, opcode_count in unique.items()})
        if remainder:
            tail = replace(
                affine,
                start=affine.start + affine.step * full_values,
                step=0,
                period=None,
                advance_every=1,
            )
            result.update(
                {
                    opcode: opcode_count * remainder
                    for opcode, opcode_count in affine_add_unique_value_counts(tail, 1).items()
                }
            )
        return +result

    def add_periodic_affine_counts(
        affine: ScheduleAffineLoad | ScheduleAffineAdd,
        visits: int,
        counter: Callable[[Any, int], Counter[str]],
    ) -> None:
        if affine.period is None:
            regular.update(counter(affine, visits))
            return
        if affine.period <= 0 or affine.period % affine.advance_every:
            raise ValueError(f"affine period must be positive and contain complete repeated-value groups: {affine!r}")
        full_periods, remainder = divmod(visits, affine.period)
        period_counts = counter(affine, affine.period)
        regular.update({opcode: count * full_periods for opcode, count in period_counts.items()})
        regular.update(counter(affine, remainder))

    visit(node, 1)
    for key, visits in affine_visits.items():
        affine = affine_specs[key]
        add_periodic_affine_counts(affine, visits, affine_sequence_counts)
    for key, visits in affine_add_visits.items():
        add_periodic_affine_counts(affine_add_specs[key], visits, affine_add_sequence_counts)
    return +regular


UNCLASSIFIED_PARALLEL_KERNEL = "__unclassified__"


def _schedule_parallel_lineage_keys(
    node: ScheduleNode,
) -> set[tuple[str, ParallelKernelTag | None]]:
    keys: set[tuple[str, ParallelKernelTag | None]] = set()

    def visit(current: ScheduleNode) -> None:
        if isinstance(
            current,
            (
                ScheduleInstruction,
                ScheduleAffineLoad,
                ScheduleAffineAdd,
                ScheduleUnavailable,
            ),
        ):
            keys.add((current.stage, current.parallel_kernel))
            return
        if isinstance(current, ScheduleSequence):
            for child in current.children:
                visit(child)
            return
        if isinstance(current, ScheduleRepeat):
            visit(current.body)
            return
        raise TypeError(type(current).__name__)

    visit(node)
    return keys


def _filter_schedule_parallel_lineage(
    node: ScheduleNode,
    *,
    stage: str,
    tag: ParallelKernelTag | None,
) -> ScheduleNode:
    if isinstance(
        node,
        (
            ScheduleInstruction,
            ScheduleAffineLoad,
            ScheduleAffineAdd,
            ScheduleUnavailable,
        ),
    ):
        return (
            node
            if node.stage == stage and node.parallel_kernel == tag
            else ScheduleSequence()
        )
    if isinstance(node, ScheduleSequence):
        return replace(
            node,
            children=tuple(
                _filter_schedule_parallel_lineage(
                    child,
                    stage=stage,
                    tag=tag,
                )
                for child in node.children
            ),
        )
    if isinstance(node, ScheduleRepeat):
        body = _filter_schedule_parallel_lineage(
            node.body,
            stage=stage,
            tag=tag,
        )
        assert isinstance(body, ScheduleSequence)
        return replace(node, body=body)
    raise TypeError(type(node).__name__)


def schedule_parallel_kernel_census(
    node: ScheduleNode,
) -> list[ParallelKernelCensusEntry]:
    """Derive exact semantic opcode counts from a transformed schedule."""

    entries: list[ParallelKernelCensusEntry] = []
    reconstructed: Counter[str] = Counter()
    for stage, tag in sorted(
        _schedule_parallel_lineage_keys(node),
        key=lambda item: (
            item[0],
            "" if item[1] is None else item[1].kernel,
            "" if item[1] is None else item[1].tp_semantics,
        ),
    ):
        filtered = _filter_schedule_parallel_lineage(
            node,
            stage=stage,
            tag=tag,
        )
        counts = _schedule_opcode_counts(filtered)
        reconstructed.update(counts)
        for opcode, count in sorted(counts.items()):
            if opcode.startswith("H_") or count <= 0:
                continue
            if tag is None:
                entries.append(
                    ParallelKernelCensusEntry(
                        stage=stage,
                        kernel=UNCLASSIFIED_PARALLEL_KERNEL,
                        opcode=opcode,
                        count=int(count),
                        tp_semantics=UNCLASSIFIED_PARALLEL_KERNEL,
                        cp_semantics=UNCLASSIFIED_PARALLEL_KERNEL,
                        ep_semantics=UNCLASSIFIED_PARALLEL_KERNEL,
                        fidelity="compiler_kernel_lineage_missing",
                    )
                )
                continue
            entries.append(
                ParallelKernelCensusEntry(
                    stage=stage,
                    kernel=tag.kernel,
                    opcode=opcode,
                    count=int(count),
                    tp_semantics=tag.tp_semantics,
                    cp_semantics=tag.cp_semantics,
                    ep_semantics=tag.ep_semantics,
                    logical_rows=tag.logical_rows,
                    logical_m=tag.logical_m,
                    logical_n=tag.logical_n,
                    logical_k=tag.logical_k,
                    matrix_mlen=tag.matrix_mlen,
                    matrix_blen=tag.matrix_blen,
                    fidelity=tag.fidelity,
                )
            )
    expected = _schedule_opcode_counts(node)
    if reconstructed != expected:
        raise ValueError(
            "parallel-kernel schedule reconstruction drifted from opcode "
            f"counts: census={reconstructed}, schedule={expected}"
        )
    return entries


def rebuild_parallel_kernel_census(trace: CostTrace) -> None:
    """Replace a trace census with final post-rewrite schedule lineage."""

    trace.parallel_kernel_census = schedule_parallel_kernel_census(
        trace.schedule
    )


def schedule_instruction_variants(
    node: ScheduleNode,
    *,
    opcodes: frozenset[str] | set[str],
    stage: str | None = None,
) -> Counter[tuple[str, tuple[str, ...]]]:
    """Count operand-sensitive instructions without expanding repeats.

    The ordinary opcode histogram deliberately omits operands for assembly
    parity. Segment-reduction latency, however, depends on the encoded
    ``segment_log2`` operand. This auxiliary view preserves that distinction
    while retaining the compact schedule representation.

    Unavailable schedule regions are skipped. Timing consumers validate that
    every selected dynamic opcode is represented here and fail closed if a
    parameterized instruction was emitted through a counts-only path.
    """

    selected = frozenset(opcodes)
    variants: Counter[tuple[str, tuple[str, ...]]] = Counter()

    def visit(current: ScheduleNode, multiplier: int) -> None:
        if isinstance(current, ScheduleInstruction):
            if current.opcode in selected and (stage is None or current.stage == stage):
                variants[(current.opcode, current.args)] += multiplier
            return
        if isinstance(current, ScheduleSequence):
            for child in current.children:
                visit(child, multiplier)
            return
        if isinstance(current, ScheduleRepeat):
            visit(current.body, multiplier * current.count)
            return
        if isinstance(
            current,
            (ScheduleAffineLoad, ScheduleAffineAdd, ScheduleUnavailable),
        ):
            return
        raise TypeError(type(current).__name__)

    visit(node, 1)
    return variants


_VECTOR_MASK_OPERAND_INDEX = {
    "V_ADD_VV": 3,
    "V_ADD_VF": 3,
    "V_SUB_VV": 3,
    "V_SUB_VF": 3,
    "V_MUL_VV": 3,
    "V_MUL_VF": 3,
    "V_EXP_V": 2,
    "V_RECI_V": 2,
    "V_RED_SUM": 2,
    "V_RED_MAX": 2,
    "V_RED_SUM_OVR": 2,
    "V_RED_MAX_OVR": 2,
}


def _vector_activity(
    opcode: str,
    args: tuple[str, ...],
    vector_mask: int | None,
) -> tuple[str, int]:
    """Return mask fidelity and active segment count for one vector opcode."""

    if opcode in {
        "V_RED_SUM_SEG",
        "V_RED_MAX_SEG",
        "V_RED_SUM_SEG_OVR",
        "V_RED_MAX_SEG_OVR",
    }:
        return "exact_single_segment", 1
    if opcode in {"V_RED_SUM_SEGS", "V_RED_MAX_SEGS", "V_SHIFT_V"}:
        return "full_width", 0
    if opcode in {"V_RED_SUM_ROWS", "V_RED_MAX_ROWS", "V_SFM_MAX_ROWS", "V_SFM_SUM_ROWS", "V_SFM_FINAL_ROWS"}:
        if len(args) < 3:
            return "clock_work_unavailable", 0
        try:
            return "exact_active_rows", int(args[2], 0)
        except ValueError:
            return "clock_work_unavailable", 0
    if opcode in {"V_SUB_ROWS", "V_EXP_ROWS", "V_MUL_ROWS_STATS", "V_MUL_ROWS_F"}:
        rows = _row_action_count(opcode, args)
        return ("configured_row_tier", rows) if rows else ("clock_work_unavailable", 0)
    if opcode in {"V_ADD_VSEG", "V_SUB_VSEG", "V_MUL_VSEG"}:
        if len(args) == 4:
            return "full_width", 0
        mask_index = 4
    elif opcode in {"V_STAT_MUL_F", "V_STAT_ADD_F", "V_STAT_RSQRT"}:
        if len(args) < 4:
            return "clock_work_unavailable", 0
        try:
            return "exact_compact_lanes", int(args[3], 0)
        except ValueError:
            return "clock_work_unavailable", 0
    else:
        mask_index = _VECTOR_MASK_OPERAND_INDEX.get(opcode, -1)
    if mask_index >= 0 and len(args) == mask_index:
        # Historical assembly permits the zero-valued mask operand to be
        # omitted. This is an architectural full-width operation, not missing
        # activity metadata.
        return "full_width", 0
    if mask_index < 0 or mask_index >= len(args):
        return "clock_work_unavailable", 0
    try:
        mask_enabled = int(args[mask_index], 0) != 0
    except ValueError:
        return "clock_work_unavailable", 0
    if not mask_enabled:
        return "full_width", 0
    if vector_mask is None:
        return "clock_work_unavailable", 0
    return "exact_segment_mask", int(vector_mask).bit_count()


def schedule_instruction_activity_variants(
    node: ScheduleNode,
    *,
    opcodes: frozenset[str] | set[str],
    stage: str | None = None,
) -> Counter[tuple[str, tuple[str, ...], str, int]]:
    """Count operand and vector-mask variants without expanding repeats.

    The compiler commonly loads a constant segment mask into a GP register and
    then emits a masked vector operation inside a compressed hardware loop.
    This small abstract interpreter propagates only integer constants needed
    for that mask. Address-like affine updates deliberately invalidate their
    destinations; an unresolved masked operation is retained with
    ``clock_work_unavailable`` fidelity instead of assuming full-width work.
    """

    selected = frozenset(opcodes)
    variants: Counter[tuple[str, tuple[str, ...], str, int]] = Counter()
    gp_constants: dict[str, int] = {"gp0": 0}
    vector_mask: int | None = None

    def parse_int(text: str) -> int | None:
        try:
            return int(text, 0)
        except ValueError:
            return None

    def write_gp(register: str, value: int | None) -> None:
        if not register.startswith("gp") or register == "gp0":
            return
        if value is None:
            gp_constants.pop(register, None)
        else:
            gp_constants[register] = int(value)

    def execute_constant_effect(instruction: ScheduleInstruction) -> None:
        nonlocal vector_mask
        opcode = instruction.opcode
        args = instruction.args
        if opcode == "S_ADDI_INT" and len(args) >= 3:
            immediate = parse_int(args[2])
            source = gp_constants.get(args[1])
            write_gp(
                args[0],
                None if immediate is None or source is None else source + immediate,
            )
        elif opcode == "S_LUI_INT" and len(args) >= 2:
            immediate = parse_int(args[1])
            write_gp(args[0], None if immediate is None else immediate << 12)
        elif opcode in {"S_ADD_INT", "S_SUB_INT", "S_MUL_INT"} and len(args) >= 3:
            left = gp_constants.get(args[1])
            right = gp_constants.get(args[2])
            value = None
            if left is not None and right is not None:
                if opcode == "S_ADD_INT":
                    value = left + right
                elif opcode == "S_SUB_INT":
                    value = left - right
                else:
                    value = left * right
            write_gp(args[0], value)
        elif opcode in {"S_LD_INT"} and args:
            write_gp(args[0], None)
        if opcode == "C_SET_V_MASK_REG":
            vector_mask = gp_constants.get(args[0]) if args else None

    def visit(current: ScheduleNode, multiplier: int) -> None:
        if isinstance(current, ScheduleInstruction):
            if current.opcode in selected and (stage is None or current.stage == stage):
                if current.opcode in VECTOR_COMPUTE_OPS:
                    fidelity, active_segments = _vector_activity(current.opcode, current.args, vector_mask)
                else:
                    fidelity, active_segments = "full_component", 0
                variants[
                    (
                        current.opcode,
                        current.args,
                        fidelity,
                        active_segments,
                    )
                ] += multiplier
            execute_constant_effect(current)
            return
        if isinstance(current, ScheduleSequence):
            for child in current.children:
                visit(child, multiplier)
            return
        if isinstance(current, ScheduleRepeat):
            # The body is represented once and algebraically multiplied. Its
            # constant mask setup is deterministic for all existing compiler
            # loops, while loop-carried address registers are irrelevant here.
            visit(current.body, multiplier * current.count)
            return
        if isinstance(current, ScheduleAffineLoad):
            write_gp(current.register, None)
            return
        if isinstance(current, ScheduleAffineAdd):
            write_gp(current.destination, None)
            write_gp(current.temp, None)
            return
        if isinstance(current, ScheduleUnavailable):
            return
        raise TypeError(type(current).__name__)

    visit(node, 1)
    return variants


def _remap_schedule_memory_streams(node: ScheduleNode, stream_indices: tuple[int, ...]) -> ScheduleNode:
    """Replace kernel-local DMA stream ordinals with trace-global indices."""
    if isinstance(node, ScheduleInstruction):
        local = node.memory_stream_index
        if local is None:
            return node
        if local < 0 or local >= len(stream_indices):
            raise ValueError(f"schedule DMA stream index {local} is outside [0, {len(stream_indices)})")
        return replace(node, memory_stream_index=stream_indices[local])
    if isinstance(node, (ScheduleAffineLoad, ScheduleAffineAdd)):
        return node
    if isinstance(node, ScheduleUnavailable):
        return node
    if isinstance(node, ScheduleSequence):
        return replace(
            node,
            children=tuple(_remap_schedule_memory_streams(child, stream_indices) for child in node.children),
        )
    if isinstance(node, ScheduleRepeat):
        body = _remap_schedule_memory_streams(node.body, stream_indices)
        assert isinstance(body, ScheduleSequence)
        return replace(
            node,
            body=body,
        )
    raise TypeError(type(node).__name__)


def _retag_schedule_stage(node: ScheduleNode, stage: str) -> ScheduleNode:
    if isinstance(node, ScheduleInstruction):
        return replace(node, stage=stage)
    if isinstance(node, (ScheduleAffineLoad, ScheduleAffineAdd)):
        return replace(node, stage=stage)
    if isinstance(node, ScheduleUnavailable):
        return replace(node, stage=stage)
    if isinstance(node, ScheduleSequence):
        return replace(
            node,
            children=tuple(_retag_schedule_stage(child, stage) for child in node.children),
        )
    if isinstance(node, ScheduleRepeat):
        body = _retag_schedule_stage(node.body, stage)
        assert isinstance(body, ScheduleSequence)
        return replace(node, body=body)
    raise TypeError(type(node).__name__)


def _retag_schedule_parallel_kernel(
    node: ScheduleNode,
    tag: ParallelKernelTag | None,
) -> ScheduleNode:
    if isinstance(
        node,
        (
            ScheduleInstruction,
            ScheduleAffineLoad,
            ScheduleAffineAdd,
            ScheduleUnavailable,
        ),
    ):
        if node.parallel_kernel is not None and node.parallel_kernel != tag:
            raise ValueError(
                "ordered schedule already carries a conflicting parallel "
                f"kernel tag: existing={node.parallel_kernel!r}, new={tag!r}"
            )
        return replace(node, parallel_kernel=tag)
    if isinstance(node, ScheduleSequence):
        return replace(
            node,
            children=tuple(
                _retag_schedule_parallel_kernel(child, tag)
                for child in node.children
            ),
        )
    if isinstance(node, ScheduleRepeat):
        body = _retag_schedule_parallel_kernel(node.body, tag)
        assert isinstance(body, ScheduleSequence)
        return replace(node, body=body)
    raise TypeError(type(node).__name__)


def _schedule_unavailable_counts(node: ScheduleNode) -> Counter[str]:
    if isinstance(node, (ScheduleInstruction, ScheduleAffineLoad, ScheduleAffineAdd)):
        return Counter()
    if isinstance(node, ScheduleUnavailable):
        return Counter({node.reason: node.dynamic_instruction_count})
    if isinstance(node, ScheduleSequence):
        result: Counter[str] = Counter()
        for child in node.children:
            result.update(_schedule_unavailable_counts(child))
        return result
    if isinstance(node, ScheduleRepeat):
        body = _schedule_unavailable_counts(node.body)
        return Counter({reason: count * node.count for reason, count in body.items()})
    raise TypeError(type(node).__name__)


def _bind_unindexed_memory_instructions(node: ScheduleNode, memory_events: list[MemoryEvent]) -> ScheduleNode:
    """Link legacy plain HBM instructions to separately recorded DMA streams.

    Older templates emit an ordinary ``Instr`` and call
    ``record_dma_stream`` afterwards.  The geometry is exact, but the two
    records are not connected until this finalization pass.  Matching is
    intentionally strict so a schedule cannot silently consume another
    operation's service-time estimate.
    """

    bound_indices: set[int] = set()

    def collect(current: ScheduleNode) -> None:
        if isinstance(current, ScheduleInstruction):
            if current.memory_stream_index is not None:
                bound_indices.add(current.memory_stream_index)
            return
        if isinstance(current, (ScheduleAffineLoad, ScheduleAffineAdd, ScheduleUnavailable)):
            return
        if isinstance(current, ScheduleSequence):
            for child in current.children:
                collect(child)
            return
        if isinstance(current, ScheduleRepeat):
            collect(current.body)
            return
        raise TypeError(type(current).__name__)

    collect(node)
    available = [event for event in memory_events if event.stream_index not in bound_indices]
    remaining_multiplicity = {event.stream_index: event.multiplicity for event in available}

    def bind(current: ScheduleNode, multiplier: int) -> ScheduleNode:
        if isinstance(current, ScheduleInstruction):
            if current.opcode not in MEMORY_OPS or current.memory_stream_index is not None:
                return current
            matches = [
                event
                for event in available
                if event.stage == current.stage
                and event.transfer.opcode == current.opcode
                and remaining_multiplicity[event.stream_index] >= multiplier
            ]
            if not matches:
                raise ValueError(
                    "no exact DMA event matches unindexed schedule instruction "
                    f"{current.opcode} in {current.stage!r} with dynamic "
                    f"multiplicity {multiplier}"
                )
            event = matches[0]
            remaining_multiplicity[event.stream_index] -= multiplier
            if remaining_multiplicity[event.stream_index] == 0:
                available.remove(event)
            return replace(current, memory_stream_index=event.stream_index)
        if isinstance(current, (ScheduleAffineLoad, ScheduleAffineAdd, ScheduleUnavailable)):
            return current
        if isinstance(current, ScheduleSequence):
            return replace(
                current,
                children=tuple(bind(child, multiplier) for child in current.children),
            )
        if isinstance(current, ScheduleRepeat):
            body = bind(current.body, multiplier * current.count)
            assert isinstance(body, ScheduleSequence)
            return replace(current, body=body)
        raise TypeError(type(current).__name__)

    result = bind(node, 1)
    if available:
        remaining = [
            (
                event.stream_index,
                event.stage,
                event.transfer.opcode,
                remaining_multiplicity[event.stream_index],
            )
            for event in available
        ]
        raise ValueError(f"DMA events are not represented in the ordered schedule: {remaining!r}")
    return result


def _compress_explicit_hardware_loops(node: ScheduleNode) -> ScheduleNode:
    """Recover structured repeats from typed ``C_LOOP_*`` marker pairs.

    Some older lowering helpers emit typed loop marker instructions instead
    of :class:`HardwareLoop`. Their order and trip count are still exact, so
    treating those regions as counts-only loses useful scheduling information.
    This post-pass turns each balanced marker pair into the same compressed
    representation produced for a first-class ``HardwareLoop``.
    """
    if isinstance(
        node,
        (ScheduleInstruction, ScheduleAffineLoad, ScheduleAffineAdd, ScheduleUnavailable),
    ):
        return node
    if isinstance(node, ScheduleRepeat):
        if node.repeat_kind == "hardware_loop":
            body_children = node.body.children
            if not body_children or not (
                isinstance(body_children[-1], ScheduleInstruction) and body_children[-1].opcode == "C_LOOP_END"
            ):
                raise RawAsmCostError(f"hardware-loop repeat {node.name!r} has no trailing C_LOOP_END")
            prefix = _compress_explicit_hardware_loops(ScheduleSequence(body_children[:-1]))
            assert isinstance(prefix, ScheduleSequence)
            return replace(
                node,
                body=ScheduleSequence((*prefix.children, body_children[-1])),
            )
        return replace(
            node,
            body=_compress_explicit_hardware_loops(node.body),
        )
    if not isinstance(node, ScheduleSequence):
        raise TypeError(type(node).__name__)

    children = node.children
    compressed: list[ScheduleNode] = []
    index = 0
    while index < len(children):
        child = children[index]
        if not (isinstance(child, ScheduleInstruction) and child.opcode in {"C_LOOP_START", "C_LOOP_END"}):
            compressed.append(_compress_explicit_hardware_loops(child))
            index += 1
            continue
        if child.opcode == "C_LOOP_END":
            raise RawAsmCostError(f"unmatched typed C_LOOP_END in compressed schedule stage {child.stage!r}")

        # First-class HardwareLoop nodes are already represented as a start
        # instruction followed by a repeat whose body contains the loop end.
        if (
            index + 1 < len(children)
            and isinstance(children[index + 1], ScheduleRepeat)
            and children[index + 1].repeat_kind == "hardware_loop"
        ):
            compressed.append(child)
            compressed.append(_compress_explicit_hardware_loops(children[index + 1]))
            index += 2
            continue

        try:
            loop_count = int(child.args[-1])
        except (IndexError, ValueError) as exc:
            raise RawAsmCostError(f"cannot parse typed hardware-loop count from schedule node {child!r}") from exc
        if loop_count <= 0:
            raise RawAsmCostError(f"typed hardware-loop count must be positive, got {loop_count}")

        depth = 1
        end_index = index + 1
        while end_index < len(children):
            candidate = children[end_index]
            if isinstance(candidate, ScheduleInstruction):
                if candidate.opcode == "C_LOOP_START":
                    if (
                        end_index + 1 < len(children)
                        and isinstance(children[end_index + 1], ScheduleRepeat)
                        and children[end_index + 1].repeat_kind == "hardware_loop"
                    ):
                        # This complete first-class loop is nested inside the
                        # explicit loop but its end marker lives in its repeat
                        # body, not at the current sequence level.
                        end_index += 2
                        continue
                    depth += 1
                elif candidate.opcode == "C_LOOP_END":
                    depth -= 1
                    if depth == 0:
                        break
            end_index += 1
        if end_index == len(children):
            raise RawAsmCostError(f"unterminated typed C_LOOP_START in compressed schedule stage {child.stage!r}")

        end = children[end_index]
        assert isinstance(end, ScheduleInstruction) and end.opcode == "C_LOOP_END"
        body = _compress_explicit_hardware_loops(ScheduleSequence(children[index + 1 : end_index]))
        assert isinstance(body, ScheduleSequence)
        compressed.append(child)
        compressed.append(
            ScheduleRepeat(
                count=loop_count,
                body=ScheduleSequence((*body.children, end)),
                name=(child.args[0] if child.args else "explicit_hardware_loop"),
                repeat_kind="hardware_loop",
            )
        )
        index = end_index + 1
    return ScheduleSequence(tuple(compressed))


def optimize_cost_trace_loop_agu(
    trace: CostTrace,
    *,
    mode: str = "loop-agu-v1",
    build_energy_actions: bool = True,
) -> CostTrace:
    """Apply the six-stream loop AGU rewrite to an ordered cost trace.

    The pass mirrors the rendered-assembly rewrite: only contiguous in-place
    ``S_ADDI_INT`` chains at the current loop level are eligible, and a bound
    register may not be read or written later in the iteration.  Rebuilding
    stage counters from the transformed schedule keeps timing, power, and DMA
    consumers on one instruction inventory.
    """

    from compiler.aten.agu import (
        AGU_MAX_LOOP_DEPTH,
        AGU_MAX_STREAMS,
        AGU_MODE_LEGACY,
        AGU_MODE_LOOP_V1,
        AGU_REPEAT_MAX_PERIOD,
        AguStats,
        encode_agu_stride,
    )

    if mode not in {AGU_MODE_LEGACY, AGU_MODE_LOOP_V1}:
        raise ValueError(f"unsupported address_generation_mode={mode!r}")
    stats = AguStats(agu_mode=mode)
    if mode == AGU_MODE_LEGACY:
        trace.metadata.update(stats.as_dict())
        trace.metadata["address_generation_mode"] = mode
        if not trace.schedule_unavailable_reasons:
            rebuild_parallel_kernel_census(trace)
        return trace
    if trace.schedule_unavailable_reasons:
        stats.agu_fallback_reasons["schedule_unavailable"] += 1
        trace.metadata.update(stats.as_dict())
        trace.metadata["address_generation_mode"] = mode
        return trace
    marker_stages: Counter[str] = Counter()

    def unique_parallel_kernel_owner(
        node: ScheduleNode,
    ) -> ParallelKernelTag | None:
        owners: set[ParallelKernelTag] = set()

        def visit(current: ScheduleNode) -> None:
            if isinstance(
                current,
                (
                    ScheduleInstruction,
                    ScheduleAffineLoad,
                    ScheduleAffineAdd,
                    ScheduleUnavailable,
                ),
            ):
                if current.parallel_kernel is not None:
                    owners.add(current.parallel_kernel)
                return
            if isinstance(current, ScheduleSequence):
                for child in current.children:
                    visit(child)
                return
            if isinstance(current, ScheduleRepeat):
                visit(current.body)
                return
            raise TypeError(type(current).__name__)

        visit(node)
        return next(iter(owners)) if len(owners) == 1 else None

    def mentions(instruction: ScheduleInstruction, register: str) -> bool:
        return register in instruction.args

    def static_words(node: ScheduleNode) -> int | None:
        if isinstance(node, ScheduleInstruction):
            return 1
        if isinstance(node, ScheduleSequence):
            values = [static_words(child) for child in node.children]
            return None if any(value is None for value in values) else sum(values)
        if isinstance(node, ScheduleRepeat):
            return static_words(node.body)
        # Affine and unavailable nodes do not have one stable rendered width.
        return None

    def candidate_groups(
        children: list[ScheduleNode],
    ) -> list[tuple[str, int, list[int]]]:
        direct = [child for child in children if isinstance(child, ScheduleInstruction)]
        candidates: list[tuple[str, int, list[int]]] = []
        index = 0
        while index < len(children):
            child = children[index]
            if not (
                isinstance(child, ScheduleInstruction)
                and child.opcode == "S_ADDI_INT"
                and len(child.args) == 3
                and child.args[0] == child.args[1]
                and child.args[0].startswith("gp")
                and child.args[0] != "gp0"
            ):
                index += 1
                continue
            register = child.args[0]
            stride = 0
            indices: list[int] = []
            while index < len(children):
                current = children[index]
                if not (
                    isinstance(current, ScheduleInstruction)
                    and current.opcode == "S_ADDI_INT"
                    and len(current.args) == 3
                    and current.args[0] == register
                    and current.args[1] == register
                ):
                    break
                try:
                    stride += int(current.args[2], 0)
                except ValueError:
                    indices = []
                    break
                indices.append(index)
                index += 1
            if indices and stride and encode_agu_stride(stride) is not None:
                last = indices[-1]
                later = [item for item in children[last + 1 :] if isinstance(item, ScheduleInstruction)]
                if not any(mentions(item, register) for item in later):
                    candidates.append((register, stride, indices))
            elif not indices:
                index += 1
        candidates.sort(key=lambda item: (-len(item[2]), item[0]))
        return candidates[:AGU_MAX_STREAMS]

    def best_exact_repeat(
        children: list[ScheduleNode],
    ) -> tuple[int, int, int] | None:
        if not all(isinstance(child, ScheduleInstruction) for child in children):
            return None
        instructions = [child for child in children if isinstance(child, ScheduleInstruction)]

        def semantic_key(
            instruction: ScheduleInstruction,
        ) -> tuple[str, tuple[str, ...], ParallelKernelTag | None]:
            # The rendered-assembly AGU pass sees only opcode and operands.
            # Stage is reporting metadata and must not prevent refolding an
            # otherwise byte-identical instruction microkernel.
            return (
                instruction.opcode,
                instruction.args,
                instruction.parallel_kernel,
            )

        keys = tuple(semantic_key(item) for item in instructions)
        removable_cache: dict[
            tuple[
                tuple[
                    str,
                    tuple[str, ...],
                    ParallelKernelTag | None,
                ],
                ...,
            ],
            tuple[int, int],
        ] = {}
        best: tuple[int, int, int] | None = None
        best_savings = 0
        instruction_count = len(instructions)
        max_global_period = min(
            AGU_REPEAT_MAX_PERIOD,
            instruction_count // 2,
        )
        for period in range(1, max_global_period + 1):
            # A repeated block of width ``period`` exists at ``start`` iff
            # keys[start:start+period] equals the following block. Scan the
            # equivalent element-wise condition keys[i] == keys[i+period]
            # once per period. Within one equality run, starts separated by a
            # full period have the same block and strictly fewer repeats, so
            # only the first start for each phase can improve the optimum.
            compare_limit = instruction_count - period
            compare_index = 0
            while compare_index < compare_limit:
                if keys[compare_index] != keys[compare_index + period]:
                    compare_index += 1
                    continue
                run_start = compare_index
                compare_index += 1
                while (
                    compare_index < compare_limit
                    and keys[compare_index]
                    == keys[compare_index + period]
                ):
                    compare_index += 1
                run_end = compare_index
                if run_end - run_start < period:
                    continue
                last_candidate = min(
                    run_start + period,
                    run_end - period + 1,
                )
                for start_index in range(run_start, last_candidate):
                    repeat_count = (
                        1 + (run_end - start_index) // period
                    )
                    if repeat_count < 2:
                        continue
                    block = instructions[
                        start_index : start_index + period
                    ]
                    block_keys = keys[
                        start_index : start_index + period
                    ]
                    cached_removal = removable_cache.get(block_keys)
                    if cached_removal is None:
                        candidates = candidate_groups(list(block))
                        removed = sum(
                            len(indices)
                            for _, _, indices in candidates
                        )
                        candidate_count = len(candidates)
                        removable_cache[block_keys] = (
                            removed,
                            candidate_count,
                        )
                    else:
                        removed, candidate_count = cached_removal
                    savings = (
                        repeat_count * removed - (candidate_count + 2)
                    )
                    candidate = (start_index, period, repeat_count)
                    if (
                        savings > best_savings
                        or (
                            savings == best_savings
                            and savings > 0
                            and best is not None
                            and candidate[:2] < best[:2]
                        )
                    ):
                        best_savings = savings
                        best = candidate
        return best

    def refold_sequence(
        sequence: ScheduleSequence,
        *,
        depth: int,
    ) -> ScheduleSequence:
        recursively_folded: list[ScheduleNode] = []
        index = 0
        while index < len(sequence.children):
            child = sequence.children[index]
            if (
                isinstance(child, ScheduleInstruction)
                and child.opcode == "C_LOOP_START"
                and index + 1 < len(sequence.children)
                and isinstance(sequence.children[index + 1], ScheduleRepeat)
                and sequence.children[index + 1].repeat_kind == "hardware_loop"
            ):
                repeat = sequence.children[index + 1]
                body = refold_sequence(repeat.body, depth=depth + 1)
                if depth + 1 < AGU_MAX_LOOP_DEPTH:
                    body_children = list(body.children)
                    marker = (
                        body_children.pop()
                        if body_children
                        and isinstance(body_children[-1], ScheduleInstruction)
                        and body_children[-1].opcode == "C_LOOP_END"
                        else None
                    )
                    if marker is not None:
                        candidate = best_exact_repeat(body_children)
                        if candidate is not None:
                            start_index, period, repeat_count = candidate
                            block = tuple(body_children[start_index : start_index + period])
                            stage = (
                                block[0].stage if block and isinstance(block[0], ScheduleInstruction) else child.stage
                            )
                            owner = (
                                block[0].parallel_kernel
                                if block
                                and isinstance(
                                    block[0],
                                    ScheduleInstruction,
                                )
                                else child.parallel_kernel
                            )
                            nested_marker = ScheduleInstruction(
                                "C_LOOP_END",
                                ("gp0",),
                                stage,
                                parallel_kernel=owner,
                            )
                            nested_start = ScheduleInstruction(
                                "C_LOOP_START",
                                ("gp0", str(repeat_count)),
                                stage,
                                parallel_kernel=owner,
                            )
                            nested_repeat = ScheduleRepeat(
                                repeat_count,
                                ScheduleSequence((*block, nested_marker)),
                                "agu_refolded_microkernel",
                                "hardware_loop",
                            )
                            body_children = [
                                *body_children[:start_index],
                                nested_start,
                                nested_repeat,
                                *body_children[start_index + repeat_count * period :],
                            ]
                            stats.agu_refolded_loop_count += 1
                            stats.agu_refolded_instruction_count += repeat_count * period - period
                        body_children.append(marker)
                        body = ScheduleSequence(tuple(body_children))
                recursively_folded.extend((child, replace(repeat, body=body)))
                index += 2
                continue
            if isinstance(child, ScheduleSequence):
                child = refold_sequence(child, depth=depth)
                # Rendering erases purely structural sequence boundaries.
                # Flatten them here so repeat discovery sees the same
                # contiguous instruction stream as the assembly pass.
                recursively_folded.extend(child.children)
                index += 1
                continue
            elif isinstance(child, ScheduleRepeat):
                child = replace(
                    child,
                    body=refold_sequence(child.body, depth=depth),
                )
                if (
                    child.repeat_kind == "compile_time"
                    and child.count > 1
                    and depth < AGU_MAX_LOOP_DEPTH
                    and all(isinstance(item, ScheduleInstruction) for item in child.body.children)
                    and candidate_groups(list(child.body.children))
                ):
                    body_children = tuple(child.body.children)
                    stage = body_children[0].stage
                    owner = (
                        body_children[0].parallel_kernel
                        or unique_parallel_kernel_owner(child.body)
                    )
                    marker = ScheduleInstruction(
                        "C_LOOP_END",
                        ("gp0",),
                        stage,
                        parallel_kernel=owner,
                    )
                    recursively_folded.extend(
                        (
                            ScheduleInstruction(
                                "C_LOOP_START",
                                ("gp0", str(child.count)),
                                stage,
                                parallel_kernel=owner,
                            ),
                            ScheduleRepeat(
                                child.count,
                                ScheduleSequence((*body_children, marker)),
                                child.name,
                                "hardware_loop",
                            ),
                        )
                    )
                    stats.agu_refolded_loop_count += 1
                    stats.agu_refolded_instruction_count += (child.count - 1) * len(body_children)
                    index += 1
                    continue
            recursively_folded.append(child)
            index += 1
        if depth >= AGU_MAX_LOOP_DEPTH:
            return ScheduleSequence(tuple(recursively_folded))

        # Match ``agu._refold_exact_repeats`` on every contiguous rendered
        # instruction run, not only inside an existing hardware loop. Native
        # lowering contains statically repeated row microkernels at sequence
        # boundaries; the assembly pass can see and refold those runs.
        result: list[ScheduleNode] = []
        index = 0
        while index < len(recursively_folded):
            if not isinstance(recursively_folded[index], ScheduleInstruction):
                result.append(recursively_folded[index])
                index += 1
                continue
            end = index
            while end < len(recursively_folded) and isinstance(recursively_folded[end], ScheduleInstruction):
                end += 1
            run = [item for item in recursively_folded[index:end] if isinstance(item, ScheduleInstruction)]
            candidate = best_exact_repeat(run)
            if candidate is None:
                result.extend(run)
                index = end
                continue
            start_index, period, repeat_count = candidate
            if start_index:
                result.extend(
                    refold_sequence(
                        ScheduleSequence(tuple(run[:start_index])),
                        depth=depth,
                    ).children
                )
            block = tuple(run[start_index : start_index + period])
            stage = block[0].stage
            owner = (
                block[0].parallel_kernel
                or unique_parallel_kernel_owner(ScheduleSequence(block))
            )
            marker = ScheduleInstruction(
                "C_LOOP_END",
                ("gp0",),
                stage,
                parallel_kernel=owner,
            )
            result.extend(
                (
                    ScheduleInstruction(
                        "C_LOOP_START",
                        ("gp0", str(repeat_count)),
                        stage,
                        parallel_kernel=owner,
                    ),
                    ScheduleRepeat(
                        repeat_count,
                        ScheduleSequence((*block, marker)),
                        "agu_refolded_microkernel",
                        "hardware_loop",
                    ),
                )
            )
            stats.agu_refolded_loop_count += 1
            stats.agu_refolded_instruction_count += repeat_count * period - period
            suffix = run[start_index + repeat_count * period :]
            if suffix:
                result.extend(
                    refold_sequence(
                        ScheduleSequence(tuple(suffix)),
                        depth=depth,
                    ).children
                )
            index = end
        return ScheduleSequence(tuple(result))

    def optimize_repeat(
        start: ScheduleInstruction,
        repeat: ScheduleRepeat,
    ) -> tuple[tuple[ScheduleNode, ...], int]:
        body = rewrite(repeat.body)
        assert isinstance(body, ScheduleSequence)
        children = list(body.children)
        if not children or not (isinstance(children[-1], ScheduleInstruction) and children[-1].opcode == "C_LOOP_END"):
            stats.agu_fallback_reasons["missing_loop_end_marker"] += 1
            return (start, replace(repeat, body=body)), 0
        marker = children.pop()
        candidates = candidate_groups(children)
        candidates = sorted(
            candidates,
            key=lambda item: (
                -(repeat.count * len(item[2]) - 1),
                item[0],
            ),
        )[:AGU_MAX_STREAMS]
        removed_per_iteration = sum(len(indices) for _, _, indices in candidates)
        setup_delta = len(candidates) + 1
        projected = repeat.count * (removed_per_iteration + 1) - setup_delta
        if projected <= 0:
            stats.agu_fallback_reasons["not_profitable"] += 1
            return (start, replace(repeat, body=body)), 0
        removed = {index for _, _, indices in candidates for index in indices}
        optimized_body = ScheduleSequence(tuple(child for index, child in enumerate(children) if index not in removed))
        body_words = static_words(optimized_body)
        if body_words is None or not 0 < body_words < (1 << 22):
            stats.agu_fallback_reasons["body_length_unavailable"] += 1
            return (start, replace(repeat, body=body)), 0

        setup_nodes: list[ScheduleNode] = []
        owner = start.parallel_kernel or unique_parallel_kernel_owner(body)
        for register, stride, indices in candidates:
            encoded = encode_agu_stride(stride)
            assert encoded is not None
            setup_nodes.append(
                ScheduleInstruction(
                    "C_AGU_BIND",
                    (register, str(encoded)),
                    start.stage,
                    parallel_kernel=owner,
                )
            )
            stats.agu_affine_updates_elided += repeat.count
            stats.agu_large_immediate_chunks_elided += repeat.count * (len(indices) - 1)
        setup_nodes.extend(
            (
                ScheduleInstruction(
                    "C_AGU_LOOP_LEN",
                    (str(body_words),),
                    start.stage,
                    parallel_kernel=owner,
                ),
                ScheduleInstruction(
                    "C_LOOP_START_AGU",
                    start.args,
                    start.stage,
                    parallel_kernel=owner,
                ),
                ScheduleRepeat(
                    count=repeat.count,
                    body=optimized_body,
                    name=repeat.name,
                    repeat_kind="agu_hardware_loop",
                ),
            )
        )
        stats.agu_loop_count += 1
        stats.agu_stream_count_histogram[len(candidates)] += 1
        stats.dynamic_loop_end_elided += repeat.count
        stats.agu_setup_instruction_count += setup_delta
        stats.agu_projected_cycle_savings += projected
        marker_stages[marker.stage] += 1
        return tuple(setup_nodes), 1

    def rewrite(node: ScheduleNode) -> ScheduleNode:
        if isinstance(node, ScheduleSequence):
            rewritten: list[ScheduleNode] = []
            children = node.children
            index = 0
            while index < len(children):
                child = children[index]
                if (
                    isinstance(child, ScheduleInstruction)
                    and child.opcode == "C_LOOP_START"
                    and index + 1 < len(children)
                    and isinstance(children[index + 1], ScheduleRepeat)
                    and children[index + 1].repeat_kind == "hardware_loop"
                ):
                    replacement, _ = optimize_repeat(child, children[index + 1])
                    rewritten.extend(replacement)
                    index += 2
                    continue
                rewritten.append(rewrite(child))
                index += 1
            return ScheduleSequence(tuple(rewritten))
        if isinstance(node, ScheduleRepeat):
            return replace(node, body=rewrite(node.body))
        return node

    refolded = refold_sequence(trace.schedule, depth=0)
    transformed = rewrite(refolded)
    assert isinstance(transformed, ScheduleSequence)
    trace.schedule = transformed

    def filter_stage(node: ScheduleNode, stage_name: str) -> ScheduleNode:
        if isinstance(
            node,
            (ScheduleInstruction, ScheduleAffineLoad, ScheduleAffineAdd),
        ):
            return node if node.stage == stage_name else ScheduleSequence()
        if isinstance(node, ScheduleUnavailable):
            return node if node.stage == stage_name else ScheduleSequence()
        if isinstance(node, ScheduleSequence):
            return ScheduleSequence(tuple(filter_stage(child, stage_name) for child in node.children))
        if isinstance(node, ScheduleRepeat):
            return replace(node, body=filter_stage(node.body, stage_name))
        raise TypeError(type(node).__name__)

    def one_static_copy(node: ScheduleNode) -> ScheduleNode:
        if isinstance(node, ScheduleSequence):
            return ScheduleSequence(tuple(one_static_copy(child) for child in node.children))
        if isinstance(node, ScheduleRepeat):
            count = 1 if node.repeat_kind in {"hardware_loop", "agu_hardware_loop"} else node.count
            return replace(node, count=count, body=one_static_copy(node.body))
        return node

    stage_names = set(trace.stages)
    stage_names.update(marker_stages)
    dynamic_by_stage = {
        stage_name: _schedule_opcode_counts(filter_stage(trace.schedule, stage_name)) for stage_name in stage_names
    }
    static_schedule = one_static_copy(trace.schedule)
    static_by_stage = {
        stage_name: _schedule_opcode_counts(filter_stage(static_schedule, stage_name)) for stage_name in stage_names
    }
    # The zero-overhead marker remains in the binary but not in dynamic flow.
    for stage_name, count in marker_stages.items():
        static_by_stage[stage_name]["C_LOOP_END"] += count

    trace.dynamic_opcodes = Counter()
    trace.static_opcodes = Counter()
    for stage_name in stage_names:
        stage = trace.stages[stage_name]
        stage.dynamic_opcodes = dynamic_by_stage[stage_name]
        stage.static_opcodes = static_by_stage[stage_name]
        trace.dynamic_opcodes.update(stage.dynamic_opcodes)
        trace.static_opcodes.update(stage.static_opcodes)
        stage.energy_actions.clear()

    derived = _schedule_opcode_counts(trace.schedule)
    if derived != trace.dynamic_opcodes:
        raise ValueError(
            f"AGU schedule opcode counts drifted from CostTrace: schedule={derived}, trace={trace.dynamic_opcodes}"
        )
    if build_energy_actions:
        trace.energy_actions = _build_energy_actions(trace)
        for action in trace.energy_actions:
            trace.stages[action.stage].energy_actions.append(action)
    stats.agu_residual_s_addi = int(trace.dynamic_opcodes.get("S_ADDI_INT", 0))
    trace.metadata.update(stats.as_dict())
    trace.metadata["address_generation_mode"] = mode
    trace.metadata["agu_isa_version"] = "loop_agu_v1"
    rebuild_parallel_kernel_census(trace)
    return trace


def _logical_dma_metadata(transfer: DmaTransfer) -> DmaTransfer:
    """Populate schema-v3 logical fields without changing rendered ISA.

    Lowerings that know an allocation-relative offset provide it explicitly.
    Legacy symbolic call sites remain valid and use their reference MXFP8
    addresses as a deterministic logical coordinate system.
    """
    source = transfer.source or "anonymous"
    return replace(
        transfer,
        memory_object=transfer.memory_object or source,
        precision_role=transfer.precision_role or transfer.precision,
        logical_element_offset=(
            transfer.element_base if transfer.logical_element_offset is None else transfer.logical_element_offset
        ),
        logical_scale_offset=(
            transfer.scale_base if transfer.logical_scale_offset is None else transfer.logical_scale_offset
        ),
        logical_stride=(transfer.stride if transfer.logical_stride is None else transfer.logical_stride),
    )


def _logical_repeat_axis(axis: RepeatAxis) -> RepeatAxis:
    return replace(
        axis,
        logical_element_delta=(
            axis.element_base_delta if axis.logical_element_delta is None else axis.logical_element_delta
        ),
        logical_scale_delta=(axis.scale_base_delta if axis.logical_scale_delta is None else axis.logical_scale_delta),
    )


@dataclass
class StageCost:
    static_opcodes: Counter[str] = field(default_factory=Counter)
    dynamic_opcodes: Counter[str] = field(default_factory=Counter)
    energy_actions: list[EnergyAction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "static_opcodes": dict(sorted(self.static_opcodes.items())),
            "dynamic_opcodes": dict(sorted(self.dynamic_opcodes.items())),
            "static_instruction_count": sum(self.static_opcodes.values()),
            "dynamic_instruction_count": sum(self.dynamic_opcodes.values()),
            "energy_actions": [action.to_dict() for action in self.energy_actions],
        }


@dataclass
class CostTrace:
    schema_version: ClassVar[int] = 7
    static_opcodes: Counter[str] = field(default_factory=Counter)
    dynamic_opcodes: Counter[str] = field(default_factory=Counter)
    memory_events: list[MemoryEvent] = field(default_factory=list)
    stages: dict[str, StageCost] = field(default_factory=lambda: defaultdict(StageCost))
    schedule: ScheduleSequence = field(default_factory=ScheduleSequence)
    schedule_unavailable_reasons: Counter[str] = field(default_factory=Counter)
    energy_actions: list[EnergyAction] = field(default_factory=list)
    parallel_kernel_census: list[ParallelKernelCensusEntry] = field(
        default_factory=list
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def static_instruction_count(self) -> int:
        return sum(self.static_opcodes.values())

    @property
    def dynamic_instruction_count(self) -> int:
        return sum(self.dynamic_opcodes.values())

    def to_dict(self) -> dict[str, Any]:
        categories: Counter[str] = Counter()
        for opcode, count in self.dynamic_opcodes.items():
            categories[opcode_category(opcode)] += count
        unavailable = _schedule_unavailable_counts(self.schedule)
        unavailable_count = sum(unavailable.values())
        ordered_count = max(0, self.dynamic_instruction_count - unavailable_count)
        return {
            "schema_version": self.schema_version,
            **self.metadata,
            "static_instruction_count": self.static_instruction_count,
            "dynamic_instruction_count": self.dynamic_instruction_count,
            "static_opcodes": dict(sorted(self.static_opcodes.items())),
            "dynamic_opcodes": dict(sorted(self.dynamic_opcodes.items())),
            "instruction_categories": dict(sorted(categories.items())),
            "compressed_memory_events": [event.to_dict() for event in self.memory_events],
            "energy_actions": [action.to_dict() for action in self.energy_actions],
            "parallel_kernel_census": [
                entry.to_dict() for entry in self.parallel_kernel_census
            ],
            "stage_breakdown": {name: stage.to_dict() for name, stage in sorted(self.stages.items())},
            "compressed_schedule": self.schedule.to_dict(),
            "schedule_fidelity": ("unavailable" if self.schedule_unavailable_reasons else "ordered_compressed"),
            "schedule_unavailable_reasons": dict(sorted(self.schedule_unavailable_reasons.items())),
            "schedule_coverage": {
                "ordered_dynamic_instructions": ordered_count,
                "unavailable_dynamic_instructions": unavailable_count,
                "ordered_fraction": (
                    1.0 if self.dynamic_instruction_count == 0 else ordered_count / self.dynamic_instruction_count
                ),
                "unavailable_by_reason": dict(sorted(unavailable.items())),
            },
        }


class CostSink:
    """Algebraically count a symbolic program without rendering it."""

    def __init__(
        self,
        *,
        strict_raw: bool = True,
        granularity: str = COST_TRACE_GRANULARITY_DETAILED,
        address_generation_mode: str = "legacy",
    ):
        if granularity not in COST_TRACE_GRANULARITIES:
            raise ValueError(
                f"cost trace granularity must be one of {sorted(COST_TRACE_GRANULARITIES)}, got {granularity!r}"
            )
        self.strict_raw = strict_raw
        self.granularity = granularity
        self.address_generation_mode = address_generation_mode
        self.trace = CostTrace()
        self._raw_loop_stack: list[int] = []
        self._typed_loop_stack: list[int] = []
        self._raw_straight_line_cache: dict[str, Counter[str]] = {}
        self._next_memory_stream_index = 0
        self._schedule_children: list[ScheduleNode] = []
        self._summary_hold_depth = 0
        self._summary_baseline_static: Counter[str] | None = None
        self._summary_baseline_dynamic: Counter[str] | None = None
        self._summary_baseline_stages: dict[str, tuple[Counter[str], Counter[str]]] | None = None
        self._summary_energy_actions: list[EnergyAction] = []
        self._summary_activity_variants: dict[
            tuple[str, ParallelKernelTag | None],
            Counter[tuple[str, tuple[str, ...], str, int]],
        ] = defaultdict(Counter)
        self._summary_parameterized_variants: Counter[tuple[str, tuple[str, ...]]] = Counter()
        self._summary_stage_parameterized_variants: dict[str, Counter[tuple[str, tuple[str, ...]]]] = defaultdict(
            Counter
        )
        self._summary_agu_metadata: dict[str, Any] = {}
        self._summary_fragment_cache: OrderedDict[tuple[Any, ...], CostTrace] = OrderedDict()
        self._summary_templates: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._summary_template_replay_counts: Counter[
            tuple[
                tuple[Any, ...],
                int,
                int,
                tuple[tuple[str, str], ...],
            ]
        ] = Counter()
        self._parallel_kernel_stack: list[ParallelKernelTag] = []

    @property
    def active_parallel_kernel(self) -> ParallelKernelTag | None:
        return (
            self._parallel_kernel_stack[-1]
            if self._parallel_kernel_stack
            else None
        )

    @contextmanager
    def parallel_kernel(
        self,
        *,
        kernel: str,
        tp_semantics: str,
        cp_semantics: str,
        ep_semantics: str = "none",
        logical_rows: int = 0,
        logical_m: int = 0,
        logical_n: int = 0,
        logical_k: int = 0,
        matrix_mlen: int = 0,
        matrix_blen: int = 0,
    ):
        tag = ParallelKernelTag(
            kernel=kernel,
            tp_semantics=tp_semantics,
            cp_semantics=cp_semantics,
            ep_semantics=ep_semantics,
            logical_rows=logical_rows,
            logical_m=logical_m,
            logical_n=logical_n,
            logical_k=logical_k,
            matrix_mlen=matrix_mlen,
            matrix_blen=matrix_blen,
        )
        self._parallel_kernel_stack.append(tag)
        try:
            yield
        finally:
            popped = self._parallel_kernel_stack.pop()
            if popped != tag:
                raise RuntimeError("parallel kernel context stack corrupted")

    @property
    def _summary_enabled(self) -> bool:
        return self.granularity == COST_TRACE_GRANULARITY_AFFINE_BLOCK_SUMMARY_V1

    def _begin_summary_fragment(self) -> None:
        if not self._summary_enabled or self._summary_baseline_static is not None:
            return
        self._summary_baseline_static = Counter(self.trace.static_opcodes)
        self._summary_baseline_dynamic = Counter(self.trace.dynamic_opcodes)
        self._summary_baseline_stages = {
            name: (Counter(stage.static_opcodes), Counter(stage.dynamic_opcodes))
            for name, stage in self.trace.stages.items()
        }

    @staticmethod
    def _counter_delta(current: Mapping[str, int], baseline: Mapping[str, int]) -> Counter[str]:
        return Counter(
            {
                key: int(current.get(key, 0)) - int(baseline.get(key, 0))
                for key in set(current) | set(baseline)
                if int(current.get(key, 0)) != int(baseline.get(key, 0))
            }
        )

    @staticmethod
    def _replace_counter_delta(
        target: Counter[str],
        old_delta: Mapping[str, int],
        new_values: Mapping[str, int],
    ) -> None:
        target.subtract(old_delta)
        target.update(new_values)
        for key in tuple(target):
            if target[key] == 0:
                del target[key]

    def _merge_summary_agu_metadata(self, metadata: Mapping[str, Any]) -> None:
        additive = {
            "agu_loop_count",
            "agu_affine_updates_elided",
            "agu_large_immediate_chunks_elided",
            "dynamic_loop_end_elided",
            "agu_setup_instruction_count",
            "agu_projected_cycle_savings",
            "agu_refolded_loop_count",
            "agu_refolded_instruction_count",
            "agu_residual_s_addi",
        }
        counter_maps = {
            "agu_stream_count_histogram",
            "agu_fallback_reasons",
        }
        for key in additive:
            self._summary_agu_metadata[key] = int(self._summary_agu_metadata.get(key, 0)) + int(metadata.get(key, 0))
        for key in counter_maps:
            merged = Counter(self._summary_agu_metadata.get(key, {}))
            merged.update({str(name): int(value) for name, value in metadata.get(key, {}).items()})
            self._summary_agu_metadata[key] = dict(sorted(merged.items()))
        self._summary_agu_metadata["agu_mode"] = self.address_generation_mode
        self._summary_agu_metadata["address_generation_mode"] = self.address_generation_mode
        if "agu_isa_version" in metadata:
            self._summary_agu_metadata["agu_isa_version"] = metadata["agu_isa_version"]

    @staticmethod
    def _nested_counter_delta(
        current: Mapping[str, Mapping[Any, int]],
        baseline: Mapping[str, Mapping[Any, int]],
    ) -> dict[str, Counter[Any]]:
        result: dict[str, Counter[Any]] = {}
        for name in set(current) | set(baseline):
            delta = Counter(current.get(name, {}))
            delta.subtract(baseline.get(name, {}))
            delta += Counter()
            if delta:
                result[name] = delta
        return result

    @classmethod
    def _summary_schedule_shape_key(cls, node: ScheduleNode) -> Any:
        """Normalize address constants that cannot affect loop-AGU decisions."""

        if isinstance(node, ScheduleInstruction):
            args = list(node.args)
            if node.opcode == "S_LUI_INT" and len(args) >= 2:
                args[1] = "<base>"
            elif node.opcode == "S_ADDI_INT" and len(args) >= 3 and args[1] == "gp0":
                # Immediate legalization already happened before this point.
                # The resulting opcode shape is therefore fixed; the concrete
                # base value is neither an affine loop update nor an AGU
                # eligibility input.
                args[2] = "<base>"
            return (
                "instruction",
                node.opcode,
                tuple(args),
                node.stage,
                node.memory_stream_index is not None,
                node.parallel_kernel,
            )
        if isinstance(node, ScheduleAffineLoad):
            return (
                "affine_load",
                node.register,
                "<base>",
                node.step,
                node.period,
                node.advance_every,
                node.stage,
                node.parallel_kernel,
            )
        if isinstance(node, ScheduleAffineAdd):
            return (
                "affine_add",
                node.destination,
                node.source,
                node.temp,
                "<base>",
                node.step,
                node.period,
                node.advance_every,
                node.stage,
                node.parallel_kernel,
            )
        if isinstance(node, ScheduleSequence):
            return (
                "sequence",
                tuple(cls._summary_schedule_shape_key(child) for child in node.children),
            )
        if isinstance(node, ScheduleRepeat):
            return (
                "repeat",
                node.count,
                node.name,
                node.repeat_kind,
                cls._summary_schedule_shape_key(node.body),
            )
        if isinstance(node, ScheduleUnavailable):
            return (
                "unavailable",
                node.reason,
                node.stage,
                node.dynamic_instruction_count,
                node.dynamic_opcodes,
                node.parallel_kernel,
            )
        raise TypeError(type(node).__name__)

    def replay_summary_template(
        self,
        key: tuple[Any, ...],
        *,
        count: int = 1,
        element_base_delta: int = 0,
        scale_base_delta: int = 0,
        memory_object_replacements: tuple[tuple[str, str], ...] = (),
    ) -> bool:
        """Replay a previously lowered algebraic kernel template."""

        if not self._summary_enabled:
            return False
        if count <= 0:
            raise ValueError(f"summary replay count must be positive, got {count}")
        template = self._summary_templates.get(key)
        if template is None:
            return False
        replacements = tuple(memory_object_replacements)
        self._replay_summary_template_memory(
            template,
            count=count,
            element_base_delta=int(element_base_delta),
            scale_base_delta=int(scale_base_delta),
            memory_object_replacements=replacements,
        )
        self._summary_template_replay_counts[
            (
                key,
                int(element_base_delta),
                int(scale_base_delta),
                replacements,
            )
        ] += count
        return True

    def _replay_summary_template_memory(
        self,
        template: Mapping[str, Any],
        *,
        count: int,
        element_base_delta: int,
        scale_base_delta: int,
        memory_object_replacements: tuple[tuple[str, str], ...],
    ) -> None:
        replacements = dict(memory_object_replacements)
        for event in template["memory_events"]:
            memory_object = event.transfer.memory_object
            source = event.transfer.source
            shifted_memory_object = replacements.get(memory_object, memory_object)
            if (
                shifted_memory_object == memory_object
                and memory_object is not None
                and memory_object.startswith("hbm:")
            ):
                parts = memory_object.split(":")
                if len(parts) == 3:
                    shifted_memory_object = f"hbm:{int(parts[1]) + element_base_delta}:{parts[2]}"
            transfer = replace(
                event.transfer,
                element_base=event.transfer.element_base + element_base_delta,
                scale_base=event.transfer.scale_base + scale_base_delta,
                memory_object=shifted_memory_object,
                source=replacements.get(source, source),
            )
            axes = event.enclosing_axes
            multiplicity = event.multiplicity
            if count > 1:
                axes = (
                    *axes,
                    RepeatAxis(
                        name="summary_template_replay",
                        count=count,
                        element_base_delta=element_base_delta,
                        scale_base_delta=scale_base_delta,
                    ),
                )
                multiplicity *= count
            self.add_memory_event(
                transfer=transfer,
                multiplicity=multiplicity,
                stage=event.stage,
                axes=axes,
                parallel_kernel=event.parallel_kernel,
            )

    def _apply_summary_template(
        self,
        key: tuple[Any, ...],
        multiplier: int,
        *,
        element_base_delta: int = 0,
        scale_base_delta: int = 0,
    ) -> None:
        template = self._summary_templates[key]
        self.trace.static_opcodes.update({name: count * multiplier for name, count in template["static"].items()})
        self.trace.dynamic_opcodes.update({name: count * multiplier for name, count in template["dynamic"].items()})
        for stage_name, (static_counts, dynamic_counts) in template["stages"].items():
            self.trace.stages[stage_name].static_opcodes.update(
                {name: count * multiplier for name, count in static_counts.items()}
            )
            self.trace.stages[stage_name].dynamic_opcodes.update(
                {name: count * multiplier for name, count in dynamic_counts.items()}
            )
        self._summary_energy_actions.extend(
            replace(
                action,
                count=action.count * multiplier,
                busy_cycles=action.busy_cycles * multiplier,
                bytes=action.bytes * multiplier,
            )
            for action in template["agu_actions"]
        )
        self._summary_parameterized_variants.update(
            {name: count * multiplier for name, count in template["parameterized"].items()}
        )
        for stage_name, variants in template["stage_parameterized"].items():
            self._summary_stage_parameterized_variants[stage_name].update(
                {name: count * multiplier for name, count in variants.items()}
            )
        for stage_name, variants in template["activity"].items():
            self._summary_activity_variants[stage_name].update(
                {name: count * multiplier for name, count in variants.items()}
            )
        self.trace.parallel_kernel_census.extend(
            replace(
                entry,
                count=entry.count * multiplier,
                multiplicity=entry.multiplicity * multiplier,
            )
            for entry in template["parallel_kernel_census"]
        )
        scaled_metadata: dict[str, Any] = {}
        for name, value in template["agu_metadata"].items():
            if isinstance(value, int):
                scaled_metadata[name] = value * multiplier
            elif isinstance(value, Mapping):
                scaled_metadata[name] = {item: int(count) * multiplier for item, count in value.items()}
            else:
                scaled_metadata[name] = value
        self._merge_summary_agu_metadata(scaled_metadata)

    @contextmanager
    def summary_template(self, key: tuple[Any, ...], *, allow_memory: bool = False):
        """Capture one exact lowering for later algebraic replay."""

        if not self._summary_enabled:
            yield
            return
        if key in self._summary_templates:
            raise ValueError(f"summary template {key!r} was already captured")
        before_static = Counter(self.trace.static_opcodes)
        before_dynamic = Counter(self.trace.dynamic_opcodes)
        before_stages = {
            name: (Counter(stage.static_opcodes), Counter(stage.dynamic_opcodes))
            for name, stage in self.trace.stages.items()
        }
        before_memory = len(self.trace.memory_events)
        before_parallel_census = len(self.trace.parallel_kernel_census)
        before_agu_actions = len(self._summary_energy_actions)
        before_parameterized = Counter(self._summary_parameterized_variants)
        before_stage_parameterized = {
            name: Counter(values) for name, values in self._summary_stage_parameterized_variants.items()
        }
        before_activity = {name: Counter(values) for name, values in self._summary_activity_variants.items()}
        before_agu_metadata = dict(self._summary_agu_metadata)
        self._summary_hold_depth += 1
        try:
            yield
        finally:
            self._summary_hold_depth -= 1
            self._flush_summary_fragment()
        if not allow_memory and len(self.trace.memory_events) != before_memory:
            raise ValueError("summary kernel templates cannot contain DMA events")
        stages: dict[str, tuple[Counter[str], Counter[str]]] = {}
        for stage_name in set(self.trace.stages) | set(before_stages):
            old_static, old_dynamic = before_stages.get(stage_name, (Counter(), Counter()))
            current = self.trace.stages.get(stage_name, StageCost())
            static_delta = self._counter_delta(current.static_opcodes, old_static)
            dynamic_delta = self._counter_delta(current.dynamic_opcodes, old_dynamic)
            if static_delta or dynamic_delta:
                stages[stage_name] = (static_delta, dynamic_delta)
        agu_metadata_delta: dict[str, Any] = {}
        for name, value in self._summary_agu_metadata.items():
            old = before_agu_metadata.get(name)
            if isinstance(value, int):
                agu_metadata_delta[name] = value - int(old or 0)
            elif isinstance(value, Mapping):
                delta = Counter(value)
                delta.subtract(old or {})
                delta += Counter()
                agu_metadata_delta[name] = dict(delta)
            elif value != old:
                agu_metadata_delta[name] = value
        self._summary_templates[key] = {
            "static": self._counter_delta(self.trace.static_opcodes, before_static),
            "dynamic": self._counter_delta(self.trace.dynamic_opcodes, before_dynamic),
            "stages": stages,
            "agu_actions": tuple(self._summary_energy_actions[before_agu_actions:]),
            "parameterized": self._counter_delta(self._summary_parameterized_variants, before_parameterized),
            "stage_parameterized": self._nested_counter_delta(
                self._summary_stage_parameterized_variants,
                before_stage_parameterized,
            ),
            "activity": self._nested_counter_delta(self._summary_activity_variants, before_activity),
            "agu_metadata": agu_metadata_delta,
            "memory_events": tuple(self.trace.memory_events[before_memory:]),
            "parallel_kernel_census": tuple(
                self.trace.parallel_kernel_census[before_parallel_census:]
            ),
        }

    def _flush_summary_fragment(self, *, force: bool = False) -> None:
        if not self._summary_enabled:
            return
        if not force and (self._summary_hold_depth or self._raw_loop_stack or self._typed_loop_stack):
            return
        if not self._schedule_children:
            self._summary_baseline_static = None
            self._summary_baseline_dynamic = None
            self._summary_baseline_stages = None
            return
        assert self._summary_baseline_static is not None
        assert self._summary_baseline_dynamic is not None
        assert self._summary_baseline_stages is not None

        schedule = ScheduleSequence(tuple(self._schedule_children))
        if self.trace.schedule_unavailable_reasons.get("explicit_loop_markers"):
            schedule = _compress_explicit_hardware_loops(schedule)
            assert isinstance(schedule, ScheduleSequence)

        old_static = self._counter_delta(self.trace.static_opcodes, self._summary_baseline_static)
        old_dynamic = self._counter_delta(self.trace.dynamic_opcodes, self._summary_baseline_dynamic)
        temporary = CostTrace(
            static_opcodes=Counter(old_static),
            dynamic_opcodes=Counter(old_dynamic),
            schedule=schedule,
        )
        stage_names = set(self.trace.stages) | set(self._summary_baseline_stages)
        for stage_name in stage_names:
            before_static, before_dynamic = self._summary_baseline_stages.get(stage_name, (Counter(), Counter()))
            current = self.trace.stages.get(stage_name, StageCost())
            static_delta = self._counter_delta(current.static_opcodes, before_static)
            dynamic_delta = self._counter_delta(current.dynamic_opcodes, before_dynamic)
            if static_delta or dynamic_delta:
                temporary.stages[stage_name] = StageCost(
                    static_opcodes=static_delta,
                    dynamic_opcodes=dynamic_delta,
                )

        cache_key = (
            self._summary_schedule_shape_key(schedule),
            tuple(sorted(old_static.items())),
            tuple(sorted(old_dynamic.items())),
            tuple(
                (
                    stage_name,
                    tuple(sorted(stage.static_opcodes.items())),
                    tuple(sorted(stage.dynamic_opcodes.items())),
                )
                for stage_name, stage in sorted(temporary.stages.items())
            ),
            self.address_generation_mode,
        )
        cached = self._summary_fragment_cache.get(cache_key)
        if cached is None:
            temporary = optimize_cost_trace_loop_agu(
                temporary,
                mode=self.address_generation_mode,
                build_energy_actions=False,
            )
            rebuild_parallel_kernel_census(temporary)
            self._summary_fragment_cache[cache_key] = temporary
            self._summary_fragment_cache.move_to_end(cache_key)
            while len(self._summary_fragment_cache) > 4096:
                self._summary_fragment_cache.popitem(last=False)
        else:
            temporary = cached
            self._summary_fragment_cache.move_to_end(cache_key)
        self.trace.parallel_kernel_census.extend(
            temporary.parallel_kernel_census
        )
        self._summary_energy_actions.extend(_agu_runtime_energy_actions(temporary.schedule))
        self._summary_parameterized_variants.update(
            schedule_instruction_variants(temporary.schedule, opcodes=MATRIX_COMPUTE_OPS | VECTOR_COMPUTE_OPS)
        )
        for stage_name, tag in _schedule_parallel_lineage_keys(
            temporary.schedule
        ):
            filtered = _filter_schedule_parallel_lineage(
                temporary.schedule,
                stage=stage_name,
                tag=tag,
            )
            counts = _schedule_opcode_counts(filtered)
            self._summary_activity_variants[(stage_name, tag)].update(
                schedule_instruction_activity_variants(
                    filtered,
                    opcodes={
                        opcode
                        for opcode, count in counts.items()
                        if count > 0 and opcode not in MEMORY_OPS
                    },
                    stage=stage_name,
                )
            )
        for stage_name in temporary.stages:
            self._summary_stage_parameterized_variants[stage_name].update(
                schedule_instruction_variants(
                    temporary.schedule,
                    opcodes=MATRIX_COMPUTE_OPS | VECTOR_COMPUTE_OPS,
                    stage=stage_name,
                )
            )
        self._merge_summary_agu_metadata(temporary.metadata)

        self._replace_counter_delta(self.trace.static_opcodes, old_static, temporary.static_opcodes)
        self._replace_counter_delta(self.trace.dynamic_opcodes, old_dynamic, temporary.dynamic_opcodes)
        all_stage_names = set(temporary.stages) | set(self._summary_baseline_stages)
        for stage_name in all_stage_names:
            before_static, before_dynamic = self._summary_baseline_stages.get(stage_name, (Counter(), Counter()))
            current = self.trace.stages[stage_name]
            stage_old_static = self._counter_delta(current.static_opcodes, before_static)
            stage_old_dynamic = self._counter_delta(current.dynamic_opcodes, before_dynamic)
            replacement = temporary.stages.get(stage_name, StageCost())
            self._replace_counter_delta(
                current.static_opcodes,
                stage_old_static,
                replacement.static_opcodes,
            )
            self._replace_counter_delta(
                current.dynamic_opcodes,
                stage_old_dynamic,
                replacement.dynamic_opcodes,
            )

        self._schedule_children.clear()
        self._summary_baseline_static = None
        self._summary_baseline_dynamic = None
        self._summary_baseline_stages = None

    def emit(self, value: IsaBuilder | Sequence | Iterable[AsmItem]) -> None:
        self._begin_summary_fragment()
        sequence = as_sequence(value)
        items = legalize_large_immediates(sequence.items)
        self._schedule_children.extend(
            self._visit(
                items,
                static_multiplier=1,
                dynamic_multiplier=1,
                stage="global",
                axes=(),
            )
        )
        self._flush_summary_fragment()

    def add_counts(
        self,
        *,
        static_opcodes: Mapping[str, int],
        dynamic_opcodes: Mapping[str, int],
        stage: str = "global",
        schedule_reason: str = "counts_only_kernel_summary",
    ) -> None:
        """Add an algebraically lowered kernel summary."""
        enclosing_multiplier = 1
        for count in (*self._raw_loop_stack, *self._typed_loop_stack):
            enclosing_multiplier *= count
        for opcode in set(static_opcodes) | set(dynamic_opcodes):
            static_count = int(static_opcodes.get(opcode, 0))
            dynamic_count = int(dynamic_opcodes.get(opcode, 0))
            if static_count < 0 or dynamic_count < 0:
                raise ValueError(f"negative cost count for {opcode}: {static_count}, {dynamic_count}")
            if static_count or dynamic_count:
                self._record(opcode, static_count, dynamic_count * enclosing_multiplier, stage)
        dynamic_instruction_count = sum(int(value) for value in dynamic_opcodes.values())
        if dynamic_instruction_count:
            reason = schedule_reason
            self.trace.schedule_unavailable_reasons[reason] += 1
            self._schedule_children.append(
                ScheduleUnavailable(
                    reason=reason,
                    stage=stage,
                    # Keep the local body count here. Any enclosing typed
                    # hardware loop is represented by ScheduleRepeat and
                    # applies its multiplicity exactly once during traversal.
                    dynamic_instruction_count=dynamic_instruction_count,
                    dynamic_opcodes=tuple(
                        sorted(
                            (
                                str(opcode),
                                int(value),
                            )
                            for opcode, value in dynamic_opcodes.items()
                            if int(value)
                        )
                    ),
                    parallel_kernel=self.active_parallel_kernel,
                )
            )

    @contextmanager
    def repeated_region(self, count: int, *, name: str, repeat_kind: str = "compile_time"):
        """Emit one ordered body and account for ``count`` identical copies.

        This is a cost-only equivalent of compiler unrolling.  It is intended
        for latency summaries whose dynamic addresses do not change opcode or
        DMA geometry.  DMA inside the region is rejected so every physical
        memory stream remains represented explicitly by ``MemoryEvent``.
        """
        if count <= 0:
            raise ValueError(f"repeated-region count must be positive, got {count}")
        self._begin_summary_fragment()
        self._summary_hold_depth += 1
        before_schedule = len(self._schedule_children)
        before_streams = len(self.trace.memory_events)
        before_static = Counter(self.trace.static_opcodes)
        before_dynamic = Counter(self.trace.dynamic_opcodes)
        before_stages = {
            stage: (Counter(cost.static_opcodes), Counter(cost.dynamic_opcodes))
            for stage, cost in self.trace.stages.items()
        }
        try:
            yield
        finally:
            self._summary_hold_depth -= 1
        if len(self.trace.memory_events) != before_streams:
            raise ValueError("repeated cost regions cannot contain DMA events")
        body = tuple(self._schedule_children[before_schedule:])
        del self._schedule_children[before_schedule:]
        self._schedule_children.append(
            ScheduleRepeat(
                count=count,
                body=ScheduleSequence(body),
                name=name,
                repeat_kind=repeat_kind,
            )
        )
        if count == 1:
            self._flush_summary_fragment()
            return
        static_delta = self.trace.static_opcodes - before_static
        dynamic_delta = self.trace.dynamic_opcodes - before_dynamic
        self.trace.static_opcodes.update({opcode: value * (count - 1) for opcode, value in static_delta.items()})
        self.trace.dynamic_opcodes.update({opcode: value * (count - 1) for opcode, value in dynamic_delta.items()})
        for stage, cost in self.trace.stages.items():
            old_static, old_dynamic = before_stages.get(stage, (Counter(), Counter()))
            stage_static_delta = cost.static_opcodes - old_static
            stage_dynamic_delta = cost.dynamic_opcodes - old_dynamic
            cost.static_opcodes.update({opcode: value * (count - 1) for opcode, value in stage_static_delta.items()})
            cost.dynamic_opcodes.update({opcode: value * (count - 1) for opcode, value in stage_dynamic_delta.items()})
        self._flush_summary_fragment()

    def add_ordered_schedule(
        self,
        *,
        static_opcodes: Mapping[str, int],
        dynamic_opcodes: Mapping[str, int],
        schedule: ScheduleSequence,
        stage: str = "global",
        memory_stream_indices: tuple[int, ...] = (),
    ) -> None:
        """Add an algebraic kernel summary whose program order is preserved.

        Kernel schedule builders use local DMA stream ordinals so they remain
        independent of the surrounding trace. They are remapped here after
        the corresponding :class:`MemoryEvent` objects have been appended.
        """
        self._begin_summary_fragment()
        enclosing_multiplier = 1
        for count in (*self._raw_loop_stack, *self._typed_loop_stack):
            enclosing_multiplier *= count
        local_dynamic = Counter({opcode: int(count) for opcode, count in dynamic_opcodes.items() if int(count)})
        derived = _schedule_opcode_counts(schedule)
        if derived != local_dynamic:
            raise ValueError(
                f"ordered kernel schedule drifted from algebraic counts: schedule={derived}, counts={local_dynamic}"
            )
        for opcode in set(static_opcodes) | set(dynamic_opcodes):
            static_count = int(static_opcodes.get(opcode, 0))
            dynamic_count = int(dynamic_opcodes.get(opcode, 0))
            if static_count < 0 or dynamic_count < 0:
                raise ValueError(f"negative cost count for {opcode}: {static_count}, {dynamic_count}")
            if static_count or dynamic_count:
                self._record(
                    opcode,
                    static_count,
                    dynamic_count * enclosing_multiplier,
                    stage,
                )
        tagged = _retag_schedule_parallel_kernel(
            _retag_schedule_stage(schedule, stage),
            self.active_parallel_kernel,
        )
        remapped = _remap_schedule_memory_streams(
            tagged,
            memory_stream_indices,
        )
        assert isinstance(remapped, ScheduleSequence)
        self._schedule_children.extend(remapped.children)
        self._flush_summary_fragment()

    def add_memory_event(
        self,
        *,
        transfer: DmaTransfer,
        multiplicity: int,
        stage: str = "global",
        axes: tuple[RepeatAxis, ...] = (),
        parallel_kernel: ParallelKernelTag | None = None,
    ) -> int:
        """Record one ordered compressed DMA stream."""
        if transfer.opcode not in MEMORY_OPS:
            raise ValueError(f"DMA stream uses non-memory opcode {transfer.opcode!r}")
        if multiplicity <= 0:
            raise ValueError(f"DMA stream multiplicity must be positive, got {multiplicity}")
        if any(axis.count <= 0 for axis in axes):
            raise ValueError(f"DMA repeat axes must be positive: {axes!r}")
        if axes:
            repeat_product = 1
            for axis in axes:
                repeat_product *= axis.count
            if repeat_product != multiplicity:
                raise ValueError(f"DMA repeat axes multiply to {repeat_product}, expected {multiplicity}")
        transfer = _logical_dma_metadata(transfer)
        axes = tuple(_logical_repeat_axis(axis) for axis in axes)
        stream_index = self._next_memory_stream_index
        self.trace.memory_events.append(
            MemoryEvent(
                stage=stage,
                transfer=transfer,
                multiplicity=multiplicity,
                enclosing_axes=axes,
                stream_index=stream_index,
                parallel_kernel=(
                    self.active_parallel_kernel
                    if parallel_kernel is None
                    else parallel_kernel
                ),
            )
        )
        self._next_memory_stream_index += 1
        return stream_index

    def _record(self, opcode: str, static_count: int, dynamic_count: int, stage: str) -> None:
        self.trace.static_opcodes[opcode] += static_count
        self.trace.dynamic_opcodes[opcode] += dynamic_count
        stage_cost = self.trace.stages[stage]
        stage_cost.static_opcodes[opcode] += static_count
        stage_cost.dynamic_opcodes[opcode] += dynamic_count

    def _visit(
        self,
        items: Iterable[AsmItem],
        *,
        static_multiplier: int,
        dynamic_multiplier: int,
        stage: str,
        axes: tuple[RepeatAxis, ...],
    ) -> list[ScheduleNode]:
        scheduled: list[ScheduleNode] = []
        for item in items:
            if isinstance(item, str):
                meaningful = [
                    line.strip() for line in item.splitlines() if line.strip() and not line.lstrip().startswith(";")
                ]
                if not meaningful:
                    continue
                if self.strict_raw:
                    preview = meaningful[0][:120]
                    raise RawAsmCostError(f"unstructured ASM reached CostSink in stage {stage!r}: {preview}")
                self._visit_raw(
                    item,
                    static_multiplier=static_multiplier,
                    dynamic_multiplier=dynamic_multiplier,
                    stage=stage,
                )
                reason = "unstructured_legacy_asm"
                self.trace.schedule_unavailable_reasons[reason] += 1
                scheduled.append(
                    ScheduleUnavailable(
                        reason=reason,
                        stage=stage,
                        dynamic_instruction_count=dynamic_multiplier,
                        parallel_kernel=self.active_parallel_kernel,
                    )
                )
                continue
            if isinstance(item, Comment):
                continue
            if isinstance(item, Instr):
                flat_multiplier = 1
                for count in self._typed_loop_stack:
                    flat_multiplier *= count
                effective_dynamic = dynamic_multiplier * flat_multiplier
                self._record(item.opcode, static_multiplier, effective_dynamic, stage)
                memory_stream_index = None
                if item.dma is not None:
                    if item.dma.opcode != item.opcode:
                        raise ValueError(f"DMA opcode {item.dma.opcode!r} does not match instruction {item.opcode!r}")
                    memory_stream_index = self.add_memory_event(
                        stage=stage,
                        transfer=item.dma,
                        multiplicity=effective_dynamic,
                        axes=axes,
                    )
                scheduled.append(
                    ScheduleInstruction(
                        opcode=item.opcode,
                        args=tuple(render_arg(arg) for arg in item.args),
                        stage=stage,
                        memory_stream_index=memory_stream_index,
                        parallel_kernel=self.active_parallel_kernel,
                    )
                )
                if item.opcode == "C_LOOP_START":
                    try:
                        count = int(item.args[-1])
                    except (IndexError, TypeError, ValueError) as exc:
                        raise RawAsmCostError(f"cannot parse typed hardware-loop count from {item!r}") from exc
                    if count <= 0:
                        raise RawAsmCostError(f"typed hardware-loop count must be positive, got {count}")
                    self._typed_loop_stack.append(count)
                    self.trace.schedule_unavailable_reasons["explicit_loop_markers"] += 1
                elif item.opcode == "C_LOOP_END":
                    if not self._typed_loop_stack:
                        raise RawAsmCostError(f"unmatched typed C_LOOP_END in stage {stage!r}")
                    self._typed_loop_stack.pop()
                continue
            if isinstance(item, Sequence):
                scheduled.extend(
                    self._visit(
                        item.items,
                        static_multiplier=static_multiplier,
                        dynamic_multiplier=dynamic_multiplier,
                        stage=stage,
                        axes=axes,
                    )
                )
                continue
            if isinstance(item, Stage):
                nested_stage = item.name if stage == "global" else f"{stage}/{item.name}"
                scheduled.extend(
                    self._visit(
                        item.body.items,
                        static_multiplier=static_multiplier,
                        dynamic_multiplier=dynamic_multiplier,
                        stage=nested_stage,
                        axes=axes,
                    )
                )
                continue
            if isinstance(item, CompileTimeRepeat):
                if item.count < 0:
                    raise ValueError(f"CompileTimeRepeat count must be >= 0, got {item.count}")
                if item.count == 0:
                    continue
                nested_axes = (*axes, item.axis or RepeatAxis("compile_time", item.count))
                body = self._visit(
                    item.body.items,
                    static_multiplier=static_multiplier * item.count,
                    dynamic_multiplier=dynamic_multiplier * item.count,
                    stage=stage,
                    axes=nested_axes,
                )
                scheduled.append(
                    ScheduleRepeat(
                        count=item.count,
                        body=ScheduleSequence(tuple(body)),
                        name=(item.axis.name if item.axis else "compile_time"),
                        repeat_kind="compile_time",
                    )
                )
                continue
            if isinstance(item, HardwareLoop):
                if item.count <= 0:
                    raise ValueError(f"HardwareLoop count must be > 0, got {item.count}")
                effective_count = item.count if item.effective_count is None else item.effective_count
                if not 0 <= effective_count <= item.count:
                    raise ValueError(
                        f"HardwareLoop effective_count must be in [0, {item.count}], got {effective_count}"
                    )
                self._record("C_LOOP_START", static_multiplier, dynamic_multiplier, stage)
                nested_axes = (*axes, item.axis or RepeatAxis("hardware_loop", effective_count))
                body = self._visit(
                    item.body.items,
                    static_multiplier=static_multiplier,
                    dynamic_multiplier=dynamic_multiplier * effective_count,
                    stage=stage,
                    axes=nested_axes,
                )
                self._record(
                    "C_LOOP_END",
                    static_multiplier,
                    dynamic_multiplier * effective_count,
                    stage,
                )
                scheduled.append(
                    ScheduleInstruction(
                        opcode="C_LOOP_START",
                        args=(render_arg(item.loop_register), str(item.count)),
                        stage=stage,
                        parallel_kernel=self.active_parallel_kernel,
                    )
                )
                if effective_count:
                    scheduled.append(
                        ScheduleRepeat(
                            count=effective_count,
                            body=ScheduleSequence(
                                (
                                    *body,
                                    ScheduleInstruction(
                                        opcode="C_LOOP_END",
                                        stage=stage,
                                        parallel_kernel=self.active_parallel_kernel,
                                    ),
                                )
                            ),
                            name=(item.axis.name if item.axis else "hardware_loop"),
                            repeat_kind="hardware_loop",
                        )
                    )
                continue
            raise TypeError(f"Unsupported symbolic item: {type(item).__name__}")
        return scheduled

    def _visit_raw(
        self,
        text: str,
        *,
        static_multiplier: int,
        dynamic_multiplier: int,
        stage: str,
    ) -> None:
        """Streaming compatibility parser for not-yet-migrated templates.

        This is deliberately unavailable in strict production cost lowering.
        It exists as a parity oracle while old string templates are migrated.
        """
        if "C_LOOP_START" not in text and "C_LOOP_END" not in text:
            summary = self._raw_straight_line_cache.get(text)
            if summary is None:
                summary = Counter()
                for raw_line in text.splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith(";"):
                        continue
                    token = line.split(maxsplit=1)[0]
                    if token.startswith(("S_", "C_", "H_", "V_", "M_")):
                        summary.update(self._raw_legalized_opcodes(line, token))
                self._raw_straight_line_cache[text] = summary
            loop_multiplier = 1
            for count in (*self._raw_loop_stack, *self._typed_loop_stack):
                loop_multiplier *= count
            for opcode, count in summary.items():
                self._record(
                    opcode,
                    static_multiplier * count,
                    dynamic_multiplier * loop_multiplier * count,
                    stage,
                )
            return

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            token = line.split(maxsplit=1)[0]
            if not token.startswith(("S_", "C_", "H_", "V_", "M_")):
                continue
            loop_multiplier = 1
            for count in (*self._raw_loop_stack, *self._typed_loop_stack):
                loop_multiplier *= count
            for legalized_opcode in self._raw_legalized_opcodes(line, token):
                self._record(
                    legalized_opcode,
                    static_multiplier,
                    dynamic_multiplier * loop_multiplier,
                    stage,
                )
            if token == "C_LOOP_START":
                try:
                    count = int(line.rsplit(",", 1)[1].strip())
                except (IndexError, ValueError) as exc:
                    raise RawAsmCostError(f"cannot parse hardware-loop count from: {line}") from exc
                if count <= 0:
                    raise RawAsmCostError(f"hardware-loop count must be positive in: {line}")
                self._raw_loop_stack.append(count)
            elif token == "C_LOOP_END":
                if not self._raw_loop_stack:
                    raise RawAsmCostError(f"unmatched C_LOOP_END in stage {stage!r}: {line}")
                self._raw_loop_stack.pop()

    @staticmethod
    def _raw_legalized_opcodes(line: str, token: str) -> tuple[str, ...]:
        if token != "S_ADDI_INT":
            return (token,)
        try:
            operands = [part.strip() for part in line.split(maxsplit=1)[1].split(",")]
            _, rs, immediate_text = operands[:3]
            immediate = int(immediate_text)
        except (IndexError, ValueError):
            return (token,)
        if immediate < IMM2_BOUND:
            return (token,)
        if rs == "gp0":
            lower = immediate & 0xFFF
            return ("S_LUI_INT", "S_ADDI_INT") if lower else ("S_LUI_INT",)
        chunks = (immediate + IMM2_BOUND - 2) // (IMM2_BOUND - 1)
        return tuple("S_ADDI_INT" for _ in range(chunks))

    def finish(self) -> CostTrace:
        if self._raw_loop_stack:
            raise RawAsmCostError(f"unterminated raw hardware loops: {self._raw_loop_stack}")
        if self._typed_loop_stack:
            raise RawAsmCostError(f"unterminated typed hardware loops: {self._typed_loop_stack}")
        self._flush_summary_fragment(force=True)
        if self._summary_enabled:
            if self._schedule_children:
                raise RawAsmCostError("affine summary retained an unterminated schedule fragment")
            for (
                key,
                element_base_delta,
                scale_base_delta,
                _memory_object_replacements,
            ), count in self._summary_template_replay_counts.items():
                self._apply_summary_template(
                    key,
                    count,
                    element_base_delta=element_base_delta,
                    scale_base_delta=scale_base_delta,
                )
            self.trace.schedule = ScheduleSequence(
                (
                    ScheduleUnavailable(
                        reason=COST_TRACE_GRANULARITY_AFFINE_BLOCK_SUMMARY_V1,
                        stage="global",
                        dynamic_instruction_count=self.trace.dynamic_instruction_count,
                    ),
                )
            )
            self.trace.schedule_unavailable_reasons = Counter({COST_TRACE_GRANULARITY_AFFINE_BLOCK_SUMMARY_V1: 1})
            summary_actions = _build_summary_energy_actions(
                self.trace,
                self._summary_activity_variants,
            )
            memory_only = CostTrace(memory_events=list(self.trace.memory_events))
            memory_actions = _build_energy_actions(memory_only)
            self.trace.energy_actions = _merge_energy_actions(
                (
                    *summary_actions,
                    *self._summary_energy_actions,
                    *memory_actions,
                )
            )
            for stage in self.trace.stages.values():
                stage.energy_actions.clear()
            for action in self.trace.energy_actions:
                self.trace.stages[action.stage].energy_actions.append(action)
            encoded_total = _encode_schedule_variants(self._summary_parameterized_variants)
            self.trace.metadata.update(
                {
                    **self._summary_agu_metadata,
                    "cost_trace_granularity": self.granularity,
                    "compute_trace_fidelity": "exact_algebraic_ideal_ii1",
                    "ordered_schedule_available": False,
                    "parameterized_timing_variants": encoded_total,
                    "one_layer_parameterized_timing_variants": encoded_total,
                    "stage_parameterized_timing_variants": {
                        stage: _encode_schedule_variants(variants)
                        for stage, variants in sorted(self._summary_stage_parameterized_variants.items())
                    },
                }
            )
            return self.trace

        self.trace.schedule = ScheduleSequence(tuple(self._schedule_children))
        if self.trace.schedule_unavailable_reasons.get("explicit_loop_markers"):
            compressed = _compress_explicit_hardware_loops(self.trace.schedule)
            assert isinstance(compressed, ScheduleSequence)
            self.trace.schedule = compressed
            del self.trace.schedule_unavailable_reasons["explicit_loop_markers"]
        if not self.trace.schedule_unavailable_reasons:
            self.trace.schedule = _bind_unindexed_memory_instructions(self.trace.schedule, self.trace.memory_events)
        if not self.trace.schedule_unavailable_reasons:
            derived = _schedule_opcode_counts(self.trace.schedule)
            if derived != self.trace.dynamic_opcodes:
                raise ValueError(
                    "compressed schedule opcode counts drifted from CostTrace: "
                    f"schedule={derived}, trace={self.trace.dynamic_opcodes}"
                )
            self.trace.dynamic_opcodes = derived
        if not self.trace.schedule_unavailable_reasons:
            rebuild_parallel_kernel_census(self.trace)
        self.trace.energy_actions = _build_energy_actions(self.trace)
        for stage in self.trace.stages.values():
            stage.energy_actions.clear()
        for action in self.trace.energy_actions:
            self.trace.stages[action.stage].energy_actions.append(action)
        return self.trace


class AsmSink:
    """Compatibility sink that renders symbolic nodes to assembly text."""

    def __init__(self):
        self.chunks: list[str] = []

    def emit(self, value) -> str:
        rendered = render_asm(value)
        self.chunks.append(rendered)
        return rendered

    def getvalue(self) -> str:
        return "".join(self.chunks)


class CompositeSink:
    """Render and count one small symbolic program for parity tests."""

    def __init__(self, asm: AsmSink | None = None, cost: CostSink | None = None):
        self.asm = asm or AsmSink()
        self.cost = cost or CostSink()

    def emit(self, value) -> str:
        self.cost.emit(value)
        return self.asm.emit(value)


__all__ = [
    "AsmSink",
    "CompositeSink",
    "ClockWork",
    "CostSink",
    "CostTrace",
    "EnergyAction",
    "MemoryEvent",
    "ParallelKernelCensusEntry",
    "ParallelKernelTag",
    "RawAsmCostError",
    "ScheduleAffineAdd",
    "ScheduleAffineLoad",
    "ScheduleInstruction",
    "ScheduleNode",
    "ScheduleRepeat",
    "ScheduleSequence",
    "ScheduleUnavailable",
    "opcode_category",
    "optimize_cost_trace_loop_agu",
    "schedule_instruction_activity_variants",
    "schedule_instruction_variants",
    "schedule_parallel_kernel_census",
    "rebuild_parallel_kernel_census",
    "parallel_kernel_lineage_id",
    "UNCLASSIFIED_PARALLEL_KERNEL",
]
