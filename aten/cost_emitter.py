"""Symbolic instruction and DMA cost collection for the ATen compiler.

The cost path consumes the same nodes as the assembly renderer.  It never
materializes compile-time repeats or hardware-loop iterations, so a program
whose physical ISA contains tens of millions of instructions stays compact.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from typing import Any
from collections.abc import Iterable, Mapping

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
    "S_LD_FP",
    "S_ST_FP",
    "S_MAP_V_FP",
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
    "C_BREAK",
}
MEMORY_OPS = {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}


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
            }
        )
        return result


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
            transfer.element_base
            if transfer.logical_element_offset is None
            else transfer.logical_element_offset
        ),
        logical_scale_offset=(
            transfer.scale_base
            if transfer.logical_scale_offset is None
            else transfer.logical_scale_offset
        ),
        logical_stride=(
            transfer.stride if transfer.logical_stride is None else transfer.logical_stride
        ),
    )


def _logical_repeat_axis(axis: RepeatAxis) -> RepeatAxis:
    return replace(
        axis,
        logical_element_delta=(
            axis.element_base_delta
            if axis.logical_element_delta is None
            else axis.logical_element_delta
        ),
        logical_scale_delta=(
            axis.scale_base_delta
            if axis.logical_scale_delta is None
            else axis.logical_scale_delta
        ),
    )


@dataclass
class StageCost:
    static_opcodes: Counter[str] = field(default_factory=Counter)
    dynamic_opcodes: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "static_opcodes": dict(sorted(self.static_opcodes.items())),
            "dynamic_opcodes": dict(sorted(self.dynamic_opcodes.items())),
            "static_instruction_count": sum(self.static_opcodes.values()),
            "dynamic_instruction_count": sum(self.dynamic_opcodes.values()),
        }


@dataclass
class CostTrace:
    static_opcodes: Counter[str] = field(default_factory=Counter)
    dynamic_opcodes: Counter[str] = field(default_factory=Counter)
    memory_events: list[MemoryEvent] = field(default_factory=list)
    stages: dict[str, StageCost] = field(default_factory=lambda: defaultdict(StageCost))
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
        return {
            "schema_version": 3,
            **self.metadata,
            "static_instruction_count": self.static_instruction_count,
            "dynamic_instruction_count": self.dynamic_instruction_count,
            "static_opcodes": dict(sorted(self.static_opcodes.items())),
            "dynamic_opcodes": dict(sorted(self.dynamic_opcodes.items())),
            "instruction_categories": dict(sorted(categories.items())),
            "compressed_memory_events": [event.to_dict() for event in self.memory_events],
            "stage_breakdown": {name: stage.to_dict() for name, stage in sorted(self.stages.items())},
        }


class CostSink:
    """Algebraically count a symbolic program without rendering it."""

    def __init__(self, *, strict_raw: bool = True):
        self.strict_raw = strict_raw
        self.trace = CostTrace()
        self._raw_loop_stack: list[int] = []
        self._typed_loop_stack: list[int] = []
        self._raw_straight_line_cache: dict[str, Counter[str]] = {}
        self._next_memory_stream_index = 0

    def emit(self, value: IsaBuilder | Sequence | Iterable[AsmItem]) -> None:
        sequence = as_sequence(value)
        items = legalize_large_immediates(sequence.items)
        self._visit(items, static_multiplier=1, dynamic_multiplier=1, stage="global", axes=())

    def add_counts(
        self,
        *,
        static_opcodes: Mapping[str, int],
        dynamic_opcodes: Mapping[str, int],
        stage: str = "global",
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

    def add_memory_event(
        self,
        *,
        transfer: DmaTransfer,
        multiplicity: int,
        stage: str = "global",
        axes: tuple[RepeatAxis, ...] = (),
    ) -> None:
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
                raise ValueError(
                    f"DMA repeat axes multiply to {repeat_product}, expected {multiplicity}"
                )
        transfer = _logical_dma_metadata(transfer)
        axes = tuple(_logical_repeat_axis(axis) for axis in axes)
        self.trace.memory_events.append(
            MemoryEvent(
                stage=stage,
                transfer=transfer,
                multiplicity=multiplicity,
                enclosing_axes=axes,
                stream_index=self._next_memory_stream_index,
            )
        )
        self._next_memory_stream_index += 1

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
    ) -> None:
        for item in items:
            if isinstance(item, str):
                meaningful = [
                    line.strip()
                    for line in item.splitlines()
                    if line.strip() and not line.lstrip().startswith(";")
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
                continue
            if isinstance(item, Comment):
                continue
            if isinstance(item, Instr):
                flat_multiplier = 1
                for count in self._typed_loop_stack:
                    flat_multiplier *= count
                effective_dynamic = dynamic_multiplier * flat_multiplier
                self._record(item.opcode, static_multiplier, effective_dynamic, stage)
                if item.dma is not None:
                    if item.dma.opcode != item.opcode:
                        raise ValueError(
                            f"DMA opcode {item.dma.opcode!r} does not match instruction {item.opcode!r}"
                        )
                    self.add_memory_event(
                        stage=stage,
                        transfer=item.dma,
                        multiplicity=effective_dynamic,
                        axes=axes,
                    )
                if item.opcode == "C_LOOP_START":
                    try:
                        count = int(item.args[-1])
                    except (IndexError, TypeError, ValueError) as exc:
                        raise RawAsmCostError(f"cannot parse typed hardware-loop count from {item!r}") from exc
                    if count <= 0:
                        raise RawAsmCostError(f"typed hardware-loop count must be positive, got {count}")
                    self._typed_loop_stack.append(count)
                elif item.opcode == "C_LOOP_END":
                    if not self._typed_loop_stack:
                        raise RawAsmCostError(f"unmatched typed C_LOOP_END in stage {stage!r}")
                    self._typed_loop_stack.pop()
                continue
            if isinstance(item, Sequence):
                self._visit(
                    item.items,
                    static_multiplier=static_multiplier,
                    dynamic_multiplier=dynamic_multiplier,
                    stage=stage,
                    axes=axes,
                )
                continue
            if isinstance(item, Stage):
                nested_stage = item.name if stage == "global" else f"{stage}/{item.name}"
                self._visit(
                    item.body.items,
                    static_multiplier=static_multiplier,
                    dynamic_multiplier=dynamic_multiplier,
                    stage=nested_stage,
                    axes=axes,
                )
                continue
            if isinstance(item, CompileTimeRepeat):
                if item.count < 0:
                    raise ValueError(f"CompileTimeRepeat count must be >= 0, got {item.count}")
                nested_axes = (*axes, item.axis or RepeatAxis("compile_time", item.count))
                self._visit(
                    item.body.items,
                    static_multiplier=static_multiplier * item.count,
                    dynamic_multiplier=dynamic_multiplier * item.count,
                    stage=stage,
                    axes=nested_axes,
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
                self._visit(
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
                continue
            raise TypeError(f"Unsupported symbolic item: {type(item).__name__}")

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
    "CostSink",
    "CostTrace",
    "MemoryEvent",
    "RawAsmCostError",
    "opcode_category",
]
