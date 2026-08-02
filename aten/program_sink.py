"""Consumers for the compiler's immutable final-schedule IR."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
import json
from typing import Any, Protocol

from compiler.aten.isa_builder import (
    Comment,
    CompileTimeRepeat,
    DmaTransfer,
    HardwareLoop,
    Instr,
    RepeatAxis,
    Sequence,
    Stage,
    parse_legacy_asm,
    render_arg,
    render_items,
)


TRACE_SCHEMA_VERSION = "plena-final-schedule-v1"
COST_TRACE_GRANULARITY_DETAILED = "detailed"
COST_TRACE_GRANULARITY_SUMMARY = "affine-block-summary-v1"
COST_TRACE_GRANULARITIES = (
    COST_TRACE_GRANULARITY_DETAILED,
    COST_TRACE_GRANULARITY_SUMMARY,
)


class ProgramSink(Protocol):
    """Minimal backend contract for final compiler schedules."""

    def begin_stage(self, stage_path: str) -> None: ...
    def end_stage(self, stage_path: str) -> None: ...
    def emit_instruction(self, instruction: Instr) -> None: ...
    def emit_dma(self, transfer: DmaTransfer) -> None: ...
    def begin_repeat(self, count: int, axis: RepeatAxis | None, kind: str) -> None: ...
    def end_repeat(self, count: int, axis: RepeatAxis | None, kind: str) -> None: ...


@dataclass(frozen=True)
class TraceInstruction:
    stage: str
    opcode: str
    operands: tuple[str, ...]
    variant: tuple[tuple[str, str], ...]
    active: dict[str, int | None] | None
    sram: tuple[dict[str, Any], ...]
    multiplicity: int

    def key(self) -> tuple[Any, ...]:
        return (
            self.stage,
            self.opcode,
            self.operands,
            self.variant,
            tuple(sorted((self.active or {}).items())),
            tuple(tuple(sorted(entry.items())) for entry in self.sram),
        )


@dataclass(frozen=True)
class TraceDma:
    stage: str
    transfer: DmaTransfer
    multiplicity: int
    repeat_axes: tuple[RepeatAxis, ...]


@dataclass(frozen=True)
class CostTrace:
    schema_version: str
    isa_hash: str
    compiler_hash: str
    instructions: tuple[TraceInstruction, ...]
    dma_events: tuple[TraceDma, ...]
    metadata: dict[str, Any]

    @property
    def dynamic_opcode_counts(self) -> Counter[str]:
        result: Counter[str] = Counter()
        for item in self.instructions:
            result[item.opcode] += item.multiplicity
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "isa_hash": self.isa_hash,
            "compiler_hash": self.compiler_hash,
            "instructions": [asdict(item) for item in self.instructions],
            "dma_events": [
                {
                    "stage": item.stage,
                    "transfer": asdict(item.transfer),
                    "multiplicity": item.multiplicity,
                    "repeat_axes": [asdict(axis) for axis in item.repeat_axes],
                }
                for item in self.dma_events
            ],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class CostTraceFragment:
    """One compiler-emitted kernel body retained for algebraic replay."""

    instructions: tuple[TraceInstruction, ...]
    dma_events: tuple[TraceDma, ...]


@dataclass
class AsmSink:
    """Reference sink used to prove IR rendering remains byte stable."""

    lines: list[str] = field(default_factory=list)

    def consume(self, schedule: Sequence) -> None:
        self.lines.extend(render_items(schedule.items))

    def render(self) -> str:
        return "\n".join(self.lines) + ("\n" if self.lines else "")


@dataclass
class SymbolicCostSink:
    """Exact multiplicity collector that never expands symbolic repeats."""

    compiler_hash: str = "unknown"
    default_stage: str | None = None
    granularity: str = COST_TRACE_GRANULARITY_DETAILED
    _stage_stack: list[str] = field(default_factory=list)
    _multiplier_stack: list[int] = field(default_factory=lambda: [1])
    _axis_stack: list[RepeatAxis] = field(default_factory=list)
    _instructions: dict[tuple[Any, ...], TraceInstruction] = field(default_factory=dict)
    _dmas: list[TraceDma] = field(default_factory=list)
    _schedule_digest: Any = field(default_factory=sha256)
    _raw_instruction_count: int = 0
    _symbolic_repeat_count: int = 0
    _suppress_dma_depth: int = 0
    _templates: dict[tuple[Any, ...], CostTraceFragment] = field(default_factory=dict)
    _capture_stack: list[
        tuple[tuple[Any, ...], dict[tuple[Any, ...], int], int]
    ] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.granularity not in COST_TRACE_GRANULARITIES:
            raise ValueError(
                f"unsupported cost-trace granularity {self.granularity!r}; "
                f"expected one of {COST_TRACE_GRANULARITIES}"
            )

    @property
    def summary_enabled(self) -> bool:
        return self.granularity == COST_TRACE_GRANULARITY_SUMMARY

    @property
    def stage(self) -> str:
        if self._stage_stack:
            return "/".join(self._stage_stack)
        if self.default_stage:
            return self.default_stage
        raise ValueError("final schedule instruction has no stage ownership")

    @property
    def multiplier(self) -> int:
        return self._multiplier_stack[-1]

    def begin_stage(self, stage_path: str) -> None:
        if not stage_path:
            raise ValueError("stage_path must be non-empty")
        self._stage_stack.append(stage_path)

    def end_stage(self, stage_path: str) -> None:
        if not self._stage_stack or self._stage_stack[-1] != stage_path:
            raise ValueError(f"unbalanced stage {stage_path!r}")
        self._stage_stack.pop()

    def emit_instruction(self, instruction: Instr) -> None:
        if not instruction.opcode.startswith(("M_", "V_", "S_", "C_", "H_")):
            raise ValueError(f"unknown final-schedule opcode {instruction.opcode!r}")
        stage = self.stage
        operands = tuple(render_arg(arg) for arg in instruction.args)
        active = None if instruction.active is None else asdict(instruction.active)
        sram = tuple(asdict(entry) for entry in instruction.sram)
        leaf = TraceInstruction(
            stage=stage,
            opcode=instruction.opcode,
            operands=operands,
            variant=instruction.variant,
            active=active,
            sram=sram,
            multiplicity=self.multiplier,
        )
        self._add_instruction(leaf)
        self._raw_instruction_count += 1
        self._schedule_digest.update(instruction.render().encode())
        self._schedule_digest.update(b"\n")
        if instruction.dma is not None:
            self.emit_dma(instruction.dma)

    def _add_instruction(self, leaf: TraceInstruction, multiplier: int = 1) -> None:
        if multiplier < 0:
            raise ValueError("instruction multiplier must be nonnegative")
        if multiplier == 0 or leaf.multiplicity == 0:
            return
        if multiplier != 1:
            leaf = TraceInstruction(
                stage=leaf.stage,
                opcode=leaf.opcode,
                operands=leaf.operands,
                variant=leaf.variant,
                active=leaf.active,
                sram=leaf.sram,
                multiplicity=leaf.multiplicity * multiplier,
            )
        key = leaf.key()
        old = self._instructions.get(key)
        if old is None:
            self._instructions[key] = leaf
        else:
            self._instructions[key] = TraceInstruction(
                stage=old.stage,
                opcode=old.opcode,
                operands=old.operands,
                variant=old.variant,
                active=old.active,
                sram=old.sram,
                multiplicity=old.multiplicity + leaf.multiplicity,
            )

    def add_opcode_counts(
        self,
        counts: Counter[str] | dict[str, int],
        *,
        provenance: str,
    ) -> None:
        """Add a compiler-owned symbolic schedule census.

        This hook is reserved for legacy templates whose production renderer
        still Python-unrolls a deterministic affine loop.  The census lives in
        the compiler adapter and is parity-tested against the rendered final
        schedule; timing consumers never carry a second opcode formula.
        """
        if not self.summary_enabled:
            raise RuntimeError("abstract opcode counts are only valid in summary mode")
        for opcode, count in counts.items():
            if count < 0:
                raise ValueError(f"negative {opcode} count")
            if count == 0:
                continue
            if not opcode.startswith(("M_", "V_", "S_", "C_", "H_")):
                raise ValueError(f"unknown final-schedule opcode {opcode!r}")
            self._add_instruction(
                TraceInstruction(
                    stage=self.stage,
                    opcode=opcode,
                    operands=(),
                    variant=(("symbolic_provenance", provenance),),
                    active=None,
                    sram=(),
                    multiplicity=int(count) * self.multiplier,
                )
            )
        self._schedule_digest.update(
            json.dumps(
                {"provenance": provenance, "counts": dict(sorted(counts.items()))},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )

    def record_symbolic_lineage(self, provenance: str, payload: Any) -> None:
        """Bind an algebraic schedule census to the trace identity."""
        if not self.summary_enabled:
            raise RuntimeError("symbolic lineage is only valid in summary mode")
        self._schedule_digest.update(
            json.dumps(
                {"provenance": provenance, "payload": payload},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )

    def emit_dma(self, transfer: DmaTransfer) -> None:
        self.record_dma(transfer)

    def record_dma(
        self,
        transfer: DmaTransfer,
        *,
        multiplicity: int = 1,
        axes: tuple[RepeatAxis, ...] = (),
    ) -> None:
        if self._suppress_dma_depth:
            return
        if transfer.opcode not in {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}:
            raise ValueError(f"unsupported DMA opcode {transfer.opcode!r}")
        if multiplicity < 0:
            raise ValueError("DMA multiplicity must be nonnegative")
        self._dmas.append(
            TraceDma(
                stage=self.stage,
                transfer=transfer,
                multiplicity=self.multiplier * multiplicity,
                repeat_axes=tuple(self._axis_stack) + axes + transfer.axes,
            )
        )

    def begin_dma_suppression(self) -> None:
        self._suppress_dma_depth += 1

    def end_dma_suppression(self) -> None:
        if self._suppress_dma_depth <= 0:
            raise ValueError("unbalanced DMA suppression")
        self._suppress_dma_depth -= 1

    def begin_template(self, key: tuple[Any, ...]) -> None:
        if not self.summary_enabled:
            raise RuntimeError("summary templates require summary granularity")
        if self._capture_stack:
            raise RuntimeError("nested summary-template capture is unsupported")
        if key in self._templates:
            raise ValueError(f"summary template {key!r} already exists")
        snapshot = {item_key: item.multiplicity for item_key, item in self._instructions.items()}
        self._capture_stack.append((key, snapshot, len(self._dmas)))

    def end_template(self, key: tuple[Any, ...]) -> None:
        if not self._capture_stack or self._capture_stack[-1][0] != key:
            raise ValueError(f"unbalanced summary template {key!r}")
        _, before, dma_start = self._capture_stack.pop()
        instructions: list[TraceInstruction] = []
        for item_key, item in self._instructions.items():
            delta = item.multiplicity - before.get(item_key, 0)
            if delta:
                instructions.append(
                    TraceInstruction(
                        stage=item.stage,
                        opcode=item.opcode,
                        operands=item.operands,
                        variant=item.variant,
                        active=item.active,
                        sram=item.sram,
                        multiplicity=delta,
                    )
                )
        self._templates[key] = CostTraceFragment(
            instructions=tuple(instructions),
            dma_events=tuple(self._dmas[dma_start:]),
        )

    def replay_template(
        self,
        key: tuple[Any, ...],
        *,
        count: int = 1,
        axes: tuple[RepeatAxis, ...] = (),
        dma_address_delta_bytes: int = 0,
    ) -> bool:
        fragment = self._templates.get(key)
        if fragment is None:
            return False
        if count < 0:
            raise ValueError("summary-template replay count must be nonnegative")
        for item in fragment.instructions:
            self._add_instruction(item, count * self.multiplier)
        for event in fragment.dma_events:
            transfer = event.transfer
            if dma_address_delta_bytes:
                transfer = replace(
                    transfer,
                    element_base_bytes=(
                        transfer.element_base_bytes + dma_address_delta_bytes
                    ),
                    scale_base_bytes=(
                        None
                        if transfer.scale_base_bytes is None
                        else transfer.scale_base_bytes + dma_address_delta_bytes
                    ),
                )
            self._dmas.append(
                TraceDma(
                    stage=event.stage,
                    transfer=transfer,
                    multiplicity=event.multiplicity * count * self.multiplier,
                    repeat_axes=tuple(self._axis_stack) + axes + event.repeat_axes,
                )
            )
        self._schedule_digest.update(
            repr(("replay", key, count, axes, dma_address_delta_bytes)).encode()
        )
        return True

    def begin_repeat(self, count: int, axis: RepeatAxis | None, kind: str) -> None:
        if count < 0:
            raise ValueError(f"repeat count must be nonnegative, got {count}")
        self._multiplier_stack.append(self.multiplier * count)
        if axis is not None:
            self._axis_stack.append(axis)
        self._symbolic_repeat_count += 1

    def end_repeat(self, count: int, axis: RepeatAxis | None, kind: str) -> None:
        if axis is not None:
            if not self._axis_stack or self._axis_stack[-1] != axis:
                raise ValueError("unbalanced affine repeat axis")
            self._axis_stack.pop()
        if len(self._multiplier_stack) == 1:
            raise ValueError("unbalanced repeat")
        self._multiplier_stack.pop()

    def consume(self, schedule: Sequence) -> None:
        _walk(schedule, self)

    def finish(self, **metadata: Any) -> CostTrace:
        if (
            self._stage_stack
            or len(self._multiplier_stack) != 1
            or self._axis_stack
            or self._capture_stack
            or self._suppress_dma_depth
        ):
            raise ValueError("cannot finish an unbalanced final schedule")
        merged_dma = _merge_dma_events(self._dmas)
        hbm_opcodes = {
            opcode: count
            for opcode, count in self.dynamic_opcode_counts().items()
            if opcode.startswith("H_")
        }
        dma_opcodes: Counter[str] = Counter()
        for event in merged_dma:
            dma_opcodes[event.transfer.opcode] += event.multiplicity
        uncovered = {
            opcode: (count, dma_opcodes.get(opcode, 0))
            for opcode, count in hbm_opcodes.items()
            if count != dma_opcodes.get(opcode, 0)
        }
        if uncovered:
            detail = ", ".join(
                f"{opcode}: instructions={counts[0]}, DMA={counts[1]}"
                for opcode, counts in sorted(uncovered.items())
            )
            raise ValueError(f"incomplete final-schedule DMA coverage: {detail}")
        trace_metadata = {
            "trace_fidelity": "exact_final_schedule",
            "cost_trace_granularity": self.granularity,
            "ordered_schedule_available": not self.summary_enabled,
            "materialized_dynamic_instructions": 0,
            "symbolic_repeat_nodes": self._symbolic_repeat_count,
            "materialized_schedule_leaves": self._raw_instruction_count,
            "summary_template_count": len(self._templates),
            "dma_opcode_coverage": {
                opcode: {
                    "instruction_count": count,
                    "described_count": dma_opcodes.get(opcode, 0),
                }
                for opcode, count in sorted(hbm_opcodes.items())
            },
            **metadata,
        }
        return CostTrace(
            schema_version=TRACE_SCHEMA_VERSION,
            isa_hash=self._schedule_digest.hexdigest(),
            compiler_hash=self.compiler_hash,
            instructions=tuple(sorted(self._instructions.values(), key=lambda item: item.key())),
            dma_events=tuple(merged_dma),
            metadata=trace_metadata,
        )

    def dynamic_opcode_counts(self) -> Counter[str]:
        result: Counter[str] = Counter()
        for item in self._instructions.values():
            result[item.opcode] += item.multiplicity
        return result


def _walk(sequence: Sequence, sink: ProgramSink) -> None:
    for item in sequence.items:
        if isinstance(item, str):
            _walk(parse_legacy_asm(item), sink)
        elif isinstance(item, Comment):
            continue
        elif isinstance(item, Instr):
            sink.emit_instruction(item)
        elif isinstance(item, Sequence):
            _walk(item, sink)
        elif isinstance(item, Stage):
            sink.begin_stage(item.path)
            _walk(item.body, sink)
            sink.end_stage(item.path)
        elif isinstance(item, CompileTimeRepeat):
            sink.begin_repeat(item.count, item.axis, "compile-time")
            _walk(item.body, sink)
            sink.end_repeat(item.count, item.axis, "compile-time")
        elif isinstance(item, HardwareLoop):
            # The loop setup executes once; the body and C_LOOP_END execute for
            # every effective iteration in the current emulator semantics.
            sink.emit_instruction(Instr("C_LOOP_START", (item.loop_register, item.count)))
            dynamic_count = item.effective_count if item.effective_count is not None else item.count
            sink.begin_repeat(dynamic_count, item.axis, "hardware")
            _walk(item.body, sink)
            sink.emit_instruction(Instr("C_LOOP_END", (item.loop_register,)))
            sink.end_repeat(dynamic_count, item.axis, "hardware")
        else:
            raise TypeError(f"unsupported final schedule node {type(item).__name__}")


def _merge_dma_events(events: list[TraceDma]) -> list[TraceDma]:
    merged: dict[str, TraceDma] = {}
    for event in events:
        payload = {
            "stage": event.stage,
            "transfer": asdict(event.transfer),
            "repeat_axes": [asdict(axis) for axis in event.repeat_axes],
        }
        key = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        old = merged.get(key)
        if old is None:
            merged[key] = event
        else:
            merged[key] = TraceDma(
                stage=old.stage,
                transfer=old.transfer,
                multiplicity=old.multiplicity + event.multiplicity,
                repeat_axes=old.repeat_axes,
            )
    return [merged[key] for key in sorted(merged)]


__all__ = [
    "AsmSink",
    "CostTrace",
    "CostTraceFragment",
    "COST_TRACE_GRANULARITIES",
    "COST_TRACE_GRANULARITY_DETAILED",
    "COST_TRACE_GRANULARITY_SUMMARY",
    "ProgramSink",
    "SymbolicCostSink",
    "TRACE_SCHEMA_VERSION",
    "TraceDma",
    "TraceInstruction",
]
