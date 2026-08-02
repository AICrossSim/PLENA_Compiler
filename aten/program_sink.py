"""Consumers for the compiler's immutable final-schedule IR."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
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
        return Counter({item.opcode: item.multiplicity for item in self.instructions})

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
    _stage_stack: list[str] = field(default_factory=list)
    _multiplier_stack: list[int] = field(default_factory=lambda: [1])
    _axis_stack: list[RepeatAxis] = field(default_factory=list)
    _instructions: dict[tuple[Any, ...], TraceInstruction] = field(default_factory=dict)
    _dmas: list[TraceDma] = field(default_factory=list)
    _schedule_digest: Any = field(default_factory=sha256)
    _raw_instruction_count: int = 0
    _symbolic_repeat_count: int = 0

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
        self._raw_instruction_count += 1
        self._schedule_digest.update(instruction.render().encode())
        self._schedule_digest.update(b"\n")
        if instruction.dma is not None:
            self.emit_dma(instruction.dma)

    def emit_dma(self, transfer: DmaTransfer) -> None:
        if transfer.opcode not in {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}:
            raise ValueError(f"unsupported DMA opcode {transfer.opcode!r}")
        self._dmas.append(
            TraceDma(
                stage=self.stage,
                transfer=transfer,
                multiplicity=self.multiplier,
                repeat_axes=tuple(self._axis_stack) + transfer.axes,
            )
        )

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
        if self._stage_stack or len(self._multiplier_stack) != 1 or self._axis_stack:
            raise ValueError("cannot finish an unbalanced final schedule")
        merged_dma = _merge_dma_events(self._dmas)
        trace_metadata = {
            "trace_fidelity": "exact_final_schedule",
            "ordered_schedule_available": True,
            "materialized_dynamic_instructions": 0,
            "symbolic_repeat_nodes": self._symbolic_repeat_count,
            "materialized_schedule_leaves": self._raw_instruction_count,
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
    "ProgramSink",
    "SymbolicCostSink",
    "TRACE_SCHEMA_VERSION",
    "TraceDma",
    "TraceInstruction",
]
