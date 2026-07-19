"""Symbolic instruction and DMA cost collection for the ATen compiler.

The cost path consumes the same nodes as the assembly renderer.  It never
materializes compile-time repeats or hardware-loop iterations, so a program
whose physical ISA contains tens of millions of instructions stays compact.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from typing import Any, ClassVar, TypeAlias
from collections.abc import Callable, Iterable, Mapping

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

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "instruction",
            "opcode": self.opcode,
            "args": list(self.args),
            "stage": self.stage,
        }
        if self.memory_stream_index is not None:
            result["memory_stream_index"] = self.memory_stream_index
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "unavailable",
            "reason": self.reason,
            "stage": self.stage,
            "dynamic_instruction_count": self.dynamic_instruction_count,
        }


ScheduleNode: TypeAlias = (
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
            if replace(previous, stage=current.stage) != current:
                raise ValueError(
                    f"affine schedule key {current.key!r} has inconsistent "
                    f"definitions: previous={previous!r}, current={current!r}"
                )
            affine_visits[current.key] += multiplier
            return
        if isinstance(current, ScheduleAffineAdd):
            previous = affine_add_specs.setdefault(current.key, current)
            if replace(previous, stage=current.stage) != current:
                raise ValueError(
                    f"affine-add schedule key {current.key!r} has "
                    f"inconsistent definitions: previous={previous!r}, "
                    f"current={current!r}"
                )
            affine_add_visits[current.key] += multiplier
            return
        if isinstance(current, ScheduleUnavailable):
            raise ValueError(
                "cannot derive opcode histogram from unavailable schedule node"
            )
        if isinstance(current, ScheduleSequence):
            for child in current.children:
                visit(child, multiplier)
            return
        if isinstance(current, ScheduleRepeat):
            visit(current.body, multiplier * current.count)
            return
        raise TypeError(type(current).__name__)

    def affine_unique_value_counts(
        affine: ScheduleAffineLoad, count: int
    ) -> Counter[str]:
        if count <= 0:
            return Counter()
        if affine.step < 0 or affine.start < 0:
            raise ValueError(f"negative affine load is unsupported: {affine!r}")
        below = (
            max(
                0,
                min(
                    count,
                    (IMM2_BOUND - affine.start + affine.step - 1)
                    // affine.step,
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
                    first = (
                        (rhs // divisor)
                        * pow(affine.step // divisor, -1, reduced_modulus)
                    ) % reduced_modulus
                    if first < below:
                        first += (
                            (below - first + reduced_modulus - 1)
                            // reduced_modulus
                        ) * reduced_modulus
                    if first < count:
                        zero_low = 1 + (count - 1 - first) // reduced_modulus
        return Counter(
            {
                "S_LUI_INT": high,
                "S_ADDI_INT": below + high - zero_low,
            }
        )

    def affine_sequence_counts(
        affine: ScheduleAffineLoad, count: int
    ) -> Counter[str]:
        if affine.advance_every <= 0:
            raise ValueError(
                f"affine advance_every must be positive: {affine!r}"
            )
        full_values, remainder = divmod(count, affine.advance_every)
        unique = affine_unique_value_counts(affine, full_values)
        result = Counter(
            {
                opcode: opcode_count * affine.advance_every
                for opcode, opcode_count in unique.items()
            }
        )
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
                    for opcode, opcode_count in affine_unique_value_counts(
                        tail, 1
                    ).items()
                }
            )
        return +result

    def affine_add_unique_value_counts(
        affine: ScheduleAffineAdd, count: int
    ) -> Counter[str]:
        if count <= 0:
            return Counter()
        if affine.step < 0 or affine.start < 0:
            raise ValueError(f"negative affine add is unsupported: {affine!r}")
        below = (
            max(
                0,
                min(
                    count,
                    (IMM2_BOUND - affine.start + affine.step - 1)
                    // affine.step,
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
                    first = (
                        (rhs // divisor)
                        * pow(affine.step // divisor, -1, reduced_modulus)
                    ) % reduced_modulus
                    if first < below:
                        first += (
                            (below - first + reduced_modulus - 1)
                            // reduced_modulus
                        ) * reduced_modulus
                    if first < count:
                        zero_low = 1 + (count - 1 - first) // reduced_modulus
        return +Counter(
            {
                "S_ADDI_INT": below + high - zero_low,
                "S_LUI_INT": high,
                "S_ADD_INT": high,
            }
        )

    def affine_add_sequence_counts(
        affine: ScheduleAffineAdd, count: int
    ) -> Counter[str]:
        if affine.advance_every <= 0:
            raise ValueError(
                f"affine-add advance_every must be positive: {affine!r}"
            )
        full_values, remainder = divmod(count, affine.advance_every)
        unique = affine_add_unique_value_counts(affine, full_values)
        result = Counter(
            {
                opcode: opcode_count * affine.advance_every
                for opcode, opcode_count in unique.items()
            }
        )
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
                    for opcode, opcode_count in affine_add_unique_value_counts(
                        tail, 1
                    ).items()
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
            raise ValueError(
                "affine period must be positive and contain complete "
                f"repeated-value groups: {affine!r}"
            )
        full_periods, remainder = divmod(visits, affine.period)
        period_counts = counter(affine, affine.period)
        regular.update(
            {opcode: count * full_periods for opcode, count in period_counts.items()}
        )
        regular.update(counter(affine, remainder))

    visit(node, 1)
    for key, visits in affine_visits.items():
        affine = affine_specs[key]
        add_periodic_affine_counts(affine, visits, affine_sequence_counts)
    for key, visits in affine_add_visits.items():
        add_periodic_affine_counts(
            affine_add_specs[key], visits, affine_add_sequence_counts
        )
    return +regular


def _remap_schedule_memory_streams(
    node: ScheduleNode, stream_indices: tuple[int, ...]
) -> ScheduleNode:
    """Replace kernel-local DMA stream ordinals with trace-global indices."""
    if isinstance(node, ScheduleInstruction):
        local = node.memory_stream_index
        if local is None:
            return node
        if local < 0 or local >= len(stream_indices):
            raise ValueError(
                f"schedule DMA stream index {local} is outside "
                f"[0, {len(stream_indices)})"
            )
        return replace(node, memory_stream_index=stream_indices[local])
    if isinstance(node, (ScheduleAffineLoad, ScheduleAffineAdd)):
        return node
    if isinstance(node, ScheduleUnavailable):
        return node
    if isinstance(node, ScheduleSequence):
        return replace(
            node,
            children=tuple(
                _remap_schedule_memory_streams(child, stream_indices)
                for child in node.children
            ),
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
            children=tuple(
                _retag_schedule_stage(child, stage) for child in node.children
            ),
        )
    if isinstance(node, ScheduleRepeat):
        body = _retag_schedule_stage(node.body, stage)
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


def _bind_unindexed_memory_instructions(
    node: ScheduleNode, memory_events: list[MemoryEvent]
) -> ScheduleNode:
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
    available = [
        event for event in memory_events if event.stream_index not in bound_indices
    ]
    remaining_multiplicity = {
        event.stream_index: event.multiplicity for event in available
    }

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
        raise ValueError(
            "DMA events are not represented in the ordered schedule: "
            f"{remaining!r}"
        )
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
                isinstance(body_children[-1], ScheduleInstruction)
                and body_children[-1].opcode == "C_LOOP_END"
            ):
                raise RawAsmCostError(
                    f"hardware-loop repeat {node.name!r} has no trailing C_LOOP_END"
                )
            prefix = _compress_explicit_hardware_loops(
                ScheduleSequence(body_children[:-1])
            )
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
        if not (
            isinstance(child, ScheduleInstruction)
            and child.opcode in {"C_LOOP_START", "C_LOOP_END"}
        ):
            compressed.append(_compress_explicit_hardware_loops(child))
            index += 1
            continue
        if child.opcode == "C_LOOP_END":
            raise RawAsmCostError(
                f"unmatched typed C_LOOP_END in compressed schedule stage {child.stage!r}"
            )

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
            raise RawAsmCostError(
                f"cannot parse typed hardware-loop count from schedule node {child!r}"
            ) from exc
        if loop_count <= 0:
            raise RawAsmCostError(
                f"typed hardware-loop count must be positive, got {loop_count}"
            )

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
            raise RawAsmCostError(
                f"unterminated typed C_LOOP_START in compressed schedule stage {child.stage!r}"
            )

        end = children[end_index]
        assert isinstance(end, ScheduleInstruction) and end.opcode == "C_LOOP_END"
        body = _compress_explicit_hardware_loops(
            ScheduleSequence(children[index + 1 : end_index])
        )
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
    schema_version: ClassVar[int] = 4
    static_opcodes: Counter[str] = field(default_factory=Counter)
    dynamic_opcodes: Counter[str] = field(default_factory=Counter)
    memory_events: list[MemoryEvent] = field(default_factory=list)
    stages: dict[str, StageCost] = field(default_factory=lambda: defaultdict(StageCost))
    schedule: ScheduleSequence = field(default_factory=ScheduleSequence)
    schedule_unavailable_reasons: Counter[str] = field(default_factory=Counter)
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
            "stage_breakdown": {name: stage.to_dict() for name, stage in sorted(self.stages.items())},
            "compressed_schedule": self.schedule.to_dict(),
            "schedule_fidelity": (
                "unavailable" if self.schedule_unavailable_reasons else "ordered_compressed"
            ),
            "schedule_unavailable_reasons": dict(
                sorted(self.schedule_unavailable_reasons.items())
            ),
            "schedule_coverage": {
                "ordered_dynamic_instructions": ordered_count,
                "unavailable_dynamic_instructions": unavailable_count,
                "ordered_fraction": (
                    1.0
                    if self.dynamic_instruction_count == 0
                    else ordered_count / self.dynamic_instruction_count
                ),
                "unavailable_by_reason": dict(sorted(unavailable.items())),
            },
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
        self._schedule_children: list[ScheduleNode] = []

    def emit(self, value: IsaBuilder | Sequence | Iterable[AsmItem]) -> None:
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
                )
            )

    @contextmanager
    def repeated_region(
        self, count: int, *, name: str, repeat_kind: str = "compile_time"
    ):
        """Emit one ordered body and account for ``count`` identical copies.

        This is a cost-only equivalent of compiler unrolling.  It is intended
        for latency summaries whose dynamic addresses do not change opcode or
        DMA geometry.  DMA inside the region is rejected so every physical
        memory stream remains represented explicitly by ``MemoryEvent``.
        """
        if count <= 0:
            raise ValueError(f"repeated-region count must be positive, got {count}")
        before_schedule = len(self._schedule_children)
        before_streams = len(self.trace.memory_events)
        before_static = Counter(self.trace.static_opcodes)
        before_dynamic = Counter(self.trace.dynamic_opcodes)
        before_stages = {
            stage: (Counter(cost.static_opcodes), Counter(cost.dynamic_opcodes))
            for stage, cost in self.trace.stages.items()
        }
        yield
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
            return
        static_delta = self.trace.static_opcodes - before_static
        dynamic_delta = self.trace.dynamic_opcodes - before_dynamic
        self.trace.static_opcodes.update(
            {opcode: value * (count - 1) for opcode, value in static_delta.items()}
        )
        self.trace.dynamic_opcodes.update(
            {opcode: value * (count - 1) for opcode, value in dynamic_delta.items()}
        )
        for stage, cost in self.trace.stages.items():
            old_static, old_dynamic = before_stages.get(stage, (Counter(), Counter()))
            stage_static_delta = cost.static_opcodes - old_static
            stage_dynamic_delta = cost.dynamic_opcodes - old_dynamic
            cost.static_opcodes.update(
                {
                    opcode: value * (count - 1)
                    for opcode, value in stage_static_delta.items()
                }
            )
            cost.dynamic_opcodes.update(
                {
                    opcode: value * (count - 1)
                    for opcode, value in stage_dynamic_delta.items()
                }
            )

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
        enclosing_multiplier = 1
        for count in (*self._raw_loop_stack, *self._typed_loop_stack):
            enclosing_multiplier *= count
        local_dynamic = Counter(
            {
                opcode: int(count)
                for opcode, count in dynamic_opcodes.items()
                if int(count)
            }
        )
        derived = _schedule_opcode_counts(schedule)
        if derived != local_dynamic:
            raise ValueError(
                "ordered kernel schedule drifted from algebraic counts: "
                f"schedule={derived}, counts={local_dynamic}"
            )
        for opcode in set(static_opcodes) | set(dynamic_opcodes):
            static_count = int(static_opcodes.get(opcode, 0))
            dynamic_count = int(dynamic_opcodes.get(opcode, 0))
            if static_count < 0 or dynamic_count < 0:
                raise ValueError(
                    f"negative cost count for {opcode}: "
                    f"{static_count}, {dynamic_count}"
                )
            if static_count or dynamic_count:
                self._record(
                    opcode,
                    static_count,
                    dynamic_count * enclosing_multiplier,
                    stage,
                )
        remapped = _remap_schedule_memory_streams(
            _retag_schedule_stage(schedule, stage), memory_stream_indices
        )
        assert isinstance(remapped, ScheduleSequence)
        self._schedule_children.extend(remapped.children)

    def add_memory_event(
        self,
        *,
        transfer: DmaTransfer,
        multiplicity: int,
        stage: str = "global",
        axes: tuple[RepeatAxis, ...] = (),
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
                raise ValueError(
                    f"DMA repeat axes multiply to {repeat_product}, expected {multiplicity}"
                )
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
                reason = "unstructured_legacy_asm"
                self.trace.schedule_unavailable_reasons[reason] += 1
                scheduled.append(
                    ScheduleUnavailable(
                        reason=reason,
                        stage=stage,
                        dynamic_instruction_count=dynamic_multiplier,
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
                        raise ValueError(
                            f"DMA opcode {item.dma.opcode!r} does not match instruction {item.opcode!r}"
                        )
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
                    self.trace.schedule_unavailable_reasons[
                        "explicit_loop_markers"
                    ] += 1
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
        self.trace.schedule = ScheduleSequence(tuple(self._schedule_children))
        if self.trace.schedule_unavailable_reasons.get("explicit_loop_markers"):
            compressed = _compress_explicit_hardware_loops(self.trace.schedule)
            assert isinstance(compressed, ScheduleSequence)
            self.trace.schedule = compressed
            del self.trace.schedule_unavailable_reasons["explicit_loop_markers"]
        if not self.trace.schedule_unavailable_reasons:
            self.trace.schedule = _bind_unindexed_memory_instructions(
                self.trace.schedule, self.trace.memory_events
            )
        if not self.trace.schedule_unavailable_reasons:
            derived = _schedule_opcode_counts(self.trace.schedule)
            if derived != self.trace.dynamic_opcodes:
                raise ValueError(
                    "compressed schedule opcode counts drifted from CostTrace: "
                    f"schedule={derived}, trace={self.trace.dynamic_opcodes}"
                )
            self.trace.dynamic_opcodes = derived
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
    "ScheduleAffineLoad",
    "ScheduleAffineAdd",
    "ScheduleInstruction",
    "ScheduleNode",
    "ScheduleRepeat",
    "ScheduleSequence",
    "ScheduleUnavailable",
    "opcode_category",
]
