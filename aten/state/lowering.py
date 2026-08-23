"""Lower semantic X_STATE schedules into descriptor memory and register writes."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Protocol

from .contract import StateDescriptor, StateSubop, decode_instruction
from .generated_contract import STATE_DESCRIPTOR_ALIGNMENT, STATE_DESCRIPTOR_SIZE
from .layout_contract import (
    LAYOUT_DESCRIPTOR_ALIGNMENT,
    LAYOUT_DESCRIPTOR_SIZE,
    LayoutScatterDescriptor,
    encode_layout_instruction,
    layout_contract_document,
    layout_contract_sha256,
)
from .projection import (
    ProjectionScatterPlan,
    projection_contract_document,
    projection_contract_sha256,
)


class TraceEventLike(Protocol):
    index: int
    operation: str
    descriptor: StateDescriptor | None
    instruction_word: int | None
    projection_scatter: ProjectionScatterPlan | None


class ScheduleTraceLike(Protocol):
    events: tuple[TraceEventLike, ...]


@dataclass(frozen=True)
class RegisterWrite:
    register_class: str
    register_index: int
    value: int


@dataclass(frozen=True)
class LoweredStateCommand:
    event_index: int
    operation: str
    instruction_word: int
    descriptor_address: int | None
    descriptor_offset: int | None
    register_writes: tuple[RegisterWrite, ...]


@dataclass(frozen=True)
class LoweredProjectionScatter:
    event_index: int
    operation: str
    plan: ProjectionScatterPlan


@dataclass(frozen=True)
class LoweredLayoutCommand:
    event_index: int
    operation: str
    instruction_word: int
    descriptor_address: int
    descriptor_offset: int
    register_writes: tuple[RegisterWrite, ...]
    descriptor: LayoutScatterDescriptor


@dataclass(frozen=True)
class LoweredStateTrace:
    descriptor_base: int
    descriptor_image: bytes
    commands: tuple[LoweredStateCommand, ...]
    projection_scatters: tuple[LoweredProjectionScatter, ...] = ()
    layout_descriptor_base: int = 0
    layout_descriptor_image: bytes = b""
    layout_commands: tuple[LoweredLayoutCommand, ...] = ()

    @property
    def descriptor_count(self) -> int:
        return len(self.descriptor_image) // STATE_DESCRIPTOR_SIZE

    @property
    def layout_descriptor_count(self) -> int:
        return len(self.layout_descriptor_image) // LAYOUT_DESCRIPTOR_SIZE

    def to_dict(self) -> dict[str, object]:
        return {
            "descriptor_base": self.descriptor_base,
            "descriptor_count": self.descriptor_count,
            "descriptor_image_sha256": hashlib.sha256(
                self.descriptor_image
            ).hexdigest(),
            "descriptor_image_hex": self.descriptor_image.hex(),
            "layout_descriptor_base": self.layout_descriptor_base,
            "layout_descriptor_count": self.layout_descriptor_count,
            "layout_descriptor_image_sha256": hashlib.sha256(
                self.layout_descriptor_image
            ).hexdigest(),
            "layout_descriptor_image_hex": self.layout_descriptor_image.hex(),
            "layout_contract": {
                **layout_contract_document(),
                "sha256": layout_contract_sha256(),
            },
            "projection_scatter_contract": {
                **projection_contract_document(),
                "sha256": projection_contract_sha256(),
            },
            "projection_scatters": [
                {
                    "event_index": scatter.event_index,
                    "operation": scatter.operation,
                    "plan": scatter.plan.to_dict(),
                }
                for scatter in self.projection_scatters
            ],
            "layout_commands": [
                {
                    "event_index": command.event_index,
                    "operation": command.operation,
                    "instruction_word": f"0x{command.instruction_word:08x}",
                    "descriptor_address": command.descriptor_address,
                    "descriptor_offset": command.descriptor_offset,
                    "register_writes": [
                        asdict(write) for write in command.register_writes
                    ],
                    "descriptor": asdict(command.descriptor),
                }
                for command in self.layout_commands
            ],
            "commands": [
                {
                    **asdict(command),
                    "instruction_word": f"0x{command.instruction_word:08x}",
                }
                for command in self.commands
            ],
        }


def lower_state_trace(
    trace: ScheduleTraceLike,
    *,
    descriptor_base: int = 0x7000_0000,
) -> LoweredStateTrace:
    if descriptor_base < 0 or descriptor_base % STATE_DESCRIPTOR_ALIGNMENT:
        raise ValueError("descriptor_base must be non-negative and descriptor aligned")
    image = bytearray()
    commands: list[LoweredStateCommand] = []
    projection_scatters: list[LoweredProjectionScatter] = []
    pending_layouts: list[
        tuple[int, str, ProjectionScatterPlan, StateDescriptor]
    ] = []
    for event in trace.events:
        if event.projection_scatter is not None:
            if event.operation != "PROJECTION_SCATTER":
                raise ValueError("projection scatter plan is attached to the wrong operation")
            if event.descriptor is None:
                raise ValueError("projection scatter event has no state descriptor")
            projection_scatters.append(
                LoweredProjectionScatter(event.index, event.operation, event.projection_scatter)
            )
            pending_layouts.append(
                (
                    event.index,
                    event.operation,
                    event.projection_scatter,
                    event.descriptor,
                )
            )
        if event.instruction_word is None:
            continue
        fields = decode_instruction(event.instruction_word)
        subop = fields["subop"]
        assert isinstance(subop, StateSubop)
        if subop == StateSubop.FENCE:
            if event.descriptor is not None:
                raise ValueError("FENCE event unexpectedly carries a descriptor")
            commands.append(
                LoweredStateCommand(
                    event_index=event.index,
                    operation=event.operation,
                    instruction_word=event.instruction_word,
                    descriptor_address=None,
                    descriptor_offset=None,
                    register_writes=(),
                )
            )
            continue
        if event.descriptor is None:
            raise ValueError(f"{event.operation} event has no descriptor")
        packed = event.descriptor.pack()
        if StateDescriptor.unpack(packed) != event.descriptor:
            raise ValueError("descriptor failed Compiler pack/unpack round trip")
        offset = len(image)
        if offset >= 1 << 32:
            raise ValueError("descriptor image offset does not fit GP register")
        address = descriptor_base + offset
        if address >= 1 << 64:
            raise ValueError("descriptor address does not fit HBM register")
        image.extend(packed)
        commands.append(
            LoweredStateCommand(
                event_index=event.index,
                operation=event.operation,
                instruction_word=event.instruction_word,
                descriptor_address=address,
                descriptor_offset=offset,
                register_writes=(
                    RegisterWrite(
                        "gp",
                        int(fields["context_gp"]),
                        event.descriptor.context_id,
                    ),
                    RegisterWrite(
                        "gp",
                        int(fields["descriptor_offset_gp"]),
                        offset,
                    ),
                    RegisterWrite(
                        "hbm",
                        int(fields["descriptor_hbm_reg"]),
                        descriptor_base,
                    ),
                ),
            )
        )
    layout_base = _align_up(
        descriptor_base + len(image), LAYOUT_DESCRIPTOR_ALIGNMENT
    )
    layout_image = bytearray()
    layout_commands: list[LoweredLayoutCommand] = []
    for event_index, operation, plan, state_descriptor in pending_layouts:
        descriptor = LayoutScatterDescriptor.from_projection_plan(
            plan, state_descriptor
        )
        packed = descriptor.pack()
        if LayoutScatterDescriptor.unpack(packed) != descriptor:
            raise ValueError("layout descriptor failed Compiler pack/unpack round trip")
        offset = len(layout_image)
        address = layout_base + offset
        if address >= 1 << 64:
            raise ValueError("layout descriptor address does not fit HBM register")
        instruction_word = encode_layout_instruction(
            context_gp=1,
            descriptor_offset_gp=2,
            descriptor_hbm_reg=0,
            buffer_id=descriptor.buffer_id,
            mode=descriptor.mode,
        )
        layout_image.extend(packed)
        layout_commands.append(
            LoweredLayoutCommand(
                event_index=event_index,
                operation=operation,
                instruction_word=instruction_word,
                descriptor_address=address,
                descriptor_offset=offset,
                register_writes=(
                    RegisterWrite("gp", 1, descriptor.context_id),
                    RegisterWrite("gp", 2, offset),
                    RegisterWrite("hbm", 0, layout_base),
                ),
                descriptor=descriptor,
            )
        )
    return LoweredStateTrace(
        descriptor_base=descriptor_base,
        descriptor_image=bytes(image),
        commands=tuple(commands),
        projection_scatters=tuple(projection_scatters),
        layout_descriptor_base=layout_base,
        layout_descriptor_image=bytes(layout_image),
        layout_commands=tuple(layout_commands),
    )


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
