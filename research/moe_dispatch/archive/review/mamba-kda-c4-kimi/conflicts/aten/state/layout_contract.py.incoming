"""Binary contract for the programmable L_SCATTER_M layout engine."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum

from .contract import PrecisionCode, StateDescriptor
from .projection import ProjectionFlow, ProjectionLayout, ProjectionScatterPlan


L_SCATTER_M_OPCODE = 0x3F
LAYOUT_DESCRIPTOR_MAGIC = 0x314D_534C
LAYOUT_DESCRIPTOR_VERSION = 1
LAYOUT_DESCRIPTOR_SIZE = 256
LAYOUT_DESCRIPTOR_ALIGNMENT = 64
LAYOUT_FIELD_OFFSET = 80
LAYOUT_FIELD_SIZE = 24
LAYOUT_MAX_FIELDS = 7
#: Byte offset of the consumer-lane trailer. The field records end at
#: ``LAYOUT_FIELD_OFFSET + LAYOUT_MAX_FIELDS * LAYOUT_FIELD_SIZE`` (248), so the
#: last eight bytes carry the packet geometry that the skew is tuned against.
#: Without it the consumer lane widths are a third, uncontracted source of truth
#: hardcoded independently in the emulator.
LAYOUT_TRAILER_OFFSET = LAYOUT_FIELD_OFFSET + LAYOUT_MAX_FIELDS * LAYOUT_FIELD_SIZE
LAYOUT_CONTRACT_NAME = "plena-l-scatter-m-v1"


class LayoutMode(IntEnum):
    ROW_MAJOR = 0
    TRANSPOSE = 1
    MAMBA_SKEW = 2
    KDA_SKEW = 3
    CUSTOM = 4


class LayoutSkew(IntEnum):
    NONE = 0
    LOCAL_ROW = 1
    FIELD = 2
    GROUP = 3


class LayoutConsumer(IntEnum):
    STATE = 0
    VECTOR = 1


class LayoutFlow(IntEnum):
    BUFFERED = 0
    FIFO_WITH_SPILL = 1


class LayoutFieldId(IntEnum):
    MAMBA_GATE = 0
    MAMBA_X = 1
    MAMBA_B = 2
    MAMBA_C = 3
    MAMBA_DT = 4
    KDA_Q = 16
    KDA_K = 17
    KDA_V = 18
    KDA_DECAY = 19
    KDA_BETA = 20


def layout_contract_document() -> dict[str, object]:
    """Return the stable executable ABI identity embedded in lowered traces."""
    return {
        "contract": LAYOUT_CONTRACT_NAME,
        "version": LAYOUT_DESCRIPTOR_VERSION,
        "instruction_opcode": L_SCATTER_M_OPCODE,
        "descriptor_magic": LAYOUT_DESCRIPTOR_MAGIC,
        "descriptor_size": LAYOUT_DESCRIPTOR_SIZE,
        "descriptor_alignment": LAYOUT_DESCRIPTOR_ALIGNMENT,
        "field_offset": LAYOUT_FIELD_OFFSET,
        "field_size": LAYOUT_FIELD_SIZE,
        "max_fields": LAYOUT_MAX_FIELDS,
        "layout_modes": {item.name: int(item) for item in LayoutMode},
        "flow_modes": {item.name: int(item) for item in LayoutFlow},
        "skew_kinds": {item.name: int(item) for item in LayoutSkew},
        "field_ids": {item.name: int(item) for item in LayoutFieldId},
    }


def layout_contract_sha256() -> str:
    encoded = json.dumps(
        layout_contract_document(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


_FIELD_IDS = {
    ("mamba2", "gate"): LayoutFieldId.MAMBA_GATE,
    ("mamba2", "x"): LayoutFieldId.MAMBA_X,
    ("mamba2", "b"): LayoutFieldId.MAMBA_B,
    ("mamba2", "c"): LayoutFieldId.MAMBA_C,
    ("mamba2", "dt"): LayoutFieldId.MAMBA_DT,
    ("kda", "q"): LayoutFieldId.KDA_Q,
    ("kda", "k"): LayoutFieldId.KDA_K,
    ("kda", "v"): LayoutFieldId.KDA_V,
    ("kda", "decay"): LayoutFieldId.KDA_DECAY,
    ("kda", "beta"): LayoutFieldId.KDA_BETA,
}
_SKEWS = {
    "none": LayoutSkew.NONE,
    "local_row_stride": LayoutSkew.LOCAL_ROW,
    "field_constant": LayoutSkew.FIELD,
    "group_stride": LayoutSkew.GROUP,
}


def _u(name: str, value: int, bits: int) -> None:
    if not isinstance(value, int) or not 0 <= value < 1 << bits:
        raise ValueError(f"{name} must fit u{bits}")


@dataclass(frozen=True)
class LayoutFieldDescriptor:
    field_id: LayoutFieldId
    consumer: LayoutConsumer
    skew_kind: LayoutSkew
    skew_stride: int
    source_offset: int
    values_per_group: int
    physical_offset: int
    physical_span: int
    local_rows: int
    local_lanes: int

    def __post_init__(self) -> None:
        _u("skew_stride", self.skew_stride, 8)
        for name in (
            "source_offset",
            "values_per_group",
            "physical_offset",
            "physical_span",
        ):
            _u(name, getattr(self, name), 32)
        _u("local_rows", self.local_rows, 16)
        _u("local_lanes", self.local_lanes, 16)
        if self.values_per_group == 0:
            raise ValueError("layout field cannot be empty")
        if self.local_rows * self.local_lanes != self.values_per_group:
            raise ValueError("layout field shape does not match values_per_group")
        if self.physical_span < self.values_per_group:
            raise ValueError("layout field physical span is too small")

    def pack(self) -> bytes:
        return struct.pack(
            "<BBBBIIIIHH",
            int(self.field_id),
            int(self.consumer),
            int(self.skew_kind),
            self.skew_stride,
            self.source_offset,
            self.values_per_group,
            self.physical_offset,
            self.physical_span,
            self.local_rows,
            self.local_lanes,
        )

    @classmethod
    def unpack(cls, data: bytes) -> LayoutFieldDescriptor:
        if len(data) != LAYOUT_FIELD_SIZE:
            raise ValueError("layout field record has the wrong size")
        values = struct.unpack("<BBBBIIIIHH", data)
        try:
            return cls(
                LayoutFieldId(values[0]),
                LayoutConsumer(values[1]),
                LayoutSkew(values[2]),
                *values[3:],
            )
        except ValueError as error:
            raise ValueError("layout field contains an unknown enum") from error


@dataclass(frozen=True)
class LayoutScatterDescriptor:
    context_id: int
    request_id: int
    layer_id: int
    token_offset: int
    source_vram_addr: int
    source_token_stride: int
    source_values_per_token: int
    logical_rows: int
    logical_cols: int
    valid_tokens: int
    chunk_size: int
    batch_size: int
    groups: int
    physical_buffer_base_row: int
    physical_token_stride_rows: int
    physical_buffer_rows: int
    group_span_values: int
    banks: int
    ports_per_bank: int
    mode: LayoutMode
    activation_precision: PrecisionCode
    buffer_id: int
    flow: LayoutFlow
    spill_write_values_per_cycle: int
    producer_burst_values: int
    fifo_capacity_values: int
    head_lanes: int
    head_dim_lanes: int
    state_dim_lanes: int
    fields: tuple[LayoutFieldDescriptor, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "context_id",
            "request_id",
            "layer_id",
            "token_offset",
            "source_vram_addr",
            "source_token_stride",
            "source_values_per_token",
            "physical_buffer_base_row",
            "physical_token_stride_rows",
            "physical_buffer_rows",
            "group_span_values",
        ):
            _u(name, getattr(self, name), 32)
        for name in (
            "logical_rows",
            "logical_cols",
            "valid_tokens",
            "chunk_size",
            "batch_size",
            "groups",
            "producer_burst_values",
            "fifo_capacity_values",
        ):
            _u(name, getattr(self, name), 16)
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "head_lanes",
            "head_dim_lanes",
            "state_dim_lanes",
        ):
            _u(name, getattr(self, name), 8)
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "banks",
            "ports_per_bank",
            "buffer_id",
            "spill_write_values_per_cycle",
        ):
            _u(name, getattr(self, name), 8)
        if not self.banks or self.banks & (self.banks - 1):
            raise ValueError("layout banks must be a nonzero power of two")
        if not self.ports_per_bank:
            raise ValueError("ports_per_bank must be positive")
        if not self.spill_write_values_per_cycle:
            raise ValueError("spill_write_values_per_cycle must be positive")
        if len(self.fields) > LAYOUT_MAX_FIELDS:
            raise ValueError("layout descriptor has too many fields")
        if self.source_token_stride < self.source_values_per_token:
            raise ValueError("source token stride is smaller than live values")
        if self.logical_rows * self.logical_cols != self.source_values_per_token:
            raise ValueError("logical matrix shape does not match source values")
        if self.physical_buffer_rows < (
            self.batch_size * self.chunk_size * self.physical_token_stride_rows
        ):
            raise ValueError("physical layout buffer is too small")
        if self.mode == LayoutMode.TRANSPOSE and self.fields:
            raise ValueError("transpose layout must not carry field records")
        if self.mode in {LayoutMode.MAMBA_SKEW, LayoutMode.KDA_SKEW, LayoutMode.CUSTOM}:
            if not self.fields:
                raise ValueError("skewed layout requires field records")
        self._validate_mapping()

    @property
    def mapping_crc32(self) -> int:
        crc = 0
        for source, row, bank in self.mapping():
            crc = zlib.crc32(struct.pack("<III", source, row, bank), crc)
        return crc

    def mapping(self) -> tuple[tuple[int, int, int], ...]:
        if self.mode == LayoutMode.ROW_MAJOR:
            return tuple(
                (
                    source,
                    self.physical_buffer_base_row + source // self.banks,
                    source % self.banks,
                )
                for source in range(self.source_values_per_token)
            )
        if self.mode == LayoutMode.TRANSPOSE:
            mapped = []
            for row in range(self.logical_rows):
                for column in range(self.logical_cols):
                    source = row * self.logical_cols + column
                    physical = column * self.logical_rows + row
                    mapped.append(
                        (
                            source,
                            self.physical_buffer_base_row + physical // self.banks,
                            physical % self.banks,
                        )
                    )
            return tuple(mapped)
        mapped = []
        for field in self.fields:
            for group in range(self.groups):
                for local_row in range(field.local_rows):
                    for lane in range(field.local_lanes):
                        local = local_row * field.local_lanes + lane
                        source = field.source_offset + group * field.values_per_group + local
                        physical = (
                            group * self.group_span_values
                            + field.physical_offset
                            + local
                        )
                        if field.skew_kind == LayoutSkew.LOCAL_ROW:
                            skew = local_row * field.skew_stride
                        elif field.skew_kind == LayoutSkew.FIELD:
                            skew = field.skew_stride
                        elif field.skew_kind == LayoutSkew.GROUP:
                            skew = group * field.skew_stride
                        else:
                            skew = 0
                        mapped.append(
                            (
                                source,
                                self.physical_buffer_base_row + physical // self.banks,
                                (local % self.banks + skew) % self.banks,
                            )
                        )
        return tuple(mapped)

    def _validate_mapping(self) -> None:
        mapping = self.mapping()
        sources = [source for source, _, _ in mapping]
        coordinates = [(row, bank) for _, row, bank in mapping]
        if sorted(sources) != list(range(self.source_values_per_token)):
            raise ValueError("layout descriptor does not cover every source exactly once")
        if len(coordinates) != len(set(coordinates)):
            raise ValueError("layout descriptor aliases two sources")
        limit = self.physical_buffer_base_row + self.physical_buffer_rows
        if any(not self.physical_buffer_base_row <= row < limit for row, _ in coordinates):
            raise ValueError("layout mapping exceeds the physical buffer")
        if any(not 0 <= bank < self.banks for _, bank in coordinates):
            raise ValueError("layout mapping selects an invalid bank")

    def pack(self) -> bytes:
        data = bytearray(LAYOUT_DESCRIPTOR_SIZE)
        struct.pack_into(
            "<IHHIIIIIIIHHHHHHIIII",
            data,
            0,
            LAYOUT_DESCRIPTOR_MAGIC,
            LAYOUT_DESCRIPTOR_VERSION,
            LAYOUT_DESCRIPTOR_SIZE,
            self.context_id,
            self.request_id,
            self.layer_id,
            self.token_offset,
            self.source_vram_addr,
            self.source_token_stride,
            self.source_values_per_token,
            self.logical_rows,
            self.logical_cols,
            self.valid_tokens,
            self.chunk_size,
            self.batch_size,
            self.groups,
            self.physical_buffer_base_row,
            self.physical_token_stride_rows,
            self.physical_buffer_rows,
            self.group_span_values,
        )
        struct.pack_into(
            "<BBBBBBBBHHI",
            data,
            64,
            self.banks,
            self.ports_per_bank,
            len(self.fields),
            int(self.mode),
            int(self.activation_precision),
            self.buffer_id,
            int(self.flow),
            self.spill_write_values_per_cycle,
            self.producer_burst_values,
            self.fifo_capacity_values,
            self.mapping_crc32,
        )
        for index, field in enumerate(self.fields):
            start = LAYOUT_FIELD_OFFSET + index * LAYOUT_FIELD_SIZE
            data[start : start + LAYOUT_FIELD_SIZE] = field.pack()
        struct.pack_into(
            "<BBB",
            data,
            LAYOUT_TRAILER_OFFSET,
            self.head_lanes,
            self.head_dim_lanes,
            self.state_dim_lanes,
        )
        return bytes(data)

    @classmethod
    def unpack(cls, data: bytes) -> LayoutScatterDescriptor:
        if len(data) != LAYOUT_DESCRIPTOR_SIZE:
            raise ValueError("layout descriptor must be exactly 256 bytes")
        header = struct.unpack_from("<IHHIIIIIIIHHHHHHIIII", data, 0)
        if header[:3] != (
            LAYOUT_DESCRIPTOR_MAGIC,
            LAYOUT_DESCRIPTOR_VERSION,
            LAYOUT_DESCRIPTOR_SIZE,
        ):
            raise ValueError("incompatible L_SCATTER_M descriptor header")
        tail = struct.unpack_from("<BBBBBBBBHHI", data, 64)
        (
            banks,
            ports,
            field_count,
            mode,
            precision,
            buffer_id,
            flow,
            spill_width,
            burst,
            fifo,
            crc,
        ) = tail
        if field_count > LAYOUT_MAX_FIELDS:
            raise ValueError("layout descriptor field count is invalid")
        fields = tuple(
            LayoutFieldDescriptor.unpack(
                data[
                    LAYOUT_FIELD_OFFSET + index * LAYOUT_FIELD_SIZE :
                    LAYOUT_FIELD_OFFSET + (index + 1) * LAYOUT_FIELD_SIZE
                ]
            )
            for index in range(field_count)
        )
        used = LAYOUT_FIELD_OFFSET + field_count * LAYOUT_FIELD_SIZE
        if any(data[used:LAYOUT_TRAILER_OFFSET]):
            raise ValueError("layout descriptor unused field bytes must be zero")
        head_lanes, head_dim_lanes, state_dim_lanes = struct.unpack_from(
            "<BBB", data, LAYOUT_TRAILER_OFFSET
        )
        if any(data[LAYOUT_TRAILER_OFFSET + 3 :]):
            raise ValueError("layout descriptor trailer padding must be zero")
        try:
            descriptor = cls(
                *header[3:10],
                *header[10:16],
                *header[16:20],
                banks,
                ports,
                LayoutMode(mode),
                PrecisionCode(precision),
                buffer_id,
                LayoutFlow(flow),
                spill_width,
                burst,
                fifo,
                head_lanes,
                head_dim_lanes,
                state_dim_lanes,
                fields,
            )
        except ValueError as error:
            raise ValueError("layout descriptor contains an unknown enum") from error
        if descriptor.mapping_crc32 != crc:
            raise ValueError("layout descriptor mapping CRC does not match")
        return descriptor

    @classmethod
    def from_projection_plan(
        cls,
        plan: ProjectionScatterPlan,
        state: StateDescriptor,
    ) -> LayoutScatterDescriptor:
        if plan.algorithm == "mamba2":
            skew_mode = LayoutMode.MAMBA_SKEW
        elif plan.algorithm == "kda":
            skew_mode = LayoutMode.KDA_SKEW
        else:
            raise ValueError(f"unsupported layout algorithm {plan.algorithm!r}")
        mode = (
            LayoutMode.ROW_MAJOR
            if plan.layout == ProjectionLayout.ROW_MAJOR
            else skew_mode
        )
        fields = tuple(
            LayoutFieldDescriptor(
                _FIELD_IDS[(plan.algorithm, field.name)],
                LayoutConsumer.STATE
                if field.consumer == "state"
                else LayoutConsumer.VECTOR,
                _SKEWS[field.skew_kind],
                field.skew_stride,
                field.source_offset,
                field.values_per_group,
                field.physical_offset,
                field.physical_span,
                field.local_rows,
                field.local_lanes,
            )
            for field in plan.fields
        )
        records = state.batch_size * state.chunk_size
        physical_rows = records * plan.physical_token_stride_rows
        base_row = plan.physical_buffer_index * physical_rows
        return cls(
            context_id=state.context_id,
            request_id=state.request_id,
            layer_id=state.layer_id,
            token_offset=state.token_offset,
            source_vram_addr=state.input_vram_addr,
            source_token_stride=state.input_token_stride,
            source_values_per_token=plan.source_values_per_token,
            logical_rows=1,
            logical_cols=plan.source_values_per_token,
            valid_tokens=state.valid_tokens,
            chunk_size=state.chunk_size,
            batch_size=state.batch_size,
            groups=plan.groups,
            physical_buffer_base_row=base_row,
            physical_token_stride_rows=plan.physical_token_stride_rows,
            physical_buffer_rows=physical_rows,
            group_span_values=plan.group_span_values,
            banks=plan.banks,
            ports_per_bank=plan.ports_per_bank,
            mode=mode,
            activation_precision=state.activation_precision,
            buffer_id=plan.physical_buffer_index,
            flow=(
                LayoutFlow.BUFFERED
                if plan.flow == ProjectionFlow.BUFFERED
                else LayoutFlow.FIFO_WITH_SPILL
            ),
            spill_write_values_per_cycle=plan.spill_write_values_per_cycle,
            producer_burst_values=plan.producer_burst_values,
            fifo_capacity_values=plan.fifo_capacity_values,
            head_lanes=plan.head_lanes,
            head_dim_lanes=plan.head_dim_lanes,
            state_dim_lanes=plan.state_dim_lanes,
            fields=fields,
        )


def encode_layout_instruction(
    context_gp: int,
    descriptor_offset_gp: int,
    descriptor_hbm_reg: int,
    buffer_id: int,
    mode: LayoutMode,
) -> int:
    for name, value in (
        ("context_gp", context_gp),
        ("descriptor_offset_gp", descriptor_offset_gp),
        ("buffer_id", buffer_id),
    ):
        _u(name, value, 4)
    if not 0 <= descriptor_hbm_reg < 8:
        raise ValueError("descriptor_hbm_reg must select a0..a7")
    return (
        L_SCATTER_M_OPCODE
        | (context_gp << 6)
        | (descriptor_offset_gp << 10)
        | (descriptor_hbm_reg << 14)
        | (buffer_id << 18)
        | (int(mode) << 22)
    )


def decode_layout_instruction(word: int) -> dict[str, int | LayoutMode]:
    _u("instruction", word, 32)
    if word & 0x3F != L_SCATTER_M_OPCODE or word >> 26:
        raise ValueError("word is not a canonical L_SCATTER_M instruction")
    result: dict[str, int | LayoutMode] = {
        "context_gp": (word >> 6) & 0xF,
        "descriptor_offset_gp": (word >> 10) & 0xF,
        "descriptor_hbm_reg": (word >> 14) & 0xF,
        "buffer_id": (word >> 18) & 0xF,
        "mode": LayoutMode((word >> 22) & 0xF),
    }
    if int(result["descriptor_hbm_reg"]) >= 8:
        raise ValueError("L_SCATTER_M descriptor_hbm_reg must select a0..a7")
    return result


__all__ = [
    "LAYOUT_DESCRIPTOR_ALIGNMENT",
    "LAYOUT_DESCRIPTOR_MAGIC",
    "LAYOUT_DESCRIPTOR_SIZE",
    "L_SCATTER_M_OPCODE",
    "LayoutConsumer",
    "LayoutFieldDescriptor",
    "LayoutFieldId",
    "LayoutFlow",
    "LayoutMode",
    "LayoutScatterDescriptor",
    "LayoutSkew",
    "decode_layout_instruction",
    "encode_layout_instruction",
]
