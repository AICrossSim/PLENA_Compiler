"""Typed Python contract for the descriptor-driven X_STATE v2 instruction."""

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum
from typing import TypeAlias

from .generated_contract import (
    STATE_ALGORITHMS,
    STATE_COMMON_FIELDS,
    STATE_COMPLETION_FIELDS,
    STATE_COMPLETION_SIZE,
    STATE_DESCRIPTOR_MAGIC,
    STATE_DESCRIPTOR_SIZE,
    STATE_DESCRIPTOR_VERSION,
    STATE_FLAG_BITS,
    STATE_NO_EVENT,
    STATE_PAYLOAD_FIELDS,
    STATE_PRECISION_BYTES,
    STATE_PRECISIONS,
    STATE_STREAMING_SRAM_OFFSET,
    STATE_SUBOPS,
    X_STATE_OPCODE,
)


class StateAlgorithm(IntEnum):
    MAMBA2 = STATE_ALGORITHMS["MAMBA2"]
    KDA = STATE_ALGORITHMS["KDA"]


class StateSubop(IntEnum):
    PRELOAD = STATE_SUBOPS["PRELOAD"]
    RESET = STATE_SUBOPS["RESET"]
    PREFILL = STATE_SUBOPS["PREFILL"]
    STEP = STATE_SUBOPS["STEP"]
    COMMIT = STATE_SUBOPS["COMMIT"]
    EVICT = STATE_SUBOPS["EVICT"]
    FENCE = STATE_SUBOPS["FENCE"]


class PrecisionCode(IntEnum):
    FP32 = STATE_PRECISIONS["FP32"]
    BF16 = STATE_PRECISIONS["BF16"]
    FP16 = STATE_PRECISIONS["FP16"]
    MX8_B128 = STATE_PRECISIONS["MX8_B128"]

    @property
    def element_bytes(self) -> int:
        return STATE_PRECISION_BYTES[int(self)]


FLAG_LAST_CHUNK = 1 << STATE_FLAG_BITS["LAST_CHUNK"]
FLAG_WRITE_COMPLETION = 1 << STATE_FLAG_BITS["WRITE_COMPLETION"]
FLAG_PROFILE = 1 << STATE_FLAG_BITS["PROFILE"]
KNOWN_FLAGS = FLAG_LAST_CHUNK | FLAG_WRITE_COMPLETION | FLAG_PROFILE
STREAMING_SRAM_OFFSET = STATE_STREAMING_SRAM_OFFSET
NO_EVENT = STATE_NO_EVENT


def _require_uint(name: str, value: int, bits: int) -> None:
    if not isinstance(value, int) or not 0 <= value < 1 << bits:
        raise ValueError(f"{name} must fit u{bits}")


def _f32_bits(value: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(f"{value!r} is not representable as binary32") from error


def _bits_f32(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value))[0]


@dataclass(frozen=True)
class Mamba2Payload:
    head_dim: int = 64
    state_dim: int = 128
    groups: int = 8
    conv_kernel: int = 4
    xbc_offset: int = 4096
    dt_offset: int = 10240
    conv_weight_addr: int = 0
    conv_bias_addr: int = 0
    a_log_addr: int = 0
    dt_bias_addr: int = 0
    d_skip_addr: int = 0
    parameter_scale_addr: int = 0
    dt_min: float = 0.0
    dt_max: float = math.inf

    algorithm = StateAlgorithm.MAMBA2

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt_min", _bits_f32(_f32_bits(self.dt_min)))
        object.__setattr__(self, "dt_max", _bits_f32(_f32_bits(self.dt_max)))

    def validate(self, num_heads: int) -> None:
        for name in ("head_dim", "state_dim", "groups", "conv_kernel"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Mamba2 {name} must be positive")
        if num_heads % self.groups:
            raise ValueError("Mamba2 num_heads must be divisible by groups")
        if self.dt_min < 0 or self.dt_max < self.dt_min:
            raise ValueError("Mamba2 dt limits are invalid")
        for name in ("xbc_offset", "dt_offset"):
            _require_uint(name, getattr(self, name), 32)
        for name in (
            "conv_weight_addr",
            "conv_bias_addr",
            "a_log_addr",
            "dt_bias_addr",
            "d_skip_addr",
            "parameter_scale_addr",
        ):
            _require_uint(name, getattr(self, name), 64)

    def state_elements(self, batch_size: int, num_heads: int) -> int:
        return batch_size * num_heads * self.head_dim * self.state_dim

    def conv_state_elements(self, batch_size: int, num_heads: int) -> int:
        d_inner = num_heads * self.head_dim
        channels = d_inner + 2 * self.groups * self.state_dim
        return batch_size * channels * self.conv_kernel

    def input_elements(self, num_heads: int) -> int:
        d_inner = num_heads * self.head_dim
        return d_inner + (d_inner + 2 * self.groups * self.state_dim) + num_heads

    def output_elements(self, num_heads: int) -> int:
        return num_heads * self.head_dim

    def state_scale_elements(self, batch_size: int, num_heads: int) -> int:
        outer = batch_size * num_heads * self.head_dim
        return outer * math.ceil(self.state_dim / 128)

    def conv_state_scale_elements(self, batch_size: int, num_heads: int) -> int:
        d_inner = num_heads * self.head_dim
        channels = d_inner + 2 * self.groups * self.state_dim
        return batch_size * channels * math.ceil(self.conv_kernel / 128)

    def wire_values(self) -> dict[str, int]:
        values = asdict(self)
        values["dt_min_f32_bits"] = _f32_bits(values.pop("dt_min"))
        values["dt_max_f32_bits"] = _f32_bits(values.pop("dt_max"))
        return values

    @classmethod
    def from_wire(cls, values: dict[str, int]) -> Mamba2Payload:
        return cls(
            **{
                name: value
                for name, value in values.items()
                if not name.startswith("reserved") and not name.endswith("_f32_bits")
            },
            dt_min=_bits_f32(values["dt_min_f32_bits"]),
            dt_max=_bits_f32(values["dt_max_f32_bits"]),
        )


@dataclass(frozen=True)
class KdaPayload:
    key_dim: int = 128
    value_dim: int = 128
    conv_kernel: int = 4
    q_offset: int = 0
    k_offset: int = 12288
    v_offset: int = 24576
    decay_offset: int = 36864
    beta_offset: int = 49152
    q_conv_weight_addr: int = 0
    k_conv_weight_addr: int = 0
    v_conv_weight_addr: int = 0
    q_conv_bias_addr: int = 0
    k_conv_bias_addr: int = 0
    v_conv_bias_addr: int = 0
    a_log_addr: int = 0
    dt_bias_addr: int = 0
    parameter_scale_addr: int = 0
    output_scale: float = 1.0 / math.sqrt(128)
    gate_lower_bound: float = -5.0

    algorithm = StateAlgorithm.KDA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "output_scale", _bits_f32(_f32_bits(self.output_scale))
        )
        object.__setattr__(
            self,
            "gate_lower_bound",
            _bits_f32(_f32_bits(self.gate_lower_bound)),
        )

    def validate(self, num_heads: int) -> None:
        for name in ("key_dim", "value_dim", "conv_kernel"):
            if getattr(self, name) <= 0:
                raise ValueError(f"KDA {name} must be positive")
        if num_heads <= 0:
            raise ValueError("KDA num_heads must be positive")
        if not math.isfinite(self.output_scale) or self.output_scale <= 0:
            raise ValueError("KDA output_scale must be finite and positive")
        if not math.isfinite(self.gate_lower_bound) or self.gate_lower_bound >= 0:
            raise ValueError("KDA gate_lower_bound must be finite and negative")
        for name in ("q_offset", "k_offset", "v_offset", "decay_offset", "beta_offset"):
            _require_uint(name, getattr(self, name), 32)
        for name in (
            "q_conv_weight_addr",
            "k_conv_weight_addr",
            "v_conv_weight_addr",
            "q_conv_bias_addr",
            "k_conv_bias_addr",
            "v_conv_bias_addr",
            "a_log_addr",
            "dt_bias_addr",
            "parameter_scale_addr",
        ):
            _require_uint(name, getattr(self, name), 64)

    def state_elements(self, batch_size: int, num_heads: int) -> int:
        return batch_size * num_heads * self.value_dim * self.key_dim

    def conv_state_elements(self, batch_size: int, num_heads: int) -> int:
        channels = num_heads * (2 * self.key_dim + self.value_dim)
        return batch_size * channels * self.conv_kernel

    def input_elements(self, num_heads: int) -> int:
        return num_heads * (3 * self.key_dim + self.value_dim + 1)

    def output_elements(self, num_heads: int) -> int:
        return num_heads * self.value_dim

    def state_scale_elements(self, batch_size: int, num_heads: int) -> int:
        outer = batch_size * num_heads * self.value_dim
        return outer * math.ceil(self.key_dim / 128)

    def conv_state_scale_elements(self, batch_size: int, num_heads: int) -> int:
        channels = num_heads * (2 * self.key_dim + self.value_dim)
        return batch_size * channels * math.ceil(self.conv_kernel / 128)

    def wire_values(self) -> dict[str, int]:
        values = asdict(self)
        values["output_scale_f32_bits"] = _f32_bits(values.pop("output_scale"))
        values["gate_lower_bound_f32_bits"] = _f32_bits(values.pop("gate_lower_bound"))
        return values

    @classmethod
    def from_wire(cls, values: dict[str, int]) -> KdaPayload:
        return cls(
            **{
                name: value
                for name, value in values.items()
                if not name.startswith("reserved") and not name.endswith("_f32_bits")
            },
            output_scale=_bits_f32(values["output_scale_f32_bits"]),
            gate_lower_bound=_bits_f32(values["gate_lower_bound_f32_bits"]),
        )


StatePayload: TypeAlias = Mamba2Payload | KdaPayload


@dataclass(frozen=True)
class StateIdentity:
    context_id: int
    request_id: int
    layer_id: int
    state_id: int = 0


def _validate_segments(
    stride: int,
    segments: tuple[tuple[int, int, str], ...],
) -> None:
    for index, (start, length, name) in enumerate(segments):
        end = start + length
        if end > stride:
            raise ValueError(f"{name} VRAM range exceeds input_token_stride")
        for other_start, other_length, other_name in segments[:index]:
            other_end = other_start + other_length
            if start < other_end and other_start < end:
                raise ValueError(f"{name} and {other_name} VRAM ranges overlap")


@dataclass(frozen=True)
class StateDescriptor:
    payload: StatePayload
    batch_size: int = 1
    num_heads: int = 64
    sequence_length: int = 1
    token_offset: int = 0
    valid_tokens: int = 1
    chunk_size: int = 128
    state_precision: PrecisionCode = PrecisionCode.FP32
    conv_state_precision: PrecisionCode | None = None
    activation_precision: PrecisionCode = PrecisionCode.BF16
    parameter_precision: PrecisionCode = PrecisionCode.BF16
    accumulator_precision: PrecisionCode = PrecisionCode.FP32
    flags: int = 0
    context_id: int = 0
    request_id: int = 0
    layer_id: int = 0
    state_id: int = 0
    state_sram_offset: int = STREAMING_SRAM_OFFSET
    input_vram_addr: int = 0
    output_vram_addr: int = 0
    input_token_stride: int = 0
    output_token_stride: int = 0
    state_hbm_addr: int = 0
    conv_state_hbm_addr: int = 0
    state_scale_addr: int = 0
    completion_addr: int = 0
    dependency_event: int = NO_EVENT
    completion_event: int = NO_EVENT

    def __post_init__(self) -> None:
        if not isinstance(self.payload, (Mamba2Payload, KdaPayload)):
            raise TypeError("payload must be Mamba2Payload or KdaPayload")
        if self.conv_state_precision is None:
            object.__setattr__(self, "conv_state_precision", self.state_precision)
        for name in ("batch_size", "num_heads", "sequence_length", "valid_tokens", "chunk_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.valid_tokens > self.chunk_size:
            raise ValueError("valid_tokens cannot exceed chunk_size")
        if self.token_offset + self.valid_tokens > self.sequence_length:
            raise ValueError("descriptor token range exceeds sequence_length")
        if self.accumulator_precision != PrecisionCode.FP32:
            raise ValueError("X_STATE v2 accumulation must remain FP32")
        if self.flags & ~KNOWN_FLAGS:
            raise ValueError("descriptor contains unknown flag bits")
        self.payload.validate(self.num_heads)
        if self.input_token_stride == 0:
            object.__setattr__(self, "input_token_stride", self.payload.input_elements(self.num_heads))
        if self.output_token_stride == 0:
            object.__setattr__(self, "output_token_stride", self.payload.output_elements(self.num_heads))
        if self.output_token_stride < self.payload.output_elements(self.num_heads):
            raise ValueError("output_token_stride is smaller than the logical tensor")
        if isinstance(self.payload, Mamba2Payload):
            d_inner = self.num_heads * self.payload.head_dim
            group_elements = self.payload.groups * self.payload.state_dim
            _validate_segments(
                self.input_token_stride,
                (
                    (0, d_inner, "gate"),
                    (self.payload.xbc_offset, d_inner + 2 * group_elements, "xbc"),
                    (self.payload.dt_offset, self.num_heads, "dt"),
                ),
            )
        else:
            key_elements = self.num_heads * self.payload.key_dim
            value_elements = self.num_heads * self.payload.value_dim
            _validate_segments(
                self.input_token_stride,
                (
                    (self.payload.q_offset, key_elements, "q"),
                    (self.payload.k_offset, key_elements, "k"),
                    (self.payload.v_offset, value_elements, "v"),
                    (self.payload.decay_offset, key_elements, "decay"),
                    (self.payload.beta_offset, self.num_heads, "beta"),
                ),
            )
        for name in (
            "context_id",
            "request_id",
            "layer_id",
            "state_id",
            "sequence_length",
            "token_offset",
            "state_sram_offset",
            "input_vram_addr",
            "output_vram_addr",
            "input_token_stride",
            "output_token_stride",
            "dependency_event",
            "completion_event",
        ):
            _require_uint(name, getattr(self, name), 32)
        _require_uint("batch_size", self.batch_size, 16)
        _require_uint("num_heads", self.num_heads, 16)
        _require_uint("valid_tokens", self.valid_tokens, 16)
        _require_uint("chunk_size", self.chunk_size, 16)
        for name in ("state_hbm_addr", "conv_state_hbm_addr", "state_scale_addr", "completion_addr"):
            _require_uint(name, getattr(self, name), 64)
        for name in ("state_hbm_addr", "conv_state_hbm_addr", "state_scale_addr", "completion_addr"):
            address = getattr(self, name)
            if address and address % 64:
                raise ValueError(f"{name} must be 64-byte aligned")
        if self.flags & FLAG_WRITE_COMPLETION and self.completion_addr == 0:
            raise ValueError("WRITE_COMPLETION requires a nonzero completion_addr")
        if (
            self.state_precision == PrecisionCode.MX8_B128
            or self.conv_state_precision == PrecisionCode.MX8_B128
        ) and self.state_scale_addr == 0:
            raise ValueError("MX8 state storage requires state_scale_addr")
        if self.parameter_precision == PrecisionCode.MX8_B128:
            if self.payload.parameter_scale_addr == 0:
                raise ValueError("MX8 parameter storage requires parameter_scale_addr")
        if not self.streaming and self.state_sram_offset % 64:
            raise ValueError("resident state_sram_offset must be 64-byte aligned")
        if any(
            value >= 1 << 32
            for value in (
                self.state_bytes,
                self.conv_state_bytes,
                self.state_scale_bytes,
                self.conv_state_scale_bytes,
                self.resident_bytes,
            )
        ):
            raise ValueError("state allocation does not fit the descriptor")
    @property
    def algorithm(self) -> StateAlgorithm:
        return self.payload.algorithm

    @property
    def identity(self) -> StateIdentity:
        return StateIdentity(self.context_id, self.request_id, self.layer_id, self.state_id)

    @property
    def streaming(self) -> bool:
        return self.state_sram_offset == STREAMING_SRAM_OFFSET

    @property
    def state_bytes(self) -> int:
        return self.payload.state_elements(self.batch_size, self.num_heads) * self.state_precision.element_bytes

    @property
    def conv_state_bytes(self) -> int:
        assert self.conv_state_precision is not None
        return (
            self.payload.conv_state_elements(self.batch_size, self.num_heads)
            * self.conv_state_precision.element_bytes
        )

    @property
    def state_scale_bytes(self) -> int:
        if self.state_precision != PrecisionCode.MX8_B128:
            return 0
        return self.payload.state_scale_elements(self.batch_size, self.num_heads)

    @property
    def conv_state_scale_bytes(self) -> int:
        if self.conv_state_precision != PrecisionCode.MX8_B128:
            return 0
        return self.payload.conv_state_scale_elements(self.batch_size, self.num_heads)

    @property
    def resident_bytes(self) -> int:
        return (
            self.state_bytes
            + self.conv_state_bytes
            + self.state_scale_bytes
            + self.conv_state_scale_bytes
        )

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["algorithm"] = self.algorithm.name.lower()
        result["state_precision"] = self.state_precision.name.lower()
        assert self.conv_state_precision is not None
        result["conv_state_precision"] = self.conv_state_precision.name.lower()
        result["activation_precision"] = self.activation_precision.name.lower()
        result["parameter_precision"] = self.parameter_precision.name.lower()
        result["accumulator_precision"] = self.accumulator_precision.name.lower()
        result["state_bytes"] = self.state_bytes
        result["conv_state_bytes"] = self.conv_state_bytes
        result["state_scale_bytes"] = self.state_scale_bytes
        result["conv_state_scale_bytes"] = self.conv_state_scale_bytes
        result["resident_bytes"] = self.resident_bytes
        result["streaming"] = self.streaming
        return result

    def pack(self) -> bytes:
        data = bytearray(STATE_DESCRIPTOR_SIZE)
        common = {
            "magic": STATE_DESCRIPTOR_MAGIC,
            "version": STATE_DESCRIPTOR_VERSION,
            "size_bytes": STATE_DESCRIPTOR_SIZE,
            "algorithm": int(self.algorithm),
            "state_precision": int(self.state_precision),
            "activation_precision": int(self.activation_precision),
            "accumulator_precision": int(self.accumulator_precision),
            "parameter_precision": int(self.parameter_precision),
            "conv_state_precision": int(self.conv_state_precision),
            "flags": self.flags,
            "context_id": self.context_id,
            "request_id": self.request_id,
            "layer_id": self.layer_id,
            "state_id": self.state_id,
            "batch_size": self.batch_size,
            "num_heads": self.num_heads,
            "sequence_length": self.sequence_length,
            "token_offset": self.token_offset,
            "valid_tokens": self.valid_tokens,
            "chunk_size": self.chunk_size,
            "state_sram_offset": self.state_sram_offset,
            "state_bytes": self.state_bytes,
            "conv_state_bytes": self.conv_state_bytes,
            "input_vram_addr": self.input_vram_addr,
            "output_vram_addr": self.output_vram_addr,
            "input_token_stride": self.input_token_stride,
            "output_token_stride": self.output_token_stride,
            "state_hbm_addr": self.state_hbm_addr,
            "conv_state_hbm_addr": self.conv_state_hbm_addr,
            "state_scale_addr": self.state_scale_addr,
            "completion_addr": self.completion_addr,
            "dependency_event": self.dependency_event,
            "completion_event": self.completion_event,
            "state_scale_bytes": self.state_scale_bytes,
            "conv_state_scale_bytes": self.conv_state_scale_bytes,
        }
        _pack_fields(data, STATE_COMMON_FIELDS, common)
        _pack_fields(data, STATE_PAYLOAD_FIELDS[self.algorithm.name], self.payload.wire_values())
        return bytes(data)

    @classmethod
    def unpack(cls, data: bytes) -> StateDescriptor:
        if len(data) != STATE_DESCRIPTOR_SIZE:
            raise ValueError(f"descriptor must be exactly {STATE_DESCRIPTOR_SIZE} bytes")
        common = _unpack_fields(data, STATE_COMMON_FIELDS)
        if (
            common["magic"],
            common["version"],
            common["size_bytes"],
        ) != (STATE_DESCRIPTOR_MAGIC, STATE_DESCRIPTOR_VERSION, STATE_DESCRIPTOR_SIZE):
            raise ValueError("incompatible X_STATE descriptor header")
        if common["reserved0"]:
            raise ValueError("common reserved descriptor bytes must be zero")
        try:
            algorithm = StateAlgorithm(common.pop("algorithm"))
            state_precision = PrecisionCode(common.pop("state_precision"))
            activation_precision = PrecisionCode(common.pop("activation_precision"))
            accumulator_precision = PrecisionCode(common.pop("accumulator_precision"))
            parameter_precision = PrecisionCode(common.pop("parameter_precision"))
            conv_state_precision = PrecisionCode(common.pop("conv_state_precision"))
        except ValueError as error:
            raise ValueError("descriptor contains an unknown enum value") from error
        payload_values = _unpack_fields(data, STATE_PAYLOAD_FIELDS[algorithm.name])
        if any(value for name, value in payload_values.items() if name.startswith("reserved")):
            raise ValueError("payload reserved descriptor bytes must be zero")
        payload = (
            Mamba2Payload.from_wire(payload_values)
            if algorithm == StateAlgorithm.MAMBA2
            else KdaPayload.from_wire(payload_values)
        )
        expected_state_bytes = common.pop("state_bytes")
        expected_conv_state_bytes = common.pop("conv_state_bytes")
        expected_state_scale_bytes = common.pop("state_scale_bytes")
        expected_conv_state_scale_bytes = common.pop("conv_state_scale_bytes")
        for name in ("magic", "version", "size_bytes", "reserved0"):
            common.pop(name)
        descriptor = cls(
            payload=payload,
            state_precision=state_precision,
            conv_state_precision=conv_state_precision,
            activation_precision=activation_precision,
            accumulator_precision=accumulator_precision,
            parameter_precision=parameter_precision,
            **common,
        )
        if descriptor.state_bytes != expected_state_bytes:
            raise ValueError("descriptor state_bytes does not match shape and precision")
        if descriptor.conv_state_bytes != expected_conv_state_bytes:
            raise ValueError("descriptor conv_state_bytes does not match shape and precision")
        if descriptor.state_scale_bytes != expected_state_scale_bytes:
            raise ValueError("descriptor state_scale_bytes does not match shape and precision")
        if descriptor.conv_state_scale_bytes != expected_conv_state_scale_bytes:
            raise ValueError(
                "descriptor conv_state_scale_bytes does not match shape and precision"
            )
        return descriptor


def _pack_fields(data: bytearray, fields: dict[str, tuple[int, str]], values: dict[str, int]) -> None:
    formats = {"u8": "B", "u16": "H", "u32": "I", "u64": "Q"}
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"unknown wire fields: {sorted(unknown)}")
    for name, value in values.items():
        offset, field_type = fields[name]
        if field_type.startswith("bytes"):
            raise ValueError(f"cannot assign reserved byte field {name}")
        try:
            struct.pack_into(f"<{formats[field_type]}", data, offset, value)
        except struct.error as error:
            raise ValueError(f"{name}={value} does not fit {field_type}") from error


def _unpack_fields(data: bytes, fields: dict[str, tuple[int, str]]) -> dict[str, int]:
    formats = {"u8": "B", "u16": "H", "u32": "I", "u64": "Q"}
    result: dict[str, int] = {}
    for name, (offset, field_type) in fields.items():
        if field_type.startswith("bytes"):
            width = int(field_type.removeprefix("bytes"))
            result[name] = int.from_bytes(data[offset : offset + width], "little")
        else:
            result[name] = struct.unpack_from(f"<{formats[field_type]}", data, offset)[0]
    return result


def encode_instruction(
    context_gp: int,
    descriptor_offset_gp: int,
    descriptor_hbm_reg: int,
    queue_id: int,
    subop: StateSubop | int,
) -> int:
    for name, value in (
        ("context_gp", context_gp),
        ("descriptor_offset_gp", descriptor_offset_gp),
        ("descriptor_hbm_reg", descriptor_hbm_reg),
        ("queue_id", queue_id),
    ):
        if not isinstance(value, int) or not 0 <= value < 16:
            raise ValueError(f"{name} must fit the four-bit instruction field")
    if descriptor_hbm_reg >= 8:
        raise ValueError("descriptor_hbm_reg must select a0..a7")
    subop = StateSubop(subop)
    if subop == StateSubop.FENCE and any((context_gp, descriptor_offset_gp, descriptor_hbm_reg)):
        raise ValueError("FENCE requires canonical zero descriptor operands")
    return (
        X_STATE_OPCODE
        | (context_gp << 6)
        | (descriptor_offset_gp << 10)
        | (descriptor_hbm_reg << 14)
        | (queue_id << 18)
        | (int(subop) << 22)
    )


def decode_instruction(word: int) -> dict[str, int | StateSubop]:
    if not isinstance(word, int) or not 0 <= word < 1 << 32:
        raise ValueError("instruction must be an unsigned 32-bit word")
    if word >> 26 or word & 0x3F != X_STATE_OPCODE:
        raise ValueError("instruction is not canonical X_STATE")
    result: dict[str, int | StateSubop] = {
        "context_gp": (word >> 6) & 0xF,
        "descriptor_offset_gp": (word >> 10) & 0xF,
        "descriptor_hbm_reg": (word >> 14) & 0xF,
        "queue_id": (word >> 18) & 0xF,
        "subop": StateSubop((word >> 22) & 0xF),
    }
    if result["descriptor_hbm_reg"] >= 8:
        raise ValueError("descriptor_hbm_reg does not select a0..a7")
    if result["subop"] == StateSubop.FENCE and any(
        result[name] for name in ("context_gp", "descriptor_offset_gp", "descriptor_hbm_reg")
    ):
        raise ValueError("FENCE has nonzero descriptor operands")
    return result


@dataclass(frozen=True)
class StateCommand:
    subop: StateSubop
    descriptor: StateDescriptor | None = None
    context_gp: int = 1
    descriptor_offset_gp: int = 2
    descriptor_hbm_reg: int = 0
    queue_id: int = 0

    def __post_init__(self) -> None:
        if self.subop == StateSubop.FENCE:
            if self.descriptor is not None:
                raise ValueError("FENCE must not fetch a descriptor")
            object.__setattr__(self, "context_gp", 0)
            object.__setattr__(self, "descriptor_offset_gp", 0)
            object.__setattr__(self, "descriptor_hbm_reg", 0)
        elif self.descriptor is None:
            raise ValueError(f"{self.subop.name} requires a descriptor")
        if self.subop == StateSubop.STEP and self.descriptor is not None:
            if self.descriptor.valid_tokens != 1:
                raise ValueError("STEP requires valid_tokens=1")

    @property
    def instruction_word(self) -> int:
        return encode_instruction(
            self.context_gp,
            self.descriptor_offset_gp,
            self.descriptor_hbm_reg,
            self.queue_id,
            self.subop,
        )


class StateLocation(StrEnum):
    HBM_CLEAN = "hbm_clean"
    RESIDENT_CLEAN = "resident_clean"
    RESIDENT_DIRTY = "resident_dirty"


class StateLifecycle:
    """Compiler-side checker for architected state ownership transitions."""

    def __init__(self) -> None:
        self.locations: dict[StateIdentity, StateLocation] = {}

    def seed_hbm(self, key: StateIdentity) -> None:
        self.locations[key] = StateLocation.HBM_CLEAN

    def apply(self, descriptor: StateDescriptor, subop: StateSubop) -> StateLocation:
        key = descriptor.identity
        location = self.locations.get(key, StateLocation.HBM_CLEAN)
        if descriptor.streaming:
            if subop in {StateSubop.PRELOAD, StateSubop.COMMIT, StateSubop.EVICT}:
                raise ValueError(f"{subop.name} is invalid for streaming state {key}")
            if subop not in {StateSubop.RESET, StateSubop.PREFILL, StateSubop.STEP}:
                raise ValueError(f"unsupported streaming transition {subop.name}")
            self.locations[key] = StateLocation.HBM_CLEAN
            return StateLocation.HBM_CLEAN
        if subop == StateSubop.PRELOAD:
            if location != StateLocation.HBM_CLEAN:
                raise ValueError(f"cannot preload {key} from {location}")
            location = StateLocation.RESIDENT_CLEAN
        elif subop == StateSubop.RESET:
            if location not in {StateLocation.HBM_CLEAN, StateLocation.RESIDENT_CLEAN}:
                raise ValueError(f"cannot reset dirty state {key}")
            location = StateLocation.RESIDENT_DIRTY
        elif subop in {StateSubop.PREFILL, StateSubop.STEP}:
            if location not in {StateLocation.RESIDENT_CLEAN, StateLocation.RESIDENT_DIRTY}:
                raise ValueError(f"cannot execute {subop.name} with nonresident state {key}")
            location = StateLocation.RESIDENT_DIRTY
        elif subop == StateSubop.COMMIT:
            if location != StateLocation.RESIDENT_DIRTY:
                raise ValueError(f"cannot commit non-dirty state {key}")
            location = StateLocation.RESIDENT_CLEAN
        elif subop == StateSubop.EVICT:
            if location != StateLocation.RESIDENT_CLEAN:
                raise ValueError(f"cannot evict non-clean state {key}")
            location = StateLocation.HBM_CLEAN
        else:
            raise ValueError(f"{subop.name} does not apply to one descriptor state")
        self.locations[key] = location
        return location


@dataclass(frozen=True)
class StateCompletion:
    status: int
    completion_event: int = NO_EVENT
    elapsed_cycles: int = 0

    def pack(self) -> bytes:
        data = bytearray(STATE_COMPLETION_SIZE)
        _pack_fields(data, STATE_COMPLETION_FIELDS, asdict(self))
        return bytes(data)

    @classmethod
    def unpack(cls, data: bytes) -> StateCompletion:
        if len(data) != STATE_COMPLETION_SIZE:
            raise ValueError(f"completion must be exactly {STATE_COMPLETION_SIZE} bytes")
        return cls(**_unpack_fields(data, STATE_COMPLETION_FIELDS))
