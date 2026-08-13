"""Executable semantic contract for the proposed descriptor-driven Mamba ISA."""

from __future__ import annotations

import struct
from dataclasses import asdict, dataclass
from enum import IntEnum, StrEnum


MAMBA_OPCODE = 0x39
DESCRIPTOR_MAGIC = 0x4D324D42  # "BM2M" as a little-endian u32
DESCRIPTOR_VERSION = 1
DESCRIPTOR_SIZE = 256


class MambaSubop(IntEnum):
    STATE_PREFETCH = 0
    STATE_RESET = 1
    PREFILL = 2
    STEP = 3
    STATE_COMMIT = 4
    STATE_EVICT = 5
    WAIT = 6


class PrecisionCode(IntEnum):
    FP32 = 0
    BF16 = 1
    FP16 = 2
    MX8_B128 = 3


class ProjectionLayout(IntEnum):
    ROW_MAJOR = 0
    GROUP_MAJOR_SKEWED = 1


class StateLocation(StrEnum):
    HBM_CLEAN = "hbm_clean"
    RESIDENT_CLEAN = "resident_clean"
    RESIDENT_DIRTY = "resident_dirty"


FLAG_BC_GROUP_SHARED = 1 << 0
FLAG_CONTINUE_STATE = 1 << 1
FLAG_LAST_CHUNK = 1 << 2


_U16_FIELDS = {
    "num_heads": 32,
    "head_dim": 34,
    "state_dim": 36,
    "groups": 38,
    "conv_kernel": 40,
    "chunk_size": 42,
    "valid_tokens": 44,
    "layer_id": 46,
    "cache_slot": 48,
}
_U32_FIELDS = {
    "flags": 8,
    "batch_size": 12,
    "sequence_length": 16,
    "request_id": 20,
    "token_offset": 24,
    "completion_id": 28,
}
_U8_FIELDS = {
    "state_precision": 50,
    "activation_precision": 51,
    "projection_layout": 52,
}
_U64_FIELDS = {
    "projection_addr": 64,
    "scan_output_addr": 72,
    "state_addr": 80,
    "conv_state_addr": 88,
    "conv_weight_addr": 96,
    "conv_bias_addr": 104,
    "a_log_addr": 112,
    "dt_bias_addr": 120,
    "d_skip_addr": 128,
    "norm_weight_addr": 136,
    "state_scale_addr": 144,
    "projection_scale_addr": 152,
    "completion_addr": 160,
}


@dataclass(frozen=True)
class MambaDescriptor:
    batch_size: int = 1
    sequence_length: int = 1
    num_heads: int = 64
    head_dim: int = 64
    state_dim: int = 128
    groups: int = 8
    conv_kernel: int = 4
    chunk_size: int = 128
    state_precision: PrecisionCode = PrecisionCode.FP32
    activation_precision: PrecisionCode = PrecisionCode.BF16
    projection_layout: ProjectionLayout = ProjectionLayout.GROUP_MAJOR_SKEWED
    flags: int = FLAG_BC_GROUP_SHARED
    request_id: int = 0
    layer_id: int = 0
    cache_slot: int = 0
    token_offset: int = 0
    valid_tokens: int = 1
    completion_id: int = 0
    projection_addr: int = 0
    scan_output_addr: int = 0
    state_addr: int = 0
    conv_state_addr: int = 0
    conv_weight_addr: int = 0
    conv_bias_addr: int = 0
    a_log_addr: int = 0
    dt_bias_addr: int = 0
    d_skip_addr: int = 0
    norm_weight_addr: int = 0
    state_scale_addr: int = 0
    projection_scale_addr: int = 0
    completion_addr: int = 0

    def __post_init__(self) -> None:
        positive = (
            "batch_size",
            "sequence_length",
            "num_heads",
            "head_dim",
            "state_dim",
            "groups",
            "conv_kernel",
            "chunk_size",
            "valid_tokens",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_heads % self.groups:
            raise ValueError("num_heads must be divisible by groups")
        if self.valid_tokens > self.chunk_size:
            raise ValueError("valid_tokens cannot exceed chunk_size")
        if self.token_offset + self.valid_tokens > self.sequence_length:
            raise ValueError("descriptor token range exceeds sequence_length")
        for name in _U64_FIELDS:
            value = getattr(self, name)
            if not 0 <= value < 2**64:
                raise ValueError(f"{name} must fit u64")

    @property
    def heads_per_group(self) -> int:
        return self.num_heads // self.groups

    @property
    def state_elements(self) -> int:
        return self.batch_size * self.num_heads * self.head_dim * self.state_dim

    def to_dict(self) -> dict:
        result = asdict(self)
        result["state_precision"] = self.state_precision.name.lower()
        result["activation_precision"] = self.activation_precision.name.lower()
        result["projection_layout"] = self.projection_layout.name.lower()
        return result

    def pack(self) -> bytes:
        data = bytearray(DESCRIPTOR_SIZE)
        struct.pack_into(
            "<IHH", data, 0, DESCRIPTOR_MAGIC, DESCRIPTOR_VERSION, DESCRIPTOR_SIZE
        )
        for name, offset in _U8_FIELDS.items():
            struct.pack_into("<B", data, offset, int(getattr(self, name)))
        for name, offset in _U16_FIELDS.items():
            struct.pack_into("<H", data, offset, getattr(self, name))
        for name, offset in _U32_FIELDS.items():
            struct.pack_into("<I", data, offset, getattr(self, name))
        for name, offset in _U64_FIELDS.items():
            struct.pack_into("<Q", data, offset, getattr(self, name))
        return bytes(data)

    @classmethod
    def unpack(cls, data: bytes) -> MambaDescriptor:
        if len(data) != DESCRIPTOR_SIZE:
            raise ValueError(f"descriptor must be exactly {DESCRIPTOR_SIZE} bytes")
        magic, version, size = struct.unpack_from("<IHH", data, 0)
        if (magic, version, size) != (
            DESCRIPTOR_MAGIC,
            DESCRIPTOR_VERSION,
            DESCRIPTOR_SIZE,
        ):
            raise ValueError("incompatible Mamba descriptor header")
        values: dict[str, int | PrecisionCode | ProjectionLayout] = {}
        for name, offset in _U8_FIELDS.items():
            values[name] = struct.unpack_from("<B", data, offset)[0]
        values["state_precision"] = PrecisionCode(values["state_precision"])
        values["activation_precision"] = PrecisionCode(values["activation_precision"])
        values["projection_layout"] = ProjectionLayout(values["projection_layout"])
        for name, offset in _U16_FIELDS.items():
            values[name] = struct.unpack_from("<H", data, offset)[0]
        for name, offset in _U32_FIELDS.items():
            values[name] = struct.unpack_from("<I", data, offset)[0]
        for name, offset in _U64_FIELDS.items():
            values[name] = struct.unpack_from("<Q", data, offset)[0]
        return cls(**values)


@dataclass(frozen=True)
class MambaCommand:
    subop: MambaSubop
    descriptor: MambaDescriptor | None = None
    context_gp: int = 1
    descriptor_offset_gp: int = 2
    descriptor_hbm_reg: int = 0
    queue_id: int = 0

    def __post_init__(self) -> None:
        if self.subop != MambaSubop.WAIT and self.descriptor is None:
            raise ValueError(f"{self.subop.name} requires a descriptor")
        if self.subop == MambaSubop.STEP and self.descriptor is not None:
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


def _check_u4(name: str, value: int) -> None:
    if not isinstance(value, int) or not 0 <= value < 16:
        raise ValueError(f"{name} must fit the 4-bit instruction field")


def encode_instruction(
    context_gp: int,
    descriptor_offset_gp: int,
    descriptor_hbm_reg: int,
    queue_id: int,
    subop: MambaSubop | int,
) -> int:
    """Encode ``X_MAMBA context, desc_off, desc_hbm, queue, subop``."""
    for name, value in (
        ("context_gp", context_gp),
        ("descriptor_offset_gp", descriptor_offset_gp),
        ("descriptor_hbm_reg", descriptor_hbm_reg),
        ("queue_id", queue_id),
        ("subop", int(subop)),
    ):
        _check_u4(name, value)
    if descriptor_hbm_reg >= 8:
        raise ValueError("descriptor_hbm_reg must select address register a0..a7")
    subop = MambaSubop(subop)
    return (
        MAMBA_OPCODE
        | (context_gp << 6)
        | (descriptor_offset_gp << 10)
        | (descriptor_hbm_reg << 14)
        | (queue_id << 18)
        | (int(subop) << 22)
    )


def decode_instruction(word: int) -> dict[str, int | MambaSubop]:
    if not isinstance(word, int) or not 0 <= word < 2**32:
        raise ValueError("instruction must be an unsigned 32-bit word")
    if word >> 26:
        raise ValueError("reserved instruction bits 26..31 must be zero")
    if word & 0x3F != MAMBA_OPCODE:
        raise ValueError("instruction is not X_MAMBA")
    result: dict[str, int | MambaSubop] = {
        "context_gp": (word >> 6) & 0xF,
        "descriptor_offset_gp": (word >> 10) & 0xF,
        "descriptor_hbm_reg": (word >> 14) & 0xF,
        "queue_id": (word >> 18) & 0xF,
        "subop": MambaSubop((word >> 22) & 0xF),
    }
    if result["descriptor_hbm_reg"] >= 8:
        raise ValueError("descriptor_hbm_reg does not select a0..a7")
    return result


class StateLifecycle:
    """Small semantic checker shared by scheduler tests and future simulator code."""

    def __init__(self) -> None:
        self.locations: dict[tuple[int, int], StateLocation] = {}

    def seed_hbm(self, key: tuple[int, int]) -> None:
        self.locations[key] = StateLocation.HBM_CLEAN

    def apply(self, key: tuple[int, int], subop: MambaSubop) -> StateLocation:
        location = self.locations.get(key, StateLocation.HBM_CLEAN)
        if subop == MambaSubop.STATE_PREFETCH:
            if location != StateLocation.HBM_CLEAN:
                raise ValueError(f"cannot prefetch {key} from {location}")
            location = StateLocation.RESIDENT_CLEAN
        elif subop == MambaSubop.STATE_RESET:
            if location not in {StateLocation.HBM_CLEAN, StateLocation.RESIDENT_CLEAN}:
                raise ValueError(f"cannot reset dirty state {key}")
            location = StateLocation.RESIDENT_DIRTY
        elif subop in {MambaSubop.PREFILL, MambaSubop.STEP}:
            if location not in {
                StateLocation.RESIDENT_CLEAN,
                StateLocation.RESIDENT_DIRTY,
            }:
                raise ValueError(
                    f"cannot execute {subop.name} with nonresident state {key}"
                )
            location = StateLocation.RESIDENT_DIRTY
        elif subop == MambaSubop.STATE_COMMIT:
            if location != StateLocation.RESIDENT_DIRTY:
                raise ValueError(f"cannot commit non-dirty state {key}")
            location = StateLocation.RESIDENT_CLEAN
        elif subop == MambaSubop.STATE_EVICT:
            if location != StateLocation.RESIDENT_CLEAN:
                raise ValueError(f"cannot evict non-clean state {key}")
            location = StateLocation.HBM_CLEAN
        self.locations[key] = location
        return location
