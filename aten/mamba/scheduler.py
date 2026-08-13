"""Capacity-aware compiler trace scheduler for Nemotron 3 Mamba layers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from enum import StrEnum

from .contract import (
    FLAG_BC_GROUP_SHARED,
    FLAG_CONTINUE_STATE,
    FLAG_LAST_CHUNK,
    MambaCommand,
    MambaDescriptor,
    MambaSubop,
    PrecisionCode,
    ProjectionLayout,
    StateLifecycle,
)


NEMOTRON3_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
NEMOTRON3_MAMBA_LAYERS = tuple(
    index for index, symbol in enumerate(NEMOTRON3_PATTERN) if symbol == "M"
)


class SchedulePhase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


class CachePolicy(StrEnum):
    NONE = "none"
    LRU = "lru"
    PINNED = "pinned"


class Resource(StrEnum):
    MATRIX = "matrix"
    VECTOR = "vector"
    LAYOUT = "l_compute"
    MAMBA = "mamba_state_engine"
    CONTROL = "control"


@dataclass(frozen=True)
class MambaScheduleConfig:
    phase: SchedulePhase
    batch_size: int = 1
    sequence_length: int = 1
    decode_tokens: int = 1
    chunk_size: int = 128
    state_cache_entries: int = 0
    cache_policy: CachePolicy = CachePolicy.NONE
    state_precision: PrecisionCode = PrecisionCode.FP32
    activation_precision: PrecisionCode = PrecisionCode.BF16
    projection_layout: ProjectionLayout = ProjectionLayout.GROUP_MAJOR_SKEWED
    flush_at_end: bool = True

    def __post_init__(self) -> None:
        for name in ("batch_size", "sequence_length", "decode_tokens", "chunk_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.state_cache_entries < 0:
            raise ValueError("state_cache_entries must be non-negative")
        if self.state_cache_entries == 0 and self.cache_policy != CachePolicy.NONE:
            raise ValueError("zero cache entries require policy=none")
        if self.state_cache_entries > 0 and self.cache_policy == CachePolicy.NONE:
            raise ValueError("non-zero cache entries require lru or pinned policy")
        if self.phase == SchedulePhase.DECODE and self.sequence_length != 1:
            raise ValueError("decode uses one input token per model pass")


@dataclass(frozen=True)
class TraceEvent:
    index: int
    resource: Resource
    operation: str
    request_id: int | None = None
    layer_id: int | None = None
    token_offset: int | None = None
    valid_tokens: int | None = None
    cache_hit: bool | None = None
    descriptor: MambaDescriptor | None = None
    instruction_word: int | None = None
    note: str = ""

    def to_dict(self) -> dict:
        result = asdict(self)
        result["resource"] = self.resource.value
        if self.descriptor is not None:
            result["descriptor"] = self.descriptor.to_dict()
        return result


@dataclass(frozen=True)
class ScheduleTrace:
    config: MambaScheduleConfig
    events: tuple[TraceEvent, ...]
    cache_hits: int
    cache_misses: int
    cache_evictions: int

    def count(self, operation: str) -> int:
        return sum(event.operation == operation for event in self.events)

    def to_dict(self) -> dict:
        return {
            "config": {
                **asdict(self.config),
                "phase": self.config.phase.value,
                "cache_policy": self.config.cache_policy.value,
                "state_precision": self.config.state_precision.name.lower(),
                "activation_precision": self.config.activation_precision.name.lower(),
                "projection_layout": self.config.projection_layout.name.lower(),
            },
            "nemotron3_mamba_layers": list(NEMOTRON3_MAMBA_LAYERS),
            "summary": {
                "event_count": len(self.events),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_evictions": self.cache_evictions,
                "operation_counts": {
                    operation: self.count(operation)
                    for operation in sorted({event.operation for event in self.events})
                },
            },
            "events": [event.to_dict() for event in self.events],
        }


class Nemotron3MambaScheduler:
    """Generate service and X_MAMBA ordering before lowering to physical assembly."""

    def __init__(self, config: MambaScheduleConfig) -> None:
        self.config = config
        self.events: list[TraceEvent] = []
        self.lifecycle = StateLifecycle()
        self.cache: OrderedDict[tuple[int, int], int] = OrderedDict()
        self.dirty: set[tuple[int, int]] = set()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.pinned = self._pinned_keys()

    def build(self) -> ScheduleTrace:
        if self.config.phase == SchedulePhase.PREFILL:
            self._build_prefill()
        else:
            self._build_decode()
        if self.config.flush_at_end:
            self._flush_cache()
        self._emit(
            Resource.CONTROL,
            "WAIT",
            note="join Matrix, Vector, layout, and Mamba queues",
        )
        return ScheduleTrace(
            self.config,
            tuple(self.events),
            self.cache_hits,
            self.cache_misses,
            self.cache_evictions,
        )

    def _keys(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (request_id, layer_id)
            for layer_id in NEMOTRON3_MAMBA_LAYERS
            for request_id in range(self.config.batch_size)
        )

    def _pinned_keys(self) -> set[tuple[int, int]]:
        if self.config.cache_policy != CachePolicy.PINNED:
            return set()
        return set(self._keys()[: self.config.state_cache_entries])

    def _base_descriptor(
        self,
        key: tuple[int, int],
        token_offset: int,
        valid_tokens: int,
        *,
        sequence_length: int,
        last_chunk: bool,
    ) -> MambaDescriptor:
        request_id, layer_id = key
        key_index = (
            NEMOTRON3_MAMBA_LAYERS.index(layer_id) * self.config.batch_size + request_id
        )
        projection_stride = 10304 * 2 * self.config.chunk_size
        scan_stride = 4096 * 4 * self.config.chunk_size
        flags = FLAG_BC_GROUP_SHARED
        if self.config.phase == SchedulePhase.DECODE or token_offset > 0:
            flags |= FLAG_CONTINUE_STATE
        if last_chunk:
            flags |= FLAG_LAST_CHUNK
        return MambaDescriptor(
            batch_size=1,
            sequence_length=sequence_length,
            chunk_size=self.config.chunk_size,
            state_precision=self.config.state_precision,
            activation_precision=self.config.activation_precision,
            projection_layout=self.config.projection_layout,
            flags=flags,
            request_id=request_id,
            layer_id=layer_id,
            cache_slot=self.cache.get(key, 0xFFFF),
            token_offset=token_offset,
            valid_tokens=valid_tokens,
            completion_id=len(self.events),
            projection_addr=0x1000_0000 + (len(self.events) & 1) * projection_stride,
            scan_output_addr=0x2000_0000 + (len(self.events) & 1) * scan_stride,
            state_addr=0x4000_0000 + key_index * 2 * 1024 * 1024,
            conv_state_addr=0x8000_0000 + key_index * 96 * 1024,
            conv_weight_addr=0x1_0000_0000 + layer_id * 0x20_0000,
            conv_bias_addr=0x1_1000_0000 + layer_id * 0x1_0000,
            a_log_addr=0x1_2000_0000 + layer_id * 0x1_0000,
            dt_bias_addr=0x1_3000_0000 + layer_id * 0x1_0000,
            d_skip_addr=0x1_4000_0000 + layer_id * 0x1_0000,
            norm_weight_addr=0x1_5000_0000 + layer_id * 0x1_0000,
            state_scale_addr=0x1_6000_0000 + key_index * 0x1_0000,
            projection_scale_addr=0x1_7000_0000,
            completion_addr=0x1_8000_0000 + len(self.events) * 64,
        )

    def _emit(
        self,
        resource: Resource,
        operation: str,
        *,
        key: tuple[int, int] | None = None,
        descriptor: MambaDescriptor | None = None,
        cache_hit: bool | None = None,
        note: str = "",
    ) -> None:
        command = None
        if operation in MambaSubop.__members__:
            command = MambaCommand(MambaSubop[operation], descriptor)
        self.events.append(
            TraceEvent(
                index=len(self.events),
                resource=resource,
                operation=operation,
                request_id=key[0] if key else None,
                layer_id=key[1] if key else None,
                token_offset=descriptor.token_offset if descriptor else None,
                valid_tokens=descriptor.valid_tokens if descriptor else None,
                cache_hit=cache_hit,
                descriptor=descriptor,
                instruction_word=command.instruction_word if command else None,
                note=note,
            )
        )

    def _command(
        self, key: tuple[int, int], subop: MambaSubop, descriptor: MambaDescriptor
    ) -> None:
        self.lifecycle.apply(key, subop)
        self._emit(Resource.MAMBA, subop.name, key=key, descriptor=descriptor)
        if subop in {MambaSubop.PREFILL, MambaSubop.STEP, MambaSubop.STATE_RESET}:
            self.dirty.add(key)
        elif subop == MambaSubop.STATE_COMMIT:
            self.dirty.discard(key)

    def _allocate_slot(
        self, key: tuple[int, int], descriptor: MambaDescriptor
    ) -> MambaDescriptor:
        capacity = self.config.state_cache_entries
        if capacity == 0 or (
            self.config.cache_policy == CachePolicy.PINNED and key not in self.pinned
        ):
            return replace(descriptor, cache_slot=0)
        if len(self.cache) >= capacity:
            victim = next(
                candidate for candidate in self.cache if candidate not in self.pinned
            )
            victim_slot = self.cache[victim]
            victim_desc = self._base_descriptor(
                victim,
                0,
                1,
                sequence_length=max(1, descriptor.sequence_length),
                last_chunk=True,
            )
            victim_desc = replace(victim_desc, cache_slot=victim_slot)
            if victim in self.dirty:
                self._command(victim, MambaSubop.STATE_COMMIT, victim_desc)
            self._command(victim, MambaSubop.STATE_EVICT, victim_desc)
            del self.cache[victim]
            self.cache_evictions += 1
            slot = victim_slot
        else:
            used = set(self.cache.values())
            slot = next(index for index in range(capacity) if index not in used)
        self.cache[key] = slot
        return replace(descriptor, cache_slot=slot)

    def _ensure_decode_state(
        self, key: tuple[int, int], descriptor: MambaDescriptor
    ) -> tuple[MambaDescriptor, bool]:
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            return replace(descriptor, cache_slot=self.cache[key]), True
        self.cache_misses += 1
        descriptor = self._allocate_slot(key, descriptor)
        self.lifecycle.seed_hbm(key)
        self._command(key, MambaSubop.STATE_PREFETCH, descriptor)
        return descriptor, False

    def _finish_streamed(
        self, key: tuple[int, int], descriptor: MambaDescriptor
    ) -> None:
        streamed = self.config.state_cache_entries == 0 or (
            self.config.cache_policy == CachePolicy.PINNED and key not in self.pinned
        )
        if streamed:
            self._command(key, MambaSubop.STATE_COMMIT, descriptor)
            self._command(key, MambaSubop.STATE_EVICT, descriptor)

    def _compute_chunk(
        self, key: tuple[int, int], descriptor: MambaDescriptor, subop: MambaSubop
    ) -> None:
        self._emit(
            Resource.MATRIX,
            "IN_PROJECTION",
            key=key,
            descriptor=descriptor,
            note="2688 -> 10304 using existing Matrix service",
        )
        self._emit(
            Resource.LAYOUT,
            "PROJECTION_SCATTER",
            key=key,
            descriptor=descriptor,
            note="linear split -> group-major bank packets; no extra user-visible opcode",
        )
        self._command(key, subop, descriptor)
        self._emit(
            Resource.VECTOR,
            "GATED_GROUP_RMSNORM",
            key=key,
            descriptor=descriptor,
            note="Mamba sequencer borrows existing Vector service after fused state update/C reduction",
        )
        self._emit(
            Resource.MATRIX,
            "OUT_PROJECTION",
            key=key,
            descriptor=descriptor,
            note="4096 -> 2688 using existing Matrix service",
        )

    def _build_decode(self) -> None:
        sequence_length = self.config.decode_tokens
        for token in range(self.config.decode_tokens):
            for key in self._keys():
                descriptor = self._base_descriptor(
                    key, token, 1, sequence_length=sequence_length, last_chunk=True
                )
                descriptor, hit = self._ensure_decode_state(key, descriptor)
                self._emit(
                    Resource.CONTROL,
                    "STATE_CACHE_HIT" if hit else "STATE_CACHE_MISS",
                    key=key,
                    descriptor=descriptor,
                    cache_hit=hit,
                )
                self._compute_chunk(key, descriptor, MambaSubop.STEP)
                self._finish_streamed(key, descriptor)

    def _build_prefill(self) -> None:
        for key in self._keys():
            descriptor = self._base_descriptor(
                key,
                0,
                min(self.config.chunk_size, self.config.sequence_length),
                sequence_length=self.config.sequence_length,
                last_chunk=self.config.sequence_length <= self.config.chunk_size,
            )
            descriptor = self._allocate_slot(key, descriptor)
            self.lifecycle.seed_hbm(key)
            self._command(key, MambaSubop.STATE_RESET, descriptor)
            for start in range(0, self.config.sequence_length, self.config.chunk_size):
                valid = min(self.config.chunk_size, self.config.sequence_length - start)
                chunk = replace(
                    descriptor,
                    token_offset=start,
                    valid_tokens=valid,
                    flags=descriptor.flags
                    | (FLAG_CONTINUE_STATE if start > 0 else 0)
                    | (
                        FLAG_LAST_CHUNK
                        if start + valid == self.config.sequence_length
                        else 0
                    ),
                )
                self._compute_chunk(key, chunk, MambaSubop.PREFILL)
            self._finish_streamed(key, descriptor)

    def _flush_cache(self) -> None:
        for key, slot in list(self.cache.items()):
            if key not in self.dirty:
                continue
            descriptor = self._base_descriptor(
                key, 0, 1, sequence_length=1, last_chunk=True
            )
            descriptor = replace(descriptor, cache_slot=slot)
            self._command(key, MambaSubop.STATE_COMMIT, descriptor)
