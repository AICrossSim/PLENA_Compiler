"""Capacity-aware compiler trace scheduler for Nemotron 3 Mamba layers."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from enum import StrEnum

from .contract import (
    FLAG_LAST_CHUNK,
    STREAMING_SRAM_OFFSET,
    Mamba2Payload,
    MambaCommand,
    MambaDescriptor,
    MambaSubop,
    PrecisionCode,
    StateIdentity,
    StateLifecycle,
)
from aten.state.projection import (
    ProjectionLayout,
    ProjectionScatterConfig,
    ProjectionScatterPlan,
    build_projection_scatter_plan,
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
    STATE = "state_engine"
    MAMBA = "state_engine"
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
    conv_state_precision: PrecisionCode | None = None
    activation_precision: PrecisionCode = PrecisionCode.BF16
    parameter_precision: PrecisionCode = PrecisionCode.BF16
    vector_tile_size: int = 64
    physical_row_tile_size: int = 4
    vector_sram_elements: int = 4 * 1024 * 1024
    async_pipeline: bool = False
    flush_at_end: bool = True
    projection_layout: ProjectionLayout = ProjectionLayout.GROUP_MAJOR_SKEWED
    projection_buffer_banks: int = 16
    projection_buffer_ports_per_bank: int = 1
    projection_fifo_values: int = 64
    matrix_result_burst_values: int = 64
    projection_spill_write_values_per_cycle: int = 16
    projection_direct_bypass: bool = True
    state_head_lanes: int = 8
    state_head_dim_lanes: int = 4
    state_dim_lanes: int = 8
    matrix_input_features: int = 2688
    kda_q_bank_rotation: int = 0
    kda_k_bank_rotation: int = 0
    kda_v_bank_rotation: int = 0
    kda_decay_bank_rotation: int = 0
    kda_beta_bank_rotation: int = 0
    kda_beta_group_stride: int = 0
    resident_state_keys: tuple[tuple[int, int], ...] = ()
    residency_capacity_bytes: int = 0
    residency_source: str | None = None
    residency_target: str | None = None
    mamba_layer_ids: tuple[int, ...] = NEMOTRON3_MAMBA_LAYERS
    #: Pack Mamba state, parameters, and Matrix weights into one byte-addressed
    #: HBM arena. ``None`` preserves the historical sparse addresses exactly.
    mamba_hbm_arena_base: int | None = None

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
        if self.resident_state_keys and self.cache_policy != CachePolicy.PINNED:
            raise ValueError("explicit resident_state_keys require policy=pinned")
        if len(self.resident_state_keys) > self.state_cache_entries:
            raise ValueError("resident_state_keys exceed state_cache_entries")
        if len(set(self.resident_state_keys)) != len(self.resident_state_keys):
            raise ValueError("resident_state_keys must be unique")
        if self.residency_capacity_bytes < 0:
            raise ValueError("residency_capacity_bytes must be non-negative")
        if not self.mamba_layer_ids:
            raise ValueError("mamba_layer_ids must not be empty")
        if any(layer < 0 for layer in self.mamba_layer_ids):
            raise ValueError("mamba_layer_ids must be non-negative")
        if len(set(self.mamba_layer_ids)) != len(self.mamba_layer_ids):
            raise ValueError("mamba_layer_ids must be unique")
        if self.mamba_hbm_arena_base is not None and (
            self.mamba_hbm_arena_base < 0 or self.mamba_hbm_arena_base % 64
        ):
            raise ValueError(
                "mamba_hbm_arena_base must be non-negative and 64-byte aligned"
            )
        if self.phase == SchedulePhase.DECODE and self.sequence_length != 1:
            raise ValueError("decode uses one input token per model pass")
        if self.vector_tile_size <= 0 or self.vector_sram_elements <= 0:
            raise ValueError("Vector SRAM geometry must be positive")
        if self.physical_row_tile_size <= 0:
            raise ValueError("physical_row_tile_size must be positive")
        if self.phase == SchedulePhase.PREFILL and self.chunk_size % self.physical_row_tile_size:
            raise ValueError("prefill chunk_size must be a multiple of physical_row_tile_size")
        if self.async_pipeline and self.batch_size < 2:
            raise ValueError(
                "async_pipeline requires at least two independent requests"
            )
        if self.async_pipeline and self.state_cache_entries:
            raise ValueError("async_pipeline currently supports streaming state only")
        for name in (
            "projection_buffer_banks",
            "projection_buffer_ports_per_bank",
            "projection_fifo_values",
            "matrix_result_burst_values",
            "projection_spill_write_values_per_cycle",
            "state_head_lanes",
            "state_head_dim_lanes",
            "state_dim_lanes",
            "matrix_input_features",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "kda_q_bank_rotation",
            "kda_k_bank_rotation",
            "kda_v_bank_rotation",
            "kda_decay_bank_rotation",
            "kda_beta_bank_rotation",
            "kda_beta_group_stride",
        ):
            if not 0 <= getattr(self, name) < self.projection_buffer_banks:
                raise ValueError(f"{name} must be in [0, projection_buffer_banks)")

    @property
    def projection_scatter_config(self) -> ProjectionScatterConfig:
        return ProjectionScatterConfig(
            layout=self.projection_layout,
            banks=self.projection_buffer_banks,
            ports_per_bank=self.projection_buffer_ports_per_bank,
            fifo_capacity_values=self.projection_fifo_values,
            producer_burst_values=self.matrix_result_burst_values,
            spill_write_values_per_cycle=self.projection_spill_write_values_per_cycle,
            direct_bypass=self.projection_direct_bypass,
            head_lanes=self.state_head_lanes,
            head_dim_lanes=self.state_head_dim_lanes,
            state_dim_lanes=self.state_dim_lanes,
            matrix_input_features=self.matrix_input_features,
            kda_q_bank_rotation=self.kda_q_bank_rotation,
            kda_k_bank_rotation=self.kda_k_bank_rotation,
            kda_v_bank_rotation=self.kda_v_bank_rotation,
            kda_decay_bank_rotation=self.kda_decay_bank_rotation,
            kda_beta_bank_rotation=self.kda_beta_bank_rotation,
            kda_beta_group_stride=self.kda_beta_group_stride,
        )


@dataclass(frozen=True)
class MambaHbmLayout:
    """Sparse-compatible or compact byte addresses for one Mamba schedule."""

    bases: dict[str, int]
    strides: dict[str, int]
    arena_base: int | None
    arena_end: int

    @classmethod
    def build(cls, config: MambaScheduleConfig) -> MambaHbmLayout:
        sparse_bases = {
            "state": 0x4000_0000,
            "conv_state": 0x8000_0000,
            "input_projection": 0x9000_0000,
            "output_projection": 0xC000_0000,
            "norm_weight": 0xD200_0000,
            "conv_weight": 0x1_0000_0000,
            "conv_bias": 0x1_1000_0000,
            "a_log": 0x1_2000_0000,
            "dt_bias": 0x1_3000_0000,
            "d_skip": 0x1_4000_0000,
            "state_scale": 0x1_6000_0000,
            "parameter_scale": 0x1_7000_0000,
            "completion": 0x1_8000_0000,
        }
        sparse_strides = {
            "state": 2 * 1024 * 1024,
            "conv_state": 96 * 1024,
            "input_projection": 0x0200_0000,
            "output_projection": 0x00C0_0000,
            "norm_weight": 0x1_0000,
            "conv_weight": 0x20_0000,
            "conv_bias": 0x1_0000,
            "a_log": 0x1_0000,
            "dt_bias": 0x1_0000,
            "d_skip": 0x1_0000,
            "state_scale": 0x1_0000,
            "parameter_scale": 0x1_0000,
            "completion": 64,
        }
        if config.mamba_hbm_arena_base is None:
            return cls(sparse_bases, sparse_strides, None, 0x1_9000_0000)

        def align64(value: int) -> int:
            return ((value + 63) // 64) * 64

        entries = len(config.mamba_layer_ids) * config.batch_size
        layers = len(config.mamba_layer_ids)
        state_bytes = 64 * 64 * 128 * config.state_precision.element_bytes
        conv_precision = config.conv_state_precision or config.state_precision
        conv_state_bytes = 96 * 1024
        # Matrix HBM addresses are byte offsets in the transactional emulator.
        # Compact numerical tests use plain BF16 weights.
        strides = {
            "state": align64(state_bytes),
            "conv_state": align64(
                max(conv_state_bytes, 6144 * 3 * conv_precision.element_bytes)
            ),
            "state_scale": 0x1_0000,
            "conv_weight": align64(6144 * 4 * 2),
            "conv_bias": align64(6144 * 2),
            "a_log": align64(64 * 4),
            "dt_bias": align64(64 * 4),
            "d_skip": align64(64 * 4),
            "parameter_scale": 0x1_0000,
            "completion": 64,
            "input_projection": align64(config.matrix_input_features * 10304 * 2),
            "output_projection": align64(4096 * config.matrix_input_features * 2),
            "norm_weight": align64(4096 * 2),
        }
        counts = {
            "state": entries,
            "conv_state": entries,
            "state_scale": entries,
            "conv_weight": layers,
            "conv_bias": layers,
            "a_log": layers,
            "dt_bias": layers,
            "d_skip": layers,
            "parameter_scale": 1,
            "completion": entries * 16,
            "input_projection": layers,
            "output_projection": layers,
            "norm_weight": layers,
        }
        bases: dict[str, int] = {}
        cursor = config.mamba_hbm_arena_base
        for region in (
            "state",
            "conv_state",
            "state_scale",
            "conv_weight",
            "conv_bias",
            "a_log",
            "dt_bias",
            "d_skip",
            "parameter_scale",
            "completion",
            "input_projection",
            "output_projection",
            "norm_weight",
        ):
            bases[region] = cursor
            cursor += strides[region] * counts[region]
        return cls(bases, strides, config.mamba_hbm_arena_base, cursor)

    def address(
        self,
        region: str,
        *,
        ordinal: int = 0,
        layer_id: int | None = None,
    ) -> int:
        if ordinal < 0:
            raise ValueError("HBM ordinal must be non-negative")
        index = ordinal
        if self.arena_base is None and region in {
            "conv_weight",
            "conv_bias",
            "a_log",
            "dt_bias",
            "d_skip",
        }:
            if layer_id is None:
                raise ValueError(f"sparse {region} address requires layer_id")
            index = layer_id
        if region == "parameter_scale":
            index = 0
        return self.bases[region] + self.strides[region] * index


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
    queue_id: int | None = None
    aux_vram_addr: int | None = None
    projection_scatter: ProjectionScatterPlan | None = None
    note: str = ""

    def to_dict(self) -> dict:
        result = asdict(self)
        result["resource"] = self.resource.value
        if self.descriptor is not None:
            result["descriptor"] = self.descriptor.to_dict()
        if self.projection_scatter is not None:
            result["projection_scatter"] = self.projection_scatter.to_dict()
        return result


@dataclass(frozen=True)
class ScheduleTrace:
    config: MambaScheduleConfig
    events: tuple[TraceEvent, ...]
    cache_hits: int
    cache_misses: int
    cache_evictions: int
    model_name: str = "nemotron3"
    state_layers: tuple[int, ...] = NEMOTRON3_MAMBA_LAYERS

    def count(self, operation: str) -> int:
        return sum(event.operation == operation for event in self.events)

    def to_dict(self) -> dict:
        result = {
            "config": {
                **asdict(self.config),
                "phase": self.config.phase.value,
                "cache_policy": self.config.cache_policy.value,
                "state_precision": self.config.state_precision.name.lower(),
                "conv_state_precision": (
                    self.config.conv_state_precision or self.config.state_precision
                ).name.lower(),
                "activation_precision": self.config.activation_precision.name.lower(),
                "parameter_precision": self.config.parameter_precision.name.lower(),
                "projection_layout": self.config.projection_layout.value,
            },
            "model": self.model_name,
            "state_layers": list(self.state_layers),
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
        if self.model_name == "nemotron3":
            result["nemotron3_mamba_layers"] = list(self.state_layers)
        return result


class Nemotron3MambaScheduler:
    """Generate Matrix/Vector/X_STATE ordering before physical lowering."""

    def __init__(self, config: MambaScheduleConfig) -> None:
        self.config = config
        self.events: list[TraceEvent] = []
        self.lifecycle = StateLifecycle()
        self.cache: OrderedDict[StateIdentity, int] = OrderedDict()
        self.dirty: set[StateIdentity] = set()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_evictions = 0
        self.pinned = self._pinned_keys()
        self.used_queues: set[int] = set()

    def build(self) -> ScheduleTrace:
        if self.config.phase == SchedulePhase.PREFILL:
            self._build_prefill()
        else:
            self._build_decode()
        if self.config.flush_at_end:
            self._flush_cache()
        for queue_id in sorted(self.used_queues or {0}):
            self._emit(
                Resource.CONTROL,
                "FENCE",
                queue_id=queue_id,
                note="drain the State Engine queue before trace completion",
            )
        return ScheduleTrace(
            self.config,
            tuple(self.events),
            self.cache_hits,
            self.cache_misses,
            self.cache_evictions,
            state_layers=self.config.mamba_layer_ids,
        )

    def _keys(self) -> tuple[StateIdentity, ...]:
        return tuple(
            StateIdentity(0, request_id, layer_id, 0)
            for layer_id in self.config.mamba_layer_ids
            for request_id in range(self.config.batch_size)
        )

    def _pinned_keys(self) -> set[StateIdentity]:
        if self.config.cache_policy != CachePolicy.PINNED:
            return set()
        if self.config.resident_state_keys:
            return {
                StateIdentity(0, request_id, layer_id, 0)
                for request_id, layer_id in self.config.resident_state_keys
            }
        return set(self._keys()[: self.config.state_cache_entries])

    def _buffer_index(self, key_index: int, token_offset: int) -> int:
        """Alternate the projection/scan double buffer once per issued chunk.

        Decode issues one buffer turn per token, so the token offset is already
        the turn counter. Prefill issues one turn per chunk and advances
        token_offset by chunk_size, so taking parity on the raw offset would
        never flip for an even chunk size and would pin every chunk to buffer 0.
        """
        issues = (
            token_offset // self.config.chunk_size
            if self.config.phase == SchedulePhase.PREFILL
            else token_offset
        )
        return (key_index + issues) & 1

    def _base_descriptor(
        self,
        key: StateIdentity,
        token_offset: int,
        valid_tokens: int,
        *,
        sequence_length: int,
        last_chunk: bool,
    ) -> MambaDescriptor:
        request_id = key.request_id
        layer_id = key.layer_id
        key_index = (
            self.config.mamba_layer_ids.index(layer_id) * self.config.batch_size
            + request_id
        )
        packed_hbm = MambaHbmLayout.build(self.config)
        input_token_stride = _align_up(10304, self.config.vector_tile_size)
        output_token_stride = _align_up(4096, self.config.vector_tile_size)
        descriptor_chunk_size = (
            self.config.chunk_size
            if self.config.phase == SchedulePhase.PREFILL
            else self.config.physical_row_tile_size
        )
        projection_buffer = input_token_stride * descriptor_chunk_size
        scan_buffer = output_token_stride * descriptor_chunk_size
        output_base = 2 * projection_buffer
        required_vram = output_base + 2 * scan_buffer
        if required_vram > self.config.vector_sram_elements:
            raise ValueError(
                f"Mamba double buffers require {required_vram} Vector SRAM elements, "
                f"only {self.config.vector_sram_elements} configured"
            )
        buffer_index = self._buffer_index(key_index, token_offset)
        flags = 0
        if last_chunk:
            flags |= FLAG_LAST_CHUNK
        return MambaDescriptor(
            payload=Mamba2Payload(
                conv_weight_addr=packed_hbm.address(
                    "conv_weight", layer_id=layer_id, ordinal=key_index // self.config.batch_size
                ),
                conv_bias_addr=packed_hbm.address(
                    "conv_bias", layer_id=layer_id, ordinal=key_index // self.config.batch_size
                ),
                a_log_addr=packed_hbm.address(
                    "a_log", layer_id=layer_id, ordinal=key_index // self.config.batch_size
                ),
                dt_bias_addr=packed_hbm.address(
                    "dt_bias", layer_id=layer_id, ordinal=key_index // self.config.batch_size
                ),
                d_skip_addr=packed_hbm.address(
                    "d_skip", layer_id=layer_id, ordinal=key_index // self.config.batch_size
                ),
                parameter_scale_addr=packed_hbm.address("parameter_scale"),
            ),
            batch_size=1,
            num_heads=64,
            sequence_length=sequence_length,
            chunk_size=descriptor_chunk_size,
            state_precision=self.config.state_precision,
            conv_state_precision=self.config.conv_state_precision,
            activation_precision=self.config.activation_precision,
            parameter_precision=self.config.parameter_precision,
            flags=flags,
            context_id=key.context_id,
            request_id=request_id,
            layer_id=layer_id,
            state_id=key.state_id,
            state_sram_offset=STREAMING_SRAM_OFFSET,
            token_offset=token_offset,
            valid_tokens=valid_tokens,
            input_vram_addr=buffer_index * projection_buffer,
            output_vram_addr=output_base + buffer_index * scan_buffer,
            input_token_stride=input_token_stride,
            output_token_stride=output_token_stride,
            state_hbm_addr=packed_hbm.address("state", ordinal=key_index),
            conv_state_hbm_addr=packed_hbm.address("conv_state", ordinal=key_index),
            state_scale_addr=packed_hbm.address("state_scale", ordinal=key_index),
            completion_addr=packed_hbm.address(
                "completion", ordinal=len(self.events)
            ),
        )

    def _emit(
        self,
        resource: Resource,
        operation: str,
        *,
        key: StateIdentity | None = None,
        descriptor: MambaDescriptor | None = None,
        cache_hit: bool | None = None,
        queue_id: int = 0,
        aux_vram_addr: int | None = None,
        projection_scatter: ProjectionScatterPlan | None = None,
        note: str = "",
    ) -> None:
        command = None
        if operation in MambaSubop.__members__:
            command = MambaCommand(MambaSubop[operation], descriptor, queue_id=queue_id)
        self.events.append(
            TraceEvent(
                index=len(self.events),
                resource=resource,
                operation=operation,
                request_id=key.request_id if key else None,
                layer_id=key.layer_id if key else None,
                token_offset=descriptor.token_offset if descriptor else None,
                valid_tokens=descriptor.valid_tokens if descriptor else None,
                cache_hit=cache_hit,
                descriptor=descriptor,
                instruction_word=command.instruction_word if command else None,
                queue_id=queue_id if command else None,
                aux_vram_addr=aux_vram_addr,
                projection_scatter=projection_scatter,
                note=note,
            )
        )

    def _command(
        self,
        key: StateIdentity,
        subop: MambaSubop,
        descriptor: MambaDescriptor,
        *,
        queue_id: int = 0,
    ) -> None:
        if descriptor.identity != key:
            raise ValueError("scheduler key and descriptor identity disagree")
        self.lifecycle.apply(descriptor, subop)
        self.used_queues.add(queue_id)
        self._emit(
            Resource.STATE,
            subop.name,
            key=key,
            descriptor=descriptor,
            queue_id=queue_id,
        )
        if not descriptor.streaming and subop in {
            MambaSubop.PREFILL,
            MambaSubop.STEP,
            MambaSubop.RESET,
        }:
            self.dirty.add(key)
        elif subop == MambaSubop.COMMIT:
            self.dirty.discard(key)

    def _allocate_slot(
        self, key: StateIdentity, descriptor: MambaDescriptor
    ) -> MambaDescriptor:
        capacity = self.config.state_cache_entries
        if capacity == 0 or (
            self.config.cache_policy == CachePolicy.PINNED and key not in self.pinned
        ):
            return descriptor
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
            victim_offset = victim_slot * victim_desc.resident_bytes
            victim_desc = replace(victim_desc, state_sram_offset=victim_offset)
            if victim in self.dirty:
                self._command(victim, MambaSubop.COMMIT, victim_desc)
            self._command(victim, MambaSubop.EVICT, victim_desc)
            del self.cache[victim]
            self.cache_evictions += 1
            slot = victim_slot
        else:
            used = set(self.cache.values())
            slot = next(index for index in range(capacity) if index not in used)
        self.cache[key] = slot
        state_sram_offset = slot * descriptor.resident_bytes
        return replace(descriptor, state_sram_offset=state_sram_offset)

    def _ensure_decode_state(
        self, key: StateIdentity, descriptor: MambaDescriptor
    ) -> tuple[MambaDescriptor, bool]:
        if key in self.cache:
            self.cache_hits += 1
            self.cache.move_to_end(key)
            slot = self.cache[key]
            offset = slot * descriptor.resident_bytes
            return replace(descriptor, state_sram_offset=offset), True
        self.cache_misses += 1
        descriptor = self._allocate_slot(key, descriptor)
        self.lifecycle.seed_hbm(key)
        if not descriptor.streaming:
            self._command(key, MambaSubop.PRELOAD, descriptor)
        return descriptor, False

    def _finish_streamed(self, key: StateIdentity, descriptor: MambaDescriptor) -> None:
        if descriptor.streaming and key in self.dirty:
            raise ValueError("streaming X_STATE commands must commit state internally")

    def _compute_chunk(
        self, key: StateIdentity, descriptor: MambaDescriptor, subop: MambaSubop
    ) -> None:
        self._issue_chunk(key, descriptor, subop, queue_id=0)
        self._consume_chunk(key, descriptor, queue_id=0)

    def _issue_chunk(
        self,
        key: StateIdentity,
        descriptor: MambaDescriptor,
        subop: MambaSubop,
        *,
        queue_id: int,
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
            projection_scatter=build_projection_scatter_plan(
                descriptor,
                phase=self.config.phase.value,
                config=self.config.projection_scatter_config,
            ),
            note="physical Matrix-result FIFO, bank scatter, and Vector-SRAM spill plan",
        )
        self._command(key, subop, descriptor, queue_id=queue_id)

    def _consume_chunk(
        self,
        key: StateIdentity,
        descriptor: MambaDescriptor,
        *,
        queue_id: int,
    ) -> None:
        self._emit(
            Resource.CONTROL,
            "FENCE",
            key=key,
            queue_id=queue_id,
            note="wait for X_STATE output before Vector consumes it",
        )
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
        if self.config.async_pipeline:
            self._build_decode_async()
            return
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

    def _build_decode_async(self) -> None:
        sequence_length = self.config.decode_tokens
        all_keys = self._keys()
        layer_ids = tuple(dict.fromkeys(key.layer_id for key in all_keys))
        keys_by_layer = {
            layer_id: tuple(key for key in all_keys if key.layer_id == layer_id)
            for layer_id in layer_ids
        }
        for token in range(self.config.decode_tokens):
            for layer_id in layer_ids:
                layer_keys = keys_by_layer[layer_id]
                for start in range(0, len(layer_keys), 2):
                    prepared: list[tuple[StateIdentity, MambaDescriptor, int]] = []
                    for queue_id, key in enumerate(layer_keys[start : start + 2]):
                        descriptor = self._base_descriptor(
                            key,
                            token,
                            1,
                            sequence_length=sequence_length,
                            last_chunk=True,
                        )
                        descriptor, hit = self._ensure_decode_state(key, descriptor)
                        self._emit(
                            Resource.CONTROL,
                            "STATE_CACHE_HIT" if hit else "STATE_CACHE_MISS",
                            key=key,
                            descriptor=descriptor,
                            cache_hit=hit,
                        )
                        prepared.append((key, descriptor, queue_id))
                    for key, descriptor, queue_id in prepared:
                        self._issue_chunk(
                            key,
                            descriptor,
                            MambaSubop.STEP,
                            queue_id=queue_id,
                        )
                    for key, descriptor, queue_id in prepared:
                        self._consume_chunk(key, descriptor, queue_id=queue_id)
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
            self._command(key, MambaSubop.RESET, descriptor)
            state_sram_offset = descriptor.state_sram_offset
            for start in range(0, self.config.sequence_length, self.config.chunk_size):
                valid = min(self.config.chunk_size, self.config.sequence_length - start)
                # Rebuild rather than `replace` the chunk descriptor: the input
                # and output Vector SRAM addresses are derived from token_offset
                # inside _base_descriptor, so reusing the first chunk's addresses
                # would pin every chunk to buffer 0 and make the second half of
                # the allocated double buffer dead. Rebuilding also gives each
                # chunk its own completion address.
                chunk = self._base_descriptor(
                    key,
                    start,
                    valid,
                    sequence_length=self.config.sequence_length,
                    last_chunk=start + valid == self.config.sequence_length,
                )
                chunk = replace(chunk, state_sram_offset=state_sram_offset)
                self._compute_chunk(key, chunk, MambaSubop.PREFILL)
            self._finish_streamed(key, descriptor)

    def _flush_cache(self) -> None:
        for key, slot in list(self.cache.items()):
            if key not in self.dirty:
                continue
            descriptor = self._base_descriptor(
                key, 0, 1, sequence_length=1, last_chunk=True
            )
            offset = slot * descriptor.resident_bytes
            descriptor = replace(descriptor, state_sram_offset=offset)
            self._command(key, MambaSubop.COMMIT, descriptor)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
