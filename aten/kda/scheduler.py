"""Capacity-aware X_STATE scheduler for the Kimi K3 KDA mixer layers."""

from __future__ import annotations

from dataclasses import dataclass, replace

from aten.mamba.scheduler import (
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    Resource,
    SchedulePhase,
    ScheduleTrace,
    _align_up,
)
from aten.state import (
    FLAG_LAST_CHUNK,
    STREAMING_SRAM_OFFSET,
    KdaPayload,
    PrecisionCode,
    StateDescriptor,
    StateIdentity,
    StateSubop,
)
from aten.state.projection import ProjectionLayout, build_projection_scatter_plan


# The official config numbers text layers from one. PLENA identities are zero based.
KIMI_K3_KDA_LAYERS = tuple(layer - 1 for layer in range(1, 93) if layer % 4 != 0)


@dataclass(frozen=True)
class KdaScheduleConfig(MambaScheduleConfig):
    chunk_size: int = 16
    state_precision: PrecisionCode = PrecisionCode.FP32
    conv_state_precision: PrecisionCode | None = PrecisionCode.BF16
    projection_layout: ProjectionLayout = ProjectionLayout.GROUP_MAJOR_SKEWED
    projection_fifo_values: int = 64
    projection_direct_bypass: bool = False
    kda_k_bank_rotation: int = 8
    kda_beta_group_stride: int = 1
    matrix_input_features: int = 7168
    kda_layer_ids: tuple[int, ...] = KIMI_K3_KDA_LAYERS
    kda_num_heads: int = 96
    kda_key_dim: int = 128
    kda_value_dim: int = 128
    kda_conv_kernel: int = 4
    projection_weight_hbm_base: int = 0x8_0000_0000
    projection_weight_layer_stride: int = 0x2000_0000
    projection_weight_offsets: tuple[int, ...] = (
        0x0000_0000,
        0x0600_0000,
        0x0C00_0000,
        0x1200_0000,
        0x1800_0000,
        0x1E00_0000,
        0x1E20_0000,
        0x1E50_0000,
        0x1E60_0000,
    )
    #: Pack every HBM region into one arena starting here instead of using the
    #: sparse 24.5 GiB default bases. Required to execute the physical program on
    #: the transactional emulator, whose HBM is a single flat allocation. Leaving
    #: this None keeps the previously emitted descriptors byte-identical.
    hbm_arena_base: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.kda_layer_ids:
            raise ValueError("kda_layer_ids must not be empty")
        if any(layer < 0 for layer in self.kda_layer_ids):
            raise ValueError("kda_layer_ids must be non-negative")
        if len(set(self.kda_layer_ids)) != len(self.kda_layer_ids):
            raise ValueError("kda_layer_ids must be unique")
        for name in (
            "kda_num_heads",
            "kda_key_dim",
            "kda_value_dim",
            "kda_conv_kernel",
            "projection_weight_layer_stride",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.projection_weight_hbm_base < 0:
            raise ValueError("projection_weight_hbm_base must be non-negative")
        if len(self.projection_weight_offsets) != 9:
            raise ValueError("projection_weight_offsets must name the 9 KDA tensors")
        if any(offset < 0 for offset in self.projection_weight_offsets):
            raise ValueError("projection_weight_offsets must be non-negative")


#: ``(sparse base, stride, index kind)`` for every HBM region a KDA descriptor
#: names. The sparse bases are readable but span 24.5 GiB, and the transactional
#: emulator's HBM is one flat allocation preloaded from a flat file -- so that
#: span means a 24.5 GiB allocation and an equally large file, which is why the
#: physical KDA program has never actually been executed. Strides are the region
#: sizes; the compact layout below reuses them unchanged, so HBM traffic, and
#: therefore timing, is identical either way.
_KDA_HBM_REGIONS: tuple[tuple[str, int, int, str], ...] = (
    ("state", 0x2_0000_0000, 0x40_0000, "entry"),
    ("conv_state", 0x3_0000_0000, 0x8_0000, "entry"),
    ("state_scale", 0x4_0000_0000, 0x1_0000, "entry"),
    ("q_conv_weight", 0x5_0000_0000, 0x4_0000, "layer"),
    ("k_conv_weight", 0x5_4000_0000, 0x4_0000, "layer"),
    ("v_conv_weight", 0x5_8000_0000, 0x4_0000, "layer"),
    ("q_conv_bias", 0x5_C000_0000, 0x1_0000, "layer"),
    ("k_conv_bias", 0x5_D000_0000, 0x1_0000, "layer"),
    ("v_conv_bias", 0x5_E000_0000, 0x1_0000, "layer"),
    ("a_log", 0x5_F000_0000, 0x1_0000, "layer"),
    ("dt_bias", 0x6_0000_0000, 0x1_0000, "layer"),
    ("parameter_scale", 0x6_1000_0000, 0x1_0000, "layer"),
    ("completion", 0x6_2000_0000, 64, "event"),
)

#: Completion records reserved per state key when packing. One record is claimed
#: per emitted event, and the event count depends on phase, chunking and cache
#: policy, so the region is placed last in the arena and allowed to grow past the
#: reservation rather than being capped -- capping it would reject schedules the
#: sparse layout has always accepted. ``realized_arena_bytes`` reports the size
#: actually needed once a trace is built.
_COMPLETION_RECORDS_PER_ENTRY = 16


@dataclass(frozen=True)
class KdaHbmLayout:
    """Base address per KDA HBM region, sparse by default or packed on request."""

    bases: dict[str, int]
    strides: dict[str, int]
    counts: dict[str, int]
    arena_base: int | None
    arena_bytes: int

    @classmethod
    def build(
        cls,
        *,
        entries: int,
        layers: int,
        arena_base: int | None,
        state_bytes: int = 0,
        conv_state_bytes: int = 0,
    ) -> KdaHbmLayout:
        if entries <= 0 or layers <= 0:
            raise ValueError("KDA HBM layout needs at least one entry and layer")
        counts = {
            "entry": entries,
            "layer": layers,
            "event": entries * _COMPLETION_RECORDS_PER_ENTRY,
        }
        strides = {name: stride for name, _, stride, _ in _KDA_HBM_REGIONS}
        # The literal strides were sized for BF16 state. FP32 KDA state is 6 MiB
        # against a 4 MiB stride, so consecutive layers overlapped by 2 MiB and
        # every layer's write corrupted the next one's state. Byte counts were
        # unaffected, which is why timing models never noticed; deriving the
        # stride from the real footprint is what makes the program executable.
        for name, size in (("state", state_bytes), ("conv_state", conv_state_bytes)):
            if size < 0:
                raise ValueError(f"{name}_bytes must be non-negative")
            if size:
                strides[name] = max(strides[name], ((size + 63) // 64) * 64)
        region_counts = {name: counts[kind] for name, _, _, kind in _KDA_HBM_REGIONS}
        if arena_base is None:
            bases = {name: base for name, base, _, _ in _KDA_HBM_REGIONS}
            end = max(
                bases[name] + strides[name] * region_counts[name]
                for name in bases
            )
            return cls(bases, strides, region_counts, None, end)
        if arena_base < 0 or arena_base % 64:
            raise ValueError("hbm_arena_base must be non-negative and 64-byte aligned")
        bases = {}
        cursor = arena_base
        for name, _, _, _ in _KDA_HBM_REGIONS:
            bases[name] = cursor
            # `strides[name]`, not the literal from the table: the state stride is
            # widened above for FP32, and advancing by the literal would leave the
            # state region running into whatever region follows it.
            cursor += strides[name] * region_counts[name]
        return cls(bases, strides, region_counts, arena_base, cursor)

    def address(self, region: str, index: int) -> int:
        if index < 0:
            raise ValueError(f"KDA HBM index for region {region!r} must be non-negative")
        # `completion` is last in the arena and intentionally unbounded; every
        # other region is fixed-size, so an out-of-range index there is a bug.
        if region != "completion" and index >= self.counts[region]:
            raise ValueError(
                f"KDA HBM index {index} is outside region {region!r} "
                f"({self.counts[region]} slots)"
            )
        return self.bases[region] + self.strides[region] * index

    def realized_arena_bytes(self, completion_records: int) -> int:
        """Arena size needed once the real completion-record count is known."""
        if completion_records < 0:
            raise ValueError("completion_records must be non-negative")
        end = self.bases["completion"] + self.strides["completion"] * completion_records
        return max(self.arena_bytes, end)


class KimiK3KdaScheduler(Nemotron3MambaScheduler):
    """Generate Matrix/Vector/X_STATE ordering for all 69 KDA mixers."""

    config: KdaScheduleConfig

    def __init__(self, config: KdaScheduleConfig) -> None:
        super().__init__(config)

    def build(self) -> ScheduleTrace:
        trace = super().build()
        return replace(
            trace,
            model_name="kimi_k3",
            state_layers=self.config.kda_layer_ids,
        )

    def _keys(self) -> tuple[StateIdentity, ...]:
        return tuple(
            StateIdentity(0, request_id, layer_id, 0)
            for layer_id in self.config.kda_layer_ids
            for request_id in range(self.config.batch_size)
        )

    def _base_descriptor(
        self,
        key: StateIdentity,
        token_offset: int,
        valid_tokens: int,
        *,
        sequence_length: int,
        last_chunk: bool,
    ) -> StateDescriptor:
        request_id = key.request_id
        layer_id = key.layer_id
        key_index = (
            self.config.kda_layer_ids.index(layer_id) * self.config.batch_size + request_id
        )
        hbm = self._hbm_layout()
        payload = KdaPayload(
            key_dim=self.config.kda_key_dim,
            value_dim=self.config.kda_value_dim,
            conv_kernel=self.config.kda_conv_kernel,
            q_offset=0,
            k_offset=self.config.kda_num_heads * self.config.kda_key_dim,
            v_offset=2 * self.config.kda_num_heads * self.config.kda_key_dim,
            decay_offset=(
                2 * self.config.kda_num_heads * self.config.kda_key_dim
                + self.config.kda_num_heads * self.config.kda_value_dim
            ),
            beta_offset=(
                3 * self.config.kda_num_heads * self.config.kda_key_dim
                + self.config.kda_num_heads * self.config.kda_value_dim
            ),
            q_conv_weight_addr=hbm.address("q_conv_weight", layer_id),
            k_conv_weight_addr=hbm.address("k_conv_weight", layer_id),
            v_conv_weight_addr=hbm.address("v_conv_weight", layer_id),
            q_conv_bias_addr=hbm.address("q_conv_bias", layer_id),
            k_conv_bias_addr=hbm.address("k_conv_bias", layer_id),
            v_conv_bias_addr=hbm.address("v_conv_bias", layer_id),
            a_log_addr=hbm.address("a_log", layer_id),
            dt_bias_addr=hbm.address("dt_bias", layer_id),
            parameter_scale_addr=hbm.address("parameter_scale", layer_id),
            output_scale=1.0 / (self.config.kda_key_dim**0.5),
        )
        input_token_stride = _align_up(
            payload.input_elements(self.config.kda_num_heads), self.config.vector_tile_size
        )
        output_token_stride = _align_up(
            payload.output_elements(self.config.kda_num_heads), self.config.vector_tile_size
        )
        descriptor_chunk_size = (
            self.config.chunk_size
            if self.config.phase == SchedulePhase.PREFILL
            else self.config.physical_row_tile_size
        )
        input_buffer_values = input_token_stride * descriptor_chunk_size
        output_buffer_values = output_token_stride * descriptor_chunk_size
        output_base = 2 * input_buffer_values
        required_vram = output_base + 4 * output_buffer_values
        if required_vram > self.config.vector_sram_elements:
            raise ValueError(
                f"KDA double buffers require {required_vram} Vector SRAM elements, "
                f"only {self.config.vector_sram_elements} configured"
            )
        buffer_index = self._buffer_index(key_index, token_offset)
        flags = FLAG_LAST_CHUNK if last_chunk else 0
        return StateDescriptor(
            payload=payload,
            batch_size=1,
            num_heads=self.config.kda_num_heads,
            sequence_length=sequence_length,
            token_offset=token_offset,
            valid_tokens=valid_tokens,
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
            input_vram_addr=buffer_index * input_buffer_values,
            output_vram_addr=output_base + buffer_index * output_buffer_values,
            input_token_stride=input_token_stride,
            output_token_stride=output_token_stride,
            state_hbm_addr=hbm.address("state", key_index),
            conv_state_hbm_addr=hbm.address("conv_state", key_index),
            state_scale_addr=hbm.address("state_scale", key_index),
            completion_addr=hbm.address("completion", len(self.events)),
        )

    def hbm_layout(self) -> KdaHbmLayout:
        """The HBM layout this schedule's descriptors were built against.

        Callers must not rebuild it: the state stride depends on the configured
        precisions, and a rebuild that omits them silently reverts to the old
        4 MiB stride and under-reports the arena.
        """
        return self._hbm_layout()

    def _hbm_layout(self) -> KdaHbmLayout:
        cached = getattr(self, "_hbm_layout_cache", None)
        if cached is None:
            heads = self.config.kda_num_heads
            key_dim = self.config.kda_key_dim
            value_dim = self.config.kda_value_dim
            kernel = self.config.kda_conv_kernel
            state_bytes = (
                heads * value_dim * key_dim * self.config.state_precision.element_bytes
            )
            conv_precision = (
                self.config.conv_state_precision or self.config.state_precision
            )
            conv_state_bytes = (
                heads * (2 * key_dim + value_dim) * kernel * conv_precision.element_bytes
            )
            cached = KdaHbmLayout.build(
                entries=len(self.config.kda_layer_ids) * self.config.batch_size,
                layers=max(self.config.kda_layer_ids) + 1,
                arena_base=self.config.hbm_arena_base,
                state_bytes=state_bytes,
                conv_state_bytes=conv_state_bytes,
            )
            object.__setattr__(self, "_hbm_layout_cache", cached)
        return cached

    def _compute_chunk(
        self,
        key: StateIdentity,
        descriptor: StateDescriptor,
        subop: StateSubop,
    ) -> None:
        self._issue_chunk(key, descriptor, subop, queue_id=0)
        self._consume_chunk(key, descriptor, queue_id=0)

    def _issue_chunk(
        self,
        key: StateIdentity,
        descriptor: StateDescriptor,
        subop: StateSubop,
        *,
        queue_id: int,
    ) -> None:
        input_buffer_values = descriptor.input_token_stride * descriptor.chunk_size
        output_buffer_values = descriptor.output_token_stride * descriptor.chunk_size
        output_base = 2 * input_buffer_values
        buffer_index = (
            descriptor.output_vram_addr - output_base
        ) // output_buffer_values
        gate_vram_addr = output_base + (2 + buffer_index) * output_buffer_values
        self._emit(
            Resource.MATRIX,
            "KDA_QKV_PROJECTION",
            key=key,
            descriptor=descriptor,
            note="three 7168 -> 12288 projections on the existing Matrix service",
        )
        self._emit(
            Resource.MATRIX,
            "KDA_DECAY_BETA_PROJECTION",
            key=key,
            descriptor=descriptor,
            note="low-rank decay and per-head beta are materialized before X_STATE",
        )
        self._emit(
            Resource.MATRIX,
            "KDA_OUTPUT_GATE_PROJECTION",
            key=key,
            descriptor=descriptor,
            aux_vram_addr=gate_vram_addr,
            note="independent output gate is computed before the recurrent result",
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
            note="physical q/k/v/decay/beta FIFO, bank scatter, and Vector-SRAM spill plan",
        )
        self._command(key, subop, descriptor, queue_id=queue_id)

    def _consume_chunk(
        self,
        key: StateIdentity,
        descriptor: StateDescriptor,
        *,
        queue_id: int,
    ) -> None:
        input_buffer_values = descriptor.input_token_stride * descriptor.chunk_size
        output_buffer_values = descriptor.output_token_stride * descriptor.chunk_size
        output_base = 2 * input_buffer_values
        buffer_index = (
            descriptor.output_vram_addr - output_base
        ) // output_buffer_values
        gate_vram_addr = output_base + (2 + buffer_index) * output_buffer_values
        self._emit(
            Resource.CONTROL,
            "FENCE",
            key=key,
            queue_id=queue_id,
            note="wait for X_STATE output before Vector consumes it",
        )
        self._emit(
            Resource.VECTOR,
            "KDA_OUTPUT_GATE_RMSNORM",
            key=key,
            descriptor=descriptor,
            aux_vram_addr=gate_vram_addr,
            note="existing Vector service gates and normalizes X_STATE output",
        )
        self._emit(
            Resource.MATRIX,
            "KDA_OUT_PROJECTION",
            key=key,
            descriptor=descriptor,
            note="12288 -> 7168 using the existing Matrix service",
        )
