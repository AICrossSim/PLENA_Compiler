"""Full 52-layer Nemotron 3 Attention/MoE/Mamba model trace."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum

from aten.mamba.scheduler import (
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    Resource,
    SchedulePhase,
    ScheduleTrace,
    TraceEvent,
)


NEMOTRON3_PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"


class HybridLayerType(StrEnum):
    MAMBA = "mamba"
    MOE = "moe"
    ATTENTION = "attention"


SYMBOL_TO_LAYER = {
    "M": HybridLayerType.MAMBA,
    "E": HybridLayerType.MOE,
    "*": HybridLayerType.ATTENTION,
}


@dataclass(frozen=True)
class Nemotron3Architecture:
    hidden_size: int = 2688
    num_layers: int = 52
    attention_heads: int = 32
    kv_heads: int = 2
    attention_head_dim: int = 128
    routed_experts: int = 128
    experts_per_token: int = 6
    moe_intermediate_size: int = 1856
    shared_experts: int = 1
    shared_intermediate_size: int = 3712
    pattern: str = NEMOTRON3_PATTERN

    def __post_init__(self) -> None:
        if len(self.pattern) != self.num_layers:
            raise ValueError("Nemotron pattern length must equal num_layers")
        if set(self.pattern) != set(SYMBOL_TO_LAYER):
            raise ValueError("Nemotron pattern must contain M, E, and * layer symbols")


@dataclass(frozen=True)
class HybridTraceEvent:
    index: int
    pass_index: int
    layer_id: int
    layer_type: HybridLayerType
    stage: str
    resource: str
    implementation: str
    token_offset: int
    valid_tokens: int
    source_state_event_index: int | None = None


@dataclass(frozen=True)
class Nemotron3HybridTrace:
    architecture: Nemotron3Architecture
    phase: SchedulePhase
    events: tuple[HybridTraceEvent, ...]
    mamba_trace: ScheduleTrace

    def count(self, stage: str) -> int:
        return sum(event.stage == stage for event in self.events)

    def to_dict(self) -> dict[str, object]:
        layer_types = [SYMBOL_TO_LAYER[symbol] for symbol in self.architecture.pattern]
        return {
            "schema_version": 1,
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            "scope": "full_52_layer_body_trace",
            "phase": self.phase.value,
            "architecture": {
                **asdict(self.architecture),
                "layer_counts": {
                    layer_type.value: layer_types.count(layer_type)
                    for layer_type in HybridLayerType
                },
                "layer_types": [layer_type.value for layer_type in layer_types],
            },
            "summary": {
                "event_count": len(self.events),
                "stage_counts": {
                    stage: self.count(stage)
                    for stage in sorted({event.stage for event in self.events})
                },
                "implementation_counts": {
                    implementation: sum(
                        event.implementation == implementation for event in self.events
                    )
                    for implementation in sorted(
                        {event.implementation for event in self.events}
                    )
                },
                "mamba_state_trace": self.mamba_trace.to_dict()["summary"],
            },
            "mamba_residency": {
                "cache_policy": self.mamba_trace.config.cache_policy.value,
                "capacity_bytes": self.mamba_trace.config.residency_capacity_bytes,
                "state_cache_entries": self.mamba_trace.config.state_cache_entries,
                "source": self.mamba_trace.config.residency_source,
                "target": self.mamba_trace.config.residency_target,
                "resident_state_keys": [
                    {"request_id": request_id, "layer_id": layer_id}
                    for request_id, layer_id in self.mamba_trace.config.resident_state_keys
                ],
            },
            "events": [
                {
                    **asdict(event),
                    "layer_type": event.layer_type.value,
                }
                for event in self.events
            ],
        }


class Nemotron3HybridScheduler:
    """Interleave the state trace with the 23 MoE and 6 Attention layers."""

    def __init__(
        self,
        mamba_config: MambaScheduleConfig,
        architecture: Nemotron3Architecture | None = None,
    ) -> None:
        self.config = mamba_config
        self.arch = architecture or Nemotron3Architecture()
        if self.config.cache_policy.value == "lru" and 0 < self.config.state_cache_entries < 23:
            raise ValueError("hybrid trace requires streaming, pinned, or full-resident LRU state")

    def build(self) -> Nemotron3HybridTrace:
        mamba_trace = Nemotron3MambaScheduler(self.config).build()
        buckets, tail = _bucket_mamba_events(mamba_trace)
        events: list[HybridTraceEvent] = []
        passes = self.config.decode_tokens if self.config.phase == SchedulePhase.DECODE else 1
        valid_tokens = 1 if self.config.phase == SchedulePhase.DECODE else self.config.sequence_length
        for pass_index in range(passes):
            token_offset = pass_index if self.config.phase == SchedulePhase.DECODE else 0
            for layer_id, symbol in enumerate(self.arch.pattern):
                layer_type = SYMBOL_TO_LAYER[symbol]
                self._append(
                    events,
                    pass_index,
                    layer_id,
                    layer_type,
                    "block_rms_norm",
                    "vector",
                    "existing_plena_service",
                    token_offset,
                    valid_tokens,
                )
                if layer_type == HybridLayerType.MAMBA:
                    for state_event in buckets[(pass_index, layer_id)]:
                        self._append_state(events, pass_index, layer_type, state_event)
                elif layer_type == HybridLayerType.MOE:
                    for stage, resource in (
                        ("moe_router_topk", "vector"),
                        ("moe_routed_experts", "matrix"),
                        ("moe_shared_expert", "matrix"),
                    ):
                        self._append(
                            events,
                            pass_index,
                            layer_id,
                            layer_type,
                            stage,
                            resource,
                            "existing_plena_service",
                            token_offset,
                            valid_tokens,
                        )
                else:
                    for stage, resource in (
                        ("attention_qkv_projection", "matrix"),
                        ("attention_qk_softmax_pv", "matrix_vector"),
                        ("attention_out_projection", "matrix"),
                    ):
                        self._append(
                            events,
                            pass_index,
                            layer_id,
                            layer_type,
                            stage,
                            resource,
                            "existing_plena_service",
                            token_offset,
                            valid_tokens,
                        )
                self._append(
                    events,
                    pass_index,
                    layer_id,
                    layer_type,
                    "block_residual",
                    "vector",
                    "existing_plena_service",
                    token_offset,
                    valid_tokens,
                )
        for state_event in tail:
            layer_type = (
                HybridLayerType.MAMBA
                if state_event.layer_id is not None
                else HybridLayerType.MAMBA
            )
            self._append_state(events, passes - 1, layer_type, state_event)
        return Nemotron3HybridTrace(self.arch, self.config.phase, tuple(events), mamba_trace)

    @staticmethod
    def _append(
        events: list[HybridTraceEvent],
        pass_index: int,
        layer_id: int,
        layer_type: HybridLayerType,
        stage: str,
        resource: str,
        implementation: str,
        token_offset: int,
        valid_tokens: int,
        source_state_event_index: int | None = None,
    ) -> None:
        events.append(
            HybridTraceEvent(
                len(events),
                pass_index,
                layer_id,
                layer_type,
                stage,
                resource,
                implementation,
                token_offset,
                valid_tokens,
                source_state_event_index,
            )
        )

    def _append_state(
        self,
        events: list[HybridTraceEvent],
        pass_index: int,
        layer_type: HybridLayerType,
        state_event: TraceEvent,
    ) -> None:
        implementation = "existing_isa"
        if state_event.operation == "PROJECTION_SCATTER":
            implementation = "l_scatter_m_v1"
        elif state_event.resource == Resource.CONTROL and state_event.instruction_word is None:
            implementation = "compiler_control"
        self._append(
            events,
            pass_index,
            state_event.layer_id if state_event.layer_id is not None else -1,
            layer_type,
            _mamba_stage_name(state_event.operation),
            state_event.resource.value,
            implementation,
            state_event.token_offset if state_event.token_offset is not None else pass_index,
            state_event.valid_tokens if state_event.valid_tokens is not None else 1,
            state_event.index,
        )


def _bucket_mamba_events(
    trace: ScheduleTrace,
) -> tuple[dict[tuple[int, int], list[TraceEvent]], list[TraceEvent]]:
    buckets: dict[tuple[int, int], list[TraceEvent]] = defaultdict(list)
    tail = []
    current_pass: dict[int, int] = {}
    for event in trace.events:
        if event.layer_id is None:
            tail.append(event)
            continue
        if event.operation == "COMMIT" and trace.config.flush_at_end:
            tail.append(event)
            continue
        if event.token_offset is not None:
            current_pass[event.layer_id] = (
                event.token_offset if trace.config.phase == SchedulePhase.DECODE else 0
            )
        pass_index = current_pass.get(event.layer_id, 0)
        buckets[(pass_index, event.layer_id)].append(event)
    return buckets, tail


def _mamba_stage_name(operation: str) -> str:
    names = {
        "IN_PROJECTION": "mamba_in_projection",
        "PROJECTION_SCATTER": "mamba_projection_scatter",
        "GATED_GROUP_RMSNORM": "mamba_gate_group_rms_norm",
        "OUT_PROJECTION": "mamba_out_projection",
    }
    return names.get(operation, f"x_state_{operation.lower()}")
