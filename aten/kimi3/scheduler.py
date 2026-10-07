"""Full 93-layer Kimi K3 KDA/MLA/LatentMoE structural trace."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import StrEnum

from aten.kda.scheduler import KIMI_K3_KDA_LAYERS, KdaScheduleConfig, KimiK3KdaScheduler
from aten.mamba.scheduler import Resource, SchedulePhase, ScheduleTrace, TraceEvent


KIMI_K3_HF_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"


class KimiMixerType(StrEnum):
    KDA = "kda"
    MLA = "mla"


class KimiFfnType(StrEnum):
    DENSE = "dense"
    LATENT_MOE = "latent_moe"


@dataclass(frozen=True)
class KimiK3Architecture:
    hidden_size: int = 7168
    num_layers: int = 93
    attention_heads: int = 96
    kda_head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    num_experts: int = 896
    experts_per_token: int = 16
    routed_expert_hidden_size: int = 3584
    moe_intermediate_size: int = 3072
    shared_experts: int = 2
    dense_intermediate_size: int = 33792
    attn_res_block_size: int = 12

    def __post_init__(self) -> None:
        if self.num_layers != 93 or self.hidden_size != 7168:
            raise ValueError(
                "the pinned Kimi K3 structural contract has 93 layers and hidden size 7168"
            )

    @property
    def kda_layers(self) -> tuple[int, ...]:
        return KIMI_K3_KDA_LAYERS

    @property
    def mla_layers(self) -> tuple[int, ...]:
        kda = set(self.kda_layers)
        return tuple(layer for layer in range(self.num_layers) if layer not in kda)

    @property
    def dense_layers(self) -> tuple[int, ...]:
        return (0,)

    @property
    def moe_layers(self) -> tuple[int, ...]:
        return tuple(range(1, self.num_layers))

    @property
    def attn_res_capture_layers(self) -> tuple[int, ...]:
        return tuple(range(0, self.num_layers, self.attn_res_block_size))


@dataclass(frozen=True)
class KimiHybridEvent:
    index: int
    pass_index: int
    layer_id: int
    mixer_type: KimiMixerType | None
    ffn_type: KimiFfnType | None
    stage: str
    resource: str
    implementation: str
    token_offset: int
    valid_tokens: int
    source_kda_event_index: int | None = None


@dataclass(frozen=True)
class KimiK3HybridTrace:
    architecture: KimiK3Architecture
    phase: SchedulePhase
    events: tuple[KimiHybridEvent, ...]
    kda_trace: ScheduleTrace

    def count(self, stage: str) -> int:
        return sum(event.stage == stage for event in self.events)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model": "moonshotai/Kimi-K3",
            "revision": KIMI_K3_HF_REVISION,
            "scope": "full_93_layer_text_backbone_structural_trace",
            "phase": self.phase.value,
            "architecture": {
                **asdict(self.architecture),
                "kda_layers": list(self.architecture.kda_layers),
                "mla_layers": list(self.architecture.mla_layers),
                "dense_layers": list(self.architecture.dense_layers),
                "moe_layers": list(self.architecture.moe_layers),
                "attn_res_capture_layers": list(
                    self.architecture.attn_res_capture_layers
                ),
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
                "kda_state_trace": self.kda_trace.to_dict()["summary"],
            },
            "kda_residency": {
                "cache_policy": self.kda_trace.config.cache_policy.value,
                "capacity_bytes": self.kda_trace.config.residency_capacity_bytes,
                "state_cache_entries": self.kda_trace.config.state_cache_entries,
                "resident_state_keys": [
                    {"request_id": request_id, "layer_id": layer_id}
                    for request_id, layer_id in self.kda_trace.config.resident_state_keys
                ],
            },
            "events": [
                {
                    **asdict(event),
                    "mixer_type": event.mixer_type.value
                    if event.mixer_type is not None
                    else None,
                    "ffn_type": event.ffn_type.value
                    if event.ffn_type is not None
                    else None,
                }
                for event in self.events
            ],
            "limits": [
                "KDA events have existing-ISA physical lowering",
                (
                    "this structural trace does not itself carry machine code; the connected decode builder lowers "
                    "compact MLA, LatentMoE, dense FFN, AttnRes, and KDA blocks for context_length=1"
                ),
                "single-token Top-16 MoE uses looped dynamic expert dispatch",
                (
                    "the full 93-layer, 96-head single-token program now assembles; "
                    "MLA head bodies remain statically emitted and are a code-size optimization target"
                ),
                (
                    "four-token compressed MLA cache append/reconstruct is executable at block level; "
                    "the full 93-layer connected builder remains context_length=1"
                ),
                "the trace is not an end-to-end GPU or RTL latency claim",
            ],
        }


class KimiK3HybridScheduler:
    """Interleave all 69 KDA states with the pinned 24 MLA and 93 FFN blocks."""

    def __init__(
        self,
        kda_config: KdaScheduleConfig,
        architecture: KimiK3Architecture | None = None,
    ) -> None:
        self.config = kda_config
        self.arch = architecture or KimiK3Architecture()

    def build(self) -> KimiK3HybridTrace:
        kda_trace = KimiK3KdaScheduler(self.config).build()
        buckets, tail = _bucket_kda_events(kda_trace)
        events: list[KimiHybridEvent] = []
        passes = (
            self.config.decode_tokens
            if self.config.phase == SchedulePhase.DECODE
            else 1
        )
        valid_tokens = (
            1
            if self.config.phase == SchedulePhase.DECODE
            else self.config.sequence_length
        )

        for pass_index in range(passes):
            token_offset = (
                pass_index if self.config.phase == SchedulePhase.DECODE else 0
            )
            captured_blocks = 0
            for layer_id in range(self.arch.num_layers):
                mixer = (
                    KimiMixerType.KDA
                    if layer_id in self.arch.kda_layers
                    else KimiMixerType.MLA
                )
                ffn = KimiFfnType.DENSE if layer_id == 0 else KimiFfnType.LATENT_MOE
                if captured_blocks:
                    self._append(
                        events,
                        pass_index,
                        layer_id,
                        mixer,
                        ffn,
                        "attn_res_before_mixer",
                        "matrix_vector",
                        "existing_plena_service",
                        token_offset,
                        valid_tokens,
                    )
                if layer_id in self.arch.attn_res_capture_layers:
                    captured_blocks += 1
                    self._append(
                        events,
                        pass_index,
                        layer_id,
                        mixer,
                        ffn,
                        "attn_res_capture_prefix",
                        "vector",
                        "existing_plena_service",
                        token_offset,
                        valid_tokens,
                    )
                self._append(
                    events,
                    pass_index,
                    layer_id,
                    mixer,
                    ffn,
                    "input_rms_norm",
                    "vector",
                    "existing_plena_service",
                    token_offset,
                    valid_tokens,
                )
                if mixer == KimiMixerType.KDA:
                    for state_event in buckets[(pass_index, layer_id)]:
                        self._append_kda(events, pass_index, ffn, state_event)
                else:
                    for stage, resource in (
                        ("mla_q_low_rank_projection", "matrix_vector"),
                        ("mla_kv_low_rank_projection", "matrix_vector"),
                        ("mla_rope_kv_cache_attention", "matrix_vector"),
                        ("mla_output_gate", "matrix_vector"),
                        ("mla_out_projection", "matrix"),
                    ):
                        self._append(
                            events,
                            pass_index,
                            layer_id,
                            mixer,
                            ffn,
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
                    mixer,
                    ffn,
                    "prefix_sum_after_mixer",
                    "vector",
                    "existing_plena_service",
                    token_offset,
                    valid_tokens,
                )
                self._append(
                    events,
                    pass_index,
                    layer_id,
                    mixer,
                    ffn,
                    "attn_res_before_ffn",
                    "matrix_vector",
                    "existing_plena_service",
                    token_offset,
                    valid_tokens,
                )
                self._append(
                    events,
                    pass_index,
                    layer_id,
                    mixer,
                    ffn,
                    "post_attention_rms_norm",
                    "vector",
                    "existing_plena_service",
                    token_offset,
                    valid_tokens,
                )
                if ffn == KimiFfnType.DENSE:
                    self._append(
                        events,
                        pass_index,
                        layer_id,
                        mixer,
                        ffn,
                        "dense_situ_ffn",
                        "matrix_vector",
                        "existing_plena_service",
                        token_offset,
                        valid_tokens,
                    )
                else:
                    for stage, resource in (
                        ("latent_moe_router_top16", "vector"),
                        ("latent_moe_down_projection", "matrix"),
                        ("latent_moe_routed_experts", "matrix_vector"),
                        ("latent_moe_up_projection", "matrix"),
                        ("latent_moe_shared_experts", "matrix_vector"),
                    ):
                        self._append(
                            events,
                            pass_index,
                            layer_id,
                            mixer,
                            ffn,
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
                    mixer,
                    ffn,
                    "prefix_sum_after_ffn",
                    "vector",
                    "existing_plena_service",
                    token_offset,
                    valid_tokens,
                )
            self._append(
                events,
                pass_index,
                -1,
                None,
                None,
                "output_attn_res",
                "matrix_vector",
                "existing_plena_service",
                token_offset,
                valid_tokens,
            )
            self._append(
                events,
                pass_index,
                -1,
                None,
                None,
                "final_rms_norm",
                "vector",
                "existing_plena_service",
                token_offset,
                valid_tokens,
            )

        for state_event in tail:
            self._append_kda(events, passes - 1, None, state_event)
        return KimiK3HybridTrace(self.arch, self.config.phase, tuple(events), kda_trace)

    @staticmethod
    def _append(
        events: list[KimiHybridEvent],
        pass_index: int,
        layer_id: int,
        mixer_type: KimiMixerType | None,
        ffn_type: KimiFfnType | None,
        stage: str,
        resource: str,
        implementation: str,
        token_offset: int,
        valid_tokens: int,
        source_kda_event_index: int | None = None,
    ) -> None:
        events.append(
            KimiHybridEvent(
                len(events),
                pass_index,
                layer_id,
                mixer_type,
                ffn_type,
                stage,
                resource,
                implementation,
                token_offset,
                valid_tokens,
                source_kda_event_index,
            )
        )

    def _append_kda(
        self,
        events: list[KimiHybridEvent],
        pass_index: int,
        ffn_type: KimiFfnType | None,
        event: TraceEvent,
    ) -> None:
        implementation = "existing_isa"
        if event.operation == "PROJECTION_SCATTER":
            implementation = "l_scatter_m_v1"
        elif event.resource == Resource.STATE:
            implementation = "x_state_v2"
        elif event.resource == Resource.CONTROL and event.instruction_word is None:
            implementation = "compiler_control"
        self._append(
            events,
            pass_index,
            event.layer_id if event.layer_id is not None else -1,
            KimiMixerType.KDA,
            ffn_type,
            _kda_stage_name(event.operation),
            event.resource.value,
            implementation,
            event.token_offset if event.token_offset is not None else pass_index,
            event.valid_tokens if event.valid_tokens is not None else 1,
            event.index,
        )


def _bucket_kda_events(
    trace: ScheduleTrace,
) -> tuple[dict[tuple[int, int], list[TraceEvent]], list[TraceEvent]]:
    buckets: dict[tuple[int, int], list[TraceEvent]] = defaultdict(list)
    tail: list[TraceEvent] = []
    current_pass: dict[int, int] = {}
    for event in trace.events:
        if event.layer_id is None or (
            event.operation == "COMMIT" and trace.config.flush_at_end
        ):
            tail.append(event)
            continue
        if event.token_offset is not None:
            current_pass[event.layer_id] = (
                event.token_offset if trace.config.phase == SchedulePhase.DECODE else 0
            )
        buckets[(current_pass.get(event.layer_id, 0), event.layer_id)].append(event)
    return buckets, tail


def _kda_stage_name(operation: str) -> str:
    names = {
        "KDA_QKV_PROJECTION": "kda_qkv_projection",
        "KDA_DECAY_BETA_PROJECTION": "kda_decay_beta_projection",
        "KDA_OUTPUT_GATE_PROJECTION": "kda_output_gate_projection",
        "PROJECTION_SCATTER": "kda_projection_scatter",
        "KDA_OUTPUT_GATE_RMSNORM": "kda_output_gate_rmsnorm",
        "KDA_OUT_PROJECTION": "kda_out_projection",
    }
    return names.get(operation, f"x_state_{operation.lower()}")
