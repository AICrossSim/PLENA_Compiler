"""Shape-only Qwen3 prefill lowering into a compiler cost trace.

The frontend deliberately reuses the native decoder lowering helpers.  Dense
and static-index MoE traces therefore describe the same ISA schedule as the
transactional compiler without materializing model weights or rendering ASM.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import compiler.aten.ops as ops
from compiler.aten.agu import AGU_MODE_LEGACY, AGU_MODE_LOOP_V1
from compiler.aten.cost_emitter import (
    COST_TRACE_GRANULARITIES,
    COST_TRACE_GRANULARITY_AFFINE_BLOCK_SUMMARY_V1,
    COST_TRACE_GRANULARITY_DETAILED,
    CostTrace,
    EnergyAction,
    MemoryEvent,
    ParallelKernelCensusEntry,
    ScheduleAffineAdd,
    ScheduleAffineLoad,
    ScheduleInstruction,
    ScheduleNode,
    ScheduleRepeat,
    ScheduleSequence,
    ScheduleUnavailable,
    UNCLASSIFIED_PARALLEL_KERNEL,
    _logic_energy_actions,
    _sram_actions,
    optimize_cost_trace_loop_agu,
    parallel_kernel_lineage_id,
)
from compiler.aten.isa_builder import RepeatAxis
from compiler.aten.model_extract import ModelConfig
from compiler.aten.moe import (
    MOE_LOWERING_SCHEDULE_COMPACT_ROUTE_V2,
    MOE_LOWERING_SCHEDULE_LEGACY_STATIC_V1,
    MOE_LOWERING_SCHEDULES,
    FixedBalancedRoutingSummary,
    MoeRoutingPlan,
    coerce_routing_plan,
)
from compiler.aten.ops.registry import Backend, OpRegistry
from compiler.aten.plena import PlenaCompiler
from compiler.asm_templates.ffn_address_plan import (
    FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1,
    FFN_ADDRESS_SCHEDULES,
)
from compiler.asm_templates._imm import load_large_int
from compiler.asm_templates.ffn_projection_plan import (
    FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2,
    FFN_PROJECTION_SCHEDULES,
)
from compiler.aten.plena.kv_residency import plan_kv_residency
from compiler.aten.plena.native_layout import (
    FP_CONSTANT_NUM_DEFAULT,
    NATIVE_LAYOUT_SCHEMA_VERSION,
    PACKED_QK_SCHEDULES,
    PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1,
    AttentionHeadPacking,
    SequencePackingPlan,
    SOFTMAX_STATE_SCHEDULES,
    SOFTMAX_STATE_SCHEDULE_STREAMED_V2,
    build_attention_head_packing,
    build_softmax_state_layout,
)
from compiler.aten.plena_frontend import (
    LayerInputVars,
    MoeExpertInputVars,
    MoeLayerInputVars,
    _add_residual,
    _emit_ffn_block,
    _emit_compact_route_weight_token,
    _emit_fpram_row_to_vram,
    _emit_load_route_weight_lane,
    _emit_moe_block,
    _emit_normalize_route_weights,
    _emit_packed_attention_block,
    _emit_router_softmax,
    _emit_scale_wide_row_from_fp_register,
    _emit_scale_wide_row_from_fpram,
    _emit_selected_probability,
    _reset_moe_fpram_scratch,
    _save_residual_and_norm,
    _linear_projection,
)


@dataclass(frozen=True)
class CompilerCostHardware:
    mlen: int
    blen: int
    vlen: int
    hlen: int
    broadcast_amount: int
    mram_tile_capacity: int
    hbm_m_prefetch_amount: int
    hbm_v_prefetch_amount: int
    hbm_v_writeback_amount: int
    hbm_channels: int = 128
    fp_sram_depth: int | None = None
    fp_constant_num: int = FP_CONSTANT_NUM_DEFAULT
    kv_residency_policy: str = "raw-tiles"

    def validate(self) -> None:
        positive = asdict(self)
        for name, value in positive.items():
            if name == "kv_residency_policy":
                continue
            if value is None:
                continue
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.vlen != self.mlen:
            raise ValueError(
                f"native decoder cost lowering currently requires VLEN==MLEN, got {self.vlen}!={self.mlen}"
            )
        if self.mlen % self.blen:
            raise ValueError(f"MLEN={self.mlen} must be divisible by BLEN={self.blen}")


@dataclass(frozen=True)
class NativeDecoderCostLayout:
    padded_seq_len: int
    rows_per_batch: int
    compile_seq_rows: int
    padded_hidden: int
    padded_inter: int
    padded_head_dim: int
    head_packing: AttentionHeadPacking
    sequence_packing: SequencePackingPlan


_ONE_LAYER_CACHE_LIMIT = 4
_ONE_LAYER_TRACE_CACHE: OrderedDict[tuple[Any, ...], CostTrace] = OrderedDict()


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _mapping_value(mapping: Mapping[str, Any], *names: str, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def load_cost_model_config(value: ModelConfig | Mapping[str, Any] | str | Path) -> tuple[ModelConfig, int | None]:
    if isinstance(value, ModelConfig):
        return value, None
    if isinstance(value, (str, Path)):
        with Path(value).open() as f:
            value = json.load(f)
    if not isinstance(value, Mapping):
        raise TypeError(f"model_config must be ModelConfig, mapping, or JSON path, got {type(value).__name__}")
    hidden = int(_mapping_value(value, "hidden_size"))
    heads = int(_mapping_value(value, "num_attention_heads", "num_heads"))
    kv_heads = int(_mapping_value(value, "num_key_value_heads", "num_kv_heads", default=heads))
    dense_inter = int(
        _mapping_value(
            value,
            "intermediate_size",
            "dense_inter_dim",
            "inter_dim",
            default=4 * hidden,
        )
    )
    raw_moe_inter = _mapping_value(value, "moe_intermediate_size", "moe_inter_dim")
    moe_inter = None if raw_moe_inter is None else int(raw_moe_inter)
    num_experts = int(_mapping_value(value, "num_experts", default=0) or 0)
    active_inter = moe_inter if num_experts and moe_inter is not None else dense_inter
    config = ModelConfig(
        hidden_size=hidden,
        inter_dim=active_inter,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=int(_mapping_value(value, "head_dim", default=hidden // heads)),
        eps=float(_mapping_value(value, "rms_norm_eps", "eps", default=1e-5)),
        rope_theta=float(_mapping_value(value, "rope_theta", default=10_000.0)),
        vocab_size=_mapping_value(value, "vocab_size"),
        model_type=str(_mapping_value(value, "model_type", default="qwen3")),
        dense_inter_dim=dense_inter,
        moe_inter_dim=moe_inter,
        num_experts=num_experts,
        experts_per_token=int(
            _mapping_value(
                value,
                "num_experts_per_tok",
                "experts_per_token",
                default=0,
            )
            or 0
        ),
        norm_topk_prob=bool(_mapping_value(value, "norm_topk_prob", default=False)),
        decoder_sparse_step=int(_mapping_value(value, "decoder_sparse_step", default=1) or 1),
        mlp_only_layers=tuple(int(layer) for layer in (_mapping_value(value, "mlp_only_layers", default=()) or ())),
    )
    layers = _mapping_value(value, "num_hidden_layers", "num_layers")
    return config, int(layers) if layers is not None else None


def _build_layout(
    model: ModelConfig,
    hardware: CompilerCostHardware,
    *,
    seq_len: int,
    batch_size: int,
    layer_idx: int,
    native_layout_mode: str,
) -> NativeDecoderCostLayout:
    if model.model_type not in {"qwen3", "qwen3_moe"}:
        raise ValueError(f"CostEmitter frontend supports model_type='qwen3' or 'qwen3_moe', got {model.model_type!r}")
    if model.num_heads % model.num_kv_heads:
        raise ValueError(f"num_heads={model.num_heads} must be divisible by num_kv_heads={model.num_kv_heads}")
    if hardware.hlen < model.head_dim:
        raise ValueError(
            "Packed GQA does not support head-dimension tiling: "
            f"HLEN={hardware.hlen} is smaller than head_dim={model.head_dim}"
        )
    if hardware.hlen > hardware.mlen:
        raise ValueError(f"HLEN={hardware.hlen} cannot exceed MLEN={hardware.mlen}")
    sequence_packing = SequencePackingPlan.build(
        batch_size=batch_size,
        seq_len=seq_len,
        mlen=hardware.mlen,
        mode=native_layout_mode,
    )
    packing = build_attention_head_packing(
        mlen=hardware.mlen,
        hlen=hardware.hlen,
        head_dim=model.head_dim,
        logical_broadcast_amount=hardware.broadcast_amount,
        gqa_ratio=model.head_ratio,
        num_kv_heads=model.num_kv_heads,
        mode=native_layout_mode,
    )
    return NativeDecoderCostLayout(
        padded_seq_len=sequence_packing.rows_per_attention_group,
        rows_per_batch=sequence_packing.rows_per_attention_group,
        compile_seq_rows=sequence_packing.compile_seq_rows,
        padded_hidden=_ceil_to_multiple(model.hidden_size, hardware.mlen),
        padded_inter=_ceil_to_multiple(
            (
                model.moe_inter_dim or model.inter_dim
                if model.is_moe_layer(layer_idx)
                else model.dense_inter_dim or model.inter_dim
            ),
            hardware.mlen,
        ),
        padded_head_dim=_ceil_to_multiple(model.head_dim, hardware.mlen),
        head_packing=packing,
        sequence_packing=sequence_packing,
    )


def _register_shape_attention_inputs(
    prog: PlenaCompiler,
    model: ModelConfig,
    layout: NativeDecoderCostLayout,
    *,
    layer_idx: int,
) -> tuple[Any, Any, list[Any], list[Any]]:
    def prefix(name: str) -> str:
        return f"{name}_{layer_idx}"

    w_q = prog.input(prefix("W_q"), (layout.padded_hidden, layout.head_packing.total_q_dim))
    w_o = prog.input(prefix("W_o"), (layout.head_packing.total_q_dim, layout.padded_hidden))
    w_k_heads = []
    w_v_heads = []
    for head in range(model.num_kv_heads):
        w_k_heads.append(
            prog.input(
                f"W_k_{layer_idx}_h{head}",
                (layout.padded_hidden, layout.head_packing.head_slot_dim),
                physical_shape=(layout.padded_hidden, layout.padded_head_dim),
            )
        )
        w_v_heads.append(
            prog.input(
                f"W_v_{layer_idx}_h{head}",
                (layout.padded_hidden, layout.head_packing.head_slot_dim),
                physical_shape=(layout.padded_hidden, layout.padded_head_dim),
            )
        )
    return w_q, w_o, w_k_heads, w_v_heads


def _register_shape_norm_inputs(
    prog: PlenaCompiler,
    layout: NativeDecoderCostLayout,
    *,
    layer_idx: int,
) -> dict[str, Any]:
    """Register the learned Qwen3 norm weights in native HBM order."""

    prefix = lambda name: f"{name}_{layer_idx}"
    return {
        "input_norm": prog.input(
            prefix("W_input_norm"),
            (layout.compile_seq_rows, layout.padded_hidden),
        ),
        "post_attn_norm": prog.input(
            prefix("W_post_attn_norm"),
            (layout.compile_seq_rows, layout.padded_hidden),
        ),
        "q_norm": prog.input(
            prefix("W_q_norm"),
            (layout.compile_seq_rows, layout.head_packing.total_q_dim),
        ),
        "k_norm": prog.input(
            prefix("W_k_norm"),
            (layout.compile_seq_rows, layout.head_packing.head_slot_dim),
            physical_shape=(layout.compile_seq_rows, layout.padded_head_dim),
        ),
    }


def _register_shape_layer_inputs(
    prog: PlenaCompiler,
    model: ModelConfig,
    layout: NativeDecoderCostLayout,
    *,
    layer_idx: int,
) -> LayerInputVars:
    w_q, w_o, w_k_heads, w_v_heads = _register_shape_attention_inputs(prog, model, layout, layer_idx=layer_idx)
    prefix = lambda name: f"{name}_{layer_idx}"
    w_gate = prog.input(prefix("W_gate"), (layout.padded_hidden, layout.padded_inter))
    w_up = prog.input(prefix("W_up"), (layout.padded_hidden, layout.padded_inter))
    w_down = prog.input(prefix("W_down"), (layout.padded_inter, layout.padded_hidden))
    norms = _register_shape_norm_inputs(prog, layout, layer_idx=layer_idx)
    return LayerInputVars(
        w_q=w_q,
        w_o=w_o,
        w_k_heads=w_k_heads,
        w_v_heads=w_v_heads,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
        **norms,
    )


def _register_shape_moe_layer_inputs(
    prog: PlenaCompiler,
    model: ModelConfig,
    layout: NativeDecoderCostLayout,
    routing: MoeRoutingPlan | FixedBalancedRoutingSummary,
    *,
    layer_idx: int,
) -> MoeLayerInputVars:
    """Register only experts selected by the static routing plan."""

    w_q, w_o, w_k_heads, w_v_heads = _register_shape_attention_inputs(prog, model, layout, layer_idx=layer_idx)
    prefix = lambda name: f"{name}_{layer_idx}"
    w_router = prog.input(
        prefix("W_router"),
        (layout.padded_hidden, prog.mlen),
    )
    norms = _register_shape_norm_inputs(prog, layout, layer_idx=layer_idx)
    experts: dict[int, MoeExpertInputVars] = {}
    for expert_id in routing.active_expert_ids:
        experts[expert_id] = MoeExpertInputVars(
            w_gate=prog.input(
                f"W_expert_gate_{layer_idx}_e{expert_id}",
                (layout.padded_hidden, layout.padded_inter),
            ),
            w_up=prog.input(
                f"W_expert_up_{layer_idx}_e{expert_id}",
                (layout.padded_hidden, layout.padded_inter),
            ),
            w_down=prog.input(
                f"W_expert_down_{layer_idx}_e{expert_id}",
                (layout.padded_inter, layout.padded_hidden),
            ),
        )
    return MoeLayerInputVars(
        w_q=w_q,
        w_o=w_o,
        w_k_heads=w_k_heads,
        w_v_heads=w_v_heads,
        w_router=w_router,
        experts=experts,
        **norms,
    )


def _emit_fixed_balanced_route_weights(
    prog: PlenaCompiler,
    *,
    router_probs,
    identity,
    summary: FixedBalancedRoutingSummary,
    physical_rows: int,
    normalize: bool,
    layer_idx: int,
):
    """Emit one route-weight template and account for all active tokens."""
    route_weights = prog.alloc(
        f"moe_route_weights_{layer_idx}",
        physical_rows,
        prog.mlen,
        strict=False,
        physical_shape=(physical_rows, prog.mlen),
    )
    prog.vram_fill_zero(route_weights)
    route_scratch = prog.alloc(
        f"moe_route_extract_{layer_idx}",
        1,
        prog.mlen,
        strict=False,
        physical_shape=(1, prog.mlen),
    )
    route_base = prog._ONLINE_SOFTMAX_FPSRAM_BASE
    route_weights_addr = prog.get_vram_addr(route_weights.name)
    with prog.cost_repeat_region(
        summary.num_tokens,
        name="moe_balanced_token_route_weights",
        repeat_kind="fixed_balanced",
    ):
        _reset_moe_fpram_scratch(prog)
        for rank in range(summary.experts_per_token):
            _emit_selected_probability(
                prog,
                router_probs=router_probs,
                identity=identity,
                scratch=route_scratch,
                physical_row=0,
                identity_row=rank,
                fpram_addr=route_base + rank,
            )
        if normalize:
            _emit_normalize_route_weights(prog, fpram_addr=route_base, count=summary.experts_per_token)
        _emit_fpram_row_to_vram(prog, fpram_addr=route_base, vram_addr=route_weights_addr)
    prog.free_tensor(route_scratch)
    return route_weights


def _emit_fixed_balanced_compact_route_weights(
    prog: PlenaCompiler,
    *,
    router_probs,
    summary: FixedBalancedRoutingSummary,
    physical_rows: int,
    normalize: bool,
    layer_idx: int,
):
    """Emit one direct-lane route template and repeat it for every token."""
    route_weights = prog.alloc(
        f"moe_route_weights_{layer_idx}",
        physical_rows,
        prog.mlen,
        strict=False,
        physical_shape=(physical_rows, prog.mlen),
    )
    with prog.cost_repeat_region(
        summary.num_tokens,
        name="moe_balanced_compact_token_route_weights",
        repeat_kind="fixed_balanced",
    ):
        _emit_compact_route_weight_token(
            prog,
            router_probs=router_probs,
            route_weights=route_weights,
            token_index=0,
            physical_row=0,
            expert_rank_pairs=tuple((rank, rank) for rank in range(summary.experts_per_token)),
            normalize=normalize,
        )
    return route_weights


def _emit_fixed_balanced_moe_block(
    prog: PlenaCompiler,
    current,
    layer_inputs: MoeLayerInputVars,
    scratch,
    *,
    router_mask,
    route_identity,
    summary: FixedBalancedRoutingSummary,
    model_cfg: ModelConfig,
    layer_idx: int,
) -> tuple[Any, dict[int, dict[str, int]]]:
    """Cost-only MoE lowering with algebraically repeated route operations."""
    compact_routes = (
        prog.moe_lowering_schedule == MOE_LOWERING_SCHEDULE_COMPACT_ROUTE_V2 and summary.experts_per_token <= 8
    )
    effective_schedule = (
        MOE_LOWERING_SCHEDULE_COMPACT_ROUTE_V2 if compact_routes else MOE_LOWERING_SCHEDULE_LEGACY_STATIC_V1
    )
    fallback_reasons = (
        {}
        if compact_routes
        else {"top_k_exceeds_compact_register_capacity": 1}
        if prog.moe_lowering_schedule == MOE_LOWERING_SCHEDULE_COMPACT_ROUTE_V2
        else {}
    )
    prog._moe_lowering_stats = {
        "moe_lowering_schedule": effective_schedule,
        "router_softmax_schedule": ("single-block-first-v2" if compact_routes else "online-general-v1"),
        "route_probability_access": ("direct_vector_lane" if compact_routes else "identity_reduce"),
        "route_identity_elided": compact_routes,
        "route_fpram_resets_elided": (summary.num_tokens * prog.mlen if compact_routes else 0),
        "route_weight_lane_loads": (2 * summary.route_count if compact_routes else 0),
        "route_weight_register_reuse": compact_routes,
        "expert_route_run_count": summary.num_experts,
        "affine_route_count": summary.route_count,
        "irregular_route_count": 0,
        "moe_fallback_reasons": fallback_reasons,
    }
    with prog.cost_stage("norm"):
        with prog.cost_parallel_kernel(
            "moe_norm",
            tp_semantics="token_replicated_hidden",
            cp_semantics="token_partitioned",
            ep_semantics="none",
            logical_rows=summary.num_tokens,
            logical_m=summary.num_tokens,
            logical_n=current.shape[1],
        ):
            _save_residual_and_norm(
                prog,
                current,
                scratch,
                layer_inputs.post_attn_norm,
            )
    physical_rows, padded_hidden = current.physical_shape
    with prog.cost_stage("router"):
        with prog.cost_parallel_kernel(
            "moe_router_projection",
            tp_semantics="row_parallel_projection",
            cp_semantics="token_partitioned",
            ep_semantics="router_replicated",
            logical_rows=summary.num_tokens,
            logical_m=summary.num_tokens,
            logical_n=model_cfg.num_experts,
            logical_k=current.shape[1],
        ):
            router_probs = _linear_projection(
                prog,
                current,
                layer_inputs.w_router,
                f"moe_router_logits_{layer_idx}",
                physical_shape=(physical_rows, prog.mlen),
            )
        with prog.cost_parallel_kernel(
            "moe_router_postprocess",
            tp_semantics="token_replicated_hidden",
            cp_semantics="token_partitioned",
            ep_semantics="router_replicated",
            logical_rows=summary.num_tokens,
            logical_m=summary.num_tokens,
            logical_n=model_cfg.num_experts,
        ):
            if compact_routes:
                _emit_router_softmax(
                    prog,
                    router_probs,
                    physical_rows=physical_rows,
                    num_experts=model_cfg.num_experts,
                    compact=True,
                )
                route_weights = _emit_fixed_balanced_compact_route_weights(
                    prog,
                    router_probs=router_probs,
                    summary=summary,
                    physical_rows=physical_rows,
                    normalize=model_cfg.norm_topk_prob,
                    layer_idx=layer_idx,
                )
            else:
                if router_mask is None or route_identity is None:
                    raise ValueError("legacy MoE lowering requires mask and identity")
                prog.vram_add(router_probs, router_mask)
                _emit_router_softmax(
                    prog,
                    router_probs,
                    physical_rows=physical_rows,
                )
                route_weights = _emit_fixed_balanced_route_weights(
                    prog,
                    router_probs=router_probs,
                    identity=route_identity,
                    summary=summary,
                    physical_rows=physical_rows,
                    normalize=model_cfg.norm_topk_prob,
                    layer_idx=layer_idx,
                )
    with prog.cost_stage("combine"):
        with prog.cost_parallel_kernel(
            "moe_combine",
            tp_semantics="token_replicated_hidden",
            cp_semantics="token_partitioned",
            ep_semantics="expert_combine",
            logical_rows=summary.num_tokens,
            logical_m=summary.num_tokens,
            logical_n=padded_hidden,
        ):
            combined = prog.alloc(
                f"moe_combined_{layer_idx}",
                physical_rows,
                padded_hidden,
                strict=False,
                physical_shape=(physical_rows, padded_hidden),
            )
            prog.vram_fill_zero(combined)
    scalar_scratch = (
        prog.alloc(
            f"moe_route_scalar_{layer_idx}",
            1,
            prog.mlen,
            strict=False,
            physical_shape=(1, prog.mlen),
        )
        if not compact_routes
        else None
    )
    scalar_addr = prog._ONLINE_SOFTMAX_FPSRAM_BASE
    bucket_metadata: dict[int, dict[str, int]] = {}
    padded_rows = summary.padded_bucket_rows(prog.blen)
    expert_template_origins: dict[tuple[Any, ...], int] = {}
    expert_template_objects: dict[tuple[Any, ...], tuple[str, str, str]] = {}
    expert_template_count = 0
    expert_template_replays = 0
    dispatch_template_count = 0
    dispatch_template_replays = 0
    combine_template_count = 0
    combine_template_replays = 0
    for expert_id, real_rows in summary.routes_per_expert.items():
        expert_padded_rows = padded_rows[expert_id]
        bucket_metadata[expert_id] = {
            "real_rows": real_rows,
            "padded_rows": expert_padded_rows,
        }
        with prog.cost_stage("dispatch"):
            with prog.cost_parallel_kernel(
                "moe_dispatch",
                tp_semantics="token_replicated_hidden",
                cp_semantics="token_partitioned",
                ep_semantics="expert_dispatch",
                logical_rows=real_rows,
                logical_m=real_rows,
                logical_n=padded_hidden,
            ):
                bucket = prog.alloc(
                    f"moe_expert_{expert_id}_bucket_l{layer_idx}",
                    expert_padded_rows,
                    padded_hidden,
                    strict=False,
                    physical_shape=(expert_padded_rows, padded_hidden),
                )
                dispatch_template_key = (
                    "moe_balanced_dispatch_compact_v2",
                    expert_padded_rows,
                    padded_hidden,
                    real_rows,
                    prog.get_vram_addr(bucket.name),
                    prog.get_vram_addr(current.name),
                )
                if (
                    prog.cost_affine_summary_enabled()
                    and prog.replay_cost_summary_template(dispatch_template_key)
                ):
                    dispatch_template_replays += 1
                else:
                    dispatch_template_count += 1
                    with prog.cost_summary_template(dispatch_template_key):
                        prog.vram_fill_zero(bucket)
                        with prog.cost_repeat_region(
                            real_rows,
                            name=f"moe_balanced_dispatch_e{expert_id}",
                            repeat_kind="fixed_balanced",
                        ):
                            prog.vram_add(bucket, current, num_rows=1)
        expert = layer_inputs.experts[expert_id]
        with prog.cost_stage("experts"):
            with prog.cost_parallel_kernel(
                "moe_expert_ffn",
                tp_semantics="expert_tensor_sharded",
                cp_semantics="token_partitioned",
                ep_semantics="expert_partitioned",
                logical_rows=real_rows,
                logical_m=real_rows,
                logical_n=model_cfg.moe_inter_dim or model_cfg.inter_dim,
                logical_k=padded_hidden,
            ):
                template_key = (
                    "moe_expert_ffn_compact_route_v2",
                    expert_padded_rows,
                    padded_hidden,
                    model_cfg.moe_inter_dim or model_cfg.inter_dim,
                    prog.mlen,
                    prog.blen,
                    prog.ffn_address_schedule,
                    prog.ffn_projection_schedule,
                    tuple(
                        tuple(
                            line.split(maxsplit=1)[0]
                            for line in load_large_int(1, address)
                        )
                        for address in (
                            expert.w_gate.hbm_addr,
                            expert.w_up.hbm_addr,
                            expert.w_down.hbm_addr,
                        )
                    ),
                )
                template_origin = expert_template_origins.get(template_key)
                if template_origin is None:
                    expert_template_origins[template_key] = expert.w_gate.hbm_addr
                    expert_template_objects[template_key] = (
                        expert.w_gate.name,
                        expert.w_up.name,
                        expert.w_down.name,
                    )
                    expert_template_count += 1
                    with prog.cost_summary_template(
                        template_key,
                        allow_memory=True,
                    ):
                        ops.ffn(
                            prog,
                            bucket,
                            expert.w_gate,
                            expert.w_up,
                            expert.w_down,
                        )
                elif prog.cost_affine_summary_enabled():
                    origin_objects = expert_template_objects[template_key]
                    replayed = prog.replay_cost_summary_template(
                        template_key,
                        element_base_delta=expert.w_gate.hbm_addr - template_origin,
                        scale_base_delta=expert.w_gate.hbm_addr - template_origin,
                        memory_object_replacements=tuple(
                            zip(
                                origin_objects,
                                (
                                    expert.w_gate.name,
                                    expert.w_up.name,
                                    expert.w_down.name,
                                ),
                                strict=True,
                            )
                        ),
                    )
                    assert replayed
                    expert_template_replays += 1
                else:
                    ops.ffn(
                        prog,
                        bucket,
                        expert.w_gate,
                        expert.w_up,
                        expert.w_down,
                    )
        with prog.cost_stage("combine"):
            with prog.cost_parallel_kernel(
                "moe_combine",
                tp_semantics="token_replicated_hidden",
                cp_semantics="token_partitioned",
                ep_semantics="expert_combine",
                logical_rows=real_rows,
                logical_m=real_rows,
                logical_n=padded_hidden,
            ):
                route_fp = (
                    prog.register_allocator.allocate_fp(1)[0]
                    if compact_routes
                    else None
                )
                combine_template_key = (
                    "moe_balanced_combine_compact_v2",
                    compact_routes,
                    expert_padded_rows,
                    padded_hidden,
                    real_rows,
                    expert_id % summary.experts_per_token,
                    prog.get_vram_addr(bucket.name),
                    prog.get_vram_addr(combined.name),
                    prog.get_vram_addr(route_weights.name),
                    route_fp,
                )
                if (
                    prog.cost_affine_summary_enabled()
                    and prog.replay_cost_summary_template(combine_template_key)
                ):
                    combine_template_replays += 1
                else:
                    combine_template_count += 1
                    with prog.cost_summary_template(combine_template_key):
                        with prog.cost_repeat_region(
                            real_rows,
                            name=f"moe_balanced_combine_e{expert_id}",
                            repeat_kind="fixed_balanced",
                        ):
                            if compact_routes:
                                assert route_fp is not None
                                _emit_load_route_weight_lane(
                                    prog,
                                    route_weights=route_weights,
                                    physical_row=0,
                                    rank=(
                                        expert_id
                                        % summary.experts_per_token
                                    ),
                                    fp_register=route_fp,
                                )
                                _emit_scale_wide_row_from_fp_register(
                                    prog,
                                    bucket,
                                    row=0,
                                    fp_register=route_fp,
                                )
                            else:
                                assert scalar_scratch is not None
                                _emit_selected_probability(
                                    prog,
                                    router_probs=route_weights,
                                    identity=route_identity,
                                    scratch=scalar_scratch,
                                    physical_row=0,
                                    identity_row=(
                                        expert_id
                                        % summary.experts_per_token
                                    ),
                                    fpram_addr=scalar_addr,
                                )
                                _emit_scale_wide_row_from_fpram(
                                    prog,
                                    bucket,
                                    row=0,
                                    fpram_addr=scalar_addr,
                                )
                            prog.vram_add(combined, bucket, num_rows=1)
                if route_fp is not None:
                    prog.register_allocator.free_fp([route_fp])
        prog.free_tensor(bucket)
    with prog.cost_stage("combine"):
        with prog.cost_parallel_kernel(
            "moe_combine",
            tp_semantics="token_replicated_hidden",
            cp_semantics="token_partitioned",
            ep_semantics="expert_combine",
            logical_rows=summary.num_tokens,
            logical_m=summary.num_tokens,
            logical_n=padded_hidden,
        ):
            prog.vram_fill_zero(current)
            prog.vram_add(current, combined)
            _add_residual(prog, current, scratch)
    if scalar_scratch is not None:
        prog.free_tensor(scalar_scratch)
    prog.free_tensor(route_weights)
    prog.free_tensor(router_probs)
    prog.free_tensor(combined)
    prog._moe_lowering_stats.update(
        {
            "expert_ffn_template_count": expert_template_count,
            "expert_ffn_template_replays": expert_template_replays,
            "expert_dispatch_template_count": dispatch_template_count,
            "expert_dispatch_template_replays": dispatch_template_replays,
            "expert_combine_template_count": combine_template_count,
            "expert_combine_template_replays": combine_template_replays,
            "moe_cost_trace_fidelity": (
                "exact_algebraic_expert_template" if prog.cost_affine_summary_enabled() else "ordered_explicit_experts"
            ),
            "moe_v4_aggregation": (
                "expert_descriptor_feature_grouped_exact" if expert_template_replays else "explicit_expert_descriptors"
            ),
        }
    )
    return current, bucket_metadata


def _schedule_stages(node: ScheduleNode) -> set[str]:
    """Return every compiler stage represented by a compressed schedule node."""
    if isinstance(
        node,
        (ScheduleInstruction, ScheduleAffineLoad, ScheduleAffineAdd, ScheduleUnavailable),
    ):
        return {node.stage}
    if isinstance(node, ScheduleSequence):
        stages: set[str] = set()
        for child in node.children:
            stages.update(_schedule_stages(child))
        return stages
    if isinstance(node, ScheduleRepeat):
        return _schedule_stages(node.body)
    raise TypeError(type(node).__name__)


def _scale_schedule(one_layer: CostTrace, num_layers: int) -> tuple[ScheduleSequence, str | None]:
    """Repeat one contiguous attention-plus-FFN region in program order."""
    if num_layers == 1:
        return one_layer.schedule, None

    hoisted_masks: list[ScheduleNode] = []
    schedule_children: list[ScheduleNode] = []
    for child in one_layer.schedule.children:
        stages = _schedule_stages(child)
        if stages and all(stage == "global/valid_col_mask" for stage in stages):
            # Mask construction is independent of decoder state and the mask
            # remains resident for the whole program.  Cost-only compilation
            # discovers it lazily in layer 0; hoist it immediately before the
            # repeated layer region so it is neither multiplied nor placed
            # between otherwise contiguous layer schedule nodes.
            hoisted_masks.append(child)
        else:
            schedule_children.append(child)

    classifications: list[str] = []
    for child in schedule_children:
        stages = _schedule_stages(child)
        has_layer = any(stage.startswith("layer/") for stage in stages)
        has_global = any(not stage.startswith("layer/") for stage in stages)
        if has_layer and has_global:
            return one_layer.schedule, "mixed_global_layer_schedule_node"
        classifications.append("layer" if has_layer else "global")

    layer_indices = [index for index, kind in enumerate(classifications) if kind == "layer"]
    if not layer_indices:
        return one_layer.schedule, "decoder_layer_schedule_missing"
    first, last = layer_indices[0], layer_indices[-1]
    if any(kind != "layer" for kind in classifications[first : last + 1]):
        return one_layer.schedule, "noncontiguous_decoder_layer_schedule"

    children: list[ScheduleNode] = list(schedule_children[:first])
    children.extend(hoisted_masks)
    children.append(
        ScheduleRepeat(
            count=num_layers,
            body=ScheduleSequence(tuple(schedule_children[first : last + 1])),
            name="decoder_layer",
            repeat_kind="model_layer",
        )
    )
    children.extend(schedule_children[last + 1 :])
    return ScheduleSequence(tuple(children)), None


def _scale_trace(one_layer: CostTrace, num_layers: int, *, layer_hbm_stride: int = 0) -> CostTrace:
    result = CostTrace(metadata=dict(one_layer.metadata))
    result.metadata["num_layers"] = num_layers
    routes_per_layer = int(one_layer.metadata.get("route_count", 0))
    result.metadata["route_count_per_layer"] = routes_per_layer
    result.metadata["decoder_route_count"] = routes_per_layer * num_layers
    stage_variants = one_layer.metadata.get("stage_parameterized_timing_variants")
    if isinstance(stage_variants, Mapping):
        scaled_stage_variants: dict[str, list[dict[str, Any]]] = {}
        total_variants: Counter[tuple[str, tuple[str, ...]]] = Counter()
        one_layer_variants: Counter[tuple[str, tuple[str, ...]]] = Counter()
        for stage_name, records in stage_variants.items():
            multiplier = num_layers if str(stage_name).startswith("layer/") else 1
            scaled_records: list[dict[str, Any]] = []
            for record in records:
                opcode = str(record["opcode"])
                args = tuple(str(arg) for arg in record.get("args", ()))
                count = int(record["count"])
                one_layer_variants[(opcode, args)] += count
                total_variants[(opcode, args)] += count * multiplier
                scaled_records.append(
                    {
                        "opcode": opcode,
                        "args": list(args),
                        "count": count * multiplier,
                    }
                )
            scaled_stage_variants[str(stage_name)] = scaled_records
        result.metadata["stage_parameterized_timing_variants"] = scaled_stage_variants
        result.metadata["one_layer_parameterized_timing_variants"] = [
            {"opcode": opcode, "args": list(args), "count": count}
            for (opcode, args), count in sorted(one_layer_variants.items())
        ]
        result.metadata["parameterized_timing_variants"] = [
            {"opcode": opcode, "args": list(args), "count": count}
            for (opcode, args), count in sorted(total_variants.items())
        ]
    for stage_name, stage in one_layer.stages.items():
        multiplier = num_layers if stage_name.startswith("layer/") else 1
        target = result.stages[stage_name]
        for opcode, count in stage.static_opcodes.items():
            target.static_opcodes[opcode] += count * multiplier
            result.static_opcodes[opcode] += count * multiplier
        for opcode, count in stage.dynamic_opcodes.items():
            target.dynamic_opcodes[opcode] += count * multiplier
            result.dynamic_opcodes[opcode] += count * multiplier
    for stream_index, event in enumerate(one_layer.memory_events):
        multiplier = num_layers if event.stage.startswith("layer/") else 1
        layer_axis = ()
        if multiplier != 1:
            address_delta = layer_hbm_stride if event.transfer.precision in {"matrix", "weight"} else 0
            layer_axis = (
                RepeatAxis(
                    "decoder_layer",
                    num_layers,
                    element_base_delta=address_delta,
                    scale_base_delta=address_delta,
                    logical_element_delta=address_delta,
                    logical_scale_delta=address_delta,
                ),
            )
        result.memory_events.append(
            MemoryEvent(
                stage=event.stage,
                transfer=event.transfer,
                multiplicity=event.multiplicity * multiplier,
                enclosing_axes=event.enclosing_axes + layer_axis,
                stream_index=stream_index,
                parallel_kernel=event.parallel_kernel,
            )
        )
    for action in one_layer.energy_actions:
        multiplier = num_layers if action.stage.startswith("layer/") else 1
        scaled = EnergyAction(
            stage=action.stage,
            component=action.component,
            action=action.action,
            count=action.count * multiplier,
            precision=action.precision,
            active_lanes=action.active_lanes,
            total_lanes=action.total_lanes,
            active_bits=action.active_bits,
            busy_cycles=action.busy_cycles * multiplier,
            bytes=action.bytes * multiplier,
            variant=action.variant,
            segment_log2=action.segment_log2,
            segment_count=action.segment_count,
            activity_fidelity=action.activity_fidelity,
            parallel_kernel=action.parallel_kernel,
        )
        result.energy_actions.append(scaled)
        result.stages[action.stage].energy_actions.append(scaled)
    for entry in one_layer.parallel_kernel_census:
        multiplier = num_layers if entry.stage.startswith("layer/") else 1
        result.parallel_kernel_census.append(
            ParallelKernelCensusEntry(
                stage=entry.stage,
                kernel=entry.kernel,
                opcode=entry.opcode,
                count=entry.count * multiplier,
                tp_semantics=entry.tp_semantics,
                cp_semantics=entry.cp_semantics,
                ep_semantics=entry.ep_semantics,
                logical_rows=entry.logical_rows,
                logical_m=entry.logical_m,
                logical_n=entry.logical_n,
                logical_k=entry.logical_k,
                matrix_mlen=entry.matrix_mlen,
                matrix_blen=entry.matrix_blen,
                multiplicity=entry.multiplicity * multiplier,
                fidelity=entry.fidelity,
            )
        )
    result.schedule, schedule_error = _scale_schedule(one_layer, num_layers)
    result.schedule_unavailable_reasons.update(one_layer.schedule_unavailable_reasons)
    if schedule_error is not None:
        result.schedule_unavailable_reasons[schedule_error] += 1
    return result


_ATTENTION_PAIR_OPS = frozenset(
    {
        "M_BTMM",
        "M_BMM_WO",
        "V_EXP_V",
        "V_RED_MAX_SEG",
        "V_RED_SUM_SEG",
        "V_RED_MAX_SEG_OVR",
        "V_RED_SUM_SEG_OVR",
        "V_SHIFT_V",
        "V_SUB_VF",
    }
)
_SEGMENTED_NORM_OPS = frozenset(
    {
        "V_RED_SUM_SEGS",
        "V_RED_MAX_SEGS",
        "V_MUL_VSEG",
        "V_STAT_MUL_F",
        "V_STAT_ADD_F",
        "V_STAT_RSQRT",
        "S_LD_VLANE_FP",
        "S_ST_VLANE_FP",
    }
)
_FFN_TENSOR_OPS = frozenset(
    {
        "V_EXP_V",
        "V_MUL_VV",
        "V_MUL_VF",
        "V_ADD_VF",
        "V_SUB_VF",
        "V_RECI_V",
    }
)


def _parallel_semantics(stage: str, opcode: str) -> tuple[str, str, str, str]:
    """Classify one emitted opcode without changing the executable schedule."""

    if stage.startswith("global/"):
        if stage == "global/mask_load":
            return (
                "attention_mask",
                "attention_head_pair_sharded",
                "causal_block_partitioned",
                "none",
            )
        if stage in {"global/input_load", "global/final_norm"}:
            return (
                "token_state",
                "token_replicated_hidden",
                "token_partitioned",
                "none",
            )
        return ("global_setup", "replicated_setup", "replicated", "none")
    if stage.startswith("layer/moe/experts"):
        return (
            "moe_expert_ffn",
            "expert_tensor_sharded",
            "token_partitioned",
            "expert_partitioned",
        )
    if stage.startswith("layer/moe/router"):
        if opcode in {"M_MM", "M_MM_WO", "M_MV", "M_MV_WO"}:
            return (
                "moe_router_projection",
                "row_parallel_projection",
                "token_partitioned",
                "router_replicated",
            )
        return (
            "moe_router_postprocess",
            "token_replicated_hidden",
            "token_partitioned",
            "router_replicated",
        )
    if stage.startswith("layer/moe/dispatch"):
        return (
            "moe_dispatch",
            "token_replicated_hidden",
            "token_partitioned",
            "expert_dispatch",
        )
    if stage.startswith("layer/moe/combine"):
        return (
            "moe_combine",
            "token_replicated_hidden",
            "token_partitioned",
            "expert_combine",
        )
    if stage.startswith("layer/moe/norm"):
        return (
            "moe_norm",
            "token_replicated_hidden",
            "token_partitioned",
            "none",
        )
    if stage == "layer/ffn":
        if opcode in {"M_MM", "M_MM_WO"}:
            return (
                "dense_ffn_projection",
                "ffn_projection_tiled",
                "token_partitioned",
                "none",
            )
        if opcode in _FFN_TENSOR_OPS:
            return (
                "dense_ffn_activation",
                "token_tensor_sharded",
                "token_partitioned",
                "none",
            )
        return (
            "dense_ffn_projection_control",
            "ffn_projection_tiled",
            "token_partitioned",
            "none",
        )
    if stage == "layer/attention":
        if opcode in _ATTENTION_PAIR_OPS:
            return (
                "attention_core",
                "attention_head_pair_sharded",
                "causal_block_partitioned",
                "none",
            )
        if opcode in _SEGMENTED_NORM_OPS:
            return (
                "attention_segmented_norm",
                "token_tensor_sharded",
                "token_partitioned",
                "none",
            )
        if opcode in {"M_MM", "M_MM_WO", "M_MV", "M_MV_WO"}:
            return (
                "attention_projection",
                "attention_projection_tiled",
                "token_partitioned",
                "none",
            )
        return (
            "attention_token_path",
            "token_replicated_hidden",
            "token_partitioned",
            "none",
        )
    raise ValueError(
        f"parallel kernel census has no semantic classification for "
        f"{stage}/{opcode}"
    )


def _build_parallel_kernel_census(
    trace: CostTrace,
    *,
    model: ModelConfig,
    hardware: CompilerCostHardware,
    layout: NativeDecoderCostLayout,
) -> list[ParallelKernelCensusEntry]:
    entries: list[ParallelKernelCensusEntry] = []
    logical_rows = layout.sequence_packing.logical_active_rows
    for raw in trace.parallel_kernel_census:
        if raw.count <= 0:
            continue
        if raw.kernel != UNCLASSIFIED_PARALLEL_KERNEL:
            entries.append(raw)
            continue
        if raw.stage.startswith("layer/"):
            raise ValueError(
                "parallel kernel lineage was lost inside a layer; refusing "
                f"stage-level fallback for {raw.stage}/{raw.opcode} "
                f"(count={raw.count})"
            )
        kernel, tp_semantics, cp_semantics, ep_semantics = (
            _parallel_semantics(raw.stage, raw.opcode)
        )
        logical_n = 0
        logical_k = 0
        if kernel in {"attention_projection", "moe_router_projection"}:
            logical_n = (
                int(model.num_experts)
                if kernel == "moe_router_projection"
                else int(layout.head_packing.total_q_dim)
            )
            logical_k = int(model.hidden_size)
        entries.append(
            replace(
                raw,
                kernel=kernel,
                tp_semantics=tp_semantics,
                cp_semantics=cp_semantics,
                ep_semantics=ep_semantics,
                logical_rows=logical_rows,
                logical_m=logical_rows,
                logical_n=logical_n,
                logical_k=logical_k,
                matrix_mlen=hardware.mlen,
                matrix_blen=hardware.blen,
                fidelity="compiler_global_semantic_explicit_v3",
            )
        )

    covered: Counter[tuple[str, str]] = Counter(
        {
            (entry.stage, entry.opcode): 0
            for entry in entries
        }
    )
    for entry in entries:
        covered[(entry.stage, entry.opcode)] += int(entry.count)
    expected = Counter(
        {
            (stage_name, opcode): int(count)
            for stage_name, stage in trace.stages.items()
            for opcode, count in stage.dynamic_opcodes.items()
            if not opcode.startswith("H_") and int(count)
        }
    )
    if covered != expected:
        mismatches = {
            f"{stage}/{opcode}": {
                "expected": expected[(stage, opcode)],
                "lineage": covered[(stage, opcode)],
            }
            for stage, opcode in sorted(set(expected) | set(covered))
            if expected[(stage, opcode)] != covered[(stage, opcode)]
        }
        raise ValueError(
            "parallel kernel lineage does not exactly cover final dynamic "
            f"compute opcodes: {mismatches}"
        )

    merged: dict[tuple[Any, ...], ParallelKernelCensusEntry] = {}
    for entry in entries:
        key = (
            entry.stage,
            entry.kernel,
            entry.opcode,
            entry.tp_semantics,
            entry.cp_semantics,
            entry.ep_semantics,
            entry.logical_rows,
            entry.logical_m,
            entry.logical_n,
            entry.logical_k,
            entry.matrix_mlen,
            entry.matrix_blen,
            entry.fidelity,
        )
        previous = merged.get(key)
        merged[key] = (
            entry
            if previous is None
            else ParallelKernelCensusEntry(
                stage=entry.stage,
                kernel=entry.kernel,
                opcode=entry.opcode,
                count=previous.count + entry.count,
                tp_semantics=entry.tp_semantics,
                cp_semantics=entry.cp_semantics,
                ep_semantics=entry.ep_semantics,
                logical_rows=entry.logical_rows,
                logical_m=entry.logical_m,
                logical_n=entry.logical_n,
                logical_k=entry.logical_k,
                matrix_mlen=entry.matrix_mlen,
                matrix_blen=entry.matrix_blen,
                multiplicity=entry.multiplicity,
                fidelity=entry.fidelity,
            )
        )
    return list(merged.values())


def _audit_memory_events(trace: CostTrace) -> dict[str, Any]:
    """Require exact, ordered DMA coverage for every dynamic HBM opcode."""
    accounted: Counter[tuple[str, str]] = Counter()
    lineage_accounted = 0
    lineage_missing: Counter[tuple[str, str]] = Counter()
    for event in trace.memory_events:
        accounted[(event.stage, event.transfer.opcode)] += event.multiplicity
        if event.parallel_kernel is None:
            lineage_missing[(event.stage, event.transfer.opcode)] += (
                event.multiplicity
            )
        else:
            lineage_accounted += event.multiplicity
        if event.transfer.geometry_fidelity != "exact":
            raise ValueError(
                f"non-exact DMA geometry in {event.stage}/{event.transfer.opcode}: "
                f"{event.transfer.source or 'unknown source'}"
            )
        if event.multiplicity > 1 and not event.enclosing_axes:
            raise ValueError(
                f"compressed DMA stream lacks repeat axes in "
                f"{event.stage}/{event.transfer.opcode}: multiplicity={event.multiplicity}"
            )
    expected: Counter[tuple[str, str]] = Counter()
    for stage_name, stage in trace.stages.items():
        for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"):
            expected[(stage_name, opcode)] = stage.dynamic_opcodes.get(opcode, 0)
    mismatches = {
        f"{stage}/{opcode}": {"expected": count, "recorded": accounted[(stage, opcode)]}
        for (stage, opcode), count in expected.items()
        if count != accounted[(stage, opcode)]
    }
    if mismatches:
        raise ValueError(f"exact DMA coverage mismatch: {json.dumps(mismatches, sort_keys=True)}")
    missing_layer = {
        f"{stage}/{opcode}": count
        for (stage, opcode), count in lineage_missing.items()
        if stage.startswith("layer/") and count
    }
    if missing_layer:
        raise ValueError(
            "layer DMA event lost parallel-kernel lineage: "
            f"{missing_layer}"
        )
    stream_indices = [event.stream_index for event in trace.memory_events]
    if stream_indices != list(range(len(stream_indices))):
        raise ValueError("DMA stream indices are not contiguous and emission ordered")
    total = sum(accounted.values())
    layer_total = sum(
        count
        for (stage, _opcode), count in accounted.items()
        if stage.startswith("layer/")
    )
    layer_missing = sum(
        count
        for (stage, _opcode), count in lineage_missing.items()
        if stage.startswith("layer/")
    )
    return {
        "geometry_fidelity": "exact",
        "stream_count": len(trace.memory_events),
        "parallel_kernel_lineage_coverage": (
            1.0 if total == 0 else lineage_accounted / total
        ),
        "layer_parallel_kernel_lineage_coverage": (
            1.0
            if layer_total == 0
            else (layer_total - layer_missing) / layer_total
        ),
        "layer_dynamic_occurrences": layer_total,
        "global_unclassified_occurrences": sum(
            count
            for (stage, _opcode), count in lineage_missing.items()
            if not stage.startswith("layer/")
        ),
        "dynamic_opcodes": {
            opcode: sum(event.multiplicity for event in trace.memory_events if event.transfer.opcode == opcode)
            for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")
        },
    }


def _finalize_energy_action_lineage(trace: CostTrace) -> dict[str, Any]:
    """Join final census ownership to every structural energy-action family."""

    lineage_by_stage_opcode: dict[tuple[str, str], set[str]] = defaultdict(set)
    expected: Counter[tuple[str, str, str, str, str]] = Counter()
    for entry in trace.parallel_kernel_census:
        lineage = parallel_kernel_lineage_id(entry.to_dict())
        lineage_by_stage_opcode[(entry.stage, entry.opcode)].add(lineage)
        for component, action, opcode, _segment_log2 in (
            _logic_energy_actions(entry.opcode)
        ):
            expected[
                (entry.stage, lineage, opcode, component, action)
            ] += entry.count
        for component, action, accesses in _sram_actions(entry.opcode):
            expected[
                (entry.stage, lineage, entry.opcode, component, action)
            ] += entry.count * accesses

    remapped: list[EnergyAction] = []
    for action in trace.energy_actions:
        lineage = action.parallel_kernel
        if lineage == UNCLASSIFIED_PARALLEL_KERNEL:
            candidates = lineage_by_stage_opcode.get(
                (action.stage, action.precision), set()
            )
            if len(candidates) == 1:
                lineage = next(iter(candidates))
        if (
            action.stage.startswith("layer/")
            and action.component != "hbm_controller"
            and lineage == UNCLASSIFIED_PARALLEL_KERNEL
        ):
            raise ValueError(
                "layer EnergyAction lost parallel-kernel lineage: "
                f"{action.stage}/{action.component}/{action.action}/"
                f"{action.precision}"
            )
        remapped.append(replace(action, parallel_kernel=lineage))
    trace.energy_actions = remapped
    for stage in trace.stages.values():
        stage.energy_actions.clear()
    for action in trace.energy_actions:
        trace.stages[action.stage].energy_actions.append(action)

    action_families: Counter[tuple[str, str, str, str, str]] = Counter()
    for action in trace.energy_actions:
        if action.component == "hbm_controller":
            continue
        if action.precision.startswith("implicit_"):
            continue
        action_families[
            (
                action.stage,
                action.parallel_kernel,
                action.precision,
                action.component,
                action.action,
            )
        ] += action.count
    mismatches = {
        f"{stage}/{lineage}/{opcode}/{component}/{action}": {
            "expected": count,
            "energy_action": action_families.get(
                (stage, lineage, opcode, component, action), 0
            ),
        }
        for (
            stage,
            lineage,
            opcode,
            component,
            action,
        ), count in expected.items()
        if action_families.get(
            (stage, lineage, opcode, component, action), 0
        )
        != count
    }
    if mismatches:
        raise ValueError(
            "EnergyAction structural-family coverage differs from final schedule "
            f"census: {mismatches}"
        )
    return {
        "schema": "energy_action_kernel_lineage_v3_structural_families",
        "coverage": 1.0,
        "expected_family_count": len(expected),
        "action_family_count": len(action_families),
    }


def clear_cost_trace_cache() -> None:
    """Clear the small in-process cache used by DSE warm evaluations."""
    _ONE_LAYER_TRACE_CACHE.clear()


def _load_moe_routing_plan(
    value: MoeRoutingPlan | Mapping[str, Any] | str | Path | None,
) -> MoeRoutingPlan | None:
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        with Path(value).open() as handle:
            value = json.load(handle)
    if isinstance(value, Mapping):
        value = dict(value)
    return coerce_routing_plan(value)


def _active_physical_rows(*, batch_size: int, seq_len: int, rows_per_batch: int) -> tuple[int, ...]:
    return tuple(
        batch_idx * rows_per_batch + token_idx for batch_idx in range(batch_size) for token_idx in range(seq_len)
    )


def _trace_cache_key(
    model: ModelConfig,
    hardware: CompilerCostHardware,
    *,
    seq_len: int,
    batch_size: int,
    layer_idx: int,
    routing_plan_hash: str | None,
    moe_routing_mode: str,
    moe_lowering_schedule: str,
    moe_layer_scaling: str,
    native_layout_mode: str,
    packed_attention_schedule: str,
    softmax_state_schedule: str,
    packed_qk_schedule: str,
    vector_scalar_schedule: str,
    selector_schedule: str,
    reduction_output_mode: str,
    gqa_pipeline_schedule: str,
    gqa_timing_calibration: str | Path | None,
    address_generation_mode: str,
    ffn_address_schedule: str,
    ffn_projection_schedule: str,
    cost_trace_granularity: str,
) -> tuple[Any, ...]:
    return (
        model,
        hardware,
        seq_len,
        batch_size,
        layer_idx,
        routing_plan_hash,
        moe_routing_mode,
        moe_lowering_schedule,
        moe_layer_scaling,
        NATIVE_LAYOUT_SCHEMA_VERSION,
        native_layout_mode,
        packed_attention_schedule,
        softmax_state_schedule,
        packed_qk_schedule,
        vector_scalar_schedule,
        selector_schedule,
        reduction_output_mode,
        gqa_pipeline_schedule,
        str(gqa_timing_calibration) if gqa_timing_calibration is not None else None,
        address_generation_mode,
        ffn_address_schedule,
        ffn_projection_schedule,
        cost_trace_granularity,
    )


def _cached_one_layer(key: tuple[Any, ...]) -> CostTrace | None:
    cached = _ONE_LAYER_TRACE_CACHE.get(key)
    if cached is None:
        return None
    _ONE_LAYER_TRACE_CACHE.move_to_end(key)
    return cached


def _store_one_layer(key: tuple[Any, ...], trace: CostTrace) -> None:
    _ONE_LAYER_TRACE_CACHE[key] = trace
    _ONE_LAYER_TRACE_CACHE.move_to_end(key)
    while len(_ONE_LAYER_TRACE_CACHE) > _ONE_LAYER_CACHE_LIMIT:
        _ONE_LAYER_TRACE_CACHE.popitem(last=False)


def _config_hash(
    model: ModelConfig,
    hardware: CompilerCostHardware,
    seq_len: int,
    batch_size: int,
    *,
    layer_idx: int,
    routing_plan_hash: str | None,
    moe_routing_mode: str,
    moe_lowering_schedule: str,
    moe_layer_scaling: str,
    native_layout_mode: str,
    packed_attention_schedule: str,
    softmax_state_schedule: str,
    packed_qk_schedule: str,
    vector_scalar_schedule: str,
    selector_schedule: str,
    reduction_output_mode: str,
    gqa_pipeline_schedule: str,
    gqa_timing_artifact_sha256: str | None,
    address_generation_mode: str,
    ffn_address_schedule: str,
    ffn_projection_schedule: str,
    cost_trace_granularity: str,
) -> str:
    payload = {
        "model": asdict(model),
        "hardware": asdict(hardware),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "layer_idx": layer_idx,
        "routing_plan_hash": routing_plan_hash,
        "moe_routing_mode": moe_routing_mode,
        "moe_lowering_schedule": moe_lowering_schedule,
        "moe_layer_scaling": moe_layer_scaling,
        "native_layout_schema_version": NATIVE_LAYOUT_SCHEMA_VERSION,
        "native_layout_mode": native_layout_mode,
        "packed_attention_schedule": packed_attention_schedule,
        "softmax_state_schedule": softmax_state_schedule,
        "packed_qk_schedule": packed_qk_schedule,
        "vector_scalar_schedule": vector_scalar_schedule,
        "selector_schedule": selector_schedule,
        "reduction_output_mode": reduction_output_mode,
        "gqa_pipeline_schedule": gqa_pipeline_schedule,
        "gqa_timing_artifact_sha256": gqa_timing_artifact_sha256,
        "address_generation_mode": address_generation_mode,
        "ffn_address_schedule": ffn_address_schedule,
        "ffn_projection_schedule": ffn_projection_schedule,
        "cost_trace_granularity": cost_trace_granularity,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()[:16]


def compile_native_decoder_cost_trace(
    model_config: ModelConfig | Mapping[str, Any] | str | Path,
    hardware_config: CompilerCostHardware | Mapping[str, Any],
    *,
    seq_len: int,
    batch_size: int = 1,
    num_layers: int | None = None,
    layer_idx: int = 0,
    moe_routing_mode: str = "static-indices",
    moe_lowering_schedule: str = MOE_LOWERING_SCHEDULE_COMPACT_ROUTE_V2,
    moe_routing_plan: MoeRoutingPlan | Mapping[str, Any] | str | Path | None = None,
    max_static_routes: int = 1024,
    moe_layer_scaling: str = "single-layer",
    native_layout_mode: str = "compact",
    packed_attention_schedule: str = "direct-first-block-v1",
    softmax_state_schedule: str = SOFTMAX_STATE_SCHEDULE_STREAMED_V2,
    packed_qk_schedule: str = PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1,
    vector_scalar_schedule: str = "rtl-v3",
    selector_schedule: str = "legacy",
    reduction_output_mode: str = "accumulate-v1",
    gqa_pipeline_schedule: str | None = None,
    gqa_timing_calibration: str | Path | None = None,
    address_generation_mode: str = AGU_MODE_LOOP_V1,
    ffn_address_schedule: str = FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1,
    ffn_projection_schedule: str = FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2,
    cost_trace_granularity: str = COST_TRACE_GRANULARITY_DETAILED,
    use_cache: bool = True,
) -> CostTrace:
    """Run the native Qwen3 schedule using shape-only variables.

    Static-index MoE layers require an explicit routing plan.  Fixed-balanced
    mode is a latency-only aggregate and never materializes token routes.
    """
    model, configured_layers = load_cost_model_config(model_config)
    hardware = (
        hardware_config
        if isinstance(hardware_config, CompilerCostHardware)
        else CompilerCostHardware(**hardware_config)
    )
    hardware.validate()
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError(f"seq_len and batch_size must be positive, got {seq_len}, {batch_size}")
    if layer_idx < 0:
        raise ValueError(f"layer_idx must be nonnegative, got {layer_idx}")
    if configured_layers is not None and layer_idx >= configured_layers:
        raise ValueError(f"layer_idx={layer_idx} outside configured num_layers={configured_layers}")
    if max_static_routes <= 0:
        raise ValueError(f"max_static_routes must be positive, got {max_static_routes}")
    if native_layout_mode not in {"compact", "legacy"}:
        raise ValueError(f"native_layout_mode must be 'compact' or 'legacy', got {native_layout_mode!r}")
    if packed_attention_schedule not in {"direct-first-block-v1", "legacy"}:
        raise ValueError(
            f"packed_attention_schedule must be 'direct-first-block-v1' or 'legacy', got {packed_attention_schedule!r}"
        )
    if softmax_state_schedule not in SOFTMAX_STATE_SCHEDULES:
        raise ValueError(
            f"softmax_state_schedule must be one of {sorted(SOFTMAX_STATE_SCHEDULES)}, got {softmax_state_schedule!r}"
        )
    if packed_qk_schedule not in PACKED_QK_SCHEDULES:
        raise ValueError(f"packed_qk_schedule must be one of {sorted(PACKED_QK_SCHEDULES)}, got {packed_qk_schedule!r}")
    if vector_scalar_schedule not in {
        "rtl-v4",
        "rtl-v3",
        "rtl-v2",
        "compiler-v1",
        "legacy",
    }:
        raise ValueError(
            "vector_scalar_schedule must be 'rtl-v4', 'rtl-v3', 'rtl-v2', "
            "'compiler-v1', or 'legacy', got "
            f"{vector_scalar_schedule!r}"
        )
    if selector_schedule not in {"hoisted-v1", "legacy"}:
        raise ValueError(f"selector_schedule must be 'hoisted-v1' or 'legacy', got {selector_schedule!r}")
    if reduction_output_mode not in {"overwrite-v1", "accumulate-v1"}:
        raise ValueError(
            f"reduction_output_mode must be 'overwrite-v1' or 'accumulate-v1', got {reduction_output_mode!r}"
        )
    if address_generation_mode not in {
        AGU_MODE_LEGACY,
        AGU_MODE_LOOP_V1,
    }:
        raise ValueError(
            "address_generation_mode must be 'loop-agu-v1' or 'legacy', got "
            f"{address_generation_mode!r}"
        )
    if ffn_address_schedule not in FFN_ADDRESS_SCHEDULES:
        raise ValueError(f"ffn_address_schedule must be one of {FFN_ADDRESS_SCHEDULES}, got {ffn_address_schedule!r}")
    if ffn_projection_schedule not in FFN_PROJECTION_SCHEDULES:
        raise ValueError(
            f"ffn_projection_schedule must be one of {FFN_PROJECTION_SCHEDULES}, got {ffn_projection_schedule!r}"
        )
    if cost_trace_granularity not in COST_TRACE_GRANULARITIES:
        raise ValueError(
            f"cost_trace_granularity must be one of {sorted(COST_TRACE_GRANULARITIES)}, got {cost_trace_granularity!r}"
        )
    if gqa_pipeline_schedule is None:
        gqa_pipeline_schedule = "row-interleaved-v1" if vector_scalar_schedule in {"rtl-v3", "rtl-v4"} else "row-serial"
    if gqa_pipeline_schedule not in {"row-interleaved-v1", "row-serial"}:
        raise ValueError(
            f"gqa_pipeline_schedule must be 'row-interleaved-v1' or 'row-serial', got {gqa_pipeline_schedule!r}"
        )
    if gqa_pipeline_schedule == "row-interleaved-v1" and vector_scalar_schedule not in {"rtl-v3", "rtl-v4"}:
        raise ValueError("row-interleaved-v1 requires vector_scalar_schedule='rtl-v3' or 'rtl-v4'")
    if moe_routing_mode not in {"static-indices", "fixed-balanced"}:
        raise ValueError(f"moe_routing_mode must be 'static-indices' or 'fixed-balanced', got {moe_routing_mode!r}")
    if moe_lowering_schedule not in MOE_LOWERING_SCHEDULES:
        raise ValueError(
            f"moe_lowering_schedule must be one of {sorted(MOE_LOWERING_SCHEDULES)}, got {moe_lowering_schedule!r}"
        )
    if moe_layer_scaling not in {
        "single-layer",
        "repeat-static-plan",
        "repeat-fixed-balanced",
    }:
        raise ValueError(f"unsupported moe_layer_scaling={moe_layer_scaling!r}")

    is_moe_model = model.num_experts > 0
    is_moe_layer = model.is_moe_layer(layer_idx)
    plan = _load_moe_routing_plan(moe_routing_plan)
    summary: FixedBalancedRoutingSummary | None = None
    if is_moe_layer:
        if model.num_experts <= 0 or model.experts_per_token <= 0:
            raise ValueError("MoE model config must define positive num_experts and num_experts_per_tok")
        if model.num_experts > hardware.mlen:
            raise ValueError(f"num_experts={model.num_experts} exceeds MLEN={hardware.mlen}")
        if moe_routing_mode == "static-indices":
            if plan is None:
                raise ValueError("static-index MoE CostEmitter requires an explicit moe_routing_plan")
            if plan.num_experts != model.num_experts:
                raise ValueError(
                    f"Routing plan num_experts={plan.num_experts} does not match model num_experts={model.num_experts}"
                )
            if plan.experts_per_token != model.experts_per_token:
                raise ValueError(
                    f"Routing plan top-k={plan.experts_per_token} does not match model top-k={model.experts_per_token}"
                )
            routing_sequence_packing = SequencePackingPlan.build(
                batch_size=batch_size,
                seq_len=seq_len,
                mlen=hardware.mlen,
                mode=native_layout_mode,
            )
            plan.validate(
                active_physical_rows=routing_sequence_packing.active_physical_rows(),
                max_routes=max_static_routes,
            )
        else:
            if plan is not None:
                raise ValueError("fixed-balanced routing cannot be combined with moe_routing_plan")
            summary = FixedBalancedRoutingSummary.build(
                num_tokens=seq_len * batch_size,
                num_experts=model.num_experts,
                experts_per_token=model.experts_per_token,
            )
        if num_layers is None:
            num_layers = 1
        expected_scaling = "repeat-static-plan" if moe_routing_mode == "static-indices" else "repeat-fixed-balanced"
        if num_layers > 1 and moe_layer_scaling != expected_scaling:
            raise ValueError(
                f"num_layers > 1 for {moe_routing_mode} MoE requires moe_layer_scaling={expected_scaling!r}"
            )
        if num_layers == 1 and moe_layer_scaling != "single-layer":
            raise ValueError("single-layer MoE costing requires moe_layer_scaling='single-layer'")
    else:
        if plan is not None:
            raise ValueError(f"layer_idx={layer_idx} is dense; moe_routing_plan is not applicable")
        if moe_routing_mode != "static-indices":
            raise ValueError("fixed-balanced routing is only valid for a MoE layer")
        if moe_layer_scaling != "single-layer":
            raise ValueError("moe_layer_scaling is only applicable to a selected MoE layer")
        if num_layers is None:
            num_layers = 1 if is_moe_model else configured_layers or 1
        if is_moe_model and num_layers > 1:
            raise ValueError(
                "hybrid Qwen3-MoE dense-layer costing models one selected layer; "
                "compile dense and MoE layer classes separately"
            )
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")

    routing_plan_hash = (
        plan.routing_plan_hash if plan is not None else summary.routing_summary_hash if summary is not None else None
    )
    cache_key = _trace_cache_key(
        model,
        hardware,
        seq_len=seq_len,
        batch_size=batch_size,
        layer_idx=layer_idx,
        routing_plan_hash=routing_plan_hash,
        moe_routing_mode=moe_routing_mode,
        moe_lowering_schedule=moe_lowering_schedule,
        moe_layer_scaling=moe_layer_scaling,
        native_layout_mode=native_layout_mode,
        packed_attention_schedule=packed_attention_schedule,
        softmax_state_schedule=softmax_state_schedule,
        packed_qk_schedule=packed_qk_schedule,
        vector_scalar_schedule=vector_scalar_schedule,
        selector_schedule=selector_schedule,
        reduction_output_mode=reduction_output_mode,
        gqa_pipeline_schedule=gqa_pipeline_schedule,
        gqa_timing_calibration=gqa_timing_calibration,
        address_generation_mode=address_generation_mode,
        ffn_address_schedule=ffn_address_schedule,
        ffn_projection_schedule=ffn_projection_schedule,
        cost_trace_granularity=cost_trace_granularity,
    )
    one_layer = _cached_one_layer(cache_key) if use_cache else None
    if one_layer is not None:
        full_trace = _scale_trace(
            one_layer,
            num_layers,
            layer_hbm_stride=int(one_layer.metadata.get("layer_hbm_stride", 0)),
        )
        full_trace.metadata["cost_cache_hit"] = True
        full_trace.metadata["layer_scaling_fidelity"] = (
            "approximate_repeated_balanced_routing"
            if summary is not None and num_layers > 1
            else "approximate_repeated_static_plan"
            if plan is not None and num_layers > 1
            else "fixed_balanced_histogram"
            if summary is not None
            else "exact_static_indices"
            if plan is not None
            else "shape_equivalent_dense_repeat"
        )
        return full_trace

    layout = _build_layout(
        model,
        hardware,
        seq_len=seq_len,
        batch_size=batch_size,
        layer_idx=layer_idx,
        native_layout_mode=native_layout_mode,
    )
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)
    prog = PlenaCompiler(
        mlen=hardware.mlen,
        blen=hardware.blen,
        mram_tile_capacity=hardware.mram_tile_capacity,
        hbm_m_prefetch_amount=hardware.hbm_m_prefetch_amount,
        hbm_v_prefetch_amount=hardware.hbm_v_prefetch_amount,
        hbm_v_writeback_amount=hardware.hbm_v_writeback_amount,
        emission_mode="cost",
        cost_strict_raw=True,
        cost_trace_granularity=cost_trace_granularity,
        cost_address_generation_mode=address_generation_mode,
        packed_attention_schedule=packed_attention_schedule,
        softmax_state_schedule=softmax_state_schedule,
        packed_qk_schedule=packed_qk_schedule,
        vector_scalar_schedule=vector_scalar_schedule,
        selector_schedule=selector_schedule,
        reduction_output_mode=reduction_output_mode,
        gqa_pipeline_schedule=gqa_pipeline_schedule,
        gqa_timing_calibration=gqa_timing_calibration,
        address_generation_mode=AGU_MODE_LEGACY,
        ffn_address_schedule=ffn_address_schedule,
        ffn_projection_schedule=ffn_projection_schedule,
        moe_lowering_schedule=moe_lowering_schedule,
        fp_sram_depth=hardware.fp_sram_depth,
        fp_constant_num=hardware.fp_constant_num,
        kv_residency_policy=hardware.kv_residency_policy,
    )
    prog._native_active_row_ranges = layout.sequence_packing.active_row_ranges()
    # Attention lowering needs the same slot geometry as the native compiler
    # to select operand-sensitive segment reductions. Keeping only active row
    # ranges here caused CostEmitter to silently fall back to full-VLEN trees.
    prog._native_sequence_packing = layout.sequence_packing
    prog.hlen = hardware.hlen
    prog.broadcast_amount = layout.head_packing.hardware_broadcast_amount
    softmax_layout = build_softmax_state_layout(
        mlen=hardware.mlen,
        active_broadcast_heads=layout.head_packing.broadcast_amount,
        schedule=softmax_state_schedule,
        fp_constant_num=hardware.fp_constant_num,
    )
    if hardware.fp_sram_depth is not None and hardware.fp_sram_depth < softmax_layout.required_depth:
        raise ValueError(
            f"FP_SRAM_DEPTH={hardware.fp_sram_depth} is smaller than required {softmax_layout.required_depth}"
        )

    sequence_shape = (layout.compile_seq_rows, layout.padded_hidden)
    rope_shape = (layout.compile_seq_rows, hardware.mlen)
    x_input = prog.input("X", sequence_shape, physical_shape=sequence_shape)
    pos_input = prog.input("POS", sequence_shape, physical_shape=sequence_shape)
    r_input = prog.input("R_rope", (hardware.mlen, hardware.mlen))
    cos_input = prog.input("COS", rope_shape, physical_shape=rope_shape)
    sin_input = prog.input("SIN", rope_shape, physical_shape=rope_shape)
    with prog.cost_stage("global/rope_load"):
        cos = prog.load_batch(cos_input, name="COS")
        sin = prog.load_batch(sin_input, name="SIN")
    causal_input = prog.input("causal_mask", (hardware.mlen, hardware.mlen))
    with prog.cost_stage("global/mask_load"):
        causal = prog.load_batch(causal_input, name="CAUSAL_MASK")

    router_mask = None
    route_identity = None
    if is_moe_layer and (
        moe_lowering_schedule == MOE_LOWERING_SCHEDULE_LEGACY_STATIC_V1 or model.experts_per_token > 8
    ):
        router_mask_input = prog.input(
            "MOE_ROUTER_MASK",
            (layout.compile_seq_rows, hardware.mlen),
            physical_shape=(layout.compile_seq_rows, hardware.mlen),
        )
        route_identity_input = prog.input("MOE_ROUTE_IDENTITY", (hardware.mlen, hardware.mlen))
        with prog.cost_stage("global/moe_setup"):
            router_mask = prog.load_batch(router_mask_input, name="MOE_ROUTER_MASK")
            route_identity = prog.load_batch(route_identity_input, name="MOE_ROUTE_IDENTITY")

    layer_hbm_start = prog._next_hbm_addr
    if is_moe_layer:
        routing = plan if plan is not None else summary
        assert routing is not None
        layer_inputs = _register_shape_moe_layer_inputs(prog, model, layout, routing, layer_idx=layer_idx)
    else:
        layer_inputs = _register_shape_layer_inputs(prog, model, layout, layer_idx=layer_idx)
    layer_hbm_stride = prog._next_hbm_addr - layer_hbm_start
    with prog.cost_stage("global/input_load"):
        current = prog.load_batch(x_input, name="X")
        pos = prog.load_batch(pos_input, name="POS")
        ops.embedding_add(prog, current, pos)

    scratch = prog.alloc(
        "residual_scratch",
        layout.compile_seq_rows,
        layout.padded_hidden,
        strict=False,
        physical_shape=sequence_shape,
    )
    scale = 1.0 / math.sqrt(model.head_dim)
    with prog.cost_stage("layer/attention"):
        current = _emit_packed_attention_block(
            prog,
            current,
            layer_inputs,
            (r_input, cos, sin),
            causal,
            scratch,
            scale,
            layer_idx,
            layout.padded_seq_len,
            model.head_dim,
            model.num_kv_heads,
            model.head_ratio,
            layout.head_packing,
            model,
            batch_size=layout.sequence_packing.attention_group_count,
            rows_per_batch=layout.rows_per_batch,
            active_seq_len_per_batch=(layout.sequence_packing.attention_group_seq_len),
            active_seq_len=batch_size * seq_len,
            active_hidden=model.hidden_size,
        )
    moe_bucket_metadata: dict[int, dict[str, int]] = {}
    if is_moe_layer:
        assert isinstance(layer_inputs, MoeLayerInputVars)
        with prog.cost_stage("layer/moe"):
            if summary is not None:
                current, moe_bucket_metadata = _emit_fixed_balanced_moe_block(
                    prog,
                    current,
                    layer_inputs,
                    scratch,
                    router_mask=router_mask,
                    route_identity=route_identity,
                    summary=summary,
                    model_cfg=model,
                    layer_idx=layer_idx,
                )
            else:
                assert plan is not None
                current, moe_bucket_metadata = _emit_moe_block(
                    prog,
                    current,
                    layer_inputs,
                    scratch,
                    router_mask=router_mask,
                    route_identity=route_identity,
                    plan=plan,
                    model_cfg=model,
                    layer_idx=layer_idx,
                    active_seq_len=batch_size * seq_len,
                    active_hidden=model.hidden_size,
                )
    else:
        assert isinstance(layer_inputs, LayerInputVars)
        with prog.cost_stage("layer/ffn"):
            current = _emit_ffn_block(
                prog,
                current,
                layer_inputs,
                scratch,
                layer_idx=layer_idx,
                active_seq_len=batch_size * seq_len,
                active_hidden=model.hidden_size,
            )
    with prog.cost_stage("global/final_norm"):
        ops.rms_norm(
            prog,
            current,
            eps_offset=3,
            reci_hid_offset=4,
            physical_rows=current.physical_shape[0],
            active_row_ranges=layout.sequence_packing.active_row_ranges(),
        )
        final_norm_input = prog.input(
            "W_final_norm",
            sequence_shape,
            physical_shape=sequence_shape,
        )
        final_norm = prog.load_batch(final_norm_input, name="W_final_norm_load")
        prog.vram_mul(current, final_norm)
        prog.free_tensor(final_norm)

    one_layer = prog.compile_cost_trace()
    if cost_trace_granularity == COST_TRACE_GRANULARITY_DETAILED:
        one_layer = optimize_cost_trace_loop_agu(
            one_layer,
            mode=address_generation_mode,
        )
    ffn_address_optimization = prog.ffn_address_stats()
    ffn_stages = tuple(stage for name, stage in one_layer.stages.items() if name in {"layer/ffn", "layer/moe/experts"})
    if ffn_stages:
        address_control_opcodes = {
            "S_ADDI_INT",
            "S_ADD_INT",
            "S_LUI_INT",
            "C_LOOP_START",
            "C_LOOP_END",
            "C_AGU_CONFIG",
            "C_LOOP_START_AGU",
        }
        ffn_address_optimization["ffn_address_cycles_after"] = sum(
            stage.dynamic_opcodes.get(opcode, 0) for stage in ffn_stages for opcode in address_control_opcodes
        )
        ffn_address_optimization["ffn_residual_address_opcodes"] = sum(
            stage.dynamic_opcodes.get("S_ADDI_INT", 0) + stage.dynamic_opcodes.get("S_ADD_INT", 0)
            for stage in ffn_stages
        )
        ffn_address_optimization["ffn_post_agu_census_source"] = (
            "+".join(name for name in ("layer/ffn", "layer/moe/experts") if name in one_layer.stages) + "_dynamic_trace"
        )
    ratio = model.num_heads // model.num_kv_heads
    physical_broadcast = layout.head_packing.broadcast_amount
    full_chunks, tail_heads = divmod(ratio, physical_broadcast)
    q_blocks = math.ceil(layout.sequence_packing.attention_group_seq_len / hardware.mlen)
    expanded_block_pairs = q_blocks * (q_blocks + 1) // 2
    resident_kv_tiles = 2 * q_blocks
    kv_residency = plan_kv_residency(
        k_blocks=q_blocks,
        mlen=hardware.mlen,
        matrix_sram_tiles=hardware.mram_tile_capacity,
        requested_residency_fraction=(0.0 if hardware.kv_residency_policy == "streaming" else None),
        policy=hardware.kv_residency_policy,
        force_streaming=hardware.kv_residency_policy == "streaming",
    )
    active_inter_dim = (
        model.moe_inter_dim or model.inter_dim if is_moe_layer else model.dense_inter_dim or model.inter_dim
    )
    routing = plan if plan is not None else summary
    routes_per_expert = (
        {} if routing is None else {str(expert_id): count for expert_id, count in routing.routes_per_expert.items()}
    )
    serialized_bucket_metadata = {str(expert_id): values for expert_id, values in moe_bucket_metadata.items()}
    one_layer.metadata.update(
        {
            "workload": {
                "model_type": model.model_type,
                "hidden_size": model.hidden_size,
                "inter_dim": active_inter_dim,
                "dense_inter_dim": model.dense_inter_dim,
                "moe_inter_dim": model.moe_inter_dim,
                "num_experts": model.num_experts,
                "experts_per_token": model.experts_per_token,
                "num_heads": model.num_heads,
                "num_kv_heads": model.num_kv_heads,
                "head_dim": model.head_dim,
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
            "hardware": asdict(hardware),
            "attention_schedule": {
                "kind": "logical_kv_group",
                "active_head_dim": model.head_dim,
                "head_slot_dim": layout.head_packing.head_slot_dim,
                "logical_broadcast": hardware.broadcast_amount,
                "physical_broadcast": physical_broadcast,
                "group_broadcast": physical_broadcast,
                "hardware_broadcast": layout.head_packing.hardware_broadcast_amount,
                "storage_block_broadcast_strategy": (
                    "replicate_single_kv_head_select_group_lanes"
                    if layout.head_packing.groups_per_storage_block > 1
                    else "single_group"
                ),
                "chunks_per_kv": layout.head_packing.chunks_per_kv,
                "full_chunks": full_chunks,
                "tail_heads": tail_heads,
                "q_blocks": q_blocks,
                "k_blocks": q_blocks,
                "policy": kv_residency.policy,
                "kv_resident": kv_residency.full_resident,
                "resident_kv_tiles": resident_kv_tiles,
                "resident_prefix_blocks": kv_residency.resident_prefix_blocks,
                "streaming_blocks": kv_residency.streaming_blocks,
                "requested_residency_fraction": (kv_residency.requested_residency_fraction),
                "realized_residency_fraction": (kv_residency.realized_residency_fraction),
                "stream_k_address": kv_residency.stream_k_address,
                "stream_v_address": kv_residency.stream_v_address,
                "peak_live_tiles": kv_residency.peak_live_tiles,
                "tile_utilization": kv_residency.tile_utilization,
                "kv_cache_fidelity": "exact_compiler_schedule_single_chip",
                "looped_batch": layout.sequence_packing.attention_group_count > 1,
                "looped_kv_heads": model.num_kv_heads > 1,
                "looped_full_chunks": full_chunks > 1,
                "rows_per_batch": layout.rows_per_batch,
                "logical_group_count": layout.head_packing.logical_group_count,
                "groups_per_storage_block": (layout.head_packing.groups_per_storage_block),
                "storage_block_count": layout.head_packing.storage_block_count,
                "logical_q_width": model.total_q_dim,
                "physical_q_width": layout.head_packing.total_q_dim,
                "head_lane_utilization": (layout.head_packing.head_lane_utilization),
            },
            "native_layout": {
                **layout.sequence_packing.metadata(),
                "head_packing": layout.head_packing.metadata(),
            },
            "packed_attention": prog.packed_attention_stats(),
            "vector_scalar_optimization": prog.vector_scalar_stats(),
            # Keep the selected lowering mode at the trace root as well as in
            # the optimization counters.  Timing consumers must be able to
            # select the rtl-v3 scoreboard without reverse-engineering an
            # optional diagnostics dictionary.
            "vector_scalar_schedule": vector_scalar_schedule,
            "selector_schedule": selector_schedule,
            "reduction_output_mode": reduction_output_mode,
            "packed_attention_schedule": packed_attention_schedule,
            "softmax_state_schedule": softmax_state_schedule,
            "packed_qk_schedule": packed_qk_schedule,
            **(getattr(prog, "_moe_lowering_stats", {}) if is_moe_layer else {"moe_lowering_schedule": None}),
            **softmax_layout.metadata(),
            "broadcast_timing_model": "ordinary_matrix_structural_equivalent",
            "broadcast_rtl_validated": (
                False if packed_qk_schedule == PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1 else None
            ),
            "broadcast_rtl_validation_status": (
                "broadcast_rtl_unvalidated"
                if packed_qk_schedule == PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1
                else "not_applicable"
            ),
            "gqa_pipeline_schedule": gqa_pipeline_schedule,
            "address_generation_mode": address_generation_mode,
            "ffn_address_schedule": ffn_address_schedule,
            "ffn_projection_schedule": prog.ffn_projection_schedule,
            "ffn_address_optimization": ffn_address_optimization,
            "cost_trace_granularity": cost_trace_granularity,
            "compute_trace_fidelity": (
                "exact_algebraic_ideal_ii1"
                if cost_trace_granularity == COST_TRACE_GRANULARITY_AFFINE_BLOCK_SUMMARY_V1
                else "ordered_detailed"
            ),
            "ordered_schedule_available": (cost_trace_granularity == COST_TRACE_GRANULARITY_DETAILED),
            "block_class_count": (
                min(96, max(1, q_blocks) * 2)
                if cost_trace_granularity == COST_TRACE_GRANULARITY_AFFINE_BLOCK_SUMMARY_V1
                else None
            ),
            "expanded_block_pair_equivalent": expanded_block_pairs,
            "materialized_block_pair_count": (
                0 if cost_trace_granularity == COST_TRACE_GRANULARITY_AFFINE_BLOCK_SUMMARY_V1 else expanded_block_pairs
            ),
            "gqa_timing_artifact": prog.packed_attention_stats().get("gqa_timing_artifact"),
            "gqa_timing_artifact_sha256": prog.packed_attention_stats().get("gqa_timing_artifact_sha256"),
            "native_layout_mode": native_layout_mode,
            "compiler_revision": os.environ.get("PLENA_COMPILER_REVISION", "working-tree"),
            "config_hash": _config_hash(
                model,
                hardware,
                seq_len,
                batch_size,
                layer_idx=layer_idx,
                routing_plan_hash=routing_plan_hash,
                moe_routing_mode=moe_routing_mode,
                moe_lowering_schedule=moe_lowering_schedule,
                moe_layer_scaling=moe_layer_scaling,
                native_layout_mode=native_layout_mode,
                packed_attention_schedule=packed_attention_schedule,
                softmax_state_schedule=softmax_state_schedule,
                packed_qk_schedule=packed_qk_schedule,
                vector_scalar_schedule=vector_scalar_schedule,
                selector_schedule=selector_schedule,
                reduction_output_mode=reduction_output_mode,
                gqa_pipeline_schedule=gqa_pipeline_schedule,
                gqa_timing_artifact_sha256=prog.packed_attention_stats().get("gqa_timing_artifact_sha256"),
                address_generation_mode=address_generation_mode,
                ffn_address_schedule=ffn_address_schedule,
                ffn_projection_schedule=ffn_projection_schedule,
                cost_trace_granularity=cost_trace_granularity,
            ),
            "dma_logical_layout": "precision_independent_v3",
            "layer_hbm_stride": layer_hbm_stride,
            "selected_layer_idx": layer_idx,
            "configured_num_layers": configured_layers,
            "moe_routing_mode": moe_routing_mode if is_moe_layer else None,
            "routing_plan_hash": routing_plan_hash,
            "routing_summary_hash": (summary.routing_summary_hash if summary is not None else None),
            "routing_summary_algorithm": (summary.algorithm_version if summary is not None else None),
            "routing_fidelity": (
                "fixed_balanced_histogram"
                if summary is not None
                else "exact_static_indices"
                if plan is not None
                else None
            ),
            "route_count": 0 if routing is None else routing.route_count,
            "active_expert_ids": ([] if routing is None else list(routing.active_expert_ids)),
            "active_expert_count": (0 if routing is None else len(routing.active_expert_ids)),
            "materialized_route_count": (len(plan.routes) if plan is not None else 0),
            "routes_per_expert": routes_per_expert,
            "expert_bucket_rows": serialized_bucket_metadata,
            "host_selected_indices": plan is not None,
            "runtime_arg_topk_included": False if is_moe_layer else None,
            "exact_token_addresses": False if summary is not None else True,
            "latency_only": summary is not None,
            "excluded_runtime_operation": "arg_topk" if is_moe_layer else None,
            "layer_scaling_mode": moe_layer_scaling,
            "layer_scaling_fidelity": (
                "fixed_balanced_histogram"
                if summary is not None
                else "exact_static_indices"
                if plan is not None
                else "shape_equivalent_dense_repeat"
            ),
            "cost_cache_hit": False,
            "one_layer_static_opcodes": dict(sorted(one_layer.static_opcodes.items())),
            "one_layer_dynamic_opcodes": dict(sorted(one_layer.dynamic_opcodes.items())),
        }
    )
    raw_missing_lineage_count = sum(
        entry.count
        for entry in one_layer.parallel_kernel_census
        if entry.kernel == UNCLASSIFIED_PARALLEL_KERNEL
    )
    one_layer.parallel_kernel_census = _build_parallel_kernel_census(
        one_layer,
        model=model,
        hardware=hardware,
        layout=layout,
    )
    census_count = sum(entry.count for entry in one_layer.parallel_kernel_census)
    compute_count = sum(
        count
        for opcode, count in one_layer.dynamic_opcodes.items()
        if not opcode.startswith("H_")
    )
    if census_count != compute_count:
        raise ValueError(
            "parallel kernel census coverage mismatch: "
            f"census={census_count}, compute={compute_count}"
        )
    one_layer.metadata.update(
        {
            "parallel_kernel_census_schema": (
                "parallel_kernel_census_v2_schedule_lineage"
            ),
            "parallel_kernel_census_coverage": (
                1.0 if compute_count == 0 else census_count / compute_count
            ),
            "parallel_kernel_lineage_coverage": (
                1.0
                if compute_count == 0
                else (compute_count - raw_missing_lineage_count) / compute_count
            ),
            "parallel_kernel_stage_fallback_count": 0,
            "parallel_kernel_global_explicit_count": raw_missing_lineage_count,
            "parallel_kernel_census_entry_count": len(
                one_layer.parallel_kernel_census
            ),
        }
    )
    one_layer.metadata["energy_action_lineage"] = (
        _finalize_energy_action_lineage(one_layer)
    )
    one_layer.metadata["dma_coverage"] = _audit_memory_events(one_layer)
    one_layer.metadata["dma_metadata_fidelity"] = "exact"
    if use_cache:
        _store_one_layer(cache_key, one_layer)
    full_trace = _scale_trace(one_layer, num_layers, layer_hbm_stride=layer_hbm_stride)
    full_trace.metadata["layer_scaling_fidelity"] = (
        "approximate_repeated_balanced_routing"
        if summary is not None and num_layers > 1
        else "approximate_repeated_static_plan"
        if plan is not None and num_layers > 1
        else "fixed_balanced_histogram"
        if summary is not None
        else "exact_static_indices"
        if plan is not None
        else "shape_equivalent_dense_repeat"
    )
    return full_trace


__all__ = [
    "CompilerCostHardware",
    "NativeDecoderCostLayout",
    "clear_cost_trace_cache",
    "compile_native_decoder_cost_trace",
    "load_cost_model_config",
]
