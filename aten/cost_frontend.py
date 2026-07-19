"""Shape-only Qwen3 prefill lowering into a compiler cost trace.

The frontend deliberately reuses the native decoder lowering helpers.  Dense
and static-index MoE traces therefore describe the same ISA schedule as the
transactional compiler without materializing model weights or rendering ASM.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import compiler.aten.ops as ops
from compiler.aten.cost_emitter import (
    CostTrace,
    MemoryEvent,
    ScheduleAffineAdd,
    ScheduleAffineLoad,
    ScheduleInstruction,
    ScheduleNode,
    ScheduleRepeat,
    ScheduleSequence,
    ScheduleUnavailable,
)
from compiler.aten.isa_builder import RepeatAxis
from compiler.aten.model_extract import ModelConfig
from compiler.aten.moe import (
    FixedBalancedRoutingSummary,
    MoeRoutingPlan,
    coerce_routing_plan,
)
from compiler.aten.ops.registry import Backend, OpRegistry
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.native_layout import (
    NATIVE_LAYOUT_SCHEMA_VERSION,
    AttentionHeadPacking,
    SequencePackingPlan,
    build_attention_head_packing,
)
from compiler.aten.plena_frontend import (
    LayerInputVars,
    MoeExpertInputVars,
    MoeLayerInputVars,
    _add_residual,
    _emit_ffn_block,
    _emit_fpram_row_to_vram,
    _emit_moe_block,
    _emit_normalize_route_weights,
    _emit_packed_attention_block,
    _emit_router_softmax,
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

    def validate(self) -> None:
        positive = asdict(self)
        for name, value in positive.items():
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


_ONE_LAYER_CACHE_LIMIT = 64
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
        decoder_sparse_step=int(
            _mapping_value(value, "decoder_sparse_step", default=1) or 1
        ),
        mlp_only_layers=tuple(
            int(layer)
            for layer in (_mapping_value(value, "mlp_only_layers", default=()) or ())
        ),
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
        raise ValueError(
            "CostEmitter frontend supports model_type='qwen3' or "
            f"'qwen3_moe', got {model.model_type!r}"
        )
    if model.num_heads % model.num_kv_heads:
        raise ValueError(
            f"num_heads={model.num_heads} must be divisible by num_kv_heads={model.num_kv_heads}"
        )
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
    w_q, w_o, w_k_heads, w_v_heads = _register_shape_attention_inputs(
        prog, model, layout, layer_idx=layer_idx
    )
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

    w_q, w_o, w_k_heads, w_v_heads = _register_shape_attention_inputs(
        prog, model, layout, layer_idx=layer_idx
    )
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
            _emit_normalize_route_weights(
                prog, fpram_addr=route_base, count=summary.experts_per_token
            )
        _emit_fpram_row_to_vram(
            prog, fpram_addr=route_base, vram_addr=route_weights_addr
        )
    prog.free_tensor(route_scratch)
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
    with prog.cost_stage("norm"):
        _save_residual_and_norm(
            prog, current, scratch, layer_inputs.post_attn_norm
        )
    physical_rows, padded_hidden = current.physical_shape
    with prog.cost_stage("router"):
        router_probs = _linear_projection(
            prog,
            current,
            layer_inputs.w_router,
            f"moe_router_logits_{layer_idx}",
            physical_shape=(physical_rows, prog.mlen),
        )
        prog.vram_add(router_probs, router_mask)
        _emit_router_softmax(prog, router_probs, physical_rows=physical_rows)
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
        combined = prog.alloc(
            f"moe_combined_{layer_idx}",
            physical_rows,
            padded_hidden,
            strict=False,
            physical_shape=(physical_rows, padded_hidden),
        )
        prog.vram_fill_zero(combined)
    scalar_scratch = prog.alloc(
        f"moe_route_scalar_{layer_idx}",
        1,
        prog.mlen,
        strict=False,
        physical_shape=(1, prog.mlen),
    )
    scalar_addr = prog._ONLINE_SOFTMAX_FPSRAM_BASE
    bucket_metadata: dict[int, dict[str, int]] = {}
    padded_rows = summary.padded_bucket_rows(prog.blen)
    for expert_id, real_rows in summary.routes_per_expert.items():
        expert_padded_rows = padded_rows[expert_id]
        bucket_metadata[expert_id] = {
            "real_rows": real_rows,
            "padded_rows": expert_padded_rows,
        }
        with prog.cost_stage("dispatch"):
            bucket = prog.alloc(
                f"moe_expert_{expert_id}_bucket_l{layer_idx}",
                expert_padded_rows,
                padded_hidden,
                strict=False,
                physical_shape=(expert_padded_rows, padded_hidden),
            )
            prog.vram_fill_zero(bucket)
            with prog.cost_repeat_region(
                real_rows,
                name=f"moe_balanced_dispatch_e{expert_id}",
                repeat_kind="fixed_balanced",
            ):
                prog.vram_add(bucket, current, num_rows=1)
        expert = layer_inputs.experts[expert_id]
        with prog.cost_stage("experts"):
            ops.ffn(prog, bucket, expert.w_gate, expert.w_up, expert.w_down)
        with prog.cost_stage("combine"):
            with prog.cost_repeat_region(
                real_rows,
                name=f"moe_balanced_combine_e{expert_id}",
                repeat_kind="fixed_balanced",
            ):
                _emit_selected_probability(
                    prog,
                    router_probs=route_weights,
                    identity=route_identity,
                    scratch=scalar_scratch,
                    physical_row=0,
                    identity_row=expert_id % summary.experts_per_token,
                    fpram_addr=scalar_addr,
                )
                _emit_scale_wide_row_from_fpram(
                    prog, bucket, row=0, fpram_addr=scalar_addr
                )
                prog.vram_add(combined, bucket, num_rows=1)
        prog.free_tensor(bucket)
    with prog.cost_stage("combine"):
        prog.vram_fill_zero(current)
        prog.vram_add(current, combined)
        _add_residual(prog, current, scratch)
    prog.free_tensor(scalar_scratch)
    prog.free_tensor(route_weights)
    prog.free_tensor(router_probs)
    prog.free_tensor(combined)
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
            )
        )
    result.schedule, schedule_error = _scale_schedule(one_layer, num_layers)
    result.schedule_unavailable_reasons.update(one_layer.schedule_unavailable_reasons)
    if schedule_error is not None:
        result.schedule_unavailable_reasons[schedule_error] += 1
    return result


def _audit_memory_events(trace: CostTrace) -> dict[str, Any]:
    """Require exact, ordered DMA coverage for every dynamic HBM opcode."""
    accounted: Counter[tuple[str, str]] = Counter()
    for event in trace.memory_events:
        accounted[(event.stage, event.transfer.opcode)] += event.multiplicity
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
    stream_indices = [event.stream_index for event in trace.memory_events]
    if stream_indices != list(range(len(stream_indices))):
        raise ValueError("DMA stream indices are not contiguous and emission ordered")
    return {
        "geometry_fidelity": "exact",
        "stream_count": len(trace.memory_events),
        "dynamic_opcodes": {
            opcode: sum(
                event.multiplicity
                for event in trace.memory_events
                if event.transfer.opcode == opcode
            )
            for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")
        },
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


def _active_physical_rows(
    *, batch_size: int, seq_len: int, rows_per_batch: int
) -> tuple[int, ...]:
    return tuple(
        batch_idx * rows_per_batch + token_idx
        for batch_idx in range(batch_size)
        for token_idx in range(seq_len)
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
    moe_layer_scaling: str,
    native_layout_mode: str,
    packed_attention_schedule: str,
    vector_scalar_schedule: str,
) -> tuple[Any, ...]:
    return (
        model,
        hardware,
        seq_len,
        batch_size,
        layer_idx,
        routing_plan_hash,
        moe_routing_mode,
        moe_layer_scaling,
        NATIVE_LAYOUT_SCHEMA_VERSION,
        native_layout_mode,
        packed_attention_schedule,
        vector_scalar_schedule,
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
    moe_layer_scaling: str,
    native_layout_mode: str,
    packed_attention_schedule: str,
    vector_scalar_schedule: str,
) -> str:
    payload = {
        "model": asdict(model),
        "hardware": asdict(hardware),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "layer_idx": layer_idx,
        "routing_plan_hash": routing_plan_hash,
        "moe_routing_mode": moe_routing_mode,
        "moe_layer_scaling": moe_layer_scaling,
        "native_layout_schema_version": NATIVE_LAYOUT_SCHEMA_VERSION,
        "native_layout_mode": native_layout_mode,
        "packed_attention_schedule": packed_attention_schedule,
        "vector_scalar_schedule": vector_scalar_schedule,
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
    moe_routing_plan: MoeRoutingPlan | Mapping[str, Any] | str | Path | None = None,
    max_static_routes: int = 1024,
    moe_layer_scaling: str = "single-layer",
    native_layout_mode: str = "compact",
    packed_attention_schedule: str = "direct-first-block-v1",
    vector_scalar_schedule: str = "compiler-v1",
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
        raise ValueError(
            f"layer_idx={layer_idx} outside configured num_layers={configured_layers}"
        )
    if max_static_routes <= 0:
        raise ValueError(
            f"max_static_routes must be positive, got {max_static_routes}"
        )
    if native_layout_mode not in {"compact", "legacy"}:
        raise ValueError(
            "native_layout_mode must be 'compact' or 'legacy', got "
            f"{native_layout_mode!r}"
        )
    if packed_attention_schedule not in {"direct-first-block-v1", "legacy"}:
        raise ValueError(
            "packed_attention_schedule must be 'direct-first-block-v1' or "
            f"'legacy', got {packed_attention_schedule!r}"
        )
    if vector_scalar_schedule not in {"compiler-v1", "legacy"}:
        raise ValueError(
            "vector_scalar_schedule must be 'compiler-v1' or 'legacy', got "
            f"{vector_scalar_schedule!r}"
        )
    if moe_routing_mode not in {"static-indices", "fixed-balanced"}:
        raise ValueError(
            "moe_routing_mode must be 'static-indices' or 'fixed-balanced', "
            f"got {moe_routing_mode!r}"
        )
    if moe_layer_scaling not in {
        "single-layer",
        "repeat-static-plan",
        "repeat-fixed-balanced",
    }:
        raise ValueError(
            "unsupported moe_layer_scaling=" f"{moe_layer_scaling!r}"
        )

    is_moe_model = model.num_experts > 0
    is_moe_layer = model.is_moe_layer(layer_idx)
    plan = _load_moe_routing_plan(moe_routing_plan)
    summary: FixedBalancedRoutingSummary | None = None
    if is_moe_layer:
        if model.num_experts <= 0 or model.experts_per_token <= 0:
            raise ValueError(
                "MoE model config must define positive num_experts and "
                "num_experts_per_tok"
            )
        if model.num_experts > hardware.mlen:
            raise ValueError(
                f"num_experts={model.num_experts} exceeds MLEN={hardware.mlen}"
            )
        if moe_routing_mode == "static-indices":
            if plan is None:
                raise ValueError(
                    "static-index MoE CostEmitter requires an explicit "
                    "moe_routing_plan"
                )
            if plan.num_experts != model.num_experts:
                raise ValueError(
                    f"Routing plan num_experts={plan.num_experts} does not match "
                    f"model num_experts={model.num_experts}"
                )
            if plan.experts_per_token != model.experts_per_token:
                raise ValueError(
                    f"Routing plan top-k={plan.experts_per_token} does not match "
                    f"model top-k={model.experts_per_token}"
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
                raise ValueError(
                    "fixed-balanced routing cannot be combined with moe_routing_plan"
                )
            summary = FixedBalancedRoutingSummary.build(
                num_tokens=seq_len * batch_size,
                num_experts=model.num_experts,
                experts_per_token=model.experts_per_token,
            )
        if num_layers is None:
            num_layers = 1
        expected_scaling = (
            "repeat-static-plan"
            if moe_routing_mode == "static-indices"
            else "repeat-fixed-balanced"
        )
        if num_layers > 1 and moe_layer_scaling != expected_scaling:
            raise ValueError(
                f"num_layers > 1 for {moe_routing_mode} MoE requires "
                f"moe_layer_scaling={expected_scaling!r}"
            )
        if num_layers == 1 and moe_layer_scaling != "single-layer":
            raise ValueError(
                "single-layer MoE costing requires moe_layer_scaling='single-layer'"
            )
    else:
        if plan is not None:
            raise ValueError(
                f"layer_idx={layer_idx} is dense; moe_routing_plan is not applicable"
            )
        if moe_routing_mode != "static-indices":
            raise ValueError("fixed-balanced routing is only valid for a MoE layer")
        if moe_layer_scaling != "single-layer":
            raise ValueError(
                "moe_layer_scaling is only applicable to a selected MoE layer"
            )
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
        plan.routing_plan_hash
        if plan is not None
        else summary.routing_summary_hash
        if summary is not None
        else None
    )
    cache_key = _trace_cache_key(
        model,
        hardware,
        seq_len=seq_len,
        batch_size=batch_size,
        layer_idx=layer_idx,
        routing_plan_hash=routing_plan_hash,
        moe_routing_mode=moe_routing_mode,
        moe_layer_scaling=moe_layer_scaling,
        native_layout_mode=native_layout_mode,
        packed_attention_schedule=packed_attention_schedule,
        vector_scalar_schedule=vector_scalar_schedule,
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
        hbm_v_prefetch_amount=hardware.hbm_v_prefetch_amount,
        hbm_v_writeback_amount=hardware.hbm_v_writeback_amount,
        emission_mode="cost",
        cost_strict_raw=True,
        packed_attention_schedule=packed_attention_schedule,
        vector_scalar_schedule=vector_scalar_schedule,
    )
    prog._native_active_row_ranges = layout.sequence_packing.active_row_ranges()
    prog.hlen = hardware.hlen
    prog.broadcast_amount = layout.head_packing.hardware_broadcast_amount
    prog.hbm_m_prefetch_amount = hardware.hbm_m_prefetch_amount

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
    if is_moe_layer:
        router_mask_input = prog.input(
            "MOE_ROUTER_MASK",
            (layout.compile_seq_rows, hardware.mlen),
            physical_shape=(layout.compile_seq_rows, hardware.mlen),
        )
        route_identity_input = prog.input(
            "MOE_ROUTE_IDENTITY", (hardware.mlen, hardware.mlen)
        )
        with prog.cost_stage("global/moe_setup"):
            router_mask = prog.load_batch(
                router_mask_input, name="MOE_ROUTER_MASK"
            )
            route_identity = prog.load_batch(
                route_identity_input, name="MOE_ROUTE_IDENTITY"
            )

    layer_hbm_start = prog._next_hbm_addr
    if is_moe_layer:
        routing = plan if plan is not None else summary
        assert routing is not None
        layer_inputs = _register_shape_moe_layer_inputs(
            prog, model, layout, routing, layer_idx=layer_idx
        )
    else:
        layer_inputs = _register_shape_layer_inputs(
            prog, model, layout, layer_idx=layer_idx
        )
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
            active_seq_len_per_batch=(
                layout.sequence_packing.attention_group_seq_len
            ),
            active_seq_len=batch_size * seq_len,
            active_hidden=model.hidden_size,
        )
    moe_bucket_metadata: dict[int, dict[str, int]] = {}
    if is_moe_layer:
        assert isinstance(layer_inputs, MoeLayerInputVars)
        assert router_mask is not None
        assert route_identity is not None
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
    ratio = model.num_heads // model.num_kv_heads
    physical_broadcast = layout.head_packing.broadcast_amount
    full_chunks, tail_heads = divmod(ratio, physical_broadcast)
    q_blocks = math.ceil(
        layout.sequence_packing.attention_group_seq_len / hardware.mlen
    )
    resident_kv_tiles = 2 * q_blocks
    active_inter_dim = (
        model.moe_inter_dim or model.inter_dim
        if is_moe_layer
        else model.dense_inter_dim or model.inter_dim
    )
    routing = plan if plan is not None else summary
    routes_per_expert = (
        {}
        if routing is None
        else {
            str(expert_id): count
            for expert_id, count in routing.routes_per_expert.items()
        }
    )
    serialized_bucket_metadata = {
        str(expert_id): values
        for expert_id, values in moe_bucket_metadata.items()
    }
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
                "kv_resident": resident_kv_tiles <= hardware.mram_tile_capacity,
                "resident_kv_tiles": resident_kv_tiles,
                "looped_batch": layout.sequence_packing.attention_group_count > 1,
                "looped_kv_heads": model.num_kv_heads > 1,
                "looped_full_chunks": full_chunks > 1,
                "rows_per_batch": layout.rows_per_batch,
                "logical_group_count": layout.head_packing.logical_group_count,
                "groups_per_storage_block": (
                    layout.head_packing.groups_per_storage_block
                ),
                "storage_block_count": layout.head_packing.storage_block_count,
                "logical_q_width": model.total_q_dim,
                "physical_q_width": layout.head_packing.total_q_dim,
                "head_lane_utilization": (
                    layout.head_packing.head_lane_utilization
                ),
            },
            "native_layout": {
                **layout.sequence_packing.metadata(),
                "head_packing": layout.head_packing.metadata(),
            },
            "packed_attention": prog.packed_attention_stats(),
            "vector_scalar_optimization": prog.vector_scalar_stats(),
            "compiler_revision": os.environ.get("PLENA_COMPILER_REVISION", "working-tree"),
            "config_hash": _config_hash(
                model,
                hardware,
                seq_len,
                batch_size,
                layer_idx=layer_idx,
                routing_plan_hash=routing_plan_hash,
                moe_routing_mode=moe_routing_mode,
                moe_layer_scaling=moe_layer_scaling,
                native_layout_mode=native_layout_mode,
                packed_attention_schedule=packed_attention_schedule,
                vector_scalar_schedule=vector_scalar_schedule,
            ),
            "dma_logical_layout": "precision_independent_v3",
            "layer_hbm_stride": layer_hbm_stride,
            "selected_layer_idx": layer_idx,
            "configured_num_layers": configured_layers,
            "moe_routing_mode": moe_routing_mode if is_moe_layer else None,
            "routing_plan_hash": routing_plan_hash,
            "routing_summary_hash": (
                summary.routing_summary_hash if summary is not None else None
            ),
            "routing_summary_algorithm": (
                summary.algorithm_version if summary is not None else None
            ),
            "routing_fidelity": (
                "fixed_balanced_histogram"
                if summary is not None
                else "exact_static_indices"
                if plan is not None
                else None
            ),
            "route_count": 0 if routing is None else routing.route_count,
            "active_expert_ids": (
                [] if routing is None else list(routing.active_expert_ids)
            ),
            "active_expert_count": (
                0 if routing is None else len(routing.active_expert_ids)
            ),
            "materialized_route_count": (
                len(plan.routes) if plan is not None else 0
            ),
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
