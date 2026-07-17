"""Shape-only dense Qwen3 prefill lowering into a compiler cost trace."""

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
from compiler.aten.ops.registry import Backend, OpRegistry
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena_frontend import (
    AttentionHeadPacking,
    LayerInputVars,
    _emit_ffn_block,
    _emit_packed_attention_block,
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
    config = ModelConfig(
        hidden_size=hidden,
        inter_dim=int(_mapping_value(value, "intermediate_size", "inter_dim", default=4 * hidden)),
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=int(_mapping_value(value, "head_dim", default=hidden // heads)),
        eps=float(_mapping_value(value, "rms_norm_eps", "eps", default=1e-5)),
        rope_theta=float(_mapping_value(value, "rope_theta", default=10_000.0)),
        vocab_size=_mapping_value(value, "vocab_size"),
        model_type=str(_mapping_value(value, "model_type", default="qwen3")),
    )
    layers = _mapping_value(value, "num_hidden_layers", "num_layers")
    return config, int(layers) if layers is not None else None


def _build_layout(
    model: ModelConfig,
    hardware: CompilerCostHardware,
    *,
    seq_len: int,
    batch_size: int,
) -> NativeDecoderCostLayout:
    if model.model_type != "qwen3":
        raise ValueError(f"first CostEmitter frontend supports model_type='qwen3', got {model.model_type!r}")
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
    ratio = model.num_heads // model.num_kv_heads
    physical_broadcast = min(hardware.broadcast_amount, hardware.mlen // hardware.hlen)
    chunks_per_kv = math.ceil(ratio / physical_broadcast)
    padded_seq_len = _ceil_to_multiple(seq_len, hardware.mlen)
    rows_per_batch = padded_seq_len if batch_size == 1 else _ceil_to_multiple(
        max(hardware.mlen, padded_seq_len), hardware.mlen
    )
    packing = AttentionHeadPacking(
        enabled=True,
        hlen=hardware.hlen,
        logical_broadcast_amount=hardware.broadcast_amount,
        broadcast_amount=physical_broadcast,
        head_slot_dim=hardware.hlen,
        group_width=hardware.mlen,
        total_q_dim=model.num_kv_heads * chunks_per_kv * hardware.mlen,
        active_head_dim=model.head_dim,
        chunks_per_kv=chunks_per_kv,
    )
    return NativeDecoderCostLayout(
        padded_seq_len=padded_seq_len,
        rows_per_batch=rows_per_batch,
        compile_seq_rows=batch_size * rows_per_batch,
        padded_hidden=_ceil_to_multiple(model.hidden_size, hardware.mlen),
        padded_inter=_ceil_to_multiple(model.inter_dim, hardware.mlen),
        padded_head_dim=_ceil_to_multiple(model.head_dim, hardware.mlen),
        head_packing=packing,
    )


def _register_shape_layer_inputs(
    prog: PlenaCompiler,
    model: ModelConfig,
    layout: NativeDecoderCostLayout,
    *,
    layer_idx: int,
) -> LayerInputVars:
    def prefix(name: str) -> str:
        return f"{name}_{layer_idx}"

    w_q = prog.input(prefix("W_q"), (layout.padded_hidden, layout.head_packing.total_q_dim))
    w_o = prog.input(prefix("W_o"), (layout.head_packing.total_q_dim, layout.padded_hidden))
    w_k_heads = []
    w_v_heads = []
    for head in range(model.num_kv_heads):
        w_k_heads.append(
            prog.input(
                prefix(f"W_k_h{head}"),
                (layout.padded_hidden, layout.head_packing.head_slot_dim),
                physical_shape=(layout.padded_hidden, layout.padded_head_dim),
            )
        )
        w_v_heads.append(
            prog.input(
                prefix(f"W_v_h{head}"),
                (layout.padded_hidden, layout.head_packing.head_slot_dim),
                physical_shape=(layout.padded_hidden, layout.padded_head_dim),
            )
        )
    return LayerInputVars(
        w_q=w_q,
        w_o=w_o,
        w_k_heads=w_k_heads,
        w_v_heads=w_v_heads,
        w_gate=prog.input(prefix("W_gate"), (layout.padded_hidden, layout.padded_inter)),
        w_up=prog.input(prefix("W_up"), (layout.padded_hidden, layout.padded_inter)),
        w_down=prog.input(prefix("W_down"), (layout.padded_inter, layout.padded_hidden)),
    )


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

    classifications: list[str] = []
    for child in one_layer.schedule.children:
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

    children: list[ScheduleNode] = list(one_layer.schedule.children[:first])
    children.append(
        ScheduleRepeat(
            count=num_layers,
            body=ScheduleSequence(one_layer.schedule.children[first : last + 1]),
            name="decoder_layer",
            repeat_kind="model_layer",
        )
    )
    children.extend(one_layer.schedule.children[last + 1 :])
    return ScheduleSequence(tuple(children)), None


def _scale_trace(one_layer: CostTrace, num_layers: int, *, layer_hbm_stride: int = 0) -> CostTrace:
    result = CostTrace(metadata=dict(one_layer.metadata))
    result.metadata["num_layers"] = num_layers
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


def _trace_cache_key(
    model: ModelConfig,
    hardware: CompilerCostHardware,
    *,
    seq_len: int,
    batch_size: int,
) -> tuple[Any, ...]:
    return (model, hardware, seq_len, batch_size)


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


def _config_hash(model: ModelConfig, hardware: CompilerCostHardware, seq_len: int, batch_size: int) -> str:
    payload = {
        "model": asdict(model),
        "hardware": asdict(hardware),
        "seq_len": seq_len,
        "batch_size": batch_size,
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
    use_cache: bool = True,
) -> CostTrace:
    """Run the real packed-GQA compiler schedule using shape-only variables."""
    model, configured_layers = load_cost_model_config(model_config)
    hardware = (
        hardware_config
        if isinstance(hardware_config, CompilerCostHardware)
        else CompilerCostHardware(**hardware_config)
    )
    hardware.validate()
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError(f"seq_len and batch_size must be positive, got {seq_len}, {batch_size}")
    if num_layers is None:
        num_layers = configured_layers or 1
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")

    cache_key = _trace_cache_key(model, hardware, seq_len=seq_len, batch_size=batch_size)
    one_layer = _cached_one_layer(cache_key) if use_cache else None
    if one_layer is not None:
        full_trace = _scale_trace(
            one_layer,
            num_layers,
            layer_hbm_stride=int(one_layer.metadata.get("layer_hbm_stride", 0)),
        )
        full_trace.metadata["cost_cache_hit"] = True
        return full_trace

    layout = _build_layout(model, hardware, seq_len=seq_len, batch_size=batch_size)
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
    )
    prog.hlen = hardware.hlen
    prog.broadcast_amount = layout.head_packing.broadcast_amount
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

    layer_hbm_start = prog._next_hbm_addr
    layer_inputs = _register_shape_layer_inputs(prog, model, layout, layer_idx=0)
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
            0,
            layout.padded_seq_len,
            model.head_dim,
            model.num_kv_heads,
            model.head_ratio,
            layout.head_packing,
            batch_size=batch_size,
            rows_per_batch=layout.rows_per_batch,
            active_seq_len_per_batch=seq_len,
            active_seq_len=batch_size * seq_len,
            active_hidden=model.hidden_size,
        )
    with prog.cost_stage("layer/ffn"):
        current = _emit_ffn_block(
            prog,
            current,
            layer_inputs,
            scratch,
            layer_idx=0,
            active_seq_len=batch_size * seq_len,
            active_hidden=model.hidden_size,
        )
    with prog.cost_stage("global/final_norm"):
        ops.rms_norm(prog, current, eps_offset=3, reci_hid_offset=4)

    one_layer = prog.compile_cost_trace()
    ratio = model.num_heads // model.num_kv_heads
    physical_broadcast = layout.head_packing.broadcast_amount
    full_chunks, tail_heads = divmod(ratio, physical_broadcast)
    q_blocks = math.ceil(seq_len / hardware.mlen)
    resident_kv_tiles = 2 * q_blocks
    one_layer.metadata.update(
        {
            "workload": {
                "model_type": model.model_type,
                "hidden_size": model.hidden_size,
                "inter_dim": model.inter_dim,
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
                "chunks_per_kv": layout.head_packing.chunks_per_kv,
                "full_chunks": full_chunks,
                "tail_heads": tail_heads,
                "q_blocks": q_blocks,
                "k_blocks": q_blocks,
                "kv_resident": resident_kv_tiles <= hardware.mram_tile_capacity,
                "resident_kv_tiles": resident_kv_tiles,
                "looped_batch": batch_size > 1,
                "looped_kv_heads": model.num_kv_heads > 1,
                "looped_full_chunks": full_chunks > 1,
                "rows_per_batch": layout.rows_per_batch,
            },
            "compiler_revision": os.environ.get("PLENA_COMPILER_REVISION", "working-tree"),
            "config_hash": _config_hash(model, hardware, seq_len, batch_size),
            "dma_logical_layout": "precision_independent_v3",
            "layer_hbm_stride": layer_hbm_stride,
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
    return full_trace


__all__ = [
    "CompilerCostHardware",
    "NativeDecoderCostLayout",
    "clear_cost_trace_cache",
    "compile_native_decoder_cost_trace",
    "load_cost_model_config",
]
