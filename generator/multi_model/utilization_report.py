"""
Trace-based compute and memory utilization analysis for VLM models.

This module consumes the hook trace returned by ``VLMModelParser.trace_leaf_modules``
or a flat node list derived from it. It reports:

- Compute-oriented estimates for tiled hardware execution
- Activation-memory usage derived from symbol liveness
- Peak live-memory utilization against an optional hardware budget
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

try:
    from .vlm_parser import VLMModelParser, flatten_call_tree, template_qwen3_vl_inputs
except ImportError:
    from vlm_parser import VLMModelParser, flatten_call_tree, template_qwen3_vl_inputs


DEFAULT_HW: dict[str, int | None] = {
    "tile_m": 64,
    "tile_k": 64,
    "tile_n": 64,
    "memory_capacity_bytes": None,
}

DTYPE_BYTES: dict[str, int] = {
    "torch.float64": 8,
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
}

_ATTENTION_HINTS = ("attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", ".q", ".k", ".v")
_FFN_HINTS = ("mlp", "ffn", "feed_forward", "gate_proj", "up_proj", "down_proj", "fc1", "fc2")
_NORM_HINTS = ("norm", "layernorm", "rmsnorm")
_VISION_HINTS = ("visual", "vision", "image", "patch", "pixel")
_TEXT_HINTS = ("language", "text", "token", "lm_head", "embed_tokens")
_CROSS_HINTS = ("projector", "merger", "fusion", "cross")


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def _numel(shape: list[int] | tuple[int, ...] | None) -> int:
    if not shape:
        return 0
    return _product(shape)


def _dtype_nbytes(dtype: str | None) -> int:
    if dtype is None:
        return 4
    return DTYPE_BYTES.get(str(dtype), 4)


def _bytes_for(shape: list[int] | tuple[int, ...] | None, dtype: str | None) -> int:
    return _numel(shape) * _dtype_nbytes(dtype)


def _human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    return f"{value} B"


def _safe_ratio(numerator: int | float, denominator: int | float | None) -> float | None:
    if denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _ceil_to(value: int, tile: int) -> int:
    if tile <= 0:
        raise ValueError(f"Tile size must be positive, got {tile}")
    return int(math.ceil(value / tile) * tile) if value > 0 else 0


def _resolve_hw(hw: dict[str, Any] | None, m: int | None, k: int | None, n: int | None) -> dict[str, Any]:
    resolved = dict(DEFAULT_HW)
    if hw is not None:
        resolved.update(hw)

    aliases = {
        "tile_m": ("tile_m", "m"),
        "tile_k": ("tile_k", "k"),
        "tile_n": ("tile_n", "n"),
        "memory_capacity_bytes": ("memory_capacity_bytes", "sram_bytes", "activation_memory_bytes"),
    }
    normalized: dict[str, Any] = {}
    for canonical, names in aliases.items():
        for name in names:
            if name in resolved and resolved[name] is not None:
                normalized[canonical] = resolved[name]
                break
        else:
            normalized[canonical] = DEFAULT_HW[canonical]

    if m is not None:
        normalized["tile_m"] = m
    if k is not None:
        normalized["tile_k"] = k
    if n is not None:
        normalized["tile_n"] = n
    return normalized


def _tensor_meta_list(field: Any) -> list[dict[str, Any]]:
    if isinstance(field, dict):
        return [field] if "shape" in field else []
    if isinstance(field, list):
        return [item for item in field if isinstance(item, dict) and "shape" in item]
    return []


def _pair_tensor_metas(symbols: list[str], field: Any) -> dict[str, dict[str, Any]]:
    metas = _tensor_meta_list(field)
    return {symbol: meta for symbol, meta in zip(symbols, metas)}


def _first_meta(node: dict[str, Any], key: str) -> dict[str, Any] | None:
    metas = _tensor_meta_list(node.get(key))
    return metas[0] if metas else None


def _guess_source_branch(name: str) -> str:
    lname = name.lower()
    if any(token in lname for token in _VISION_HINTS):
        return "vision"
    if any(token in lname for token in _TEXT_HINTS):
        return "text"
    if any(token in lname for token in _CROSS_HINTS):
        return "cross_modal"
    return "shared"


def _classify_op_family(module_type: str) -> str:
    if module_type == "Embedding":
        return "embedding"
    if module_type == "Linear":
        return "matmul"
    if module_type.startswith("Conv"):
        return "convolution"
    if module_type in {"LayerNorm", "RMSNorm", "Qwen3VLTextRMSNorm"} or "Norm" in module_type:
        return "normalization"
    if module_type in {"SiLU", "GELU", "ReLU", "Softmax"}:
        return "elementwise"
    return "module"


def _guess_semantic_group(node: dict[str, Any]) -> str:
    name = str(node.get("name", "")).lower()
    module_type = str(node.get("type", ""))
    if module_type == "Embedding" or "embed" in name:
        return "embedding"
    if any(token in name for token in _ATTENTION_HINTS) or "Attention" in module_type:
        return "attention"
    if any(token in name for token in _FFN_HINTS) or any(token in module_type for token in ("MLP", "FFN")):
        return "ffn"
    if any(token in name for token in _NORM_HINTS) or "Norm" in module_type:
        return "normalization"
    if module_type.startswith("Conv"):
        return "vision_conv"
    if module_type in {"SiLU", "GELU", "ReLU", "Softmax"}:
        return "elementwise"
    return "other"


def _flatten_trace(trace_or_nodes: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(trace_or_nodes, list):
        return sorted([deepcopy(node) for node in trace_or_nodes], key=lambda node: int(node["order"]))
    if isinstance(trace_or_nodes, dict):
        if "children" in trace_or_nodes:
            return flatten_call_tree(deepcopy(trace_or_nodes))
        if "nodes" in trace_or_nodes:
            return sorted([deepcopy(node) for node in trace_or_nodes["nodes"]], key=lambda node: int(node["order"]))
    raise TypeError("Expected a trace tree dict or a flat list of nodes")


def _estimate_parameter_count(weights: list[dict[str, Any]]) -> int:
    total = 0
    for weight in weights:
        shape = weight.get("shape")
        if isinstance(shape, list):
            total += _numel(shape)
    return total


def _estimate_linear_compute(
    node: dict[str, Any],
    input_meta: dict[str, Any] | None,
    output_meta: dict[str, Any] | None,
    hw: dict[str, Any],
) -> dict[str, Any]:
    weight_shape = None
    for weight in node.get("weights", []):
        if weight.get("name") == "weight":
            weight_shape = weight.get("shape")
            break
    if weight_shape is None and node.get("weights"):
        weight_shape = node["weights"][0].get("shape")

    input_shape = input_meta.get("shape") if input_meta else None
    output_shape = output_meta.get("shape") if output_meta else None

    if not isinstance(weight_shape, list) or len(weight_shape) != 2 or not isinstance(input_shape, list):
        return {
            "effective_m": None,
            "effective_k": None,
            "effective_n": None,
            "padded_m": None,
            "padded_k": None,
            "padded_n": None,
            "useful_flops": 0,
            "scheduled_flops": 0,
            "compute_utilization": None,
        }

    effective_n = int(weight_shape[0])
    effective_k = int(weight_shape[1])
    if len(input_shape) == 0:
        effective_m = 1
    elif len(input_shape) == 1:
        effective_m = 1
        effective_k = int(input_shape[0])
    else:
        effective_m = _product(input_shape[:-1])
        effective_k = int(input_shape[-1])

    if isinstance(output_shape, list) and output_shape:
        effective_n = int(output_shape[-1])

    padded_m = _ceil_to(effective_m, int(hw["tile_m"]))
    padded_k = _ceil_to(effective_k, int(hw["tile_k"]))
    padded_n = _ceil_to(effective_n, int(hw["tile_n"]))

    useful_flops = 2 * effective_m * effective_k * effective_n
    scheduled_flops = 2 * padded_m * padded_k * padded_n
    return {
        "effective_m": effective_m,
        "effective_k": effective_k,
        "effective_n": effective_n,
        "padded_m": padded_m,
        "padded_k": padded_k,
        "padded_n": padded_n,
        "useful_flops": useful_flops,
        "scheduled_flops": scheduled_flops,
        "compute_utilization": _safe_ratio(useful_flops, scheduled_flops),
    }


def _estimate_conv3d_compute(
    node: dict[str, Any],
    input_meta: dict[str, Any] | None,
    output_meta: dict[str, Any] | None,
    hw: dict[str, Any],
) -> dict[str, Any]:
    if input_meta is None or output_meta is None:
        return {
            "effective_m": None,
            "effective_k": None,
            "effective_n": None,
            "padded_m": None,
            "padded_k": None,
            "padded_n": None,
            "useful_flops": 0,
            "scheduled_flops": 0,
            "compute_utilization": None,
        }

    input_shape = input_meta.get("shape")
    output_shape = output_meta.get("shape")
    if not isinstance(input_shape, list) or not isinstance(output_shape, list) or len(input_shape) != 5 or len(output_shape) != 5:
        return {
            "effective_m": None,
            "effective_k": None,
            "effective_n": None,
            "padded_m": None,
            "padded_k": None,
            "padded_n": None,
            "useful_flops": 0,
            "scheduled_flops": 0,
            "compute_utilization": None,
        }

    weight_shape = None
    for weight in node.get("weights", []):
        if weight.get("name") == "weight":
            weight_shape = weight.get("shape")
            break

    if not isinstance(weight_shape, list) or len(weight_shape) != 5:
        return {
            "effective_m": None,
            "effective_k": None,
            "effective_n": None,
            "padded_m": None,
            "padded_k": None,
            "padded_n": None,
            "useful_flops": 0,
            "scheduled_flops": 0,
            "compute_utilization": None,
        }

    batch, out_channels, out_d, out_h, out_w = [int(v) for v in output_shape]
    in_channels = int(weight_shape[1])
    kernel_volume = _product(weight_shape[2:])
    effective_m = batch * out_d * out_h * out_w
    effective_k = in_channels * kernel_volume
    effective_n = out_channels

    padded_m = _ceil_to(effective_m, int(hw["tile_m"]))
    padded_k = _ceil_to(effective_k, int(hw["tile_k"]))
    padded_n = _ceil_to(effective_n, int(hw["tile_n"]))

    useful_flops = 2 * effective_m * effective_k * effective_n
    scheduled_flops = 2 * padded_m * padded_k * padded_n
    return {
        "effective_m": effective_m,
        "effective_k": effective_k,
        "effective_n": effective_n,
        "padded_m": padded_m,
        "padded_k": padded_k,
        "padded_n": padded_n,
        "useful_flops": useful_flops,
        "scheduled_flops": scheduled_flops,
        "compute_utilization": _safe_ratio(useful_flops, scheduled_flops),
    }


def _estimate_embedding_compute(
    input_meta: dict[str, Any] | None,
    output_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    input_shape = input_meta.get("shape") if input_meta else None
    output_shape = output_meta.get("shape") if output_meta else None
    token_count = _numel(input_shape) if isinstance(input_shape, list) else 0
    embedding_dim = int(output_shape[-1]) if isinstance(output_shape, list) and output_shape else None
    return {
        "token_count": token_count,
        "embedding_dim": embedding_dim,
        "useful_flops": 0,
        "scheduled_flops": 0,
        "compute_utilization": None,
    }


def _compute_node_stats(node: dict[str, Any], hw: dict[str, Any]) -> dict[str, Any]:
    input_meta = _first_meta(node, "in")
    output_meta = _first_meta(node, "out")
    op_family = _classify_op_family(str(node.get("type", "")))

    result = {
        "op_family": op_family,
        "semantic_group": _guess_semantic_group(node),
        "source_branch": _guess_source_branch(str(node.get("name", ""))),
        "is_leaf": not node.get("children"),
        "input_shape": input_meta.get("shape") if input_meta else None,
        "output_shape": output_meta.get("shape") if output_meta else None,
        "input_dtype": input_meta.get("dtype") if input_meta else None,
        "output_dtype": output_meta.get("dtype") if output_meta else None,
        "input_bytes": _bytes_for(input_meta.get("shape") if input_meta else None, input_meta.get("dtype") if input_meta else None),
        "output_bytes": _bytes_for(output_meta.get("shape") if output_meta else None, output_meta.get("dtype") if output_meta else None),
        "parameter_count": _estimate_parameter_count(node.get("weights", [])),
        "parameter_bytes_estimate": 0,
        "compute": {
            "effective_m": None,
            "effective_k": None,
            "effective_n": None,
            "padded_m": None,
            "padded_k": None,
            "padded_n": None,
            "useful_flops": 0,
            "scheduled_flops": 0,
            "compute_utilization": None,
        },
    }

    param_dtype = result["output_dtype"] or result["input_dtype"]
    result["parameter_bytes_estimate"] = result["parameter_count"] * _dtype_nbytes(param_dtype)

    if op_family == "matmul":
        result["compute"] = _estimate_linear_compute(node, input_meta, output_meta, hw)
    elif op_family == "convolution":
        result["compute"] = _estimate_conv3d_compute(node, input_meta, output_meta, hw)
    elif op_family == "embedding":
        result["compute"] = _estimate_embedding_compute(input_meta, output_meta)

    result["compute"]["padding_waste_flops"] = (
        result["compute"]["scheduled_flops"] - result["compute"]["useful_flops"]
    )
    return result


def _build_symbol_table(
    ordered_nodes: list[dict[str, Any]],
    keep_model_outputs_live: bool,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    symbols: dict[str, dict[str, Any]] = {}
    symbol_uses: dict[str, list[int]] = defaultdict(list)

    for node in ordered_nodes:
        input_pairs = _pair_tensor_metas(node.get("in_syms", []), node.get("in"))
        output_pairs = _pair_tensor_metas(node.get("out_syms", []), node.get("out"))

        for symbol, meta in input_pairs.items():
            info = symbols.setdefault(
                symbol,
                {
                    "symbol": symbol,
                    "producer": "__input__",
                    "birth_order": -1,
                    "shape": meta.get("shape"),
                    "dtype": meta.get("dtype"),
                    "device": meta.get("device"),
                    "bytes": _bytes_for(meta.get("shape"), meta.get("dtype")),
                    "alias_producers": [],
                },
            )
            if info.get("shape") is None and meta.get("shape") is not None:
                info["shape"] = meta.get("shape")
                info["dtype"] = meta.get("dtype")
                info["device"] = meta.get("device")
                info["bytes"] = _bytes_for(meta.get("shape"), meta.get("dtype"))
            symbol_uses[symbol].append(int(node["order"]))

        for symbol, meta in output_pairs.items():
            if symbol not in symbols:
                symbols[symbol] = {
                    "symbol": symbol,
                    "producer": node["name"],
                    "birth_order": int(node["order"]),
                    "shape": meta.get("shape"),
                    "dtype": meta.get("dtype"),
                    "device": meta.get("device"),
                    "bytes": _bytes_for(meta.get("shape"), meta.get("dtype")),
                    "alias_producers": [],
                }
            else:
                existing = symbols[symbol]
                existing_birth = int(existing["birth_order"])
                current_order = int(node["order"])
                if existing_birth < 0:
                    if existing["producer"] != node["name"]:
                        existing["alias_producers"].append(node["name"])
                elif current_order >= existing_birth:
                    if existing["producer"] != node["name"]:
                        existing["alias_producers"].append(existing["producer"])
                    existing["producer"] = node["name"]
                    existing["birth_order"] = current_order
                    existing["shape"] = meta.get("shape") or existing.get("shape")
                    existing["dtype"] = meta.get("dtype") or existing.get("dtype")
                    existing["device"] = meta.get("device") or existing.get("device")
                    existing["bytes"] = _bytes_for(existing.get("shape"), existing.get("dtype"))
                elif existing["producer"] != node["name"]:
                    existing["alias_producers"].append(node["name"])

    last_order = max((int(node["order"]) for node in ordered_nodes), default=-1)
    output_symbols = [symbol for symbol, info in symbols.items() if not symbol_uses.get(symbol)]

    for symbol, info in symbols.items():
        uses = symbol_uses.get(symbol, [])
        if uses:
            info["last_use_order"] = max(uses)
        elif keep_model_outputs_live and symbol in output_symbols:
            info["last_use_order"] = last_order
        else:
            info["last_use_order"] = int(info["birth_order"])

        info["retired_after_order"] = info["last_use_order"]
        info["is_model_input"] = info["birth_order"] < 0
        info["is_model_output"] = symbol in output_symbols

    return symbols, output_symbols


def _analyze_memory_timeline(
    ordered_nodes: list[dict[str, Any]],
    symbols: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    live_symbols = {
        symbol
        for symbol, info in symbols.items()
        if int(info["birth_order"]) < 0 and int(info["bytes"]) > 0
    }
    timeline: list[dict[str, Any]] = []

    def _live_bytes(active: set[str]) -> int:
        return sum(int(symbols[symbol]["bytes"]) for symbol in active)

    peak_bytes = _live_bytes(live_symbols)
    peak_step = -1 if peak_bytes else None
    peak_symbols = sorted(live_symbols)

    for node in ordered_nodes:
        order = int(node["order"])
        live_before = _live_bytes(live_symbols)

        allocated_symbols: list[str] = []
        for symbol in node.get("out_syms", []):
            info = symbols.get(symbol)
            if info is None:
                continue
            if int(info["birth_order"]) == order and symbol not in live_symbols and int(info["bytes"]) > 0:
                live_symbols.add(symbol)
                allocated_symbols.append(symbol)

        live_after_alloc = _live_bytes(live_symbols)
        if live_after_alloc > peak_bytes:
            peak_bytes = live_after_alloc
            peak_step = order
            peak_symbols = sorted(live_symbols)

        retired_symbols = [
            symbol
            for symbol in sorted(live_symbols)
            if int(symbols[symbol]["retired_after_order"]) == order and symbol not in node.get("out_syms", [])
        ]
        for symbol in retired_symbols:
            live_symbols.remove(symbol)

        live_after_retire = _live_bytes(live_symbols)
        timeline.append(
            {
                "order": order,
                "name": node.get("name", ""),
                "type": node.get("type", ""),
                "live_bytes_before_op": live_before,
                "live_bytes_after_alloc": live_after_alloc,
                "live_bytes_after_retire": live_after_retire,
                "allocated_symbols": allocated_symbols,
                "retired_symbols": retired_symbols,
            }
        )

    summary = {
        "input_live_bytes": _live_bytes(
            {
                symbol
                for symbol, info in symbols.items()
                if info.get("is_model_input") and int(info["bytes"]) > 0
            }
        ),
        "peak_live_bytes": peak_bytes,
        "peak_live_bytes_human": _human_bytes(peak_bytes),
        "peak_live_step": peak_step,
        "peak_live_symbols": peak_symbols,
        "final_live_bytes": _live_bytes(live_symbols),
        "final_live_symbols": sorted(live_symbols),
    }
    return timeline, summary


def _init_bucket() -> dict[str, Any]:
    return {
        "node_count": 0,
        "input_bytes": 0,
        "output_bytes": 0,
        "parameter_bytes_estimate": 0,
        "useful_flops": 0,
        "scheduled_flops": 0,
        "padding_waste_flops": 0,
    }


def _bucketize(nodes: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = defaultdict(_init_bucket)
    for node in nodes:
        bucket = buckets[str(node[key])]
        bucket["node_count"] += 1
        bucket["input_bytes"] += int(node["input_bytes"])
        bucket["output_bytes"] += int(node["output_bytes"])
        bucket["parameter_bytes_estimate"] += int(node["parameter_bytes_estimate"])
        bucket["useful_flops"] += int(node["compute"]["useful_flops"])
        bucket["scheduled_flops"] += int(node["compute"]["scheduled_flops"])
        bucket["padding_waste_flops"] += int(node["compute"]["padding_waste_flops"])

    result: dict[str, Any] = {}
    for name, bucket in buckets.items():
        result[name] = {
            **bucket,
            "compute_utilization": _safe_ratio(bucket["useful_flops"], bucket["scheduled_flops"]),
        }
    return dict(sorted(result.items()))


def analyse_trace_utilization(
    trace_or_nodes: dict[str, Any] | list[dict[str, Any]],
    model_info: dict[str, Any] | None = None,
    hw: dict[str, Any] | None = None,
    *,
    include_non_leaf: bool = False,
    keep_model_outputs_live: bool = True,
    m: int | None = None,
    k: int | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    """
    Analyse compute and memory utilization from a hook trace.

    Parameters
    ----------
    trace_or_nodes:
        Tree returned by ``trace_leaf_modules`` or a flat list of nodes.
    model_info:
        Optional model metadata, preserved in the final report for context.
    hw:
        Optional hardware configuration. Supported keys: ``tile_m``, ``tile_k``,
        ``tile_n``, ``memory_capacity_bytes``. ``m``/``k``/``n`` aliases are also
        accepted.
    include_non_leaf:
        If ``False`` only leaf nodes contribute to compute summaries. Memory
        analysis still considers the full trace.
    keep_model_outputs_live:
        If ``True`` graph outputs stay live until the end of the trace.
    m, k, n:
        Optional direct tile-dimension overrides.
    """
    ordered_nodes = _flatten_trace(trace_or_nodes)
    resolved_hw = _resolve_hw(hw, m=m, k=k, n=n)
    symbols, output_symbols = _build_symbol_table(ordered_nodes, keep_model_outputs_live=keep_model_outputs_live)
    memory_timeline, memory_summary = _analyze_memory_timeline(ordered_nodes, symbols)

    analysed_nodes: list[dict[str, Any]] = []
    for node in ordered_nodes:
        if not include_non_leaf and node.get("children"):
            continue
        node_report = {
            "name": node.get("name", ""),
            "type": node.get("type", ""),
            "order": int(node["order"]),
            "in_syms": list(node.get("in_syms", [])),
            "out_syms": list(node.get("out_syms", [])),
            "weights": deepcopy(node.get("weights", [])),
        }
        node_report.update(_compute_node_stats(node, resolved_hw))

        timeline_entry = next((entry for entry in memory_timeline if entry["order"] == node_report["order"]), None)
        node_report["memory"] = {
            "live_bytes_before_op": timeline_entry["live_bytes_before_op"] if timeline_entry else 0,
            "live_bytes_after_alloc": timeline_entry["live_bytes_after_alloc"] if timeline_entry else 0,
            "live_bytes_after_retire": timeline_entry["live_bytes_after_retire"] if timeline_entry else 0,
            "allocated_symbols": timeline_entry["allocated_symbols"] if timeline_entry else [],
            "retired_symbols": timeline_entry["retired_symbols"] if timeline_entry else [],
        }
        analysed_nodes.append(node_report)

    total_useful_flops = sum(int(node["compute"]["useful_flops"]) for node in analysed_nodes)
    total_scheduled_flops = sum(int(node["compute"]["scheduled_flops"]) for node in analysed_nodes)
    memory_capacity = resolved_hw.get("memory_capacity_bytes")

    report = {
        "model_info": deepcopy(model_info) if model_info is not None else {},
        "hardware": resolved_hw,
        "analysis_scope": {
            "compute_nodes": "all_nodes" if include_non_leaf else "leaf_nodes",
            "memory_nodes": "all_nodes",
            "keep_model_outputs_live": keep_model_outputs_live,
        },
        "summary": {
            "node_count_total": len(ordered_nodes),
            "node_count_analysed": len(analysed_nodes),
            "symbol_count": len(symbols),
            "model_output_symbol_count": len(output_symbols),
            "total_useful_flops": total_useful_flops,
            "total_scheduled_flops": total_scheduled_flops,
            "overall_compute_utilization": _safe_ratio(total_useful_flops, total_scheduled_flops),
            "peak_live_bytes": memory_summary["peak_live_bytes"],
            "peak_live_bytes_human": memory_summary["peak_live_bytes_human"],
            "memory_capacity_bytes": memory_capacity,
            "memory_capacity_human": _human_bytes(int(memory_capacity)) if memory_capacity is not None else None,
            "memory_utilization_ratio": _safe_ratio(memory_summary["peak_live_bytes"], memory_capacity),
            "by_op_family": _bucketize(analysed_nodes, "op_family"),
            "by_semantic_group": _bucketize(analysed_nodes, "semantic_group"),
            "by_source_branch": _bucketize(analysed_nodes, "source_branch"),
        },
        "memory": {
            **memory_summary,
            "timeline": memory_timeline,
            "symbols": [
                {
                    **info,
                    "bytes_human": _human_bytes(int(info["bytes"])),
                }
                for _, info in sorted(symbols.items(), key=lambda item: (int(item[1]["birth_order"]), item[0]))
            ],
        },
        "nodes": analysed_nodes,
    }
    return report


def analyze_trace_utilization(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """US spelling alias for ``analyse_trace_utilization``."""
    return analyse_trace_utilization(*args, **kwargs)


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _format_top_rows(rows: list[tuple[str, dict[str, Any]]], *, title: str) -> list[str]:
    lines = [f"## {title}", "", "| Name | Nodes | Compute Utilization | Useful FLOPs | Scheduled FLOPs |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, metrics in rows:
        lines.append(
            "| "
            f"{name} | "
            f"{metrics['node_count']} | "
            f"{_format_ratio(metrics.get('compute_utilization'))} | "
            f"{metrics['useful_flops']} | "
            f"{metrics['scheduled_flops']} |"
        )
    if len(rows) == 0:
        lines.append("| (none) | 0 | N/A | 0 | 0 |")
    return lines


def render_markdown_report(report: dict[str, Any], *, top_k: int = 12) -> str:
    summary = report["summary"]
    memory = report["memory"]
    model_info = report.get("model_info", {})
    hardware = report.get("hardware", {})
    top_nodes = sorted(
        report.get("nodes", []),
        key=lambda node: (
            int(node["compute"]["scheduled_flops"]),
            int(node["memory"]["live_bytes_after_alloc"]),
        ),
        reverse=True,
    )[:top_k]

    lines = [
        "# Utilization Report",
        "",
        "## Model Context",
        "",
        f"- Model: {model_info.get('model_name', 'Unknown')}",
        f"- Architecture: {model_info.get('architecture', 'Unknown')}",
        f"- Trace Nodes: {summary['node_count_total']}",
        f"- Analysed Nodes: {summary['node_count_analysed']}",
        f"- Symbols: {summary['symbol_count']}",
        "",
        "## Hardware",
        "",
        f"- Tile M: {hardware.get('tile_m')}",
        f"- Tile K: {hardware.get('tile_k')}",
        f"- Tile N: {hardware.get('tile_n')}",
        f"- Memory Capacity: {summary.get('memory_capacity_human') or 'N/A'}",
        "",
        "## Summary",
        "",
        f"- Overall Compute Utilization: {_format_ratio(summary.get('overall_compute_utilization'))}",
        f"- Useful FLOPs: {summary['total_useful_flops']}",
        f"- Scheduled FLOPs: {summary['total_scheduled_flops']}",
        f"- Peak Live Memory: {summary['peak_live_bytes_human']}",
        f"- Peak Memory Utilization: {_format_ratio(summary.get('memory_utilization_ratio'))}",
        f"- Peak Live Step: {memory.get('peak_live_step')}",
        "",
    ]

    lines.extend(_format_top_rows(list(summary["by_op_family"].items()), title="By Op Family"))
    lines.extend([""])
    lines.extend(_format_top_rows(list(summary["by_semantic_group"].items()), title="By Semantic Group"))
    lines.extend([""])
    lines.extend(_format_top_rows(list(summary["by_source_branch"].items()), title="By Source Branch"))
    lines.extend(["", "## Hot Nodes", "", "| Order | Name | Type | Semantic Group | Compute Utilization | Live Bytes After Alloc |", "| ---: | --- | --- | --- | ---: | ---: |"])

    for node in top_nodes:
        lines.append(
            "| "
            f"{node['order']} | "
            f"{node['name'] or '(root)'} | "
            f"{node['type']} | "
            f"{node['semantic_group']} | "
            f"{_format_ratio(node['compute'].get('compute_utilization'))} | "
            f"{node['memory']['live_bytes_after_alloc']} |"
        )
    if len(top_nodes) == 0:
        lines.append("| 0 | (none) | N/A | other | N/A | 0 |")

    return "\n".join(lines) + "\n"


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_output_path(fmt: str) -> Path:
    suffix = "json" if fmt == "json" else "md"
    return _module_dir() / "outputs" / f"report.{suffix}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a compute/memory utilization report from a VLM trace.")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct", help="Model name or path.")
    parser.add_argument(
        "--image",
        default=str(_module_dir() / "inputs" / "img" / "image.png"),
        help="Image path used to build Qwen3-VL inputs.",
    )
    parser.add_argument("--text", default="Describe this image.", help="Prompt text for the VLM input template.")
    parser.add_argument("--format", choices=("md", "json"), default="md", help="Report output format.")
    parser.add_argument("--output", default=None, help="Output file path. Defaults to outputs/report.<format>.")
    parser.add_argument("--tile-m", type=int, default=64, help="Hardware tile size M.")
    parser.add_argument("--tile-k", type=int, default=64, help="Hardware tile size K.")
    parser.add_argument("--tile-n", type=int, default=64, help="Hardware tile size N.")
    parser.add_argument(
        "--memory-capacity-bytes",
        type=int,
        default=None,
        help="Optional hardware activation-memory budget for memory utilization reporting.",
    )
    parser.add_argument(
        "--include-non-leaf",
        action="store_true",
        help="Include non-leaf modules in compute summaries.",
    )
    parser.add_argument(
        "--drop-outputs-live",
        action="store_true",
        help="Do not keep final output symbols live until the end of the trace.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of hottest nodes to include in the Markdown report.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    parser = VLMModelParser(args.model)
    parser.load_model()
    if parser.processor is None:
        raise RuntimeError(
            "utilization_report main currently expects a processor-backed VLM model "
            "that can build inputs from --image/--text."
        )
    inputs = template_qwen3_vl_inputs(parser.processor, args.image, text=args.text)
    model = parser.model.model if hasattr(parser.model, "model") else parser.model

    trace_tree = parser.trace_leaf_modules(
        model,
        {**inputs, "use_cache": False, "return_dict": False},
    )
    model_info = parser.extract_model_info(inputs=inputs)

    report = analyse_trace_utilization(
        trace_tree,
        model_info=model_info,
        hw={
            "tile_m": args.tile_m,
            "tile_k": args.tile_k,
            "tile_n": args.tile_n,
            "memory_capacity_bytes": args.memory_capacity_bytes,
        },
        include_non_leaf=args.include_non_leaf,
        keep_model_outputs_live=not args.drop_outputs_live,
    )

    output_path = Path(args.output).expanduser() if args.output else _default_output_path(args.format)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        output_path.write_text(json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")
    else:
        output_path.write_text(render_markdown_report(report, top_k=args.top_k), encoding="utf-8")

    print(f"Trace nodes: {report['summary']['node_count_total']}")
    print(f"Peak live memory: {report['summary']['peak_live_bytes_human']}")
    print(f"Overall compute utilization: {_format_ratio(report['summary']['overall_compute_utilization'])}")
    print(f"Report written to: {output_path}")
    return 0


__all__ = [
    "DEFAULT_HW",
    "analyse_trace_utilization",
    "analyze_trace_utilization",
    "main",
    "render_markdown_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
