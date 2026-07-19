"""Local-cache partial loader for one Qwen3-MoE decoder layer.

This loader intentionally does not construct ``Qwen3MoeForCausalLM``.  It
loads the common tensors needed by one layer and leaves fused expert tensors
behind a slicing provider.  The resulting lightweight object implements the
small attribute surface consumed by :func:`compile_native_hf_decoder`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from compiler.aten.model_extract import MoeExpertWeights


@dataclass
class _WeightModule:
    weight: torch.Tensor


@dataclass
class _NormModule:
    weight: torch.Tensor
    eps: float

    @property
    def variance_epsilon(self) -> float:
        return self.eps


class SafetensorsMoeExpertProvider:
    """Read one fused expert slice from local safetensors shards on demand."""

    def __init__(
        self,
        *,
        gate_up_path: Path,
        gate_up_name: str,
        down_path: Path,
        down_name: str,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
    ):
        self.gate_up_path = gate_up_path
        self.gate_up_name = gate_up_name
        self.down_path = down_path
        self.down_name = down_name
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.materialized: list[int] = []

    @staticmethod
    def _slice(path: Path, tensor_name: str, expert_id: int) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            tensor_slice = handle.get_slice(tensor_name)
            return tensor_slice[expert_id : expert_id + 1].squeeze(0).contiguous()

    def materialize(self, expert_id: int) -> MoeExpertWeights:
        if not 0 <= expert_id < self.num_experts:
            raise IndexError(
                f"expert_id={expert_id} outside [0, {self.num_experts})"
            )
        self.materialized.append(expert_id)
        fused = self._slice(self.gate_up_path, self.gate_up_name, expert_id)
        down = self._slice(self.down_path, self.down_name, expert_id)
        expected_fused = (2 * self.intermediate_size, self.hidden_size)
        expected_down = (self.hidden_size, self.intermediate_size)
        if tuple(fused.shape) != expected_fused or tuple(down.shape) != expected_down:
            raise ValueError(
                f"Unexpected expert {expert_id} shapes: gate_up={tuple(fused.shape)} "
                f"(expected {expected_fused}), down={tuple(down.shape)} "
                f"(expected {expected_down})"
            )
        gate, up = fused.split(self.intermediate_size, dim=0)
        return MoeExpertWeights(
            w_gate=gate.T.contiguous(),
            w_up=up.T.contiguous(),
            w_down=down.T.contiguous(),
        )


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _resolve_metadata_root(model_id_or_path: str | Path, *, local_files_only: bool) -> Path:
    path = Path(model_id_or_path).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                str(model_id_or_path),
                allow_patterns=["config.json", "model.safetensors.index.json"],
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        mode = "local Hugging Face cache" if local_files_only else "Hugging Face Hub"
        raise FileNotFoundError(
            f"Cannot resolve {model_id_or_path!s} config/index from {mode}. "
            "Cache config.json and model.safetensors.index.json first."
        ) from exc


def _resolve_shard(
    metadata_root: Path,
    *,
    model_id_or_path: str | Path,
    filename: str,
    local_files_only: bool,
) -> Path:
    direct = metadata_root / filename
    if direct.exists():
        return direct.resolve()
    if Path(model_id_or_path).expanduser().is_dir():
        raise FileNotFoundError(
            f"Required Qwen3-MoE shard is missing: {direct}. "
            "Copy/cache the shard referenced by model.safetensors.index.json."
        )
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                repo_id=str(model_id_or_path),
                filename=filename,
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Required Qwen3-MoE shard {filename!r} is not in the local cache. "
            "Download the layer shard before running partial compilation."
        ) from exc


def load_qwen3_moe_partial_model(
    model_id_or_path: str | Path,
    *,
    layer_idx: int = 0,
    local_files_only: bool = True,
):
    """Build a lightweight one-layer Qwen3-MoE model from local shards."""
    metadata_root = _resolve_metadata_root(
        model_id_or_path, local_files_only=local_files_only
    )
    config_path = metadata_root / "config.json"
    index_path = metadata_root / "model.safetensors.index.json"
    if not config_path.exists() or not index_path.exists():
        raise FileNotFoundError(
            f"Expected config.json and model.safetensors.index.json under {metadata_root}"
        )
    config_dict = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    config = _namespace(config_dict)
    prefix = f"model.layers.{layer_idx}"

    shard_cache: dict[str, Path] = {}

    def shard_for(tensor_name: str) -> Path:
        try:
            filename = weight_map[tensor_name]
        except KeyError as exc:
            raise KeyError(
                f"Tensor {tensor_name!r} is absent from model.safetensors.index.json"
            ) from exc
        if filename not in shard_cache:
            shard_cache[filename] = _resolve_shard(
                metadata_root,
                model_id_or_path=model_id_or_path,
                filename=filename,
                local_files_only=local_files_only,
            )
        return shard_cache[filename]

    def load_tensor(name: str) -> torch.Tensor:
        from safetensors import safe_open

        path = shard_for(name)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).contiguous()

    def linear(suffix: str) -> _WeightModule:
        return _WeightModule(load_tensor(f"{prefix}.{suffix}.weight"))

    eps = float(getattr(config, "rms_norm_eps", 1e-6))
    attention = SimpleNamespace(
        q_proj=linear("self_attn.q_proj"),
        k_proj=linear("self_attn.k_proj"),
        v_proj=linear("self_attn.v_proj"),
        o_proj=linear("self_attn.o_proj"),
        q_norm=_NormModule(
            load_tensor(f"{prefix}.self_attn.q_norm.weight"), eps
        ),
        k_norm=_NormModule(
            load_tensor(f"{prefix}.self_attn.k_norm.weight"), eps
        ),
    )
    gate_up_name = f"{prefix}.mlp.experts.gate_up_proj"
    down_name = f"{prefix}.mlp.experts.down_proj"
    provider = SafetensorsMoeExpertProvider(
        gate_up_path=shard_for(gate_up_name),
        gate_up_name=gate_up_name,
        down_path=shard_for(down_name),
        down_name=down_name,
        num_experts=int(config.num_experts),
        hidden_size=int(config.hidden_size),
        intermediate_size=int(config.moe_intermediate_size),
    )
    experts = SimpleNamespace(_plena_expert_provider=provider)
    mlp = SimpleNamespace(gate=linear("mlp.gate"), experts=experts)
    layer = SimpleNamespace(
        self_attn=attention,
        mlp=mlp,
        input_layernorm=_NormModule(
            load_tensor(f"{prefix}.input_layernorm.weight"), eps
        ),
        post_attention_layernorm=_NormModule(
            load_tensor(f"{prefix}.post_attention_layernorm.weight"), eps
        ),
    )

    final_norm_name = "model.norm.weight"
    final_norm = (
        _NormModule(load_tensor(final_norm_name), eps)
        if final_norm_name in weight_map
        else None
    )
    return SimpleNamespace(
        config=config,
        layers=[layer],
        norm=final_norm,
        _plena_expert_provider=provider,
        _plena_source_layer_idx=layer_idx,
        _plena_partial_model=True,
        _plena_metadata_root=str(metadata_root),
    )


__all__ = [
    "SafetensorsMoeExpertProvider",
    "load_qwen3_moe_partial_model",
]
