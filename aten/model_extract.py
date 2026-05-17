"""HuggingFace decoder model extraction helpers for the PLENA frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelConfig:
    """Decoder dimensions needed by the PLENA ATen frontend."""

    hidden_size: int
    inter_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    eps: float
    rope_theta: float
    vocab_size: int | None
    model_type: str

    @property
    def total_q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def head_ratio(self) -> int:
        return self.num_heads // self.num_kv_heads


@dataclass(frozen=True)
class LayerWeights:
    """One decoder layer in PLENA's (in, out) linear-weight convention."""

    w_q: torch.Tensor
    w_o: torch.Tensor
    w_k_heads: list[torch.Tensor]
    w_v_heads: list[torch.Tensor]
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor
    eps: float

    def tensor_entries(self, layer_idx: int) -> list[tuple[str, torch.Tensor]]:
        entries = [
            (f"W_q_{layer_idx}", self.w_q),
            (f"W_o_{layer_idx}", self.w_o),
        ]
        for kv_h, (w_k, w_v) in enumerate(zip(self.w_k_heads, self.w_v_heads, strict=True)):
            entries.extend(
                [
                    (f"W_k_{layer_idx}_h{kv_h}", w_k),
                    (f"W_v_{layer_idx}_h{kv_h}", w_v),
                ]
            )
        entries.extend(
            [
                (f"W_gate_{layer_idx}", self.w_gate),
                (f"W_up_{layer_idx}", self.w_up),
                (f"W_down_{layer_idx}", self.w_down),
            ]
        )
        return entries


def find_model_root(model: Any) -> Any:
    """Find the transformer backbone (model.model or model.model.text_model)."""
    for candidate in [
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "text_model", None),
        getattr(model, "language_model", getattr(model, "text_model", None)),
    ]:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise ValueError(f"Cannot find decoder layers on {type(model).__name__}")


def embedding_module(root: Any) -> Any | None:
    """Return the token embedding module when the backbone exposes one."""
    return getattr(root, "embed_tokens", getattr(root, "wte", None))


def extract_model_config(model: Any) -> ModelConfig:
    """Extract decoder dimensions, resolving text_config for VLM wrappers."""
    config = getattr(model.config, "text_config", model.config)
    hidden = config.hidden_size
    num_heads = config.num_attention_heads
    return ModelConfig(
        hidden_size=hidden,
        inter_dim=getattr(config, "intermediate_size", 4 * hidden),
        num_heads=num_heads,
        num_kv_heads=getattr(config, "num_key_value_heads", num_heads),
        head_dim=hidden // num_heads,
        eps=getattr(config, "rms_norm_eps", 1e-5),
        rope_theta=getattr(config, "rope_theta", 10000.0),
        vocab_size=getattr(config, "vocab_size", None),
        model_type=getattr(config, "model_type", "unknown"),
    )


def extract_layer_weights(layer: Any, config: ModelConfig) -> LayerWeights:
    """Extract one decoder layer in PLENA's (in, out) convention."""
    hidden = config.hidden_size
    total_kv_dim = config.num_kv_heads * config.head_dim
    w_k_full = _linear_weight(layer.self_attn.k_proj, hidden, total_kv_dim)
    w_v_full = _linear_weight(layer.self_attn.v_proj, hidden, total_kv_dim)
    norm = layer.input_layernorm

    return LayerWeights(
        w_q=_linear_weight(layer.self_attn.q_proj, hidden, config.total_q_dim),
        w_o=_linear_weight(layer.self_attn.o_proj, config.total_q_dim, hidden),
        w_k_heads=_split_heads(w_k_full, config.head_dim, config.num_kv_heads),
        w_v_heads=_split_heads(w_v_full, config.head_dim, config.num_kv_heads),
        w_gate=_linear_weight(layer.mlp.gate_proj, hidden, config.inter_dim),
        w_up=_linear_weight(layer.mlp.up_proj, hidden, config.inter_dim),
        w_down=_linear_weight(layer.mlp.down_proj, config.inter_dim, hidden),
        eps=getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)),
    )


def _linear_weight(module: Any, rows: int, cols: int) -> torch.Tensor:
    """HF Linear stores (out, in); PLENA uses (in, out)."""
    return module.weight.detach().T.contiguous()[:rows, :cols]


def _split_heads(weight: torch.Tensor, head_dim: int, num_heads: int) -> list[torch.Tensor]:
    return [weight[:, h * head_dim:(h + 1) * head_dim].contiguous() for h in range(num_heads)]
