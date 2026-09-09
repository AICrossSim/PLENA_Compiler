"""HuggingFace decoder model extraction helpers for the PLENA frontend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

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
    qk_norm: bool = False
    num_hidden_layers: int = 0
    dense_inter_dim: int | None = None
    moe_inter_dim: int | None = None
    num_experts: int = 0
    experts_per_token: int = 0
    norm_topk_prob: bool = False
    decoder_sparse_step: int = 1
    mlp_only_layers: tuple[int, ...] = ()

    @property
    def total_q_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def head_ratio(self) -> int:
        return self.num_heads // self.num_kv_heads

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.num_experts <= 0 or layer_idx in self.mlp_only_layers:
            return False
        return (layer_idx + 1) % self.decoder_sparse_step == 0


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
    input_norm_weight: torch.Tensor | None = None
    post_attn_norm_weight: torch.Tensor | None = None
    q_norm_weight: torch.Tensor | None = None
    k_norm_weight: torch.Tensor | None = None

    def tensor_entries(self, layer_idx: int) -> list[tuple[str, torch.Tensor]]:
        entries = [
            (f"W_q_{layer_idx}", self.w_q),
            (f"W_o_{layer_idx}", self.w_o),
        ]
        if len(self.w_k_heads) != len(self.w_v_heads):
            raise ValueError(
                f"w_k_heads/w_v_heads length mismatch: {len(self.w_k_heads)} != {len(self.w_v_heads)}"
            )
        for kv_h, (w_k, w_v) in enumerate(zip(self.w_k_heads, self.w_v_heads)):
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
        for suffix, tensor in (
            ("input_norm", self.input_norm_weight),
            ("post_attn_norm", self.post_attn_norm_weight),
            ("q_norm", self.q_norm_weight),
            ("k_norm", self.k_norm_weight),
        ):
            if tensor is not None:
                entries.append((f"W_{suffix}_{layer_idx}", tensor))
        return entries


@dataclass(frozen=True)
class MoeExpertWeights:
    """One routed expert in PLENA's ``(input, output)`` convention."""

    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


class MoeExpertProvider(Protocol):
    num_experts: int
    hidden_size: int
    intermediate_size: int

    def materialize(self, expert_id: int) -> MoeExpertWeights: ...


@dataclass
class TensorMoeExpertProvider:
    """Lazy views over Qwen3's two fused expert tensors."""

    gate_up_proj: torch.Tensor
    down_proj: torch.Tensor
    num_experts: int
    hidden_size: int
    intermediate_size: int

    def materialize(self, expert_id: int) -> MoeExpertWeights:
        if not 0 <= expert_id < self.num_experts:
            raise IndexError(
                f"expert_id={expert_id} outside [0, {self.num_experts})"
            )
        fused = self.gate_up_proj[expert_id]
        down = self.down_proj[expert_id]
        expected_fused = (2 * self.intermediate_size, self.hidden_size)
        expected_down = (self.hidden_size, self.intermediate_size)
        if tuple(fused.shape) != expected_fused or tuple(down.shape) != expected_down:
            raise ValueError(
                f"expert {expert_id} shapes gate_up={tuple(fused.shape)}, "
                f"down={tuple(down.shape)}; expected {expected_fused}, {expected_down}"
            )
        gate, up = fused.split(self.intermediate_size, dim=0)
        return MoeExpertWeights(
            w_gate=gate.detach().T.contiguous(),
            w_up=up.detach().T.contiguous(),
            w_down=down.detach().T.contiguous(),
        )


@dataclass(frozen=True)
class MoeLayerWeights:
    """Attention and routed weights; router rows are ``(expert, hidden)``."""

    w_q: torch.Tensor
    w_o: torch.Tensor
    w_k_heads: list[torch.Tensor]
    w_v_heads: list[torch.Tensor]
    w_router_rows: torch.Tensor
    expert_provider: MoeExpertProvider
    eps: float
    input_norm_weight: torch.Tensor | None = None
    post_attn_norm_weight: torch.Tensor | None = None
    q_norm_weight: torch.Tensor | None = None
    k_norm_weight: torch.Tensor | None = None

    def tensor_entries(self, layer_idx: int) -> list[tuple[str, torch.Tensor]]:
        entries = [(f"W_q_{layer_idx}", self.w_q), (f"W_o_{layer_idx}", self.w_o)]
        for kv_head, (w_k, w_v) in enumerate(
            zip(self.w_k_heads, self.w_v_heads, strict=True)
        ):
            entries.extend(
                [
                    (f"W_k_{layer_idx}_h{kv_head}", w_k),
                    (f"W_v_{layer_idx}_h{kv_head}", w_v),
                ]
            )
        entries.append((f"W_router_rows_{layer_idx}", self.w_router_rows))
        # Runtime routing can select any expert, so every expert must remain
        # addressable.  There is intentionally no active-expert subset API.
        for expert_id in range(self.expert_provider.num_experts):
            expert = self.expert_provider.materialize(expert_id)
            entries.extend(
                [
                    (f"W_expert_gate_{layer_idx}_e{expert_id}", expert.w_gate),
                    (f"W_expert_up_{layer_idx}_e{expert_id}", expert.w_up),
                    (f"W_expert_down_{layer_idx}_e{expert_id}", expert.w_down),
                ]
            )
        for suffix, tensor in (
            ("input_norm", self.input_norm_weight),
            ("post_attn_norm", self.post_attn_norm_weight),
            ("q_norm", self.q_norm_weight),
            ("k_norm", self.k_norm_weight),
        ):
            if tensor is not None:
                entries.append((f"W_{suffix}_{layer_idx}", tensor))
        return entries


DecoderLayerWeights: TypeAlias = LayerWeights | MoeLayerWeights


@dataclass(frozen=True)
class VisionConfig:
    """Vision encoder dimensions needed by the PLENA ATen frontend."""

    hidden_size: int
    inter_dim: int
    num_heads: int
    head_dim: int
    eps: float
    image_size: int
    patch_size: int
    num_channels: int
    model_type: str

    @property
    def total_q_dim(self) -> int:
        return self.num_heads * self.head_dim


@dataclass(frozen=True)
class VisionPatchWeights:
    """Patch embedding weights in PLENA im2col convention."""

    weight_2d: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True)
class VisionLayerWeights:
    """One SigLIP/ViT vision layer in PLENA's (in, out) convention."""

    w_q: torch.Tensor
    w_k: torch.Tensor
    w_v: torch.Tensor
    w_o: torch.Tensor
    b_q: torch.Tensor
    b_k: torch.Tensor
    b_v: torch.Tensor
    b_o: torch.Tensor
    w_fc1: torch.Tensor
    w_fc2: torch.Tensor
    b_fc1: torch.Tensor
    b_fc2: torch.Tensor
    ln1_weight: torch.Tensor
    ln1_bias: torch.Tensor
    ln2_weight: torch.Tensor
    ln2_bias: torch.Tensor
    eps: float

    def tensor_entries(self, layer_idx: int) -> list[tuple[str, torch.Tensor]]:
        return [
            (f"V_W_q_{layer_idx}", self.w_q),
            (f"V_W_k_{layer_idx}", self.w_k),
            (f"V_W_v_{layer_idx}", self.w_v),
            (f"V_W_o_{layer_idx}", self.w_o),
            (f"V_B_q_{layer_idx}", self.b_q),
            (f"V_B_k_{layer_idx}", self.b_k),
            (f"V_B_v_{layer_idx}", self.b_v),
            (f"V_B_o_{layer_idx}", self.b_o),
            (f"V_W_fc1_{layer_idx}", self.w_fc1),
            (f"V_W_fc2_{layer_idx}", self.w_fc2),
            (f"V_B_fc1_{layer_idx}", self.b_fc1),
            (f"V_B_fc2_{layer_idx}", self.b_fc2),
            (f"V_LN1_weight_{layer_idx}", self.ln1_weight),
            (f"V_LN1_bias_{layer_idx}", self.ln1_bias),
            (f"V_LN2_weight_{layer_idx}", self.ln2_weight),
            (f"V_LN2_bias_{layer_idx}", self.ln2_bias),
        ]


@dataclass(frozen=True)
class VisionPostNormWeights:
    weight: torch.Tensor
    bias: torch.Tensor
    eps: float

    def tensor_entries(self) -> list[tuple[str, torch.Tensor]]:
        return [
            ("V_POST_LN_weight", self.weight),
            ("V_POST_LN_bias", self.bias),
        ]


@dataclass(frozen=True)
class VisionConnectorWeights:
    """SmolVLM-style vision-to-text connector projection."""

    weight: torch.Tensor
    bias: torch.Tensor | None
    scale_factor: int
    input_dim: int
    output_dim: int


def find_model_root(model: Any) -> Any:
    """Find the transformer backbone (model.model or model.model.text_model).

    Handles standard CausalLM models, VLMs like SmolVLM2, diffusion models like LLaDA,
    and direct-wrapper models like Fast_dLLM_QwenModel.
    """
    # Check the model itself first, then common nested attributes
    candidates = [
        model,  # Direct wrapper models (e.g., Fast_dLLM_QwenModel)
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "text_model", None),
        getattr(model, "language_model", getattr(model, "text_model", None)),
    ]
    for candidate in candidates:
        if candidate is not None:
            # Standard case: .layers attribute
            if hasattr(candidate, "layers"):
                return candidate
            # LLaDA-style: transformer.blocks instead of layers
            if hasattr(candidate, "transformer") and hasattr(
                candidate.transformer, "blocks"
            ):
                # Create a wrapper that exposes .layers for compatibility
                class LayersWrapper:
                    def __init__(self, obj):
                        self._obj = obj

                    def __getattr__(self, name):
                        if name == "layers":
                            return self._obj.transformer.blocks
                        return getattr(self._obj, name)

                return LayersWrapper(candidate)
    raise ValueError(f"Cannot find decoder layers on {type(model).__name__}")


def embedding_module(root: Any) -> Any | None:
    """Return the token embedding module when the backbone exposes one."""
    return getattr(root, "embed_tokens", getattr(root, "wte", None))


def find_vision_model(model: Any) -> Any:
    """Find the HF vision tower on a multimodal wrapper."""
    for candidate in [
        getattr(model, "vision_model", None),
        getattr(getattr(model, "model", None), "vision_model", None),
    ]:
        if candidate is not None:
            return candidate
    raise ValueError(f"Cannot find vision_model on {type(model).__name__}")


def find_vision_connector(model: Any) -> Any | None:
    """Find an optional VLM connector after the vision tower."""
    for candidate in [
        getattr(model, "connector", None),
        getattr(getattr(model, "model", None), "connector", None),
    ]:
        if candidate is not None:
            return candidate
    return None


def extract_model_config(model: Any) -> ModelConfig:
    """Extract decoder dimensions, resolving text_config for VLM wrappers."""
    # Robust config lookup for custom wrappers
    cfg = getattr(model, "config", None)
    if cfg is None:
        cfg = getattr(getattr(model, "model", None), "config", None)
    if cfg is None:
        raise ValueError(
            f"Cannot find .config on {type(model).__name__} or its submodules"
        )

    config = getattr(cfg, "text_config", cfg)
    hidden = config.hidden_size
    num_heads = config.num_attention_heads
    dense_inter_dim = getattr(config, "intermediate_size", None)
    if dense_inter_dim is None:
        dense_inter_dim = getattr(config, "mlp_hidden_size", None)
    moe_inter_dim = getattr(config, "moe_intermediate_size", None)
    num_experts = int(getattr(config, "num_experts", 0) or 0)
    inter_dim = (
        moe_inter_dim
        if num_experts > 0 and moe_inter_dim is not None
        else dense_inter_dim
    )
    if inter_dim is None:
        inter_dim = 4 * hidden
    if dense_inter_dim is None:
        dense_inter_dim = inter_dim
    num_kv_heads = getattr(config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = getattr(config, "n_kv_heads", num_heads)
    model_type = str(getattr(config, "model_type", "unknown"))
    qk_norm = getattr(config, "qk_norm", None)
    if qk_norm is None:
        qk_norm = getattr(config, "use_qk_norm", None)
    if qk_norm is None:
        qk_norm = model_type in {"qwen3", "qwen3_moe"}
    return ModelConfig(
        hidden_size=hidden,
        inter_dim=inter_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=int(getattr(config, "head_dim", hidden // num_heads)),
        eps=getattr(config, "rms_norm_eps", 1e-5),
        rope_theta=getattr(config, "rope_theta", 10000.0),
        vocab_size=getattr(config, "vocab_size", None),
        model_type=model_type,
        qk_norm=bool(qk_norm),
        num_hidden_layers=int(getattr(config, "num_hidden_layers", 0) or 0),
        dense_inter_dim=int(dense_inter_dim),
        moe_inter_dim=None if moe_inter_dim is None else int(moe_inter_dim),
        num_experts=num_experts,
        experts_per_token=int(getattr(config, "num_experts_per_tok", 0) or 0),
        norm_topk_prob=bool(getattr(config, "norm_topk_prob", False)),
        decoder_sparse_step=int(getattr(config, "decoder_sparse_step", 1) or 1),
        mlp_only_layers=tuple(
            int(index) for index in (getattr(config, "mlp_only_layers", None) or ())
        ),
    )


def extract_vision_config(model: Any) -> VisionConfig:
    """Extract SigLIP/ViT vision dimensions from a VLM wrapper."""
    config = getattr(model.config, "vision_config", None)
    if config is None:
        vision_model = find_vision_model(model)
        config = getattr(vision_model, "config", None)
    if config is None:
        raise ValueError(f"Cannot find vision_config on {type(model).__name__}")

    hidden = int(config.hidden_size)
    num_heads = int(config.num_attention_heads)
    head_dim = int(getattr(config, "head_dim", hidden // num_heads))
    return VisionConfig(
        hidden_size=hidden,
        inter_dim=int(getattr(config, "intermediate_size", 4 * hidden)),
        num_heads=num_heads,
        head_dim=head_dim,
        eps=float(getattr(config, "layer_norm_eps", getattr(config, "norm_eps", 1e-6))),
        image_size=int(getattr(config, "image_size", 224)),
        patch_size=int(getattr(config, "patch_size", 16)),
        num_channels=int(getattr(config, "num_channels", 3)),
        model_type=str(getattr(config, "model_type", "vision")),
    )


def extract_layer_weights(layer: Any, config: ModelConfig) -> DecoderLayerWeights:
    """Extract one decoder layer in PLENA's (in, out) convention.

    Supports both standard Llama/Qwen layers (with self_attn and mlp) and
    LLaDA-style layers (with direct projections).
    """
    if is_qwen3_moe_layer(layer):
        return _extract_qwen3_moe_layer_weights(layer, config)
    if config.model_type == "qwen3_moe":
        raise ValueError(
            "Qwen3-MoE extraction requires the Transformers 5.5.0 fused "
            "Qwen3MoeExperts ABI (gate_up_proj/down_proj); dense or "
            "ModuleList-expert fallback is forbidden"
        )

    # Detect layer type
    is_llada = hasattr(layer, "q_proj") and not hasattr(layer, "self_attn")

    if is_llada:
        return _extract_llada_layer_weights(layer, config)
    else:
        return _extract_llama_layer_weights(layer, config)


def extract_vision_patch_weights(
    vision_model: Any, config: VisionConfig
) -> VisionPatchWeights:
    """Extract Conv2d patch embedding as im2col weight + bias."""
    patch = vision_model.embeddings.patch_embedding
    weight = (
        patch.weight.detach().float().reshape(config.hidden_size, -1).T.contiguous()
    )
    bias = _linear_bias(patch, config.hidden_size)
    return VisionPatchWeights(weight_2d=weight, bias=bias)


def extract_vision_layer_weights(
    layer: Any, config: VisionConfig
) -> VisionLayerWeights:
    """Extract one SigLIP/ViT encoder layer."""
    hidden = config.hidden_size
    inter = config.inter_dim
    attn = layer.self_attn
    mlp = layer.mlp
    return VisionLayerWeights(
        w_q=_linear_weight(attn.q_proj, hidden, config.total_q_dim),
        w_k=_linear_weight(attn.k_proj, hidden, config.total_q_dim),
        w_v=_linear_weight(attn.v_proj, hidden, config.total_q_dim),
        w_o=_linear_weight(attn.out_proj, config.total_q_dim, hidden),
        b_q=_linear_bias(attn.q_proj, config.total_q_dim),
        b_k=_linear_bias(attn.k_proj, config.total_q_dim),
        b_v=_linear_bias(attn.v_proj, config.total_q_dim),
        b_o=_linear_bias(attn.out_proj, hidden),
        w_fc1=_linear_weight(mlp.fc1, hidden, inter),
        w_fc2=_linear_weight(mlp.fc2, inter, hidden),
        b_fc1=_linear_bias(mlp.fc1, inter),
        b_fc2=_linear_bias(mlp.fc2, hidden),
        ln1_weight=layer.layer_norm1.weight.detach().float().contiguous()[:hidden],
        ln1_bias=layer.layer_norm1.bias.detach().float().contiguous()[:hidden],
        ln2_weight=layer.layer_norm2.weight.detach().float().contiguous()[:hidden],
        ln2_bias=layer.layer_norm2.bias.detach().float().contiguous()[:hidden],
        eps=float(getattr(layer.layer_norm1, "eps", config.eps)),
    )


def extract_vision_post_norm_weights(
    vision_model: Any, config: VisionConfig
) -> VisionPostNormWeights:
    norm = vision_model.post_layernorm
    hidden = config.hidden_size
    return VisionPostNormWeights(
        weight=norm.weight.detach().float().contiguous()[:hidden],
        bias=norm.bias.detach().float().contiguous()[:hidden],
        eps=float(getattr(norm, "eps", config.eps)),
    )


def extract_vision_connector_weights(
    model: Any, config: VisionConfig
) -> VisionConnectorWeights | None:
    connector = find_vision_connector(model)
    if connector is None:
        return None

    projection = getattr(getattr(connector, "modality_projection", None), "proj", None)
    if projection is None:
        projection = getattr(connector, "modality_projection", None)
    if projection is None or not hasattr(projection, "weight"):
        raise ValueError(
            f"Unsupported vision connector on {type(model).__name__}: missing modality projection"
        )

    scale_factor = int(getattr(connector, "scale_factor", 1))
    input_dim = int(
        getattr(
            projection, "in_features", config.hidden_size * scale_factor * scale_factor
        )
    )
    output_dim = int(getattr(projection, "out_features", config.hidden_size))
    expected_input = config.hidden_size * scale_factor * scale_factor
    if input_dim != expected_input:
        raise ValueError(
            f"Unsupported vision connector input dim {input_dim}; expected "
            f"hidden_size({config.hidden_size}) * scale_factor({scale_factor})^2 = {expected_input}"
        )

    bias = getattr(projection, "bias", None)
    return VisionConnectorWeights(
        weight=_linear_weight(projection, input_dim, output_dim),
        bias=None if bias is None else bias.detach().float().contiguous()[:output_dim],
        scale_factor=scale_factor,
        input_dim=input_dim,
        output_dim=output_dim,
    )


def _extract_llama_layer_weights(layer: Any, config: ModelConfig) -> LayerWeights:
    """Extract standard Llama/Qwen layer weights."""
    hidden = config.hidden_size
    inter_dim = config.dense_inter_dim or config.inter_dim
    total_kv_dim = config.num_kv_heads * config.head_dim
    w_k_full = _linear_weight(layer.self_attn.k_proj, hidden, total_kv_dim)
    w_v_full = _linear_weight(layer.self_attn.v_proj, hidden, total_kv_dim)
    norm = layer.input_layernorm

    return LayerWeights(
        w_q=_linear_weight(layer.self_attn.q_proj, hidden, config.total_q_dim),
        w_o=_linear_weight(layer.self_attn.o_proj, config.total_q_dim, hidden),
        w_k_heads=_split_heads(w_k_full, config.head_dim, config.num_kv_heads),
        w_v_heads=_split_heads(w_v_full, config.head_dim, config.num_kv_heads),
        w_gate=_linear_weight(layer.mlp.gate_proj, hidden, inter_dim),
        w_up=_linear_weight(layer.mlp.up_proj, hidden, inter_dim),
        w_down=_linear_weight(layer.mlp.down_proj, inter_dim, hidden),
        eps=getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)),
        input_norm_weight=_optional_norm_weight(layer, "input_layernorm"),
        post_attn_norm_weight=_optional_norm_weight(
            layer, "post_attention_layernorm"
        ),
        q_norm_weight=_optional_norm_weight(layer.self_attn, "q_norm"),
        k_norm_weight=_optional_norm_weight(layer.self_attn, "k_norm"),
    )


def is_qwen3_moe_layer(layer: Any) -> bool:
    mlp = getattr(layer, "mlp", None)
    experts = getattr(mlp, "experts", None)
    return bool(
        hasattr(layer, "self_attn")
        and mlp is not None
        and hasattr(mlp, "gate")
        and experts is not None
        and hasattr(experts, "gate_up_proj")
        and hasattr(experts, "down_proj")
    )


def _extract_qwen3_moe_layer_weights(
    layer: Any, config: ModelConfig
) -> MoeLayerWeights:
    if config.model_type != "qwen3_moe":
        raise ValueError(
            f"Qwen3-MoE layer found under model_type={config.model_type!r}"
        )
    if config.num_experts <= 0 or config.moe_inter_dim is None:
        raise ValueError("Qwen3-MoE config is missing expert dimensions")
    hidden = config.hidden_size
    total_kv_dim = config.num_kv_heads * config.head_dim
    attention = layer.self_attn
    experts = layer.mlp.experts
    provider = TensorMoeExpertProvider(
        gate_up_proj=experts.gate_up_proj,
        down_proj=experts.down_proj,
        num_experts=config.num_experts,
        hidden_size=hidden,
        intermediate_size=config.moe_inter_dim,
    )
    w_k = _linear_weight(attention.k_proj, hidden, total_kv_dim)
    w_v = _linear_weight(attention.v_proj, hidden, total_kv_dim)
    norm = layer.input_layernorm
    return MoeLayerWeights(
        w_q=_linear_weight(attention.q_proj, hidden, config.total_q_dim),
        w_o=_linear_weight(attention.o_proj, config.total_q_dim, hidden),
        w_k_heads=_split_heads(w_k, config.head_dim, config.num_kv_heads),
        w_v_heads=_split_heads(w_v, config.head_dim, config.num_kv_heads),
        w_router_rows=layer.mlp.gate.weight.detach().contiguous(),
        expert_provider=provider,
        eps=getattr(norm, "variance_epsilon", getattr(norm, "eps", config.eps)),
        input_norm_weight=_optional_norm_weight(layer, "input_layernorm"),
        post_attn_norm_weight=_optional_norm_weight(
            layer, "post_attention_layernorm"
        ),
        q_norm_weight=_optional_norm_weight(attention, "q_norm"),
        k_norm_weight=_optional_norm_weight(attention, "k_norm"),
    )


def _optional_norm_weight(parent: Any, name: str) -> torch.Tensor | None:
    norm = getattr(parent, name, None)
    weight = getattr(norm, "weight", None)
    if weight is None:
        return None
    return weight.detach().float().contiguous()


def _extract_llada_layer_weights(layer: Any, config: ModelConfig) -> LayerWeights:
    """Extract LLaDA-style layer weights.

    LLaDA uses ff_proj and up_proj both mapping to intermediate dimension,
    with ff_out as the down projection. We treat ff_proj as gate and up_proj
    as the traditional up projection for compatibility with PLENA's FFN ops.
    """
    hidden = config.hidden_size
    total_kv_dim = config.num_kv_heads * config.head_dim
    w_k_full = _linear_weight(layer.k_proj, hidden, total_kv_dim)
    w_v_full = _linear_weight(layer.v_proj, hidden, total_kv_dim)

    # Get epsilon from either attn_norm or ff_norm
    norm = getattr(layer, "attn_norm", getattr(layer, "ff_norm", None))

    return LayerWeights(
        w_q=_linear_weight(layer.q_proj, hidden, config.total_q_dim),
        w_o=_linear_weight(layer.attn_out, config.total_q_dim, hidden),
        w_k_heads=_split_heads(w_k_full, config.head_dim, config.num_kv_heads),
        w_v_heads=_split_heads(w_v_full, config.head_dim, config.num_kv_heads),
        w_gate=_linear_weight(layer.ff_proj, hidden, config.inter_dim),
        w_up=_linear_weight(layer.up_proj, hidden, config.inter_dim),
        w_down=_linear_weight(layer.ff_out, config.inter_dim, hidden),
        eps=(
            getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5))
            if norm
            else 1e-5
        ),
    )


def _linear_weight(module: Any, rows: int, cols: int) -> torch.Tensor:
    """HF Linear stores (out, in); PLENA uses (in, out)."""
    return module.weight.detach().T.contiguous()[:rows, :cols]


def _linear_bias(module: Any, cols: int) -> torch.Tensor:
    """Return a linear/conv bias vector, or zeros for bias-free modules."""
    bias = getattr(module, "bias", None)
    if bias is None:
        return torch.zeros(cols)
    return bias.detach().float().contiguous()[:cols]


def _split_heads(
    weight: torch.Tensor, head_dim: int, num_heads: int
) -> list[torch.Tensor]:
    return [
        weight[:, h * head_dim : (h + 1) * head_dim].contiguous()
        for h in range(num_heads)
    ]
