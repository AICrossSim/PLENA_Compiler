"""CPU decoder references used by the PLENA ATen frontend."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from compiler.aten.model_extract import LayerWeights, ModelConfig
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware


_HW_MAX_K_TILES = 4


@dataclass(frozen=True)
class ReferencePrecision:
    """Precision policy for the shared decoder reference runner."""

    name: str
    label: str
    quantize_hbm: bool
    bf16_intermediates: bool
    use_ksplit: bool

    @classmethod
    def from_mode(cls, mode: str) -> ReferencePrecision:
        modes = {
            "hardware": cls("hardware", "MXFP8 weights + BF16 intermediates", True, True, True),
            "no_weight_quant": cls(
                "no_weight_quant",
                "float32 weights + BF16 intermediates",
                False,
                True,
                True,
            ),
            "no_bf16": cls("no_bf16", "MXFP8 weights + float32 intermediates", True, False, True),
            "fp32": cls("fp32", "float32 weights + float32 intermediates", False, False, True),
            "hf_fp32": cls("hf_fp32", "float32, no quantization", False, False, False),
        }
        try:
            return modes[mode]
        except KeyError as exc:
            valid = ", ".join(modes)
            raise ValueError(f"Unknown reference precision '{mode}'. Expected one of: {valid}") from exc

    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        return quantize_to_mxfp(tensor) if self.quantize_hbm else tensor

    def to_inter(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.bfloat16) if self.bf16_intermediates else tensor

    def from_inter(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.float() if self.bf16_intermediates else tensor


def quantize_to_mxfp(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to MXFP8 matching HBM hardware format; return dequantized result."""
    orig_shape = tensor.shape
    tensor_2d = tensor.float().reshape(-1, tensor.shape[-1])
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor_2d,
        width=8,
        exponent_width=4,
        exponent_bias_width=8,
        block_size=[1, 8],
    )
    return bm_x.reshape(orig_shape)


def _make_rope_tables(seq_len: int, head_dim: int, theta: float = 10000.0):
    """Compute RoPE cos/sin tables, shape (seq_len, head_dim)."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin


def _make_rotate_half_matrix(head_dim: int) -> torch.Tensor:
    """Build the (head_dim, head_dim) matrix that computes rotate_half."""
    rotate = torch.zeros(head_dim, head_dim)
    half = head_dim // 2
    for i in range(half):
        rotate[i + half, i] = -1.0
        rotate[i, i + half] = 1.0
    return rotate


def make_rope_inputs(seq_len: int, config: ModelConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the rotate_half matrix and RoPE tables used by CPU and PLENA paths."""
    rotate = _make_rotate_half_matrix(config.head_dim)
    cos_table, sin_table = _make_rope_tables(seq_len, config.head_dim, config.rope_theta)
    return rotate, cos_table, sin_table


def run_decoder_reference(
    token_embeds: torch.Tensor,
    pos_weight: torch.Tensor,
    weights: list[LayerWeights],
    config: ModelConfig,
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    *,
    mlen: int,
    precision: ReferencePrecision,
    trace: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Run the compiled decoder blocks under a given precision policy."""
    quantize = precision.quantize
    x = quantize(token_embeds.clone()) + quantize(pos_weight)
    rope_ref = quantize(rope_matrix)

    for layer_idx, layer in enumerate(weights):
        x = _attention_block_ref(
            x,
            layer,
            config,
            rope_ref,
            cos_table,
            sin_table,
            mlen,
            precision,
        )
        x = _ffn_block_ref(x, layer, mlen, precision)

        if trace is not None:
            trace(layer_idx, x)

    return _rms_norm_ref(x, weights[0].eps, precision)


def _attention_block_ref(
    x: torch.Tensor,
    layer: LayerWeights,
    config: ModelConfig,
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    mlen: int,
    precision: ReferencePrecision,
) -> torch.Tensor:
    quantize = precision.quantize
    residual = x.clone()
    x_normed = _rms_norm_ref(x, layer.eps, precision)
    q_full = _round(_linear_ref(x_normed, quantize(layer.w_q), mlen, precision), precision)

    k_heads = []
    v_heads = []
    for kv_h in range(config.num_kv_heads):
        k_h = _linear_ref(x_normed, quantize(layer.w_k_heads[kv_h]), mlen, precision)
        v_h = _linear_ref(x_normed, quantize(layer.w_v_heads[kv_h]), mlen, precision)
        k_h = _rope_ref(k_h, rope_matrix, cos_table, sin_table, precision)
        k_heads.append(_hbm_round_ref(k_h, precision))
        v_heads.append(_hbm_round_ref(v_h, precision))

    scale = 1.0 / math.sqrt(config.head_dim)
    o_heads = []
    for h in range(config.num_heads):
        kv_h = h // config.head_ratio
        q_h = q_full[:, h * config.head_dim:(h + 1) * config.head_dim]
        q_h = _rope_ref(q_h, rope_matrix, cos_table, sin_table, precision)
        o_heads.append(_flash_attn_ref(q_h, k_heads[kv_h], v_heads[kv_h], scale, causal=True))

    attn_out = _round(torch.cat(o_heads, dim=1), precision)
    o_proj = _round(_linear_ref(attn_out, quantize(layer.w_o), mlen, precision), precision)
    return _residual_add_ref(o_proj, residual, precision)


def _ffn_block_ref(
    x: torch.Tensor,
    layer: LayerWeights,
    mlen: int,
    precision: ReferencePrecision,
) -> torch.Tensor:
    quantize = precision.quantize
    residual = x.clone()
    x_normed = _rms_norm_ref(x, layer.eps, precision)
    up_out = _linear_ref(x_normed, quantize(layer.w_up), mlen, precision)
    gate_out = _linear_ref(x_normed, quantize(layer.w_gate), mlen, precision)
    silu_gate = precision.to_inter(F.silu(_round(up_out, precision)) * _round(gate_out, precision))
    x = _linear_ref(precision.from_inter(silu_gate), quantize(layer.w_down), mlen, precision)
    return _residual_add_ref(_round(x, precision), residual, precision)


def _round(x: torch.Tensor, precision: ReferencePrecision) -> torch.Tensor:
    return precision.from_inter(precision.to_inter(x))


def _rms_norm_ref(x: torch.Tensor, eps: float, precision: ReferencePrecision) -> torch.Tensor:
    x_inter = precision.to_inter(x)
    rms = torch.rsqrt(precision.from_inter(x_inter).pow(2).mean(-1, keepdim=True) + eps)
    return _round(precision.from_inter(x_inter) * rms, precision)


def _linear_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    mlen: int,
    precision: ReferencePrecision,
) -> torch.Tensor:
    if not precision.use_ksplit:
        return x @ weight.float()
    return _ksplit_matmul(
        x,
        weight,
        mlen=mlen,
        max_k_tiles=_HW_MAX_K_TILES,
        to_inter=precision.to_inter,
        from_inter=precision.from_inter,
    )


def _rope_ref(
    x: torch.Tensor,
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    precision: ReferencePrecision,
) -> torch.Tensor:
    x_inter = _round(x, precision)
    x_rot = _round(
        torch.matmul(x_inter, _round(rope_matrix, precision)),
        precision,
    )
    x_cos = _round(x_inter * _round(cos_table, precision), precision)
    x_rot_sin = _round(x_rot * _round(sin_table, precision), precision)
    return _round(x_cos + x_rot_sin, precision)


def _hbm_round_ref(x: torch.Tensor, precision: ReferencePrecision) -> torch.Tensor:
    return _round(precision.quantize(x), precision)


def _residual_add_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    precision: ReferencePrecision,
) -> torch.Tensor:
    return _round(x + residual, precision)


def _flash_attn_ref(Q, K, V, scale, causal=False):
    """CPU reference: scaled dot-product attention matching hardware BF16 precision."""
    scores = (Q @ K.T).to(torch.bfloat16).float() * scale
    scores = scores.to(torch.bfloat16).float()
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1], device=scores.device), diagonal=1).bool()
        scores.masked_fill_(mask, float("-inf"))
    attn = F.softmax(scores, dim=-1).to(torch.bfloat16).float()
    return (attn @ V).to(torch.bfloat16).float()


def _ksplit_matmul(A, B, mlen=64, max_k_tiles=_HW_MAX_K_TILES, to_inter=None, from_inter=None):
    """Matrix multiply matching hardware K-split BF16 precision."""
    if to_inter is None:
        def to_inter(x):
            return x.to(torch.bfloat16)

    if from_inter is None:
        def from_inter(x):
            return x.float()

    k_total = A.shape[1]
    num_k_tiles = math.ceil(k_total / mlen)

    if num_k_tiles <= max_k_tiles:
        return from_inter(to_inter(torch.matmul(from_inter(to_inter(A)), from_inter(to_inter(B)))))

    result = None
    k_start = 0
    while k_start < k_total:
        k_end = min(k_start + max_k_tiles * mlen, k_total)
        a_chunk = A[:, k_start:k_end]
        b_chunk = B[k_start:k_end, :]
        partial = from_inter(to_inter(torch.matmul(from_inter(to_inter(a_chunk)), from_inter(to_inter(b_chunk)))))
        if result is None:
            result = partial
        else:
            result = from_inter(to_inter(result) + to_inter(partial))
        k_start = k_end

    return result
