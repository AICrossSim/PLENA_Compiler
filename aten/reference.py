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


@dataclass(frozen=True)
class ScheduledReferenceConfig:
    """Physical/native decoder dimensions used by the scheduled PLENA reference."""

    seq_len: int
    padded_seq_len: int
    hidden_size: int
    padded_hidden_size: int
    inter_dim: int
    padded_inter_dim: int
    head_dim: int
    padded_head_dim: int
    num_heads: int
    num_kv_heads: int
    mlen: int
    blen: int
    max_k_tiles: int = _HW_MAX_K_TILES
    attention_head_packing: bool = False
    head_slot_dim: int | None = None
    broadcast_amount: int | None = None
    total_q_dim: int | None = None
    chunks_per_kv: int = 1
    batch_size: int = 1
    rows_per_batch: int | None = None

    @property
    def head_ratio(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def packed_group_width(self) -> int:
        return self.mlen

    @property
    def attention_width(self) -> int:
        if self.attention_head_packing:
            return self.total_q_dim if self.total_q_dim is not None else self.num_kv_heads * self.mlen
        return self.num_heads * self.padded_head_dim

    @property
    def kv_head_width(self) -> int:
        if self.attention_head_packing:
            return self.head_slot_dim or self.head_dim
        return self.padded_head_dim

    @property
    def physical_rows_per_batch(self) -> int:
        return self.rows_per_batch if self.rows_per_batch is not None else self.padded_seq_len

    @property
    def total_physical_rows(self) -> int:
        return self.batch_size * self.physical_rows_per_batch


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
    max_k_tiles: int = _HW_MAX_K_TILES,
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
            max_k_tiles,
            precision,
        )
        x = _ffn_block_ref(x, layer, mlen, max_k_tiles, precision)

        if trace is not None:
            trace(layer_idx, x)

    return _rms_norm_ref(x, weights[0].eps, precision)


def run_native_decoder_scheduled_reference(
    token_embeds: torch.Tensor,
    pos_weight: torch.Tensor,
    weights: list[LayerWeights],
    config: ScheduledReferenceConfig,
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    *,
    precision: ReferencePrecision,
    trace: Callable[[int, torch.Tensor], None] | None = None,
) -> torch.Tensor:
    """Run the native decoder on the same padded tensors the compiler emits.

    This reference starts from physical padded inputs and weights rather than
    padding an unpadded CPU/HF result afterward. It mirrors the major hardware
    boundaries that affect the final output: HBM MXFP load/store, BF16
    vector/matrix writeback, active-hidden RMS denominators, K-split GEMMs,
    packed GQA layout, and padded RoPE lanes.
    """
    x = _vram_load_ref(token_embeds.clone(), precision)
    pos = _vram_load_ref(pos_weight, precision)
    x = _round(x + pos, precision)

    # R_rope is consumed as an HBM matrix by the projection helper, while COS
    # and SIN are loaded into VRAM batches before vector ops.
    rope_ref = rope_matrix
    cos_ref = _vram_load_ref(cos_table, precision)
    sin_ref = _vram_load_ref(sin_table, precision)

    for layer_idx, layer in enumerate(weights):
        x = _scheduled_attention_block_ref(
            x,
            layer,
            config,
            rope_ref,
            cos_ref,
            sin_ref,
            precision,
        )
        x = _scheduled_ffn_block_ref(x, layer, config, precision)

        if trace is not None:
            trace(layer_idx, x)

    return _rms_norm_scheduled_ref(x, config.hidden_size, weights[0].eps, config.mlen, precision)


def _attention_block_ref(
    x: torch.Tensor,
    layer: LayerWeights,
    config: ModelConfig,
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    mlen: int,
    max_k_tiles: int,
    precision: ReferencePrecision,
) -> torch.Tensor:
    quantize = precision.quantize
    residual = x.clone()
    x_normed = _rms_norm_ref(x, layer.eps, precision)
    q_full = _round(_linear_ref(x_normed, quantize(layer.w_q), mlen, max_k_tiles, precision), precision)

    k_heads = []
    v_heads = []
    for kv_h in range(config.num_kv_heads):
        k_h = _linear_ref(x_normed, quantize(layer.w_k_heads[kv_h]), mlen, max_k_tiles, precision)
        v_h = _linear_ref(x_normed, quantize(layer.w_v_heads[kv_h]), mlen, max_k_tiles, precision)
        k_h = _rope_ref(k_h, rope_matrix, cos_table, sin_table, precision)
        k_heads.append(_hbm_round_ref(k_h, precision))
        v_heads.append(_hbm_round_ref(v_h, precision))

    scale = 1.0 / math.sqrt(config.head_dim)
    o_heads = []
    for h in range(config.num_heads):
        kv_h = h // config.head_ratio
        q_h = q_full[:, h * config.head_dim : (h + 1) * config.head_dim]
        q_h = _rope_ref(q_h, rope_matrix, cos_table, sin_table, precision)
        o_heads.append(_flash_attn_ref(q_h, k_heads[kv_h], v_heads[kv_h], scale, causal=True))

    attn_out = _round(torch.cat(o_heads, dim=1), precision)
    o_proj = _round(_linear_ref(attn_out, quantize(layer.w_o), mlen, max_k_tiles, precision), precision)
    return _residual_add_ref(o_proj, residual, precision)


def _scheduled_attention_block_ref(
    x: torch.Tensor,
    layer: LayerWeights,
    config: ScheduledReferenceConfig,
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    precision: ReferencePrecision,
) -> torch.Tensor:
    residual = x.clone()
    x_normed = _rms_norm_scheduled_ref(x, config.hidden_size, layer.eps, config.mlen, precision)
    q_full = _linear_scheduled_ref(x_normed, layer.w_q, config, precision)

    if config.attention_head_packing:
        attn_out = _packed_attention_scheduled_ref(
            q_full,
            x_normed,
            layer,
            config,
            rope_matrix,
            cos_table,
            sin_table,
            precision,
        )
    else:
        attn_out = _mha_attention_scheduled_ref(
            q_full,
            x_normed,
            layer,
            config,
            rope_matrix,
            cos_table,
            sin_table,
            precision,
        )

    o_proj = _linear_scheduled_ref(attn_out, layer.w_o, config, precision)
    return _residual_add_ref(o_proj, residual, precision)


def _packed_attention_scheduled_ref(
    q_full: torch.Tensor,
    x_normed: torch.Tensor,
    layer: LayerWeights,
    config: ScheduledReferenceConfig,
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    precision: ReferencePrecision,
) -> torch.Tensor:
    if config.head_slot_dim is None or config.broadcast_amount is None:
        raise ValueError("Scheduled packed attention requires head_slot_dim and broadcast_amount")

    rows = q_full.shape[0]
    group_width = config.packed_group_width
    head_slot_dim = config.head_slot_dim
    broadcast_amount = config.broadcast_amount
    chunks_per_kv = config.chunks_per_kv
    ratio = config.head_ratio
    scale = 1.0 / math.sqrt(config.head_dim)
    out = torch.zeros((rows, config.attention_width), dtype=q_full.dtype, device=q_full.device)
    if chunks_per_kv <= 0:
        raise ValueError(f"chunks_per_kv must be positive, got {chunks_per_kv}")
    if broadcast_amount <= 0:
        raise ValueError(f"broadcast_amount must be positive, got {broadcast_amount}")
    if chunks_per_kv * broadcast_amount < ratio:
        raise ValueError(
            f"packed chunks cannot cover ratio={ratio}: "
            f"chunks_per_kv={chunks_per_kv}, broadcast_amount={broadcast_amount}"
        )

    rows_per_batch = config.physical_rows_per_batch
    if rows_per_batch < config.seq_len:
        raise ValueError(
            f"rows_per_batch={rows_per_batch} cannot cover seq_len={config.seq_len}"
        )
    if rows < config.batch_size * rows_per_batch:
        raise ValueError(
            f"q_full rows={rows} cannot cover batch_size*rows_per_batch="
            f"{config.batch_size * rows_per_batch}"
        )

    k_heads = []
    v_heads = []
    for kv_h in range(config.num_kv_heads):
        k_h = _linear_scheduled_ref(x_normed, layer.w_k_heads[kv_h], config, precision)
        v_h = _linear_scheduled_ref(x_normed, layer.w_v_heads[kv_h], config, precision)
        k_h = _pad_cols_ref(k_h, group_width)
        v_h = _pad_cols_ref(v_h, group_width)
        k_h = _rope_scheduled_ref(k_h, rope_matrix, cos_table, sin_table, config, precision)
        k_heads.append(_hbm_round_ref(k_h, precision))
        v_heads.append(_hbm_round_ref(v_h, precision))

    for batch_idx in range(config.batch_size):
        row_start = batch_idx * rows_per_batch
        row_end = row_start + config.seq_len
        for kv_h in range(config.num_kv_heads):
            k_h = k_heads[kv_h][row_start:row_end]
            v_h = v_heads[kv_h][row_start:row_end]

            for chunk in range(chunks_per_kv):
                group_idx = kv_h * chunks_per_kv + chunk
                group_start = group_idx * group_width
                q_group = q_full[row_start:row_end, group_start:group_start + group_width]
                q_group = _rope_scheduled_ref(
                    q_group,
                    rope_matrix,
                    cos_table[row_start:row_end],
                    sin_table[row_start:row_end],
                    config,
                    precision,
                )

                chunk_head_start = chunk * broadcast_amount
                chunk_heads = min(broadcast_amount, ratio - chunk_head_start)
                for lane in range(chunk_heads):
                    lane_start = lane * head_slot_dim
                    lane_end = lane_start + head_slot_dim
                    q_h = q_group[:, lane_start:lane_end]
                    k_lane = k_h[:, :head_slot_dim]
                    v_lane = v_h[:, :head_slot_dim]
                    o_h = _flash_attn_scheduled_ref(
                        q_h,
                        k_lane,
                        v_lane,
                        scale,
                        precision,
                        causal=True,
                        matmul_scale=0.25,
                    )
                    out[row_start:row_end, group_start + lane_start:group_start + lane_end] = o_h

    return _round(out, precision)


def _mha_attention_scheduled_ref(
    q_full: torch.Tensor,
    x_normed: torch.Tensor,
    layer: LayerWeights,
    config: ScheduledReferenceConfig,
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    precision: ReferencePrecision,
) -> torch.Tensor:
    rows = q_full.shape[0]
    head_width = config.padded_head_dim
    scale = 1.0 / math.sqrt(config.head_dim)
    rows_per_batch = config.physical_rows_per_batch
    if rows_per_batch < config.seq_len:
        raise ValueError(
            f"rows_per_batch={rows_per_batch} cannot cover seq_len={config.seq_len}"
        )
    if rows < config.batch_size * rows_per_batch:
        raise ValueError(
            f"q_full rows={rows} cannot cover batch_size*rows_per_batch="
            f"{config.batch_size * rows_per_batch}"
        )

    k_heads = []
    v_heads = []
    for kv_h in range(config.num_kv_heads):
        k_h = _linear_scheduled_ref(x_normed, layer.w_k_heads[kv_h], config, precision)
        v_h = _linear_scheduled_ref(x_normed, layer.w_v_heads[kv_h], config, precision)
        k_h = _pad_cols_ref(k_h, head_width)
        v_h = _pad_cols_ref(v_h, head_width)
        k_h = _rope_scheduled_ref(k_h, rope_matrix, cos_table, sin_table, config, precision)
        k_heads.append(_hbm_round_ref(k_h, precision))
        v_heads.append(_hbm_round_ref(v_h, precision))

    out = torch.zeros((rows, config.attention_width), dtype=q_full.dtype, device=q_full.device)
    for batch_idx in range(config.batch_size):
        row_start = batch_idx * rows_per_batch
        row_end = row_start + config.seq_len
        for h in range(config.num_heads):
            kv_h = h // config.head_ratio
            start = h * head_width
            end = start + head_width
            q_h = q_full[row_start:row_end, start:end]
            q_h = _rope_scheduled_ref(
                q_h,
                rope_matrix,
                cos_table[row_start:row_end],
                sin_table[row_start:row_end],
                config,
                precision,
            )
            out[row_start:row_end, start:end] = _flash_attn_scheduled_ref(
                q_h,
                k_heads[kv_h][row_start:row_end],
                v_heads[kv_h][row_start:row_end],
                scale,
                precision,
                causal=True,
            )

    return _round(out, precision)


def _ffn_block_ref(
    x: torch.Tensor,
    layer: LayerWeights,
    mlen: int,
    max_k_tiles: int,
    precision: ReferencePrecision,
) -> torch.Tensor:
    quantize = precision.quantize
    residual = x.clone()
    x_normed = _rms_norm_ref(x, layer.eps, precision)
    up_out = _linear_ref(x_normed, quantize(layer.w_up), mlen, max_k_tiles, precision)
    gate_out = _linear_ref(x_normed, quantize(layer.w_gate), mlen, max_k_tiles, precision)
    silu_gate = precision.to_inter(F.silu(_round(up_out, precision)) * _round(gate_out, precision))
    x = _linear_ref(precision.from_inter(silu_gate), quantize(layer.w_down), mlen, max_k_tiles, precision)
    return _residual_add_ref(_round(x, precision), residual, precision)


def _scheduled_ffn_block_ref(
    x: torch.Tensor,
    layer: LayerWeights,
    config: ScheduledReferenceConfig,
    precision: ReferencePrecision,
) -> torch.Tensor:
    residual = x.clone()
    x_normed = _rms_norm_scheduled_ref(x, config.hidden_size, layer.eps, config.mlen, precision)
    up_out = _linear_scheduled_ref(x_normed, layer.w_up, config, precision)
    gate_out = _linear_scheduled_ref(x_normed, layer.w_gate, config, precision)
    silu_gate = _silu_gate_scheduled_ref(up_out, gate_out, precision)
    ffn_out = _linear_scheduled_ref(silu_gate, layer.w_down, config, precision)
    return _residual_add_ref(ffn_out, residual, precision)


def _round(x: torch.Tensor, precision: ReferencePrecision) -> torch.Tensor:
    return precision.from_inter(precision.to_inter(x))


def _scalar_round(value: float, precision: ReferencePrecision) -> float:
    if not precision.bf16_intermediates:
        return float(value)
    return float(torch.tensor(float(value), dtype=torch.float32).to(torch.bfloat16).float())


def _scalar_preload_ref(value: float, precision: ReferencePrecision) -> float:
    if not precision.bf16_intermediates:
        return float(value)
    fp16_value = torch.tensor(float(value), dtype=torch.float16).float()
    return _scalar_round(float(fp16_value), precision)


def _vram_load_ref(tensor: torch.Tensor, precision: ReferencePrecision) -> torch.Tensor:
    return _round(precision.quantize(tensor), precision)


def _rms_norm_ref(x: torch.Tensor, eps: float, precision: ReferencePrecision) -> torch.Tensor:
    x_inter = precision.to_inter(x)
    rms = torch.rsqrt(precision.from_inter(x_inter).pow(2).mean(-1, keepdim=True) + eps)
    return _round(precision.from_inter(x_inter) * rms, precision)


def _rms_norm_scheduled_ref(
    x: torch.Tensor,
    active_hidden: int,
    eps: float,
    mlen: int,
    precision: ReferencePrecision,
) -> torch.Tensor:
    if not precision.bf16_intermediates:
        rms = torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / float(active_hidden) + eps)
        return x * rms

    x_inter = _round(x, precision)
    out = torch.empty_like(x_inter.float())
    eps_scalar = _scalar_preload_ref(eps, precision)
    reci_hidden = _scalar_preload_ref(1.0 / active_hidden, precision)

    for row in range(x_inter.shape[0]):
        acc = _scalar_round(0.0, precision)
        for col in range(0, x_inter.shape[1], mlen):
            chunk = x_inter[row, col:col + mlen].float()
            sq = _round(chunk * chunk, precision)
            acc = _scalar_round(acc + float(sq.sum().item()), precision)
        mean_sq = _scalar_round(acc * reci_hidden, precision)
        denom = _scalar_round(math.sqrt(_scalar_round(mean_sq + eps_scalar, precision)), precision)
        inv = _scalar_round(1.0 / denom, precision)
        for col in range(0, x_inter.shape[1], mlen):
            out[row, col:col + mlen] = _round(x_inter[row, col:col + mlen].float() * inv, precision)

    return out


def _linear_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    mlen: int,
    max_k_tiles: int,
    precision: ReferencePrecision,
) -> torch.Tensor:
    if not precision.use_ksplit:
        return x @ weight.float()
    return _ksplit_matmul(
        x,
        weight,
        mlen=mlen,
        max_k_tiles=max_k_tiles,
        to_inter=precision.to_inter,
        from_inter=precision.from_inter,
    )


def _linear_scheduled_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    config: ScheduledReferenceConfig,
    precision: ReferencePrecision,
) -> torch.Tensor:
    return _round(
        _linear_ref(
            x,
            precision.quantize(weight),
            config.mlen,
            config.max_k_tiles,
            precision,
        ),
        precision,
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


def _rope_scheduled_ref(
    x: torch.Tensor,
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    config: ScheduledReferenceConfig,
    precision: ReferencePrecision,
) -> torch.Tensor:
    cols = x.shape[1]
    rows = x.shape[0]
    x_inter = _round(x, precision)
    x_rot = _linear_scheduled_ref(x_inter, rope_matrix[:cols, :cols], config, precision)
    x_cos = _round(x_inter * _round(cos_table[:rows, :cols], precision), precision)
    x_rot_sin = _round(x_rot * _round(sin_table[:rows, :cols], precision), precision)
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


def _flash_attn_scheduled_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    precision: ReferencePrecision,
    *,
    causal: bool,
    matmul_scale: float = 1.0,
) -> torch.Tensor:
    q = _round(q, precision)
    k = _round(k, precision)
    v = _round(v, precision)
    scores = q @ k.T
    if matmul_scale != 1.0:
        scores = scores * matmul_scale
    scores = _round(scores, precision)
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1], device=scores.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    scores = _round(scores * _scalar_preload_ref(scale, precision), precision)
    attn = _round(torch.softmax(scores, dim=-1), precision)
    return _round(attn @ v, precision)


def _silu_gate_scheduled_ref(
    up_out: torch.Tensor,
    gate_out: torch.Tensor,
    precision: ReferencePrecision,
) -> torch.Tensor:
    up = _round(up_out, precision)
    gate = _round(gate_out, precision)
    neg = _round(0.0 - up, precision)
    exp_neg = _round(torch.clamp(neg, -88.0, 88.0).exp(), precision)
    denom = _round(exp_neg + 1.0, precision)
    sigmoid = _round(denom.reciprocal(), precision)
    silu = _round(sigmoid * up, precision)
    return _round(silu * gate, precision)


def _pad_cols_ref(x: torch.Tensor, cols: int) -> torch.Tensor:
    if x.shape[1] == cols:
        return x
    if x.shape[1] > cols:
        raise ValueError(f"Cannot pad tensor with {x.shape[1]} cols to {cols} cols")
    out = torch.zeros((x.shape[0], cols), dtype=x.dtype, device=x.device)
    out[:, :x.shape[1]] = x
    return out


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
