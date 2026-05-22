"""HuggingFace decoder model to PLENA ISA compiler."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from compiler.aten.model_extract import (
    LayerWeights,
    VisionConnectorWeights,
    VisionLayerWeights,
    VisionPostNormWeights,
    VisionConfig,
    embedding_module,
    extract_layer_weights,
    extract_model_config,
    extract_vision_config,
    extract_vision_connector_weights,
    extract_vision_layer_weights,
    extract_vision_patch_weights,
    extract_vision_post_norm_weights,
    find_model_root,
    find_vision_model,
)
import compiler.aten.ops as ops
from compiler.asm_templates.gelu_asm import gelu_asm
from asm_templates._imm import add_large_int as _add_large_int_lines
from asm_templates._imm import load_large_int as _load_large_int_lines
from compiler.aten.ops.registry import Backend, OpRegistry
from compiler.aten.plena import PlenaCompiler
from compiler.aten.reference import (
    ReferencePrecision,
    ScheduledReferenceConfig,
    _ksplit_matmul,
    _make_rotate_half_matrix,
    _round,
    _scalar_preload_ref,
    make_rope_inputs,
    quantize_to_mxfp,
    run_decoder_reference,
    run_native_decoder_scheduled_reference,
)

__all__ = [
    "_fix_large_immediates",
    "_ksplit_matmul",
    "_make_rotate_half_matrix",
    "compile_hf_model",
    "compile_native_hf_decoder",
    "compile_native_hf_vision_encoder",
    "quantize_to_mxfp",
]

_IMM2_BOUND = 1 << 18  # S_ADDI_INT max immediate


def _fix_large_immediates(isa_code: str) -> str:
    """Post-process ISA so every S_ADDI_INT immediate fits in the 18-bit field.

    Absolute loads from gp0 use asm_templates._imm.load_large_int. Relative
    adds use asm_templates._imm.add_large_int without a temp register, which
    lowers to bounded S_ADDI_INT chunks and is safe as a compiler-wide pass.
    """
    pattern = re.compile(r'^(\s*)S_ADDI_INT gp(\d+), gp(\d+), (\d+)(.*)')
    out = []
    for line in isa_code.split('\n'):
        m = pattern.match(line)
        if m:
            indent, rd_str, rs_str, imm_str, rest = m.groups()
            rd = int(rd_str)
            rs = int(rs_str)
            imm = int(imm_str)
            if imm >= _IMM2_BOUND:
                replacement = (
                    _load_large_int_lines(rd, imm)
                    if rs == 0
                    else _add_large_int_lines(rd, rs, imm, temp_reg=None)
                )
                out.extend(f"{indent}{replacement_line}{rest}" for replacement_line in replacement)
                continue
        out.append(line)
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)


@dataclass(frozen=True)
class LayerInputVars:
    """PLENA input variables for one extracted decoder layer."""

    w_q: Any
    w_o: Any
    w_k_heads: list[Any]
    w_v_heads: list[Any]
    w_gate: Any
    w_up: Any
    w_down: Any


@dataclass(frozen=True)
class VisionLayerInputVars:
    """PLENA input variables for one extracted vision layer."""

    w_q_heads: list[Any]
    w_k_heads: list[Any]
    w_v_heads: list[Any]
    w_o: Any
    b_q_heads: list[Any]
    b_k_heads: list[Any]
    b_v_heads: list[Any]
    b_o: Any
    w_fc1: Any
    w_fc2: Any
    b_fc1: Any
    b_fc2: Any
    ln1_weight: Any
    ln1_bias: Any
    ln2_weight: Any
    ln2_bias: Any


@dataclass(frozen=True)
class VisionPostNormInputVars:
    weight: Any
    bias: Any


@dataclass(frozen=True)
class VisionConnectorInputVars:
    weight: Any
    bias: Any | None


@dataclass(frozen=True)
class AttentionHeadPacking:
    """Packed-head layout for D < MLEN attention lowering."""

    enabled: bool
    hlen: int
    broadcast_amount: int
    head_slot_dim: int
    group_width: int
    total_q_dim: int


@dataclass
class StageCheckpointRecorder:
    """Optional debug recorder for preserving VRAM stage boundaries.

    The normal compiler reuses a small set of VRAM buffers. That is good for
    execution, but bad for post-run validation because the final VRAM dump no
    longer contains most intermediate values. When enabled, this recorder emits
    explicit VRAM copies after selected stages and records their addresses.
    """

    enabled: bool = False
    checkpoints: list[dict[str, Any]] | None = None

    def __post_init__(self):
        if self.checkpoints is None:
            self.checkpoints = []

    def record(
        self,
        prog,
        *,
        layer_idx: int | None,
        stage: str,
        tensor,
        active_shape: tuple[int, int],
        semantic: str,
    ):
        if not self.enabled:
            return None

        layer_part = "global" if layer_idx is None else f"l{layer_idx}"
        safe_stage = re.sub(r"[^0-9A-Za-z_]+", "_", stage).strip("_")
        name = f"checkpoint_{layer_part}_{safe_stage}"
        checkpoint = prog.alloc(
            name,
            tensor.shape[0],
            tensor.shape[1],
            strict=False,
            physical_shape=tensor.physical_shape,
        )
        prog.vram_fill_zero(checkpoint)
        prog.vram_add(checkpoint, tensor)
        addr = prog.get_vram_addr(checkpoint.name)
        self.checkpoints.append(
            {
                "name": checkpoint.name,
                "source": tensor.name,
                "layer_idx": layer_idx,
                "stage": stage,
                "semantic": semantic,
                "vram_addr": addr,
                "active_shape": [int(active_shape[0]), int(active_shape[1])],
                "logical_shape": [int(tensor.shape[0]), int(tensor.shape[1])],
                "physical_shape": [int(tensor.physical_shape[0]), int(tensor.physical_shape[1])],
            }
        )
        return checkpoint

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "native_decoder_vram_stage_checkpoints",
            "checkpoints": list(self.checkpoints or []),
        }


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _pad_2d(tensor: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Zero-pad a 2D tensor to the requested shape, preserving the top-left data."""
    src_rows, src_cols = tensor.shape
    if src_rows > rows or src_cols > cols:
        raise ValueError(f"Cannot pad tensor of shape {tuple(tensor.shape)} to smaller shape {(rows, cols)}")
    if src_rows == rows and src_cols == cols:
        return tensor.contiguous()
    out = torch.zeros((rows, cols), dtype=tensor.dtype, device=tensor.device)
    out[:src_rows, :src_cols] = tensor
    return out.contiguous()


def _pad_batched_sequence_storage(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    rows_per_batch: int,
    cols: int,
) -> torch.Tensor:
    """Pad (B, S, C) data into independent per-batch row slabs."""
    if tensor.dim() == 2:
        if batch_size != 1:
            raise ValueError(f"2D tensor storage is only valid for batch_size=1, got {batch_size}")
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        raise ValueError(f"Expected 2D or 3D sequence tensor, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] != batch_size or tensor.shape[1] != seq_len:
        raise ValueError(
            f"Expected tensor shape ({batch_size}, {seq_len}, C), got {tuple(tensor.shape)}"
        )
    if rows_per_batch < seq_len:
        raise ValueError(f"rows_per_batch={rows_per_batch} cannot cover seq_len={seq_len}")
    if tensor.shape[2] > cols:
        raise ValueError(f"Cannot pad tensor with {tensor.shape[2]} cols to {cols}")

    out = torch.zeros((batch_size * rows_per_batch, cols), dtype=tensor.dtype, device=tensor.device)
    for batch_idx in range(batch_size):
        start = batch_idx * rows_per_batch
        out[start:start + seq_len, : tensor.shape[2]] = tensor[batch_idx]
    return out.contiguous()


def _repeat_sequence_storage(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    rows_per_batch: int,
) -> torch.Tensor:
    """Repeat a per-position table into per-batch row slabs."""
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D sequence table, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] < seq_len:
        raise ValueError(f"Table rows {tensor.shape[0]} cannot cover seq_len={seq_len}")
    out = torch.zeros(
        (batch_size * rows_per_batch, tensor.shape[1]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    for batch_idx in range(batch_size):
        start = batch_idx * rows_per_batch
        out[start:start + seq_len] = tensor[:seq_len]
    return out.contiguous()


def _repeat_feature_vector_storage(
    vector: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    rows_per_batch: int,
    cols: int,
) -> torch.Tensor:
    """Repeat a feature vector across active rows in padded per-batch slabs."""
    if vector.dim() != 1:
        raise ValueError(f"Expected 1D feature vector, got shape {tuple(vector.shape)}")
    if vector.numel() > cols:
        raise ValueError(f"Cannot pad feature vector with {vector.numel()} elements to {cols}")
    out = torch.zeros(
        (batch_size * rows_per_batch, cols),
        dtype=vector.dtype,
        device=vector.device,
    )
    for batch_idx in range(batch_size):
        start = batch_idx * rows_per_batch
        out[start:start + seq_len, : vector.numel()] = vector
    return out.contiguous()


@dataclass(frozen=True)
class LazyRepeatedFeatureVector:
    """Compact descriptor for vectors repeated across active padded rows."""

    vector: torch.Tensor
    batch_size: int
    seq_len: int
    rows_per_batch: int
    cols: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.batch_size * self.rows_per_batch, self.cols)

    def numel(self) -> int:
        rows, cols = self.shape
        return rows * cols

    def materialize(self) -> torch.Tensor:
        return _repeat_feature_vector_storage(
            self.vector,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            rows_per_batch=self.rows_per_batch,
            cols=self.cols,
        )


def _lazy_repeat_feature_vector_storage(
    vector: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    rows_per_batch: int,
    cols: int,
) -> LazyRepeatedFeatureVector:
    """Return a compact repeat descriptor instead of eagerly expanding storage."""
    if vector.dim() != 1:
        raise ValueError(f"Expected 1D feature vector, got shape {tuple(vector.shape)}")
    if vector.numel() > cols:
        raise ValueError(f"Cannot pad feature vector with {vector.numel()} elements to {cols}")
    return LazyRepeatedFeatureVector(
        vector=vector,
        batch_size=batch_size,
        seq_len=seq_len,
        rows_per_batch=rows_per_batch,
        cols=cols,
    )


def _repeat_matrix_rows_storage(
    matrix: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    rows_per_batch: int,
    cols: int,
) -> torch.Tensor:
    """Pad and repeat a per-position matrix into per-batch row slabs."""
    if matrix.dim() != 2:
        raise ValueError(f"Expected 2D matrix, got shape {tuple(matrix.shape)}")
    if matrix.shape[0] < seq_len:
        raise ValueError(f"Matrix rows {matrix.shape[0]} cannot cover seq_len={seq_len}")
    if matrix.shape[1] > cols:
        raise ValueError(f"Cannot pad matrix with {matrix.shape[1]} cols to {cols}")
    out = torch.zeros(
        (batch_size * rows_per_batch, cols),
        dtype=matrix.dtype,
        device=matrix.device,
    )
    for batch_idx in range(batch_size):
        start = batch_idx * rows_per_batch
        out[start:start + seq_len, : matrix.shape[1]] = matrix[:seq_len]
    return out.contiguous()


def _compact_active_sequence_rows(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    rows_per_batch: int,
    cols: int,
) -> torch.Tensor:
    """Gather active sequence rows from per-batch slabs into compact B*S rows."""
    rows = []
    for batch_idx in range(batch_size):
        start = batch_idx * rows_per_batch
        rows.append(tensor[start:start + seq_len, :cols])
    return torch.cat(rows, dim=0).contiguous()


def _pad_q_weight_grouped(weight: torch.Tensor, num_heads: int, head_dim: int, padded_hidden: int, padded_head_dim: int):
    """Pad Q weights while keeping each head in its own padded head block."""
    hidden, _ = weight.shape
    out = torch.zeros((padded_hidden, num_heads * padded_head_dim), dtype=weight.dtype, device=weight.device)
    for h in range(num_heads):
        src_start = h * head_dim
        dst_start = h * padded_head_dim
        out[:hidden, dst_start:dst_start + head_dim] = weight[:, src_start:src_start + head_dim]
    return out.contiguous()


def _pad_o_weight_grouped(weight: torch.Tensor, num_heads: int, head_dim: int, padded_head_dim: int, padded_hidden: int):
    """Pad O-projection weights from packed native heads into padded head blocks."""
    _, hidden = weight.shape
    out = torch.zeros((num_heads * padded_head_dim, padded_hidden), dtype=weight.dtype, device=weight.device)
    for h in range(num_heads):
        src_start = h * head_dim
        dst_start = h * padded_head_dim
        out[dst_start:dst_start + head_dim, :hidden] = weight[src_start:src_start + head_dim, :]
    return out.contiguous()


def _pad_q_weight_grouped_by_kv(
    weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    padded_hidden: int,
    group_width: int,
    head_slot_dim: int,
):
    """Pad Q weights into KV-group MLEN rows with HLEN-sized head slots."""
    hidden, _ = weight.shape
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    ratio = num_heads // num_kv_heads
    if ratio * head_slot_dim > group_width:
        raise ValueError(
            f"Q head slots do not fit one group: ratio={ratio}, "
            f"head_slot_dim={head_slot_dim}, group_width={group_width}"
        )
    out = torch.zeros((padded_hidden, num_kv_heads * group_width), dtype=weight.dtype, device=weight.device)
    for h in range(num_heads):
        kv_h = h // ratio
        lane = h % ratio
        src_start = h * head_dim
        dst_start = kv_h * group_width + lane * head_slot_dim
        out[:hidden, dst_start:dst_start + head_dim] = weight[:, src_start:src_start + head_dim]
    return out.contiguous()


def _pad_o_weight_grouped_by_kv(
    weight: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    group_width: int,
    head_slot_dim: int,
    padded_hidden: int,
):
    """Pad O weights from KV-group HLEN head slots into hidden projection."""
    _, hidden = weight.shape
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
    ratio = num_heads // num_kv_heads
    if ratio * head_slot_dim > group_width:
        raise ValueError(
            f"O head slots do not fit one group: ratio={ratio}, "
            f"head_slot_dim={head_slot_dim}, group_width={group_width}"
        )
    out = torch.zeros((num_kv_heads * group_width, padded_hidden), dtype=weight.dtype, device=weight.device)
    for h in range(num_heads):
        kv_h = h // ratio
        lane = h % ratio
        src_start = h * head_dim
        dst_start = kv_h * group_width + lane * head_slot_dim
        out[dst_start:dst_start + head_dim, :hidden] = weight[src_start:src_start + head_dim, :]
    return out.contiguous()


def _pad_decoder_weights_for_tiles(
    weights: LayerWeights,
    model_cfg,
    *,
    padded_hidden: int,
    padded_inter: int,
    padded_head_dim: int,
    head_packing: AttentionHeadPacking | None = None,
) -> LayerWeights:
    head_dim = model_cfg.head_dim
    num_heads = model_cfg.num_heads
    num_kv_heads = model_cfg.num_kv_heads

    if head_packing is not None and head_packing.enabled:
        w_q = _pad_q_weight_grouped_by_kv(
            weights.w_q,
            num_heads,
            num_kv_heads,
            head_dim,
            padded_hidden,
            head_packing.group_width,
            head_packing.head_slot_dim,
        )
        w_o = _pad_o_weight_grouped_by_kv(
            weights.w_o,
            num_heads,
            num_kv_heads,
            head_dim,
            head_packing.group_width,
            head_packing.head_slot_dim,
            padded_hidden,
        )
    else:
        w_q = _pad_q_weight_grouped(weights.w_q, num_heads, head_dim, padded_hidden, padded_head_dim)
        w_o = _pad_o_weight_grouped(weights.w_o, num_heads, head_dim, padded_head_dim, padded_hidden)

    kv_head_dim = head_packing.head_slot_dim if head_packing is not None and head_packing.enabled else padded_head_dim
    return LayerWeights(
        w_q=w_q,
        w_o=w_o,
        w_k_heads=[_pad_2d(w, padded_hidden, kv_head_dim) for w in weights.w_k_heads],
        w_v_heads=[_pad_2d(w, padded_hidden, kv_head_dim) for w in weights.w_v_heads],
        w_gate=_pad_2d(weights.w_gate, padded_hidden, padded_inter),
        w_up=_pad_2d(weights.w_up, padded_hidden, padded_inter),
        w_down=_pad_2d(weights.w_down, padded_inter, padded_hidden),
        eps=weights.eps,
    )


def _pad_vector(vector: torch.Tensor, cols: int) -> torch.Tensor:
    if vector.numel() > cols:
        raise ValueError(f"Cannot pad vector of length {vector.numel()} to {cols}")
    if vector.numel() == cols:
        return vector.contiguous()
    out = torch.zeros((cols,), dtype=vector.dtype, device=vector.device)
    out[: vector.numel()] = vector
    return out.contiguous()


def _pad_vision_patch_weight_for_tiles(
    weight: torch.Tensor,
    *,
    padded_k_col: int,
    padded_hidden: int,
) -> torch.Tensor:
    return _pad_2d(weight, padded_k_col, padded_hidden)


def _pad_vision_layer_weights_for_tiles(
    weights: VisionLayerWeights,
    *,
    num_heads: int,
    head_dim: int,
    padded_hidden: int,
    padded_inter: int,
    padded_head_dim: int,
    padded_total_q_dim: int,
) -> VisionLayerWeights:
    def pad_head_bias(bias: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((padded_total_q_dim,), dtype=bias.dtype, device=bias.device)
        for h in range(num_heads):
            src_start = h * head_dim
            dst_start = h * padded_head_dim
            out[dst_start:dst_start + head_dim] = bias[src_start:src_start + head_dim]
        return out.contiguous()

    return VisionLayerWeights(
        w_q=_pad_q_weight_grouped(weights.w_q, num_heads, head_dim, padded_hidden, padded_head_dim),
        w_k=_pad_q_weight_grouped(weights.w_k, num_heads, head_dim, padded_hidden, padded_head_dim),
        w_v=_pad_q_weight_grouped(weights.w_v, num_heads, head_dim, padded_hidden, padded_head_dim),
        w_o=_pad_o_weight_grouped(weights.w_o, num_heads, head_dim, padded_head_dim, padded_hidden),
        b_q=pad_head_bias(weights.b_q),
        b_k=pad_head_bias(weights.b_k),
        b_v=pad_head_bias(weights.b_v),
        b_o=_pad_vector(weights.b_o, padded_hidden),
        w_fc1=_pad_2d(weights.w_fc1, padded_hidden, padded_inter),
        w_fc2=_pad_2d(weights.w_fc2, padded_inter, padded_hidden),
        b_fc1=_pad_vector(weights.b_fc1, padded_inter),
        b_fc2=_pad_vector(weights.b_fc2, padded_hidden),
        ln1_weight=_pad_vector(weights.ln1_weight, padded_hidden),
        ln1_bias=_pad_vector(weights.ln1_bias, padded_hidden),
        ln2_weight=_pad_vector(weights.ln2_weight, padded_hidden),
        ln2_bias=_pad_vector(weights.ln2_bias, padded_hidden),
        eps=weights.eps,
    )


def _pad_vision_post_norm_for_tiles(
    weights: VisionPostNormWeights,
    *,
    padded_hidden: int,
) -> VisionPostNormWeights:
    return VisionPostNormWeights(
        weight=_pad_vector(weights.weight, padded_hidden),
        bias=_pad_vector(weights.bias, padded_hidden),
        eps=weights.eps,
    )


def _pad_vision_connector_weight_for_tiles(
    weights: VisionConnectorWeights,
    *,
    hidden: int,
    padded_hidden: int,
    padded_output_dim: int,
) -> torch.Tensor:
    """Pad connector projection rows by pixel-shuffle segment slot."""
    scale_area = weights.scale_factor * weights.scale_factor
    storage_input_dim = padded_hidden * scale_area
    out = torch.zeros(
        (storage_input_dim, padded_output_dim),
        dtype=weights.weight.dtype,
        device=weights.weight.device,
    )
    for segment_idx in range(scale_area):
        src_start = segment_idx * hidden
        src_end = src_start + hidden
        dst_start = segment_idx * padded_hidden
        out[dst_start:dst_start + hidden, :weights.output_dim] = weights.weight[src_start:src_end]
    return out.contiguous()


def _pad_optional_vector(vector: torch.Tensor | None, cols: int) -> torch.Tensor | None:
    if vector is None:
        return None
    return _pad_vector(vector, cols)


def _pad_rope_inputs_for_tiles(
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    *,
    padded_seq_len: int,
    padded_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        _pad_2d(rope_matrix, padded_head_dim, padded_head_dim),
        _pad_2d(cos_table, padded_seq_len, padded_head_dim),
        _pad_2d(sin_table, padded_seq_len, padded_head_dim),
    )


def _pad_rope_inputs_for_head_slots(
    rope_matrix: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    *,
    padded_seq_len: int,
    group_width: int,
    head_slot_dim: int,
    broadcast_amount: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an MLEN-wide RoPE table repeated into each packed head lane."""
    head_dim = rope_matrix.shape[0]
    if broadcast_amount * head_slot_dim != group_width:
        raise ValueError(
            f"broadcast_amount*head_slot_dim must equal group_width "
            f"({broadcast_amount}*{head_slot_dim} != {group_width})"
        )
    if head_dim > head_slot_dim:
        raise ValueError(f"head_dim {head_dim} exceeds head_slot_dim {head_slot_dim}")

    r = torch.zeros((group_width, group_width), dtype=rope_matrix.dtype, device=rope_matrix.device)
    cos = torch.zeros((padded_seq_len, group_width), dtype=cos_table.dtype, device=cos_table.device)
    sin = torch.zeros((padded_seq_len, group_width), dtype=sin_table.dtype, device=sin_table.device)
    seq = cos_table.shape[0]
    for lane in range(broadcast_amount):
        start = lane * head_slot_dim
        r[start:start + head_dim, start:start + head_dim] = rope_matrix
        cos[:seq, start:start + head_dim] = cos_table
        sin[:seq, start:start + head_dim] = sin_table
    return r.contiguous(), cos.contiguous(), sin.contiguous()


def _save_residual_and_norm(prog, source, scratch):
    """Emit the common decoder pre-norm residual prologue."""
    prog.vram_fill_zero(scratch)
    prog.vram_add(scratch, source)
    ops.rms_norm(prog, source, eps_offset=3, reci_hid_offset=4)


def _add_residual(prog, target, scratch):
    prog.vram_add(target, scratch)
    return target


def _linear_projection(prog, input_var, weight_var, name: str, physical_shape: tuple[int, int] | None = None):
    if physical_shape is not None:
        return prog.linear_projection(input_var, weight_var, name=name, physical_shape=physical_shape)
    return ops.linear(prog, input_var, weight_var, name=name)


def _apply_rope_projection(prog, x_var, rope_matrix, cos_var, sin_var, name):
    x_rot = _linear_projection(prog, x_var, rope_matrix, name)
    ops.rope(prog, x_var, x_rot, cos_var, sin_var)
    prog.free_tensor(x_rot)
    return x_var


def _copy_into_vram_view(prog, source, name, rows, cols, vram_addr, physical_shape=None):
    target = prog.alloc_at(
        name,
        rows,
        cols,
        vram_addr,
        physical_shape=physical_shape or source.physical_shape,
    )
    prog.vram_fill_zero(target)
    prog.vram_add(target, source)
    return target


def _free_named_tensors(prog, names):
    for name in names:
        tensor = prog._tensors.get(name)
        if tensor is not None:
            prog.free_tensor(tensor)
            prog._tensors.pop(name, None)


def _emit_kv_stores(
    prog,
    current,
    layer_inputs,
    rope_inputs,
    layer_idx,
    num_kv_heads,
    physical_rows: int | None = None,
    checkpoint_recorder: StageCheckpointRecorder | None = None,
    active_seq_len: int | None = None,
    active_head_dim: int | None = None,
):
    rope_matrix, cos_var, sin_var = rope_inputs
    kv_stored = []
    active_seq_len = active_seq_len or current.shape[0]
    for kv_h in range(num_kv_heads):
        kv_physical_shape = None
        if physical_rows is not None:
            kv_physical_shape = (physical_rows, layer_inputs.w_k_heads[kv_h].physical_shape[1])
        K_h = _linear_projection(
            prog,
            current,
            layer_inputs.w_k_heads[kv_h],
            f"K_{layer_idx}_h{kv_h}",
            physical_shape=kv_physical_shape,
        )
        v_physical_shape = None
        if physical_rows is not None:
            v_physical_shape = (physical_rows, layer_inputs.w_v_heads[kv_h].physical_shape[1])
        V_h = _linear_projection(
            prog,
            current,
            layer_inputs.w_v_heads[kv_h],
            f"V_{layer_idx}_h{kv_h}",
            physical_shape=v_physical_shape,
        )
        checkpoint_cols = active_head_dim or K_h.shape[1]
        if checkpoint_recorder is not None:
            checkpoint_recorder.record(
                prog,
                layer_idx=layer_idx,
                stage=f"k_proj_h{kv_h}",
                tensor=K_h,
                active_shape=(active_seq_len, checkpoint_cols),
                semantic=f"K projection for KV head {kv_h} before RoPE",
            )
            checkpoint_recorder.record(
                prog,
                layer_idx=layer_idx,
                stage=f"v_proj_h{kv_h}",
                tensor=V_h,
                active_shape=(active_seq_len, checkpoint_cols),
                semantic=f"V projection for KV head {kv_h}",
            )

        _apply_rope_projection(
            prog,
            K_h,
            rope_matrix,
            cos_var,
            sin_var,
            f"K_rot_{layer_idx}_h{kv_h}",
        )
        if checkpoint_recorder is not None:
            checkpoint_recorder.record(
                prog,
                layer_idx=layer_idx,
                stage=f"k_rope_h{kv_h}",
                tensor=K_h,
                active_shape=(active_seq_len, checkpoint_cols),
                semantic=f"K projection for KV head {kv_h} after RoPE",
            )

        K_stored = prog.store(K_h, name=f"K_stored_{layer_idx}_h{kv_h}")
        V_stored = prog.store(V_h, name=f"V_stored_{layer_idx}_h{kv_h}")
        kv_stored.append((K_stored, V_stored))

        prog.free_tensor(K_h)
        prog.free_tensor(V_h)

    return kv_stored


def _emit_packed_attention_block(
    prog,
    current,
    layer_inputs,
    rope_inputs,
    causal_mask,
    scratch,
    scale,
    layer_idx,
    seq_len,
    head_dim,
    num_kv_heads,
    ratio,
    head_packing: AttentionHeadPacking,
    checkpoint_recorder: StageCheckpointRecorder | None = None,
    active_seq_len: int | None = None,
    active_hidden: int | None = None,
    batch_size: int = 1,
    rows_per_batch: int | None = None,
    active_seq_len_per_batch: int | None = None,
):
    active_seq_len = active_seq_len or seq_len
    active_hidden = active_hidden or current.shape[1]
    active_seq_len_per_batch = active_seq_len_per_batch or seq_len
    rows_per_batch = rows_per_batch or max(prog.mlen, seq_len)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if rows_per_batch < active_seq_len_per_batch:
        raise ValueError(
            f"rows_per_batch={rows_per_batch} cannot cover active_seq_len={active_seq_len_per_batch}"
        )
    total_physical_rows = batch_size * rows_per_batch if batch_size > 1 else max(prog.mlen, seq_len)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="attn_input",
            tensor=current,
            active_shape=(active_seq_len, active_hidden),
            semantic="decoder layer input before attention RMS norm",
        )

    _save_residual_and_norm(prog, current, scratch)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="attn_norm",
            tensor=current,
            active_shape=(active_seq_len, active_hidden),
            semantic="attention RMS-normalized input",
        )

    q_physical_shape = (total_physical_rows, head_packing.total_q_dim)
    Q = _linear_projection(
        prog,
        current,
        layer_inputs.w_q,
        f"Q_{layer_idx}",
        physical_shape=q_physical_shape,
    )
    q_full_addr = prog.get_vram_addr(Q.name)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="q_full",
            tensor=Q,
            active_shape=(active_seq_len, head_packing.total_q_dim),
            semantic="packed Q projection before RoPE",
        )

    O_full = prog.alloc(
        f"O_full_{layer_idx}",
        current.shape[0],
        head_packing.total_q_dim,
        strict=False,
        physical_shape=(total_physical_rows, head_packing.total_q_dim),
    )
    prog.vram_fill_zero(O_full)
    o_full_addr = prog.get_vram_addr(O_full.name)

    kv_stored = _emit_kv_stores(
        prog,
        current,
        layer_inputs,
        rope_inputs,
        layer_idx,
        num_kv_heads,
        physical_rows=total_physical_rows,
        checkpoint_recorder=checkpoint_recorder,
        active_seq_len=active_seq_len,
        active_head_dim=head_packing.head_slot_dim,
    )

    scratch_rows = prog.mlen * (head_packing.broadcast_amount + ratio)
    attn_scratch = prog.alloc(
        f"packed_attn_scratch_{layer_idx}",
        scratch_rows,
        prog.mlen,
        strict=True,
    )
    attn_scratch_addr = prog.get_vram_addr(attn_scratch.name)

    rope_matrix, cos_var, sin_var = rope_inputs
    q_group_stride = Q.physical_shape[0] * prog.mlen
    o_group_stride = O_full.physical_shape[0] * prog.mlen
    q_groups = []
    if batch_size == 1:
        for kv_h in range(num_kv_heads):
            q_group_addr = q_full_addr + kv_h * q_group_stride
            Q_group = prog.alloc_at(
                f"Q_group{kv_h}_{layer_idx}",
                seq_len,
                prog.mlen,
                q_group_addr,
                physical_shape=(Q.physical_shape[0], prog.mlen),
            )
            _apply_rope_projection(
                prog,
                Q_group,
                rope_matrix,
                cos_var,
                sin_var,
                f"Q_rot_{layer_idx}_g{kv_h}",
            )
            q_groups.append(Q_group)

    roll_kv_groups = os.environ.get("PLENA_ROLL_KV_GROUPS", "1") != "0"
    if batch_size == 1 and roll_kv_groups and num_kv_heads > 1:
        prog.flash_attention_packed_groups_looped(
            Q,
            kv_stored,
            group_heads=ratio,
            head_slot_dim=head_packing.head_slot_dim,
            output_base_address=o_full_addr,
            scratch_base_address=attn_scratch_addr,
            broadcast_amount=head_packing.broadcast_amount,
            scale=scale,
            causal_mask=causal_mask,
        )
    elif batch_size == 1:
        for kv_h, Q_group in enumerate(q_groups):
            K_stored, V_stored = kv_stored[kv_h]
            prog.flash_attention_packed_group(
                Q_group,
                K_stored,
                V_stored,
                group_heads=ratio,
                head_slot_dim=head_packing.head_slot_dim,
                output_base_address=o_full_addr + kv_h * o_group_stride,
                scratch_base_address=attn_scratch_addr,
                broadcast_amount=head_packing.broadcast_amount,
                scale=scale,
                causal_mask=causal_mask,
                valid_cols=active_seq_len_per_batch,
            )
    else:
        row_block_stride = rows_per_batch // prog.mlen
        if rows_per_batch % prog.mlen != 0:
            raise ValueError(f"rows_per_batch={rows_per_batch} must be a multiple of MLEN={prog.mlen}")
        batch_vram_stride = rows_per_batch * prog.mlen
        for batch_idx in range(batch_size):
            batch_row_offset = batch_idx * batch_vram_stride
            for kv_h in range(num_kv_heads):
                q_group_addr = q_full_addr + kv_h * q_group_stride + batch_row_offset
                Q_group = prog.alloc_at(
                    f"Q_group{kv_h}_b{batch_idx}_{layer_idx}",
                    active_seq_len_per_batch,
                    prog.mlen,
                    q_group_addr,
                    physical_shape=(rows_per_batch, prog.mlen),
                )
                _apply_rope_projection(
                    prog,
                    Q_group,
                    rope_matrix,
                    cos_var,
                    sin_var,
                    f"Q_rot_{layer_idx}_g{kv_h}_b{batch_idx}",
                )
                K_stored, V_stored = kv_stored[kv_h]
                prog.flash_attention_packed_group(
                    Q_group,
                    K_stored,
                    V_stored,
                    group_heads=ratio,
                    head_slot_dim=head_packing.head_slot_dim,
                    output_base_address=o_full_addr + kv_h * o_group_stride + batch_row_offset,
                    scratch_base_address=attn_scratch_addr,
                    broadcast_amount=head_packing.broadcast_amount,
                    scale=scale,
                    causal_mask=causal_mask,
                    k_idx=batch_idx * row_block_stride,
                    valid_cols=active_seq_len_per_batch,
                )
    prog.free_tensor(attn_scratch)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="o_full",
            tensor=O_full,
            active_shape=(active_seq_len, head_packing.total_q_dim),
            semantic="packed attention output before output projection",
        )
    O_proj = _linear_projection(prog, O_full, layer_inputs.w_o, f"O_proj_{layer_idx}")
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="o_proj",
            tensor=O_proj,
            active_shape=(active_seq_len, active_hidden),
            semantic="attention output projection before residual add",
        )
    out = _add_residual(prog, O_proj, scratch)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="attn_residual",
            tensor=out,
            active_shape=(active_seq_len, active_hidden),
            semantic="attention block output after residual add",
        )
    return out


def _emit_attention_block(
    prog,
    current,
    layer_inputs,
    rope_inputs,
    causal_mask,
    scratch,
    scale,
    layer_idx,
    seq_len,
    head_dim,
    total_q_dim,
    num_heads,
    num_kv_heads,
    ratio,
    checkpoint_recorder: StageCheckpointRecorder | None = None,
    active_seq_len: int | None = None,
    active_hidden: int | None = None,
):
    active_seq_len = active_seq_len or seq_len
    active_hidden = active_hidden or current.shape[1]
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="attn_input",
            tensor=current,
            active_shape=(active_seq_len, active_hidden),
            semantic="decoder layer input before attention RMS norm",
        )

    _save_residual_and_norm(prog, current, scratch)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="attn_norm",
            tensor=current,
            active_shape=(active_seq_len, active_hidden),
            semantic="attention RMS-normalized input",
        )

    Q = _linear_projection(prog, current, layer_inputs.w_q, f"Q_{layer_idx}")
    q_full_addr = prog.get_vram_addr(Q.name)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="q_full",
            tensor=Q,
            active_shape=(active_seq_len, total_q_dim),
            semantic="Q projection before per-head RoPE",
        )

    O_full = prog.alloc(f"O_full_{layer_idx}", seq_len, total_q_dim, strict=False)
    o_full_addr = prog.get_vram_addr(O_full.name)

    kv_stored = _emit_kv_stores(
        prog,
        current,
        layer_inputs,
        rope_inputs,
        layer_idx,
        num_kv_heads,
        checkpoint_recorder=checkpoint_recorder,
        active_seq_len=active_seq_len,
        active_head_dim=head_dim,
    )

    rope_matrix, cos_var, sin_var = rope_inputs
    for h in range(num_heads):
        kv_h = h // ratio
        K_stored, V_stored = kv_stored[kv_h]

        q_h_addr = q_full_addr + h * seq_len * prog.mlen
        Q_h = prog.alloc_at(f"Q_h{h}_{layer_idx}", seq_len, head_dim, q_h_addr)
        _apply_rope_projection(
            prog,
            Q_h,
            rope_matrix,
            cos_var,
            sin_var,
            f"Q_rot_{layer_idx}_h{h}",
        )

        O_h = ops.flash_attention(
            prog,
            Q_h,
            K_stored,
            V_stored,
            scale,
            causal_mask=causal_mask,
        )

        o_h_dest_addr = o_full_addr + h * seq_len * prog.mlen
        _copy_into_vram_view(
            prog,
            O_h,
            f"O_dest_h{h}_{layer_idx}",
            seq_len,
            head_dim,
            o_h_dest_addr,
        )
        _free_named_tensors(prog, ("O", "S", "PV"))

    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="o_full",
            tensor=O_full,
            active_shape=(active_seq_len, total_q_dim),
            semantic="attention output before output projection",
        )
    O_proj = _linear_projection(prog, O_full, layer_inputs.w_o, f"O_proj_{layer_idx}")
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="o_proj",
            tensor=O_proj,
            active_shape=(active_seq_len, active_hidden),
            semantic="attention output projection before residual add",
        )
    out = _add_residual(prog, O_proj, scratch)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="attn_residual",
            tensor=out,
            active_shape=(active_seq_len, active_hidden),
            semantic="attention block output after residual add",
        )
    return out


def _emit_ffn_block(
    prog,
    current,
    layer_inputs,
    scratch,
    layer_idx: int | None = None,
    checkpoint_recorder: StageCheckpointRecorder | None = None,
    active_seq_len: int | None = None,
    active_hidden: int | None = None,
):
    active_seq_len = active_seq_len or current.shape[0]
    active_hidden = active_hidden or current.shape[1]
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="ffn_input",
            tensor=current,
            active_shape=(active_seq_len, active_hidden),
            semantic="FFN block input before RMS norm",
        )
    _save_residual_and_norm(prog, current, scratch)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="ffn_norm",
            tensor=current,
            active_shape=(active_seq_len, active_hidden),
            semantic="FFN RMS-normalized input",
        )
    ops.ffn(prog, current, layer_inputs.w_gate, layer_inputs.w_up, layer_inputs.w_down)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="ffn_out",
            tensor=current,
            active_shape=(active_seq_len, active_hidden),
            semantic="FFN projection output before residual add",
        )
    out = _add_residual(prog, current, scratch)
    if checkpoint_recorder is not None:
        checkpoint_recorder.record(
            prog,
            layer_idx=layer_idx,
            stage="ffn_residual",
            tensor=out,
            active_shape=(active_seq_len, active_hidden),
            semantic="FFN block output after residual add",
        )
    return out


def _load_and_add(prog, target, input_var, name: str):
    loaded = prog.load_batch(input_var, name=name)
    prog.vram_add(target, loaded)
    prog.free_tensor(loaded)
    return target


def _load_and_mul(prog, target, input_var, name: str):
    loaded = prog.load_batch(input_var, name=name)
    prog.vram_mul(target, loaded)
    prog.free_tensor(loaded)
    return target


def _apply_vision_layer_norm(prog, current, gamma, beta, *, name: str):
    ops.layer_norm(prog, current, eps_offset=3, reci_hid_offset=4)
    _load_and_mul(prog, current, gamma, f"{name}_gamma")
    _load_and_add(prog, current, beta, f"{name}_beta")
    return current


def _gelu_inplace(prog, tensor, *, const_one_fp_address: int = 5, const_1702_fp_address: int = 6):
    scratch = prog.alloc(
        f"gelu_scratch_{tensor.name}",
        1,
        prog.mlen,
        strict=False,
        physical_shape=(1, prog.mlen),
    )
    gp_regs = prog.register_allocator.allocate_gp(3)
    try:
        prog.emit(
            gelu_asm(
                const_one_fp_address=const_one_fp_address,
                const_1702_fp_address=const_1702_fp_address,
                alive_registers=gp_regs,
                activation_base_address=prog.get_vram_addr(tensor.name),
                scratchpad_base_address=prog.get_vram_addr(scratch.name),
                vlen=prog.mlen,
                batch_size=tensor.physical_shape[0],
                hidden_dim=tensor.physical_shape[1],
            )
        )
    finally:
        prog.register_allocator.free_gp(gp_regs)
        prog.free_tensor(scratch)
    return tensor


def _gelu_1702_fp_address(prog) -> int:
    """FP SRAM slot for GELU's 1.702 constant, kept clear of im2col/softmax scratch."""
    return prog._ONLINE_SOFTMAX_FPSRAM_BASE + 3 * prog.mlen


def _emit_vision_attention_block(
    prog,
    current,
    layer_inputs: VisionLayerInputVars,
    scratch,
    *,
    layer_idx: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
    padded_total_q_dim: int,
    scale: float,
    stop_after: str | None = None,
):
    prog.vram_fill_zero(scratch)
    prog.vram_add(scratch, current)
    _apply_vision_layer_norm(
        prog,
        current,
        layer_inputs.ln1_weight,
        layer_inputs.ln1_bias,
        name=f"vision_l{layer_idx}_ln1",
    )
    emitted_stage = f"layer{layer_idx}_ln1"
    if stop_after == emitted_stage:
        return current, emitted_stage

    O_full = prog.alloc(
        f"V_O_full_{layer_idx}",
        current.shape[0],
        padded_total_q_dim,
        strict=False,
        physical_shape=(current.physical_shape[0], padded_total_q_dim),
    )
    prog.vram_fill_zero(O_full)
    o_full_addr = prog.get_vram_addr(O_full.name)
    o_head_stride = O_full.physical_shape[0] * prog.mlen

    for h in range(num_heads):
        Q_h = _linear_projection(
            prog,
            current,
            layer_inputs.w_q_heads[h],
            f"V_Q_{layer_idx}_h{h}",
            physical_shape=(current.physical_shape[0], padded_head_dim),
        )
        _load_and_add(prog, Q_h, layer_inputs.b_q_heads[h], f"V_B_q_{layer_idx}_h{h}_load")

        K_h = _linear_projection(
            prog,
            current,
            layer_inputs.w_k_heads[h],
            f"V_K_{layer_idx}_h{h}",
            physical_shape=(current.physical_shape[0], padded_head_dim),
        )
        _load_and_add(prog, K_h, layer_inputs.b_k_heads[h], f"V_B_k_{layer_idx}_h{h}_load")

        V_h = _linear_projection(
            prog,
            current,
            layer_inputs.w_v_heads[h],
            f"V_V_{layer_idx}_h{h}",
            physical_shape=(current.physical_shape[0], padded_head_dim),
        )
        _load_and_add(prog, V_h, layer_inputs.b_v_heads[h], f"V_B_v_{layer_idx}_h{h}_load")

        K_stored = prog.store(K_h, name=f"V_K_stored_{layer_idx}_h{h}")
        V_stored = prog.store(V_h, name=f"V_V_stored_{layer_idx}_h{h}")
        O_h = ops.flash_attention(
            prog,
            Q_h,
            K_stored,
            V_stored,
            scale,
            causal_mask=None,
            batch_size=1,
            seq_len=seq_len,
            kv_seq_len=seq_len,
        )

        _copy_into_vram_view(
            prog,
            O_h,
            f"V_O_dest_{layer_idx}_h{h}",
            seq_len,
            padded_head_dim,
            o_full_addr + h * o_head_stride,
            physical_shape=(O_full.physical_shape[0], padded_head_dim),
        )
        _free_named_tensors(prog, ("O", "S", "PV"))
        for tensor in (Q_h, K_h, V_h):
            prog.free_tensor(tensor)

    emitted_stage = f"layer{layer_idx}_attn_o_full"
    if stop_after == emitted_stage:
        return O_full, emitted_stage

    O_proj = _linear_projection(prog, O_full, layer_inputs.w_o, f"V_O_proj_{layer_idx}")
    _load_and_add(prog, O_proj, layer_inputs.b_o, f"V_B_o_{layer_idx}_load")
    emitted_stage = f"layer{layer_idx}_attn_proj"
    if stop_after == emitted_stage:
        return O_proj, emitted_stage

    out = _add_residual(prog, O_proj, scratch)
    return out, f"layer{layer_idx}_attn_residual"


def _emit_vision_mlp_block(
    prog,
    current,
    layer_inputs: VisionLayerInputVars,
    scratch,
    *,
    layer_idx: int,
    stop_after: str | None = None,
):
    prog.vram_fill_zero(scratch)
    prog.vram_add(scratch, current)
    _apply_vision_layer_norm(
        prog,
        current,
        layer_inputs.ln2_weight,
        layer_inputs.ln2_bias,
        name=f"vision_l{layer_idx}_ln2",
    )
    emitted_stage = f"layer{layer_idx}_ln2"
    if stop_after == emitted_stage:
        return current, emitted_stage

    fc1 = _linear_projection(prog, current, layer_inputs.w_fc1, f"V_FC1_{layer_idx}")
    _load_and_add(prog, fc1, layer_inputs.b_fc1, f"V_B_fc1_{layer_idx}_load")
    emitted_stage = f"layer{layer_idx}_fc1"
    if stop_after == emitted_stage:
        return fc1, emitted_stage

    _gelu_inplace(prog, fc1, const_1702_fp_address=_gelu_1702_fp_address(prog))
    emitted_stage = f"layer{layer_idx}_gelu"
    if stop_after == emitted_stage:
        return fc1, emitted_stage

    fc2 = _linear_projection(prog, fc1, layer_inputs.w_fc2, f"V_FC2_{layer_idx}")
    _load_and_add(prog, fc2, layer_inputs.b_fc2, f"V_B_fc2_{layer_idx}_load")
    emitted_stage = f"layer{layer_idx}_fc2"
    if stop_after == emitted_stage:
        prog.free_tensor(fc1)
        return fc2, emitted_stage

    out = _add_residual(prog, fc2, scratch)
    prog.free_tensor(fc1)
    return out, f"layer{layer_idx}_mlp_residual"


def _add_vram_row_chunk(
    prog,
    *,
    source,
    target,
    source_addr: int,
    target_addr: int,
    name: str,
):
    src_view = prog.alloc_at(
        f"{name}_src",
        1,
        prog.mlen,
        source_addr,
        physical_shape=(source.physical_shape[0], prog.mlen),
    )
    dst_view = prog.alloc_at(
        f"{name}_dst",
        1,
        prog.mlen,
        target_addr,
        physical_shape=(target.physical_shape[0], prog.mlen),
    )
    prog.vram_add(dst_view, src_view, num_rows=1)
    prog.free_tensor(src_view)
    prog.free_tensor(dst_view)


def _emit_vision_pixel_shuffle(
    prog,
    current,
    *,
    seq_len: int,
    hidden: int,
    padded_hidden: int,
    scale_factor: int,
    connector_rows: int,
    connector_storage_dim: int,
):
    del hidden
    grid = int(math.isqrt(seq_len))
    if grid * grid != seq_len:
        raise ValueError(f"vision connector pixel_shuffle requires square seq_len, got {seq_len}")
    if grid % scale_factor != 0:
        raise ValueError(
            f"vision connector scale_factor={scale_factor} must divide patch grid {grid}x{grid}"
        )
    if padded_hidden % prog.mlen != 0:
        raise ValueError(f"padded_hidden={padded_hidden} must be a multiple of MLEN={prog.mlen}")

    out_grid = grid // scale_factor
    connector_seq_len = seq_len // (scale_factor ** 2)
    shuffled = prog.alloc(
        "V_CONNECTOR_SHUFFLED",
        connector_seq_len,
        connector_storage_dim,
        strict=False,
        physical_shape=(connector_rows, connector_storage_dim),
    )
    prog.vram_fill_zero(shuffled)

    source_base = prog.get_vram_addr(current.name)
    target_base = prog.get_vram_addr(shuffled.name)
    hidden_blocks = padded_hidden // prog.mlen

    for out_y in range(out_grid):
        for out_x in range(out_grid):
            target_row = out_y * out_grid + out_x
            for dy in range(scale_factor):
                for dx in range(scale_factor):
                    source_row = (out_y * scale_factor + dy) * grid + (out_x * scale_factor + dx)
                    segment_idx = dy * scale_factor + dx
                    for col_block in range(hidden_blocks):
                        source_addr = (
                            source_base
                            + col_block * current.physical_shape[0] * prog.mlen
                            + source_row * prog.mlen
                        )
                        target_col_block = segment_idx * hidden_blocks + col_block
                        target_addr = (
                            target_base
                            + target_col_block * shuffled.physical_shape[0] * prog.mlen
                            + target_row * prog.mlen
                        )
                        _add_vram_row_chunk(
                            prog,
                            source=current,
                            target=shuffled,
                            source_addr=source_addr,
                            target_addr=target_addr,
                            name=(
                                f"V_connector_shuffle_o{target_row}_"
                                f"s{segment_idx}_c{col_block}"
                            ),
                        )

    return shuffled


def _emit_vision_connector(
    prog,
    current,
    connector_inputs: VisionConnectorInputVars,
    *,
    seq_len: int,
    hidden: int,
    padded_hidden: int,
    scale_factor: int,
    connector_rows: int,
    connector_storage_dim: int,
    padded_output_dim: int,
):
    shuffled = _emit_vision_pixel_shuffle(
        prog,
        current,
        seq_len=seq_len,
        hidden=hidden,
        padded_hidden=padded_hidden,
        scale_factor=scale_factor,
        connector_rows=connector_rows,
        connector_storage_dim=connector_storage_dim,
    )
    projected = _linear_projection(
        prog,
        shuffled,
        connector_inputs.weight,
        "V_CONNECTOR_OUT",
        physical_shape=(connector_rows, padded_output_dim),
    )
    if connector_inputs.bias is not None:
        _load_and_add(prog, projected, connector_inputs.bias, "V_CONNECTOR_B_load")
    prog.free_tensor(shuffled)
    return projected


def _register_layer_inputs(
    prog,
    layer_idx: int,
    weights: LayerWeights,
    physical_shapes: dict[str, tuple[int, int]] | None = None,
) -> LayerInputVars:
    named_vars = {}
    w_k_heads = []
    w_v_heads = []
    physical_shapes = physical_shapes or {}
    for tensor_name, tensor in weights.tensor_entries(layer_idx):
        var = prog.input(tensor_name, shape=tuple(tensor.shape), physical_shape=physical_shapes.get(tensor_name))
        if tensor_name.startswith(f"W_k_{layer_idx}_h"):
            w_k_heads.append(var)
        elif tensor_name.startswith(f"W_v_{layer_idx}_h"):
            w_v_heads.append(var)
        else:
            named_vars[tensor_name[: tensor_name.rfind(f"_{layer_idx}")]] = var

    return LayerInputVars(
        w_q=named_vars["W_q"],
        w_o=named_vars["W_o"],
        w_k_heads=w_k_heads,
        w_v_heads=w_v_heads,
        w_gate=named_vars["W_gate"],
        w_up=named_vars["W_up"],
        w_down=named_vars["W_down"],
    )


def _register_vision_layer_inputs(
    prog,
    layer_idx: int,
    weights: VisionLayerWeights,
    *,
    num_heads: int,
    padded_head_dim: int,
    compile_seq_rows: int,
) -> VisionLayerInputVars:
    w_q_heads = []
    w_k_heads = []
    w_v_heads = []
    b_q_heads = []
    b_k_heads = []
    b_v_heads = []

    for h in range(num_heads):
        w_q_heads.append(
            prog.input(
                f"V_W_q_{layer_idx}_h{h}",
                shape=(weights.w_q.shape[0], padded_head_dim),
            )
        )
        w_k_heads.append(
            prog.input(
                f"V_W_k_{layer_idx}_h{h}",
                shape=(weights.w_k.shape[0], padded_head_dim),
            )
        )
        w_v_heads.append(
            prog.input(
                f"V_W_v_{layer_idx}_h{h}",
                shape=(weights.w_v.shape[0], padded_head_dim),
            )
        )
        b_q_heads.append(
            prog.input(
                f"V_B_q_{layer_idx}_h{h}",
                shape=(compile_seq_rows, padded_head_dim),
                physical_shape=(compile_seq_rows, padded_head_dim),
            )
        )
        b_k_heads.append(
            prog.input(
                f"V_B_k_{layer_idx}_h{h}",
                shape=(compile_seq_rows, padded_head_dim),
                physical_shape=(compile_seq_rows, padded_head_dim),
            )
        )
        b_v_heads.append(
            prog.input(
                f"V_B_v_{layer_idx}_h{h}",
                shape=(compile_seq_rows, padded_head_dim),
                physical_shape=(compile_seq_rows, padded_head_dim),
            )
        )

    return VisionLayerInputVars(
        w_q_heads=w_q_heads,
        w_k_heads=w_k_heads,
        w_v_heads=w_v_heads,
        w_o=prog.input("V_W_o_%d" % layer_idx, shape=tuple(weights.w_o.shape)),
        b_q_heads=b_q_heads,
        b_k_heads=b_k_heads,
        b_v_heads=b_v_heads,
        b_o=prog.input(
            "V_B_o_%d" % layer_idx,
            shape=(compile_seq_rows, weights.b_o.numel()),
            physical_shape=(compile_seq_rows, weights.b_o.numel()),
        ),
        w_fc1=prog.input("V_W_fc1_%d" % layer_idx, shape=tuple(weights.w_fc1.shape)),
        w_fc2=prog.input("V_W_fc2_%d" % layer_idx, shape=tuple(weights.w_fc2.shape)),
        b_fc1=prog.input(
            "V_B_fc1_%d" % layer_idx,
            shape=(compile_seq_rows, weights.b_fc1.numel()),
            physical_shape=(compile_seq_rows, weights.b_fc1.numel()),
        ),
        b_fc2=prog.input(
            "V_B_fc2_%d" % layer_idx,
            shape=(compile_seq_rows, weights.b_fc2.numel()),
            physical_shape=(compile_seq_rows, weights.b_fc2.numel()),
        ),
        ln1_weight=prog.input(
            "V_LN1_weight_%d" % layer_idx,
            shape=(compile_seq_rows, weights.ln1_weight.numel()),
            physical_shape=(compile_seq_rows, weights.ln1_weight.numel()),
        ),
        ln1_bias=prog.input(
            "V_LN1_bias_%d" % layer_idx,
            shape=(compile_seq_rows, weights.ln1_bias.numel()),
            physical_shape=(compile_seq_rows, weights.ln1_bias.numel()),
        ),
        ln2_weight=prog.input(
            "V_LN2_weight_%d" % layer_idx,
            shape=(compile_seq_rows, weights.ln2_weight.numel()),
            physical_shape=(compile_seq_rows, weights.ln2_weight.numel()),
        ),
        ln2_bias=prog.input(
            "V_LN2_bias_%d" % layer_idx,
            shape=(compile_seq_rows, weights.ln2_bias.numel()),
            physical_shape=(compile_seq_rows, weights.ln2_bias.numel()),
        ),
    )


def _register_vision_post_norm_inputs(
    prog,
    weights: VisionPostNormWeights,
    *,
    compile_seq_rows: int,
) -> VisionPostNormInputVars:
    return VisionPostNormInputVars(
        weight=prog.input(
            "V_POST_LN_weight",
            shape=(compile_seq_rows, weights.weight.numel()),
            physical_shape=(compile_seq_rows, weights.weight.numel()),
        ),
        bias=prog.input(
            "V_POST_LN_bias",
            shape=(compile_seq_rows, weights.bias.numel()),
            physical_shape=(compile_seq_rows, weights.bias.numel()),
        ),
    )


def _register_vision_connector_inputs(
    prog,
    weights: VisionConnectorWeights,
    *,
    storage_input_dim: int,
    padded_output_dim: int,
    connector_rows: int,
    has_bias: bool,
) -> VisionConnectorInputVars:
    return VisionConnectorInputVars(
        weight=prog.input(
            "V_CONNECTOR_W",
            shape=(storage_input_dim, weights.output_dim),
            physical_shape=(storage_input_dim, padded_output_dim),
        ),
        bias=(
            prog.input(
                "V_CONNECTOR_B",
                shape=(connector_rows, padded_output_dim),
                physical_shape=(connector_rows, padded_output_dim),
            )
            if has_bias else None
        ),
    )


def _vision_position_ids(vision_model, *, batch_size: int, patches_h: int, patches_w: int, device) -> torch.Tensor:
    """Match SmolVLM dynamic position bucketing for an all-valid patch mask."""
    embeddings = vision_model.embeddings
    num_patches_per_side = int(getattr(embeddings, "num_patches_per_side", patches_h))
    boundaries = torch.arange(
        1 / num_patches_per_side,
        1.0,
        1 / num_patches_per_side,
        device=device,
    )
    nb_patches_h = torch.full((batch_size,), patches_h, device=device, dtype=torch.float32)
    nb_patches_w = torch.full((batch_size,), patches_w, device=device, dtype=torch.float32)
    step_h = 1.0 / nb_patches_h
    step_w = 1.0 / nb_patches_w
    h_indices = torch.arange(patches_h, device=device, dtype=torch.float32)
    w_indices = torch.arange(patches_w, device=device, dtype=torch.float32)
    fractional_coords_h = torch.clamp(h_indices[None, :] * step_h[:, None], max=(1.0 - 1e-6))
    fractional_coords_w = torch.clamp(w_indices[None, :] * step_w[:, None], max=(1.0 - 1e-6))
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = bucket_coords_h[:, :, None] * num_patches_per_side + bucket_coords_w[:, None, :]
    return pos_ids.reshape(batch_size, -1)


def _pixel_values_to_raw_storage(pixel_values: torch.Tensor, *, w_padded: int) -> torch.Tensor:
    """Convert B=1 NCHW pixels to conv2d_plena's raw HBM row layout."""
    if pixel_values.shape[0] != 1:
        raise NotImplementedError("native vision compile currently supports batch_size=1")
    _, channels, height, width = pixel_values.shape
    if w_padded < width:
        raise ValueError(f"w_padded={w_padded} cannot cover image width {width}")
    raw = torch.zeros((channels * height, w_padded), dtype=pixel_values.dtype, device=pixel_values.device)
    for c in range(channels):
        raw[c * height:(c + 1) * height, :width] = pixel_values[0, c]
    return raw.contiguous()


def _im2col_vision(pixel_values: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    col = F.unfold(pixel_values.float(), kernel_size=patch_size, stride=patch_size)
    return col.permute(0, 2, 1).reshape(-1, pixel_values.shape[1] * patch_size * patch_size).contiguous()


def _quantize_vision_pixels_like_hbm(
    pixel_values: torch.Tensor,
    *,
    precision: ReferencePrecision,
) -> torch.Tensor:
    if not precision.quantize_hbm:
        return pixel_values
    width = pixel_values.shape[-1]
    w_padded = _ceil_to_multiple(width, 64)
    raw = _pixel_values_to_raw_storage(pixel_values, w_padded=w_padded)
    raw_q = precision.quantize(raw)
    out = torch.zeros_like(pixel_values.float())
    _, channels, height, _ = pixel_values.shape
    for c in range(channels):
        out[0, c] = raw_q[c * height:(c + 1) * height, :width]
    return out


def _layer_norm_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    *,
    precision: ReferencePrecision,
) -> torch.Tensor:
    gamma = precision.quantize(weight)
    beta = precision.quantize(bias)
    y = _round_vision_intermediate(F.layer_norm(x.float(), (x.shape[-1],), eps=eps), precision)
    y = _round_vision_intermediate(y.float() * gamma.float() + beta.float(), precision)
    return y


def _round_vision_intermediate(x: torch.Tensor, precision: ReferencePrecision) -> torch.Tensor:
    return precision.from_inter(precision.to_inter(x))


def _gelu_hw_ref(x: torch.Tensor, precision: ReferencePrecision) -> torch.Tensor:
    # Match gelu_asm: each vector op writes BF16-like intermediate state before
    # the next op consumes it.
    step1 = _round_vision_intermediate(1.702 * x.float(), precision)
    step2 = _round_vision_intermediate(-step1.float(), precision)
    step3 = _round_vision_intermediate(torch.exp(step2.float()), precision)
    step4 = _round_vision_intermediate(1.0 + step3.float(), precision)
    step5 = _round_vision_intermediate(1.0 / step4.float(), precision)
    return _round_vision_intermediate(x.float() * step5.float(), precision)


def _vision_linear_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    precision: ReferencePrecision,
) -> torch.Tensor:
    w = precision.quantize(weight)
    b = precision.quantize(bias)
    return _round_vision_intermediate(x.float() @ w.float() + b.float(), precision)


def _vision_attention_ref(
    x: torch.Tensor,
    weights: VisionLayerWeights,
    config: VisionConfig,
    *,
    precision: ReferencePrecision,
) -> torch.Tensor:
    o_full = _vision_attention_heads_ref(x, weights, config, precision=precision)
    return _vision_linear_ref(o_full, weights.w_o, weights.b_o, precision=precision)


def _vision_attention_heads_ref(
    x: torch.Tensor,
    weights: VisionLayerWeights,
    config: VisionConfig,
    *,
    precision: ReferencePrecision,
) -> torch.Tensor:
    heads = []
    for h in range(config.num_heads):
        start = h * config.head_dim
        end = start + config.head_dim
        q = _vision_linear_ref(x, weights.w_q[:, start:end], weights.b_q[start:end], precision=precision)
        k = _vision_linear_ref(x, weights.w_k[:, start:end], weights.b_k[start:end], precision=precision)
        v = _vision_linear_ref(x, weights.w_v[:, start:end], weights.b_v[start:end], precision=precision)
        scores = _round(q.float() @ k.float().T, precision)
        scale = _scalar_preload_ref(1.0 / math.sqrt(config.head_dim), precision)
        scores = _round(scores.float() * scale, precision)
        attn = _round(torch.softmax(scores.float(), dim=-1), precision)
        heads.append(_round(attn.float() @ v.float(), precision))
    return torch.cat(heads, dim=-1)


def _pad_vision_attention_heads_ref(
    o_full: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int | None,
) -> torch.Tensor:
    if padded_head_dim is None or padded_head_dim == head_dim:
        return o_full
    out = torch.zeros(
        o_full.shape[0],
        num_heads * padded_head_dim,
        dtype=o_full.dtype,
        device=o_full.device,
    )
    for h in range(num_heads):
        src_start = h * head_dim
        dst_start = h * padded_head_dim
        out[:, dst_start:dst_start + head_dim] = o_full[:, src_start:src_start + head_dim]
    return out.contiguous()


def _run_vision_reference(
    pixel_values: torch.Tensor,
    position_embeddings: torch.Tensor,
    patch_weights,
    all_weights: list[VisionLayerWeights],
    post_norm: VisionPostNormWeights,
    config: VisionConfig,
    *,
    precision: ReferencePrecision,
    gelu_mode: str,
    padded_head_dim: int | None = None,
) -> torch.Tensor:
    return _run_vision_reference_trace(
        pixel_values,
        position_embeddings,
        patch_weights,
        all_weights,
        post_norm,
        config,
        precision=precision,
        gelu_mode=gelu_mode,
        padded_head_dim=padded_head_dim,
    )["post_ln"]


def _run_vision_reference_trace(
    pixel_values: torch.Tensor,
    position_embeddings: torch.Tensor,
    patch_weights,
    all_weights: list[VisionLayerWeights],
    post_norm: VisionPostNormWeights,
    config: VisionConfig,
    *,
    precision: ReferencePrecision,
    gelu_mode: str,
    padded_head_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    """Return hardware-shaped vision reference values at compiler stop points."""
    trace: dict[str, torch.Tensor] = {}
    patch_w = precision.quantize(patch_weights.weight_2d)
    patch_bias_pos = precision.quantize(position_embeddings + patch_weights.bias.detach().float().view(1, -1))
    pixel_values = _quantize_vision_pixels_like_hbm(pixel_values, precision=precision)
    x_col = _round_vision_intermediate(_im2col_vision(pixel_values, patch_size=config.patch_size), precision)
    trace["patch_im2col"] = x_col.float().contiguous()
    x = _round_vision_intermediate(x_col.float() @ patch_w.float(), precision)
    trace["patch"] = x.float().contiguous()
    x = _round_vision_intermediate(x.float() + patch_bias_pos.float(), precision)
    trace["patch_bias"] = x.float().contiguous()

    for layer_idx, layer in enumerate(all_weights):
        residual = x
        x_norm = _layer_norm_ref(
            x,
            layer.ln1_weight,
            layer.ln1_bias,
            layer.eps,
            precision=precision,
        )
        trace[f"layer{layer_idx}_ln1"] = x_norm.float().contiguous()
        attn_heads = _vision_attention_heads_ref(
            x_norm,
            layer,
            config,
            precision=precision,
        )
        trace[f"layer{layer_idx}_attn_o_full"] = _pad_vision_attention_heads_ref(
            attn_heads,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            padded_head_dim=padded_head_dim,
        ).float().contiguous()
        attn = _vision_linear_ref(attn_heads, layer.w_o, layer.b_o, precision=precision)
        trace[f"layer{layer_idx}_attn_proj"] = attn.float().contiguous()
        x = _round_vision_intermediate(residual.float() + attn.float(), precision)
        trace[f"layer{layer_idx}_attn_residual"] = x.float().contiguous()

        residual = x
        x_norm = _layer_norm_ref(
            x,
            layer.ln2_weight,
            layer.ln2_bias,
            layer.eps,
            precision=precision,
        )
        trace[f"layer{layer_idx}_ln2"] = x_norm.float().contiguous()
        fc1 = _vision_linear_ref(x_norm, layer.w_fc1, layer.b_fc1, precision=precision)
        trace[f"layer{layer_idx}_fc1"] = fc1.float().contiguous()
        if gelu_mode == "hardware":
            act = _gelu_hw_ref(fc1, precision)
        else:
            act = _round_vision_intermediate(F.gelu(fc1.float(), approximate="tanh"), precision)
        trace[f"layer{layer_idx}_gelu"] = act.float().contiguous()
        mlp = _vision_linear_ref(act, layer.w_fc2, layer.b_fc2, precision=precision)
        trace[f"layer{layer_idx}_fc2"] = mlp.float().contiguous()
        x = _round_vision_intermediate(residual.float() + mlp.float(), precision)
        trace[f"layer{layer_idx}_mlp_residual"] = x.float().contiguous()

    x = _layer_norm_ref(
        x,
        post_norm.weight,
        post_norm.bias,
        post_norm.eps,
        precision=precision,
    )
    trace["post_ln"] = x.float().contiguous()
    return trace


def _vision_pixel_shuffle_ref(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Match SmolVLMConnector.pixel_shuffle for compact sequence tensors."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze = True
    elif x.dim() == 3:
        squeeze = False
    else:
        raise ValueError(f"Expected 2D/3D vision sequence tensor, got {tuple(x.shape)}")

    batch_size, seq_len, embed_dim = x.shape
    height = width = int(math.isqrt(seq_len))
    if height * width != seq_len:
        raise ValueError(f"pixel_shuffle requires square seq_len, got {seq_len}")
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError(
            f"pixel_shuffle scale_factor={scale_factor} must divide grid {height}x{width}"
        )

    y = x.view(batch_size, height, width, embed_dim)
    y = y.view(batch_size, height, width // scale_factor, embed_dim * scale_factor)
    y = y.permute(0, 2, 1, 3)
    y = y.reshape(
        batch_size,
        width // scale_factor,
        height // scale_factor,
        embed_dim * (scale_factor ** 2),
    )
    y = y.permute(0, 2, 1, 3)
    y = y.reshape(batch_size, seq_len // (scale_factor ** 2), embed_dim * (scale_factor ** 2))
    return y[0].contiguous() if squeeze else y.contiguous()


def _run_vision_connector_reference(
    encoder_output: torch.Tensor,
    weights: VisionConnectorWeights,
    *,
    precision: ReferencePrecision,
) -> torch.Tensor:
    shuffled = _vision_pixel_shuffle_ref(encoder_output, weights.scale_factor)
    w = precision.quantize(weights.weight)
    y = shuffled.float() @ w.float()
    if weights.bias is not None:
        y = y + precision.quantize(weights.bias).float()
    return _round_vision_intermediate(y, precision).float().contiguous()


def _weight_physical_shapes_for_layer(
    layer_idx: int,
    weights: LayerWeights,
    *,
    head_packing: AttentionHeadPacking | None,
    padded_head_dim: int,
) -> dict[str, tuple[int, int]]:
    if head_packing is None or not head_packing.enabled:
        return {}

    physical_shapes = {}
    for kv_h, (w_k, w_v) in enumerate(zip(weights.w_k_heads, weights.w_v_heads)):
        physical_shapes[f"W_k_{layer_idx}_h{kv_h}"] = (w_k.shape[0], padded_head_dim)
        physical_shapes[f"W_v_{layer_idx}_h{kv_h}"] = (w_v.shape[0], padded_head_dim)
    return physical_shapes


def _tensor_layout_metadata(prog, input_tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, list[int] | int]]:
    layouts = {}
    for name, tensor in input_tensors.items():
        hbm_layout = prog.hbm_matrices.get(name)
        if hbm_layout is None:
            continue
        tensor_shape = getattr(tensor, "shape", None)
        if tensor_shape is None:
            continue
        source_shape = tuple(int(dim) for dim in tensor_shape)
        storage_shape = tuple(int(dim) for dim in (hbm_layout.physical_shape or hbm_layout.full_shape))
        if len(source_shape) < 2 or len(storage_shape) < 2:
            continue
        layouts[name] = {
            "source_shape": list(source_shape),
            "storage_shape": list(storage_shape),
            "logical_shape": list(hbm_layout.full_shape),
            "source_row_elements": source_shape[-1],
            "storage_row_elements": storage_shape[-1],
        }
    return layouts


# ---------------------------------------------------------------------------
# Native vision compilation
# ---------------------------------------------------------------------------
def compile_native_hf_vision_encoder(
    model,
    seq_len: int = 64,
    batch_size: int = 1,
    hidden_size: int | None = None,
    inter_dim: int | None = None,
    num_layers: int | None = None,
    layer_idx_start: int = 0,
    mlen: int = 64,
    blen: int = 4,
    mram_tile_capacity: int = 4,
    seed: int = 42,
    golden_precision: str = "hardware",
    reference_backend: str = "scheduled",
    include_connector: bool = True,
    stop_after: str | None = None,
    verbose: bool = False,
    **_unused,
) -> dict:
    """Compile a HuggingFace SigLIP/ViT vision encoder to PLENA ISA metadata."""
    del reference_backend

    def _verbose(message: str = ""):
        if verbose:
            print(message)

    if batch_size != 1:
        raise NotImplementedError("native vision compile currently supports batch_size=1")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    vision_model = find_vision_model(model)
    model_cfg = extract_vision_config(model)
    if hidden_size is not None and hidden_size != model_cfg.hidden_size:
        raise ValueError(
            f"compile_native_hf_vision_encoder currently supports native hidden size only: "
            f"requested {hidden_size}, model has {model_cfg.hidden_size}"
        )
    if inter_dim is not None and inter_dim != model_cfg.inter_dim:
        raise ValueError(
            f"compile_native_hf_vision_encoder currently supports native inter_dim only: "
            f"requested {inter_dim}, model has {model_cfg.inter_dim}"
        )

    grid = int(math.isqrt(seq_len))
    if grid * grid != seq_len:
        raise ValueError(
            f"vision seq_len must be a square patch count for now, got {seq_len}"
        )
    image_h = grid * model_cfg.patch_size
    image_w = image_h
    patches_h = grid
    patches_w = grid
    w_padded = _ceil_to_multiple(image_w, 64)
    k_col = model_cfg.num_channels * model_cfg.patch_size * model_cfg.patch_size

    hidden = model_cfg.hidden_size
    inter = model_cfg.inter_dim
    head_dim = model_cfg.head_dim
    num_heads = model_cfg.num_heads
    padded_seq_len = _ceil_to_multiple(seq_len, blen)
    rows_per_batch = max(mlen, padded_seq_len)
    compile_seq_rows = rows_per_batch
    padded_hidden = _ceil_to_multiple(hidden, mlen)
    padded_inter = _ceil_to_multiple(inter, mlen)
    padded_head_dim = _ceil_to_multiple(head_dim, mlen)
    padded_total_q_dim = num_heads * padded_head_dim
    padded_k_col = _ceil_to_multiple(k_col, mlen)
    scale = 1.0 / math.sqrt(head_dim)
    connector_weights = extract_vision_connector_weights(model, model_cfg) if include_connector else None
    connector_seq_len = seq_len
    connector_rows = rows_per_batch
    connector_storage_dim = padded_hidden
    connector_output_dim = hidden
    padded_connector_output_dim = padded_hidden
    if connector_weights is not None:
        connector_scale = connector_weights.scale_factor
        if grid % connector_scale != 0:
            raise ValueError(
                f"vision connector scale_factor={connector_scale} must divide patch grid {grid}x{grid}"
            )
        connector_seq_len = seq_len // (connector_scale ** 2)
        connector_padded_seq_len = _ceil_to_multiple(connector_seq_len, blen)
        connector_rows = max(mlen, connector_padded_seq_len)
        connector_storage_dim = padded_hidden * (connector_scale ** 2)
        connector_output_dim = connector_weights.output_dim
        padded_connector_output_dim = _ceil_to_multiple(connector_output_dim, mlen)

    layers = vision_model.encoder.layers
    n_layers = num_layers if num_layers is not None else len(layers)
    assert layer_idx_start + n_layers <= len(layers), (
        f"Requested vision layers [{layer_idx_start}, {layer_idx_start + n_layers}) "
        f"but model only has {len(layers)} layers"
    )

    print("=" * 80)
    print(f"Vision Compiler - {model_cfg.model_type} ({n_layers} layer{'s' if n_layers != 1 else ''})")
    print(
        f"  vision: hidden={hidden}, inter={inter}, heads={num_heads}, "
        f"head_dim={head_dim}, patch={model_cfg.patch_size}, channels={model_cfg.num_channels}"
    )
    print(
        f"  compile: batch_size={batch_size}, seq_len={seq_len}, image={image_h}x{image_w}, "
        f"mlen={mlen}, blen={blen}, mram_tile_capacity={mram_tile_capacity}"
    )
    print(
        "  tile padding: "
        f"seq_len={seq_len}->{padded_seq_len}, hidden={hidden}->{padded_hidden}, "
        f"inter={inter}->{padded_inter}, head_dim={head_dim}->{padded_head_dim}, "
        f"total_q_dim={model_cfg.total_q_dim}->{padded_total_q_dim}, k_col={k_col}->{padded_k_col}"
    )
    if connector_weights is not None:
        print(
            "  connector: "
            f"pixel_shuffle scale={connector_weights.scale_factor}, "
            f"seq_len={seq_len}->{connector_seq_len}, "
            f"proj={connector_storage_dim}->{connector_output_dim}"
        )
    print("=" * 80)

    print(f"\nExtracting vision weights from layers {layer_idx_start}..{layer_idx_start + n_layers - 1}...")
    patch_weights = extract_vision_patch_weights(vision_model, model_cfg)
    all_weights = [
        extract_vision_layer_weights(layers[layer_idx_start + i], model_cfg)
        for i in range(n_layers)
    ]
    post_norm = extract_vision_post_norm_weights(vision_model, model_cfg)
    compile_patch_weight = _pad_vision_patch_weight_for_tiles(
        patch_weights.weight_2d,
        padded_k_col=padded_k_col,
        padded_hidden=padded_hidden,
    )
    compile_weights = [
        _pad_vision_layer_weights_for_tiles(
            w,
            num_heads=num_heads,
            head_dim=head_dim,
            padded_hidden=padded_hidden,
            padded_inter=padded_inter,
            padded_head_dim=padded_head_dim,
            padded_total_q_dim=padded_total_q_dim,
        )
        for w in all_weights
    ]
    compile_post_norm = _pad_vision_post_norm_for_tiles(post_norm, padded_hidden=padded_hidden)
    compile_connector_weight = None
    compile_connector_bias = None
    if connector_weights is not None:
        compile_connector_weight = _pad_vision_connector_weight_for_tiles(
            connector_weights,
            hidden=hidden,
            padded_hidden=padded_hidden,
            padded_output_dim=padded_connector_output_dim,
        )
        compile_connector_bias = _pad_optional_vector(
            connector_weights.bias,
            padded_connector_output_dim,
        )
    eps = post_norm.eps

    torch.manual_seed(seed)
    pixel_values = torch.randn(batch_size, model_cfg.num_channels, image_h, image_w)
    raw_pixels = _pixel_values_to_raw_storage(pixel_values, w_padded=w_padded)
    position_ids = _vision_position_ids(
        vision_model,
        batch_size=batch_size,
        patches_h=patches_h,
        patches_w=patches_w,
        device=pixel_values.device,
    )
    position_embeddings = vision_model.embeddings.position_embedding(position_ids)[0].detach().float()
    patch_bias_pos = position_embeddings + patch_weights.bias.detach().float().view(1, -1)
    compile_patch_bias_pos = _repeat_matrix_rows_storage(
        _pad_2d(patch_bias_pos, seq_len, padded_hidden),
        batch_size=batch_size,
        seq_len=seq_len,
        rows_per_batch=rows_per_batch,
        cols=padded_hidden,
    )

    golden_policy = ReferencePrecision.from_mode(golden_precision)
    requested_stop = None if stop_after in (None, "", "final") else str(stop_after).lower().replace("-", "_")
    print(f"\nComputing CPU golden vision reference ({golden_policy.label})")
    encoder_trace = _run_vision_reference_trace(
        pixel_values,
        position_embeddings,
        patch_weights,
        all_weights,
        post_norm,
        model_cfg,
        precision=golden_policy,
        gelu_mode="hardware",
        padded_head_dim=padded_head_dim,
    )
    encoder_hf_trace = _run_vision_reference_trace(
        pixel_values,
        position_embeddings,
        patch_weights,
        all_weights,
        post_norm,
        model_cfg,
        precision=ReferencePrecision.from_mode("hf_fp32"),
        gelu_mode="hf",
        padded_head_dim=padded_head_dim,
    )
    default_output_stage = "connector" if connector_weights is not None else "post_ln"
    output_stage = requested_stop or default_output_stage
    valid_stages = set(encoder_trace)
    if connector_weights is not None:
        valid_stages.update({"connector_shuffle", "connector"})
    if output_stage not in valid_stages:
        raise ValueError(
            f"Unsupported vision stop_after={stop_after!r}; valid stages are: "
            f"{', '.join(sorted(valid_stages))}"
        )

    encoder_golden_out = encoder_trace["post_ln"]
    encoder_hf_ground_truth = encoder_hf_trace["post_ln"]
    if output_stage in encoder_trace:
        golden_out = encoder_trace[output_stage]
        hf_ground_truth = encoder_hf_trace[output_stage]
        output_seq_len = seq_len
        if output_stage == "patch_im2col":
            output_rows = _ceil_to_multiple(seq_len, blen)
            output_hidden = k_col
            output_padded_hidden = padded_k_col
        elif output_stage.endswith("_attn_o_full"):
            output_rows = rows_per_batch
            output_hidden = padded_total_q_dim
            output_padded_hidden = padded_total_q_dim
        elif output_stage.endswith("_fc1") or output_stage.endswith("_gelu"):
            output_rows = rows_per_batch
            output_hidden = inter
            output_padded_hidden = padded_inter
        else:
            output_rows = rows_per_batch
            output_hidden = hidden
            output_padded_hidden = padded_hidden
    elif output_stage == "connector_shuffle":
        golden_out = _vision_pixel_shuffle_ref(
            encoder_golden_out,
            connector_weights.scale_factor,
        )
        hf_ground_truth = _vision_pixel_shuffle_ref(
            encoder_hf_ground_truth,
            connector_weights.scale_factor,
        )
        output_seq_len = connector_seq_len
        output_rows = connector_rows
        output_hidden = connector_weights.input_dim
        output_padded_hidden = connector_storage_dim
    elif output_stage == "connector":
        golden_out = _run_vision_connector_reference(
            encoder_golden_out,
            connector_weights,
            precision=golden_policy,
        )
        hf_ground_truth = _run_vision_connector_reference(
            encoder_hf_ground_truth,
            connector_weights,
            precision=ReferencePrecision.from_mode("hf_fp32"),
        )
        output_seq_len = connector_seq_len
        output_rows = connector_rows
        output_hidden = connector_output_dim
        output_padded_hidden = padded_connector_output_dim
    else:
        raise AssertionError(f"Unhandled vision output stage {output_stage!r}")
    padded_golden_output = _pad_batched_sequence_storage(
        golden_out,
        batch_size=batch_size,
        seq_len=output_seq_len,
        rows_per_batch=output_rows,
        cols=output_padded_hidden,
    )
    print(f"  golden_out: {golden_out.shape}")
    _verbose(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    print("\n--- PLENA Vision Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        real_data_ratio=REAL_DATA_RATIO,
        mram_tile_capacity=mram_tile_capacity,
    )

    sequence_physical_shape = (compile_seq_rows, padded_hidden)
    input_raw_var = prog.input("V_PIXELS", shape=tuple(raw_pixels.shape))
    patch_w_var = prog.input(
        "V_PATCH_W",
        shape=(k_col, hidden),
        physical_shape=(padded_k_col, padded_hidden),
    )
    patch_bias_pos_var = prog.input(
        "V_PATCH_BIAS_POS",
        shape=(compile_seq_rows, padded_hidden),
        physical_shape=sequence_physical_shape,
    )

    layer_inputs = [
        _register_vision_layer_inputs(
            prog,
            i,
            compile_weights[i],
            num_heads=num_heads,
            padded_head_dim=padded_head_dim,
            compile_seq_rows=compile_seq_rows,
        )
        for i in range(n_layers)
    ]
    post_norm_inputs = _register_vision_post_norm_inputs(
        prog,
        compile_post_norm,
        compile_seq_rows=compile_seq_rows,
    )
    connector_inputs = None
    if connector_weights is not None:
        connector_inputs = _register_vision_connector_inputs(
            prog,
            connector_weights,
            storage_input_dim=connector_storage_dim,
            padded_output_dim=padded_connector_output_dim,
            connector_rows=connector_rows,
            has_bias=compile_connector_bias is not None,
        )

    conv_out = ops.conv2d(
        prog,
        input_raw_var,
        patch_w_var,
        C_in=model_cfg.num_channels,
        H=image_h,
        W=image_w,
        K=model_cfg.patch_size,
        OH=patches_h,
        OW=patches_w,
        M=seq_len,
        W_padded=w_padded,
        fp_one_reg=5,
        stride=model_cfg.patch_size,
        return_im2col=output_stage == "patch_im2col",
    )
    if output_stage == "patch_im2col":
        current = conv_out
        emitted_stage = "patch_im2col"
    else:
        current = prog.alloc(
            "V_PATCH_OUT",
            compile_seq_rows,
            padded_hidden,
            strict=False,
            physical_shape=sequence_physical_shape,
        )
        prog.vram_fill_zero(current)
        prog.vram_add(current, conv_out, num_rows=seq_len)
        prog.free_tensor(conv_out)
        emitted_stage = "patch"

    if output_stage not in {"patch_im2col", "patch"}:
        _load_and_add(prog, current, patch_bias_pos_var, "V_PATCH_BIAS_POS_load")
        emitted_stage = "patch_bias"

    if output_stage not in {"patch_im2col", "patch", "patch_bias"}:
        scratch = prog.alloc(
            "vision_residual_scratch",
            compile_seq_rows,
            padded_hidden,
            strict=False,
            physical_shape=sequence_physical_shape,
        )

        for i, li in enumerate(layer_inputs):
            prog.emit_comment(f"=== VISION LAYER {i}/{n_layers} START ===")
            current, emitted_stage = _emit_vision_attention_block(
                prog,
                current,
                li,
                scratch,
                layer_idx=i,
                seq_len=seq_len,
                num_heads=num_heads,
                head_dim=head_dim,
                padded_head_dim=padded_head_dim,
                padded_total_q_dim=padded_total_q_dim,
                scale=scale,
                stop_after=output_stage,
            )
            if output_stage == emitted_stage:
                break

            current, emitted_stage = _emit_vision_mlp_block(
                prog,
                current,
                li,
                scratch,
                layer_idx=i,
                stop_after=output_stage,
            )
            prog.emit_comment(f"=== VISION LAYER {i}/{n_layers} COMPLETE ===")
            if output_stage == emitted_stage:
                break

        if output_stage == "post_ln" or output_stage in {"connector_shuffle", "connector"}:
            _apply_vision_layer_norm(
                prog,
                current,
                post_norm_inputs.weight,
                post_norm_inputs.bias,
                name="vision_post_ln",
            )
            emitted_stage = "post_ln"

        if output_stage == "connector_shuffle":
            current = _emit_vision_pixel_shuffle(
                prog,
                current,
                seq_len=seq_len,
                hidden=hidden,
                padded_hidden=padded_hidden,
                scale_factor=connector_weights.scale_factor,
                connector_rows=connector_rows,
                connector_storage_dim=connector_storage_dim,
            )
            emitted_stage = "connector_shuffle"
        elif output_stage == "connector":
            if connector_inputs is None:
                raise ValueError("connector stop requested but model has no connector")
            current = _emit_vision_connector(
                prog,
                current,
                connector_inputs,
                seq_len=seq_len,
                hidden=hidden,
                padded_hidden=padded_hidden,
                scale_factor=connector_weights.scale_factor,
                connector_rows=connector_rows,
                connector_storage_dim=connector_storage_dim,
                padded_output_dim=padded_connector_output_dim,
            )
            emitted_stage = "connector"

    if emitted_stage != output_stage:
        raise AssertionError(f"Vision compiler emitted {emitted_stage!r}, expected {output_stage!r}")

    isa_code = prog.compile()
    isa_code = _fix_large_immediates(isa_code)
    lines = isa_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of vision ISA code")

    input_tensors = {
        "V_PIXELS": raw_pixels.float(),
        "V_PATCH_W": patch_weights.weight_2d,
        "V_PATCH_BIAS_POS": compile_patch_bias_pos,
    }
    data_order = ["V_PIXELS", "V_PATCH_W", "V_PATCH_BIAS_POS"]
    for i, w in enumerate(compile_weights):
        for h in range(num_heads):
            start = h * padded_head_dim
            end = start + padded_head_dim
            for prefix, tensor in (
                ("V_W_q", w.w_q[:, start:end]),
                ("V_W_k", w.w_k[:, start:end]),
                ("V_W_v", w.w_v[:, start:end]),
            ):
                name = f"{prefix}_{i}_h{h}"
                input_tensors[name] = tensor
                data_order.append(name)
            for prefix, vector in (
                ("V_B_q", w.b_q[start:end]),
                ("V_B_k", w.b_k[start:end]),
                ("V_B_v", w.b_v[start:end]),
            ):
                name = f"{prefix}_{i}_h{h}"
                input_tensors[name] = _lazy_repeat_feature_vector_storage(
                    vector,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    rows_per_batch=rows_per_batch,
                    cols=padded_head_dim,
                )
                data_order.append(name)

        for name, tensor in (
            (f"V_W_o_{i}", w.w_o),
            (f"V_B_o_{i}", _lazy_repeat_feature_vector_storage(
                w.b_o,
                batch_size=batch_size,
                seq_len=seq_len,
                rows_per_batch=rows_per_batch,
                cols=padded_hidden,
            )),
            (f"V_W_fc1_{i}", w.w_fc1),
            (f"V_W_fc2_{i}", w.w_fc2),
            (f"V_B_fc1_{i}", _lazy_repeat_feature_vector_storage(
                w.b_fc1,
                batch_size=batch_size,
                seq_len=seq_len,
                rows_per_batch=rows_per_batch,
                cols=padded_inter,
            )),
            (f"V_B_fc2_{i}", _lazy_repeat_feature_vector_storage(
                w.b_fc2,
                batch_size=batch_size,
                seq_len=seq_len,
                rows_per_batch=rows_per_batch,
                cols=padded_hidden,
            )),
            (f"V_LN1_weight_{i}", _lazy_repeat_feature_vector_storage(
                w.ln1_weight,
                batch_size=batch_size,
                seq_len=seq_len,
                rows_per_batch=rows_per_batch,
                cols=padded_hidden,
            )),
            (f"V_LN1_bias_{i}", _lazy_repeat_feature_vector_storage(
                w.ln1_bias,
                batch_size=batch_size,
                seq_len=seq_len,
                rows_per_batch=rows_per_batch,
                cols=padded_hidden,
            )),
            (f"V_LN2_weight_{i}", _lazy_repeat_feature_vector_storage(
                w.ln2_weight,
                batch_size=batch_size,
                seq_len=seq_len,
                rows_per_batch=rows_per_batch,
                cols=padded_hidden,
            )),
            (f"V_LN2_bias_{i}", _lazy_repeat_feature_vector_storage(
                w.ln2_bias,
                batch_size=batch_size,
                seq_len=seq_len,
                rows_per_batch=rows_per_batch,
                cols=padded_hidden,
            )),
        ):
            input_tensors[name] = tensor
            data_order.append(name)

    for name, tensor in (
        ("V_POST_LN_weight", _lazy_repeat_feature_vector_storage(
            compile_post_norm.weight,
            batch_size=batch_size,
            seq_len=seq_len,
            rows_per_batch=rows_per_batch,
            cols=padded_hidden,
        )),
        ("V_POST_LN_bias", _lazy_repeat_feature_vector_storage(
            compile_post_norm.bias,
            batch_size=batch_size,
            seq_len=seq_len,
            rows_per_batch=rows_per_batch,
            cols=padded_hidden,
        )),
    ):
        input_tensors[name] = tensor
        data_order.append(name)
    if connector_weights is not None:
        input_tensors["V_CONNECTOR_W"] = compile_connector_weight
        data_order.append("V_CONNECTOR_W")
        if compile_connector_bias is not None:
            input_tensors["V_CONNECTOR_B"] = _lazy_repeat_feature_vector_storage(
                compile_connector_bias,
                batch_size=batch_size,
                seq_len=connector_seq_len,
                rows_per_batch=connector_rows,
                cols=padded_connector_output_dim,
            )
            data_order.append("V_CONNECTOR_B")

    tensor_layouts = _tensor_layout_metadata(prog, input_tensors)
    gelu_1702_fp_address = prog._ONLINE_SOFTMAX_FPSRAM_BASE + 3 * mlen
    fp_preload = [0.0, scale, float("-inf"), eps, 1.0 / hidden, 1.0, 1.702] + [0.0] * 3
    if len(fp_preload) <= gelu_1702_fp_address:
        fp_preload.extend([0.0] * (gelu_1702_fp_address + 1 - len(fp_preload)))
    fp_preload[gelu_1702_fp_address] = 1.702

    o_vram_addr = prog.get_vram_addr(current.name)
    comparison_rows = output_seq_len
    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (comparison_rows * output_padded_hidden) // mlen,
        "num_batches": comparison_rows,
        "elements_per_batch": output_padded_hidden,
        "row_dim": mlen,
        "physical_rows": current.physical_shape[0],
        "use_stride_mode": output_padded_hidden > mlen,
    }

    info = {
        "model_type": model_cfg.model_type,
        "component": "vision",
        "hidden_size": hidden,
        "inter_dim": inter,
        "padded_hidden_size": padded_hidden,
        "padded_inter_dim": padded_inter,
        "num_layers": n_layers,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "padded_seq_len": padded_seq_len,
        "output_seq_len": output_seq_len,
        "output_hidden_size": output_hidden,
        "padded_output_hidden_size": output_padded_hidden,
        "compile_seq_rows": compile_seq_rows,
        "output_rows": output_rows,
        "image_h": image_h,
        "image_w": image_w,
        "patch_size": model_cfg.patch_size,
        "num_channels": model_cfg.num_channels,
        "head_dim": head_dim,
        "padded_head_dim": padded_head_dim,
        "num_heads": num_heads,
        "mlen": mlen,
        "blen": blen,
        "mram_tile_capacity": mram_tile_capacity,
        "include_connector": connector_weights is not None,
        "vision_output_stage": output_stage,
        "vision_stop_after": requested_stop,
        "golden_precision": golden_precision,
        "isa_lines": len(lines),
    }
    if connector_weights is not None:
        info.update(
            {
                "connector_scale_factor": connector_weights.scale_factor,
                "connector_input_dim": connector_weights.input_dim,
                "connector_storage_input_dim": connector_storage_dim,
                "connector_output_dim": connector_output_dim,
            }
        )

    print(f"\nVision compilation complete: {info['isa_lines']} ISA lines, "
          f"{n_layers} layers, output at VRAM row {o_vram_addr // mlen}")
    return {
        "isa": isa_code,
        "golden_output": golden_out,
        "padded_golden_output": padded_golden_output,
        "hf_ground_truth": hf_ground_truth,
        "input_tensors": input_tensors,
        "tensor_layouts": tensor_layouts,
        "data_order": data_order,
        "fp_preload": fp_preload,
        "comparison_params": comparison_params,
        "info": info,
        "sim_golden_result": {
            "original_output": padded_golden_output,
            "tensor_layouts": tensor_layouts,
            "data_order": data_order,
            "compile_info": info,
        },
        "golden_precision": golden_precision,
        "reference_backend": "vision_scheduled",
    }


# ---------------------------------------------------------------------------
# Main compilation function
# ---------------------------------------------------------------------------
def compile_native_hf_decoder(
    model,
    seq_len: int = 64,
    batch_size: int = 1,
    hidden_size: int | None = None,
    inter_dim: int | None = None,
    num_layers: int | None = None,
    layer_idx_start: int = 0,
    mlen: int = 64,
    blen: int = 4,
    hlen: int | None = None,
    broadcast_amount: int | None = None,
    attention_head_packing: bool | None = None,
    mram_tile_capacity: int = 4,
    seed: int = 42,
    golden_precision: str = "hardware",
    reference_backend: str = "scheduled",
    stage_checkpoints: bool = False,
    verbose: bool = False,
    component: str = "decoder",
    vision_stop_after: str | None = None,
    decoder_input_embeds: torch.Tensor | None = None,
) -> dict:
    """Compile a HuggingFace decoder model at native dimensions to PLENA ISA metadata."""
    component = component.lower()
    if component in {"vision", "vision_model", "vision_encoder"}:
        return compile_native_hf_vision_encoder(
            model,
            seq_len=seq_len,
            batch_size=batch_size,
            hidden_size=hidden_size,
            inter_dim=inter_dim,
            num_layers=num_layers,
            layer_idx_start=layer_idx_start,
            mlen=mlen,
            blen=blen,
            mram_tile_capacity=mram_tile_capacity,
            seed=seed,
            golden_precision=golden_precision,
            reference_backend=reference_backend,
            include_connector=component != "vision_encoder",
            stop_after=vision_stop_after,
            stage_checkpoints=stage_checkpoints,
            verbose=verbose,
        )
    if component not in {"decoder", "text", "text_decoder"}:
        raise ValueError("component must be 'decoder' or 'vision'")

    def _verbose(message: str = ""):
        if verbose:
            print(message)

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    model_cfg = extract_model_config(model)
    if hidden_size is not None and hidden_size != model_cfg.hidden_size:
        raise ValueError(
            f"compile_native_hf_decoder currently supports native hidden size only: "
            f"requested {hidden_size}, model has {model_cfg.hidden_size}"
        )
    if inter_dim is not None and inter_dim != model_cfg.inter_dim:
        raise ValueError(
            f"compile_native_hf_decoder currently supports native inter_dim only: "
            f"requested {inter_dim}, model has {model_cfg.inter_dim}"
        )
    hidden = model_cfg.hidden_size
    inter = model_cfg.inter_dim
    num_heads = model_cfg.num_heads
    num_kv_heads = model_cfg.num_kv_heads
    head_dim = model_cfg.head_dim
    total_q_dim = model_cfg.total_q_dim
    ratio = model_cfg.head_ratio
    # Rows only need enough physical lanes for the matrix writeback group
    # (BLEN), not a full MLEN block. Columns/K dimensions still use MLEN
    # until the vector, norm, RoPE, and FFN templates learn true tail masks.
    #
    # Keep an opt-in compatibility path for reproducing older tile-scaling
    # reports that padded sequence rows to MLEN.
    seq_padding_multiple = mlen if os.environ.get("PLENA_PAD_SEQ_TO_MLEN") == "1" else blen
    padded_seq_len = _ceil_to_multiple(seq_len, seq_padding_multiple)
    rows_per_batch = (
        padded_seq_len
        if batch_size == 1
        else _ceil_to_multiple(max(mlen, padded_seq_len), mlen)
    )
    compile_seq_rows = batch_size * rows_per_batch
    active_rows = batch_size * seq_len
    checkpoint_rows = compile_seq_rows if batch_size > 1 else seq_len
    padded_hidden = _ceil_to_multiple(hidden, mlen)
    padded_inter = _ceil_to_multiple(inter, mlen)
    padded_head_dim = _ceil_to_multiple(head_dim, mlen)
    if attention_head_packing is None:
        attention_head_packing = hlen is not None and broadcast_amount is not None and head_dim < mlen
    if attention_head_packing:
        if hlen is None or broadcast_amount is None:
            raise ValueError("attention_head_packing requires hlen and broadcast_amount")
        if hlen <= 0 or broadcast_amount <= 0:
            raise ValueError(f"hlen and broadcast_amount must be positive, got {hlen}, {broadcast_amount}")
        if broadcast_amount * hlen != mlen:
            raise ValueError(
                f"Packed attention requires broadcast_amount*hlen == mlen "
                f"({broadcast_amount}*{hlen} != {mlen})"
            )
        if head_dim > hlen:
            raise ValueError(f"head_dim={head_dim} exceeds hlen={hlen}; cannot head-pack")
        if ratio > broadcast_amount:
            raise ValueError(f"GQA ratio={ratio} exceeds broadcast_amount={broadcast_amount}")
        head_packing = AttentionHeadPacking(
            enabled=True,
            hlen=hlen,
            broadcast_amount=broadcast_amount,
            head_slot_dim=hlen,
            group_width=mlen,
            total_q_dim=num_kv_heads * mlen,
        )
    else:
        head_packing = None
    padded_total_q_dim = head_packing.total_q_dim if head_packing is not None else num_heads * padded_head_dim
    padding_enabled = (
        padded_seq_len != seq_len
        or rows_per_batch != seq_len
        or batch_size != 1
        or padded_hidden != hidden
        or padded_inter != inter
        or padded_head_dim != head_dim
        or head_packing is not None
    )

    root = find_model_root(model)
    layers = root.layers
    n_layers = num_layers if num_layers is not None else len(layers)
    assert layer_idx_start + n_layers <= len(layers), (
        f"Requested layers [{layer_idx_start}, {layer_idx_start + n_layers}) "
        f"but model only has {len(layers)} layers"
    )

    scale = 1.0 / math.sqrt(head_dim)
    embed = embedding_module(root)

    print("=" * 80)
    print(f"Model Compiler - {model_cfg.model_type} ({n_layers} layer{'s' if n_layers != 1 else ''})")
    print(
        f"  decoder: hidden={hidden}, inter={inter}, heads={num_heads}/{num_kv_heads}, "
        f"head_dim={head_dim}"
    )
    print(
        f"  compile: batch_size={batch_size}, seq_len={seq_len}, mlen={mlen}, blen={blen}, "
        f"mram_tile_capacity={mram_tile_capacity}, total_q_dim={total_q_dim}"
    )
    if padding_enabled:
        print(
            "  tile padding: "
            f"seq_len={seq_len}->{padded_seq_len}, rows_per_batch={rows_per_batch}, "
            f"hidden={hidden}->{padded_hidden}, "
            f"inter={inter}->{padded_inter}, "
            f"head_dim={head_dim}->{padded_head_dim}, "
            f"total_q_dim={total_q_dim}->{padded_total_q_dim}"
        )
    if head_packing is not None:
        print(
            "  attention head packing: "
            f"head_dim={head_dim}, hlen={head_packing.hlen}, "
            f"broadcast_amount={head_packing.broadcast_amount}, "
            f"group_width={head_packing.group_width}, "
            f"q_groups={num_kv_heads}"
        )
    print("=" * 80)

    # ----------------------------------------------------------- weights
    print(f"\nExtracting weights from layers {layer_idx_start}..{layer_idx_start + n_layers - 1}...")
    all_weights = []
    for i in range(n_layers):
        layer_module = layers[layer_idx_start + i]
        w = extract_layer_weights(layer_module, model_cfg)
        all_weights.append(w)
        _verbose(
            f"  Layer {i}: W_q={w.w_q.shape}, W_o={w.w_o.shape}, "
            f"W_gate={w.w_gate.shape}, "
            f"K_heads={len(w.w_k_heads)}x{w.w_k_heads[0].shape}, eps={w.eps}"
        )
    compile_weights = [
        _pad_decoder_weights_for_tiles(
            w,
            model_cfg,
            padded_hidden=padded_hidden,
            padded_inter=padded_inter,
            padded_head_dim=padded_head_dim,
            head_packing=head_packing,
        )
        for w in all_weights
    ]

    eps = all_weights[0].eps

    torch.manual_seed(seed)

    decoder_input_source = "decoder_input_embeds" if decoder_input_embeds is not None else None
    if decoder_input_embeds is not None:
        token_embeds = decoder_input_embeds.detach().float()
        if token_embeds.dim() == 2:
            if batch_size != 1:
                raise ValueError(
                    f"2D decoder_input_embeds are only valid for batch_size=1, got {batch_size}"
                )
            if token_embeds.shape[0] != seq_len:
                raise ValueError(
                    f"Expected decoder_input_embeds shape ({seq_len}, C), got {tuple(token_embeds.shape)}"
                )
        elif token_embeds.dim() == 3:
            if token_embeds.shape[0] != batch_size or token_embeds.shape[1] != seq_len:
                raise ValueError(
                    f"Expected decoder_input_embeds shape ({batch_size}, {seq_len}, C), "
                    f"got {tuple(token_embeds.shape)}"
                )
        else:
            raise ValueError(
                f"Expected decoder_input_embeds to be 2D or 3D, got shape {tuple(token_embeds.shape)}"
            )
        if token_embeds.shape[-1] < hidden:
            raise ValueError(
                f"decoder_input_embeds hidden dim {token_embeds.shape[-1]} is smaller than model hidden {hidden}"
            )
        if batch_size == 1 and token_embeds.dim() == 3:
            token_embeds = token_embeds.squeeze(0)
        token_embeds = token_embeds[..., :hidden].contiguous()
        _verbose(f"\nDecoder input override: token_embeds={token_embeds.shape}")
    elif embed is not None:
        decoder_input_source = "embed_tokens"
        input_shape = (seq_len,) if batch_size == 1 else (batch_size, seq_len)
        input_ids = torch.randint(0, model_cfg.vocab_size or 32000, input_shape)
        with torch.no_grad():
            token_embeds = embed(input_ids).float()
        if batch_size == 1 and token_embeds.dim() == 3:
            token_embeds = token_embeds.squeeze(0)
        if batch_size > 1 and token_embeds.dim() == 2:
            token_embeds = token_embeds.unsqueeze(0)
        # Slice to the model hidden width.
        token_embeds = token_embeds[..., :hidden]
        _verbose(
            f"\nEmbedding lookup: input_ids shape={input_ids.shape}, "
            f"token_embeds={token_embeds.shape}"
        )
    else:
        decoder_input_source = "random"
        token_embeds = torch.randn(seq_len, hidden) if batch_size == 1 else torch.randn(batch_size, seq_len, hidden)
        print(f"\nNo embed_tokens found; using random token_embeds: {token_embeds.shape}")

    # Llama-style models use RoPE (not learned position embeddings).
    # Set pos_weight to zeros so embedding_add is a no-op for position.
    pos_weight = torch.zeros_like(token_embeds)
    compile_token_embeds = _pad_batched_sequence_storage(
        token_embeds,
        batch_size=batch_size,
        seq_len=seq_len,
        rows_per_batch=rows_per_batch,
        cols=padded_hidden,
    )
    compile_pos_weight = _pad_batched_sequence_storage(
        pos_weight,
        batch_size=batch_size,
        seq_len=seq_len,
        rows_per_batch=rows_per_batch,
        cols=padded_hidden,
    )

    _verbose(f"pos_weight: zeros {pos_weight.shape} (RoPE model; learned position add is a no-op)")
    for i in range(n_layers):
        for kv_h in range(num_kv_heads):
            _verbose(
                f"  W_k_{i}_h{kv_h}: {all_weights[i].w_k_heads[kv_h].shape}, "
                f"W_v_{i}_h{kv_h}: {all_weights[i].w_v_heads[kv_h].shape}"
            )
    print(f"attn_scale: {scale:.6f}")

    R_matrix, cos_table, sin_table = make_rope_inputs(seq_len, model_cfg)
    if head_packing is not None:
        compile_R_matrix, per_sequence_cos_table, per_sequence_sin_table = _pad_rope_inputs_for_head_slots(
            R_matrix,
            cos_table,
            sin_table,
            padded_seq_len=padded_seq_len,
            group_width=head_packing.group_width,
            head_slot_dim=head_packing.head_slot_dim,
            broadcast_amount=head_packing.broadcast_amount,
        )
        rope_width = head_packing.group_width
    else:
        compile_R_matrix, per_sequence_cos_table, per_sequence_sin_table = _pad_rope_inputs_for_tiles(
            R_matrix,
            cos_table,
            sin_table,
            padded_seq_len=padded_seq_len,
            padded_head_dim=padded_head_dim,
        )
        rope_width = padded_head_dim
    compile_cos_table = _repeat_sequence_storage(
        per_sequence_cos_table,
        batch_size=batch_size,
        seq_len=seq_len,
        rows_per_batch=rows_per_batch,
    )
    compile_sin_table = _repeat_sequence_storage(
        per_sequence_sin_table,
        batch_size=batch_size,
        seq_len=seq_len,
        rows_per_batch=rows_per_batch,
    )

    golden_policy = ReferencePrecision.from_mode(golden_precision)
    reference_backend = reference_backend.lower()
    print(f"\nComputing CPU golden reference ({golden_policy.label}, backend={reference_backend})")
    if reference_backend == "scheduled":
        scheduled_cfg = ScheduledReferenceConfig(
            seq_len=seq_len,
            padded_seq_len=padded_seq_len,
            batch_size=batch_size,
            rows_per_batch=rows_per_batch,
            hidden_size=hidden,
            padded_hidden_size=padded_hidden,
            inter_dim=inter,
            padded_inter_dim=padded_inter,
            head_dim=head_dim,
            padded_head_dim=padded_head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            mlen=mlen,
            blen=blen,
            max_k_tiles=mram_tile_capacity,
            attention_head_packing=head_packing is not None,
            head_slot_dim=head_packing.head_slot_dim if head_packing is not None else padded_head_dim,
            broadcast_amount=head_packing.broadcast_amount if head_packing is not None else None,
            total_q_dim=padded_total_q_dim,
        )
        padded_golden_output = run_native_decoder_scheduled_reference(
            compile_token_embeds,
            compile_pos_weight,
            compile_weights,
            scheduled_cfg,
            compile_R_matrix,
            compile_cos_table,
            compile_sin_table,
            precision=golden_policy,
            trace=lambda i, x: _verbose(f"  After layer {i}: X_gold[0,:4] = {x[0, :4].tolist()}"),
        )
        golden_out = _compact_active_sequence_rows(
            padded_golden_output,
            batch_size=batch_size,
            seq_len=seq_len,
            rows_per_batch=rows_per_batch,
            cols=hidden,
        )
    elif reference_backend == "legacy":
        if batch_size != 1:
            raise NotImplementedError("legacy reference_backend does not support batch_size > 1")
        golden_out = run_decoder_reference(
            token_embeds,
            pos_weight,
            all_weights,
            model_cfg,
            R_matrix,
            cos_table,
            sin_table,
            mlen=mlen,
            max_k_tiles=mram_tile_capacity,
            precision=golden_policy,
            trace=lambda i, x: _verbose(f"  After layer {i}: X_gold[0,:4] = {x[0, :4].tolist()}"),
        )
        padded_golden_output = _pad_2d(golden_out, padded_seq_len, padded_hidden)
    else:
        raise ValueError("reference_backend must be 'scheduled' or 'legacy'")
    print(f"  golden_out: {golden_out.shape}")
    _verbose(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    print(f"\nComputing HF reference (float32, {n_layers} layer{'s' if n_layers != 1 else ''}, no quantization)")
    with torch.no_grad():
        if batch_size == 1:
            hf_ground_truth = run_decoder_reference(
                token_embeds,
                pos_weight,
                all_weights,
                model_cfg,
                R_matrix,
                cos_table,
                sin_table,
                mlen=mlen,
                max_k_tiles=mram_tile_capacity,
                precision=ReferencePrecision.from_mode("hf_fp32"),
                trace=lambda i, x: _verbose(f"  After layer {i}: X_hf[0,:4] = {x[0, :4].tolist()}"),
            )
        else:
            padded_hf_output = run_native_decoder_scheduled_reference(
                compile_token_embeds,
                compile_pos_weight,
                compile_weights,
                scheduled_cfg,
                compile_R_matrix,
                compile_cos_table,
                compile_sin_table,
                precision=ReferencePrecision.from_mode("hf_fp32"),
                trace=lambda i, x: _verbose(f"  After layer {i}: X_hf[0,:4] = {x[0, :4].tolist()}"),
            )
            hf_ground_truth = _compact_active_sequence_rows(
                padded_hf_output,
                batch_size=batch_size,
                seq_len=seq_len,
                rows_per_batch=rows_per_batch,
                cols=hidden,
            )

    print(f"  hf_ground_truth: {hf_ground_truth.shape}")
    _verbose(f"  hf_ground_truth[0,:4]: {hf_ground_truth[0, :4].tolist()}")

    # ----------------------------------------------------------- PLENA ISA
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        real_data_ratio=REAL_DATA_RATIO,
        mram_tile_capacity=mram_tile_capacity,
    )
    if hlen is not None:
        prog.hlen = hlen
    if broadcast_amount is not None:
        prog.broadcast_amount = broadcast_amount
    checkpoints = StageCheckpointRecorder(enabled=stage_checkpoints)

    # Shared inputs
    sequence_physical_shape = (compile_seq_rows, padded_hidden)
    rope_table_physical_shape = (compile_seq_rows, rope_width)
    x_input = prog.input("X", shape=(compile_seq_rows, padded_hidden), physical_shape=sequence_physical_shape)
    pos_input = prog.input("POS", shape=(compile_seq_rows, padded_hidden), physical_shape=sequence_physical_shape)

    r_input = prog.input("R_rope", shape=(rope_width, rope_width))
    cos_input = prog.input("COS", shape=(compile_seq_rows, rope_width), physical_shape=rope_table_physical_shape)
    sin_input = prog.input("SIN", shape=(compile_seq_rows, rope_width), physical_shape=rope_table_physical_shape)
    COS = prog.load_batch(cos_input, name="COS")
    SIN = prog.load_batch(sin_input, name="SIN")

    # Causal mask: (mlen, mlen) with 0 on/below diagonal, -inf above
    causal_mask_data = torch.zeros(mlen, mlen)
    causal_mask_data.masked_fill_(
        torch.triu(torch.ones(mlen, mlen), diagonal=1).bool(), float('-inf')
    )
    causal_mask_input = prog.input("causal_mask", shape=(mlen, mlen))
    CAUSAL_MASK = prog.load_batch(causal_mask_input, name="CAUSAL_MASK")

    # Per-layer weight inputs (order determines HBM layout)
    layer_inputs = []
    for i in range(n_layers):
        layer_inputs.append(
            _register_layer_inputs(
                prog,
                i,
                compile_weights[i],
                physical_shapes=_weight_physical_shapes_for_layer(
                    i,
                    compile_weights[i],
                    head_packing=head_packing,
                    padded_head_dim=padded_head_dim,
                ),
            )
        )

    # Load activations to VRAM
    X_batch = prog.load_batch(x_input, name="X")
    POS_batch = prog.load_batch(pos_input, name="POS")
    ops.embedding_add(prog, X_batch, POS_batch)  # X += POS in-place
    checkpoints.record(
        prog,
        layer_idx=None,
        stage="embedding_add",
        tensor=X_batch,
        active_shape=(checkpoint_rows, hidden),
        semantic="token embedding plus position tensor",
    )

    # Allocate scratch buffer for residual save/restore (reused across layers)
    scratch = prog.alloc(
        "residual_scratch",
        compile_seq_rows,
        padded_hidden,
        strict=False,
        physical_shape=sequence_physical_shape,
    )

    # Chain layers
    current = X_batch

    for i in range(n_layers):
        li = layer_inputs[i]

        # Layer progress marker (visible in non-quiet emulator output)
        prog.emit_comment(f"=== LAYER {i}/{n_layers} START ===")

        if head_packing is not None:
            current_after_attn = _emit_packed_attention_block(
                prog,
                current,
                li,
                (r_input, COS, SIN),
                CAUSAL_MASK,
                scratch,
                scale,
                i,
                padded_seq_len,
                head_dim,
                num_kv_heads,
                ratio,
                head_packing,
                checkpoints,
                checkpoint_rows,
                hidden,
                batch_size=batch_size,
                rows_per_batch=rows_per_batch,
                active_seq_len_per_batch=seq_len,
            )
        else:
            if batch_size != 1:
                raise NotImplementedError("native non-packed MHA decoder lowering does not yet support batch_size > 1")
            current_after_attn = _emit_attention_block(
                prog,
                current,
                li,
                (r_input, COS, SIN),
                CAUSAL_MASK,
                scratch,
                scale,
                i,
                padded_seq_len,
                padded_head_dim,
                padded_total_q_dim,
                num_heads,
                num_kv_heads,
                ratio,
                checkpoints,
                checkpoint_rows,
                hidden,
            )

        current = _emit_ffn_block(
            prog,
            current_after_attn,
            li,
            scratch,
            layer_idx=i,
            checkpoint_recorder=checkpoints,
            active_seq_len=checkpoint_rows,
            active_hidden=hidden,
        )
        prog.emit_comment(f"=== LAYER {i}/{n_layers} COMPLETE ===")

    # Final norm
    ops.rms_norm(prog, current, eps_offset=3, reci_hid_offset=4)
    checkpoints.record(
        prog,
        layer_idx=n_layers - 1,
        stage="final_norm",
        tensor=current,
        active_shape=(checkpoint_rows, hidden),
        semantic="decoder final RMS norm output",
    )

    isa_code = prog.compile()
    isa_code = _fix_large_immediates(isa_code)
    lines = isa_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ----------------------------------------------------------- build return
    input_tensors = {
        "X": compile_token_embeds,
        "POS": compile_pos_weight,
        "R_rope": compile_R_matrix,
        "COS": compile_cos_table,
        "SIN": compile_sin_table,
    }
    data_order = ["X", "POS", "R_rope", "COS", "SIN"]
    input_tensors["causal_mask"] = causal_mask_data
    data_order.append("causal_mask")
    for i in range(n_layers):
        for name, tensor in compile_weights[i].tensor_entries(i):
            input_tensors[name] = tensor
            data_order.append(name)
    tensor_layouts = _tensor_layout_metadata(prog, input_tensors)

    # FPRAM layout (same as single-layer decoder):
    #   slot 0 = 0.0        (reserved)
    #   slot 1 = attn_scale  (flash_attention)
    #   slot 2 = -inf        (flash_attention softmax mask)
    #   slot 3 = eps         (rms_norm, offset=3)
    #   slot 4 = 1/hidden    (rms_norm, offset=4)
    #   slot 5 = 1.0         (FFN SiLU)
    #   slots 6-9 = 0.0      (padding)
    fp_preload = [0.0, scale, float("-inf"), eps, 1.0 / hidden, 1.0] + [0.0] * 4

    # Result is at current's VRAM location
    o_vram_addr = prog.get_vram_addr(current.name)

    out_features = padded_hidden
    comparison_rows = compile_seq_rows if batch_size > 1 else seq_len

    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (comparison_rows * out_features) // mlen,
        "num_batches": comparison_rows,
        "elements_per_batch": out_features,
        "row_dim": mlen,
        "physical_rows": current.physical_shape[0],
        "use_stride_mode": out_features > mlen,
    }

    info = {
        "model_type": model_cfg.model_type,
        "hidden_size": hidden,
        "inter_dim": inter,
        "padded_hidden_size": padded_hidden,
        "padded_inter_dim": padded_inter,
        "num_layers": n_layers,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "padded_seq_len": padded_seq_len,
        "rows_per_batch": rows_per_batch,
        "compile_seq_rows": compile_seq_rows,
        "active_rows": active_rows,
        "head_dim": head_dim,
        "padded_head_dim": padded_head_dim,
        "attention_head_packing": head_packing is not None,
        "attention_head_slot_dim": head_packing.head_slot_dim if head_packing is not None else padded_head_dim,
        "attention_kv_storage_head_dim": padded_head_dim,
        "attention_broadcast_amount": head_packing.broadcast_amount if head_packing is not None else None,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "mlen": mlen,
        "blen": blen,
        "mram_tile_capacity": mram_tile_capacity,
        "stage_checkpoints_enabled": stage_checkpoints,
        "stage_checkpoint_count": len(checkpoints.checkpoints or []),
        "reference_backend": reference_backend,
        "golden_precision": golden_precision,
        "decoder_input_source": decoder_input_source,
        "padding_enabled": padding_enabled,
        "isa_lines": len(lines),
    }
    stage_checkpoint_metadata = checkpoints.metadata()
    stage_checkpoint_metadata["compile_info"] = {
        key: info[key]
        for key in (
            "model_type",
            "hidden_size",
            "inter_dim",
            "padded_hidden_size",
            "padded_inter_dim",
            "num_layers",
            "batch_size",
            "seq_len",
            "padded_seq_len",
            "rows_per_batch",
            "compile_seq_rows",
            "active_rows",
            "head_dim",
            "padded_head_dim",
            "attention_head_packing",
            "attention_head_slot_dim",
            "attention_kv_storage_head_dim",
            "attention_broadcast_amount",
            "num_heads",
            "num_kv_heads",
            "mlen",
            "blen",
            "mram_tile_capacity",
        )
    }

    print(f"\nCompilation complete: {info['isa_lines']} ISA lines, "
          f"{n_layers} layers, output at VRAM row {o_vram_addr // mlen}")
    return {
        "isa": isa_code,
        "golden_output": golden_out,
        "padded_golden_output": padded_golden_output,
        "hf_ground_truth": hf_ground_truth,
        "input_tensors": input_tensors,
        "tensor_layouts": tensor_layouts,
        "data_order": data_order,
        "fp_preload": fp_preload,
        "comparison_params": comparison_params,
        "info": info,
        "stage_checkpoints": stage_checkpoint_metadata,
        "sim_golden_result": {
            "original_output": padded_golden_output,
            "tensor_layouts": tensor_layouts,
            "data_order": data_order,
            "compile_info": info,
            "stage_checkpoints": stage_checkpoint_metadata,
        },
        "golden_precision": golden_precision,
        "reference_backend": reference_backend,
    }
# Backwards-compatible alias for older callers.
compile_hf_model = compile_native_hf_decoder
