"""HuggingFace decoder model to PLENA ISA compiler."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any

import torch

from compiler.aten.model_extract import (
    LayerWeights,
    embedding_module,
    extract_layer_weights,
    extract_model_config,
    find_model_root,
)
import compiler.aten.ops as ops
from asm_templates._imm import add_large_int as _add_large_int_lines
from asm_templates._imm import load_large_int as _load_large_int_lines
from compiler.aten.ops.registry import Backend, OpRegistry
from compiler.aten.plena import PlenaCompiler
from compiler.aten.reference import (
    ReferencePrecision,
    ScheduledReferenceConfig,
    _ksplit_matmul,
    _make_rotate_half_matrix,
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


def _copy_into_vram_view(prog, source, name, rows, cols, vram_addr):
    target = prog.alloc_at(name, rows, cols, vram_addr)
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
        source_shape = tuple(int(dim) for dim in tensor.shape)
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
) -> dict:
    """Compile a HuggingFace decoder model at native dimensions to PLENA ISA metadata."""
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

    if embed is not None:
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
