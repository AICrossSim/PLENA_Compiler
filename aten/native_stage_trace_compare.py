"""Compare native decoder checkpoints against a propagated scheduled reference.

The existing VRAM stage checker validates most kernels using checkpointed
emulator inputs. That is useful for isolating local kernel bugs, but it can
miss chain-level errors. This diagnostic starts from the original build inputs,
propagates the scheduled reference stage by stage, and compares each saved
checkpoint to that propagated tensor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_COMPILER_ROOT = Path(__file__).resolve().parents[1]
_SIM_ROOT = _COMPILER_ROOT.parent
for _path in (_SIM_ROOT, _SIM_ROOT / "tools", _COMPILER_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from compiler.aten.model_extract import LayerWeights
from compiler.aten.reference import (  # noqa: E402
    ReferencePrecision,
    ScheduledReferenceConfig,
    _flash_attn_scheduled_ref,
    _hbm_round_ref,
    _linear_scheduled_ref,
    _pad_cols_ref,
    _residual_add_ref,
    _rms_norm_scheduled_ref,
    _rope_scheduled_ref,
    _round,
    _silu_gate_scheduled_ref,
    _vram_load_ref,
)
from aten.vram_stage_compare import _read_checkpoint  # noqa: E402


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _load_tensor(build: Path, name: str) -> torch.Tensor:
    return torch.load(build / name, map_location="cpu", weights_only=True).float()


def _load_layer(build: Path, layer_idx: int, num_kv_heads: int) -> LayerWeights:
    return LayerWeights(
        w_q=_load_tensor(build, f"W_q_{layer_idx}.pt"),
        w_o=_load_tensor(build, f"W_o_{layer_idx}.pt"),
        w_k_heads=[_load_tensor(build, f"W_k_{layer_idx}_h{h}.pt") for h in range(num_kv_heads)],
        w_v_heads=[_load_tensor(build, f"W_v_{layer_idx}_h{h}.pt") for h in range(num_kv_heads)],
        w_gate=_load_tensor(build, f"W_gate_{layer_idx}.pt"),
        w_up=_load_tensor(build, f"W_up_{layer_idx}.pt"),
        w_down=_load_tensor(build, f"W_down_{layer_idx}.pt"),
        eps=1e-5,
    )


def _lookup(metadata: dict) -> dict[tuple[int | None, str], dict]:
    return {
        (entry.get("layer_idx"), entry["stage"]): entry
        for entry in metadata.get("checkpoints", [])
    }


def _pad_to_actual(expected: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    if expected.shape == actual.shape:
        return expected
    if expected.shape[0] > actual.shape[0] or expected.shape[1] > actual.shape[1]:
        raise ValueError(f"Expected shape {tuple(expected.shape)} exceeds actual {tuple(actual.shape)}")
    out = torch.zeros_like(actual)
    out[: expected.shape[0], : expected.shape[1]] = expected
    return out


def _metrics(expected: torch.Tensor, actual: torch.Tensor, active_rows: int, active_cols: int) -> dict:
    expected = _pad_to_actual(expected, actual)
    exp = expected[:active_rows, :active_cols].bfloat16()
    act = actual[:active_rows, :active_cols].bfloat16()
    close = torch.isclose(act, exp, atol=0.2, rtol=0.2)
    diff = (act.float() - exp.float()).abs()
    return {
        "active_allclose": float(close.float().mean().item() * 100.0),
        "mse": float(((act.float() - exp.float()) ** 2).mean().item()),
        "mae": float(diff.mean().item()),
        "max_error": float(diff.max().item()),
    }


def _compare(
    results: list[dict],
    lookup: dict[tuple[int | None, str], dict],
    vram_path: Path,
    mlen: int,
    layer_idx: int | None,
    stage: str,
    expected: torch.Tensor,
) -> torch.Tensor | None:
    entry = lookup.get((layer_idx, stage))
    if entry is None:
        return None
    actual = _read_checkpoint(vram_path, entry, mlen)
    active_rows, active_cols = entry["active_shape"]
    row = {
        "layer_idx": layer_idx,
        "stage": stage,
        **_metrics(expected, actual, int(active_rows), int(active_cols)),
    }
    results.append(row)
    return actual


def _packed_attention_expected(
    q_full: torch.Tensor,
    x_normed: torch.Tensor,
    layer: LayerWeights,
    config: ScheduledReferenceConfig,
    rope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    precision: ReferencePrecision,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    if config.head_slot_dim is None or config.broadcast_amount is None:
        raise ValueError("packed attention config requires head_slot_dim and broadcast_amount")

    rows = q_full.shape[0]
    group_width = config.packed_group_width
    head_slot_dim = config.head_slot_dim
    ratio = config.head_ratio
    scale = 1.0 / (config.head_dim ** 0.5)
    out = torch.zeros((rows, config.attention_width), dtype=q_full.dtype)

    k_proj = []
    v_proj = []
    k_rope = []
    for kv_h in range(config.num_kv_heads):
        k_h = _linear_scheduled_ref(x_normed, layer.w_k_heads[kv_h], config, precision)
        v_h = _linear_scheduled_ref(x_normed, layer.w_v_heads[kv_h], config, precision)
        k_proj.append(k_h)
        v_proj.append(v_h)
        k_h_full = _pad_cols_ref(k_h, group_width)
        v_h_full = _pad_cols_ref(v_h, group_width)
        k_h_full = _rope_scheduled_ref(k_h_full, rope, cos, sin, config, precision)
        k_rope.append(k_h_full)
        k_for_attn = _hbm_round_ref(k_h_full, precision)
        v_for_attn = _hbm_round_ref(v_h_full, precision)

        group_start = kv_h * group_width
        q_group = q_full[:, group_start:group_start + group_width]
        q_group = _rope_scheduled_ref(q_group, rope, cos, sin, config, precision)
        for lane in range(ratio):
            lane_start = lane * head_slot_dim
            lane_end = lane_start + head_slot_dim
            o_h = _flash_attn_scheduled_ref(
                q_group[:, lane_start:lane_end],
                k_for_attn[:, :head_slot_dim],
                v_for_attn[:, :head_slot_dim],
                scale,
                precision,
                causal=True,
                matmul_scale=0.25,
            )
            out[:, group_start + lane_start:group_start + lane_end] = o_h

    return _round(out, precision), k_proj, v_proj, k_rope


def compare_trace(build_dir: Path, vram_path: Path) -> dict:
    info = _load_json(build_dir / "compile_info.json")
    metadata = _load_json(build_dir / "stage_checkpoints.json")
    lookup = _lookup(metadata)

    precision = ReferencePrecision.from_mode(info.get("golden_precision", "hardware"))
    config = ScheduledReferenceConfig(
        seq_len=int(info["seq_len"]),
        padded_seq_len=int(info["padded_seq_len"]),
        hidden_size=int(info["hidden_size"]),
        padded_hidden_size=int(info["padded_hidden_size"]),
        inter_dim=int(info["inter_dim"]),
        padded_inter_dim=int(info["padded_inter_dim"]),
        head_dim=int(info["head_dim"]),
        padded_head_dim=int(info["padded_head_dim"]),
        num_heads=int(info["num_heads"]),
        num_kv_heads=int(info["num_kv_heads"]),
        mlen=int(info["mlen"]),
        blen=int(info["blen"]),
        max_k_tiles=int(info.get("mram_tile_capacity", 4)),
        attention_head_packing=bool(info.get("attention_head_packing", False)),
        head_slot_dim=info.get("attention_head_slot_dim"),
        broadcast_amount=info.get("attention_broadcast_amount"),
        total_q_dim=info.get("attention_broadcast_amount") and int(info["num_kv_heads"]) * int(info["mlen"]),
        batch_size=int(info.get("batch_size", 1)),
        rows_per_batch=int(info.get("rows_per_batch", info["padded_seq_len"])),
    )

    x = _vram_load_ref(_load_tensor(build_dir, "X.pt").clone(), precision)
    pos = _vram_load_ref(_load_tensor(build_dir, "POS.pt"), precision)
    x = _round(x + pos, precision)
    rope = _load_tensor(build_dir, "R_rope.pt")
    cos = _vram_load_ref(_load_tensor(build_dir, "COS.pt"), precision)
    sin = _vram_load_ref(_load_tensor(build_dir, "SIN.pt"), precision)

    results: list[dict] = []
    _compare(results, lookup, vram_path, config.mlen, None, "embedding_add", x)

    for layer_idx in range(int(info["num_layers"])):
        layer = _load_layer(build_dir, layer_idx, config.num_kv_heads)
        _compare(results, lookup, vram_path, config.mlen, layer_idx, "attn_input", x)
        residual = x.clone()
        x_normed = _rms_norm_scheduled_ref(x, config.hidden_size, layer.eps, config.mlen, precision)
        _compare(results, lookup, vram_path, config.mlen, layer_idx, "attn_norm", x_normed)
        q_full = _linear_scheduled_ref(x_normed, layer.w_q, config, precision)
        _compare(results, lookup, vram_path, config.mlen, layer_idx, "q_full", q_full)

        if not config.attention_head_packing:
            raise NotImplementedError("native_stage_trace_compare currently handles packed attention only")
        o_full, k_proj, v_proj, k_rope = _packed_attention_expected(
            q_full,
            x_normed,
            layer,
            config,
            rope,
            cos,
            sin,
            precision,
        )
        for kv_h in range(config.num_kv_heads):
            _compare(results, lookup, vram_path, config.mlen, layer_idx, f"k_proj_h{kv_h}", k_proj[kv_h])
            _compare(results, lookup, vram_path, config.mlen, layer_idx, f"v_proj_h{kv_h}", v_proj[kv_h])
            _compare(results, lookup, vram_path, config.mlen, layer_idx, f"k_rope_h{kv_h}", k_rope[kv_h])
        _compare(results, lookup, vram_path, config.mlen, layer_idx, "o_full", o_full)

        o_proj = _linear_scheduled_ref(o_full, layer.w_o, config, precision)
        _compare(results, lookup, vram_path, config.mlen, layer_idx, "o_proj", o_proj)
        x = _residual_add_ref(o_proj, residual, precision)
        _compare(results, lookup, vram_path, config.mlen, layer_idx, "attn_residual", x)
        _compare(results, lookup, vram_path, config.mlen, layer_idx, "ffn_input", x)

        residual = x.clone()
        x_normed = _rms_norm_scheduled_ref(x, config.hidden_size, layer.eps, config.mlen, precision)
        _compare(results, lookup, vram_path, config.mlen, layer_idx, "ffn_norm", x_normed)
        up = _linear_scheduled_ref(x_normed, layer.w_up, config, precision)
        gate = _linear_scheduled_ref(x_normed, layer.w_gate, config, precision)
        ffn_mid = _silu_gate_scheduled_ref(up, gate, precision)
        ffn_out = _linear_scheduled_ref(ffn_mid, layer.w_down, config, precision)
        _compare(results, lookup, vram_path, config.mlen, layer_idx, "ffn_out", ffn_out)
        x = _residual_add_ref(ffn_out, residual, precision)
        _compare(results, lookup, vram_path, config.mlen, layer_idx, "ffn_residual", x)

    final_norm = _rms_norm_scheduled_ref(x, config.hidden_size, layer.eps, config.mlen, precision)
    _compare(results, lookup, vram_path, config.mlen, int(info["num_layers"]) - 1, "final_norm", final_norm)

    first_bad = next((r for r in results if r["active_allclose"] < 99.0), None)
    return {
        "build_dir": str(build_dir),
        "vram_path": str(vram_path),
        "first_bad": first_bad,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vram_path", type=Path)
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = compare_trace(args.build_dir, args.vram_path)
    for row in report["results"]:
        prefix = "global" if row["layer_idx"] is None else f"layer={row['layer_idx']}"
        status = "BAD" if row["active_allclose"] < 99.0 else "ok "
        print(
            f"{status} {prefix:<8} {row['stage']:<16} "
            f"{row['active_allclose']:6.2f}% mse={row['mse']:.6g} "
            f"mae={row['mae']:.6g} max={row['max_error']:.6g}"
        )
    if report["first_bad"] is not None:
        bad = report["first_bad"]
        print(f"FIRST_BAD layer={bad['layer_idx']} stage={bad['stage']} allclose={bad['active_allclose']:.2f}%")
    else:
        print("FIRST_BAD none")

    if args.json_out is not None:
        with args.json_out.open("w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
