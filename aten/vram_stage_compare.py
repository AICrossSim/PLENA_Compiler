"""VRAM-in-the-loop stage comparison for native decoder compiles.

When a build contains ``stage_checkpoints.json``, this checker validates the
preserved VRAM stage boundaries directly. For older builds it falls back to the
legacy final-layer FFN check that starts from the emulator's residual scratch.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_COMPILER_ROOT = Path(__file__).resolve().parents[1]
_SIM_ROOT = _COMPILER_ROOT.parent
for _path in (_SIM_ROOT, _SIM_ROOT / "tools", _COMPILER_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from compiler.aten.plena_frontend import _ksplit_matmul, quantize_to_mxfp


_HW_MAX_K_TILES = 4


def _read_bf16_matrix(vram_path, addr: int, rows: int, cols: int, mlen: int = 64) -> torch.Tensor:
    """Read a VRAM matrix stored in column-tile layout as float32."""
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows and cols must be positive, got rows={rows}, cols={cols}")
    if cols % mlen != 0:
        raise ValueError(f"cols ({cols}) must be a multiple of mlen ({mlen})")

    out = np.zeros((rows, cols), dtype=np.float32)
    with open(vram_path, "rb") as f:
        for tile in range(cols // mlen):
            for row in range(rows):
                f.seek((addr + tile * rows * mlen + row * mlen) * 2)
                raw = f.read(mlen * 2)
                if len(raw) != mlen * 2:
                    raise EOFError(
                        f"Could not read {mlen * 2} bytes at VRAM element offset "
                        f"{addr + tile * rows * mlen + row * mlen}"
                    )
                for col in range(mlen):
                    u16 = raw[col * 2] | (raw[col * 2 + 1] << 8)
                    out[row, tile * mlen + col] = struct.unpack("f", struct.pack("I", u16 << 16))[0]
    return torch.tensor(out)


def _allclose_pct(a: torch.Tensor, b: torch.Tensor, atol: float = 0.2, rtol: float = 0.2) -> float:
    return torch.isclose(a.bfloat16(), b.bfloat16(), atol=atol, rtol=rtol).float().mean().item() * 100


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()) ** 2).mean().item()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def _infer_final_layer_idx(build: Path) -> int:
    indices = []
    for path in build.glob("W_o_*.pt"):
        match = re.fullmatch(r"W_o_(\d+)\.pt", path.name)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        raise FileNotFoundError(f"No W_o_<layer>.pt files found in {build}")
    return max(indices)


def _read_alloc_addr(asm: str, name: str) -> int | None:
    match = re.search(rf"Allocate VRAM Matrix {re.escape(name)}: .*?VRAM\[(\d+)\]", asm)
    return int(match.group(1)) if match else None


def _load_weight(build: Path, name: str) -> torch.Tensor:
    return quantize_to_mxfp(torch.load(build / name, map_location="cpu", weights_only=True))


def _load_tensor(build: Path, name: str, *, quantized: bool = False) -> torch.Tensor:
    tensor = torch.load(build / name, map_location="cpu", weights_only=True).float()
    return quantize_to_mxfp(tensor) if quantized else tensor


def _round_hw(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16).float()


def _linear_hw(A: torch.Tensor, B: torch.Tensor, mlen: int, max_k_tiles: int) -> torch.Tensor:
    return _round_hw(
        _ksplit_matmul(
            A,
            B,
            mlen,
            max_k_tiles,
            to_inter=lambda x: x.to(torch.bfloat16),
            from_inter=lambda x: x.float(),
        )
    )


def _rope_hw(x: torch.Tensor, rope: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    rope_slice = rope[:cols, :cols]
    x_inter = _round_hw(x)
    cos_slice = _pad_expected_to_actual(cos[:rows, :cols], x_inter)
    sin_slice = _pad_expected_to_actual(sin[:rows, :cols], x_inter)
    x_rot = _round_hw(torch.matmul(x_inter, _round_hw(rope_slice)))
    x_cos = _round_hw(x_inter * _round_hw(cos_slice))
    x_rot_sin = _round_hw(x_rot * _round_hw(sin_slice))
    return _round_hw(x_cos + x_rot_sin)


def _rms_norm_padded(x: torch.Tensor, active_hidden: int, eps: float) -> torch.Tensor:
    """RMS norm for padded physical hidden storage.

    Native compiler output can store hidden columns padded to MLEN, while FPRAM
    still stores 1/native_hidden. Padded columns are expected to be zero, so the
    scalar denominator must remain the active hidden width.
    """
    x_bf = x.to(torch.bfloat16)
    rms = torch.rsqrt(x_bf.float().pow(2).sum(-1, keepdim=True) / float(active_hidden) + eps)
    return (x_bf.float() * rms).to(torch.bfloat16).float()


def _compare_region(
    expected: torch.Tensor,
    actual: torch.Tensor,
    active_rows: int,
    active_cols: int,
) -> dict[str, float]:
    active_expected = expected[:active_rows, :active_cols]
    active_actual = actual[:active_rows, :active_cols]
    return {
        "full_allclose": _allclose_pct(expected, actual),
        "active_allclose": _allclose_pct(active_expected, active_actual),
        "full_mse": _mse(expected, actual),
        "active_mse": _mse(active_expected, active_actual),
    }


def _pad_expected_to_actual(expected: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    if expected.shape == actual.shape:
        return expected
    if expected.shape[0] > actual.shape[0] or expected.shape[1] > actual.shape[1]:
        raise ValueError(f"Expected shape {tuple(expected.shape)} exceeds actual shape {tuple(actual.shape)}")
    padded = torch.zeros_like(actual)
    padded[: expected.shape[0], : expected.shape[1]] = expected
    return padded


def _load_stage_checkpoint_metadata(build: Path) -> dict:
    metadata = _load_json(build / "stage_checkpoints.json")
    if not metadata:
        return {}
    if isinstance(metadata, list):
        return {"schema_version": 0, "checkpoints": metadata}
    return metadata


def _checkpoint_lookup(metadata: dict) -> dict[tuple[int | None, str], dict]:
    lookup = {}
    for entry in metadata.get("checkpoints", []):
        lookup[(entry.get("layer_idx"), entry["stage"])] = entry
    return lookup


def _read_checkpoint(vram_path, entry: dict, mlen: int) -> torch.Tensor:
    physical_rows, physical_cols = entry["physical_shape"]
    return _read_bf16_matrix(vram_path, int(entry["vram_addr"]), physical_rows, physical_cols, mlen=mlen)


def _stage_result(stage: str, expected: torch.Tensor, actual: torch.Tensor, entry: dict) -> dict[str, float | str]:
    expected = _pad_expected_to_actual(expected, actual)
    active_rows, active_cols = entry["active_shape"]
    result = _compare_region(expected, actual, int(active_rows), int(active_cols))
    return {
        "stage": stage,
        "active_allclose": result["active_allclose"],
        "full_allclose": result["full_allclose"],
        "active_mse": result["active_mse"],
        "full_mse": result["full_mse"],
        "semantic": entry.get("semantic", ""),
    }


def _append_stage_result(
    stage_results: list[dict],
    lookup: dict[tuple[int | None, str], dict],
    vram_path,
    mlen: int,
    layer_idx: int | None,
    stage: str,
    expected: torch.Tensor,
) -> torch.Tensor | None:
    entry = lookup.get((layer_idx, stage))
    if entry is None:
        return None
    actual = _read_checkpoint(vram_path, entry, mlen)
    stage_results.append(_stage_result(stage, expected, actual, entry))
    return actual


def _compare_checkpointed_stages(
    vram_path,
    build: Path,
    info: dict,
    params: dict,
    metadata: dict,
    *,
    hidden: int,
    inter: int,
    seq_len: int,
    mlen: int,
    eps: float,
    verbose: bool,
    layer_idx: int | None,
) -> dict:
    del params

    lookup = _checkpoint_lookup(metadata)
    if layer_idx is None:
        layer_indices = sorted(
            {
                int(entry["layer_idx"])
                for entry in metadata.get("checkpoints", [])
                if entry.get("layer_idx") is not None
            }
        )
    else:
        layer_indices = [layer_idx]
    layer_indices = [idx for idx in layer_indices if (idx, "attn_input") in lookup or (idx, "ffn_input") in lookup]
    if not layer_indices:
        raise ValueError(f"No layer checkpoints found in {build / 'stage_checkpoints.json'}")

    padded_hidden = int(info.get("padded_hidden_size", hidden))
    padded_inter = int(info.get("padded_inter_dim", inter))
    padded_seq_len = int(info.get("padded_seq_len", seq_len))
    max_k_tiles = int(info.get("mram_tile_capacity", _HW_MAX_K_TILES))
    num_kv_heads = int(info.get("num_kv_heads", 1))

    rope = _load_tensor(build, "R_rope.pt", quantized=True)
    cos = _load_tensor(build, "COS.pt", quantized=True)
    sin = _load_tensor(build, "SIN.pt", quantized=True)

    stage_results = []
    results = {
        "mode": "checkpointed",
        "layers": layer_indices,
        "seq_len": seq_len,
        "hidden": hidden,
        "inter": inter,
        "mlen": mlen,
        "padded_seq_len": padded_seq_len,
        "padded_hidden": padded_hidden,
        "padded_inter": padded_inter,
        "stage_results": stage_results,
    }

    for idx in layer_indices:
        if verbose:
            print(f"  Validating checkpointed layer {idx}")
        W_q = _load_weight(build, f"W_q_{idx}.pt")
        W_o = _load_weight(build, f"W_o_{idx}.pt")
        W_gate = _load_weight(build, f"W_gate_{idx}.pt")
        W_up = _load_weight(build, f"W_up_{idx}.pt")
        W_down = _load_weight(build, f"W_down_{idx}.pt")

        attn_input_entry = lookup.get((idx, "attn_input"))
        attn_norm_entry = lookup.get((idx, "attn_norm"))
        if attn_input_entry is not None and attn_norm_entry is not None:
            attn_input = _read_checkpoint(vram_path, attn_input_entry, mlen)
            attn_norm_expected = _rms_norm_padded(attn_input, hidden, eps)
            attn_norm = _append_stage_result(
                stage_results,
                lookup,
                vram_path,
                mlen,
                idx,
                "attn_norm",
                attn_norm_expected,
            )
            if attn_norm is not None:
                q_expected = _linear_hw(attn_norm, W_q, mlen, max_k_tiles)
                _append_stage_result(stage_results, lookup, vram_path, mlen, idx, "q_full", q_expected)

                for kv_h in range(num_kv_heads):
                    W_k = _load_weight(build, f"W_k_{idx}_h{kv_h}.pt")
                    W_v = _load_weight(build, f"W_v_{idx}_h{kv_h}.pt")
                    k_expected = _linear_hw(attn_norm, W_k, mlen, max_k_tiles)
                    v_expected = _linear_hw(attn_norm, W_v, mlen, max_k_tiles)
                    _append_stage_result(stage_results, lookup, vram_path, mlen, idx, f"k_proj_h{kv_h}", k_expected)
                    _append_stage_result(stage_results, lookup, vram_path, mlen, idx, f"v_proj_h{kv_h}", v_expected)
                    k_rope_expected = _rope_hw(k_expected, rope, cos, sin)
                    _append_stage_result(stage_results, lookup, vram_path, mlen, idx, f"k_rope_h{kv_h}", k_rope_expected)

        o_full_entry = lookup.get((idx, "o_full"))
        o_proj_entry = lookup.get((idx, "o_proj"))
        if o_full_entry is not None and o_proj_entry is not None:
            o_full = _read_checkpoint(vram_path, o_full_entry, mlen)
            o_proj_expected = _linear_hw(o_full, W_o, mlen, max_k_tiles)
            o_proj = _append_stage_result(stage_results, lookup, vram_path, mlen, idx, "o_proj", o_proj_expected)

            attn_input_entry = lookup.get((idx, "attn_input"))
            if o_proj is not None and attn_input_entry is not None:
                attn_input = _read_checkpoint(vram_path, attn_input_entry, mlen)
                attn_input = _pad_expected_to_actual(attn_input, o_proj)
                attn_resid_expected = _round_hw(o_proj + attn_input)
                _append_stage_result(
                    stage_results,
                    lookup,
                    vram_path,
                    mlen,
                    idx,
                    "attn_residual",
                    attn_resid_expected,
                )

        ffn_input_entry = lookup.get((idx, "ffn_input"))
        ffn_norm_entry = lookup.get((idx, "ffn_norm"))
        if ffn_input_entry is not None and ffn_norm_entry is not None:
            ffn_input = _read_checkpoint(vram_path, ffn_input_entry, mlen)
            ffn_norm_expected = _rms_norm_padded(ffn_input, hidden, eps)
            ffn_norm = _append_stage_result(
                stage_results,
                lookup,
                vram_path,
                mlen,
                idx,
                "ffn_norm",
                ffn_norm_expected,
            )
            if ffn_norm is not None:
                up_out = _linear_hw(ffn_norm, W_up, mlen, max_k_tiles)
                gate_out = _linear_hw(ffn_norm, W_gate, mlen, max_k_tiles)
                silu_gate = _round_hw(F.silu(up_out) * gate_out)
                ffn_out_expected = _linear_hw(silu_gate, W_down, mlen, max_k_tiles)
                ffn_out = _append_stage_result(stage_results, lookup, vram_path, mlen, idx, "ffn_out", ffn_out_expected)
                if ffn_out is not None:
                    ffn_input = _pad_expected_to_actual(ffn_input, ffn_out)
                    ffn_resid_expected = _round_hw(ffn_out + ffn_input)
                    ffn_resid = _append_stage_result(
                        stage_results,
                        lookup,
                        vram_path,
                        mlen,
                        idx,
                        "ffn_residual",
                        ffn_resid_expected,
                    )
                    if ffn_resid is not None and (idx, "final_norm") in lookup:
                        final_expected = _rms_norm_padded(ffn_resid, hidden, eps)
                        _append_stage_result(
                            stage_results,
                            lookup,
                            vram_path,
                            mlen,
                            idx,
                            "final_norm",
                            final_expected,
                        )

    if not stage_results:
        raise ValueError("Checkpoint metadata was present, but no comparable stages were found")

    min_active = min(float(row["active_allclose"]) for row in stage_results)
    results["min_active_allclose"] = min_active
    results["passed"] = min_active >= 99.0

    if verbose:
        print("  Stage results:")
        for row in stage_results:
            print(
                f"    {row['stage']:<16} {row['active_allclose']:6.1f}% active "
                f"{row['full_allclose']:6.1f}% full  mse={row['active_mse']:.4e}"
            )
        print(f"  ==> {'PASS' if results['passed'] else 'DIVERGENCE'}: checkpointed stage chain")

    return results


def compare_stages(
    vram_path,
    build_dir,
    hidden,
    inter,
    num_heads,
    num_kv_heads,
    seq_len=64,
    mlen=64,
    head_dim=64,
    eps=1e-5,
    verbose=True,
    layer_idx=None,
    padded_hidden=None,
    padded_inter=None,
    padded_seq_len=None,
):
    """Compare the final layer's FFN segment from emulator VRAM intermediates.

    ``hidden``/``inter``/``seq_len`` are active dimensions. The optional
    ``padded_*`` dimensions describe storage. If ``compile_info.json`` is
    present, it overrides these arguments.
    """
    del num_heads, num_kv_heads, head_dim

    build = Path(build_dir)
    params = _load_json(build / "comparison_params.json")
    info = _load_json(build / "compile_info.json")
    if not params:
        raise FileNotFoundError(f"Missing comparison_params.json in {build}")

    checkpoint_metadata = _load_stage_checkpoint_metadata(build)
    if not info and checkpoint_metadata.get("compile_info"):
        info = checkpoint_metadata["compile_info"]

    hidden = int(info.get("hidden_size", hidden))
    inter = int(info.get("inter_dim", inter))
    seq_len = int(info.get("seq_len", seq_len))
    mlen = int(info.get("mlen", mlen))
    padded_hidden = int(info.get("padded_hidden_size", padded_hidden or hidden))
    padded_inter = int(info.get("padded_inter_dim", padded_inter or inter))
    padded_seq_len = int(info.get("padded_seq_len", padded_seq_len or params.get("physical_rows", seq_len)))

    if checkpoint_metadata.get("checkpoints"):
        return _compare_checkpointed_stages(
            vram_path,
            build,
            info,
            params,
            checkpoint_metadata,
            hidden=hidden,
            inter=inter,
            seq_len=seq_len,
            mlen=mlen,
            eps=eps,
            verbose=verbose,
            layer_idx=layer_idx,
        )

    if layer_idx is None:
        layer_idx = _infer_final_layer_idx(build)

    final_addr = int(params["start_row_idx"]) * mlen
    results = {
        "layer_idx": layer_idx,
        "seq_len": seq_len,
        "hidden": hidden,
        "inter": inter,
        "mlen": mlen,
        "padded_seq_len": padded_seq_len,
        "padded_hidden": padded_hidden,
        "padded_inter": padded_inter,
    }

    to_inter = lambda x: x.to(torch.bfloat16)
    from_inter = lambda x: x.float()

    W_o = _load_weight(build, f"W_o_{layer_idx}.pt")
    W_gate = _load_weight(build, f"W_gate_{layer_idx}.pt")
    W_up = _load_weight(build, f"W_up_{layer_idx}.pt")
    W_down = _load_weight(build, f"W_down_{layer_idx}.pt")

    asm_path = build / "generated_asm_code.asm"
    scratch_addr = None
    o_full_addr = final_addr - 2 * padded_seq_len * padded_hidden
    if asm_path.exists():
        asm = asm_path.read_text()
        parsed_o_full_addr = _read_alloc_addr(asm, f"O_full_{layer_idx}")
        if parsed_o_full_addr is not None:
            o_full_addr = parsed_o_full_addr
        scratch_addr = _read_alloc_addr(asm, "residual_scratch")

    if scratch_addr is None:
        raise ValueError("Could not find residual_scratch allocation in generated ASM")

    if verbose:
        print(f"  Validating layer {layer_idx}")
        print(
            f"  active shape=({seq_len}, {hidden}), "
            f"storage shape=({padded_seq_len}, {padded_hidden}), mlen={mlen}"
        )

    O_full = _read_bf16_matrix(vram_path, o_full_addr, padded_seq_len, padded_hidden, mlen=mlen)
    O_proj_expected = _ksplit_matmul(O_full, W_o, mlen, _HW_MAX_K_TILES, to_inter, from_inter)
    O_proj_expected = from_inter(to_inter(O_proj_expected))

    scratch_vram = _read_bf16_matrix(vram_path, scratch_addr, padded_seq_len, padded_hidden, mlen=mlen)

    # The residual itself is not preserved after the FFN prologue. Recovering it
    # from scratch makes this a sanity check for shape/address math, not a
    # proof of O-projection correctness.
    x_residual = from_inter(to_inter(scratch_vram - O_proj_expected))
    o_proj_resid = from_inter(to_inter(O_proj_expected + x_residual))
    resid_result = _compare_region(o_proj_resid, scratch_vram, seq_len, hidden)
    results["O_proj+resid"] = resid_result["active_allclose"]
    results["O_proj+resid_full"] = resid_result["full_allclose"]
    results["O_proj+resid_mse"] = resid_result["active_mse"]

    X_pre_ffn = scratch_vram.clone()
    X_normed = _rms_norm_padded(X_pre_ffn, hidden, eps)

    up_out = to_inter(_ksplit_matmul(X_normed, W_up, mlen, _HW_MAX_K_TILES, to_inter, from_inter))
    gate_out = to_inter(_ksplit_matmul(X_normed, W_gate, mlen, _HW_MAX_K_TILES, to_inter, from_inter))
    silu_gate = to_inter(F.silu(from_inter(up_out)) * from_inter(gate_out))
    ffn_out = _ksplit_matmul(from_inter(silu_gate), W_down, mlen, _HW_MAX_K_TILES, to_inter, from_inter)
    ffn_out = from_inter(to_inter(ffn_out))

    post_ffn = from_inter(to_inter(ffn_out + X_pre_ffn))
    expected_final = _rms_norm_padded(post_ffn, hidden, eps)

    final_vram = _read_bf16_matrix(vram_path, final_addr, padded_seq_len, padded_hidden, mlen=mlen)
    final_result = _compare_region(expected_final, final_vram, seq_len, hidden)
    results["norm+FFN+norm"] = final_result["active_allclose"]
    results["norm+FFN+norm_full"] = final_result["full_allclose"]
    results["final_mse"] = final_result["active_mse"]
    results["final_mse_full"] = final_result["full_mse"]

    if verbose:
        print(
            f"  O_proj+resid:  {resid_result['active_allclose']:.1f}% active "
            f"({resid_result['full_allclose']:.1f}% full; residual reconstructed)"
        )
        print(f"  norm+FFN+norm: {final_result['active_allclose']:.1f}% active allclose (atol=0.2)")
        print(f"                 {final_result['full_allclose']:.1f}% full allclose")
        print(
            f"                 active MSE={final_result['active_mse']:.4e}, "
            f"full MSE={final_result['full_mse']:.4e}"
        )
        if final_result["active_allclose"] >= 99.0:
            print("  ==> PASS: emulator's FFN chain matches golden from its own intermediate")
        else:
            print("  ==> DIVERGENCE in FFN chain")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vram", nargs="?", default="transactional_emulator/vram_dump.bin")
    parser.add_argument("build", nargs="?", default="/tmp/smolvlm2_1layer_f32regs")
    parser.add_argument("layer_idx", nargs="?", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=576)
    parser.add_argument("--inter", type=int, default=1536)
    parser.add_argument("--num-heads", type=int, default=9)
    parser.add_argument("--num-kv-heads", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--mlen", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    print("=== VRAM Stage Comparison ===")
    results = compare_stages(
        vram_path=args.vram,
        build_dir=args.build,
        hidden=args.hidden,
        inter=args.inter,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        seq_len=args.seq_len,
        mlen=args.mlen,
        head_dim=args.head_dim,
        eps=args.eps,
        layer_idx=args.layer_idx,
    )
    report_path = Path(args.build) / "vram_stage_compare.json"
    report_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"\nReport: {report_path}")
    passed = bool(results.get("passed", results.get("norm+FFN+norm", 0) >= 99.0))
    print(f"\nOverall: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
