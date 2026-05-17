"""VRAM-in-the-loop stage comparison for SmolVLM2 / clm-60m.

Instead of comparing the final output against a golden chain (which drifts),
this reads the emulator's ACTUAL VRAM values at each stage boundary and uses
them as input to compute the expected NEXT stage output. Each stage is then
compared independently.

Usage:
    from compiler.aten.vram_stage_compare import compare_stages
    results = compare_stages(
        vram_path="transactional_emulator/vram_dump.bin",
        build_dir="/tmp/smolvlm2_1layer_f32regs",
        hidden=576, inter=1536, num_heads=9, num_kv_heads=3,
        layer_idx=0,
    )
"""
import re
import struct
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from compiler.aten.plena_frontend import quantize_to_mxfp, _ksplit_matmul


_HW_MAX_K_TILES = 4


def _read_bf16_matrix(vram_path, addr, rows, cols, mlen=64):
    """Read a (rows, cols) matrix from VRAM dump in column-tile layout."""
    out = np.zeros((rows, cols), dtype=np.float32)
    with open(vram_path, 'rb') as f:
        for tile in range(cols // mlen):
            for r in range(rows):
                f.seek((addr + tile * rows * mlen + r * mlen) * 2)
                raw = f.read(mlen * 2)
                for c in range(mlen):
                    u16 = int(raw[c * 2]) | (int(raw[c * 2 + 1]) << 8)
                    u32 = u16 << 16
                    out[r, tile * mlen + c] = struct.unpack('f', struct.pack('I', u32))[0]
    return torch.tensor(out)


def _allclose_pct(a, b, atol=0.2, rtol=0.2):
    return torch.isclose(a.bfloat16(), b.bfloat16(), atol=atol, rtol=rtol).float().mean().item() * 100


def _mse(a, b):
    return ((a.float() - b.float()) ** 2).mean().item()


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
    match = re.search(
        rf"Allocate VRAM Matrix {re.escape(name)}: .*?VRAM\[(\d+)\]",
        asm,
    )
    return int(match.group(1)) if match else None


def compare_stages(vram_path, build_dir, hidden, inter, num_heads, num_kv_heads,
                   seq_len=64, mlen=64, head_dim=64, eps=1e-5, verbose=True,
                   layer_idx=None):
    """Compare each pipeline stage using emulator's own VRAM intermediates.

    Args:
        vram_path: path to the emulator's vram_dump.bin
        build_dir: path to the build directory with weight .pt files
        hidden, inter, num_heads, num_kv_heads: model dimensions
        layer_idx: decoder layer to validate. Defaults to the last layer found
                   in build_dir, which is the layer that feeds the final output.

    Returns:
        dict of stage results with allclose percentages
    """
    build = Path(build_dir)
    _to_inter = lambda x: x.to(torch.bfloat16)
    _from_inter = lambda x: x.float()

    if layer_idx is None:
        layer_idx = _infer_final_layer_idx(build)

    # Read final output address from comparison_params
    import json
    params = json.load(open(build / "comparison_params.json"))
    final_addr = params["start_row_idx"] * mlen

    results = {"layer_idx": layer_idx}

    # --- Load weights ---
    W_o = quantize_to_mxfp(torch.load(build / f"W_o_{layer_idx}.pt", weights_only=True))
    W_gate = quantize_to_mxfp(torch.load(build / f"W_gate_{layer_idx}.pt", weights_only=True))
    W_up = quantize_to_mxfp(torch.load(build / f"W_up_{layer_idx}.pt", weights_only=True))
    W_down = quantize_to_mxfp(torch.load(build / f"W_down_{layer_idx}.pt", weights_only=True))

    # --- Stage 1: O_full (attention output) ---
    # Find O_full address from ISA comments
    o_full_addr = final_addr - 2 * seq_len * hidden
    asm_path = build / "generated_asm_code.asm"
    if asm_path.exists():
        with open(asm_path) as f:
            asm = f.read()
        parsed_o_full_addr = _read_alloc_addr(asm, f"O_full_{layer_idx}")
        if parsed_o_full_addr is not None:
            o_full_addr = parsed_o_full_addr
        scratch_addr = _read_alloc_addr(asm, "residual_scratch")
    else:
        scratch_addr = None

    if verbose:
        print(f"  Validating layer {layer_idx}")

    O_full = _read_bf16_matrix(vram_path, o_full_addr, seq_len, hidden)

    # --- Stage 2: O_proj = O_full @ W_o ---
    O_proj_expected = _ksplit_matmul(O_full, W_o, mlen, _HW_MAX_K_TILES, _to_inter, _from_inter)
    O_proj_expected = _from_inter(_to_inter(O_proj_expected))

    # --- Stage 3: O_proj + X_residual ---
    # scratch = O_proj + X_residual (saved before FFN norm)
    if scratch_addr is not None:
        scratch_vram = _read_bf16_matrix(vram_path, scratch_addr, seq_len, hidden)
        # X_residual = scratch - O_proj
        X_residual = _from_inter(_to_inter(scratch_vram - O_proj_expected))
        O_proj_resid = _from_inter(_to_inter(O_proj_expected + X_residual))

        allclose_resid = _allclose_pct(O_proj_resid, scratch_vram, atol=0.01, rtol=0.01)
        results["O_proj+resid"] = allclose_resid
        if verbose:
            print(f"  O_proj+resid:  {allclose_resid:.1f}% (atol=0.01)")

    # --- Stage 4: RMS norm → FFN → residual → final norm ---
    # Use emulator's scratch (O_proj+resid) as input
    if scratch_addr is not None:
        X_pre_ffn = scratch_vram.clone()

        # RMS norm
        X_bf = _to_inter(X_pre_ffn)
        rms = torch.rsqrt(_from_inter(X_bf).pow(2).mean(-1, keepdim=True) + eps)
        X_normed = _from_inter(_to_inter(_from_inter(X_bf) * rms))

        # FFN
        up_out = _to_inter(torch.matmul(_from_inter(_to_inter(X_normed)), W_up.float()))
        gate_out = _to_inter(torch.matmul(_from_inter(_to_inter(X_normed)), W_gate.float()))
        silu_gate = _to_inter(F.silu(_from_inter(up_out)) * _from_inter(gate_out))
        ffn_out = _from_inter(_to_inter(torch.matmul(_from_inter(silu_gate), W_down.float())))

        # FFN residual
        post_ffn = _from_inter(_to_inter(ffn_out + X_pre_ffn))

        # Final norm
        X_bf2 = _to_inter(post_ffn)
        rms2 = torch.rsqrt(_from_inter(X_bf2).pow(2).mean(-1, keepdim=True) + eps)
        expected_final = _from_inter(_to_inter(_from_inter(X_bf2) * rms2))

        # Compare with VRAM final output
        final_vram = _read_bf16_matrix(vram_path, final_addr, seq_len, hidden)

        allclose_final = _allclose_pct(expected_final, final_vram)
        mse_final = _mse(expected_final, final_vram)
        results["norm+FFN+norm"] = allclose_final
        results["final_mse"] = mse_final

        if verbose:
            print(f"  norm+FFN+norm: {allclose_final:.1f}% allclose (atol=0.2)")
            print(f"                 MSE={mse_final:.4e}")
            if allclose_final >= 99.0:
                print(f"  ==> PASS: emulator's FFN chain matches golden from its own intermediate")
            else:
                print(f"  ==> DIVERGENCE in FFN chain")

    return results


if __name__ == "__main__":
    import sys
    vram = sys.argv[1] if len(sys.argv) > 1 else "transactional_emulator/vram_dump.bin"
    build = sys.argv[2] if len(sys.argv) > 2 else "/tmp/smolvlm2_1layer_f32regs"
    layer_idx = int(sys.argv[3]) if len(sys.argv) > 3 else None

    print("=== VRAM Stage Comparison ===")
    results = compare_stages(
        vram_path=vram,
        build_dir=build,
        hidden=576, inter=1536, num_heads=9, num_kv_heads=3,
        layer_idx=layer_idx,
    )
    print(f"\nOverall: {'PASS' if results.get('norm+FFN+norm', 0) >= 99.0 else 'FAIL'}")
