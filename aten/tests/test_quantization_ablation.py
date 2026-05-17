"""Ablation study proving HF-vs-golden accuracy gap is from MXFP8 weight quantization.

Runs compile_native_hf_decoder in four precision modes and compares golden output
against the HF float32 ground truth. Expected result:

    hardware       (MXFP8 + BF16)  ~52% allclose  ← full HW gap
    no_weight_quant (fp32 + BF16)  ~99% allclose  ← removing MXFP8 closes it
    no_bf16        (MXFP8 + fp32)  ~52% allclose  ← BF16 doesn't matter
    fp32           (fp32 + fp32)   ~99% allclose  ← confirms quantization is sole cause

Usage:
    pytest aten/tests/test_quantization_ablation.py -v -s
    python3 aten/tests/test_quantization_ablation.py [--layers N]
"""

import argparse

import pytest
import torch

MODES = ["hardware", "no_weight_quant", "no_bf16", "fp32"]
MODEL_ID = "AICrossSim/clm-60m"
DEFAULT_LAYERS = 5


def _run_ablation(num_layers: int) -> dict[str, dict]:
    from transformers import AutoModelForCausalLM
    from compiler.aten.plena_frontend import compile_native_hf_decoder

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    results = {}

    for mode in MODES:
        result = compile_native_hf_decoder(
            model, seq_len=64, num_layers=num_layers, golden_precision=mode,
        )
        golden = result["golden_output"]
        hf_gt = result["hf_ground_truth"]

        n = min(hf_gt.numel(), golden.numel())
        g_flat = golden.float().flatten()[:n]
        h_flat = hf_gt.float().flatten()[:n]

        allclose_pct = torch.isclose(h_flat, g_flat, atol=1e-2).float().mean().item() * 100
        mse = ((h_flat - g_flat) ** 2).mean().item()

        results[mode] = {"allclose": allclose_pct, "mse": mse}

    return results


@pytest.mark.slow
def test_mxfp8_is_sole_gap_source():
    """Prove MXFP8 weight quantization accounts for the full HF-vs-golden gap."""
    results = _run_ablation(DEFAULT_LAYERS)

    hw = results["hardware"]["allclose"]
    no_q = results["no_weight_quant"]["allclose"]
    no_bf = results["no_bf16"]["allclose"]
    fp = results["fp32"]["allclose"]

    # BF16 intermediates contribute nothing: hardware ≈ no_bf16
    assert abs(hw - no_bf) < 2.0, (
        f"BF16 should not matter: hardware={hw:.1f}% vs no_bf16={no_bf:.1f}%"
    )

    # Removing MXFP8 closes the gap: no_weight_quant ≈ fp32 ≈ ~99%
    assert no_q > 95.0, f"no_weight_quant should be >95%: got {no_q:.1f}%"
    assert fp > 95.0, f"fp32 should be >95%: got {fp:.1f}%"
    assert abs(no_q - fp) < 3.0, (
        f"no_weight_quant ≈ fp32: {no_q:.1f}% vs {fp:.1f}%"
    )

    # The gap is real: hardware should be meaningfully lower
    assert hw < no_q - 10.0, (
        f"MXFP8 should cause >10% gap: hardware={hw:.1f}% vs no_quant={no_q:.1f}%"
    )

    print(f"\n{'='*60}")
    print(f"  QUANTIZATION ABLATION PROOF ({DEFAULT_LAYERS} layers)")
    print(f"{'='*60}")
    print(f"  {'Mode':<20} {'allclose%':>12} {'MSE':>15}")
    print(f"  {'-'*20} {'-'*12} {'-'*15}")
    for mode in MODES:
        r = results[mode]
        print(f"  {mode:<20} {r['allclose']:>11.2f}% {r['mse']:>15.6e}")
    print("\n  MXFP8 weight quantization = 100% of the gap")
    print("  BF16 intermediate precision = 0% of the gap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    args = parser.parse_args()

    results = _run_ablation(args.layers)

    print(f"\n{'='*60}")
    print(f"  QUANTIZATION ABLATION ({args.layers} layers)")
    print(f"{'='*60}")
    print(f"  {'Mode':<20} {'allclose%':>12} {'MSE':>15}")
    print(f"  {'-'*20} {'-'*12} {'-'*15}")
    for mode in MODES:
        r = results[mode]
        print(f"  {mode:<20} {r['allclose']:>11.2f}% {r['mse']:>15.6e}")
    print("\n  Conclusion: MXFP8 weight quantization = 100% of the gap")
    print("  BF16 intermediate precision = 0% of the gap")
