"""Measure weight quantization and BF16 intermediate error independently.

Four precision modes use the same checkpoint, seed, input and five layers.
The prior sole-gap claim assumed BF16 was numerically irrelevant. Current
scheduled-reference measurements show it contributes measurable error; this
ablation reports both components and checks the actual quantization effect.
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
            model,
            seq_len=64,
            num_layers=num_layers,
            golden_precision=mode,
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
def test_precision_ablation_measures_both_error_sources():
    """Verify the float32 baseline and measure each isolated precision cost."""
    import math
    results = _run_ablation(DEFAULT_LAYERS)
    for mode, metrics in results.items():
        assert 0 <= metrics["allclose"] <= 100
        assert math.isfinite(metrics["mse"]) and metrics["mse"] >= 0
    # The same scheduled float32 calculation must agree with the independent
    # float32 reference. No BF16 mode is assigned that float32 tolerance.
    assert results["fp32"]["allclose"] > 99.0
    assert results["fp32"]["mse"] < results["no_weight_quant"]["mse"]
    assert results["fp32"]["mse"] < results["no_bf16"]["mse"]
    # On this frozen five-layer workload, removing MX weight quantization
    # reduces MSE in the BF16-intermediate comparison. Report the BF16 cost
    # separately instead of claiming it is zero or the whole gap is MXFP8.
    assert results["no_weight_quant"]["mse"] < results["hardware"]["mse"]
    print(f"\nPrecision ablation ({DEFAULT_LAYERS} layers, seed=42, {MODEL_ID})")
    for mode in MODES:
        print(f"  {mode:<20} allclose={results[mode]['allclose']:.2f}% MSE={results[mode]['mse']:.6e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    args = parser.parse_args()

    results = _run_ablation(args.layers)

    print(f"\n{'=' * 60}")
    print(f"  QUANTIZATION ABLATION ({args.layers} layers)")
    print(f"{'=' * 60}")
    print(f"  {'Mode':<20} {'allclose%':>12} {'MSE':>15}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 15}")
    for mode in MODES:
        r = results[mode]
        print(f"  {mode:<20} {r['allclose']:>11.2f}% {r['mse']:>15.6e}")
    print("\nBoth weight quantization and BF16 intermediates contribute measurable error.")
