"""Controlled attribution of decoder error to weight and activation precision.

Each precision factor is changed independently while keeping the scheduled
execution path fixed. This separates MXFP8 weight error from BF16 rounding
instead of attributing both effects through an end-to-end allclose percentage.

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


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Return scale-aware error metrics for two shape-identical outputs."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"output shape mismatch: actual={tuple(actual.shape)}, "
            f"expected={tuple(expected.shape)}"
        )

    actual_flat = actual.float().flatten()
    expected_flat = expected.float().flatten()
    error = actual_flat - expected_flat
    mse = error.square().mean()
    reference_rms = expected_flat.square().mean().sqrt().clamp_min(
        torch.finfo(torch.float32).eps
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual_flat.unsqueeze(0),
        expected_flat.unsqueeze(0),
    )

    return {
        "allclose": (
            torch.isclose(actual_flat, expected_flat, atol=1e-2)
            .float()
            .mean()
            .item()
            * 100
        ),
        "mse": mse.item(),
        "nrmse": (mse.sqrt() / reference_rms).item(),
        "cosine": cosine.item(),
    }


def _run_ablation(num_layers: int) -> dict[str, dict[str, dict[str, float]]]:
    from transformers import AutoModelForCausalLM
    from compiler.aten.plena_frontend import compile_native_hf_decoder

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    outputs: dict[str, torch.Tensor] = {}
    baseline: dict[str, dict[str, float]] = {}

    for mode in MODES:
        result = compile_native_hf_decoder(
            model,
            seq_len=64,
            num_layers=num_layers,
            golden_precision=mode,
        )
        golden = result["golden_output"]
        hf_gt = result["hf_ground_truth"]
        outputs[mode] = golden
        baseline[mode] = _comparison(golden, hf_gt)

    pairs = {
        # Change only intermediate precision at each fixed weight precision.
        "bf16_quantized_weights": _comparison(
            outputs["hardware"], outputs["no_bf16"]
        ),
        "bf16_unquantized_weights": _comparison(
            outputs["no_weight_quant"], outputs["fp32"]
        ),
        # Change only weight precision at each fixed intermediate precision.
        "mxfp8_bf16_intermediates": _comparison(
            outputs["hardware"], outputs["no_weight_quant"]
        ),
        "mxfp8_fp32_intermediates": _comparison(
            outputs["no_bf16"], outputs["fp32"]
        ),
    }
    return {"baseline": baseline, "pairs": pairs}


def _print_results(
    results: dict[str, dict[str, dict[str, float]]],
    num_layers: int,
) -> None:
    print(f"\n{'=' * 78}")
    print(f"  QUANTIZATION ABLATION ({num_layers} layers)")
    print(f"{'=' * 78}")
    print(
        f"  {'Comparison':<31} {'allclose%':>10} {'NRMSE':>12} "
        f"{'cosine':>12} {'MSE':>13}"
    )
    print(f"  {'-' * 31} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 13}")
    for name, metrics in results["baseline"].items():
        print(
            f"  {name + ' vs fp32 reference':<31} "
            f"{metrics['allclose']:>9.2f}% {metrics['nrmse']:>12.6f} "
            f"{metrics['cosine']:>12.8f} {metrics['mse']:>13.6e}"
        )
    for name, metrics in results["pairs"].items():
        print(
            f"  {name:<31} {metrics['allclose']:>9.2f}% "
            f"{metrics['nrmse']:>12.6f} {metrics['cosine']:>12.8f} "
            f"{metrics['mse']:>13.6e}"
        )


@pytest.mark.slow
def test_mxfp8_weight_error_dominates_bf16_rounding():
    """Attribute the dominant error with matched, single-factor contrasts."""
    results = _run_ablation(DEFAULT_LAYERS)
    baseline = results["baseline"]
    pairs = results["pairs"]

    # The scheduled implementation itself remains an accurate FP32 control.
    fp32_allclose = baseline["fp32"]["allclose"]
    assert fp32_allclose > 95.0, (
        f"FP32 scheduled control should be >95%: got {fp32_allclose:.1f}%"
    )

    bf16_pairs = (
        pairs["bf16_quantized_weights"],
        pairs["bf16_unquantized_weights"],
    )
    mxfp8_pairs = (
        pairs["mxfp8_bf16_intermediates"],
        pairs["mxfp8_fp32_intermediates"],
    )

    # BF16 rounding is small and independently bounded under both weight modes.
    assert max(pair["nrmse"] for pair in bf16_pairs) < 0.01
    assert min(pair["cosine"] for pair in bf16_pairs) > 0.9999

    # MXFP8 remains dominant under either intermediate-precision control.
    largest_bf16_mse = max(pair["mse"] for pair in bf16_pairs)
    smallest_mxfp8_mse = min(pair["mse"] for pair in mxfp8_pairs)
    assert smallest_mxfp8_mse > 10.0 * largest_bf16_mse, (
        "MXFP8 weight error must exceed BF16 rounding by at least 10x MSE: "
        f"MXFP8={smallest_mxfp8_mse:.6e}, BF16={largest_bf16_mse:.6e}"
    )

    _print_results(results, DEFAULT_LAYERS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    args = parser.parse_args()

    _print_results(_run_ablation(args.layers), args.layers)
