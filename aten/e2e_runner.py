"""ATen-backed end-to-end runner.

This wraps the verified ATen compilation path:

    HuggingFace model -> PlenaCompiler + ops.* -> ISA -> emulator -> golden check

The symbolic generator path is separate and remains under ``generator.runner
codegen``.

Usage:
    python -m compiler.aten.e2e_runner AICrossSim/clm-60m --seq-len 32
"""

import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root bootstrap — mirror the same sys.path setup used by the existing
# test infrastructure so imports resolve regardless of cwd.
# ---------------------------------------------------------------------------
_COMPILER_ROOT = Path(__file__).resolve().parents[1]  # PLENA_Compiler/
_REPO_ROOT = _COMPILER_ROOT.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "tools"), str(_COMPILER_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_aten_e2e(
    model_id: str,
    seq_len: int = 64,
    num_layers: int = 1,
    build_dir: str | None = None,
    layer_idx: int = 0,
    hidden_size: int = 64,
    inter_dim: int = 128,
    trust_remote_code: bool = False,
    partial_load: bool = False,
) -> dict:
    """Run a HF model through the ATen compilation path end-to-end.

    Steps:
      1. Load model config + layer weights from HuggingFace
      2. Build ISA via PlenaCompiler + ops.* (numerically verified path)
      3. Set up sim environment (ASM + HBM weights + FPRAM constants)
      4. Run Rust emulator
      5. Compare VRAM output against golden PyTorch reference

    Returns dict with:
        passed:             bool
        allclose_match_rate: float (percentage)
        max_error:          float
        mae:                float
        mse:                float
        elapsed_s:          float (wall-clock seconds)
        model_id:           str
        layer_idx:          int
        num_layers:         int
        seq_len:            int
        hidden_size:        int
        inter_dim:          int
        build_dir:          str
    """
    from transactional_emulator.testbench.emulator_runner import compare_emulator_output
    from transactional_emulator.testbench.model_layer_test_builder import (
        build_and_run_decoder_test,
        build_and_run_multi_layer_test,
        get_model_dims,
        slice_dims_for_sim,
    )

    t0 = time.time()

    # Resolve build directory
    if build_dir is None:
        safe_name = model_id.replace("/", "_")
        build_dir = str(
            Path("/tmp") / f"aten_e2e_{safe_name}_sl{seq_len}_l{layer_idx}"
        )
    build_path = Path(build_dir)

    # ------------------------------------------------------------------
    # [1/5] Probe model config
    # ------------------------------------------------------------------
    print(f"[1/5] Probing model config: {model_id}")
    try:
        full_dims = get_model_dims(model_id)
    except (OSError, ConnectionError) as exc:
        print(f"[SKIP] HuggingFace model '{model_id}' unavailable: {exc}")
        return {
            "passed": False,
            "error": str(exc),
            "model_id": model_id,
        }
    sim_dims = slice_dims_for_sim(full_dims, hidden_slice=hidden_size, inter_slice=inter_dim)
    print(f"       Full dims: hidden={full_dims.hidden_size}, inter={full_dims.inter_dim}, "
          f"heads={full_dims.num_heads}, kv_heads={full_dims.num_kv_heads}, head_dim={full_dims.head_dim}")
    print(f"       Sim  dims: hidden={sim_dims.hidden_size}, inter={sim_dims.inter_dim}")

    # ------------------------------------------------------------------
    # [2/5] Build ISA + golden reference + sim env via build_and_run_decoder_test
    #
    # We call the proven function directly — it handles:
    #   - Weight loading + slicing
    #   - PlenaCompiler ISA generation
    #   - create_sim_env + create_mem_for_sim
    #   - Golden reference computation
    #   - Emulator execution + comparison
    #
    # For multi-layer: iterate layers (each is independent at sim scale).
    # ------------------------------------------------------------------
    results_per_layer = []

    if num_layers == 1:
        # Single layer: use proven single-layer path (with RoPE)
        current_layer = layer_idx
        asm_name = f"aten_{model_id.split('/')[-1]}_l{current_layer}"
        layer_build = build_path / f"layer_{current_layer}"

        print(f"\n[2/5] Building ISA for layer {current_layer} via PlenaCompiler + ops.*")
        print(f"[3/5] Setting up sim environment: {layer_build}")
        print("[4/5] Running Rust transactional emulator")

        extra_kwargs = {}
        if trust_remote_code:
            extra_kwargs["trust_remote_code"] = True
        if partial_load:
            extra_kwargs["partial_load"] = True

        try:
            build_and_run_decoder_test(
                model_id=model_id,
                asm_name=asm_name,
                build_dir=layer_build,
                layer_idx=current_layer,
                seq_len=seq_len,
                hidden_size=hidden_size,
                inter_dim=inter_dim,
                **extra_kwargs,
            )
            comp_results, _comp_params = compare_emulator_output(layer_build)
            results_per_layer.append({
                "layer": current_layer,
                "passed": True,
                "allclose_match_rate": comp_results["allclose_match_rate"],
                "max_error": comp_results["max_error"],
                "mae": comp_results["mae"],
                "mse": comp_results["mse"],
            })
        except SystemExit as e:
            if e.code == 0:
                return {
                    "passed": False,
                    "error": "HuggingFace model unavailable (skipped)",
                    "model_id": model_id,
                }
            try:
                comp_results, _comp_params = compare_emulator_output(layer_build)
                results_per_layer.append({
                    "layer": current_layer,
                    "passed": False,
                    "allclose_match_rate": comp_results["allclose_match_rate"],
                    "max_error": comp_results["max_error"],
                    "mae": comp_results["mae"],
                    "mse": comp_results["mse"],
                })
            except Exception:
                results_per_layer.append({
                    "layer": current_layer,
                    "passed": False,
                    "error": f"Emulator comparison failed after exit code {e.code}",
                })
    else:
        # Multi-layer: chain N layers with residual connections (no RoPE)
        asm_name = f"aten_{model_id.split('/')[-1]}_chain{num_layers}"
        chain_build = build_path / f"chain_{num_layers}layers"

        print(f"\n[2/5] Building chained {num_layers}-layer ISA via PlenaCompiler + ops.*")
        print(f"[3/5] Setting up sim environment: {chain_build}")
        print("[4/5] Running Rust transactional emulator")

        extra_kwargs = {}
        if trust_remote_code:
            extra_kwargs["trust_remote_code"] = True
        if partial_load:
            extra_kwargs["partial_load"] = True

        try:
            build_and_run_multi_layer_test(
                model_id=model_id,
                asm_name=asm_name,
                build_dir=chain_build,
                num_layers=num_layers,
                layer_idx_start=layer_idx,
                seq_len=seq_len,
                hidden_size=hidden_size,
                inter_dim=inter_dim,
                **extra_kwargs,
            )
            comp_results, _comp_params = compare_emulator_output(chain_build)
            results_per_layer.append({
                "layer": f"chain_{num_layers}",
                "passed": True,
                "allclose_match_rate": comp_results["allclose_match_rate"],
                "max_error": comp_results["max_error"],
                "mae": comp_results["mae"],
                "mse": comp_results["mse"],
            })
        except SystemExit as e:
            if e.code == 0:
                return {
                    "passed": False,
                    "error": "HuggingFace model unavailable (skipped)",
                    "model_id": model_id,
                }
            try:
                comp_results, _comp_params = compare_emulator_output(chain_build)
                results_per_layer.append({
                    "layer": f"chain_{num_layers}",
                    "passed": False,
                    "allclose_match_rate": comp_results["allclose_match_rate"],
                    "max_error": comp_results["max_error"],
                    "mae": comp_results["mae"],
                    "mse": comp_results["mse"],
                })
            except Exception:
                results_per_layer.append({
                    "layer": f"chain_{num_layers}",
                    "passed": False,
                    "error": f"Emulator comparison failed after exit code {e.code}",
                })

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # [5/5] Aggregate results
    # ------------------------------------------------------------------
    print(f"\n[5/5] Results summary ({elapsed:.1f}s elapsed)")
    all_passed = all(r.get("passed", False) for r in results_per_layer)

    # Use first layer's metrics for the top-level result
    first = results_per_layer[0] if results_per_layer else {}

    summary = {
        "passed": all_passed,
        "allclose_match_rate": first.get("allclose_match_rate", 0.0),
        "max_error": first.get("max_error", float("inf")),
        "mae": first.get("mae", float("inf")),
        "mse": first.get("mse", float("inf")),
        "elapsed_s": elapsed,
        "model_id": model_id,
        "layer_idx": layer_idx,
        "num_layers": num_layers,
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "inter_dim": inter_dim,
        "build_dir": str(build_path),
        "layers": results_per_layer,
    }

    for r in results_per_layer:
        status = "PASS" if r.get("passed") else "FAIL"
        match = r.get("allclose_match_rate", "N/A")
        if isinstance(match, float):
            match = f"{match:.2f}%"
        print(f"  Layer {r.get('layer', '?')}: [{status}] allclose={match}")

    if all_passed:
        print(f"\n[ATen e2e PASSED] {model_id} — {num_layers} layer(s), "
              f"allclose={first.get('allclose_match_rate', 0):.2f}%")
    else:
        print(f"\n[ATen e2e FAILED] {model_id} — see per-layer results above")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run HF model through ATen compilation path (PlenaCompiler + ops.*)",
        prog="python -m compiler.aten.e2e_runner",
    )
    parser.add_argument("model_id", help="HuggingFace model ID (e.g. AICrossSim/clm-60m)")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Sequence length (default: 64)")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Number of decoder layers to test (default: 1)")
    parser.add_argument("--layer-idx", type=int, default=0,
                        help="Starting layer index (default: 0)")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="Hidden dimension clipped to sim limits (default: 64)")
    parser.add_argument("--inter-dim", type=int, default=128,
                        help="FFN intermediate dimension clipped to sim limits (default: 128)")
    parser.add_argument("--build-dir", type=str, default=None,
                        help="Build directory for sim artifacts (default: /tmp/aten_e2e_...)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code for HF model loading")
    parser.add_argument("--partial-load", action="store_true",
                        help="Load only needed weight shards (for large models)")

    args = parser.parse_args()

    result = run_aten_e2e(
        model_id=args.model_id,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        build_dir=args.build_dir,
        layer_idx=args.layer_idx,
        hidden_size=args.hidden_size,
        inter_dim=args.inter_dim,
        trust_remote_code=args.trust_remote_code,
        partial_load=args.partial_load,
    )

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
