#!/usr/bin/env python3
"""
PLENA generator runner -- symbolic codegen and ATen e2e convenience entry.

Modes:
  codegen      -- Pipeline 2 (Generator): HF config -> symbolic graph ->
                  code_gen_pass -> ASM. Valid ISA for analysis/profiling;
                  numerically incomplete (HBM address registers not set).
  aten         -- Pipeline 1 (ATen): HF model -> PlenaCompiler -> verified
                  ISA. Numerically verified, 98-100% allclose.
  utilization  -- PE utilization analysis report (no ISA emitted).

Examples:
    python -m generator.runner codegen AICrossSim/clm-60m output.asm --seq-len 512
    python -m compiler.aten.e2e_runner AICrossSim/clm-60m --seq-len 32 --num-layers 1

See docs/COMPILATION_PIPELINES.md for the full architecture overview.
"""

import argparse
import sys
from pathlib import Path

from generator.parser import LLMModelParser, hardware_parser
from generator.passes.code_gen import code_gen_pass
from generator.passes.utilization_report import analyse_overall_utilization
from generator.scheduler import gen_scheduler


def _run_aten(args) -> int:
    """ATen-backed end-to-end: PlenaCompiler + ops.* -> emulator -> numerical check."""
    from compiler.aten.e2e_runner import run_aten_e2e

    result = run_aten_e2e(
        model_id=args.model_path,
        seq_len=args.seq_len,
        num_layers=args.num_layers if args.num_layers is not None else 1,
        build_dir=args.build_dir,
        trust_remote_code=args.trust_remote_code,
        partial_load=args.partial_load,
    )
    return 0 if result["passed"] else 1


def _run_codegen(args) -> int:
    """Original generator codegen path — symbolic graph → ISA → .asm file."""
    model_path = args.model_path
    output_file = args.output_file
    seq_len = args.seq_len
    num_layers_override = args.num_layers

    if output_file is None:
        print("Error: codegen mode requires an output .asm file")
        print("Usage: python -m generator.runner codegen <model> <output.asm> [--seq-len N]")
        return 1

    hardware_config_path = Path(__file__).resolve().parents[1] / "doc" / "configuration.svh"
    precision_config_path = Path(__file__).resolve().parents[1] / "doc" / "precision.svh"
    mem_layout_lib_path = Path(__file__).resolve().parents[0] / "scheduler" / "mem_layout_lib.json"
    reg_assignment_lib_path = Path(__file__).resolve().parents[0] / "scheduler" / "reg_assignment_lib.json"

    if not output_file.endswith(".asm"):
        print("Error: Output file must end with .asm extension")
        print("Example: python -m generator.runner codegen AICrossSim/clm-60m output.asm")
        return 1

    print(f"Loading model: {model_path}")
    parser = LLMModelParser(model_path)

    parser.load_model()

    # Apply num_hidden_layers override before the symbolic graph is built.
    if num_layers_override is not None:
        text_cfg = parser._resolve_text_config()
        original = getattr(text_cfg, "num_hidden_layers", None)
        text_cfg.num_hidden_layers = num_layers_override
        print(f"[override] num_hidden_layers: {original} -> {num_layers_override} (via --num-layers)")

    parser.print_summary()

    # Create symbolic graph
    symbolic_graph = parser.create_symbolic_graph(seq_len=seq_len)

    dimensions = parser.extract_critical_dimensions()

    # For multimodal models, prepend the vision encoder graph so the emitted ASM
    # actually exercises the SigLIP / ViT layers + connector before the text decoder.
    vision_graph = parser.create_vision_symbolic_graph(batch_size=1)
    if vision_graph is not None:
        vision_nodes = vision_graph["nodes"]
        vision_order = vision_graph["execution_order"]
        # Renumber text nodes so execution_order remains globally monotonic.
        offset = len(vision_nodes)
        for node in symbolic_graph["nodes"]:
            node["execution_order"] = node["execution_order"] + offset
        symbolic_graph["nodes"] = vision_nodes + symbolic_graph["nodes"]
        symbolic_graph["execution_order"] = vision_order + symbolic_graph["execution_order"]
        symbolic_graph["total_nodes"] = len(symbolic_graph["nodes"])
        print(f"\n[vision] Prepended {len(vision_nodes)} vision encoder nodes to symbolic graph")

    # Print detailed symbolic graph
    parser.print_symbolic_graph_details()

    # Prepare model info for code generation
    model_info = {
        "model_name": model_path,
        "architecture": getattr(parser.config, "architectures", ["Unknown"])[0] if parser.config else "Unknown",
        "batch_size": 4,
        "context_length": dimensions.get("max_position_embeddings", "Unknown"),
        "vocab_size": dimensions.get("vocab_size", "Unknown"),
        "hidden_size": dimensions.get("hidden_size", "Unknown"),
        "intermediate_size": dimensions.get("ffn", {}).get("intermediate_size", 4096),
        "num_key_value_heads": dimensions.get("attention", {}).get("num_key_value_heads", "Unknown"),
        "num_attention_heads": dimensions.get("attention", {}).get("num_attention_heads", "Unknown"),
        "num_layers": dimensions.get("num_hidden_layers", "Unknown"),
        # FIXME(code-smell): defaulting head_dim to 0 silently; downstream codegen
        # divides by this, so a missing value will explode rather than fail here.
        # Should raise / fall back to hidden_size // num_heads instead.
        "head_dim": dimensions.get("attention", {}).get("head_dim", 0),
        "eps": dimensions.get("rms_norm", {}).get("eps", 1e-6),
        "seq_len": seq_len,
    }

    # Expose vision encoder dims when present (so code_gen can parameterise
    # conv2d, bidirectional attention and the connector projection).
    if "vision" in dimensions:
        model_info["vision"] = dimensions["vision"]
        model_info["has_vision"] = True
    else:
        model_info["has_vision"] = False

    # Run code generation pass
    if args.mode == "utilization":
        m_dim = 64
        k_dim = 64
        n_dim = 64
        print("\nRunning utilization analysis...")
        utilization_report = analyse_overall_utilization(symbolic_graph, model_info, m_dim, k_dim, n_dim)
        print(f"Utilization Report:\n{utilization_report}")
        return 0

    hardware_config = hardware_parser(hardware_config_path, precision_config_path)
    scheduler = gen_scheduler(hardware_config, model_info, mem_layout_lib_path, reg_assignment_lib_path)
    print("\nRunning code generation pass...")
    generated_asm = code_gen_pass(symbolic_graph, model_info, hardware_config, scheduler)

    # Save generated code
    with open(output_file, "w") as f:
        f.write(generated_asm)

    print(f"Generated assembly code saved to: {output_file}")
    print(f"seq_len used: {seq_len}")

    # Print a preview of the generated code
    print("\nGenerated code preview (first 20 lines):")
    print("=" * 50)
    lines = generated_asm.split("\n")
    for i, line in enumerate(lines[:20]):
        print(f"{i + 1:3d}: {line}")
    if len(lines) > 20:
        print(f"... and {len(lines) - 20} more lines")
    print("=" * 50)
    return 0


def run():
    # Use proper argparse for all modes
    parser = argparse.ArgumentParser(
        description="Generator runner — codegen and ATen compilation modes",
        prog="python -m generator.runner",
    )
    parser.add_argument("mode", choices=["codegen", "aten", "utilization"],
                        help="Execution mode: codegen (ASM generation), "
                             "aten (PlenaCompiler e2e with emulator), "
                             "utilization (PE utilization report)")
    parser.add_argument("model_path", help="HuggingFace model ID (e.g. AICrossSim/clm-60m)")
    parser.add_argument("output_file", nargs="?", default=None,
                        help="Output .asm file (required for codegen/utilization, ignored for aten)")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length (default: 512 for codegen, 64 for aten)")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override num_hidden_layers in model config")
    parser.add_argument("--build-dir", type=str, default=None,
                        help="Build directory for aten mode sim artifacts")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code for HF model loading (aten mode)")
    parser.add_argument("--partial-load", action="store_true",
                        help="Load only needed weight shards for large models (aten mode)")

    args = parser.parse_args()

    if args.mode == "aten":
        # Default seq_len for aten mode is 64 (sim-optimal), not 512
        if "--seq-len" not in sys.argv:
            args.seq_len = 64
        return _run_aten(args)
    else:
        return _run_codegen(args)


if __name__ == "__main__":
    sys.exit(run())
