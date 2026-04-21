#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from generator.parser import LLMModelParser, hardware_parser
from generator.passes.code_gen import code_gen_pass
from generator.passes.utilization_report import analyse_overall_utilization
from generator.scheduler import gen_scheduler


def run():
    if len(sys.argv) < 4:
        print("Usage: python -m generator.runner <mode> <model_name_or_path> <output_file.asm> [--seq-len N]")
        print("Example: python -m generator.runner codegen AICrossSim/clm-60m output.asm --seq-len 512")
        return
    mode = sys.argv[1]
    model_path = sys.argv[2]
    output_file = sys.argv[3]

    # Parse optional arguments after the positional ones
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("--seq-len", type=int, default=512)
    extra_args, _ = arg_parser.parse_known_args(sys.argv[4:])
    seq_len = extra_args.seq_len
    hardware_config_path = Path(__file__).resolve().parents[1] / "doc" / "configuration.svh"
    precision_config_path = Path(__file__).resolve().parents[1] / "doc" / "precision.svh"
    mem_layout_lib_path = Path(__file__).resolve().parents[0] / "scheduler" / "mem_layout_lib.json"
    reg_assignment_lib_path = Path(__file__).resolve().parents[0] / "scheduler" / "reg_assignment_lib.json"
    # Validate that output file ends with .asm
    if not output_file.endswith(".asm"):
        print("Error: Output file must end with .asm extension")
        print("Example: python runner.py AICrossSim/clm-60m output.asm")
        return

    print(f"Loading model: {model_path}")
    parser = LLMModelParser(model_path)

    parser.load_model()
    parser.print_summary()

    # Create symbolic graph
    symbolic_graph = parser.create_symbolic_graph(seq_len=seq_len)

    dimensions = parser.extract_critical_dimensions()

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

    # Run code generation pass
    if mode == "utilization":
        m_dim = 64
        k_dim = 64
        n_dim = 64
        print("\nRunning utilization analysis...")
        utilization_report = analyse_overall_utilization(symbolic_graph, model_info, m_dim, k_dim, n_dim)
        print(f"Utilization Report:\n{utilization_report}")
        return

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


if __name__ == "__main__":
    sys.exit(run())
