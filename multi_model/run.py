import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from multi_model import OUTPUTS_DIR
    from multi_model.model.TRANSFOMER_ENCODER import TRANSFOMER_ENCODER, MultiLayerEncoder
    from multi_model.utilization_report import DEFAULT_HW, analyse_trace_utilization, render_markdown_report
    from multi_model.vlm_codegen import vlm_codegen
    from multi_model.vlm_parser import VLMModelParser, template_qwen3_vl_inputs
else:
    from . import OUTPUTS_DIR
    from .model.TRANSFOMER_ENCODER import TRANSFOMER_ENCODER, MultiLayerEncoder
    from .utilization_report import DEFAULT_HW, analyse_trace_utilization, render_markdown_report
    from .vlm_codegen import vlm_codegen
    from .vlm_parser import VLMModelParser, template_qwen3_vl_inputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace a demo model or a Hugging Face model and generate PLENA ASM."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model name/path to load. If omitted, run the local MultiLayerEncoder demo.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image path for processor-backed vision-language models such as Qwen3-VL.",
    )
    parser.add_argument(
        "--text",
        default="Describe this image.",
        help="Prompt text used together with --image for VLM models.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Synthetic input batch size.")
    parser.add_argument("--seq-len", type=int, default=4, help="Synthetic text sequence length.")
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden size for the local MultiLayerEncoder demo model.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Layer count for the local MultiLayerEncoder demo model.",
    )
    parser.add_argument("--seed", type=int, default=2061452, help="Torch random seed.")
    parser.add_argument(
        "--output-prefix",
        default=str(OUTPUTS_DIR / "run"),
        help="Output prefix. Writes trace/model-info/ASM/report files under this prefix.",
    )
    parser.add_argument(
        "--report-format",
        choices=("md", "json", "both"),
        default="md",
        help="Utilization report output format.",
    )
    parser.add_argument("--tile-m", type=int, default=int(DEFAULT_HW["tile_m"]), help="Hardware tile M.")
    parser.add_argument("--tile-k", type=int, default=int(DEFAULT_HW["tile_k"]), help="Hardware tile K.")
    parser.add_argument("--tile-n", type=int, default=int(DEFAULT_HW["tile_n"]), help="Hardware tile N.")
    parser.add_argument(
        "--memory-capacity-bytes",
        type=int,
        default=None,
        help="Optional hardware activation-memory capacity for utilization analysis.",
    )
    parser.add_argument(
        "--include-non-leaf",
        action="store_true",
        help="Include non-leaf modules in utilization summaries.",
    )
    parser.add_argument(
        "--drop-outputs-live",
        action="store_true",
        help="Do not keep final outputs live until the end of the trace.",
    )
    return parser.parse_args()


def _build_demo_model(args: argparse.Namespace) -> tuple[nn.Module, dict[str, torch.Tensor]]:
    x = torch.rand(args.batch_size, args.seq_len, args.hidden_size)
    model = MultiLayerEncoder(num_layers=args.num_layers, hidden_size=args.hidden_size)
    return model, {"x": x}


def _build_text_inputs(parser: VLMModelParser, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    if parser.config is None:
        raise RuntimeError("Model config is required to synthesize text inputs.")

    vocab_size = max(2, int(getattr(parser.config, "vocab_size", 32000)))
    input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def _build_hf_inputs(parser: VLMModelParser, args: argparse.Namespace) -> dict[str, Any]:
    if parser.processor is not None:
        if not args.image:
            raise ValueError(
                "This model uses a processor. Pass --image for vision-language inputs."
            )
        return template_qwen3_vl_inputs(parser.processor, args.image, text=args.text)

    return _build_text_inputs(parser, args)


def main() -> int:
    args = _parse_args()
    torch.manual_seed(args.seed)

    if args.model:
        parser = VLMModelParser(args.model)
        parser.load_model()
        model = parser.model
        inputs = _build_hf_inputs(parser, args)
    else:
        model, inputs = _build_demo_model(args)
        parser = VLMModelParser()
        parser.load_model(model)

    parser.load_inputs(inputs)
    trace_tree = parser.trace()
    nodes = parser.flattened_traced_tree
    if nodes is None:
        raise RuntimeError("flattened_traced_tree was not populated after tracing")
    model_info = parser.extract_model_info(inputs=inputs)

    output_prefix = Path(args.output_prefix)
    trace_path = output_prefix.with_name(f"{output_prefix.name}_trace.json")
    model_info_path = output_prefix.with_name(f"{output_prefix.name}_model_info.json")
    asm_path = output_prefix.with_suffix(".asm")
    report_md_path = output_prefix.with_name(f"{output_prefix.name}_report.md")
    report_json_path = output_prefix.with_name(f"{output_prefix.name}_report.json")

    parser.export_flatten_call_tree(nodes, trace_path)
    parser.export_model_info(model_info, model_info_path)

    asm = vlm_codegen(nodes, model_info, mode="default")
    asm_path.parent.mkdir(parents=True, exist_ok=True)
    asm_path.write_text(asm, encoding="utf-8")

    report = analyse_trace_utilization(
        trace_tree,
        model_info=model_info,
        hw={
            "tile_m": args.tile_m,
            "tile_k": args.tile_k,
            "tile_n": args.tile_n,
            "memory_capacity_bytes": args.memory_capacity_bytes,
        },
        include_non_leaf=args.include_non_leaf,
        keep_model_outputs_live=not args.drop_outputs_live,
    )

    if args.report_format in {"md", "both"}:
        report_md_path.parent.mkdir(parents=True, exist_ok=True)
        report_md_path.write_text(render_markdown_report(report), encoding="utf-8")

    if args.report_format in {"json", "both"}:
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    n_lines = len(asm.splitlines())
    print(f"Model: {args.model or type(model).__name__}")
    print(f"Inputs: {sorted(inputs.keys())}")
    print(f"Generated {n_lines} ASM lines -> {asm_path}")
    print(f"Trace nodes: {report['summary']['node_count_total']}")
    print(f"Peak live memory: {report['summary']['peak_live_bytes_human']}")
    print(f"Overall compute utilization: {report['summary']['overall_compute_utilization']}")
    if args.report_format in {"md", "both"}:
        print(f"Utilization report (Markdown) -> {report_md_path}")
    if args.report_format in {"json", "both"}:
        print(f"Utilization report (JSON) -> {report_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
