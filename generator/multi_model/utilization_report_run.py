from pathlib import Path

import torch

from model import TRANSFOMER_ENCODER
from utilization_report import analyse_trace_utilization, render_markdown_report
from vlm_parser import VLMModelParser


def main() -> int:
    torch.manual_seed(2061452)
    x = torch.rand(2, 4, 4096)
    model = TRANSFOMER_ENCODER.TRANSFOMER_ENCODER()

    parser = VLMModelParser()
    parser.load_model(model)
    trace_tree = parser.trace_leaf_modules(model, {"x": x})
    model_info = parser.extract_model_info(model=model, inputs={"x": x})

    report = analyse_trace_utilization(
        trace_tree,
        model_info=model_info,
        hw={"tile_m": 64, "tile_k": 64, "tile_n": 64},
    )

    out_path = Path(__file__).resolve().parent / "outputs" / "report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown_report(report), encoding="utf-8")

    print(f"Trace nodes: {report['summary']['node_count_total']}")
    print(f"Peak live memory: {report['summary']['peak_live_bytes_human']}")
    print(f"Overall compute utilization: {report['summary']['overall_compute_utilization']}")
    print(f"Markdown report written to: {out_path}")
    return 0


if __name__ == "__main__":
    main()
