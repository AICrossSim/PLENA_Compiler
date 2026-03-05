import torch
import torch.nn as nn
import math
from model import TRANSFOMER_ENCODER
from vlm_parser import VLMModelParser
from vlm_codegen import *

if __name__ == "__main__":
    torch.manual_seed(2061452)
    x = torch.rand(2, 4, 4096)
    print(x.shape)
    transformer = TRANSFOMER_ENCODER.TRANSFOMER_ENCODER()
    parser = VLMModelParser()
    parser.load_model(transformer)
    parser.load_inputs({"x": x})
    trace_tree = parser.trace()
    nodes = flatten_call_tree(trace_tree)
    model_info = {
        "name": "TRANSFOMER_ENCODER",
        "input_shapes": {"x": list(x.shape)},
        "output_shapes": {"output": list(x.shape)},
    }
    asm = vlm_codegen(nodes, model_info)

    out_path = Path("./outputs/simple_run_output.asm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(asm)

    n_lines = len(asm.splitlines())
    print(f"Generated {n_lines} ASM lines → {out_path}")