import torch
import torch.nn as nn
import math
from model import TRANSFOMER_ENCODER
from vlm_parser import * 
from vlm_codegen import *


class MultiLayerEncoder(nn.Module):
    def __init__(self, num_layers=2, hidden_size=4096):
        super().__init__()
        self.layers = nn.ModuleList([TRANSFOMER_ENCODER.TRANSFOMER_ENCODER(hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

if __name__ == "__main__":
    torch.manual_seed(2061452)
    hidden_size = 128
    x = torch.rand(2, 4, hidden_size)
    print(x.shape)
    #model = TRANSFOMER_ENCODER.TRANSFOMER_ENCODER(hidden_size = hidden_size)
    model = MultiLayerEncoder(num_layers=3, hidden_size=hidden_size)
    parser = VLMModelParser()
    parser.load_model(model)
    parser.load_inputs({"x": x})
    trace_tree = parser.trace()
    nodes = flatten_call_tree(trace_tree)
    model_info = parser.extract_model_info()
    
    export_flatten_call_tree(nodes, "./outputs/trace.json")
    export_model_info(model_info, "./outputs/model_info.json")
    
    asm = vlm_codegen(nodes, model_info, mode = "default")

    out_path = Path("./outputs/simple_run_output.asm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(asm)

    n_lines = len(asm.splitlines())
    print(f"Generated {n_lines} ASM lines → {out_path}")
