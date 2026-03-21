from unittest import case
import torch
import torch.fx as fx
import torch.nn as nn
import transformers

from pathlib import Path

if __package__ in (None, ""):
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from multi_model.vlm_codegen_context import matrix_shape_from_meta
else:
    from .vlm_codegen_context import matrix_shape_from_meta

def play_matrix_shape_from_meta():
    meta = {"shape": [2, 4, 4096], "dtype": "torch.float32", "device": "cpu"}
    shape = matrix_shape_from_meta(meta)
    print(f"Meta: {meta}")
    print(f"Shape: {shape}")
    
def print_fx_graph(model: nn.Module, inputs: dict[str, torch.Tensor]):
    traced = fx.symbolic_trace(model)
    print(traced.graph)

if __name__ == "__main__":
    run_index = 1
    if run_index == 0:
        play_matrix_shape_from_meta()
    elif run_index == 1:
        print_fx_graph(nn.Linear(4, 4), {"x": torch.randn(1, 4)})
