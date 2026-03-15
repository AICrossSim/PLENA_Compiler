import sys
from pathlib import Path
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from generator.multi_model import *
from generator.multi_model.vlm_codegen_context import matrix_shape_from_meta

def play_matrix_shape_from_meta():
    meta = {"shape": [2, 4, 4096], "dtype": "torch.float32", "device": "cpu"}
    shape = matrix_shape_from_meta(meta)
    print(f"Meta: {meta}")
    print(f"Shape: {shape}")

if __name__ == "__main__":
    play_matrix_shape_from_meta()