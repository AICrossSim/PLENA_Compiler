"""compiler.aten — ATen-style PLENA compiler path.

PlenaCompiler DSL + compile_module (torch.export → ISA) + op backend
registry. Pairs with compiler.generator for the two-path compiler
(template vs. aten).
"""

from pathlib import Path

PLENA_PKG_DIR = Path(__file__).parent
NATIVE_OPS_YAML = PLENA_PKG_DIR / "native_ops.yaml"

from compiler.aten.plena_compiler import (  # noqa: E402, F401
    PlenaCompiler,
    TileCompiler,
    DeveloperCompiler,
    RegisterAllocator,
    TensorVar,
    InputVar,
    VRAMMatrixVar,
    FPVar,
    TensorKind,
    tensor_kind,
    Tensor,
    TensorInfo,
    TileLayout,
    MemoryBlock,
    VirtualMemoryManager,
    MRAMAllocator,
    VRAMAllocator,
    FPRAMAllocator,
    SubMatrixInfo,
    MatrixBlockLayout,
    VRAMSubMatrixInfo,
    VRAMMatrixBlockLayout,
    MemoryObjectInfo,
    FPRAMObjectLayout,
)
from compiler.aten.ops.registry import OpRegistry, Backend  # noqa: E402, F401
from compiler.aten.compile_module import compile_module, quantize_to_mxfp  # noqa: E402, F401
