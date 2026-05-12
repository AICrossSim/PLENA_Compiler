"""compiler.aten — ATen-style PLENA compiler path.

PlenaCompiler program builder + op backend registry. Pairs with compiler.generator
for the two-path compiler (template vs. aten).
"""

from pathlib import Path

PLENA_PKG_DIR = Path(__file__).parent
NATIVE_OPS_YAML = PLENA_PKG_DIR / "native_ops.yaml"

from compiler.aten.isa_builder import (  # noqa: E402, F401
    Comment,
    Instr,
    IsaBuilder,
    Register,
    addr,
    fp,
    gp,
)
from compiler.aten.plena_compiler import (  # noqa: E402, F401
    FPVar,
    InputVar,
    IsaCompiler,
    MemoryStateMixin,
    PlenaCompiler,
    TensorVar,
    TileCompiler,
    VRAMMatrixVar,
)
from compiler.aten.ops.registry import OpRegistry, Backend  # noqa: E402, F401
