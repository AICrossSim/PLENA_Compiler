"""
PLENA ATen-style operator registration package.

Usage:
    from plena.ops.registry import OpRegistry, Backend
    import plena.ops as ops

    OpRegistry.load()  # loads native_ops.yaml from this package
    prog = PLENAProgram(mlen=64, blen=4)
    result = ops.softmax(prog, X_batch, scale=1.0)
"""
from pathlib import Path

PLENA_PKG_DIR = Path(__file__).parent
NATIVE_OPS_YAML = PLENA_PKG_DIR / "native_ops.yaml"


def __getattr__(name: str):
    if name == "ops":
        from . import ops as ops_module

        return ops_module
    if name in {"Backend", "OpRegistry"}:
        from .ops.registry import Backend, OpRegistry

        return {"Backend": Backend, "OpRegistry": OpRegistry}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
