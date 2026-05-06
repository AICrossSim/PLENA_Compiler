"""tilelang -> PLENA-flavored TIR frontend.

Lowers a tilelang `@T.prim_func` (with `T.Kernel`, `T.alloc_shared`,
`T.copy`, `T.gemm`, ...) into the same TIR shape that
`tilelang_tvm_compiler.codegen.PlenaCodegen` consumes.

Public entry: `compile_func(func) -> tir.PrimFunc`
"""

from .pipeline import compile_func, compile_to_tir_text

__all__ = ["compile_func", "compile_to_tir_text"]
