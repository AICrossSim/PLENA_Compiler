"""Minimal FPRAM/FP-op smoke kernel.

Exercises:
  * arbitrary-shaped FPRAM alloc buffers
  * VRAM <-> FPRAM mapping
  * scalar/elementwise FP ops on FPRAM
  * row-wise VRAM reduction into FPRAM
  * row-wise VRAM op with FPRAM scalar RHS
"""

import tvm
from tvm.script import tir as T


@T.prim_func
def fpram_smoke():
    V_src = T.alloc_buffer((2, 64), "float16", scope="vram")
    V_dst = T.alloc_buffer((2, 64), "float16", scope="vram")
    F_src = T.alloc_buffer((2, 64), "float16", scope="fpram")
    F_tmp = T.alloc_buffer((2, 64), "float16", scope="fpram")
    Row_max = T.alloc_buffer((2,), "float16", scope="fpram")

    T.evaluate(T.call_extern(
        "handle", "plena.map_v_to_fp",
        V_src.data, F_src.data,
    ))
    T.evaluate(T.call_extern(
        "handle", "plena.fp_exp",
        F_src.data, F_tmp.data,
    ))
    T.evaluate(T.call_extern(
        "handle", "plena.map_fp_to_v",
        F_tmp.data, V_dst.data,
    ))
    T.evaluate(T.call_extern(
        "handle", "plena.row_reduce_max",
        V_src.data, Row_max.data,
    ))
    T.evaluate(T.call_extern(
        "handle", "plena.row_sub_fp",
        V_src.data, Row_max.data, V_dst.data,
    ))


def build_module() -> tvm.IRModule:
    return tvm.IRModule({"fpram_smoke": fpram_smoke})
