"""Reference kernel: single 64×64 @ 64×64 matmul.

Demonstrates the simplest happy-path through the new tilelang frontend
pipeline:

  * `T.copy` from HBM into per-operand shared / fragment buffers
  * `T.gemm` with the default kind (overwrite) → ``plena.matmul``
  * `T.copy` from the output fragment back to HBM

Lowering route::

    tl.tileop.copy        --[lower_to_hlir]-->  plena.dma_h2v_slice / h2m / v2h
    tl.tileop.gemm_py     --[lower_to_hlir]-->  plena.matmul (M_tiles=K_tiles=1, N=64)

Entry point: ``make_mm64(rows=64, cols=64) -> tir.PrimFunc``.
"""

from __future__ import annotations

import tilelang.language as T


def make_mm64(rows: int = 64, cols: int = 64) -> "T.prim_func":
    if rows != 64 or cols != 64:
        raise ValueError(f"mm64 reference fixed at 64×64 (got {rows}×{cols})")

    @T.prim_func
    def mm64(
        A: T.Tensor((1, 64, 1, 64), "float16"),
        B: T.Tensor((1, 64, 1, 64), "float16"),
        C: T.Tensor((1, 64, 1, 64), "float16"),
    ):
        with T.Kernel(1, threads=128) as bx:
            A_sh = T.alloc_shared((64, 64), "float16")
            B_sh = T.alloc_shared((64, 64), "float16")
            C_loc = T.alloc_fragment((64, 64), "float16")
            T.copy(A[0, 0, 0, 0], A_sh)
            T.copy(B[0, 0, 0, 0], B_sh)
            T.gemm(A_sh, B_sh, C_loc)
            T.copy(C_loc, C[0, 0, 0, 0])

    return mm64


__all__ = ["make_mm64"]
