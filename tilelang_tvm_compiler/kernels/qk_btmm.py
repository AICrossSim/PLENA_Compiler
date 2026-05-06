"""Reference kernel: head-fused Q @ K^T via BTMM.

Demonstrates the lane-fusion path of the new frontend:

  * ``T.Kernel(1, lane_count)`` — the ``by`` axis is a head_like grid
    binding which becomes a lane group of extent ``lane_count``.
  * Per-head DMAs ``T.copy(Q[..., by, ...], Q_sh)`` get sync-wrapped and
    fused — the resulting ``plena.dma_h2v_slice`` is a single multi-lane
    DMA covering all four heads.
  * The gemm carries ``T.attr(0, KIND, "btmm")`` so it lowers through
    the head-fused ``M_BTMM`` / ``M_BMM_WO`` hardware path.

Lowering route::

    T.copy(Q[..., by, ...], Q_sh)
        + sync + plena.group(lane_count)
        --[lower_to_hlir]-->
            plena.dma_h2v_slice(Q.data, Q_sh.data, ndim=4,
                                 0, 0, 0, 0, 1, rows, lane_count, hlen)

    T.gemm(Q_sh, K_sh, S_loc, transpose_B=True) under KIND="btmm"
        --[lower_to_hlir]-->  plena.btmm(Q_sh.data, K_sh.data, S_loc.data, lane_count)

The for-loop iterating ``by`` is dropped after lane fusion — every op
inside has been collapsed into a single multi-lane HW op.

Entry point: ``make_qk_btmm(rows=64, hlen=16, lane_count=4) -> tir.PrimFunc``.
"""

from __future__ import annotations

import tilelang.language as T

from tilelang_tvm_compiler.frontend.gemm_macros import KIND


def make_qk_btmm(rows: int = 64, hlen: int = 16, lane_count: int = 4) -> "T.prim_func":
    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"rows must equal mlen={MLEN}, got {rows}")
    if lane_count * hlen != MLEN:
        raise ValueError(
            f"lane_count*hlen must equal mlen={MLEN}; got {lane_count}*{hlen}"
        )

    @T.prim_func
    def qk_btmm(
        Q: T.Tensor((1, rows, lane_count, hlen), "float16"),
        K: T.Tensor((1, rows, lane_count, hlen), "float16"),
        S: T.Tensor((1, rows, lane_count, MLEN), "float16"),
    ):
        with T.Kernel(1, lane_count, threads=128) as (bx, by):
            Q_sh = T.alloc_shared((rows, hlen), "float16")
            K_sh = T.alloc_shared((rows, hlen), "float16")
            S_loc = T.alloc_fragment((rows, MLEN), "float16")
            T.copy(Q[0, 0, by, 0], Q_sh)
            T.copy(K[0, 0, by, 0], K_sh)
            with T.attr(0, KIND, "btmm"):
                T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)
            T.copy(S_loc, S[0, 0, by, 0])

    return qk_btmm


__all__ = ["make_qk_btmm"]
