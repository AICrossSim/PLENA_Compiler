"""flash_attention gemm-only debug kernel — with a step dial.

Base: BTMM(Q@K^T) + matmul(S@V), no softmax. ``fd_steps`` (0..6) adds
the softmax of flash_attention_min.py back ONE BLOCK AT A TIME, where
each block is copied VERBATIM from flash_attention_min (so every level
is a strict subset of that kernel and is guaranteed to compile):

    0  O = (Q@K^T) @ V                       (gemm-only base)
    1  + block A : S *= scale ; M_CURR = M_OLD
    2  + reduce_max -> M_CURR
    3  + block B : M_RES = exp(M_OLD-M_CURR) ; S = exp(S-M_CURR) ; P_SUM=0
    4  + reduce_sum -> P_SUM
    5  + block C : L_NEW = L_OLD*M_RES+P_SUM ; O *= M_RES ; advance state
    6  + block D : L_INV = 1/L_NEW ; O *= L_INV   (== full flash_attention)

These are flash_attention_min.py's own natural code blocks (lines
190-229), not an arbitrary split — so each level always compiles.
The testbench golden mirrors the same fd_steps.

NOTE the O at intermediate levels:
  * 1/2  : O == (scale * Q@K^T) @ V    (M_CURR computed, unused for O)
  * 3/4  : O == exp(scale*S - M_CURR) @ V
  * 5    : O == M_RES * (exp(...) @ V)
  * 6    : O == (M_RES * (exp(...) @ V)) / L_NEW
"""

import math

import tilelang.language as T

from ..frontend.gemm_macros import KIND
from ..plena_settings import load_sizes as _load_sizes


def make_flash_attention_gemm_only(
    *,
    rows: int | None = None,
    hlen: int | None = None,
    head_count: int | None = None,
    num_kv_blocks: int = 1,
    num_q_blocks: int = 1,
    fd_steps: int = 0,
):
    # Hardware sizes default to plena_settings.toml's active mode.
    _hw = _load_sizes()
    MLEN = _hw.mlen
    if hlen is None:
        hlen = _hw.hlen
    if rows is None:
        rows = MLEN
    if rows != MLEN:
        raise ValueError(
            f"flash_attention_gemm_only requires rows == MLEN ({MLEN}), got {rows}"
        )
    if MLEN % hlen != 0:
        raise ValueError(
            f"hlen must divide MLEN ({MLEN}); got hlen={hlen}"
        )
    if not (0 <= fd_steps <= 6):
        raise ValueError(f"fd_steps must be in [0, 6], got {fd_steps}")
    hardware_lane_count = MLEN // hlen
    if head_count is None:
        head_count = hardware_lane_count
    if head_count % hardware_lane_count != 0:
        raise ValueError(
            f"head_count must be a multiple of MLEN/hlen={hardware_lane_count}; "
            f"got {head_count}"
        )

    kv_seq = num_kv_blocks * rows
    q_seq = num_q_blocks * rows
    scale_val = 1.0 / math.sqrt(hlen)

    # DMA-IN-ONLY probe: HBM -> VRAM, nothing else. No writeback, no
    # gemm, no FPRAM. Isolates a single dma_h2v_slice (H_PREFETCH_V
    # chain). The result lives in Q_sh (VRAM); compare it directly,
    # no O_hbm, no compare-staging. ``fd_steps`` and K/V/O are kept
    # for signature compatibility but unused.
    @T.prim_func
    def flash_attention_gemm_only(
        Q_hbm: T.Tensor((1, q_seq,  head_count, hlen), "float16"),
        K_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        V_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        O_hbm: T.Tensor((1, q_seq,  head_count, hlen), "float16"),
    ):
        with T.Kernel(num_q_blocks, head_count, threads=128) as (q_block, by):
            Q_sh = T.alloc_shared((rows, hlen), "float16")

            # HBM -> VRAM. That's it.
            T.copy(
                Q_hbm[0, q_block * rows : (q_block + 1) * rows, by, 0:hlen],
                Q_sh,
            )

    lowered = flash_attention_gemm_only
    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
        "NUM_KV_BLOCKS": num_kv_blocks,
        "NUM_Q_BLOCKS": num_q_blocks,
        "FD_STEPS": fd_steps,
    }
    return lowered, constants


__all__ = ["make_flash_attention_gemm_only"]
