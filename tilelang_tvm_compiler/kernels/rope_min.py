"""RoPE-min kernel — written in tilelang style.

Multi-S × multi-head RoPE with branchless pair-swap and FPRAM-scalar
lowering. The pair-swap (output element d depends on input elements
d and d^1) is expressed as loop fission over half_dim — no
``T.if_then_else``, no per-element predicate. Each iteration writes both
the even (``2*i``) and odd (``2*i+1``) output slots in straight-line
code.

Lowering path:
  * ``T.copy`` for HBM↔VRAM tile transfer (existing).
  * Per-row inner: ``T.copy(shared_2d[row, 0], frag_1d)`` lowers to a
    contiguous DMA from VRAM into an FPRAM-resident 1D fragment (v2f).
    Same fragments are reused across rows — one FPRAM region per slot,
    bound once by ``allocate_group_memory``.
  * FPRAM-only scalar FMA over half_dim pairs. ``X_FP[e]`` and
    ``X_FP[o]`` are different elements of the same fragment addressed
    as scalars; no cross-element hardware shuffle needed.
  * ``T.copy(frag_1d, shared_2d[row, 0])`` lowers to f2v symmetrically.

What this kernel needs from the compiler that isn't already in place:
  * ``T.copy(shared, fragment)`` and ``T.copy(fragment, shared)`` lowered
    to ``plena.dma_v2f`` / ``plena.dma_f2v`` (length=hlen, contiguous,
    dynamic row index).
  * Scalar FPRAM lvalue stores (``OUT_FP[e] = ...``) where ``e`` is an
    affine expression of an enclosing loop var. flash_attention_min
    already lvalue-stores into a 1D fragment with a loop-var index
    (``M_RES[row] = ...``); this just reuses that with ``e = 2*i``.

K-side RoPE has identical structure (XK→K, SIN↔NEG_SIN role swap).
Kept out of this minimal kernel.
"""

import tvm
import tilelang.language as T

from ..frontend import compile_func


def make_rope_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int = 8,
    half_dim: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
):
    full_dim = half_dim * 2
    if full_dim != hlen:
        raise ValueError(
            f"full_dim (= 2*half_dim = {full_dim}) must equal hlen ({hlen})"
        )
    MLEN = 64
    if rows != MLEN:
        raise ValueError(
            f"rope_min requires rows == MLEN ({MLEN}), got {rows}"
        )
    if MLEN % hlen != 0:
        raise ValueError(f"hlen must divide MLEN ({MLEN}); got hlen={hlen}")
    hardware_lane_count = MLEN // hlen
    if head_count % hardware_lane_count != 0:
        raise ValueError(
            f"head_count must be a multiple of MLEN/hlen={hardware_lane_count}; "
            f"got {head_count}"
        )
    if num_s_blocks < 1:
        raise ValueError(f"num_s_blocks must be >= 1, got {num_s_blocks}")

    seq_len = num_s_blocks * rows

    @T.prim_func
    def rope_min(
        XQ_hbm:      T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        COS_hbm:     T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        SIN_hbm:     T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        NEG_SIN_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Q_OUT_hbm:   T.Tensor((batch, seq_len, head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            XQ_sh      = T.alloc_shared((rows, hlen), "float16")
            COS_sh     = T.alloc_shared((rows, hlen), "float16")
            SIN_sh     = T.alloc_shared((rows, hlen), "float16")
            NEG_SIN_sh = T.alloc_shared((rows, hlen), "float16")
            Q_OUT_sh   = T.alloc_shared((rows, hlen), "float16")

            # FPRAM scratch — one (hlen,) fragment per source.
            # Allocated at kernel scope; same FPRAM offsets are reused
            # across every row of every (s_block, head) tile.
            X_FP   = T.alloc_fragment((hlen,), "float16")
            C_FP   = T.alloc_fragment((hlen,), "float16")
            S_FP   = T.alloc_fragment((hlen,), "float16")
            NS_FP  = T.alloc_fragment((hlen,), "float16")
            OUT_FP = T.alloc_fragment((hlen,), "float16")

            T.copy(XQ_hbm     [0, s_block * rows, by, 0], XQ_sh)
            T.copy(COS_hbm    [0, s_block * rows, by, 0], COS_sh)
            T.copy(SIN_hbm    [0, s_block * rows, by, 0], SIN_sh)
            T.copy(NEG_SIN_hbm[0, s_block * rows, by, 0], NEG_SIN_sh)

            for row in T.serial(rows):
                T.copy(XQ_sh     [row, 0], X_FP)
                T.copy(COS_sh    [row, 0], C_FP)
                T.copy(SIN_sh    [row, 0], S_FP)
                T.copy(NEG_SIN_sh[row, 0], NS_FP)

                for i in T.unroll(half_dim):
                    e = 2 * i
                    o = 2 * i + 1
                    OUT_FP[e] = X_FP[e] * C_FP[e] + X_FP[o] * NS_FP[e]
                    OUT_FP[o] = X_FP[o] * C_FP[o] + X_FP[e] * S_FP[o]

                T.copy(OUT_FP, Q_OUT_sh[row, 0])

            T.copy(Q_OUT_sh, Q_OUT_hbm[0, s_block * rows, by, 0])

    lowered = compile_func(rope_min)

    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "HALF_DIM": half_dim,
        "FULL_DIM": full_dim,
        "BATCH": batch,
        "NUM_S_BLOCKS": num_s_blocks,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
    }
    return lowered, constants
