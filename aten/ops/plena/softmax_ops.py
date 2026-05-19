"""
PLENA backend implementation for softmax operator.

This encapsulates the online softmax algorithm that was previously
written inline in fpvar_softmax_test.py (Tilelang-style).

ATen-style: the function receives a PlenaCompiler context and TensorVar
arguments, orchestrates PlenaCompiler calls, and returns the result TensorVar.
"""

from __future__ import annotations


def softmax_plena(prog, input_var, scale: float = 1.0):
    """
    PLENA backend: online softmax (numerically stable, row-wise).

    Algorithm (matches fpvar_softmax_test.py):
        Initialize: m_old[row] = -inf, l_old[row] = 0
        for row in range(mlen):
            m_old_saved = m_old[row]
            S[row] *= scale
            row_max = max(S[row])
            m_old[row] = max(m_old[row], row_max)  # m_curr
            m_res[row] = exp(m_old_saved - m_curr)
            S[row] -= m_curr
            P[row] = exp(S[row])
            sum_p = sum(P[row])
            l_old[row] = l_old[row] * m_res[row] + sum_p
        P[row] /= l_old[row]  # final normalization

    Args:
        prog:      PlenaCompiler instance (compilation context)
        input_var: BatchVar or VRAMMatrixVar — the input matrix in VRAM
                   Shape: (mlen, mlen)
        scale:     Multiplicative scale applied before softmax (default 1.0)

    Returns:
        The same input_var (in-place modification) after softmax is applied.
        The result is stored in S (VRAM), which is also returned.

    Note:
        fp_preload layout expected: [0]=0.0, [1]=scale, [2]=-inf
        The caller must set fp_preload = [0.0, scale, float('-inf'), ...]
    """
    mlen = prog.mlen

    # Allocate output matrix in VRAM (S holds the softmax result).
    S = prog.alloc("S", mlen, mlen)

    # Step 0: Initialize S = input.  The allocated VRAM starts zeroed in the
    # simulator, matching the previous implementation's S += input behavior.
    prog.vram_add(S, input_var)

    # Reuse the flash-attention online-softmax state layout instead of
    # allocating separate FPVars for m_old/m_res/l_old/inv_l.  At MLEN=256 the
    # old FPVar-heavy lowering needed 1028 slots plus reserved constants and
    # overflowed the 1024-slot FPRAM.  The shared layout uses exactly 3*MLEN
    # slots at _ONLINE_SOFTMAX_FPSRAM_BASE and scalar FP registers for temps.
    fp_sram_start = prog._ONLINE_SOFTMAX_FPSRAM_BASE
    prog.emit(prog._reset_fpsram_asm(fp_sram_start, mlen, 2))  # m_old = -inf
    prog.emit(prog._reset_fpsram_asm(fp_sram_start + 2 * mlen, mlen, 0))  # l_old = 0
    prog.emit(
        prog._online_softmax_asm(
            mlen=mlen,
            s_address=prog.get_vram_addr(S.name),
            m_start_address=fp_sram_start,
            scale=scale,
            rows=mlen,
        )
    )
    prog.final_scale_o(0, S, rows=mlen)

    return S
