"""
PLENA backend implementation for softmax operator.

ATen-style: the function receives a PlenaCompiler context and TensorVar
arguments, orchestrates PlenaCompiler calls, and returns the result TensorVar.
"""

from __future__ import annotations

from compiler.aten.isa_builder import IsaBuilder, fp, gp


def softmax_plena(prog, input_var, scale: float = 1.0):
    """
    PLENA backend: single-pass, per-row, register-direct row-wise softmax.

    The input is one (mlen, mlen) block whose rows each fit in a single vector
    tile (mlen == VLEN), so the whole softmax can be computed in one pass over
    the row without the streaming running-max / running-sum machinery that the
    flash-attention path (``_online_softmax_asm``) needs for multi-tile
    sequences.

    This mirrors the hand-written softmax_asm template EXACTLY: it emits, per
    row, the 6-op sequence with the reduction results (max, sum) held in FP
    REGISTERS and fed DIRECTLY to their immediate vector consumers -- no
    ``S_ST_FP`` / ``S_LD_FP`` round-trip through FP SRAM:

        V_RED_MAX f2, row, 0          ; row max -> FP register f2
        V_SUB_VF  row, row, f2, 0, 0  ; x - max   (f2 used directly)
        V_EXP_V   row, row, 0         ; exp(x - max)
        S_ADD_FP  f3, f0, f0          ; clear sum accumulator
        V_RED_SUM f3, row             ; row sum -> FP register f3
        S_RECI_FP f3, f3, 0           ; 1 / sum   (in register)
        V_MUL_VF  row, row, f3, 0     ; normalize (f3 used directly)

    Batching the reductions and round-tripping max/sum through FP SRAM (the
    ``tile_row_max`` / ``tile_row_sum`` helpers) trips a reduction-result
    capture hazard on back-to-back reductions (the stored sum comes out ~half
    on alternate rows -> result ~2x golden). Keeping the reduction result in a
    register and consuming it immediately avoids that hazard entirely.

    No scalar running-max (no ``S_MAX_FP``, which is a silent no-op in RTL) and
    no correction-factor bookkeeping.

    Args:
        prog:      PlenaCompiler instance (compilation context)
        input_var: BatchVar or VRAMMatrixVar — the input matrix in VRAM
                   Shape: (mlen, mlen)
        scale:     Multiplicative scale applied before softmax (default 1.0).
                   When != 1.0 it is read from FP SRAM slot 1.

    Returns:
        The output VRAM matrix ``S`` (the softmax result is written in place
        into S, tile (0, 0), across all mlen rows).

    Note:
        fp_preload layout expected: [0]=0.0, [1]=scale.
    """
    mlen = prog.mlen

    # Allocate output matrix in VRAM (S holds the softmax result).
    S = prog.alloc("S", mlen, mlen)

    # Step 0: Initialize S = input.  The allocated VRAM starts zeroed in the
    # simulator, matching the previous implementation's S += input behavior.
    prog.vram_add(S, input_var)

    s_addr = prog.get_vram_addr(S.name)

    # f2 = row max, f3 = row sum; each reduction result stays in its register and
    # feeds the next vector op directly (no S_ST_FP/S_LD_FP round-trip).
    fp_max = 2
    fp_sum = 3

    (gp_row,) = prog._reg.allocate_gp(1)
    try:
        asm = IsaBuilder().comment("=== Single-pass per-row softmax on S (register-direct) ===")
        for row in range(mlen):
            row_addr = s_addr + row * mlen
            asm.comment(f"row {row}: softmax over VRAM[{row_addr}:{row_addr + mlen}]")
            asm.instr("S_ADDI_INT", gp(gp_row), gp(0), row_addr)

            # Optional pre-scale: row *= scale (FP SRAM slot 1 holds scale).
            if scale != 1.0:
                asm.instr("S_LD_FP", fp(fp_max), gp(0), 1)
                asm.instr("V_MUL_VF", gp(gp_row), gp(gp_row), fp(fp_max), 0)

            # max -> sub -> exp -> (clear acc) sum -> reci -> mul, register-direct
            asm.instr("V_RED_MAX", fp(fp_max), gp(gp_row), 0)
            asm.instr("V_SUB_VF", gp(gp_row), gp(gp_row), fp(fp_max), 0, 0)
            asm.instr("V_EXP_V", gp(gp_row), gp(gp_row), 0)
            asm.instr("S_ADD_FP", fp(fp_sum), fp(0), fp(0))
            asm.instr("V_RED_SUM", fp(fp_sum), gp(gp_row))
            asm.instr("S_RECI_FP", fp(fp_sum), fp(fp_sum), 0)
            asm.instr("V_MUL_VF", gp(gp_row), gp(gp_row), fp(fp_sum), 0)

        prog.emit(asm)
    finally:
        prog._reg.free_gp([gp_row])

    return S
