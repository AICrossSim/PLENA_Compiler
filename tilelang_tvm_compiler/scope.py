"""PLENA storage scopes.

In TVM TIR, a buffer's "storage scope" is just a string attached to the
buffer. We pick a fixed vocabulary here so different parts of the compiler
agree on which physical memory each buffer lives in.

Scope semantics (mirrors PLENA hardware):
    HBM   -- main DRAM, source/sink for DMA
    VRAM  -- vector SRAM, LHS operand of BTMM/MM and target of vector ops
    MRAM  -- matrix SRAM, RHS operand of BTMM/MM
    FPRAM -- on-chip FP buffer (small, used for staging)

PrimFunc parameters (function arguments) are treated as HBM by default.
"""

HBM = "hbm"
VRAM = "vram"
MRAM = "mram"
FPRAM = "fpram"

ALL_SCOPES = (HBM, VRAM, MRAM, FPRAM)


def is_known(scope: str) -> bool:
    return scope in ALL_SCOPES
