"""Loop + dynamic-start slice: validates Phase 7 end-to-end.

Kernel intent (attention.py-style):
    for i in T.serial(NUM_BLOCKS):
        copy A_hbm[0, i*MLEN : (i+1)*MLEN, :, :]  ->  A_v

Each iteration loads a different mlen-row band of A. The slice's
seq-dim start is `i * MLEN`, which is a runtime-computed PrimExpr --
ExprMaterializer must produce ISA that reads `i` (gp_idx) and
strength-reduces `i * MLEN`  (MLEN=64=2^6) to `S_SLLI_INT gp_off, gp_idx, 6`.
The DMA is then issued with `hbm_start_offset_reg=gp_off`.

This is the smallest demonstration of:
    * loop var binding -> symbol_table -> ExprMaterializer
    * dynamic offset expression: `i * (MLEN * H * D)` with strength
      reduction against PLENA's S_SLLI_INT
    * isa_emitter accepting a register-sourced offset
"""

from __future__ import annotations

import tvm
from tvm.script import tir as T

BATCH = 1
SEQ_TOTAL = 256       # 4 mlen tiles in seq dim
GROUP_HEADS = 4
HLEN = 16
MLEN = 64             # GROUP_HEADS * HLEN must equal MLEN
NUM_BLOCKS = SEQ_TOTAL // MLEN  # = 4


@T.prim_func
def loop_slice_dma(
    A_hbm: T.Buffer((BATCH, SEQ_TOTAL, GROUP_HEADS, HLEN), "float16"),
    A_v_dummy: T.Buffer((BATCH, SEQ_TOTAL, GROUP_HEADS, HLEN), "float16"),
):
    A_v = T.alloc_buffer((BATCH, MLEN, GROUP_HEADS, HLEN), "float16", scope="vram")
    for i in T.serial(NUM_BLOCKS):
        T.evaluate(T.call_extern(
            "handle", "plena.dma_h2v_slice",
            A_hbm.data, A_v.data,
            4,                                  # ndim
            0, i * MLEN, 0, 0,                  # starts (seq start = i*MLEN)
            BATCH, MLEN, GROUP_HEADS, HLEN,     # extents
        ))


def build_module() -> tvm.IRModule:
    return tvm.IRModule({"loop_slice_dma": loop_slice_dma})
