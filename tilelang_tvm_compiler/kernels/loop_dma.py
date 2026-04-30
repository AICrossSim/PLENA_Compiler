"""Minimal loop kernel: a for-loop wrapping a DMA.

Goal: validate the Phase 4 ForOp lowering end-to-end.

Body is intentionally degenerate (the same DMA every iteration, no slice
indices) -- we want to test that the LOOP STRUCTURE lowers correctly:
    C_LOOP_START gp_loop, 4
      <DMA body>
      S_ADDI_INT gp_idx, gp_idx, 1
    C_LOOP_END gp_loop

A meaningful loop would slice the buffer using `i` (e.g.
`A_hbm[i*M:(i+1)*M, ...]`). That requires BufferSlice in HLIR + Pass 3
slice support, which is the NEXT phase. Until then, the body just
re-runs the same DMA -- functionally pointless but a clean structural
check on the loop machinery.
"""

from __future__ import annotations

import tvm
from tvm.script import tir as T

# Same shape conventions as minimal_btmm so the loop body uses an already-
# debugged DMA pattern (BSHD on HBM, mlen-tile-aligned).
BATCH = 1
SEQ = 64
GROUP_HEADS = 4
HLEN = 16
MLEN = 64
ITERS = 4   # matches q_block_count in attention.py for these shapes


@T.prim_func
def loop_dma(
    A_hbm: T.Buffer((BATCH, SEQ, GROUP_HEADS, HLEN), "float16"),
    A_v_out: T.Buffer((BATCH, SEQ, GROUP_HEADS, HLEN), "float16"),  # unused; HBM placeholder
):
    A_v = T.alloc_buffer((BATCH, SEQ, GROUP_HEADS, HLEN), "float16", scope="vram")
    for i in T.serial(ITERS):
        T.evaluate(T.call_extern(
            "handle", "plena.dma_h2v",
            A_hbm.data, A_v.data, BATCH * SEQ * GROUP_HEADS * HLEN,
        ))


def build_module() -> tvm.IRModule:
    return tvm.IRModule({"loop_dma": loop_dma})
