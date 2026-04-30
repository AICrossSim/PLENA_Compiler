"""Static-slice DMA kernel: validates Phase 6 BufferSlice end-to-end.

The HBM source A_hbm has shape (1, 128, 4, 16) -- twice as many sequence
positions as one mlen tile. We DMA only the SECOND half (`A_hbm[0,
64:128, :, :]`) into VRAM. Logical-2D collapse:
    parent: (B*S, H*D) = (128, 64)
    slice:  rows 64..128 (row_start=64), all cols (col_start=0, col_ext=64)
    -> single mlen*mlen tile starting at element offset 64*64 = 4096.

We expect the emitted ISA to do an H_PREFETCH_V whose hbm_start_offset
loads `4096` (i.e. `S_ADDI_INT gpX, gp0, 4096` before the prefetch),
proving the slice arithmetic flowed through correctly.
"""

from __future__ import annotations

import tvm
from tvm.script import tir as T

BATCH = 1
SEQ_TOTAL = 128         # parent has 2 mlen-tiles in the seq dim
SLICE_START = 64        # take the second half
SLICE_EXTENT = 64       # one mlen tile's worth
GROUP_HEADS = 4
HLEN = 16
MLEN = 64               # GROUP_HEADS * HLEN must == MLEN


@T.prim_func
def static_slice_dma(
    A_hbm: T.Buffer((BATCH, SEQ_TOTAL, GROUP_HEADS, HLEN), "float16"),
    A_hbm_dummy: T.Buffer((BATCH, SEQ_TOTAL, GROUP_HEADS, HLEN), "float16"),
):
    A_v = T.alloc_buffer((BATCH, SLICE_EXTENT, GROUP_HEADS, HLEN), "float16", scope="vram")
    # plena.dma_h2v_slice signature:
    #   src_buf, dst_buf, ndim, *starts, *extents
    T.evaluate(T.call_extern(
        "handle", "plena.dma_h2v_slice",
        A_hbm.data, A_v.data,
        4,                                                # ndim
        0, SLICE_START, 0, 0,                             # starts (B, S, H, D)
        BATCH, SLICE_EXTENT, GROUP_HEADS, HLEN,           # extents
    ))


def build_module() -> tvm.IRModule:
    return tvm.IRModule({"static_slice_dma": static_slice_dma})
