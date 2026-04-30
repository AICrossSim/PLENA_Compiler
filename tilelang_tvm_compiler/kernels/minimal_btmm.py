"""Minimal kernel: one BTMM with explicit DMA staging.

Intentionally trivial -- no loops, no softmax, no accumulation. The point
is to validate the full path:

    TIR PrimFunc
      -> custom storage scopes ("vram"/"mram"/"hbm")
      -> plena.* extern calls
      -> PlenaCodegen
      -> textual ISA

Shape conventions:

    - HBM buffers are ALWAYS BSHD = (Batch, Seq, Heads, Dim).
      This is the canonical layout the runtime kernels (attention.py /
      linear.py) use and the only thing `create_mem_for_sim` knows how
      to pack into hbm_for_behave_sim.bin.

    - VRAM/MRAM buffers reflect the PHYSICAL layout the hardware
      produces/consumes, which is sometimes different from BSHD:
        * inputs after H_PREFETCH_V land BSHD (DMA preserves layout)
        * BTMM/BMM_WO writes its output BHSD: head is the outermost
          dimension because the hardware writes one full mlen*mlen
          tile per head, head-major. See main.rs:bmm_wo() for proof.
      The dma_v2h pass is what reconciles "BHSD in VRAM" with
      "BSHD in HBM" via a tile reorder during the store.

    Constraint: GROUP_HEADS * HLEN must equal MLEN, otherwise the merged
    tile width does not match the BTMM hardware shape.
"""

from __future__ import annotations

import tvm
from tvm.script import tir as T

# BTMM shape constants. Match what attention.py uses for one head-group.
BATCH = 1
SEQ = 64           # mirrors mlen for this minimal kernel
MLEN = 64          # hardware tile width
GROUP_HEADS = 4
HLEN = 16

assert GROUP_HEADS * HLEN == MLEN, (
    f"GROUP_HEADS*HLEN ({GROUP_HEADS}*{HLEN}={GROUP_HEADS*HLEN}) must equal "
    f"MLEN ({MLEN}); BTMM expects merged head tiles to fill one mlen tile."
)


@T.prim_func
def minimal_btmm(
    # ---- HBM buffers: BSHD (canonical) ----
    A_hbm: T.Buffer((BATCH, SEQ, GROUP_HEADS, HLEN), "float16"),
    B_hbm: T.Buffer((BATCH, SEQ, GROUP_HEADS, HLEN), "float16"),
    C_hbm: T.Buffer((BATCH, SEQ, GROUP_HEADS, MLEN), "float16"),
):
    # ---- VRAM/MRAM buffers reflect physical layout ----
    # A_v / B_m: input DMA preserves BSHD.
    # C_v: BMM_WO writes head-major, so the physical layout is BHSD.
    #      dma_v2h reorders to BSHD when committing to C_hbm.
    A_v = T.alloc_buffer((BATCH, SEQ, GROUP_HEADS, HLEN), "float16", scope="vram")
    B_m = T.alloc_buffer((BATCH, SEQ, GROUP_HEADS, HLEN), "float16", scope="mram")
    C_v = T.alloc_buffer((BATCH, GROUP_HEADS, SEQ, MLEN), "float16", scope="vram")

    T.evaluate(T.call_extern(
        "handle", "plena.dma_h2v",
        A_hbm.data, A_v.data, BATCH * SEQ * GROUP_HEADS * HLEN,
    ))
    T.evaluate(T.call_extern(
        "handle", "plena.dma_h2m",
        B_hbm.data, B_m.data, BATCH * SEQ * GROUP_HEADS * HLEN,
    ))
    T.evaluate(T.call_extern(
        "handle", "plena.btmm",
        A_v.data, B_m.data, C_v.data, GROUP_HEADS,
    ))
    T.evaluate(T.call_extern(
        "handle", "plena.dma_v2h",
        C_v.data, C_hbm.data, BATCH * SEQ * GROUP_HEADS * MLEN,
    ))


def build_module() -> tvm.IRModule:
    return tvm.IRModule({"minimal_btmm": minimal_btmm})
