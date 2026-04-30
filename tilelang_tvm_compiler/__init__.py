"""TVM-based PLENA compiler.

Pipeline (3 passes):

    TIR PrimFunc (with PLENA scopes + plena.* extern calls)
        |
        v   PASS 1  PlenaCodegen.lower_to_hlir()
    HLIR (Buffer + Op stream, no addresses)
        |
        v   PASS 2  AddressAllocationPass
    HLIR with HBM/VRAM/MRAM addresses + stride/scale + tile annotations
        |
        v   PASS 3  IsaEmitterPass (re-uses runtime's ISAEmitter via shim)
    Real PLENA ISA text -> assembler -> .mem -> emulator

==============================================================================
HARD CONVENTIONS -- READ THIS FIRST WHEN ADDING A KERNEL
==============================================================================

These three rules are load-bearing. Violating any of them silently produces
emulator output that does not match golden, with no immediate error.

1) HBM IS ALWAYS BSHD.
   Every HBM-resident T.Buffer must declare shape as (Batch, Seq, Heads, Dim).
   `create_mem_for_sim` -> `map_mx_data_to_hbm_for_behave_sim` packs tensors
   into hbm_for_behave_sim.bin assuming this layout. The address_alloc pass
   computes per-tile stride from `H*D` (last two dims merged). If you
   declare an HBM buffer in any other order, the H_PREFETCH_*/H_STORE_V
   addresses will be wrong and the emulator will read/write the wrong cells.

2) VRAM/MRAM REFLECTS PHYSICAL HARDWARE LAYOUT.
   Different ops produce different physical layouts in VRAM/MRAM:
     - DMA from HBM (H_PREFETCH_V/M) preserves BSHD.
     - BTMM/BMM_WO writes BHSD (head is the outermost stored dimension --
       see transactional_emulator/src/main.rs:bmm_wo()).
   Declare the buffer's shape to match what the hardware actually produces.
   The dma_v2h pass is what reconciles "VRAM=BHSD" with "HBM=BSHD" via
   tile-level reorder during the store -- it walks col-block-major over the
   BSHD HBM dst, which lands vram_off = idx * tile_elems exactly on each
   head's tile boundary in BHSD VRAM, transparently transposing.
   Lying about the VRAM shape (e.g. labelling a BHSD VRAM buffer as BSHD)
   may "work" by coincidence for one kernel but breaks as soon as another
   op tries to index it.

3) COMPARISON IS ALWAYS BSHD.
   At the testbench boundary, both golden and the post-staging VRAM dump
   must be in BSHD-flat form (B*S rows x H*D cols):
     - golden: `golden_4d.reshape(B*S, H*D)` before passing to
       `create_sim_env(golden_result=...)`.
     - simulated: the `--stage-output BUFFER` flag emits per-tile DMAs
       that lay out HBM-BSHD into VRAM[0..] in a stride-mode-compatible
       arrangement. `view_mem.py` reassembles via `chunks_per_batch=H`,
       producing a BSHD-flat (B*S, H*D) tensor for diff against golden.

==============================================================================
"""

from .codegen import PlenaCodegen, compile_module
from .test_helper import emit_single_output_testbench
from . import scope
from . import intrinsics

__all__ = [
    "PlenaCodegen",
    "compile_module",
    "emit_single_output_testbench",
    "scope",
    "intrinsics",
]
