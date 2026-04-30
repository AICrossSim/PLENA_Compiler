"""Parameterised "tiled BTMM" kernel.

This generalises minimal_btmm in two ways:
  1. A / B / C HBM shapes are kernel-time parameters (Python ints chosen
     when the kernel is constructed; baked into the TIR before lowering).
  2. The kernel does ONE BTMM per (q_block, kv_block) iteration of the
     two outer loops. Inputs are sliced from A and B; output is written
     to a multi-tile slice of C (one tile per head).

Layout assumptions (BSHD on HBM, one-tile constraints from the BTMM ISA):
    A_hbm: (BATCH, SEQ_Q,  GROUP_HEADS, HLEN)
    B_hbm: (BATCH, SEQ_K,  GROUP_HEADS, HLEN)
    C_hbm: (BATCH, SEQ_Q,  GROUP_HEADS, SEQ_K)

    With GROUP_HEADS * HLEN == MLEN, A and B slices of shape
    (1, MLEN, GROUP_HEADS, HLEN) each fit a single mlen*mlen tile. The
    C slice (1, MLEN, GROUP_HEADS, MLEN) splits into GROUP_HEADS tiles
    in the parent's H*D-merged 2D layout when SEQ_K > MLEN -- this is
    the case Phase 8 unlocks via per-head multi-tile writeback.

    When SEQ_K == MLEN (degenerate case), the slice still has GROUP_HEADS
    tiles but they are physically adjacent in 2D -- our per-head iterator
    handles both cases uniformly because each head's tile lives at a
    distinct column offset h_idx * D regardless.
"""

import tvm
from tvm.script import tir as T


def make_tiled_btmm(
    *,
    batch: int = 1,
    seq_q: int = 128,
    seq_k: int = 128,
    head_count: int = 4,    # total heads in the tensors (multiple of LANE_COUNT)
    hlen: int = 16,
):
    """Build a parameterised tiled-BTMM PrimFunc.

    Hardware constants (hardwired, NOT user-tunable):
        * MLEN       = 64   -- PLENA tile width
        * LANE_COUNT = 4    -- BTMM lane count (heads processed per BTMM)

    Each BTMM op consumes exactly LANE_COUNT heads at a time. When
    `head_count > LANE_COUNT` we add a third loop level (`hg`) that
    iterates over head groups; each iteration loads the slice of A/B
    covering the current group's LANE_COUNT heads, runs BTMM, and
    writes back the per-head tiles.

    Constraints:
        * hlen * LANE_COUNT == MLEN          (BTMM hardware shape)
        * head_count % LANE_COUNT == 0       (clean head grouping)
        * seq_q % MLEN == 0, seq_k % MLEN == 0
    """
    MLEN = 64
    LANE_COUNT = 4
    if hlen * LANE_COUNT != MLEN:
        raise ValueError(
            f"hlen*LANE_COUNT ({hlen}*{LANE_COUNT}={hlen*LANE_COUNT}) must "
            f"equal MLEN ({MLEN})"
        )
    if head_count % LANE_COUNT:
        raise ValueError(
            f"head_count ({head_count}) must be a multiple of LANE_COUNT "
            f"({LANE_COUNT})"
        )
    if seq_q % MLEN or seq_k % MLEN:
        raise ValueError(
            f"seq_q ({seq_q}) and seq_k ({seq_k}) must be MLEN-aligned"
        )

    BATCH = batch
    SEQ_Q = seq_q
    SEQ_K = seq_k
    HEAD_COUNT = head_count
    HLEN = hlen
    NUM_Q = SEQ_Q // MLEN
    NUM_K = SEQ_K // MLEN
    NUM_HG = HEAD_COUNT // LANE_COUNT

    # Pre-compute shape tuples so the @T.prim_func parser doesn't have to
    # resolve closure variables at type-annotation parse time.
    A_SHAPE = (BATCH, SEQ_Q, HEAD_COUNT, HLEN)
    B_SHAPE = (BATCH, SEQ_K, HEAD_COUNT, HLEN)
    C_SHAPE = (BATCH, SEQ_Q, HEAD_COUNT, SEQ_K)
    # Working buffers are sized to ONE head-group (LANE_COUNT heads).
    A_V_SHAPE = (1, MLEN, LANE_COUNT, HLEN)
    B_M_SHAPE = (1, MLEN, LANE_COUNT, HLEN)
    C_V_SHAPE = (1, LANE_COUNT, MLEN, MLEN)

    @T.prim_func
    def tiled_btmm(
        A_hbm: T.Buffer(A_SHAPE, "float16"),
        B_hbm: T.Buffer(B_SHAPE, "float16"),
        C_hbm: T.Buffer(C_SHAPE, "float16"),
    ):
        A_v = T.alloc_buffer(A_V_SHAPE, "float16", scope="vram")
        B_m = T.alloc_buffer(B_M_SHAPE, "float16", scope="mram")
        C_v = T.alloc_buffer(C_V_SHAPE, "float16", scope="vram")

        for q_block in T.serial(NUM_Q):
            for hg in T.serial(NUM_HG):           # head group: 0..head_count/LANE_COUNT - 1
                for kv_block in T.serial(NUM_K):
                    # A's slice: head start = hg * LANE_COUNT, eh = LANE_COUNT
                    T.evaluate(T.call_extern(
                        "handle", "plena.dma_h2v_slice",
                        A_hbm.data, A_v.data,
                        4,
                        0, q_block * MLEN, hg * LANE_COUNT, 0,
                        1, MLEN, LANE_COUNT, HLEN,
                    ))
                    # B's slice: same head-group offset
                    T.evaluate(T.call_extern(
                        "handle", "plena.dma_h2m_slice",
                        B_hbm.data, B_m.data,
                        4,
                        0, kv_block * MLEN, hg * LANE_COUNT, 0,
                        1, MLEN, LANE_COUNT, HLEN,
                    ))
                    T.evaluate(T.call_extern(
                        "handle", "plena.btmm",
                        A_v.data, B_m.data, C_v.data, LANE_COUNT,
                    ))
                    # C writeback: per-head multi-tile, head start = hg * LANE_COUNT
                    T.evaluate(T.call_extern(
                        "handle", "plena.dma_v2h_slice",
                        C_v.data, C_hbm.data,
                        4,
                        0, q_block * MLEN, hg * LANE_COUNT, kv_block * MLEN,
                        1, MLEN, LANE_COUNT, MLEN,
                    ))

    constants = {
        "BATCH": BATCH, "SEQ_Q": SEQ_Q, "SEQ_K": SEQ_K,
        "HEAD_COUNT": HEAD_COUNT, "LANE_COUNT": LANE_COUNT,
        "HLEN": HLEN, "MLEN": MLEN,
        "NUM_Q": NUM_Q, "NUM_K": NUM_K, "NUM_HG": NUM_HG,
    }
    return tiled_btmm, constants


def build_module(
    *, batch: int = 1, seq_q: int = 128, seq_k: int = 128,
    head_count: int = 4, hlen: int = 16,
) -> tvm.IRModule:
    func, _ = make_tiled_btmm(
        batch=batch, seq_q=seq_q, seq_k=seq_k,
        head_count=head_count, hlen=hlen,
    )
    return tvm.IRModule({"tiled_btmm": func})


# ---------------------------------------------------------------------------
# Default-parameterised PrimFunc, exposed at module level so the CLI can
# fetch it via `--kernel tilelang_tvm_compiler.kernels.tiled_btmm:tiled_btmm_default`.
# Shape choices satisfy the testbench's stride-mode comparator:
#   * SEQ_Q == MLEN  -> single row block in the output (chunks_per_batch
#                       == col_blocks, no row-wise interleaving in VRAM)
#   * SEQ_K  > MLEN  -> exercises multi-tile slice writeback (per-head)
#   * head_count == LANE_COUNT -> single head-group iteration
#
# Test drivers should pass shape parameters explicitly via --kernel-kwargs
# to keep the compiled HBM layout in lock-step with their input data.
# ---------------------------------------------------------------------------
TILED_BTMM_DEFAULT_PARAMS = dict(
    batch=1,
    seq_q=64,
    seq_k=128,
    head_count=4,
    hlen=16,
)
tiled_btmm_default, TILED_BTMM_DEFAULT_CONSTANTS = make_tiled_btmm(**TILED_BTMM_DEFAULT_PARAMS)
