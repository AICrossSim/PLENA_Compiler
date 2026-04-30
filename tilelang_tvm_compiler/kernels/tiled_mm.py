"""Tiled regular matrix multiply (BSHT @ BTHD = BSHD).

Per-head GEMM contracted over T:

    A_hbm[b, s, h, t]  *  B_hbm[b, t, h, d]  ->  C_hbm[b, s, h, d]
    C[b, s, h, d] = sum_t A[b, s, h, t] * B[b, t, h, d]

Hardware uses M_MM (single-head, mlen*mlen output tile, contraction
runs through the M_MM/M_MM_WO accumulator), so the kernel walks heads
explicitly — there is no LANE_COUNT pack like BTMM.

Tiling (per output (mlen, mlen) tile, per head):
    1. zero_v   accumulator C_v
    2. for kv_block in NUM_K:                   # contract T in mlen chunks
           dma_h2v_slice  A_hbm -> A_v   (1, MLEN, 1, MLEN)
           dma_h2m_slice  B_hbm -> B_m   (1, MLEN, 1, MLEN)
           plena.mm        A_v @ B_m -> C_partial   (overwrites)
           plena.v_add     C_v += C_partial
    3. dma_v2h_slice C_v -> C_hbm   (1, MLEN, 1, MLEN)

Contraction across kv_blocks is done in software via V_ADD against a
separate accumulator tile because emit_matmul commits with M_MM_WO at
the end of each call (overwriting dst). A future optimisation would
pre-stage all NUM_K tiles into a multi-tile VRAM/MRAM region and
hand them to emit_matmul as a single accumulation chain — would save
NUM_K-1 tile adds per output tile but needs multi-tile slice DMA
support first.

Constraints:
    * seq_q  % MLEN == 0
    * seq_k  % MLEN == 0
    * Either:
        - d_dim % MLEN == 0                               (regular full-tile MM), or
        - d_dim < MLEN and LANE_COUNT * d_dim == MLEN     (grouped narrow-tile MM)
"""

import tvm
from tvm.script import tir as T


def make_tiled_mm(
    *,
    batch: int = 1,
    seq_q: int = 64,
    seq_k: int = 128,        # contracted dim T
    head_count: int = 4,
    d_dim: int = 64,         # output last dim
):
    MLEN = 64
    LANE_COUNT = 4
    if seq_q % MLEN:
        raise ValueError(f"seq_q ({seq_q}) must be a multiple of MLEN ({MLEN})")
    if seq_k % MLEN:
        raise ValueError(f"seq_k ({seq_k}) must be a multiple of MLEN ({MLEN})")
    grouped_narrow = d_dim < MLEN
    if grouped_narrow:
        if d_dim <= 0 or MLEN % d_dim != 0:
            raise ValueError(
                f"grouped narrow d_dim ({d_dim}) must be a positive divisor of MLEN ({MLEN})"
            )
        lane_count = MLEN // d_dim
        if lane_count != LANE_COUNT:
            raise ValueError(
                f"grouped narrow tiled_mm currently requires d_dim * LANE_COUNT == MLEN "
                f"({d_dim} * {LANE_COUNT} == {MLEN})"
            )
        if head_count % lane_count:
            raise ValueError(
                f"head_count ({head_count}) must be a multiple of lane_count ({lane_count})"
            )
    else:
        if d_dim % MLEN:
            raise ValueError(f"d_dim ({d_dim}) must be a multiple of MLEN ({MLEN})")
        lane_count = 1

    BATCH = batch
    SEQ_Q = seq_q
    SEQ_K = seq_k
    HEAD_COUNT = head_count
    D = d_dim
    NUM_Q = SEQ_Q // MLEN
    NUM_K = SEQ_K // MLEN
    NUM_D = D // MLEN if not grouped_narrow else 1
    NUM_HG = HEAD_COUNT // lane_count if grouped_narrow else HEAD_COUNT

    A_SHAPE = (BATCH, SEQ_Q, HEAD_COUNT, SEQ_K)        # BSHT
    B_SHAPE = (BATCH, SEQ_K, HEAD_COUNT, D)            # BTHD
    C_SHAPE = (BATCH, SEQ_Q, HEAD_COUNT, D)            # BSHD
    A_V_SHAPE = (1, MLEN, 1, MLEN)
    if grouped_narrow:
        B_M_SHAPE = (1, MLEN, lane_count, D)
        TILE_SHAPE = (1, MLEN, lane_count, D)
    else:
        B_M_SHAPE = (1, MLEN, 1, MLEN)
        TILE_SHAPE = (MLEN, MLEN)

    @T.prim_func
    def tiled_mm(
        A_hbm: T.Buffer(A_SHAPE, "float16"),
        B_hbm: T.Buffer(B_SHAPE, "float16"),
        C_hbm: T.Buffer(C_SHAPE, "float16"),
    ):
        A_v = T.alloc_buffer(A_V_SHAPE, "float16", scope="vram")
        B_m = T.alloc_buffer(B_M_SHAPE, "float16", scope="mram")
        C_partial = T.alloc_buffer(TILE_SHAPE, "float16", scope="vram")
        C_v = T.alloc_buffer(TILE_SHAPE, "float16", scope="vram")

        # NOTE on loop kinds: each plena.mm lowers (via the hw-loop
        # emitter) to one nested 16x16 hardware loop running ~256 M_MM
        # / M_MM_WO pairs == ~1.1k dynamic instructions. Adding DMAs +
        # V_ADD pushes one kv_block iter to ~1.5k dyn, one d_block iter
        # to ~3k, one h iter to ~6.5k -- all comfortably under the
        # emulator's 10000-per-iter cap. The OUTERMOST loop (q_block)
        # is the only one whose body dispatches all of (h * d * kv)
        # work in a single iteration, so its dyn count scales as
        # HEAD_COUNT * NUM_D * NUM_K * inner (~26k for the default
        # config) and would blow the cap. We unroll q_block at compile
        # time to dodge that; the remaining three levels stay as
        # hardware loops to keep the static ISA short.
        for q_block in T.unroll(NUM_Q):
            if grouped_narrow:
                for hg in T.serial(NUM_HG):
                    T.evaluate(T.call_extern(
                        "handle", "plena.zero_v",
                        C_v.data,
                    ))
                    for kv_block in T.serial(NUM_K):
                        T.evaluate(T.call_extern(
                            "handle", "plena.dma_h2m_slice",
                            B_hbm.data, B_m.data,
                            4,
                            0, kv_block * MLEN, hg * lane_count, 0,
                            1, MLEN, lane_count, D,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.zero_v",
                            C_partial.data,
                        ))
                        # Narrow grouped path: each lane contributes one
                        # D-wide slot within the packed 64x64 B/C tiles.
                        # `lane * D` now lowers through ExprMaterializer,
                        # so we can keep this as a regular TIR loop.
                        for lane in T.serial(lane_count):
                            T.evaluate(T.call_extern(
                                "handle", "plena.dma_h2v_slice",
                                A_hbm.data, A_v.data,
                                4,
                                0, q_block * MLEN, hg * lane_count + lane, kv_block * MLEN,
                                1, MLEN, 1, MLEN,
                            ))
                            T.evaluate(T.call_extern(
                                "handle", "plena.mm_slot",
                                A_v.data, B_m.data, C_partial.data,
                                0,                 # lhs_row_offset (single-tile A_v)
                                lane * D,          # rhs_col_offset
                                lane * D,          # dst_col_offset
                                D,                 # col_count
                            ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.v_add",
                            C_v.data, C_partial.data, C_v.data,
                        ))
                    T.evaluate(T.call_extern(
                        "handle", "plena.dma_v2h_slice",
                        C_v.data, C_hbm.data,
                        4,
                        0, q_block * MLEN, hg * lane_count, 0,
                        1, MLEN, lane_count, D,
                    ))
            else:
                for h in T.serial(HEAD_COUNT):
                    for d_block in T.serial(NUM_D):
                        T.evaluate(T.call_extern(
                            "handle", "plena.zero_v",
                            C_v.data,
                        ))
                        for kv_block in T.serial(NUM_K):
                            T.evaluate(T.call_extern(
                                "handle", "plena.dma_h2v_slice",
                                A_hbm.data, A_v.data,
                                4,
                                0, q_block * MLEN, h, kv_block * MLEN,
                                1, MLEN, 1, MLEN,
                            ))
                            T.evaluate(T.call_extern(
                                "handle", "plena.dma_h2m_slice",
                                B_hbm.data, B_m.data,
                                4,
                                0, kv_block * MLEN, h, d_block * MLEN,
                                1, MLEN, 1, MLEN,
                            ))
                            T.evaluate(T.call_extern(
                                "handle", "plena.mm",
                                A_v.data, B_m.data, C_partial.data,
                            ))
                            T.evaluate(T.call_extern(
                                "handle", "plena.v_add",
                                C_v.data, C_partial.data, C_v.data,
                            ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.dma_v2h_slice",
                            C_v.data, C_hbm.data,
                            4,
                            0, q_block * MLEN, h, d_block * MLEN,
                            1, MLEN, 1, MLEN,
                        ))

    constants = {
        "BATCH": BATCH, "SEQ_Q": SEQ_Q, "SEQ_K": SEQ_K,
        "HEAD_COUNT": HEAD_COUNT, "D": D, "MLEN": MLEN,
        "NUM_Q": NUM_Q, "NUM_K": NUM_K, "NUM_D": NUM_D,
        "LANE_COUNT": lane_count, "NUM_HG": NUM_HG,
        "GROUPED_NARROW": grouped_narrow,
    }
    return tiled_mm, constants


def build_module(
    *, batch: int = 1, seq_q: int = 64, seq_k: int = 128,
    head_count: int = 4, d_dim: int = 64,
) -> tvm.IRModule:
    func, _ = make_tiled_mm(
        batch=batch, seq_q=seq_q, seq_k=seq_k,
        head_count=head_count, d_dim=d_dim,
    )
    return tvm.IRModule({"tiled_mm": func})
