"""Minimal FlashAttention kernel (single Q-block x single KV-block).

Mirrors `transactional_emulator/testbench/tile_tensor_kernel_programs/attention.py`
but expressed in TIR + plena.* intrinsics. Intentionally simple:

  * one q_block, one kv_block (so no outer KV loop yet)
  * no softmax scale
  * no causal mask
  * all lanes run the same online-softmax update

Dataflow (per kv_block, here only one):
  Q_v   = DMA(Q_hbm)
  K_m   = DMA(K_hbm)              # MRAM for BTMM rhs
  V_m   = DMA(V_hbm)
  zero(O_v)
  S_v   = BTMM(Q_v, K_m)          # Q @ K^T per head
  -- online softmax in place on S_v --
  for row in 0..mlen:
      M_curr = max(M_old, max(S_v[row]))    # masked row-reduce
      M_res  = exp(M_old - M_curr)
      S_v[row] = exp(S_v[row] - M_curr)     # this becomes P
      P_sum  = sum(S_v[row])
      L_new  = L_old * M_res + P_sum
      O_v[row] *= M_res                     # rescale running output
      M_old <- M_curr ; L_old <- L_new
  PV_v  = BTMM(S_v, V_m)
  O_v   += PV_v
  -- (final O / L_new is left to a follow-up; only matters once we have
     the outer KV loop and accumulate over multiple blocks.)
  DMA(O_v, O_hbm)

FP-state preload requirements (handled in testbench):
  Scale[h, :]  = 1 / sqrt(d_k)
  M_init[h, :] = -inf surrogate
  L_init[h, :] = 0
"""

import tvm
from tvm.script import tir as T

from ..address_alloc import FPRAM_USER_BASE


def make_flash_attention_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    lane_count: int = 4,
    active_lane: int = 0,
    num_kv_blocks: int = 2,
    num_q_blocks: int = 2,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"flash_attention_min currently requires rows == MLEN ({MLEN}), got {rows}")
    if lane_count * hlen != MLEN:
        raise ValueError(f"lane_count*hlen must == MLEN ({MLEN})")
    if not (0 <= active_lane < lane_count):
        raise ValueError(f"active_lane out of range")
    if num_kv_blocks < 1:
        raise ValueError(f"num_kv_blocks must be >= 1, got {num_kv_blocks}")
    if num_q_blocks < 1:
        raise ValueError(f"num_q_blocks must be >= 1, got {num_q_blocks}")

    grouped = hlen < MLEN
    kv_seq = num_kv_blocks * rows
    q_seq = num_q_blocks * rows
    # Q and O cover all Q blocks back-to-back along the seq dim.
    Q_HBM_SHAPE = (1, q_seq, lane_count, hlen)
    O_HBM_SHAPE = (1, q_seq, lane_count, hlen)
    # On-chip Q / O tiles hold one Q block at a time.
    Q_TILE_SHAPE = (1, rows, lane_count, hlen)
    O_TILE_SHAPE = (1, rows, lane_count, hlen)
    # K and V cover all KV blocks back-to-back along the seq dim.
    KV_HBM_SHAPE = (1, kv_seq, lane_count, hlen)
    # On-chip K/V tiles hold ONE block at a time -- we re-DMA per kv iter.
    KV_TILE_SHAPE = (1, rows, lane_count, hlen)
    # BTMM #1 writes a (B, H, M, M) tile; flat the last dim into lane_count*hlen
    # for HBM compatibility (BHSD layout). For our intermediate VRAM tile, we
    # use the BHSD shape directly so per-head P[h] starts at h*mlen*mlen.
    S_SHAPE = (1, lane_count, rows, MLEN)
    # PV mirrors O's BSHD layout so the v_add accumulator has identical
    # per-head column-slot striding. mm_slot writes head h's hlen
    # columns at dst_col_offset = h*hlen within the mlen-wide row.
    PV_SHAPE = (1, rows, lane_count, hlen)
    FP_STATE_SHAPE = (lane_count, rows)

    @T.prim_func
    def flash_attention_min(
        Q_hbm: T.Buffer(Q_HBM_SHAPE, "float16"),
        K_hbm: T.Buffer(KV_HBM_SHAPE, "float16"),
        V_hbm: T.Buffer(KV_HBM_SHAPE, "float16"),
        O_hbm: T.Buffer(O_HBM_SHAPE, "float16"),
    ):
        Q_v   = T.alloc_buffer(Q_TILE_SHAPE, "float16", scope="vram")
        K_m   = T.alloc_buffer(KV_TILE_SHAPE, "float16", scope="mram")
        V_m   = T.alloc_buffer(KV_TILE_SHAPE, "float16", scope="mram")
        S_v   = T.alloc_buffer(S_SHAPE, "float16", scope="vram")
        PV_v  = T.alloc_buffer(PV_SHAPE, "float16", scope="vram")
        O_v   = T.alloc_buffer(O_TILE_SHAPE, "float16", scope="vram")
        M_old = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        M_curr = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        M_res = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        L_old = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        L_new = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        P_sum = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        # Softmax scale (= 1 / sqrt(d_k)). Preloaded by the testbench for
        # every head segment with all-equal `1/sqrt(hlen)` values.
        Scale = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        # Reciprocal of L_new, used for the final O = O / L_new step.
        L_inv = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        # Per-q_block reset constants. Preloaded by the testbench:
        #   M_init[h, :] = -inf surrogate
        #   L_init[h, :] = 0
        # The kernel copies these into M_old / L_old at the start of each
        # q_block iteration so the FP state carrying online softmax across
        # KV blocks is correctly reset between Q tiles.
        M_init = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        L_init = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")

        # ---- Q outer loop ----
        # Per Q tile we (re)stage Q, reset the running m/l state, run all
        # KV blocks through the online softmax, finalize O = O / L_new,
        # and DMA the result out at the q_block-th slot of O_hbm. Unrolled
        # so q_block is a compile-time constant in DMA scalars.
        for q_block in T.unroll(num_q_blocks):
            # DMA Q[q_block] -> Q_v.
            T.evaluate(T.call_extern(
                "handle", "plena.dma_h2v_slice",
                Q_hbm.data, Q_v.data, 4,
                0, q_block * rows, 0, 0,
                1, rows, lane_count, hlen,
            ))

            # Clear running output accumulator for this Q tile.
            T.evaluate(T.call_extern("handle", "plena.zero_v", O_v.data))

            # Reset M_old / L_old for this Q tile by copying the preloaded
            # constants (M_init = -inf, L_init = 0) into every head's FP
            # segment. Without this every q_block past the first would
            # inherit the previous tile's m_old / l_old.
            for h in T.serial(lane_count):
                for row in T.serial(rows):
                    T.evaluate(T.call_extern(
                        "handle", "plena.fp_copy_at",
                        M_init.data, M_old.data, h * rows + row,
                    ))
                    T.evaluate(T.call_extern(
                        "handle", "plena.fp_copy_at",
                        L_init.data, L_old.data, h * rows + row,
                    ))

            # ---- KV outer loop ----
            # Software-unroll so kv_block becomes a compile-time constant.
            # Per-iter body:
            #     1. DMA K[kv], V[kv] -> on-chip K_m / V_m
            #     2. BTMM #1: Q @ K^T -> S_v
            #     3. online softmax over every head-row in S_v
            #        (also rescales O_v by exp(m_old - m_curr))
            #     4. BTMM #2: per head P @ V -> PV_v
            #     5. v_add: O_v += PV_v
            for kv_block in T.serial(num_kv_blocks):
                T.evaluate(T.call_extern(
                    "handle", "plena.dma_h2m_slice",
                    K_hbm.data, K_m.data, 4,
                    0, kv_block * rows, 0, 0,
                    1, rows, lane_count, hlen,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.dma_h2m_slice",
                    V_hbm.data, V_m.data, 4,
                    0, kv_block * rows, 0, 0,
                    1, rows, lane_count, hlen,
                ))

                # Q @ K^T -> S_v (lane_count heads, mlen x mlen score per head).
                T.evaluate(T.call_extern(
                    "handle", "plena.btmm",
                    Q_v.data, K_m.data, S_v.data, lane_count,
                ))

                # ---- online softmax over S_v + rescale O_v ----
                # `_at` row ops now take logical (dim2, dim3) coordinates and
                # let the emitter derive physical row packing automatically.
                # For S_v (BHSD) we address (head, row); for O_v (BSHD) we
                # address (row, head).
                # ---- online softmax over S_v + per-head P @ V ----
                # Each head's softmax state is independent, so we can finish the
                # row-wise update for one head and immediately launch mm_slot for
                # that same head. v_add stays outside because it consumes the
                # whole packed PV_v tile once every head slot has been overwritten.
                for h in T.serial(lane_count):
                    for row in T.serial(rows):
                        # Scale: S_v[h, row, :] *= 1/sqrt(d_k).
                        T.evaluate(T.call_extern(
                            "handle", "plena.row_mul_fp_at",
                            S_v.data, Scale.data, S_v.data,
                            h, row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.fp_copy_at",
                            M_old.data, M_curr.data, h * rows + row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.row_reduce_max_at",
                            S_v.data, M_curr.data, h, row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.fp_sub_at",
                            M_old.data, M_curr.data, M_res.data, h * rows + row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.fp_exp_at",
                            M_res.data, M_res.data, h * rows + row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.row_sub_fp_at",
                            S_v.data, M_curr.data, S_v.data,
                            h, row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.row_exp_at",
                            S_v.data, S_v.data,
                            h, row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.fp_copy_at",
                            L_init.data, P_sum.data, h * rows + row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.row_reduce_sum_at",
                            S_v.data, P_sum.data,
                            h, row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.fp_mul_at",
                            L_old.data, M_res.data, L_new.data, h * rows + row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.fp_add_at",
                            L_new.data, P_sum.data, L_new.data, h * rows + row,
                        ))
                        # Rescale running output: O_v[row, h, :] *= M_res
                        T.evaluate(T.call_extern(
                            "handle", "plena.row_mul_fp_at",
                            O_v.data, M_res.data, O_v.data,
                            row, h,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.fp_copy_at",
                            M_curr.data, M_old.data, h * rows + row,
                        ))
                        T.evaluate(T.call_extern(
                            "handle", "plena.fp_copy_at",
                            L_new.data, L_old.data, h * rows + row,
                        ))

                    T.evaluate(T.call_extern(
                        "handle", "plena.mm_slot",
                        S_v.data, V_m.data, PV_v.data,
                        h * MLEN * MLEN,   # lhs_row_offset (head h's tile in S_v)
                        h * hlen,          # rhs_col_offset (head h's V columns)
                        h * hlen,          # dst_col_offset (head h's PV columns)
                        hlen,              # col_count
                    ))
                T.evaluate(T.call_extern(
                    "handle", "plena.v_add",
                    O_v.data, PV_v.data, O_v.data,
                ))

            # Final softmax normalization: O[row, h, :] /= L_new[h, row].
            for h in T.serial(lane_count):
                for row in T.serial(rows):
                    T.evaluate(T.call_extern(
                        "handle", "plena.fp_reci_at",
                        L_new.data, L_inv.data, h * rows + row,
                    ))
                    T.evaluate(T.call_extern(
                        "handle", "plena.row_mul_fp_at",
                        O_v.data, L_inv.data, O_v.data,
                        row, h,
                    ))

            # DMA this Q tile's normalized output back to O_hbm[q_block].
            T.evaluate(T.call_extern(
                "handle", "plena.dma_v2h_slice",
                O_v.data, O_hbm.data, 4,
                0, q_block * rows, 0, 0,
                1, rows, lane_count, hlen,
            ))

    fp_state_elems = lane_count * rows
    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "LANE_COUNT": lane_count,
        "ACTIVE_LANE": active_lane,
        "GROUPED": grouped,
        "FPRAM_USER_BASE": FPRAM_USER_BASE,
        "FP_STATE_ELEMS": fp_state_elems,
        # FP buffer ordering matches T.alloc_buffer declarations above.
        "M_OLD_ADDR":  FPRAM_USER_BASE + 0 * fp_state_elems,
        "M_CURR_ADDR": FPRAM_USER_BASE + 1 * fp_state_elems,
        "M_RES_ADDR":  FPRAM_USER_BASE + 2 * fp_state_elems,
        "L_OLD_ADDR":  FPRAM_USER_BASE + 3 * fp_state_elems,
        "L_NEW_ADDR":  FPRAM_USER_BASE + 4 * fp_state_elems,
        "P_SUM_ADDR":  FPRAM_USER_BASE + 5 * fp_state_elems,
        "SCALE_ADDR":  FPRAM_USER_BASE + 6 * fp_state_elems,
        "L_INV_ADDR":  FPRAM_USER_BASE + 7 * fp_state_elems,
        "M_INIT_ADDR": FPRAM_USER_BASE + 8 * fp_state_elems,
        "L_INIT_ADDR": FPRAM_USER_BASE + 9 * fp_state_elems,
        "NUM_KV_BLOCKS": num_kv_blocks,
        "NUM_Q_BLOCKS": num_q_blocks,
    }
    return flash_attention_min, constants


def build_module(
    *, rows: int = 64, hlen: int = 16, lane_count: int = 4, active_lane: int = 0,
) -> tvm.IRModule:
    func, _ = make_flash_attention_min(
        rows=rows, hlen=hlen, lane_count=lane_count, active_lane=active_lane,
    )
    return tvm.IRModule({"flash_attention_min": func})
