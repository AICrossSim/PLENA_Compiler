"""Minimal online-softmax kernel over one VRAM score tile.

This is not full FlashAttention yet. It only covers the score update:
  m_curr = max(score_row)
  m_new  = max(m_old, m_curr)
  m_res  = exp(m_old - m_new)
  score  = exp(score - m_new)
  l_new  = l_old * m_res + sum(score)
  m_old/l_old updated in-place
"""

import tvm
from tvm.script import tir as T

from ..address_alloc import FPRAM_USER_BASE


def make_online_softmax_min(*, rows: int = 64, cols: int = 64):
    MLEN = 64
    if rows <= 0 or rows > MLEN:
        raise ValueError(f"rows must be in (0, {MLEN}], got {rows}")
    if cols != MLEN:
        raise ValueError(f"minimal online softmax currently expects cols == MLEN ({MLEN}), got {cols}")

    SCORE_SHAPE = (rows, cols)
    FP_STATE_SHAPE = (rows,)

    @T.prim_func
    def online_softmax_min():
        Score_v = T.alloc_buffer(SCORE_SHAPE, "float16", scope="vram")
        M_old = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        M_curr = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        M_res = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        L_old = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        L_new = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        P_sum = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")

        T.evaluate(T.call_extern(
            "handle", "plena.row_reduce_max",
            Score_v.data, M_curr.data,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.fp_max",
            M_old.data, M_curr.data, M_curr.data,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.fp_sub",
            M_old.data, M_curr.data, M_res.data,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.fp_exp",
            M_res.data, M_res.data,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.row_sub_fp",
            Score_v.data, M_curr.data, Score_v.data,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.row_exp",
            Score_v.data, Score_v.data,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.row_reduce_sum",
            Score_v.data, P_sum.data,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.fp_mul",
            L_old.data, M_res.data, L_new.data,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.fp_add",
            L_new.data, P_sum.data, L_new.data,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.fp_copy",
            M_curr.data, M_old.data,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.fp_copy",
            L_new.data, L_old.data,
        ))

    constants = {"ROWS": rows, "COLS": cols, "MLEN": MLEN}
    return online_softmax_min, constants


def build_module(*, rows: int = 64, cols: int = 64) -> tvm.IRModule:
    func, _ = make_online_softmax_min(rows=rows, cols=cols)
    return tvm.IRModule({"online_softmax_min": func})


def make_online_softmax_hbm(
    *,
    rows: int = 64,
    hlen: int = 16,
    lane_count: int = 4,
    active_lane: int = 0,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"online_softmax_hbm currently requires rows == MLEN ({MLEN}), got {rows}")
    if hlen <= 0 or hlen > MLEN or MLEN % hlen != 0:
        raise ValueError(f"hlen must be a positive divisor of MLEN={MLEN}, got {hlen}")
    if lane_count * hlen != MLEN:
        raise ValueError(
            f"lane_count * hlen must equal MLEN ({lane_count} * {hlen} == {MLEN})"
        )
    if not (0 <= active_lane < lane_count):
        raise ValueError(f"active_lane must be in [0, {lane_count}), got {active_lane}")

    grouped = hlen < MLEN
    mask_val = 1 << active_lane
    SCORE_SHAPE = (1, rows, lane_count, hlen)
    FP_STATE_SHAPE = (lane_count, rows)

    @T.prim_func
    def online_softmax_hbm(
        Score_hbm: T.Buffer(SCORE_SHAPE, "float16"),
        Score_out_hbm: T.Buffer(SCORE_SHAPE, "float16"),
    ):
        Score_v = T.alloc_buffer(SCORE_SHAPE, "float16", scope="vram")
        M_old = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        M_curr = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        M_res = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        L_old = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        L_new = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")
        P_sum = T.alloc_buffer(FP_STATE_SHAPE, "float16", scope="fpram")

        T.evaluate(T.call_extern(
            "handle", "plena.dma_h2v_slice",
            Score_hbm.data, Score_v.data,
            4,
            0, 0, 0, 0,
            1, rows, lane_count, hlen,
        ))
        for lane in T.serial(lane_count):
            for row in T.serial(rows):
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_copy_at",
                    M_old.data, M_curr.data, lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.row_reduce_max_at",
                    Score_v.data, M_curr.data, row, lane,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_sub_at",
                    M_old.data, M_curr.data, M_res.data, lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_exp_at",
                    M_res.data, M_res.data, lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.row_sub_fp_at",
                    Score_v.data, M_curr.data, Score_v.data, row, lane,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.row_exp_at",
                    Score_v.data, Score_v.data, row, lane,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_sub_at",
                    P_sum.data, P_sum.data, P_sum.data, lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.row_reduce_sum_at",
                    Score_v.data, P_sum.data, row, lane,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_mul_at",
                    L_old.data, M_res.data, L_new.data, lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_add_at",
                    L_new.data, P_sum.data, L_new.data, lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_copy_at",
                    M_curr.data, M_old.data, lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_copy_at",
                    L_new.data, L_old.data, lane * rows + row,
                ))
        T.evaluate(T.call_extern(
            "handle", "plena.dma_v2h_slice",
            Score_v.data, Score_out_hbm.data,
            4,
            0, 0, 0, 0,
            1, rows, lane_count, hlen,
        ))

    fp_state_elems = lane_count * rows
    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "LANE_COUNT": lane_count,
        "ACTIVE_LANE": active_lane,
        "MASK_VAL": mask_val,
        "GROUPED": grouped,
        "FPRAM_USER_BASE": FPRAM_USER_BASE,
        "FP_STATE_ELEMS": fp_state_elems,
        "M_OLD_ADDR": FPRAM_USER_BASE + 0 * fp_state_elems,
        "M_CURR_ADDR": FPRAM_USER_BASE + 1 * fp_state_elems,
        "M_RES_ADDR": FPRAM_USER_BASE + 2 * fp_state_elems,
        "L_OLD_ADDR": FPRAM_USER_BASE + 3 * fp_state_elems,
        "L_NEW_ADDR": FPRAM_USER_BASE + 4 * fp_state_elems,
        "P_SUM_ADDR": FPRAM_USER_BASE + 5 * fp_state_elems,
    }
    return online_softmax_hbm, constants


def build_hbm_module(
    *, rows: int = 64, hlen: int = 16, lane_count: int = 4, active_lane: int = 0,
) -> tvm.IRModule:
    func, _ = make_online_softmax_hbm(
        rows=rows, hlen=hlen, lane_count=lane_count, active_lane=active_lane,
    )
    return tvm.IRModule({"online_softmax_hbm": func})
