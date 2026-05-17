"""Minimal online-softmax kernel over one Score tile (HBM round-trip).

  m_curr = max(score_row)
  m_new  = max(m_old, m_curr)
  m_res  = exp(m_old - m_new)
  score  = exp(score - m_new)
  l_new  = l_old * m_res + sum(score)
  m_old/l_old updated in-place

FPRAM layout: one flat region starting at FPRAM_USER_BASE; each slot
takes lane_count*rows elements; addresses passed directly to the FP /
row `_at` intrinsics.
"""

import tvm
from tvm.script import tir as T

from ..address_alloc import FPRAM_USER_BASE


_SLOTS = ("M_OLD", "M_CURR", "M_RES", "L_OLD", "L_NEW", "P_SUM")


def _slot_bases(fp_state_elems: int) -> dict[str, int]:
    return {name: FPRAM_USER_BASE + i * fp_state_elems for i, name in enumerate(_SLOTS)}


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

    fp_state_elems = lane_count * rows
    bases = _slot_bases(fp_state_elems)
    M_OLD  = bases["M_OLD"]
    M_CURR = bases["M_CURR"]
    M_RES  = bases["M_RES"]
    L_OLD  = bases["L_OLD"]
    L_NEW  = bases["L_NEW"]
    P_SUM  = bases["P_SUM"]

    @T.prim_func
    def online_softmax_hbm(
        Score_hbm: T.Buffer(SCORE_SHAPE, "float16"),
        Score_out_hbm: T.Buffer(SCORE_SHAPE, "float16"),
    ):
        Score_v = T.alloc_buffer(SCORE_SHAPE, "float16", scope="vram")

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
                    M_OLD + lane * rows + row, M_CURR + lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.row_reduce_max_at",
                    Score_v.data, M_CURR + lane * rows + row, row, lane,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_sub_at",
                    M_OLD + lane * rows + row, M_CURR + lane * rows + row, M_RES + lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_exp_at",
                    M_RES + lane * rows + row, M_RES + lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.row_sub_fp_at",
                    Score_v.data, M_CURR + lane * rows + row, Score_v.data, row, lane,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.row_exp_at",
                    Score_v.data, Score_v.data, row, lane,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_sub_at",
                    P_SUM + lane * rows + row, P_SUM + lane * rows + row, P_SUM + lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.row_reduce_sum_at",
                    Score_v.data, P_SUM + lane * rows + row, row, lane,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_mul_at",
                    L_OLD + lane * rows + row, M_RES + lane * rows + row, L_NEW + lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_add_at",
                    L_NEW + lane * rows + row, P_SUM + lane * rows + row, L_NEW + lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_copy_at",
                    M_CURR + lane * rows + row, M_OLD + lane * rows + row,
                ))
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_copy_at",
                    L_NEW + lane * rows + row, L_OLD + lane * rows + row,
                ))
        T.evaluate(T.call_extern(
            "handle", "plena.dma_v2h_slice",
            Score_v.data, Score_out_hbm.data,
            4,
            0, 0, 0, 0,
            1, rows, lane_count, hlen,
        ))

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
        "M_OLD_ADDR": M_OLD,
        "M_CURR_ADDR": M_CURR,
        "M_RES_ADDR": M_RES,
        "L_OLD_ADDR": L_OLD,
        "L_NEW_ADDR": L_NEW,
        "P_SUM_ADDR": P_SUM,
    }
    return online_softmax_hbm, constants


def build_hbm_module(
    *, rows: int = 64, hlen: int = 16, lane_count: int = 4, active_lane: int = 0,
) -> tvm.IRModule:
    func, _ = make_online_softmax_hbm(
        rows=rows, hlen=hlen, lane_count=lane_count, active_lane=active_lane,
    )
    return tvm.IRModule({"online_softmax_hbm": func})
