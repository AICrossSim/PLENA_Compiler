"""Packed-HLEN row-op smoke kernel using V_MASK."""

import tvm
from tvm.script import tir as T


def make_row_mask_smoke(*, rows: int = 64, lane_count: int = 4, hlen: int = 16, active_lane: int = 0):
    MLEN = 64
    if lane_count * hlen != MLEN:
        raise ValueError(
            f"lane_count * hlen must equal MLEN ({lane_count} * {hlen} == {MLEN})"
        )
    if rows <= 0 or rows > MLEN:
        raise ValueError(f"rows must be in (0, {MLEN}], got {rows}")
    if not (0 <= active_lane < lane_count):
        raise ValueError(f"active_lane must be in [0, {lane_count}), got {active_lane}")

    mask_val = 1 << active_lane
    PACKED_SHAPE = (1, rows, lane_count, hlen)
    FP_ROW_SHAPE = (rows,)

    @T.prim_func
    def row_mask_smoke():
        Packed_v = T.alloc_buffer(PACKED_SHAPE, "float16", scope="vram")
        Scale = T.alloc_buffer(FP_ROW_SHAPE, "float16", scope="fpram")
        Row_sum = T.alloc_buffer(FP_ROW_SHAPE, "float16", scope="fpram")

        T.evaluate(T.call_extern(
            "handle", "plena.row_mul_fp_mask",
            Packed_v.data, Scale.data, Packed_v.data, mask_val,
        ))
        T.evaluate(T.call_extern(
            "handle", "plena.row_reduce_sum_mask",
            Packed_v.data, Row_sum.data, mask_val,
        ))

    constants = {
        "ROWS": rows, "LANE_COUNT": lane_count, "HLEN": hlen,
        "ACTIVE_LANE": active_lane, "MASK_VAL": mask_val, "MLEN": MLEN,
    }
    return row_mask_smoke, constants


def build_module(
    *, rows: int = 64, lane_count: int = 4, hlen: int = 16, active_lane: int = 0,
) -> tvm.IRModule:
    func, _ = make_row_mask_smoke(
        rows=rows, lane_count=lane_count, hlen=hlen, active_lane=active_lane,
    )
    return tvm.IRModule({"row_mask_smoke": func})
