"""Flash-attention-GQA-min kernel — grouped-query attention.

Variant of ``flash_attention_min`` where the KV head count is smaller
than the Q head count: ``kv_head_count = head_count // group_size``.
Each Q head ``by`` shares a KV head with ``group_size - 1`` siblings.

KV-head index mapping
---------------------
Q head ``by`` reads KV head ``by % kv_head_count``.

We use ``%`` (modulo), NOT ``//`` (floordiv), on purpose:

  * The PLENA ISA implements a modulo op but has **no integer-divide
    op**, so ``by // group_size`` cannot be lowered to hardware.
  * ``%`` gives an *interleaved* group layout: Q heads
    ``h, h+kv_head_count, h+2*kv_head_count, ...`` all map to KV head
    ``h``. (A ``//``-based layout would be contiguous groups
    ``0..G-1 -> 0`` — that layout is not expressible on this HW.)

So callers must lay Q heads out interleaved-by-KV-head in HBM.

KV tensors are declared with ``kv_head_count`` heads — this is the real
GQA memory saving, K/V genuinely store fewer heads. The grid still
iterates ``head_count`` Q heads; the ``by % kv_head_count`` index picks
the shared KV head.

Everything else (online softmax, BTMM Q@K^T, per-head P@V, lane fusion)
is identical to ``flash_attention_min`` — see that file's docstring.
The frontend's ``_subst_lane_var`` recurses through arbitrary index
``op`` nodes, so ``by % kv_head_count`` survives the lane-axis split
(``by`` -> ``phase + number*cluster_count``) with no pass changes.
"""

import math

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes

from ..address_alloc import FPRAM_USER_BASE
from ..frontend.gemm_macros import KIND


def make_flash_attention_gqa_min(
    *,
    rows: int | None = None,
    hlen: int | None = None,
    head_count: int | None = None,
    group_size: int = 2,
    lane_count: int | None = None,
    active_lane: int = 0,
    num_kv_blocks: int = 1,
    num_q_blocks: int = 2,
    o_head_count: int | None = None,
    o_head_offset: int = 0,
):
    """Grouped-query flash attention with online softmax.

    ``group_size`` Q heads share one KV head; ``kv_head_count =
    head_count // group_size``. ``group_size == 1`` degenerates to plain
    MHA (identical to ``flash_attention_min``).

    ``o_head_count`` / ``o_head_offset`` behave exactly as in
    ``flash_attention_min`` — they let the kernel drop its output into a
    head-slice of a wider output tensor.
    """
    # Hardware sizes default to plena_settings.toml's active mode.
    _hw = _load_sizes()
    MLEN = _hw.mlen
    if hlen is None:
        hlen = _hw.hlen
    if rows is None:
        rows = MLEN
    if rows != MLEN:
        raise ValueError(
            f"flash_attention_gqa_min requires rows == MLEN ({MLEN}), got {rows}"
        )
    if MLEN % hlen != 0:
        raise ValueError(
            f"hlen must divide MLEN ({MLEN}); got hlen={hlen}"
        )
    hardware_lane_count = MLEN // hlen
    if head_count is None:
        head_count = lane_count if lane_count is not None else hardware_lane_count
    elif lane_count is not None and lane_count != head_count:
        raise ValueError(
            f"head_count and legacy lane_count disagree: {head_count} vs {lane_count}"
        )
    if head_count < 1:
        raise ValueError(f"head_count must be >= 1, got {head_count}")
    if head_count % hardware_lane_count != 0:
        raise ValueError(
            f"head_count must be a multiple of hardware lane width "
            f"MLEN/hlen={hardware_lane_count}; got {head_count}"
        )
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if head_count % group_size != 0:
        raise ValueError(
            f"head_count ({head_count}) must be a multiple of group_size "
            f"({group_size})"
        )
    kv_head_count = head_count // group_size
    if not (0 <= active_lane < hardware_lane_count):
        raise ValueError(
            f"active_lane out of hardware lane range [0, {hardware_lane_count}): "
            f"{active_lane}"
        )
    if num_kv_blocks < 1:
        raise ValueError(f"num_kv_blocks must be >= 1, got {num_kv_blocks}")
    if num_q_blocks < 1:
        raise ValueError(f"num_q_blocks must be >= 1, got {num_q_blocks}")

    if o_head_count is None:
        o_head_count = head_count
    if o_head_count < head_count:
        raise ValueError(
            f"o_head_count ({o_head_count}) must be >= head_count "
            f"({head_count})"
        )
    if not (0 <= o_head_offset <= o_head_count - head_count):
        raise ValueError(
            f"o_head_offset ({o_head_offset}) + head_count ({head_count}) "
            f"must fit within o_head_count ({o_head_count})"
        )

    grouped = hlen < MLEN
    kv_seq = num_kv_blocks * rows
    q_seq = num_q_blocks * rows

    fp_state_elems = hardware_lane_count * rows
    scale_val = 1.0 / math.sqrt(hlen)

    @T.prim_func
    def flash_attention_gqa_min(
        Q_hbm: T.Tensor((1, q_seq,  head_count,    hlen), "float16"),
        K_hbm: T.Tensor((1, kv_seq, kv_head_count, hlen), "float16"),
        V_hbm: T.Tensor((1, kv_seq, kv_head_count, hlen), "float16"),
        O_hbm: T.Tensor((1, q_seq,  o_head_count,  hlen), "float16"),
    ):
        with T.Kernel(num_q_blocks, head_count, threads=128) as (q_block, by):
            # KV head shared across the group. ``%`` (modulo) — the ISA
            # has no integer divide, so an interleaved group layout is
            # the only HW-expressible GQA mapping.
            kv_by = by % kv_head_count

            # Per-lane (rows, hlen) — col-pack expanded to 4D BSHD-packed.
            Q_sh = T.alloc_shared((rows, hlen), "float16")
            K_sh = T.alloc_shared((rows, hlen), "float16")  # gemm RHS → mram
            V_sh = T.alloc_shared((rows, hlen), "float16")  # matmul RHS → mram (via DMA + gemm)
            PV_loc = T.alloc_fragment((rows, hlen), "float16")
            O_loc  = T.alloc_fragment((rows, hlen), "float16")
            # BTMM output: per-lane (rows, MLEN), row-stack expanded to 4D BHSD.
            S_loc = T.alloc_fragment((rows, MLEN), "float16")
            # Per-lane FP softmax state — expanded to (lane_count, rows).
            M_OLD = T.alloc_fragment((rows,), "float16")
            M_CURR = T.alloc_fragment((rows,), "float16")
            M_RES = T.alloc_fragment((rows,), "float16")
            L_OLD = T.alloc_fragment((rows,), "float16")
            L_NEW = T.alloc_fragment((rows,), "float16")
            P_SUM = T.alloc_fragment((rows,), "float16")
            L_INV = T.alloc_fragment((rows,), "float16")

            # Q DMA — sync, fires once per q_block (multi-lane). Q is
            # indexed by the full Q head ``by``.
            T.copy(
                Q_hbm[0, q_block * rows : (q_block + 1) * rows, by, 0:hlen],
                Q_sh,
            )

            # Zero running output.
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    O_loc[row, col] = T.float16(0)

            # Reset per-lane FP softmax state for this q tile.
            for row in T.serial(rows):
                M_OLD[row] = T.float16(-1.0e4)
                L_OLD[row] = T.float16(0)

            for kv_block in T.serial(num_kv_blocks):
                # K, V DMAs — sync, multi-lane. Indexed by the SHARED
                # KV head ``kv_by`` (= by % kv_head_count).
                T.copy(
                    K_hbm[0, kv_block * rows : (kv_block + 1) * rows, kv_by, 0:hlen],
                    K_sh,
                )
                T.copy(
                    V_hbm[0, kv_block * rows : (kv_block + 1) * rows, kv_by, 0:hlen],
                    V_sh,
                )

                # BTMM Q @ K^T → S_loc.
                with T.attr(0, KIND, "btmm"):
                    T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)

                # Scale S_loc by 1/sqrt(d_k) per row.
                for row in T.serial(rows):
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = S_loc[row, col] * T.float16(scale_val)
                    M_CURR[row] = M_OLD[row]

                # M_CURR = max(M_OLD, rowmax(S_loc)).
                T.reduce_max(S_loc, M_CURR, dim=1, clear=False)

                for row in T.serial(rows):
                    M_RES[row] = M_OLD[row] - M_CURR[row]
                    M_RES[row] = T.exp(M_RES[row])
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = S_loc[row, col] - M_CURR[row]
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = T.exp(S_loc[row, col])
                    P_SUM[row] = T.float16(0)

                # P_SUM = rowsum(exp(S - M_CURR)).
                T.reduce_sum(S_loc, P_SUM, dim=1, clear=False)

                for row in T.serial(rows):
                    L_NEW[row] = L_OLD[row] * M_RES[row]
                    L_NEW[row] = L_NEW[row] + P_SUM[row]
                    for col in T.Parallel(hlen):
                        O_loc[row, col] = O_loc[row, col] * M_RES[row]
                    M_OLD[row] = M_CURR[row]
                    L_OLD[row] = L_NEW[row]

                # Per-head P @ V → PV_loc, then O += PV_loc.
                T.gemm(S_loc, V_sh, PV_loc)

                for row in T.serial(rows):
                    for col in T.Parallel(hlen):
                        O_loc[row, col] = O_loc[row, col] + PV_loc[row, col]

            # Final O = O / L_new for this q_block.
            for row in T.serial(rows):
                L_INV[row] = 1.0 / L_NEW[row]
                for col in T.Parallel(hlen):
                    O_loc[row, col] = O_loc[row, col] * L_INV[row]

            # Write O back to HBM at this q_block slot. O is indexed by
            # the full Q head ``by`` (+ o_head_offset) — every Q head
            # produces its own output, only K/V are shared.
            T.copy(
                O_loc,
                O_hbm[0, q_block * rows : (q_block + 1) * rows,
                      by + o_head_offset, 0:hlen],
            )

    lowered = flash_attention_gqa_min

    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "KV_HEAD_COUNT": kv_head_count,
        "GROUP_SIZE": group_size,
        "LANE_COUNT": hardware_lane_count,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
        "ACTIVE_LANE": active_lane,
        "GROUPED": grouped,
        "FPRAM_USER_BASE": FPRAM_USER_BASE,
        "FP_STATE_ELEMS": fp_state_elems,
        "NUM_KV_BLOCKS": num_kv_blocks,
        "NUM_Q_BLOCKS": num_q_blocks,
    }
    return lowered, constants


__all__ = ["make_flash_attention_gqa_min"]
