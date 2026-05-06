"""Flash-attention-min kernel — written in tilelang style.

Multi-q-block × multi-kv-block flash attention with online softmax,
head-fused via ``T.Kernel(num_q_blocks, head_count)``.

Tilelang-DSL parts:
  * ``T.Kernel(num_q_blocks, head_count) as (q_block, by)`` — grid axes;
    ``by`` is the logical head axis. The frontend splits it into hardware
    sync domains of width ``MLEN / hlen`` when DMAs / BTMM need fusion.
  * ``T.copy`` for HBM↔VRAM/MRAM transfers.
  * ``T.gemm(..., transpose_B=True)`` under ``T.attr(0, KIND, "btmm")``
    for Q@K^T with head fusion.
  * Per-lane buffers declared as 2D shapes get auto-expanded by the
    ``allocate_group_memory`` pass into 4D BSHD-packed (column-direction
    head packing) or BHSD-stacked (row-direction head stacking).

Direct ``T.call_extern("plena.*")`` parts (no tilelang DSL equivalent
yet):
  * Per-head ``plena.matmul`` for ``P @ V``.
  * ``plena.v_add`` / ``plena.zero_v`` for output accumulation.

The frontend pipeline handles lane-fusion segmentation automatically:
each sync point (DMA / BTMM / vector op) fires once as a multi-lane HW
op outside the per-lane for-by loop; per-lane FP / matmul / row ops run
inside their own for-by loop.

FP slot layout (1 flat FPRAM region starting at FPRAM_USER_BASE; 10
slots, each ``hardware_lane_count*rows`` wide). Users declare each slot as a
1D per-lane fragment ``(rows,)`` and the compiler expands it to
``(hardware_lane_count, rows)`` inside the lane group. The testbench preloads
the read-only slots:
  Scale[h, :]  = 1 / sqrt(d_k)
  M_init[h, :] = -inf surrogate
  L_init[h, :] = 0
"""

import tilelang.language as T

from ..address_alloc import FPRAM_USER_BASE
from ..frontend import compile_func
from ..frontend.gemm_macros import KIND


def make_flash_attention_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int | None = None,
    lane_count: int | None = None,
    active_lane: int = 0,
    num_kv_blocks: int = 2,
    num_q_blocks: int = 2,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(
            f"flash_attention_min requires rows == MLEN ({MLEN}), got {rows}"
        )
    if MLEN % hlen != 0:
        raise ValueError(
            f"hlen must divide MLEN ({MLEN}); got hlen={hlen}"
        )
    hardware_lane_count = MLEN // hlen
    # Backward compatibility for older scripts: `lane_count` used to mean
    # logical head count. New callers should pass `head_count`.
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
    if not (0 <= active_lane < hardware_lane_count):
        raise ValueError(
            f"active_lane out of hardware lane range [0, {hardware_lane_count}): "
            f"{active_lane}"
        )
    if num_kv_blocks < 1:
        raise ValueError(f"num_kv_blocks must be >= 1, got {num_kv_blocks}")
    if num_q_blocks < 1:
        raise ValueError(f"num_q_blocks must be >= 1, got {num_q_blocks}")

    grouped = hlen < MLEN
    kv_seq = num_kv_blocks * rows
    q_seq = num_q_blocks * rows

    fp_state_elems = hardware_lane_count * rows

    @T.prim_func
    def flash_attention_min(
        Q_hbm: T.Tensor((1, q_seq,  head_count, hlen), "float16"),
        K_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        V_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        O_hbm: T.Tensor((1, q_seq,  head_count, hlen), "float16"),
    ):
        with T.Kernel(num_q_blocks, head_count, threads=128) as (q_block, by):
            # Per-lane (rows, hlen) — col-pack expanded to 4D BSHD-packed.
            Q_sh = T.alloc_shared((rows, hlen), "float16")
            K_sh = T.alloc_shared((rows, hlen), "float16")  # gemm RHS → mram
            V_sh = T.alloc_shared((rows, hlen), "float16")  # matmul RHS → mram (via DMA + gemm)
            # Per-lane (rows, hlen) for output / per-head P@V — also col-packed.
            PV_loc = T.alloc_fragment((rows, hlen), "float16")
            O_loc  = T.alloc_fragment((rows, hlen), "float16")
            # BTMM output: per-lane (rows, MLEN), row-stack expanded to 4D BHSD.
            S_loc = T.alloc_fragment((rows, MLEN), "float16")
            # Per-lane FP softmax state. The compiler expands these
            # inside the lane group to (lane_count, rows) in FPRAM.
            M_OLD = T.alloc_fragment((rows,), "float16")
            M_CURR = T.alloc_fragment((rows,), "float16")
            M_RES = T.alloc_fragment((rows,), "float16")
            L_OLD = T.alloc_fragment((rows,), "float16")
            L_NEW = T.alloc_fragment((rows,), "float16")
            P_SUM = T.alloc_fragment((rows,), "float16")
            SCALE = T.alloc_fragment((rows,), "float16")
            L_INV = T.alloc_fragment((rows,), "float16")
            M_INIT = T.alloc_fragment((rows,), "float16")
            L_INIT = T.alloc_fragment((rows,), "float16")

            # Q DMA — sync, fires once per q_block (multi-lane).
            T.copy(Q_hbm[0, q_block * rows, by, 0], Q_sh)

            # Zero running output. The nested ``T.serial(rows) +
            # T.Parallel(hlen)`` pattern is folded by fuse_elementwise
            # into a single whole-buffer plena.zero_v: with lane fusion
            # the two loops together iterate exactly rows*hlen*lane_count
            # = post-expansion-buffer elements, matching the HW op's
            # whole-buffer scope. Source code stays semantically faithful
            # — no "name only row 0 to trick the compiler" hack.
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    O_loc[row, col] = T.float16(0)

            # Reset per-lane FP softmax state for this q tile.
            for row in T.serial(rows):
                M_OLD[row] = M_INIT[row]
                L_OLD[row] = L_INIT[row]

            for kv_block in T.unroll(num_kv_blocks):
                # K, V DMAs — sync, multi-lane.
                T.copy(K_hbm[0, kv_block * rows, by, 0], K_sh)
                T.copy(V_hbm[0, kv_block * rows, by, 0], V_sh)

                # BTMM Q @ K^T → S_loc (head-fused, sync, multi-lane).
                with T.attr(0, KIND, "btmm"):
                    T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)

                # Per-lane online softmax body.
                # S_loc is BHSD (last dim == mlen) → (dim2=head, dim3=row)
                # O_loc is BSHD-packed-narrow → (dim2=row, dim3=lane)
                for row in T.serial(rows):
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = S_loc[row, col] * SCALE[row]
                    M_CURR[row] = M_OLD[row]

                T.reduce_max(S_loc, M_CURR, dim=1, clear=False)

                for row in T.serial(rows):
                    M_RES[row] = M_OLD[row] - M_CURR[row]
                    M_RES[row] = T.exp(M_RES[row])
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = S_loc[row, col] - M_CURR[row]
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = T.exp(S_loc[row, col])
                    P_SUM[row] = L_INIT[row]

                T.reduce_sum(S_loc, P_SUM, dim=1, clear=False)

                for row in T.serial(rows):
                    L_NEW[row] = L_OLD[row] * M_RES[row]
                    L_NEW[row] = L_NEW[row] + P_SUM[row]
                    for col in T.Parallel(hlen):
                        O_loc[row, col] = O_loc[row, col] * M_RES[row]
                    M_OLD[row] = M_CURR[row]
                    L_OLD[row] = L_NEW[row]

                # Per-head P @ V — default kind. S_loc has rows=rows>1
                # so the compiler picks plena.matmul (M_MM); per-lane
                # offsets (LHS=by*MLEN*MLEN row-stacked, RHS / DST=by*hlen
                # col-packed) are auto-injected from each buffer's
                # lane-axis stride. PV_loc is fragment-only and gets
                # marked COL_PACK by the gemm itself (no surrounding
                # DMA / extern to do it).
                T.gemm(S_loc, V_sh, PV_loc)

                # O += PV. The nested ``T.serial(rows) + T.Parallel(hlen)``
                # pattern is folded by fuse_elementwise into a single
                # whole-buffer plena.v_add — semantically faithful (no
                # "name only row 0" hack) and matches the HW op's
                # whole-buffer scope after lane fusion.
                for row in T.serial(rows):
                    for col in T.Parallel(hlen):
                        O_loc[row, col] = O_loc[row, col] + PV_loc[row, col]

            # Final O = O / L_new for this q_block.
            for row in T.serial(rows):
                L_INV[row] = 1.0 / L_NEW[row]
                for col in T.Parallel(hlen):
                    O_loc[row, col] = O_loc[row, col] * L_INV[row]

            # Write O back to HBM at this q_block slot.
            T.copy(O_loc, O_hbm[0, q_block * rows, by, 0])

    # The factory must return a TIR PrimFunc already lowered through
    # the new tilelang frontend, since the CLI's `compile_kernel`
    # consumes plain TIR (post-frontend) directly.
    lowered = compile_func(flash_attention_min)

    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "LANE_COUNT": hardware_lane_count,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
        "ACTIVE_LANE": active_lane,
        "GROUPED": grouped,
        "FPRAM_USER_BASE": FPRAM_USER_BASE,
        "FP_STATE_ELEMS": fp_state_elems,
        # FPRAM scalar-slot addresses are exposed via the compiler's
        # --dump-buffer-addrs JSON (single source of truth — see
        # PIPELINE_ARCHITECTURE.md § 5.6). Don't add ``*_ADDR`` keys
        # back here; they were a hand-rolled mirror of
        # AddressAllocationPass and were the root cause of the
        # flash_decode_min FPRAM bug when they drifted.
        "NUM_KV_BLOCKS": num_kv_blocks,
        "NUM_Q_BLOCKS": num_q_blocks,
    }
    return lowered, constants


__all__ = ["make_flash_attention_min"]
