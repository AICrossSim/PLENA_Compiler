"""Flash-attention decode kernel — single Q token, lane-fused multi-head.

Mirrors `flash_attention_min`'s structure with three key differences:

  1. Q is a **single token** (rows=1). The kernel doesn't loop over
     q_blocks; only over kv_blocks for the online-softmax accumulation.
  2. Q does NOT come from HBM. It lives in a **VRAM "tensor cache"**
     region (``Q_cache``) sized ``(head_count, hlen)``. The testbench
     preloads Q values to FPRAM and a pre-kernel ASM stub copies them
     to the VRAM cache via ``S_MAP_V_FP`` before the kernel proper
     starts. From the kernel's perspective this is just a normal shared
     buffer it reads from. Per-by_o iteration the kernel pulls one
     MLEN-wide chunk via ``T.copy(Q_cache[by, 0], Q_sh[0, 0])``, which
     lowers to a single ``V_ADD_VF`` with f0=0 (vram→vram row copy).
  3. Q @ K^T uses **BTMV** (rows=1 LHS triggers the dispatch in
     `_lower_gemm`). P @ V uses **plena.mv** (per-head M_MV), since
     the row-stacked S_loc layout is incompatible with M_BMV's
     lane-packed input — exactly the same reason flash_attention_min
     uses per-head M_MM for its P @ V.

Supports any ``head_count`` that is a multiple of
``hardware_lane_count`` (single by_o iter when equal; multi-by_o
otherwise). Q_cache holds all heads' Q values laid out head-major
(head 0's hlen elements, then head 1's, etc.); the by-indexed read
naturally selects the right MLEN-wide chunk per by_o.

FP slot layout (1 flat FPRAM region starting at FPRAM_USER_BASE):

    Q_FP_STAGE  (head_count, hlen)  — staging area, preloaded by testbench;
                                       pre-kernel stub copies to Q_cache
    M_OLD       (lane_count, 1)
    M_CURR      (lane_count, 1)
    M_RES       (lane_count, 1)
    L_OLD       (lane_count, 1)
    L_NEW       (lane_count, 1)
    P_SUM       (lane_count, 1)
    SCALE       (lane_count, 1)     — preloaded: 1 / sqrt(d_k)
    L_INV       (lane_count, 1)
    M_INIT      (lane_count, 1)     — preloaded: -inf surrogate
    L_INIT      (lane_count, 1)     — preloaded: 0
    O_FP        (lane_count, hlen)  — final per-head output, drained from VRAM

The kernel does NOT write back to HBM. The output ends up in FPRAM at
``O_FP``; the testbench reads FPRAM directly to compare against golden
(``compare_fpsram_output=True`` in comparison_params).
"""

import math

import tilelang.language as T

from ..address_alloc import FPRAM_USER_BASE
from ..frontend.gemm_macros import KIND


def make_flash_decode_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int | None = None,
    num_kv_blocks: int = 2,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(
            f"flash_decode_min requires rows == MLEN ({MLEN}), got {rows}"
        )
    if MLEN % hlen != 0:
        raise ValueError(f"hlen must divide MLEN ({MLEN}); got hlen={hlen}")
    hardware_lane_count = MLEN // hlen
    if head_count is None:
        head_count = hardware_lane_count
    if head_count % hardware_lane_count != 0:
        raise ValueError(
            f"head_count must be a multiple of hardware_lane_count "
            f"({hardware_lane_count}); got head_count={head_count}"
        )
    if num_kv_blocks < 1:
        raise ValueError(f"num_kv_blocks must be >= 1, got {num_kv_blocks}")

    kv_seq = num_kv_blocks * rows
    # Softmax scale 1/sqrt(d_k). Embedded directly as a FloatImm via
    # ``T.float16(...)`` in the kernel body — the ``hoist_float_constants``
    # pre-pass turns it into a 1-slot global.fpram buffer at compile
    # time, no SCALE alloc / SCALE preload required.
    scale_val = 1.0 / math.sqrt(hlen)

    @T.prim_func
    def flash_decode_min(
        K_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        V_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
    ):
        with T.Kernel(1, head_count, threads=128) as (_, by):
            # Q lives in a VRAM "tensor cache" region — a global tensor
            # populated by the testbench pre-kernel stub via S_MAP_V_FP
            # from FPRAM. Layout is head-major (head_count rows, hlen
            # cols), so by-indexed reads naturally select per-by_o slices
            # of MLEN = lane_count * hlen. Marked global.vram so
            # allocate_group_memory does not try to re-expand its head
            # axis (the head dim is already explicit in the shape).
            Q_cache = T.alloc_shared((head_count, hlen), "float16",
                                      scope="global.vram")
            # Symmetric output cache. Kernel writes O_loc -> O_cache[by, 0]
            # via vram→vram T.copy (V_ADD_VF f0=0). Testbench compares the
            # VRAM region directly — no FPRAM round-trip needed. Same
            # global.vram rationale as Q_cache.
            O_cache = T.alloc_shared((head_count, hlen), "float16",
                                      scope="global.vram")
            # VRAM staging so BTMV can read Q from VRAM.
            # 2D rows=1 → col-packed to (1, 1, lane_count, hlen).
            Q_sh   = T.alloc_shared((1, hlen), "float16")
            # MRAM tiles for K and V (gemm RHS).
            K_sh   = T.alloc_shared((rows, hlen), "float16")
            V_sh   = T.alloc_shared((rows, hlen), "float16")
            # BTMV output: 2D rows=1 → row-stacked to (1, lane_count, 1, MLEN).
            S_loc  = T.alloc_fragment((1, MLEN), "float16")
            # P @ V partial output and running accumulator: 2D rows=1.
            PV_loc = T.alloc_fragment((1, hlen), "float16")
            O_loc  = T.alloc_fragment((1, hlen), "float16")
            # Online softmax state: rank-1 → lane-stacked (lane_count, 1).
            M_OLD  = T.alloc_fragment((1,), "float16")
            M_CURR = T.alloc_fragment((1,), "float16")
            M_RES  = T.alloc_fragment((1,), "float16")
            L_OLD  = T.alloc_fragment((1,), "float16")
            L_NEW  = T.alloc_fragment((1,), "float16")
            P_SUM  = T.alloc_fragment((1,), "float16")
            L_INV  = T.alloc_fragment((1,), "float16")
            # SCALE / M_INIT / L_INIT are no longer declared buffers —
            # the kernel body embeds the literals directly as
            # ``T.float16(...)`` and the ``hoist_float_constants``
            # pre-pass synthesises an equivalent ``global.fpram``
            # 1-slot buffer per unique constant at compile time.
            # ``test_helper`` auto-preloads the values from the
            # buffer-addrs dump.

            # VRAM cache → VRAM staging: pull this by_o's MLEN-wide chunk
            # of Q into Q_sh. Lowers to one V_ADD_VF (f0=0) row copy.
            # ``by`` after split_lane_groups is by_o*lane_count + by_i; sync
            # wrap substitutes by_i -> 0, so the source offset becomes
            # by_o*lane_count*hlen = by_o*MLEN — exactly the per-by_o chunk.
            # NOTE: dst is the whole Q_sh buffer (NOT Q_sh[0, 0]) so tilelang's
            # copy_op doesn't degenerate to a scalar BufferStore.
            T.copy(Q_cache[by, 0], Q_sh)

            # Zero output accumulator. T.Parallel + constant fill is
            # picked up by fuse_elementwise → plena.zero_v (multi-lane,
            # sync) — kernel never sees the plena op directly.
            for col in T.Parallel(hlen):
                O_loc[0, col] = T.float16(0)

            # Init online softmax state from -inf / 0 literals; the
            # pre-pass hoists -1e4 into a shared global.fpram slot.
            for row in T.serial(1):
                M_OLD[row] = T.float16(-1.0e4)
                L_OLD[row] = T.float16(0)

            for kv_block in T.unroll(num_kv_blocks):
                # K, V DMAs — sync, multi-lane. Explicit slice form so
                # mid_ir's ranged_slice inference produces clean
                # (extent=rows, extent=hlen) tile shapes.
                T.copy(
                    K_hbm[0, kv_block * rows : (kv_block + 1) * rows, by, 0:hlen],
                    K_sh,
                )
                T.copy(
                    V_hbm[0, kv_block * rows : (kv_block + 1) * rows, by, 0:hlen],
                    V_sh,
                )

                # Q @ K^T → BTMV (rows=1 LHS auto-routes to plena.btmv).
                with T.attr(0, KIND, "btmm"):
                    T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)

                # Scale + grab current max baseline.
                for row in T.serial(1):
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = S_loc[row, col] * T.float16(scale_val)
                    M_CURR[row] = M_OLD[row]

                T.reduce_max(S_loc, M_CURR, dim=1, clear=False)

                for row in T.serial(1):
                    M_RES[row] = M_OLD[row] - M_CURR[row]
                    M_RES[row] = T.exp(M_RES[row])
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = S_loc[row, col] - M_CURR[row]
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = T.exp(S_loc[row, col])
                    P_SUM[row] = T.float16(0)

                T.reduce_sum(S_loc, P_SUM, dim=1, clear=False)

                for row in T.serial(1):
                    L_NEW[row] = L_OLD[row] * M_RES[row]
                    L_NEW[row] = L_NEW[row] + P_SUM[row]
                    for col in T.Parallel(hlen):
                        O_loc[row, col] = O_loc[row, col] * M_RES[row]
                    M_OLD[row] = M_CURR[row]
                    L_OLD[row] = L_NEW[row]

                # P @ V — default kind. Compiler picks plena.mv (M_MV)
                # because S_loc has rows=1; per-head lane offset
                # (S_loc row-stacked at by*MLEN, V_sh / PV_loc
                # col-packed at by*hlen) is auto-injected from each
                # buffer's lane-axis stride.
                T.gemm(S_loc, V_sh, PV_loc)

                # O += PV. T.Parallel + add is picked up by
                # fuse_elementwise → plena.v_add (multi-lane, sync).
                for col in T.Parallel(hlen):
                    O_loc[0, col] = O_loc[0, col] + PV_loc[0, col]

            # Final O = O / L_new.
            for row in T.serial(1):
                L_INV[row] = 1.0 / L_NEW[row]
                for col in T.Parallel(hlen):
                    O_loc[row, col] = O_loc[row, col] * L_INV[row]

            # Write this by_o's MLEN-wide chunk of O into O_cache[by, 0].
            # vram→vram copy (V_ADD_VF f0=0); after lane fusion sync wrap,
            # by_i drops to 0 so the dst offset becomes by_o*MLEN — exactly
            # the per-by_o slice in the head-major O_cache layout.
            # NOTE: src is the whole O_loc buffer (NOT O_loc[0, 0]) so
            # tilelang's copy_op doesn't degenerate to a scalar BufferStore.
            T.copy(O_loc, O_cache[by, 0])

    # Return the raw PrimFunc — ``compile_kernel`` runs stmt prep + the
    # mid_ir pipeline itself, so factories don't pre-lower anymore.
    lowered = flash_decode_min

    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
        "FPRAM_USER_BASE": FPRAM_USER_BASE,
        "NUM_KV_BLOCKS": num_kv_blocks,
        "CACHE_NUM_MLEN_ROWS": (head_count * hlen) // MLEN,
        # Buffer addresses are exposed via the compiler's
        # --dump-buffer-addrs JSON (single source of truth — see
        # PIPELINE_ARCHITECTURE.md § 5.6). The previous ``*_ADDR``
        # entries here were a hand-rolled mirror of
        # AddressAllocationPass / `_slot_addresses` and were the root
        # cause of the flash_decode_min FPRAM bug when they drifted.
    }
    return lowered, constants
