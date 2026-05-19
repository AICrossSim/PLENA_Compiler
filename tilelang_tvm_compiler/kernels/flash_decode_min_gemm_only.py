"""flash_decode gemm-only debug kernel — with a step dial.

Base: BTMV(Q@K^T) + MV(S@V), no softmax. ``fd_steps`` (0..8) adds the
stripped softmax ops back ONE AT A TIME so the testbench can bisect
which step introduces the sim/golden mismatch.

Each level produces a MATHEMATICALLY WELL-DEFINED output (the testbench
golden mirrors it exactly), so there are no half-computed states to
guess at:

    0  O = (Q@K^T) @ V                          (gemm-only base)
    1  O = (scale * Q@K^T) @ V                   (+ S *= scale)
    2  same O as 1, but reduce_max -> M_CURR is issued (result unused)
    3  same O as 1, but M_RES = exp(M_OLD-M_CURR) is issued (unused)
    4  O = exp(scale*S - M_CURR) @ V              (+ S exp/sub)
    5  same O as 4, but reduce_sum -> P_SUM is issued (unused)
    6  same O as 4, but L_NEW = L_OLD*M_RES+P_SUM is issued (unused)
    7  O = M_RES * (exp(...) @ V)                 (+ O *= M_RES)
    8  O = (M_RES * (exp(...) @ V)) / L_NEW       (== full flash_decode)

Levels 2/3/5/6 add an op whose result does NOT change O — so if the
correctness drops at one of those, that op's hardware (reduce_max /
scalar exp / reduce_sum / scalar L chain) is itself the error source.
Levels 1/4/7/8 genuinely change O and the golden tracks them.

NOTE: this is the single-q-block decode shape (rows of S used = 1).
"""

import math

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes

from ..frontend.gemm_macros import KIND


def make_flash_decode_min_gemm_only(
    *,
    rows: int | None = None,
    hlen: int | None = None,
    head_count: int | None = None,
    num_kv_blocks: int = 2,
    fd_steps: int = 0,
):
    # Hardware sizes default to plena_settings.toml's active mode.
    _hw = _load_sizes()
    MLEN = _hw.mlen
    if hlen is None:
        hlen = _hw.hlen
    if rows is None:
        rows = MLEN
    if rows != MLEN:
        raise ValueError(f"requires rows == MLEN ({MLEN}), got {rows}")
    if MLEN % hlen != 0:
        raise ValueError(f"hlen must divide MLEN ({MLEN}); got hlen={hlen}")
    if not (0 <= fd_steps <= 8):
        raise ValueError(f"fd_steps must be in [0, 8], got {fd_steps}")
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
    scale_val = 1.0 / math.sqrt(hlen)

    @T.prim_func
    def flash_decode_min_gemm_only(
        K_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        V_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
    ):
        with T.Kernel(1, head_count, threads=128) as (_, by):
            Q_cache = T.alloc_shared((head_count, hlen), "float16",
                                      scope="global.vram")
            O_cache = T.alloc_shared((head_count, hlen), "float16",
                                      scope="global.vram")
            Q_sh   = T.alloc_shared((1, hlen), "float16")
            K_sh   = T.alloc_shared((rows, hlen), "float16")
            V_sh   = T.alloc_shared((rows, hlen), "float16")
            S_loc  = T.alloc_fragment((1, MLEN), "float16")
            PV_loc = T.alloc_fragment((1, hlen), "float16")
            O_loc  = T.alloc_fragment((1, hlen), "float16")
            # Online-softmax FPRAM scalars. Allocated for fd_steps >= 2;
            # cheap to always declare, only written when the step needs it.
            M_OLD  = T.alloc_fragment((1,), "float16")
            M_CURR = T.alloc_fragment((1,), "float16")
            M_RES  = T.alloc_fragment((1,), "float16")
            L_OLD  = T.alloc_fragment((1,), "float16")
            L_NEW  = T.alloc_fragment((1,), "float16")
            P_SUM  = T.alloc_fragment((1,), "float16")
            L_INV  = T.alloc_fragment((1,), "float16")

            T.copy(Q_cache[by, 0], Q_sh)

            for col in T.Parallel(hlen):
                O_loc[0, col] = T.float16(0)

            # Init softmax state (needed from step 2 onward).
            if fd_steps >= 2:
                for row in T.serial(1):
                    M_OLD[row] = T.float16(-1.0e4)
                    L_OLD[row] = T.float16(0)

            for kv_block in T.unroll(num_kv_blocks):
                T.copy(
                    K_hbm[0, kv_block * rows : (kv_block + 1) * rows, by, 0:hlen],
                    K_sh,
                )
                T.copy(
                    V_hbm[0, kv_block * rows : (kv_block + 1) * rows, by, 0:hlen],
                    V_sh,
                )

                with T.attr(0, KIND, "btmm"):
                    T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)

                # STEP 1: S *= scale
                if fd_steps >= 1:
                    for row in T.serial(1):
                        for col in T.Parallel(MLEN):
                            S_loc[row, col] = S_loc[row, col] * T.float16(scale_val)

                # STEP 2: reduce_max -> M_CURR (result unused for O at <4).
                if fd_steps >= 2:
                    for row in T.serial(1):
                        M_CURR[row] = M_OLD[row]
                    T.reduce_max(S_loc, M_CURR, dim=1, clear=False)

                # STEP 3: M_RES = exp(M_OLD - M_CURR)  (unused for O at <7).
                if fd_steps >= 3:
                    for row in T.serial(1):
                        M_RES[row] = M_OLD[row] - M_CURR[row]
                        M_RES[row] = T.exp(M_RES[row])

                # STEP 4: S = exp(S - M_CURR)
                if fd_steps >= 4:
                    for row in T.serial(1):
                        for col in T.Parallel(MLEN):
                            S_loc[row, col] = S_loc[row, col] - M_CURR[row]
                        for col in T.Parallel(MLEN):
                            S_loc[row, col] = T.exp(S_loc[row, col])

                # STEP 5: reduce_sum -> P_SUM  (unused for O at <6/8).
                if fd_steps >= 5:
                    for row in T.serial(1):
                        P_SUM[row] = T.float16(0)
                    T.reduce_sum(S_loc, P_SUM, dim=1, clear=False)

                # STEP 6: L_NEW = L_OLD*M_RES + P_SUM  (unused for O at <8).
                if fd_steps >= 6:
                    for row in T.serial(1):
                        L_NEW[row] = L_OLD[row] * M_RES[row]
                        L_NEW[row] = L_NEW[row] + P_SUM[row]

                # STEP 7: O_loc *= M_RES (rescale running output).
                if fd_steps >= 7:
                    for row in T.serial(1):
                        for col in T.Parallel(hlen):
                            O_loc[row, col] = O_loc[row, col] * M_RES[row]

                # Advance online state (only meaningful once it is used).
                if fd_steps >= 6:
                    for row in T.serial(1):
                        M_OLD[row] = M_CURR[row]
                        L_OLD[row] = L_NEW[row]

                T.gemm(S_loc, V_sh, PV_loc)

                for col in T.Parallel(hlen):
                    O_loc[0, col] = O_loc[0, col] + PV_loc[0, col]

            # STEP 8: O = O / L_NEW
            if fd_steps >= 8:
                for row in T.serial(1):
                    L_INV[row] = 1.0 / L_NEW[row]
                    for col in T.Parallel(hlen):
                        O_loc[row, col] = O_loc[row, col] * L_INV[row]

            T.copy(O_loc, O_cache[by, 0])

    lowered = flash_decode_min_gemm_only
    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
        "NUM_KV_BLOCKS": num_kv_blocks,
        "CACHE_NUM_MLEN_ROWS": (head_count * hlen) // MLEN,
        "FD_STEPS": fd_steps,
    }
    return lowered, constants


__all__ = ["make_flash_decode_min_gemm_only"]
