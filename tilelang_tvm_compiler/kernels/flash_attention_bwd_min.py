"""Flash-attention BACKWARD kernel.

A flash-style FlashAttention-2 backward. It carries the five head-fused
MLEN x MLEN x MLEN GEMMs that dominate the FLOP/energy bill, the dQ
transpose-A GEMM (``transpose_A=True`` -> M_TMM_A), and the row-level
softmax-gradient re-derivation, which now reads the real forward statistics
L (logsumexp) and D (rowsum(dO.O)) from VRAM tensor caches rather than
zeroing them.

Structure follows tilelang's official ``example_mha_bwd_bshd.py``: the
outer grid axis is the KV block, the inner serial loop is the query block,
and the scores are computed in *transposed* form so that dV and dK become
plain (non-transposed) GEMMs — only dQ needs a transpose, which is taken on
the VRAM/A operand via ``transpose_a=True`` (lowers to M_TMM_A).

Per (kv_block fixed, q_block) tile:

    sT  = K @ Q^T * scale         # transposed scores  [kv, q]   (btmm, Q@K^T form)
    pT  = exp(sT - L)             # transposed probs               (cheap)
    dpT = V @ dO^T                # transposed dP       [kv, q]   (btmm_mm, transpose_B)
    dv += pT @ dO                 # value grad          [kv, d]   (matmul, no transpose)
    dsT = pT . (dpT - D)          # transposed dS                  (cheap)
    dk += dsT @ Q                 # key grad            [kv, d]   (matmul, no transpose)
    dq  = dsT^T @ K  -> dQ        # query grad, scattered to HBM  (matmul, transpose_a -> M_TMM_A)

Head fusion / grid mirror flash_attention_min: T.Kernel(num_kv_blocks,
head_count); the head axis (``by``) is the lane axis the frontend
cluster-fuses. All HBM tensors are BSHD (1, seq, head, hlen).
"""

import math

import tilelang.language as T

from ..frontend.gemm_macros import KIND, BTMM_MM
from ..plena_settings import load_sizes as _load_sizes


def make_flash_attention_bwd_min(
    *,
    rows: int | None = None,
    hlen: int | None = None,
    head_count: int | None = None,
    num_kv_blocks: int = 1,
    num_q_blocks: int = 2,
):
    _hw = _load_sizes()
    MLEN = _hw.mlen
    if hlen is None:
        hlen = _hw.hlen
    if rows is None:
        rows = MLEN
    if rows != MLEN:
        raise ValueError(
            f"flash_attention_bwd_min requires rows == MLEN ({MLEN}), got {rows}"
        )
    if MLEN % hlen != 0:
        raise ValueError(f"hlen must divide MLEN ({MLEN}); got hlen={hlen}")
    hardware_lane_count = MLEN // hlen
    if head_count is None:
        head_count = hardware_lane_count
    if head_count % hardware_lane_count != 0:
        raise ValueError(
            f"head_count must be a multiple of MLEN/hlen={hardware_lane_count}; "
            f"got {head_count}"
        )
    if num_kv_blocks < 1 or num_q_blocks < 1:
        raise ValueError("num_kv_blocks / num_q_blocks must be >= 1")

    kv_seq = num_kv_blocks * rows
    q_seq = num_q_blocks * rows
    scale_val = 1.0 / math.sqrt(hlen)

    @T.prim_func
    def flash_attention_bwd_min(
        Q_hbm:  T.Tensor((1, q_seq,  head_count, hlen), "float16"),
        K_hbm:  T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        V_hbm:  T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        dO_hbm: T.Tensor((1, q_seq,  head_count, hlen), "float16"),  # output grad
        # L / D (per-(head,q-token) logsumexp and rowsum(dO.O)) are NOT read
        # from HBM. Like flash_decode_min's Q_cache they live in a VRAM
        # "tensor cache" (L_cache / D_cache, global.vram) populated by the
        # testbench from the forward's saved statistics; the kernel reads the
        # real values from there. See body.
        dQ_hbm: T.Tensor((1, q_seq,  head_count, hlen), "float16"),
        dK_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        dV_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
    ):
        # Outer grid over KV blocks: one program accumulates dK/dV for its
        # kv block over all q blocks, and scatters dQ contributions to HBM.
        with T.Kernel(num_kv_blocks, head_count, threads=128) as (kv_block, by):
            head = by
            kv0 = kv_block * rows

            # --- per-(kv) buffers ---
            # K plays two roles with different scopes, so it gets two copies:
            #   K_sh : LHS/A of #1 (K@Q^T)        -> VRAM
            #   K_mr : RHS/B of #7 (dS@K)          -> MRAM
            # (V only ever acts as an A operand, Q/dO only ever as B operands,
            #  so they need just one buffer each.)
            K_sh   = T.alloc_shared((rows, hlen), "float16")  # A of #1   -> vram
            K_mr   = T.alloc_shared((rows, hlen), "float16")  # B of #7   -> mram
            V_sh   = T.alloc_shared((rows, hlen), "float16")  # A of #3   -> vram
            dK_loc = T.alloc_fragment((rows, hlen), "float16")
            dV_loc = T.alloc_fragment((rows, hlen), "float16")

            # --- per-(q) tile buffers ---
            Q_sh   = T.alloc_shared((rows, hlen), "float16")   # gemm RHS -> mram
            dO_sh  = T.alloc_shared((rows, hlen), "float16")   # gemm RHS -> mram
            dQ_sh  = T.alloc_shared((rows, hlen), "float16")

            sT_loc  = T.alloc_fragment((rows, MLEN), "float16")  # K@Q^T  (btmm out)
            pT_loc  = T.alloc_fragment((rows, MLEN), "float16")  # exp(sT-L)
            dpT_loc = T.alloc_fragment((rows, MLEN), "float16")  # V@dO^T
            dsT_loc = T.alloc_fragment((rows, MLEN), "float16")  # pT.(dpT-D) = dS^T [k,q]
            dvp     = T.alloc_fragment((rows, hlen), "float16")  # pT@dO   partial
            dkp     = T.alloc_fragment((rows, hlen), "float16")  # dsT@Q   partial
            dQp     = T.alloc_fragment((rows, hlen), "float16")  # dS@K    partial

            L_loc   = T.alloc_fragment((rows,), "float16")       # per-q-row logsumexp
            D_loc   = T.alloc_fragment((rows,), "float16")       # per-q-row delta

            # L / D "tensor caches" — VRAM global tensors holding the
            # forward's saved per-(head, q-token) statistics, laid out
            # head-major (head_count rows, q_seq cols). Mirrors
            # flash_decode_min's Q_cache: the testbench preloads them and
            # the kernel reads its (head, q0:q0+rows) slice per q block.
            # global.vram so allocate_group_memory does not re-expand the
            # head axis (already explicit in the shape).
            L_cache = T.alloc_shared((head_count, q_seq), "float16",
                                     scope="global.vram")
            D_cache = T.alloc_shared((head_count, q_seq), "float16",
                                     scope="global.vram")

            T.copy(K_hbm[0, kv0:kv0 + rows, head, 0:hlen], K_sh)
            T.copy(K_hbm[0, kv0:kv0 + rows, head, 0:hlen], K_mr)  # MRAM copy of K for #7
            T.copy(V_hbm[0, kv0:kv0 + rows, head, 0:hlen], V_sh)

            # Zero dK / dV accumulators. One buffer per loop so each folds to
            # a single whole-tile v_zero (writing two buffers in one
            # for-col body breaks the fold and explodes into per-element ops).
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    dV_loc[row, col] = T.float16(0)
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    dK_loc[row, col] = T.float16(0)

            for q_block in T.serial(num_q_blocks):
                q0 = q_block * rows
                T.copy(Q_hbm[0, q0:q0 + rows, head, 0:hlen], Q_sh)
                T.copy(dO_hbm[0, q0:q0 + rows, head, 0:hlen], dO_sh)
                # L (logsumexp) and D (rowsum(dO.O)): pull this q block's
                # rows-long slice of the real forward statistics out of the
                # VRAM caches (head-major, so [head, q0:q0+rows]). vram→vram
                # row copies, exactly like flash_decode_min's Q_cache read.
                T.copy(L_cache[head, q0:q0 + rows], L_loc)
                T.copy(D_cache[head, q0:q0 + rows], D_loc)

                # 1) sT = K @ Q^T * scale   (transposed scores; head-fused btmm).
                with T.attr(0, KIND, "btmm"):
                    T.gemm(K_sh, Q_sh, sT_loc, transpose_B=True)

                # 2) pT = exp(sT * scale - L)   (rebuild transposed probs; minimal).
                for row in T.serial(rows):
                    for col in T.Parallel(MLEN):
                        pT_loc[row, col] = sT_loc[row, col] * T.float16(scale_val)
                    for col in T.Parallel(MLEN):
                        pT_loc[row, col] = pT_loc[row, col] - L_loc[row]
                    for col in T.Parallel(MLEN):
                        pT_loc[row, col] = T.exp(pT_loc[row, col])

                # 3) dpT = V @ dO^T   (score-class: output [MLEN,MLEN],
                #    contracts over d=hlen, same shape as S^T -> btmm, NOT btmm_mm).
                with T.attr(0, KIND, "btmm"):
                    T.gemm(V_sh, dO_sh, dpT_loc, transpose_B=True)

                # 4) dv += pT @ dO    (NO transpose — pT is already P^T).
                T.gemm(pT_loc, dO_sh, dvp)
                for row in T.serial(rows):
                    for col in T.Parallel(hlen):
                        dV_loc[row, col] = dV_loc[row, col] + dvp[row, col]

                # 5) dsT = pT . (dpT - D)   (cheap elementwise + broadcast).
                for row in T.serial(rows):
                    for col in T.Parallel(MLEN):
                        dpT_loc[row, col] = dpT_loc[row, col] - D_loc[row]
                    for col in T.Parallel(MLEN):
                        dsT_loc[row, col] = pT_loc[row, col] * dpT_loc[row, col]

                # 6) dk += dsT @ Q    (NO transpose — dsT is already dS^T).
                T.gemm(dsT_loc, Q_sh, dkp)
                for row in T.serial(rows):
                    for col in T.Parallel(hlen):
                        dK_loc[row, col] = dK_loc[row, col] + dkp[row, col]

                # 7) dQ = dS @ K   (the ONLY transpose-A point in the kernel).
                #    We hold the score gradient in transposed form dsT = dS^T
                #    = [k, q]; dQ needs the non-transposed dS = [q, k] as the
                #    VRAM (A) operand. We get it with a real transpose-A GEMM:
                #    ``transpose_a=True`` makes the matrix core transpose the
                #    VRAM/activation (A) tile on the fly as it streams into the
                #    array — the symmetric counterpart of the MRAM-side M_TMM
                #    used for transpose_B. The lowering emits M_TMM_A, so the
                #    GEMM is both numerically exact (dQ = (dsT)^T @ K = dS @ K)
                #    and carries the correct MLEN×MLEN×MLEN matmul cost.
                T.gemm(dsT_loc, K_mr, dQp, transpose_A=True)   # dQ = (dS^T)^T @ K = dS @ K
                T.copy(dQ_hbm[0, q0:q0 + rows, head, 0:hlen], dQ_sh)
                for row in T.serial(rows):
                    for col in T.Parallel(hlen):
                        dQ_sh[row, col] = dQ_sh[row, col] + dQp[row, col]
                T.copy(dQ_sh, dQ_hbm[0, q0:q0 + rows, head, 0:hlen])

            T.copy(dK_loc, dK_hbm[0, kv0:kv0 + rows, head, 0:hlen])
            T.copy(dV_loc, dV_hbm[0, kv0:kv0 + rows, head, 0:hlen])

    lowered = flash_attention_bwd_min

    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "LANE_COUNT": hardware_lane_count,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
        "NUM_KV_BLOCKS": num_kv_blocks,
        "NUM_Q_BLOCKS": num_q_blocks,
    }
    return lowered, constants


__all__ = ["make_flash_attention_bwd_min"]
