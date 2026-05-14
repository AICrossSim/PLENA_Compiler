"""Linear-min kernel — multi-tile GEMM (+ optional bias).

PLENA-flavored counterpart to GPU-tilelang's ``_build_gemm_kernel`` /
``_build_gemm_bias_kernel``. Shape constraints: M, N, K must each be a
multiple of MLEN (= 64) — PLENA's matmul tile granularity. The natural
mapping mirrors the GPU version's (block_M, block_N, block_K) where each
block equals MLEN:

    block_M = block_N = block_K = MLEN = 64
    m_blocks = M / MLEN
    n_blocks = N / MLEN
    k_blocks = K / MLEN

What it computes (output per (m_block, n_block) tile):

    C[m_block, n_block] = sum_{k_block} A[m_block, k_block]
                                       @ B[n_block, k_block]^T
                        + bias[m_block, n_block]    (optional)

B is K-inner ``(N, K)`` to match ``nn.Linear.weight``; the gemm uses
``transpose_B=True``.

Lowering path (kind="overwrite" default, NOT btmm):
  PLENA's BTMM is the head-fused Q@K^T path with packed-head layout
  (btmm_hlen < MLEN). Plain MLEN×MLEN×MLEN matmul without head packing
  lowers through ``plena.matmul`` (``M_MM_WO`` drain) — the same path
  flash_attention's second gemm (P @ V) takes. So no ``T.attr(KIND,
  "btmm")`` wrap here.

  ``transpose_B=True`` is honoured by the lowering: emit_matmul_general
  swaps ``M_MM`` for ``M_TMM`` (which transposes the (mlen, mlen) MRAM
  tile inside the systolic array), so the kernel takes ``B`` in
  ``(N, K)`` row-major layout — the standard nn.Linear weight format,
  no host-side transpose required.

K accumulation:
  ``kind="add"`` is reserved but not implemented yet (see
  gemm_macros.py docstring). For now do the documented workaround
  manually:

      T.gemm(A_blk, B_blk, SCR_loc)        # overwrite into scratch
      for r, c: C_loc[r, c] += SCR_loc[r, c]    # fuse_elementwise → v_add

  Before the K-loop starts, ``C_loc`` is zeroed by an inline parallel
  loop so the first k_block's add behaves like a clear+write.

Bias (optional):
  ``bias`` broadcasts along N (cols) in nn.Linear, which is the wrong
  axis for ``row_*_fp_at``'s "one FP scalar per row" semantics. So the
  testbench host-broadcasts ``bias[N]`` → ``(M, N)`` and the kernel
  consumes it as a full-shape VRAM tile (one tile_add per (m, n) block).
  Same trick rmsnorm_min uses for its ``(hlen,)`` scale weight.
"""

import tilelang.language as T


def make_linear_min(
    *,
    m_blocks: int = 1,
    n_blocks: int = 1,
    k_blocks: int = 1,
    with_bias: bool = False,
):
    MLEN = 64
    if m_blocks < 1 or n_blocks < 1 or k_blocks < 1:
        raise ValueError(
            f"m_blocks/n_blocks/k_blocks must be >= 1; "
            f"got m={m_blocks}, n={n_blocks}, k={k_blocks}"
        )
    M = m_blocks * MLEN
    N = n_blocks * MLEN
    K = k_blocks * MLEN

    # PLENA's DMA-slice lowering expects HBM tensors to carry the full
    # 4D BSHD shape (batch, seq, head, hlen). Linear has no real head
    # axis, so we degenerate head=1 and lay (M, K) / (N, K) / (M, N)
    # along the (seq, hlen) pair: A_hbm[1, M, 1, K], B_hbm[1, N, 1, K],
    # C_hbm[1, M, 1, N], BIAS_hbm[1, M, 1, N].
    if with_bias:
        @T.prim_func
        def linear_min(
            A_hbm:    T.Tensor((1, M, 1, K), "float16"),
            B_hbm:    T.Tensor((1, N, 1, K), "float16"),
            BIAS_hbm: T.Tensor((1, M, 1, N), "float16"),
            C_hbm:    T.Tensor((1, M, 1, N), "float16"),
        ):
            # Grid: one program per (n_block, m_block) tile — same axis
            # order tilelang_kernels/linear.py uses (bx along N, by along
            # M) so cache-line / coalescing intuition carries over.
            with T.Kernel(n_blocks, m_blocks, threads=128) as (bx, by):
                A_sh    = T.alloc_shared((MLEN, MLEN), "float16")
                B_sh    = T.alloc_shared((MLEN, MLEN), "float16")
                BIAS_sh = T.alloc_shared((MLEN, MLEN), "float16")
                C_sh    = T.alloc_shared((MLEN, MLEN), "float16")

                C_loc   = T.alloc_fragment((MLEN, MLEN), "float16")
                SCR_loc = T.alloc_fragment((MLEN, MLEN), "float16")

                # Zero C_loc so the first K iteration's add behaves as
                # clear+write.
                for row in T.serial(MLEN):
                    for col in T.Parallel(MLEN):
                        C_loc[row, col] = T.float16(0)

                for k_block in T.serial(k_blocks):
                    T.copy(
                        A_hbm[0,
                              by * MLEN : (by + 1) * MLEN,
                              0,
                              k_block * MLEN : (k_block + 1) * MLEN],
                        A_sh,
                    )
                    # B is (N, K) row-major — same convention as
                    # nn.Linear.weight. The lowering issues M_TMM when
                    # transpose_B is set, which transposes the (mlen,
                    # mlen) MRAM tile on the fly inside the systolic
                    # array. The slice walks N along the seq axis and K
                    # along the hlen axis.
                    T.copy(
                        B_hbm[0,
                              bx * MLEN : (bx + 1) * MLEN,
                              0,
                              k_block * MLEN : (k_block + 1) * MLEN],
                        B_sh,
                    )

                    T.gemm(A_sh, B_sh, SCR_loc, transpose_B=True)

                    for row in T.serial(MLEN):
                        for col in T.Parallel(MLEN):
                            C_loc[row, col] = C_loc[row, col] + SCR_loc[row, col]

                T.copy(
                    BIAS_hbm[0,
                             by * MLEN : (by + 1) * MLEN,
                             0,
                             bx * MLEN : (bx + 1) * MLEN],
                    BIAS_sh,
                )
                for row in T.serial(MLEN):
                    for col in T.Parallel(MLEN):
                        C_loc[row, col] = C_loc[row, col] + BIAS_sh[row, col]

                T.copy(C_loc, C_sh)
                T.copy(
                    C_sh,
                    C_hbm[0,
                          by * MLEN : (by + 1) * MLEN,
                          0,
                          bx * MLEN : (bx + 1) * MLEN],
                )
    else:
        @T.prim_func
        def linear_min(
            A_hbm: T.Tensor((1, M, 1, K), "float16"),
            B_hbm: T.Tensor((1, N, 1, K), "float16"),
            C_hbm: T.Tensor((1, M, 1, N), "float16"),
        ):
            with T.Kernel(n_blocks, m_blocks, threads=128) as (bx, by):
                A_sh = T.alloc_shared((MLEN, MLEN), "float16")
                B_sh = T.alloc_shared((MLEN, MLEN), "float16")
                C_sh = T.alloc_shared((MLEN, MLEN), "float16")

                C_loc   = T.alloc_fragment((MLEN, MLEN), "float16")
                SCR_loc = T.alloc_fragment((MLEN, MLEN), "float16")

                for row in T.serial(MLEN):
                    for col in T.Parallel(MLEN):
                        C_loc[row, col] = T.float16(0)

                for k_block in T.serial(k_blocks):
                    T.copy(
                        A_hbm[0,
                              by * MLEN : (by + 1) * MLEN,
                              0,
                              k_block * MLEN : (k_block + 1) * MLEN],
                        A_sh,
                    )
                    T.copy(
                        B_hbm[0,
                              bx * MLEN : (bx + 1) * MLEN,
                              0,
                              k_block * MLEN : (k_block + 1) * MLEN],
                        B_sh,
                    )

                    T.gemm(A_sh, B_sh, SCR_loc, transpose_B=True)

                    for row in T.serial(MLEN):
                        for col in T.Parallel(MLEN):
                            C_loc[row, col] = C_loc[row, col] + SCR_loc[row, col]

                T.copy(C_loc, C_sh)
                T.copy(
                    C_sh,
                    C_hbm[0,
                          by * MLEN : (by + 1) * MLEN,
                          0,
                          bx * MLEN : (bx + 1) * MLEN],
                )

    lowered = linear_min
    constants = {
        "M": M, "N": N, "K": K, "MLEN": MLEN,
        "M_BLOCKS": m_blocks, "N_BLOCKS": n_blocks, "K_BLOCKS": k_blocks,
        "WITH_BIAS": with_bias,
    }
    return lowered, constants


__all__ = ["make_linear_min"]
