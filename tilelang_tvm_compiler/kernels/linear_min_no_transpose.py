"""Linear-min kernel — multi-tile GEMM (no-transpose-B variant).

Same shape semantics as ``linear_min`` but ``B`` is laid out as
``(K, N)`` row-major rather than ``(N, K)`` — i.e. the host has already
transposed the weight. The lowering uses plain ``M_MM`` (no ``M_TMM``),
exercising the matmul path that does NOT need the transpose flag.

What it computes (output per (m_block, n_block) tile):

    C[m_block, n_block] = sum_{k_block} A[m_block, k_block]
                                       @ B[k_block, n_block]
                        + bias[m_block, n_block]    (optional)

Differences vs ``linear_min`` (``transpose_B=True``):
  * ``B_hbm`` shape:     ``(1, K, 1, N)`` (was ``(1, N, 1, K)``)
  * Slice over B:        ``[k_block * MLEN : ., bx * MLEN : .]``
                         (was ``[bx * MLEN : ., k_block * MLEN : .]``)
  * ``T.gemm`` call:     no ``transpose_B`` argument (default False)
  * Inner ISA:           ``M_MM`` instead of ``M_TMM``; per-oc B step
                         is ``blen`` (cols of (K, N)) instead of
                         ``blen * mlen`` (rows of (N, K)).

Everything else (K-acc via SCR_loc + tile_add, grid layout, bias as
host-broadcast (M, N) tile) is identical to ``linear_min``.
"""

import tilelang.language as T


def make_linear_min_no_transpose(
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

    if with_bias:
        @T.prim_func
        def linear_min_no_transpose(
            A_hbm:    T.Tensor((1, M, 1, K), "float16"),
            B_hbm:    T.Tensor((1, K, 1, N), "float16"),
            BIAS_hbm: T.Tensor((1, M, 1, N), "float16"),
            C_hbm:    T.Tensor((1, M, 1, N), "float16"),
        ):
            with T.Kernel(n_blocks, m_blocks, threads=128) as (bx, by):
                A_sh    = T.alloc_shared((MLEN, MLEN), "float16")
                B_sh    = T.alloc_shared((MLEN, MLEN), "float16")
                BIAS_sh = T.alloc_shared((MLEN, MLEN), "float16")
                C_sh    = T.alloc_shared((MLEN, MLEN), "float16")

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
                    # B is (K, N) row-major: walk K along the seq axis and
                    # N along the hlen axis. No transpose needed in the
                    # matmul itself.
                    T.copy(
                        B_hbm[0,
                              k_block * MLEN : (k_block + 1) * MLEN,
                              0,
                              bx * MLEN : (bx + 1) * MLEN],
                        B_sh,
                    )

                    T.gemm(A_sh, B_sh, SCR_loc)

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
        def linear_min_no_transpose(
            A_hbm: T.Tensor((1, M, 1, K), "float16"),
            B_hbm: T.Tensor((1, K, 1, N), "float16"),
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
                              k_block * MLEN : (k_block + 1) * MLEN,
                              0,
                              bx * MLEN : (bx + 1) * MLEN],
                        B_sh,
                    )

                    T.gemm(A_sh, B_sh, SCR_loc)

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

    lowered = linear_min_no_transpose
    constants = {
        "M": M, "N": N, "K": K, "MLEN": MLEN,
        "M_BLOCKS": m_blocks, "N_BLOCKS": n_blocks, "K_BLOCKS": k_blocks,
        "WITH_BIAS": with_bias,
    }
    return lowered, constants


__all__ = ["make_linear_min_no_transpose"]
