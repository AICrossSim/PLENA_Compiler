"""LayerNorm BACKWARD — exact analytic gradient.

Forward (layernorm_min):  y = xhat * g + b,  xhat = (x - mu)/sigma,
                          mu = mean(x), sigma = sqrt(var(x) + eps).
Exact backward (gradient wrt x), with g_i = scale_i:
    let  a_i = dy_i * g_i
         s1  = (1/H) sum_j a_j
         s2  = (1/H) sum_j a_j * xhat_j
    dx_i = (1/sigma) * ( a_i - s1 - xhat_i * s2 )

This is the exact LayerNorm input gradient (matches PyTorch). We recompute
mu, sigma (two reductions: sum x, sum x^2) and the two gradient reductions
(sum a, sum a*xhat), then combine. Vector-only, no matmul.
"""

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes


def make_layernorm_bwd_min(*, rows=None, hidden_size=None, num_s_blocks=2,
                           batch=1, eps=1e-5, **_ignore):
    _hw = _load_sizes()
    MLEN = _hw.mlen
    rows = rows if rows is not None else MLEN
    H = hidden_size if hidden_size is not None else MLEN
    seq_len = num_s_blocks * rows
    inv_h = 1.0 / H

    @T.prim_func
    def layernorm_bwd_min(
        X_hbm:     T.Tensor((batch, seq_len, 1, H), "float16"),
        SCALE_hbm: T.Tensor((batch, seq_len, 1, H), "float16"),
        dY_hbm:    T.Tensor((batch, seq_len, 1, H), "float16"),
        dX_hbm:    T.Tensor((batch, seq_len, 1, H), "float16"),
    ):
        with T.Kernel(num_s_blocks, threads=128) as s_block:
            X_sh   = T.alloc_shared((rows, H), "float16")
            SC_sh  = T.alloc_shared((rows, H), "float16")
            dY_sh  = T.alloc_shared((rows, H), "float16")
            SQ     = T.alloc_shared((rows, H), "float16")   # x^2
            XH     = T.alloc_shared((rows, H), "float16")   # xhat
            A      = T.alloc_shared((rows, H), "float16")   # dy*g
            AX     = T.alloc_shared((rows, H), "float16")   # a*xhat
            dX_sh  = T.alloc_shared((rows, H), "float16")
            sx     = T.alloc_fragment((rows,), "float16")   # sum x
            sx2    = T.alloc_fragment((rows,), "float16")   # sum x^2
            sa     = T.alloc_fragment((rows,), "float16")   # sum a
            sax    = T.alloc_fragment((rows,), "float16")   # sum a*xhat
            mu     = T.alloc_fragment((rows,), "float16")
            invsig = T.alloc_fragment((rows,), "float16")
            T.copy(X_hbm[0, s_block*rows:(s_block+1)*rows, 0, 0:H], X_sh)
            T.copy(SCALE_hbm[0, s_block*rows:(s_block+1)*rows, 0, 0:H], SC_sh)
            T.copy(dY_hbm[0, s_block*rows:(s_block+1)*rows, 0, 0:H], dY_sh)
            # mu, var: sum x and sum x^2
            for row in T.serial(rows):
                for col in T.Parallel(H):
                    SQ[row, col] = X_sh[row, col] * X_sh[row, col]
                sx[row] = T.float16(0)
                sx2[row] = T.float16(0)
            T.reduce_sum(X_sh, sx, dim=1, clear=False)
            T.reduce_sum(SQ, sx2, dim=1, clear=False)
            for row in T.serial(rows):
                mu[row] = sx[row] * T.float16(inv_h)
                invsig[row] = sx2[row] * T.float16(inv_h)        # mean(x^2)
                invsig[row] = invsig[row] - mu[row] * mu[row]    # var
                invsig[row] = invsig[row] + T.float16(eps)
                invsig[row] = T.sqrt(invsig[row])
                invsig[row] = T.float16(1.0) / invsig[row]       # 1/sigma
            # xhat = (x-mu)/sigma ; a = dy*g ; ax = a*xhat
            for row in T.serial(rows):
                for col in T.Parallel(H):
                    XH[row, col] = X_sh[row, col] - mu[row]
                for col in T.Parallel(H):
                    XH[row, col] = XH[row, col] * invsig[row]
                for col in T.Parallel(H):
                    A[row, col] = dY_sh[row, col] * SC_sh[row, col]
                for col in T.Parallel(H):
                    AX[row, col] = A[row, col] * XH[row, col]
                sa[row] = T.float16(0)
                sax[row] = T.float16(0)
            T.reduce_sum(A, sa, dim=1, clear=False)
            T.reduce_sum(AX, sax, dim=1, clear=False)
            # dx = (1/sigma)*( a - mean(a) - xhat*mean(a*xhat) )
            for row in T.serial(rows):
                sa[row] = sa[row] * T.float16(inv_h)
                sax[row] = sax[row] * T.float16(inv_h)
            for row in T.serial(rows):
                for col in T.Parallel(H):
                    dX_sh[row, col] = A[row, col] - sa[row]
                for col in T.Parallel(H):
                    dX_sh[row, col] = dX_sh[row, col] - XH[row, col] * sax[row]
                for col in T.Parallel(H):
                    dX_sh[row, col] = dX_sh[row, col] * invsig[row]
            T.copy(dX_sh, dX_hbm[0, s_block*rows:(s_block+1)*rows, 0, 0:H])

    return layernorm_bwd_min, {"MLEN": MLEN, "H": H, "NUM_S_BLOCKS": num_s_blocks}


__all__ = ["make_layernorm_bwd_min"]
