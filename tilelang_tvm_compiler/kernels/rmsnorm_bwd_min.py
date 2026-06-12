"""RMSNorm BACKWARD — exact analytic gradient.

Forward (rmsnorm_min):  y_i = x_i / rms * g_i,  rms = sqrt(mean(x^2) + eps).
Exact backward (gradient wrt x):
    let inv = 1/rms,  c = (1/H) * inv^2 * sum_j (dy_j * g_j * x_j)
    dx_i = inv * (dy_i * g_i) - c * x_i
         = inv * dy_i * g_i - x_i * inv^3 * (1/H) * sum_j dy_j g_j x_j

This is the exact RMSNorm input gradient (matches PyTorch). We recompute
rms from x (one Sigma x^2 reduce + sqrt) and the cross term (one
Sigma dy*g*x reduce), then combine. Vector-only, no matmul.
"""

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes


def make_rmsnorm_bwd_min(*, rows=None, hlen=None, head_count=8,
                         num_s_blocks=2, batch=1, eps=1e-6, **_ignore):
    _hw = _load_sizes()
    MLEN = _hw.mlen
    hlen = hlen if hlen is not None else _hw.hlen
    rows = rows if rows is not None else MLEN
    seq_len = num_s_blocks * rows
    inv_n = 1.0 / hlen

    @T.prim_func
    def rmsnorm_bwd_min(
        X_hbm:     T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        SCALE_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        dY_hbm:    T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        dX_hbm:    T.Tensor((batch, seq_len, head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh  = T.alloc_shared((rows, hlen), "float16")
            SC_sh = T.alloc_shared((rows, hlen), "float16")
            dY_sh = T.alloc_shared((rows, hlen), "float16")
            SQ    = T.alloc_shared((rows, hlen), "float16")   # x^2
            CR    = T.alloc_shared((rows, hlen), "float16")   # dy*g*x
            dX_sh = T.alloc_shared((rows, hlen), "float16")
            ss    = T.alloc_fragment((rows,), "float16")      # sum x^2
            cross = T.alloc_fragment((rows,), "float16")      # sum dy*g*x
            inv   = T.alloc_fragment((rows,), "float16")      # 1/rms
            c     = T.alloc_fragment((rows,), "float16")      # (1/H)*inv^3*cross
            T.copy(X_hbm[0, s_block*rows:(s_block+1)*rows, by, 0:hlen], X_sh)
            T.copy(SCALE_hbm[0, s_block*rows:(s_block+1)*rows, by, 0:hlen], SC_sh)
            T.copy(dY_hbm[0, s_block*rows:(s_block+1)*rows, by, 0:hlen], dY_sh)
            # SQ = x^2 ; CR = dy*g*x
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    SQ[row, col] = X_sh[row, col] * X_sh[row, col]
                for col in T.Parallel(hlen):
                    CR[row, col] = dY_sh[row, col] * SC_sh[row, col]
                for col in T.Parallel(hlen):
                    CR[row, col] = CR[row, col] * X_sh[row, col]
                ss[row] = T.float16(0)
                cross[row] = T.float16(0)
            T.reduce_sum(SQ, ss, dim=1, clear=False)
            T.reduce_sum(CR, cross, dim=1, clear=False)
            # inv = 1/sqrt(mean(x^2)+eps) ; c = (1/H)*inv^3*cross
            for row in T.serial(rows):
                inv[row] = ss[row] * T.float16(inv_n)
                inv[row] = inv[row] + T.float16(eps)
                inv[row] = T.sqrt(inv[row])
                inv[row] = T.float16(1.0) / inv[row]
                c[row] = inv[row] * inv[row]
                c[row] = c[row] * inv[row]
                c[row] = c[row] * cross[row]
                c[row] = c[row] * T.float16(inv_n)
            # dx_i = inv * dy_i * g_i - c * x_i
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    dX_sh[row, col] = dY_sh[row, col] * SC_sh[row, col]
                for col in T.Parallel(hlen):
                    dX_sh[row, col] = dX_sh[row, col] * inv[row]
                for col in T.Parallel(hlen):
                    dX_sh[row, col] = dX_sh[row, col] - X_sh[row, col] * c[row]
            T.copy(dX_sh, dX_hbm[0, s_block*rows:(s_block+1)*rows, by, 0:hlen])

    return rmsnorm_bwd_min, {"MLEN": MLEN, "HLEN": hlen, "HEAD_COUNT": head_count,
                             "NUM_S_BLOCKS": num_s_blocks}


__all__ = ["make_rmsnorm_bwd_min"]
