"""GELU BACKWARD — exact derivative of the tanh-approx GELU.

Forward (gelu_min, tanh approximation):
    u    = sqrt(2/pi) * (x + 0.044715 * x^3)
    t    = tanh(u) = 1 - 2/(exp(2u)+1)          (no native tanh; exp/reci)
    gelu = 0.5 * x * (1 + t)

Backward is the exact analytic derivative of that same forward:
    u'        = sqrt(2/pi) * (1 + 3*0.044715 * x^2)
    sech^2(u) = 1 - t^2
    gelu'(x)  = 0.5*(1 + t) + 0.5 * x * (1 - t^2) * u'
    dX        = dY * gelu'(x)

This is mathematically exact for the tanh-GELU (the same form PyTorch /
GPT use); tanh is expanded via exp/reci exactly as the forward kernel does,
so there is no approximation beyond the tanh-GELU itself. Vector-only, no
matmul.
"""

import math

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes

_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_C = 0.044715


def make_gelu_bwd_min(*, rows=None, hlen=None, head_count=8,
                      num_s_blocks=2, batch=1, **_ignore):
    _hw = _load_sizes()
    MLEN = _hw.mlen
    hlen = hlen if hlen is not None else _hw.hlen
    rows = rows if rows is not None else MLEN
    seq_len = num_s_blocks * rows

    @T.prim_func
    def gelu_bwd_min(
        X_hbm:  T.Tensor((batch, seq_len, head_count, hlen), "float16"),  # fwd input
        dY_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),  # upstream grad
        dX_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh  = T.alloc_shared((rows, hlen), "float16")
            dY_sh = T.alloc_shared((rows, hlen), "float16")
            dX_sh = T.alloc_shared((rows, hlen), "float16")
            T.copy(X_hbm[0, s_block*rows:(s_block+1)*rows, by, 0:hlen], X_sh)
            T.copy(dY_hbm[0, s_block*rows:(s_block+1)*rows, by, 0:hlen], dY_sh)
            for row in T.serial(rows):
                for i in T.Parallel(hlen):
                    x = X_sh[row, i]
                    u = T.float16(_SQRT_2_OVER_PI) * (x + T.float16(_C) * (x * x * x))
                    t = T.float16(1.0) - T.float16(2.0) * (
                        T.float16(1.0) / (T.exp(T.float16(2.0) * u) + T.float16(1.0)))
                    up = T.float16(_SQRT_2_OVER_PI) * (
                        T.float16(1.0) + T.float16(3.0 * _C) * (x * x))
                    # gelu'(x) = 0.5(1+t) + 0.5*x*(1-t^2)*u'
                    dgelu = T.float16(0.5) * (T.float16(1.0) + t) \
                        + T.float16(0.5) * x * (T.float16(1.0) - t * t) * up
                    dX_sh[row, i] = dY_sh[row, i] * dgelu
            T.copy(dX_sh, dX_hbm[0, s_block*rows:(s_block+1)*rows, by, 0:hlen])

    return gelu_bwd_min, {"MLEN": MLEN, "HLEN": hlen, "HEAD_COUNT": head_count,
                          "NUM_S_BLOCKS": num_s_blocks}


__all__ = ["make_gelu_bwd_min"]
