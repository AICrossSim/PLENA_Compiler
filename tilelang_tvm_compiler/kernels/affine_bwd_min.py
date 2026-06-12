"""Affine BACKWARD — vector-only, exact input gradient.

Covers the input gradient of modulate (Y = X*(1+scale)+shift) and
residual_gate (OUT = X+gate*Y): a single whole-tile elementwise multiply by
the (broadcast) affine factor, dX = dY * factor. This is the exact dX — no
matmul, no reduction, the cheapest backward in the block. (The dscale /
dshift / dgate parameter gradients live in the adaLN modulation path, not
here.)
"""

import tilelang.language as T

from ..plena_settings import load_sizes as _load_sizes


def make_affine_bwd_min(*, rows=None, hlen=None, head_count=8,
                        num_s_blocks=2, batch=1, **_ignore):
    _hw = _load_sizes()
    MLEN = _hw.mlen
    hlen = hlen if hlen is not None else _hw.hlen
    rows = rows if rows is not None else MLEN
    seq_len = num_s_blocks * rows

    @T.prim_func
    def affine_bwd_min(
        dY_hbm:  T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        FAC_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),  # (1+scale) or gate
        dX_hbm:  T.Tensor((batch, seq_len, head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            dY_sh  = T.alloc_shared((rows, hlen), "float16")
            FAC_sh = T.alloc_shared((rows, hlen), "float16")
            dX_sh  = T.alloc_shared((rows, hlen), "float16")
            T.copy(dY_hbm[0, s_block*rows:(s_block+1)*rows, by, 0:hlen], dY_sh)
            T.copy(FAC_hbm[0, s_block*rows:(s_block+1)*rows, by, 0:hlen], FAC_sh)
            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    dX_sh[row, col] = dY_sh[row, col] * FAC_sh[row, col]
            T.copy(dX_sh, dX_hbm[0, s_block*rows:(s_block+1)*rows, by, 0:hlen])

    return affine_bwd_min, {"MLEN": MLEN, "HLEN": hlen, "HEAD_COUNT": head_count,
                            "NUM_S_BLOCKS": num_s_blocks}


__all__ = ["make_affine_bwd_min"]
