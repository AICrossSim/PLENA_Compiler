"""Copy-offset-min kernel — staged probe for the o_head_offset path.

Reads a compact ``[B, S, H, D]`` input and writes it into a head-slice
``[o_head_offset : o_head_offset + H]`` of a wider ``[B, S, o_head_count,
D]`` output. The DMA structure matches gelu_min's offset variant; the
``compute`` knob dials how much per-element FPRAM math runs in between,
so a binary search over compute stages can pin down which operator's
interaction with the offset writeback is broken.

``compute`` stages (each adds one GELU-relevant operator):
    "copy"      VRAM->VRAM verbatim, no FPRAM at all.
    "id"        per-row VRAM->FPRAM->VRAM, identity (Y_FP = X_FP).
    "mul"       Y_FP[i] = X_FP[i] * X_FP[i]            (plain fp mul)
    "const_mul" Y_FP[i] = 0.5 * X_FP[i]                (hoisted-const mul)
    "exp"       Y_FP[i] = exp(X_FP[i])                 (fp exp)
    "reci"      Y_FP[i] = 1.0 / X_FP[i]                (fp reciprocal)
"""

import tilelang.language as T


_COMPUTE_STAGES = ("copy", "id", "mul", "const_mul", "exp", "reci")


def make_copy_offset_min(
    *,
    rows: int = 64,
    hlen: int = 16,
    head_count: int = 8,
    num_s_blocks: int = 2,
    batch: int = 1,
    o_head_count: int | None = None,
    o_head_offset: int = 0,
    compute: str = "copy",
    # Back-compat: ``fp_roundtrip=True`` is the old name for compute="id".
    fp_roundtrip: bool | None = None,
):
    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"copy_offset_min requires rows == MLEN ({MLEN}), got {rows}")
    if MLEN % hlen != 0:
        raise ValueError(f"hlen must divide MLEN ({MLEN}); got hlen={hlen}")
    hardware_lane_count = MLEN // hlen
    if head_count % hardware_lane_count != 0:
        raise ValueError(
            f"head_count must be a multiple of MLEN/hlen={hardware_lane_count}; "
            f"got {head_count}"
        )
    if num_s_blocks < 1:
        raise ValueError(f"num_s_blocks must be >= 1, got {num_s_blocks}")
    if o_head_count is None:
        o_head_count = head_count
    if o_head_count < head_count:
        raise ValueError(
            f"o_head_count ({o_head_count}) must be >= head_count ({head_count})"
        )
    if not (0 <= o_head_offset <= o_head_count - head_count):
        raise ValueError(
            f"o_head_offset ({o_head_offset}) + head_count ({head_count}) "
            f"must fit within o_head_count ({o_head_count})"
        )
    if fp_roundtrip is not None:
        compute = "id" if fp_roundtrip else "copy"
    if compute not in _COMPUTE_STAGES:
        raise ValueError(
            f"compute must be one of {_COMPUTE_STAGES}; got {compute!r}"
        )

    seq_len = num_s_blocks * rows

    # ----- compute == "copy": plain VRAM->VRAM, no FPRAM -----
    @T.prim_func
    def copy_offset_copy(
        X_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm: T.Tensor((batch, seq_len, o_head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh = T.alloc_shared((rows, hlen), "float16")
            Y_sh = T.alloc_shared((rows, hlen), "float16")
            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )
            T.copy(X_sh, Y_sh)
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows,
                      by + o_head_offset, 0:hlen],
            )

    # ----- compute == "id": per-row FPRAM round-trip, identity -----
    @T.prim_func
    def copy_offset_id(
        X_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm: T.Tensor((batch, seq_len, o_head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh = T.alloc_shared((rows, hlen), "float16")
            Y_sh = T.alloc_shared((rows, hlen), "float16")
            X_FP = T.alloc_fragment((hlen,), "float16")
            Y_FP = T.alloc_fragment((hlen,), "float16")
            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )
            for row in T.serial(rows):
                T.copy(X_sh[row, 0], X_FP)
                for i in T.unroll(hlen):
                    Y_FP[i] = X_FP[i]
                T.copy(Y_FP, Y_sh[row, 0])
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows,
                      by + o_head_offset, 0:hlen],
            )

    # ----- compute == "mul": Y = X * X -----
    @T.prim_func
    def copy_offset_mul(
        X_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm: T.Tensor((batch, seq_len, o_head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh = T.alloc_shared((rows, hlen), "float16")
            Y_sh = T.alloc_shared((rows, hlen), "float16")
            X_FP = T.alloc_fragment((hlen,), "float16")
            Y_FP = T.alloc_fragment((hlen,), "float16")
            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )
            for row in T.serial(rows):
                T.copy(X_sh[row, 0], X_FP)
                for i in T.unroll(hlen):
                    Y_FP[i] = X_FP[i] * X_FP[i]
                T.copy(Y_FP, Y_sh[row, 0])
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows,
                      by + o_head_offset, 0:hlen],
            )

    # ----- compute == "const_mul": Y = 0.5 * X  (hoisted-const mul) -----
    @T.prim_func
    def copy_offset_const_mul(
        X_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm: T.Tensor((batch, seq_len, o_head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh = T.alloc_shared((rows, hlen), "float16")
            Y_sh = T.alloc_shared((rows, hlen), "float16")
            X_FP = T.alloc_fragment((hlen,), "float16")
            Y_FP = T.alloc_fragment((hlen,), "float16")
            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )
            for row in T.serial(rows):
                T.copy(X_sh[row, 0], X_FP)
                for i in T.unroll(hlen):
                    Y_FP[i] = T.float16(0.5) * X_FP[i]
                T.copy(Y_FP, Y_sh[row, 0])
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows,
                      by + o_head_offset, 0:hlen],
            )

    # ----- compute == "exp": Y = exp(X) -----
    @T.prim_func
    def copy_offset_exp(
        X_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm: T.Tensor((batch, seq_len, o_head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh = T.alloc_shared((rows, hlen), "float16")
            Y_sh = T.alloc_shared((rows, hlen), "float16")
            X_FP = T.alloc_fragment((hlen,), "float16")
            Y_FP = T.alloc_fragment((hlen,), "float16")
            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )
            for row in T.serial(rows):
                T.copy(X_sh[row, 0], X_FP)
                for i in T.unroll(hlen):
                    Y_FP[i] = T.exp(X_FP[i])
                T.copy(Y_FP, Y_sh[row, 0])
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows,
                      by + o_head_offset, 0:hlen],
            )

    # ----- compute == "reci": Y = 1.0 / X -----
    @T.prim_func
    def copy_offset_reci(
        X_hbm: T.Tensor((batch, seq_len, head_count, hlen), "float16"),
        Y_hbm: T.Tensor((batch, seq_len, o_head_count, hlen), "float16"),
    ):
        with T.Kernel(num_s_blocks, head_count, threads=128) as (s_block, by):
            X_sh = T.alloc_shared((rows, hlen), "float16")
            Y_sh = T.alloc_shared((rows, hlen), "float16")
            X_FP = T.alloc_fragment((hlen,), "float16")
            Y_FP = T.alloc_fragment((hlen,), "float16")
            T.copy(
                X_hbm[0, s_block * rows : (s_block + 1) * rows, by, 0:hlen],
                X_sh,
            )
            for row in T.serial(rows):
                T.copy(X_sh[row, 0], X_FP)
                for i in T.unroll(hlen):
                    Y_FP[i] = T.float16(1.0) / X_FP[i]
                T.copy(Y_FP, Y_sh[row, 0])
            T.copy(
                Y_sh,
                Y_hbm[0, s_block * rows : (s_block + 1) * rows,
                      by + o_head_offset, 0:hlen],
            )

    _STAGE_FUNCS = {
        "copy": copy_offset_copy,
        "id": copy_offset_id,
        "mul": copy_offset_mul,
        "const_mul": copy_offset_const_mul,
        "exp": copy_offset_exp,
        "reci": copy_offset_reci,
    }
    lowered = _STAGE_FUNCS[compute]

    constants = {
        "ROWS": rows,
        "MLEN": MLEN,
        "HLEN": hlen,
        "HEAD_COUNT": head_count,
        "O_HEAD_COUNT": o_head_count,
        "O_HEAD_OFFSET": o_head_offset,
        "COMPUTE": compute,
        "BATCH": batch,
        "NUM_S_BLOCKS": num_s_blocks,
        "HARDWARE_LANE_COUNT": hardware_lane_count,
    }
    return lowered, constants


__all__ = ["make_copy_offset_min"]
