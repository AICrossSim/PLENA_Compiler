from __future__ import annotations

from ._imm import load_large_int


def sequence_scan_asm(
    mlen: int,
    vlen: int,
    seq_len: int,
    feature_dim: int,
    alive_registers: list[int],
    input_hbm_base_addr_reg: int,
    output_vram_base: int,
    prefetch_block_size: int = 1,
    input_row_stride: int | None = None,
    output_row_stride: int | None = None,
    reverse: bool = False,
) -> str:
    """
    Emit assembly that scans a flattened sequence (B=1, S, D) layout and writes
    it into VRAM. When `reverse=True`, the generated code walks source blocks
    from the end of S toward the beginning and uses `H_PREFETCH_R_V` so each
    prefetch loads a block in reverse order. This gives the same effect as
    `x.flip(dims=[1])` but with fewer instructions than a per-element loop.

    The generated code uses `rstride=1` for block prefetching. When the total
    sequence length is not divisible by `prefetch_block_size`, the tail falls
    back to single-row prefetches.
    """
    _ = mlen

    if len(alive_registers) < 3:
        raise ValueError("sequence_scan_asm requires at least 3 GP registers")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")

    if input_row_stride is None:
        input_row_stride = feature_dim
    if output_row_stride is None:
        output_row_stride = input_row_stride

    if input_row_stride < feature_dim:
        raise ValueError("input_row_stride must be >= feature_dim")
    if output_row_stride < feature_dim:
        raise ValueError("output_row_stride must be >= feature_dim")

    if input_row_stride % vlen != 0:
        raise ValueError("input_row_stride must be a multiple of vlen")
    if output_row_stride % vlen != 0:
        raise ValueError("output_row_stride must be a multiple of vlen")

    src_reg = alive_registers[0]
    dst_reg = alive_registers[1]
    temp_reg = alive_registers[2]

    feature_tiles = output_row_stride // vlen

    lines: list[str] = []
    lines.append("; Sequence scan generation")
    lines.append(f"; Scan (B=1, S={seq_len}, D={feature_dim})")
    lines.append(f"; Source stride={input_row_stride}, destination stride={output_row_stride}")

    total_input_size = seq_len * input_row_stride
    lines.extend(load_large_int(temp_reg, total_input_size))
    lines.append(f"C_SET_SCALE_REG gp{temp_reg}")
    lines.extend(load_large_int(temp_reg, input_row_stride))
    lines.append(f"C_SET_STRIDE_REG gp{temp_reg}")

    prefetch_mnemonic = "H_PREFETCH_R_V" if reverse else "H_PREFETCH_V"

    full_block_count = seq_len // prefetch_block_size
    tail_start = full_block_count * prefetch_block_size

    block_indices = range(full_block_count - 1, -1, -1) if reverse else range(full_block_count)

    output_cursor = 0
    for block_idx in block_indices:
        if reverse:
            source_start = (block_idx + 1) * prefetch_block_size - 1
        else:
            source_start = block_idx * prefetch_block_size
        dest_start = output_cursor
        output_cursor += prefetch_block_size

        for tile in range(feature_tiles):
            source_offset = source_start * input_row_stride + tile * vlen
            dest_offset = output_vram_base + dest_start * output_row_stride + tile * vlen

            lines.extend(load_large_int(src_reg, source_offset))
            lines.extend(load_large_int(dst_reg, dest_offset))
            lines.append(f"{prefetch_mnemonic} gp{dst_reg}, gp{src_reg}, a{input_hbm_base_addr_reg}, 1, 0")

    if tail_start < seq_len:
        tail_indices = range(seq_len - 1, tail_start - 1, -1) if reverse else range(tail_start, seq_len)
        for out_idx, source_idx in enumerate(tail_indices, start=output_cursor):
            for tile in range(feature_tiles):
                source_offset = source_idx * input_row_stride + tile * vlen
                dest_offset = output_vram_base + out_idx * output_row_stride + tile * vlen

                lines.extend(load_large_int(src_reg, source_offset))
                lines.extend(load_large_int(dst_reg, dest_offset))
                lines.append(f"{prefetch_mnemonic} gp{dst_reg}, gp{src_reg}, a{input_hbm_base_addr_reg}, 0, 0")

    return "\n".join(lines) + "\n"
