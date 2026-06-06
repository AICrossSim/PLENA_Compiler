from __future__ import annotations

from ._imm import load_large_int


def columnwise_scan_stride_asm(
    mlen: int,
    vlen: int,
    rows: int,
    cols: int,
    feature_dim: int,
    alive_registers: list[int],
    input_hbm_base_addr_reg: int,
    output_vram_base: int,
    prefetch_block_size: int = 4,
    input_row_stride: int | None = None,
    output_row_stride: int | None = None,
    reverse: bool = False,
) -> str:
    """
    Stride-optimized reordering of batch=1 matrix to column-major token order
    with chunk-major VRAM layout [NB, S, V].
    
    This version uses H_PREFETCH_V with rstride=1 to load multiple rows from a column block with
    a single instruction call, reducing the number of prefetch instructions compared to the
    per-row loop approach.
    
    The key optimization: When rstride=1, H_PREFETCH_V reads from addresses spaced by STRIDE_REG,
    loading prefetch_block_size (equivalent to HBM_V_Prefetch_Amount) rows at once. We set 
    STRIDE_REG = cols * input_row_stride to step through rows within a column, and load all rows' 
    tiles for that column.
    
    The input is treated as a flattened (rows * cols, feature_dim) matrix.
    Destination token order is transposed so that logical index (r, c) maps
    to token index c * rows + r. When `reverse=True`, the transposed token
    order is reversed to match a sequence flip after the columnwise reorder.

    Args:
        mlen: Matrix tile size
        vlen: Vector length / VRAM row width for D dimension (assumed to divide feature_dim)
        rows: Number of rows in the logical 2D grid
        cols: Number of columns in the logical 2D grid
        feature_dim: Feature width per row (D) - should be divisible by vlen
        alive_registers: GP registers available for use [src_reg, dst_reg, ..., temp_reg]
        input_hbm_base_addr_reg: HBM base address register index.
        output_vram_base: Base VRAM address for the chunk-major output.
        prefetch_block_size: Number of rows to prefetch per H_PREFETCH_V call (HBM_V_Prefetch_Amount parameter in plena_settings.toml).
        input_row_stride: Physical row stride for the source matrix.
        output_row_stride: Physical row stride for the destination matrix.

    Returns:
        Assembly string that writes chunk-major [NB, S, V] layout using stride-based prefetch.
    """
    _ = mlen

    if len(alive_registers) < 4:
        raise ValueError("columnwise_scan_stride_asm requires at least 4 GP registers")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must both be positive")
    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive")
    if prefetch_block_size <= 0:
        raise ValueError("prefetch_block_size must be positive")

    if input_row_stride is None:
        input_row_stride = feature_dim  #i.e. consecutive packed rows
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
    loop_reg = alive_registers[2]
    temp_reg = alive_registers[3]

    seq_len = rows * cols
    feature_tiles = output_row_stride // vlen
    prefetch_mnemonic = "H_PREFETCH_R_V" if reverse else "H_PREFETCH_V"

    lines: list[str] = []
    lines.append("; Columnwise scan with stride-based prefetch optimization (chunk-major output)")
    if reverse:
        lines.append(
            f"; Reorder token order (rows={rows}, cols={cols}, D={feature_dim}) -> reverse((cols, rows, D))"
        )
    else:
        lines.append(f"; Reorder token order (rows={rows}, cols={cols}, D={feature_dim}) -> (cols, rows, D)")
    lines.append(f"; Prefetch block size: {prefetch_block_size} rows per full H_PREFETCH_V block")
    lines.append(f"; Source stride={input_row_stride}, destination stride={output_row_stride}")
    lines.append(f"; STRIDE_REG will be set to {cols * input_row_stride} (cols * input_row_stride)")

    # Set up stride register: stride between consecutive rows within the same column
    # To go from (row_r, col_c) to (row_r+1, col_c), we move by:
    # ((r+1)*cols + c - (r*cols + c)) * input_row_stride = cols * input_row_stride
    col_stride = cols * input_row_stride
    total_input_size = seq_len * input_row_stride
    lines.extend(load_large_int(temp_reg, total_input_size))
    lines.append(f"C_SET_SCALE_REG gp{temp_reg}")
    lines.extend(load_large_int(temp_reg, col_stride))
    lines.append(f"C_SET_STRIDE_REG gp{temp_reg}")

    if not reverse:
        # For each column
        for col in range(cols):
            full_block_count = rows // prefetch_block_size
            tail_start = full_block_count * prefetch_block_size
            tile_dst_step = seq_len * vlen

            lines.append(f"; Column {col}: full blocks={full_block_count}, tail={rows - tail_start}")

            # Full blocks: keep row-block iteration in Python, but loop tiles in hardware.
            for block_idx in range(full_block_count):
                row_start = block_idx * prefetch_block_size
                source_offset = (row_start * cols + col) * input_row_stride
                dest_token_index_first = col * rows + row_start
                # Chunk-major [NB, S, V] for B=1:
                # flat_idx = ((tile * seq_len) + token_idx) * vlen
                dest_offset = output_vram_base + dest_token_index_first * vlen

                lines.extend(load_large_int(src_reg, source_offset))
                lines.extend(load_large_int(dst_reg, dest_offset))
                lines.extend(load_large_int(temp_reg, tile_dst_step))
                lines.append(f"C_LOOP_START gp{loop_reg}, {feature_tiles}")
                lines.append(f"{prefetch_mnemonic} gp{dst_reg}, gp{src_reg}, a{input_hbm_base_addr_reg}, 1, 0")
                lines.append(f"S_ADDI_INT gp{src_reg}, gp{src_reg}, {vlen}")
                lines.append(f"S_ADD_INT gp{dst_reg}, gp{dst_reg}, gp{temp_reg}")
                lines.append(f"C_LOOP_END gp{loop_reg}")

            # Tail rows: per-row source progression, with tile loop in hardware.
            for row in range(tail_start, rows):
                source_offset = (row * cols + col) * input_row_stride
                dest_token_index = col * rows + row
                dest_offset = output_vram_base + dest_token_index * vlen

                lines.extend(load_large_int(src_reg, source_offset))
                lines.extend(load_large_int(dst_reg, dest_offset))
                lines.extend(load_large_int(temp_reg, tile_dst_step))
                lines.append(f"C_LOOP_START gp{loop_reg}, {feature_tiles}")
                lines.append(f"{prefetch_mnemonic} gp{dst_reg}, gp{src_reg}, a{input_hbm_base_addr_reg}, 0, 0")
                lines.append(f"S_ADDI_INT gp{src_reg}, gp{src_reg}, {vlen}")
                lines.append(f"S_ADD_INT gp{dst_reg}, gp{dst_reg}, gp{temp_reg}")
                lines.append(f"C_LOOP_END gp{loop_reg}")

        return "\n".join(lines) + "\n"

    block_src_step = prefetch_block_size * col_stride
    for col in range(cols):
        _ = col

    for reverse_col_idx, col in enumerate(range(cols - 1, -1, -1)):
        full_block_count = rows // prefetch_block_size
        tail_start = full_block_count * prefetch_block_size
        tile_dst_step = seq_len * vlen
        output_cursor = reverse_col_idx * rows

        lines.append(f"; Reverse column {col}: full blocks={full_block_count}, tail={rows - tail_start}")

        if tail_start < rows:
            reverse_tail_indices = range(rows - 1, tail_start - 1, -1)
            for out_idx, row in enumerate(reverse_tail_indices):
                source_offset = (row * cols + col) * input_row_stride
                dest_token_index = output_cursor + out_idx
                dest_offset = output_vram_base + dest_token_index * vlen

                lines.extend(load_large_int(src_reg, source_offset))
                lines.extend(load_large_int(dst_reg, dest_offset))
                lines.extend(load_large_int(temp_reg, tile_dst_step))
                lines.append(f"C_LOOP_START gp{loop_reg}, {feature_tiles}")
                lines.append(f"{prefetch_mnemonic} gp{dst_reg}, gp{src_reg}, a{input_hbm_base_addr_reg}, 0, 0")
                lines.append(f"S_ADDI_INT gp{src_reg}, gp{src_reg}, {vlen}")
                lines.append(f"S_ADD_INT gp{dst_reg}, gp{dst_reg}, gp{temp_reg}")
                lines.append(f"C_LOOP_END gp{loop_reg}")
            output_cursor += rows - tail_start

        for block_idx in range(full_block_count):
            row_end = tail_start - 1 - block_idx * prefetch_block_size
            source_offset = (row_end * cols + col) * input_row_stride
            dest_token_index_first = output_cursor + block_idx * prefetch_block_size
            dest_offset = output_vram_base + dest_token_index_first * vlen

            lines.extend(load_large_int(src_reg, source_offset))
            lines.extend(load_large_int(dst_reg, dest_offset))
            lines.extend(load_large_int(temp_reg, tile_dst_step))
            lines.append(f"C_LOOP_START gp{loop_reg}, {feature_tiles}")
            lines.append(f"{prefetch_mnemonic} gp{dst_reg}, gp{src_reg}, a{input_hbm_base_addr_reg}, 1, 0")
            lines.append(f"S_ADDI_INT gp{src_reg}, gp{src_reg}, {vlen}")
            lines.append(f"S_ADD_INT gp{dst_reg}, gp{dst_reg}, gp{temp_reg}")
            lines.append(f"C_LOOP_END gp{loop_reg}")
            lines.extend(load_large_int(temp_reg, block_src_step))
            lines.append(f"S_SUB_INT gp{src_reg}, gp{src_reg}, gp{temp_reg}")

    return "\n".join(lines) + "\n"
