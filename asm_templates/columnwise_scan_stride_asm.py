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
) -> str:
    """
    Stride-optimized reordering of batch=1 matrix from row-major to column-major layout.
    
    This version uses H_PREFETCH_V with rstride=1 to load multiple rows from a column block with
    a single instruction call, reducing the number of prefetch instructions compared to the
    per-row loop approach.
    
    The key optimization: When rstride=1, H_PREFETCH_V reads from addresses spaced by STRIDE_REG,
    loading prefetch_block_size (equivalent to HBM_V_Prefetch_Amount) rows at once. We set 
    STRIDE_REG = cols * input_row_stride to step through rows within a column, and load all rows' 
    tiles for that column.
    
    The input is treated as a flattened (rows * cols, feature_dim) matrix.
    Output rows are written in transposed order so that the logical index
    (r, c) maps to output row c * rows + r.

    Args:
        mlen: Matrix tile size
        vlen: Vector length / VRAM row width for D dimension (assumed to divide feature_dim)
        rows: Number of rows in the logical 2D grid
        cols: Number of columns in the logical 2D grid
        feature_dim: Feature width per row (D) - should be divisible by vlen
        alive_registers: GP registers available for use [src_reg, dst_reg, ..., temp_reg]
        input_hbm_base_addr_reg: HBM base address register index.
        output_vram_base: Base VRAM address for the transposed output.
        prefetch_block_size: Number of rows to prefetch per H_PREFETCH_V call (HBM_V_Prefetch_Amount parameter in plena_settings.toml).
        input_row_stride: Physical row stride for the source matrix.
        output_row_stride: Physical row stride for the destination matrix.

    Returns:
        Assembly string that writes the transposed layout into VRAM using stride-based prefetch.
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
    temp_reg = alive_registers[3]

    feature_tiles = output_row_stride // vlen

    lines: list[str] = []
    lines.append("; Columnwise scan with stride-based prefetch optimization")
    lines.append(f"; Reorder (rows={rows}, cols={cols}, D={feature_dim}) -> (cols, rows, D)")
    lines.append(f"; Prefetch block size: {prefetch_block_size} rows per full H_PREFETCH_V block")
    lines.append(f"; Source stride={input_row_stride}, destination stride={output_row_stride}")
    lines.append(f"; STRIDE_REG will be set to {cols * input_row_stride} (cols * input_row_stride)")

    # Set up stride register: stride between consecutive rows within the same column
    # To go from (row_r, col_c) to (row_r+1, col_c), we move by:
    # ((r+1)*cols + c - (r*cols + c)) * input_row_stride = cols * input_row_stride
    col_stride = cols * input_row_stride
    total_input_size = rows * cols * input_row_stride
    lines.extend(load_large_int(temp_reg, total_input_size))
    lines.append(f"C_SET_SCALE_REG gp{temp_reg}")
    lines.extend(load_large_int(temp_reg, col_stride))
    lines.append(f"C_SET_STRIDE_REG gp{temp_reg}")

    # For each column
    for col in range(cols):
        full_block_count = rows // prefetch_block_size
        tail_start = full_block_count * prefetch_block_size

        for block_idx in range(full_block_count):
            row_start = block_idx * prefetch_block_size
            row_end = row_start + prefetch_block_size
            
            lines.append(f"; Column {col}, row block [{row_start}, {row_end}), full stride block")
            
            # For each feature tile within this row block
            for tile in range(feature_tiles):
                # Calculate source address for the first row in the block
                source_row_index_first = row_start * cols + col
                source_row_base_first = source_row_index_first * input_row_stride
                source_offset = source_row_base_first + tile * vlen
                
                # Calculate destination address for the first row in the block
                # (where the output will start writing)
                dest_row_index_first = col * rows + row_start
                dest_row_base_first = output_vram_base + dest_row_index_first * output_row_stride
                dest_offset = dest_row_base_first + tile * vlen
                
                lines.extend(load_large_int(src_reg, source_offset))
                lines.extend(load_large_int(dst_reg, dest_offset))
                
                # Use H_PREFETCH_V with rstride=1 to load a full block with a single call.
                # Source: reads from source_offset + 0, + stride, + 2*stride, ...
                # Destination: writes consecutively to VRAM starting at dest_offset
                # (which is exactly where we want row_start's data for this tile)
                lines.append(
                    f"H_PREFETCH_V gp{dst_reg}, gp{src_reg}, a{input_hbm_base_addr_reg}, 1, 0"
                )

        # Tail rows cannot use the stride path safely because H_PREFETCH_V always loads the
        # configured prefetch width. Fall back to the original row-wise pattern for the tail.
        for row in range(tail_start, rows):
            source_row_index = row * cols + col
            dest_row_index = col * rows + row

            source_row_base = source_row_index * input_row_stride
            dest_row_base = output_vram_base + dest_row_index * output_row_stride

            for tile in range(feature_tiles):
                source_offset = source_row_base + tile * vlen
                dest_offset = dest_row_base + tile * vlen

                lines.extend(load_large_int(src_reg, source_offset))
                lines.extend(load_large_int(dst_reg, dest_offset))
                lines.append(
                    f"H_PREFETCH_V gp{dst_reg}, gp{src_reg}, a{input_hbm_base_addr_reg}, 0, 0"
                )

    return "\n".join(lines) + "\n"
