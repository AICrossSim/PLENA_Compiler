from __future__ import annotations

from ._imm import load_large_int


def columnwise_scan_asm(
    mlen: int,
    vlen: int,
    rows: int,
    cols: int,
    feature_dim: int,
    alive_registers: list[int],
    input_hbm_base_addr_reg: int,
    output_vram_base: int,
    input_row_stride: int | None = None,
    output_row_stride: int | None = None,
) -> str:
    """
    Reorder a batch=1 matrix from row-major (rows, cols, D) layout to
    column-major logical layout (cols, rows, D).

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
        input_row_stride: Physical row stride for the source matrix.
        output_row_stride: Physical row stride for the destination matrix.

    Returns:
        Assembly string that writes the transposed layout into VRAM.
    """
    _ = mlen

    if len(alive_registers) < 4:
        raise ValueError("columnwise_scan_asm requires at least 4 GP registers")
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must both be positive")
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
    temp_reg = alive_registers[3]

    feature_tiles = output_row_stride // vlen

    lines: list[str] = []
    lines.append("; Columnwise scan generation")
    lines.append(f"; Reorder (rows={rows}, cols={cols}, D={feature_dim}) -> (cols, rows, D)")
    lines.append(f"; Source stride={input_row_stride}, destination stride={output_row_stride}")

    total_input_size = rows * cols * input_row_stride
    lines.extend(load_large_int(temp_reg, total_input_size))
    lines.append(f"C_SET_SCALE_REG gp{temp_reg}")
    lines.extend(load_large_int(temp_reg, input_row_stride))
    lines.append(f"C_SET_STRIDE_REG gp{temp_reg}")

    for col in range(cols):
        for row in range(rows):
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
                    f"H_PREFETCH_V gp{dst_reg}, gp{src_reg}, a{input_hbm_base_addr_reg}, 1, 0"
                )

    return "\n".join(lines) + "\n"
