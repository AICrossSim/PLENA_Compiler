"""PLENA backend stubs for linear projection operators."""

import math


MAX_K_TILES = 4  # MRAM capacity: 4 x mlen^2 elements


def _iter_k_chunks(num_k_tiles: int):
    k_start = 0
    while k_start < num_k_tiles:
        k_end = min(k_start + MAX_K_TILES, num_k_tiles)
        yield k_start, k_end - k_start
        k_start = k_end


def linear_projection_plena(prog, input_var, weight_var, name: str = "linear_out"):
    """Emit tiled PLENA linear projection, including K-split accumulation."""
    mlen = prog.mlen

    rows, k_total = input_var.shape
    _, out_features = weight_var.shape
    num_row_blocks = math.ceil(rows / mlen)
    assert out_features % mlen == 0, (
        f"out_features ({out_features}) must be a multiple of mlen ({mlen})"
    )
    num_col_blocks = out_features // mlen
    num_k_tiles = math.ceil(k_total / mlen)

    # Allocate output matrix.  When batch is not a multiple of mlen we pass
    # strict=False so the allocator doesn't reject the shape; the hardware will
    # operate on full mlen-wide tiles (HBM zero-pads unused rows) and only the
    # first `rows` rows of the output contain valid results.
    output_strict = rows % mlen == 0
    output = prog.alloc(name, rows, out_features, strict=output_strict)

    def emit_projection(row_idx, col_idx, target, target_row_idx, target_col_idx, **k_split):
        prog.vram_sub_projection_to(
            input_var,
            row_idx,
            weight_var,
            col_idx,
            target,
            target_row_idx,
            target_col_idx,
            **k_split,
        )

    if num_k_tiles <= MAX_K_TILES:
        for col_idx in range(num_col_blocks):
            for row_idx in range(num_row_blocks):
                emit_projection(row_idx, col_idx, output, row_idx, col_idx)
    else:
        # Temp buffer for partial sums — only needs one (mlen, mlen) tile since
        # each sub_projection_to writes a single tile before accumulating.
        # Using the full output shape would cause VRAM overlap with the output
        # when out_features > mlen (column-block-major layout).
        temp = prog.alloc(f"{name}_temp", mlen, mlen)

        for k_chunk_idx, (k_block_start, k_block_count) in enumerate(_iter_k_chunks(num_k_tiles)):
            k_split = {
                "k_block_start": k_block_start,
                "k_block_count": k_block_count,
            }
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    if k_chunk_idx == 0:
                        emit_projection(row_idx, col_idx, output, row_idx, col_idx, **k_split)
                    else:
                        emit_projection(row_idx, col_idx, temp, 0, 0, **k_split)
                        prog.vram_block_add_to(
                            output,
                            row_idx,
                            col_idx,
                            temp,
                            0,
                            0,
                            output,
                            row_idx,
                            col_idx,
                        )
        prog.free_tensor(temp)

    return output


def linear_plena(prog, input_var, weight_var):
    return linear_projection_plena(prog, input_var, weight_var)
