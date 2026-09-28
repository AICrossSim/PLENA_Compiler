"""Allocation and tail failures that corrupt connected layer programs."""

from dataclasses import replace

import pytest

from compiler.aten.plena.isa_matrix_projection import (
    Projection,
    lower_b1_projection,
    lower_resident_projection,
)
from compiler.aten.plena.recurrent_coefficients import (
    L2NormRows,
    lower_bf16_gather,
    lower_l2norm_rows,
    lower_pointwise_rows,
)


def projection():
    return Projection(0, 65536, 131072, 196608, 257, 65, 256)


def test_matrix_tail_dma_and_cross_k_partial_sum_lifetime():
    p = projection()
    packets = list(p.packets())
    assert [(c, k, r) for c, k, r, _ in packets] == [
        (0, 0, 256),
        (0, 256, 32),
        (32, 0, 256),
        (32, 256, 32),
        (64, 0, 256),
        (64, 256, 32),
    ]
    assert p.weight_bytes == 3 * (256 + 32) * 32 * 2
    assert p.input_values == 2304
    assembly = lower_b1_projection(p).splitlines()
    matrix = [line.split()[0] for line in assembly if line.startswith("M_")]
    assert matrix == ["M_MV", "M_MV", "M_MV_WO"] * 3
    assert sum(line.startswith("H_STORE_V ") for line in assembly) == 1


def test_projection_rejects_dma_overread_into_a_neighbor_allocation():
    p = projection()
    # 257 input values fit, but the last ordinary Vector DMA reads 2048.
    with pytest.raises(ValueError, match="overlap"):
        lower_b1_projection(replace(p, weights=1024))
    with pytest.raises(ValueError, match="aligned"):
        lower_b1_projection(replace(p, weights=65538))


def test_resident_projection_preserves_private_lifetimes_and_capacity():
    p = projection()
    for inputs, outputs in [([0, 0], [131072, 139264]), ([0, 8192], [131072, 8192])]:
        with pytest.raises(ValueError, match="overlap"):
            lower_resident_projection(p, inputs, outputs)
    with pytest.raises(ValueError, match="workspace"):
        lower_resident_projection(p, [0, 8192], [131072, 139264], vector_rows=2)
    with pytest.raises(ValueError, match="workspace"):
        lower_resident_projection(p, vector_rows=65)
    # A 65-packet K panel cannot occupy 65 banks, even when K < 16384.
    # Fall back to serial request packets instead of overcommitting Matrix.
    p = Projection(0, 1048576, 4194304, 8388608, 8193, 33, 128)
    code = lower_resident_projection(p, [0, 32768], [4194304, 4198400])
    assert code.count("@stage=matrix_projection_resident_rows") == 2


def test_gather_rejects_source_destruction_and_out_of_range_lanes():
    with pytest.raises(ValueError, match="aliases"):
        lower_bf16_gather([(0, 7)], 0, 8192, 12288)
    with pytest.raises(ValueError, match="out of range"):
        lower_bf16_gather([(0, 2048)], 4096, 8192, 12288)


def test_head_norm_uses_actual_group_width_and_private_output():
    p = L2NormRows(0, 8192, 4096, 16384, 32768, 0, 1, 512)
    assembly = lower_l2norm_rows(p)
    assert assembly.count("V_RED_SUM") == 8
    with pytest.raises(ValueError, match="overlap"):
        lower_l2norm_rows(replace(p, output=4096))
    with pytest.raises(ValueError, match="complete"):
        lower_l2norm_rows(replace(p, channels=4095))


def test_pointwise_exact_inplace_is_safe_but_partial_overlap_is_not():
    lower_pointwise_rows(0, 16384, 0, 2)
    with pytest.raises(ValueError, match="partially overlaps"):
        lower_pointwise_rows(0, 16384, 4096, 2)
    with pytest.raises(ValueError, match="static constants"):
        lower_pointwise_rows(
            65536, 81920, 4096, 1, sigmoid_input=True, constants_base=0
        )


def test_packed_kv_append_rejects_live_aliases_and_missing_masks():
    from compiler.aten.plena.recurrent_coefficients import lower_packed_matrix_append, packed_append_edits

    edits = packed_append_edits(65, 257, column=True)
    # The 65th key channel is in a second Vector row; a final partial
    # packet still needs its own preservation mask and RMW ownership.
    assert sum(map(len, edits.values())) == 65
    masks = {row: 0x100000 + i * 4096 for i, row in enumerate(edits)}
    args = (0, 65, 257, 0x20000, 0x30000, 0x40000, 0x50000)
    lower_packed_matrix_append(*args, column=True, keep_masks=masks)
    with pytest.raises(ValueError, match="preservation mask"):
        lower_packed_matrix_append(*args, column=True, keep_masks={})
    with pytest.raises(ValueError, match="overlap"):
        lower_packed_matrix_append(*args[:-1], 0, column=True, keep_masks=masks)
    with pytest.raises(ValueError, match="32-bit"):
        lower_packed_matrix_append(*args, column=True, keep_masks={row: 2**32 for row in edits})
