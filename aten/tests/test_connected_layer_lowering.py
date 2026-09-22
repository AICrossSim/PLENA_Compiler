"""Allocation and tail failures that corrupt connected layer programs."""

from dataclasses import replace

import pytest

from compiler.aten.plena.isa_matrix_projection import Projection, lower_b1_projection
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
