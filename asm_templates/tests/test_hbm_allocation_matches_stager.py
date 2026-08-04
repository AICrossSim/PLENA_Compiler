"""The compiler's HBM addresses must be the ones the stager writes to.

`create_mem_for_sim` lays tensors down back to back as `[elements][scales]`
pairs, each plane padded only up to an HBM row. `PlenaCompiler._allocate_hbm`
must advance its bump cursor by exactly that footprint: any extra padding puts
every following tensor at an address the stager never wrote, so weights and the
KV cache are read from a neighbour's bytes and decode to arbitrary magnitudes.

That is not a graceful failure. At MLEN=256 the allocator used to round every
allocation up to an `MLEN * MLEN` boundary, which put W_Q 286,720 bytes past its
staged position; the projection then read weights peaking at 2.3e19 and the
first `M_BTMM` drain wrote infinity into the score buffer.
"""

from __future__ import annotations

import pytest

from aten.plena import PlenaCompiler

# (MLEN, BLEN) rungs of the decode geometry ladder.
GEOMETRIES = ((64, 4), (128, 8), (256, 16), (512, 64), (1024, 32))

# One decode layer's HBM tensors as element counts, parameterised by MLEN.
def tensor_elements(mlen: int, kv_size: int = 512, vocab: int = 256) -> list[int]:
    hidden = mlen
    inter = 2 * mlen
    batch = mlen
    return [
        batch * hidden,      # X
        batch * hidden,      # QROT
        batch * hidden,      # COS
        batch * hidden,      # SIN
        batch * mlen,        # KROT
        hidden * hidden,     # W_Q
        hidden * mlen,       # W_K
        hidden * mlen,       # W_V
        hidden * hidden,     # W_O
        kv_size * mlen,      # K
        kv_size * mlen,      # V
        hidden * inter,      # W_gate
        hidden * inter,      # W_up
        inter * hidden,      # W_down
        vocab * hidden,      # W_lm_head
    ]


def staged_offsets(program: PlenaCompiler, elements: list[int]) -> list[int]:
    """Where the stager puts each tensor: the running sum of its footprints."""
    offsets = []
    cursor = 0
    for count in elements:
        offsets.append(cursor)
        cursor += program.hbm_tensor_size(count)
    return offsets


@pytest.mark.parametrize("mlen,blen", GEOMETRIES)
def test_allocator_matches_the_staged_layout(mlen: int, blen: int) -> None:
    program = PlenaCompiler(mlen=mlen, blen=blen)
    elements = tensor_elements(mlen)
    expected = staged_offsets(program, elements)
    got = [
        program._allocate_hbm(program.hbm_tensor_size(count)) for count in elements
    ]
    assert got == expected, (
        f"MLEN={mlen}: allocator diverges from the staged layout at tensor "
        f"{next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b)}"
    )


@pytest.mark.parametrize("mlen,blen", GEOMETRIES)
def test_no_gap_is_left_between_tensors(mlen: int, blen: int) -> None:
    """The cursor advances by the footprint exactly, with no padding."""
    program = PlenaCompiler(mlen=mlen, blen=blen)
    for count in tensor_elements(mlen):
        size = program.hbm_tensor_size(count)
        start = program._allocate_hbm(size)
        following = program._allocate_hbm(program.hbm_tensor_size(count))
        assert following - start == size, (
            f"MLEN={mlen}: a {size}-byte tensor advanced the cursor by "
            f"{following - start}"
        )


def test_footprint_is_element_rows_plus_scale_rows() -> None:
    """The footprint is the stager's own rule, not an approximation of it."""
    program = PlenaCompiler(mlen=256, blen=16)
    row_bytes = program.hbm_row_width // 8
    for count in (4096, 65536, 131072):
        element_bytes = -(-count * program.hbm_element_width // 8 // row_bytes) * row_bytes
        scale_bytes = (
            -(-(count // program.hbm_block_size) * program.hbm_scale_width // 8 // row_bytes)
            * row_bytes
        )
        assert program.hbm_tensor_size(count) == element_bytes + scale_bytes
