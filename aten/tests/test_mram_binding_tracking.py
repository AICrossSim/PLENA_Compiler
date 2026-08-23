from __future__ import annotations

from compiler.aten.plena import PlenaCompiler


def test_mram_reset_only_tracks_tiles_that_were_actually_bound() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    weight = prog.input(
        "wide_weight",
        shape=(64 * 32, 64 * 32),
        physical_shape=(64 * 32, 64 * 32),
    )
    prog._ensure_hbm_sub_matrix_registered(weight)
    layout = prog.get_hbm_layout(weight.name)
    first = layout.get_sub_block(0, 0)
    second = layout.get_sub_block(1, 1)

    prog.bind_mram_subblock(first, 0)
    prog.bind_mram_subblock(second, 64 * 64)
    prog.bind_mram_subblock(first, 2 * 64 * 64)

    assert len(prog._mram_bound_subblocks) == 2
    assert first.mram_addr == 2 * 64 * 64
    assert second.mram_addr == 64 * 64

    prog.clear_mram_bindings()

    assert first.mram_addr is None
    assert second.mram_addr is None
    assert prog._mram_bound_subblocks == []


def test_reset_does_not_scan_unbound_weight_tiles() -> None:
    class NoValuesScan(dict):
        def values(self):
            raise AssertionError("reset scanned an unbound weight layout")

    prog = PlenaCompiler(mlen=64, blen=4)
    unused = prog.input(
        "unused_weight",
        shape=(64 * 8, 64 * 8),
        physical_shape=(64 * 8, 64 * 8),
    )
    prog._ensure_hbm_sub_matrix_registered(unused)
    layout = prog.get_hbm_layout(unused.name)
    layout.sub_blocks = NoValuesScan(layout.sub_blocks)

    prog.clear_mram_bindings()
