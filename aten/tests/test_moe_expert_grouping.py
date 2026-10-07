from __future__ import annotations

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.memory import MatrixBlockLayout


def test_hbm_storage_order_preserves_row_major_default() -> None:
    layout = MatrixBlockLayout("w", (128, 128), block_size=64)
    assert layout.storage_order == "row_major"
    assert layout.hbm_row_stride_elements == 128
    assert layout.get_sub_block(0, 1).hbm_offset == 64
    assert layout.get_sub_block(1, 0).hbm_offset == 8192


def test_tile_major_places_each_matrix_tile_contiguously() -> None:
    layout = MatrixBlockLayout(
        "w", (128, 128), block_size=64, storage_order="tile_major"
    )
    assert layout.hbm_row_stride_elements == 64
    assert layout.get_sub_block(0, 0).hbm_offset == 0
    assert layout.get_sub_block(0, 1).hbm_offset == 4096
    assert layout.get_sub_block(1, 0).hbm_offset == 8192
    assert layout.get_sub_block(1, 1).hbm_offset == 12288


def test_tile_major_rejects_partial_physical_tiles() -> None:
    try:
        MatrixBlockLayout("w", (65, 64), block_size=64, storage_order="tile_major")
    except ValueError as exc:
        assert "multiples of block_size=64" in str(exc)
    else:
        raise AssertionError("tile-major layout must reject partial physical tiles")


def _program() -> tuple[
    PlenaCompiler, object, tuple[object, object, object], tuple[object, ...]
]:
    prog = PlenaCompiler(mlen=64, blen=4, real_data_ratio=1.125)
    x = prog.alloc("x", rows=4, cols=64, strict=False, physical_shape=(4, 64))
    weights = (
        prog.input("w_gate", shape=(64, 64), hbm_storage_order="tile_major"),
        prog.input("w_up", shape=(64, 64), hbm_storage_order="tile_major"),
        prog.input("w_down", shape=(64, 64), hbm_storage_order="tile_major"),
    )
    zero = prog.fp_var("zero", size=1)
    limit_pos = prog.fp_var("limit_pos", size=4)
    limit_neg = prog.fp_var("limit_neg", size=4)
    one = prog.fp_var("one", size=4)
    neg_one = prog.fp_var("neg_one", size=4)
    return prog, x, weights, (zero, limit_pos, limit_neg, one, neg_one)


def _emit_group(pair_indices: list[int]) -> str:
    prog, x, weights, constants = _program()
    gathered = prog.moe_gather_token_rows_compact_from_vram_v0(
        x,
        token_indices=list(range(len(pair_indices))),
        hidden=64,
        zero_row=constants[0],
        name="grouped_gather",
    )
    route_scratch = prog.fp_var("route_scratch", size=64)
    prog.moe_dynamic_expert_group_v0(
        gathered,
        weights,
        weight_table_bases=(0, 4096, 8192),
        weight_table_strides=(4096, 4096, 4096),
        expert_indices_int_base=0,
        weights_fp_base=128,
        pair_indices=pair_indices,
        bias_tables=None,
        rows=len(pair_indices),
        intermediate=64,
        constants=constants,
        zero_row=constants[0],
        route_fp_scratch=route_scratch,
        activation_policy="standard_swiglu",
        name="grouped_expert",
    )
    return prog.compile()


def test_compact_gather_packs_rows_without_blen_holes() -> None:
    asm = _emit_group([1, 4, 7])
    assert "compact expert-major VRAM gather: rows=3" in asm
    assert "apply grouped route weights: rows=3" in asm
    assert "active_row=0" in asm
    assert "active_row=1" in asm
    assert "active_row=2" in asm


def test_group_fetches_one_weight_triple_for_multiple_pairs() -> None:
    asm = _emit_group([1, 4, 7])
    prefetch_markers = [
        line for line in asm.splitlines() if "dynamic HBM weight prefetch" in line
    ]
    assert len(prefetch_markers) == 3
    assert all("pair=1" in line for line in prefetch_markers)


def test_hbm_prefetch_is_attributed_after_dynamic_address_calculation() -> None:
    lines = _emit_group([1, 4, 7]).splitlines()
    for line_index, line in enumerate(lines):
        if not line.startswith("H_PREFETCH_M"):
            continue
        latest_stage = next(
            previous
            for previous in reversed(lines[:line_index])
            if previous.startswith("; @stage=")
        )
        assert latest_stage.startswith("; @stage=expert_weight_prefetch")


def test_group_rejects_duplicate_pair_indices() -> None:
    try:
        _emit_group([1, 1])
    except ValueError as exc:
        assert "distinct non-negative" in str(exc)
    else:
        raise AssertionError("duplicate pair indices must be rejected")


def _emit_buffered_projection(
    *,
    mode: str = "pingpong",
    mram_tile_capacity: int = 8,
) -> str:
    prog = PlenaCompiler(
        mlen=64,
        blen=4,
        real_data_ratio=1.125,
        mram_tile_capacity=mram_tile_capacity,
    )
    # Six four-tile panels force both the depth-two and depth-four paths to
    # consume a live slot before wrapping around and refilling it.
    k = 768
    x = prog.alloc("x", rows=4, cols=k, strict=False, physical_shape=(64, k))
    weight = prog.input("w", shape=(k, 128), hbm_storage_order="tile_major")
    prog.moe_dynamic_linear_projection_v0(
        x,
        weight,
        expert_indices_int_base=0,
        pair_idx=0,
        table_base=0,
        per_expert_stride=k * 128,
        name="buffered_projection",
        reuse_weight_across_row_blocks=True,
        weight_panel_mode=mode,
        panel_k_tiles=4,
    )
    return prog.compile()


def test_pingpong_prefetches_next_panel_before_consuming_current_panel() -> None:
    lines = _emit_buffered_projection().splitlines()
    prefetch_markers = [
        index
        for index, line in enumerate(lines)
        if "dynamic HBM weight prefetch" in line
    ]
    projection_markers = [
        index for index, line in enumerate(lines) if "pingpong resident weight" in line
    ]
    assert len(prefetch_markers) == 6
    assert len(projection_markers) == 6
    assert prefetch_markers[0] < prefetch_markers[1] < projection_markers[0]
    assert projection_markers[0] < prefetch_markers[2] < projection_markers[1]
    assert "mram=0" in lines[projection_markers[0]]
    assert "mram=16384" in lines[projection_markers[1]]
    assert "mram=0" in lines[projection_markers[2]]


def test_pingpong_requires_two_four_tile_panels() -> None:
    try:
        _emit_buffered_projection(mram_tile_capacity=4)
    except ValueError as exc:
        assert "exceed Matrix-SRAM capacity" in str(exc)
    else:
        raise AssertionError("pingpong must reject insufficient Matrix-SRAM capacity")


def test_blocking_panel_size_is_independent_of_total_mram_capacity() -> None:
    prog = PlenaCompiler(
        mlen=64,
        blen=4,
        real_data_ratio=1.125,
        mram_tile_capacity=16,
    )
    x = prog.alloc("x", rows=4, cols=512, strict=False, physical_shape=(64, 512))
    weight = prog.input("w", shape=(512, 64), hbm_storage_order="tile_major")
    prog.moe_dynamic_linear_projection_v0(
        x,
        weight,
        expert_indices_int_base=0,
        pair_idx=0,
        table_base=0,
        per_expert_stride=512 * 64,
        name="fixed_blocking_panel",
        reuse_weight_across_row_blocks=True,
        weight_panel_mode="blocking",
        panel_k_tiles=4,
    )
    asm = prog.compile()
    assert asm.count("dynamic HBM weight prefetch") == 2
