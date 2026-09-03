from pathlib import Path

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.matrix_recurrence_lowering import (
    BF16_BYTES,
    KIMI_KDA,
    NEMOTRON_MAMBA,
    ONE_MIB,
    MatrixSramPoint,
    RecurrenceLayout,
    build_matrix_recurrence_report,
    build_recurrence_field_manifest,
    build_recurrence_working_set,
    lower_matrix_recurrence,
    lowering_metrics,
    validate_recurrence_field_loads,
    validate_recurrence_output_stores,
)
from compiler.aten.plena.mview import MatrixViewFlags, validate_matrix_view_dominance
from compiler.aten.plena.program_mamba_common import Mamba2Shape


ROOT = Path(__file__).resolve().parents[2]


def test_official_shapes_and_plena_bf16_state_sizes_are_explicit() -> None:
    assert (NEMOTRON_MAMBA.heads, NEMOTRON_MAMBA.row_elements) == (64, 64)
    assert (KIMI_KDA.heads, KIMI_KDA.row_elements) == (96, 128)
    assert NEMOTRON_MAMBA.recurrence_rows == KIMI_KDA.recurrence_rows == 128
    assert NEMOTRON_MAMBA.state_bytes_per_layer == ONE_MIB
    assert KIMI_KDA.state_bytes_per_layer == 3 * ONE_MIB


@pytest.mark.parametrize(
    ("capacity_bytes", "spec", "fixed_heads", "affine_heads"),
    [
        (ONE_MIB, NEMOTRON_MAMBA, 32, 32),
        (ONE_MIB, KIMI_KDA, 16, 16),
        (2 * ONE_MIB, NEMOTRON_MAMBA, 32, 32),
        (2 * ONE_MIB, KIMI_KDA, 16, 16),
    ],
)
def test_capacity_point_sets_an_honest_head_group(
    capacity_bytes: int,
    spec,
    fixed_heads: int,
    affine_heads: int,
) -> None:
    point = MatrixSramPoint(capacity_bytes=capacity_bytes)
    for layout, group_heads in (
        (RecurrenceLayout.FIXED, fixed_heads),
        (RecurrenceLayout.AFFINE, affine_heads),
    ):
        working_set = build_recurrence_working_set(spec, layout=layout, point=point)
        assert working_set.group_heads == group_heads
        assert working_set.packet_values == group_heads * spec.row_elements
        assert working_set.capacity_facts["bank_words"] <= point.capacity_bank_words
        assert working_set.capacity_facts["max_bank_row"] <= point.depth_rows


@pytest.mark.parametrize(
    ("capacity_bytes", "spec", "expected_scalar_cols"),
    [
        (ONE_MIB, NEMOTRON_MAMBA, 64),
        (ONE_MIB, KIMI_KDA, 32),
        (2 * ONE_MIB, NEMOTRON_MAMBA, 64),
        (2 * ONE_MIB, KIMI_KDA, 32),
    ],
)
def test_recurrence_scalar_and_head_major_fields_match_their_consumers(
    capacity_bytes: int,
    spec,
    expected_scalar_cols: int,
) -> None:
    point = MatrixSramPoint(capacity_bytes=capacity_bytes)
    for layout in RecurrenceLayout:
        working_set = build_recurrence_working_set(spec, layout=layout, point=point)
        compact = {"beta"} if spec is KIMI_KDA else {"dt", "update", "c", "d"}
        for name in compact:
            descriptor = working_set.allocation(name).descriptor
            assert descriptor.shape.tile_count == 1
            assert descriptor.shape.cols == expected_scalar_cols
            assert descriptor.mapping.flags & MatrixViewFlags.BROADCAST_MINOR
            # One row stores one [a,b] pair per head at most; it must not
            # replicate each scalar across a complete recurrent row.
            assert descriptor.shape.cols < (working_set.group_heads * spec.row_elements)
        if spec is KIMI_KDA:
            for name, rows in (("decay", 2), ("key", 1), ("query", 1)):
                descriptor = working_set.allocation(name).descriptor
                assert descriptor.shape.tile_count == working_set.group_heads
                assert descriptor.shape.rows == rows
                assert descriptor.shape.cols >= working_set.state_rows_per_chunk
                assert descriptor.mapping.flags & MatrixViewFlags.BROADCAST_MINOR


def test_fixed_control_has_every_freedom_except_programmable_skew() -> None:
    point = MatrixSramPoint()
    expected_diagnostic_chunks = {NEMOTRON_MAMBA.name: 16, KIMI_KDA.name: 8}
    expected_fixed_chunks = {NEMOTRON_MAMBA.name: 64, KIMI_KDA.name: 32}
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        diagnostic = build_recurrence_working_set(
            spec,
            layout=RecurrenceLayout.FIXED_ROW_PITCH,
            point=point,
        )
        fixed = build_recurrence_working_set(spec, layout="fixed", point=point)
        affine = build_recurrence_working_set(spec, layout="affine", point=point)
        assert diagnostic.chunks == expected_diagnostic_chunks[spec.name]
        assert fixed.chunks == expected_fixed_chunks[spec.name]
        assert affine.chunks == 1
        for control in (diagnostic, fixed):
            assert all(
                allocation.descriptor.mapping.row_skew == 0
                and allocation.descriptor.mapping.tile_skew == 0
                for allocation in control.allocations
            )
        assert any(
            allocation.descriptor.mapping.tile_skew != 0
            for allocation in affine.allocations
        )


def test_every_active_recurrence_view_uses_uniform_bf16() -> None:
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        assert spec.state_bytes_per_layer == (
            spec.heads
            * spec.recurrence_rows
            * spec.row_elements
            * BF16_BYTES
        )
        for layout in RecurrenceLayout:
            working_set = build_recurrence_working_set(spec, layout=layout)
            assert working_set.point.element_bytes == BF16_BYTES
            assert all(
                not allocation.descriptor.mapping.flags & MatrixViewFlags.FP32
                for allocation in working_set.allocations
            )


def test_full_recurrence_emits_all_generic_primitives() -> None:
    report = build_matrix_recurrence_report()["models"]
    mamba = report[NEMOTRON_MAMBA.name]["capacity_points"][str(ONE_MIB)]
    kimi = report[KIMI_KDA.name]["capacity_points"][str(ONE_MIB)]

    assert mamba["affine"]["metrics"]["primitive_census"] == {
        "DOT_REDUCE": 2,
        "SCALE_ACCUM": 6,
    }
    assert kimi["affine"]["metrics"]["primitive_census"] == {
        "DOT_REDUCE": 12,
        "OUTER_UPDATE": 6,
        "SCALE_ACCUM": 12,
    }
    assert mamba["affine"]["metrics"]["l_tile_exec_count"] == 8
    assert kimi["affine"]["metrics"]["l_tile_exec_count"] == 30
    for variants in (mamba, kimi):
        for variant in variants.values():
            assert variant["metrics"]["contains_l_tile"] is True


def test_state_transfer_contract_is_explicit_and_has_no_cache_events() -> None:
    report = build_matrix_recurrence_report()["models"]
    mamba = report[NEMOTRON_MAMBA.name]["capacity_points"][str(ONE_MIB)]
    kimi = report[KIMI_KDA.name]["capacity_points"][str(ONE_MIB)]

    # Values count both directions. Mamba fixed and affine each load/store the
    # complete 1 MiB BF16 state once. KDA fixed needs two passes through the state;
    # affine keeps a complete group for both passes and only transfers it once.
    assert mamba["fixed"]["metrics"]["state_transfer_values"] == 2 * (
        NEMOTRON_MAMBA.state_bytes_per_layer // BF16_BYTES
    )
    assert mamba["affine"]["metrics"]["state_transfer_values"] == 2 * (
        NEMOTRON_MAMBA.state_bytes_per_layer // BF16_BYTES
    )
    assert kimi["fixed"]["metrics"]["state_transfer_values"] == 4 * (
        KIMI_KDA.state_bytes_per_layer // BF16_BYTES
    )
    assert kimi["affine"]["metrics"]["state_transfer_values"] == 2 * (
        KIMI_KDA.state_bytes_per_layer // BF16_BYTES
    )
    state_values = {
        "mamba": NEMOTRON_MAMBA.state_bytes_per_layer // BF16_BYTES,
        "kimi": KIMI_KDA.state_bytes_per_layer // BF16_BYTES,
    }
    assert mamba["fixed"]["metrics"]["state_transfer_values_by_direction"] == {
        "load": state_values["mamba"],
        "store": state_values["mamba"],
    }
    assert mamba["affine"]["metrics"]["state_transfer_values_by_direction"] == {
        "load": state_values["mamba"],
        "store": state_values["mamba"],
    }
    assert kimi["fixed"]["metrics"]["state_transfer_values_by_direction"] == {
        "load": state_values["kimi"],
        "reload_intermediate": state_values["kimi"],
        "store": state_values["kimi"],
        "store_intermediate": state_values["kimi"],
    }
    assert kimi["affine"]["metrics"]["state_transfer_values_by_direction"] == {
        "load": state_values["kimi"],
        "store": state_values["kimi"],
    }
    for model in report.values():
        for variants in model["capacity_points"].values():
            for variant in variants.values():
                assembly = variant["assembly"].lower()
                assert "cache_hit" not in assembly
                assert "cache_miss" not in assembly
                assert all(
                    not line.strip().startswith("x_state")
                    for line in assembly.splitlines()
                )


def test_state_transfers_are_real_viewed_dma_not_timing_comments() -> None:
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        assembly = lower_matrix_recurrence(
            spec,
            layout="affine",
            state_hbm_base=0x20000,
            hbm_address_register=2,
        )
        lines = assembly.splitlines()
        markers = [
            index
            for index, line in enumerate(lines)
            if line.startswith("; @matrix_state_")
        ]
        assert markers
        assert assembly.count("H_PREFETCH_V.MV") > 0
        assert assembly.count("H_STORE_V.MV") > 0
        for index in markers:
            marker = lines[index]
            assert "precision=bf16" in marker
            assert "hbm_byte_offset=" in marker
            next_marker = next(
                (
                    candidate
                    for candidate in range(index + 1, len(lines))
                    if lines[candidate].startswith("; @matrix_state_")
                    or lines[candidate].startswith("; @matrix_field_load")
                    or lines[candidate].startswith("; @l_tile_step=")
                ),
                len(lines),
            )
            transfer = "\n".join(lines[index:next_marker])
            assert "L_TILE_CFG 3" in transfer
            assert ("H_PREFETCH_V.MV" in transfer) ^ ("H_STORE_V.MV" in transfer)
            assert ", a2, 0, 2, 3" in transfer


def test_every_recurrence_field_is_loaded_into_a_consumer_view() -> None:
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        for layout in RecurrenceLayout:
            working_set = build_recurrence_working_set(spec, layout=layout)
            manifest = build_recurrence_field_manifest(
                working_set,
                field_hbm_base=spec.state_bytes_per_layer,
            )
            assembly = lower_matrix_recurrence(spec, layout=layout)
            assert validate_recurrence_field_loads(
                assembly,
                expected={
                    packet.key
                    for packet in manifest.packets
                    if packet.field != "output_result"
                },
            )
            assert assembly.count("H_PREFETCH_V.MV") > working_set.groups

    mamba = lower_matrix_recurrence(NEMOTRON_MAMBA, layout="affine")
    for field in ("x", "scratch_zero", "dt", "update", "c", "output_zero", "d"):
        assert f"@matrix_field_load field={field} " in mamba
    for step in (
        "mamba_dt_times_x",
        "mamba_state_decay_rank1_update",
        "mamba_c_readout",
        "mamba_skip",
    ):
        assert f"@l_tile_step={step}" in mamba

    kimi = lower_matrix_recurrence(KIMI_KDA, layout="affine")
    for field in (
        "prediction_zero",
        "decay",
        "key",
        "value",
        "beta",
        "output_zero",
        "query",
    ):
        assert f"@matrix_field_load field={field} " in kimi
    for step in (
        "kda_decay",
        "kda_prediction",
        "kda_beta_error",
        "kda_rank1_update",
        "kda_readout",
    ):
        assert f"@l_tile_step={step}" in kimi


def test_comment_only_field_claim_is_rejected() -> None:
    with pytest.raises(ValueError, match="comment-only"):
        validate_recurrence_field_loads(
            "; @matrix_field_write=x,dt layout=consumer_view\nL_TILE_EXEC gp1, gp2, gp3, 0\n"
        )


@pytest.mark.parametrize("spec", [NEMOTRON_MAMBA, KIMI_KDA])
@pytest.mark.parametrize("layout", list(RecurrenceLayout))
def test_every_head_group_output_is_stored_once_without_aliasing(spec, layout) -> None:
    working_set = build_recurrence_working_set(spec, layout=layout)
    assembly = lower_matrix_recurrence(spec, layout=layout)
    stores = validate_recurrence_output_stores(
        assembly,
        expected_groups=working_set.groups,
    )
    assert len(stores) == working_set.groups
    assert len(set(stores.values())) == working_set.groups
    assert lowering_metrics(assembly)["field_store_census"] == {
        "output_result": working_set.groups
    }


def test_missing_head_group_output_store_is_rejected() -> None:
    working_set = build_recurrence_working_set(
        NEMOTRON_MAMBA,
        layout=RecurrenceLayout.AFFINE,
    )
    assembly = lower_matrix_recurrence(
        NEMOTRON_MAMBA,
        layout=RecurrenceLayout.AFFINE,
    )
    marker = "; @matrix_field_store field=output_result"
    first = assembly.index(marker)
    next_group = assembly.index("; @head_group=1/", first)
    broken = assembly[:first] + assembly[next_group:]
    with pytest.raises(ValueError, match="output coverage differs"):
        validate_recurrence_output_stores(
            broken,
            expected_groups=working_set.groups,
        )


@pytest.mark.parametrize("spec", [NEMOTRON_MAMBA, KIMI_KDA])
def test_fixed_chunks_have_distinct_field_packets(spec) -> None:
    working_set = build_recurrence_working_set(spec, layout="fixed")
    assert working_set.chunks > 1
    manifest = build_recurrence_field_manifest(
        working_set,
        field_hbm_base=spec.state_bytes_per_layer,
    )
    chunk_fields = (
        ("update", "c") if spec is NEMOTRON_MAMBA else ("decay", "key", "query")
    )
    for group in range(working_set.groups):
        for field in chunk_fields:
            packets = [
                manifest.packet(field, group=group, chunk=chunk)
                for chunk in range(working_set.chunks)
            ]
            assert (
                len({packet.hbm_byte_offset for packet in packets})
                == working_set.chunks
            )
            assert all(packet.logical_values > 0 for packet in packets)


def test_complete_lowerings_assemble_and_views_dominate_every_exec(
    tmp_path: Path,
) -> None:
    assembler = AssemblyToBinary(
        str(ROOT / "doc" / "operation.svh"),
        str(ROOT / "doc" / "configuration.svh"),
    )
    for capacity_bytes in (ONE_MIB, 2 * ONE_MIB):
        point = MatrixSramPoint(capacity_bytes=capacity_bytes)
        for spec in (NEMOTRON_MAMBA, KIMI_KDA):
            for layout in RecurrenceLayout:
                assembly = lower_matrix_recurrence(spec, layout=layout, point=point)
                validate_matrix_view_dominance(assembly)
                assert lowering_metrics(assembly)["contains_l_tile"] is True
                stem = f"{spec.name}_{capacity_bytes}_{layout}"
                asm_path = tmp_path / f"{stem}.asm"
                mem_path = tmp_path / f"{stem}.mem"
                asm_path.write_text(assembly)
                assembler.generate_binary(str(asm_path), str(mem_path))
                words = [line for line in mem_path.read_text().splitlines() if line]
                assert len(words) == lowering_metrics(assembly)["static_instructions"]


def test_real_model_program_mixins_emit_l_tile_instead_of_a_side_report() -> None:
    mamba_program = PlenaCompiler(mlen=2048, blen=32, mram_tile_capacity=1)
    mamba_shape = Mamba2Shape(
        hidden_size=2688,
        num_heads=64,
        head_dim=64,
        state_size=128,
        n_groups=8,
        conv_kernel=4,
        chunk_size=128,
        seq_len=1,
    )
    emitted_mamba = mamba_program.ssm_decode_step_l_tile_v0(
        shape=mamba_shape,
        layout="affine",
    )
    assert "@stage=nemotron3_mamba2_matrix_recurrence" in emitted_mamba
    assert lowering_metrics(emitted_mamba)["l_tile_exec_count"] == 8

    kimi_program = PlenaCompiler(mlen=2048, blen=32, mram_tile_capacity=1)
    emitted_kimi = kimi_program.kda_decode_step_l_tile_v0(
        shape=KdaShape.kimi_k3(),
        layout="affine",
    )
    assert "@stage=kimi_k3_kda_matrix_recurrence" in emitted_kimi
    assert lowering_metrics(emitted_kimi)["l_tile_exec_count"] == 30


def test_model_program_mixins_reject_toy_shapes() -> None:
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=1)
    with pytest.raises(ValueError, match="L_TILE KDA decode expects"):
        program.kda_decode_step_l_tile_v0(
            shape=KdaShape(64, 2, 8, 8, 4),
            layout="affine",
        )
