from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from aten.plena import (
    FullModelProgram,
    PlenaCompiler,
    SymbolicHbmBinding,
    assert_registers_are_free,
    reserve_expert_weight_table,
    validate_symbolic_hbm_bindings,
)
from aten.state.hbm_image import assemble_words
from compiler.aten.plena.artifact import write_symbolic_decode_artifact


def test_full_model_summary_reports_physical_artifact_sizes() -> None:
    program = FullModelProgram(
        model="test",
        phase="decode",
        layer_counts={"state": 1},
        assembly="S_ADDI_INT gp1, gp0, 1\n",
        instruction_count=1,
        descriptor_base=0x1000,
        descriptor_image=bytes(256),
        stage_instruction_counts={"state": 1},
        layout_descriptor_base=0x2000,
        layout_descriptor_image=bytes(256),
    )

    summary = program.to_dict()

    assert summary["contract"] == "plena.full_model_program/v1"
    assert summary["machine_code_bytes"] == 4
    assert summary["descriptor_count"] == 1
    assert summary["layout_descriptor_count"] == 1


def _symbolic_program(
    *bindings: SymbolicHbmBinding, hbm_size: int = 256
) -> FullModelProgram:
    return FullModelProgram(
        model="test",
        phase="decode",
        layer_counts={"state": 1},
        assembly="",
        instruction_count=0,
        descriptor_base=0,
        descriptor_image=bytes(256),
        stage_instruction_counts={},
        hbm_size=hbm_size,
        symbolic_hbm_bindings=bindings,
    )


def _binding(name: str, address: int, size: int) -> SymbolicHbmBinding:
    return SymbolicHbmBinding(
        name=name,
        hbm_addr=address,
        byte_size=size,
        logical_shape=(size,),
        physical_shape=(size,),
        storage_format="bf16_le",
    )


def test_symbolic_hbm_manifest_is_unambiguous_and_bounded() -> None:
    program = _symbolic_program(_binding("a", 64, 32), _binding("b", 128, 64))

    validate_symbolic_hbm_bindings(program)
    manifest = program.symbolic_hbm_manifest()

    assert manifest["weights_bound"] is False
    assert manifest["binding_count"] == 2
    assert manifest["bindings"][1]["hbm_end"] == 192


@pytest.mark.parametrize(
    ("bindings", "match"),
    [
        ((_binding("same", 0, 16), _binding("same", 32, 16)), "duplicate"),
        ((_binding("a", 0, 48), _binding("b", 32, 16)), "overlap"),
        ((_binding("a", 240, 32),), "beyond HBM span"),
    ],
)
def test_symbolic_hbm_manifest_rejects_invalid_ranges(
    bindings: tuple[SymbolicHbmBinding, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_symbolic_hbm_bindings(_symbolic_program(*bindings))


def test_symbolic_artifact_writer_assembles_and_hashes_every_image(
    tmp_path: Path,
) -> None:
    program = _symbolic_program(_binding("weight", 64, 32))
    program = replace(
        program,
        assembly="S_ADDI_INT gp1, gp0, 7\n",
        instruction_count=1,
        fpram_preload=(0.0, 1.0),
    )

    summary = write_symbolic_decode_artifact(
        program,
        tmp_path,
        stem="tiny",
        artifact_contract="test.artifact/v1",
        scope="unit_test",
        max_machine_code_mib=1.0,
        claims={"machine_code_is_legal": True, "weights_are_bound": False},
    )

    assert (tmp_path / "tiny.mem").read_text().startswith("0x")
    assert summary["machine_code_raw_bytes"] == 4
    assert summary["symbolic_hbm_binding_count"] == 1
    assert len(summary["sha256"]["machine_code"]) == 64


def test_layer_boundary_rejects_a_live_register() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    assert_registers_are_free(prog, "empty program")

    leaked = prog.register_allocator.allocate_gp(1)
    with pytest.raises(AssertionError, match="left registers allocated"):
        assert_registers_are_free(prog, "test layer")

    prog.register_allocator.free_gp(leaked)
    assert_registers_are_free(prog, "released layer")


def test_tile_major_expert_table_assembles_above_16_gib() -> None:
    """Runtime expert selection must not materialize a 64-bit HBM base in one GP."""
    prog = PlenaCompiler(mlen=64, blen=4)
    prog._next_hbm_addr = (1 << 34) + 123
    table = reserve_expert_weight_table(
        prog,
        name="large_expert_table",
        num_experts=896,
        rows=64,
        cols=64,
    )

    # One MXFP8 tile is 4,096 element bytes plus 512 scale bytes. The 896
    # dynamic offsets fit in an aligned 4-MiB tile group.
    assert prog.hbm_tensor_size(64 * 64) == 4_608
    assert table.tile_group_stride == 4 * 1024 * 1024
    assert table.base > 1 << 34
    assert table.base % table.tile_group_stride == 0
    assert prog._next_hbm_addr == table.base + table.tile_group_stride

    prog._moe_dynamic_load_sub_matrix_col_v0(
        weight_template=table.template,
        col_idx=0,
        expert_indices_int_base=0,
        pair_idx=0,
        table_base=table.base,
        per_expert_stride=table.stride,
        num_experts=table.num_experts,
        tile_group_stride=table.tile_group_stride,
        name="large_expert_load",
    )
    assembly = prog.compile()

    assert "tile-major pair=0, experts=896, group=4194304" in assembly
    assert re.search(r"S_ADDI_INT gp\d+, gp0, 4608$", assembly, re.MULTILINE)
    assert "C_SET_ADDR_REG" in assembly
    words = assemble_words(assembly)
    assert words
    assert all(0 < word <= 0xFFFF_FFFF for word in words)


@pytest.mark.parametrize("num_experts", [0, -1])
def test_expert_table_rejects_nonpositive_expert_counts(num_experts: int) -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    with pytest.raises(ValueError, match="num_experts must be positive"):
        reserve_expert_weight_table(
            prog,
            name="invalid_expert_table",
            num_experts=num_experts,
            rows=64,
            cols=64,
        )
