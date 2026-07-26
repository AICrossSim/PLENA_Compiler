from __future__ import annotations

from pathlib import Path

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import parse_asm_file
from compiler.aten.agu import (
    decode_agu_stride,
    encode_agu_stride,
    optimize_agu_assembly,
)
from compiler.aten.cost_emitter import (
    CostTrace,
    ScheduleInstruction,
    ScheduleRepeat,
    ScheduleSequence,
    optimize_cost_trace_loop_agu,
)


def test_agu_stride_round_trip() -> None:
    for stride in (-16_777_216, -6144, -3, 1, 3, 2048, 6_291_456, 16_777_216):
        encoded = encode_agu_stride(stride)
        assert encoded is not None
        assert decode_agu_stride(encoded) == stride


def test_agu_rewrites_profitable_tail_updates() -> None:
    source = """\
S_ADDI_INT gp1, gp0, 0
S_ADDI_INT gp2, gp0, 64
C_LOOP_START gp7, 8
V_ADD_VV gp3, gp1, gp2
S_ADDI_INT gp1, gp1, 2048
S_ADDI_INT gp2, gp2, 262143
S_ADDI_INT gp2, gp2, 262143
S_ADDI_INT gp2, gp2, 2
C_LOOP_END gp7
"""
    optimized, stats = optimize_agu_assembly(source)
    assert "C_AGU_BIND gp1" in optimized
    assert "C_AGU_BIND gp2" in optimized
    assert "C_AGU_LOOP_LEN 1" in optimized
    assert "C_LOOP_START_AGU gp7, 8" in optimized
    assert "S_ADDI_INT gp1, gp1" not in optimized
    assert "S_ADDI_INT gp2, gp2" not in optimized
    assert optimized.count("C_LOOP_END gp7") == 1
    assert stats["agu_loop_count"] == 1
    assert stats["dynamic_loop_end_elided"] == 8
    assert stats["agu_large_immediate_chunks_elided"] == 16


def test_agu_keeps_non_tail_or_overwritten_registers() -> None:
    source = """\
C_LOOP_START gp7, 8
S_ADDI_INT gp1, gp1, 64
V_ADD_VV gp3, gp1, gp2
    C_LOOP_END gp7
"""
    optimized, stats = optimize_agu_assembly(source)
    assert "S_ADDI_INT gp1, gp1, 64" in optimized
    assert "C_AGU_BIND" not in optimized
    assert "C_LOOP_START_AGU gp7, 8" in optimized
    assert stats["agu_stream_count_histogram"] == {"0": 1}
    assert stats["agu_affine_updates_elided"] == 0


def test_agu_allows_earlier_write_that_clears_the_previous_offset() -> None:
    source = """\
C_LOOP_START gp7, 8
S_ADDI_INT gp1, gp1, 16
V_ADD_VV gp3, gp1, gp2
S_ADDI_INT gp1, gp1, 64
C_LOOP_END gp7
"""
    optimized, stats = optimize_agu_assembly(source)
    assert "S_ADDI_INT gp1, gp1, 16" in optimized
    assert "S_ADDI_INT gp1, gp1, 64" not in optimized
    assert "C_AGU_BIND gp1" in optimized
    assert stats["agu_affine_updates_elided"] == 8


def test_agu_refolds_exact_unrolled_microkernel() -> None:
    source = """\
C_LOOP_START gp7, 3
V_SHIFT_V gp3, gp1, gp5
S_ADDI_INT gp3, gp3, 2048
S_ADDI_INT gp1, gp1, 2048
V_SHIFT_V gp3, gp1, gp5
S_ADDI_INT gp3, gp3, 2048
S_ADDI_INT gp1, gp1, 2048
V_SHIFT_V gp3, gp1, gp5
S_ADDI_INT gp3, gp3, 2048
S_ADDI_INT gp1, gp1, 2048
C_LOOP_END gp7
"""
    optimized, stats = optimize_agu_assembly(source)
    assert "C_LOOP_START_AGU gp0, 3" in optimized
    assert optimized.count("V_SHIFT_V") == 1
    assert "S_ADDI_INT gp3, gp3" not in optimized
    assert "S_ADDI_INT gp1, gp1" not in optimized
    assert stats["agu_refolded_loop_count"] == 1


def test_agu_refolds_root_level_exact_repeat() -> None:
    source = """\
V_ADD_VV gp3, gp3, gp4
S_ADDI_INT gp3, gp3, 16
V_ADD_VV gp3, gp3, gp4
S_ADDI_INT gp3, gp3, 16
V_ADD_VV gp3, gp3, gp4
S_ADDI_INT gp3, gp3, 16
V_ADD_VV gp3, gp3, gp4
S_ADDI_INT gp3, gp3, 16
"""
    optimized, stats = optimize_agu_assembly(source)
    assert "C_LOOP_START_AGU gp0, 4" in optimized
    assert optimized.count("V_ADD_VV") == 1
    assert "S_ADDI_INT gp3, gp3, 16" not in optimized
    assert stats["agu_refolded_loop_count"] == 1


def test_agu_selects_six_streams_and_leaves_the_rest() -> None:
    body = ["C_LOOP_START gp15, 16", "V_ADD_VV gp14, gp1, gp2"]
    body.extend(f"S_ADDI_INT gp{index}, gp{index}, {index}" for index in range(1, 9))
    body.append("C_LOOP_END gp15")
    optimized, stats = optimize_agu_assembly("\n".join(body) + "\n")
    assert optimized.count("C_AGU_BIND") == 6
    assert optimized.count("S_ADDI_INT") == 2
    assert stats["agu_stream_count_histogram"] == {"6": 1}


def test_legacy_mode_is_byte_identical() -> None:
    source = "C_LOOP_START gp2, 4\nS_ADDI_INT gp1, gp1, 64\nC_LOOP_END gp2\n"
    optimized, stats = optimize_agu_assembly(source, mode="legacy")
    assert optimized == source
    assert stats["agu_mode"] == "legacy"


def test_cost_trace_agu_rebuilds_dynamic_and_static_counts() -> None:
    stage = "layer/test"
    body = ScheduleSequence(
        (
            ScheduleInstruction("V_ADD_VV", ("gp3", "gp1", "gp2"), stage),
            ScheduleInstruction("S_ADDI_INT", ("gp1", "gp1", "64"), stage),
            ScheduleInstruction("S_ADDI_INT", ("gp1", "gp1", "64"), stage),
            ScheduleInstruction("C_LOOP_END", ("gp7",), stage),
        )
    )
    trace = CostTrace(
        schedule=ScheduleSequence(
            (
                ScheduleInstruction("C_LOOP_START", ("gp7", "8"), stage),
                ScheduleRepeat(8, body, "gp7", "hardware_loop"),
            )
        )
    )
    trace.stages[stage].dynamic_opcodes.update(
        {"C_LOOP_START": 1, "V_ADD_VV": 8, "S_ADDI_INT": 16, "C_LOOP_END": 8}
    )
    trace.stages[stage].static_opcodes.update(
        {"C_LOOP_START": 1, "V_ADD_VV": 1, "S_ADDI_INT": 2, "C_LOOP_END": 1}
    )
    trace.dynamic_opcodes.update(trace.stages[stage].dynamic_opcodes)
    trace.static_opcodes.update(trace.stages[stage].static_opcodes)

    optimized = optimize_cost_trace_loop_agu(trace)

    assert optimized.dynamic_opcodes == {
        "V_ADD_VV": 8,
        "C_AGU_BIND": 1,
        "C_AGU_LOOP_LEN": 1,
        "C_LOOP_START_AGU": 1,
    }
    assert optimized.static_opcodes["C_LOOP_END"] == 1
    assert optimized.metadata["agu_projected_cycle_savings"] == 22


def test_cost_trace_agu_refolds_compile_time_repeat() -> None:
    stage = "layer/ffn"
    body = ScheduleSequence(
        (
            ScheduleInstruction("V_ADD_VV", ("gp3", "gp3", "gp4"), stage),
            ScheduleInstruction("S_ADDI_INT", ("gp3", "gp3", "64"), stage),
        )
    )
    trace = CostTrace(
        schedule=ScheduleSequence(
            (ScheduleRepeat(32, body, "ffn_accumulate", "compile_time"),)
        )
    )
    trace.stages[stage].dynamic_opcodes.update(
        {"V_ADD_VV": 32, "S_ADDI_INT": 32}
    )
    trace.stages[stage].static_opcodes.update(
        {"V_ADD_VV": 32, "S_ADDI_INT": 32}
    )
    trace.dynamic_opcodes.update(trace.stages[stage].dynamic_opcodes)
    trace.static_opcodes.update(trace.stages[stage].static_opcodes)

    optimized = optimize_cost_trace_loop_agu(trace)

    assert optimized.dynamic_opcodes == {
        "V_ADD_VV": 32,
        "C_AGU_BIND": 1,
        "C_AGU_LOOP_LEN": 1,
        "C_LOOP_START_AGU": 1,
    }
    assert optimized.metadata["agu_refolded_loop_count"] == 1


def test_agu_aliases_assemble_to_reserved_opcodes(tmp_path: Path) -> None:
    compiler_root = Path(__file__).resolve().parents[2]
    source = tmp_path / "agu.asm"
    source.write_text(
        "C_AGU_BIND gp3, 917556\n"
        "C_AGU_LOOP_LEN 17\n"
        "C_LOOP_START_AGU gp7, 1234\n"
        "C_LOOP_END gp7\n"
    )
    assembler = AssemblyToBinary(
        str(compiler_root / "doc/operation.svh"),
        str(compiler_root / "doc/configuration.svh"),
    )
    instructions = parse_asm_file(str(source))
    words = [assembler._convert_to_binary(item) for item in instructions]

    assert [word & 0x3F for word in words] == [0x3E, 0x3E, 0x3F, 0x30]
    assert (words[0] >> 6) & 0xF == 3
    assert (words[1] >> 6) & 0xF == 0
    assert words[1] >> 10 == 17
    assert (words[2] >> 6) & 0xF == 7
