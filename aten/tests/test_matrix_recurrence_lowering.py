from pathlib import Path

from assembler.assembly_to_binary import AssemblyToBinary
from compiler.aten.plena.matrix_recurrence_lowering import (
    KIMI_KDA,
    NEMOTRON_MAMBA,
    build_matrix_recurrence_report,
    lower_matrix_recurrence,
    lowering_metrics,
)


ROOT = Path(__file__).resolve().parents[2]


def test_official_packets_fill_the_paper_matrix_width() -> None:
    assert (NEMOTRON_MAMBA.packet_heads, NEMOTRON_MAMBA.row_elements) == (32, 64)
    assert (KIMI_KDA.packet_heads, KIMI_KDA.row_elements) == (16, 128)
    assert NEMOTRON_MAMBA.packet_values == KIMI_KDA.packet_values == 2048
    assert (
        NEMOTRON_MAMBA.affine_alpha(bank_width=32),
        KIMI_KDA.affine_alpha(bank_width=32),
    ) == (2, 4)


def test_layout_changes_no_instruction_or_arithmetic_count() -> None:
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        fixed = lowering_metrics(lower_matrix_recurrence(spec, affine=False))
        affine = lowering_metrics(lower_matrix_recurrence(spec, affine=True))
        assert fixed == affine


def test_packet_counts_come_from_the_full_official_recurrence() -> None:
    report = build_matrix_recurrence_report()["models"]
    mamba = report[NEMOTRON_MAMBA.name]["metrics"]
    kimi = report[KIMI_KDA.name]["metrics"]

    assert (mamba["packet_reads"], mamba["packet_writes"]) == (1536, 512)
    assert (kimi["packet_reads"], kimi["packet_writes"]) == (6144, 1536)
    assert sum(
        count
        for opcode, count in mamba["opcode_census"].items()
        if opcode in {"V_MUL_VV", "V_ADD_VV", "V_MUL_VV.MV", "V_ADD_VV.MV"}
    ) == 1024
    assert sum(
        count
        for opcode, count in kimi["opcode_census"].items()
        if opcode in {"V_MUL_VV", "V_ADD_VV", "V_MUL_VV.MV", "V_ADD_VV.MV"}
    ) == 4608


def test_real_recurrence_fields_have_distinct_packet_addresses() -> None:
    mamba = lower_matrix_recurrence(NEMOTRON_MAMBA, affine=True)
    assert "V_MUL_VV.MV gp1, gp1, gp2" in mamba
    assert "V_ADD_VV.MV gp1, gp1, gp3" in mamba
    assert "V_MUL_VV.MV gp5, gp1, gp4" in mamba
    assert "S_ADDI_INT gp4, gp4, 65536" in mamba

    kimi = lower_matrix_recurrence(KIMI_KDA, affine=True)
    assert "V_MUL_VV.MV gp5, gp1, gp3" in kimi
    assert "S_ADDI_INT gp3, gp3, 32768" in kimi


def test_both_layouts_assemble_to_canonical_machine_words(tmp_path: Path) -> None:
    assembler = AssemblyToBinary(
        str(ROOT / "doc" / "operation.svh"),
        str(ROOT / "doc" / "configuration.svh"),
    )
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        lengths = []
        for affine in (False, True):
            assembly = lower_matrix_recurrence(spec, affine=affine)
            asm_path = tmp_path / f"{spec.name}_{affine}.asm"
            mem_path = tmp_path / f"{spec.name}_{affine}.mem"
            asm_path.write_text(assembly)
            assembler.generate_binary(str(asm_path), str(mem_path))
            words = [line for line in mem_path.read_text().splitlines() if line]
            assert words
            lengths.append(len(words))
        assert lengths[0] == lengths[1]
