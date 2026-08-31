"""Guards that recurrent math, not just a layout micro-test, consumes packets."""

from __future__ import annotations

from pathlib import Path

import pytest

from compiler.assembler.assembly_to_binary import AssemblyToBinary
from compiler.aten.plena.hybrid_compile_report import (
    AssemblyMetrics,
    kimi_k3_mixer_assembly,
    nemotron3_mamba_decode_assembly,
)
from compiler.aten.plena.lstream import StreamConfigField, StreamFlags


ROOT = Path(__file__).resolve().parents[2]


def _packet_flag_lines(assembly: str) -> list[str]:
    marker = f", {int(StreamConfigField.FLAGS)}"
    return [
        line
        for line in assembly.splitlines()
        if line.startswith("L_CFG") and line.endswith(marker)
    ]


def _packet_setup(assembly: str, marker: str) -> list[str]:
    lines = assembly.splitlines()
    start = next(index for index, line in enumerate(lines) if marker in line)
    end = next(
        index
        for index in range(start + 1, len(lines))
        if lines[index].startswith("C_LOOP_START")
    )
    return lines[start:end]


def _small_config_immediate(
    setup: list[str], field: StreamConfigField, *, default: int | None = None
) -> int:
    indices = [
        index
        for index, line in enumerate(setup)
        if line.startswith("L_CFG") and line.endswith(f", {int(field)}")
    ]
    if not indices:
        assert default is not None
        return default
    index = indices[0]
    setter = setup[index - 1]
    assert setter.startswith("S_ADDI_INT")
    return int(setter.rsplit(",", 1)[1])


def test_nemotron_recurrence_uses_packetized_decay_and_rank_update() -> None:
    assembly = nemotron3_mamba_decode_assembly(
        stream=True, affine=True, packetized=True
    )
    assert "Packetized multi-row Mul" in assembly
    assert "Packetized multi-row FMA" in assembly
    assert f", {int(StreamConfigField.PACKET_STRIDE)}" in assembly
    assert _packet_flag_lines(assembly)
    assert "X_STATE" not in assembly


def test_kimi_recurrence_packetizes_update_but_not_cross_row_reductions() -> None:
    assembly = kimi_k3_mixer_assembly(stream=True, affine=True, packetized=True)
    assert "Packetized multi-row Mul" in assembly
    assert "Packetized multi-row FMA" in assembly
    # Prediction and readout have a pinned destination. They must remain regular
    # reduction loops instead of treating repeated destination lanes as independent.
    assert "predict head=" in assembly
    assert "kda_readout" in assembly
    assert "X_STATE" not in assembly


def test_packet_mode_is_a_stream_flag_not_a_model_specific_opcode() -> None:
    assert int(StreamFlags.PACKETIZED) == 1 << 7
    for builder in (nemotron3_mamba_decode_assembly, kimi_k3_mixer_assembly):
        assembly = builder(stream=True, affine=False, packetized=True)
        assert "MAMBA_STEP" not in assembly
        assert "KDA_STEP" not in assembly
        assert "L_CFG" in assembly


def test_affine_rotation_is_bound_only_to_the_physically_skewed_state() -> None:
    assembly = nemotron3_mamba_decode_assembly(
        stream=True, affine=True, packetized=True
    )
    for marker in ("Packetized multi-row Mul", "Packetized multi-row FMA"):
        setup = _packet_setup(assembly, marker)
        alpha_writes = [
            line
            for line in setup
            if line.startswith("L_CFG")
            and line.endswith(f", {int(StreamConfigField.ALPHA)}")
        ]
        # The moving state is skewed. The pinned source and segmented FPRAM
        # scalars remain identity-layout operands; rotating them would silently
        # change the recurrence values.
        assert len(alpha_writes) == 1


def test_real_shape_report_counts_only_packet_fed_vector_operations() -> None:
    nemotron = AssemblyMetrics.from_assembly(
        nemotron3_mamba_decode_assembly(stream=True, affine=True, packetized=True)
    )
    kimi = AssemblyMetrics.from_assembly(
        kimi_k3_mixer_assembly(stream=True, affine=True, packetized=True)
    )

    assert nemotron.packetized_opcode_census == {
        "V_FMA_VF": 8 * 8 * 128,
        "V_MUL_VF": 8 * 8 * 128,
    }
    assert kimi.packetized_opcode_census == {
        "V_FMA_VF": 96 * 2 * 128,
        "V_MUL_VF": 96 * 2 * 128,
    }


def test_paper_width_packet_coalesces_exact_real_shape_element_work() -> None:
    kwargs = {
        "stream": True,
        "affine": True,
        "packetized": True,
        "packet_elements": 2048,
        "storage_atom": 64,
        "blen": 32,
    }
    nemotron_assembly = nemotron3_mamba_decode_assembly(**kwargs)
    kimi_assembly = kimi_k3_mixer_assembly(**kwargs)
    nemotron = AssemblyMetrics.from_assembly(nemotron_assembly)
    kimi = AssemblyMetrics.from_assembly(kimi_assembly)

    # Nemotron: 64 heads x 128 state rows x 64 values, for decay and update.
    assert nemotron.packetized_opcode_census == {
        "V_FMA_VF": 64 * 128 * 64 // 2048,
        "V_MUL_VF": 64 * 128 * 64 // 2048,
    }
    # Kimi K3: 96 heads x 128 key rows x 128 values, for decay and update.
    assert kimi.packetized_opcode_census == {
        "V_FMA_VF": 96 * 128 * 128 // 2048,
        "V_MUL_VF": 96 * 128 * 128 // 2048,
    }
    for assembly in (nemotron_assembly, kimi_assembly):
        setup = _packet_setup(assembly, "Packetized multi-row Mul")
        logical_base = _small_config_immediate(setup, StreamConfigField.BASE)
        physical_base = _small_config_immediate(
            setup, StreamConfigField.PHYSICAL_BASE_ROW, default=0
        )
        assert logical_base == physical_base * 2048


@pytest.mark.parametrize(
    "name,builder",
    [
        ("nemotron3", nemotron3_mamba_decode_assembly),
        ("kimi_k3", kimi_k3_mixer_assembly),
    ],
)
def test_real_shape_packet_recurrence_assembles_to_machine_words(
    tmp_path: Path, name: str, builder
) -> None:
    assembly = builder(stream=True, affine=True, packetized=True)
    asm_path = tmp_path / f"{name}.asm"
    mem_path = tmp_path / f"{name}.mem"
    asm_path.write_text(assembly)
    words = AssemblyToBinary(
        str(ROOT / "doc/operation.svh"),
        str(ROOT / "doc/configuration.svh"),
    ).generate_binary(str(asm_path), str(mem_path))

    assert words
    assert sum((word & 0x3F) == 0x3F for word in words) > 0
    assert mem_path.stat().st_size > 0


@pytest.mark.parametrize(
    "name,builder,row_elements",
    [
        ("nemotron3", nemotron3_mamba_decode_assembly, 64),
        ("kimi_k3", kimi_k3_mixer_assembly, 128),
    ],
)
def test_paper_width_packet_recurrence_assembles_to_machine_words(
    tmp_path: Path, name: str, builder, row_elements: int
) -> None:
    assembly = builder(
        stream=True,
        affine=True,
        packetized=True,
        packet_elements=2048,
        storage_atom=64,
        blen=32,
        recurrent_row_elements=row_elements,
    )
    asm_path = tmp_path / f"{name}_paper2048.asm"
    mem_path = tmp_path / f"{name}_paper2048.mem"
    asm_path.write_text(assembly)
    words = AssemblyToBinary(
        str(ROOT / "doc/operation.svh"),
        str(ROOT / "doc/configuration.svh"),
    ).generate_binary(str(asm_path), str(mem_path))

    assert words
    assert sum((word & 0x3F) == 0x3F for word in words) > 0
    assert mem_path.stat().st_size > 0
