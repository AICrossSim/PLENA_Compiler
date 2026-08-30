from pathlib import Path

import pytest

from compiler.assembler.assembly_to_binary import AssemblyToBinary
from compiler.aten.plena.affine_layout import AffineLayout, LayoutKind
from compiler.aten.plena.lstream import (
    L_STREAM_CFG_OPCODE,
    StreamBinding,
    StreamConfigField,
    decode_l_stream_cfg_word,
    emit_stream_configuration,
    encode_l_stream_cfg_word,
)


ROOT = Path(__file__).resolve().parents[2]


def test_l_stream_cfg_golden_word_and_assembler_agree(tmp_path):
    expected = encode_l_stream_cfg_word(
        value_register=7,
        target_register=3,
        slot=2,
        field=StreamConfigField.ALPHA,
    )
    assert expected & 0x3F == L_STREAM_CFG_OPCODE
    assert decode_l_stream_cfg_word(expected) == (7, 3, 2, StreamConfigField.ALPHA)

    asm_path = tmp_path / "lstream.asm"
    mem_path = tmp_path / "lstream.mem"
    asm_path.write_text("L_STREAM_CFG gp7, gp3, 2, 8\n")
    assembler = AssemblyToBinary(str(ROOT / "doc/operation.svh"), str(ROOT / "doc/configuration.svh"))
    assert assembler.generate_binary(str(asm_path), str(mem_path)) == [expected]


@pytest.mark.parametrize("slot", [4, 7, 15])
def test_l_stream_cfg_rejects_unimplemented_slots(slot):
    with pytest.raises(ValueError, match="slot"):
        encode_l_stream_cfg_word(
            value_register=1,
            target_register=2,
            slot=slot,
            field=StreamConfigField.BASE,
        )


def test_l_stream_cfg_rejects_noncanonical_high_bits():
    word = encode_l_stream_cfg_word(
        value_register=1,
        target_register=2,
        slot=0,
        field=StreamConfigField.BASE,
    )
    with pytest.raises(ValueError, match="canonical"):
        decode_l_stream_cfg_word(word | (1 << 31))


def test_l_stream_cfg_field_15_encodes_packet_stride():
    word = encode_l_stream_cfg_word(
        value_register=1,
        target_register=2,
        slot=3,
        field=StreamConfigField.PACKET_STRIDE,
    )
    assert word == 0x003C_C87C
    assert decode_l_stream_cfg_word(word) == (
        1,
        2,
        3,
        StreamConfigField.PACKET_STRIDE,
    )


def test_configuration_is_model_independent_and_enables_last():
    layout = AffineLayout(
        LayoutKind.AFFINE_SKEW,
        groups=8,
        fields=4,
        majors=8,
        minors=64,
        alpha=1,
        beta=4,
        gamma=2,
    )
    binding = StreamBinding(
        slot=0,
        target_register=3,
        target_is_fp=False,
        base=4096,
        advance=64,
        packet_elements=64,
        storage_atom=4,
    )
    code = emit_stream_configuration(value_gp=7, binding=binding, layout=layout).render()
    cfg_lines = [line for line in code.splitlines() if line.startswith("L_STREAM_CFG")]
    assert len(cfg_lines) == 13
    assert cfg_lines[0].endswith(f", {int(StreamConfigField.RESET)}")
    assert cfg_lines[-1].endswith(f", {int(StreamConfigField.FLAGS)}")
    assert "MAMBA" not in code.upper()
    assert "KDA" not in code.upper()


def test_linear_configuration_elides_default_dimensions_and_zero_coefficients():
    layout = AffineLayout(LayoutKind.ROW_MAJOR, 1, 1, 8, 64)
    binding = StreamBinding(
        slot=0,
        target_register=3,
        target_is_fp=False,
        base=1024,
        advance=64,
        packet_elements=64,
        storage_atom=4,
    )
    code = emit_stream_configuration(value_gp=7, binding=binding, layout=layout).render()
    cfg_lines = [line for line in code.splitlines() if line.startswith("L_STREAM_CFG")]
    fields = {int(line.rsplit(",", 1)[1]) for line in cfg_lines}
    assert fields == {
        int(StreamConfigField.RESET),
        int(StreamConfigField.BASE),
        int(StreamConfigField.EXTENT_MINOR),
        int(StreamConfigField.EXTENT_MAJOR),
        int(StreamConfigField.ADVANCE),
        int(StreamConfigField.PACKET_ELEMENTS),
        int(StreamConfigField.STORAGE_ATOM),
        int(StreamConfigField.FLAGS),
    }


def test_packet_configuration_emits_stride_and_packet_flag():
    layout = AffineLayout(LayoutKind.AFFINE_SKEW, 1, 1, 16, 64, alpha=1)
    binding = StreamBinding(
        slot=0,
        target_register=3,
        target_is_fp=False,
        base=1024,
        advance=4,
        packet_elements=64,
        storage_atom=4,
        packet_stride=64,
        packetized=True,
    )
    code = emit_stream_configuration(value_gp=7, binding=binding, layout=layout).render()
    assert f", {int(StreamConfigField.PACKET_STRIDE)}" in code
    flags_line = next(
        line
        for line in code.splitlines()
        if line.startswith("L_STREAM_CFG")
        and line.endswith(f", {int(StreamConfigField.FLAGS)}")
    )
    assert flags_line


def test_configuration_legalizes_real_scale_addresses_before_assembly(tmp_path):
    """A full-model VRAM base is not guaranteed to fit S_ADDI_INT's 18 bits."""

    layout = AffineLayout(LayoutKind.ROW_MAJOR, 1, 1, 8, 64)
    binding = StreamBinding(
        slot=0,
        target_register=3,
        target_is_fp=False,
        base=(1 << 22) + 0x345,
        advance=64,
        packet_elements=64,
        storage_atom=4,
    )
    code = emit_stream_configuration(value_gp=7, binding=binding, layout=layout).render()

    assert "S_LUI_INT gp7" in code
    assert f"S_ADDI_INT gp7, gp0, {binding.base}" not in code
    asm_path = tmp_path / "large_lstream.asm"
    mem_path = tmp_path / "large_lstream.mem"
    asm_path.write_text(code)
    assembler = AssemblyToBinary(
        str(ROOT / "doc/operation.svh"), str(ROOT / "doc/configuration.svh")
    )
    words = assembler.generate_binary(str(asm_path), str(mem_path))
    assert any(word & 0x3F == L_STREAM_CFG_OPCODE for word in words)
