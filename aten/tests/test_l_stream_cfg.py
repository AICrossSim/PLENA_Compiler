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
