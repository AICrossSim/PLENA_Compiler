import pytest
from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import parse_asm_file


def test_dot_encoding_is_explicit_and_canonical(tmp_path):
    asm = AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")
    path = tmp_path / "dot.asm"
    path.write_text("V_DOT_RESET gp0, gp0, gp0, 0\nV_DOT_ACC gp0, gp2, gp3, 0\nV_DOT_WRITE gp1, gp0, gp0, 0\n")
    words = [asm._convert_to_binary(i) for i in parse_asm_file(str(path))]
    assert words == [0x0D | (8 << 22), 0x11 | (2 << 10) | (3 << 14) | (8 << 22), 0x0F | (1 << 6) | (8 << 22)]
    for line in ["V_DOT_RESET gp1, gp0, gp0, 0", "V_DOT_ACC gp1, gp2, gp3, 0", "V_DOT_WRITE gp1, gp2, gp0, 0", "V_DOT_ACC gp0, gp2, gp3, 1"]:
        path.write_text(line + "\n")
        with pytest.raises(ValueError, match="unused registers and mask"):
            asm._convert_to_binary(parse_asm_file(str(path))[0])


def test_kda_dot_optin_preserves_legacy_default():
    from compiler.aten.plena.matrix_recurrence_lowering import KIMI_KDA
    from compiler.aten.plena.prepared_vector_recurrence import PreparedVectorGroup, lower_prepared_vector_recurrence
    fields = {name: (i + 4) * 1024 * 1024 for i, name in enumerate(("value", "decay", "key", "query", "beta", "zero", "output"))}
    groups = tuple(PreparedVectorGroup(g * 524288, fields) for g in range(6))
    legacy = lower_prepared_vector_recurrence(KIMI_KDA, groups)
    explicit = lower_prepared_vector_recurrence(KIMI_KDA, groups, experimental_fp32_dot=False)
    promoted = lower_prepared_vector_recurrence(KIMI_KDA, groups, experimental_fp32_dot=True)
    assert legacy == explicit and "V_DOT" not in legacy
    assert promoted.count("V_DOT_RESET") == promoted.count("V_DOT_WRITE") == 12
    assert promoted.count("V_DOT_ACC") == 12 * 128
