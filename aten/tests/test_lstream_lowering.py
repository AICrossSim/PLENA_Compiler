from pathlib import Path

from compiler.assembler.assembly_to_binary import AssemblyToBinary
from compiler.aten.plena.compiler import PlenaCompiler
from compiler.aten.plena.instruction_stream import dynamic_count


ROOT = Path(__file__).resolve().parents[2]


def _loop_body(code: str) -> list[str]:
    lines = [line.strip() for line in code.splitlines() if line.strip() and not line.startswith(";")]
    start = next(i for i, line in enumerate(lines) if line.startswith("C_LOOP_START"))
    end = next(i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("C_LOOP_END"))
    return lines[start + 1 : end]


def test_fma_stream_lowering_removes_address_and_scalar_issue_from_loop(tmp_path):
    baseline = PlenaCompiler(mlen=64, blen=4)
    streamed = PlenaCompiler(mlen=64, blen=4, stream_addressing=True, stream_affine_alpha=1)
    rows = list(range(128))
    base_code = baseline.tile_row_fma_fp_sweep_asm(0, 8192, 0, [0] * 128, rows)
    stream_code = streamed.tile_row_fma_fp_sweep_asm(0, 8192, 0, [0] * 128, rows)

    assert _loop_body(base_code) == [
        "S_LD_FP f1, gp3, 0",
        "V_FMA_VF gp1, gp2, f1, 0",
        "S_ADDI_INT gp2, gp2, 64",
        "S_ADDI_INT gp3, gp3, 1",
    ]
    assert _loop_body(stream_code) == ["V_FMA_VF gp1, gp2, f1, 0, 7"]
    assert "L_CFG" in stream_code
    assert dynamic_count(stream_code) < dynamic_count(base_code)

    asm = tmp_path / "stream.asm"
    mem = tmp_path / "stream.mem"
    asm.write_text(stream_code)
    assembler = AssemblyToBinary(str(ROOT / "doc/operation.svh"), str(ROOT / "doc/configuration.svh"))
    words = assembler.generate_binary(str(asm), str(mem))
    assert words
    assert sum(word & 0x3F == 0x3F for word in words) > 0


def test_stream_lowering_is_opt_in_and_baseline_is_byte_stable():
    rows = list(range(8))
    before = PlenaCompiler(mlen=64, blen=4).tile_row_mul_fp_asm(0, [(row, row) for row in rows])
    after = PlenaCompiler(mlen=64, blen=4, stream_addressing=False).tile_row_mul_fp_asm(
        0, [(row, row) for row in rows]
    )
    assert before == after


def test_stream_lowering_is_explicit_for_vector_ops_and_map_uses_static_fallback():
    compiler = PlenaCompiler(mlen=64, blen=4, stream_addressing=True)
    rows = list(range(128))

    unary = compiler.tile_row_exp_asm(0, rows)
    binary = compiler.tile_row_add_asm(0, 8192, rows)
    mapped = compiler.tile_row_to_fpram_asm(0, [(row, row * 64) for row in rows])

    assert _loop_body(unary) == ["V_EXP_V gp1, gp1, 0, 0, 1"]
    assert _loop_body(binary) == ["V_ADD_VV gp1, gp1, gp2, 0, 3"]
    assert _loop_body(mapped) == [
        "S_MAP_FP_V gp2, gp1, 0",
        "S_ADDI_INT gp1, gp1, 64",
        "S_ADDI_INT gp2, gp2, 64",
    ]
    assert unary.count("L_CFG") > 0
    assert binary.count("L_CFG") > unary.count("L_CFG")
    assert mapped.count("L_CFG") == 0
    assert mapped.count("L_CFG") < binary.count("L_CFG")


def test_short_and_reverse_walks_fall_back_to_the_plain_static_path():
    compiler = PlenaCompiler(mlen=64, blen=4, stream_addressing=True)
    short = compiler.tile_row_exp_asm(0, list(range(8)))
    reverse = compiler.tile_row_exp_asm(0, list(reversed(range(128))))

    assert "L_CFG" not in short
    assert "L_CFG" not in reverse
    assert any(line.startswith("S_ADDI_INT") for line in _loop_body(short))
    assert any(line.startswith("S_ADDI_INT") for line in _loop_body(reverse))
