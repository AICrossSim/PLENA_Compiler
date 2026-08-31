from pathlib import Path

from compiler.assembler.assembly_to_binary import AssemblyToBinary
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.affine_layout import AffineLayout, LayoutKind
from compiler.aten.plena.lstream import StreamConfigField


ROOT = Path(__file__).resolve().parents[2]


def _projection(*, affine: bool) -> str:
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=64)
    x_hbm = program.input(
        "x",
        shape=(4, 64),
        physical_shape=(4, 64),
        real_data_ratio=1.0,
    )
    x = program.load_batch(x_hbm, name="x_vram")
    weight = program.input(
        "weight",
        shape=(64, 64),
        physical_shape=(64, 64),
        real_data_ratio=1.0,
    )
    output = program.alloc(
        "output",
        4,
        64,
        strict=False,
        physical_shape=(4, 64),
    )
    layout = None
    if affine:
        layout = AffineLayout(
            kind=LayoutKind.AFFINE_SKEW,
            groups=1,
            fields=1,
            majors=4,
            minors=64,
            alpha=1,
        )
    program.vram_sub_projection_to(
        x,
        0,
        weight,
        0,
        output,
        0,
        0,
        output_layout=layout,
    )
    return program.get_code()


def test_projection_affine_writeback_is_opt_in_and_brackets_matrix_writeout(tmp_path):
    baseline = _projection(affine=False)
    affine = _projection(affine=True)

    assert "L_CFG" not in baseline
    assert "L_CFG" in affine
    writeout = affine.index("M_MM_WO")
    setup = affine.index("L_CFG")
    reset = affine.rindex(
        f", 3, {int(StreamConfigField.RESET)}"
    )
    assert setup < writeout < reset

    asm_path = tmp_path / "affine_projection.asm"
    mem_path = tmp_path / "affine_projection.mem"
    asm_path.write_text(affine)
    assembler = AssemblyToBinary(
        str(ROOT / "doc/operation.svh"),
        str(ROOT / "doc/configuration.svh"),
    )
    words = assembler.generate_binary(str(asm_path), str(mem_path))
    assert words
    assert sum(word & 0x3F == 0x3F for word in words) >= 2


def test_projection_write_layout_does_not_replace_the_static_result_pointer():
    affine = _projection(affine=True)
    # Matrix writeback owns slot 3 by convention. Existing scalar address
    # arithmetic remains authoritative; the view changes only physical bank
    # placement at M_MM_WO.
    flags_cfg = [
        line
        for line in affine.splitlines()
        if line.startswith("L_CFG")
        and line.endswith(f", {int(StreamConfigField.FLAGS)}")
    ]
    assert len(flags_cfg) == 1
    assert "S_ADDI_INT" in affine
    assert "C_LOOP_START" in affine


def test_wide_k_projection_applies_affine_layout_only_at_final_writeback():
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=1)
    x_hbm = program.input("wide_x", shape=(4, 128), physical_shape=(4, 128))
    x = program.load_batch(x_hbm, name="wide_x_vram")
    weight = program.input("wide_w", shape=(128, 64), physical_shape=(128, 64))
    layout = AffineLayout(LayoutKind.AFFINE_SKEW, 1, 1, 4, 64, alpha=1)

    program.linear_projection(x, weight, name="wide_y", output_layout=layout)
    code = program.get_code()

    assert "VRAM Sub Projection microtile accumulate" in code
    assert "VRAM Matrix Add" not in code
    assert code.count("M_MM_WO") == 16
    # One setup/reset pair covers all sixteen 4x4 microtiles in the output tile.
    cfg_fields = [
        int(line.rsplit(",", 1)[1])
        for line in code.splitlines()
        if line.startswith("L_CFG")
    ]
    assert cfg_fields.count(int(StreamConfigField.RESET)) == 2
    assert cfg_fields.count(int(StreamConfigField.FLAGS)) == 1
