"""Guards for the decode-only compact row-major Matrix lowering."""

from __future__ import annotations

from pathlib import Path

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import parse_asm_file
from compiler.aten.plena import PlenaCompiler


REPO_ROOT = Path(__file__).resolve().parents[2]


def _program(
    *, compact: bool, weight_base: int = 0x1_2345_0000
) -> tuple[PlenaCompiler, object, object]:
    prog = PlenaCompiler(
        mlen=64,
        blen=4,
        mram_tile_capacity=4,
        compact_matrix_loops=compact,
    )
    hidden = prog.load_batch(
        prog.input(
            "hidden",
            shape=(1, 5 * 64),
            physical_shape=(4, 5 * 64),
            prestaged_vram_addr=0,
        ),
        name="hidden",
    )
    weight = prog.input(
        "weight",
        shape=(5 * 64, 6 * 64),
        physical_shape=(5 * 64, 6 * 64),
        hbm_addr=weight_base,
    )
    return prog, hidden, weight


def _instruction_lines(assembly: str) -> list[str]:
    return [
        line.strip()
        for line in assembly.splitlines()
        if line.strip() and not line.lstrip().startswith(";")
    ]


def test_compact_projection_rolls_n_and_reduces_static_machine_code() -> None:
    compact, compact_hidden, compact_weight = _program(compact=True)
    compact.linear_projection(compact_hidden, compact_weight, name="compact_out")
    compact_asm = compact.compile()

    static, static_hidden, static_weight = _program(compact=False)
    static.linear_projection(static_hidden, static_weight, name="static_out")
    static_asm = static.compile()

    assert (
        "compact row-major Matrix projection compact_out: Ktiles=5, Ntiles=6"
        in compact_asm
    )
    assert (
        len(_instruction_lines(compact_asm)) < len(_instruction_lines(static_asm)) / 3
    )
    # Two K chunks each carry one N loop; the loop body itself is emitted once.
    assert compact_asm.count("compact K chunk") == 2
    assert compact_asm.count("H_PREFETCH_M") == 5


def test_compact_slice_uses_the_requested_bf16_weight_columns() -> None:
    prog, hidden, weight = _program(compact=True)
    output = prog.linear_projection_slice(
        hidden,
        weight,
        output_col_offset=2 * 64,
        output_features=2 * 64,
        name="bf16_slice",
        physical_shape=(64, 2 * 64),
        matrix_precision="keyvalue",
        set_scale=False,
        hbm_element_bytes=2,
    )
    assembly = prog.compile()

    assert output.shape == (1, 2 * 64)
    assert "compact row-major Matrix projection bf16_slice" in assembly
    assert "C_SET_SCALE_REG" not in assembly
    assert any(
        line.startswith("H_PREFETCH_M") and line.endswith(", 1")
        for line in _instruction_lines(assembly)
    )


def test_compact_stream_k_router_writes_once_after_all_chunks() -> None:
    prog, hidden, weight = _program(compact=True)
    prog.linear_projection_bf16_stream_k_accum(
        hidden,
        weight,
        name="router_logits",
    )
    assembly = prog.compile()

    assert "compact BF16 stream-K Matrix projection router_logits" in assembly
    assert assembly.count("stream-K chunk") == 2
    # M_MM_WO sits inside the N/micro-column loops but appears once in code.
    assert assembly.count("M_MM_WO") == 1
    assert assembly.count("H_PREFETCH_M") == 5


def test_compact_projection_with_64_bit_weight_base_assembles(tmp_path: Path) -> None:
    prog, hidden, weight = _program(
        compact=True,
        weight_base=0x12_3456_7000,
    )
    prog.linear_projection(hidden, weight, name="high_hbm_out")
    assembly_path = tmp_path / "compact_high_hbm.asm"
    assembly_path.write_text(prog.compile())
    instructions = parse_asm_file(str(assembly_path))
    assembler = AssemblyToBinary(
        str(REPO_ROOT / "doc" / "operation.svh"),
        str(REPO_ROOT / "doc" / "configuration.svh"),
    )

    words = [assembler._convert_to_binary(line) for line in instructions]
    assert words
    assert all(0 <= word < 1 << 32 for word in words)


def test_streaming_assembler_matches_the_legacy_list_path(tmp_path: Path) -> None:
    prog, hidden, weight = _program(compact=True)
    prog.linear_projection(hidden, weight, name="streamed_out")
    assembly_path = tmp_path / "streamed.asm"
    assembly_path.write_text(prog.compile())
    legacy_path = tmp_path / "legacy.mem"
    streaming_path = tmp_path / "streaming.mem"
    assembler = AssemblyToBinary(
        str(REPO_ROOT / "doc" / "operation.svh"),
        str(REPO_ROOT / "doc" / "configuration.svh"),
    )

    legacy_words = assembler.generate_binary(str(assembly_path), str(legacy_path))
    streaming_count = assembler.generate_binary_streaming(
        str(assembly_path),
        str(streaming_path),
    )

    assert streaming_count == len(legacy_words)
    assert streaming_path.read_bytes() == legacy_path.read_bytes()
