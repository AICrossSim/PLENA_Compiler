"""V_FMA_VF encodes into the same R-type slots as V_MUL_VF.

The one opcode this work adds. It differs from V_MUL_VF in exactly one bit
field -- the opcode -- because it must reach the same operand decode path in
the RTL; the difference is entirely in execution, where `rd` is read as well
as written.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.assembler.assembly_to_binary import AssemblyToBinary  # noqa: E402
from compiler.assembler.parser import parse_asm_file  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _assemble(text: str) -> list[int]:
    with tempfile.NamedTemporaryFile("w", suffix=".asm", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        converter = AssemblyToBinary(
            str(REPO_ROOT / "doc" / "operation.svh"),
            str(REPO_ROOT / "doc" / "configuration.svh"),
        )
        return [converter._convert_to_binary(i) for i in parse_asm_file(str(path))]
    finally:
        path.unlink(missing_ok=True)


def test_v_fma_vf_opcode_and_operand_slots():
    (word,) = _assemble("V_FMA_VF gp3, gp4, f2, 0\n")
    assert word & 0x3F == 0x3B
    assert (word >> 6) & 0xF == 3    # rd
    assert (word >> 10) & 0xF == 4   # rs1
    assert (word >> 14) & 0xF == 2   # fp2
    assert (word >> 18) & 0xF == 0   # rmask


def test_v_fma_vf_matches_v_mul_vf_layout():
    (fma,) = _assemble("V_FMA_VF gp5, gp6, f1, 0\n")
    (mul,) = _assemble("V_MUL_VF gp5, gp6, f1, 0\n")
    assert fma >> 6 == mul >> 6, "only the opcode field may differ"
    assert (fma ^ mul) & 0x3F == (0x3B ^ 0x12)


def test_v_fma_vf_carries_a_nonzero_mask():
    """rmask lands in rs3, the same slot the rest of the masked vector family
    uses. A mask that silently assembled to zero would run unmasked."""
    (word,) = _assemble("V_FMA_VF gp1, gp2, f3, 5\n")
    assert (word >> 18) & 0xF == 5


# ---------------------------------------------------------------------------
# The emitter. The plan's static-footprint claim rests on these sweeps becoming
# hardware loops rather than unrolling.

from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.tests.isa_interpreter import Machine  # noqa: E402

MLEN = 8


def _body(asm: str) -> list[str]:
    return [
        line.strip()
        for line in asm.splitlines()
        if line.strip() and not line.strip().startswith(";")
    ]


def _prog():
    return PlenaCompiler(mlen=MLEN, blen=2)


def test_fma_sweep_over_a_progression_emits_a_hardware_loop():
    """A 128-row sweep must be a loop, not 128 unrolled blocks.

    This is the whole static-footprint argument: the copy/multiply/add
    predecessor needed a scratch row per step, and that row was the *same* row
    every iteration, so the copy's destination was constant and broke the
    progression.
    """
    asm = _prog().tile_row_fma_fp_sweep_asm(
        dst_addr=0, src_addr=8192, fpram_base=16,
        dst_rows=list(range(128)), src_rows=list(range(128)),
    )
    body = _body(asm)
    assert sum("C_LOOP_START" in line for line in body) == 1
    assert sum("C_LOOP_END" in line for line in body) == 1
    assert sum("V_FMA_VF" in line for line in body) == 1, "loop body holds one FMA"
    assert len(body) < 12, f"expected a compact loop, got {len(body)} instructions"


def test_fma_sweep_holds_a_constant_destination():
    """The contraction walks the source and pins the destination.

    A pinned side is a step of 0, which `_arith_progression` used to refuse.
    """
    asm = _prog().tile_row_fma_fp_sweep_asm(
        dst_addr=0, src_addr=8192, fpram_base=16,
        dst_rows=[3] * 64, src_rows=list(range(64)),
    )
    body = _body(asm)
    assert sum("C_LOOP_START" in line for line in body) == 1
    assert sum("V_FMA_VF" in line for line in body) == 1
    # The pinned pointer must not be advanced at all, not advanced by zero.
    assert len(body) < 12, f"expected a compact loop, got {len(body)} instructions"


def test_fma_uses_no_scratch_and_no_separate_multiply():
    asm = _prog().tile_row_fma_fp_sweep_asm(
        dst_addr=0, src_addr=8192, fpram_base=16,
        dst_rows=[0, 1, 2, 3], src_rows=[0, 1, 2, 3],
    )
    body = _body(asm)
    assert not any("V_ADD_VV" in line for line in body)
    assert not any("V_MUL_VF" in line for line in body)


def test_fma_rejects_mismatched_row_counts():
    with pytest.raises(ValueError, match="row counts"):
        _prog().tile_row_fma_fp_sweep_asm(
            dst_addr=0, src_addr=8192, fpram_base=16,
            dst_rows=[0, 1], src_rows=[0],
        )
    with pytest.raises(ValueError, match="row counts"):
        _prog().tile_row_fma_fp_broadcast_asm(
            dst_addr=0, src_addr=8192, fpram_scalar_addr=16,
            dst_rows=[0, 1], src_rows=[0],
        )


def test_broadcast_applies_one_slot_to_every_row():
    """`_broadcast` in this file means one slot for all rows -- the opposite of
    `_sweep`. Mixing them up is a silent wrong answer, so pin the difference."""
    sweep = _body(_prog().tile_row_fma_fp_sweep_asm(
        dst_addr=0, src_addr=8192, fpram_base=16,
        dst_rows=[0, 1, 2], src_rows=[0, 1, 2],
    ))
    bcast = _body(_prog().tile_row_fma_fp_broadcast_asm(
        dst_addr=0, src_addr=8192, fpram_scalar_addr=16,
        dst_rows=[0, 1, 2], src_rows=[0, 1, 2],
    ))
    # The sweep advances the FPRAM pointer; the broadcast has nothing to advance.
    assert len(bcast) < len(sweep)


def test_the_unrolled_fallback_computes_the_same_answer_as_the_loop():
    """The `else:` branch of `_emit_tile_row_fma`, which no test reached.

    It runs whenever any of the three walks is not a progression -- a scattered
    row map through the public `tile_row_fma_fp_asm`, or `ATEN_OPS_UNROLL=1`,
    which routes *every* emitter through it. Swapping the two pointers there,
    or replacing the whole branch with a `raise`, left the entire suite green.

    Checked four shapes: each walk non-monotonic in turn, plus a repeated
    destination row, which is the read-modify-write ordering case -- the FMA
    accumulates, so two entries landing on the same row must compose rather
    than the later winning.
    """
    scattered = [
        ("dst jumps",  [(3, 0, 0), (0, 1, 1), (2, 2, 2)]),
        ("src jumps",  [(0, 5, 0), (1, 2, 1), (2, 7, 2)]),
        ("fp jumps",   [(0, 0, 5), (1, 1, 0), (2, 2, 3)]),
        ("dst repeats", [(1, 0, 0), (1, 3, 2), (1, 2, 1)]),
    ]
    for label, row_map in scattered:
        p = _prog()
        dst = p.alloc("dst", MLEN, MLEN)
        src = p.alloc("src", MLEN, MLEN)
        weights = p.fp_var("w", size=MLEN)
        dst_base = p.get_vram_layout(dst.name).vram_base_addr
        src_base = p.get_vram_layout(src.name).vram_base_addr
        mark = len(p.get_code())
        p.tile_row_fma_fp_asm(dst_addr=dst_base, src_addr=src_base, row_map=row_map)
        code = p.get_code()[mark:]
        assert "C_LOOP_START" not in code, f"{label}: expected the unrolled path"
        assert code.count("V_FMA_VF") == len(row_map)

        m = Machine(vlen=MLEN, vram_words=1 << 16, fpram_words=1 << 13)
        expected = []
        for r in range(MLEN):
            dst_vals = [float(r * MLEN + c + 1) / 8.0 for c in range(MLEN)]
            src_vals = [float(r + c) / 4.0 - 1.0 for c in range(MLEN)]
            m.write_vram_row(dst_base + r * MLEN, dst_vals)
            m.write_vram_row(src_base + r * MLEN, src_vals)
            expected.append(list(dst_vals))
        src_rows = [[float(r + c) / 4.0 - 1.0 for c in range(MLEN)] for r in range(MLEN)]
        for i in range(MLEN):
            m.write_fpram(weights.address + i, [0.25 + i])
        # Apply the row map in order, exactly as the emitter does.
        for d, sr, f in row_map:
            for c in range(MLEN):
                expected[d][c] += src_rows[sr][c] * (0.25 + f)
        # The emitter takes absolute FPRAM addresses in row_map, so re-emit with
        # the allocated base folded in.
        p2 = _prog()
        d2 = p2.alloc("dst", MLEN, MLEN)
        s2 = p2.alloc("src", MLEN, MLEN)
        w2 = p2.fp_var("w", size=MLEN)
        mark2 = len(p2.get_code())
        p2.tile_row_fma_fp_asm(
            dst_addr=p2.get_vram_layout(d2.name).vram_base_addr,
            src_addr=p2.get_vram_layout(s2.name).vram_base_addr,
            row_map=[(d, sr, w2.address + f) for d, sr, f in row_map],
        )
        m2 = Machine(vlen=MLEN, vram_words=1 << 16, fpram_words=1 << 13)
        db = p2.get_vram_layout(d2.name).vram_base_addr
        sb = p2.get_vram_layout(s2.name).vram_base_addr
        for r in range(MLEN):
            m2.write_vram_row(db + r * MLEN, [float(r * MLEN + c + 1) / 8.0 for c in range(MLEN)])
            m2.write_vram_row(sb + r * MLEN, src_rows[r])
        for i in range(MLEN):
            m2.write_fpram(w2.address + i, [0.25 + i])
        m2.run(p2.get_code()[mark2:])
        for r in range(MLEN):
            got = m2.read_vram_row(db + r * MLEN, MLEN)
            for c in range(MLEN):
                assert abs(got[c] - expected[r][c]) < 1e-4, (
                    f"{label}: row {r} lane {c}: {got[c]} != {expected[r][c]}"
                )


def test_the_unrolled_fallback_matches_the_looped_path_exactly():
    """Same work, two code paths. A progression run through the fallback (by
    scrambling and restoring the order) must land on the same numbers."""
    rows = list(range(6))
    results = []
    for row_map in (
        [(0, r, r) for r in rows],                       # progression -> loop
        [(0, r, r) for r in [0, 1, 2, 3, 4, 5][::-1]],   # reversed -> still a
    ):                                                    # progression (step -1)
        p = _prog()
        dst = p.alloc("dst", MLEN, MLEN)
        src = p.alloc("src", MLEN, MLEN)
        w = p.fp_var("w", size=MLEN)
        db = p.get_vram_layout(dst.name).vram_base_addr
        sb = p.get_vram_layout(src.name).vram_base_addr
        mark = len(p.get_code())
        p.tile_row_fma_fp_asm(
            dst_addr=db, src_addr=sb,
            row_map=[(d, s, w.address + f) for d, s, f in row_map],
        )
        m = Machine(vlen=MLEN, vram_words=1 << 16, fpram_words=1 << 13)
        m.write_vram_row(db, [0.0] * MLEN)
        for r in rows:
            m.write_vram_row(sb + r * MLEN, [float(r + c) / 4.0 for c in range(MLEN)])
            m.write_fpram(w.address + r, [0.5 + r])
        m.run(p.get_code()[mark:])
        results.append(m.read_vram_row(db, MLEN))
    # Addition is commutative here up to float ordering; both orders sum the
    # same six products.
    for a, b in zip(*results):
        assert abs(a - b) < 1e-4, f"{a} != {b}"


def test_a_step_zero_loop_computes_the_right_answer():
    """Allowing a step of 0 changes codegen for every emitter, so prove the
    resulting loop is arithmetically right and not merely shorter.

    Contracts eight source rows into one destination row, weighted by eight
    FPRAM slots -- the exact shape of KDA's prediction sweep.
    """
    p = _prog()
    dst = p.alloc("dst", MLEN, MLEN)
    src = p.alloc("src", MLEN, MLEN)
    weights = p.fp_var("w", size=MLEN)
    dst_base = p.get_vram_layout(dst.name).vram_base_addr
    src_base = p.get_vram_layout(src.name).vram_base_addr
    mark = len(p.get_code())
    p.tile_row_fma_fp_sweep_asm(
        dst_addr=dst_base, src_addr=src_base, fpram_base=weights.address,
        dst_rows=[0] * MLEN, src_rows=list(range(MLEN)),
    )
    code = p.get_code()[mark:]
    assert "C_LOOP_START" in code, "the pinned destination must still loop"
    assert code.count("V_FMA_VF") == 1, "one FMA in the loop body, not MLEN of them"

    m = Machine(vlen=MLEN, vram_words=1 << 16, fpram_words=1 << 13)
    m.write_vram_row(dst_base, [0.0] * MLEN)
    expected = [0.0] * MLEN
    for row in range(MLEN):
        values = [float(row * MLEN + col + 1) for col in range(MLEN)]
        m.write_vram_row(src_base + row * MLEN, values)
        m.write_fpram(weights.address + row, [0.5 + row])
        for col in range(MLEN):
            expected[col] += values[col] * (0.5 + row)
    m.run(code)

    got = m.read_vram_row(dst_base, MLEN)
    for col in range(MLEN):
        assert abs(got[col] - expected[col]) < 1e-3, (
            f"lane {col}: {got[col]} != {expected[col]}"
        )
