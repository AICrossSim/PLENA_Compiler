"""Unit tests for the ISA oracle in `aten/tests/isa_interpreter.py`.

The interpreter is what makes `test_kda_decode_step.py` a numerical check
rather than a shape check, so it needs anchoring independently of what the KDA
lowering happens to emit. Mutating it and running only the KDA tests is not
enough: making `V_MUL_VF` read its destination instead of its source left those
green, because the lowering only ever emits the in-place form `V_MUL_VF gpX,
gpX, fY`. Every opcode is therefore exercised here in a form the lowering does
not currently produce.

Semantics are taken from the emulator, not from `doc/plena_isa_spec.md` --
`V_SUB_VV` is the case where the two disagreed.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.tests.isa_interpreter import Machine, UnsupportedInstruction  # noqa: E402

VLEN = 4


def _machine() -> Machine:
    return Machine(vlen=VLEN, vram_words=256, fpram_words=64)


# ---------------------------------------------------------------------------
# Vector ops -- out of place, so dst and src cannot be confused
# ---------------------------------------------------------------------------


def test_v_mul_vf_reads_its_source_not_its_destination():
    m = _machine()
    m.write_vram_row(0, [9.0, 9.0, 9.0, 9.0])   # destination, must be overwritten
    m.write_vram_row(4, [1.0, 2.0, 3.0, 4.0])   # source
    m.write_fpram(0, [2.5])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
        V_MUL_VF gp1, gp2, f1, 0
        """
    )
    assert m.read_vram_row(0) == [2.5, 5.0, 7.5, 10.0]
    assert m.read_vram_row(4) == [1.0, 2.0, 3.0, 4.0], "source must be untouched"


def test_v_sub_vv_is_rs1_minus_rs2():
    """The emulator computes rs1 - rs2. doc/plena_isa_spec.md said rs2 - rs1
    until this was found; subtraction does not commute, and KDA's error term
    flips sign under the wrong one."""
    m = _machine()
    m.write_vram_row(4, [10.0, 10.0, 10.0, 10.0])
    m.write_vram_row(8, [1.0, 2.0, 3.0, 4.0])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
        S_ADDI_INT gp3, gp0, 8
        V_SUB_VV gp1, gp2, gp3, 0
        """
    )
    assert m.read_vram_row(0) == [9.0, 8.0, 7.0, 6.0]


def test_v_add_vv_writes_a_third_destination():
    m = _machine()
    m.write_vram_row(4, [1.0, 2.0, 3.0, 4.0])
    m.write_vram_row(8, [10.0, 20.0, 30.0, 40.0])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
        S_ADDI_INT gp3, gp0, 8
        V_ADD_VV gp1, gp2, gp3, 0
        """
    )
    assert m.read_vram_row(0) == [11.0, 22.0, 33.0, 44.0]
    assert m.read_vram_row(4) == [1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# Scalar and control
# ---------------------------------------------------------------------------


def test_s_addi_int_accumulates():
    m = _machine()
    m.run(
        """
        S_ADDI_INT gp1, gp0, 7
        S_ADDI_INT gp1, gp1, 5
        """
    )
    assert m.gp[1] == 12


def test_gp0_and_f0_are_writable_registers_not_hardwired_zero():
    """Faithful to the emulator, where both are plain array slots. Their zero
    value is a compiler convention (`RegisterAllocator` never hands them out),
    so an emitter that wrote one would break address arithmetic on real
    hardware -- and a hardwired oracle would keep passing it."""
    m = _machine()
    m.run(
        """
        S_ADDI_INT gp0, gp0, 99
        S_ADDI_INT gp1, gp0, 1
        """
    )
    assert m.gp[0] == 99
    assert m.gp[1] == 100


def test_s_ld_fp_addresses_through_the_base_register():
    m = _machine()
    m.write_fpram(0, [10.0, 11.0, 12.0, 13.0])
    m.write_vram_row(0, [1.0, 1.0, 1.0, 1.0])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 2
        S_LD_FP f1, gp2, 1
        V_MUL_VF gp1, gp1, f1, 0
        """
    )
    # base 2 + imm 1 -> slot 3
    assert m.read_vram_row(0) == [13.0, 13.0, 13.0, 13.0]


def test_loop_runs_exactly_the_declared_count():
    m = _machine()
    m.write_vram_row(0, [1.0, 1.0, 1.0, 1.0])
    m.write_fpram(0, [2.0])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
        C_LOOP_START gp2, 3
        V_MUL_VF gp1, gp1, f1, 0
        C_LOOP_END gp2
        """
    )
    assert m.read_vram_row(0) == [8.0, 8.0, 8.0, 8.0], "2^3, so exactly 3 iterations"


def test_loop_body_advances_a_pointer_each_iteration():
    """The pattern every tile_row_* sweep emits: the loop body bumps the
    address register, so an off-by-one in the trip count shows up as a row that
    was skipped or written twice."""
    m = _machine()
    for row in range(3):
        m.write_vram_row(row * VLEN, [1.0] * VLEN)
    m.write_vram_row(3 * VLEN, [5.0] * VLEN)  # must be left alone
    m.write_fpram(0, [3.0])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
        C_LOOP_START gp2, 3
        V_MUL_VF gp1, gp1, f1, 0
        S_ADDI_INT gp1, gp1, 4
        C_LOOP_END gp2
        """
    )
    for row in range(3):
        assert m.read_vram_row(row * VLEN) == [3.0] * VLEN
    assert m.read_vram_row(3 * VLEN) == [5.0] * VLEN


# ---------------------------------------------------------------------------
# Failure modes: the oracle must refuse rather than guess
# ---------------------------------------------------------------------------


def test_unmodelled_opcode_raises():
    # V_FMA_VF used to be the example here; it is modelled now, so this needs an
    # opcode that genuinely is not. V_MAX_VF exists in the ISA and no KDA or
    # Mamba lowering emits it.
    with pytest.raises(UnsupportedInstruction, match="outside the modelled subset"):
        _machine().run("V_MAX_VF gp1, gp2, f1, 0")


def test_fma_accumulates_into_its_destination():
    """The oracle has to model the accumulate, or a lowering that used FMA where
    it meant multiply would agree with it and disagree with the hardware."""
    m = _machine()
    m.write_vram_row(0, [1.0, 2.0, 3.0, 4.0])
    m.write_vram_row(4, [10.0, 20.0, 30.0, 40.0])
    m.write_fpram(0, [0.5])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
        V_FMA_VF gp1, gp2, f1, 0
        """
    )
    assert m.read_vram_row(0, 4) == [6.0, 12.0, 18.0, 24.0]
    assert m.read_vram_row(4, 4) == [10.0, 20.0, 30.0, 40.0], "source is read-only"


def test_fma_with_the_destination_as_its_own_source():
    """`x += x * f` is `x * (1 + f)`. Reading and writing the same row must not
    see a partially-updated value part-way through the lane sweep."""
    m = _machine()
    m.write_vram_row(0, [1.0, 2.0, 3.0, 4.0])
    m.write_fpram(0, [2.0])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
        V_FMA_VF gp1, gp1, f1, 0
        """
    )
    assert m.read_vram_row(0, 4) == [3.0, 6.0, 9.0, 12.0]


def test_nonzero_rmask_raises():
    with pytest.raises(UnsupportedInstruction, match="rmask"):
        _machine().run(
            """
            S_ADDI_INT gp1, gp0, 0
            S_ADDI_INT gp2, gp0, 4
            V_ADD_VV gp1, gp2, gp2, 1
            """
        )


def test_misaligned_vector_address_raises():
    """The emulator's vector SRAM asserts VLEN alignment; reading across a row
    boundary is a panic there, not a silent wrap."""
    with pytest.raises(UnsupportedInstruction, match="not a multiple of vlen"):
        _machine().run(
            """
            S_ADDI_INT gp1, gp0, 3
            S_ADDI_INT gp2, gp0, 0
            V_ADD_VV gp1, gp2, gp2, 0
            """
        )


def test_comments_and_stage_markers_are_ignored():
    m = _machine()
    m.write_vram_row(0, [1.0, 1.0, 1.0, 1.0])
    m.run(
        """
        ; @stage=kda_decay head=0
        S_ADDI_INT gp1, gp0, 0
        // trailing style comment
        V_ADD_VV gp1, gp1, gp1, 0
        """
    )
    assert m.read_vram_row(0) == [2.0, 2.0, 2.0, 2.0]


def test_runaway_loop_is_bounded():
    """A loop whose counter is rewritten inside the body would otherwise hang
    the suite instead of failing it."""
    with pytest.raises(RuntimeError, match="runaway loop"):
        _machine().run(
            """
            C_LOOP_START gp2, 2
            S_ADDI_INT gp2, gp0, 2
            C_LOOP_END gp2
            """
        )


# ---------------------------------------------------------------------------
# Ops added for the causal conv. Each is exercised in a form the lowering does
# not emit, for the same reason as V_MUL_VF above.
# ---------------------------------------------------------------------------


def test_v_mul_vv_is_elementwise_and_out_of_place():
    m = _machine()
    m.write_vram_row(0, [9.0] * VLEN)
    m.write_vram_row(4, [1.0, 2.0, 3.0, 4.0])
    m.write_vram_row(8, [10.0, 100.0, 1000.0, 0.5])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
        S_ADDI_INT gp3, gp0, 8
        V_MUL_VV gp1, gp2, gp3, 0
        """
    )
    assert m.read_vram_row(0) == [10.0, 200.0, 3000.0, 2.0]
    assert m.read_vram_row(4) == [1.0, 2.0, 3.0, 4.0]


def test_v_add_vf_broadcasts_the_scalar():
    m = _machine()
    m.write_vram_row(4, [1.0, 2.0, 3.0, 4.0])
    m.write_fpram(0, [0.5])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
        V_ADD_VF gp1, gp2, f1, 0
        """
    )
    assert m.read_vram_row(0) == [1.5, 2.5, 3.5, 4.5]


def test_v_sub_vf_honours_rorder():
    """rorder=1 reverses the operands. The negate idiom `V_SUB_VF rd, rs1, f0, 0, 1`
    depends on it, and getting it backwards flips the sign of everything after."""
    m = _machine()
    m.write_vram_row(4, [1.0, 2.0, 3.0, 4.0])
    m.write_fpram(0, [10.0])
    prologue = """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
    """
    m.run(prologue + "V_SUB_VF gp1, gp2, f1, 0, 0\n")
    assert m.read_vram_row(0) == [-9.0, -8.0, -7.0, -6.0]

    m2 = _machine()
    m2.write_vram_row(4, [1.0, 2.0, 3.0, 4.0])
    m2.write_fpram(0, [10.0])
    m2.run(prologue + "V_SUB_VF gp1, gp2, f1, 0, 1\n")
    assert m2.read_vram_row(0) == [9.0, 8.0, 7.0, 6.0]


def test_v_exp_v_saturates_rather_than_overflowing():
    """vector_machine.rs clamps to [-88, 88] before exp, so bf16 cannot go
    infinite. An unclamped model would return inf and any comparison against it
    would then pass or fail for the wrong reason."""
    import math

    m = _machine()
    m.write_vram_row(4, [0.0, 1.0, 200.0, -200.0])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
        V_EXP_V gp1, gp2, 0
        """
    )
    got = m.read_vram_row(0)
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(math.e)
    assert got[2] == pytest.approx(math.exp(88.0))
    assert got[3] == pytest.approx(math.exp(-88.0))
    assert all(math.isfinite(x) for x in got)


def test_v_reci_v_on_zero_returns_inf_like_the_emulator():
    """vector_machine.rs uses tensor.reciprocal(), which yields inf rather than
    trapping. A model that raises would crash the oracle on a lowering the
    hardware runs fine -- e.g. one that reciprocates an unused lane."""
    import math

    m = _machine()
    m.write_vram_row(4, [0.0, 2.0, 0.0, -4.0])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
        V_RECI_V gp1, gp2, 0
        """
    )
    got = m.read_vram_row(0)
    assert math.isinf(got[0]) and math.isinf(got[2])
    assert got[1] == 0.5 and got[3] == -0.25


def test_v_reci_v_is_elementwise_reciprocal():
    m = _machine()
    m.write_vram_row(4, [1.0, 2.0, 4.0, -0.5])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
        V_RECI_V gp1, gp2, 0
        """
    )
    assert m.read_vram_row(0) == [1.0, 0.5, 0.25, -2.0]


# ---------------------------------------------------------------------------
# Scalar FP and the reduction.
#
# These carry the two semantics this oracle took from the emulator rather than
# from doc/plena_isa_spec.md, and neither was reachable through the KDA
# lowerings: every V_RED_SUM the emitters produce is preceded by
# `S_ADD_FP f1, f0, f0`, so f1 always enters at zero and accumulate-vs-overwrite
# never shows; and no emitter ever names f0 as a destination, because
# RegisterAllocator hands out f1..f7. Both are pinned here directly.
# ---------------------------------------------------------------------------


def _fp_prog(body: str) -> str:
    return """
        S_ADDI_INT gp1, gp0, 0
        S_ADDI_INT gp2, gp0, 4
    """ + body


def test_v_red_sum_accumulates_into_its_fp_register():
    """dispatch.rs seeds reduce_sum with the current f[rd], so a second
    reduction adds to the first. The compiler relies on the opposite -- it
    clears f1 before every V_RED_SUM -- so nothing in the lowering exercises
    this, and an oracle that overwrote would go unnoticed."""
    m = _machine()
    m.write_vram_row(0, [1.0, 2.0, 3.0, 4.0])   # sums to 10
    m.write_vram_row(4, [10.0, 20.0, 30.0, 40.0])  # sums to 100
    m.run(
        _fp_prog(
            """
        V_RED_SUM f1, gp1, 0, 0
        V_RED_SUM f1, gp2, 0, 0
        S_ADDI_INT gp3, gp0, 0
        S_ST_FP f1, gp3, 0
        """
        )
    )
    assert m.fpram[0] == 110.0, "second V_RED_SUM must add, not replace"


@pytest.mark.parametrize("form", ["V_RED_SUM f1, gp1, 1, 0", "V_RED_SUM f1, gp1, 0, 1"])
def test_v_red_sum_refuses_a_mask_it_does_not_model(form):
    """The masked reduce_sum in vector_machine.rs is a materially different
    computation -- it broadcasts each head's sum back over its slice before
    summing. Refuse rather than model it wrongly, whichever operand slot the
    assembler puts the mask in."""
    with pytest.raises(UnsupportedInstruction, match="mask"):
        _machine().run(_fp_prog(form + "\n"))


def test_scalar_fp_writes_to_f0_are_discarded():
    """dispatch.rs:417-424 makes every scalar FP op a no-op when rd == 0. That
    is what keeps f0 zero for the `V_MUL_VF gpX, gpX, f0` zeroing idiom --
    RegisterAllocator never handing out f0 is the other half."""
    m = _machine()
    m.write_fpram(0, [7.0, 3.0])
    m.write_vram_row(0, [5.0, 5.0, 5.0, 5.0])
    m.run(
        """
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
        S_LD_FP f2, gp3, 1
        S_ADD_FP f0, f1, f2
        S_ADDI_INT gp1, gp0, 0
        V_MUL_VF gp1, gp1, f0, 0
        """
    )
    assert m.fp[0] == 0.0, "S_ADD_FP with rd=0 must be discarded"
    assert m.read_vram_row(0) == [0.0] * VLEN, "f0 must still zero the row"


def test_s_ld_fp_is_not_guarded_against_f0():
    """S_LD_FP is deliberately absent from dispatch.rs's rd==0 guard list, so
    modelling it as guarded would diverge."""
    m = _machine()
    m.write_fpram(0, [9.0])
    m.run(
        """
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f0, gp3, 0
        """
    )
    assert m.fp[0] == 9.0


def test_s_st_fp_source_and_address_roles():
    """S_ST_FP { rd, rs1, imm } stores f[rd] to FPRAM[gp[rs1] + imm] -- rd is
    the source, which is the reverse of the usual destination-first reading."""
    m = _machine()
    m.write_fpram(0, [42.0])
    m.run(
        """
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
        S_ADDI_INT gp4, gp0, 5
        S_ST_FP f1, gp4, 2
        """
    )
    assert m.fpram[7] == 42.0, "address is gp[rs1] + imm = 5 + 2"
    assert m.fpram[5] == 0.0 and m.fpram[0] == 42.0


def test_scalar_fp_arithmetic():
    m = _machine()
    m.write_fpram(0, [10.0, 4.0])
    m.run(
        """
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
        S_LD_FP f2, gp3, 1
        S_ADD_FP f3, f1, f2
        S_SUB_FP f4, f1, f2
        S_MUL_FP f5, f1, f2
        S_ADDI_INT gp4, gp0, 10
        S_ST_FP f3, gp4, 0
        S_ST_FP f4, gp4, 1
        S_ST_FP f5, gp4, 2
        """
    )
    assert m.fpram[10] == 14.0
    assert m.fpram[11] == 6.0, "S_SUB_FP is fs1 - fs2, not the reverse"
    assert m.fpram[12] == 40.0


def test_s_sqrt_and_s_reci_edge_cases():
    """sqrt of a negative is NaN (f32::sqrt), reciprocal of zero is inf
    (bf16::ONE / 0), and -0.0 gives -inf rather than +inf."""
    import math

    m = _machine()
    m.write_fpram(0, [9.0, -4.0, 0.0, -0.0])
    m.run(
        """
        S_ADDI_INT gp3, gp0, 0
        S_LD_FP f1, gp3, 0
        S_SQRT_FP f2, f1
        S_LD_FP f3, gp3, 1
        S_SQRT_FP f4, f3
        S_LD_FP f5, gp3, 2
        S_RECI_FP f6, f5
        S_LD_FP f7, gp3, 3
        S_RECI_FP f1, f7
        S_ADDI_INT gp4, gp0, 10
        S_ST_FP f2, gp4, 0
        S_ST_FP f4, gp4, 1
        S_ST_FP f6, gp4, 2
        S_ST_FP f1, gp4, 3
        """
    )
    assert m.fpram[10] == 3.0
    assert math.isnan(m.fpram[11])
    assert m.fpram[12] == math.inf
    assert m.fpram[13] == -math.inf


def test_v_red_sum_rejects_a_misaligned_address():
    with pytest.raises(UnsupportedInstruction, match="not a multiple of vlen"):
        _machine().run(
            """
            S_ADDI_INT gp1, gp0, 3
            V_RED_SUM f1, gp1, 0, 0
            """
        )


def test_s_map_fp_v_moves_a_vram_row_into_fpram():
    """Operand roles mirror S_MAP_V_FP: rs1 is the VRAM source row and rd the
    FP_MEM base, so that in both instructions rd names the destination memory.
    Reading them the usual destination-first way gets it backwards."""
    m = _machine()
    m.write_vram_row(8, [1.0, 2.0, 3.0, 4.0])
    m.run(
        """
        S_ADDI_INT gp1, gp0, 5
        S_ADDI_INT gp2, gp0, 8
        S_MAP_FP_V gp1, gp2, 2
        """
    )
    # destination is gp[rd] + imm = 5 + 2
    assert m.fpram[7:11] == [1.0, 2.0, 3.0, 4.0]
    assert m.fpram[5] == 0.0 and m.fpram[6] == 0.0
    assert m.read_vram_row(8) == [1.0, 2.0, 3.0, 4.0], "source untouched"


def test_s_map_fp_v_rejects_a_misaligned_source():
    with pytest.raises(UnsupportedInstruction, match="not a multiple of vlen"):
        _machine().run(
            """
            S_ADDI_INT gp1, gp0, 0
            S_ADDI_INT gp2, gp0, 3
            S_MAP_FP_V gp1, gp2, 0
            """
        )


def test_s_map_fp_v_refuses_to_write_past_the_modelled_fpram():
    """Python slice assignment past the end *extends* the list rather than
    failing, so an out-of-bounds write would silently land at the wrong address
    and stay green. The emulator asserts on the FP SRAM bound."""
    m = Machine(vlen=VLEN, vram_words=64, fpram_words=16)
    m.write_vram_row(0, [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(UnsupportedInstruction, match="past the modelled"):
        m.run(
            """
            S_ADDI_INT gp1, gp0, 14
            S_ADDI_INT gp2, gp0, 0
            S_MAP_FP_V gp1, gp2, 0
            """
        )
    assert len(m.fpram) == 16, "the FPRAM must not have grown"
