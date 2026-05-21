"""Tests for the PreIsaIR data structures (pre_isa_ir.py).

Covers:
  * PreIsaOp construction + opcode validation
  * PreIsaModule.append / .comment
  * format_pre_isa indentation around C_LOOP_START / C_LOOP_END
  * loop_regions pairing logic (nested, unmatched)

These tests use ONLY the pre_isa_ir module — no TVM kernel, no
shim. Cheap and fast.
"""

import pytest
from tvm import tir

from tilelang_tvm_compiler.pre_isa_ir import (
    PreIsaOp, PreIsaModule, format_pre_isa, loop_regions, KNOWN_OPCODES,
)


def test_known_opcodes_include_core_set():
    """Sanity: every PreIsaIR opcode passes use must appear in
    KNOWN_OPCODES. Note ``C_LOOP_START`` / ``C_LOOP_END`` are PLENA
    ISA mnemonics (emitted as text by BackendEmit's serial-loop
    handler), NOT PreIsaIR opcodes — the PreIsaIR control marker is
    the unified ``LOOP_START`` / ``LOOP_END`` (with kind in
    ``annotations["loop_kind"]``)."""
    must_have = {
        "S_ADDI_INT", "S_LD_FP", "S_ST_FP", "S_MUL_FP",
        "V_ADD_VV", "V_MUL_VF", "V_EXP_V",
        "M_BTMM", "M_MM",
        "H_LOAD_V", "H_STORE_V", "H_PREFETCH_V",
        "LOOP_START", "LOOP_END",
        "C_SET_SCALE_REG", "C_SET_STRIDE_REG",
    }
    missing = must_have - KNOWN_OPCODES
    assert not missing, f"opcodes missing from KNOWN_OPCODES: {missing}"


def test_pre_isa_op_rejects_unknown_opcode():
    with pytest.raises(ValueError, match="not a known PLENA mnemonic"):
        PreIsaOp(opcode="NOT_A_REAL_INSTR")


def test_pre_isa_op_accepts_known_opcode():
    op = PreIsaOp(opcode="S_ADDI_INT", operands=["gp1", "gp0", "256"])
    assert op.opcode == "S_ADDI_INT"
    assert op.operands == ["gp1", "gp0", "256"]
    assert op.binds is None
    assert op.annotations == {}


def test_module_append_and_comment():
    mod = PreIsaModule(name="k")
    mod.append(PreIsaOp(opcode="S_ADDI_INT", operands=["gp1", "gp0", "1"]))
    mod.comment("hello world")
    assert len(mod.ops) == 2
    assert mod.ops[0].opcode == "S_ADDI_INT"
    assert mod.ops[1].opcode == "_COMMENT"
    assert mod.ops[1].operands == ["hello world"]


def test_format_pre_isa_indents_inside_loop():
    mod = PreIsaModule(name="k")
    var = tir.Var("i", "int32")
    mod.append(PreIsaOp(opcode="LOOP_START", operands=["gp2", "8"], binds=var))
    mod.append(PreIsaOp(opcode="V_ADD_VV", operands=["gp3", "gp4", "gp5", "0"]))
    mod.append(PreIsaOp(opcode="LOOP_END", operands=["gp2"]))
    text = format_pre_isa(mod)
    lines = [ln for ln in text.split("\n") if "V_ADD_VV" in ln or "LOOP_" in ln]
    # The body line must be indented further than the LOOP_START line.
    start_indent = len(lines[0]) - len(lines[0].lstrip())
    body_indent = len(lines[1]) - len(lines[1].lstrip())
    end_indent = len(lines[2]) - len(lines[2].lstrip())
    assert body_indent > start_indent, f"body should indent past START:\n{text}"
    assert end_indent == start_indent, f"END should align with START:\n{text}"


def test_format_pre_isa_handles_empty():
    mod = PreIsaModule(name="empty")
    text = format_pre_isa(mod)
    assert "PreIsaModule(name='empty')" in text
    assert "Ops:" in text


def test_loop_regions_flat_pair():
    var = tir.Var("i", "int32")
    ops = [
        PreIsaOp(opcode="LOOP_START", operands=["gp2", "4"], binds=var),
        PreIsaOp(opcode="V_ADD_VV", operands=["gp3", "gp4", "gp5", "0"]),
        PreIsaOp(opcode="LOOP_END", operands=["gp2"]),
    ]
    regions = loop_regions(ops)
    assert regions == [(0, 2, var)]


def test_loop_regions_nested():
    v_outer = tir.Var("outer", "int32")
    v_inner = tir.Var("inner", "int32")
    ops = [
        PreIsaOp(opcode="LOOP_START", operands=["gp1", "4"], binds=v_outer),
        PreIsaOp(opcode="LOOP_START", operands=["gp2", "8"], binds=v_inner),
        PreIsaOp(opcode="V_ADD_VV", operands=["gp3", "gp4", "gp5", "0"]),
        PreIsaOp(opcode="LOOP_END", operands=["gp2"]),
        PreIsaOp(opcode="LOOP_END", operands=["gp1"]),
    ]
    regions = loop_regions(ops)
    # Inner pair closes first; outer second. Both reported.
    assert (1, 3, v_inner) in regions
    assert (0, 4, v_outer) in regions


def test_loop_regions_unmatched_end_raises():
    ops = [PreIsaOp(opcode="LOOP_END", operands=["gp1"])]
    with pytest.raises(ValueError, match="no matching loop-start"):
        loop_regions(ops)


def test_loop_regions_unclosed_start_raises():
    var = tir.Var("i", "int32")
    ops = [PreIsaOp(opcode="LOOP_START", operands=["gp1", "4"], binds=var)]
    with pytest.raises(ValueError, match="unclosed loop-start"):
        loop_regions(ops)


def test_loop_regions_tolerates_missing_binds_from_text_capture():
    """When PreIsaIR is built by text capture (CapturingCode), the
    iteration var is already lowered away — binds is None. loop_regions
    must NOT require it; LICM is the only consumer that needs binds and
    it operates on the var-ref construction path only."""
    ops = [
        PreIsaOp(opcode="LOOP_START", operands=["gp1", "4"]),  # binds=None
        PreIsaOp(opcode="LOOP_END", operands=["gp1"]),
    ]
    regions = loop_regions(ops)
    assert regions == [(0, 1, None)]
