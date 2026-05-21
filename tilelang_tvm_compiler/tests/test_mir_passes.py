"""Tests for MIR optimisation passes.

Each pass is verified two ways:
  1. **Unit**: build a tiny MIR by hand, run the pass, check
     structure + counts.
  2. **Integration**: run on a real PreIsaIR-v2-produced MIR (via
     v2 e2e helpers), check that the pass doesn't break MIR
     verifier and that resulting ISA matches pre-pass ISA
     structurally (HW op set unchanged).
"""

import re

import pytest
from tvm import tir

from tilelang_tvm_compiler import mir
from tilelang_tvm_compiler import mir_passes as P


# ---------------------------------------------------------------------
# Tiny MIR builders for unit tests
# ---------------------------------------------------------------------

def _mk_fn(name="t"):
    fn = mir.MirFunction(name=name)
    entry = mir.MirBlock(name="entry")
    fn.blocks.append(entry)
    fn.make_gp0_const()
    return fn, entry


def _addi(blk, src, imm, hint=""):
    fn = _enclosing_fn(blk)
    dst = fn.mint_value("i32", hint=hint)
    blk.append(mir.MirInstr("S_ADDI_INT", [src, imm], result=dst))
    return dst


def _slli(blk, src, k, hint=""):
    fn = _enclosing_fn(blk)
    dst = fn.mint_value("i32", hint=hint)
    blk.append(mir.MirInstr("S_SLLI_INT", [src, k], result=dst))
    return dst


def _add(blk, a, b, hint=""):
    fn = _enclosing_fn(blk)
    dst = fn.mint_value("i32", hint=hint)
    blk.append(mir.MirInstr("S_ADD_INT", [a, b], result=dst))
    return dst


def _enclosing_fn(blk):
    # Climb out via parent_loop pointers.
    # For unit tests the block is always one of fn.blocks or a
    # loop body block — both reachable from fn.blocks. We stash
    # the fn on the block via a side attr the helpers know.
    return blk._test_fn


def _attach_fn(blk, fn):
    blk._test_fn = fn


def _mk_loop(fn, parent_blk, init, extent, kind="unroll", lvar_hint="i"):
    lvar = fn.mint_value("i32", hint=lvar_hint)
    body = mir.MirBlock(name=f"body.{lvar_hint}")
    body.add_argument(lvar)
    _attach_fn(body, fn)
    lp = mir.MirLoop(
        name=f"L_{lvar_hint}", loop_var=lvar,
        init=init, extent=extent, body=[body],
        loop_kind=kind,
    )
    parent_blk.append(lp)
    return lp, body, lvar


# ---------------------------------------------------------------------
# dead_loop_elim
# ---------------------------------------------------------------------

def test_dle_extent_one_peels_body():
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    lp, body, lvar = _mk_loop(fn, entry, init=5, extent=1, lvar_hint="i")
    # Body: %x = ADDI lvar, 3
    x = _addi(body, lvar, 3, hint="x")
    # Outer block uses nothing — pass should peel and replace
    # lvar uses with IntImm(5).

    changed = P.dead_loop_elim(fn)
    assert changed
    mir.verify(fn)

    # After peeling: entry has no loop, has the ADDI instr.
    assert all(not isinstance(it, mir.MirLoop) for it in entry.items)
    adds = [it for it in entry.items if isinstance(it, mir.MirInstr)
            and it.opcode == "S_ADDI_INT"]
    assert len(adds) == 1
    # ADDI's first operand is now IntImm(5), not the lvar.
    op0 = adds[0].operands[0]
    assert isinstance(op0, tir.IntImm) and int(op0.value) == 5


def test_dle_extent_zero_deletes_loop():
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    lp, body, lvar = _mk_loop(fn, entry, init=0, extent=0, lvar_hint="i")
    # Body empty — extent=0 means body never runs.

    changed = P.dead_loop_elim(fn)
    assert changed
    mir.verify(fn)
    assert all(not isinstance(it, mir.MirLoop) for it in entry.items)


def test_dle_nested_extent_one_collapses_both():
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    outer, outer_body, outer_lv = _mk_loop(
        fn, entry, init=0, extent=1, lvar_hint="o",
    )
    inner, inner_body, inner_lv = _mk_loop(
        fn, outer_body, init=0, extent=1, lvar_hint="i",
    )
    _addi(inner_body, outer_lv, 0, hint="x")

    P.dead_loop_elim(fn)
    mir.verify(fn)
    # Both loops gone, one ADDI remains in entry.
    assert all(not isinstance(it, mir.MirLoop) for it in entry.items)


def test_dle_extent_two_left_alone():
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    lp, body, lvar = _mk_loop(fn, entry, init=0, extent=2, lvar_hint="i")
    _addi(body, lvar, 1)
    changed = P.dead_loop_elim(fn)
    assert not changed
    assert sum(1 for it in entry.items if isinstance(it, mir.MirLoop)) == 1


# ---------------------------------------------------------------------
# const_fold
# ---------------------------------------------------------------------

def test_const_fold_addi_chain():
    """Folded instr is rewritten to ``S_ADDI_INT gp0, K``. The
    consumer's MirValue operand is unchanged; what changed is
    whose defining instr it points to. Two chained ADDIs both
    fold to ``ADDI gp0, K`` where K is the running constant."""
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    # gp0 + 5  →  5  →  rewrites to ADDI gp0, 5
    # %a + 10  →  15 →  rewrites to ADDI gp0, 15
    a = _addi(entry, fn.gp0_value, 5, hint="a")
    b = _addi(entry, a, 10, hint="b")
    # Side-effecting consumer to keep b alive.
    entry.append(mir.MirInstr(
        "C_SET_SCALE_REG", [b], result=None,
    ))
    changed = P.const_fold(fn)
    assert changed
    # ``b`` is now produced by ``ADDI gp0, 15``.
    assert b.defined_by.opcode == "S_ADDI_INT"
    assert b.defined_by.operands[0] is fn.gp0_value
    assert b.defined_by.operands[1] == 15


def test_const_fold_slli_with_gp0():
    """``gp0 << K`` is gp0 (identity peephole). The setreg
    consumer's operand is now ``fn.gp0_value`` directly; the
    SLLI instr has no users and gets swept by DCE."""
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    s = _slli(entry, fn.gp0_value, 5, hint="s")
    entry.append(mir.MirInstr(
        "C_SET_STRIDE_REG", [s], result=None,
    ))
    P.const_fold(fn)
    setreg = [it for it in entry.items
              if isinstance(it, mir.MirInstr)
              and it.opcode == "C_SET_STRIDE_REG"][0]
    assert setreg.operands[0] is fn.gp0_value


# ---------------------------------------------------------------------
# dce
# ---------------------------------------------------------------------

def test_dce_drops_unused_addi():
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    _addi(entry, fn.gp0_value, 5, hint="dead")
    # No consumer — DCE should drop the ADDI.
    changed = P.dce(fn)
    assert changed
    assert not any(
        isinstance(it, mir.MirInstr) and it.opcode == "S_ADDI_INT"
        for it in entry.items
    )


def test_dce_keeps_used_addi():
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    x = _addi(entry, fn.gp0_value, 5, hint="live")
    entry.append(mir.MirInstr("C_SET_SCALE_REG", [x], result=None))
    changed = P.dce(fn)
    assert not changed
    assert sum(
        1 for it in entry.items
        if isinstance(it, mir.MirInstr) and it.opcode == "S_ADDI_INT"
    ) == 1


def test_dce_keeps_side_effecting():
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    # C_SET_SCALE_REG has no result; even so DCE must keep it.
    entry.append(mir.MirInstr(
        "C_SET_SCALE_REG", [fn.gp0_value], result=None,
    ))
    changed = P.dce(fn)
    assert not changed


# ---------------------------------------------------------------------
# cse
# ---------------------------------------------------------------------

def test_cse_collapses_identical_addi():
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    a = _addi(entry, fn.gp0_value, 100, hint="a")
    b = _addi(entry, fn.gp0_value, 100, hint="b")
    # Two consumers — one for each result. After CSE both should
    # point at ``a``.
    entry.append(mir.MirInstr("C_SET_SCALE_REG", [a], result=None))
    entry.append(mir.MirInstr("C_SET_STRIDE_REG", [b], result=None))

    changed = P.cse(fn)
    assert changed
    addis = [it for it in entry.items
             if isinstance(it, mir.MirInstr) and it.opcode == "S_ADDI_INT"]
    assert len(addis) == 1
    setregs = [it for it in entry.items
               if isinstance(it, mir.MirInstr)
               and it.opcode in ("C_SET_SCALE_REG", "C_SET_STRIDE_REG")]
    # Both setregs reference the same surviving ADDI's result.
    assert setregs[0].operands[0] is setregs[1].operands[0]
    assert setregs[0].operands[0] is addis[0].result


def test_cse_respects_operand_differences():
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    a = _addi(entry, fn.gp0_value, 100, hint="a")
    b = _addi(entry, fn.gp0_value, 200, hint="b")  # different imm
    entry.append(mir.MirInstr("C_SET_SCALE_REG", [a], result=None))
    entry.append(mir.MirInstr("C_SET_STRIDE_REG", [b], result=None))

    changed = P.cse(fn)
    assert not changed
    addis = [it for it in entry.items
             if isinstance(it, mir.MirInstr) and it.opcode == "S_ADDI_INT"]
    assert len(addis) == 2


# ---------------------------------------------------------------------
# default pipeline
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# licm
# ---------------------------------------------------------------------

def test_licm_hoists_invariant_addi():
    """``for i in [0, 4): %x = ADDI gp0, 100; ADDI lvar, x``
    — ``x`` is invariant; LICM hoists it out of the loop."""
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    lp, body, lvar = _mk_loop(fn, entry, init=0, extent=4, lvar_hint="i")
    x = _addi(body, fn.gp0_value, 100, hint="x")  # invariant
    y = _add(body, lvar, x, hint="y")  # depends on lvar — NOT invariant
    body.append(mir.MirInstr("C_SET_SCALE_REG", [y], result=None))

    changed = P.licm(fn)
    assert changed
    mir.verify(fn)
    # ``x``'s defining ADDI should now sit in ``entry`` BEFORE the loop.
    entry_addis = [it for it in entry.items
                   if isinstance(it, mir.MirInstr)
                   and it.opcode == "S_ADDI_INT"]
    assert len(entry_addis) == 1
    assert entry_addis[0].result is x
    # ``y`` (lvar-dependent) stays inside.
    body_adds = [it for it in body.items
                 if isinstance(it, mir.MirInstr)
                 and it.opcode == "S_ADD_INT"]
    assert len(body_adds) == 1


def test_licm_doesnt_hoist_side_effecting():
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    lp, body, lvar = _mk_loop(fn, entry, init=0, extent=4, lvar_hint="i")
    # C_SET_SCALE_REG with gp0 operand is "invariant" in the
    # operand sense but side-effecting — must NOT be hoisted.
    body.append(mir.MirInstr(
        "C_SET_SCALE_REG", [fn.gp0_value], result=None,
    ))
    changed = P.licm(fn)
    assert not changed
    # setreg still in body.
    assert any(
        isinstance(it, mir.MirInstr) and it.opcode == "C_SET_SCALE_REG"
        for it in body.items
    )


def test_licm_nested_loops():
    """Inner-loop invariant w.r.t. inner but NOT outer should be
    hoisted only to inner's parent (= outer's body), not all the
    way out."""
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    outer, outer_body, outer_lv = _mk_loop(
        fn, entry, init=0, extent=4, lvar_hint="o",
    )
    inner, inner_body, inner_lv = _mk_loop(
        fn, outer_body, init=0, extent=4, lvar_hint="i",
    )
    # x depends on outer_lv — invariant in inner, but not in outer.
    x = _addi(inner_body, outer_lv, 5, hint="x")
    inner_body.append(mir.MirInstr("C_SET_SCALE_REG", [x], result=None))

    P.licm(fn)
    mir.verify(fn)
    # x's ADDI now sits in outer_body, before the inner loop.
    outer_body_addis = [it for it in outer_body.items
                        if isinstance(it, mir.MirInstr)
                        and it.opcode == "S_ADDI_INT"]
    assert len(outer_body_addis) == 1
    assert outer_body_addis[0].result is x
    # The setreg consumer is still inside inner_body.
    assert any(
        isinstance(it, mir.MirInstr) and it.opcode == "C_SET_SCALE_REG"
        for it in inner_body.items
    )


def test_pipeline_dle_then_fold():
    """``for i in [0, 1): %x = ADDI lvar, 5; setreg %x`` → after
    pipeline:
      * DLE peels the loop, RAUW'ing ``lvar`` to ``IntImm(3)``;
      * const_fold rewrites ``ADDI IntImm(3), 5`` to ``ADDI gp0, 8``;
      * DCE has nothing to do (the ADDI is still used);
      * setreg's operand still points at the same MirValue, now
        produced by ``ADDI gp0, 8``.
    """
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    lp, body, lvar = _mk_loop(fn, entry, init=3, extent=1, lvar_hint="i")
    x = _addi(body, lvar, 5, hint="x")
    body.append(mir.MirInstr("C_SET_SCALE_REG", [x], result=None))

    P.run_default_pipeline(fn)
    mir.verify(fn)
    # Loop gone.
    assert all(not isinstance(it, mir.MirLoop) for it in entry.items)
    # Exactly one ADDI (the folded gp0+8) and one setreg referencing it.
    addis = [it for it in entry.items
             if isinstance(it, mir.MirInstr) and it.opcode == "S_ADDI_INT"]
    setregs = [it for it in entry.items
               if isinstance(it, mir.MirInstr)
               and it.opcode == "C_SET_SCALE_REG"]
    assert len(addis) == 1
    assert addis[0].operands[0] is fn.gp0_value
    assert addis[0].operands[1] == 8
    assert len(setregs) == 1
    assert setregs[0].operands[0] is addis[0].result


# ---------------------------------------------------------------------
# reassociate
# ---------------------------------------------------------------------

def _setreg(blk, src):
    blk.append(mir.MirInstr("C_SET_SCALE_REG", [src], result=None))


def _mk_loop_arg(fn, parent_blk, lvar_hint):
    """Helper: mk a serial loop with a non-trivial extent so its
    loop_var stays alive (used to give us "named" block-arg leaves
    in reassoc tests)."""
    lvar = fn.mint_value("i32", hint=lvar_hint)
    body = mir.MirBlock(name=f"body.{lvar_hint}")
    body.add_argument(lvar)
    _attach_fn(body, fn)
    lp = mir.MirLoop(
        name=f"L_{lvar_hint}", loop_var=lvar,
        init=0, extent=4, body=[body],
        loop_kind="serial",
    )
    parent_blk.append(lp)
    return lp, body, lvar


def test_reassociate_duplicate_full_chain_collapses():
    """Two chains over the same leaves+const collapse to one."""
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    lp, body, _ = _mk_loop_arg(fn, entry, "i")
    a = _addi(body, fn.gp0_value, 5, hint="a")
    b = _addi(body, fn.gp0_value, 7, hint="b")
    # chain1 = a + b + 3
    s1a = _add(body, a, b, hint="s1a")
    s1 = _addi(body, s1a, 3, hint="s1")
    _setreg(body, s1)
    # chain2 = same — different ADD order
    s2a = _addi(body, b, 3, hint="s2a")
    s2 = _add(body, s2a, a, hint="s2")
    _setreg(body, s2)

    P.reassociate(fn)
    mir.verify(fn)
    # Both setregs reference the SAME MirValue now.
    setregs = [it for it in body.items
               if isinstance(it, mir.MirInstr)
               and it.opcode == "C_SET_SCALE_REG"]
    assert len(setregs) == 2
    assert setregs[0].operands[0] is setregs[1].operands[0]


def _sub(blk, a, b, hint=""):
    fn = _enclosing_fn(blk)
    dst = fn.mint_value("i32", hint=hint)
    blk.append(mir.MirInstr("S_SUB_INT", [a, b], result=dst))
    return dst


def test_reassociate_sub_treated_as_negated_add():
    """``a - b`` and ``-b + a`` and ``a + (b - 2b)`` should all
    canonicalise to the same {+a, -b} multiset. Two chains
    expressing the same canonical form collapse to one."""
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    lp, body, _ = _mk_loop_arg(fn, entry, "i")
    a = _addi(body, fn.gp0_value, 100, hint="a")
    b = _addi(body, fn.gp0_value, 200, hint="b")
    # chain1: a - b
    s1 = _sub(body, a, b, hint="s1")
    _setreg(body, s1)
    # chain2: also a - b, but written as (a + 0) - b via a ADD
    # detour; reassoc should still see {+a, -b}, 0.
    z = _add(body, a, fn.gp0_value, hint="z")   # = a
    s2 = _sub(body, z, b, hint="s2")
    _setreg(body, s2)

    P.run_default_pipeline(fn)
    mir.verify(fn)
    # Two setregs should point at the same MirValue.
    setregs = [it for it in body.items
               if isinstance(it, mir.MirInstr)
               and it.opcode == "C_SET_SCALE_REG"]
    assert len(setregs) == 2
    assert setregs[0].operands[0] is setregs[1].operands[0]


def test_reassociate_sub_cancellation():
    """``a + b - a`` should simplify so the only remaining work in
    the body is producing ``b`` (a constant or a copy of ``b``).

    Concretely after pipeline: pure ADDs/SUBs in the body that
    survive should produce ``200`` (the value of ``b`` in this
    test). We don't insist on a specific MirValue identity — DCE
    may have collapsed the original ``b`` definition into the
    final sum's instr.
    """
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    lp, body, _ = _mk_loop_arg(fn, entry, "i")
    a = _addi(body, fn.gp0_value, 100, hint="a")
    b = _addi(body, fn.gp0_value, 200, hint="b")
    s_ab = _add(body, a, b, hint="ab")
    s = _sub(body, s_ab, a, hint="s")
    _setreg(body, s)

    P.run_default_pipeline(fn)
    mir.verify(fn)
    # No SUB / no ADD instr should remain — the whole thing
    # collapses to ``ADDI gp0, 200`` (after const-fold sees both
    # ``a`` and ``a`` cancel via reassoc).
    arith_ops = [it for it in body.items
                 if isinstance(it, mir.MirInstr)
                 and it.opcode in ("S_ADD_INT", "S_SUB_INT")]
    assert len(arith_ops) == 0, (
        f"expected no surviving ADD/SUB; got "
        f"{[it.opcode for it in arith_ops]}"
    )
    # The setreg's operand value should be 200 (we check via the
    # producer instr).
    setregs = [it for it in body.items
               if isinstance(it, mir.MirInstr)
               and it.opcode == "C_SET_SCALE_REG"]
    assert len(setregs) == 1
    src = setregs[0].operands[0]
    # src is produced by some ADDI gp0, K.
    d = src.defined_by
    assert d is not None and d.opcode == "S_ADDI_INT"
    assert d.operands[0] is fn.gp0_value
    assert d.operands[1] == 200


def test_reassociate_prefix_share():
    """Two chains where one's leaves are a prefix of the other's
    share the partial sum."""
    fn, entry = _mk_fn()
    _attach_fn(entry, fn)
    lp, body, _ = _mk_loop_arg(fn, entry, "i")
    a = _addi(body, fn.gp0_value, 5, hint="a")  # leaf 1
    b = _addi(body, fn.gp0_value, 7, hint="b")  # leaf 2
    c = _addi(body, fn.gp0_value, 11, hint="c") # leaf 3
    # chain1 = a + b
    s1 = _add(body, a, b, hint="s1")
    _setreg(body, s1)
    # chain2 = a + b + c (should reuse s1)
    s2a = _add(body, a, b, hint="s2a")
    s2 = _add(body, s2a, c, hint="s2")
    _setreg(body, s2)

    before = sum(1 for it in body.items
                 if isinstance(it, mir.MirInstr)
                 and it.opcode in ("S_ADD_INT", "S_ADDI_INT"))
    # reassoc alone only RAUWs; DCE in the full pipeline sweeps
    # the now-dead instr — so test the pipeline, not the pass.
    P.run_default_pipeline(fn)
    mir.verify(fn)
    after = sum(1 for it in body.items
                if isinstance(it, mir.MirInstr)
                and it.opcode in ("S_ADD_INT", "S_ADDI_INT"))
    assert after < before, f"before={before} after={after}"
