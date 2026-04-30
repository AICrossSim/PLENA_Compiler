"""Standalone tests for ExprMaterializer.

Run:
    LD_LIBRARY_PATH="" \\
    PYTHONPATH=/home/.../PLENA_Simulator/compiler \\
    /home/.../PLENA_Simulator/.venv-tvm/bin/python -m \\
        tilelang_tvm_compiler.tests.test_expr_materializer

These tests do NOT touch the BTMM pipeline -- they exercise expr lowering
in isolation so we can iterate on it before wiring it into Pass 3.
"""

from __future__ import annotations

import sys

from tvm import tir

from tilelang_tvm_compiler.expr_materializer import (
    ExprMaterializeError,
    ExprMaterializer,
)
from tilelang_tvm_compiler.program_shim import make_shim


def _new_materializer():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    return ExprMaterializer(shim, symbol_table={}), shim


# ---------------------------------------------------------------------------
# Test 1: literal int
# ---------------------------------------------------------------------------
def test_literal_int_small():
    mat, _ = _new_materializer()
    m = mat.materialize(tir.IntImm("int32", 42))
    assert m.owns_register, "expected fresh reg for literal"
    assert "S_ADDI_INT" in m.isa and ", 42" in m.isa, f"bad isa: {m.isa!r}"
    print(f"[ok] literal small: reg=gp{m.register}, isa={m.isa.strip()}")


def test_literal_int_large():
    mat, _ = _new_materializer()
    m = mat.materialize(tir.IntImm("int32", 1234567))  # > 262143
    assert "S_LUI_INT" in m.isa and "S_ADDI_INT" in m.isa, f"bad isa: {m.isa!r}"
    upper = 1234567 >> 12
    lower = 1234567 & 0xFFF
    assert f", {upper}" in m.isa and f", {lower}" in m.isa
    print(f"[ok] literal large: reg=gp{m.register}, two-instr load")


# ---------------------------------------------------------------------------
# Test 2: bound var lookup -- no register allocated
# ---------------------------------------------------------------------------
def test_var_lookup_uses_bound_register():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    v = tir.Var("kv_block", "int32")
    mat = ExprMaterializer(shim, symbol_table={v: 7})  # pretend gp7 already holds it
    m = mat.materialize(v)
    assert m.register == 7
    assert m.isa == ""
    assert not m.owns_register
    print(f"[ok] var lookup: reg=gp{m.register} (no isa, no alloc)")


def test_var_unbound_raises():
    mat, _ = _new_materializer()
    raised = None
    try:
        mat.materialize(tir.Var("oops", "int32"))
    except ExprMaterializeError as e:
        raised = e
    assert raised is not None
    assert "unbound" in str(raised)
    print(f"[ok] unbound var raises: {raised}")


# ---------------------------------------------------------------------------
# Test 3: constant folding
# ---------------------------------------------------------------------------
def test_constant_fold_add():
    mat, _ = _new_materializer()
    expr = tir.Add(tir.IntImm("int32", 64), tir.IntImm("int32", 16))
    m = mat.materialize(expr)
    assert ", 80" in m.isa and "S_ADD_INT" not in m.isa, (
        f"expected folded literal 80, got: {m.isa!r}"
    )
    print(f"[ok] constant fold: 64+16=80 in single S_ADDI_INT")


def test_constant_fold_mul():
    mat, _ = _new_materializer()
    expr = tir.Mul(tir.IntImm("int32", 4), tir.IntImm("int32", 64))
    m = mat.materialize(expr)
    assert ", 256" in m.isa and "S_MUL_INT" not in m.isa
    print(f"[ok] constant fold: 4*64=256 in single S_ADDI_INT")


def test_mul_by_one_identity():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    v = tir.Var("x", "int32")
    mat = ExprMaterializer(shim, symbol_table={v: 5})
    m = mat.materialize(tir.Mul(v, tir.IntImm("int32", 1)))
    assert m.register == 5  # passed through, no S_MUL_INT
    assert "S_MUL_INT" not in m.isa
    print(f"[ok] x * 1 identity: returns same reg gp{m.register}")


# ---------------------------------------------------------------------------
# Test 4: compound expression -- the canonical "kv_block * 64 + 16"
# ---------------------------------------------------------------------------
def test_compound_loop_offset():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    kv = tir.Var("kv_block", "int32")
    # gp7 pretends to hold the loop counter. The materialiser must NOT
    # try to allocate or emit ISA for it -- only for the multiplication
    # by 64 and the +16.
    mat = ExprMaterializer(shim, symbol_table={kv: 7})
    expr = kv * tir.IntImm("int32", 64) + tir.IntImm("int32", 16)
    m = mat.materialize(expr)
    print(f"[compound] reg=gp{m.register}")
    print(f"[compound] isa:")
    for line in m.isa.strip().split("\n"):
        print(f"           {line}")
    # `kv * 64` strength-reduces to S_SLLI_INT (since 64 is a power of 2),
    # and `(kv<<6) + 16` collapses into one S_ADDI_INT (immediate fits).
    assert "S_SLLI_INT" in m.isa, f"kv*64 should use SLLI, got: {m.isa!r}"
    assert "S_MUL_INT" not in m.isa, "should not need a multiplier here"
    assert "S_ADDI_INT" in m.isa, "expected S_ADDI_INT for (kv<<6) + 16"
    assert "S_ADD_INT" not in m.isa, "non-immediate add should not appear here"
    print(f"[ok] compound: kv * 64 + 16 lowered correctly (uses SLLI + ADDI fast-path)")


# ---------------------------------------------------------------------------
# Test 5: register accounting -- after release(), free pool restored
# ---------------------------------------------------------------------------
def test_register_release_frees_pool():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    ra = shim.compiler.register_allocator
    free_before = len(ra._gp_free)
    mat = ExprMaterializer(shim, symbol_table={})
    m = mat.materialize(tir.IntImm("int32", 100))
    assert len(ra._gp_free) == free_before - 1
    m.release()
    assert len(ra._gp_free) == free_before, "release() must give the reg back"
    print(f"[ok] register release: pool restored ({free_before} free again)")


def test_compound_release_frees_all():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    ra = shim.compiler.register_allocator
    free_before = len(ra._gp_free)
    kv = tir.Var("kv_block", "int32")
    mat = ExprMaterializer(shim, symbol_table={kv: 7})
    m = mat.materialize(kv * tir.IntImm("int32", 64) + tir.IntImm("int32", 16))
    # During emission, intermediates were freed eagerly -- only the final
    # output reg should remain checked out.
    assert len(ra._gp_free) == free_before - 1, (
        f"expected only output reg held, got pool delta "
        f"{free_before - len(ra._gp_free)}"
    )
    m.release()
    assert len(ra._gp_free) == free_before
    print(f"[ok] compound release: full pool restored after release()")


# ---------------------------------------------------------------------------
# Test 6: FloorDiv / FloorMod -- fold when possible, raise when not
# ---------------------------------------------------------------------------
def test_floordiv_constant_fold():
    mat, _ = _new_materializer()
    expr = tir.FloorDiv(tir.IntImm("int32", 256), tir.IntImm("int32", 64))
    m = mat.materialize(expr)
    assert ", 4" in m.isa, f"expected literal 4, got {m.isa!r}"
    print(f"[ok] FloorDiv fold: 256 // 64 = 4")


def test_floormod_constant_fold():
    mat, _ = _new_materializer()
    expr = tir.FloorMod(tir.IntImm("int32", 100), tir.IntImm("int32", 64))
    m = mat.materialize(expr)
    assert ", 36" in m.isa
    print(f"[ok] FloorMod fold: 100 % 64 = 36")


def test_floordiv_by_one_identity():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    v = tir.Var("x", "int32")
    mat = ExprMaterializer(shim, symbol_table={v: 5})
    m = mat.materialize(tir.FloorDiv(v, tir.IntImm("int32", 1)))
    assert m.register == 5
    assert "S_DIV" not in m.isa
    print(f"[ok] x // 1 identity: returns same reg gp{m.register}")


def test_floordiv_runtime_non_pow2_raises():
    """Non-power-of-2 divisor: cannot strength-reduce to shift, must raise."""
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    a = tir.Var("a", "int32")
    mat = ExprMaterializer(shim, symbol_table={a: 3})
    raised = None
    try:
        # 7 is not a power of 2 -- can't be lowered to S_SRLI_INT, no
        # other integer-divide path exists, so this should still fail.
        mat.materialize(tir.FloorDiv(a, tir.IntImm("int32", 7)))
    except ExprMaterializeError as e:
        raised = e
    assert raised is not None
    msg = str(raised)
    assert "no integer divide" in msg, f"unexpected msg: {msg!r}"
    print(f"[ok] runtime non-pow2 FloorDiv raises: {msg[:60]}...")


def test_floordiv_div_by_zero_raises():
    mat, _ = _new_materializer()
    expr = tir.FloorDiv(tir.IntImm("int32", 5), tir.IntImm("int32", 0))
    raised = None
    try:
        mat.materialize(expr)
    except ExprMaterializeError as e:
        raised = e
    assert raised is not None
    print(f"[ok] div by zero raises: {raised}")


# ---------------------------------------------------------------------------
# Test 7: shift strength reduction (multiply / divide by power of 2)
# ---------------------------------------------------------------------------
def test_mul_by_pow2_uses_slli():
    """x * 64 should lower to a single S_SLLI_INT, not a load + S_MUL_INT."""
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    v = tir.Var("kv_block", "int32")
    mat = ExprMaterializer(shim, symbol_table={v: 7})
    m = mat.materialize(tir.Mul(v, tir.IntImm("int32", 64)))
    assert "S_SLLI_INT" in m.isa, f"expected SLLI, got: {m.isa!r}"
    assert "S_MUL_INT" not in m.isa
    assert ", 6" in m.isa, f"expected shift amount 6 (=log2(64)): {m.isa!r}"
    print(f"[ok] kv_block * 64 -> SLLI 6: {m.isa.strip()}")


def test_mul_by_pow2_when_lhs_is_const():
    """4 * x should still lower to S_SLLI_INT 2 (commutative)."""
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    v = tir.Var("x", "int32")
    mat = ExprMaterializer(shim, symbol_table={v: 5})
    m = mat.materialize(tir.Mul(tir.IntImm("int32", 4), v))
    assert "S_SLLI_INT" in m.isa and ", 2" in m.isa
    print(f"[ok] 4 * x -> SLLI 2: {m.isa.strip()}")


def test_mul_by_pow2_two_literals_still_folds():
    """Both-literal mul still folds, doesn't use SLLI."""
    mat, _ = _new_materializer()
    m = mat.materialize(tir.Mul(tir.IntImm("int32", 4), tir.IntImm("int32", 64)))
    assert "S_SLLI_INT" not in m.isa
    assert "S_MUL_INT" not in m.isa
    assert ", 256" in m.isa
    print(f"[ok] 4 * 64 still folds to literal 256")


def test_floordiv_by_pow2_uses_srli():
    """x // 8 should now succeed (was previously a hard error) via SRLI."""
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    v = tir.Var("idx", "int32")
    mat = ExprMaterializer(shim, symbol_table={v: 9})
    m = mat.materialize(tir.FloorDiv(v, tir.IntImm("int32", 8)))
    assert "S_SRLI_INT" in m.isa
    assert ", 3" in m.isa, f"expected shift amount 3 (=log2(8)): {m.isa!r}"
    print(f"[ok] idx // 8 -> SRLI 3: {m.isa.strip()}")


def test_floormod_by_pow2_still_raises():
    """x % 2^k requires AND, which PLENA doesn't have. Must still error."""
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    v = tir.Var("idx", "int32")
    mat = ExprMaterializer(shim, symbol_table={v: 9})
    raised = None
    try:
        mat.materialize(tir.FloorMod(v, tir.IntImm("int32", 8)))
    except ExprMaterializeError as e:
        raised = e
    assert raised is not None
    print(f"[ok] x % 8 still raises (no AND): {str(raised)[:60]}...")


def test_mul_by_non_pow2_still_uses_mul():
    """x * 7 (non-pow2) falls through to S_MUL_INT."""
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    v = tir.Var("x", "int32")
    mat = ExprMaterializer(shim, symbol_table={v: 5})
    m = mat.materialize(tir.Mul(v, tir.IntImm("int32", 7)))
    assert "S_MUL_INT" in m.isa
    assert "S_SLLI_INT" not in m.isa
    print(f"[ok] x * 7 (non-pow2) uses S_MUL_INT")


def test_shift_by_zero_is_identity():
    """x * 1 already handled by identity check; check x * 1 doesn't shift."""
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    v = tir.Var("x", "int32")
    mat = ExprMaterializer(shim, symbol_table={v: 5})
    m = mat.materialize(tir.Mul(v, tir.IntImm("int32", 1)))
    assert "S_SLLI_INT" not in m.isa
    assert m.register == 5
    print(f"[ok] x * 1 is identity (not SLLI 0)")


# ---------------------------------------------------------------------------
def main() -> int:
    tests = [
        test_literal_int_small,
        test_literal_int_large,
        test_var_lookup_uses_bound_register,
        test_var_unbound_raises,
        test_constant_fold_add,
        test_constant_fold_mul,
        test_mul_by_one_identity,
        test_compound_loop_offset,
        test_register_release_frees_pool,
        test_compound_release_frees_all,
        test_floordiv_constant_fold,
        test_floormod_constant_fold,
        test_floordiv_by_one_identity,
        test_floordiv_runtime_non_pow2_raises,
        test_floordiv_div_by_zero_raises,
        test_mul_by_pow2_uses_slli,
        test_mul_by_pow2_when_lhs_is_const,
        test_mul_by_pow2_two_literals_still_folds,
        test_floordiv_by_pow2_uses_srli,
        test_floormod_by_pow2_still_raises,
        test_mul_by_non_pow2_still_uses_mul,
        test_shift_by_zero_is_identity,
    ]
    print("=" * 60)
    print(f"ExprMaterializer tests ({len(tests)} cases)")
    print("=" * 60)
    for t in tests:
        t()
    print("=" * 60)
    print(f"ALL {len(tests)} TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
