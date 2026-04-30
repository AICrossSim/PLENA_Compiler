"""Lower a `tir.PrimExpr` tree into ISA + a register holding the value.

This is the Phase-1 foundation for handling dynamic / symbolic values
(loop vars, slice offsets, tensor strides that depend on shape vars).
It is intentionally NOT yet wired into the main pipeline -- the existing
BTMM end-to-end test does not exercise any PrimExpr arg, so adding this
module is purely additive and cannot regress that path.

Typical use, once wired in:

    sym_table: dict[tir.Var, int] = {}     # var -> currently-bound GP reg
    mat = ExprMaterializer(shim, sym_table)
    m = mat.materialize(my_expr)
    shim.compiler.generated_code += m.isa
    # ... emit an instruction that uses gp{m.register} ...
    m.release()                            # frees register + intermediates

Design notes:
    - We do NOT try to be a peephole optimizer. Constant folding for
      pure-literal subtrees and a handful of trivial identities (mul by
      1, add 0) are the only "smarts". Everything else compiles to the
      obvious ADD/SUB/MUL chain.
    - Every materialised value lives in exactly one GP register at the
      end. Intermediate registers are freed eagerly to keep pressure low
      on the small (16-entry) GP pool.
    - PrimExpr nodes we don't handle yet raise loudly. Better to fail
      visibly than silently produce wrong ISA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from tvm import tir

from .program_shim import ProgramShim


# Maximum unsigned literal that fits in a single S_ADDI_INT immediate.
# (preload_addr_reg.py uses the same bound.)
_S_ADDI_MAX = 262143


class ExprMaterializeError(RuntimeError):
    pass


@dataclass
class MaterializedExpr:
    """Result of materialising a PrimExpr.

    `register` holds the value AFTER `isa` is emitted. The caller is
    responsible for emitting `isa` into the ISA stream and then either
    consuming `register` (using it in a subsequent instruction) and
    calling `release()` when done, or copying its value elsewhere first.
    """
    register: int
    isa: str
    owns_register: bool             # caller may free `register` via release()
    intermediates: List[int] = field(default_factory=list)
    _materializer: "ExprMaterializer | None" = None

    def release(self) -> None:
        """Free `register` (if owned) and any intermediates we held on to."""
        if self._materializer is None:
            return
        ra = self._materializer.shim.compiler.register_allocator
        for r in self.intermediates:
            ra.free_gp([r])
        if self.owns_register:
            ra.free_gp([self.register])
        self.intermediates = []
        self.owns_register = False
        self._materializer = None


class ExprMaterializer:
    """Lowers `tir.PrimExpr` -> ISA text + GP register.

    The `symbol_table` maps already-bound `tir.Var` instances (typically
    loop indices) to the GP register currently holding their value. The
    ForOp emit code in the ISA pass is responsible for installing /
    removing entries when entering / leaving a loop body.
    """

    def __init__(self, shim: ProgramShim, symbol_table: Dict[tir.Var, int]) -> None:
        self.shim = shim
        self.symbol_table = symbol_table

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def materialize(self, expr) -> MaterializedExpr:
        """Top-level entry. Always returns a MaterializedExpr."""
        return self._materialize(expr)

    # ------------------------------------------------------------------
    # core dispatch
    # ------------------------------------------------------------------
    def _materialize(self, expr) -> MaterializedExpr:
        # Plain Python ints sneak in via address-allocation; treat them
        # as IntImm.
        if isinstance(expr, int):
            return self._materialize_int(int(expr))

        if isinstance(expr, tir.IntImm):
            return self._materialize_int(int(expr.value))

        if isinstance(expr, tir.Var):
            return self._materialize_var(expr)

        if isinstance(expr, tir.Add):
            # Flatten one nested Add of constants:  Add(c1, Add(c2, x))  ->
            # Add(c1+c2, x). Saves a load+add when the inner constant comes
            # from a buffer base and the outer from a lane offset (or vice
            # versa). Also helps the `Add(imm, var)` fast-path below kick in.
            a, b = expr.a, expr.b
            if _is_intlike(a) and isinstance(b, tir.Add):
                if _is_intlike(b.a):
                    a, b = tir.IntImm("int32", _int_value(a) + _int_value(b.a)), b.b
                elif _is_intlike(b.b):
                    a, b = tir.IntImm("int32", _int_value(a) + _int_value(b.b)), b.a
            elif _is_intlike(b) and isinstance(a, tir.Add):
                if _is_intlike(a.a):
                    a, b = tir.IntImm("int32", _int_value(b) + _int_value(a.a)), a.b
                elif _is_intlike(a.b):
                    a, b = tir.IntImm("int32", _int_value(b) + _int_value(a.b)), a.a
            # Fast path: Add(IntImm, X)  -> S_ADDI_INT (one instr) when the
            # immediate fits and the OTHER side isn't itself a literal (the
            # both-literal case should constant-fold via _materialize_binop).
            if _is_intlike(a) and not _is_intlike(b) and 0 <= _int_value(a) <= _S_ADDI_MAX:
                return self._materialize_unary_imm(b, "S_ADDI_INT", _int_value(a))
            if _is_intlike(b) and not _is_intlike(a) and 0 <= _int_value(b) <= _S_ADDI_MAX:
                return self._materialize_unary_imm(a, "S_ADDI_INT", _int_value(b))
            return self._materialize_binop(
                a, b, "S_ADD_INT", lambda x, y: x + y, identity_const=0
            )

        if isinstance(expr, tir.Sub):
            # x - 0 -> x (but NOT 0 - x, which would be negation).
            if _is_intlike(expr.b) and _int_value(expr.b) == 0:
                return self._materialize(expr.a)
            return self._materialize_binop(expr.a, expr.b, "S_SUB_INT", lambda x, y: x - y)

        if isinstance(expr, tir.Mul):
            # Fold both-literal subtrees FIRST (before strength reduction),
            # so e.g. 4*64 collapses to a single LI of 256 rather than to
            # an S_SLLI_INT we don't actually need.
            if _is_intlike(expr.a) and _is_intlike(expr.b):
                return self._materialize_int(_int_value(expr.a) * _int_value(expr.b))
            # Strength reduce `x * 2^k` -> S_SLLI_INT. One instr instead
            # of two (avoids the LI for the literal multiplier and avoids
            # using the multiplier itself).
            shift = _try_pow2_shift_amount(expr.b)
            if shift is not None:
                return self._materialize_unary_imm(expr.a, "S_SLLI_INT", shift)
            shift = _try_pow2_shift_amount(expr.a)
            if shift is not None:
                return self._materialize_unary_imm(expr.b, "S_SLLI_INT", shift)
            return self._materialize_binop(
                expr.a, expr.b, "S_MUL_INT", lambda x, y: x * y, identity_const=1
            )

        # FloorDiv / FloorMod: PLENA ISA has no integer divide and no
        # shift, so we can ONLY handle the case where both operands are
        # literals (compile-time fold) or where the divisor is 1.
        # Anything else surfaces as an error -- the kernel author has to
        # restructure their code to avoid runtime division.
        if isinstance(expr, tir.FloorDiv):
            return self._materialize_floordivmod(expr.a, expr.b, "//", lambda x, y: x // y)

        if isinstance(expr, tir.FloorMod):
            return self._materialize_floordivmod(expr.a, expr.b, "%", lambda x, y: x % y)

        # tir.shift_left / shift_right surface as Call nodes (Op("tir.shift_*")).
        # Constant-fold both-literal cases; lower the rest to PLENA shift
        # instructions (S_SLLI_INT for literal RHS, S_SLL_INT for register RHS).
        if isinstance(expr, tir.Call):
            op_name = getattr(expr.op, "name", None) if hasattr(expr, "op") else None
            if op_name in ("tir.shift_left", "tir.shift_right"):
                lhs, rhs = expr.args[0], expr.args[1]
                py = (lambda a, b: a << b) if op_name == "tir.shift_left" else (lambda a, b: a >> b)
                if _is_intlike(lhs) and _is_intlike(rhs):
                    return self._materialize_int(py(_int_value(lhs), _int_value(rhs)))
                imm_op = "S_SLLI_INT" if op_name == "tir.shift_left" else "S_SRLI_INT"
                reg_op = "S_SLL_INT" if op_name == "tir.shift_left" else "S_SRL_INT"
                if _is_intlike(rhs):
                    return self._materialize_unary_imm(lhs, imm_op, _int_value(rhs))
                # Variable shift amount: S_SLL_INT rd, rs1, rs2.
                return self._materialize_binop(lhs, rhs, reg_op, py)

        raise ExprMaterializeError(
            f"unsupported PrimExpr node: {type(expr).__name__} ({expr!r})"
        )

    # ------------------------------------------------------------------
    # leaf cases
    # ------------------------------------------------------------------
    def _materialize_int(self, n: int) -> MaterializedExpr:
        """Produce a register holding integer literal `n`."""
        ra = self.shim.compiler.register_allocator
        r = ra.allocate_gp(1)[0]
        if 0 <= n <= _S_ADDI_MAX:
            isa = f"S_ADDI_INT gp{r}, gp0, {n}\n"
        elif n > _S_ADDI_MAX:
            # Two-instruction form: load upper 20 bits, then add lower 12.
            upper = n >> 12
            lower = n & 0xFFF
            isa = (
                f"S_LUI_INT gp{r}, {upper}\n"
                f"S_ADDI_INT gp{r}, gp{r}, {lower}\n"
            )
        else:
            # Negative immediates aren't part of typical PLENA use cases
            # (offsets, sizes are >=0). Surface this loudly when it
            # eventually comes up so we can decide on a proper encoding.
            raise ExprMaterializeError(
                f"negative int literal not supported yet: {n}"
            )
        return MaterializedExpr(
            register=r, isa=isa, owns_register=True, _materializer=self
        )

    def _materialize_var(self, v: tir.Var) -> MaterializedExpr:
        """Look up a bound var in the symbol table; do not allocate."""
        if v not in self.symbol_table:
            raise ExprMaterializeError(
                f"unbound tir.Var {v.name!r}; not in symbol_table "
                f"(known: {[x.name for x in self.symbol_table]!r})"
            )
        reg = self.symbol_table[v]
        return MaterializedExpr(
            register=reg, isa="", owns_register=False, _materializer=self
        )

    # ------------------------------------------------------------------
    # binary ops (Add / Sub / Mul share this skeleton)
    # ------------------------------------------------------------------
    def _materialize_binop(
        self,
        lhs,
        rhs,
        opcode: str,
        py_op,
        identity_const: int | None = None,
    ) -> MaterializedExpr:
        # Constant fold both-literal subtrees so we don't burn a register
        # on something the compiler already knows.
        if _is_intlike(lhs) and _is_intlike(rhs):
            return self._materialize_int(py_op(_int_value(lhs), _int_value(rhs)))

        # Trivial identity: x * 1 / 1 * x  -- skip the multiplication.
        if identity_const is not None:
            if _is_intlike(rhs) and _int_value(rhs) == identity_const:
                return self._materialize(lhs)
            if _is_intlike(lhs) and _int_value(lhs) == identity_const:
                return self._materialize(rhs)

        m_lhs = self._materialize(lhs)
        m_rhs = self._materialize(rhs)

        ra = self.shim.compiler.register_allocator
        out_reg = ra.allocate_gp(1)[0]
        isa = m_lhs.isa + m_rhs.isa + (
            f"{opcode} gp{out_reg}, gp{m_lhs.register}, gp{m_rhs.register}\n"
        )

        # Eagerly free operand registers we own; the result is in out_reg.
        if m_lhs.owns_register:
            ra.free_gp([m_lhs.register])
        if m_rhs.owns_register:
            ra.free_gp([m_rhs.register])
        # Inherit any intermediates the operands collected (so the caller
        # can release them transitively if they want).
        intermediates = list(m_lhs.intermediates) + list(m_rhs.intermediates)

        return MaterializedExpr(
            register=out_reg,
            isa=isa,
            owns_register=True,
            intermediates=intermediates,
            _materializer=self,
        )


    def _materialize_floordivmod(self, lhs, rhs, op_str: str, py_op) -> MaterializedExpr:
        """FloorDiv / FloorMod: only fold-or-identity-or-shift, never an
        actual hardware divide (PLENA has no integer divide instruction).
        """
        if _is_intlike(lhs) and _is_intlike(rhs):
            b = _int_value(rhs)
            if b == 0:
                raise ExprMaterializeError(f"division by zero in expr ({lhs} {op_str} {rhs})")
            return self._materialize_int(py_op(_int_value(lhs), b))

        # x // 1  -> x   ;   x % 1 -> 0
        if _is_intlike(rhs) and _int_value(rhs) == 1:
            if op_str == "//":
                return self._materialize(lhs)
            else:  # mod 1 is always 0
                return self._materialize_int(0)

        # x // 2^k  ->  S_SRLI_INT x, k.  Unblocks the most common
        # "runtime divide" case (block index from element index).
        if op_str == "//":
            shift = _try_pow2_shift_amount(rhs)
            if shift is not None:
                return self._materialize_unary_imm(lhs, "S_SRLI_INT", shift)

        # x % 2^k would normally be `x & ((1<<k)-1)`, but PLENA has no AND,
        # so we cannot lower this. Fall through to the error.

        raise ExprMaterializeError(
            f"cannot lower runtime {op_str}: PLENA ISA has no integer divide and "
            f"no bitwise-AND. The only supported runtime forms are `x // 2^k` "
            f"(via S_SRLI_INT). Got `{lhs} {op_str} {rhs}`. Restructure the "
            f"kernel so this is computed at compile time, or use a power-of-2 "
            f"divisor."
        )


    def _materialize_unary_imm(
        self,
        operand,
        opcode: str,
        imm: int,
        _identity_when_zero: bool = True,
    ) -> MaterializedExpr:
        """Common shape for `<opcode> rd, rs1, imm` where `rs1` comes from
        materialising `operand` and `imm` is an integer baked into the ISA
        text.

        Used for shifts (S_SLLI_INT / S_SRLI_INT) where the shift amount
        is a compile-time literal.
        """
        # Shift by zero is a no-op -- skip the instruction entirely.
        if _identity_when_zero and imm == 0:
            return self._materialize(operand)

        m_operand = self._materialize(operand)
        ra = self.shim.compiler.register_allocator
        out_reg = ra.allocate_gp(1)[0]
        isa = m_operand.isa + (
            f"{opcode} gp{out_reg}, gp{m_operand.register}, {imm}\n"
        )
        if m_operand.owns_register:
            ra.free_gp([m_operand.register])
        return MaterializedExpr(
            register=out_reg,
            isa=isa,
            owns_register=True,
            intermediates=list(m_operand.intermediates),
            _materializer=self,
        )


def _is_intlike(x) -> bool:
    return isinstance(x, int) or isinstance(x, tir.IntImm)


def _int_value(x) -> int:
    return int(x.value) if isinstance(x, tir.IntImm) else int(x)


def _try_pow2_shift_amount(x) -> int | None:
    """If `x` is a positive int literal that is a power of two, return its
    log2 (i.e. the shift amount). Otherwise None.

    Caps at 31 because the PLENA shift instructions take the shift amount
    mod 32, so anything >= 32 would silently misbehave; we'd rather force
    the caller down the regular MUL/DIV path (which still folds at compile
    time if both sides are literal).
    """
    if not _is_intlike(x):
        return None
    n = _int_value(x)
    if n <= 1 or (n & (n - 1)) != 0:
        return None
    k = n.bit_length() - 1
    if k > 31:
        return None
    return k


__all__ = ["ExprMaterializer", "MaterializedExpr", "ExprMaterializeError"]
