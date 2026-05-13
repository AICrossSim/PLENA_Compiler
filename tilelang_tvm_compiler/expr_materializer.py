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
        # Eager flush: write the ISA to generated_code immediately so
        # call-order matches emit-order. Lazy isa strings caused
        # cross-call register clobbering (a later allocate_gp triggered
        # auto_spill or reuse that interleaved with an earlier lazy
        # "S_LD_INT gp{r}, ..." -- the same physical reg ended up being
        # loaded twice with different values, second one winning).
        self.shim.compiler.generated_code += isa
        return MaterializedExpr(
            register=r, isa="", owns_register=True, _materializer=self
        )

    def _materialize_var(self, v: tir.Var) -> MaterializedExpr:
        """Look up a bound var in the symbol table.

        Two binding forms:
          * ``int``                  — GP reg already holding the value
            (legacy / unroll loop idx). No alloc, no ISA.
          * ``("ram", intram_addr)`` — value lives in IntRAM (serial
            loop idx; see ``_emit_for``). Borrow a fresh GP, emit a
            ``S_LD_INT`` to load it, and return as ``owns_register=True``
            so the caller's release() returns the GP to the pool. Every
            use re-loads (cheap; avoids pinning a permanent GP for the
            idx in deeply nested kernels).
        """
        if v not in self.symbol_table:
            raise ExprMaterializeError(
                f"unbound tir.Var {v.name!r}; not in symbol_table "
                f"(known: {[x.name for x in self.symbol_table]!r})"
            )
        binding = self.symbol_table[v]
        if isinstance(binding, int):
            return MaterializedExpr(
                register=binding, isa="", owns_register=False, _materializer=self
            )
        if isinstance(binding, tuple) and len(binding) == 2 and binding[0] == "ram":
            ram_addr = int(binding[1])
            ra = self.shim.compiler.register_allocator
            reg = ra.allocate_gp(1)[0]
            # IMPORTANT: write the load ISA directly to ``generated_code``
            # rather than the lazy ``isa`` field. Auto-spill (triggered by
            # later allocs) emits S_ST/LD_INT eagerly into generated_code,
            # so a later isa-string concatenation would reorder relative
            # to those spill instructions and silently corrupt the value
            # this load was supposed to deliver.
            self.shim.compiler.generated_code += (
                f"; load ram-backed idx {v.name} <- intram[{ram_addr}]\n"
                f"S_LD_INT gp{reg}, gp0, {ram_addr}\n"
            )
            return MaterializedExpr(
                register=reg, isa="", owns_register=True, _materializer=self
            )
        raise ExprMaterializeError(
            f"symbol_table[{v.name!r}] has unsupported binding {binding!r}"
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
        # Pin m_lhs.register across both the m_rhs materialise AND the
        # out_reg alloc: any allocate_gp in between may trigger auto_spill,
        # which picks non-pinned in-use regs as victims. Without the pin,
        # m_lhs's value could be silently displaced to IntRAM and the same
        # physical register handed out as m_rhs's or out_reg's, aliasing
        # operands.
        ra = self.shim.compiler.register_allocator
        lhs_was_owned = m_lhs.owns_register
        if lhs_was_owned:
            ra.pin_gp(m_lhs.register)
        m_rhs = None
        try:
            m_rhs = self._materialize(rhs)
            # Same protection for m_rhs while we alloc out_reg.
            rhs_was_owned = m_rhs.owns_register
            if rhs_was_owned:
                ra.pin_gp(m_rhs.register)
            try:
                out_reg = ra.allocate_gp(1)[0]
            finally:
                if rhs_was_owned:
                    ra.unpin_gp(m_rhs.register)
        finally:
            if lhs_was_owned:
                ra.unpin_gp(m_lhs.register)

        # Eager flush. m_lhs.isa and m_rhs.isa are already empty under
        # the eager-emit invariant (their ISA was written to
        # generated_code at construction time); we keep the `+ m.isa`
        # bits for any legacy MaterializedExpr that might still carry a
        # non-empty isa string.
        self.shim.compiler.generated_code += m_lhs.isa + m_rhs.isa + (
            f"{opcode} gp{out_reg}, gp{m_lhs.register}, gp{m_rhs.register}\n"
        )
        isa = ""

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

        # x % 2^k = x - (x // 2^k) * 2^k. PLENA has no bitwise AND, but
        # the shift + multiply + subtract sequence uses only ops we
        # already support. Lower by rewriting the PrimExpr and
        # re-entering the materializer — FloorDiv hits the S_SRLI_INT
        # branch above and Mul-by-pow2 hits the S_SLLI_INT path the
        # binop dispatcher already handles.
        if op_str == "%":
            shift = _try_pow2_shift_amount(rhs)
            if shift is not None:
                m = 1 << shift
                shifted = tir.FloorDiv(lhs, tir.IntImm("int32", m))
                scaled = tir.Mul(shifted, tir.IntImm("int32", m))
                return self._materialize(tir.Sub(lhs, scaled))

        raise ExprMaterializeError(
            f"cannot lower runtime {op_str}: PLENA ISA has no integer divide and "
            f"no bitwise-AND. The only supported runtime forms are `x // 2^k` "
            f"(via S_SRLI_INT) and `x % 2^k` (lowered as `x - (x // 2^k) * 2^k`). "
            f"Got `{lhs} {op_str} {rhs}`. Restructure the kernel so this is "
            f"computed at compile time, or use a power-of-2 divisor."
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
        # Pin m_operand's register across the out_reg alloc -- same race
        # as _materialize_binop: an auto_spill triggered here could
        # otherwise displace m_operand's value to IntRAM and reuse the
        # same physical register for out_reg, causing reads from
        # m_operand.register to silently return out_reg's content.
        if m_operand.owns_register:
            ra.pin_gp(m_operand.register)
        try:
            out_reg = ra.allocate_gp(1)[0]
        finally:
            if m_operand.owns_register:
                ra.unpin_gp(m_operand.register)
        # Eager flush (see _materialize_binop / _materialize_var for
        # the rationale -- lazy isa strings interleave incorrectly with
        # eager auto-spill / ram-idx loads).
        self.shim.compiler.generated_code += m_operand.isa + (
            f"{opcode} gp{out_reg}, gp{m_operand.register}, {imm}\n"
        )
        isa = ""
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
