"""PreIsaIR (v2) → MIR conversion pass.

This is the "explicit conversion" layer the user asked for: every
``tir.PrimExpr`` operand of a PreIsaOp is expanded into a chain of
MIR instructions producing SSA values, then the PreIsaOp itself
becomes a MirInstr referencing those SSA values.

Pipeline contract:

  PreIsaIR (v2)   — opcode + PrimExpr/int/str operands, no registers
                    yet, address algebra fully symbolic.
       │
       │  this pass
       ▼
  MIR             — SSA values, def/use chains, loops with loop_kind.
                    Ready for LICM / CSE / DCE / regalloc.

Three things this pass does:

  1. **PrimExpr lowering**.  Each PrimExpr operand is recursively
     turned into a sequence of MirInstrs producing one MirValue per
     subexpression. ``tir.Add`` → ``S_ADD_INT`` / ``S_ADDI_INT``
     (folded if the constant fits), ``tir.Mul`` → ``S_MUL_INT`` /
     ``S_SLLI_INT`` (power-of-2 strength reduction), ``tir.Var``
     looked up in the symbol table.

  2. **Symbol table threading**.  The hw_consts (MLEN_VAR / BLEN_VAR
     / ...) are bound to constant-int MirValues at pass start.
     Loop variables (``tir.Var`` introduced by LoopRegion) are bound
     to body-block-argument MirValues at loop entry, unbound on exit.

  3. **arith.Analyzer simplification**.  Before lowering each
     PrimExpr we run ``arith.Analyzer().simplify`` to fold constants,
     normalise ``Add`` order, etc. Hw consts are pre-substituted
     to their IntImm bindings so e.g. ``BLEN_VAR * MLEN_VAR`` folds
     to a literal IntImm when the shim says blen=4, mlen=64. Loop
     vars are NOT substituted — they stay symbolic so MIR sees the
     loop-dependent algebra structurally.

Caching:
  Within a single PreIsaOp lowering we cache subexpression results
  by ``tvm.ir.structural_equal`` — two operands that are the same
  expression structurally share one MirValue chain. This is a free
  cheap CSE; the bigger CSE pass later does it across PreIsaOps.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union  # noqa: F401

from tvm import arith, tir
from tvm.tir import stmt_functor

from . import mir
from .pre_isa_ir_v2 import (
    LoopRegion, PreIsaOp, PreIsaModule, KNOWN_OPCODES,
)


class PreIsaToMirError(RuntimeError):
    pass


# ---------------------------------------------------------------------
# Operand classifier — what MIR operand kind a PreIsaOp operand maps to.
# ---------------------------------------------------------------------

# For each PreIsaIR opcode, we'd ideally cross-check against
# ``mir.OPCODES``'s operand kinds. We do that on the fly in
# ``_lower_one_preop`` — see ``_kind_at``.

def _kind_at(opcode: str, i: int) -> str:
    """Look up the MIR operand kind expected at slot ``i`` of
    ``opcode``."""
    spec = mir.OPCODES.get(opcode)
    if spec is None:
        raise PreIsaToMirError(
            f"PreIsaIR opcode {opcode!r} not in mir.OPCODES — add a "
            f"matching MIR entry."
        )
    if i >= len(spec.operand_kinds):
        raise PreIsaToMirError(
            f"{opcode}: PreIsaOp gave operand[{i}] but mir.OPCODES "
            f"arity is {len(spec.operand_kinds)}"
        )
    return spec.operand_kinds[i]


# ---------------------------------------------------------------------
# Conversion class
# ---------------------------------------------------------------------

class PreIsaToMir:
    """State for one PreIsaModule → MirFunction conversion."""

    def __init__(self, mod: PreIsaModule, shim) -> None:
        self.preisa = mod
        self.shim = shim
        self.fn = mir.MirFunction(name=mod.name)
        self.fn.metadata["buffers"] = dict(mod.buffers)
        # Current block where new MirInstrs / MirLoops are appended.
        self.cur_block: Optional[mir.MirBlock] = None
        # Symbol tables.
        # tir.Var (loop vars + hw_consts) → MirValue (or IntImm for
        # hw_consts that get pre-substituted).
        self.var_to_value: Dict[tir.Var, mir.MirValue] = {}
        # hw_const tir.Var → constant int (the shim's current value).
        # Used by arith.Analyzer to pre-substitute when simplifying.
        self.hw_const_values: Dict[tir.Var, int] = {}
        self._init_hw_consts()
        # ``gp0`` is a hardware-fixed constant-zero register used as a
        # source on instructions like ``S_ADDI_INT %dst, gp0, imm``.
        # We model it as a function-level constant MirValue (its
        # ``is_function_const`` flag is True; no instr produces it,
        # no block argument).
        self.gp0_value: Optional[mir.MirValue] = None
        # Per-PreIsaOp expression cache (structural_equal → MirValue).
        # Reset by ``_begin_preop``.
        self._expr_cache: List = []
        # PreIsaOp identity → its produced MirValue (for ops that
        # define an addr_reg or other typed result that downstream
        # PreIsaOps reference via ``PreIsaOp`` as an operand).
        self._preop_result: Dict[int, mir.MirValue] = {}
        # Analyzer reused across all simplifies.
        self._analyzer = arith.Analyzer()

    def _init_hw_consts(self) -> None:
        # Lazy import to avoid cycle.
        from .hw_consts import HW_CONST_ATTRS
        for var, attr in HW_CONST_ATTRS.items():
            self.hw_const_values[var] = int(getattr(self.shim, attr))

    def run(self) -> mir.MirFunction:
        # Set up the top block.
        top = mir.MirBlock(name="entry")
        self.fn.blocks.append(top)
        self.cur_block = top
        # gp0 is a function-level constant (MLIR-style: a value that
        # just "exists"; no instr produces it; no block argument). It
        # represents the hardware-fixed zero register.
        self.gp0_value = self.fn.make_gp0_const()

        # Walk top-level body.
        self._lower_items(self.preisa.body)
        return self.fn

    # -----------------------------------------------------------------
    # Body walk
    # -----------------------------------------------------------------
    def _lower_items(self, items) -> None:
        for it in items:
            if isinstance(it, PreIsaOp):
                self._lower_one_preop(it)
            elif isinstance(it, LoopRegion):
                self._lower_loop(it)
            else:
                raise PreIsaToMirError(
                    f"unexpected body item type: {type(it).__name__}"
                )

    def _lower_loop(self, lp: LoopRegion) -> None:
        # Create body block with loop_var as its block argument
        # (MLIR-style: the region supplies the induction var on each
        # entry; from the body's perspective it's just an SSA value).
        body_blk = mir.MirBlock(name=f"loop.body.{lp.loop_var.name}")
        lvar = self.fn.mint_value("i32", hint=lp.loop_var.name)
        body_blk.add_argument(lvar)
        # Build MirLoop + attach to current block.
        loop_obj = mir.MirLoop(
            name=f"L_{lp.loop_var.name}",
            loop_var=lvar,
            init=lp.init_imm,
            extent=lp.extent_imm,
            body=[body_blk],
            loop_kind=lp.loop_kind,
            # Forward any PreIsaIR LoopRegion annotations the
            # producer set (e.g. ``order_independent`` for the
            # reverse-iter optimisation in mir_to_isa).
            annotations=dict(lp.annotations),
        )
        self.cur_block.append(loop_obj)
        # Push symbol table.
        if lp.loop_var in self.var_to_value:
            raise PreIsaToMirError(
                f"loop_var {lp.loop_var.name!r} already bound — nested "
                f"loops using the same tir.Var aren't supported (producer "
                f"must mint a fresh tir.Var per LoopRegion)"
            )
        self.var_to_value[lp.loop_var] = lvar
        prev_block = self.cur_block
        self.cur_block = body_blk
        try:
            self._lower_items(lp.body)
        finally:
            self.cur_block = prev_block
            self.var_to_value.pop(lp.loop_var, None)

    # -----------------------------------------------------------------
    # PreIsaOp lowering
    # -----------------------------------------------------------------
    def _lower_one_preop(self, op: PreIsaOp) -> None:
        # Comment passes through as a meta MirInstr.
        if op.opcode == "_COMMENT":
            text = op.operands[0] if op.operands else ""
            self.cur_block.append(mir.MirInstr(
                opcode="_COMMENT",
                operands=[text],
                result=None,
            ))
            return

        # Per-op expr cache.
        self._expr_cache = []
        spec = mir.OPCODES.get(op.opcode)
        if spec is None:
            raise PreIsaToMirError(
                f"PreIsaOp opcode {op.opcode!r} not in mir.OPCODES"
            )
        if len(op.operands) != len(spec.operand_kinds):
            raise PreIsaToMirError(
                f"{op.opcode}: PreIsaOp has {len(op.operands)} operands "
                f"but mir.OPCODES expects {len(spec.operand_kinds)}"
            )

        mir_operands: List = []
        for i, (val, kind) in enumerate(
            zip(op.operands, spec.operand_kinds),
        ):
            mir_operands.append(self._lower_operand(val, kind))

        # Allocate result MirValue if non-void.
        result: Optional[mir.MirValue] = None
        if spec.result_type != "void":
            result = self.fn.mint_value(spec.result_type)
        self.cur_block.append(mir.MirInstr(
            opcode=op.opcode,
            operands=mir_operands,
            result=result,
        ))
        # Record the result so downstream PreIsaOps referencing this
        # op via ``PreIsaOp`` operand can resolve it.
        if result is not None:
            self._preop_result[id(op)] = result

    # -----------------------------------------------------------------
    # Operand kind dispatch
    # -----------------------------------------------------------------
    def _lower_operand(self, val, kind: str):
        """Return the MIR-form operand for this PreIsaOp operand."""
        if kind == "i32":
            return self._lower_i32_operand(val)
        if kind == "literal_int":
            return self._lower_literal_int(val)
        if kind == "fp_reg":
            return self._lower_verbatim_str(val, kind)
        if kind == "verbatim_str":
            return self._lower_verbatim_str(val, kind)
        if kind == "addr_reg":
            return self._lower_addr_reg_operand(val)
        raise PreIsaToMirError(
            f"_lower_operand: unknown operand kind {kind!r}"
        )

    def _lower_i32_operand(self, val) -> mir.MirValue:
        """Turn a PrimExpr / int into an i32 MirValue."""
        if isinstance(val, int):
            return self._lower_primexpr(tir.IntImm("int32", int(val)))
        if isinstance(val, tir.PrimExpr):
            return self._lower_primexpr(val)
        raise PreIsaToMirError(
            f"i32 operand expects PrimExpr / int; got "
            f"{type(val).__name__} {val!r}"
        )

    def _lower_literal_int(self, val):
        """Pass-through compile-time int literal."""
        if isinstance(val, int):
            return int(val)
        if isinstance(val, tir.IntImm):
            return int(val.value)
        raise PreIsaToMirError(
            f"literal_int operand expects int / IntImm; got "
            f"{type(val).__name__} {val!r}"
        )

    def _lower_verbatim_str(self, val, kind: str) -> str:
        if isinstance(val, str):
            return val
        raise PreIsaToMirError(
            f"{kind} operand expects str; got {type(val).__name__} {val!r}"
        )

    def _lower_addr_reg_operand(self, val):
        # ``PreIsaOp`` operand → the MirValue produced by that op's
        # earlier lowering. The producer must have emitted the
        # referenced op before this consumer; we look it up in
        # ``_preop_result``.
        if isinstance(val, PreIsaOp):
            mv = self._preop_result.get(id(val))
            if mv is None:
                raise PreIsaToMirError(
                    f"addr_reg operand: PreIsaOp {val.opcode!r} was not "
                    f"lowered before this consumer (or it does not "
                    f"produce an addr_reg result)"
                )
            if mv.dtype != "addr_reg":
                raise PreIsaToMirError(
                    f"addr_reg operand: referenced op {val.opcode!r} "
                    f"produces dtype {mv.dtype!r}, not addr_reg"
                )
            return mv
        raise PreIsaToMirError(
            f"addr_reg operand expects a PreIsaOp reference to a "
            f"producer (typically C_SET_ADDR_REG); got "
            f"{type(val).__name__} {val!r}"
        )

    # -----------------------------------------------------------------
    # PrimExpr → SSA chain (the heart of the conversion)
    # -----------------------------------------------------------------
    def _lower_primexpr(self, expr) -> mir.MirValue:
        """Lower a PrimExpr into a chain of MirInstrs ending in an
        i32 MirValue."""
        # Simplify first (substitute hw_consts to IntImm + arith.Analyzer
        # simplify). Loop vars stay symbolic.
        expr = self._simplify(expr)

        # Structural-equal cache lookup.
        for cached_expr, cached_val in self._expr_cache:
            try:
                from tvm import ir as _ir
                if _ir.structural_equal(expr, cached_expr):
                    return cached_val
            except Exception:
                pass

        val = self._emit_primexpr(expr)
        self._expr_cache.append((expr, val))
        return val

    def _simplify(self, expr):
        """Substitute hw_const Vars to their IntImm values, then run
        arith.Analyzer().simplify. Loop var Vars are intentionally
        NOT substituted — they must stay symbolic in MIR so loop-
        invariant analysis can spot them."""
        if not isinstance(expr, tir.PrimExpr):
            return expr
        # Substitute only hw consts.
        var_map = {
            v: tir.IntImm("int32", n)
            for v, n in self.hw_const_values.items()
        }
        if var_map:
            try:
                expr = stmt_functor.substitute(expr, var_map)
            except Exception:
                pass
        try:
            expr = self._analyzer.simplify(expr)
        except Exception:
            pass
        return expr

    def _emit_primexpr(self, expr) -> mir.MirValue:
        """Recursive emit; assumes ``expr`` has been simplified."""
        if isinstance(expr, tir.IntImm):
            return self._emit_intimm(int(expr.value))
        if isinstance(expr, tir.Var):
            # Loop var lookup.
            mv = self.var_to_value.get(expr)
            if mv is None:
                # Hw const that escaped substitution? Shouldn't happen
                # after _simplify.
                if expr in self.hw_const_values:
                    return self._emit_intimm(self.hw_const_values[expr])
                raise PreIsaToMirError(
                    f"unbound tir.Var {expr.name!r} in PrimExpr; not "
                    f"a loop var and not in hw_consts"
                )
            return mv
        if isinstance(expr, tir.Add):
            return self._emit_add(expr.a, expr.b)
        if isinstance(expr, tir.Sub):
            return self._emit_sub(expr.a, expr.b)
        if isinstance(expr, tir.Mul):
            return self._emit_mul(expr.a, expr.b)
        if isinstance(expr, tir.FloorDiv):
            return self._emit_floordiv(expr.a, expr.b)
        if isinstance(expr, tir.FloorMod):
            return self._emit_floormod(expr.a, expr.b)
        if isinstance(expr, tir.Call):
            return self._emit_call(expr)
        # tir.Cast / Min / Max etc. — extend as we hit them.
        raise PreIsaToMirError(
            f"unsupported PrimExpr node: {type(expr).__name__} ({expr!r})"
        )

    def _emit_call(self, expr: "tir.Call") -> mir.MirValue:
        """Lower a TIR Call to a MIR instruction. Currently supports:
          * ``tir.shift_left(x, k)``  → ``S_SLLI_INT %x, k`` (k literal)
                                       or ``S_SLL_INT %x, %k`` (k reg)
          * ``tir.shift_right(x, k)`` → ``S_SRLI_INT %x, k`` / ``S_SRL_INT %x, %k``
        """
        op_name = expr.op.name if hasattr(expr.op, "name") else str(expr.op)
        if op_name in ("tir.shift_left", "tir.shift_right"):
            if len(expr.args) != 2:
                raise PreIsaToMirError(
                    f"{op_name}: expected 2 args; got {len(expr.args)}"
                )
            x, k = expr.args
            is_left = (op_name == "tir.shift_left")
            if _is_intlike(k):
                # Immediate shift amount.
                k_int = _intval(k)
                if k_int == 0:
                    return self._lower_primexpr(x)
                lhs = self._lower_primexpr(x)
                dst = self.fn.mint_value("i32")
                self.cur_block.append(mir.MirInstr(
                    opcode="S_SLLI_INT" if is_left else "S_SRLI_INT",
                    operands=[lhs, k_int],
                    result=dst,
                ))
                return dst
            # Reg-amount shift.
            lhs = self._lower_primexpr(x)
            rhs = self._lower_primexpr(k)
            dst = self.fn.mint_value("i32")
            self.cur_block.append(mir.MirInstr(
                opcode="S_SLL_INT" if is_left else "S_SRL_INT",
                operands=[lhs, rhs],
                result=dst,
            ))
            return dst
        raise PreIsaToMirError(
            f"unsupported PrimExpr Call: {op_name} ({expr!r})"
        )

    # ---- leaves and small helpers ----
    def _emit_intimm(self, n: int) -> mir.MirValue:
        if n == 0:
            return self.gp0_value
        dst = self.fn.mint_value("i32")
        # Fits in S_ADDI_INT immediate? bound 65535 (imm16).
        if 0 <= n <= 65535:
            self.cur_block.append(mir.MirInstr(
                opcode="S_ADDI_INT",
                operands=[self.gp0_value, n],
                result=dst,
            ))
            return dst
        # Two-instr form: S_LUI_INT upper; S_ADDI_INT lower.
        upper = n >> 12
        lower = n & 0xFFF
        hi = self.fn.mint_value("i32")
        self.cur_block.append(mir.MirInstr(
            opcode="S_LUI_INT",
            operands=[upper],
            result=hi,
        ))
        self.cur_block.append(mir.MirInstr(
            opcode="S_ADDI_INT",
            operands=[hi, lower],
            result=dst,
        ))
        return dst

    def _emit_add(self, a, b) -> mir.MirValue:
        # x + 0 → x  /  Both intlike → fold.
        if _is_intlike(a) and _is_intlike(b):
            return self._emit_intimm(_intval(a) + _intval(b))
        if _is_intlike(b) and _intval(b) == 0:
            return self._lower_primexpr(a)
        if _is_intlike(a) and _intval(a) == 0:
            return self._lower_primexpr(b)
        # Imm form: S_ADDI_INT %a, IMM  (fits in immediate).
        if _is_intlike(b) and 0 <= _intval(b) <= 65535:
            lhs = self._lower_primexpr(a)
            dst = self.fn.mint_value("i32")
            self.cur_block.append(mir.MirInstr(
                opcode="S_ADDI_INT",
                operands=[lhs, _intval(b)],
                result=dst,
            ))
            return dst
        if _is_intlike(a) and 0 <= _intval(a) <= 65535:
            rhs = self._lower_primexpr(b)
            dst = self.fn.mint_value("i32")
            self.cur_block.append(mir.MirInstr(
                opcode="S_ADDI_INT",
                operands=[rhs, _intval(a)],
                result=dst,
            ))
            return dst
        # General form.
        lhs = self._lower_primexpr(a)
        rhs = self._lower_primexpr(b)
        dst = self.fn.mint_value("i32")
        self.cur_block.append(mir.MirInstr(
            opcode="S_ADD_INT",
            operands=[lhs, rhs],
            result=dst,
        ))
        return dst

    def _emit_sub(self, a, b) -> mir.MirValue:
        if _is_intlike(a) and _is_intlike(b):
            return self._emit_intimm(_intval(a) - _intval(b))
        if _is_intlike(b) and _intval(b) == 0:
            return self._lower_primexpr(a)
        lhs = self._lower_primexpr(a)
        rhs = self._lower_primexpr(b)
        dst = self.fn.mint_value("i32")
        self.cur_block.append(mir.MirInstr(
            opcode="S_SUB_INT",
            operands=[lhs, rhs],
            result=dst,
        ))
        return dst

    def _emit_mul(self, a, b) -> mir.MirValue:
        if _is_intlike(a) and _is_intlike(b):
            return self._emit_intimm(_intval(a) * _intval(b))
        # x * 0 / 0 * x → 0.
        if _is_intlike(b) and _intval(b) == 0:
            return self.gp0_value
        if _is_intlike(a) and _intval(a) == 0:
            return self.gp0_value
        # x * 1 / 1 * x → x.
        if _is_intlike(b) and _intval(b) == 1:
            return self._lower_primexpr(a)
        if _is_intlike(a) and _intval(a) == 1:
            return self._lower_primexpr(b)
        # Strength reduce x * 2^k → SLLI_INT %x, k.
        if _is_intlike(b):
            k = _try_pow2_shift(_intval(b))
            if k is not None:
                lhs = self._lower_primexpr(a)
                dst = self.fn.mint_value("i32")
                self.cur_block.append(mir.MirInstr(
                    opcode="S_SLLI_INT",
                    operands=[lhs, k],
                    result=dst,
                ))
                return dst
        if _is_intlike(a):
            k = _try_pow2_shift(_intval(a))
            if k is not None:
                rhs = self._lower_primexpr(b)
                dst = self.fn.mint_value("i32")
                self.cur_block.append(mir.MirInstr(
                    opcode="S_SLLI_INT",
                    operands=[rhs, k],
                    result=dst,
                ))
                return dst
        # General S_MUL_INT.
        lhs = self._lower_primexpr(a)
        rhs = self._lower_primexpr(b)
        dst = self.fn.mint_value("i32")
        self.cur_block.append(mir.MirInstr(
            opcode="S_MUL_INT",
            operands=[lhs, rhs],
            result=dst,
        ))
        return dst

    def _emit_floordiv(self, a, b) -> mir.MirValue:
        if _is_intlike(a) and _is_intlike(b):
            return self._emit_intimm(_intval(a) // _intval(b))
        # x // 2^k → SRLI_INT.
        if _is_intlike(b):
            k = _try_pow2_shift(_intval(b))
            if k is not None:
                lhs = self._lower_primexpr(a)
                dst = self.fn.mint_value("i32")
                self.cur_block.append(mir.MirInstr(
                    opcode="S_SRLI_INT",
                    operands=[lhs, k],
                    result=dst,
                ))
                return dst
        # Non-power-of-2 divisor: explicit unsigned integer divide.
        # (Added for real Open-Sora dims, e.g. hidden/MLEN=3, head_count=24.)
        lhs = self._lower_primexpr(a)
        rhs = self._lower_primexpr(b)
        dst = self.fn.mint_value("i32")
        self.cur_block.append(mir.MirInstr(
            opcode="S_DIV_INT",
            operands=[lhs, rhs],
            result=dst,
        ))
        return dst

    def _emit_floormod(self, a, b) -> mir.MirValue:
        if _is_intlike(a) and _is_intlike(b):
            return self._emit_intimm(_intval(a) % _intval(b))
        # x % 2^k = x - (x >> k) << k. Emit directly in MIR
        # without re-running arith.simplify (which would fold the
        # SRLI+SLLI+Sub chain straight back into FloorMod and
        # loop forever).
        if _is_intlike(b):
            k = _try_pow2_shift(_intval(b))
            if k is not None:
                lhs = self._lower_primexpr(a)
                shifted = self.fn.mint_value("i32")
                self.cur_block.append(mir.MirInstr(
                    opcode="S_SRLI_INT",
                    operands=[lhs, k],
                    result=shifted,
                ))
                scaled = self.fn.mint_value("i32")
                self.cur_block.append(mir.MirInstr(
                    opcode="S_SLLI_INT",
                    operands=[shifted, k],
                    result=scaled,
                ))
                dst = self.fn.mint_value("i32")
                self.cur_block.append(mir.MirInstr(
                    opcode="S_SUB_INT",
                    operands=[lhs, scaled],
                    result=dst,
                ))
                return dst
        # Non-power-of-2 modulus: explicit unsigned remainder.
        lhs = self._lower_primexpr(a)
        rhs = self._lower_primexpr(b)
        dst = self.fn.mint_value("i32")
        self.cur_block.append(mir.MirInstr(
            opcode="S_REM_INT",
            operands=[lhs, rhs],
            result=dst,
        ))
        return dst


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _is_intlike(x) -> bool:
    return isinstance(x, (int, tir.IntImm))


def _intval(x) -> int:
    if isinstance(x, tir.IntImm):
        return int(x.value)
    return int(x)


def _try_pow2_shift(n: int) -> Optional[int]:
    if n <= 1 or (n & (n - 1)) != 0:
        return None
    k = n.bit_length() - 1
    if k > 31:
        return None
    return k


# ---------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------

def convert(mod: PreIsaModule, shim) -> mir.MirFunction:
    """Convert one PreIsaIR v2 module to a MirFunction. ``shim`` is
    used for hw_const substitution (mlen / blen / etc.)."""
    return PreIsaToMir(mod, shim).run()


__all__ = ["convert", "PreIsaToMir", "PreIsaToMirError"]
