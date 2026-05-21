"""MIR — machine IR for the PLENA backend.

This sits between :mod:`pre_isa_ir` (PrimExpr-operand form, machine-
neutral) and the final ISA text. It is the "explicit conversion" layer
the user wanted: where the abstract address algebra of PreIsaIR becomes
named SSA values, and where loop structure / def-use chains are
explicit enough to support standard machine-IR optimisations
(LICM, CSE, DCE, register allocation, spill-to-IntRAM).

Design summary
--------------
*  **SSA**.  Every value computed by an instruction is a unique
   :class:`MirValue` with a string name (``%0``, ``%1``, ...).
   Operands of subsequent instructions reference values by identity
   (the Python ``MirValue`` object), NOT by name; the name is a debug
   string only.  This mirrors LLVM's ``llvm::Value*`` model.

*  **def/use chain**.  Every ``MirValue`` knows the single instruction
   that produced it (``defined_by``) and every instruction that
   currently uses it (``used_by`` — kept up-to-date by operand mutators
   on ``MirInstr``).  This is what makes LICM cheap (free-vars of an
   expression are the ``defined_by`` set of its operand values), and
   what makes register allocation a graph problem on the live-range
   intervals.

*  **Block + terminators**.  Instructions live in :class:`MirBlock`s.
   Each block ends in exactly ONE terminator (a branch / loop-back /
   loop-end / return).  Non-terminator instructions in the middle are
   straight-line.

*  **Loop is a region**.  PLENA has a hardware loop primitive
   (``C_LOOP_START`` / ``C_LOOP_END``), and unrolled loops are a
   separate codegen choice.  We model both as :class:`MirLoop`
   regions tagged with ``loop_kind`` ∈ ``{"serial", "unroll"}`` — the
   loop IS the region; backend chooses how to lower it.

*  **Types**.  Three concrete types for now:
      - ``"i32"``      — an integer (occupies one GP register at lowering)
      - ``"addr_reg"`` — a PLENA address-register value (``aN``)
      - ``"fp_reg"``   — a PLENA FPU-register value (``fN``); these are
                          a hardware-fixed file (f0=0, f1, f2, ...) and
                          almost never live more than one instruction,
                          so we encode them as pinned named tokens
                          rather than real SSA values
   ``"void"`` is used as a result type for instructions that don't
   produce a value (``M_BTMM`` for example writes to the systolic
   array, not a GP).

The verifier (``verify``) catches:
   - operand type / count mismatches per opcode
   - def-before-use (every use must come after its def, modulo loop
     back-edges that the loop-region structure resolves)
   - dangling uses (used_by entries pointing to deleted instructions)
   - terminator-in-the-middle / non-terminator-at-end

Lowering passes (PreIsaPass → MIR, MIR optimise, MIR → ISA) live in
separate modules; this file is the data model + dump + verifier only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tvm import tir


# ----------------------------------------------------------------------
# Opcode table
# ----------------------------------------------------------------------

# Each opcode declares:
#   - result_type: "i32" / "addr_reg" / "fp_reg" / "void"
#   - operand_kinds: tuple of expected operand kinds, where each kind is
#       "i32"          — an i32 MirValue (or an int / IntImm immediate
#                         that will fold; the verifier accepts both)
#       "addr_reg"     — an addr_reg MirValue
#       "fp_reg"       — a verbatim FP register token (``"f0"`` / ``"f1"``)
#                         held as a Python str on the operand list
#       "literal_int"  — a Python int / IntImm bound at construction
#                         (loop bounds, mask flags, immediate fields)
#       "verbatim_str" — a Python str dropped into the ISA template
#   The same opcode may appear in two forms (different operand
#   counts) — those are encoded as separate _OpcodeSpec entries
#   with disambiguating internal names ('_'-prefixed for variants).
@dataclass(frozen=True)
class _OpcodeSpec:
    result_type: str
    operand_kinds: Tuple[str, ...]
    isa_mnemonic: str   # what the backend writes into the ASM text
    # Per-operand position in the emitted ISA, given as a list of
    # operand-list indices. Default = identity (0, 1, 2, ...). Used by
    # opcodes whose emit order differs from the construction order.
    isa_operand_order: Optional[Tuple[int, ...]] = None


# Note: this table is the SOURCE OF TRUTH for what a valid MIR
# instruction looks like. Anything not here will fail verify().
OPCODES: Dict[str, _OpcodeSpec] = {
    # ---- integer scalar ----
    "S_ADDI_INT": _OpcodeSpec(
        result_type="i32",
        operand_kinds=("i32", "literal_int"),
        isa_mnemonic="S_ADDI_INT",
    ),
    "S_ADD_INT": _OpcodeSpec(
        result_type="i32",
        operand_kinds=("i32", "i32"),
        isa_mnemonic="S_ADD_INT",
    ),
    "S_SUB_INT": _OpcodeSpec(
        result_type="i32",
        operand_kinds=("i32", "i32"),
        isa_mnemonic="S_SUB_INT",
    ),
    "S_MUL_INT": _OpcodeSpec(
        result_type="i32",
        operand_kinds=("i32", "i32"),
        isa_mnemonic="S_MUL_INT",
    ),
    "S_LUI_INT": _OpcodeSpec(
        result_type="i32",
        operand_kinds=("literal_int",),
        isa_mnemonic="S_LUI_INT",
    ),
    "S_LD_INT": _OpcodeSpec(
        # gp_dst = intram[gp_base + imm]
        result_type="i32",
        operand_kinds=("i32", "literal_int"),
        isa_mnemonic="S_LD_INT",
    ),
    "S_ST_INT": _OpcodeSpec(
        # intram[gp_base + imm] = gp_value (no result)
        result_type="void",
        operand_kinds=("i32", "i32", "literal_int"),
        isa_mnemonic="S_ST_INT",
    ),
    "S_SLLI_INT": _OpcodeSpec(
        result_type="i32",
        operand_kinds=("i32", "literal_int"),
        isa_mnemonic="S_SLLI_INT",
    ),
    "S_SRLI_INT": _OpcodeSpec(
        result_type="i32",
        operand_kinds=("i32", "literal_int"),
        isa_mnemonic="S_SRLI_INT",
    ),
    # Reg-amount shifts — the shift count comes from another GP rather
    # than an immediate. Used by packed-head mask expressions like
    # ``1 << (lane_var % lane_count)``.
    "S_SLL_INT": _OpcodeSpec(
        result_type="i32",
        operand_kinds=("i32", "i32"),
        isa_mnemonic="S_SLL_INT",
    ),
    "S_SRL_INT": _OpcodeSpec(
        result_type="i32",
        operand_kinds=("i32", "i32"),
        isa_mnemonic="S_SRL_INT",
    ),
    # ---- FP scalar ----
    "S_LD_FP": _OpcodeSpec(
        # f_dst = fpram[gp_addr + 0]; no MIR-level SSA result for fpregs
        # (they're a fixed file). The producer carries the fp register
        # as a verbatim_str on operand 0.
        result_type="void",
        operand_kinds=("fp_reg", "i32", "literal_int"),
        isa_mnemonic="S_LD_FP",
    ),
    "S_ST_FP": _OpcodeSpec(
        result_type="void",
        operand_kinds=("fp_reg", "i32", "literal_int"),
        isa_mnemonic="S_ST_FP",
    ),
    "S_ADD_FP": _OpcodeSpec(
        result_type="void",
        operand_kinds=("fp_reg", "fp_reg", "fp_reg"),
        isa_mnemonic="S_ADD_FP",
    ),
    "S_SUB_FP": _OpcodeSpec(
        result_type="void",
        operand_kinds=("fp_reg", "fp_reg", "fp_reg"),
        isa_mnemonic="S_SUB_FP",
    ),
    "S_MUL_FP": _OpcodeSpec(
        result_type="void",
        operand_kinds=("fp_reg", "fp_reg", "fp_reg"),
        isa_mnemonic="S_MUL_FP",
    ),
    "S_MAX_FP": _OpcodeSpec(
        result_type="void",
        operand_kinds=("fp_reg", "fp_reg", "fp_reg"),
        isa_mnemonic="S_MAX_FP",
    ),
    "S_EXP_FP": _OpcodeSpec(
        result_type="void",
        operand_kinds=("fp_reg", "fp_reg", "literal_int"),
        isa_mnemonic="S_EXP_FP",
    ),
    "S_RECI_FP": _OpcodeSpec(
        result_type="void",
        operand_kinds=("fp_reg", "fp_reg"),
        isa_mnemonic="S_RECI_FP",
    ),
    "S_SQRT_FP": _OpcodeSpec(
        result_type="void",
        operand_kinds=("fp_reg", "fp_reg"),
        isa_mnemonic="S_SQRT_FP",
    ),
    "S_MAP_FP_V": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "i32", "literal_int"),
        isa_mnemonic="S_MAP_FP_V",
    ),
    "S_MAP_V_FP": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "i32", "literal_int"),
        isa_mnemonic="S_MAP_V_FP",
    ),
    # ---- vector ----
    "V_ADD_VV": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "i32", "i32", "literal_int"),
        isa_mnemonic="V_ADD_VV",
    ),
    "V_SUB_VV": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "i32", "i32", "literal_int"),
        isa_mnemonic="V_SUB_VV",
    ),
    "V_MUL_VV": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "i32", "i32", "literal_int"),
        isa_mnemonic="V_MUL_VV",
    ),
    "V_ADD_VF": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "i32", "fp_reg", "literal_int"),
        isa_mnemonic="V_ADD_VF",
    ),
    "V_SUB_VF": _OpcodeSpec(
        # PLENA V_SUB_VF takes 5 operands: dst, src, fp_scalar,
        # mask_flag, reverse_flag (legacy quirk — always 0). The
        # extra trailing flag distinguishes it from V_ADD_VF /
        # V_MUL_VF which are 4-operand.
        result_type="void",
        operand_kinds=(
            "i32", "i32", "fp_reg", "literal_int", "literal_int",
        ),
        isa_mnemonic="V_SUB_VF",
    ),
    "V_MUL_VF": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "i32", "fp_reg", "literal_int"),
        isa_mnemonic="V_MUL_VF",
    ),
    "V_EXP_V": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "i32", "literal_int"),
        isa_mnemonic="V_EXP_V",
    ),
    "V_RECI_V": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "i32", "literal_int"),
        isa_mnemonic="V_RECI_V",
    ),
    "V_SQRT_V": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "i32", "literal_int"),
        isa_mnemonic="V_SQRT_V",
    ),
    "V_RED_MAX": _OpcodeSpec(
        result_type="void",
        operand_kinds=("fp_reg", "i32", "literal_int"),
        isa_mnemonic="V_RED_MAX",
    ),
    "V_RED_SUM": _OpcodeSpec(
        result_type="void",
        operand_kinds=("fp_reg", "i32", "literal_int"),
        isa_mnemonic="V_RED_SUM",
    ),
    # ---- matrix ----
    "M_BTMM": _OpcodeSpec(
        result_type="void",
        operand_kinds=("verbatim_str", "i32", "i32"),
        isa_mnemonic="M_BTMM",
    ),
    "M_BMM_WO": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "literal_int"),
        isa_mnemonic="M_BMM_WO",
    ),
    "M_BTMV": _OpcodeSpec(
        result_type="void",
        operand_kinds=("verbatim_str", "i32", "i32"),
        isa_mnemonic="M_BTMV",
    ),
    "M_BMV_WO": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "literal_int"),
        isa_mnemonic="M_BMV_WO",
    ),
    "M_MV": _OpcodeSpec(
        result_type="void",
        operand_kinds=("verbatim_str", "i32", "i32"),
        isa_mnemonic="M_MV",
    ),
    "M_MV_WO": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "literal_int"),
        isa_mnemonic="M_MV_WO",
    ),
    "M_MM": _OpcodeSpec(
        result_type="void",
        operand_kinds=("literal_int", "i32", "i32"),
        isa_mnemonic="M_MM",
    ),
    "M_MM_WO": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32", "verbatim_str", "literal_int"),
        isa_mnemonic="M_MM_WO",
    ),
    "M_TMM": _OpcodeSpec(
        result_type="void",
        operand_kinds=("literal_int", "i32", "i32"),
        isa_mnemonic="M_TMM",
    ),
    # ---- control / setup ----
    "C_SET_SCALE_REG": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32",),
        isa_mnemonic="C_SET_SCALE_REG",
    ),
    "C_SET_STRIDE_REG": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32",),
        isa_mnemonic="C_SET_STRIDE_REG",
    ),
    "C_SET_ADDR_REG": _OpcodeSpec(
        # Bind a PLENA addr-reg slot. HW builds the 64-bit address as
        # ``(gp[rs1] << 32) | gp[rs2]`` (see main.rs C_SET_ADDR_REG), so
        # the ISA needs TWO gp sources: high word then low word. Our
        # addresses are 32-bit, so the high word is the hardwired-zero
        # ``gp0`` (matches the legacy backend_emit form
        # ``C_SET_ADDR_REG aN, gp0, gp{addr}``).
        # operand 0: verbatim "gp0" — the constant-zero high word.
        # operand 1: the source i32 SSA value (the low/address word).
        # result   : an addr_reg SSA value (the bound aN).
        result_type="addr_reg",
        operand_kinds=("verbatim_str", "i32"),
        isa_mnemonic="C_SET_ADDR_REG",
    ),
    "C_SET_V_MASK_REG": _OpcodeSpec(
        result_type="void",
        operand_kinds=("i32",),
        isa_mnemonic="C_SET_V_MASK_REG",
    ),
    # ---- HBM ----
    "H_PREFETCH_V": _OpcodeSpec(
        result_type="void",
        operand_kinds=(
            "i32", "i32", "addr_reg",
            "literal_int", "literal_int",
        ),
        isa_mnemonic="H_PREFETCH_V",
    ),
    "H_PREFETCH_M": _OpcodeSpec(
        result_type="void",
        operand_kinds=(
            "i32", "i32", "addr_reg",
            "literal_int", "literal_int",
        ),
        isa_mnemonic="H_PREFETCH_M",
    ),
    "H_STORE_V": _OpcodeSpec(
        result_type="void",
        operand_kinds=(
            "i32", "i32", "addr_reg",
            "literal_int", "literal_int",
        ),
        isa_mnemonic="H_STORE_V",
    ),
    "H_LOAD_V": _OpcodeSpec(
        result_type="void",
        operand_kinds=(
            "i32", "i32", "addr_reg",
            "literal_int", "literal_int",
        ),
        isa_mnemonic="H_LOAD_V",
    ),
    # ---- pseudo / meta ----
    # ``_COMMENT`` does not emit a real instruction; backend prints it
    # as a ``; ...`` line. operands = (one verbatim_str).
    "_COMMENT": _OpcodeSpec(
        result_type="void",
        operand_kinds=("verbatim_str",),
        isa_mnemonic=";",
    ),
}


# ----------------------------------------------------------------------
# Core data structures
# ----------------------------------------------------------------------

class MirValue:
    """An SSA value.

    Mirrors MLIR-style SSA: each value has exactly one definition site.
    There are three legal definition sites:

      1. ``defined_by`` set to a :class:`MirInstr` — the standard case:
         a producer instruction computes this value.
      2. ``defined_by is None`` AND ``block_arg_of`` set to a
         :class:`MirBlock` — a block argument, supplied by the
         enclosing region (loop header) on each entry to that block.
         loop_var values are of this kind.
      3. ``defined_by is None`` AND ``is_function_const`` is True — a
         function-level constant value (today: ``gp0``, the
         hardware-fixed zero). Treated by passes as a "value that just
         exists" — no def site to schedule.

    ``dtype`` ∈ ``{"i32", "addr_reg", "fp_reg"}``. void instructions
    don't produce a MirValue at all (their ``MirInstr.result`` is
    None).
    """

    __slots__ = (
        "name", "dtype", "defined_by", "used_by",
        "block_arg_of", "is_function_const",
    )

    def __init__(self, name: str, dtype: str) -> None:
        self.name: str = name
        self.dtype: str = dtype
        self.defined_by: Optional["MirInstr"] = None
        self.used_by: List["MirInstr"] = []
        # Set when this value is the block argument of a particular
        # MirBlock. ``defined_by`` stays None in that case.
        self.block_arg_of: Optional["MirBlock"] = None
        # True for function-level constants (e.g. gp0). ``defined_by``
        # and ``block_arg_of`` both stay None.
        self.is_function_const: bool = False

    def __repr__(self) -> str:
        return f"%{self.name}:{self.dtype}"


# An operand can be:
#   * MirValue        — reference to an SSA value produced earlier
#   * int             — compile-time literal int (for literal_int kinds)
#   * tir.IntImm      — TVM integer immediate; treated as int by passes
#   * str             — verbatim token (e.g. "f0", "f1", "gp0" for the
#                        constant-zero source on instructions that hard-
#                        code it)
MirOperand = Union[MirValue, int, "tir.IntImm", str]


class MirInstr:
    """One MIR instruction.

    ``opcode`` is a key in ``OPCODES`` above. ``operands`` is a list of
    ``MirOperand``s whose kinds match ``OPCODES[opcode].operand_kinds``.
    ``result`` is a ``MirValue`` for non-void opcodes, else None.

    Use ``set_operand(i, val)`` to mutate operands so the def-use chain
    stays consistent (the old operand's ``used_by`` loses this instr,
    the new operand's gains it).
    """

    __slots__ = (
        "opcode", "operands", "result", "parent",
        "annotations",
    )

    def __init__(
        self,
        opcode: str,
        operands: List[MirOperand],
        result: Optional[MirValue] = None,
    ) -> None:
        if opcode not in OPCODES:
            raise ValueError(
                f"MirInstr: unknown opcode {opcode!r}. Add an entry to "
                f"mir.OPCODES."
            )
        self.opcode = opcode
        self.operands: List[MirOperand] = list(operands)
        self.result = result
        self.parent: Optional["MirBlock"] = None
        # Free-form per-pass scratch (debug source-PreIsaOp index,
        # optimisation hints, etc.).
        self.annotations: Dict[str, Any] = {}
        # Wire result.defined_by + each MirValue operand's used_by.
        if result is not None:
            if result.defined_by is not None:
                raise ValueError(
                    f"MirInstr: result {result!r} is already defined "
                    f"by another instruction {result.defined_by!r}; "
                    f"each SSA value must have exactly one def."
                )
            result.defined_by = self
        for op in self.operands:
            if isinstance(op, MirValue):
                op.used_by.append(self)

    def set_operand(self, i: int, new: MirOperand) -> None:
        """Replace the i-th operand, updating def-use chains."""
        old = self.operands[i]
        if isinstance(old, MirValue):
            try:
                old.used_by.remove(self)
            except ValueError:
                pass
        self.operands[i] = new
        if isinstance(new, MirValue):
            new.used_by.append(self)

    def replace_all_uses_of(
        self, old: MirValue, new: MirOperand,
    ) -> None:
        """Replace every occurrence of ``old`` in this instr's operands
        with ``new``. Updates def-use chains. Common subroutine for
        the same-named LLVM API used by CSE / value-replacement
        rewrites."""
        for i, op in enumerate(self.operands):
            if op is old:
                self.set_operand(i, new)

    def __repr__(self) -> str:
        op_strs = [_fmt_operand(o) for o in self.operands]
        if self.result is not None:
            return (
                f"%{self.result.name} = {self.opcode} "
                f"{', '.join(op_strs)}"
            )
        return f"{self.opcode} {', '.join(op_strs)}"


@dataclass
class MirBlock:
    """A basic block — a straight-line interleaved sequence of
    :class:`MirInstr`s and nested :class:`MirLoop` regions, prefixed
    by an optional list of block arguments (MLIR-style).

    Block arguments are SSA values that the enclosing region
    (function entry, or a loop header) supplies on each entry. A
    MirLoop's body block has exactly one argument — the loop_var
    SSA value — and that argument's ``block_arg_of`` points back at
    this block. From the body's perspective the argument is a
    normal SSA value with no in-block def: it just "is there".

    ``items`` preserves source order so loops can appear interleaved
    with instructions; the dump walks them in sequence.

    PLENA kernels are loop-nested DAGs only — we don't model
    arbitrary branching (no MirIf / phi); a block has no terminator
    instruction. Control flow is implicit in the loop-region
    nesting.
    """

    name: str
    items: List[Union["MirInstr", "MirLoop"]] = field(default_factory=list)
    # Block arguments — SSA values supplied by the enclosing region
    # at entry. For loop body blocks, this is ``[loop_var]``.
    arguments: List[MirValue] = field(default_factory=list)
    # Parent loop region (None = function-level top scope). Set when
    # this block is added to a MirLoop.body.
    parent_loop: Optional["MirLoop"] = None

    def append(self, item: Union["MirInstr", "MirLoop"]) -> Union["MirInstr", "MirLoop"]:
        if isinstance(item, MirInstr):
            item.parent = self
        elif isinstance(item, MirLoop):
            item.parent_block = self
        else:
            raise TypeError(
                f"MirBlock.append: expected MirInstr or MirLoop, "
                f"got {type(item).__name__}"
            )
        self.items.append(item)
        return item

    def add_argument(self, v: MirValue) -> MirValue:
        """Register ``v`` as a block argument of this block. The
        value's ``block_arg_of`` is set; its ``defined_by`` stays
        None (block arguments have no in-block def)."""
        if v.defined_by is not None:
            raise ValueError(
                f"add_argument: {v!r} is already defined by "
                f"{v.defined_by!r}; cannot also be a block argument"
            )
        if v.block_arg_of is not None and v.block_arg_of is not self:
            raise ValueError(
                f"add_argument: {v!r} is already an argument of "
                f"block {v.block_arg_of.name!r}"
            )
        v.block_arg_of = self
        if v not in self.arguments:
            self.arguments.append(v)
        return v

    # Back-compat helper for code that iterates only MirInstrs.
    @property
    def instrs(self) -> List["MirInstr"]:
        return [it for it in self.items if isinstance(it, MirInstr)]


@dataclass
class MirLoop:
    """A loop region.

    A loop's body is a list of MirBlocks (currently always exactly
    one in the lower passes, but the structure is here so a future
    "if-then" inside a loop body can land cleanly).

    ``loop_var`` is the SSA value supplied as the BLOCK ARGUMENT of
    the first body block — exactly like MLIR's ``scf.for`` induction
    variable. From the body's perspective ``loop_var`` is a normal
    SSA value with no in-block def; the region (this MirLoop) is the
    notional "definer", injecting a fresh value each iteration.

    Whether the lowering physically unrolls the body (binding
    loop_var to a literal int per iter) or emits a hardware loop
    (binding loop_var to an IntRAM-backed counter) is the
    ``loop_kind`` switch — a backend decision, not an IR-level
    distinction. Optimisation passes may REWRITE ``loop_kind`` to
    flip the strategy; the IR structure is unchanged.

    ``init`` / ``extent`` are compile-time ints (matching PLENA's
    ``C_LOOP_START`` immediate-only iteration count). Runtime-bounded
    loops would need a different lowering and are not modelled here.

    ``loop_kind`` ∈ ``{"serial", "unroll"}``.
    """

    name: str
    loop_var: MirValue
    init: int
    extent: int
    body: List[MirBlock] = field(default_factory=list)
    loop_kind: str = "serial"
    # Set by ``MirBlock.append`` when this loop is added to a block's
    # items list. None at top-of-function (the function's top block
    # is the parent in that case).
    parent_block: Optional[MirBlock] = None
    # Free-form scratch (loop_gp choice if hand-pinned, source
    # PreIsaOp idx, etc.).
    annotations: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Wire reverse parent pointer on each body block. Producers
        # build a MirLoop with ``body=[blk]``; we make sure
        # ``blk.parent_loop`` points back, so scope-walks (verify
        # dominance, last-use loop-stack, etc.) can climb out of a
        # body block to find its enclosing loop region. The same
        # invariant applies to any blocks appended later — see
        # ``add_body_block``.
        for blk in self.body:
            blk.parent_loop = self

    def add_body_block(self, blk: MirBlock) -> MirBlock:
        """Append a block to this loop's body, setting its
        ``parent_loop`` back-pointer."""
        blk.parent_loop = self
        self.body.append(blk)
        return blk


@dataclass
class MirFunction:
    """Top-level MIR container — one kernel.

    The function has a sequence of top-level :class:`MirBlock`s.
    Loops live inside blocks (via ``MirBlock.append(MirLoop(...))``);
    a loop's ``body`` is a list of nested blocks. Walking
    ``walk_loops()`` yields all loops in pre-order for passes that
    care.
    """

    name: str
    blocks: List[MirBlock] = field(default_factory=list)
    # The SSA name counter. Auto-incremented by ``mint_value``.
    _next_id: int = 0
    # Free-form metadata forwarded from HLIR / PreIsaIR (buffer table
    # for the dump, etc.).
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Function-level constant: hardware-fixed gp0 zero. Available to
    # passes as a "value that just exists" — its
    # ``is_function_const`` flag is True, ``defined_by`` is None.
    # Created by the converter via ``make_gp0_const``.
    gp0_value: Optional[MirValue] = None

    def make_gp0_const(self) -> MirValue:
        """Mint (or return existing) the function-level gp0 constant."""
        if self.gp0_value is not None:
            return self.gp0_value
        v = self.mint_value("i32", hint="gp0")
        v.is_function_const = True
        self.gp0_value = v
        return v

    def mint_value(self, dtype: str, hint: str = "") -> MirValue:
        """Create a fresh, undefined MirValue. Caller must wire it as
        the ``result`` of exactly one MirInstr (the constructor does
        the binding).

        ``hint`` is appended to the auto-generated name for readability
        (e.g. ``mint_value("i32", "lhs_addr")`` → ``%5_lhs_addr``).
        """
        nm = f"{self._next_id}"
        if hint:
            nm = f"{nm}_{hint}"
        self._next_id += 1
        return MirValue(nm, dtype)

    def walk_loops(self):
        """Yield every :class:`MirLoop` in the function in pre-order
        (outer loops before their nested children)."""
        def _recurse(blocks: List[MirBlock]):
            for blk in blocks:
                for item in blk.items:
                    if isinstance(item, MirLoop):
                        yield item
                        yield from _recurse(item.body)
        yield from _recurse(self.blocks)

    def walk_instrs(self):
        """Yield every :class:`MirInstr` in the function in source
        order, recursing into nested loop bodies."""
        def _recurse(blocks: List[MirBlock]):
            for blk in blocks:
                for item in blk.items:
                    if isinstance(item, MirInstr):
                        yield item
                    elif isinstance(item, MirLoop):
                        yield from _recurse(item.body)
        yield from _recurse(self.blocks)


# ----------------------------------------------------------------------
# Dump
# ----------------------------------------------------------------------

def _fmt_operand(op: MirOperand) -> str:
    if isinstance(op, MirValue):
        return f"%{op.name}"
    if isinstance(op, tir.IntImm):
        return str(int(op.value))
    if isinstance(op, int):
        return str(op)
    if isinstance(op, str):
        return op
    return repr(op)


def format_mir(fn: MirFunction) -> str:
    """Pretty-print one MirFunction. Used for ``<kernel>.mir.txt``."""
    lines = [f"MirFunction({fn.name!r}):"]
    if fn.metadata.get("buffers"):
        lines.append("  Buffers:")
        bufs = fn.metadata["buffers"]
        name_w = max((len(n) for n in bufs), default=4)
        for nm, b in bufs.items():
            scope = getattr(b, "scope", "?")
            shape = getattr(b, "shape", ())
            addr = getattr(b, "address", None)
            shape_s = "x".join(str(s) for s in shape) if shape else "()"
            addr_s = "?" if addr is None else str(addr)
            lines.append(
                f"    {nm:<{name_w}}  scope={scope:<5}  addr={addr_s}  "
                f"shape={shape_s}"
            )
    if fn.gp0_value is not None:
        lines.append(
            f"  Function constants:  %{fn.gp0_value.name}:i32 = "
            f"<gp0_const>"
        )
    lines.append("  Body:")
    for blk in fn.blocks:
        _format_block(blk, lines, indent=4)
    return "\n".join(lines) + "\n"


def _format_block_header(blk: MirBlock) -> str:
    """``^body(%2: i32, %3: i32):`` MLIR-style block header."""
    if blk.arguments:
        args = ", ".join(
            f"%{a.name}: {a.dtype}" for a in blk.arguments
        )
        return f"^{blk.name}({args}):"
    return f"^{blk.name}:"


def _format_block(blk: MirBlock, lines: List[str], indent: int) -> None:
    ind = " " * indent
    lines.append(f"{ind}{_format_block_header(blk)}")
    body_ind = " " * (indent + 2)
    for item in blk.items:
        if isinstance(item, MirInstr):
            lines.append(f"{body_ind}{_format_instr(item)}")
        else:
            _format_loop(item, lines, indent + 2)


def _format_loop(lp: MirLoop, lines: List[str], indent: int) -> None:
    ind = " " * indent
    lines.append(
        f"{ind}loop {lp.name} in [{lp.init}, {lp.init + lp.extent}) "
        f"[kind={lp.loop_kind}]"
    )
    for body_blk in lp.body:
        _format_block(body_blk, lines, indent + 2)


def _format_instr(instr: MirInstr) -> str:
    op_strs = [_fmt_operand(o) for o in instr.operands]
    body = f"{instr.opcode} {', '.join(op_strs)}".rstrip()
    if instr.result is not None:
        prefix = f"%{instr.result.name}:{instr.result.dtype} = "
        return prefix + body
    return body


# ----------------------------------------------------------------------
# Verifier
# ----------------------------------------------------------------------

class MirVerifyError(RuntimeError):
    pass


def verify(fn: MirFunction) -> None:
    """Sanity-check the MIR. Raises :class:`MirVerifyError` on the
    first problem found.

    Checks:
      * every instruction's opcode is in ``OPCODES``
      * operand counts + kinds match the opcode spec
      * non-void instructions have a non-None result of the right type
      * void instructions have ``result is None``
      * every MirValue has exactly one defining instruction
      * every operand-as-MirValue reference is reflected in the
        defining value's ``used_by``
      * loop_var SSA values are defined by ``_LOOP_VAR_DEF`` instrs
        sitting in the loop's body
    """
    # Collect every MirValue that is the result of some MirInstr,
    # and every MirValue that appears as an operand. The set of
    # operand-referenced MirValues that AREN'T defined anywhere in
    # the function is a hard error.
    defined: Set[int] = set()       # id() of MirValues we've seen as results
    declared_vals: List[MirValue] = []

    def _walk_instrs(instrs: List[MirInstr]) -> None:
        for instr in instrs:
            spec = OPCODES.get(instr.opcode)
            if spec is None:
                raise MirVerifyError(
                    f"unknown opcode {instr.opcode!r} on instr {instr!r}"
                )
            if len(instr.operands) != len(spec.operand_kinds):
                raise MirVerifyError(
                    f"{instr.opcode}: operand count "
                    f"{len(instr.operands)} != spec arity "
                    f"{len(spec.operand_kinds)} ({spec.operand_kinds})"
                )
            for i, (op, kind) in enumerate(
                zip(instr.operands, spec.operand_kinds),
            ):
                _check_operand_kind(instr, i, op, kind)
            if spec.result_type == "void":
                if instr.result is not None:
                    raise MirVerifyError(
                        f"{instr.opcode}: void opcode but result "
                        f"{instr.result!r} is not None"
                    )
            else:
                if instr.result is None:
                    raise MirVerifyError(
                        f"{instr.opcode}: non-void opcode but result "
                        f"is None"
                    )
                if instr.result.dtype != spec.result_type:
                    raise MirVerifyError(
                        f"{instr.opcode}: result dtype "
                        f"{instr.result.dtype!r} != spec result "
                        f"{spec.result_type!r}"
                    )
                if instr.result.defined_by is not instr:
                    raise MirVerifyError(
                        f"{instr.opcode}: result {instr.result!r} "
                        f"defined_by mismatch (claims "
                        f"{instr.result.defined_by!r}, real {instr!r})"
                    )
                if id(instr.result) in defined:
                    raise MirVerifyError(
                        f"{instr.opcode}: result {instr.result!r} is "
                        f"already defined by a previous instr "
                        f"(double-def)"
                    )
                defined.add(id(instr.result))
                declared_vals.append(instr.result)

    def _walk_block(blk: MirBlock) -> None:
        # Block arguments count as "defined" — they have no in-block
        # def site but the enclosing region supplies them.
        for arg in blk.arguments:
            if arg.defined_by is not None:
                raise MirVerifyError(
                    f"block {blk.name!r}: argument {arg!r} also has "
                    f"defined_by {arg.defined_by!r} — block arguments "
                    f"must have defined_by=None"
                )
            if arg.block_arg_of is not blk:
                raise MirVerifyError(
                    f"block {blk.name!r}: argument {arg!r} has "
                    f"block_arg_of={arg.block_arg_of!r}, not this block"
                )
            if id(arg) in defined:
                raise MirVerifyError(
                    f"block argument {arg!r} double-defined"
                )
            defined.add(id(arg))
            declared_vals.append(arg)
        for item in blk.items:
            if isinstance(item, MirInstr):
                _walk_instrs([item])
            elif isinstance(item, MirLoop):
                # loop_var is the body's first block argument; verify
                # the binding before recursing into the body.
                if not item.body:
                    raise MirVerifyError(
                        f"loop {item.name}: empty body"
                    )
                first_body = item.body[0]
                if not first_body.arguments or \
                   first_body.arguments[0] is not item.loop_var:
                    raise MirVerifyError(
                        f"loop {item.name}: loop_var must be the first "
                        f"argument of body block {first_body.name!r}; "
                        f"got arguments={first_body.arguments!r}"
                    )
                for body_blk in item.body:
                    _walk_block(body_blk)
            else:
                raise MirVerifyError(
                    f"block {blk.name}: unexpected item type "
                    f"{type(item).__name__}"
                )

    # Function-level constants count as defined too.
    if fn.gp0_value is not None:
        if not fn.gp0_value.is_function_const:
            raise MirVerifyError(
                f"function gp0 {fn.gp0_value!r} missing "
                f"is_function_const flag"
            )
        defined.add(id(fn.gp0_value))
        declared_vals.append(fn.gp0_value)

    for blk in fn.blocks:
        _walk_block(blk)

    # Cross-check: every MirValue used as an operand is defined,
    # used_by chains consistent, AND the def site is an ancestor
    # scope of the use site (SCF-style dominance — a value defined
    # inside a loop body cannot be referenced from outside the
    # body, since the loop has no yield/iter_args mechanism in
    # our MIR today).
    def _block_chain(blk: MirBlock) -> List[MirBlock]:
        """Ancestor chain from ``blk`` outward, crossing one MirLoop
        boundary per step."""
        chain = []
        cur = blk
        while cur is not None:
            chain.append(cur)
            if cur.parent_loop is None:
                break
            cur = cur.parent_loop.parent_block
        return chain

    def _def_block(v: MirValue) -> Optional[MirBlock]:
        if v.is_function_const:
            return None  # always in scope, no specific block
        if v.block_arg_of is not None:
            return v.block_arg_of
        if v.defined_by is not None:
            return v.defined_by.parent
        return None

    def _check_uses_block(blk: MirBlock) -> None:
        ancestors = _block_chain(blk)
        ancestor_set = {id(b) for b in ancestors}
        for item in blk.items:
            if isinstance(item, MirInstr):
                for i, op in enumerate(item.operands):
                    if isinstance(op, MirValue):
                        if id(op) not in defined:
                            raise MirVerifyError(
                                f"{item.opcode} operand[{i}] uses "
                                f"undefined SSA value {op!r}"
                            )
                        if item not in op.used_by:
                            raise MirVerifyError(
                                f"{item.opcode} operand[{i}]: SSA "
                                f"value {op!r}'s used_by chain doesn't "
                                f"include this instr"
                            )
                        if not op.is_function_const:
                            db = _def_block(op)
                            if db is not None and id(db) not in ancestor_set:
                                raise MirVerifyError(
                                    f"{item.opcode} operand[{i}]: SSA "
                                    f"dominance violation. Value "
                                    f"{op!r} is defined in block "
                                    f"{db.name!r}, which is NOT an "
                                    f"ancestor of the use site block "
                                    f"{blk.name!r}. Cross-scope use "
                                    f"is illegal: a value defined "
                                    f"inside a loop body can only be "
                                    f"referenced within that body."
                                )
            elif isinstance(item, MirLoop):
                for body_blk in item.body:
                    _check_uses_block(body_blk)

    for blk in fn.blocks:
        _check_uses_block(blk)

    # SCF-style scope check: every operand's defining-site scope must
    # be an ANCESTOR of the using instruction's scope. A scope is the
    # block (or its enclosing loop-region chain) containing a value's
    # def. Concretely:
    #   * a function-level constant (gp0) is in scope everywhere
    #   * a value defined by an instr in block B is in scope inside
    #     B and inside any block nested under B (via MirLoop body)
    #   * a block argument of block B is in scope inside B and below
    #
    # Equivalently: walking outward from the use site through
    # parent_loop / parent_block pointers must eventually reach the
    # block where the value is defined / is an argument of.
    def _block_chain(blk: MirBlock) -> List[MirBlock]:
        """List of blocks from ``blk`` outward to the function root.
        Each step crosses one MirLoop boundary (the block's
        parent_loop's parent_block is the next outer block)."""
        chain = []
        cur = blk
        while cur is not None:
            chain.append(cur)
            if cur.parent_loop is None:
                break
            cur = cur.parent_loop.parent_block
        return chain

    def _def_block(v: MirValue) -> Optional[MirBlock]:
        if v.is_function_const:
            return None  # always in scope
        if v.block_arg_of is not None:
            return v.block_arg_of
        if v.defined_by is not None:
            return v.defined_by.parent
        return None

    def _check_scope_block(blk: MirBlock) -> None:
        for item in blk.items:
            if isinstance(item, MirInstr):
                ancestors = _block_chain(blk)
                ancestor_set = {id(b) for b in ancestors}
                for i, op in enumerate(item.operands):
                    if not isinstance(op, MirValue):
                        continue
                    if op.is_function_const:
                        continue
                    db = _def_block(op)
                    if db is None:
                        continue  # already handled
                    if id(db) not in ancestor_set:
                        raise MirVerifyError(
                            f"{item.opcode} operand[{i}]: SSA value "
                            f"{op!r} is defined in block "
                            f"{db.name!r} which is NOT an ancestor "
                            f"of the use site block {blk.name!r}. "
                            f"Cross-scope use violates SSA/SCF "
                            f"dominance. Producer should yield this "
                            f"value via a (TODO) loop-result mechanism "
                            f"or hoist the def to a common ancestor."
                        )
            elif isinstance(item, MirLoop):
                for body_blk in item.body:
                    _check_scope_block(body_blk)

    for blk in fn.blocks:
        _check_scope_block(blk)


def _check_operand_kind(
    instr: MirInstr, i: int, op: MirOperand, kind: str,
) -> None:
    if kind == "i32":
        if isinstance(op, MirValue):
            if op.dtype != "i32":
                raise MirVerifyError(
                    f"{instr.opcode} operand[{i}]: expected i32 SSA "
                    f"value, got {op!r} (dtype={op.dtype!r})"
                )
            return
        if isinstance(op, (int, tir.IntImm)):
            return
        raise MirVerifyError(
            f"{instr.opcode} operand[{i}]: expected i32 SSA value or "
            f"int literal; got {type(op).__name__} {op!r}"
        )
    if kind == "addr_reg":
        if not isinstance(op, MirValue) or op.dtype != "addr_reg":
            raise MirVerifyError(
                f"{instr.opcode} operand[{i}]: expected addr_reg SSA "
                f"value; got {op!r}"
            )
        return
    if kind == "fp_reg":
        if not isinstance(op, str):
            raise MirVerifyError(
                f"{instr.opcode} operand[{i}]: expected fp_reg verbatim "
                f"string (e.g. 'f1'); got {type(op).__name__} {op!r}"
            )
        return
    if kind == "literal_int":
        if not isinstance(op, (int, tir.IntImm)):
            raise MirVerifyError(
                f"{instr.opcode} operand[{i}]: expected literal_int; "
                f"got {type(op).__name__} {op!r}"
            )
        return
    if kind == "verbatim_str":
        if not isinstance(op, str):
            raise MirVerifyError(
                f"{instr.opcode} operand[{i}]: expected verbatim_str; "
                f"got {type(op).__name__} {op!r}"
            )
        return
    raise MirVerifyError(
        f"{instr.opcode} operand[{i}]: unknown operand kind {kind!r}"
    )


__all__ = [
    "MirValue", "MirInstr", "MirBlock", "MirLoop", "MirFunction",
    "MirOperand", "MirVerifyError",
    "OPCODES", "_OpcodeSpec",
    "format_mir", "verify",
]
