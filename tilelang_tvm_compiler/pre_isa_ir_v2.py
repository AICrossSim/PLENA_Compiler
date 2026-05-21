"""PreIsaIR v2 — the clean rewrite.

The original :mod:`pre_isa_ir` accumulated PreIsaOps for *register
materialisation control* (``_PRELOAD_ADDR``, ``_BUMP_CACHED_GP``,
``_PRELOAD_ADDR_REG``, ``_slot_expr_cached`` family, ``group_id``,
``unroll_scope``, ``scope_floor``, etc.). Those concerns were forced
into this layer because the original "BackendEmit" did
register-allocation and ISA emission in one mixed step.

The new architecture splits responsibilities:

  PreIsaIR (this file)  — what HW instruction, with what symbolic
                          operand. PrimExpr operands stay in their
                          most abstract form (full address algebra).
                          NO register materialisation concept.
  MIR (mir.py)         — explicit conversion of PrimExpr → SSA value
                          chains, def/use, loops with loop_kind, ready
                          for LICM / CSE / DCE / register allocation.
  Backend (mir_to_isa) — mechanical SSA → physical-register dispatch.

What's IN this PreIsaIR:
  * PreIsaOp(opcode, operands) where:
      - opcode is a literal PLENA ISA mnemonic ("M_MM", "H_PREFETCH_V",
        "S_ADDI_INT", "S_LD_FP", ...). NO ``_*`` prefixed variants.
      - operands is a list of:
          ``tir.PrimExpr``  — any address algebra; PreIsaIR doesn't
                              fold or evaluate; the next pass does.
          ``int``           — compile-time literal immediate.
          ``str``           — verbatim token like "f0" / "f1" / "f2"
                              (FPU register names) or "gp0" (the
                              hardware-fixed constant-zero source on
                              instructions where it's encoded as part
                              of the ISA, e.g. ``S_ADDI_INT _, gp0, _``
                              — and only those cases).
  * LoopRegion(start, end, loop_var, init_imm, extent_imm, loop_kind)
    where ``loop_kind`` is ``"serial"`` (HW C_LOOP) or ``"unroll"``
    (compile-time unrolled). The loop_var is a ``tir.Var`` that body
    PreIsaOps reference in their operand PrimExprs.
  * Comment lines (``_COMMENT`` opcode) for the human-readable dump.

What's NOT in this PreIsaIR:
  * NO ``_PRELOAD_ADDR`` / ``_PRELOAD_ADDR_REG`` / ``_BUMP_CACHED_GP``
  * NO ``group_id`` / ``close_order`` / ``unroll_scope`` annotations
  * NO ``_slot_*_cached`` / ``_*_ROW`` opcode variants
  * NO ``addr_reg`` / ``gp_reg`` numbers — those don't exist yet

Producer contract:
  For an HLIR op, the PreIsaPass producer emits a sequence of
  PreIsaOps + LoopRegions that, taken together, semantically realise
  the legacy ISA emission BUT with addresses left as PrimExprs.
  Whether one M_MM ends up with a fresh ``S_ADDI_INT`` ahead of it,
  or shares a hoisted register with another M_MM — none of that is
  the producer's concern. PreIsaIR → MIR conversion + MIR LICM
  decide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from tvm import tir


# ---------------------------------------------------------------------
# Loop-strategy switch
# ---------------------------------------------------------------------
# When True, every LoopRegion built with ``loop_kind="unroll"`` is
# downgraded to ``"serial"`` (a hardware C_LOOP) at construction time.
# Emit-time unrolling was unsound: it cloned each iter's body into a
# scratch block (minting MirValues the precomputed last_use table never
# saw) and the linear-scan allocator then read released / never-tracked
# operands. Until a real MIR-level unroll pass exists, run everything as
# hardware loops. Flip to False to restore the (broken) emit-time
# unroll path for A/B debugging.
FORCE_SERIAL_LOOPS = True


# ---------------------------------------------------------------------
# Opcode set
# ---------------------------------------------------------------------

# PreIsaIR opcodes are PLENA ISA mnemonics in their *atomic* form —
# one PreIsaOp per single PLENA instruction. No internal variants.
# The PreIsaIR → MIR pass turns each into a MIR instruction with
# operands lowered to SSA values; mir.OPCODES is the source of truth
# for what arguments each PLENA instruction takes.
#
# This set must stay a SUBSET of mir.OPCODES (every PreIsaIR opcode
# must have a corresponding MIR opcode). We don't import mir.OPCODES
# here to avoid a circular import; the conversion pass cross-checks.
KNOWN_OPCODES = frozenset({
    # control
    "C_SET_V_MASK_REG", "C_SET_ADDR_REG",
    "C_SET_SCALE_REG", "C_SET_STRIDE_REG",
    # scalar int
    "S_ADDI_INT", "S_ADD_INT", "S_SUB_INT", "S_MUL_INT",
    "S_LUI_INT", "S_LD_INT", "S_ST_INT",
    "S_SLLI_INT", "S_SRLI_INT",
    # scalar fp
    "S_LD_FP", "S_ST_FP",
    "S_ADD_FP", "S_SUB_FP", "S_MUL_FP", "S_MAX_FP",
    "S_EXP_FP", "S_RECI_FP", "S_SQRT_FP",
    "S_MAP_FP_V", "S_MAP_V_FP",
    # vector
    "V_ADD_VV", "V_SUB_VV", "V_MUL_VV",
    "V_ADD_VF", "V_SUB_VF", "V_MUL_VF",
    "V_EXP_V", "V_RECI_V", "V_SQRT_V",
    "V_RED_MAX", "V_RED_SUM",
    # matrix
    "M_BTMM", "M_BMM_WO",
    "M_BTMV", "M_BMV_WO",
    "M_MV", "M_MV_WO",
    "M_MM", "M_MM_WO", "M_TMM",
    # HBM
    "H_LOAD_V", "H_STORE_V", "H_PREFETCH_V", "H_PREFETCH_M",
    # meta — emitted as ``; ...`` comment, not a real instruction.
    "_COMMENT",
})


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

# Operand of a PreIsaOp.
#   * tir.PrimExpr — address algebra / value expression; loop_vars
#       stay as tir.Var, hw constants as MLEN_VAR / BLEN_VAR
#       (defined in hw_consts), IntImm for literals already known.
#   * int          — compile-time integer immediate (flag bits, loop
#       counts on legacy callsites we haven't migrated yet).
#   * str          — verbatim token. The ONLY allowed verbatim
#       strings are PLENA FPU register names ("f0", "f1", "f2") and
#       the hardware-encoded constant-zero source "gp0" — these are
#       part of the ISA encoding, not register-allocation decisions.
#   * PreIsaOp     — reference to the result of a previously emitted
#       op. Used for ``addr_reg`` operands (the only currently-supported
#       result-producing opcode is ``C_SET_ADDR_REG``). The MIR
#       converter substitutes the producer's MirValue at lowering
#       time. Producer must emit the referenced op BEFORE the
#       consumer in the same module body.
PreIsaOperand = Union[tir.PrimExpr, int, str, "PreIsaOp"]


@dataclass
class PreIsaOp:
    """One PLENA ISA instruction with symbolic operands.

    ``opcode`` is a PLENA mnemonic from :data:`KNOWN_OPCODES`. NO
    ``_*``-prefixed variants — those were a leak of register-
    materialisation concerns from PreIsaIR v1; in v2 the conversion
    pass handles those concerns instead.
    """

    opcode: str
    operands: List[PreIsaOperand] = field(default_factory=list)
    # Free-form debug annotations (source HLIR op index, intrinsic
    # name, etc.). No semantic meaning to any pass — feel free to
    # add fields without coordinating.
    annotations: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.opcode not in KNOWN_OPCODES:
            raise ValueError(
                f"PreIsaOp opcode {self.opcode!r} is not a known PLENA "
                f"mnemonic in pre_isa_ir_v2. If this is a real HW "
                f"instruction, add it to KNOWN_OPCODES (and ensure "
                f"mir.OPCODES has a matching entry)."
            )


@dataclass
class LoopRegion:
    """A loop block in the PreIsaIR program structure.

    ``loop_var`` is a ``tir.Var``. Body PreIsaOps reference it via
    PrimExpr operands; the PreIsaIR → MIR conversion will lower it
    to a MirValue defined by a ``_LOOP_VAR_DEF`` MirInstr at the top
    of the loop body block.

    ``loop_kind`` is ``"serial"`` (HW C_LOOP_START / C_LOOP_END) or
    ``"unroll"`` (compile-time body replay with loop_var bound to
    IntImm per iteration). Optimisation passes may rewrite this
    attribute — it's a strategy hint, not a structural property.

    ``body`` is a flat sequence of PreIsaOps + LoopRegions (nested
    loops). Order is source order; the conversion pass walks it
    verbatim.

    Producer contract for ``loop_var`` choice: every LoopRegion's
    loop_var must be a FRESH ``tir.Var`` instance (don't reuse a Var
    across LoopRegions; the conversion pass relies on identity).
    """

    loop_var: tir.Var
    init_imm: int
    extent_imm: int
    body: List[Union["PreIsaOp", "LoopRegion"]] = field(default_factory=list)
    loop_kind: str = "serial"   # "serial" or "unroll"
    annotations: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.loop_kind not in ("serial", "unroll"):
            raise ValueError(
                f"LoopRegion.loop_kind must be 'serial' or 'unroll'; "
                f"got {self.loop_kind!r}"
            )
        # Global force-serial: emit-time unrolling (clone body into a
        # scratch block, fold loop_var→const) created emit-time MirValues
        # the single-walk last_use table never saw, corrupting register
        # allocation. Until a proper MIR-level unroll pass exists, every
        # LoopRegion lowers to a hardware C_LOOP. The address arithmetic
        # (base + loop_var*stride) that unroll used to const-fold is
        # computed at runtime instead — mathematically equivalent.
        # Python-side static unrolls (plain ``for`` in the generator,
        # e.g. matmul's partial-tile n_mlen sweep) are NOT LoopRegions
        # and are unaffected.
        if FORCE_SERIAL_LOOPS and self.loop_kind == "unroll":
            self.loop_kind = "serial"


@dataclass
class PreIsaModule:
    """One kernel's PreIsaIR — a flat sequence of PreIsaOps and
    LoopRegions at the top level. Loops can nest arbitrarily.
    """

    name: str
    body: List[Union[PreIsaOp, LoopRegion]] = field(default_factory=list)
    # Buffer table forwarded from HLIR. Used by the dump + the
    # MIR conversion pass for buffer-address resolution.
    buffers: Dict[str, Any] = field(default_factory=dict)

    def append(self, item: Union[PreIsaOp, LoopRegion]) -> None:
        self.body.append(item)

    def comment(self, text: str) -> None:
        self.body.append(PreIsaOp(opcode="_COMMENT", operands=[text]))


# ---------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------

def _fmt_operand(op: PreIsaOperand) -> str:
    if isinstance(op, str):
        return op
    if isinstance(op, int):
        return str(op)
    if isinstance(op, tir.IntImm):
        return str(int(op.value))
    if isinstance(op, tir.Var):
        return op.name
    # Generic PrimExpr — str() preserves loop var names.
    return str(op)


def format_pre_isa_v2(mod: PreIsaModule) -> str:
    """Pretty-print PreIsaIR v2 — used for ``<kernel>.pre_isa.txt``."""
    lines = [f"PreIsaModule(name={mod.name!r}):"]
    if mod.buffers:
        lines.append("  Buffers:")
        name_w = max((len(n) for n in mod.buffers), default=4)
        for nm, b in mod.buffers.items():
            scope = getattr(b, "scope", "?")
            shape = getattr(b, "shape", ())
            addr = getattr(b, "address", None)
            shape_s = "x".join(str(s) for s in shape) if shape else "()"
            addr_s = "?" if addr is None else str(addr)
            lines.append(
                f"    {nm:<{name_w}}  scope={scope:<5}  addr={addr_s}  "
                f"shape={shape_s}"
            )
    lines.append("  Body:")
    for item in mod.body:
        _fmt_item(item, lines, indent=4)
    return "\n".join(lines) + "\n"


def _fmt_item(item, lines, indent):
    ind = " " * indent
    if isinstance(item, PreIsaOp):
        if item.opcode == "_COMMENT":
            text = item.operands[0] if item.operands else ""
            lines.append(f"{ind}; {text}")
            return
        ops_s = ", ".join(_fmt_operand(o) for o in item.operands)
        lines.append(f"{ind}{item.opcode:<18} {ops_s}")
        return
    if isinstance(item, LoopRegion):
        lines.append(
            f"{ind}loop {item.loop_var.name} in "
            f"[{item.init_imm}, {item.init_imm + item.extent_imm}) "
            f"[kind={item.loop_kind}]"
        )
        for inner in item.body:
            _fmt_item(inner, lines, indent + 2)
        return
    raise TypeError(
        f"_fmt_item: expected PreIsaOp or LoopRegion, got "
        f"{type(item).__name__}"
    )


# ---------------------------------------------------------------------
# Sanity / structural helpers
# ---------------------------------------------------------------------

def loop_regions(
    body: List[Union[PreIsaOp, LoopRegion]],
) -> List[Tuple[LoopRegion, int]]:
    """Return ``(loop, depth)`` for every LoopRegion in pre-order."""
    out: List[Tuple[LoopRegion, int]] = []

    def _walk(items, depth):
        for it in items:
            if isinstance(it, LoopRegion):
                out.append((it, depth))
                _walk(it.body, depth + 1)
    _walk(body, 0)
    return out


__all__ = [
    "PreIsaOp", "LoopRegion", "PreIsaModule",
    "KNOWN_OPCODES", "PreIsaOperand",
    "format_pre_isa_v2", "loop_regions",
]
