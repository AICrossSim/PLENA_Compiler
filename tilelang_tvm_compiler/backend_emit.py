"""BackendEmit — consume PreIsaIR, produce final ISA text.

This is the second half of the IsaPass split:

    Old:  HLIR -> IsaEmitterPass (algebra + materialise + emit) -> ISA

    New:  HLIR -> PreIsaPass (algebra; produce PreIsaIR)
                 -> pre_isa_optimize (arith.simplify; CSE; LICM)
                 -> BackendEmit (materialise each operand; emit ISA)

BackendEmit owns the GP register / ISA-text wiring. For each PreIsaOp
in the input stream:
  * Materialise every ``tir.PrimExpr`` operand via the existing
    ``ExprMaterializer`` (which handles symbol_table lookup, GP alloc,
    eager auto-spill, constant folding, the lot).
  * Plug the resulting GP register numbers into a per-opcode ISA
    template string and append one ISA line to
    ``shim.compiler.generated_code``.
  * Release the operand GPs after the emit.

The opcode dispatch table lives in ``_TEMPLATES`` below. Each entry
declares the ISA mnemonic's operand layout — which operand slots are
materialised (PrimExpr -> GP) vs dropped in verbatim (already a
literal int / a hard-coded ``f0`` / ``a3`` token / a flag).
Adding a new HW instruction to the BackendEmit means adding one row
to ``_TEMPLATES`` and (if needed) the mnemonic to ``KNOWN_OPCODES``
in ``pre_isa_ir.py``.

Loop handling: ``LOOP_START`` / ``LOOP_END`` are the PreIsaIR
control markers. The PreIsaOp on ``LOOP_START`` carries
``annotations["loop_kind"]`` which selects the codegen strategy:

  * ``"serial"`` (default) — emit a hardware loop. Claims an IntRAM
    idx slot, binds the loop_var into ``symbol_table`` as
    ``("ram", idx_addr)``, and emits the counter init + the literal
    ISA mnemonic ``C_LOOP_START gp_loop, extent``. The matching
    ``LOOP_END`` emits the idx-increment + ``C_LOOP_END gp_loop``
    and releases the slot. Byte-equal to legacy ``_emit_for``'s
    serial branch.

  * ``"unroll"`` — bind loop_var to a per-iteration ``tir.IntImm``
    and recursively emit the body PreIsaOps N times. No idx slot,
    no loop_gp use, no hardware C_LOOP_* lines. Byte-equal to legacy
    ``_emit_for``'s unrolled branch.

Switching the strategy is a single annotation edit at any PreIsaIR
optimisation pass — the IR shape doesn't change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from tvm import tir

from .expr_materializer import ExprMaterializer, MaterializedExpr
from .pre_isa_ir import PreIsaModule, PreIsaOp
from .program_shim import ProgramShim


class BackendEmitError(RuntimeError):
    pass


# Sentinel used by ``_invoke_slot`` to signal that a slot's GP register
# came from the group cache, NOT from a fresh materialise(). The
# caller's per-emit cleanup must skip these (they're released at
# group close, not per emit).
_CACHED_SENTINEL = object()


# Operand-slot descriptor: a Python callable that, given the slot
# value (PrimExpr / int / str) and the materialiser, returns the
# token to drop into the ISA template ("gpN" / literal / "f0" / ...)
# AND an optional ``MaterializedExpr`` to release after the emit.
def _slot_expr(
    val: Any, mat: ExprMaterializer,
) -> Tuple[str, Optional[MaterializedExpr]]:
    """A ``PrimExpr`` operand → materialise to gpN, return (``"gpN"``,
    handle). A plain ``int`` also goes through materialise so loop-vars
    and large literals get the proper S_ADDI/S_LUI sequence the
    existing emitter expects."""
    if isinstance(val, (int, tir.PrimExpr)):
        m = mat.materialize(val)
        return f"gp{m.register}", m
    raise BackendEmitError(
        f"slot_expr: expected PrimExpr / int, got {type(val).__name__} {val!r}"
    )


def _slot_literal_int(
    val: Any, mat: ExprMaterializer,
) -> Tuple[str, Optional[MaterializedExpr]]:
    """An immediate field of the ISA encoding — must be a compile-time
    int literal, dropped verbatim into the template."""
    if isinstance(val, int):
        return str(val), None
    if isinstance(val, tir.IntImm):
        return str(int(val.value)), None
    raise BackendEmitError(
        f"slot_literal_int: ISA literal must be a compile-time int; "
        f"got {type(val).__name__} {val!r}"
    )


def _slot_verbatim(
    val: Any, mat: ExprMaterializer,
) -> Tuple[str, Optional[MaterializedExpr]]:
    """A hard-coded token (e.g. ``"f0"`` for the zero FPRAM register,
    ``"a3"`` for an addr-reg slot). Caller already wrote the string
    they want in the ISA; we just drop it in."""
    if isinstance(val, str):
        return val, None
    raise BackendEmitError(
        f"slot_verbatim: expected str token; got {type(val).__name__} {val!r}"
    )


# Marker slot kind: the BackendEmit handler treats it specially in
# _invoke_slot. Looks up the operand PrimExpr in the current group
# cache and returns the cached GP token; never materialises fresh.
# Used by row_*_at's destructive in-place stride bump pattern: the
# V_*_VF / V_RED_* / V_EXP_V iterations all read the same GP that an
# earlier _PRELOAD_ADDR established, and _BUMP_CACHED_GP mutates that
# GP between iterations.
def _slot_expr_cached(
    val: Any, mat: ExprMaterializer,
) -> Tuple[str, Optional[MaterializedExpr]]:
    raise BackendEmitError(
        "_slot_expr_cached must be intercepted by BackendEmit._invoke_slot"
    )


# Marker slot kind: the BackendEmit handler looks up the operand
# PrimExpr in the ADDR-REG cache (populated by ``_PRELOAD_ADDR_REG``)
# and returns the cached ``aN`` token. Used by DMA HW instructions
# (H_PREFETCH_V / H_STORE_V / H_PREFETCH_M).
def _slot_addr_reg_cached(
    val: Any, mat: ExprMaterializer,
) -> Tuple[str, Optional[MaterializedExpr]]:
    raise BackendEmitError(
        "_slot_addr_reg_cached must be intercepted by BackendEmit._invoke_slot"
    )


@dataclass
class _Template:
    """One opcode's emit template.

    ``slots`` is a list of (slot-kind-callable) entries, one per
    operand on the PreIsaOp. ``fmt`` is the ISA-line format string
    with positional placeholders ``{0}`` ... ``{N-1}`` corresponding
    to the slots' rendered tokens.
    """
    slots: List[Callable[[Any, ExprMaterializer], Tuple[str, Optional[MaterializedExpr]]]]
    fmt: str


# Per-opcode dispatch table. **Only opcodes whose handler has been
# migrated to the PreIsaPass producer appear here.** As more handlers
# migrate (one PR-or-commit at a time, byte-equal verified per op),
# new rows land in this table.
_TEMPLATES: Dict[str, _Template] = {
    # FP scalar load/store: ``S_LD_FP fX, gp{addr}, 0`` /
    # ``S_ST_FP fX, gp{addr}, 0``. First operand is the FP register
    # token (verbatim), second is the address (PrimExpr), third is the
    # element offset literal (always 0 in current emit, kept as
    # operand for future flexibility).
    "S_LD_FP": _Template(
        slots=[_slot_verbatim, _slot_expr, _slot_literal_int],
        fmt="S_LD_FP {0}, {1}, {2}",
    ),
    "S_ST_FP": _Template(
        slots=[_slot_verbatim, _slot_expr, _slot_literal_int],
        fmt="S_ST_FP {0}, {1}, {2}",
    ),
    # FP scalar binary ops: ``OP f_dst, f_lhs, f_rhs`` — all three
    # operands are FP register tokens.
    "S_ADD_FP": _Template(
        slots=[_slot_verbatim, _slot_verbatim, _slot_verbatim],
        fmt="S_ADD_FP {0}, {1}, {2}",
    ),
    "S_SUB_FP": _Template(
        slots=[_slot_verbatim, _slot_verbatim, _slot_verbatim],
        fmt="S_SUB_FP {0}, {1}, {2}",
    ),
    "S_MUL_FP": _Template(
        slots=[_slot_verbatim, _slot_verbatim, _slot_verbatim],
        fmt="S_MUL_FP {0}, {1}, {2}",
    ),
    "S_MAX_FP": _Template(
        slots=[_slot_verbatim, _slot_verbatim, _slot_verbatim],
        fmt="S_MAX_FP {0}, {1}, {2}",
    ),
    # FP scalar unary ops. Note legacy emitter is inconsistent: S_EXP_FP
    # takes a trailing ``0`` flag operand, S_RECI_FP / S_SQRT_FP do not.
    "S_EXP_FP": _Template(
        slots=[_slot_verbatim, _slot_verbatim, _slot_literal_int],
        fmt="S_EXP_FP {0}, {1}, {2}",
    ),
    "S_RECI_FP": _Template(
        slots=[_slot_verbatim, _slot_verbatim],
        fmt="S_RECI_FP {0}, {1}",
    ),
    "S_SQRT_FP": _Template(
        slots=[_slot_verbatim, _slot_verbatim],
        fmt="S_SQRT_FP {0}, {1}",
    ),
    # Vector ops. ``V_*_VV`` / ``V_*_VF`` all take a trailing literal
    # flag (almost always 0 in current emit). The "VV" variant takes
    # gp-gp-gp (dst, lhs, rhs); the "VF" variant takes gp-gp-fpram_reg.
    "V_ADD_VV": _Template(
        slots=[_slot_expr, _slot_expr, _slot_expr, _slot_literal_int],
        fmt="V_ADD_VV {0}, {1}, {2}, {3}",
    ),
    "V_SUB_VV": _Template(
        slots=[_slot_expr, _slot_expr, _slot_expr, _slot_literal_int],
        fmt="V_SUB_VV {0}, {1}, {2}, {3}",
    ),
    "V_MUL_VV": _Template(
        slots=[_slot_expr, _slot_expr, _slot_expr, _slot_literal_int],
        fmt="V_MUL_VV {0}, {1}, {2}, {3}",
    ),
    "V_MUL_VF": _Template(
        slots=[_slot_expr, _slot_expr, _slot_verbatim, _slot_literal_int],
        fmt="V_MUL_VF {0}, {1}, {2}, {3}",
    ),
    "V_ADD_VF": _Template(
        slots=[_slot_expr, _slot_expr, _slot_verbatim, _slot_literal_int],
        fmt="V_ADD_VF {0}, {1}, {2}, {3}",
    ),
    "V_SUB_VF": _Template(
        slots=[_slot_expr, _slot_expr, _slot_verbatim, _slot_literal_int],
        fmt="V_SUB_VF {0}, {1}, {2}, {3}",
    ),
    "V_EXP_V": _Template(
        slots=[_slot_expr, _slot_expr, _slot_literal_int],
        fmt="V_EXP_V {0}, {1}, {2}",
    ),
    "V_RECI_V": _Template(
        slots=[_slot_expr, _slot_expr, _slot_literal_int],
        fmt="V_RECI_V {0}, {1}, {2}",
    ),
    "V_SQRT_V": _Template(
        slots=[_slot_expr, _slot_expr, _slot_literal_int],
        fmt="V_SQRT_V {0}, {1}, {2}",
    ),
    # VRAM <-> FPRAM transfer (S_MAP_*_*). Both directions take
    # (gp_dst_addr, gp_src_addr, 0) — the gp values are FPRAM /
    # VRAM addresses, so both slots are PrimExpr.
    "S_MAP_FP_V": _Template(
        slots=[_slot_expr, _slot_expr, _slot_literal_int],
        fmt="S_MAP_FP_V {0}, {1}, {2}",
    ),
    "S_MAP_V_FP": _Template(
        slots=[_slot_expr, _slot_expr, _slot_literal_int],
        fmt="S_MAP_V_FP {0}, {1}, {2}",
    ),
    # Mask register control. The operand is a GP holding the new mask
    # value. legacy emits this twice per masked row op: once before to
    # arm, once after to reset to 0.
    "C_SET_V_MASK_REG": _Template(
        slots=[_slot_expr_cached],
        fmt="C_SET_V_MASK_REG {0}",
    ),
    # Vector reduce — accumulates into f1. Legacy ``V_RED_*`` takes
    # (f_acc, gp_src_vec, mask_flag).
    "V_RED_MAX": _Template(
        slots=[_slot_verbatim, _slot_expr_cached, _slot_literal_int],
        fmt="V_RED_MAX {0}, {1}, {2}",
    ),
    "V_RED_SUM": _Template(
        slots=[_slot_verbatim, _slot_expr_cached, _slot_literal_int],
        fmt="V_RED_SUM {0}, {1}, {2}",
    ),
    # S_SUB_VF for row_sub_fp: legacy emits a 5-operand form
    # ``V_SUB_VF gp{dst}, gp{src}, f1, mask_flag, 0``. The trailing 0
    # is a literal that doesn't appear in V_ADD_VF / V_MUL_VF (those
    # are 4-operand).
    "_V_SUB_VF_ROW": _Template(
        slots=[_slot_expr_cached, _slot_expr_cached, _slot_verbatim,
               _slot_literal_int, _slot_literal_int],
        fmt="V_SUB_VF {0}, {1}, {2}, {3}, {4}",
    ),
    # Same as _V_SUB_VF_ROW but for V_ADD_VF / V_MUL_VF with the cached
    # GP operand pattern (used by row_*_at's d_tile unroll loop).
    "_V_ADD_VF_ROW": _Template(
        slots=[_slot_expr_cached, _slot_expr_cached, _slot_verbatim,
               _slot_literal_int],
        fmt="V_ADD_VF {0}, {1}, {2}, {3}",
    ),
    "_V_MUL_VF_ROW": _Template(
        slots=[_slot_expr_cached, _slot_expr_cached, _slot_verbatim,
               _slot_literal_int],
        fmt="V_MUL_VF {0}, {1}, {2}, {3}",
    ),
    # V_EXP_V / V_RECI_V with cached GPs (no fresh materialise).
    "_V_EXP_V_ROW": _Template(
        slots=[_slot_expr_cached, _slot_expr_cached, _slot_literal_int],
        fmt="V_EXP_V {0}, {1}, {2}",
    ),
    "_V_RECI_V_ROW": _Template(
        slots=[_slot_expr_cached, _slot_expr_cached, _slot_literal_int],
        fmt="V_RECI_V {0}, {1}, {2}",
    ),
    # S_ADDI_INT for setting up mask reset (gp_mask, gp0, 0). Cached
    # form (gp_mask is in the group cache from an earlier
    # _PRELOAD_ADDR / materialise). Prefix avoids colliding with future
    # uses of S_ADDI_INT that take freshly-materialised GPs.
    "_S_ADDI_INT_RESET_MASK": _Template(
        slots=[_slot_expr_cached, _slot_verbatim, _slot_literal_int],
        fmt="S_ADDI_INT {0}, {1}, {2}",
    ),
    # S_LD_FP / S_ST_FP variants where the FP-side GP is cached (an
    # FPRAM destination address staying live across multiple ops).
    # ``f1, gp_cached, 0`` and ``gp_cached, f1, 0`` patterns.
    "_S_LD_FP_CACHED": _Template(
        slots=[_slot_verbatim, _slot_expr_cached, _slot_literal_int],
        fmt="S_LD_FP {0}, {1}, {2}",
    ),
    "_S_ST_FP_CACHED": _Template(
        slots=[_slot_verbatim, _slot_expr_cached, _slot_literal_int],
        fmt="S_ST_FP {0}, {1}, {2}",
    ),
    # Matrix ops. Legacy emit_btmm form:
    #   M_BTMM gp0, gp{rhs_mram_base}, gp{lhs_packed_vram_base}
    # All three operand slots use cached-GP (the two base addresses
    # were just preloaded via S_ADDI_INT; gp0 is the constant-zero
    # verbatim token used as a dummy result accumulator).
    "M_BTMM": _Template(
        slots=[_slot_verbatim, _slot_expr_cached, _slot_expr_cached],
        fmt="M_BTMM {0}, {1}, {2}",
    ),
    # Legacy emit_btmm_wo form: M_BMM_WO gp{out_base}, 0
    "M_BMM_WO": _Template(
        slots=[_slot_expr_cached, _slot_literal_int],
        fmt="M_BMM_WO {0}, {1}",
    ),
    # Legacy emit_btmv form (clone of emit_btmm with M_BTMV mnemonic):
    #   M_BTMV gp0, gp{rhs_mram_base}, gp{lhs_packed_vram_base}
    "M_BTMV": _Template(
        slots=[_slot_verbatim, _slot_expr_cached, _slot_expr_cached],
        fmt="M_BTMV {0}, {1}, {2}",
    ),
    # Legacy emit_bmv_wo form: M_BMV_WO gp{out}, 0
    "M_BMV_WO": _Template(
        slots=[_slot_expr_cached, _slot_literal_int],
        fmt="M_BMV_WO {0}, {1}",
    ),
    # M_MV / M_MV_WO — per-iteration HW ops in emit_mv.
    #   M_MV gp0, gp{rhs_base}, gp{lhs_base}    -- both bases cached
    #   M_MV_WO gp{dst_base}, 0
    "M_MV": _Template(
        slots=[_slot_verbatim, _slot_expr_cached, _slot_expr_cached],
        fmt="M_MV {0}, {1}, {2}",
    ),
    "M_MV_WO": _Template(
        slots=[_slot_expr_cached, _slot_literal_int],
        fmt="M_MV_WO {0}, {1}",
    ),
    # M_MM / M_MM_WO — emitted by mm / matmul handlers.
    #   M_MM 0, gp{mat_col_base}, gp{act_row_base}
    #   M_MM_WO gp{result}, gp0, 0
    "M_MM": _Template(
        slots=[_slot_literal_int, _slot_expr_cached, _slot_expr_cached],
        fmt="M_MM {0}, {1}, {2}",
    ),
    "M_MM_WO": _Template(
        slots=[_slot_expr_cached, _slot_verbatim, _slot_literal_int],
        fmt="M_MM_WO {0}, {1}, {2}",
    ),
    # M_TMM — transposed matmul:
    #   M_TMM 0, gp{act_vram}, gp{mat_mram}
    # (note: operand order swapped vs M_MM — rs1 is lhs, rs2 is rhs).
    "M_TMM": _Template(
        slots=[_slot_literal_int, _slot_expr_cached, _slot_expr_cached],
        fmt="M_TMM {0}, {1}, {2}",
    ),
    # M_TMM_A — transpose-A matmul:
    #   M_TMM_A 0, gp{mat_mram}, gp{act_vram}
    # (operand order matches M_MM — rs1 is mram rhs, rs2 is vram lhs; the
    # VRAM/A tile is transposed on the fly).
    "M_TMM_A": _Template(
        slots=[_slot_literal_int, _slot_expr_cached, _slot_expr_cached],
        fmt="M_TMM_A {0}, {1}, {2}",
    ),
    # DMA control / data-movement instructions.
    #
    # C_SET_SCALE_REG gp{r}       — set scale length (gp{r} holds it)
    # C_SET_STRIDE_REG gp{r}      — set stride length
    # C_SET_ADDR_REG aN, gp0, gp{r} — load addr-reg ``aN`` from gp{r}
    #
    # The ``aN`` (PLENA addr register) appears as a verbatim string
    # operand on the PreIsaOp (chosen by the producer at allocation
    # time). The legacy ``gp0`` constant-zero source in C_SET_ADDR_REG
    # stays hardcoded in the template (matches PLENA HW encoding).
    "C_SET_SCALE_REG": _Template(
        slots=[_slot_expr_cached],
        fmt="C_SET_SCALE_REG {0}",
    ),
    "C_SET_STRIDE_REG": _Template(
        slots=[_slot_expr_cached],
        fmt="C_SET_STRIDE_REG {0}",
    ),
    # C_SET_ADDR_REG is handled internally by _PRELOAD_ADDR_REG; no
    # separate template needed (the meta-op emits the C_SET_ADDR_REG
    # text directly during its handler).
    #
    # H_PREFETCH_V — HBM→VRAM tile prefetch. Two operand-count forms
    # in legacy emit (5-op batch>1 / 6-op batch=1):
    #   5-op:  H_PREFETCH_V gp{result}, gp{a_off}, aN, 1, 0
    #   6-op:  H_PREFETCH_V gp{result}, gp{a_actual}, aN, 0, 0, 0
    # ``aN`` (3rd slot) is the addr-reg token resolved from the
    # addr-reg cache (populated by an earlier _PRELOAD_ADDR_REG of the
    # same PrimExpr object).
    "H_PREFETCH_V": _Template(
        slots=[_slot_expr_cached, _slot_expr_cached, _slot_addr_reg_cached,
               _slot_literal_int, _slot_literal_int],
        fmt="H_PREFETCH_V {0}, {1}, {2}, {3}, {4}",
    ),
    "_H_PREFETCH_V_6OP": _Template(
        slots=[_slot_expr_cached, _slot_expr_cached, _slot_addr_reg_cached,
               _slot_literal_int, _slot_literal_int, _slot_literal_int],
        fmt="H_PREFETCH_V {0}, {1}, {2}, {3}, {4}, {5}",
    ),
    # H_PREFETCH_M — HBM→MRAM tile prefetch:
    #   H_PREFETCH_M gp{mram}, gp{scale}, aN, 1, 0
    "H_PREFETCH_M": _Template(
        slots=[_slot_expr_cached, _slot_expr_cached, _slot_addr_reg_cached,
               _slot_literal_int, _slot_literal_int],
        fmt="H_PREFETCH_M {0}, {1}, {2}, {3}, {4}",
    ),
    # H_STORE_V — VRAM→HBM tile store:
    #   H_STORE_V gp{vram}, gp{hbm_off}, aN, <flag>, 0
    "H_STORE_V": _Template(
        slots=[_slot_expr_cached, _slot_expr_cached, _slot_addr_reg_cached,
               _slot_literal_int, _slot_literal_int],
        fmt="H_STORE_V {0}, {1}, {2}, {3}, {4}",
    ),
    # More entries land here as handlers migrate.
}


class BackendEmit:
    """Walks a ``PreIsaModule`` and produces final ISA text.

    Construction takes a fully wired ``ProgramShim`` (same one the old
    ``IsaEmitterPass`` uses) so the materialiser sees the same register
    allocator + generated_code sink.

    Call ``run(pre_isa_mod)`` to drive emission; the resulting ISA text
    is read from ``shim.compiler.generated_code`` (a string under the
    new architecture — no CapturingCode proxy).

    Materialisation grouping
    ------------------------
    The legacy ``IsaEmitterPass`` would materialise each HLIR op's
    address operands ONCE per ``begin_op`` / ``end_op`` scope and reuse
    the resulting GP registers across every ISA line that op emitted
    (e.g. an FP binary ``_at`` op reuses 3 GPs across 5 ISA lines
    via ``ra.pin_gp``).

    In PreIsaIR each of those 5 ISA lines is its OWN PreIsaOp, so a
    naive per-PreIsaOp ``begin_op`` / ``end_op`` would re-materialise
    the same address 5 times — emitting 5x as much address-setup ISA
    and breaking byte-equality with the legacy path.

    BackendEmit therefore groups consecutive PreIsaOps that share a
    materialisation scope. The grouping is driven by an integer
    ``annotations["group_id"]`` stamped by the PreIsaPass producer:
    every PreIsaOp produced by ONE call to a legacy-style handler
    (i.e. ONE HLIR op) shares one ``group_id``. BackendEmit opens a
    fresh materialiser scope on a ``group_id`` transition and closes
    it on the next transition; PreIsaOps with no ``group_id`` (e.g.
    free-standing comments) inherit the current scope without
    transitioning.

    Within a group, repeated occurrences of the SAME Python expression
    object (id()) are materialised once and the resulting GP register
    cached — this is how an FP-binary's 3 addresses survive across its
    5 ISA lines with one S_ADDI_INT each rather than five.
    """

    def __init__(self, shim: ProgramShim) -> None:
        self.shim = shim
        self.symbol_table: Dict[tir.Var, Any] = {}
        # Bind the hw-shape constants (mlen, blen, btmm_hlen, ...) into
        # the symbol_table as IntImms taken from the shim. PreIsaIR
        # producers use the symbolic tir.Vars from ``hw_consts``;
        # ExprMaterializer's ``_peephole_const_fold`` substitutes them
        # at materialise time so the final ISA has the hardware's
        # current numeric values folded in — but PreIsaIR itself stays
        # algebraic. See ``hw_consts.py`` for the design.
        from .hw_consts import HW_CONST_ATTRS
        for var, attr in HW_CONST_ATTRS.items():
            self.symbol_table[var] = tir.IntImm(
                "int32", int(getattr(shim, attr)),
            )
        self.materializer = ExprMaterializer(shim, self.symbol_table)
        # Group materialisation cache STACK. Each entry represents one
        # open materialisation scope: a dict keyed by ``id(prim_expr)``
        # with values ``(gp_reg, MaterializedExpr)``.
        #
        # The stack supports nested scopes — outer scopes' cached
        # entries are visible to lookups while inner scopes are open
        # (lookup walks the stack from top to bottom). This is what
        # lets a producer preload an address in an outer iteration
        # (e.g. matmul's per-oc ``mat_addr``) and reference it from
        # PreIsaOps inside an inner unroll body without the inner
        # ``per_iter`` close clobbering the outer entry.
        #
        # When a group is opened with ``_open_group`` a fresh empty
        # dict is pushed; ``_close_group`` pops the top dict and frees
        # its entries (NOT entries in any outer scope). ``_slot_expr``
        # caches into the TOP scope; ``_slot_expr_cached`` looks up
        # through the whole stack.
        self._group_stack: List[Dict[int, Tuple[int, MaterializedExpr]]] = []
        self._group_id_stack: List[Optional[Any]] = []
        # Per-scope close order. Indexed in parallel with _group_stack.
        # Default "reverse" matches the ``for m in reversed(mats):
        # m.release()`` pattern; PreIsaPass producers may set
        # "insertion" via annotations on the group's first PreIsaOp.
        self._group_close_order_stack: List[str] = []
        # Stack of open hardware loops. Each entry is a dict produced
        # by _emit_loop_start_serial and consumed by the matching
        # _emit_loop_end_serial; tracks the loop_var, loop_gp,
        # idx_addr so nested loops compose correctly.
        self._loop_stack: List[Dict[str, Any]] = []
        # Addr-register cache: id(addr_value_expr) -> (a_reg_int, token)
        # — populated by ``_PRELOAD_ADDR_REG`` and looked up by DMA
        # PreIsaOps that need to reference an ``aN`` token. Lives in a
        # per-scope stack mirroring ``_group_stack`` so nested DMA
        # contexts compose. Each entry's addr reg is released on
        # scope close.
        self._addr_reg_stack: List[Dict[int, Tuple[int, str]]] = []
        # Scope-floor stack — minimum allowed scope depth that
        # ``_enter_group_for`` may close down to. Each entry is an
        # int. Pushed by ``_emit_unroll`` (so sibling-gid transitions
        # inside an unroll iter body can't clobber the outer scope
        # that holds e.g. an addr-reg binding). Default floor is 0
        # — no constraint at the top level.
        self._scope_floor: List[int] = []

    # Back-compat properties used by legacy methods (now read top of
    # stack instead of the old flat single-cache state).
    @property
    def _group_cache(self) -> Dict[int, Tuple[int, MaterializedExpr]]:
        if not self._group_stack:
            # No scope open — return an empty dict to make cache hit
            # checks always fail. Writes via this property still go
            # somewhere, but no scope means nothing to write to; we
            # require a scope to be open before any _slot_expr.
            return {}
        return self._group_stack[-1]

    @property
    def _group_open(self) -> bool:
        return bool(self._group_stack)

    @property
    def _current_group_id(self) -> Optional[Any]:
        if not self._group_id_stack:
            return None
        return self._group_id_stack[-1]

    @property
    def _group_close_order(self) -> str:
        if not self._group_close_order_stack:
            return "reverse"
        return self._group_close_order_stack[-1]

    @_group_close_order.setter
    def _group_close_order(self, value: str) -> None:
        if not self._group_close_order_stack:
            return
        self._group_close_order_stack[-1] = value

    def run(self, mod: PreIsaModule) -> str:
        """Emit ``mod`` into ``shim.compiler.generated_code``. Returns
        the final ISA text (str).

        The walker has to handle ``loop_kind="unroll"`` LOOP_STARTs
        specially: when it sees one, it locates the matching LOOP_END
        (respecting nesting) and recursively re-emits the body N
        times with ``loop_var`` rebound to ``tir.IntImm(init + i)``.
        ``loop_kind="serial"`` LOOP_STARTs go straight through
        ``_emit_one`` which calls the hardware-loop path.
        """
        self._run_ops(mod.ops)
        return self.shim.compiler.generated_code

    def _run_ops(self, ops: List[PreIsaOp], _close_at_end: bool = True) -> None:
        """Walk a (sub)sequence of PreIsaOps, handling unroll
        LOOP_START specially. Called recursively for each unrolled
        iteration's body, so nested unrolls compose.

        ``_close_at_end`` controls whether scopes opened DURING this
        call's body are flushed at the end. Snapshots stack depth on
        entry; only scopes pushed since are popped (outer scopes are
        not touched). The outer ``run`` call leaves it True.
        ``_emit_unroll`` in ``"shared"`` mode passes False — the
        unroll body's scope must stay open across iterations.
        """
        entry_depth = len(self._group_stack)
        try:
            i = 0
            n = len(ops)
            while i < n:
                op = ops[i]
                if (
                    op.opcode == "LOOP_START"
                    and op.annotations.get("loop_kind", "serial") == "unroll"
                ):
                    j, body, init_imm, extent_imm, loop_var = (
                        self._locate_unroll_body(ops, i)
                    )
                    self._emit_unroll(
                        body, init_imm, extent_imm, loop_var,
                        annotations=op.annotations,
                    )
                    i = j + 1
                    continue
                self._enter_group_for(op, i)
                self._emit_one(op)
                i += 1
        finally:
            if _close_at_end:
                while len(self._group_stack) > entry_depth:
                    self._close_group()

    def _locate_unroll_body(
        self,
        ops: List[PreIsaOp],
        start_idx: int,
    ) -> Tuple[int, List[PreIsaOp], int, int, tir.Var]:
        """Find the LOOP_END matching the LOOP_START at ``start_idx``
        (must be loop_kind="unroll"). Returns
        ``(end_idx, body_ops, init_imm, extent_imm, loop_var)``.
        Body is the slice ``ops[start_idx+1 .. end_idx-1]``.
        """
        start_op = ops[start_idx]
        if len(start_op.operands) != 2:
            raise BackendEmitError(
                f"LOOP_START at [{start_idx}] expects "
                f"[init_imm, extent_imm]; got {start_op.operands!r}"
            )
        init_imm = int(start_op.operands[0])
        extent_imm = int(start_op.operands[1])
        loop_var = start_op.binds
        if loop_var is None:
            raise BackendEmitError(
                f"LOOP_START at [{start_idx}] has no binds "
                f"(loop iteration var)"
            )

        depth = 1
        j = start_idx + 1
        n = len(ops)
        while j < n:
            opc = ops[j].opcode
            if opc == "LOOP_START":
                depth += 1
            elif opc == "LOOP_END":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j == n:
            raise BackendEmitError(
                f"LOOP_START at [{start_idx}] has no matching LOOP_END"
            )
        body = ops[start_idx + 1:j]
        return j, body, init_imm, extent_imm, loop_var

    def _emit_unroll(
        self,
        body: List[PreIsaOp],
        init_imm: int,
        extent_imm: int,
        loop_var: tir.Var,
        annotations: Dict[str, Any],
    ) -> None:
        """Emit ``extent_imm`` copies of ``body`` with ``loop_var``
        bound to ``IntImm(init_imm + iter)`` in the materialiser's
        symbol_table.

        Two scope-management modes, selected by
        ``annotations["unroll_scope"]``:

          * ``"per_iter"`` (default) — each iteration is a fresh
            materialiser scope; the group cache resets between iters.
            Matches legacy ``_emit_for``'s unrolled branch where
            ``begin_op`` / ``end_op`` runs once per body sub-op per
            iter (no GP carry-over across iters).
          * ``"shared"`` — the materialiser scope OPEN at unroll-loop
            entry is kept open across all iterations. The body's
            id()-keyed cache hits the same PrimExpr objects across
            iters and reuses GPs. Used by handlers that need
            destructive in-place state (e.g. mv's per-tile bump of
            gp_m / gp_o across iters): the bumps mutate the cached
            GP value, the next iter's body picks up the bumped value
            via the SAME cached entry.
        """
        scope_mode = annotations.get("unroll_scope", "per_iter")
        if scope_mode not in ("per_iter", "shared"):
            raise BackendEmitError(
                f"unknown unroll_scope {scope_mode!r}; "
                f"expected 'per_iter' or 'shared'"
            )
        if loop_var in self.symbol_table:
            raise BackendEmitError(
                f"loop_var {loop_var.name!r} already bound — nested "
                f"unroll reusing the same Var is unsupported"
            )
        # Header comment matching legacy.
        self.shim.compiler.generated_code += (
            f"; unroll for {loop_var.name} in "
            f"[{init_imm}, {init_imm + extent_imm}) -- idx is a literal\n"
        )
        # Snapshot scope-stack depth at entry. ``per_iter`` mode resets
        # the stack to THIS depth between iters (closing only scopes
        # OPENED inside the body), preserving outer scopes (e.g.
        # matmul narrow's per-oc mat_addr scope spans the inner t
        # unroll body even though inner uses per_iter).
        entry_depth = len(self._group_stack)
        # Push a scope-floor watermark so ``_enter_group_for``'s
        # sibling-gid-transition close is bounded by the entry depth.
        # Without this, the first body PreIsaOp's gid != outer gid
        # triggers a close of the outer scope itself, killing any
        # addr-reg / GP caches the inner body relies on.
        self._scope_floor.append(entry_depth)
        try:
            for k in range(extent_imm):
                iter_val = init_imm + k
                self.symbol_table[loop_var] = tir.IntImm(
                    "int32", iter_val,
                )
                self.shim.compiler.generated_code += (
                    f"; ... unroll iter {k} -> "
                    f"{loop_var.name}={iter_val}\n"
                )
                if scope_mode == "per_iter":
                    # Close any inner scopes opened during the previous
                    # iter's body. We must NOT close scopes that were
                    # open at unroll entry.
                    while len(self._group_stack) > entry_depth:
                        self._close_group()
                # In shared mode we leave the stack alone.
                self._run_ops(body, _close_at_end=(scope_mode != "shared"))
        finally:
            self.symbol_table.pop(loop_var, None)
            # On exit close down to entry depth.
            while len(self._group_stack) > entry_depth:
                self._close_group()
            # Pop the floor.
            self._scope_floor.pop()

    # ------------------------------------------------------------------
    # group management
    # ------------------------------------------------------------------
    def _enter_group_for(self, op: PreIsaOp, default_idx: int) -> None:
        """Open / transition / close the materialisation scope around
        ``op`` based on its ``annotations["group_id"]``.

        Rules:
          * No group_id: behaves like a singleton group keyed on the
            op's index. Used for _COMMENT and any leaf op the producer
            didn't bother tagging.
          * Same group_id as the open scope: keep the scope open.
          * Different group_id: close the open scope, open a fresh one.
        """
        gid = op.annotations.get("group_id", None)
        # _COMMENT ops never disturb the open scope — they're not real
        # HW instructions and never materialise operands. Lets a
        # comment land in the middle of a multi-line group (e.g. the
        # ``; fp scalar task ... op=mul`` header inside an
        # fp_mul_at's 5-line burst) without forcing a scope flush.
        if op.opcode == "_COMMENT":
            return
        if gid is None:
            # Singleton — close any open scope, then open a fresh
            # one keyed on the default index so each "ungrouped" op
            # gets its own materialiser.begin_op cycle (matches the
            # pre-grouping behaviour for handlers that emit a single
            # ISA line per op).
            gid = ("_singleton", default_idx)
        if self._group_open and gid == self._current_group_id:
            return
        # Close the top scope only if it sits ABOVE the current
        # scope-floor watermark. Inside an unroll iter body, the
        # outer scope (which holds e.g. an _PRELOAD_ADDR_REG'd addr
        # register the inner H_PREFETCH_V needs) is protected by a
        # floor pushed by _emit_unroll — sibling-gid transitions in
        # the body PUSH new scopes on top instead of replacing.
        floor = self._scope_floor[-1] if self._scope_floor else 0
        if len(self._group_stack) > floor:
            self._close_group()
        self._open_group(gid, default_idx)
        # First PreIsaOp of a group may carry close_order in its
        # annotations. Honour it for the rest of the group's lifetime.
        order = op.annotations.get("close_order")
        if order is not None:
            if order not in ("reverse", "insertion"):
                raise BackendEmitError(
                    f"close_order must be 'reverse' or 'insertion'; "
                    f"got {order!r}"
                )
            self._group_close_order = order

    def _open_group(self, gid: Any, idx: int) -> None:
        """Push a fresh materialisation scope onto the stack."""
        self.materializer.set_lowir_op_idx(idx)
        self.materializer.begin_op()
        self._group_stack.append({})
        self._group_id_stack.append(gid)
        self._group_close_order_stack.append("reverse")
        self._addr_reg_stack.append({})

    def _close_group(self) -> None:
        """Pop the TOP materialisation scope, freeing its cached GPs.
        Outer scopes (deeper in the stack) remain open and their
        cached GPs visible to subsequent lookups.

        Legacy emitters use two different release patterns:
          * fp_*_at, v_*, row_*_at: ``for m in reversed(mats):
            m.release()`` — releases in REVERSE insertion order, so
            the FIRST-inserted reg ends up on top.
          * emit_btmm, emit_btmm_wo, emit_mv: ``ra.free_gp(gp_regs)``
            — passes a list in insertion order; free_gp iterates that
            list, so the LAST-inserted reg ends up on top.

        The PreIsaPass producer stamps each group with the desired
        order via ``annotations["close_order"]`` on the group's FIRST
        PreIsaOp. The setting was snapshotted into
        ``_group_close_order_stack`` at open time.
        """
        if not self._group_stack:
            return
        top_cache = self._group_stack.pop()
        self._group_id_stack.pop()
        close_order = self._group_close_order_stack.pop()
        # Release any addr-regs allocated in this scope BEFORE popping
        # so the allocator's free_addr happens before the gp release.
        addr_top = (
            self._addr_reg_stack.pop()
            if self._addr_reg_stack
            else {}
        )
        ra = self.shim.compiler.register_allocator
        for _key, (a_reg, _tok) in addr_top.items():
            ra.free_addr([a_reg])
        items = list(top_cache.values())
        if close_order == "insertion":
            iter_order = items
        else:
            iter_order = list(reversed(items))
        for _gp, m in iter_order:
            if m.owns_register:
                ra.unpin_gp(m.register)
            m.release()
        self.materializer.end_op()

    # ------------------------------------------------------------------
    # per-op emit
    # ------------------------------------------------------------------
    def _emit_one(self, op: PreIsaOp) -> None:
        if op.opcode == "_COMMENT":
            text = op.operands[0] if op.operands else ""
            self.shim.compiler.generated_code += f"; {text}\n"
            return

        if op.opcode == "LOOP_START":
            # Strategy is on the PreIsaOp; default = "serial".
            kind = op.annotations.get("loop_kind", "serial")
            if kind == "serial":
                self._emit_loop_start_serial(op)
            elif kind == "unroll":
                # Unrolled loops are handled by ``run`` because they
                # need to drive the body-replay loop themselves. The
                # main ``run`` walker detects this opcode + kind and
                # never calls ``_emit_one`` on it directly.
                raise BackendEmitError(
                    "LOOP_START with loop_kind='unroll' must be "
                    "expanded by run()'s outer walker, not reach "
                    "_emit_one — internal invariant violated"
                )
            else:
                raise BackendEmitError(
                    f"LOOP_START: unknown loop_kind {kind!r} "
                    f"(expected 'serial' or 'unroll')"
                )
            return

        if op.opcode == "LOOP_END":
            # Mirror of LOOP_START dispatch — serial closes the HW
            # loop. Unroll's LOOP_END is consumed by run() outside.
            kind = op.annotations.get("loop_kind", "serial")
            if kind == "serial":
                self._emit_loop_end_serial(op)
            elif kind == "unroll":
                raise BackendEmitError(
                    "LOOP_END with loop_kind='unroll' must be "
                    "skipped by run()'s outer walker"
                )
            else:
                raise BackendEmitError(
                    f"LOOP_END: unknown loop_kind {kind!r}"
                )
            return

        if op.opcode == "_PRELOAD_ADDR_REG":
            # Allocate a PLENA addr register, materialise the operand
            # value into a scratch GP, emit the C_SET_ADDR_REG bind,
            # and cache (id(operand) -> ("aN", a_reg_int)) for any
            # later DMA PreIsaOp that references the same operand.
            if len(op.operands) != 1:
                raise BackendEmitError(
                    f"_PRELOAD_ADDR_REG expects 1 operand (the addr "
                    f"value expr); got {len(op.operands)}"
                )
            val = op.operands[0]
            if not self._addr_reg_stack:
                raise BackendEmitError(
                    "_PRELOAD_ADDR_REG: no open scope to cache the "
                    "addr-register binding. Open a group_id first."
                )
            top = self._addr_reg_stack[-1]
            key = id(val)
            if key in top:
                # Already bound — no-op (legacy would emit a redundant
                # C_SET_ADDR_REG; producer should not duplicate).
                return
            ra = self.shim.compiler.register_allocator
            a_reg = ra.allocate_addr(1)[0]
            tok = f"a{a_reg}"
            # Materialise the value into a GP first (this puts the
            # GP into the current group cache; subsequent ops in the
            # same scope can reference ``val`` via _slot_expr_cached
            # to get the same GP back, or use the cached addr-reg
            # token through the addr-reg cache).
            m_val_tok, _h = self._invoke_slot(_slot_expr, val)
            # Emit C_SET_ADDR_REG aN, gp0, gp{r}
            self.shim.compiler.generated_code += (
                f"C_SET_ADDR_REG {tok}, gp0, {m_val_tok}\n"
            )
            top[key] = (a_reg, tok)
            return

        if op.opcode == "_PRELOAD_ADDR":
            # Materialise the operand PrimExpr into a GP now and stash
            # it in the group cache. Mirrors legacy
            # _emit_fp_scalar_op_at's upfront materialisation loop —
            # without this, addresses materialise lazily on first ISA
            # use, which interleaves S_ADDI_INTs with FP ops and breaks
            # byte-equality with the legacy emitter.
            if len(op.operands) != 1:
                raise BackendEmitError(
                    f"_PRELOAD_ADDR expects 1 operand (the address "
                    f"PrimExpr); got {len(op.operands)}"
                )
            val = op.operands[0]
            # Drive through _invoke_slot so the cache machinery runs
            # exactly the way it would for a real ISA op.
            self._invoke_slot(_slot_expr, val)
            return

        if op.opcode == "_BUMP_CACHED_GP":
            # ``S_ADDI_INT gp{N}, gp{N}, stride`` where gp{N} is the
            # cached GP for the operand PrimExpr. Mutates the cached
            # value — subsequent _slot_expr / _slot_expr_cached
            # lookups of the same expr return the same GP but its
            # value is now ``orig + stride``. Producer must arrange
            # iteration semantics accordingly (mirrors legacy's
            # destructive in-place stride bump in row_*_at).
            if len(op.operands) != 2:
                raise BackendEmitError(
                    f"_BUMP_CACHED_GP expects [cached_expr, stride]; "
                    f"got {op.operands!r}"
                )
            cached_expr, stride = op.operands
            # Stride may be a compile-time int OR a PrimExpr that
            # references hw-shape consts (BLEN_VAR / MLEN_VAR etc).
            # The latter goes through ``_peephole_const_fold`` to
            # substitute the symbol_table-bound IntImms and simplify
            # to a literal integer at emit time.
            if isinstance(stride, int):
                stride_int = stride
            elif isinstance(stride, tir.IntImm):
                stride_int = int(stride.value)
            elif isinstance(stride, tir.PrimExpr):
                folded = self.materializer._peephole_const_fold(stride)
                if isinstance(folded, tir.IntImm):
                    stride_int = int(folded.value)
                else:
                    raise BackendEmitError(
                        f"_BUMP_CACHED_GP stride PrimExpr did not fold "
                        f"to an IntImm at emit time: {stride!r} -> "
                        f"{folded!r}. All free vars must be bound in "
                        f"symbol_table."
                    )
            else:
                raise BackendEmitError(
                    f"_BUMP_CACHED_GP stride must be int / IntImm / "
                    f"PrimExpr; got {type(stride).__name__} {stride!r}"
                )
            key = id(cached_expr)
            hit = self._lookup_cache(key)
            if hit is None:
                raise BackendEmitError(
                    f"_BUMP_CACHED_GP: PrimExpr not in any open scope "
                    f"(no preceding _PRELOAD_ADDR for the same Python "
                    f"PrimExpr object)"
                )
            gp, _m = hit
            self.shim.compiler.generated_code += (
                f"S_ADDI_INT gp{gp}, gp{gp}, {stride_int}\n"
            )
            return

        tmpl = _TEMPLATES.get(op.opcode)
        if tmpl is None:
            raise BackendEmitError(
                f"no BackendEmit template for opcode {op.opcode!r}. "
                f"The handler that produced this PreIsaOp must be "
                f"matched by a row in pre_isa_emit._TEMPLATES."
            )

        if len(op.operands) != len(tmpl.slots):
            raise BackendEmitError(
                f"{op.opcode}: operand count {len(op.operands)} does "
                f"not match template arity {len(tmpl.slots)}"
            )

        tokens: List[str] = []
        # Per-emit list of "fresh" handles to release at the end.
        # Cached handles (group-shared) are NOT released here — they
        # live until _close_group.
        fresh_handles: List[MaterializedExpr] = []
        try:
            ra = self.shim.compiler.register_allocator
            for slot_fn, val in zip(tmpl.slots, op.operands):
                tok, handle = self._invoke_slot(slot_fn, val)
                tokens.append(tok)
                if handle is not None and handle is not _CACHED_SENTINEL:
                    fresh_handles.append(handle)
                    # Pin until end of emit so the next operand's
                    # materialise() can't auto-spill it (mirrors the
                    # legacy ``ra.pin_gp(m.register)`` pattern in
                    # _emit_fp_scalar_op_at).
                    if handle.owns_register:
                        ra.pin_gp(handle.register)
            self.shim.compiler.generated_code += tmpl.fmt.format(*tokens) + "\n"
        finally:
            ra = self.shim.compiler.register_allocator
            for h in reversed(fresh_handles):
                if h.owns_register:
                    ra.unpin_gp(h.register)
            # Fresh (non-cached) handles release immediately; cached
            # ones stay alive in self._group_cache.

    # ------------------------------------------------------------------
    # loop handling — LOOP_START / LOOP_END
    # ------------------------------------------------------------------
    def _emit_loop_start_serial(self, op: PreIsaOp) -> None:
        """Open a hardware (serial) loop. Emits the literal PLENA ISA
        ``C_LOOP_START gp_loop, extent`` along with the idx-init.

        operands = [init_imm:int, extent_imm:int]
        binds    = the loop iteration tir.Var (body PreIsaOps reference
                   it via PrimExpr operands)
        annotations["loop_gp"] = the GP reserved by loop_register_alloc
                                  for this loop's HW counter

        Mirrors legacy ``_emit_for``'s serial branch byte-for-byte.
        """
        if len(op.operands) != 2:
            raise BackendEmitError(
                f"LOOP_START expects [init_imm, extent_imm]; "
                f"got {op.operands!r}"
            )
        init_imm = int(op.operands[0])
        extent_imm = int(op.operands[1])
        loop_var = op.binds
        loop_gp = op.annotations.get("loop_gp")
        if loop_var is None:
            raise BackendEmitError(
                f"LOOP_START has no binds (loop iteration tir.Var)"
            )
        if loop_gp is None:
            raise BackendEmitError(
                f"LOOP_START (serial) has no 'loop_gp' annotation "
                f"(must be set by PreIsaPass from the HLIR op's "
                f"loop_register_alloc stamp)"
            )

        self._close_group()

        ra = self.shim.compiler.register_allocator
        if loop_var in self.symbol_table:
            raise BackendEmitError(
                f"loop_var {loop_var.name!r} already bound — nested "
                f"loops reusing the same Var aren't supported"
            )
        ra.pin_gp(loop_gp)
        idx_addr = ra.claim_idx_slot()

        if init_imm == 0:
            self.shim.compiler.generated_code += (
                f"; for {loop_var.name} in [{init_imm}, "
                f"{init_imm + extent_imm}) -- hw counter gp{loop_gp}, "
                f"idx ram[{idx_addr}]\n"
                f"S_ST_INT gp0, gp0, {idx_addr}\n"
                f"C_LOOP_START gp{loop_gp}, {extent_imm}\n"
            )
        else:
            init_gp = ra.allocate_gp(1)[0]
            self.shim.compiler.generated_code += (
                f"; for {loop_var.name} in [{init_imm}, "
                f"{init_imm + extent_imm}) -- hw counter gp{loop_gp}, "
                f"idx ram[{idx_addr}]\n"
                f"S_ADDI_INT gp{init_gp}, gp0, {init_imm}\n"
                f"S_ST_INT gp{init_gp}, gp0, {idx_addr}\n"
                f"C_LOOP_START gp{loop_gp}, {extent_imm}\n"
            )
            ra.free_gp([init_gp])

        self.symbol_table[loop_var] = ("ram", idx_addr)
        # Stash state for the matching LOOP_END.
        self._loop_stack.append({
            "loop_var": loop_var,
            "loop_gp": loop_gp,
            "idx_addr": idx_addr,
        })

    def _emit_loop_end_serial(self, op: PreIsaOp) -> None:
        """Close the most-recent serial loop opened by
        ``_emit_loop_start_serial``. Emits the literal PLENA ISA
        ``C_LOOP_END gp_loop`` along with the idx-increment epilogue.
        Matches legacy ``_emit_for`` byte-for-byte.
        """
        if not self._loop_stack:
            raise BackendEmitError(
                f"LOOP_END with no matching LOOP_START on the stack"
            )
        self._close_group()

        st = self._loop_stack.pop()
        loop_var = st["loop_var"]
        loop_gp = st["loop_gp"]
        idx_addr = st["idx_addr"]

        ra = self.shim.compiler.register_allocator
        inc_gp = ra.allocate_gp(1)[0]
        self.shim.compiler.generated_code += (
            f"; idx {loop_var.name} += 1 (ram[{idx_addr}])\n"
            f"S_LD_INT gp{inc_gp}, gp0, {idx_addr}\n"
            f"S_ADDI_INT gp{inc_gp}, gp{inc_gp}, 1\n"
            f"S_ST_INT gp{inc_gp}, gp0, {idx_addr}\n"
            f"C_LOOP_END gp{loop_gp}\n"
        )
        ra.free_gp([inc_gp])

        self.symbol_table.pop(loop_var, None)
        ra.unpin_gp(loop_gp)
        ra.release_idx_slot(idx_addr)

    def _lookup_cache(
        self, key: int,
    ) -> Optional[Tuple[int, MaterializedExpr]]:
        """Walk the scope stack top-to-bottom looking for ``key``. The
        first hit wins — inner scope shadows outer. Returns ``None``
        when ``key`` is in no open scope."""
        for scope in reversed(self._group_stack):
            hit = scope.get(key)
            if hit is not None:
                return hit
        return None

    def _invoke_slot(
        self,
        slot_fn: Callable[[Any, ExprMaterializer], Tuple[str, Optional[MaterializedExpr]]],
        val: Any,
    ) -> Tuple[str, Optional[MaterializedExpr]]:
        """Resolve a slot.

        ``_slot_expr`` caches into the TOP open scope on first sight;
        subsequent occurrences of the same Python expression object
        (``id`` match) anywhere in the scope stack return the cached
        GP. Other slot kinds (literal int, verbatim string) never
        cache.

        ``_slot_expr_cached`` is the explicit "must already be in
        SOME open scope" form — looks up the PrimExpr through the
        whole scope stack. Used by destructive in-place patterns
        (row_*_at's d_tile stride bump) and nested-scope patterns
        (matmul narrow's outer-scope ``mat_addr`` referenced from the
        inner unroll body).
        """
        if slot_fn is _slot_expr_cached:
            key = id(val)
            hit = self._lookup_cache(key)
            if hit is None:
                raise BackendEmitError(
                    f"_slot_expr_cached: PrimExpr {val!r} not in any "
                    f"open group cache. The producer must have emitted "
                    f"a _PRELOAD_ADDR for this expr earlier in an open "
                    f"scope."
                )
            gp, _m = hit
            return f"gp{gp}", _CACHED_SENTINEL
        if slot_fn is _slot_addr_reg_cached:
            key = id(val)
            # Walk the addr-reg stack top-to-bottom.
            for scope in reversed(self._addr_reg_stack):
                hit = scope.get(key)
                if hit is not None:
                    _a_reg, tok = hit
                    return tok, _CACHED_SENTINEL
            raise BackendEmitError(
                f"_slot_addr_reg_cached: PrimExpr {val!r} not in any "
                f"open addr-reg cache. The producer must have emitted "
                f"a _PRELOAD_ADDR_REG for this expr earlier in an open "
                f"scope."
            )
        if slot_fn is _slot_expr:
            key = id(val)
            hit = self._lookup_cache(key)
            if hit is not None:
                gp, _m = hit
                return f"gp{gp}", _CACHED_SENTINEL
            # First sight — materialise and cache into TOP scope.
            if not self._group_stack:
                # Defensive: an _PRELOAD_ADDR / _slot_expr outside any
                # scope is a producer bug.
                raise BackendEmitError(
                    f"_slot_expr: no open scope to cache materialised "
                    f"PrimExpr {val!r} into. The producer must open a "
                    f"group (via group_id annotation) before its first "
                    f"PrimExpr operand."
                )
            tok, handle = slot_fn(val, self.materializer)
            if handle is not None:
                ra = self.shim.compiler.register_allocator
                if handle.owns_register:
                    ra.pin_gp(handle.register)
                self._group_stack[-1][key] = (handle.register, handle)
                return tok, _CACHED_SENTINEL
            return tok, handle
        return slot_fn(val, self.materializer)


__all__ = ["BackendEmit", "BackendEmitError"]
