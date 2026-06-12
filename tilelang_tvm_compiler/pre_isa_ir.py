"""PreIsaIR — the var-ref IR layer between IsaPass and the ISA emitter.

Pipeline position:

    HLIR
      |
      v   IsaPass — every handler appends PreIsaOps. Operand math
      |             (the PrimExpr addresses) is identical to the
      |             pre-PreIsaIR code path; what changes is the SINK:
      |             instead of materialising the expr to a GP register
      |             and emitting an ISA line, the handler records a
      |             PreIsaOp(opcode, operands=[PrimExpr, ...]).
      v
    PreIsaIR  -- one PreIsaOp == one PLENA ISA instruction.
      |          operands are tir.PrimExpr / int / str (loop vars are
      |          still ``tir.Var``; nothing is bound to a GP yet).
      |
      v   pre_isa_optimize — arith.simplify on every operand; CSE
      |                      across PreIsaOps within a loop region;
      |                      LICM hoisting subexprs that don't depend
      |                      on the enclosing C_LOOP_START's binds var.
      v
    PreIsaIR (optimised)
      |
      v   BackendEmit — walks the PreIsaOp stream linearly:
      |                 for each op, materialises each PrimExpr operand
      |                 to a GP register via the existing
      |                 ExprMaterializer, then emits one ISA text line
      |                 per the opcode's template.
      v
    ISA text

Iron rule: every PreIsaOp emits exactly one PLENA HW instruction.
C_LOOP_START / C_LOOP_END are themselves PreIsaOps — a "for-loop" in
PreIsaIR is two HW-loop PreIsaOps with body PreIsaOps between them,
sitting in one flat stream. There is no PreIsaFor container — structure
is by C_LOOP_START / C_LOOP_END matching the way the HW executes.

Operand semantics:
    * tir.PrimExpr — symbolic address / value. Loop vars are unresolved
      tir.Var; BackendEmit calls ExprMaterializer at lowering time.
    * int — compile-time-known immediate (loop count, mask bit, flag).
    * str — hard-coded token (e.g. "f0", "a3") dropped verbatim into
      the ISA template by BackendEmit.

This layer carries NO register decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tvm import tir


# All PLENA hardware mnemonics that may appear as a PreIsaOp.opcode.
# Sourced from the inventory of every single-ISA-line mnemonic the
# pre-PreIsaIR isa_emitter.py / isa_pass.py write into generated_code.
KNOWN_OPCODES = frozenset({
    # control — UNIFIED loop pair. The kind of loop is carried by
    # ``annotations["loop_kind"]`` on the LOOP_START PreIsaOp:
    #   * "serial" (default) → BackendEmit emits a hardware loop:
    #     S_ST_INT (idx init) + C_LOOP_START gp_loop, extent
    #     + body once + idx-inc + C_LOOP_END gp_loop.
    #   * "unroll" → BackendEmit expands the body N times inline,
    #     binding loop_var to tir.IntImm(init+i) per iteration (no
    #     hardware C_LOOP, no idx slot, no loop_gp use).
    # Optimisation passes treat both kinds uniformly (they're the
    # same data shape); a "switch" pass can change ``loop_kind`` on
    # any LOOP_START to flip the codegen strategy without touching
    # the surrounding IR.
    #
    # IMPORTANT: PreIsaIR opcodes ``LOOP_START`` / ``LOOP_END`` are
    # PreIsaIR-level control markers. The PLENA ISA *mnemonics*
    # ``C_LOOP_START`` and ``C_LOOP_END`` that the hardware actually
    # executes are emitted as plain text by BackendEmit; they don't
    # appear as PreIsaIR opcodes (and need no entry here — they're
    # just strings inside emit templates).
    "LOOP_START", "LOOP_END", "C_BREAK",
    "C_SET_V_MASK_REG", "C_SET_ADDR_REG",
    "C_SET_SCALE_REG", "C_SET_STRIDE_REG",
    "C_SET_FP_REG", "C_RUN_FP_KERNEL",
    # scalar int
    "S_ADDI_INT", "S_ADD_INT", "S_SUB_INT", "S_MUL_INT",
    "S_LUI_INT", "S_LD_INT", "S_ST_INT",
    "S_SLLI_INT", "S_SRLI_INT", "S_SLL_INT", "S_SRL_INT",
    "S_MV_INT",
    # scalar fp
    "S_LD_FP", "S_ST_FP",
    "S_ADD_FP", "S_SUB_FP", "S_MUL_FP", "S_MAX_FP",
    "S_EXP_FP", "S_RECI_FP", "S_SQRT_FP",
    "S_MAP_FP_V", "S_MAP_V_FP",
    "S_MV_FP",
    # vector
    "V_ADD_VV", "V_SUB_VV", "V_MUL_VV",
    "V_ADD_VF", "V_SUB_VF", "V_MUL_VF",
    "V_EXP_V", "V_RECI_V", "V_SQRT_V",
    "V_RED_MAX", "V_RED_SUM",
    "V_AND_VV", "V_OR_VV", "V_XOR_VV", "V_NOT_V",
    "V_MAX_VV", "V_MIN_VV", "V_MAX_VF", "V_MIN_VF",
    "V_SHIFT_V", "V_SHFTL_V",
    # matrix
    "M_BTMM", "M_BTMM_WO", "M_BMM_WO",
    "M_BTMV", "M_BMV_WO",
    "M_MV", "M_MV_WO",
    "M_MM", "M_MM_WO",
    "M_TMM", "M_TMM_A",
    # HBM
    "H_LOAD_V", "H_STORE_V", "H_PREFETCH_V", "H_PREFETCH_M",
    # meta — translated by BackendEmit into a "; ..." comment line.
    "_COMMENT",
    # meta — forces BackendEmit to materialise the operand PrimExpr
    # into a GP register NOW (and cache it in the current group scope),
    # without emitting its own ISA line. The S_ADDI_INT / S_LUI_INT
    # that the materialiser writes is the actual on-disk evidence;
    # this meta-op exists so the PreIsaPass producer can dictate the
    # ORDER of address materialisations relative to the HW ops that
    # use them, matching the legacy "materialise all addresses first,
    # then emit all HW ops" pattern (see _emit_fp_scalar_op_at).
    "_PRELOAD_ADDR",
    # meta — emit ``S_ADDI_INT gp{N}, gp{N}, stride`` where ``gp{N}`` is
    # the register the operand PrimExpr currently lives in (must already
    # be in the BackendEmit group cache; produced earlier via a
    # _PRELOAD_ADDR or implicit materialise). After the bump the cached
    # GP holds ``expr + stride``, NOT the original value of ``expr``.
    # This mirrors legacy's destructive in-place stride bump pattern in
    # _emit_row_scalar_op_at (and emit_matmul / similar) where a single
    # GP is walked across d_tile iterations via repeated S_ADDI_INTs.
    # Operands: [cached_addr_expr (PrimExpr), stride (int)].
    "_BUMP_CACHED_GP",
    # meta — allocate a PLENA addr register, load its value from the
    # operand PrimExpr, and cache the binding under the operand's
    # ``id()`` so subsequent PreIsaOps referencing the same Python
    # object get the same ``aN`` token. Side-effect emit:
    #   * ``_load_large_int`` style S_ADDI_INT / S_LUI_INT sequence to
    #     materialise the value into a scratch GP
    #   * ``C_SET_ADDR_REG aN, gp0, gp{scratch}``
    # Operand: [addr_value_expr] (tir.PrimExpr).
    # The producer must use the SAME Python PrimExpr object across
    # all DMA PreIsaOps that reference the same addr register.
    "_PRELOAD_ADDR_REG",
    # Variant opcodes — same emitted ISA mnemonic as the corresponding
    # non-prefixed entry but with cached-GP operand semantics for
    # patterns that read the SAME GP across multiple PreIsaOps
    # (row_*_at's d_tile unroll). The leading underscore avoids
    # colliding with the canonical form's slot signature.
    "_V_SUB_VF_ROW", "_V_ADD_VF_ROW", "_V_MUL_VF_ROW",
    "_V_EXP_V_ROW", "_V_RECI_V_ROW",
    "_S_ADDI_INT_RESET_MASK",
    "_S_LD_FP_CACHED", "_S_ST_FP_CACHED",
    # H_PREFETCH_V 6-operand variant (batch=1 path in legacy
    # _emit_preload_tile_isa); see backend_emit._TEMPLATES.
    "_H_PREFETCH_V_6OP",
})


@dataclass
class PreIsaOp:
    opcode: str
    operands: List[Any] = field(default_factory=list)
    binds: Optional[tir.Var] = None
    annotations: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.opcode not in KNOWN_OPCODES:
            raise ValueError(
                f"PreIsaOp opcode {self.opcode!r} is not a known PLENA "
                f"mnemonic. If this is a real HW instruction add it to "
                f"KNOWN_OPCODES (pre_isa_ir.py) after confirming it is "
                f"a single-ISA-line atom (one PreIsaOp == one ISA line)."
            )


@dataclass
class PreIsaModule:
    name: str
    ops: List[PreIsaOp] = field(default_factory=list)
    buffers: Dict[str, Any] = field(default_factory=dict)

    def append(self, op: PreIsaOp) -> None:
        self.ops.append(op)

    def comment(self, text: str) -> None:
        self.ops.append(PreIsaOp(opcode="_COMMENT", operands=[text]))


def _fmt_operand(x: Any) -> str:
    if isinstance(x, (int, float)):
        return str(x)
    if isinstance(x, str):
        return x
    if isinstance(x, tir.Var):
        return x.name
    return str(x)


def format_pre_isa(mod: PreIsaModule) -> str:
    lines = [f"PreIsaModule(name={mod.name!r})", ""]
    if mod.buffers:
        lines.append("Buffers:")
        name_w = max((len(n) for n in mod.buffers), default=4)
        for name, b in mod.buffers.items():
            addr = getattr(b, "address", None)
            scope = getattr(b, "scope", "?")
            shape = getattr(b, "shape", ())
            shape_s = "x".join(str(s) for s in shape) if shape else "()"
            addr_s = "<unalloc>" if addr is None else str(addr)
            lines.append(
                f"  {name:<{name_w}}  scope={scope:<5}  addr={addr_s:<8}  "
                f"shape={shape_s}"
            )
        lines.append("")
    lines.append("Ops:")
    indent = 2
    for idx, op in enumerate(mod.ops):
        if op.opcode in _LOOP_END_OPCODES:
            indent = max(2, indent - 4)
        ind = " " * indent
        if op.opcode == "_COMMENT":
            text = op.operands[0] if op.operands else ""
            lines.append(f"{ind}[{idx:4d}]  ; {text}")
        else:
            ops_s = ", ".join(_fmt_operand(o) for o in op.operands)
            binds_s = (
                f"  binds={op.binds.name}"
                if op.binds is not None
                else ""
            )
            note = op.annotations.get("comment", "")
            note_s = f"   ; {note}" if note else ""
            lines.append(
                f"{ind}[{idx:4d}]  {op.opcode:<18} {ops_s}{binds_s}{note_s}"
            )
        if op.opcode in _LOOP_START_OPCODES:
            indent += 4
    return "\n".join(lines) + "\n"


_LOOP_START_OPCODES = ("LOOP_START",)
_LOOP_END_OPCODES = ("LOOP_END",)


def loop_regions(
    ops: List[PreIsaOp],
) -> List[Tuple[int, int, Optional[tir.Var]]]:
    """Yield ``(start_idx, end_idx, loop_var)`` for every
    ``LOOP_START`` / ``LOOP_END`` pair. Both ``loop_kind="serial"``
    and ``loop_kind="unroll"`` loops use the same opcode pair, so
    LICM / CSE / other passes can iterate this uniformly."""
    out: List[Tuple[int, int, Optional[tir.Var]]] = []
    stack: List[Tuple[int, Optional[tir.Var]]] = []
    for i, op in enumerate(ops):
        if op.opcode == "LOOP_START":
            stack.append((i, op.binds))
        elif op.opcode == "LOOP_END":
            if not stack:
                raise ValueError(
                    f"LOOP_END at [{i}] with no matching loop-start"
                )
            start_idx, var = stack.pop()
            out.append((start_idx, i, var))
    if stack:
        raise ValueError(
            f"unclosed loop-start(s) at indices "
            f"{[s for s, _ in stack]} — missing end marker"
        )
    return out


__all__ = [
    "PreIsaOp", "PreIsaModule",
    "KNOWN_OPCODES", "format_pre_isa", "loop_regions",
]
