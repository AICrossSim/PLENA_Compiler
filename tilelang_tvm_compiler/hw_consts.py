"""Hardware-shape constants as symbolic ``tir.Var``s.

PreIsaPass producers write address expressions in terms of these
symbolic vars rather than baking the current hardware values
(``shim.mlen``, ``shim.blen``, etc.) into ``tir.IntImm``s. BackendEmit
binds each symbolic var to the shim's current numeric value via
``symbol_table`` at run start; the materialiser's
``_peephole_const_fold`` then substitutes the IntImms back in and
folds the address algebra at emit time.

Why symbolic
------------
The hardware shape parameters are NOT compile-time constants: the
chip variant being targeted changes them, parameterised tests
sweep them, and the ``plena_settings.toml`` active mode dictates
their current values. PreIsaIR keeps them symbolic so optimisations
that operate on PreIsaIR (LICM, CSE, stride-detection) see the
algebra structure — e.g. ``Mul(oc, BLEN_VAR)`` — rather than a
post-substitution ``Mul(oc, IntImm(4))`` where the ``4`` is
indistinguishable from any other unrelated literal.

The vars are MODULE-LEVEL SINGLETONS so every producer + every
BackendEmit consumer references the same Python object (``id()``
match). This lets BackendEmit's group-cache + lookup machinery do
its job — ``id(BLEN_VAR)`` is the symbol_table key on both sides.

Usage in producers
------------------
Import the relevant vars (or grab the dict via ``hw_const_vars()``)
and use them in PrimExpr operands:

    from .hw_consts import BLEN_VAR
    mat_col_expr = tir.Add(
        tir.IntImm("int32", int(rhs.address)),
        tir.Mul(oc_var, BLEN_VAR),
    )

For values derived from the shape constants
(``output_row_stride = blen * mlen``) write the derivation
explicitly as a PrimExpr — let the materialiser simplify:

    output_row_stride_expr = tir.Mul(BLEN_VAR, MLEN_VAR)
    Mul(orow_var, output_row_stride_expr)

Usage in BackendEmit
--------------------
``BackendEmit.__init__`` binds every hw_const var into
``symbol_table`` using the values from its shim:

    for var, attr in HW_CONST_ATTRS.items():
        self.symbol_table[var] = tir.IntImm("int32", int(getattr(shim, attr)))
"""

from __future__ import annotations

from typing import Dict

from tvm import tir


# Singletons. Same Python objects in every module that imports.
MLEN_VAR = tir.Var("mlen", "int32")
BLEN_VAR = tir.Var("blen", "int32")
BTMM_HLEN_VAR = tir.Var("btmm_hlen", "int32")
BTMM_LANE_COUNT_VAR = tir.Var("btmm_lane_count", "int32")
# Rows transferred per H_PREFETCH_V / H_STORE_V instruction —
# the emulator's PREFETCH_V_AMOUNT / STORE_V_AMOUNT. The DMA
# helpers use these as the per-instruction VLEN-row count.
# Different chip variants have different values.
V_PREFETCH_AMOUNT_VAR = tir.Var("v_prefetch_amount", "int32")
V_WRITEBACK_AMOUNT_VAR = tir.Var("v_writeback_amount", "int32")


# Map of hw-const tir.Var -> ProgramShim attribute name. BackendEmit
# iterates this at startup to populate symbol_table.
HW_CONST_ATTRS: Dict[tir.Var, str] = {
    MLEN_VAR: "mlen",
    BLEN_VAR: "blen",
    BTMM_HLEN_VAR: "btmm_hlen",
    BTMM_LANE_COUNT_VAR: "btmm_lane_count",
    V_PREFETCH_AMOUNT_VAR: "v_prefetch_amount",
    V_WRITEBACK_AMOUNT_VAR: "v_writeback_amount",
}


def hw_const_vars() -> Dict[str, tir.Var]:
    """Convenience dict by name (for producers that prefer string keys)."""
    return {
        "mlen": MLEN_VAR,
        "blen": BLEN_VAR,
        "btmm_hlen": BTMM_HLEN_VAR,
        "btmm_lane_count": BTMM_LANE_COUNT_VAR,
        "v_prefetch_amount": V_PREFETCH_AMOUNT_VAR,
        "v_writeback_amount": V_WRITEBACK_AMOUNT_VAR,
    }


__all__ = [
    "MLEN_VAR", "BLEN_VAR", "BTMM_HLEN_VAR", "BTMM_LANE_COUNT_VAR",
    "V_PREFETCH_AMOUNT_VAR", "V_WRITEBACK_AMOUNT_VAR",
    "HW_CONST_ATTRS", "hw_const_vars",
]
