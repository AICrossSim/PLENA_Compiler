"""Byte-equal verification for ``for`` (HLIR structured op).

A serial for-loop wraps one ``fp_zero_at`` body op. Legacy emits a
3-line prelude (init=0 case: S_ST_INT + C_LOOP_START with the
``;`` header), the body, and a 4-line epilogue (S_LD_INT /
S_ADDI_INT / S_ST_INT / C_LOOP_END). The PreIsaIR path must emit
byte-equal.

The body PrimExpr references the loop variable so this also exercises
the symbol_table binding cycle: PreIsaPass leaves ``loop_var`` symbolic
in the operand; BackendEmit's C_LOOP_START handler binds it via
``symbol_table[loop_var] = ("ram", idx_addr)`` and the body op's
materialise resolves via S_LD_INT.
"""

from tvm import tir

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler.backend_emit import BackendEmit
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass import PreIsaPass
from tilelang_tvm_compiler.program_shim import make_shim


def _instr_lines(text: str):
    return [
        ln.strip()
        for ln in text.split("\n")
        if ln.strip() and not ln.strip().startswith(";")
    ]


def test_for_serial_byte_equal_body_uses_loop_var():
    """``for i in [0, 4): fp_zero_at dst_fp[i]`` — the body references
    ``i`` in its scalar_args, so the inner S_LD_INT must come from
    BackendEmit's symbol_table binding ``i -> ("ram", idx_addr)`` set
    up by C_LOOP_START."""
    i = tir.Var("i", "int32")
    # The dst FPRAM has 4 slots; body op writes f0 to slot index ``i``.
    buf = _hlir.Buffer(
        name="dst_fp", scope=_scope.FPRAM,
        shape=(4,), dtype="float16", address=128,
    )
    body = _hlir.Op(
        kind="fp_zero_at",
        buffer_args=[],
        scalar_args=[_hlir.BufferElement(buffer="dst_fp", indices=(i,))],
    )
    # loop_register_alloc stamps loop_gp; we mimic it here.
    for_op = _hlir.Op(
        kind="for",
        buffer_args=[],
        scalar_args=[],
        annotations={
            "loop_var": i,
            "extent": 4,
            "init": 0,
            "loop_kind": "serial",
            "loop_gp": 15,  # legacy loop_register_alloc usually reserves high GPs
        },
        body=[body],
    )
    hlir = _hlir.HLIRModule(
        name="for_smoke", buffers={"dst_fp": buf},
        ops=[for_op], param_names=[],
    )

    shim_legacy = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    # legacy pass needs the loop_gp reserved in its allocator too. Reserve
    # by allocating it upfront (mirrors what loop_register_alloc does
    # via the RegisterAllocator constructor's gp_reserved). The simplest
    # cross-test approach is to pre-pin gp15 — that ensures both paths
    # see the same allocator state when they enter the for-op.
    shim_legacy.compiler.register_allocator.pin_gp(15)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)
    shim_legacy.compiler.register_allocator.unpin_gp(15)

    shim_new = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    shim_new.compiler.register_allocator.pin_gp(15)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)
    shim_new.compiler.register_allocator.unpin_gp(15)

    assert _instr_lines(legacy_isa) == _instr_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nnew:\n{new_isa}"
    )
