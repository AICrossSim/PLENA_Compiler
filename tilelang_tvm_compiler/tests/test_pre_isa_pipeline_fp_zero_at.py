"""End-to-end PreIsaIR pipeline verification for ``fp_zero_at`` (the
first migrated handler).

Two paths produce ISA text for the same minimal HLIRModule:

  legacy:  HLIR -> IsaEmitterPass.run -> ISA text
  new:     HLIR -> PreIsaPass.run -> PreIsaModule
                -> BackendEmit.run -> ISA text

With no optimisation enabled, the new path must produce
**byte-equal** ISA to the legacy path (modulo the header comment block
emitted directly by IsaEmitterPass.run vs the same comments produced
as _COMMENT PreIsaOps; we compare only the non-comment lines, which
are the actual HW instructions).

This is the proof-of-concept that the migration architecture works:
PreIsaIR + BackendEmit round-trips a real ISA emission with no loss
or drift.
"""

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler.backend_emit import BackendEmit
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass import PreIsaPass
from tilelang_tvm_compiler.program_shim import make_shim


def _build_minimal_hlir():
    """One ``fp_zero_at`` op against an FPRAM buffer at addr=128.
    ``fp_zero_at`` is the simplest leaf — one scalar arg, two ISA
    lines (comment + S_ST_FP)."""
    buf = _hlir.Buffer(
        name="dst_fp",
        scope=_scope.FPRAM,
        shape=(1,),
        dtype="float16",
        address=128,
    )
    op = _hlir.Op(
        kind="fp_zero_at",
        buffer_args=[],
        scalar_args=[_hlir.BufferElement(buffer="dst_fp", indices=(0,))],
    )
    return _hlir.HLIRModule(
        name="fpz",
        buffers={"dst_fp": buf},
        ops=[op],
        param_names=[],
    )


def _instr_lines(text: str):
    """Return the non-comment, non-blank instruction lines from ``text``.
    These are the actual HW instructions whose count + form must match
    byte-for-byte between the legacy and PreIsaIR paths."""
    return [
        ln.strip()
        for ln in text.split("\n")
        if ln.strip() and not ln.strip().startswith(";")
    ]


def test_legacy_path_emits_s_st_fp():
    """Legacy path: the materialiser writes one S_ADDI_INT to load the
    FPRAM address into a GP, then the handler emits the S_ST_FP."""
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    isa = IsaEmitterPass(shim).run(_build_minimal_hlir())
    instrs = _instr_lines(isa)
    assert instrs == ["S_ADDI_INT gp1, gp0, 128", "S_ST_FP f0, gp1, 0"], (
        f"legacy expected S_ADDI_INT + S_ST_FP; got {instrs!r}"
    )


def test_pre_isa_path_produces_pre_isa_module():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPass(shim).run(_build_minimal_hlir())
    real_ops = [op for op in pre.ops if op.opcode != "_COMMENT"]
    assert len(real_ops) == 1
    assert real_ops[0].opcode == "S_ST_FP"
    # Operands: [str "f0", PrimExpr dst_addr, int 0]
    operands = real_ops[0].operands
    assert operands[0] == "f0"
    assert isinstance(operands[2], int) and operands[2] == 0
    # The middle operand is a PrimExpr in var-ref form (the BufferElement
    # resolved to ``128 + 0 == 128``, an IntImm — but the materialiser
    # will be the one to lower it later).
    import tvm.tir as _tir
    assert isinstance(operands[1], (_tir.PrimExpr, int))


def test_backend_emit_produces_byte_equal_isa():
    """The whole point of this proof-of-concept: drive PreIsaPass +
    BackendEmit and confirm the ISA instruction lines are byte-equal
    to the legacy IsaEmitterPass output."""
    hlir = _build_minimal_hlir()

    shim_legacy = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    legacy_instrs = _instr_lines(legacy_isa)
    new_instrs = _instr_lines(new_isa)
    assert legacy_instrs == new_instrs, (
        "PreIsaIR path must produce byte-equal HW instructions to "
        "the legacy path when no optimisation is enabled.\n"
        f"legacy: {legacy_instrs!r}\n"
        f"new:    {new_instrs!r}"
    )
