"""End-to-end test for the v2 pipeline (the clean rewrite).

Path under test:
    HLIR
      ↓ PreIsaPassV2
    PreIsaIR v2 (clean — PrimExpr operands only)
      ↓ pre_isa_to_mir
    MIR (SSA values, def/use, loops)
      ↓ mir_to_isa (trivial regalloc POC)
    ISA text

Compared against legacy:
    HLIR → IsaEmitterPass → ISA text

The check is *structural*, not byte-equal: legacy and v2 must emit
the same set of PLENA mnemonics in the same order, modulo
GP-renumbering and (for v2) a single extra S_ADDI_INT zero-load
that the legacy avoids (because legacy materialises addresses
lazily; v2 always synthesises an explicit ``%addr = S_ADDI_INT
gp0, IMM`` SSA def before the use).

We use a tolerant equality: the set of HW mnemonics that the two
paths emit must be byte-equal, ignoring lines whose mnemonic is
``S_ADDI_INT`` (those are address-setup boilerplate that
allocator/peephole work changes the count of).
"""

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler import mir
from tilelang_tvm_compiler import pre_isa_to_mir as p2m
from tilelang_tvm_compiler import mir_to_isa as m2i
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass_v2 import PreIsaPassV2
from tilelang_tvm_compiler.program_shim import make_shim


def _build_fp_zero_at_hlir():
    """One ``fp_zero_at`` op against an FPRAM scalar buffer at addr=128."""
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


import re

_GP_RE = re.compile(r"\bgp\d+\b")


def _non_addi_lines(isa: str):
    """Non-S_ADDI ISA lines with gpN → gpX canonicalisation."""
    out = []
    for ln in isa.split("\n"):
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        head = s.split(None, 1)[0]
        if head == "S_ADDI_INT":
            continue
        out.append(_GP_RE.sub("gpX", s))
    return out


def test_fp_zero_at_v2_pipeline_runs():
    """Smoke: the v2 pipeline (HLIR → PreIsaIR v2 → MIR → ISA) runs
    end-to-end and produces non-empty ISA text."""
    hlir = _build_fp_zero_at_hlir()
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    mir.verify(fn)
    isa = m2i.emit(fn, shim)
    assert "S_ST_FP" in isa


def test_fp_zero_at_v2_matches_legacy_hw_mnemonics():
    """The set of non-S_ADDI HW mnemonics in v2 == legacy."""
    hlir = _build_fp_zero_at_hlir()

    shim_legacy = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPassV2(shim_new).run(hlir)
    fn = p2m.convert(pre, shim_new)
    mir.verify(fn)
    new_isa = m2i.emit(fn, shim_new)

    legacy_lines = _non_addi_lines(legacy_isa)
    new_lines = _non_addi_lines(new_isa)
    assert legacy_lines == new_lines, (
        f"\nlegacy non-S_ADDI:\n  " + "\n  ".join(legacy_lines)
        + f"\nv2 non-S_ADDI:\n  " + "\n  ".join(new_lines)
    )


def test_fp_zero_at_v2_mir_dump():
    """Sanity that the MIR dump has the expected structure:
       Function constants:  %0_gp0:i32 = <gp0_const>
       ^entry:
         _COMMENT
         %1 = S_ADDI_INT %0_gp0, 128
         S_ST_FP f0, %1, 0
    """
    hlir = _build_fp_zero_at_hlir()
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    text = mir.format_mir(fn)
    # The address 128 should appear as the immediate of an S_ADDI_INT.
    assert "S_ADDI_INT" in text and "128" in text, text
    assert "S_ST_FP" in text and "f0" in text, text
