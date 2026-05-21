"""End-to-end v2 tests for the HLIR ``for`` op — serial + unroll.

Loops in PreIsaIR v2 are LoopRegions; the MIR conversion turns them
into MirLoops whose body is a MirBlock — the SSA scope of the loop
body. No GP pinning, no symbol-table state machine, no scope_floor.
Just nested blocks, the way LLVM IR (and TVM) models loops.
"""

import re

import pytest
from tvm import tir

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler import mir
from tilelang_tvm_compiler import pre_isa_to_mir as p2m
from tilelang_tvm_compiler import mir_to_isa as m2i
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass_v2 import PreIsaPassV2
from tilelang_tvm_compiler.program_shim import make_shim


_GP_RE = re.compile(r"\bgp\d+\b")


def _strip(isa: str):
    """Comparable lines: non-S_ADDI, gpN canonicalised."""
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


def _v2_emit(hlir):
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    mir.verify(fn)
    return m2i.emit(fn, shim)


def _legacy_emit(hlir):
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    return IsaEmitterPass(shim).run(hlir)


def _fpram(name, addr, shape=(4,)):
    return _hlir.Buffer(
        name=name, scope=_scope.FPRAM,
        shape=shape, dtype="float16", address=addr,
    )


def test_for_unroll_with_fp_zero_at():
    """``for i in [0, 3) unroll: fp_zero_at dst[i]`` — body op uses
    the loop var in its address. Unroll expands 3 body copies."""
    i = tir.Var("i", "int32")
    buf = _fpram("dst", 128)
    body_op = _hlir.Op(
        kind="fp_zero_at",
        buffer_args=[],
        scalar_args=[_hlir.BufferElement(buffer="dst", indices=(i,))],
    )
    for_op = _hlir.Op(
        kind="for",
        buffer_args=[],
        scalar_args=[],
        annotations={
            "loop_var": i, "extent": 3, "init": 0,
            "loop_kind": "unroll",
        },
        body=[body_op],
    )
    hlir = _hlir.HLIRModule(
        name="for_unroll_fp_zero",
        buffers={"dst": buf},
        ops=[for_op],
        param_names=[],
    )

    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    # Both paths should emit 3 S_ST_FP lines (one per unrolled iter).
    assert _strip(legacy_isa) == _strip(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )
    # Sanity — 3 stores.
    assert sum(1 for l in new_isa.split("\n") if "S_ST_FP" in l) == 3


@pytest.mark.parametrize("ext", [3, 5])
def test_for_unroll_extent_param(ext):
    """Vary the unroll extent — verify v2 matches legacy at multiple
    sizes."""
    i = tir.Var(f"i_{ext}", "int32")
    buf = _fpram("dst", 256, shape=(ext + 1,))
    body_op = _hlir.Op(
        kind="fp_zero_at",
        buffer_args=[],
        scalar_args=[_hlir.BufferElement(buffer="dst", indices=(i,))],
    )
    for_op = _hlir.Op(
        kind="for",
        buffer_args=[],
        scalar_args=[],
        annotations={
            "loop_var": i, "extent": ext, "init": 0,
            "loop_kind": "unroll",
        },
        body=[body_op],
    )
    hlir = _hlir.HLIRModule(
        name="for_unroll_param",
        buffers={"dst": buf},
        ops=[for_op],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _strip(legacy_isa) == _strip(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )


def test_for_serial_with_fp_zero_at():
    """``for i in [0, 4) serial: fp_zero_at dst[i]`` — serial loop
    emits HW C_LOOP_START/C_LOOP_END + idx slot. Body's S_LD_INT
    on each iter reads loop_var from IntRAM."""
    i = tir.Var("i_serial", "int32")
    buf = _fpram("dst", 512, shape=(8,))
    body_op = _hlir.Op(
        kind="fp_zero_at",
        buffer_args=[],
        scalar_args=[_hlir.BufferElement(buffer="dst", indices=(i,))],
    )
    for_op = _hlir.Op(
        kind="for",
        buffer_args=[],
        scalar_args=[],
        annotations={
            "loop_var": i, "extent": 4, "init": 0,
            "loop_kind": "serial",
            "loop_gp": 15,  # legacy needs this annotation
        },
        body=[body_op],
    )
    hlir = _hlir.HLIRModule(
        name="for_serial_fp_zero",
        buffers={"dst": buf},
        ops=[for_op],
        param_names=[],
    )

    # Legacy run: pre-pin gp15 so its allocator state matches.
    shim_legacy = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    shim_legacy.compiler.register_allocator.pin_gp(15)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)
    shim_legacy.compiler.register_allocator.unpin_gp(15)

    new_isa = _v2_emit(hlir)

    # Structural: both should have exactly 1 C_LOOP_START and 1
    # C_LOOP_END, and the same number of S_ST_FP (body executes once
    # per source-code body, the HW loop runs it 4 times — so 1
    # S_ST_FP in the static text).
    def _count(isa, mnem):
        return sum(1 for l in isa.split("\n") if l.strip().startswith(mnem))
    assert _count(legacy_isa, "C_LOOP_START") == _count(new_isa, "C_LOOP_START") == 1
    assert _count(legacy_isa, "C_LOOP_END") == _count(new_isa, "C_LOOP_END") == 1
    assert _count(legacy_isa, "S_ST_FP") == _count(new_isa, "S_ST_FP") == 1
