"""End-to-end v2 tests for the fp_*_at family
(copy/exp/reci/sqrt/add/sub/mul/max).

Path:  HLIR  →  PreIsaPassV2  →  PreIsaIR v2
            →  pre_isa_to_mir  →  MIR
            →  mir_to_isa      →  ISA text

Structural compare against legacy: same non-S_ADDI mnemonics in
same order. S_ADDI_INT count may differ (legacy mixes addresses
with the body; v2's trivial allocator emits all setup S_ADDIs
up front via the SSA chain).
"""

import pytest

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler import mir
from tilelang_tvm_compiler import pre_isa_to_mir as p2m
from tilelang_tvm_compiler import mir_to_isa as m2i
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass_v2 import PreIsaPassV2
from tilelang_tvm_compiler.program_shim import make_shim


import re

_GP_RE = re.compile(r"\bgp\d+\b")


def _non_addi_mnemonic_skeleton(isa: str):
    """Return the list of non-S_ADDI ISA lines with GP numbers
    canonicalised away — every ``gpN`` becomes ``gpX``. Allows v2's
    aggressive GP reuse to compare equal to legacy's separate GPs
    when the algorithm is the same."""
    out = []
    for ln in isa.split("\n"):
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        head = s.split(None, 1)[0]
        if head == "S_ADDI_INT":
            continue
        # Canonicalise gp numbers — both paths get the same skeleton.
        out.append(_GP_RE.sub("gpX", s))
    return out


def _mk_fpram(name, addr):
    return _hlir.Buffer(
        name=name, scope=_scope.FPRAM, shape=(1,), dtype="float16",
        address=addr,
    )


def _hlir_unary(kind: str) -> _hlir.HLIRModule:
    return _hlir.HLIRModule(
        name=kind,
        buffers={
            "src": _mk_fpram("src", 64),
            "dst": _mk_fpram("dst", 128),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[],
            scalar_args=[
                _hlir.BufferElement(buffer="src", indices=(0,)),
                _hlir.BufferElement(buffer="dst", indices=(0,)),
            ],
        )],
        param_names=[],
    )


def _hlir_binary(kind: str) -> _hlir.HLIRModule:
    return _hlir.HLIRModule(
        name=kind,
        buffers={
            "lhs": _mk_fpram("lhs", 32),
            "rhs": _mk_fpram("rhs", 64),
            "dst": _mk_fpram("dst", 128),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[],
            scalar_args=[
                _hlir.BufferElement(buffer="lhs", indices=(0,)),
                _hlir.BufferElement(buffer="rhs", indices=(0,)),
                _hlir.BufferElement(buffer="dst", indices=(0,)),
            ],
        )],
        param_names=[],
    )


def _v2_emit(hlir):
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    mir.verify(fn)
    return m2i.emit(fn, shim)


def _legacy_emit(hlir):
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    return IsaEmitterPass(shim).run(hlir)


@pytest.mark.parametrize("kind", [
    "fp_copy_at", "fp_exp_at", "fp_reci_at", "fp_sqrt_at",
])
def test_fp_unary_v2_matches_legacy(kind):
    hlir = _hlir_unary(kind)
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _non_addi_mnemonic_skeleton(legacy_isa) == _non_addi_mnemonic_skeleton(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )


@pytest.mark.parametrize("kind", [
    "fp_add_at", "fp_sub_at", "fp_mul_at", "fp_max_at",
])
def test_fp_binary_v2_matches_legacy(kind):
    hlir = _hlir_binary(kind)
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _non_addi_mnemonic_skeleton(legacy_isa) == _non_addi_mnemonic_skeleton(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )
