"""Byte-equal verification for the migrated ``fp_*_at`` family
(``fp_copy_at`` / ``fp_exp_at`` / ``fp_reci_at`` / ``fp_sqrt_at`` /
``fp_add_at`` / ``fp_sub_at`` / ``fp_mul_at`` / ``fp_max_at``).

Each variant builds the same minimal HLIRModule, runs it through both
the legacy ``IsaEmitterPass`` and the new ``PreIsaPass`` + ``BackendEmit``
path, and asserts the emitted HW-instruction lines are byte-equal.

These tests are the proof that PreIsaIR's "group_id" materialisation
scoping correctly reuses the legacy path's pin/release pattern across
the multi-line burst each ``_at`` op emits.
"""

import pytest

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


def _mk_buf(name: str, addr: int) -> _hlir.Buffer:
    return _hlir.Buffer(
        name=name, scope=_scope.FPRAM, shape=(1,), dtype="float16",
        address=addr,
    )


def _hlir_unary(kind: str) -> _hlir.HLIRModule:
    buffers = {
        "src": _mk_buf("src", 64),
        "dst": _mk_buf("dst", 128),
    }
    op = _hlir.Op(
        kind=kind,
        buffer_args=[],
        scalar_args=[
            _hlir.BufferElement(buffer="src", indices=(0,)),
            _hlir.BufferElement(buffer="dst", indices=(0,)),
        ],
    )
    return _hlir.HLIRModule(
        name=kind, buffers=buffers, ops=[op], param_names=[],
    )


def _hlir_binary(kind: str) -> _hlir.HLIRModule:
    buffers = {
        "lhs": _mk_buf("lhs", 32),
        "rhs": _mk_buf("rhs", 64),
        "dst": _mk_buf("dst", 128),
    }
    op = _hlir.Op(
        kind=kind,
        buffer_args=[],
        scalar_args=[
            _hlir.BufferElement(buffer="lhs", indices=(0,)),
            _hlir.BufferElement(buffer="rhs", indices=(0,)),
            _hlir.BufferElement(buffer="dst", indices=(0,)),
        ],
    )
    return _hlir.HLIRModule(
        name=kind, buffers=buffers, ops=[op], param_names=[],
    )


def _byte_equal(hlir: _hlir.HLIRModule) -> None:
    """Run both paths against ``hlir`` and assert HW-instruction
    byte-equality."""
    shim_legacy = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    assert _instr_lines(legacy_isa) == _instr_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nnew:\n{new_isa}"
    )


@pytest.mark.parametrize("kind", ["fp_copy_at", "fp_exp_at", "fp_reci_at", "fp_sqrt_at"])
def test_fp_unary_at_byte_equal(kind):
    _byte_equal(_hlir_unary(kind))


@pytest.mark.parametrize("kind", ["fp_add_at", "fp_sub_at", "fp_mul_at", "fp_max_at"])
def test_fp_binary_at_byte_equal(kind):
    _byte_equal(_hlir_binary(kind))
