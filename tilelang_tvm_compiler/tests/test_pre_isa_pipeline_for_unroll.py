"""Byte-equal verification for the HLIR for-loop with
``loop_kind="unroll"``.

Legacy ``_emit_for`` unrolled branch (isa_pass.py:3189-3231):
  * binds loop_var to ``tir.IntImm(init+i)`` for each iter;
  * emits one ``; ... unroll iter`` header per iter;
  * replays the body sub-ops with the IntImm-bound loop_var so
    materialise() constant-folds it.

PreIsaPass migrated unroll branch:
  * emits a single ``LOOP_START`` PreIsaOp with
    ``annotations["loop_kind"] = "unroll"``;
  * walks the HLIR body sub-ops once (producing one set of body
    PreIsaOps inside the LOOP_START/LOOP_END pair);
  * BackendEmit's run() detects unroll-kind and replays the body
    PreIsaOps N times, binding loop_var to IntImm per iter.

This test exercises that whole pipeline against legacy on a small
HLIR: ``for i in [0, 3) unroll: fp_zero_at dst[i]`` — i.e. one
unrolled for-loop with a single fp_zero_at body that references the
loop var in its scalar_args.
"""

from tvm import tir

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler.backend_emit import BackendEmit
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass import PreIsaPass
from tilelang_tvm_compiler.program_shim import make_shim


def _instr_lines(text):
    return [
        ln.strip() for ln in text.split("\n")
        if ln.strip() and not ln.strip().startswith(";")
    ]


def test_for_unroll_byte_equal():
    """Unrolled for-loop body uses ``i`` in its scalar_args. Each iter's
    materialise sees ``i`` bound to IntImm(0/1/2) and constant-folds
    the address."""
    i = tir.Var("i", "int32")
    buf = _hlir.Buffer(
        name="dst_fp", scope=_scope.FPRAM,
        shape=(4,), dtype="float16", address=128,
    )
    body = _hlir.Op(
        kind="fp_zero_at",
        buffer_args=[],
        scalar_args=[_hlir.BufferElement(buffer="dst_fp", indices=(i,))],
    )
    for_op = _hlir.Op(
        kind="for",
        buffer_args=[],
        scalar_args=[],
        annotations={
            "loop_var": i,
            "extent": 3,
            "init": 0,
            "loop_kind": "unroll",
        },
        body=[body],
    )
    hlir = _hlir.HLIRModule(
        name="for_unroll_smoke",
        buffers={"dst_fp": buf},
        ops=[for_op],
        param_names=[],
    )

    shim_legacy = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    assert _instr_lines(legacy_isa) == _instr_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nnew:\n{new_isa}"
    )


def test_for_unroll_body_with_loop_var_in_addr():
    """Same as above but with a 4-iter range to make sure the
    sequence of materialised addresses (128, 129, 130, 131) shows
    up in the expected order on both paths."""
    i = tir.Var("i", "int32")
    buf = _hlir.Buffer(
        name="dst_fp", scope=_scope.FPRAM,
        shape=(8,), dtype="float16", address=256,
    )
    body = _hlir.Op(
        kind="fp_zero_at",
        buffer_args=[],
        scalar_args=[_hlir.BufferElement(buffer="dst_fp", indices=(i,))],
    )
    for_op = _hlir.Op(
        kind="for",
        buffer_args=[],
        scalar_args=[],
        annotations={
            "loop_var": i,
            "extent": 4,
            "init": 0,
            "loop_kind": "unroll",
        },
        body=[body],
    )
    hlir = _hlir.HLIRModule(
        name="for_unroll_4",
        buffers={"dst_fp": buf},
        ops=[for_op],
        param_names=[],
    )

    shim_legacy = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    assert _instr_lines(legacy_isa) == _instr_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nnew:\n{new_isa}"
    )
    # Sanity: 4 iterations -> 4 S_ST_FP instructions.
    instrs = _instr_lines(new_isa)
    assert sum(1 for s in instrs if s.startswith("S_ST_FP")) == 4, instrs
