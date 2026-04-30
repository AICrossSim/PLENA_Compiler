"""Structural tests for minimal online softmax and masked row ops."""

import re
import sys

from tilelang_tvm_compiler.kernels.online_softmax_min import (
    make_online_softmax_hbm,
    make_online_softmax_min,
)
from tilelang_tvm_compiler.kernels.row_mask_smoke import make_row_mask_smoke
from tilelang_tvm_compiler.pipeline import compile_kernel, PlenaTarget


def test_online_softmax_hlir_sequence():
    fn, _ = make_online_softmax_min()
    ck = compile_kernel(fn, target=PlenaTarget(), name="online_softmax_min")
    kinds = [op.kind for op in ck.hlir.ops]
    assert kinds == [
        "row_reduce_max",
        "fp_max",
        "fp_sub",
        "fp_exp",
        "row_sub_fp",
        "row_exp",
        "row_reduce_sum",
        "fp_mul",
        "fp_add",
        "fp_copy",
        "fp_copy",
    ], kinds
    print("[ok] online softmax HLIR sequence matches expected update order")


def test_online_softmax_isa_contains_expected_ops():
    fn, _ = make_online_softmax_min()
    ck = compile_kernel(fn, target=PlenaTarget(), name="online_softmax_min")
    asm = ck.isa_text
    for needle in ["V_RED_MAX", "V_SUB_VF", "V_EXP_V", "V_RED_SUM", "S_MAX_FP", "S_SUB_FP", "S_EXP_FP", "S_MUL_FP", "S_ADD_FP"]:
        assert needle in asm, needle
    print("[ok] online softmax ISA contains vector reduce/transform and scalar FP update ops")


def test_masked_row_ops_emit_vmask_sequence():
    fn, c = make_row_mask_smoke(active_lane=2)
    ck = compile_kernel(fn, target=PlenaTarget(), name="row_mask_smoke")
    asm = ck.isa_text
    assert re.search(rf"S_ADDI_INT gp\d+, gp0, {c['MASK_VAL']}\b", asm), asm
    assert "C_SET_V_MASK_REG" in asm, asm
    assert "V_MUL_VF" in asm and "V_RED_SUM" in asm, asm
    print("[ok] masked row ops emit V_MASK setup and masked vector instructions")


def test_row_at_ops_derive_vmask_from_logical_dims():
    fn, _ = make_online_softmax_hbm(active_lane=2)
    ck = compile_kernel(fn, target=PlenaTarget(), name="online_softmax_hbm")
    asm = ck.isa_text
    assert "C_SET_V_MASK_REG" in asm, asm
    assert "V_RED_MAX" in asm and "V_RED_SUM" in asm, asm
    print("[ok] row_*_at ops derive packed-head V_MASK from logical dims")


def main():
    tests = [
        test_online_softmax_hlir_sequence,
        test_online_softmax_isa_contains_expected_ops,
        test_masked_row_ops_emit_vmask_sequence,
        test_row_at_ops_derive_vmask_from_logical_dims,
    ]
    print("=" * 60)
    print(f"online softmax structural tests ({len(tests)} cases)")
    print("=" * 60)
    for test in tests:
        test()
    print("=" * 60)
    print(f"ALL {len(tests)} TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
