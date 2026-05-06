"""Structural tests for the minimal online-softmax (HBM round-trip) kernel."""

import sys

from tilelang_tvm_compiler.kernels.online_softmax_min import make_online_softmax_hbm
from tilelang_tvm_compiler.pipeline import compile_kernel, PlenaTarget


def test_online_softmax_hbm_isa_contains_expected_ops():
    fn, _ = make_online_softmax_hbm(active_lane=2)
    ck = compile_kernel(fn, target=PlenaTarget(), name="online_softmax_hbm")
    asm = ck.isa_text
    for needle in [
        "H_PREFETCH_V",
        "V_RED_MAX", "V_RED_SUM",
        "V_SUB_VF", "V_EXP_V",
        "S_LD_FP", "S_ST_FP",
        "S_SUB_FP", "S_EXP_FP", "S_MUL_FP", "S_ADD_FP",
    ]:
        assert needle in asm, needle
    print("[ok] online_softmax_hbm ISA contains DMA + vector + scalar FP instructions")


def test_online_softmax_hbm_has_no_fpram_buffers():
    fn, _ = make_online_softmax_hbm(active_lane=2)
    ck = compile_kernel(fn, target=PlenaTarget(), name="online_softmax_hbm")
    fpram_bufs = [b for b in ck.hlir.buffers.values() if b.scope == "fpram"]
    assert fpram_bufs == [], [b.name for b in fpram_bufs]
    print("[ok] online_softmax_hbm exposes no fpram buffers (scalar fpram addressing)")


def test_packed_row_at_emits_vmask_setup():
    fn, _ = make_online_softmax_hbm(active_lane=2)
    ck = compile_kernel(fn, target=PlenaTarget(), name="online_softmax_hbm")
    asm = ck.isa_text
    assert "C_SET_V_MASK_REG" in asm, asm
    print("[ok] row_*_at synthesizes V_MASK setup for packed-head dim3")


def main():
    tests = [
        test_online_softmax_hbm_isa_contains_expected_ops,
        test_online_softmax_hbm_has_no_fpram_buffers,
        test_packed_row_at_emits_vmask_setup,
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
