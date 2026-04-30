"""Structural tests for FPRAM-backed FP ops."""

import sys

from tilelang_tvm_compiler.address_alloc import FPRAM_USER_BASE
from tilelang_tvm_compiler.kernels.fpram_smoke import fpram_smoke
from tilelang_tvm_compiler.pipeline import compile_kernel, PlenaTarget


def _compile():
    return compile_kernel(fpram_smoke, target=PlenaTarget(), name="fpram_smoke")


def test_hlir_collects_fpram_buffers():
    ck = _compile()
    fpram_bufs = [b for b in ck.hlir.buffers.values() if b.scope == "fpram"]
    names = [b.name for b in fpram_bufs]
    assert names == ["F_src", "F_tmp", "Row_max"], names
    print(f"[ok] HLIR records FPRAM buffers: {names}")


def test_fpram_buffers_get_distinct_addresses():
    ck = _compile()
    f_src = ck.hlir.buffers["F_src"]
    f_tmp = ck.hlir.buffers["F_tmp"]
    row_max = ck.hlir.buffers["Row_max"]
    assert (f_src.address, f_tmp.address, row_max.address) == (
        FPRAM_USER_BASE,
        FPRAM_USER_BASE + 128,
        FPRAM_USER_BASE + 256,
    )
    print(f"[ok] FPRAM addresses are sequential: {f_src.address}, {f_tmp.address}, {row_max.address}")


def test_isa_contains_map_fp_and_scalar_fp_ops():
    ck = _compile()
    asm = ck.isa_text
    assert "S_MAP_FP_V" in asm, asm
    assert "S_MAP_V_FP" in asm, asm
    assert "S_LD_FP" in asm, asm
    assert "S_ST_FP" in asm, asm
    assert "S_EXP_FP" in asm, asm
    print("[ok] ISA contains FP map/load/store/exp instructions")


def test_isa_contains_row_reduce_and_row_scalar_vector_op():
    ck = _compile()
    asm = ck.isa_text
    assert "V_RED_MAX" in asm, asm
    assert "V_SUB_VF" in asm, asm
    assert "C_LOOP_START" not in asm, asm
    print("[ok] ISA contains row reduce and row scalar-vector op without emitter-side row loops")


def main():
    tests = [
        test_hlir_collects_fpram_buffers,
        test_fpram_buffers_get_distinct_addresses,
        test_isa_contains_map_fp_and_scalar_fp_ops,
        test_isa_contains_row_reduce_and_row_scalar_vector_op,
    ]
    print("=" * 60)
    print(f"fpram structural tests ({len(tests)} cases)")
    print("=" * 60)
    for test in tests:
        test()
    print("=" * 60)
    print(f"ALL {len(tests)} TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
