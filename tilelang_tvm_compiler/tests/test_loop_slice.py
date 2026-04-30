"""Structural tests for loop_slice_dma: validates Phase 7 dynamic-start
slice + ExprMaterializer + register-sourced offset emit path.

Run:
    LD_LIBRARY_PATH="" \\
    PYTHONPATH=/.../compiler \\
    .venv-tvm/bin/python -m tilelang_tvm_compiler.tests.test_loop_slice
"""

from __future__ import annotations

import re
import sys

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler.kernels.loop_slice_dma import (
    GROUP_HEADS,
    HLEN,
    MLEN,
    NUM_BLOCKS,
    SEQ_TOTAL,
    loop_slice_dma,
)
from tilelang_tvm_compiler.pipeline import PlenaTarget, compile_kernel


def test_hlir_records_for_then_slice():
    ck = compile_kernel(loop_slice_dma, target=PlenaTarget(), name="loop_slice")
    ops = ck.hlir.ops
    assert len(ops) == 1 and ops[0].kind == "for"
    body = ops[0].body
    assert len(body) == 1 and body[0].kind == "dma_h2v_slice"
    sl = body[0].buffer_args[0]
    assert isinstance(sl, _hlir.BufferSlice)
    # The slice's seq-dim start is dynamic (a PrimExpr) -- NOT an int.
    assert not isinstance(sl.starts[1], int), (
        f"expected dynamic PrimExpr at starts[1], got {type(sl.starts[1]).__name__}"
    )
    print(f"[ok] HLIR: for-op containing dma_h2v_slice with dynamic seq start")


def test_isa_emits_outer_loop():
    ck = compile_kernel(loop_slice_dma, target=PlenaTarget(), name="loop_slice")
    asm = ck.isa_text
    starts = re.findall(rf"C_LOOP_START gp(\d+), {NUM_BLOCKS}\b", asm)
    assert len(starts) == 1, f"expected one outer C_LOOP_START extent={NUM_BLOCKS}, got {starts}"
    print(f"[ok] outer C_LOOP_START gp{starts[0]}, {NUM_BLOCKS}")


def test_isa_strength_reduces_dynamic_offset():
    """`i * MLEN` should compile to S_SLLI_INT (since MLEN is a power of 2)."""
    ck = compile_kernel(loop_slice_dma, target=PlenaTarget(), name="loop_slice")
    asm = ck.isa_text
    # Find the loop body
    loop_body = asm.split("C_LOOP_START")[1]  # everything after first outer-loop start
    # We expect at least one S_SLLI_INT inside the body for the dynamic
    # offset computation. (Strict count omitted because TVM may or may
    # not pre-simplify (i*64)*64 -> i*4096; either way SLLI is used.)
    assert "S_SLLI_INT" in loop_body, "expected S_SLLI_INT for dynamic offset (i * power-of-2)"
    print(f"[ok] dynamic offset uses S_SLLI_INT (strength-reduced)")


def test_isa_uses_register_sourced_offset_in_dma():
    """The DMA's offset must be COPIED from a dynamic register, not loaded
    as a literal (`S_ADDI_INT gpX, gpY, 0` rather than `gpX, gp0, K`)."""
    ck = compile_kernel(loop_slice_dma, target=PlenaTarget(), name="loop_slice")
    asm = ck.isa_text
    # Find the slice comment marker; it should mention `parent_off=gpN dyn`.
    m = re.search(r"parent_off=gp(\d+) dyn", asm)
    assert m is not None, "expected 'parent_off=gpN dyn' comment for dynamic slice"
    off_reg = m.group(1)
    # And the emitter must do a register copy: `S_ADDI_INT gpX, gp{off_reg}, 0`.
    copy_pat = re.compile(rf"S_ADDI_INT gp\d+, gp{off_reg}, 0\b")
    assert copy_pat.search(asm), (
        f"expected register copy from gp{off_reg} (dynamic offset) into emitter scratch"
    )
    print(f"[ok] DMA reads dynamic offset from gp{off_reg} via S_ADDI_INT mov")


def test_isa_scale_is_parent_full_size_not_slice():
    """SCALE_REG should be parent's full element count, not the slice's."""
    ck = compile_kernel(loop_slice_dma, target=PlenaTarget(), name="loop_slice")
    asm = ck.isa_text
    parent_scale = SEQ_TOTAL * GROUP_HEADS * HLEN  # B=1 so just S*H*D = 16384
    assert re.search(
        rf"S_ADDI_INT gp\d+, gp0, {parent_scale}\s*\n\s*C_SET_SCALE_REG", asm
    ), f"expected SCALE_REG = parent_full_size = {parent_scale}"
    print(f"[ok] SCALE_REG <- parent full size {parent_scale}")


def test_isa_loop_increment_present():
    """idx register manually incremented before C_LOOP_END (loop machinery)."""
    ck = compile_kernel(loop_slice_dma, target=PlenaTarget(), name="loop_slice")
    asm = ck.isa_text
    m = re.search(r"-- hw counter gp(\d+), idx gp(\d+)", asm)
    hw_reg, idx_reg = m.group(1), m.group(2)
    inc_then_end = re.compile(
        rf"S_ADDI_INT gp{idx_reg}, gp{idx_reg}, 1\s*\n\s*C_LOOP_END gp{hw_reg}"
    )
    assert inc_then_end.search(asm)
    print(f"[ok] loop tail: gp{idx_reg} += 1 then C_LOOP_END gp{hw_reg}")


def main() -> int:
    tests = [
        test_hlir_records_for_then_slice,
        test_isa_emits_outer_loop,
        test_isa_strength_reduces_dynamic_offset,
        test_isa_uses_register_sourced_offset_in_dma,
        test_isa_scale_is_parent_full_size_not_slice,
        test_isa_loop_increment_present,
    ]
    print("=" * 60)
    print(f"loop_slice_dma structural tests ({len(tests)} cases)")
    print("=" * 60)
    for t in tests:
        t()
    print("=" * 60)
    print(f"ALL {len(tests)} TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
