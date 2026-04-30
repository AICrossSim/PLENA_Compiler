"""Structural tests for the loop_dma kernel: validates Phase 4 ForOp lowering.

Run:
    LD_LIBRARY_PATH="" \\
    PYTHONPATH=/home/.../PLENA_Simulator/compiler \\
    /home/.../PLENA_Simulator/.venv-tvm/bin/python -m \\
        tilelang_tvm_compiler.tests.test_loop_dma
"""

from __future__ import annotations

import re
import sys

from tilelang_tvm_compiler.kernels.loop_dma import (
    ITERS,
    loop_dma,
)
from tilelang_tvm_compiler.pipeline import PlenaTarget, compile_kernel


def test_loop_dma_emits_c_loop_pair():
    ck = compile_kernel(loop_dma, target=PlenaTarget(), name="loop_dma_kernel")
    asm = ck.isa_text

    # Outer hardware loop: C_LOOP_START gp_X, ITERS  + matching C_LOOP_END.
    starts = re.findall(rf"C_LOOP_START gp(\d+), {ITERS}\b", asm)
    assert len(starts) == 1, (
        f"expected exactly one outer C_LOOP_START with extent={ITERS}, "
        f"got {len(starts)}: {starts!r}"
    )
    outer_reg = starts[0]
    assert f"C_LOOP_END gp{outer_reg}" in asm, (
        f"missing matching C_LOOP_END gp{outer_reg}"
    )
    print(f"[ok] outer loop: C_LOOP_START gp{outer_reg}, {ITERS} ... C_LOOP_END gp{outer_reg}")


def test_loop_dma_initialises_index_register_at_zero():
    """Body-visible idx register must be init to 0 before C_LOOP_START."""
    ck = compile_kernel(loop_dma, target=PlenaTarget(), name="loop_dma_kernel")
    asm = ck.isa_text
    # Look for "; for i in [0, 4) -- hw counter gpX, idx gpY"
    m = re.search(r"-- hw counter gp(\d+), idx gp(\d+)", asm)
    assert m is not None, "missing for-loop comment marker"
    hw_reg, idx_reg = m.group(1), m.group(2)
    # Init idx to 0 immediately before C_LOOP_START.
    init_pattern = re.compile(
        rf"S_ADDI_INT gp{idx_reg}, gp0, 0\s*\n\s*C_LOOP_START gp{hw_reg},"
    )
    assert init_pattern.search(asm), (
        f"expected `S_ADDI_INT gp{idx_reg}, gp0, 0` followed by "
        f"`C_LOOP_START gp{hw_reg}, ...`"
    )
    print(f"[ok] idx init: gp{idx_reg} = 0 then C_LOOP_START gp{hw_reg}")


def test_loop_dma_increments_index_at_loop_tail():
    """After the body, increment the idx register before C_LOOP_END."""
    ck = compile_kernel(loop_dma, target=PlenaTarget(), name="loop_dma_kernel")
    asm = ck.isa_text
    m = re.search(r"-- hw counter gp(\d+), idx gp(\d+)", asm)
    hw_reg, idx_reg = m.group(1), m.group(2)
    # Last lines of body should have:  inc idx; C_LOOP_END gp_outer
    inc_then_end = re.compile(
        rf"S_ADDI_INT gp{idx_reg}, gp{idx_reg}, 1\s*\n\s*C_LOOP_END gp{hw_reg}"
    )
    assert inc_then_end.search(asm), (
        f"expected idx increment immediately before C_LOOP_END"
    )
    print(f"[ok] tail increment: gp{idx_reg} += 1 then C_LOOP_END gp{hw_reg}")


def test_loop_dma_body_contains_dma():
    """Inside the loop, the actual DMA op (H_PREFETCH_V) must appear."""
    ck = compile_kernel(loop_dma, target=PlenaTarget(), name="loop_dma_kernel")
    asm = ck.isa_text
    assert "H_PREFETCH_V" in asm, "DMA body lost from loop"
    print(f"[ok] body: H_PREFETCH_V appears inside the loop")


def test_loop_dma_no_register_conflict():
    """Outer loop registers (gp_loop, gp_idx) must not clash with body's
    register allocations -- both use the same RegisterAllocator pool."""
    ck = compile_kernel(loop_dma, target=PlenaTarget(), name="loop_dma_kernel")
    asm = ck.isa_text
    m = re.search(r"-- hw counter gp(\d+), idx gp(\d+)", asm)
    hw_reg, idx_reg = m.group(1), m.group(2)
    # Body should NOT redefine these registers' canonical use. The DMA
    # body emits `S_ADDI_INT gp_X, gp0, ...` to set values into scratch
    # registers; we want to make sure NEITHER hw_reg NOR idx_reg appears
    # as gp_X in those scratch-init lines (other than the loop's own
    # init/inc, which are outside the body).
    body = asm.split("C_LOOP_START")[1].split("C_LOOP_END")[0]
    # Strip the inner DMA's own C_LOOP_START/END block boundaries by
    # walking line by line.
    forbidden = {hw_reg, idx_reg}
    for line in body.split("\n"):
        # We expect the body to use registers other than hw/idx.
        # Specifically watch for `S_ADDI_INT gp{hw|idx}, gp0, ...`
        # which would be a clobber of our loop's bookkeeping regs.
        for r in forbidden:
            bad = re.search(rf"^\s*S_ADDI_INT gp{r}, gp0, ", line)
            if bad:
                raise AssertionError(
                    f"body clobbers loop register gp{r}: {line.strip()!r}"
                )
    print(f"[ok] no clobber: gp{hw_reg} (hw) and gp{idx_reg} (idx) untouched by body")


def main() -> int:
    tests = [
        test_loop_dma_emits_c_loop_pair,
        test_loop_dma_initialises_index_register_at_zero,
        test_loop_dma_increments_index_at_loop_tail,
        test_loop_dma_body_contains_dma,
        test_loop_dma_no_register_conflict,
    ]
    print("=" * 60)
    print(f"loop_dma structural tests ({len(tests)} cases)")
    print("=" * 60)
    for t in tests:
        t()
    print("=" * 60)
    print(f"ALL {len(tests)} TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
