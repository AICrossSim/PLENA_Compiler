"""Structural tests for static_slice_dma: validates Phase 6 BufferSlice
with all-static slice starts.

Run:
    LD_LIBRARY_PATH="" \\
    PYTHONPATH=/.../compiler \\
    .venv-tvm/bin/python -m tilelang_tvm_compiler.tests.test_static_slice
"""

from __future__ import annotations

import re
import sys

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler.kernels.static_slice_dma import (
    BATCH,
    GROUP_HEADS,
    HLEN,
    MLEN,
    SEQ_TOTAL,
    SLICE_EXTENT,
    SLICE_START,
    static_slice_dma,
)
from tilelang_tvm_compiler.pipeline import PlenaTarget, compile_kernel


def test_hlir_carries_buffer_slice():
    """Pass 1 should pack starts/extents into a BufferSlice attached to the
    sliced DMA op."""
    ck = compile_kernel(static_slice_dma, target=PlenaTarget(), name="static_slice")
    ops = ck.hlir.ops
    assert len(ops) == 1, f"expected one op, got {len(ops)}"
    op = ops[0]
    assert op.kind == "dma_h2v_slice"
    sl = op.buffer_args[0]
    assert isinstance(sl, _hlir.BufferSlice)
    assert sl.parent == "A_hbm"
    assert sl.starts == (0, SLICE_START, 0, 0)
    assert sl.extents == (BATCH, SLICE_EXTENT, GROUP_HEADS, HLEN)
    print(f"[ok] HLIR slice: parent={sl.parent}  starts={sl.starts}  ext={sl.extents}")


def test_isa_loads_correct_offset():
    """The hbm_start_offset must equal slice_start * (group_heads*hlen)
    in elements."""
    ck = compile_kernel(static_slice_dma, target=PlenaTarget(), name="static_slice")
    asm = ck.isa_text
    # row_start in 2D logical = batch*seq_total + slice_start (with batch=0)
    # since we do H*D merge, the offset in elements = row_start * cols = slice_start * (H*D)
    expected_off = SLICE_START * (GROUP_HEADS * HLEN)
    assert f"parent_off={expected_off} elems" in asm, (
        f"expected slice comment to mention parent_off={expected_off}"
    )
    # And the literal must be loaded into a register before the prefetch.
    assert re.search(rf"S_ADDI_INT gp\d+, gp0, {expected_off}\b", asm), (
        f"expected `S_ADDI_INT gpX, gp0, {expected_off}` (offset literal)"
    )
    print(f"[ok] hbm_start_offset = {expected_off} (= {SLICE_START} * {GROUP_HEADS*HLEN})")


def test_isa_uses_parent_scale_not_slice_scale():
    """SCALE_REG must be set to the PARENT's full-tensor element count
    (B*S * H*D), not just the slice's."""
    ck = compile_kernel(static_slice_dma, target=PlenaTarget(), name="static_slice")
    asm = ck.isa_text
    parent_scale = BATCH * SEQ_TOTAL * GROUP_HEADS * HLEN  # = 8192 for our shapes
    # The HLIR dump records it cleanly; sanity-check via the HLIR module.
    parent = ck.hlir.get_buffer("A_hbm")
    assert parent.hbm_scale_size == parent_scale, (
        f"HLIR parent.hbm_scale_size={parent.hbm_scale_size}, want {parent_scale}"
    )
    # And the value must be loaded for C_SET_SCALE_REG.
    assert re.search(
        rf"S_ADDI_INT gp\d+, gp0, {parent_scale}\s*\n\s*C_SET_SCALE_REG", asm
    ), f"expected `S_ADDI_INT ... {parent_scale}` then `C_SET_SCALE_REG`"
    print(f"[ok] SCALE_REG <- parent_full_size {parent_scale}")


def test_isa_uses_parent_stride():
    """STRIDE_REG must be the parent's row width (H*D), not anything
    derived from the slice."""
    ck = compile_kernel(static_slice_dma, target=PlenaTarget(), name="static_slice")
    asm = ck.isa_text
    parent_stride = GROUP_HEADS * HLEN  # = 64
    assert re.search(
        rf"S_ADDI_INT gp\d+, gp0, {parent_stride}\s*\n\s*C_SET_STRIDE_REG", asm
    )
    print(f"[ok] STRIDE_REG <- parent_stride {parent_stride}")


def test_isa_calls_h_prefetch_v():
    """The actual DMA instruction is H_PREFETCH_V."""
    ck = compile_kernel(static_slice_dma, target=PlenaTarget(), name="static_slice")
    asm = ck.isa_text
    assert "H_PREFETCH_V" in asm
    print(f"[ok] H_PREFETCH_V emitted")


def main() -> int:
    tests = [
        test_hlir_carries_buffer_slice,
        test_isa_loads_correct_offset,
        test_isa_uses_parent_scale_not_slice_scale,
        test_isa_uses_parent_stride,
        test_isa_calls_h_prefetch_v,
    ]
    print("=" * 60)
    print(f"static_slice_dma structural tests ({len(tests)} cases)")
    print("=" * 60)
    for t in tests:
        t()
    print("=" * 60)
    print(f"ALL {len(tests)} TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
