"""Structural tests for tiled_btmm: validates Phase 8 multi-tile slice
writeback (per-head non-contiguous in 2D)."""

import re
import sys

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler.kernels.tiled_btmm import make_tiled_btmm
from tilelang_tvm_compiler.pipeline import compile_kernel, PlenaTarget


def _compile(seq_q=128, seq_k=128):
    fn, c = make_tiled_btmm(seq_q=seq_q, seq_k=seq_k)
    ck = compile_kernel(fn, target=PlenaTarget(), name="tiled_btmm")
    return ck, c


def test_kernel_has_nested_for_loops():
    """3-level nesting: q_block -> hg (head_group) -> kv_block."""
    ck, c = _compile()
    ops = ck.hlir.ops
    assert len(ops) == 1 and ops[0].kind == "for", "outer must be a for"
    hg_ops = ops[0].body
    assert len(hg_ops) == 1 and hg_ops[0].kind == "for", "second level must be a for (head_group)"
    kv_ops = hg_ops[0].body
    assert len(kv_ops) == 1 and kv_ops[0].kind == "for", "third level must be a for (kv_block)"
    inner_body = kv_ops[0].body
    kinds = [op.kind for op in inner_body]
    assert kinds == ["dma_h2v_slice", "dma_h2m_slice", "btmm", "dma_v2h_slice"], (
        f"unexpected inner-body kinds: {kinds}"
    )
    print(f"[ok] nested for: q_block -> hg -> kv_block -> [4 inner ops]")


def test_v2h_slice_emits_per_head_tile_comments():
    """The Phase 8 multi-tile dispatcher should emit `; ... tile h=K`
    comment markers for each of LANE_COUNT tiles per BTMM (= one BTMM
    body emits LANE_COUNT writeback tiles, regardless of total head_count)."""
    ck, c = _compile()
    asm = ck.isa_text
    tile_markers = re.findall(r"; \.\.\. tile h=(\d+)", asm)
    # The body emits 4 per-head tiles; ASM is shared across the loop
    # (hardware loop runs body N_q*N_k times), so we expect exactly 4.
    assert len(tile_markers) == c["LANE_COUNT"], (
        f"expected {c['LANE_COUNT']} per-head tile markers, got {len(tile_markers)}"
    )
    assert tile_markers == [str(i) for i in range(c["LANE_COUNT"])]
    print(f"[ok] v2h_slice emits {c['LANE_COUNT']} per-head tiles in order")


def test_v2h_slice_tile_const_offsets_match_per_head_layout():
    """Per-head tile h has hbm offset = base + h*D where D = SEQ_K."""
    ck, c = _compile(seq_q=128, seq_k=128)
    asm = ck.isa_text
    SEQ_K = c["SEQ_K"]
    # For SEQ_K=128, per-head offsets are 0, 128, 256, 384
    expected_offsets = [h * SEQ_K for h in range(c["LANE_COUNT"])]
    actual_offsets = [int(m) for m in re.findall(r"hbm\[base\+(\d+)\]", asm)]
    assert actual_offsets == expected_offsets, (
        f"per-head hbm offsets: expected {expected_offsets}, got {actual_offsets}"
    )
    print(f"[ok] per-head offsets: {actual_offsets} (each = h * SEQ_K = h * {SEQ_K})")


def test_v2h_slice_vram_offsets_are_head_major():
    """Per-head tile h reads from vram_off = h * tile_elems = h * MLEN^2."""
    ck, c = _compile()
    asm = ck.isa_text
    expected_vram = [h * c["MLEN"] * c["MLEN"] for h in range(c["LANE_COUNT"])]
    actual_vram = [int(m) for m in re.findall(r"vram\[\+(\d+)\]", asm)]
    assert actual_vram == expected_vram, (
        f"per-head vram offsets: expected {expected_vram}, got {actual_vram}"
    )
    print(f"[ok] per-head vram offsets: {actual_vram} (head-major BHSD)")


def test_dma_v2h_uses_dynamic_base_reg():
    """The slice base offset depends on q_block and kv_block (loop vars),
    so it must be computed into a register and the per-tile DMAs must
    reuse that register (with optional + tile_const adds)."""
    ck, _ = _compile()
    asm = ck.isa_text
    m = re.search(r"dynamic base gp(\d+)", asm)
    assert m is not None, "expected '; ... dynamic base gpN' marker"
    base_reg = m.group(1)
    # And we should see at least 3 `S_ADDI_INT gp_X, gp_base, K` lines for
    # h=1,2,3 (h=0 reuses base directly so no extra ADDI on it).
    extra_adds = re.findall(rf"S_ADDI_INT gp\d+, gp{base_reg}, \d+\b", asm)
    assert len(extra_adds) >= 3, (
        f"expected >=3 `S_ADDI_INT _, gp{base_reg}, K` for per-head offsets, "
        f"got {len(extra_adds)}"
    )
    print(f"[ok] dynamic base gp{base_reg} reused across {len(extra_adds)} per-head adds")


def test_scale_is_parent_full_size():
    ck, c = _compile()
    asm = ck.isa_text
    # Parent C_hbm 2D collapse uses head_count, not lane_count:
    # cols = HEAD_COUNT * SEQ_K, rows = BATCH * SEQ_Q.
    parent_full = c["BATCH"] * c["SEQ_Q"] * c["HEAD_COUNT"] * c["SEQ_K"]
    assert re.search(
        rf"S_ADDI_INT gp\d+, gp0, {parent_full}\s*\n\s*C_SET_SCALE_REG", asm
    ), f"expected SCALE_REG = {parent_full} (parent full element count)"
    print(f"[ok] SCALE_REG <- {parent_full} (parent full size)")


def test_stride_is_parent_row_width():
    ck, c = _compile()
    asm = ck.isa_text
    parent_stride = c["HEAD_COUNT"] * c["SEQ_K"]
    assert re.search(
        rf"S_ADDI_INT gp\d+, gp0, {parent_stride}\s*\n\s*C_SET_STRIDE_REG", asm
    )
    print(f"[ok] STRIDE_REG <- {parent_stride} (parent row width = HEAD_COUNT*SEQ_K)")


def test_kernel_has_btmm_pair():
    ck, _ = _compile()
    asm = ck.isa_text
    assert asm.count("M_BTMM ") == 1
    assert asm.count("M_BMM_WO ") == 1
    print(f"[ok] M_BTMM + M_BMM_WO emitted exactly once each (inside loop body)")


def main():
    tests = [
        test_kernel_has_nested_for_loops,
        test_v2h_slice_emits_per_head_tile_comments,
        test_v2h_slice_tile_const_offsets_match_per_head_layout,
        test_v2h_slice_vram_offsets_are_head_major,
        test_dma_v2h_uses_dynamic_base_reg,
        test_scale_is_parent_full_size,
        test_stride_is_parent_row_width,
        test_kernel_has_btmm_pair,
    ]
    print("=" * 60)
    print(f"tiled_btmm structural tests ({len(tests)} cases)")
    print("=" * 60)
    for t in tests:
        t()
    print("=" * 60)
    print(f"ALL {len(tests)} TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
