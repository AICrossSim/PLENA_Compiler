"""Structural tests for narrow M_MM emission (`mlen x mlen @ mlen x hlen`)."""

import re
import sys

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler.isa_emitter import ISAEmitter
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.program_shim import make_shim


def _emit_narrow(*, hlen=16, rhs_col_offset=0, dst_col_offset=0, zero_dst=False):
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    emitter = ISAEmitter(shim)
    emitter.emit_matmul_narrow_tile_hwloop(
        lhs_vram_addr=128,
        rhs_mram_addr=512,
        dst_vram_addr=1024,
        hlen=hlen,
        rhs_col_offset=rhs_col_offset,
        dst_col_offset=dst_col_offset,
        task_id="narrow_mm",
        zero_dst=zero_dst,
    )
    return shim.compiler.generated_code


def test_narrow_mm_emits_expected_column_count():
    asm = _emit_narrow(hlen=16)
    assert asm.count("M_MM ") == 16 // 4, asm
    assert asm.count("M_MM_WO ") == 16 // 4, asm
    print("[ok] narrow mm emits one M_MM/M_MM_WO pair per hlen/blen column block")


def test_narrow_mm_uses_full_row_hwloop():
    asm = _emit_narrow(hlen=16)
    assert "C_LOOP_START" in asm
    assert re.search(r"C_LOOP_START gp\d+, 16\b", asm), asm
    print("[ok] narrow mm keeps the full mlen/blen row sweep in hardware loop form")


def test_narrow_mm_respects_slot_offsets():
    asm = _emit_narrow(hlen=16, rhs_col_offset=32, dst_col_offset=48)
    assert "S_ADDI_INT gp" in asm
    assert re.search(r"S_ADDI_INT gp\d+, gp0, 544\b", asm), asm
    assert re.search(r"S_ADDI_INT gp\d+, gp0, 1072\b", asm), asm
    print("[ok] narrow mm biases rhs/dst bases by explicit slot offsets")


def test_narrow_mm_uses_narrow_row_stride_by_default():
    asm = _emit_narrow(hlen=16)
    assert re.search(r"S_ADDI_INT gp\d+, gp\d+, 64\b", asm), asm
    print("[ok] narrow mm advances dst rows by blen*hlen for standalone narrow tiles")


def test_narrow_mm_can_zero_dst():
    asm = _emit_narrow(hlen=16, zero_dst=True)
    assert "; zero tile vram[1024]" in asm, asm
    print("[ok] narrow mm can optionally zero the destination backing tile first")


def test_narrow_mm_rejects_unaligned_hlen():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    emitter = ISAEmitter(shim)
    try:
        emitter.emit_matmul_narrow_tile_hwloop(
            lhs_vram_addr=0,
            rhs_mram_addr=0,
            dst_vram_addr=0,
            hlen=10,
        )
    except ValueError as exc:
        assert "divisible by blen" in str(exc)
        print("[ok] narrow mm rejects hlen values that are not blen-aligned")
        return
    raise AssertionError("expected ValueError for hlen=10")


def test_mm_lowering_routes_narrow_shapes_to_narrow_emitter():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    emitter_pass = IsaEmitterPass(shim)
    mod = _hlir.HLIRModule(
        name="narrow_mm",
        buffers={
            "lhs": _hlir.Buffer(name="lhs", scope="vram", shape=(64, 64), dtype="float16", address=128),
            "rhs": _hlir.Buffer(name="rhs", scope="mram", shape=(64, 16), dtype="float16", address=512),
            "dst": _hlir.Buffer(name="dst", scope="vram", shape=(64, 16), dtype="float16", address=1024),
        },
        ops=[],
    )
    op = _hlir.Op(kind="mm", buffer_args=["lhs", "rhs", "dst"], annotations={"intrinsic": "plena.mm"})
    emitter_pass._emit_mm(mod, op)
    asm = shim.compiler.generated_code
    assert "; narrow matmul task plena.mm" in asm, asm
    assert re.search(r"S_ADDI_INT gp\d+, gp\d+, 64\b", asm), asm
    print("[ok] plena.mm lowering routes 64x16 rhs/dst tiles to the narrow emitter")


def test_mm_slot_lowering_targets_packed_slots():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    emitter_pass = IsaEmitterPass(shim)
    mod = _hlir.HLIRModule(
        name="mm_slot",
        buffers={
            "lhs": _hlir.Buffer(name="lhs", scope="vram", shape=(64, 64), dtype="float16", address=128),
            "rhs": _hlir.Buffer(name="rhs", scope="mram", shape=(1, 64, 4, 16), dtype="float16", address=512),
            "dst": _hlir.Buffer(name="dst", scope="vram", shape=(1, 64, 4, 16), dtype="float16", address=1024),
        },
        ops=[],
    )
    op = _hlir.Op(
        kind="mm_slot",
        buffer_args=["lhs", "rhs", "dst"],
        scalar_args=[0, 16, 16, 16],   # lhs_row_offset, rhs_col_offset, dst_col_offset, col_count
        annotations={"intrinsic": "plena.mm_slot"},
    )
    emitter_pass._emit_mm_slot(mod, op)
    asm = shim.compiler.generated_code
    assert "; slot matmul task plena.mm_slot" in asm, asm
    assert re.search(r"S_ADDI_INT gp\d+, gp0, 528\b", asm), asm
    assert re.search(r"S_ADDI_INT gp\d+, gp0, 1040\b", asm), asm
    print("[ok] plena.mm_slot lowering emits packed-slot matmul with explicit column offsets")


def test_grouped_narrow_v2h_slice_writes_back_as_single_tile():
    shim = make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)
    emitter_pass = IsaEmitterPass(shim)
    parent = _hlir.Buffer(
        name="C_hbm",
        scope="hbm",
        shape=(1, 128, 4, 16),
        dtype="float16",
        address=0,
        hbm_stride=64,
        hbm_scale_size=8192,
    )
    src = _hlir.Buffer(
        name="C_v",
        scope="vram",
        shape=(1, 64, 4, 16),
        dtype="float16",
        address=4096,
    )
    mod = _hlir.HLIRModule(
        name="grouped_narrow_v2h",
        buffers={"C_hbm": parent, "C_v": src},
        ops=[],
    )
    op = _hlir.Op(
        kind="dma_v2h_slice",
        buffer_args=[
            "C_v",
            _hlir.BufferSlice(
                parent="C_hbm",
                starts=(0, 0, 0, 0),
                extents=(1, 64, 4, 16),
            ),
        ],
        annotations={"intrinsic": "plena.dma_v2h_slice"},
    )
    emitter_pass._emit_dma_v2h_slice(mod, op)
    asm = shim.compiler.generated_code
    assert "grouped narrow writeback as one logical mlen*mlen tile" in asm, asm
    assert "; ... tile h=" not in asm, asm
    print("[ok] grouped narrow v2h_slice writes back one packed 64x64 tile")


def main():
    tests = [
        test_narrow_mm_emits_expected_column_count,
        test_narrow_mm_uses_full_row_hwloop,
        test_narrow_mm_respects_slot_offsets,
        test_narrow_mm_uses_narrow_row_stride_by_default,
        test_narrow_mm_can_zero_dst,
        test_narrow_mm_rejects_unaligned_hlen,
        test_mm_lowering_routes_narrow_shapes_to_narrow_emitter,
        test_mm_slot_lowering_targets_packed_slots,
        test_grouped_narrow_v2h_slice_writes_back_as_single_tile,
    ]
    print("=" * 60)
    print(f"narrow mm emitter tests ({len(tests)} cases)")
    print("=" * 60)
    for test in tests:
        test()
    print("=" * 60)
    print(f"ALL {len(tests)} TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
