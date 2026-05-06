"""Structural tests for the unified `emit_matmul_general` and `plena.matmul`
HLIR op (Phase 1 of the matmul rewrite).
"""

import re

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler.isa_emitter import ISAEmitter
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.program_shim import make_shim


def _shim():
    return make_shim(mlen=64, blen=4, btmm_lane_count=4, btmm_hlen=16)


def _emit_general(*, M_tiles, K_tiles, N):
    shim = _shim()
    emitter = ISAEmitter(shim)
    emitter.emit_matmul_general(
        M_tiles=M_tiles,
        K_tiles=K_tiles,
        N=N,
        lhs_vram_base=128,
        rhs_mram_base=4096,
        dst_vram_base=2048,
        task_id="t",
    )
    return shim.compiler.generated_code


def test_emit_matmul_general_single_tile():
    """N=mlen, M_tiles=K_tiles=1 collapses to one orow loop with one
    M_MM accumulation per iter."""
    asm = _emit_general(M_tiles=1, K_tiles=1, N=64)
    # tiles_per_n = 64/4 = 16 unrolled (m,oc) groups
    # per group: one orow hw loop containing one M_MM and one M_MM_WO.
    assert asm.count("M_MM ") == 16
    assert asm.count("M_MM_WO ") == 16
    # K_tiles=1 still emits a C_LOOP for K but with bound 1.
    assert re.search(r"C_LOOP_START gp\d+, 1\b", asm), asm
    # orow loop bound is mlen/blen = 16.
    assert re.search(r"C_LOOP_START gp\d+, 16\b", asm), asm


def test_emit_matmul_general_K_accumulates():
    """K_tiles=2 issues 2 M_MMs per output sub-tile then 1 M_MM_WO."""
    asm = _emit_general(M_tiles=1, K_tiles=2, N=64)
    # The K hw loop body is 1 M_MM, repeated K_tiles=2 dynamically.
    # Static count: still 16 M_MMs (one per (oc, orow) anchor) and 16 drains.
    assert asm.count("M_MM ") == 16, asm
    assert asm.count("M_MM_WO ") == 16, asm
    # K loop bound shows 2.
    assert re.search(r"C_LOOP_START gp\d+, 2\b", asm), asm


def test_emit_matmul_general_narrow_N():
    """N=hlen=16 -> tiles_per_n=4 unrolled groups."""
    asm = _emit_general(M_tiles=1, K_tiles=1, N=16)
    assert asm.count("M_MM ") == 4
    assert asm.count("M_MM_WO ") == 4


def test_emit_matmul_general_M_tiles_unroll():
    """M_tiles=2, N=mlen -> 2 * 16 = 32 unrolled groups."""
    asm = _emit_general(M_tiles=2, K_tiles=1, N=64)
    assert asm.count("M_MM ") == 32
    assert asm.count("M_MM_WO ") == 32


def test_emit_matmul_general_supports_N_larger_than_mlen():
    """N=128 = 2*mlen produces 2 N-mlen tile blocks, each contributing
    16 (oc) sub-tiles -> 32 anchors per M_tile."""
    asm = _emit_general(M_tiles=1, K_tiles=1, N=128)
    assert asm.count("M_MM ") == 32, asm
    assert asm.count("M_MM_WO ") == 32, asm


def test_emit_matmul_general_supports_N_partial_last_mlen_tile():
    """N=80 = 1*mlen + 16 -> 1 full mlen block (16 sub-tiles) +
    1 partial mlen block carrying hlen=16 valid cols (= 4 sub-tiles)."""
    asm = _emit_general(M_tiles=1, K_tiles=1, N=80)
    assert asm.count("M_MM ") == 16 + 4, asm
    assert asm.count("M_MM_WO ") == 16 + 4, asm


def test_emit_matmul_general_rejects_N_not_hlen_aligned():
    shim = _shim()
    emitter = ISAEmitter(shim)
    try:
        emitter.emit_matmul_general(
            M_tiles=1, K_tiles=1, N=20,   # not a multiple of hlen=16
            lhs_vram_base=0, rhs_mram_base=0, dst_vram_base=0,
        )
    except ValueError as exc:
        assert "divisible by hlen" in str(exc)
        return
    raise AssertionError("expected ValueError for non-hlen-aligned N")


def test_isa_pass_dispatches_matmul_op():
    """plena.matmul HLIR op routes through `_emit_matmul` and produces
    the same M_MM/M_MM_WO structure as a direct `emit_matmul_general` call."""
    shim = _shim()
    isa_pass = IsaEmitterPass(shim)
    mod = _hlir.HLIRModule(
        name="matmul_smoke",
        buffers={
            "A": _hlir.Buffer(name="A", scope="vram", shape=(64, 64), dtype="float16", address=128),
            "B": _hlir.Buffer(name="B", scope="mram", shape=(64, 64), dtype="float16", address=4096),
            "C": _hlir.Buffer(name="C", scope="vram", shape=(64, 64), dtype="float16", address=2048),
        },
        ops=[
            _hlir.Op(
                kind="matmul",
                buffer_args=["A", "B", "C"],
                # M_tiles, K_tiles, N, lhs_off, rhs_off, dst_off, dst_row_stride
                scalar_args=[1, 1, 64, 0, 0, 0, 0],
                annotations={"intrinsic": "plena.matmul"},
            ),
        ],
    )
    asm = isa_pass.run(mod)
    assert "MATMUL" not in asm  # MATMUL is the friendly intrinsic printer, not in the real ISA
    assert asm.count("M_MM ") == 16, asm
    assert asm.count("M_MM_WO ") == 16, asm


def test_codegen_handles_plena_matmul_call():
    """Build a tiny TIR PrimFunc with a `plena.matmul` extern call and
    drive it through the full pipeline (codegen -> address_alloc ->
    isa_pass). Verifies that codegen auto-handles the new intrinsic
    without any special-casing."""
    import tvm
    from tvm import tir
    from tilelang_tvm_compiler.codegen import PlenaCodegen
    from tilelang_tvm_compiler.address_alloc import (
        AddressAllocationPass, AddressAllocConfig,
    )

    extern_op = tvm.ir.Op.get("tir.call_extern")

    A_data = tir.Var("A", tvm.ir.PointerType(tvm.ir.PrimType("float16"), "vram"))
    B_data = tir.Var("B", tvm.ir.PointerType(tvm.ir.PrimType("float16"), "mram"))
    C_data = tir.Var("C", tvm.ir.PointerType(tvm.ir.PrimType("float16"), "vram"))
    A_buf = tir.decl_buffer(shape=(64, 64), dtype="float16", name="A", data=A_data, scope="vram")
    B_buf = tir.decl_buffer(shape=(64, 64), dtype="float16", name="B", data=B_data, scope="mram")
    C_buf = tir.decl_buffer(shape=(64, 64), dtype="float16", name="C", data=C_data, scope="vram")

    call = tir.Call(
        "handle", extern_op,
        [
            tir.StringImm("plena.matmul"),
            A_data, B_data, C_data,
            tir.IntImm("int32", 1),   # M_tiles
            tir.IntImm("int32", 1),   # K_tiles
            tir.IntImm("int32", 64),  # N
            tir.IntImm("int32", 0),   # lhs_offset
            tir.IntImm("int32", 0),   # rhs_offset
            tir.IntImm("int32", 0),   # dst_offset
            tir.IntImm("int32", 0),   # dst_row_stride (0 -> default = N)
        ],
    )
    body = tir.Block(
        iter_vars=[], reads=[], writes=[], name_hint="root",
        body=tir.Evaluate(call),
        alloc_buffers=[A_buf, B_buf, C_buf],
    )
    body = tir.BlockRealize(
        iter_values=[], predicate=tir.IntImm("bool", True), block=body,
    )
    func = tir.PrimFunc(params=[], body=body, ret_type=None, buffer_map={})

    cg = PlenaCodegen(func, name="cg_smoke")
    mod = cg.lower_to_hlir()
    assert any(op.kind == "matmul" for op in mod.ops), [op.kind for op in mod.ops]

    AddressAllocationPass(AddressAllocConfig(mlen=64, blen=4)).run(mod)

    shim = _shim()
    asm = IsaEmitterPass(shim).run(mod)
    assert asm.count("M_MM ") == 16, asm
    assert asm.count("M_MM_WO ") == 16, asm


if __name__ == "__main__":
    test_emit_matmul_general_single_tile()
    test_emit_matmul_general_K_accumulates()
    test_emit_matmul_general_narrow_N()
    test_emit_matmul_general_M_tiles_unroll()
    test_emit_matmul_general_supports_N_larger_than_mlen()
    test_emit_matmul_general_supports_N_partial_last_mlen_tile()
    test_emit_matmul_general_rejects_N_not_hlen_aligned()
    test_isa_pass_dispatches_matmul_op()
    test_codegen_handles_plena_matmul_call()
    print("all phase-1 matmul emitter tests passed")
