"""Unit tests for PlenaCompiler ATen path (no emulator needed).

Run: PYTHONPATH=.:tools:compiler python3 compiler/aten/tests/test_plena_compiler.py
"""

import sys
import os

# Insert PLENA_Simulator root and tools/ so imports resolve correctly regardless
# of how the test is invoked (direct python3 or via PYTHONPATH=.:tools:compiler).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))))
_SIM_ROOT = os.path.join(_SIM_ROOT, "PLENA_Simulator")
# Fallback: if the above doesn't exist (different layout), walk up to find it
if not os.path.isdir(os.path.join(_SIM_ROOT, "compiler")):
    _SIM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
for _p in [_SIM_ROOT, os.path.join(_SIM_ROOT, "tools"), os.path.join(_SIM_ROOT, "compiler")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch


def test_vram_fill_zero_all_column_blocks():
    """vram_fill_zero must zero ALL column blocks of a wide matrix."""
    from compiler.aten.plena_compiler import PlenaCompiler

    prog = PlenaCompiler()
    x = prog.alloc("X", 64, 384)
    prog.vram_fill_zero(x)
    code = prog.get_code()

    # 384/64 = 6 column blocks, each needs a V_MUL_VF zeroing loop
    assert code.count("V_MUL_VF") >= 6, (
        f"Expected >= 6 V_MUL_VF (one per column block), got {code.count('V_MUL_VF')}"
    )
    print("  PASS test_vram_fill_zero_all_column_blocks")


def test_vram_add_all_column_blocks():
    """vram_add must add ALL column blocks of wide matrices."""
    from compiler.aten.plena_compiler import PlenaCompiler

    prog = PlenaCompiler()
    x = prog.alloc("X", 64, 384)
    y = prog.alloc("Y", 64, 384)
    prog.vram_add(x, y)
    code = prog.get_code()

    # 6 column blocks = 6 V_ADD_VV loops
    assert code.count("V_ADD_VV") >= 6, (
        f"Expected >= 6 V_ADD_VV (one per column block), got {code.count('V_ADD_VV')}"
    )
    print("  PASS test_vram_add_all_column_blocks")


def test_alloc_at_correct_address():
    """alloc_at must create a VRAM view at the specified address."""
    from compiler.aten.plena_compiler import PlenaCompiler

    prog = PlenaCompiler()
    # Allocate a matrix, then create a view into its second column block
    x = prog.alloc("X", 64, 384)
    x_addr = prog.get_vram_addr(x.name)

    # View of column block 2 (offset = 2 * 64 * 64 = 8192)
    view_addr = x_addr + 2 * 64 * 64
    view = prog.alloc_at("X_cb2_view", 64, 64, view_addr)

    actual_addr = prog._compiler.get_vram_addr(view.name)
    assert actual_addr == view_addr, (
        f"alloc_at address mismatch: expected {view_addr}, got {actual_addr}"
    )
    print("  PASS test_alloc_at_correct_address")


def test_fix_large_immediates_roundtrip():
    """_fix_large_immediates must preserve exact address values."""
    from compiler.aten.plena_frontend import _fix_large_immediates

    # Test addresses that need S_LUI_INT conversion
    test_values = [0, 4096, 262143, 262144, 290880, 315456, 1000000]

    for val in test_values:
        asm = f"S_ADDI_INT gp5, gp0, {val}\n"
        fixed = _fix_large_immediates(asm)

        if val < (1 << 18):
            # Should remain as S_ADDI_INT
            assert f"S_ADDI_INT gp5, gp0, {val}" in fixed, (
                f"Small value {val} was incorrectly converted"
            )
        else:
            # Should be S_LUI_INT + optional S_ADDI_INT
            upper = val >> 12
            lower = val & 0xFFF
            assert f"S_LUI_INT gp5, {upper}" in fixed, (
                f"Large value {val}: missing S_LUI_INT gp5, {upper}"
            )
            # Verify roundtrip: upper << 12 + lower == val
            reconstructed = (upper << 12) + lower
            assert reconstructed == val, (
                f"Address roundtrip failed: {val} -> upper={upper}, lower={lower} -> {reconstructed}"
            )

    print("  PASS test_fix_large_immediates_roundtrip")


def test_fix_large_immediates_preserves_relative_adds():
    """_fix_large_immediates must NOT convert relative S_ADDI_INT (non-gp0 source)."""
    from compiler.aten.plena_frontend import _fix_large_immediates

    # Relative add: gp5 = gp3 + 300000 — should NOT be converted
    asm = "S_ADDI_INT gp5, gp3, 300000\n"
    fixed = _fix_large_immediates(asm)
    assert "S_ADDI_INT gp5, gp3, 300000" in fixed, (
        "Relative S_ADDI_INT was incorrectly converted"
    )

    # Absolute load: gp5 = gp0 + 300000 — SHOULD be converted
    asm2 = "S_ADDI_INT gp5, gp0, 300000\n"
    fixed2 = _fix_large_immediates(asm2)
    assert "S_LUI_INT" in fixed2, (
        "Absolute S_ADDI_INT with large value was not converted"
    )

    print("  PASS test_fix_large_immediates_preserves_relative_adds")


def test_rotate_half_matrix_identity():
    """R_rope @ R_rope should be -I (rotate_half applied twice negates)."""
    from compiler.aten.plena_frontend import _make_rotate_half_matrix

    R = _make_rotate_half_matrix(64)
    RR = R @ R
    expected = -torch.eye(64)
    assert torch.allclose(RR, expected, atol=1e-6), (
        f"R @ R should be -I, got max diff {(RR - expected).abs().max()}"
    )
    print("  PASS test_rotate_half_matrix_identity")


def test_compile_hf_model_golden_vs_hf():
    """Golden (MXFP8+BF16) should closely match HF float32 at native dims."""
    from compiler.aten.plena_frontend import compile_hf_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "AICrossSim/clm-60m", torch_dtype=torch.float32
    )
    model.eval()

    r = compile_hf_model(model, seq_len=64, hidden_size=None, inter_dim=None, num_layers=1)
    golden = r["golden_output"]
    hf = r["hf_ground_truth"]

    diff = (golden - hf).abs()
    pct = (diff <= 0.2 + 0.2 * hf.abs()).float().mean() * 100
    cos = torch.nn.functional.cosine_similarity(
        golden.flatten().unsqueeze(0), hf.flatten().unsqueeze(0)
    )

    assert pct >= 95.0, f"Golden vs HF allclose {pct:.1f}% < 95%"
    assert cos.item() >= 0.99, f"Golden vs HF cosine {cos.item():.4f} < 0.99"
    print(f"  PASS test_compile_hf_model_golden_vs_hf ({pct:.1f}% allclose, cos={cos.item():.4f})")


def test_native_compile_assembles():
    """Native-dim ISA must assemble without overflow."""
    from compiler.aten.plena_frontend import compile_hf_model, _fix_large_immediates
    from transformers import AutoModelForCausalLM
    import tempfile, os

    model = AutoModelForCausalLM.from_pretrained(
        "AICrossSim/clm-60m", torch_dtype=torch.float32
    )
    model.eval()

    r = compile_hf_model(model, seq_len=64, hidden_size=None, inter_dim=None, num_layers=1)
    isa = r["isa"]

    # Assemble — should not raise ValueError (u32 overflow)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from assembler import AssemblyToBinary

    compiler_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    asm_tool = AssemblyToBinary(
        os.path.join(compiler_root, "doc", "operation.svh"),
        os.path.join(compiler_root, "doc", "precision.svh"),
    )

    with tempfile.NamedTemporaryFile(suffix=".asm", mode="w", delete=False) as f:
        f.write(isa)
        asm_path = f.name
    mem_path = asm_path.replace(".asm", ".mem")

    try:
        asm_tool.generate_binary(asm_path, mem_path)
        mem_bytes = os.path.getsize(mem_path)
        assert mem_bytes > 0, "Assembled .mem is empty"
        print(f"  PASS test_native_compile_assembles ({mem_bytes} bytes)")
    except ValueError as exc:
        raise AssertionError(f"Assembly overflow: {exc}") from exc
    finally:
        os.unlink(asm_path)
        if os.path.exists(mem_path):
            os.unlink(mem_path)


if __name__ == "__main__":
    print("=" * 60)
    print("PlenaCompiler ATen path unit tests")
    print("=" * 60)

    tests = [
        test_vram_fill_zero_all_column_blocks,
        test_vram_add_all_column_blocks,
        test_alloc_at_correct_address,
        test_fix_large_immediates_roundtrip,
        test_fix_large_immediates_preserves_relative_adds,
        test_rotate_half_matrix_identity,
        test_compile_hf_model_golden_vs_hf,
        test_native_compile_assembles,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL {test_fn.__name__}: {e}")
            failed += 1

    print("=" * 60)
    print(f"{passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
