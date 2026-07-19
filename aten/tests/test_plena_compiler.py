"""Unit tests for PlenaCompiler ATen path (no emulator needed).

Run: PYTHONPATH=.:tools:.. python3 aten/tests/test_plena_compiler.py
"""

import sys
import os
import re

# Insert PLENA_Simulator root and tools/ so imports resolve correctly regardless
# of how the test is invoked (direct python3 or via PYTHONPATH=.:tools:..).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SIM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))))
_SIM_ROOT = os.path.join(_SIM_ROOT, "PLENA_Simulator")
# Fallback: if the above doesn't exist (different layout), walk up to find it
if not os.path.isdir(os.path.join(_SIM_ROOT, "compiler")):
    _SIM_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
for _p in [_SIM_ROOT, os.path.join(_SIM_ROOT, "tools"), os.path.join(_SIM_ROOT, "compiler")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402


def test_isa_builder_renders_typed_instruction():
    """Typed ISA builder should render the existing asm syntax."""
    from compiler.aten.isa_builder import IsaBuilder, fp, gp

    asm = IsaBuilder()
    asm.comment("typed builder smoke")
    asm.instr("S_ADDI_INT", gp(3), gp(0), 4096)
    asm.instr("S_ADD_FP", fp(1), fp(1), fp(2))

    rendered = asm.render()
    assert "; typed builder smoke" in rendered
    assert "S_ADDI_INT gp3, gp0, 4096" in rendered
    assert "S_ADD_FP f1, f1, f2" in rendered
    print("  PASS test_isa_builder_renders_typed_instruction")


def test_isa_builder_legalizes_large_absolute_immediates():
    """Typed ISA builder should split large absolute S_ADDI_INT loads."""
    from compiler.aten.isa_builder import IsaBuilder, gp

    rendered = IsaBuilder().instr("S_ADDI_INT", gp(5), gp(0), 300000).render()
    assert "S_LUI_INT gp5, 73" in rendered
    assert "S_ADDI_INT gp5, gp5, 992" in rendered
    assert "S_ADDI_INT gp5, gp0, 300000" not in rendered
    print("  PASS test_isa_builder_legalizes_large_absolute_immediates")


def test_isa_builder_legalizes_relative_large_immediates():
    """Typed ISA builder should split relative S_ADDI_INT instructions safely."""
    from compiler.aten.isa_builder import IsaBuilder, gp

    rendered = IsaBuilder().instr("S_ADDI_INT", gp(5), gp(3), 300000).render()
    assert "S_ADDI_INT gp5, gp3, 262143" in rendered
    assert "S_ADDI_INT gp5, gp5, 37857" in rendered
    assert "S_ADDI_INT gp5, gp3, 300000" not in rendered
    assert "S_LUI_INT" not in rendered
    print("  PASS test_isa_builder_legalizes_relative_large_immediates")


def test_fpvar_helper_uses_canonical_emit_path():
    """Converted FPVar helpers should still append and return asm text."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler()
    code = prog.fpvar_add_asm(src1_addr=0, src2_addr=4, dst_addr=8, count=2)

    assert code == prog.get_code()
    assert "S_ADD_FP f1, f1, f2" in code
    assert "C_LOOP_START" in code
    print("  PASS test_fpvar_helper_uses_canonical_emit_path")


def test_tile_row_minmax_fp_helpers_emit_vector_scalar_clamp_ops():
    """GPT-OSS clamp helpers should lower to V_MIN_VF/V_MAX_VF."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler(mlen=8, blen=2)
    x = prog.alloc("X", 8, 8)
    limit = prog.fp_var("limit", size=1)

    prog.tile_row_min_fp(x, limit.address, rows=[0, 1])
    prog.tile_row_max_fp(x, limit.address, rows=[0, 1])
    code = prog.get_code()

    assert "V_MIN_VF" in code
    assert "V_MAX_VF" in code
    assert "S_LD_FP" in code
    print("  PASS test_tile_row_minmax_fp_helpers_emit_vector_scalar_clamp_ops")


def test_hbm_load_helper_uses_typed_legalization():
    """Converted HBM load helpers should legalize typed large immediates."""
    from compiler.aten.plena import IsaCompiler

    compiler = IsaCompiler()
    compiler.register_matrix("W", (512, 512), hbm_base_addr=0)

    code = compiler.load_sub_matrix_asm("W", row_idx=0, col_idx=0, mram_dest_addr=0)
    assert "S_LUI_INT gp1, 64" in code
    assert "S_ADDI_INT gp1, gp0, 262144" not in code
    assert "H_PREFETCH_M gp3, gp1, a1, 1, 0" in code
    print("  PASS test_hbm_load_helper_uses_typed_legalization")


def test_vram_fill_zero_all_column_blocks():
    """vram_fill_zero must zero ALL column blocks of a wide matrix."""
    from compiler.aten.plena import PlenaCompiler

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
    from compiler.aten.plena import PlenaCompiler

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


def test_stage_checkpoint_recorder_emits_stable_vram_copy_metadata():
    """Debug checkpoints should preserve source tensors at new VRAM addresses."""
    from compiler.aten.plena import PlenaCompiler
    from compiler.aten.plena_frontend import StageCheckpointRecorder

    prog = PlenaCompiler(mlen=8, blen=2)
    x = prog.alloc("X", 8, 8)
    recorder = StageCheckpointRecorder(enabled=True)
    checkpoint = recorder.record(
        prog,
        layer_idx=0,
        stage="attn_norm",
        tensor=x,
        active_shape=(8, 8),
        semantic="unit test checkpoint",
    )
    metadata = recorder.metadata()

    assert checkpoint is not None
    assert len(metadata["checkpoints"]) == 1
    entry = metadata["checkpoints"][0]
    assert entry["stage"] == "attn_norm"
    assert entry["layer_idx"] == 0
    assert entry["source"] == x.name
    assert entry["vram_addr"] == prog.get_vram_addr(checkpoint.name)
    assert entry["physical_shape"] == [8, 8]
    assert "VRAM Matrix Add" in prog.get_code()
    print("  PASS test_stage_checkpoint_recorder_emits_stable_vram_copy_metadata")


def test_alloc_at_correct_address():
    """alloc_at must create a VRAM view at the specified address."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler()
    # Allocate a matrix, then create a view into its second column block
    x = prog.alloc("X", 64, 384)
    x_addr = prog.get_vram_addr(x.name)

    # View of column block 2 (offset = 2 * 64 * 64 = 8192)
    view_addr = x_addr + 2 * 64 * 64
    view = prog.alloc_at("X_cb2_view", 64, 64, view_addr)

    actual_addr = prog.get_vram_addr(view.name)
    assert actual_addr == view_addr, (
        f"alloc_at address mismatch: expected {view_addr}, got {actual_addr}"
    )
    print("  PASS test_alloc_at_correct_address")


def test_mram_allocator_scales_with_runtime_mlen():
    """MRAMAllocator capacity/alignment must use runtime mlen, not module MLEN=64."""
    from compiler.aten.plena.memory import MRAMAllocator

    alloc = MRAMAllocator(mlen=256, tile_capacity=4)
    tile_elems = 256 * 256

    assert alloc.alignment == tile_elems
    assert alloc.total_size == 4 * tile_elems
    assert alloc.tile_elems == tile_elems
    assert alloc.tile_capacity == 4

    addrs = [alloc.allocate(f"tile_{i}", tile_elems) for i in range(4)]
    assert addrs == [0, tile_elems, 2 * tile_elems, 3 * tile_elems]

    try:
        alloc.allocate("overflow", tile_elems)
    except MemoryError:
        pass
    else:
        raise AssertionError("MRAMAllocator accepted a fifth tile despite tile_capacity=4")

    print("  PASS test_mram_allocator_scales_with_runtime_mlen")


def test_compiler_threads_runtime_memory_geometry():
    """PlenaCompiler must pass runtime mlen/capacity into VRAM and MRAM allocators."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler(mlen=256, blen=64, mram_tile_capacity=3)
    tile_elems = 256 * 256

    assert prog.mram_tile_capacity == 3
    assert prog.mram_tile_elems == tile_elems
    assert prog.mram_capacity_elems == 3 * tile_elems
    assert prog.mram_allocator.alignment == tile_elems
    assert prog.mram_allocator.total_size == 3 * tile_elems
    assert prog.vram_allocator.alignment == tile_elems

    print("  PASS test_compiler_threads_runtime_memory_geometry")


def test_linear_projection_uses_runtime_mram_tile_capacity():
    """K-split should follow prog.mram_tile_capacity instead of a module constant."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler(mlen=128, blen=4, mram_tile_capacity=2)
    x_input = prog.input("X", shape=(128, 384), prestaged_vram_addr=0)
    x = prog.load_batch(x_input, name="X")
    w = prog.input("W", shape=(384, 128))

    prog.linear_projection(x, w, name="Y")
    code = prog.compile()

    # K=384 at mlen=128 is 3 K-tiles. With runtime capacity 2 this must split
    # into two projection chunks and accumulate the second partial sum.
    assert prog.mram_allocator.total_size == 2 * 128 * 128
    assert "V_ADD_VV" in code
    assert "linear_out_temp" not in code
    assert "Y_temp" in code

    print("  PASS test_linear_projection_uses_runtime_mram_tile_capacity")


def test_packed_skinny_stream_k_probe_compiles_cap8_under_cap4_mram():
    """Packed-skinny router probe keeps eight K slices in one MRAM tile."""
    from compiler.aten.plena import PlenaCompiler

    mlen = 128
    blen = 4
    tiles_per_mlen = mlen // blen
    hidden = 2048
    num_k_tiles = hidden // mlen
    max_k_tiles_per_packed_tile = 8
    num_groups = num_k_tiles // max_k_tiles_per_packed_tile

    prog = PlenaCompiler(mlen=mlen, blen=blen, mram_tile_capacity=4)
    x_input = prog.input(
        "X",
        shape=(4, hidden),
        physical_shape=(4, hidden),
        real_data_ratio=1.0,
    )
    x = prog.load_batch(x_input, name="X")
    packed_weight = prog.input(
        "W_router_packed_skinny_probe",
        shape=(num_groups * mlen, tiles_per_mlen * mlen),
        physical_shape=(num_groups * mlen, tiles_per_mlen * mlen),
        real_data_ratio=1.0,
    )
    logits = prog.alloc("router_logits", 4, 128, strict=False, physical_shape=(4, 128))

    prog.vram_sub_projection_packed_skinny_stream_k_accum_to(
        x,
        0,
        packed_weight,
        0,
        logits,
        0,
        0,
        max_k_tiles_per_packed_tile=max_k_tiles_per_packed_tile,
    )
    code = prog.get_code()

    # One HBM->MRAM tile per (micro-column, K group).  A non-packed cap8
    # implementation would need eight full tiles live and overflow cap4 MRAM.
    assert code.count("H_PREFETCH_M") == tiles_per_mlen * num_groups
    assert code.count("M_MM 0,") == tiles_per_mlen * num_k_tiles
    assert code.count("M_MM_WO") == tiles_per_mlen
    assert "VRAM Sub Projection packed skinny microtile" in code
    for offset in range(0, max_k_tiles_per_packed_tile * blen, blen):
        assert f"S_ADDI_INT gp2, gp0, {offset}" in code

    print("  PASS test_packed_skinny_stream_k_probe_compiles_cap8_under_cap4_mram")


def test_qwen_packed_skinny_router_rowpacked_compiles_for_128_experts():
    """Integrated Qwen packed-skinny router emits V_TOPK rowpacked logits."""
    from compiler.aten.plena import PlenaCompiler

    mlen = 64
    blen = 4
    rows = 4
    hidden = 128
    num_experts = 128
    k_tiles_per_packed_tile = 8
    num_k_tiles = hidden // mlen
    num_groups = 1
    output_blocks = num_experts // mlen
    microcols = num_experts // blen

    prog = PlenaCompiler(mlen=mlen, blen=blen, mram_tile_capacity=4)
    x_input = prog.input("X", shape=(rows, hidden), physical_shape=(rows, hidden), real_data_ratio=1.0)
    x = prog.load_batch(x_input, name="X")
    packed_weight = prog.input(
        "W_router_packed_skinny_qwen",
        shape=(num_groups * mlen, microcols * mlen),
        physical_shape=(num_groups * mlen, microcols * mlen),
        real_data_ratio=1.0,
    )

    logits = prog.qwen3_router_logits_packed_skinny_bf16_rowpacked_v0(
        x,
        packed_weight,
        rows=rows,
        hidden=hidden,
        num_experts=num_experts,
        k_tiles_per_packed_tile=k_tiles_per_packed_tile,
    )
    code = prog.compile()

    assert logits.shape == (rows * output_blocks, mlen)
    assert code.count("H_PREFETCH_M") == microcols * num_groups
    assert code.count("M_MM 0,") == microcols * num_k_tiles
    assert code.count("M_MM_WO") == microcols
    assert code.count("V_ADD_VF") >= rows * output_blocks
    assert "Qwen router packed-skinny logits pack" in code

    print("  PASS test_qwen_packed_skinny_router_rowpacked_compiles_for_128_experts")


def _build_dynamic_expert_projection(hidden, out_features=64, mlen=64, blen=4):
    """Compile one runtime-expert-id linear projection and return (code, output)."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler(mlen=mlen, blen=blen, mram_tile_capacity=4)
    x_input = prog.input("X", shape=(blen, hidden), physical_shape=(blen, hidden), real_data_ratio=1.0)
    x = prog.load_batch(x_input, name="X")
    weight = prog.input(
        "W_expert",
        shape=(hidden, out_features),
        physical_shape=(hidden, out_features),
        real_data_ratio=1.0,
    )
    output = prog.gpt_oss_dynamic_linear_projection_v0(
        x,
        weight,
        expert_indices_int_base=0,
        pair_idx=0,
        table_base=0,
        per_expert_stride=hidden * out_features,
        name="proj",
    )
    return prog.get_code(), output


def test_gpt_oss_dynamic_linear_projection_single_k_group_compiles():
    """Runtime-expert projection with num_k_tiles <= MRAM capacity: single-shot, no K-split."""
    # hidden=256 -> 4 K-tiles == mram_tile_capacity, so the non-split branch is taken.
    code, output = _build_dynamic_expert_projection(hidden=256)
    assert output.shape == (4, 64)
    # Runtime expert id is read from int SRAM and drives the dynamic weight load.
    assert "S_LD_INT" in code
    assert code.count("M_MM 0,") == 1
    assert code.count("M_MM_WO") == 1
    # No K-split -> no temp accumulator and no block-add.
    assert "proj_temp" not in code
    assert code.count("V_ADD_VV") == 0
    print("  PASS test_gpt_oss_dynamic_linear_projection_single_k_group_compiles")


def test_gpt_oss_dynamic_linear_projection_k_split_compiles():
    """Runtime-expert projection with num_k_tiles > MRAM capacity exercises the K-split
    accumulation branch — the live path at real GPT-OSS dims (hidden=2880 -> 45 K-tiles)."""
    # hidden=320 -> 5 K-tiles > mram_tile_capacity=4, so K-split into chunks [4, 1].
    code, output = _build_dynamic_expert_projection(hidden=320)
    assert output.shape == (4, 64)
    assert "S_LD_INT" in code
    # Two K-chunks: chunk 0 writes the output tile, chunk 1 writes a temp tile that is
    # block-added into the output accumulator.
    assert code.count("M_MM 0,") == 2
    assert code.count("M_MM_WO") == 2
    assert "proj_temp" in code
    assert code.count("V_ADD_VV") >= 1
    print("  PASS test_gpt_oss_dynamic_linear_projection_k_split_compiles")


def test_vram_layout_tracks_logical_and_physical_shape():
    """Layouts should keep native logical rows while allocating physical BLEN row storage."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler(mlen=256, blen=64)
    x = prog.alloc("X", 4, 256, strict=False, physical_shape=(64, 256))
    layout = prog.get_vram_layout(x.name)

    assert layout.full_shape == (4, 256)
    assert layout.physical_shape == (64, 256)
    block = layout.get_sub_block(0, 0)
    assert block.valid_shape == (4, 256)
    assert x.shape == (4, 256)
    assert x.physical_shape == (64, 256)

    print("  PASS test_vram_layout_tracks_logical_and_physical_shape")


def test_partial_row_linear_uses_one_blen_row_group():
    """M < BLEN on a large-MLEN config should emit one row group, not force MLEN rows."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler(mlen=256, blen=64)
    x_input = prog.input("X", shape=(4, 256), physical_shape=(64, 256))
    w = prog.input("W", shape=(256, 256), physical_shape=(256, 256))
    x = prog.load_batch(x_input, name="X")
    y = prog.linear_projection(x, w, name="Y")
    code = prog.get_code()

    assert y.shape == (4, 256)
    assert y.physical_shape == (64, 256)
    assert "M_MM" in code
    # The projection template receives valid_rows=4 and BLEN=64, so the
    # middle row-group loop should run once.
    assert ", 1" in code

    print("  PASS test_partial_row_linear_uses_one_blen_row_group")


def test_ffn_workspace_uses_allocator_and_avoids_rope_tables():
    """FFN temps must be allocator-managed, not hardcoded into low VRAM."""
    from compiler.aten.plena import PlenaCompiler

    def overlaps(a0, a1, b0, b1):
        return a0 < b1 and b0 < a1

    prog = PlenaCompiler(mlen=256, blen=64, mram_tile_capacity=16)
    cos_input = prog.input("COS", shape=(64, 256), physical_shape=(64, 256))
    sin_input = prog.input("SIN", shape=(64, 256), physical_shape=(64, 256))
    cos = prog.load_batch(cos_input, name="COS")
    sin = prog.load_batch(sin_input, name="SIN")
    x_input = prog.input("X", shape=(64, 768), physical_shape=(64, 768))
    x = prog.load_batch(x_input, name="X")
    w_gate = prog.input("W_gate", shape=(768, 1536), physical_shape=(768, 1536))
    w_up = prog.input("W_up", shape=(768, 1536), physical_shape=(768, 1536))
    w_down = prog.input("W_down", shape=(1536, 768), physical_shape=(1536, 768))

    cos_range = (prog.get_vram_addr(cos.name), prog.get_vram_addr(cos.name) + 64 * 256)
    sin_range = (prog.get_vram_addr(sin.name), prog.get_vram_addr(sin.name) + 64 * 256)
    prog.ffn(x, w_gate, w_up, w_down)
    code = prog.compile()

    match = re.search(r"Allocate VRAM Matrix _ffn_workspace:.* at VRAM\[(\d+)\]", code)
    assert match is not None, "FFN workspace allocation comment missing"
    workspace_base = int(match.group(1))
    workspace_elems = 64 * (2 * 1536 + 1536)
    workspace_range = (workspace_base, workspace_base + workspace_elems)

    assert not overlaps(*workspace_range, *cos_range), (
        f"FFN workspace {workspace_range} overlaps COS {cos_range}"
    )
    assert not overlaps(*workspace_range, *sin_range), (
        f"FFN workspace {workspace_range} overlaps SIN {sin_range}"
    )
    assert "_ffn_absolute_workspace_padding" not in code

    print("  PASS test_ffn_workspace_uses_allocator_and_avoids_rope_tables")


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


def test_fix_large_immediates_legalizes_relative_adds():
    """_fix_large_immediates must legalize relative S_ADDI_INT without a scratch register."""
    from compiler.aten.plena_frontend import _fix_large_immediates

    # Relative add: gp5 = gp3 + 300000. With no liveness information available
    # in the post-pass, this lowers to bounded ADDI chunks instead of using a
    # scratch register that could clobber live state.
    asm = "S_ADDI_INT gp5, gp3, 300000\n"
    fixed = _fix_large_immediates(asm)
    assert "S_ADDI_INT gp5, gp3, 262143" in fixed
    assert "S_ADDI_INT gp5, gp5, 37857" in fixed
    assert "S_ADDI_INT gp5, gp3, 300000" not in fixed

    # Absolute load: gp5 = gp0 + 300000 — SHOULD be converted
    asm2 = "S_ADDI_INT gp5, gp0, 300000\n"
    fixed2 = _fix_large_immediates(asm2)
    assert "S_LUI_INT" in fixed2, (
        "Absolute S_ADDI_INT with large value was not converted"
    )

    print("  PASS test_fix_large_immediates_legalizes_relative_adds")


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


def test_grouped_attention_weight_padding_preserves_head_slots():
    """Padded Q/O weights must keep native head lanes in padded per-head slots."""
    from compiler.aten.plena_frontend import _pad_o_weight_grouped, _pad_q_weight_grouped

    w_q = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6)
    padded_q = _pad_q_weight_grouped(w_q, num_heads=3, head_dim=2, padded_hidden=4, padded_head_dim=4)

    assert padded_q.shape == (4, 12)
    assert torch.equal(padded_q[:2, 0:2], w_q[:, 0:2])
    assert torch.equal(padded_q[:2, 4:6], w_q[:, 2:4])
    assert torch.equal(padded_q[:2, 8:10], w_q[:, 4:6])
    assert torch.count_nonzero(padded_q[:2, 2:4]) == 0
    assert torch.count_nonzero(padded_q[:2, 6:8]) == 0
    assert torch.count_nonzero(padded_q[:2, 10:12]) == 0
    assert torch.count_nonzero(padded_q[2:, :]) == 0

    w_o = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    padded_o = _pad_o_weight_grouped(w_o, num_heads=3, head_dim=2, padded_head_dim=4, padded_hidden=4)

    assert padded_o.shape == (12, 4)
    assert torch.equal(padded_o[0:2, :2], w_o[0:2, :])
    assert torch.equal(padded_o[4:6, :2], w_o[2:4, :])
    assert torch.equal(padded_o[8:10, :2], w_o[4:6, :])
    assert torch.count_nonzero(padded_o[2:4, :]) == 0
    assert torch.count_nonzero(padded_o[6:8, :]) == 0
    assert torch.count_nonzero(padded_o[10:12, :]) == 0
    assert torch.count_nonzero(padded_o[:, 2:]) == 0

    print("  PASS test_grouped_attention_weight_padding_preserves_head_slots")


def test_kv_grouped_head_packing_preserves_slots():
    """Packed attention should place Q/O heads in KV-group HLEN slots."""
    from compiler.aten.plena_frontend import (
        _pad_o_weight_grouped_by_kv,
        _pad_q_weight_grouped_by_kv,
        _pad_rope_inputs_for_head_slots,
    )

    w_q = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6)
    padded_q = _pad_q_weight_grouped_by_kv(
        w_q,
        num_heads=3,
        num_kv_heads=1,
        head_dim=2,
        padded_hidden=4,
        group_width=8,
        head_slot_dim=2,
    )
    assert padded_q.shape == (4, 8)
    assert torch.equal(padded_q[:2, 0:2], w_q[:, 0:2])
    assert torch.equal(padded_q[:2, 2:4], w_q[:, 2:4])
    assert torch.equal(padded_q[:2, 4:6], w_q[:, 4:6])
    assert torch.count_nonzero(padded_q[:2, 6:8]) == 0
    assert torch.count_nonzero(padded_q[2:, :]) == 0

    w_o = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2)
    padded_o = _pad_o_weight_grouped_by_kv(
        w_o,
        num_heads=3,
        num_kv_heads=1,
        head_dim=2,
        group_width=8,
        head_slot_dim=2,
        padded_hidden=4,
    )
    assert padded_o.shape == (8, 4)
    assert torch.equal(padded_o[0:2, :2], w_o[0:2, :])
    assert torch.equal(padded_o[2:4, :2], w_o[2:4, :])
    assert torch.equal(padded_o[4:6, :2], w_o[4:6, :])
    assert torch.count_nonzero(padded_o[6:8, :]) == 0
    assert torch.count_nonzero(padded_o[:, 2:]) == 0

    rope = torch.eye(2)
    cos = torch.ones(3, 2)
    sin = torch.full((3, 2), 2.0)
    packed_rope, packed_cos, packed_sin = _pad_rope_inputs_for_head_slots(
        rope,
        cos,
        sin,
        padded_seq_len=4,
        group_width=8,
        head_slot_dim=2,
        broadcast_amount=4,
    )
    assert packed_rope.shape == (8, 8)
    assert torch.equal(packed_rope[0:2, 0:2], rope)
    assert torch.equal(packed_rope[2:4, 2:4], rope)
    assert torch.equal(packed_cos[:3, 4:6], cos)
    assert torch.equal(packed_sin[:3, 6:8], sin)
    assert torch.count_nonzero(packed_cos[3, :]) == 0

    print("  PASS test_kv_grouped_head_packing_preserves_slots")


def test_packed_kv_weights_keep_compact_logical_width_with_physical_storage():
    """Packed K/V weights should be logical HLEN while reserving MLEN HBM storage."""
    from compiler.aten.model_extract import LayerWeights, ModelConfig
    from compiler.aten.plena import PlenaCompiler
    from compiler.aten.plena_frontend import (
        AttentionHeadPacking,
        _pad_decoder_weights_for_tiles,
        _tensor_layout_metadata,
        _weight_physical_shapes_for_layer,
    )

    model_cfg = ModelConfig(
        hidden_size=2,
        inter_dim=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        eps=1e-5,
        rope_theta=10000.0,
        vocab_size=None,
        model_type="unit",
    )
    weights = LayerWeights(
        w_q=torch.ones(2, 4),
        w_o=torch.ones(4, 2),
        w_k_heads=[torch.ones(2, 2)],
        w_v_heads=[torch.ones(2, 2)],
        w_gate=torch.ones(2, 4),
        w_up=torch.ones(2, 4),
        w_down=torch.ones(4, 2),
        eps=1e-5,
    )
    head_packing = AttentionHeadPacking(
        enabled=True,
        hlen=4,
        broadcast_amount=2,
        head_slot_dim=4,
        group_width=8,
        total_q_dim=8,
    )
    padded = _pad_decoder_weights_for_tiles(
        weights,
        model_cfg,
        padded_hidden=8,
        padded_inter=8,
        padded_head_dim=8,
        head_packing=head_packing,
    )

    assert padded.w_k_heads[0].shape == (8, 4)
    assert padded.w_v_heads[0].shape == (8, 4)

    prog = PlenaCompiler(mlen=8, blen=2)
    physical_shapes = _weight_physical_shapes_for_layer(
        0,
        padded,
        head_packing=head_packing,
        padded_head_dim=8,
    )
    prog.input("W_k_0_h0", shape=tuple(padded.w_k_heads[0].shape), physical_shape=physical_shapes["W_k_0_h0"])
    layouts = _tensor_layout_metadata(prog, {"W_k_0_h0": padded.w_k_heads[0]})

    assert layouts["W_k_0_h0"]["source_shape"] == [8, 4]
    assert layouts["W_k_0_h0"]["storage_shape"] == [8, 8]
    assert layouts["W_k_0_h0"]["source_row_elements"] == 4
    assert layouts["W_k_0_h0"]["storage_row_elements"] == 8

    print("  PASS test_packed_kv_weights_keep_compact_logical_width_with_physical_storage")


def test_packed_gqa_kv_group_loop_reduces_static_code():
    """Packed GQA can emit one attention-core body under a KV-head loop."""
    from compiler.aten.plena import PlenaCompiler

    def instruction_lines(asm: str) -> int:
        prefixes = ("S_", "C_", "H_", "V_", "M_")
        return sum(1 for line in asm.splitlines() if line.strip().startswith(prefixes))

    def emit(looped: bool) -> str:
        prog = PlenaCompiler(mlen=256, blen=64)
        prog.hlen = 64
        prog.broadcast_amount = 4
        q = prog.alloc("Q", 64, 512, strict=False, physical_shape=(256, 512))
        o = prog.alloc("O", 64, 512, strict=False, physical_shape=(256, 512))
        scratch = prog.alloc("S_scratch", 256 * (4 + 3), 256, strict=True)
        mask = prog.alloc("mask", 256, 256, strict=True)
        kv_pairs = []
        for kv_h in range(2):
            k = prog.input(f"K{kv_h}", shape=(64, 64), physical_shape=(256, 256))
            v = prog.input(f"V{kv_h}", shape=(64, 64), physical_shape=(256, 256))
            kv_pairs.append((k, v))

        q_base = prog.get_vram_addr(q.name)
        o_base = prog.get_vram_addr(o.name)
        scratch_base = prog.get_vram_addr(scratch.name)
        if looped:
            prog.flash_attention_packed_groups_looped(
                q,
                kv_pairs,
                group_heads=3,
                head_slot_dim=64,
                output_base_address=o_base,
                scratch_base_address=scratch_base,
                broadcast_amount=4,
                causal_mask=mask,
            )
        else:
            for kv_h, (k, v) in enumerate(kv_pairs):
                q_group = prog.alloc_at(
                    f"Q_group{kv_h}",
                    64,
                    256,
                    q_base + kv_h * 256 * 256,
                    physical_shape=(256, 256),
                )
                prog.flash_attention_packed_group(
                    q_group,
                    k,
                    v,
                    group_heads=3,
                    head_slot_dim=64,
                    output_base_address=o_base + kv_h * 256 * 256,
                    scratch_base_address=scratch_base,
                    broadcast_amount=4,
                    causal_mask=mask,
                )
        return prog.compile()

    baseline = emit(looped=False)
    looped = emit(looped=True)

    assert baseline.count("Packed GQA QK^T") == 2
    assert looped.count("Packed GQA attention core loop over KV groups") == 1
    assert looped.count("Packed GQA QK^T") == 1
    assert instruction_lines(looped) < instruction_lines(baseline)

    print("  PASS test_packed_gqa_kv_group_loop_reduces_static_code")


def test_packed_gqa_fused_accepts_batch_slabs():
    """Packed GQA should isolate true B>1 slabs instead of attending across B*S rows."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler(mlen=64, blen=4)
    prog.hlen = 16
    prog.broadcast_amount = 4

    q_input = prog.input("Q", shape=(128, 64), physical_shape=(128, 64), prestaged_vram_addr=0)
    k_input = prog.input("K", shape=(128, 16), physical_shape=(128, 64))
    v_input = prog.input("V", shape=(128, 16), physical_shape=(128, 64))
    q = prog.load_batch(q_input, name="Q")

    o = prog.flash_attention(
        q,
        k_input,
        v_input,
        scale=1.0 / 4.0,
        hq=4,
        hkv=1,
        h_qkv=16,
        batch_size=2,
        seq_len=64,
        kv_seq_len=64,
    )
    asm = prog.compile()

    assert o.shape == (128, 64)
    assert o.physical_shape == (128, 64)
    assert "VRAM View _gqa_Q_b0" in asm
    assert "VRAM View _gqa_Q_b1" in asm
    assert "Load SubMatrix K[0][0]" in asm
    assert "Load SubMatrix K[1][0]" in asm
    assert "Compute PV = P @ V[k_idx=0]" in asm
    assert "Compute PV = P @ V[k_idx=1]" in asm

    print("  PASS test_packed_gqa_fused_accepts_batch_slabs")


def test_mha_accepts_batch_slabs():
    """MHA should isolate true B>1 slabs instead of flattening into one sequence."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler(mlen=64, blen=4)
    q_input = prog.input("Q", shape=(128, 64), physical_shape=(128, 64), prestaged_vram_addr=0)
    k_input = prog.input("K", shape=(128, 64), physical_shape=(128, 64))
    v_input = prog.input("V", shape=(128, 64), physical_shape=(128, 64))
    q = prog.load_batch(q_input, name="Q")

    o = prog.flash_attention(
        q,
        k_input,
        v_input,
        scale=1.0 / 8.0,
        batch_size=2,
        seq_len=64,
        kv_seq_len=64,
    )
    asm = prog.compile()

    assert o.shape == (128, 64)
    assert o.physical_shape == (128, 64)
    assert "VRAM View _mha_Q_b0" in asm
    assert "VRAM View _mha_Q_b1" in asm
    assert "Load SubMatrix Row K[0][:]" in asm
    assert "Load SubMatrix Row K[1][:]" in asm
    assert "Compute PV = P @ V[k_idx=0]" in asm
    assert "Compute PV = P @ V[k_idx=1]" in asm

    print("  PASS test_mha_accepts_batch_slabs")


def test_mha_causal_skips_future_tiles_and_masks_only_diagonal():
    """Causal MHA across multiple seq tiles (seq_len > mlen).

    Regression for the seq>mlen causal bug: the static (mlen, mlen) triangular
    mask used to be added to *every* score tile, which leaked future keys on
    above-diagonal tiles and dropped valid past keys on below-diagonal tiles
    (correct only for the single-tile seq_len <= mlen case). The kernel must now
    mask only the diagonal tile, skip strictly-future key tiles entirely, and
    leave strictly-past tiles unmasked.
    """
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler(mlen=64, blen=4)
    q_input = prog.input("Q", shape=(128, 64), physical_shape=(128, 64), prestaged_vram_addr=0)
    k_input = prog.input("K", shape=(128, 64), physical_shape=(128, 64))
    v_input = prog.input("V", shape=(128, 64), physical_shape=(128, 64))
    q = prog.load_batch(q_input, name="Q")
    mask_input = prog.input("causal_mask", shape=(64, 64))
    mask = prog.load_batch(mask_input, name="CAUSAL_MASK")

    o = prog.flash_attention(
        q,
        k_input,
        v_input,
        scale=1.0 / 8.0,
        causal_mask=mask,
        batch_size=1,
        seq_len=128,
        kv_seq_len=128,
    )
    asm = prog.compile()

    assert o.shape == (128, 64)
    # 2 query tiles x 2 key tiles. Causal keeps (q0,k0), (q1,k0), (q1,k1) and
    # drops the strictly-future (q0,k1): key tile 0 is consumed by both query
    # tiles, key tile 1 only by the second query tile.
    assert asm.count("Compute PV = P @ V[k_idx=0]") == 2
    assert asm.count("Compute PV = P @ V[k_idx=1]") == 1
    # The triangular mask is added on the two diagonal tiles only, never on the
    # below-diagonal fully-visible tile (pre-fix this was 4).
    assert asm.count("+= CAUSAL_MASK") == 2

    print("  PASS test_mha_causal_skips_future_tiles_and_masks_only_diagonal")


def test_compile_native_hf_decoder_golden_vs_hf():
    """Golden (MXFP8+BF16) should closely match HF float32 at native dims."""
    from compiler.aten.plena_frontend import compile_native_hf_decoder
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "AICrossSim/clm-60m", torch_dtype=torch.float32
    )
    model.eval()

    r = compile_native_hf_decoder(model, seq_len=64, num_layers=1)
    golden = r["golden_output"]
    hf = r["hf_ground_truth"]

    diff = (golden - hf).abs()
    pct = (diff <= 0.2 + 0.2 * hf.abs()).float().mean() * 100
    cos = torch.nn.functional.cosine_similarity(
        golden.flatten().unsqueeze(0), hf.flatten().unsqueeze(0)
    )

    assert pct >= 95.0, f"Golden vs HF allclose {pct:.1f}% < 95%"
    assert cos.item() >= 0.99, f"Golden vs HF cosine {cos.item():.4f} < 0.99"
    print(f"  PASS test_compile_native_hf_decoder_golden_vs_hf ({pct:.1f}% allclose, cos={cos.item():.4f})")


def test_native_compile_assembles():
    """Native-dim ISA must assemble without overflow."""
    import os
    import tempfile

    from compiler.aten.plena_frontend import compile_native_hf_decoder
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "AICrossSim/clm-60m", torch_dtype=torch.float32
    )
    model.eval()

    r = compile_native_hf_decoder(model, seq_len=64, num_layers=1)
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
        test_isa_builder_renders_typed_instruction,
        test_isa_builder_legalizes_large_absolute_immediates,
        test_isa_builder_legalizes_relative_large_immediates,
        test_fpvar_helper_uses_canonical_emit_path,
        test_tile_row_minmax_fp_helpers_emit_vector_scalar_clamp_ops,
        test_hbm_load_helper_uses_typed_legalization,
        test_vram_fill_zero_all_column_blocks,
        test_vram_add_all_column_blocks,
        test_stage_checkpoint_recorder_emits_stable_vram_copy_metadata,
        test_alloc_at_correct_address,
        test_mram_allocator_scales_with_runtime_mlen,
        test_compiler_threads_runtime_memory_geometry,
        test_linear_projection_uses_runtime_mram_tile_capacity,
        test_packed_skinny_stream_k_probe_compiles_cap8_under_cap4_mram,
        test_gpt_oss_dynamic_linear_projection_single_k_group_compiles,
        test_gpt_oss_dynamic_linear_projection_k_split_compiles,
        test_vram_layout_tracks_logical_and_physical_shape,
        test_partial_row_linear_uses_one_blen_row_group,
        test_ffn_workspace_uses_allocator_and_avoids_rope_tables,
        test_fix_large_immediates_roundtrip,
        test_fix_large_immediates_legalizes_relative_adds,
        test_rotate_half_matrix_identity,
        test_grouped_attention_weight_padding_preserves_head_slots,
        test_kv_grouped_head_packing_preserves_slots,
        test_packed_kv_weights_keep_compact_logical_width_with_physical_storage,
        test_packed_gqa_kv_group_loop_reduces_static_code,
        test_packed_gqa_fused_accepts_batch_slabs,
        test_mha_accepts_batch_slabs,
        test_compile_native_hf_decoder_golden_vs_hf,
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
