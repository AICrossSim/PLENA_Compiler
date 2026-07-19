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
    from compiler.aten.plena.native_layout import build_attention_head_packing

    packing = build_attention_head_packing(
        mlen=8,
        hlen=2,
        head_dim=2,
        logical_broadcast_amount=3,
        gqa_ratio=3,
        num_kv_heads=1,
    )

    w_q = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6)
    padded_q = _pad_q_weight_grouped_by_kv(
        w_q,
        num_heads=3,
        num_kv_heads=1,
        head_dim=2,
        padded_hidden=4,
        head_packing=packing,
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
        padded_hidden=4,
        head_packing=packing,
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

    assert baseline.count("Packed GQA QK^T") > looped.count("Packed GQA QK^T")
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


def test_packed_gqa_schedule_covers_chunks_tails_and_mram_modes():
    """Logical GQA scheduling should separate ratio, physical lanes, and MRAM policy."""
    from compiler.aten.plena.program_attention import PackedGQASchedule

    def build(ratio, physical_broadcast, capacity=8):
        return PackedGQASchedule.build(
            batch_size=2,
            seq_len=129,
            kv_seq_len=129,
            rows_per_batch=192,
            num_kv_heads=3,
            gqa_ratio=ratio,
            physical_broadcast=physical_broadcast,
            mlen=64,
            mram_tile_capacity=capacity,
        )

    single_lane = build(8, 1)
    assert single_lane.chunks_per_kv == 8
    assert single_lane.full_chunks == 8
    assert single_lane.tail_heads == 0
    assert single_lane.q_blocks == 3
    assert single_lane.k_blocks == 3
    assert single_lane.resident_kv_tiles == 6
    assert single_lane.resident_kv is True

    four_lanes = build(8, 4)
    assert four_lanes.chunks_per_kv == 2
    assert four_lanes.full_chunks == 2
    assert four_lanes.tail_heads == 0

    tail = build(6, 4, capacity=5)
    assert tail.chunks_per_kv == 2
    assert tail.full_chunks == 1
    assert tail.tail_heads == 2
    assert tail.resident_kv is False

    print("  PASS test_packed_gqa_schedule_covers_chunks_tails_and_mram_modes")


def test_canonical_packed_gqa_codegen_reuses_resident_kv_and_direct_o():
    """Resident single-lane GQA should cache K/V, loop chunks, and avoid O packing."""
    from compiler.aten.plena import PlenaCompiler

    def emit(capacity):
        prog = PlenaCompiler(mlen=8, blen=2, mram_tile_capacity=capacity)
        prog.hlen = 8
        prog.broadcast_amount = 1
        q = prog.alloc("Q", 18, 48, strict=False, physical_shape=(32, 48))
        o = prog.alloc("O", 18, 48, strict=False, physical_shape=(32, 48))
        prog.vram_fill_zero(o)
        kv_pairs = [
            (
                prog.input(f"K{head}", shape=(18, 8), physical_shape=(32, 8)),
                prog.input(f"V{head}", shape=(18, 8), physical_shape=(32, 8)),
            )
            for head in range(2)
        ]
        scratch = prog.alloc("S", 32, 8, strict=True)
        schedule = prog.flash_attention_packed_gqa(
            q,
            o,
            kv_pairs,
            batch_size=2,
            seq_len=9,
            rows_per_batch=16,
            gqa_ratio=3,
            physical_broadcast=1,
            head_slot_dim=8,
            scratch_base_address=prog.get_vram_addr(scratch.name),
            causal_mask=None,
        )
        return schedule, prog.compile()

    resident_schedule, resident = emit(4)
    streaming_schedule, streaming = emit(3)

    assert resident_schedule.resident_kv is True
    assert streaming_schedule.resident_kv is False
    assert resident.count("H_PREFETCH_M") == 16
    assert streaming.count("H_PREFETCH_M") == 96
    assert resident.count("Resident packed GQA full-chunk loop: chunks=3") == 4
    assert "Resident packed GQA full-chunk loop" not in streaming
    assert resident.count("S_MAP_V_FP") == 1
    assert streaming.count("S_MAP_V_FP") == 1
    # The resident single-lane loop already targets the final lane and needs no
    # shift.  The streaming path uses direct packed-O shift+masked-add, while
    # both avoid the former temporary-O pack pass.
    assert "V_SHIFT_V" not in resident
    assert "V_SHIFT_V" in streaming
    assert "Pack O head lane" not in resident
    assert "Pack O head lane" not in streaming
    for asm in (resident, streaming):
        assert all(line.rstrip().endswith("1, 1") for line in asm.splitlines() if "H_PREFETCH_M" in line)

    print("  PASS test_canonical_packed_gqa_codegen_reuses_resident_kv_and_direct_o")


def test_canonical_packed_gqa_codegen_emits_nondivisible_tail():
    """A non-divisible ratio should emit only valid heads in the final chunk."""
    from compiler.aten.plena import PlenaCompiler

    prog = PlenaCompiler(mlen=8, blen=2, mram_tile_capacity=2)
    prog.hlen = 2
    prog.broadcast_amount = 3
    q = prog.alloc("Q", 7, 16, strict=False, physical_shape=(8, 16))
    o = prog.alloc("O", 7, 16, strict=False, physical_shape=(8, 16))
    prog.vram_fill_zero(o)
    k = prog.input("K", shape=(7, 2), physical_shape=(8, 8))
    v = prog.input("V", shape=(7, 2), physical_shape=(8, 8))
    scratch = prog.alloc("S", 64, 8, strict=True)
    schedule = prog.flash_attention_packed_gqa(
        q,
        o,
        [(k, v)],
        batch_size=1,
        seq_len=7,
        rows_per_batch=8,
        gqa_ratio=5,
        physical_broadcast=3,
        head_slot_dim=2,
        scratch_base_address=prog.get_vram_addr(scratch.name),
        causal_mask=None,
    )
    asm = prog.compile()

    assert schedule.chunks_per_kv == 2
    assert schedule.full_chunks == 1
    assert schedule.tail_heads == 2
    assert "Pack O head lane" not in asm
    # All five valid heads, including the two-head tail chunk, are accumulated
    # directly into their destination lane.
    assert asm.count("V_SHIFT_V") >= 5

    print("  PASS test_canonical_packed_gqa_codegen_emits_nondivisible_tail")


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


def test_native_packed_gqa_logical_broadcast_exceeds_mlen():
    """Native packed GQA should split logical broadcast across physical chunks."""
    import tempfile
    from types import SimpleNamespace

    from compiler.aten.plena_frontend import compile_native_hf_decoder

    hidden = 128
    inter = 128
    head_dim = 128
    num_heads = 8
    num_kv_heads = 1

    class TinyQwenStyleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=hidden,
                intermediate_size=inter,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
                rms_norm_eps=1e-6,
                rope_theta=1000000.0,
                vocab_size=32000,
                model_type="tiny-qwen-style",
            )
            layer = torch.nn.Module()
            layer.input_layernorm = SimpleNamespace(eps=1e-6)
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(hidden, num_heads * head_dim, bias=False)
            layer.self_attn.k_proj = torch.nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
            layer.self_attn.v_proj = torch.nn.Linear(hidden, num_kv_heads * head_dim, bias=False)
            layer.self_attn.o_proj = torch.nn.Linear(num_heads * head_dim, hidden, bias=False)
            layer.mlp = torch.nn.Module()
            layer.mlp.gate_proj = torch.nn.Linear(hidden, inter, bias=False)
            layer.mlp.up_proj = torch.nn.Linear(hidden, inter, bias=False)
            layer.mlp.down_proj = torch.nn.Linear(inter, hidden, bias=False)
            self.model = SimpleNamespace(layers=torch.nn.ModuleList([layer]))

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[TRANSACTIONAL.CONFIG.VLEN]\nvalue = 128\n")
        settings_path = f.name

    old_settings = os.environ.get("PLENA_SETTINGS_TOML")
    os.environ["PLENA_SETTINGS_TOML"] = settings_path
    try:
        result = compile_native_hf_decoder(
            TinyQwenStyleModel(),
            seq_len=4,
            num_layers=1,
            mlen=128,
            blen=8,
            hlen=128,
            broadcast_amount=8,
            attention_head_packing=True,
            mram_tile_capacity=16,
            golden_precision="fp32",
        )
    finally:
        if old_settings is None:
            os.environ.pop("PLENA_SETTINGS_TOML", None)
        else:
            os.environ["PLENA_SETTINGS_TOML"] = old_settings
        os.unlink(settings_path)

    info = result["info"]
    assert info["attention_head_packing"] is True
    assert info["attention_logical_broadcast_amount"] == 8
    assert info["attention_broadcast_amount"] == 1
    assert info["attention_physical_broadcast_amount"] == 1
    assert info["attention_chunks_per_kv"] == 8
    assert info["attention_schedule"] == "logical_kv_group"
    assert info["attention_active_head_dim"] == 128
    assert info["attention_head_slot_dim"] == 128
    assert info["attention_kv_resident"] is True
    assert info["attention_resident_kv_tiles"] == 2
    assert info["attention_looped_full_chunks"] is True
    assert info["num_heads"] * info["head_dim"] == 8 * 128

    print("  PASS test_native_packed_gqa_logical_broadcast_exceeds_mlen")


def test_native_packed_gqa_supports_head_padding_and_rejects_dimension_tiling():
    """Native packed GQA should pad D<HLEN and reject HLEN<D with a clear error."""
    import tempfile
    from types import SimpleNamespace

    from compiler.aten.plena_frontend import compile_native_hf_decoder

    class TinyHeadPaddingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=8,
                intermediate_size=8,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=4,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                vocab_size=32,
                model_type="tiny-head-padding",
            )
            layer = torch.nn.Module()
            layer.input_layernorm = SimpleNamespace(eps=1e-6)
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(8, 8, bias=False)
            layer.self_attn.k_proj = torch.nn.Linear(8, 4, bias=False)
            layer.self_attn.v_proj = torch.nn.Linear(8, 4, bias=False)
            layer.self_attn.o_proj = torch.nn.Linear(8, 8, bias=False)
            layer.mlp = torch.nn.Module()
            layer.mlp.gate_proj = torch.nn.Linear(8, 8, bias=False)
            layer.mlp.up_proj = torch.nn.Linear(8, 8, bias=False)
            layer.mlp.down_proj = torch.nn.Linear(8, 8, bias=False)
            self.model = SimpleNamespace(layers=torch.nn.ModuleList([layer]))

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[TRANSACTIONAL.CONFIG.VLEN]\nvalue = 8\n")
        settings_path = f.name

    old_settings = os.environ.get("PLENA_SETTINGS_TOML")
    os.environ["PLENA_SETTINGS_TOML"] = settings_path
    try:
        result = compile_native_hf_decoder(
            TinyHeadPaddingModel(),
            seq_len=4,
            num_layers=1,
            mlen=8,
            blen=2,
            hlen=8,
            broadcast_amount=2,
            attention_head_packing=True,
            mram_tile_capacity=2,
            golden_precision="fp32",
        )
        try:
            compile_native_hf_decoder(
                TinyHeadPaddingModel(),
                seq_len=4,
                num_layers=1,
                mlen=8,
                blen=2,
                hlen=2,
                broadcast_amount=1,
                attention_head_packing=True,
                golden_precision="fp32",
            )
        except ValueError as exc:
            message = str(exc)
            assert "does not support head-dimension tiling" in message
            assert "HLEN=2" in message
            assert "head_dim=4" in message
        else:
            raise AssertionError("HLEN < head_dim should have been rejected")
    finally:
        if old_settings is None:
            os.environ.pop("PLENA_SETTINGS_TOML", None)
        else:
            os.environ["PLENA_SETTINGS_TOML"] = old_settings
        os.unlink(settings_path)

    info = result["info"]
    assert info["attention_active_head_dim"] == 4
    assert info["attention_head_slot_dim"] == 8
    assert info["attention_logical_broadcast_amount"] == 2
    assert info["attention_physical_broadcast_amount"] == 1
    assert info["attention_chunks_per_kv"] == 2
    assert "Packed Q RoPE: slabs=2" in result["isa"]

    print("  PASS test_native_packed_gqa_supports_head_padding_and_rejects_dimension_tiling")


def test_native_decoder_compacts_batch_rows_and_gqa_groups():
    """Native lowering should consume the shared compact row/column layout."""
    import math
    import tempfile
    from collections import Counter
    from types import SimpleNamespace

    from compiler.aten.cost_frontend import (
        CompilerCostHardware,
        compile_native_decoder_cost_trace,
    )
    from compiler.aten.model_extract import extract_model_config
    from compiler.aten.plena_frontend import compile_native_hf_decoder

    def dynamic_opcode_histogram(asm: str) -> Counter[str]:
        histogram: Counter[str] = Counter()
        loop_stack: list[int] = []
        for raw_line in asm.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            opcode = line.split(maxsplit=1)[0]
            histogram[opcode] += math.prod(loop_stack)
            if opcode == "C_LOOP_START":
                loop_stack.append(int(line.rsplit(",", 1)[1]))
            elif opcode == "C_LOOP_END":
                loop_stack.pop()
        assert not loop_stack
        return histogram

    class TinyCompactQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=16,
                intermediate_size=16,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                vocab_size=32,
                model_type="qwen3",
            )
            layer = torch.nn.Module()
            layer.input_layernorm = SimpleNamespace(
                eps=1e-6, weight=torch.ones(16)
            )
            layer.post_attention_layernorm = SimpleNamespace(
                eps=1e-6, weight=torch.ones(16)
            )
            layer.self_attn = torch.nn.Module()
            layer.self_attn.q_proj = torch.nn.Linear(16, 16, bias=False)
            layer.self_attn.k_proj = torch.nn.Linear(16, 8, bias=False)
            layer.self_attn.v_proj = torch.nn.Linear(16, 8, bias=False)
            layer.self_attn.o_proj = torch.nn.Linear(16, 16, bias=False)
            layer.self_attn.q_norm = SimpleNamespace(weight=torch.ones(4))
            layer.self_attn.k_norm = SimpleNamespace(weight=torch.ones(4))
            layer.mlp = torch.nn.Module()
            layer.mlp.gate_proj = torch.nn.Linear(16, 16, bias=False)
            layer.mlp.up_proj = torch.nn.Linear(16, 16, bias=False)
            layer.mlp.down_proj = torch.nn.Linear(16, 16, bias=False)
            self.model = SimpleNamespace(
                layers=torch.nn.ModuleList([layer]),
                norm=SimpleNamespace(weight=torch.ones(16)),
            )

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[TRANSACTIONAL.CONFIG.VLEN]\nvalue = 16\n")
        settings_path = f.name
    old_settings = os.environ.get("PLENA_SETTINGS_TOML")
    os.environ["PLENA_SETTINGS_TOML"] = settings_path
    try:
        decoder_input = torch.arange(4 * 7 * 16, dtype=torch.float32).reshape(4, 7, 16) / 100
        model = TinyCompactQwen()
        result = compile_native_hf_decoder(
            model,
            seq_len=7,
            batch_size=4,
            num_layers=1,
            mlen=16,
            blen=4,
            hlen=4,
            broadcast_amount=2,
            attention_head_packing=True,
            decoder_input_embeds=decoder_input,
            golden_precision="fp32",
        )
    finally:
        if old_settings is None:
            os.environ.pop("PLENA_SETTINGS_TOML", None)
        else:
            os.environ["PLENA_SETTINGS_TOML"] = old_settings
        os.unlink(settings_path)

    info = result["info"]
    assert info["batch_pack_factor"] == 2
    assert info["attention_group_count"] == 2
    assert info["physical_token_rows"] == 32
    assert info["logical_token_rows"] == 28
    assert info["attention_groups_per_storage_block"] == 2
    assert info["attention_physical_q_width"] == 16
    assert result["input_tensors"]["X"].shape == (32, 16)
    mask = result["input_tensors"]["causal_mask"]
    assert torch.isneginf(mask[0, 7])
    assert torch.isneginf(mask[7, 0])
    assert mask[6, 0] == 0
    assert result["golden_output"].shape == (28, 16)

    trace = compile_native_decoder_cost_trace(
        extract_model_config(model),
        CompilerCostHardware(
            mlen=16,
            blen=4,
            vlen=16,
            hlen=4,
            broadcast_amount=2,
            mram_tile_capacity=4,
            hbm_m_prefetch_amount=16,
            hbm_v_prefetch_amount=4,
            hbm_v_writeback_amount=4,
            hbm_channels=8,
        ),
        seq_len=7,
        batch_size=4,
        num_layers=1,
        native_layout_mode="compact",
        use_cache=False,
    )
    assert trace.dynamic_opcodes == dynamic_opcode_histogram(result["isa"])
    dma_counts: Counter[str] = Counter()
    for event in trace.memory_events:
        dma_counts[event.transfer.opcode] += event.multiplicity
    assert dma_counts == {
        opcode: trace.dynamic_opcodes[opcode]
        for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")
    }

    print("  PASS test_native_decoder_compacts_batch_rows_and_gqa_groups")


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
        test_hbm_load_helper_uses_typed_legalization,
        test_vram_fill_zero_all_column_blocks,
        test_vram_add_all_column_blocks,
        test_stage_checkpoint_recorder_emits_stable_vram_copy_metadata,
        test_alloc_at_correct_address,
        test_mram_allocator_scales_with_runtime_mlen,
        test_compiler_threads_runtime_memory_geometry,
        test_linear_projection_uses_runtime_mram_tile_capacity,
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
        test_packed_gqa_schedule_covers_chunks_tails_and_mram_modes,
        test_canonical_packed_gqa_codegen_reuses_resident_kv_and_direct_o,
        test_canonical_packed_gqa_codegen_emits_nondivisible_tail,
        test_mha_accepts_batch_slabs,
        test_native_packed_gqa_logical_broadcast_exceeds_mlen,
        test_native_packed_gqa_supports_head_padding_and_rejects_dimension_tiling,
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
