import re

from aten.nemotron3.blocks import (
    NemotronAttentionShape,
    NemotronAttentionWeights,
    NemotronMoeConstants,
    NemotronMoeShape,
    NemotronMoeWeights,
    emit_nemotron_attention_block,
    emit_nemotron_moe_block,
)
from aten.plena import PlenaCompiler, reserve_expert_weight_table
from aten.state.hbm_image import assemble_words


def _program() -> tuple[PlenaCompiler, object]:
    prog = PlenaCompiler(mlen=64, blen=4)
    hidden = prog.load_batch(
        prog.input(
            "hidden",
            shape=(1, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=0,
        ),
        name="hidden",
    )
    return prog, hidden


def _weight(prog: PlenaCompiler, name: str, rows: int, cols: int, *, bf16=False):
    return prog.input(
        name,
        shape=(rows, cols),
        physical_shape=(rows, cols),
        real_data_ratio=2.0 if bf16 else None,
    )


def test_connected_nemotron_gqa_consumes_projected_qkv() -> None:
    prog, hidden = _program()
    output = emit_nemotron_attention_block(
        prog,
        hidden,
        shape=NemotronAttentionShape(
            hidden=64,
            query_heads=2,
            kv_heads=1,
            head_dim=64,
        ),
        weights=NemotronAttentionWeights(
            q=_weight(prog, "w_q", 64, 128),
            k=_weight(prog, "w_k", 64, 64),
            v=_weight(prog, "w_v", 64, 64),
            out=_weight(prog, "w_o", 128, 64),
        ),
        rows=1,
    )
    assembly = prog.compile()

    assert output.shape == (1, 64)
    assert "nemotron_attention_q" in assembly
    assert "nemotron_attention_k_scratch0" in assembly
    assert assembly.count("VRAM Sub Projection T To") == 2


def test_connected_nemotron_moe_executes_routed_and_shared_relu2() -> None:
    prog, hidden = _program()
    correction = prog.load_batch(
        prog.input(
            "correction",
            shape=(4, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=256,
        ),
        name="correction",
    )
    zero = prog.fp_var("zero_row", 64)
    routed_scale = prog.fp_var("routed_scale", 2)
    output = emit_nemotron_moe_block(
        prog,
        hidden,
        shape=NemotronMoeShape(
            hidden=64,
            intermediate=64,
            shared_intermediate=64,
            num_experts=4,
            top_k=2,
        ),
        weights=NemotronMoeWeights(
            router=_weight(prog, "router", 64, 4, bf16=True),
            routed_up=reserve_expert_weight_table(
                prog, name="expert_up", num_experts=4, rows=64, cols=64
            ),
            routed_down=reserve_expert_weight_table(
                prog, name="expert_down", num_experts=4, rows=64, cols=64
            ),
            shared_up=_weight(prog, "shared_up", 64, 64),
            shared_down=_weight(prog, "shared_down", 64, 64),
        ),
        correction_bias=correction,
        constants=NemotronMoeConstants(
            zero_row=zero,
            routed_scale=routed_scale,
        ),
        rows=1,
    )
    assembly = prog.compile()

    assert output.shape == (1, 64)
    assert sum(line.startswith("V_TOPK ") for line in assembly.splitlines()) == 1
    assert "relu2 nemotron_moe_pair0" in assembly
    assert "relu2 nemotron_moe_pair1" in assembly
    assert "relu2 nemotron_moe_shared" in assembly
    assert "nemotron_moe_combine" in assembly


def test_tile_major_expert_table_assembles_above_16_gib() -> None:
    """Dynamic expert selection must never materialize a 64-bit HBM base in one GP."""
    prog = PlenaCompiler(mlen=64, blen=4)
    prog._next_hbm_addr = (1 << 34) + 123
    table = reserve_expert_weight_table(
        prog,
        name="large_kimi_expert_table",
        num_experts=896,
        rows=64,
        cols=64,
    )

    # 4,096 MXFP8 element bytes + 512 scale bytes per expert tile.  The 896
    # tiles fit in one aligned 4-MiB group, so their dynamic offset is u32-safe.
    assert prog.hbm_tensor_size(64 * 64) == 4_608
    assert table.tile_group_stride == 4 * 1024 * 1024
    assert table.base > 1 << 34
    assert table.base % table.tile_group_stride == 0
    assert prog._next_hbm_addr == table.base + table.tile_group_stride

    prog._moe_dynamic_load_sub_matrix_col_v0(
        weight_template=table.template,
        col_idx=0,
        expert_indices_int_base=0,
        pair_idx=0,
        table_base=table.base,
        per_expert_stride=table.stride,
        num_experts=table.num_experts,
        tile_group_stride=table.tile_group_stride,
        name="large_kimi_expert_load",
    )
    assembly = prog.compile()

    assert "tile-major pair=0, experts=896, group=4194304" in assembly
    assert re.search(r"S_ADDI_INT gp\d+, gp0, 4608$", assembly, re.MULTILINE)
    assert "C_SET_ADDR_REG" in assembly
    for line in assembly.splitlines():
        if line.startswith("S_LUI_INT"):
            assert int(line.rsplit(" ", 1)[1]) < 1 << 22

    words = assemble_words(assembly)
    assert words
    assert all(0 < word <= 0xFFFF_FFFF for word in words)
