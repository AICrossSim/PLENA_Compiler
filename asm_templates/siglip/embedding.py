"""SigLIP vision embedding stage assembly emitter."""

from __future__ import annotations

from compiler.asm_templates import (
    elementwise_add_vram_asm,
    preload_addr_reg_asm,
    projection_asm,
    reset_reg_asm,
)


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def build_embedding_stage_asm(
    *,
    config: dict,
    vram_layout: dict,
    hbm_layout: tuple,
    mlen: int,
    blen: int,
    vlen: int,
) -> str:
    """Build embedding projection + position add + repack to encoder input layout."""
    hbm_layout_dict, _ = hbm_layout
    seq_len = int(vram_layout["seq_len"])
    hidden_size = int(vram_layout["hidden_size"])
    patch_size = int(config["patch_size"])
    num_channels = int(config["num_channels"])
    in_features = num_channels * patch_size * patch_size

    patch_weight_offset = int(hbm_layout_dict["embedding"]["patch_weight"])

    patch_input_base = int(vram_layout["embedding_patch_input_base"])
    patch_bias_base = int(vram_layout["embedding_patch_bias_base"])
    position_base = int(vram_layout["embedding_position_base"])
    embedding_output_base = int(vram_layout["embedding_base"])

    asm = "; --- STAGE 0: Embedding Projection + Position + Repack ---\n"
    asm += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[patch_weight_offset, patch_weight_offset],
    )
    asm += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

    asm += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=seq_len,
        hidden_size=_align_up(in_features, mlen),
        vlen=vlen,
        alive_registers=[1, 2, 3, 4, 5, 6],
        w_base_hbm_offset_reg=1,
        activation_base_address=patch_input_base,
        result_base_address=embedding_output_base,
        out_features=hidden_size,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])

    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=(seq_len * hidden_size) // vlen,
        alive_registers=[10, 11],
        dst_base_address=embedding_output_base,
        src_base_address=patch_bias_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11])

    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=(seq_len * hidden_size) // vlen,
        alive_registers=[10, 11],
        dst_base_address=embedding_output_base,
        src_base_address=position_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11])

    return asm
