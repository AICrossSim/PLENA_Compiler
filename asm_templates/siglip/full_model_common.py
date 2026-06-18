"""Shared helpers for the SigLIP full-model ASM builders.

Both the persistent (``full_model_pipeline``) and streaming
(``full_model_streaming``) builders emit the same per-layer encoder call and
share identical geometry/layout math. Centralizing it here keeps the two
builders in lock-step so a fix only has to be made once.
"""

from .encoder_layer import build_encoder_layer_asm


def align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def compute_flash_vram_layout(
    mlen: int, blen: int, q_len: int, hq: int, hkv: int, d: int, vector_sram_base: int
) -> dict:
    """Compute flash-attn scratch layout for one encoder layer."""
    q_size = q_len * hq * d
    q_start = vector_sram_base
    s_base = q_start + q_size
    s_tile_count = blen if (hq // hkv == blen) else 1
    pv_base = s_base + mlen * mlen * s_tile_count
    o_old_base = pv_base + mlen * mlen * (hq // hkv)
    return {
        "q_base": q_start,
        "s_base": s_base,
        "pv_base": pv_base,
        "o_old_base": o_old_base,
        "q_size": q_size,
        "s_tile_count": s_tile_count,
    }


def resolve_model_geometry(config: dict, vram_layout: dict, mlen: int) -> dict:
    """Parse config/vram_layout into padded geometry shared by both builders."""
    seq_len = int(vram_layout["seq_len"])
    hidden_size = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    head_dim = hidden_size // num_heads
    geo = {
        "seq_len": seq_len,
        "seq_len_valid": int(config.get("seq_len_valid", seq_len)),
        "hidden_size": hidden_size,
        "inter_size": int(config["intermediate_size"]),
        "num_heads": num_heads,
        "num_kv_heads": int(config["num_key_value_heads"]),
        "head_dim": head_dim,
        "hidden_padded": align_up(hidden_size, mlen),
        "inter_padded": align_up(int(config["intermediate_size"]), mlen),
        "d_padded": align_up(head_dim, mlen),
    }
    attn_hidden_padded = geo["num_heads"] * geo["d_padded"]
    if geo["hidden_padded"] != attn_hidden_padded:
        raise ValueError(
            "Internal hidden geometry mismatch after runtime repack: "
            f"hidden_padded={geo['hidden_padded']}, attn_hidden_padded={attn_hidden_padded}."
        )
    return geo


def collect_bias_bases(vram_layout: dict) -> dict:
    """Extract the per-layer affine/bias base-address maps from the VRAM layout."""
    return {
        "q_bias": vram_layout.get("q_bias_bases", {}),
        "ln1_weight": vram_layout.get("ln1_weight_bases", {}),
        "ln1_bias": vram_layout.get("ln1_bias_bases", {}),
        "ln2_weight": vram_layout.get("ln2_weight_bases", {}),
        "ln2_bias": vram_layout.get("ln2_bias_bases", {}),
        "out_bias": vram_layout.get("out_bias_bases", {}),
        "fc1_bias": vram_layout.get("fc1_bias_bases", {}),
        "fc2_bias": vram_layout.get("fc2_bias_bases", {}),
    }


def collect_layer_hbm_offsets(hbm_layout_dict: dict, layer_idx: int) -> dict:
    """Extract the per-layer HBM weight offsets for one encoder layer."""
    layer_hbm = hbm_layout_dict.get("layers", {}).get(layer_idx, {})
    return {
        "wq": layer_hbm.get("q_proj_weight", 0),
        "k": layer_hbm.get("k_proj_weight", 0),
        "v": layer_hbm.get("v_proj_weight", 0),
        "w1": layer_hbm.get("fc1_weight", 0),
        "w2": layer_hbm.get("fc2_weight", 0),
        "out": layer_hbm.get("out_proj_weight", 0),
    }


def emit_encoder_layer(
    *,
    geo: dict,
    mlen: int,
    blen: int,
    vlen: int,
    layer_idx: int,
    x_base: int,
    mlp_out_base: int,
    q_base: int,
    attn_base: int,
    residual_base: int,
    mlp_inter_base: int,
    scratch_base: int,
    hbm_offsets: dict,
    bias_bases: dict,
    debug_stage0_snapshot_base: int | None = None,
    debug_attn_snapshot_base: int | None = None,
    debug_outproj_snapshot_base: int | None = None,
) -> str:
    """Emit one encoder layer with the constant slot wiring both builders share."""
    return build_encoder_layer_asm(
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        batch=1,
        s_q=geo["seq_len"],
        s_kv=geo["seq_len"],
        s_kv_valid=geo["seq_len_valid"],
        hq=geo["num_heads"],
        hkv=geo["num_kv_heads"],
        h_qkv=geo["d_padded"],
        hidden_size=geo["hidden_padded"],
        inter_dim=geo["inter_padded"],
        x_base=x_base,
        q_base=q_base,
        attn_base=attn_base,
        residual_base=residual_base,
        mlp_inter_base=mlp_inter_base,
        mlp_out_base=mlp_out_base,
        scratch_base=scratch_base,
        k_hbm_offset=hbm_offsets["k"],
        v_hbm_offset=hbm_offsets["v"],
        out_hbm_offset=hbm_offsets["out"],
        wq_hbm_offset=hbm_offsets["wq"],
        w1_hbm_offset=hbm_offsets["w1"],
        w2_hbm_offset=hbm_offsets["w2"],
        ln_eps_fp_slot=2,
        ln_reci_hid_fp_slot=3,
        gelu_one_fp_slot=4,
        gelu_1702_fp_slot=5,
        attn_scale_fp_slot=1,
        attn_ninf_fp_slot=6,
        flash_temp_fp_start=64,
        q_bias_base=bias_bases["q_bias"].get(layer_idx),
        ln1_affine_weight_base=bias_bases["ln1_weight"].get(layer_idx),
        ln1_affine_bias_base=bias_bases["ln1_bias"].get(layer_idx),
        ln2_affine_weight_base=bias_bases["ln2_weight"].get(layer_idx),
        ln2_affine_bias_base=bias_bases["ln2_bias"].get(layer_idx),
        out_bias_base=bias_bases["out_bias"].get(layer_idx),
        fc1_bias_base=bias_bases["fc1_bias"].get(layer_idx),
        fc2_bias_base=bias_bases["fc2_bias"].get(layer_idx),
        debug_stage0_snapshot_base=debug_stage0_snapshot_base,
        debug_attn_snapshot_base=debug_attn_snapshot_base,
        debug_outproj_snapshot_base=debug_outproj_snapshot_base,
        include_final_residual=True,
        include_gelu=True,
    )
