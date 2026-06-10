"""Full-model SigLIP ASM pipeline generator.

This module contains stateless code-generation helpers for assembling
multi-layer SigLIP encoder programs.
"""

from .embedding import build_embedding_stage_asm
from .encoder_layer import build_encoder_layer_asm
from ..elementwise_add_vram_asm import elementwise_add_vram_asm
from ..flashattn.reset import reset_vssram_code
from ..reset_reg_asm import reset_reg_asm
from compiler.asm_templates.normalization_asm import layer_norm_asm


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _compute_vram_layout(mlen: int, blen: int, q_len: int, hq: int, hkv: int, d: int, vector_sram_base: int) -> dict:
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


def _compute_persistent_end(
    *,
    vram_layout: dict,
    seq_len: int,
    hidden_size: int,
    inter_size: int,
    num_layers: int,
) -> int:
    """Compute end of persistent VRAM region before scratch/workspace allocations."""
    persistent_end = int(vram_layout["embedding_base"]) + int(vram_layout["embedding_size"])
    layer_bases = vram_layout.get("layer_bases", {})
    layer_sizes = vram_layout.get("layer_sizes", {})
    if layer_bases:
        last_layer_idx = max(layer_bases.keys())
        persistent_end = max(
            persistent_end,
            int(layer_bases[last_layer_idx]) + int(layer_sizes[last_layer_idx]),
        )

    q_bias_bases = vram_layout.get("q_bias_bases", {})
    ln1_weight_bases = vram_layout.get("ln1_weight_bases", {})
    ln1_bias_bases = vram_layout.get("ln1_bias_bases", {})
    ln2_weight_bases = vram_layout.get("ln2_weight_bases", {})
    ln2_bias_bases = vram_layout.get("ln2_bias_bases", {})
    out_bias_bases = vram_layout.get("out_bias_bases", {})
    fc1_bias_bases = vram_layout.get("fc1_bias_bases", {})
    fc2_bias_bases = vram_layout.get("fc2_bias_bases", {})

    for layer_idx in range(num_layers):
        q_bias_base = q_bias_bases.get(layer_idx)
        if q_bias_base is not None:
            persistent_end = max(persistent_end, int(q_bias_base) + seq_len * hidden_size)

        for base_dict in (ln1_weight_bases, ln1_bias_bases, ln2_weight_bases, ln2_bias_bases):
            base = base_dict.get(layer_idx)
            if base is not None:
                persistent_end = max(persistent_end, int(base) + seq_len * hidden_size)

        out_base = out_bias_bases.get(layer_idx)
        if out_base is not None:
            persistent_end = max(persistent_end, int(out_base) + seq_len * hidden_size)

        fc1_base = fc1_bias_bases.get(layer_idx)
        if fc1_base is not None:
            persistent_end = max(persistent_end, int(fc1_base) + seq_len * inter_size)

        fc2_base = fc2_bias_bases.get(layer_idx)
        if fc2_base is not None:
            persistent_end = max(persistent_end, int(fc2_base) + seq_len * hidden_size)

    return max(persistent_end, int(vram_layout.get("total_vram_elements", persistent_end)))


def _copy_chunk_major_probe(*, seq_len: int, hidden_size: int, vlen: int, src_base: int, dst_base: int) -> str:
    """Copy one [seq_len, hidden_size] chunk-major activation buffer into a stable probe region."""
    total_vectors = (seq_len * hidden_size) // vlen
    asm = reset_vssram_code(
        reset_start_address=dst_base,
        vect_dim=vlen,
        per_stride_dim=hidden_size // vlen,
        reset_stride=hidden_size,
        reset_amount=seq_len,
        alive_registers_int=[10, 11, 12],
    )
    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=total_vectors,
        alive_registers=[10, 11],
        dst_base_address=dst_base,
        src_base_address=src_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11, 12])
    return asm


def build_full_model_asm(
    config: dict,
    layer_weights_list: list,
    vram_layout: dict,
    hbm_layout: tuple,
    mlen: int = 128,
    vlen: int = 128,
    blen: int = 4,
    max_layers: int = 27,
    embedding_mode: str = "bypass",
) -> str:
    """Emit complete ASM code for SigLIP embedding + encoder layers."""
    hbm_layout_dict, _total_hbm = hbm_layout

    seq_len = vram_layout["seq_len"]
    seq_len_valid = int(config.get("seq_len_valid", seq_len))
    hidden_size = config["hidden_size"]
    inter_size = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]

    hidden_padded = _align_up(hidden_size, mlen)
    inter_padded = _align_up(inter_size, mlen)
    head_dim = hidden_size // num_heads
    d_padded = _align_up(head_dim, mlen)
    attn_hidden_padded = num_heads * d_padded

    if hidden_padded != attn_hidden_padded:
        raise ValueError(
            "Internal hidden geometry mismatch after runtime repack: "
            f"hidden_padded={hidden_padded}, attn_hidden_padded={attn_hidden_padded}."
        )

    asm_code = ""
    asm_code += "; ============================================================\n"
    asm_code += "; SigLIP Full Model (27 Encoder Layers)\n"
    asm_code += f"; Hidden={hidden_size}, Heads={num_heads}, Inter={inter_size}\n"
    asm_code += f"; Sequence length={seq_len}, MLEN={mlen}, VLEN={vlen}\n"
    asm_code += "; ============================================================\n\n"

    if embedding_mode == "bypass":
        asm_code += "; --- STAGE 0: Embedding Preloaded (ASM bypass) ---\n"
    elif embedding_mode == "asm":
        asm_code += build_embedding_stage_asm(
            config=config,
            vram_layout=vram_layout,
            hbm_layout=hbm_layout,
            mlen=mlen,
            blen=blen,
            vlen=vlen,
        )
    else:
        raise ValueError(f"Unsupported embedding_mode={embedding_mode!r}")

    embedding_output_base = vram_layout["embedding_base"]
    num_layers_to_emit = min(max_layers, len(layer_weights_list))

    q_bias_bases = vram_layout.get("q_bias_bases", {})
    ln1_weight_bases = vram_layout.get("ln1_weight_bases", {})
    ln1_bias_bases = vram_layout.get("ln1_bias_bases", {})
    ln2_weight_bases = vram_layout.get("ln2_weight_bases", {})
    ln2_bias_bases = vram_layout.get("ln2_bias_bases", {})
    out_bias_bases = vram_layout.get("out_bias_bases", {})
    fc1_bias_bases = vram_layout.get("fc1_bias_bases", {})
    fc2_bias_bases = vram_layout.get("fc2_bias_bases", {})

    persistent_end = _compute_persistent_end(
        vram_layout=vram_layout,
        seq_len=seq_len,
        hidden_size=hidden_padded,
        inter_size=inter_padded,
        num_layers=num_layers_to_emit,
    )

    layer0_probe_bases: dict[str, int] = {}
    if num_layers_to_emit > 0:
        probe_span = seq_len * hidden_padded
        layer0_probe_bases = {
            "input_chunk_major": int(persistent_end),
            "attn_chunk_major": int(persistent_end + probe_span),
            "outproj_chunk_major": int(persistent_end + 2 * probe_span),
            "final_chunk_major": int(persistent_end + 3 * probe_span),
        }
        persistent_end += 4 * probe_span
        vram_layout["layer0_probe_bases"] = dict(layer0_probe_bases)

    workspace_base = _align_up(persistent_end, mlen)

    for layer_idx in range(num_layers_to_emit):
        asm_code += f"\n; --- LAYER {layer_idx}: Encoder Layer ---\n"

        layer_input_base = embedding_output_base if layer_idx == 0 else vram_layout["layer_bases"][layer_idx - 1]
        layer_output_base = vram_layout["layer_bases"][layer_idx]

        layer_hbm = hbm_layout_dict.get("layers", {}).get(layer_idx, {})
        wq_offset = layer_hbm.get("q_proj_weight", 0)
        k_offset = layer_hbm.get("k_proj_weight", 0)
        v_offset = layer_hbm.get("v_proj_weight", 0)
        w1_offset = layer_hbm.get("fc1_weight", 0)
        w2_offset = layer_hbm.get("fc2_weight", 0)
        out_offset = layer_hbm.get("out_proj_weight", 0)

        q_base = workspace_base
        q_vram_base = q_base + seq_len * hidden_padded

        flash_layout = _compute_vram_layout(
            mlen=mlen,
            blen=blen,
            q_len=seq_len,
            hq=num_heads,
            hkv=num_kv_heads,
            d=d_padded,
            vector_sram_base=q_vram_base,
        )
        attn_vram_base = flash_layout["o_old_base"]
        residual_vram_base = attn_vram_base + seq_len * hidden_padded
        mlp_inter_vram_base = residual_vram_base + seq_len * hidden_padded
        scratch_base = _align_up(mlp_inter_vram_base + seq_len * inter_padded, mlen)

        layer_asm = build_encoder_layer_asm(
            mlen=mlen,
            blen=blen,
            vlen=vlen,
            batch=1,
            s_q=seq_len,
            s_kv=seq_len,
            s_kv_valid=seq_len_valid,
            hq=num_heads,
            hkv=num_kv_heads,
            h_qkv=d_padded,
            hidden_size=hidden_padded,
            inter_dim=inter_padded,
            x_base=layer_input_base,
            q_base=q_base,
            attn_base=attn_vram_base,
            residual_base=residual_vram_base,
            mlp_inter_base=mlp_inter_vram_base,
            mlp_out_base=layer_output_base,
            scratch_base=scratch_base,
            k_hbm_offset=k_offset,
            v_hbm_offset=v_offset,
            out_hbm_offset=out_offset,
            wq_hbm_offset=wq_offset,
            w1_hbm_offset=w1_offset,
            w2_hbm_offset=w2_offset,
            ln_eps_fp_slot=2,
            ln_reci_hid_fp_slot=3,
            gelu_one_fp_slot=4,
            gelu_1702_fp_slot=5,
            attn_scale_fp_slot=1,
            attn_ninf_fp_slot=6,
            flash_temp_fp_start=64,
            q_bias_base=q_bias_bases.get(layer_idx),
            ln1_affine_weight_base=ln1_weight_bases.get(layer_idx),
            ln1_affine_bias_base=ln1_bias_bases.get(layer_idx),
            ln2_affine_weight_base=ln2_weight_bases.get(layer_idx),
            ln2_affine_bias_base=ln2_bias_bases.get(layer_idx),
            out_bias_base=out_bias_bases.get(layer_idx),
            fc1_bias_base=fc1_bias_bases.get(layer_idx),
            fc2_bias_base=fc2_bias_bases.get(layer_idx),
            debug_stage0_snapshot_base=(
                layer0_probe_bases.get("input_chunk_major") if layer_idx == 0 else None
            ),
            debug_attn_snapshot_base=(
                layer0_probe_bases.get("attn_chunk_major") if layer_idx == 0 else None
            ),
            debug_outproj_snapshot_base=(
                layer0_probe_bases.get("outproj_chunk_major") if layer_idx == 0 else None
            ),
            include_final_residual=True,
            include_gelu=True,
        )

        asm_code += layer_asm + "\n"

        if layer_idx == 0:
            asm_code += _copy_chunk_major_probe(
                seq_len=seq_len,
                hidden_size=hidden_padded,
                vlen=vlen,
                src_base=layer_output_base,
                dst_base=int(layer0_probe_bases["final_chunk_major"]),
            )
            asm_code += "\n"

    # Optional terminal post-encoder LayerNorm stage.
    if bool(config.get("apply_post_layernorm", False)):
        final_input_base = (
            vram_layout["layer_bases"][num_layers_to_emit - 1]
            if num_layers_to_emit > 0
            else embedding_output_base
        )
        final_output_base = int(vram_layout.get("final_output_base", final_input_base))
        final_ln_weight_base = vram_layout.get("final_ln_weight_base")
        final_ln_bias_base = vram_layout.get("final_ln_bias_base")
        if final_ln_weight_base is None or final_ln_bias_base is None:
            raise ValueError("apply_post_layernorm=True requires final_ln_weight_base/final_ln_bias_base in vram_layout")
        final_ln_scratch_base = _align_up(final_output_base + seq_len * hidden_padded, mlen)
        ln_activation_base = final_output_base

        asm_code += "\n; --- FINAL: Post-Encoder LayerNorm ---\n"
        if final_output_base != int(final_input_base):
            # Final compare reads from final_output_base; copy current activation there first.
            asm_code += _copy_chunk_major_probe(
                seq_len=seq_len,
                hidden_size=hidden_padded,
                vlen=vlen,
                src_base=int(final_input_base),
                dst_base=final_output_base,
            )
            asm_code += "\n"

        asm_code += layer_norm_asm(
            _eps_offset=2,
            reci_hid_offset=3,
            alive_registers=[5, 6, 7],
            activation_base_address=ln_activation_base,
            scratchpad_base_address=final_ln_scratch_base,
            vlen=vlen,
            batch_size=seq_len,
            hidden_dim=hidden_padded,
            affine_weight_base_address=final_ln_weight_base,
            affine_bias_base_address=final_ln_bias_base,
        )
        asm_code += "\n"

    asm_code += "\n; ============================================================\n"
    asm_code += "; End of SigLIP Full Model ASM\n"
    asm_code += "; ============================================================\n"

    return asm_code


def compute_hbm_data_order(num_layers: int = 27) -> list[str]:
    """Compute HBM tensor load order for embedding and per-layer weights."""
    order = [
        "patch_weight",
        "patch_bias",
        "position_table",
    ]

    for layer_idx in range(num_layers):
        order.extend(
            [
                f"layer_{layer_idx}_ln1_weight",
                f"layer_{layer_idx}_ln1_bias",
                f"layer_{layer_idx}_q_proj_weight",
                f"layer_{layer_idx}_q_proj_bias",
                f"layer_{layer_idx}_k_proj_weight",
                f"layer_{layer_idx}_k_proj_bias",
                f"layer_{layer_idx}_v_proj_weight",
                f"layer_{layer_idx}_v_proj_bias",
                f"layer_{layer_idx}_out_proj_weight",
                f"layer_{layer_idx}_out_proj_bias",
                f"layer_{layer_idx}_ln2_weight",
                f"layer_{layer_idx}_ln2_bias",
                f"layer_{layer_idx}_fc1_weight",
                f"layer_{layer_idx}_fc1_bias",
                f"layer_{layer_idx}_fc2_weight",
                f"layer_{layer_idx}_fc2_bias",
            ]
        )

    return order
