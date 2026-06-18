"""Streaming SigLIP ASM pipeline generator.

This module emits a full SigLIP encoder pipeline that minimizes persistent
vector-SRAM usage by reusing a shared workspace and ping-ponging activations.
"""

from .embedding import build_embedding_stage_asm
from .full_model_common import (
    align_up,
    collect_bias_bases,
    collect_layer_hbm_offsets,
    compute_flash_vram_layout,
    emit_encoder_layer,
    resolve_model_geometry,
)
from ..elementwise_add_vram_asm import elementwise_add_vram_asm
from ..reset_reg_asm import reset_reg_asm
from ..flashattn.reset import reset_vssram_code


def _compute_streaming_persistent_end(
    *,
    vram_layout: dict,
    seq_len: int,
    hidden_size: int,
    inter_size: int,
    num_layers: int,
) -> int:
    """Compute persistent VRAM end excluding per-layer activation outputs."""
    persistent_end = int(vram_layout["embedding_base"]) + int(vram_layout["embedding_size"])

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


def _copy_contiguous_vram(*, seq_len: int, hidden_size: int, vlen: int, src_base: int, dst_base: int) -> str:
    """Copy [seq_len, hidden_size] contiguous chunk-major vectors src -> dst."""
    total_vectors = (seq_len * hidden_size) // vlen
    asm = reset_vssram_code(
        reset_start_address=dst_base,
        vect_dim=vlen,
        per_stride_dim=total_vectors,
        reset_stride=vlen,
        reset_amount=1,
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


def build_full_model_streaming_asm(
    config: dict,
    layer_weights_list: list,
    vram_layout: dict,
    hbm_layout: tuple,
    mlen: int = 128,
    vlen: int = 128,
    blen: int = 4,
    max_layers: int = 27,
    embedding_mode: str = "bypass",
    final_output_base: int | None = None,
    enable_layer0_probes: bool = False,
) -> str:
    """Emit full SigLIP encoder ASM with non-persistent streaming activations."""
    hbm_layout_dict, _total_hbm = hbm_layout

    geo = resolve_model_geometry(config, vram_layout, mlen)
    seq_len = geo["seq_len"]
    seq_len_valid = geo["seq_len_valid"]
    hidden_size = geo["hidden_size"]
    inter_size = geo["inter_size"]
    num_heads = geo["num_heads"]
    num_kv_heads = geo["num_kv_heads"]
    hidden_padded = geo["hidden_padded"]
    inter_padded = geo["inter_padded"]
    head_dim = geo["head_dim"]
    d_padded = geo["d_padded"]

    asm_code = ""
    asm_code += "; ============================================================\n"
    asm_code += "; SigLIP Full Model Streaming (Minified SRAM Footprint)\n"
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

    embedding_output_base = int(vram_layout["embedding_base"])
    num_layers_to_emit = min(max_layers, len(layer_weights_list))
    if num_layers_to_emit <= 0:
        return asm_code

    bias_bases = collect_bias_bases(vram_layout)

    persistent_end = _compute_streaming_persistent_end(
        vram_layout=vram_layout,
        seq_len=seq_len,
        hidden_size=hidden_padded,
        inter_size=inter_padded,
        num_layers=num_layers_to_emit,
    )

    layer0_probe_bases: dict[str, int] = {}
    if enable_layer0_probes:
        probe_span = seq_len * hidden_padded
        layer0_probe_bases = {
            "input_chunk_major": int(persistent_end),
            "attn_chunk_major": int(persistent_end + probe_span),
            "outproj_chunk_major": int(persistent_end + 2 * probe_span),
            "final_chunk_major": int(persistent_end + 3 * probe_span),
        }
        persistent_end += 4 * probe_span
        vram_layout["layer0_probe_bases"] = dict(layer0_probe_bases)

    activation_span = seq_len * hidden_padded
    workspace_base = align_up(persistent_end, mlen)

    q_base = workspace_base
    # Flash attention runs with vector_sram_base_address=q_base (see
    # build_encoder_layer_asm), so the attention output (o_old) lands at
    # o_old_base(q_base). Deriving attn_base from any other base would point
    # out_proj and the attn snapshot at the wrong (zero) region.
    flash_layout = compute_flash_vram_layout(
        mlen=mlen,
        blen=blen,
        q_len=seq_len,
        hq=num_heads,
        hkv=num_kv_heads,
        d=d_padded,
        vector_sram_base=q_base,
    )
    attn_vram_base = flash_layout["o_old_base"]
    residual_vram_base = attn_vram_base + activation_span
    mlp_inter_vram_base = residual_vram_base + activation_span
    scratch_base = align_up(mlp_inter_vram_base + seq_len * inter_padded, mlen)

    activation_a_base = embedding_output_base
    activation_b_base = int(vram_layout.get("streaming_activation_base", scratch_base))
    if activation_b_base == activation_a_base:
        activation_b_base = align_up(activation_b_base + activation_span, mlen)

    # Track streaming layout for harnesses/debug tooling.
    vram_layout["streaming_runtime_layout"] = {
        "activation_a_base": int(activation_a_base),
        "activation_b_base": int(activation_b_base),
        "workspace_base": int(workspace_base),
        "q_seq_base": int(q_base),
        "attn_base": int(attn_vram_base),
        "residual_base": int(residual_vram_base),
        "mlp_inter_base": int(mlp_inter_vram_base),
        "scratch_base": int(scratch_base),
    }

    for layer_idx in range(num_layers_to_emit):
        asm_code += f"\n; --- LAYER {layer_idx}: Encoder Layer (Streaming) ---\n"

        layer_input_base = activation_a_base if (layer_idx % 2 == 0) else activation_b_base
        layer_output_base = activation_b_base if (layer_idx % 2 == 0) else activation_a_base

        hbm_offsets = collect_layer_hbm_offsets(hbm_layout_dict, layer_idx)

        layer_asm = emit_encoder_layer(
            geo=geo,
            mlen=mlen,
            blen=blen,
            vlen=vlen,
            layer_idx=layer_idx,
            x_base=layer_input_base,
            mlp_out_base=layer_output_base,
            q_base=q_base,
            attn_base=attn_vram_base,
            residual_base=residual_vram_base,
            mlp_inter_base=mlp_inter_vram_base,
            scratch_base=scratch_base,
            hbm_offsets=hbm_offsets,
            bias_bases=bias_bases,
            debug_stage0_snapshot_base=(
                layer0_probe_bases.get("input_chunk_major") if (enable_layer0_probes and layer_idx == 0) else None
            ),
            debug_attn_snapshot_base=(
                layer0_probe_bases.get("attn_chunk_major") if (enable_layer0_probes and layer_idx == 0) else None
            ),
            debug_outproj_snapshot_base=(
                layer0_probe_bases.get("outproj_chunk_major") if (enable_layer0_probes and layer_idx == 0) else None
            ),
        )

        asm_code += layer_asm + "\n"

        if layer_idx == 0 and enable_layer0_probes:
            asm_code += _copy_contiguous_vram(
                seq_len=seq_len,
                hidden_size=hidden_padded,
                vlen=vlen,
                src_base=layer_output_base,
                dst_base=int(layer0_probe_bases["final_chunk_major"]),
            )
            asm_code += "\n"

    final_stream_base = activation_b_base if (num_layers_to_emit % 2 == 1) else activation_a_base
    if final_output_base is None:
        layer_bases = vram_layout.get("layer_bases", {})
        final_output_base = int(layer_bases.get(num_layers_to_emit - 1, final_stream_base))

    if int(final_output_base) != int(final_stream_base):
        asm_code += "\n; --- Final output copy (streaming -> requested base) ---\n"
        asm_code += _copy_contiguous_vram(
            seq_len=seq_len,
            hidden_size=hidden_padded,
            vlen=vlen,
            src_base=int(final_stream_base),
            dst_base=int(final_output_base),
        )

    asm_code += "\n; ============================================================\n"
    asm_code += "; End of SigLIP Full Model Streaming ASM\n"
    asm_code += "; ============================================================\n"

    return asm_code
