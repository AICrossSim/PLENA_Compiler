"""
Code generation pass for LLM symbolic graph to assembly transformation.

This module transforms the symbolic graph representation of a LLM model
into assembly code using predefined templates for different operation types.
"""

from pathlib import Path
from typing import Any

from asm_templates import (
    elementwise_add_asm,
    embedding_asm,
    ffn_asm,
    flash_attn_asm,
    gelu_asm,
    im2col_asm,
    layer_norm_asm,
    lm_head_asm,
    projection_asm,
    rms_norm_asm,
)
from asm_templates._imm import load_large_int


def _load_template(template_name: str) -> str:
    """Load assembly template from file."""
    templates_dir = Path(__file__).parent.parent / "asm_templates"
    template_path = templates_dir / f"{template_name}.asm"

    if not template_path.exists():
        raise FileNotFoundError(f"Template {template_name}.asm not found in {templates_dir}")

    with open(template_path) as f:
        return f.read()


def _generate_embedding_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly code for embedding operations."""
    vocab_size = model_info["vocab_size"]
    dim = node["dimensions"]
    hidden_size = dim["hidden_size"]

    # Hardware-precision fields are unconditionally populated by
    # hardware_parser() (see generator/parser/hardware_parser.py).  Use direct
    # indexing so a missing key surfaces as a clear KeyError instead of being
    # silently masked by a default.
    assert "act_block_width" in hardware_config, (
        "hardware_config missing 'act_block_width' — was hardware_parser() called?"
    )
    assert "block_dim" in hardware_config, "hardware_config missing 'block_dim'"
    block_dim = hardware_config["block_dim"]
    act_block_width = hardware_config["act_block_width"]
    # scale_width is intentionally excluded from voc_table_row_bytes; the
    # emulator auto-derives scale byte offsets from the data byte offset
    # (see main.rs:2031-2037).

    # HBM row stride for the vocab table.  The Rust emulator's H_PREFETCH_V
    # derives the scale byte-offset automatically from the data byte-offset
    # (main.rs:2031-2037), so the ``offset`` register advanced by the assembly
    # must count *data bytes only* — advancing by data+scale would cause the
    # auto-derived scale pointer to double-count.
    # act_block_width is total data bits per (block_dim) block.  Data bytes
    # per element = act_block_width / block_dim / 8.
    assert act_block_width % (block_dim * 8) == 0, (
        f"act_block_width={act_block_width} must be a multiple of block_dim*8={block_dim * 8}"
    )
    elem_bytes = act_block_width // (block_dim * 8)
    voc_table_row_bytes = hidden_size * elem_bytes

    batch_size = model_info.get("batch_size", 1)
    seq_len = model_info.get("seq_len", 1)
    # Embedding must produce ``batch * seq_len * hidden`` elements in VRAM — one
    # row per token.  Generate placeholder ids covering the full sequence; the
    # pattern (sequential modulo vocab_size) matches the token pattern used by
    # the earlier `_build_vram_preload` path.
    input_ids = [(i % max(1, vocab_size)) for i in range(batch_size * seq_len)]

    code = f"""
; Embedding lookup: vocab_size={vocab_size} batch={batch_size} seq_len={seq_len}
; Input: token_ids ({batch_size * seq_len} total), Output: embedded_vectors
"""
    code += embedding_asm(
        vlen=hardware_config.get("VLEN", 64),
        batch=batch_size * seq_len,
        hidden_size=hidden_size,
        alive_registers=hardware_config.get("alive_registers", [1, 2, 3, 4]),
        activation_base_address=scheduler.get("memory_layout", {}).get("vector_sram_addr", {}).get("block1", 0),
        voc_table_base_addr_reg_index=scheduler.get("register_assignment", {})
        .get("hbm_addr_reg", {})
        .get("token_table_offset", 0),
        input_ids=input_ids,
        voc_table_row_bytes=voc_table_row_bytes,
    )

    return code.strip()


def _generate_attention_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly code for attention operations.

    Handles both causal (Llama-style decoder) and bidirectional (SigLIP/ViT)
    attention with any GQA ratio (hq/hkv need not equal blen).
    When ``dims["causal_mask"]`` is False we skip RoPE on Q/K and pass
    ``causal_mask=False`` to ``flash_attn_asm`` for bidirectional softmax.
    """

    dims = node["dimensions"]
    hidden_size = dims["hidden_size"]
    num_heads = dims["num_attention_heads"]
    head_dim = dims["head_dim"]
    causal_mask = dims.get("causal_mask", True)

    # Honor per-node out_features so SigLIP (num_heads * head_dim != hidden_size
    # is unusual but possible) and GQA/MQA stay correct.
    q_out = dims.get("q_proj", {}).get("out_features", num_heads * head_dim)
    k_out = dims.get("k_proj", {}).get("out_features", num_heads * head_dim)
    v_out = dims.get("v_proj", {}).get("out_features", num_heads * head_dim)

    attn_kind = "bidirectional (SigLIP/ViT)" if not causal_mask else "causal (decoder)"
    code = f"""
; Self-attention ({attn_kind}): hidden_size={hidden_size}, heads={num_heads}, head_dim={head_dim}
; Q, K, V projections + attention.  RoPE={'off' if not causal_mask else 'on Q/K'}.
"""
    mlen = hardware_config.get("MLEN", 64)
    blen = hardware_config.get("BLEN", 4)
    batch = model_info.get("batch", 1)
    hbm_addr_reg = scheduler["register_assignment"].get("hbm_addr_reg", {})
    vsram = scheduler["memory_layout"].get("vector_sram_addr", {})

    _proj_matrix_sram = hardware_config.get("MATRIX_SRAM_SIZE", 1024)
    _proj_vlen = hardware_config.get("VLEN", 64)
    # Use dedicated k_split_scratch (placed after all activation/intermediate regions)
    # to prevent scratch/activation aliasing at batch_size=1 where block4 == block1.
    # Fall back to block4 for scheduler dicts pre-dating the new key.
    _proj_scratch = vsram.get("k_split_scratch", vsram.get("block4", 0))

    # Q, K, V must land in distinct VRAM regions so the attention stage can
    # read all three back.  Prior versions aliased all three onto ``block2``
    # which caused K and V writes to overwrite Q.  The new dedicated
    # q_scratch/k_scratch/v_scratch regions sit past all activation + FFN
    # intermediate blocks.  We fall back to block2/3/4 only when the scheduler
    # pre-dates the new keys (legacy compatibility).
    _q_scratch = vsram.get("q_scratch", vsram.get("block2", 0))
    _k_scratch = vsram.get("k_scratch", vsram.get("block3", 0))
    _v_scratch = vsram.get("v_scratch", vsram.get("block4", 0))

    # Q projection
    code += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
        w_base_hbm_offset_reg=hbm_addr_reg.get("q_weight_offset", 0),
        rope_hbm_offset_reg=hbm_addr_reg.get("rope_params_offset", 0),
        rope_on_chip_address=vsram.get("block3", 0),
        activation_base_address=vsram.get("block1", 0),
        result_base_address=_q_scratch,
        rope_enabled=causal_mask,
        out_features=q_out,
        matrix_sram_size=_proj_matrix_sram,
        scratch_base_address=_proj_scratch,
        vlen=_proj_vlen,
    )

    # K projection
    code += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
        w_base_hbm_offset_reg=hbm_addr_reg.get("k_weight_offset", 0),
        rope_hbm_offset_reg=hbm_addr_reg.get("rope_params_offset", 0),
        rope_on_chip_address=vsram.get("block3", 0),
        activation_base_address=vsram.get("block1", 0),
        result_base_address=_k_scratch,
        rope_enabled=causal_mask,
        out_features=k_out,
        matrix_sram_size=_proj_matrix_sram,
        scratch_base_address=_proj_scratch,
        vlen=_proj_vlen,
    )

    # V projection (no RoPE ever)
    code += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
        w_base_hbm_offset_reg=hbm_addr_reg.get("v_weight_offset", 0),
        rope_hbm_offset_reg=hbm_addr_reg.get("rope_params_offset", 0),
        rope_on_chip_address=vsram.get("block3", 0),
        activation_base_address=vsram.get("block1", 0),
        result_base_address=_v_scratch,
        rope_enabled=False,
        out_features=v_out,
        matrix_sram_size=_proj_matrix_sram,
        scratch_base_address=_proj_scratch,
        vlen=_proj_vlen,
    )

    num_kv_heads = dims.get("num_key_value_heads", num_heads)
    seq_len = model_info.get("seq_len", model_info.get("context_length", mlen))
    # flash_attn_asm uses ``vector_sram_base_address`` as the Q base; feed it
    # the dedicated q_scratch region rather than the old block2 alias.
    vsram_fa_base = _q_scratch
    fp_sram_map = scheduler["memory_layout"].get("fp_sram", {})
    # ``silu_one`` is the canonical name (value = 1.0); fall back to legacy
    # ``silu_e`` for older mem_layout_lib snapshots that pre-date the rename.
    fp_sram_fa_base = fp_sram_map.get("silu_one", fp_sram_map.get("silu_e", 3))
    # Drive the flash-attn template's QK-scale and -inf slot indices from the
    # mem_layout_lib.json source of truth.  Previously these were hardcoded to
    # 1 (eps) and 2 (hid_reciprocal) inside flash_attn_asm/online_softmax,
    # which produced a worse-than-zero-fill state once PR #16 began seeding
    # fp_sram.bin per the JSON convention.
    attn_scale_fp = fp_sram_map.get("attn_scale", 5)
    inf_fp = fp_sram_map.get("infinity", 0)
    k_hbm_reg = hbm_addr_reg.get("k_weight_offset", 0)
    v_hbm_reg = hbm_addr_reg.get("v_weight_offset", 0)

    attn_kind = "bidirectional" if not causal_mask else "causal decoder"
    code += f"\n; -- Flash attention ({attn_kind}, GQA-aware) --\n"
    code += flash_attn_asm(
        mlen=mlen,
        vlen=hardware_config.get("VLEN", 64),
        blen=blen,
        batch=batch,
        hq=num_heads,
        hkv=num_kv_heads,
        d=head_dim,
        q_len=seq_len,
        kv_len=seq_len,
        alive_registers_int=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        alive_registers_fp=[1, 2, 3, 4, 5, 6, 7],
        vector_sram_base_address=vsram_fa_base,
        fp_sram_start_address=fp_sram_fa_base,
        k_base_hbm_offset_reg=k_hbm_reg,
        v_base_hbm_offset_reg=v_hbm_reg,
        attn_scale_fp_address=attn_scale_fp,
        inf_fp_address=inf_fp,
        causal_mask=causal_mask,
    )

    return code.strip()


def _generate_ffn_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly code for FFN/MLP operations.

    Dispatches on ``dims["arch"]``:
      - ``"vit"`` — SigLIP/ViT two-linear FFN (fc1 -> activation -> fc2),
        emitted as two ``projection_asm`` calls.
      - default — Llama-style gated FFN (gate/up/down) via ``ffn_asm``.
    """

    dims = node["dimensions"]
    hidden_size = dims["hidden_size"]
    intermediate_size = dims["intermediate_size"]
    activation = dims["activation"]
    arch = dims.get("arch", "gated")

    mlen = hardware_config.get("MLEN", 64)
    blen = hardware_config.get("BLEN", 4)
    vsram = scheduler["memory_layout"].get("vector_sram_addr", {})
    hbm_addr_reg = scheduler["register_assignment"].get("hbm_addr_reg", {})

    _vit_matrix_sram = hardware_config.get("MATRIX_SRAM_SIZE", 1024)
    _vit_vlen = hardware_config.get("VLEN", 64)
    # Use dedicated k_split_scratch to prevent scratch/activation aliasing
    # at batch_size=1. Fall back to block4 for legacy scheduler dicts.
    _vit_scratch = vsram.get("k_split_scratch", vsram.get("block4", 0))

    if arch == "vit":
        code = f"""
; Vision FFN (ViT-style): hidden={hidden_size} -> {intermediate_size} -> {hidden_size}, act={activation}
; Emitted as fc1 (projection) + GELU-activation (implicit) + fc2 (projection).
"""
        # fc1: hidden -> intermediate
        code += projection_asm(
            mlen=mlen,
            blen=blen,
            batch=model_info.get("batch", 1),
            hidden_size=hidden_size,
            alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
            w_base_hbm_offset_reg=hbm_addr_reg.get("ffn_up_offset", 0),
            activation_base_address=vsram.get("block1", 0),
            result_base_address=vsram.get("block5", vsram.get("block2", 0)),
            rope_enabled=False,
            out_features=intermediate_size,
            matrix_sram_size=_vit_matrix_sram,
            scratch_base_address=_vit_scratch,
            vlen=_vit_vlen,
        )
        # Activation between fc1 and fc2. For GELU we emit the sigmoid-approx
        # body (x * sigmoid(1.702 * x)); other activations fall back to an
        # annotated no-op (SigLIP/ViT in practice always uses GELU variants).
        fp_sram = scheduler["memory_layout"].get("fp_sram", {})
        if activation in ("gelu", "gelu_pytorch_tanh", "quick_gelu"):
            code += f"\n; -- {activation} activation (sigmoid-approx GELU) --\n"
            # GELU scratch must not overlap Q/K/V attention scratches (which
            # are live until the end of the attention block).  FFN runs
            # strictly after attention so k_split_scratch is free here; fall
            # back to block5 (fc1 output region) for legacy scheduler dicts.
            _gelu_scratch = vsram.get("k_split_scratch", vsram.get("block5", vsram.get("block4", 0)))
            code += gelu_asm(
                # harness seeds FP slot 3 with 1.0 for SiLU sigmoid base; GELU
                # reuses that same slot as its multiplicative identity.  Do
                # not rename without also retargeting the harness preload.
                const_one_fp_address=fp_sram.get("silu_one", fp_sram.get("silu_e", 3)),
                const_1702_fp_address=fp_sram.get("gelu_1702", 4),
                alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
                # fc1 wrote here; GELU reads/writes in-place.
                activation_base_address=vsram.get("block5", vsram.get("block2", 0)),
                scratchpad_base_address=_gelu_scratch,
                vlen=_vit_vlen,
                batch_size=model_info.get("batch", 1),
                hidden_dim=intermediate_size,
            )
        else:
            code += f"\n; -- {activation} activation (unrecognized; no ASM emitted) --\n"
        # fc2: intermediate -> hidden
        code += projection_asm(
            mlen=mlen,
            blen=blen,
            batch=model_info.get("batch", 1),
            hidden_size=intermediate_size,
            alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
            w_base_hbm_offset_reg=hbm_addr_reg.get("ffn_down_offset", 0),
            activation_base_address=vsram.get("block5", vsram.get("block2", 0)),
            result_base_address=vsram.get("block1", 0),
            rope_enabled=False,
            out_features=hidden_size,
            matrix_sram_size=_vit_matrix_sram,
            scratch_base_address=_vit_scratch,
            vlen=_vit_vlen,
        )
        return code.strip()

    code = f"""
; FFN/MLP (gated): hidden={hidden_size}, inter={intermediate_size}, activation={activation}
; Gate and Up projections
"""

    ffn_gate_reg = hbm_addr_reg.get("ffn_gate_offset", 0)
    ffn_up_reg = hbm_addr_reg.get("ffn_up_offset", 0)
    ffn_down_reg = hbm_addr_reg.get("ffn_down_offset", 0)
    code += ffn_asm(
        mlen=mlen,
        vlen=hardware_config.get("VLEN", 64),
        blen=blen,
        batch=model_info.get("batch", 1),
        seq_len=model_info.get("seq_len", 1),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        alive_registers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        gate_weight_hbm_offset_reg=ffn_gate_reg,
        up_weight_hbm_offset_reg=ffn_up_reg,
        down_weight_hbm_offset_reg=ffn_down_reg,
        const_one_fp_address=scheduler.get("memory_layout", {}).get("fp_sram", {}).get("silu_one",
            scheduler.get("memory_layout", {}).get("fp_sram", {}).get("silu_e", 3)),
        activation_base_address=vsram.get("block1", 0),
        matrix_sram_size=hardware_config.get("MATRIX_SRAM_SIZE", 1024),
    )
    return code.strip()


def _generate_normalization_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly code for normalization operations.

    Dispatches to layer_norm_asm when the node requests layer_norm (ViT / SigLIP);
    otherwise defaults to rms_norm_asm (Llama-style text decoder).
    """

    dims = node["dimensions"]
    hidden_size = dims["normalized_shape"]
    norm_type = dims.get("norm_type", "rms_norm")
    _fp_sram = scheduler.get("memory_layout", {}).get("fp_sram", {})
    eps_offset = _fp_sram.get("eps", 1)
    reci_hid_offset = _fp_sram.get("hid_reciprocal", 2)
    vlen = hardware_config.get("VLEN", 64)
    batch_size = model_info.get("batch_size", 1)
    activation_base = scheduler.get("vector_sram_addr", {}).get("block1", 0)
    scratchpad_base = scheduler.get("vector_sram_addr", {}).get("block2", 0)

    if norm_type == "layer_norm":
        code = f"""
; LayerNorm: hidden_size={hidden_size}  (vision encoder)
"""
        code += layer_norm_asm(
            _eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            alive_registers=[1, 2, 3],
            activation_base_address=activation_base,
            scratchpad_base_address=scratchpad_base,
            vlen=vlen,
            batch_size=batch_size,
            hidden_dim=hidden_size,
        )
        return code.strip()

    code = f"""
; RMSNorm: hidden_size={hidden_size}
"""
    code += rms_norm_asm(
        _eps_offset=eps_offset,
        reci_hid_offset=reci_hid_offset,
        alive_registers=[1, 2, 3],
        activation_base_address=activation_base,
        scratchpad_base_address=scratchpad_base,
        vlen=vlen,
        batch_size=batch_size,
        hidden_dim=hidden_size,
    )

    return code.strip()


def _generate_conv2d_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly code for a Conv2d patch-embedding operation.

    The PLENA ISA has no native Conv2d, so we lower it to:
        1. im2col_asm   — reshape NCHW patches into a (M, C_in*K*K) matrix in VRAM
        2. projection_asm — matmul by the Conv2d weight matrix (C_out, C_in*K*K).

    For SigLIP: C_in=3, K=patch_size, stride=patch_size.  The kernel is emitted
    as a single ASM block so we're honest about what the HW would run, even
    though the orchestration over multiple patch tiles is left to the compiler
    integration.
    """

    dims = node["dimensions"]
    in_channels = dims["in_channels"]
    out_channels = dims["out_channels"]
    image_size = dims["image_size"]
    patch_size = dims["patch_size"]
    num_patches = dims["num_patches"]
    K_col = in_channels * patch_size * patch_size  # im2col row width

    mlen = hardware_config.get("MLEN", 64)
    vlen = hardware_config.get("VLEN", 64)
    blen = hardware_config.get("BLEN", 4)

    # im2col produces one VRAM row per patch; stride == patch_size so OH=OW=image/patch.
    OH = OW = image_size // patch_size
    M = num_patches

    # Pick safe default registers / VRAM addresses (kept disjoint from rest of pipeline).
    alive_registers = [10, 11, 12, 13, 14, 15]
    mask_vec_vram_addr = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block3", 0)
    scratch_vram_addr = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block4", 0)
    output_vram_base = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block1", 0)
    input_hbm_base_addr_reg = (
        scheduler["register_assignment"].get("hbm_addr_reg", {}).get("token_table_offset", 1)
    )

    code = f"""
; === Conv2d patch embedding (lowered to im2col + matmul) ===
; in_channels={in_channels}, out_channels={out_channels}
; image={image_size}x{image_size}, patch={patch_size}x{patch_size}, num_patches={num_patches}
; im2col output shape: ({M}, {K_col})
"""

    # Step 1: im2col  (requires K | vlen for multi-tile; patch_size=16, vlen=64 satisfies this)
    code += im2col_asm(
        mlen=mlen,
        vlen=vlen,
        C_in=in_channels,
        H=image_size,
        W=image_size,
        K=patch_size,
        OH=OH,
        OW=OW,
        M=M,
        alive_registers=alive_registers,
        input_hbm_base_addr_reg=input_hbm_base_addr_reg,
        mask_vec_vram_addr=mask_vec_vram_addr,
        scratch_vram_addr=scratch_vram_addr,
        output_vram_base=output_vram_base,
    )

    # Step 2: matmul against the Conv2d weight (C_out, K_col).
    # Reuse projection_asm with out_features=C_out.
    w_base_hbm_offset_reg = (
        scheduler["register_assignment"].get("hbm_addr_reg", {}).get("q_weight_offset", 2)
    )
    result_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block2", 0)
    code += "\n; -- Conv2d weight matmul: (num_patches, K_col) @ (K_col, out_channels) --\n"
    code += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=model_info.get("batch_size", 1),
        hidden_size=K_col,
        alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
        w_base_hbm_offset_reg=w_base_hbm_offset_reg,
        activation_base_address=output_vram_base,
        result_base_address=result_base_address,
        rope_enabled=False,
        out_features=out_channels,
    )

    return code.strip()


def _generate_vision_projection_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly for the vision -> text connector (pixel-shuffle + linear).

    The pixel-shuffle is a pure reshape and has no ASM cost; we annotate it and
    emit the linear projection via projection_asm.
    """

    dims = node["dimensions"]
    in_features = dims["in_features"]
    out_features = dims["out_features"]
    scale_factor = dims.get("scale_factor", 1)
    num_patches_in = dims.get("num_patches_in", 0)
    num_patches_out = dims.get("num_patches_out", 0)

    mlen = hardware_config.get("MLEN", 64)
    blen = hardware_config.get("BLEN", 4)

    w_base_hbm_offset_reg = (
        scheduler["register_assignment"].get("hbm_addr_reg", {}).get("q_weight_offset", 2)
    )
    activation_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block1", 0)
    result_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block2", 0)

    code = f"""
; === Vision -> text connector ===
; pixel_shuffle: scale_factor={scale_factor},  patches {num_patches_in} -> {num_patches_out}
; linear: in_features={in_features} -> out_features={out_features}
; (reshape has no ASM cost; emit linear projection only.)
"""
    code += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=model_info.get("batch_size", 1),
        hidden_size=in_features,
        alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
        w_base_hbm_offset_reg=w_base_hbm_offset_reg,
        activation_base_address=activation_base_address,
        result_base_address=result_base_address,
        rope_enabled=False,
        out_features=out_features,
    )

    return code.strip()


def _generate_elementwise_add_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly code for elementwise addition (residual connections)."""
    dims = node["dimensions"]
    shape = dims["shape"]

    code = f"""
    ; Elementwise addition (residual connection): shape={shape}
    """
    code += elementwise_add_asm(
        vlen=hardware_config.get("VLEN", 64),
        hidden_size=model_info["hidden_size"],
        batch=model_info.get("batch", 1),
        alive_registers=hardware_config.get("alive_registers", [1, 2, 3]),
        stored_activation_base_address=scheduler.get("vector_sram_addr", {}).get("block1", 0),
        previous_activation_base_address=scheduler.get("vector_sram_addr", {}).get("block2", 0),
        previous_act_on_chip_addr_reg_index=scheduler["register_assignment"]
        .get("hbm_addr_reg", {})
        .get("previous_activation_offset", 0),
    )
    return code.strip()


def _generate_lm_head_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly code for the LM head (hidden→vocab_size projection)."""
    dims = node["dimensions"]
    hidden_size = dims["hidden_size"]
    vocab_size = dims["vocab_size"]

    code = f"""
; LM head projection: hidden_size={hidden_size}, vocab_size={vocab_size}
; logits = hidden_states @ lm_head.weight.T
"""
    code += lm_head_asm(
        mlen=hardware_config.get("MLEN", 64),
        blen=hardware_config.get("BLEN", 4),
        batch=model_info.get("batch_size", 1),
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        alive_registers=hardware_config.get("alive_registers", [1, 2, 3, 4]),
        lm_head_weight_hbm_offset_reg=scheduler["register_assignment"]
        .get("hbm_addr_reg", {})
        .get("lm_head_weight_offset", 0),
        activation_base_address=scheduler.get("vector_sram_addr", {}).get("block1", 0),
        result_base_address=scheduler.get("vector_sram_addr", {}).get("block2", 0),
    )
    return code.strip()


def _generate_node_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly code for a single symbolic graph node."""
    operation_type = node["operation_type"]
    node_name = node["name"]

    header = f"\n; === {node_name} ({operation_type}) ===\n"

    if operation_type == "embedding":
        return header + _generate_embedding_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "attention":
        return header + _generate_attention_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "ffn":
        return header + _generate_ffn_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "normalization":
        return header + _generate_normalization_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "elementwise_add":
        return header + _generate_elementwise_add_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "lm_head":
        return header + _generate_lm_head_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "conv2d":
        return header + _generate_conv2d_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "vision_projection":
        return header + _generate_vision_projection_code(node, model_info, hardware_config, scheduler)
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")


def _generate_program_header(model_info: dict[str, Any]) -> str:
    """Generate program header with model information."""
    return f"""
; Generated assembly code for LLM model
; Model: {model_info.get("model_name", "Unknown")}
; Architecture: {model_info.get("architecture", "Unknown")}
; Hidden size: {model_info.get("hidden_size", "Unknown")}
; Number of layers: {model_info.get("num_layers", "Unknown")}
; Generated by LLM Compiler
"""


def _generate_program_footer() -> str:
    """Generate program footer."""
    return """
    ; Program termination
"""


def code_gen_pass(
    symbolic_graph: dict[str, Any],
    model_info: dict[str, Any],
    hardware_config: dict[str, Any],
    scheduler: dict[str, Any],
) -> str:
    """
    Transform the complete symbolic graph into assembly code.

    Args:
        symbolic_graph: The symbolic graph from LLMModelParser
        model_info: Model metadata for header generation

    Returns:
        Complete assembly program as string
    """
    # Generate program header
    asm_code = [_generate_program_header(model_info)]

    # Process each node in execution order
    nodes = symbolic_graph["nodes"]
    execution_order = symbolic_graph["execution_order"]

    # Create a mapping from node names to nodes for efficient lookup
    node_map = {node["name"]: node for node in nodes}

    # Generate code for each node in execution order
    for node_name in execution_order:
        if node_name in node_map:
            node = node_map[node_name]
            node_code = _generate_node_code(node, model_info, hardware_config, scheduler)
            asm_code.append(node_code)

    # Add program footer

    asm_code.append(_generate_program_footer())
    return "\n".join(asm_code)
