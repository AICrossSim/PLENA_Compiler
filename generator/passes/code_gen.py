"""
Code generation pass -- Generator Pipeline (Pipeline 2).

Transforms a symbolic graph (from LLMModelParser) into PLENA ISA by
dispatching each node to the appropriate asm_template function. This is
the generator's own compilation backend, separate from the ATen path's
PlenaCompiler.

HBM address registers are initialized via C_SET_ADDR_REG at the top of
the generated program, with byte offsets matching the HBM weight layout
that test_generator_e2e._build_hbm_from_hf_weights writes.

See docs/COMPILATION_PIPELINES.md for the full architecture overview.
"""

import math
from pathlib import Path
from typing import Any

from asm_templates import (
    elementwise_add_asm,
    embedding_asm,
    ffn_asm,
    flash_attn_asm,
    gelu_asm,
    im2col_asm,
    im2col_asm_no_shift,
    layer_norm_asm,
    lm_head_asm,
    preload_addr_reg_asm,
    projection_asm,
    rms_norm_asm,
    silu_asm,
)
from asm_templates._imm import load_large_int as _load_large_int

# Imported from the module rather than the package: adding Mamba-2 to
# asm_templates/__init__.py is out of scope for this change, and a direct module
# import is equivalent for the caller.
from asm_templates.mamba_conv1d_asm import mamba_conv1d_asm, mamba_ssd_scan_asm


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
; Q, K, V projections + attention.  RoPE={"off" if not causal_mask else "on Q/K"}.
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
        const_one_fp_address=scheduler.get("memory_layout", {})
        .get("fp_sram", {})
        .get("silu_one", scheduler.get("memory_layout", {}).get("fp_sram", {}).get("silu_e", 3)),
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
    input_hbm_base_addr_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("token_table_offset", 1)

    code = f"""
; === Conv2d patch embedding (lowered to im2col + matmul) ===
; in_channels={in_channels}, out_channels={out_channels}
; image={image_size}x{image_size}, patch={patch_size}x{patch_size}, num_patches={num_patches}
; im2col output shape: ({M}, {K_col})
"""

    conv_stride = dims.get("stride", patch_size)

    # Check whether every output column produces a 64-aligned HBM pixel
    # offset.  When it does, the fast V_SHFT_V template can be used;
    # otherwise fall back to the no-shift (basis-vector) template which
    # handles arbitrary alignment via per-element extraction.
    _all_cols_aligned = all((ow * conv_stride) % 64 == 0 for ow in range(OW))

    # Step 1: im2col
    if _all_cols_aligned:
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
            stride=conv_stride,
        )
    else:
        # Fall back to no-shift template: tolerates non-64-aligned pixel
        # columns by loading from the aligned row base and offsetting the
        # basis-vector index.
        # basis_vram_base sits after the mask area; temp_vram sits after basis vectors.
        _max_intra = max((ow * conv_stride) % 64 for ow in range(OW))
        _num_basis = len({(ow * conv_stride) % 64 + kc for ow in range(OW) for kc in range(patch_size)})
        basis_vram_base = mask_vec_vram_addr + vlen  # after mask row
        temp_vram_addr = basis_vram_base + _num_basis * vlen
        code += im2col_asm_no_shift(
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
            basis_vram_base=basis_vram_base,
            scratch_vram_addr=scratch_vram_addr,
            temp_vram_addr=temp_vram_addr,
            output_vram_base=output_vram_base,
            stride=conv_stride,
        )

    # Step 2: matmul against the Conv2d weight (C_out, K_col).
    # Reuse projection_asm with out_features=C_out.
    w_base_hbm_offset_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("q_weight_offset", 2)
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

    w_base_hbm_offset_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("q_weight_offset", 2)
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


# ---------------------------------------------------------------------------
# Mamba-2 (selective state-space) lowering
# ---------------------------------------------------------------------------

#: HBM address registers the Mamba-2 path reuses.  A Mamba-2 program has no
#: attention and no gated FFN, so the q/k/v and ffn_* registers assigned by
#: generator/scheduler/reg_assignment_lib.json are dead for it.  Rather than
#: invent register indices (only a0-a7 exist), the Mamba weights and the two SSD
#: operand-spill areas are mapped onto those free slots.  The names below say
#: what each one actually holds in a Mamba-2 program.
_MAMBA_ADDR_REG_ROLES = {
    "token_table": "token_table_offset",
    "in_proj": "q_weight_offset",
    "conv1d": "k_weight_offset",
    "out_proj": "v_weight_offset",
    "ssd_act_spill": "ffn_gate_offset",
    "ssd_wt_spill": "ffn_up_offset",
}


def _mamba_shape_from_graph(symbolic_graph: dict[str, Any]) -> dict[str, Any] | None:
    """Return the Mamba-2 shape carried by a graph, or None for other families.

    The parser stamps it on the graph *and* on every mixer node, so a graph
    assembled by hand (as tests do) still resolves.
    """
    shape = symbolic_graph.get("mamba_shape")
    if shape:
        return shape
    for node in symbolic_graph.get("nodes", []):
        candidate = node.get("dimensions", {}).get("mamba_shape")
        if candidate:
            return candidate
    return None


def _mamba_vram_regions(shape: dict[str, Any], vlen: int) -> dict[str, int]:
    """Bump-allocate one layer's VRAM regions as VLEN-aligned element addresses.

    Every emitter for a given layer derives its addresses from this one function,
    so the conv output the SSD reads is by construction the conv output the conv
    wrote.  Regions are laid out with the sequence on the row axis (see
    ``asm_templates/mamba_conv1d_asm``'s layout contract).

    ``x``/``B``/``C``/``dt``/``z`` are column slices of the fused ``in_proj``
    output and get their own regions here.  That is not a copy the ISA would have
    to make: ``projection_asm`` writes its result one BLEN-wide output column
    block at a time via ``M_MM_WO``, so retargeting the destination per slice
    lands them in separate regions for free.  The generator does not model that
    retargeting, which is the one place the addresses below diverge from a
    numerically executable program.

    KNOWN LIMIT: this keeps the whole sequence resident, so ``_end`` exceeds the
    physical Vector SRAM (``VECTOR_SRAM_DEPTH * VLEN`` = 64 Ki elements) for any
    realistic seq_len.  The generator's other paths address VRAM the same
    unbounded way, and the ASM still assembles because these are element
    addresses rather than encoded immediates -- but a lowering meant to *run*
    would have to tile the sequence, which is exactly what ``chunk_size`` exists
    to enable and what this structural model does not yet do.
    """
    tokens = shape["batch_size"] * shape["seq_len"]
    hidden = shape["hidden_size"]
    d_inner = shape["d_inner"]
    conv_dim = shape["conv_dim"]
    in_proj_out = shape["in_proj_out"]
    state = shape["state_size"]
    heads = shape["num_heads"]
    head_dim = shape["head_dim"]
    groups = shape["n_groups"]
    chunk = shape["chunk_size"]
    conv_kernel = shape["conv_kernel"]

    regions: dict[str, int] = {}
    cursor = 0

    def alloc(name: str, size: int) -> None:
        nonlocal cursor
        regions[name] = cursor
        padded = max(size, vlen)
        cursor += ((padded + vlen - 1) // vlen) * vlen

    alloc("residual", tokens * hidden)  # layer input, also the residual operand
    alloc("in_proj", tokens * in_proj_out)  # fused [z, x, B, C, dt]
    alloc("conv_w", (conv_kernel + 1) * conv_dim)  # conv_kernel tap rows + bias row
    alloc("conv_in", tokens * conv_dim)  # the [x, B, C] slices of in_proj
    alloc("conv_out", tokens * conv_dim)
    alloc("z", tokens * d_inner)  # gate, consumed by the gated RMSNorm
    alloc("x", tokens * d_inner)
    alloc("B", tokens * groups * state)
    alloc("C", tokens * groups * state)
    alloc("dt", tokens * heads)
    alloc("decay", tokens * heads)
    alloc("score", heads * chunk * chunk)
    alloc("state", heads * state * head_dim)  # carried across chunks
    alloc("y", tokens * d_inner)
    alloc(
        "scratch",
        max(tokens * conv_dim, tokens * d_inner, heads * chunk * chunk, heads * state * head_dim),
    )
    # Single-row temporaries.  Every VRAM-destination operand needs a live
    # address register pointing at a real row; giving the one-row temporaries
    # their own regions keeps them from aliasing the multi-row "scratch" buffer
    # that the SSD GEMMs write.
    alloc("row0", vlen)
    alloc("row1", vlen)
    alloc("out", tokens * hidden)
    regions["_end"] = cursor
    return regions


def _mamba_fp_slots(scheduler: dict[str, Any]) -> dict[str, int]:
    """FP_MEM slot indices for the Mamba-2 scalar constants.

    ``generator/scheduler/mem_layout_lib.json`` has no Mamba entries, so these
    are allocated immediately after the highest named slot rather than
    hardcoded -- adding a slot to the JSON later shifts these instead of
    colliding with them.  The host must seed them the same way it seeds
    ``attn_scale`` / ``eps``.
    """
    fp_sram = scheduler.get("memory_layout", {}).get("fp_sram", {})
    named = [v for v in fp_sram.values() if isinstance(v, int)]
    base = (max(named) + 1) if named else 6
    return {
        "dt_min": base,
        "dt_max": base + 1,
        "a_decay": base + 2,
        "d_skip": base + 3,
        "one": fp_sram.get("silu_one", fp_sram.get("silu_e", 3)),
        "eps": fp_sram.get("eps", 1),
        "reci_group": fp_sram.get("hid_reciprocal", 2),
    }


def _mamba_addr_reg(scheduler: dict[str, Any], role: str, default: int) -> int:
    """Resolve the HBM address register a Mamba-2 tensor is addressed through."""
    hbm_addr_reg = scheduler.get("register_assignment", {}).get("hbm_addr_reg", {})
    return hbm_addr_reg.get(_MAMBA_ADDR_REG_ROLES[role], default)


def _prefetch_vram_asm(
    *,
    vram_base: int,
    elements: int,
    hbm_addr_reg: int,
    regs: list[int],
    vlen: int,
    prefetch_amount: int,
    label: str,
) -> str:
    """H_PREFETCH_V loop bringing ``elements`` elements from HBM into VRAM."""
    per_issue = max(1, prefetch_amount) * vlen
    issues = math.ceil(elements / per_issue) if elements > 0 else 0
    if issues == 0:
        return ""
    dst, off, loop = regs[0], regs[1], regs[2]
    lines = [f"; load {label}: {elements} elements from HBM[a{hbm_addr_reg}] ({issues} x H_PREFETCH_V)"]
    lines.extend(_load_large_int(dst, vram_base))
    lines.append(f"S_ADDI_INT gp{off}, gp0, 0")
    lines.append(f"C_LOOP_START gp{loop}, {issues}")
    lines.append(f"H_PREFETCH_V gp{dst}, gp{off}, a{hbm_addr_reg}, 1, 0")
    lines.append(f"S_ADDI_INT gp{dst}, gp{dst}, {per_issue}")
    lines.append(f"S_ADDI_INT gp{off}, gp{off}, {per_issue}")
    lines.append(f"C_LOOP_END gp{loop}")
    return "\n".join(lines) + "\n"


def _mamba_rowwise_asm(
    *,
    body: list[str],
    ptr_regs: list[int],
    rows: int,
    loop_reg: int,
    vlen: int,
    comment: str,
) -> str:
    """Row loop helper for the code_gen-side Mamba vector stages."""
    if rows <= 0:
        return ""
    lines = [f"; {comment} ({rows} rows)", f"C_LOOP_START gp{loop_reg}, {rows}"]
    lines.extend(body)
    for reg in ptr_regs:
        lines.append(f"S_ADDI_INT gp{reg}, gp{reg}, {vlen}")
    lines.append(f"C_LOOP_END gp{loop_reg}")
    return "\n".join(lines) + "\n"


def _generate_projection_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly for a plain linear layer (Mamba-2 in_proj / out_proj).

    Reuses ``projection_asm`` unchanged: in_proj and out_proj are ordinary dense
    GEMMs against a static weight, which is exactly what that template emits.
    The GEMM's batch axis is the token axis (``batch_size * seq_len``), not the
    model batch -- a Mamba-2 prefill projects every token in one pass.
    """
    dims = node["dimensions"]
    in_features = dims["in_features"]
    out_features = dims["out_features"]
    role = dims.get("role", "projection")

    mlen = hardware_config.get("MLEN", 64)
    blen = hardware_config.get("BLEN", 4)
    vlen = hardware_config.get("VLEN", 64)
    vsram = scheduler["memory_layout"].get("vector_sram_addr", {})

    shape = dims.get("mamba_shape")
    if shape is not None:
        regions = _mamba_vram_regions(shape, vlen)
        batch = shape["batch_size"] * shape["seq_len"]
        if role == "mamba_out_proj":
            activation_base = regions["y"]
            result_base = regions["out"]
            w_reg = _mamba_addr_reg(scheduler, "out_proj", 4)
        else:
            activation_base = regions["residual"]
            result_base = regions["in_proj"]
            w_reg = _mamba_addr_reg(scheduler, "in_proj", 2)
        scratch_base = regions["scratch"]
    else:
        regions = None
        batch = model_info.get("batch", 1)
        activation_base = vsram.get("block1", 0)
        result_base = vsram.get("block2", 0)
        scratch_base = vsram.get("k_split_scratch", vsram.get("block4", 0))
        w_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("q_weight_offset", 2)

    code = f"""
; Linear projection ({role}): ({batch}, {in_features}) @ ({in_features}, {out_features})
"""
    if out_features % blen:
        code += (
            f"; NOTE: out_features={out_features} is not a multiple of BLEN={blen}; "
            f"projection_asm emits whole BLEN column blocks, so the final "
            f"{out_features % blen} column(s) are not covered.\n"
        )
    code += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=in_features,
        alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
        w_base_hbm_offset_reg=w_reg,
        activation_base_address=activation_base,
        result_base_address=result_base,
        rope_enabled=False,
        out_features=out_features,
        matrix_sram_size=hardware_config.get("MATRIX_SRAM_SIZE", 1024),
        scratch_base_address=scratch_base,
        vlen=vlen,
    )
    return code.strip()


def _generate_conv1d_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly for the Mamba-2 causal depthwise conv1d + its SiLU.

    Deliberately not routed through ``im2col_asm``/``_no_shift``: those hardcode a
    square KxK patch, have no padding (so they cannot express the causal left
    pad), and would build a dense ``(M, C_in*K*K)`` GEMM against a block-diagonal
    weight.  ``mamba_conv1d_asm`` instead exploits the sequence-on-rows layout,
    where the causal shift is a row-address offset and each tap is a VLEN-wide
    ``V_MUL_VV`` operand.
    """
    dims = node["dimensions"]
    shape = dims["mamba_shape"]
    vlen = hardware_config.get("VLEN", 64)
    regions = _mamba_vram_regions(shape, vlen)
    conv_dim = dims["conv_dim"]
    conv_kernel = dims["kernel_size"]
    tokens = shape["batch_size"] * shape["seq_len"]
    use_bias = dims.get("use_conv_bias", True)
    fp_slots = _mamba_fp_slots(scheduler)

    code = f"""
; === Mamba-2 depthwise causal conv1d ===
; channels={conv_dim} (groups={dims["groups"]}), kernel={conv_kernel}, padding={dims["padding"]} (causal)
; Operates on the [x, B, C] slices of in_proj only; z and dt bypass the conv.
"""
    # The conv weight is a static per-layer parameter: taps then (optionally) bias.
    code += _prefetch_vram_asm(
        vram_base=regions["conv_w"],
        elements=(conv_kernel + (1 if use_bias else 0)) * conv_dim,
        hbm_addr_reg=_mamba_addr_reg(scheduler, "conv1d", 3),
        regs=[1, 2, 3],
        vlen=vlen,
        prefetch_amount=hardware_config.get("HBM_V_Prefetch_Amount", 16),
        label="conv1d taps + bias",
    )
    code += mamba_conv1d_asm(
        vlen=vlen,
        seq_len=tokens,
        conv_dim=conv_dim,
        conv_kernel=conv_kernel,
        alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
        input_base_address=regions["conv_in"],
        output_base_address=regions["conv_out"],
        weight_base_address=regions["conv_w"],
        scratch_base_address=regions["row0"],
        bias_base_address=(regions["conv_w"] + conv_kernel * conv_dim) if use_bias else None,
    )

    activation = dims.get("activation", "silu")
    if activation == "silu":
        code += f"\n; -- {activation} on the conv output (fused: Mamba-2 never separates them) --\n"
        code += silu_asm(
            const_one_fp_address=fp_slots["one"],
            alive_registers=[1, 2, 3],
            activation_base_address=regions["conv_out"],
            # silu_asm holds its sigmoid in a single reused row, so one row is enough.
            scratchpad_base_address=regions["row1"],
            vlen=vlen,
            batch_size=tokens,
            hidden_dim=conv_dim,
        )
    else:
        code += f"\n; -- {activation} activation (unrecognized; no ASM emitted) --\n"
    return code.strip()


def _generate_ssd_scan_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly for the chunked state-space-duality scan."""
    dims = node["dimensions"]
    shape = dims["mamba_shape"]
    vlen = hardware_config.get("VLEN", 64)
    regions = _mamba_vram_regions(shape, vlen)
    fp_slots = _mamba_fp_slots(scheduler)

    code = f"""
; === Mamba-2 SSD scan ===
; chunk_size={dims["chunk_size"]} -> {dims["num_chunks"]} chunk(s) over seq_len={dims["seq_len"]}
; time_step_limit=({dims["time_step_min"]}, {dims["time_step_max"]})
; The x / B / C / dt operands are the corresponding column slices of the conv
; output; they are addressed as separate VRAM regions here (see
; _mamba_vram_regions for why that costs nothing on real hardware).
"""
    code += mamba_ssd_scan_asm(
        mlen=hardware_config.get("MLEN", 64),
        vlen=vlen,
        blen=hardware_config.get("BLEN", 4),
        seq_len=shape["batch_size"] * shape["seq_len"],
        chunk_size=dims["chunk_size"],
        num_heads=dims["num_heads"],
        head_dim=dims["head_dim"],
        state_size=dims["state_size"],
        n_groups=dims["n_groups"],
        alive_registers=[1, 2, 3, 4, 5, 6, 7, 8],
        vram=regions,
        act_spill_addr_reg=_mamba_addr_reg(scheduler, "ssd_act_spill", 6),
        wt_spill_addr_reg=_mamba_addr_reg(scheduler, "ssd_wt_spill", 7),
        writeback_amount=hardware_config.get("HBM_V_Writeback_Amount", 4),
        dt_min_fp_address=fp_slots["dt_min"],
        dt_max_fp_address=fp_slots["dt_max"],
        a_decay_fp_address=fp_slots["a_decay"],
        d_skip_fp_address=fp_slots["d_skip"],
    )
    return code.strip()


def _generate_gated_rmsnorm_code(
    node: dict[str, Any], model_info: dict[str, Any], hardware_config: dict[str, Any], scheduler: dict[str, Any]
) -> str:
    """Generate assembly for Mamba-2's gated RMSNorm: ``y = RMSNorm(y) * silu(z)``.

    Cannot reuse ``_generate_normalization_code``: Mamba-2 normalises over
    ``d_inner / n_groups``, not over ``hidden_size``, and the gate multiply has no
    counterpart in the plain norm node.  The RMSNorm body itself is
    ``rms_norm_asm`` with the group width as its ``hidden_dim`` and one "batch"
    row per (token, group).
    """
    dims = node["dimensions"]
    shape = dims["mamba_shape"]
    vlen = hardware_config.get("VLEN", 64)
    regions = _mamba_vram_regions(shape, vlen)
    fp_slots = _mamba_fp_slots(scheduler)

    group_size = dims["group_size"]
    n_groups = dims["n_groups"]
    d_inner = dims["normalized_shape"]
    tokens = shape["batch_size"] * shape["seq_len"]

    code = f"""
; === Mamba-2 gated RMSNorm ===
; normalize over {group_size} (= d_inner {d_inner} / n_groups {n_groups}), then
; multiply by {dims["gate_activation"]}(z).  1/group_size must be seeded into the
; FP_MEM slot below -- the plain-norm path seeds 1/hidden_size there instead.
"""
    code += rms_norm_asm(
        _eps_offset=fp_slots["eps"],
        reci_hid_offset=fp_slots["reci_group"],
        alive_registers=[1, 2, 3],
        activation_base_address=regions["y"],
        scratchpad_base_address=regions["row0"],
        vlen=vlen,
        batch_size=tokens * n_groups,
        hidden_dim=group_size,
    )
    code += f"\n; -- gate = {dims['gate_activation']}(z), computed in place on the z slice --\n"
    code += silu_asm(
        const_one_fp_address=fp_slots["one"],
        alive_registers=[1, 2, 3],
        activation_base_address=regions["z"],
        scratchpad_base_address=regions["row1"],
        vlen=vlen,
        batch_size=tokens,
        hidden_dim=d_inner,
    )
    code += "\n; -- y *= gate --\n"
    code += "".join(line + "\n" for line in _load_large_int(1, regions["y"]))
    code += "".join(line + "\n" for line in _load_large_int(2, regions["z"]))
    code += _mamba_rowwise_asm(
        body=["V_MUL_VV gp1, gp1, gp2, 0"],
        ptr_regs=[1, 2],
        rows=max(1, (tokens * d_inner) // vlen),
        loop_reg=3,
        vlen=vlen,
        comment="gate multiply",
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
    elif operation_type == "projection":
        return header + _generate_projection_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "conv1d":
        return header + _generate_conv1d_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "ssd_scan":
        return header + _generate_ssd_scan_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "gated_rmsnorm":
        return header + _generate_gated_rmsnorm_code(node, model_info, hardware_config, scheduler)
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


def _weight_hbm_bytes(rows: int, cols: int, hardware_config: dict) -> int:
    """Compute MXFP8 HBM byte size for a (rows, cols) weight tensor.

    Format: data blocks + scale blocks, each padded to HBM row width.
    Matches the byte output of tools/memory_mapping/memory_map.py's
    map_mx_data_to_hbm_for_behave_sim.
    """
    block_dim = hardware_config.get("block_dim", 4)
    assert cols % block_dim == 0, f"cols ({cols}) must be a multiple of block_dim ({block_dim})"
    wt_block_width = hardware_config.get("wt_block_width", 32)
    scale_width = hardware_config.get("scale_width", 8)
    return rows * (cols // block_dim) * ((wt_block_width // 8) + (scale_width // 8))


def _mamba_addr_reg_layout(
    model_info: dict[str, Any],
    shape: dict[str, Any],
    hardware_config: dict[str, Any],
) -> list[tuple[str, int, int, str | None]]:
    """HBM weight layout for a Mamba-2 program, as (name, rows, cols, reg_key).

    Mamba-2 has no q/k/v and no gated FFN, so the register keys those names map
    to in reg_assignment_lib.json are reused here for the tensors a Mamba-2
    program actually needs (see ``_MAMBA_ADDR_REG_ROLES``).  The last two entries
    are not weights at all but the two SSD operand-spill areas -- their size is
    the largest single operand any of the four per-chunk GEMMs spills.
    """
    block_dim = hardware_config.get("block_dim", 4)

    def pad(cols: int) -> int:
        # _weight_hbm_bytes requires whole MX blocks; a Mamba width that is not a
        # multiple of block_dim (e.g. in_proj_out with num_heads % block_dim != 0)
        # is rounded up rather than tripping its assert.
        return ((cols + block_dim - 1) // block_dim) * block_dim

    vocab_size = model_info.get("vocab_size", 1024)
    hidden = shape["hidden_size"]
    d_inner = shape["d_inner"]
    conv_dim = shape["conv_dim"]
    in_proj_out = shape["in_proj_out"]
    chunk = shape["chunk_size"]
    state = shape["state_size"]
    heads = shape["num_heads"]
    head_dim = shape["head_dim"]
    groups = shape["n_groups"]

    spill_elements = max(
        groups * chunk * state,
        heads * chunk * chunk,
        heads * chunk * head_dim,
        heads * chunk * state,
        heads * state * head_dim,
    )
    spill_rows = max(1, math.ceil(spill_elements / pad(max(head_dim, block_dim))))

    return [
        ("token_table", vocab_size, pad(hidden), _MAMBA_ADDR_REG_ROLES["token_table"]),
        ("in_proj_weight", hidden, pad(in_proj_out), _MAMBA_ADDR_REG_ROLES["in_proj"]),
        # conv_kernel tap rows + one bias row, conv_dim wide.
        ("conv1d_weight", shape["conv_kernel"] + 1, pad(conv_dim), _MAMBA_ADDR_REG_ROLES["conv1d"]),
        ("out_proj_weight", d_inner, pad(hidden), _MAMBA_ADDR_REG_ROLES["out_proj"]),
        ("ssd_act_spill", spill_rows, pad(max(head_dim, block_dim)), _MAMBA_ADDR_REG_ROLES["ssd_act_spill"]),
        ("ssd_wt_spill", spill_rows, pad(max(head_dim, block_dim)), _MAMBA_ADDR_REG_ROLES["ssd_wt_spill"]),
    ]


def _generate_addr_reg_init(
    model_info: dict[str, Any],
    hardware_config: dict[str, Any],
    scheduler: dict[str, Any],
    mamba_shape: dict[str, Any] | None = None,
) -> str:
    """Emit C_SET_ADDR_REG instructions for all weight HBM address registers.

    Computes cumulative MXFP8 byte offsets matching the HBM layout that
    _build_hbm_from_hf_weights writes.  Must be emitted before any node
    that references HBM weights.

    When ``mamba_shape`` is supplied the Mamba-2 layout is used instead of the
    attention + FFN one; emitting the latter for a Mamba-2 program would reserve
    HBM for q/k/v/ffn tensors that do not exist and leave the registers the
    Mamba nodes actually read pointing into the wrong regions.
    """
    hidden_size = model_info.get("hidden_size", 64)
    intermediate_size = model_info.get("intermediate_size", 256)
    vocab_size = model_info.get("vocab_size", 1024)
    num_heads = model_info.get("num_attention_heads", 4)
    num_kv_heads = model_info.get("num_key_value_heads", num_heads)
    head_dim = model_info.get("head_dim", hidden_size // num_heads)

    hbm_addr_reg = scheduler["register_assignment"].get("hbm_addr_reg", {})

    if mamba_shape is not None:
        layout = _mamba_addr_reg_layout(model_info, mamba_shape, hardware_config)
        return _emit_addr_reg_init(layout, hardware_config, scheduler)

    # Weight layout: ordered list of (name, rows, cols, addr_reg_key).
    # addr_reg_key is None for weights without a dedicated register.
    layout = [
        ("token_table", vocab_size, hidden_size, "token_table_offset"),
        ("q_weight", hidden_size, num_heads * head_dim, "q_weight_offset"),
        ("k_weight", hidden_size, num_kv_heads * head_dim, "k_weight_offset"),
        ("v_weight", hidden_size, num_kv_heads * head_dim, "v_weight_offset"),
        ("o_weight", num_heads * head_dim, hidden_size, None),  # no addr reg
        ("ffn_gate", hidden_size, intermediate_size, "ffn_gate_offset"),
        ("ffn_up", hidden_size, intermediate_size, "ffn_up_offset"),
        ("ffn_down", intermediate_size, hidden_size, "ffn_down_offset"),
    ]
    _ = hbm_addr_reg  # resolved inside _emit_addr_reg_init
    return _emit_addr_reg_init(layout, hardware_config, scheduler)


def _emit_addr_reg_init(
    layout: list[tuple[str, int, int, str | None]],
    hardware_config: dict[str, Any],
    scheduler: dict[str, Any],
) -> str:
    """Turn an ordered HBM tensor layout into C_SET_ADDR_REG initialisation.

    Shared by the attention and Mamba-2 layouts so the two cannot drift in how
    they compute cumulative MXFP8 byte offsets.
    """
    hbm_addr_reg = scheduler["register_assignment"].get("hbm_addr_reg", {})

    # Compute cumulative offsets
    offset = 0
    addr_regs_to_set: list[int] = []
    addr_reg_vals: list[int] = []

    for name, rows, cols, reg_key in layout:
        if reg_key is not None:
            reg_idx = hbm_addr_reg.get(reg_key, 0)
            if reg_idx > 0:  # skip if reg not assigned
                addr_regs_to_set.append(reg_idx)
                addr_reg_vals.append(offset)
        size = _weight_hbm_bytes(rows, cols, hardware_config)
        offset += size

    if not addr_regs_to_set:
        return ""

    code = "\n; --- HBM address register initialization ---\n"
    code += f"; Total HBM weight footprint: {offset} bytes ({offset / 1024:.1f} KiB)\n"
    code += preload_addr_reg_asm(
        addr_reg_to_set=addr_regs_to_set,
        available_registers=[9, 10, 11, 12, 13, 14, 15][: len(addr_regs_to_set)],
        addr_reg_val=addr_reg_vals,
    )
    return code


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

    # Initialize HBM address registers with weight byte offsets.  A Mamba-2 graph
    # needs a different HBM layout (no q/k/v, no FFN), detected from the graph
    # itself rather than from model_info so callers need no new contract.
    mamba_shape = _mamba_shape_from_graph(symbolic_graph)
    addr_reg_init = _generate_addr_reg_init(model_info, hardware_config, scheduler, mamba_shape=mamba_shape)
    if addr_reg_init:
        asm_code.append(addr_reg_init)

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
