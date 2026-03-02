"""
vlm_codegen.py — ASM code generation from VLMModelParser.trace_leaf_modules() output.

Design
------
trace_leaf_modules() returns a call-tree whose nodes carry:

    name           – dotted module path  ("language_model.layers.0.self_attn")
    type           – module class name   ("Qwen3VLTextAttention")
    attrs          – static hyperparams  {"hidden_size": 2048, "eps": 1e-6, …}
    in             – list of runtime input shapes
    out            – runtime output shape(s)
    weights        – direct parameter shapes [{"name": "weight", "shape": [N, M]}, …]
    in_syms        – SSA symbol names for each input tensor  ["%12", …]
    out_syms       – SSA symbol names for each output tensor ["%13"]
    in_sym_sources – {sym: producer_module_name}

vlm_codegen() flattens this tree, iterates in execution order, and dispatches
each node to an ASM-template call based on its type.

Granularity rule
----------------
Composite semantic modules (Attention, MLP, RMSNorm, Embedding, elementwise_add)
generate code directly.  Sub-modules inside a composite (e.g. the individual
Linear layers inside Qwen3VLTextAttention) are automatically skipped to avoid
double-counting — once a node is processed its entire subtree is "covered".

Usage
-----
    python vlm_codegen.py          # runs __main__ demo with Qwen3-VL-2B-Instruct
"""

import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup: make vlm_parser, plena_backend, and asm_templates importable
# ---------------------------------------------------------------------------
_PARSER_DIR   = Path(__file__).parent
_GENERATOR_DIR = _PARSER_DIR.parent
_PROJECT_ROOT  = _GENERATOR_DIR.parent

for _p in [str(_PARSER_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vlm_parser import VLMModelParser, flatten_call_tree, template_qwen3_vl_inputs  # noqa: E402
from plena_backend import MODULE_TYPE_TO_TEMPLATE  # noqa: E402

# ---------------------------------------------------------------------------
# ASM template imports — fall back to no-op stubs when not on path
# ---------------------------------------------------------------------------
try:
    from asm_templates import (  # type: ignore[import]
        elementwise_add_asm,
        embedding_asm,
        ffn_asm,
        flash_attn_asm,
        projection_asm,
        rms_norm_asm,
        layer_norm_asm,
        batched_matmul_asm,
    )
    _TEMPLATES_OK = True
except ImportError:
    _TEMPLATES_OK = False

    def _stub_asm(template_name: str):
        def _fn(*args, **kwargs) -> str:
            return f"; [stub] {template_name} — asm_templates not on PYTHONPATH\n"
        _fn.__name__ = template_name
        return _fn

    elementwise_add_asm = _stub_asm("elementwise_add_asm")
    embedding_asm       = _stub_asm("embedding_asm")
    ffn_asm             = _stub_asm("ffn_asm")
    flash_attn_asm      = _stub_asm("flash_attn_asm")
    projection_asm      = _stub_asm("projection_asm")
    rms_norm_asm        = _stub_asm("rms_norm_asm")
    layer_norm_asm      = _stub_asm("layer_norm_asm")
    batched_matmul_asm  = _stub_asm("batched_matmul_asm")

# ---------------------------------------------------------------------------
# Default hardware configuration and scheduler
# These mirror the PLENA accelerator defaults used in code_gen.py.
# Pass custom dicts to vlm_codegen() to override.
# ---------------------------------------------------------------------------
DEFAULT_HW: dict[str, Any] = {
    "mlen": 16,
    "blen": 16,
    "vlen": 16,
    "alive_registers": [1, 2, 3, 4, 5, 6, 7, 8],
}

DEFAULT_SCHED: dict[str, Any] = {
    "activation_base_address": 0x0000,
    "memory_layout": {
        "vector_sram_addr": {
            "block1": 0x0000,
            "block2": 0x1000,
            "block3": 0x2000,
            "block5": 0x3000,
        },
        "fp_sram": {
            "silu_e":         0x0100,
            "eps":            0x0200,
            "hid_reciprocal": 0x0300,
        },
    },
    "register_assignment": {
        "hbm_addr_reg": {
            "token_table_offset":         1,
            "q_weight_offset":            2,
            "k_weight_offset":            3,
            "v_weight_offset":            4,
            "ffn_weight_offset":          5,
            "rope_params_offset":         6,
            "previous_activation_offset": 7,
        }
    },
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hw(cfg: dict, key: str, default: Any) -> Any:
    return cfg.get(key, default)


def _reg(sched: dict, name: str, default: int = 0) -> int:
    return (
        sched.get("register_assignment", {})
             .get("hbm_addr_reg", {})
             .get(name, default)
    )


def _mem(sched: dict, block: str, default: int = 0) -> int:
    return (
        sched.get("memory_layout", {})
             .get("vector_sram_addr", {})
             .get(block, default)
    )


def _fp_mem(sched: dict, key: str, default: int = 0) -> int:
    return (
        sched.get("memory_layout", {})
             .get("fp_sram", {})
             .get(key, default)
    )


def _in_shape(node: dict, idx: int = 0) -> list[int] | None:
    """Return the shape list of the idx-th input tensor, or None."""
    in_list = node.get("in") or []
    if idx < len(in_list) and in_list[idx]:
        return in_list[idx]["shape"]
    return None


def _out_shape(node: dict) -> list[int] | None:
    """Return the shape list of the first output tensor, or None."""
    out = node.get("out")
    if isinstance(out, dict):
        return out.get("shape")
    if isinstance(out, list):
        for sh in out:
            if sh is not None:
                return sh["shape"]
    return None


def _batch_from_node(node: dict, model_info: dict) -> int:
    sh = _in_shape(node)
    if sh and len(sh) >= 1:
        return sh[0]
    return model_info.get("batch_size", 1)


def _hidden_from_node(node: dict, model_info: dict) -> int:
    """Best-effort hidden size: attrs first, then last dim of input shape."""
    attrs = node.get("attrs") or {}
    if "hidden_size" in attrs:
        return attrs["hidden_size"]
    sh = _in_shape(node)
    if sh:
        return sh[-1]
    return model_info.get("hidden_size", 2048)

# ---------------------------------------------------------------------------
# Per-type codegen handlers
# Each returns a raw ASM string (no surrounding header/footer).
# ---------------------------------------------------------------------------

def _codegen_embedding(node: dict, hw: dict, sched: dict, model_info: dict) -> str:
    """Embedding lookup → embedding_asm().

    Context extracted from node:
        attrs["num_embeddings"]  → vocab / table size
        attrs["embedding_dim"]   → hidden size
        in[0]["shape"][0]        → batch size
    """
    attrs   = node.get("attrs") or {}
    n_emb   = attrs.get("num_embeddings", model_info.get("vocab_size", 32000))
    emb_dim = attrs.get("embedding_dim",  model_info.get("hidden_size", 2048))
    batch   = _batch_from_node(node, model_info)

    return embedding_asm(
        mlen=_hw(hw, "mlen", 16),
        blen=_hw(hw, "blen", 16),
        batch=batch,
        hidden_size=emb_dim,
        alive_registers=_hw(hw, "alive_registers", [1, 2, 3, 4]),
        voc_table_row_size=n_emb,
        activation_base_address=sched.get("activation_base_address", 0),
        voc_table_base_addr_reg_index=_reg(sched, "token_table_offset", 1),
        input_ids=[1 for _ in range(batch)],
    )


def _codegen_rms_norm(node: dict, hw: dict, sched: dict, model_info: dict) -> str:
    """RMS normalisation → rms_norm_asm().

    Context extracted from node:
        attrs["hidden_size"]  → hidden dimension (set by _static_attrs for Qwen3VLTextRMSNorm)
        in[0]["shape"][0]     → batch size
    """
    hidden = _hidden_from_node(node, model_info)
    batch  = _batch_from_node(node, model_info)

    return rms_norm_asm(
        _eps_offset=_fp_mem(sched, "eps", 0),
        reci_hid_offset=_fp_mem(sched, "hid_reciprocal", 0),
        alive_registers=_hw(hw, "alive_registers", [1, 2, 3, 4])[:3],
        activation_base_address=_mem(sched, "block1", 0),
        scratchpad_base_address=_mem(sched, "block2", 0),
        vlen=_hw(hw, "vlen", 16),
        batch_size=batch,
        hidden_dim=hidden,
    )


def _codegen_layer_norm(node: dict, hw: dict, sched: dict, model_info: dict) -> str:
    """LayerNorm → rms_norm_asm() (close enough for the dummy pass).

    Context extracted from node:
        attrs["normalized_shape"][0] → hidden dimension
    """
    attrs  = node.get("attrs") or {}
    ns     = attrs.get("normalized_shape", [model_info.get("hidden_size", 2048)])
    hidden = ns[0] if isinstance(ns, list) else ns
    batch  = _batch_from_node(node, model_info)

    return rms_norm_asm(
        _eps_offset=_fp_mem(sched, "eps", 0),
        reci_hid_offset=_fp_mem(sched, "hid_reciprocal", 0),
        alive_registers=_hw(hw, "alive_registers", [1, 2, 3, 4])[:3],
        activation_base_address=_mem(sched, "block1", 0),
        scratchpad_base_address=_mem(sched, "block2", 0),
        vlen=_hw(hw, "vlen", 16),
        batch_size=batch,
        hidden_dim=hidden,
    )


def _codegen_text_attention(node: dict, hw: dict, sched: dict, model_info: dict) -> str:
    """Text self-attention → 3 × projection_asm() (Q, K, V) + flash_attn_asm().

    Context extracted from node:
        in[0]["shape"][-1]         → hidden_size
        model_info["head_dim"]     → head dimension
    """
    sh          = _in_shape(node)
    hidden_size = sh[-1] if sh else model_info.get("hidden_size", 2048)
    head_dim    = model_info.get("head_dim", hidden_size // model_info.get("num_attention_heads", 16))
    batch       = sh[0] if sh else model_info.get("batch_size", 1)
    regs        = _hw(hw, "alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])

    _proj_common = dict(
        mlen=_hw(hw, "mlen", 16),
        blen=_hw(hw, "blen", 16),
        batch=batch,
        hidden_size=hidden_size,
        alive_registers=regs,
        head_dim=head_dim,
        rope_hbm_offset_reg=_reg(sched, "rope_params_offset", 6),
        rope_on_chip_address=_mem(sched, "block3", 0),
        activation_base_address=_mem(sched, "block1", 0),
        result_base_address=_mem(sched, "block2", 0),
    )

    code  = "; --- Q projection\n"
    code += projection_asm(**_proj_common,
                           w_base_hbm_offset_reg=_reg(sched, "q_weight_offset", 2),
                           rope_enabled=True)
    code += "; --- K projection\n"
    code += projection_asm(**_proj_common,
                           w_base_hbm_offset_reg=_reg(sched, "k_weight_offset", 3),
                           rope_enabled=True)
    code += "; --- V projection\n"
    code += projection_asm(**_proj_common,
                           w_base_hbm_offset_reg=_reg(sched, "v_weight_offset", 4),
                           rope_enabled=False)
    code += "; --- Flash Attention (TODO: wire flash_attn_asm)\n"
    # code += flash_attn_asm(...)   # TODO: finalize flash_attn_asm signature
    return code


def _codegen_vision_attention(node: dict, hw: dict, sched: dict, model_info: dict) -> str:
    """Vision self-attention → same pattern as text attention without RoPE.

    Vision tokens are packed as [num_tokens, hidden_size], so seq_len = in[0]["shape"][0].
    """
    sh          = _in_shape(node)
    hidden_size = sh[-1] if sh else model_info.get("vision_hidden_size", model_info.get("hidden_size", 2048))
    head_dim    = model_info.get("vision_head_dim", hidden_size // model_info.get("num_attention_heads", 16))
    batch       = 1   # vision tokens already merged; treat as batch=1
    regs        = _hw(hw, "alive_registers", [1, 2, 3, 4, 5, 6, 7, 8])

    _proj_common = dict(
        mlen=_hw(hw, "mlen", 16),
        blen=_hw(hw, "blen", 16),
        batch=batch,
        hidden_size=hidden_size,
        alive_registers=regs,
        head_dim=head_dim,
        rope_hbm_offset_reg=_reg(sched, "rope_params_offset", 6),
        rope_on_chip_address=_mem(sched, "block3", 0),
        activation_base_address=_mem(sched, "block1", 0),
        result_base_address=_mem(sched, "block2", 0),
        rope_enabled=False,   # vision attention uses 2D RoPE handled externally
    )

    code  = "; --- Vision Q projection\n"
    code += projection_asm(**_proj_common, w_base_hbm_offset_reg=_reg(sched, "q_weight_offset", 2))
    code += "; --- Vision K projection\n"
    code += projection_asm(**_proj_common, w_base_hbm_offset_reg=_reg(sched, "k_weight_offset", 3))
    code += "; --- Vision V projection\n"
    code += projection_asm(**_proj_common, w_base_hbm_offset_reg=_reg(sched, "v_weight_offset", 4))
    code += "; --- Vision Flash Attention (TODO: wire flash_attn_asm)\n"
    return code


def _codegen_ffn(node: dict, hw: dict, sched: dict, model_info: dict) -> str:
    """FFN / MLP → ffn_asm().

    Context extracted from node:
        in[0]["shape"][-1]         → hidden_size
        model_info["intermediate_size"] → intermediate size
    """
    sh              = _in_shape(node)
    hidden_size     = sh[-1] if sh else model_info.get("hidden_size", 2048)
    inter_size      = model_info.get("intermediate_size", hidden_size * 4)
    batch           = sh[0] if sh else model_info.get("batch_size", 1)

    return ffn_asm(
        mlen=_hw(hw, "mlen", 16),
        blen=_hw(hw, "blen", 16),
        batch=batch,
        hidden_size=hidden_size,
        alive_registers=_hw(hw, "alive_registers", [1, 2, 3, 4, 5, 6]),
        weight_hbm_offset_reg=_reg(sched, "ffn_weight_offset", 5),
        intermediate_size=inter_size,
        activation_base_address=_mem(sched, "block1", 0),
        const_address=_fp_mem(sched, "silu_e", 0),
        result_base_address=_mem(sched, "block5", 0),
    )


def _codegen_linear(node: dict, hw: dict, sched: dict, model_info: dict) -> str:
    """Standalone Linear (e.g. PatchMerger or bare projection) → projection_asm().

    Context extracted from node:
        attrs["in_features"]   → input dim
        attrs["out_features"]  → output dim (used as head_dim here)
        in[0]["shape"][0]      → batch
    """
    attrs       = node.get("attrs") or {}
    in_features = attrs.get("in_features",  model_info.get("hidden_size", 2048))
    out_features = attrs.get("out_features", model_info.get("hidden_size", 2048))
    batch       = _batch_from_node(node, model_info)

    return projection_asm(
        mlen=_hw(hw, "mlen", 16),
        blen=_hw(hw, "blen", 16),
        batch=batch,
        hidden_size=in_features,
        alive_registers=_hw(hw, "alive_registers", [1, 2, 3, 4, 5, 6, 7, 8]),
        head_dim=out_features,
        w_base_hbm_offset_reg=_reg(sched, "q_weight_offset", 2),
        rope_hbm_offset_reg=_reg(sched, "rope_params_offset", 6),
        rope_on_chip_address=_mem(sched, "block3", 0),
        activation_base_address=_mem(sched, "block1", 0),
        result_base_address=_mem(sched, "block2", 0),
        rope_enabled=False,
    )


def _codegen_conv3d(node: dict, hw: dict, sched: dict, model_info: dict) -> str:
    """Conv3d (vision patch embed) → batched_matmul_asm().

    Context extracted from node:
        attrs["out_channels"]  → output features
        attrs["kernel_size"]   → patch size
        in[0]["shape"]         → video / image input shape
        out["shape"]           → patch token shape
    """
    attrs       = node.get("attrs") or {}
    out_ch      = attrs.get("out_channels", model_info.get("hidden_size", 2048))
    in_sh       = _in_shape(node)   # e.g. [batch, C, T, H, W]
    out_sh      = _out_shape(node)  # e.g. [batch, num_tokens, out_ch]

    return batched_matmul_asm(
        mlen=_hw(hw, "mlen", 16),
        blen=_hw(hw, "blen", 16),
        hidden_size=out_ch,
        alive_registers=_hw(hw, "alive_registers", [1, 2, 3, 4, 5, 6, 7, 8]),
        activation_base_address=_mem(sched, "block1", 0),
        result_base_address=_mem(sched, "block2", 0),
        weight_hbm_offset_reg=_reg(sched, "q_weight_offset", 2),
    )


def _codegen_elementwise_add(node: dict, hw: dict, sched: dict, model_info: dict) -> str:
    """Elementwise add (residual connection) → elementwise_add_asm().

    Context extracted from node:
        in[0]["shape"][-1]   → hidden_size  (may be None for FX nodes)
        in[0]["shape"][0]    → batch        (may be None for FX nodes)
    """
    sh          = _in_shape(node)
    hidden_size = (sh[-1] if sh else None) or model_info.get("hidden_size", 2048)
    batch       = (sh[0]  if sh else None) or model_info.get("batch_size", 1)

    return elementwise_add_asm(
        vlen=_hw(hw, "vlen", 16),
        hidden_size=hidden_size,
        batch=batch,
        alive_registers=_hw(hw, "alive_registers", [1, 2, 3, 4])[:3],
        stored_activation_base_address=_mem(sched, "block1", 0),
        previous_activation_base_address=_mem(sched, "block2", 0),
        previous_act_on_chip_addr_reg_index=_reg(sched, "previous_activation_offset", 7),
    )


def _codegen_unsupported(node: dict, hw: dict, sched: dict, model_info: dict) -> str:
    return f"; [unsupported] {node['type']} — no codegen handler registered\n"

# ---------------------------------------------------------------------------
# Dispatch table: module type name → handler function
# Extend this dict when adding support for new types.
# ---------------------------------------------------------------------------
_TYPE_TO_HANDLER: dict[str, Any] = {
    # PyTorch built-ins
    "Embedding":                    _codegen_embedding,
    "Linear":                       _codegen_linear,
    "LayerNorm":                    _codegen_layer_norm,
    "Conv3d":                       _codegen_conv3d,
    # Qwen3-VL text decoder
    "Qwen3VLTextRMSNorm":           _codegen_rms_norm,
    "Qwen3VLTextAttention":         _codegen_text_attention,
    "Qwen3VLTextMLP":               _codegen_ffn,
    # Qwen3-VL vision encoder
    "Qwen3VLVisionAttention":       _codegen_vision_attention,
    "Qwen3VLVisionMLP":             _codegen_ffn,
    "Qwen3VLVisionPatchEmbed":      _codegen_conv3d,
    "Qwen3VLVisionPatchMerger":     _codegen_linear,
    # Qwen2-VL (same handlers, different class names)
    "Qwen2VLTextRMSNorm":           _codegen_rms_norm,
    "Qwen2VLTextAttention":         _codegen_text_attention,
    "Qwen2VLTextMLP":               _codegen_ffn,
    "Qwen2VLVisionAttention":       _codegen_vision_attention,
    "Qwen2VLVisionMLP":             _codegen_ffn,
    # FX-derived functional ops (from hardcoded text-decoder pattern)
    "elementwise_add":              _codegen_elementwise_add,
}


# ---------------------------------------------------------------------------
# Program header / footer helpers
# ---------------------------------------------------------------------------

def _program_header(model_info: dict) -> str:
    return (
        "; ============================================================\n"
        f"; PLENA VLM Assembly  —  {model_info.get('model_name', 'unknown')}\n"
        f"; hidden_size={model_info.get('hidden_size', '?')}  "
        f"layers={model_info.get('num_layers', '?')}  "
        f"heads={model_info.get('num_attention_heads', '?')}\n"
        "; ============================================================\n"
    )


def _program_footer() -> str:
    return "; ============================================================\n; END\n"


# ---------------------------------------------------------------------------
# Node header (emitted before each codegen block)
# ---------------------------------------------------------------------------

def _node_header(node: dict) -> str:
    name     = node["name"] or "(root)"
    ntype    = node["type"]
    in_syms  = node.get("in_syms") or []
    out_syms = node.get("out_syms") or []
    sh_in    = _in_shape(node)
    sh_out   = _out_shape(node)

    lines = [f"; --- [{ntype}]  {name}"]
    if sh_in:
        lines.append(f";     in  {sh_in}  sym={in_syms}")
    if sh_out:
        lines.append(f";     out {sh_out}  sym={out_syms}")

    weights = node.get("weights") or []
    if weights:
        w_str = "  ".join(f"{w['name']}:{w['shape']}" for w in weights)
        lines.append(f";     weights  {w_str}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main codegen entry point
# ---------------------------------------------------------------------------

def vlm_codegen(
    nodes: list[dict],
    model_info: dict[str, Any],
    hw: dict[str, Any] | None = None,
    sched: dict[str, Any] | None = None,
) -> str:
    """
    Generate PLENA assembly from a flat list of trace nodes.

    Parameters
    ----------
    nodes      : flat list returned by flatten_call_tree(trace_leaf_modules(...))
                 or the combined list from combine_traces().
    model_info : model-level metadata (hidden_size, num_layers, head_dim, …).
    hw         : hardware config dict (mlen, blen, vlen, alive_registers).
                 Defaults to DEFAULT_HW.
    sched      : scheduler dict (memory addresses, register assignments).
                 Defaults to DEFAULT_SCHED.

    Returns
    -------
    str : complete assembly program.
    """
    hw    = hw    or DEFAULT_HW
    sched = sched or DEFAULT_SCHED

    asm_parts = [_program_header(model_info)]

    # covered_prefixes: once a node is processed, its children are skipped.
    covered: list[str] = []
    stats = {"generated": 0, "skipped_covered": 0, "skipped_unknown": 0}
    unknown: list[tuple[str, str]] = []

    for node in nodes:
        name      = node.get("name", "")
        node_type = node.get("type", "")

        # --- skip unknown types ---
        if node_type not in _TYPE_TO_HANDLER:
            asm_block = _codegen_unsupported(node, hw, sched, model_info)
            stats["skipped_unknown"] += 1
            unknown.append((name, node_type))
            continue

        # --- skip if already covered by a parent ---
        if name and any(name.startswith(pfx) for pfx in covered):
            stats["skipped_covered"] += 1
            continue

        # --- generate code ---
        handler = _TYPE_TO_HANDLER[node_type]
        try:
            asm_block = handler(node, hw, sched, model_info)
        except Exception as exc:
            asm_block = f"; [ERROR in {node_type} handler: {exc}]\n"

        asm_parts.append(_node_header(node) + asm_block)
        stats["generated"] += 1

        # cover this node's subtree (only named nodes, not FX ops like add_attn)
        if name:
            covered.append(name + ".")

    asm_parts.append(_program_footer())
    asm_parts.append(
        f"; stats: generated={stats['generated']}  "
        f"skipped_covered={stats['skipped_covered']}  "
        f"skipped_unknown={stats['skipped_unknown']}\n"
    )
    if unknown:
        asm_parts.append("; unknown nodes:\n")
        for name, node_type in unknown:
            asm_parts.append(f";   {name} ({node_type})\n")

    return "\n".join(asm_parts)


# ---------------------------------------------------------------------------
# __main__ — demonstration with Qwen3-VL-2B-Instruct
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
    IMAGE_PATH = "./inputs/img/image.png"

    # ── 1. load model and trace ──────────────────────────────────────────
    parser = VLMModelParser(MODEL_NAME)
    parser.load_model()
    inputs = template_qwen3_vl_inputs(parser.processor, IMAGE_PATH)

    trace_tree = parser.trace_leaf_modules(
        parser.model.model,
        {**inputs, "use_cache": False, "return_dict": False},
    )
    nodes = flatten_call_tree(trace_tree)

    # ── 2. extract model_info from config ────────────────────────────────
    cfg      = parser.config
    txt_cfg  = cfg.text_config
    vis_cfg  = cfg.vision_config

    model_info: dict[str, Any] = {
        "model_name":          MODEL_NAME,
        "batch_size":          1,
        "hidden_size":         txt_cfg.hidden_size,
        "intermediate_size":   txt_cfg.intermediate_size,
        "num_attention_heads": txt_cfg.num_attention_heads,
        "num_key_value_heads": txt_cfg.num_key_value_heads,
        "num_layers":          txt_cfg.num_hidden_layers,
        "head_dim":            getattr(txt_cfg, "head_dim",
                                       txt_cfg.hidden_size // txt_cfg.num_attention_heads),
        "vocab_size":          txt_cfg.vocab_size,
        # Vision-specific
        "vision_hidden_size":  getattr(vis_cfg, "hidden_size", txt_cfg.hidden_size),
        "vision_head_dim":     getattr(vis_cfg, "head_dim",
                                       getattr(vis_cfg, "hidden_size", txt_cfg.hidden_size)
                                       // getattr(vis_cfg, "num_heads", txt_cfg.num_attention_heads)),
    }

    print(f"Model info: hidden={model_info['hidden_size']}, "
          f"layers={model_info['num_layers']}, "
          f"head_dim={model_info['head_dim']}")
    print(f"Trace nodes: {len(nodes)}")

    # ── 3. run codegen ───────────────────────────────────────────────────
    asm = vlm_codegen(nodes, model_info)

    out_path = Path("./outputs/vlm_output.asm")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(asm)

    n_lines = len(asm.splitlines())
    print(f"Generated {n_lines} ASM lines → {out_path}")
