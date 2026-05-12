"""HuggingFace decoder model to PLENA ISA compiler."""

import math
import re

import torch
import torch.nn.functional as F

from compiler.aten.plena_compiler import PlenaCompiler
from compiler.aten.ops.registry import OpRegistry, Backend
from compiler.aten.ops.plena.linear_ops import linear_projection_plena as _linear_projection
import compiler.aten.ops as ops
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware

_IMM2_BOUND = 1 << 18  # S_ADDI_INT max immediate

def _fix_large_immediates(isa_code: str) -> str:
    """Post-process ISA: replace S_ADDI_INT gp{r}, gp0, {large} with S_LUI_INT + S_ADDI_INT.

    PlenaCompiler emits raw S_ADDI_INT for VRAM/HBM addresses. At large
    model dimensions these can exceed the 18-bit immediate limit. This pass
    splits them into S_LUI_INT (upper 22 bits, shifted <<12) + S_ADDI_INT
    (lower 12 bits), matching asm_templates._imm.load_large_int.
    """
    pattern = re.compile(r'^(\s*)S_ADDI_INT gp(\d+), gp0, (\d+)(.*)')
    out = []
    for line in isa_code.split('\n'):
        m = pattern.match(line)
        if m:
            indent, rd, imm_str, rest = m.groups()
            imm = int(imm_str)
            if imm >= _IMM2_BOUND:
                upper = imm >> 12
                lower = imm & 0xFFF
                out.append(f"{indent}S_LUI_INT gp{rd}, {upper}{rest}")
                if lower:
                    out.append(f"{indent}S_ADDI_INT gp{rd}, gp{rd}, {lower}{rest}")
                continue
        out.append(line)
    return '\n'.join(out)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)

# Hardware K-split tile limit (matches the PLENA linear backend)
_HW_MAX_K_TILES = 4


def quantize_to_mxfp(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to MXFP8 matching HBM hardware format; return dequantized result."""
    orig_shape = tensor.shape
    tensor_2d = tensor.float().reshape(-1, tensor.shape[-1])
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor_2d,
        width=8,
        exponent_width=4,
        exponent_bias_width=8,
        block_size=[1, 8],
    )
    return bm_x.reshape(orig_shape)


def _make_rope_tables(seq_len: int, head_dim: int, theta: float = 10000.0):
    """Compute RoPE cos/sin tables, shape (seq_len, head_dim)."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin


def _ksplit_matmul(A, B, mlen=64, max_k_tiles=_HW_MAX_K_TILES, to_inter=None, from_inter=None):
    """Matrix multiply matching hardware K-split BF16 precision.

    When the K dimension exceeds max_k_tiles * mlen, the hardware splits the
    inner product into chunks, writing each partial sum to BF16 VRAM and
    accumulating via BF16 add.  This function replicates that precision loss
    so the golden reference matches the emulator output.

    If the K dimension fits in a single pass (num_k_tiles <= max_k_tiles),
    this is equivalent to a single matmul with BF16 cast.
    """
    if to_inter is None:
        def to_inter(x):
            return x.to(torch.bfloat16)

    if from_inter is None:
        def from_inter(x):
            return x.float()

    k_total = A.shape[1]
    num_k_tiles = math.ceil(k_total / mlen)

    if num_k_tiles <= max_k_tiles:
        # Single pass — no K-split precision loss
        # Hardware path: MXFP8 (HBM) → BF16 (MRAM) → float32 (M_MM)
        # Cast B to BF16 then float32 to match MRAM precision loss
        return from_inter(to_inter(torch.matmul(from_inter(to_inter(A)), from_inter(to_inter(B)))))

    # K-split: chunk K tiles into groups of max_k_tiles
    result = None
    k_start = 0
    while k_start < k_total:
        k_end = min(k_start + max_k_tiles * mlen, k_total)
        A_chunk = A[:, k_start:k_end]
        B_chunk = B[k_start:k_end, :]
        # Cast B_chunk to BF16 then float32 to match MRAM precision loss
        partial = from_inter(to_inter(torch.matmul(from_inter(to_inter(A_chunk)), from_inter(to_inter(B_chunk)))))
        if result is None:
            result = partial
        else:
            # Hardware accumulates in BF16 VRAM (vram_block_add_to)
            result = from_inter(to_inter(result) + to_inter(partial))
        k_start = k_end

    return result


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------
def _make_rotate_half_matrix(head_dim: int) -> torch.Tensor:
    """Build the (head_dim, head_dim) matrix that computes rotate_half.

    rotate_half(x) = x @ R, where R permutes the halves with a sign flip:
      output[:d//2] = -input[d//2:]
      output[d//2:] = +input[:d//2]
    """
    R = torch.zeros(head_dim, head_dim)
    half = head_dim // 2
    for i in range(half):
        R[i + half, i] = -1.0   # first half of output = negated second half of input
        R[i, i + half] = 1.0    # second half of output = first half of input
    return R


# ---------------------------------------------------------------------------
# Model structure helpers
# ---------------------------------------------------------------------------
def _linear_weight(module, rows, cols):
    """HF Linear stores (out, in); PLENA uses (in, out)."""
    return module.weight.detach().T.contiguous()[:rows, :cols]


def _split_heads(weight, head_dim, num_heads):
    return [weight[:, h * head_dim:(h + 1) * head_dim].contiguous() for h in range(num_heads)]


def _find_model_root(model):
    """Find the transformer backbone (model.model or model.model.text_model).

    Handles standard CausalLM models and VLMs like SmolVLM2.
    """
    for candidate in [
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "text_model", None),
        getattr(model, "language_model", getattr(model, "text_model", None)),
    ]:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise ValueError(f"Cannot find decoder layers on {type(model).__name__}")


def _extract_config(model):
    """Extract config dimensions from the model, resolving text_config for VLMs."""
    config = getattr(model.config, "text_config", model.config)
    hidden = config.hidden_size
    inter = getattr(config, "intermediate_size", 4 * hidden)
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = hidden // num_heads
    eps = getattr(config, "rms_norm_eps", 1e-5)
    rope_theta = getattr(config, "rope_theta", 10000.0)
    vocab_size = getattr(config, "vocab_size", None)
    return {
        "hidden_size": hidden,
        "inter_dim": inter,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "eps": eps,
        "rope_theta": rope_theta,
        "vocab_size": vocab_size,
        "model_type": getattr(config, "model_type", "unknown"),
    }


def _extract_layer_weights(layer, hidden, inter, num_heads, head_dim, num_kv_heads=1):
    """Extract one decoder layer in PLENA's (in, out) linear-weight convention."""
    norm = layer.input_layernorm
    total_q_dim = num_heads * head_dim
    total_kv_dim = num_kv_heads * head_dim
    W_k_full = _linear_weight(layer.self_attn.k_proj, hidden, total_kv_dim)
    W_v_full = _linear_weight(layer.self_attn.v_proj, hidden, total_kv_dim)

    weights = {
        "W_q": _linear_weight(layer.self_attn.q_proj, hidden, total_q_dim),
        "W_o": _linear_weight(layer.self_attn.o_proj, total_q_dim, hidden),
        "W_k_heads": _split_heads(W_k_full, head_dim, num_kv_heads),
        "W_v_heads": _split_heads(W_v_full, head_dim, num_kv_heads),
        "W_gate": _linear_weight(layer.mlp.gate_proj, hidden, inter),
        "W_up": _linear_weight(layer.mlp.up_proj, hidden, inter),
        "W_down": _linear_weight(layer.mlp.down_proj, inter, hidden),
        "eps": getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)),
    }
    return weights


# ---------------------------------------------------------------------------
# Golden reference helpers (match hardware: MXFP8 HBM + BF16 intermediates)
# ---------------------------------------------------------------------------
def _flash_attn_ref(Q, K, V, scale, causal=False):
    """CPU reference: scaled dot-product attention matching hardware BF16 precision.

    Hardware path:
      S = Q @ K^T (M_TMM in f32, written to VRAM as BF16 via M_MM_WO)
      S *= scale (V_MUL_VF: BF16 * f32 -> BF16)
      P = softmax(S) (online softmax on BF16 data)
      O = P @ V (M_MM in f32, written to VRAM as BF16 via M_MM_WO)

    We model the key BF16 truncation points to match hardware precision.
    """
    # S = Q @ K^T, then truncate to BF16 (M_MM_WO writes BF16)
    scores = (Q @ K.T).to(torch.bfloat16).float() * scale
    scores = scores.to(torch.bfloat16).float()  # V_MUL_VF result is BF16
    if causal:
        mask = torch.triu(torch.ones(scores.shape[-2], scores.shape[-1],
                                     device=scores.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
    # Softmax output written to BF16 VRAM
    attn = F.softmax(scores, dim=-1).to(torch.bfloat16).float()
    # O = P @ V, result written to BF16 VRAM
    return (attn @ V).to(torch.bfloat16).float()


def _inter_round(x, to_inter, from_inter):
    return from_inter(to_inter(x))


def _hardware_rms_norm_ref(x, eps, to_inter, from_inter):
    x_inter = to_inter(x)
    rms = torch.rsqrt(from_inter(x_inter).pow(2).mean(-1, keepdim=True) + eps)
    return _inter_round(from_inter(x_inter) * rms, to_inter, from_inter)


def _hardware_linear_ref(x, weight, mlen, to_inter, from_inter):
    return _ksplit_matmul(x, weight, mlen, _HW_MAX_K_TILES, to_inter, from_inter)


def _hardware_rope_ref(x, rope_matrix, cos_table, sin_table, to_inter, from_inter):
    x_inter = _inter_round(x, to_inter, from_inter)
    x_rot = _inter_round(
        torch.matmul(x_inter, _inter_round(rope_matrix, to_inter, from_inter)),
        to_inter,
        from_inter,
    )
    x_cos = _inter_round(x_inter * _inter_round(cos_table, to_inter, from_inter), to_inter, from_inter)
    x_rot_sin = _inter_round(x_rot * _inter_round(sin_table, to_inter, from_inter), to_inter, from_inter)
    return _inter_round(x_cos + x_rot_sin, to_inter, from_inter)


def _hardware_hbm_round_ref(x, quantize_weight, to_inter, from_inter):
    return _inter_round(quantize_weight(x), to_inter, from_inter)


def _hardware_residual_add_ref(x, residual, to_inter, from_inter):
    return _inter_round(x + residual, to_inter, from_inter)


def _hardware_ffn_ref(x, W_gate, W_up, W_down, eps, mlen, to_inter, from_inter):
    residual = x.clone()
    x = _hardware_rms_norm_ref(x, eps, to_inter, from_inter)
    up_out = _hardware_linear_ref(x, W_up, mlen, to_inter, from_inter)
    gate_out = _hardware_linear_ref(x, W_gate, mlen, to_inter, from_inter)
    silu_gate = to_inter(
        F.silu(_inter_round(up_out, to_inter, from_inter)) * _inter_round(gate_out, to_inter, from_inter)
    )
    x = _hardware_linear_ref(from_inter(silu_gate), W_down, mlen, to_inter, from_inter)
    return _hardware_residual_add_ref(_inter_round(x, to_inter, from_inter), residual, to_inter, from_inter)


def _fp32_rms_norm_ref(x, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _fp32_linear_ref(x, weight):
    return x @ weight.float()


def _fp32_rope_ref(x, rope_matrix, cos_table, sin_table):
    return x * cos_table.float() + (x @ rope_matrix.float()) * sin_table.float()


def _fp32_ffn_ref(x, weights, eps):
    residual = x.clone()
    x_normed = _fp32_rms_norm_ref(x, eps)
    up_out = F.silu(_fp32_linear_ref(x_normed, weights["W_up"]))
    gate_out = _fp32_linear_ref(x_normed, weights["W_gate"])
    return _fp32_linear_ref(up_out * gate_out, weights["W_down"]) + residual


def _save_residual_and_norm(prog, source, scratch):
    """Emit the common decoder pre-norm residual prologue."""
    prog.vram_fill_zero(scratch)
    prog.vram_add(scratch, source)
    prog.rms_norm(source, eps_offset=3, reci_hid_offset=4)


def _add_residual(prog, target, scratch):
    prog.vram_add(target, scratch)
    return target


def _apply_rope_projection(prog, x_var, rope_matrix, cos_var, sin_var, name):
    x_rot = _linear_projection(prog, x_var, rope_matrix, name)
    prog.rope(x_var, x_rot, cos_var, sin_var)
    prog.free_tensor(x_rot)
    return x_var


def _copy_into_vram_view(prog, source, name, rows, cols, vram_addr):
    target = prog.alloc_at(name, rows, cols, vram_addr)
    prog.vram_fill_zero(target)
    prog.vram_add(target, source)
    return target


def _free_named_tensors(prog, names):
    for name in names:
        tensor = prog._tensors.get(name)
        if tensor is not None:
            prog.free_tensor(tensor)
            prog._tensors.pop(name, None)


def _emit_kv_stores(prog, current, layer_inputs, rope_inputs, layer_idx, num_kv_heads):
    rope_matrix, cos_var, sin_var = rope_inputs
    kv_stored = []
    for kv_h in range(num_kv_heads):
        K_h = _linear_projection(
            prog,
            current,
            layer_inputs["W_k_heads"][kv_h],
            f"K_{layer_idx}_h{kv_h}",
        )
        V_h = _linear_projection(
            prog,
            current,
            layer_inputs["W_v_heads"][kv_h],
            f"V_{layer_idx}_h{kv_h}",
        )

        _apply_rope_projection(
            prog,
            K_h,
            rope_matrix,
            cos_var,
            sin_var,
            f"K_rot_{layer_idx}_h{kv_h}",
        )

        K_stored = prog.store(K_h, name=f"K_stored_{layer_idx}_h{kv_h}")
        V_stored = prog.store(V_h, name=f"V_stored_{layer_idx}_h{kv_h}")
        kv_stored.append((K_stored, V_stored))

        prog.free_tensor(K_h)
        prog.free_tensor(V_h)

    return kv_stored


def _emit_attention_block(
    prog,
    current,
    layer_inputs,
    rope_inputs,
    causal_mask,
    scratch,
    scale,
    layer_idx,
    seq_len,
    head_dim,
    total_q_dim,
    num_heads,
    num_kv_heads,
    ratio,
):
    _save_residual_and_norm(prog, current, scratch)

    Q = _linear_projection(prog, current, layer_inputs["W_q"], f"Q_{layer_idx}")
    q_full_addr = prog.get_vram_addr(Q.name)

    O_full = prog.alloc(f"O_full_{layer_idx}", seq_len, total_q_dim)
    o_full_addr = prog.get_vram_addr(O_full.name)

    kv_stored = _emit_kv_stores(
        prog,
        current,
        layer_inputs,
        rope_inputs,
        layer_idx,
        num_kv_heads,
    )

    rope_matrix, cos_var, sin_var = rope_inputs
    for h in range(num_heads):
        kv_h = h // ratio
        K_stored, V_stored = kv_stored[kv_h]

        q_h_addr = q_full_addr + h * seq_len * prog.mlen
        Q_h = prog.alloc_at(f"Q_h{h}_{layer_idx}", seq_len, head_dim, q_h_addr)
        _apply_rope_projection(
            prog,
            Q_h,
            rope_matrix,
            cos_var,
            sin_var,
            f"Q_rot_{layer_idx}_h{h}",
        )

        O_h = ops.flash_attention(
            prog,
            Q_h,
            K_stored,
            V_stored,
            scale,
            causal_mask=causal_mask,
        )

        o_h_dest_addr = o_full_addr + h * seq_len * prog.mlen
        _copy_into_vram_view(
            prog,
            O_h,
            f"O_dest_h{h}_{layer_idx}",
            seq_len,
            head_dim,
            o_h_dest_addr,
        )
        _free_named_tensors(prog, ("O", "S", "PV"))

    O_proj = _linear_projection(prog, O_full, layer_inputs["W_o"], f"O_proj_{layer_idx}")
    return _add_residual(prog, O_proj, scratch)


def _emit_ffn_block(prog, current, layer_inputs, scratch):
    _save_residual_and_norm(prog, current, scratch)
    ops.ffn(prog, current, layer_inputs["W_gate"], layer_inputs["W_up"], layer_inputs["W_down"])
    return _add_residual(prog, current, scratch)


def _layer_tensor_entries(layer_idx, weights, num_kv_heads):
    entries = [
        (f"W_q_{layer_idx}", weights["W_q"]),
        (f"W_o_{layer_idx}", weights["W_o"]),
    ]
    for kv_h in range(num_kv_heads):
        entries.extend(
            [
                (f"W_k_{layer_idx}_h{kv_h}", weights["W_k_heads"][kv_h]),
                (f"W_v_{layer_idx}_h{kv_h}", weights["W_v_heads"][kv_h]),
            ]
        )
    entries.extend(
        [
            (f"W_gate_{layer_idx}", weights["W_gate"]),
            (f"W_up_{layer_idx}", weights["W_up"]),
            (f"W_down_{layer_idx}", weights["W_down"]),
        ]
    )
    return entries


# ---------------------------------------------------------------------------
# Main compilation function
# ---------------------------------------------------------------------------
def compile_hf_model(
    model,
    seq_len: int = 64,
    num_layers: int | None = None,
    layer_idx_start: int = 0,
    mlen: int = 64,
    blen: int = 4,
    seed: int = 42,
    golden_precision: str = "hardware",
    verbose: bool = False,
) -> dict:
    """Compile a HuggingFace decoder model to PLENA ISA and simulation metadata."""
    def _verbose(message: str = ""):
        if verbose:
            print(message)

    # -------------------------------------------------------------- config
    model_cfg = _extract_config(model)
    hidden = model_cfg["hidden_size"]
    inter = model_cfg["inter_dim"]
    num_heads = model_cfg["num_heads"]
    num_kv_heads = model_cfg["num_kv_heads"]
    head_dim = model_cfg["head_dim"]
    total_q_dim = num_heads * head_dim

    root = _find_model_root(model)
    layers = root.layers
    n_layers = num_layers if num_layers is not None else len(layers)
    assert layer_idx_start + n_layers <= len(layers), (
        f"Requested layers [{layer_idx_start}, {layer_idx_start + n_layers}) "
        f"but model only has {len(layers)} layers"
    )

    scale = 1.0 / math.sqrt(head_dim)

    # ----------------------------------------------------------- embedding
    embed = getattr(root, "embed_tokens", getattr(root, "wte", None))

    print("=" * 80)
    print(f"Model Compiler - {model_cfg['model_type']} ({n_layers} layer{'s' if n_layers != 1 else ''})")
    print(
        f"  decoder: hidden={hidden}, inter={inter}, heads={num_heads}/{num_kv_heads}, "
        f"head_dim={head_dim}"
    )
    print(f"  compile: seq_len={seq_len}, mlen={mlen}, blen={blen}, total_q_dim={total_q_dim}")
    print("=" * 80)

    # ----------------------------------------------------------- weights
    print(f"\nExtracting weights from layers {layer_idx_start}..{layer_idx_start + n_layers - 1}...")
    all_weights = []
    for i in range(n_layers):
        layer_module = layers[layer_idx_start + i]
        w = _extract_layer_weights(
            layer_module,
            hidden,
            inter,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
        )
        all_weights.append(w)
        _verbose(
            f"  Layer {i}: W_q={w['W_q'].shape}, W_o={w['W_o'].shape}, "
            f"W_gate={w['W_gate'].shape}, "
            f"K_heads={len(w['W_k_heads'])}x{w['W_k_heads'][0].shape}, eps={w['eps']}"
        )

    eps = all_weights[0]["eps"]

    # ----------------------------------------------------------- test data
    torch.manual_seed(seed)

    # Embedding lookup: use real HF embedding table if available
    if embed is not None:
        input_ids = torch.randint(0, model_cfg["vocab_size"] or 32000, (seq_len,))
        with torch.no_grad():
            token_embeds = embed(input_ids).float()
        if token_embeds.dim() == 3:
            token_embeds = token_embeds.squeeze(0)
        # Slice to the model hidden width.
        token_embeds = token_embeds[:, :hidden]
        _verbose(
            f"\nEmbedding lookup: input_ids shape={input_ids.shape}, "
            f"token_embeds={token_embeds.shape}"
        )
    else:
        token_embeds = torch.randn(seq_len, hidden)
        print(f"\nNo embed_tokens found; using random token_embeds: {token_embeds.shape}")

    # Llama-style models use RoPE (not learned position embeddings).
    # Set pos_weight to zeros so embedding_add is a no-op for position.
    pos_weight = torch.zeros(seq_len, hidden)

    _verbose(f"pos_weight: zeros {pos_weight.shape} (RoPE model; learned position add is a no-op)")
    for i in range(n_layers):
        for kv_h in range(num_kv_heads):
            _verbose(
                f"  W_k_{i}_h{kv_h}: {all_weights[i]['W_k_heads'][kv_h].shape}, "
                f"W_v_{i}_h{kv_h}: {all_weights[i]['W_v_heads'][kv_h].shape}"
            )
    print(f"attn_scale: {scale:.6f}")

    # ----------------------------------------------------------- golden ref
    _do_quant = golden_precision in ("hardware", "no_bf16")
    _do_bf16 = golden_precision in ("hardware", "no_weight_quant")
    _qw = quantize_to_mxfp if _do_quant else (lambda x: x)
    _to_inter = (lambda x: x.to(torch.bfloat16)) if _do_bf16 else (lambda x: x)
    _from_inter = (lambda x: x.float()) if _do_bf16 else (lambda x: x)
    _prec_label = {"hardware": "MXFP8 weights + BF16 intermediates",
                   "no_weight_quant": "float32 weights + BF16 intermediates",
                   "no_bf16": "MXFP8 weights + float32 intermediates",
                   "fp32": "float32 weights + float32 intermediates"}[golden_precision]
    print(f"\nComputing CPU golden reference ({_prec_label})")

    W_k_q_heads = [[_qw(all_weights[i]["W_k_heads"][h]) for h in range(num_kv_heads)] for i in range(n_layers)]
    W_v_q_heads = [[_qw(all_weights[i]["W_v_heads"][h]) for h in range(num_kv_heads)] for i in range(n_layers)]
    R_matrix = _make_rotate_half_matrix(head_dim)
    R_rope_q = _qw(R_matrix)
    cos_table, sin_table = _make_rope_tables(seq_len, head_dim, model_cfg["rope_theta"])

    X_gold = _qw(token_embeds.clone()) + _qw(pos_weight)  # embedding_add (MXFP8-quantized, matching HBM)
    ratio = num_heads // num_kv_heads

    for i in range(n_layers):
        w = all_weights[i]
        W_q_q = _qw(w["W_q"])
        W_o_q = _qw(w["W_o"])
        W_gate_q = _qw(w["W_gate"])
        W_up_q = _qw(w["W_up"])
        W_down_q = _qw(w["W_down"])

        # --- Attention block ---
        residual = X_gold.clone()
        X_gold = _hardware_rms_norm_ref(X_gold, eps, _to_inter, _from_inter)
        Q_gold = _inter_round(_hardware_linear_ref(X_gold, W_q_q, mlen, _to_inter, _from_inter), _to_inter, _from_inter)

        K_q_heads_i = []
        V_q_heads_i = []
        for kv_h in range(num_kv_heads):
            K_h = _hardware_linear_ref(X_gold, W_k_q_heads[i][kv_h], mlen, _to_inter, _from_inter)
            V_h = _hardware_linear_ref(X_gold, W_v_q_heads[i][kv_h], mlen, _to_inter, _from_inter)
            K_h = _hardware_rope_ref(K_h, R_rope_q, cos_table, sin_table, _to_inter, _from_inter)
            K_q_heads_i.append(_hardware_hbm_round_ref(K_h, _qw, _to_inter, _from_inter))
            V_q_heads_i.append(_hardware_hbm_round_ref(V_h, _qw, _to_inter, _from_inter))

        O_heads = []
        for h in range(num_heads):
            kv_h = h // ratio
            Q_h = Q_gold[:, h * head_dim:(h + 1) * head_dim]
            Q_h = _hardware_rope_ref(Q_h, R_rope_q, cos_table, sin_table, _to_inter, _from_inter)
            O_h = _flash_attn_ref(Q_h, K_q_heads_i[kv_h], V_q_heads_i[kv_h], scale, causal=True)
            O_heads.append(O_h)
        attn_out = _inter_round(torch.cat(O_heads, dim=1), _to_inter, _from_inter)
        O_gold = _inter_round(_hardware_linear_ref(attn_out, W_o_q, mlen, _to_inter, _from_inter), _to_inter, _from_inter)
        X_gold = _hardware_residual_add_ref(O_gold, residual, _to_inter, _from_inter)

        # --- FFN block ---
        X_gold = _hardware_ffn_ref(X_gold, W_gate_q, W_up_q, W_down_q, eps, mlen, _to_inter, _from_inter)

        _verbose(f"  After layer {i}: X_gold[0,:4] = {X_gold[0, :4].tolist()}")

    # Final norm
    X_gold = _hardware_rms_norm_ref(X_gold, eps, _to_inter, _from_inter)

    golden_out = X_gold
    print(f"  golden_out: {golden_out.shape}")
    _verbose(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ----------------------------------------------------------- HF ground truth
    # Same pipeline as golden but pure float32: no MXFP8 quantization, no BF16 casting.
    # This is the float32 reference for the same decoder blocks being compiled.
    print(f"\nComputing HF reference (float32, {n_layers} layer{'s' if n_layers != 1 else ''}, no quantization)")
    with torch.no_grad():
        X_hf = token_embeds.clone() + pos_weight  # embedding_add

        for i in range(n_layers):
            w = all_weights[i]
            eps_i = w["eps"]

            residual = X_hf.clone()
            X_normed = _fp32_rms_norm_ref(X_hf, eps_i)

            Q_hf = _fp32_linear_ref(X_normed, w["W_q"])

            K_hf_heads = []
            V_hf_heads = []
            for kv_h in range(num_kv_heads):
                K_hf_heads.append(
                    _fp32_rope_ref(
                        _fp32_linear_ref(X_normed, w["W_k_heads"][kv_h]),
                        R_matrix,
                        cos_table,
                        sin_table,
                    )
                )
                V_hf_heads.append(_fp32_linear_ref(X_normed, w["W_v_heads"][kv_h]))

            O_heads_hf = []
            for h in range(num_heads):
                kv_h = h // ratio
                Q_h = Q_hf[:, h * head_dim:(h + 1) * head_dim]
                Q_h = _fp32_rope_ref(Q_h, R_matrix, cos_table, sin_table)
                O_h = _flash_attn_ref(Q_h, K_hf_heads[kv_h], V_hf_heads[kv_h], scale, causal=True)
                O_heads_hf.append(O_h)
            attn_out_hf = torch.cat(O_heads_hf, dim=1)
            O_hf = _fp32_linear_ref(attn_out_hf, w["W_o"])
            X_hf = O_hf + residual

            X_hf = _fp32_ffn_ref(X_hf, w, eps_i)

            _verbose(f"  After layer {i}: X_hf[0,:4] = {X_hf[0, :4].tolist()}")

        X_hf = _fp32_rms_norm_ref(X_hf, eps)

        hf_ground_truth = X_hf

    print(f"  hf_ground_truth: {hf_ground_truth.shape}")
    _verbose(f"  hf_ground_truth[0,:4]: {hf_ground_truth[0, :4].tolist()}")

    # ----------------------------------------------------------- PLENA ISA
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=REAL_DATA_RATIO)

    # Shared inputs
    x_input = prog.input("X", shape=(seq_len, hidden))
    pos_input = prog.input("POS", shape=(seq_len, hidden))

    r_input = prog.input("R_rope", shape=(head_dim, head_dim))
    cos_input = prog.input("COS", shape=(seq_len, head_dim))
    sin_input = prog.input("SIN", shape=(seq_len, head_dim))
    COS = prog.load_batch(cos_input, name="COS")
    SIN = prog.load_batch(sin_input, name="SIN")

    # Causal mask: (mlen, mlen) with 0 on/below diagonal, -inf above
    causal_mask_data = torch.zeros(mlen, mlen)
    causal_mask_data.masked_fill_(
        torch.triu(torch.ones(mlen, mlen), diagonal=1).bool(), float('-inf')
    )
    causal_mask_input = prog.input("causal_mask", shape=(mlen, mlen))
    CAUSAL_MASK = prog.load_batch(causal_mask_input, name="CAUSAL_MASK")

    # Per-layer weight inputs (order determines HBM layout)
    layer_inputs = []
    for i in range(n_layers):
        li_entry = {"W_k_heads": [], "W_v_heads": []}
        for tensor_name, tensor in _layer_tensor_entries(i, all_weights[i], num_kv_heads):
            var = prog.input(tensor_name, shape=tuple(tensor.shape))
            if tensor_name.startswith(f"W_k_{i}_h"):
                li_entry["W_k_heads"].append(var)
            elif tensor_name.startswith(f"W_v_{i}_h"):
                li_entry["W_v_heads"].append(var)
            else:
                li_entry[tensor_name[: tensor_name.rfind(f"_{i}")]] = var
        layer_inputs.append(li_entry)

    # Load activations to VRAM
    X_batch = prog.load_batch(x_input, name="X")
    POS_batch = prog.load_batch(pos_input, name="POS")
    ops.embedding_add(prog, X_batch, POS_batch)  # X += POS in-place

    # VRAM layout hazard: ffn_asm writes gate/up intermediates at absolute
    # address batch*hidden spanning up to batch*hidden + 2*inter*batch.
    # The residual scratch buffer must be placed ABOVE this region.
    _ffn_intermediate_end = seq_len * hidden + 2 * inter * seq_len
    _current_bump = 2 * seq_len * hidden  # X + POS already allocated
    _current_bump += 2 * seq_len * head_dim  # COS + SIN loaded to VRAM
    _current_bump += mlen * mlen  # CAUSAL_MASK loaded to VRAM
    if _current_bump < _ffn_intermediate_end:
        _pad_elems = _ffn_intermediate_end - _current_bump
        # Allocate enough mlen-wide rows to cover the padding
        _pad_rows = (_pad_elems + mlen - 1) // mlen
        # Round up to mlen for VRAM alignment
        _pad_rows = ((_pad_rows + mlen - 1) // mlen) * mlen
        prog.alloc("_vram_padding", _pad_rows, mlen, strict=False)

    # Allocate scratch buffer for residual save/restore (reused across layers)
    scratch = prog.alloc("residual_scratch", seq_len, hidden)

    # Chain layers
    current = X_batch

    for i in range(n_layers):
        li = layer_inputs[i]

        # Layer progress marker (visible in non-quiet emulator output)
        prog.emit_comment(f"=== LAYER {i}/{n_layers} START ===")

        current_after_attn = _emit_attention_block(
            prog,
            current,
            li,
            (r_input, COS, SIN),
            CAUSAL_MASK,
            scratch,
            scale,
            i,
            seq_len,
            head_dim,
            total_q_dim,
            num_heads,
            num_kv_heads,
            ratio,
        )

        current = _emit_ffn_block(prog, current_after_attn, li, scratch)
        prog.emit_comment(f"=== LAYER {i}/{n_layers} COMPLETE ===")

    # Final norm
    prog.rms_norm(current, eps_offset=3, reci_hid_offset=4)

    isa_code = prog.compile()
    isa_code = _fix_large_immediates(isa_code)
    lines = isa_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ----------------------------------------------------------- build return
    input_tensors = {"X": token_embeds, "POS": pos_weight, "R_rope": R_matrix, "COS": cos_table, "SIN": sin_table}
    data_order = ["X", "POS", "R_rope", "COS", "SIN"]
    input_tensors["causal_mask"] = causal_mask_data
    data_order.append("causal_mask")
    for i in range(n_layers):
        for name, tensor in _layer_tensor_entries(i, all_weights[i], num_kv_heads):
            input_tensors[name] = tensor
            data_order.append(name)

    # FPRAM layout (same as single-layer decoder):
    #   slot 0 = 0.0        (reserved)
    #   slot 1 = attn_scale  (flash_attention)
    #   slot 2 = -inf        (flash_attention softmax mask)
    #   slot 3 = eps         (rms_norm, offset=3)
    #   slot 4 = 1/hidden    (rms_norm, offset=4)
    #   slot 5 = 1.0         (FFN SiLU)
    #   slots 6-9 = 0.0      (padding)
    fp_preload = [0.0, scale, float("-inf"), eps, 1.0 / hidden, 1.0] + [0.0] * 4

    # Result is at current's VRAM location
    o_vram_addr = prog.get_vram_addr(current.name)

    out_features = hidden

    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (seq_len * out_features) // mlen,
        "num_batches": seq_len,
        "elements_per_batch": out_features,
        "row_dim": mlen,
        "use_stride_mode": out_features > mlen,
    }

    info = {
        "model_type": model_cfg["model_type"],
        "hidden_size": hidden,
        "inter_dim": inter,
        "num_layers": n_layers,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "mlen": mlen,
        "blen": blen,
        "isa_lines": len(lines),
    }

    print(f"\nCompilation complete: {info['isa_lines']} ISA lines, "
          f"{n_layers} layers, output at VRAM row {o_vram_addr // mlen}")
    return {
        "isa": isa_code,
        "golden_output": golden_out,
        "hf_ground_truth": hf_ground_truth,
        "input_tensors": input_tensors,
        "data_order": data_order,
        "fp_preload": fp_preload,
        "comparison_params": comparison_params,
        "info": info,
        "golden_precision": golden_precision,
    }
