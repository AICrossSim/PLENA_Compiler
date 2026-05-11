"""
Automatic HuggingFace model -> PLENA ISA compiler.

Walks an HF nn.Module tree, extracts weights, and generates ISA with
proper residual connections for multi-layer decoder pipelines.

Usage:
    from transformers import AutoModelForCausalLM
    from compiler.aten.plena_frontend import compile_hf_model, compile_and_run

    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M",
                                                  torch_dtype=torch.float32)
    # Just compile (no emulator run):
    result = compile_hf_model(model, seq_len=64, hidden_size=64, inter_dim=128, num_layers=2)

    # Compile + run emulator + compare:
    result = compile_and_run(model, "/tmp/build", seq_len=64, hidden_size=64, inter_dim=128, num_layers=2)
"""

import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from compiler.aten.plena_compiler import PlenaCompiler
from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware

_IMM2_BOUND = 1 << 18  # S_ADDI_INT max immediate

def _fix_large_immediates(isa_code: str) -> str:
    """Post-process ISA: replace S_ADDI_INT gp{r}, gp0, {large} with S_LUI_INT + S_ADDI_INT.

    PlenaCompiler emits raw S_ADDI_INT for VRAM/HBM addresses. At native
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

# Hardware K-split tile limit (matches _linear_projection MAX_K_TILES)
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
    native_hidden = config.hidden_size
    native_inter = getattr(config, "intermediate_size", 4 * native_hidden)
    native_heads = config.num_attention_heads
    native_kv_heads = getattr(config, "num_key_value_heads", native_heads)
    native_head_dim = native_hidden // native_heads
    eps = getattr(config, "rms_norm_eps", 1e-5)
    rope_theta = getattr(config, "rope_theta", 10000.0)
    vocab_size = getattr(config, "vocab_size", None)
    return {
        "hidden_size": native_hidden,
        "inter_dim": native_inter,
        "num_heads": native_heads,
        "num_kv_heads": native_kv_heads,
        "head_dim": native_head_dim,
        "eps": eps,
        "rope_theta": rope_theta,
        "vocab_size": vocab_size,
        "model_type": getattr(config, "model_type", "unknown"),
    }


def _extract_layer_weights(layer, hidden_slice, inter_slice, head_dim_slice, num_heads, head_dim,
                            num_kv_heads=1, native_mode=False):
    """Extract and slice weights from a single decoder layer.

    Transposes from HF's (out_features, in_features) to PLENA's (in, out) convention.

    In native mode (hidden_slice == native hidden, multi-head):
      - W_q: (hidden, num_heads * head_dim) — full Q projection
      - W_o: (num_heads * head_dim, hidden) — full O projection
      - W_k_heads: list of (hidden, head_dim) per KV head
      - W_v_heads: list of (hidden, head_dim) per KV head

    In sliced mode (legacy single-head):
      - W_q: (hidden_slice, head_dim_slice)
      - W_o: (head_dim_slice, hidden_slice)
      - W_k: (hidden_slice, head_dim_slice)
      - W_v: (hidden_slice, head_dim_slice)

    Args:
        layer: nn.Module for a single decoder layer
        hidden_slice: target hidden dimension
        inter_slice: target intermediate dimension
        head_dim_slice: target head dimension (min of native head_dim, hidden_slice)
        num_heads: number of attention heads (native config)
        head_dim: native head dimension
        num_kv_heads: number of KV heads (native config)
        native_mode: if True, extract full multi-head weights

    Returns:
        dict with W_q, W_o, W_gate, W_up, W_down, W_k/W_k_heads, W_v/W_v_heads, eps
    """
    # FFN weights: HF stores (out, in) -> transpose to (in, out) -> slice
    W_gate = layer.mlp.gate_proj.weight.detach().T.contiguous()[:hidden_slice, :inter_slice]
    W_up = layer.mlp.up_proj.weight.detach().T.contiguous()[:hidden_slice, :inter_slice]
    W_down = layer.mlp.down_proj.weight.detach().T.contiguous()[:inter_slice, :hidden_slice]

    # eps from input_layernorm
    norm = layer.input_layernorm
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5))

    if native_mode:
        # Full multi-head weights (no slicing)
        total_q_dim = num_heads * head_dim
        total_kv_dim = num_kv_heads * head_dim

        W_q = layer.self_attn.q_proj.weight.detach().T.contiguous()[:hidden_slice, :total_q_dim]
        W_o = layer.self_attn.o_proj.weight.detach().T.contiguous()[:total_q_dim, :hidden_slice]

        # Per-KV-head K and V weights
        W_k_full = layer.self_attn.k_proj.weight.detach().T.contiguous()[:hidden_slice, :total_kv_dim]
        W_v_full = layer.self_attn.v_proj.weight.detach().T.contiguous()[:hidden_slice, :total_kv_dim]

        W_k_heads = [W_k_full[:, h * head_dim:(h + 1) * head_dim].contiguous()
                     for h in range(num_kv_heads)]
        W_v_heads = [W_v_full[:, h * head_dim:(h + 1) * head_dim].contiguous()
                     for h in range(num_kv_heads)]

        return {
            "W_q": W_q,
            "W_o": W_o,
            "W_gate": W_gate,
            "W_up": W_up,
            "W_down": W_down,
            "W_k_heads": W_k_heads,
            "W_v_heads": W_v_heads,
            "eps": eps,
        }
    else:
        # Legacy single-head sliced mode
        W_q = layer.self_attn.q_proj.weight.detach().T.contiguous()[:hidden_slice, :head_dim_slice]
        W_o = layer.self_attn.o_proj.weight.detach().T.contiguous()[:head_dim_slice, :hidden_slice]
        W_k = layer.self_attn.k_proj.weight.detach().T.contiguous()[:hidden_slice, :head_dim_slice]
        W_v = layer.self_attn.v_proj.weight.detach().T.contiguous()[:hidden_slice, :head_dim_slice]

        return {
            "W_q": W_q,
            "W_o": W_o,
            "W_gate": W_gate,
            "W_up": W_up,
            "W_down": W_down,
            "W_k": W_k,
            "W_v": W_v,
            "eps": eps,
        }


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


def _rms_norm_ref(x, eps):
    """CPU reference: RMS normalization matching PLENA hardware.

    Hardware path: V_RED_SUM accumulates sum-of-squares into f32 scalar register,
    S_MUL_FP / S_ADD_FP / S_SQRT_FP / S_RECI_FP all operate in f32,
    then V_MUL_VF multiplies BF16 vector by f32 scalar -> BF16 result.
    The scalar rms factor stays in f32 throughout; only the vector data is BF16.
    """
    x_bf = x.to(torch.bfloat16)
    # Compute rms in f32 (matching hardware scalar register precision)
    rms = torch.rsqrt(x_bf.float().pow(2).mean(-1, keepdim=True) + eps)
    # V_MUL_VF: BF16 vector * f32 scalar -> quantized back to BF16
    return (x_bf.float() * rms).to(torch.bfloat16).float()


# ---------------------------------------------------------------------------
# PLENA ISA helper: named linear projection (avoids name conflicts)
# ---------------------------------------------------------------------------
def _linear_projection(prog, input_var, weight_var, name):
    """Emit a linear projection with a custom VRAM output name.

    Equivalent to ops.linear but uses *name* for the output allocation so
    that multiple projections in the same scope don't collide on the
    default "linear_out" name.

    Supports K-split: when K tiles exceed MRAM capacity (4 tiles), splits
    into chunks and accumulates partial sums via a temporary buffer.
    """
    import math as _math

    mlen = prog.mlen
    MAX_K_TILES = 4  # MRAM capacity: 4 x mlen^2 elements

    rows, k_total = input_var.shape
    _, out_features = weight_var.shape
    num_row_blocks = _math.ceil(rows / mlen)
    assert out_features % mlen == 0, (
        f"out_features ({out_features}) must be a multiple of mlen ({mlen})"
    )
    num_col_blocks = out_features // mlen
    num_k_tiles = _math.ceil(k_total / mlen)

    output_strict = rows % mlen == 0
    output = prog.alloc(name, rows, out_features, strict=output_strict)

    if num_k_tiles <= MAX_K_TILES:
        # Single pass: all K tiles fit in MRAM
        for col_idx in range(num_col_blocks):
            for row_idx in range(num_row_blocks):
                prog.vram_sub_projection_to(
                    input_var,
                    row_idx,
                    weight_var,
                    col_idx,
                    output,
                    row_idx,
                    col_idx,
                )
    else:
        # K-split: chunk K tiles into groups of MAX_K_TILES
        k_chunks = []
        k_start = 0
        while k_start < num_k_tiles:
            k_end = min(k_start + MAX_K_TILES, num_k_tiles)
            k_chunks.append((k_start, k_end - k_start))
            k_start = k_end

        temp = prog.alloc(f"{name}_ksplit_tmp", rows, out_features, strict=output_strict)

        for k_chunk_idx, (k_block_start, k_block_count) in enumerate(k_chunks):
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    if k_chunk_idx == 0:
                        prog.vram_sub_projection_to(
                            input_var, row_idx, weight_var, col_idx,
                            output, row_idx, col_idx,
                            k_block_start=k_block_start,
                            k_block_count=k_block_count,
                        )
                    else:
                        prog.vram_sub_projection_to(
                            input_var, row_idx, weight_var, col_idx,
                            temp, row_idx, col_idx,
                            k_block_start=k_block_start,
                            k_block_count=k_block_count,
                        )
                        prog.vram_block_add_to(
                            output, row_idx, col_idx,
                            temp, row_idx, col_idx,
                            output, row_idx, col_idx,
                        )

        prog.free_tensor(temp)

    return output


# ---------------------------------------------------------------------------
# Main compilation function
# ---------------------------------------------------------------------------
def compile_hf_model(
    model,
    seq_len: int = 64,
    hidden_size: int | None = None,
    inter_dim: int | None = None,
    num_layers: int | None = None,
    layer_idx_start: int = 0,
    mlen: int = 64,
    blen: int = 4,
    seed: int = 42,
    include_lm_head: bool = False,
    golden_precision: str = "hardware",
) -> dict:
    """Compile a HuggingFace model to PLENA ISA via PlenaCompiler.

    Walks the nn.Module tree, extracts weights, and generates ISA with
    proper residual connections for multi-layer decoders.

    The pipeline implemented (per-layer, with pre-norm + residual):
        X = embed_tokens(input_ids)      # real HF embedding lookup
        X = embedding_add(X, zeros)      # pos_weight=0 for Llama (RoPE handles position)
        for each layer:
            residual = X
            X = rms_norm(X)
            Q = linear(X, W_q)           # Q projection
            K_h = linear(X, W_k_h)       # K projection (per KV head, on-chip)
            V_h = linear(X, W_v_h)       # V projection (per KV head, on-chip)
            # RoPE (native mode only):
            Q_rot_h = linear(Q_h, R_rope)  # rotate_half via matmul
            Q_h = Q_h * cos + Q_rot_h * sin
            K_rot_h = linear(K_h, R_rope)
            K_h = K_h * cos + K_rot_h * sin
            store K_h, V_h to HBM
            O = flash_attention(Q, K, V, scale)
            O = linear(O, W_o)           # O projection
            X = O + residual
            residual = X
            X = rms_norm(X)
            X = ffn(X, gate, up, down)
            X = X + residual
        X = rms_norm(X)  # final norm
        if include_lm_head:
            logits = linear(X, W_lm_head)  # (seq, vocab_size)

    RoPE is applied in native mode via rotate_half matrix multiplication.
    Q, K, V, and O linear projections are all computed on-chip.

    Args:
        model:          nn.Module (HF CausalLM model, already loaded)
        seq_len:        Sequence length (default 64)
        hidden_size:    Target hidden dimension (None = use model's native)
        inter_dim:      Target intermediate dimension (None = use model's native)
        num_layers:     Number of layers to compile (None = all layers)
        layer_idx_start: First layer index to use (default 0)
        mlen:           Matrix tile length (default 64)
        blen:           Batch tile length (default 4)
        seed:           Random seed for test data generation
        include_lm_head: If True, add lm_head projection after final norm (default False)
        golden_precision: Precision mode for golden reference computation.
            "hardware"       — MXFP8 weights + BF16 intermediates (default, matches HW)
            "no_weight_quant" — float32 weights + BF16 intermediates (isolates MXFP8 effect)
            "no_bf16"        — MXFP8 weights + float32 intermediates (isolates BF16 effect)
            "fp32"           — float32 weights + float32 intermediates (should match HF exactly)

    Returns:
        dict with:
            isa: str                    - generated ISA code
            golden_output: torch.Tensor - CPU golden reference output
            input_tensors: dict         - {name: tensor} for sim env setup
            data_order: list[str]       - HBM tensor ordering
            fp_preload: list[float]     - FPRAM constants
            comparison_params: dict     - for emulator comparison
            info: dict                  - model dims, VRAM usage, etc.
    """
    # -------------------------------------------------------------- config
    native_cfg = _extract_config(model)
    hidden = hidden_size if hidden_size is not None else native_cfg["hidden_size"]
    inter = inter_dim if inter_dim is not None else native_cfg["inter_dim"]

    num_heads = native_cfg["num_heads"]
    num_kv_heads = native_cfg["num_kv_heads"]
    native_head_dim = native_cfg["head_dim"]

    # Native mode: use all heads at full dimension when hidden_size is not overridden
    native_mode = (hidden_size is None) and (num_heads > 1)
    if native_mode:
        head_dim = native_head_dim
        total_q_dim = num_heads * head_dim       # e.g. 6*64 = 384
        total_kv_dim = num_kv_heads * head_dim   # e.g. 2*64 = 128
    else:
        head_dim = min(native_head_dim, hidden)
        total_q_dim = head_dim
        total_kv_dim = head_dim

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

    # ----------------------------------------------------------- lm_head
    lm_head_weight = None
    vocab_size = native_cfg.get("vocab_size")
    if include_lm_head:
        lm_head_mod = getattr(model, "lm_head", None)
        if lm_head_mod is None:
            lm_head_mod = getattr(
                getattr(model, "language_model", model), "lm_head", None
            )
        if lm_head_mod is not None and hasattr(lm_head_mod, "weight"):
            # nn.Linear stores (vocab, hidden) -> transpose to (hidden, vocab)
            lm_head_weight_raw = lm_head_mod.weight.detach().T.contiguous()
            # Slice to hidden if we're in sliced mode
            lm_head_weight = lm_head_weight_raw[:hidden, :]
            vocab_size = lm_head_weight.shape[1]
            # Ensure vocab_size is a multiple of mlen (pad if needed)
            if vocab_size % mlen != 0:
                pad_cols = mlen - (vocab_size % mlen)
                lm_head_weight = F.pad(lm_head_weight, (0, pad_cols))
                vocab_size = lm_head_weight.shape[1]
        else:
            print("WARNING: include_lm_head=True but no lm_head module found; skipping")
            include_lm_head = False

    print("=" * 80)
    print(f"Model Compiler - {native_cfg['model_type']}  ({n_layers} layers)")
    print(f"  native: hidden={native_cfg['hidden_size']}, inter={native_cfg['inter_dim']}, "
          f"heads={native_cfg['num_heads']}/{native_cfg['num_kv_heads']}, "
          f"head_dim={native_cfg['head_dim']}")
    print(f"  sim:    hidden={hidden}, inter={inter}, head_dim={head_dim}, "
          f"seq_len={seq_len}, mlen={mlen}, native_mode={native_mode}")
    if native_mode:
        print(f"  MHA:    num_heads={num_heads}, num_kv_heads={num_kv_heads}, "
              f"total_q_dim={total_q_dim}, total_kv_dim={total_kv_dim}")
    if include_lm_head:
        print(f"  lm_head: W_lm_head={lm_head_weight.shape}, vocab_size={vocab_size}")
    print("=" * 80)

    # ----------------------------------------------------------- weights
    print(f"\nExtracting weights from layers {layer_idx_start}..{layer_idx_start + n_layers - 1}...")
    all_weights = []
    for i in range(n_layers):
        layer_module = layers[layer_idx_start + i]
        w = _extract_layer_weights(
            layer_module, hidden, inter, head_dim,
            num_heads, native_head_dim,
            num_kv_heads=num_kv_heads,
            native_mode=native_mode,
        )
        all_weights.append(w)
        if native_mode:
            print(f"  Layer {i}: W_q={w['W_q'].shape}, W_o={w['W_o'].shape}, "
                  f"W_gate={w['W_gate'].shape}, "
                  f"K_heads={len(w['W_k_heads'])}x{w['W_k_heads'][0].shape}, eps={w['eps']}")
        else:
            print(f"  Layer {i}: W_q={w['W_q'].shape}, W_o={w['W_o'].shape}, "
                  f"W_gate={w['W_gate'].shape}, W_k={w['W_k'].shape}, eps={w['eps']}")

    eps = all_weights[0]["eps"]

    # ----------------------------------------------------------- test data
    torch.manual_seed(seed)

    # Embedding lookup: use real HF embedding table if available
    if embed is not None:
        input_ids = torch.randint(0, native_cfg["vocab_size"] or 32000, (seq_len,))
        with torch.no_grad():
            token_embeds = embed(input_ids).float()
        if token_embeds.dim() == 3:
            token_embeds = token_embeds.squeeze(0)
        # Slice to target hidden dim (native mode uses full hidden)
        token_embeds = token_embeds[:, :hidden]
        print(f"\nEmbedding lookup: input_ids shape={input_ids.shape}, "
              f"token_embeds={token_embeds.shape}")
    else:
        token_embeds = torch.randn(seq_len, hidden)
        print(f"\nNo embed_tokens found; using random token_embeds: {token_embeds.shape}")

    # Llama-style models use RoPE (not learned position embeddings).
    # Set pos_weight to zeros so embedding_add is a no-op for position.
    pos_weight = torch.zeros(seq_len, hidden)

    # K/V test data: native mode computes K/V on-chip, legacy mode uses precomputed
    if native_mode:
        # K/V are computed on-chip from X_normed @ W_k/W_v — no precomputed K/V needed
        print(f"pos_weight:   zeros {pos_weight.shape} (Llama uses RoPE, not learned pos embed)")
        for i in range(n_layers):
            for kv_h in range(num_kv_heads):
                print(f"  W_k_{i}_h{kv_h}: {all_weights[i]['W_k_heads'][kv_h].shape}, "
                      f"W_v_{i}_h{kv_h}: {all_weights[i]['W_v_heads'][kv_h].shape}")
    else:
        K_mats = []
        V_mats = []
        for i in range(n_layers):
            X_ctx = torch.randn(seq_len, hidden)
            K_mats.append(X_ctx @ all_weights[i]["W_k"])
            V_mats.append(X_ctx @ all_weights[i]["W_v"])

        print(f"pos_weight:   zeros {pos_weight.shape} (Llama uses RoPE, not learned pos embed)")
        for i in range(n_layers):
            print(f"  K_{i}: {K_mats[i].shape}, V_{i}: {V_mats[i].shape}")
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
    print(f"\n--- CPU Golden Reference ({_prec_label}) ---")

    if native_mode:
        W_k_q_heads = [[_qw(all_weights[i]["W_k_heads"][h])
                        for h in range(num_kv_heads)]
                       for i in range(n_layers)]
        W_v_q_heads = [[_qw(all_weights[i]["W_v_heads"][h])
                        for h in range(num_kv_heads)]
                       for i in range(n_layers)]
    if native_mode:
        R_matrix = _make_rotate_half_matrix(head_dim)
        R_rope_q = _qw(R_matrix)
        cos_table, sin_table = _make_rope_tables(seq_len, head_dim, native_cfg["rope_theta"])

    else:
        K_q_list = [_qw(K_mats[i]) for i in range(n_layers)]
        V_q_list = [_qw(V_mats[i]) for i in range(n_layers)]

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
        X_bf = _to_inter(X_gold)
        # Hardware: scalar rms stays in f32, V_MUL_VF multiplies BF16 * f32 -> BF16
        rms = torch.rsqrt(_from_inter(X_bf).pow(2).mean(-1, keepdim=True) + eps)
        X_gold = (_from_inter(X_bf) * rms).to(torch.bfloat16).float()
        if native_mode:
            Q_gold = _ksplit_matmul(X_gold, W_q_q, mlen, _HW_MAX_K_TILES, _to_inter, _from_inter)
            Q_gold = _from_inter(_to_inter(Q_gold))  # VRAM write: BF16

            K_q_heads_i = []
            V_q_heads_i = []
            for kv_h in range(num_kv_heads):
                K_h = _ksplit_matmul(X_gold, W_k_q_heads[i][kv_h], mlen, _HW_MAX_K_TILES, _to_inter, _from_inter)
                V_h = _ksplit_matmul(X_gold, W_v_q_heads[i][kv_h], mlen, _HW_MAX_K_TILES, _to_inter, _from_inter)
                K_rot_h = _from_inter(_to_inter(torch.matmul(_from_inter(_to_inter(K_h)), _from_inter(_to_inter(R_rope_q)))))
                # Hardware RoPE: V_MUL_VV (BF16*BF16->BF16), V_ADD_VV (BF16+BF16->BF16)
                K_cos = _from_inter(_to_inter(K_h.to(torch.bfloat16).float() * cos_table.to(torch.bfloat16).float()))
                K_rot_sin = _from_inter(_to_inter(K_rot_h.to(torch.bfloat16).float() * sin_table.to(torch.bfloat16).float()))
                K_h = _from_inter(_to_inter(K_cos) + _to_inter(K_rot_sin))
                # Hardware stores K/V to HBM as MXFP8, loads back as BF16 for attention
                K_q_heads_i.append(_from_inter(_to_inter(_qw(K_h))))
                V_q_heads_i.append(_from_inter(_to_inter(_qw(V_h))))

            O_heads = []
            for h in range(num_heads):
                kv_h = h // ratio
                Q_h = Q_gold[:, h * head_dim:(h + 1) * head_dim]
                Q_rot_h = _from_inter(_to_inter(torch.matmul(_from_inter(_to_inter(Q_h)), _from_inter(_to_inter(R_rope_q)))))
                # Hardware RoPE: V_MUL_VV (BF16*BF16->BF16), V_ADD_VV (BF16+BF16->BF16)
                Q_cos = _from_inter(_to_inter(Q_h.to(torch.bfloat16).float() * cos_table.to(torch.bfloat16).float()))
                Q_rot_sin = _from_inter(_to_inter(Q_rot_h.to(torch.bfloat16).float() * sin_table.to(torch.bfloat16).float()))
                Q_h = _from_inter(_to_inter(Q_cos) + _to_inter(Q_rot_sin))
                O_h = _flash_attn_ref(Q_h, K_q_heads_i[kv_h], V_q_heads_i[kv_h], scale, causal=True)
                O_heads.append(O_h)
            attn_out = _from_inter(_to_inter(torch.cat(O_heads, dim=1)))  # VRAM write: BF16
            O_gold = _ksplit_matmul(attn_out, W_o_q, mlen, _HW_MAX_K_TILES, _to_inter, _from_inter)
            O_gold = _from_inter(_to_inter(O_gold))  # VRAM write: BF16
            X_gold = _from_inter(_to_inter(O_gold + residual))  # residual add -> VRAM write: BF16
        else:
            attn_out = _flash_attn_ref(X_gold, K_q_list[i], V_q_list[i], scale, causal=True)
            X_gold = _from_inter(_to_inter(attn_out + residual))  # residual add -> VRAM write: BF16

        # --- FFN block ---
        residual = X_gold.clone()
        X_bf = _to_inter(X_gold)
        # Hardware: scalar rms stays in f32, V_MUL_VF multiplies BF16 * f32 -> BF16
        rms = torch.rsqrt(_from_inter(X_bf).pow(2).mean(-1, keepdim=True) + eps)
        X_gold = (_from_inter(X_bf) * rms).to(torch.bfloat16).float()
        # Hardware path: MXFP8 (HBM) → BF16 (MRAM) → float32 (M_MM)
        # Use _ksplit_matmul to match hardware K-split BF16 precision loss for
        # projections that exceed MAX_K_TILES (e.g., hidden=576 → 9 tiles, inter=1536 → 24 tiles)
        up_out = _ksplit_matmul(X_gold, W_up_q, mlen, _HW_MAX_K_TILES, _to_inter, _from_inter)
        gate_out = _ksplit_matmul(X_gold, W_gate_q, mlen, _HW_MAX_K_TILES, _to_inter, _from_inter)
        # Hardware: SiLU(up) * gate -> BF16 VRAM write before down projection
        silu_gate = _to_inter(F.silu(_from_inter(_to_inter(up_out))) * _from_inter(_to_inter(gate_out)))
        X_gold = _ksplit_matmul(_from_inter(silu_gate), W_down_q, mlen, _HW_MAX_K_TILES, _to_inter, _from_inter)
        X_gold = _from_inter(_to_inter(X_gold))  # VRAM write: BF16 after down proj
        X_gold = _from_inter(_to_inter(X_gold + residual))  # residual add -> VRAM write: BF16

        print(f"  After layer {i}: X_gold[0,:4] = {X_gold[0, :4].tolist()}")

    # Final norm
    X_gold = _rms_norm_ref(X_gold, eps)

    # lm_head projection (optional)
    if include_lm_head and lm_head_weight is not None:
        W_lm_q = quantize_to_mxfp(lm_head_weight)
        # Hardware: MXFP8 → BF16 (MRAM) → float32 (M_MM)
        logits_gold = torch.matmul(
            X_gold.to(torch.bfloat16).float(), W_lm_q.to(torch.bfloat16).float()
        ).to(torch.bfloat16).float()
        golden_out = logits_gold
        print(f"  logits_gold: {golden_out.shape}")
    else:
        golden_out = X_gold
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ----------------------------------------------------------- HF ground truth
    # Same pipeline as golden but pure float32: no MXFP8 quantization, no BF16 casting.
    # This is the "best possible" reference for the sliced weight dimensions being tested.
    print(f"\n--- HF Ground Truth (float32, {n_layers} layers, no quantization) ---")
    with torch.no_grad():
        X_hf = token_embeds.clone() + pos_weight  # embedding_add

        for i in range(n_layers):
            w = all_weights[i]
            eps_i = w["eps"]

            # --- Attention block (float32) ---
            residual = X_hf.clone()
            rms = torch.rsqrt(X_hf.pow(2).mean(-1, keepdim=True) + eps_i)
            X_normed = X_hf * rms

            if native_mode:
                Q_hf = X_normed @ w["W_q"].float()

                K_hf_heads = []
                V_hf_heads = []
                R_mat_f32 = R_matrix.float()
                cos_f32 = cos_table.float()
                sin_f32 = sin_table.float()
                for kv_h in range(num_kv_heads):
                    K_h = X_normed @ w["W_k_heads"][kv_h].float()
                    V_h = X_normed @ w["W_v_heads"][kv_h].float()
                    # RoPE on K_h (float32)
                    K_rot = K_h @ R_mat_f32
                    K_h = K_h * cos_f32 + K_rot * sin_f32
                    K_hf_heads.append(K_h)
                    V_hf_heads.append(V_h)

                O_heads_hf = []
                for h in range(num_heads):
                    kv_h = h // ratio
                    Q_h = Q_hf[:, h * head_dim:(h + 1) * head_dim]
                    # RoPE on Q_h (float32)
                    Q_rot = Q_h @ R_mat_f32
                    Q_h = Q_h * cos_f32 + Q_rot * sin_f32
                    O_h = _flash_attn_ref(Q_h, K_hf_heads[kv_h], V_hf_heads[kv_h], scale, causal=True)
                    O_heads_hf.append(O_h)
                attn_out_hf = torch.cat(O_heads_hf, dim=1)
                O_hf = attn_out_hf @ w["W_o"].float()
                X_hf = O_hf + residual
            else:
                attn_out_hf = _flash_attn_ref(X_normed, K_mats[i].float(), V_mats[i].float(), scale, causal=True)
                X_hf = attn_out_hf + residual

            # --- FFN block (float32) ---
            residual = X_hf.clone()
            rms = torch.rsqrt(X_hf.pow(2).mean(-1, keepdim=True) + eps_i)
            X_normed = X_hf * rms
            up_out = F.silu(X_normed @ w["W_up"].float())
            gate_out = X_normed @ w["W_gate"].float()
            X_hf = (up_out * gate_out) @ w["W_down"].float() + residual

            print(f"  After layer {i}: X_hf[0,:4] = {X_hf[0, :4].tolist()}")

        # Final norm (float32)
        rms = torch.rsqrt(X_hf.pow(2).mean(-1, keepdim=True) + eps)
        X_hf = X_hf * rms

        if include_lm_head and lm_head_weight is not None:
            hf_ground_truth = (X_hf @ lm_head_weight.float())
        else:
            hf_ground_truth = X_hf

    print(f"  hf_ground_truth: {hf_ground_truth.shape}")
    print(f"  hf_ground_truth[0,:4]: {hf_ground_truth[0, :4].tolist()}")

    # ----------------------------------------------------------- PLENA ISA
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=REAL_DATA_RATIO)

    # Shared inputs
    x_input = prog.input("X", shape=(seq_len, hidden))
    pos_input = prog.input("POS", shape=(seq_len, hidden))

    # RoPE inputs (native mode only: R_rope matrix + cos/sin tables)
    if native_mode:
        rope_theta = native_cfg["rope_theta"]
        R_matrix = _make_rotate_half_matrix(head_dim)
        cos_table, sin_table = _make_rope_tables(seq_len, head_dim, rope_theta)

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
        wq = prog.input(f"W_q_{i}", shape=(hidden, total_q_dim))
        wo = prog.input(f"W_o_{i}", shape=(total_q_dim, hidden))
        if native_mode:
            wk_heads = []
            wv_heads = []
            for kv_h in range(num_kv_heads):
                wk_heads.append(prog.input(f"W_k_{i}_h{kv_h}", shape=(hidden, head_dim)))
                wv_heads.append(prog.input(f"W_v_{i}_h{kv_h}", shape=(hidden, head_dim)))
            li_entry = {
                "W_q": wq, "W_o": wo,
                "W_k_heads": wk_heads, "W_v_heads": wv_heads,
            }
        else:
            ki = prog.input(f"K_{i}", shape=(seq_len, head_dim))
            vi = prog.input(f"V_{i}", shape=(seq_len, head_dim))
            li_entry = {
                "W_q": wq, "W_o": wo,
                "K": ki, "V": vi,
            }
        wg = prog.input(f"W_gate_{i}", shape=(hidden, inter))
        wu = prog.input(f"W_up_{i}", shape=(hidden, inter))
        wd = prog.input(f"W_down_{i}", shape=(inter, hidden))
        li_entry.update({"W_gate": wg, "W_up": wu, "W_down": wd})
        layer_inputs.append(li_entry)

    # lm_head weight input (after all layer weights in HBM layout)
    lm_head_input = None
    if include_lm_head and lm_head_weight is not None:
        lm_head_input = prog.input("W_lm_head", shape=(hidden, vocab_size))

    # Load activations to VRAM
    X_batch = prog.load_batch(x_input, name="X")
    POS_batch = prog.load_batch(pos_input, name="POS")
    ops.embedding_add(prog, X_batch, POS_batch)  # X += POS in-place

    # VRAM layout hazard: ffn_asm writes gate/up intermediates at absolute
    # address batch*hidden spanning up to batch*hidden + 2*inter*batch.
    # The residual scratch buffer must be placed ABOVE this region.
    _ffn_intermediate_end = seq_len * hidden + 2 * inter * seq_len
    _current_bump = 2 * seq_len * hidden  # X + POS already allocated
    if native_mode:
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
        prog._compiler.emit_comment(f"=== LAYER {i}/{n_layers} START ===")

        # --- Attention block ---
        # Save residual: scratch = current (zero then add)
        prog.vram_fill_zero(scratch)
        prog.vram_add(scratch, current)

        # Norm (in-place on current)
        prog.rms_norm(current, eps_offset=3, reci_hid_offset=4)

        if native_mode:
            # Q projection: current (seq, hidden) @ W_q (hidden, total_q_dim)
            Q = _linear_projection(prog, current, li["W_q"], f"Q_{i}")

            # Per-KV-head: compute K/V on-chip, store to HBM, then run Q-heads
            q_full_addr = prog._compiler.get_vram_addr(Q.name)

            # Allocate O_full for concatenated head outputs
            O_full = prog.alloc(f"O_full_{i}", seq_len, total_q_dim)
            o_full_addr = prog._compiler.get_vram_addr(O_full.name)

            # Store K/V InputVars for each KV head (filled during loop below)
            kv_stored = []

            for kv_h in range(num_kv_heads):
                # K projection: current (seq, hidden) @ W_k_h (hidden, head_dim)
                K_h = _linear_projection(prog, current, li["W_k_heads"][kv_h], f"K_{i}_h{kv_h}")
                # V projection: current (seq, hidden) @ W_v_h (hidden, head_dim)
                V_h = _linear_projection(prog, current, li["W_v_heads"][kv_h], f"V_{i}_h{kv_h}")

                # RoPE on K_h: K_rot = linear(K_h, R_rope), then K_h = K_h * cos + K_rot * sin
                K_rot = _linear_projection(prog, K_h, r_input, f"K_rot_{i}_h{kv_h}")
                prog.rope(K_h, K_rot, COS, SIN)
                prog.free_tensor(K_rot)

                # Store K/V from VRAM to HBM (auto-allocates HBM space)
                K_stored = prog.store(K_h, name=f"K_stored_{i}_h{kv_h}")
                V_stored = prog.store(V_h, name=f"V_stored_{i}_h{kv_h}")
                kv_stored.append((K_stored, V_stored))

                # Free K/V VRAM — data is in HBM now
                prog.free_tensor(K_h)
                prog.free_tensor(V_h)

            for h in range(num_heads):
                kv_h = h // ratio
                K_stored, V_stored = kv_stored[kv_h]

                # View for this head's Q slice: column block h in Q_full
                q_h_addr = q_full_addr + h * seq_len * mlen
                Q_h = prog.alloc_at(f"Q_h{h}_{i}", seq_len, head_dim, q_h_addr)

                # RoPE on Q_h: Q_rot = linear(Q_h, R_rope), then Q_h = Q_h * cos + Q_rot * sin
                Q_rot = _linear_projection(prog, Q_h, r_input, f"Q_rot_{i}_h{h}")
                prog.rope(Q_h, Q_rot, COS, SIN)
                prog.free_tensor(Q_rot)

                # Single-head flash attention (K/V read from HBM) with causal mask
                O_h = ops.flash_attention(prog, Q_h, K_stored, V_stored, scale,
                                          causal_mask=CAUSAL_MASK)

                # Copy O_h to the right column block of O_full
                o_h_dest_addr = o_full_addr + h * seq_len * mlen
                O_h_dest = prog.alloc_at(f"O_dest_h{h}_{i}", seq_len, head_dim, o_h_dest_addr)
                prog.vram_fill_zero(O_h_dest)
                prog.vram_add(O_h_dest, O_h)

                # Free flash_attention intermediates (S, PV, O) to reclaim VRAM
                for _tmp_name in ("O", "S", "PV"):
                    _tmp_var = prog._tensors.get(_tmp_name)
                    if _tmp_var is not None:
                        prog.free_tensor(_tmp_var)
                        prog._tensors.pop(_tmp_name, None)

            # O projection: O_full @ W_o -> O_proj (seq, hidden)
            O_proj = _linear_projection(prog, O_full, li["W_o"], f"O_proj_{i}")

            # Attention residual: O_proj += scratch
            prog.vram_add(O_proj, scratch)
            current_after_attn = O_proj
        else:
            # Legacy: X is Q directly (no projections), with causal mask
            O = ops.flash_attention(prog, current, li["K"], li["V"], scale,
                                    causal_mask=CAUSAL_MASK)
            prog.vram_add(O, scratch)
            current_after_attn = O

        # --- FFN block ---
        # Save residual: scratch = current_after_attn (zero then add)
        prog.vram_fill_zero(scratch)
        prog.vram_add(scratch, current_after_attn)

        # Norm (in-place)
        prog.rms_norm(current_after_attn, eps_offset=3, reci_hid_offset=4)

        # FFN (in-place)
        ops.ffn(prog, current_after_attn, li["W_gate"], li["W_up"], li["W_down"])

        # FFN residual
        prog.vram_add(current_after_attn, scratch)

        current = current_after_attn  # carry forward
        prog._compiler.emit_comment(f"=== LAYER {i}/{n_layers} COMPLETE ===")

    # Final norm
    prog.rms_norm(current, eps_offset=3, reci_hid_offset=4)

    # lm_head projection (optional): logits = linear(current, W_lm_head)
    if include_lm_head and lm_head_input is not None:
        logits = _linear_projection(prog, current, lm_head_input, "logits")
        current = logits

    isa_code = prog.compile()
    isa_code = _fix_large_immediates(isa_code)
    lines = isa_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ----------------------------------------------------------- build return
    input_tensors = {"X": token_embeds, "POS": pos_weight}
    data_order = ["X", "POS"]
    if native_mode:
        input_tensors["R_rope"] = R_matrix
        input_tensors["COS"] = cos_table
        input_tensors["SIN"] = sin_table
        data_order.extend(["R_rope", "COS", "SIN"])
    input_tensors["causal_mask"] = causal_mask_data
    data_order.append("causal_mask")
    for i in range(n_layers):
        input_tensors[f"W_q_{i}"] = all_weights[i]["W_q"]
        input_tensors[f"W_o_{i}"] = all_weights[i]["W_o"]
        if native_mode:
            for kv_h in range(num_kv_heads):
                input_tensors[f"W_k_{i}_h{kv_h}"] = all_weights[i]["W_k_heads"][kv_h]
                input_tensors[f"W_v_{i}_h{kv_h}"] = all_weights[i]["W_v_heads"][kv_h]
        else:
            input_tensors[f"K_{i}"] = K_mats[i]
            input_tensors[f"V_{i}"] = V_mats[i]
        input_tensors[f"W_gate_{i}"] = all_weights[i]["W_gate"]
        input_tensors[f"W_up_{i}"] = all_weights[i]["W_up"]
        input_tensors[f"W_down_{i}"] = all_weights[i]["W_down"]

        kv_keys = []
        if native_mode:
            for kv_h in range(num_kv_heads):
                kv_keys.extend([f"W_k_{i}_h{kv_h}", f"W_v_{i}_h{kv_h}"])
        else:
            kv_keys = [f"K_{i}", f"V_{i}"]
        data_order.extend([
            f"W_q_{i}", f"W_o_{i}",
            *kv_keys,
            f"W_gate_{i}", f"W_up_{i}", f"W_down_{i}",
        ])

    # lm_head weight in input_tensors + data_order
    if include_lm_head and lm_head_weight is not None:
        input_tensors["W_lm_head"] = lm_head_weight
        data_order.append("W_lm_head")

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
    o_vram_addr = prog._compiler.get_vram_addr(current.name)

    # Output dimensions depend on whether lm_head is included
    if include_lm_head and lm_head_weight is not None:
        out_features = vocab_size
    else:
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
        "model_type": native_cfg["model_type"],
        "hidden_size": hidden,
        "inter_dim": inter,
        "num_layers": n_layers,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "num_heads": num_heads if native_mode else 1,
        "num_kv_heads": num_kv_heads if native_mode else 1,
        "native_mode": native_mode,
        "include_lm_head": include_lm_head and lm_head_weight is not None,
        "vocab_size": vocab_size if (include_lm_head and lm_head_weight is not None) else None,
        "mlen": mlen,
        "blen": blen,
        "isa_lines": len(lines),
    }

    print(f"\nCompilation complete: {info['isa_lines']} ISA lines, "
          f"{n_layers} layers, output at VRAM row {o_vram_addr // mlen}")
    if include_lm_head and lm_head_weight is not None:
        print(f"  lm_head: output shape=({seq_len}, {vocab_size})")

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


# ---------------------------------------------------------------------------
# Convenience: compile + run emulator + compare
# ---------------------------------------------------------------------------
def compile_and_run(
    model,
    build_dir,
    **kwargs,
) -> dict:
    """Compile, run emulator, and compare against golden.

    Convenience wrapper that calls compile_hf_model, sets up simulation
    environment, runs the Rust transactional emulator, and compares output.

    Args:
        model:     nn.Module (HF CausalLM model, already loaded)
        build_dir: Directory for simulation artifacts
        **kwargs:  Forwarded to compile_hf_model (seq_len, hidden_size, etc.)

    Returns:
        dict with compilation info + comparison results including
        'allclose_match_rate' percentage.
    """
    from transactional_emulator.tools.create_sim_env import create_sim_env
    from compiler.sim_env_utils.build_env import create_mem_for_sim
    from transactional_emulator.testbench.emulator_runner import (
        compare_emulator_output,
    )

    result = compile_hf_model(model, **kwargs)
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    mlen = kwargs.get("mlen", 64)
    blen = kwargs.get("blen", 4)
    asm_name = f"model_{result['info']['model_type']}_{result['info']['num_layers']}L"

    # Write sim env files
    create_sim_env(
        result["input_tensors"],
        result["isa"],
        {"original_output": result["golden_output"]},
        result["fp_preload"],
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=asm_name,
        data=None,
        specified_data_order=result["data_order"],
        build_path=build_dir,
    )

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(result["comparison_params"], f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(result["isa"])

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {result['comparison_params']['start_row_idx']}")
    print(f"  Layers: {result['info']['num_layers']}, data_order: {result['data_order']}")

    # Run emulator and compare (don't exit on failure — VRAM stage comparison follows)
    from transactional_emulator.testbench.emulator_runner import update_plena_config, run_emulator
    update_plena_config(vlen=mlen, mlen=mlen, blen=blen, verbose=False)
    print("\n--- Running Rust transactional emulator ---")
    run_emulator(build_dir)

    print("\n--- Comparing emulator output vs golden ---")
    comp_results, _params = compare_emulator_output(build_dir)
    from transactional_emulator.tools.check_mem import print_comparison_results
    print_comparison_results(comp_results, verbose=True, comparison_params=_params)

    if comp_results["allclose_pass"]:
        print(f"\n[ATen-style {asm_name} test PASSED - ISA generated + emulator verified]")
    else:
        print(f"\n[ATen-style {asm_name} test FAILED - emulator numerical check failed]")

    # Three-way comparison
    golden = result["golden_output"]
    hf_gt = result["hf_ground_truth"]
    print("\n--- Three-way comparison ---")
    if hf_gt is not None and golden is not None:
        # HF float32 vs golden (MXFP8 + BF16)
        n = min(hf_gt.numel(), golden.numel())
        allclose_hf_vs_gold = (
            torch.isclose(hf_gt.float().flatten()[:n],
                          golden.float().flatten()[:n], atol=1e-2)
            .float().mean().item() * 100
        )
        print(f"  HF float32 vs golden (MXFP8+BF16):  {allclose_hf_vs_gold:.2f}% allclose")
    # Emulator vs golden: reported by compare_emulator_output
    emu_match = comp_results.get("allclose_match_rate", None)
    if emu_match is not None:
        print(f"  Emulator vs golden (MXFP8+BF16):    {emu_match:.2f}% allclose")

    # VRAM stage comparison: validates each pipeline segment using
    # emulator's own intermediates as golden input (immune to accumulation drift)
    try:
        from compiler.aten.vram_stage_compare import compare_stages
        emulator_dir = Path(__file__).parent.parent.parent / "transactional_emulator"
        vram_path = str(emulator_dir / "vram_dump.bin")
        print("\n--- VRAM stage comparison (authoritative) ---")
        stage_results = compare_stages(
            vram_path=vram_path,
            build_dir=str(build_dir),
            hidden=result["info"]["hidden_size"],
            inter=result["info"].get("inter_dim", result["info"]["hidden_size"] * 4),
            num_heads=result["info"]["num_heads"],
            num_kv_heads=result["info"]["num_kv_heads"],
        )
        stage_pass = stage_results.get("norm+FFN+norm", 0) >= 99.0
        comp_results["vram_stage_allclose"] = stage_results.get("norm+FFN+norm", None)
        comp_results["vram_stage_pass"] = stage_pass
    except Exception as e:
        print(f"  (skipped: {e})")

    comp_results, _params = compare_emulator_output(build_dir)

    # Three-way comparison
    golden = result["golden_output"]
    hf_gt = result["hf_ground_truth"]
    print("\n--- Three-way comparison ---")
    if hf_gt is not None and golden is not None:
        # HF float32 vs golden (MXFP8 + BF16)
        n = min(hf_gt.numel(), golden.numel())
        allclose_hf_vs_gold = (
            torch.isclose(hf_gt.float().flatten()[:n],
                          golden.float().flatten()[:n], atol=1e-2)
            .float().mean().item() * 100
        )
        print(f"  HF float32 vs golden (MXFP8+BF16):  {allclose_hf_vs_gold:.2f}% allclose")
    # Emulator vs golden: reported by compare_emulator_output
    emu_match = comp_results.get("allclose_match_rate", None)
    if emu_match is not None:
        print(f"  Emulator vs golden (MXFP8+BF16):    {emu_match:.2f}% allclose")

    return {**result["info"], **comp_results}
