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
from pathlib import Path

import torch
import torch.nn.functional as F

from compiler.aten.plena_compiler import PlenaCompiler
from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops
from transactional_emulator.testbench.model_layer_test_builder import quantize_to_mxfp
import re

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
    return {
        "hidden_size": native_hidden,
        "inter_dim": native_inter,
        "num_heads": native_heads,
        "num_kv_heads": native_kv_heads,
        "head_dim": native_head_dim,
        "eps": eps,
        "rope_theta": rope_theta,
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
def _flash_attn_ref(Q, K, V, scale):
    """CPU reference: scaled dot-product attention."""
    scores = (Q @ K.T) * scale
    attn = F.softmax(scores, dim=-1)
    return attn @ V


def _rms_norm_ref(x, eps):
    """CPU reference: RMS normalization matching PLENA hardware (BF16 intermediate)."""
    x_bf = x.to(torch.bfloat16)
    rms = torch.rsqrt(x_bf.float().pow(2).mean(-1, keepdim=True) + eps).to(torch.bfloat16)
    return (x_bf * rms).float()


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
) -> dict:
    """Compile a HuggingFace model to PLENA ISA via PlenaCompiler.

    Walks the nn.Module tree, extracts weights, and generates ISA with
    proper residual connections for multi-layer decoders.

    The pipeline implemented (per-layer, with pre-norm + residual):
        X = embedding_add(token_embeds, pos_weight)
        for each layer:
            residual = X
            X = rms_norm(X)
            Q = linear(X, W_q)           # Q projection
            O = flash_attention(Q, K, V, scale)
            O = linear(O, W_o)           # O projection
            X = O + residual
            residual = X
            X = rms_norm(X)
            X = ffn(X, gate, up, down)
            X = X + residual
        X = rms_norm(X)  # final norm

    RoPE is omitted (orthogonal to multi-layer compilation testing).
    Q and O linear projections are included for the attention block.

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
    token_embeds = torch.randn(seq_len, hidden)
    pos_weight = torch.randn(seq_len, hidden)

    # K/V test data: per-KV-head in native mode, single matrix in legacy mode
    if native_mode:
        # List of lists: K_head_mats[layer][kv_head] = (seq_len, head_dim)
        K_head_mats = []
        V_head_mats = []
        for i in range(n_layers):
            k_heads_i = []
            v_heads_i = []
            for kv_h in range(num_kv_heads):
                X_ctx = torch.randn(seq_len, hidden)
                k_heads_i.append(X_ctx @ all_weights[i]["W_k_heads"][kv_h])
                v_heads_i.append(X_ctx @ all_weights[i]["W_v_heads"][kv_h])
            K_head_mats.append(k_heads_i)
            V_head_mats.append(v_heads_i)

        print(f"\ntoken_embeds: {token_embeds.shape}")
        print(f"pos_weight:   {pos_weight.shape}")
        for i in range(n_layers):
            for kv_h in range(num_kv_heads):
                print(f"  K_{i}_h{kv_h}: {K_head_mats[i][kv_h].shape}, "
                      f"V_{i}_h{kv_h}: {V_head_mats[i][kv_h].shape}")
    else:
        K_mats = []
        V_mats = []
        for i in range(n_layers):
            X_ctx = torch.randn(seq_len, hidden)
            K_mats.append(X_ctx @ all_weights[i]["W_k"])
            V_mats.append(X_ctx @ all_weights[i]["W_v"])

        print(f"\ntoken_embeds: {token_embeds.shape}")
        print(f"pos_weight:   {pos_weight.shape}")
        for i in range(n_layers):
            print(f"  K_{i}: {K_mats[i].shape}, V_{i}: {V_mats[i].shape}")
    print(f"attn_scale: {scale:.6f}")

    # ----------------------------------------------------------- golden ref
    print("\n--- CPU Golden Reference (MXFP8 quantized HBM + BF16 intermediates) ---")

    if native_mode:
        # Quantize per-head K/V
        K_q_heads = [[quantize_to_mxfp(K_head_mats[i][h]) for h in range(num_kv_heads)]
                     for i in range(n_layers)]
        V_q_heads = [[quantize_to_mxfp(V_head_mats[i][h]) for h in range(num_kv_heads)]
                     for i in range(n_layers)]
    else:
        K_q_list = [quantize_to_mxfp(K_mats[i]) for i in range(n_layers)]
        V_q_list = [quantize_to_mxfp(V_mats[i]) for i in range(n_layers)]

    X_gold = token_embeds.clone() + pos_weight  # embedding_add
    ratio = num_heads // num_kv_heads

    for i in range(n_layers):
        w = all_weights[i]
        W_q_q = quantize_to_mxfp(w["W_q"])
        W_o_q = quantize_to_mxfp(w["W_o"])
        W_gate_q = quantize_to_mxfp(w["W_gate"])
        W_up_q = quantize_to_mxfp(w["W_up"])
        W_down_q = quantize_to_mxfp(w["W_down"])

        # --- Attention block ---
        residual = X_gold.clone()
        # rms_norm with bfloat16 to match PLENA
        X_bf = X_gold.to(torch.bfloat16)
        rms = torch.rsqrt(X_bf.float().pow(2).mean(-1, keepdim=True) + eps).to(torch.bfloat16)
        X_gold = (X_bf * rms).float()
        if native_mode:
            # Q projection: X @ W_q (MXFP8-quantized weight, BF16 intermediate)
            Q_gold = torch.matmul(X_gold.to(torch.bfloat16).float(), W_q_q.float()).to(torch.bfloat16).float()
            # Per-head flash attention
            O_heads = []
            for h in range(num_heads):
                kv_h = h // ratio
                Q_h = Q_gold[:, h * head_dim:(h + 1) * head_dim]
                O_h = _flash_attn_ref(Q_h, K_q_heads[i][kv_h], V_q_heads[i][kv_h], scale)
                O_heads.append(O_h)
            attn_out = torch.cat(O_heads, dim=1)  # (seq, num_heads * head_dim)
            # O projection
            O_gold = torch.matmul(attn_out.to(torch.bfloat16).float(), W_o_q.float()).to(torch.bfloat16).float()
            X_gold = O_gold + residual
        else:
            # Legacy: X is Q directly (no projection)
            attn_out = _flash_attn_ref(X_gold, K_q_list[i], V_q_list[i], scale)
            X_gold = attn_out + residual

        # --- FFN block ---
        residual = X_gold.clone()
        # rms_norm with bfloat16
        X_bf = X_gold.to(torch.bfloat16)
        rms = torch.rsqrt(X_bf.float().pow(2).mean(-1, keepdim=True) + eps).to(torch.bfloat16)
        X_gold = (X_bf * rms).float()
        # FFN with MXFP8 weights + BF16 intermediates
        up_out = torch.matmul(X_gold.to(torch.bfloat16).float(), W_up_q.float()).to(torch.bfloat16)
        gate_out = torch.matmul(X_gold.to(torch.bfloat16).float(), W_gate_q.float()).to(torch.bfloat16)
        silu_gate = (F.silu(up_out.float()) * gate_out.float()).to(torch.bfloat16)
        X_gold = torch.matmul(silu_gate.float(), W_down_q.float()).to(torch.bfloat16).float()
        # FFN residual
        X_gold = X_gold + residual

        print(f"  After layer {i}: X_gold[0,:4] = {X_gold[0, :4].tolist()}")

    # Final norm
    X_gold = _rms_norm_ref(X_gold, eps)

    golden_out = X_gold
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ----------------------------------------------------------- PLENA ISA
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=REAL_DATA_RATIO)

    # Shared inputs
    x_input = prog.input("X", shape=(seq_len, hidden))
    pos_input = prog.input("POS", shape=(seq_len, hidden))

    # Per-layer weight inputs (order determines HBM layout)
    layer_inputs = []
    for i in range(n_layers):
        wq = prog.input(f"W_q_{i}", shape=(hidden, total_q_dim))
        wo = prog.input(f"W_o_{i}", shape=(total_q_dim, hidden))
        if native_mode:
            k_heads = []
            v_heads = []
            for kv_h in range(num_kv_heads):
                k_heads.append(prog.input(f"K_{i}_h{kv_h}", shape=(seq_len, head_dim)))
                v_heads.append(prog.input(f"V_{i}_h{kv_h}", shape=(seq_len, head_dim)))
            li_entry = {
                "W_q": wq, "W_o": wo,
                "K_heads": k_heads, "V_heads": v_heads,
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

    # Load activations to VRAM
    X_batch = prog.load_batch(x_input, name="X")
    POS_batch = prog.load_batch(pos_input, name="POS")
    ops.embedding_add(prog, X_batch, POS_batch)  # X += POS in-place

    # VRAM layout hazard: ffn_asm writes gate/up intermediates at absolute
    # address batch*hidden spanning up to batch*hidden + 2*inter*batch.
    # The residual scratch buffer must be placed ABOVE this region.
    _ffn_intermediate_end = seq_len * hidden + 2 * inter * seq_len
    _current_bump = 2 * seq_len * hidden  # X + POS already allocated
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

        # --- Attention block ---
        # Save residual: scratch = current (zero then add)
        prog.vram_fill_zero(scratch)
        prog.vram_add(scratch, current)

        # Norm (in-place on current)
        prog.rms_norm(current, eps_offset=3, reci_hid_offset=4)

        if native_mode:
            # Q projection: current (seq, hidden) @ W_q (hidden, total_q_dim)
            Q = _linear_projection(prog, current, li["W_q"], f"Q_{i}")

            # Per-head flash attention with VRAM views
            q_full_addr = prog._compiler.get_vram_addr(Q.name)

            # Allocate O_full for concatenated head outputs
            O_full = prog.alloc(f"O_full_{i}", seq_len, total_q_dim)
            o_full_addr = prog._compiler.get_vram_addr(O_full.name)

            for h in range(num_heads):
                kv_h = h // ratio

                # View for this head's Q slice: column block h in Q_full
                q_h_addr = q_full_addr + h * seq_len * mlen
                Q_h = prog.alloc_at(f"Q_h{h}_{i}", seq_len, head_dim, q_h_addr)

                # Single-head flash attention (allocates S, PV, O internally)
                O_h = ops.flash_attention(prog, Q_h, li["K_heads"][kv_h], li["V_heads"][kv_h], scale)

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
            # Legacy: X is Q directly (no projections)
            O = ops.flash_attention(prog, current, li["K"], li["V"], scale)
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

    # Final norm
    prog.rms_norm(current, eps_offset=3, reci_hid_offset=4)

    isa_code = prog.compile()
    isa_code = _fix_large_immediates(isa_code)
    lines = isa_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ----------------------------------------------------------- build return
    input_tensors = {"X": token_embeds, "POS": pos_weight}
    data_order = ["X", "POS"]
    for i in range(n_layers):
        input_tensors[f"W_q_{i}"] = all_weights[i]["W_q"]
        input_tensors[f"W_o_{i}"] = all_weights[i]["W_o"]
        if native_mode:
            for kv_h in range(num_kv_heads):
                input_tensors[f"K_{i}_h{kv_h}"] = K_head_mats[i][kv_h]
                input_tensors[f"V_{i}_h{kv_h}"] = V_head_mats[i][kv_h]
        else:
            input_tensors[f"K_{i}"] = K_mats[i]
            input_tensors[f"V_{i}"] = V_mats[i]
        input_tensors[f"W_gate_{i}"] = all_weights[i]["W_gate"]
        input_tensors[f"W_up_{i}"] = all_weights[i]["W_up"]
        input_tensors[f"W_down_{i}"] = all_weights[i]["W_down"]

        kv_keys = []
        if native_mode:
            for kv_h in range(num_kv_heads):
                kv_keys.extend([f"K_{i}_h{kv_h}", f"V_{i}_h{kv_h}"])
        else:
            kv_keys = [f"K_{i}", f"V_{i}"]
        data_order.extend([
            f"W_q_{i}", f"W_o_{i}",
            *kv_keys,
            f"W_gate_{i}", f"W_up_{i}", f"W_down_{i}",
        ])

    # FPRAM layout (same as single-layer decoder):
    #   slot 0 = 0.0        (reserved)
    #   slot 1 = attn_scale  (flash_attention)
    #   slot 2 = -inf        (flash_attention softmax mask)
    #   slot 3 = eps         (rms_norm, offset=3)
    #   slot 4 = 1/hidden    (rms_norm, offset=4)
    #   slot 5 = 1.0         (FFN SiLU)
    #   slots 6-9 = 0.0      (padding)
    fp_preload = [0.0, scale, float("-inf"), eps, 1.0 / hidden, 1.0] + [0.0] * 4

    # Result is at current's VRAM location (last O from flash_attention chain)
    o_vram_addr = prog._compiler.get_vram_addr(current.name)

    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (seq_len * hidden) // mlen,
        "num_batches": seq_len,
        "elements_per_batch": hidden,
        "row_dim": mlen,
        "use_stride_mode": hidden > mlen,
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
        "mlen": mlen,
        "blen": blen,
        "isa_lines": len(lines),
    }

    print(f"\nCompilation complete: {info['isa_lines']} ISA lines, "
          f"{n_layers} layers, output at VRAM row {o_vram_addr // mlen}")

    return {
        "isa": isa_code,
        "golden_output": golden_out,
        "input_tensors": input_tensors,
        "data_order": data_order,
        "fp_preload": fp_preload,
        "comparison_params": comparison_params,
        "info": info,
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
        run_and_assert,
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

    # Run emulator and compare
    run_and_assert(build_dir, asm_name, mlen=mlen, blen=blen)

    comp_results, _ = compare_emulator_output(build_dir)
    return {**result["info"], **comp_results}
