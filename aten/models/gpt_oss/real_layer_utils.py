"""Reusable GPT-OSS real-layer loading helpers.

These helpers are shared by the correctness tests that need a real GPT-OSS
layer without making those tests depend on the standalone checkpoint probe
script.  They intentionally use cached HuggingFace files only; callers decide
which tensors to load and how to compare them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


REPO_ID = "openai/gpt-oss-20b"
SHARD0 = "model-00000-of-00002.safetensors"
SHARD2 = "model-00002-of-00002.safetensors"
SHARDS = (SHARD0, SHARD2)
_TORCHVISION_NMS_LIB = None


def ensure_torchvision_nms_schema() -> None:
    """Define a minimal torchvision::nms schema when torchvision C ops are absent."""
    global _TORCHVISION_NMS_LIB
    if _TORCHVISION_NMS_LIB is not None:
        return
    try:
        if importlib.util.find_spec("torchvision._C") is not None:
            return
    except Exception:
        pass
    try:
        _TORCHVISION_NMS_LIB = torch.library.Library("torchvision", "DEF")
        _TORCHVISION_NMS_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass


def cached_file(filename: str) -> Path:
    return Path(hf_hub_download(REPO_ID, filename, local_files_only=True))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_revision(path: Path) -> str | None:
    parts = path.parts
    try:
        idx = parts.index("snapshots")
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    return parts[idx + 1]


def load_json(filename: str) -> dict:
    return json.loads(cached_file(filename).read_text())


@lru_cache(maxsize=1)
def weight_map() -> dict[str, str]:
    index_path = cached_file("model.safetensors.index.json")
    return json.loads(index_path.read_text())["weight_map"]


def load_sharded_tensor(tensor_name: str) -> torch.Tensor:
    filename = weight_map().get(tensor_name)
    candidates = (filename,) if filename is not None else SHARDS
    for candidate in candidates:
        if candidate is None:
            continue
        shard_path = cached_file(candidate)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if tensor_name in f.keys():
                return f.get_tensor(tensor_name)
    raise KeyError(f"tensor {tensor_name!r} not found in cached GPT-OSS shards")


def load_token_ids(prompt: str, max_tokens: int) -> torch.Tensor:
    ensure_torchvision_nms_schema()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, local_files_only=True)
    token_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if token_ids.numel() == 0:
        raise ValueError("Prompt tokenized to zero tokens")
    return token_ids[:max_tokens].to(torch.long)


def slice_rows(shard_path: Path, tensor_name: str, rows: torch.Tensor) -> torch.Tensor:
    pieces = []
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(tensor_name)
        for row in rows.tolist():
            pieces.append(tensor_slice[int(row) : int(row) + 1])
    return torch.cat(pieces, dim=0)


def rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(input_dtype)


def load_layer_tensors(layer_index: int) -> dict[str, torch.Tensor]:
    """Load one GPT-OSS layer's MoE tensors in raw HuggingFace orientation.

    ``router_weight`` is returned as HF ``[num_experts, hidden]`` — the layout
    ``build_hf_mlp`` copies straight into ``mlp.router.weight``. Note that
    ``gpt_oss_moe_golden_a`` (moe_reference.py) instead expects the transposed
    PLENA layout ``[hidden, experts]`` for its ``x @ router_weight``; a caller
    feeding this dict into Golden A must pass ``router_weight.t()``.
    """
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    prefix = f"model.layers.{layer_index}"
    gate_blocks = load_sharded_tensor(f"{prefix}.mlp.experts.gate_up_proj_blocks")
    gate_scales = load_sharded_tensor(f"{prefix}.mlp.experts.gate_up_proj_scales")
    down_blocks = load_sharded_tensor(f"{prefix}.mlp.experts.down_proj_blocks")
    down_scales = load_sharded_tensor(f"{prefix}.mlp.experts.down_proj_scales")
    tensors = {
        "router_weight": load_sharded_tensor(f"{prefix}.mlp.router.weight").to(torch.bfloat16),
        "router_bias": load_sharded_tensor(f"{prefix}.mlp.router.bias").to(torch.bfloat16),
        "gate_up_bias": load_sharded_tensor(f"{prefix}.mlp.experts.gate_up_proj_bias").to(torch.bfloat16),
        "down_bias": load_sharded_tensor(f"{prefix}.mlp.experts.down_proj_bias").to(torch.bfloat16),
        "input_layernorm_weight": load_sharded_tensor(f"{prefix}.input_layernorm.weight").to(torch.bfloat16),
    }

    tensors["gate_up_weight"] = convert_moe_packed_tensors(gate_blocks, gate_scales).to(torch.bfloat16)
    tensors["down_weight"] = convert_moe_packed_tensors(down_blocks, down_scales).to(torch.bfloat16)
    return tensors


def load_layer0_tensors() -> dict[str, torch.Tensor]:
    return load_layer_tensors(0)


def build_hf_mlp(config, tensors: dict[str, torch.Tensor]):
    ensure_torchvision_nms_schema()
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

    mlp = GptOssMLP(config).eval().to(torch.bfloat16)
    with torch.no_grad():
        mlp.router.weight.copy_(tensors["router_weight"])
        mlp.router.bias.copy_(tensors["router_bias"])
        mlp.experts.gate_up_proj.copy_(tensors["gate_up_weight"])
        mlp.experts.gate_up_proj_bias.copy_(tensors["gate_up_bias"])
        mlp.experts.down_proj.copy_(tensors["down_weight"])
        mlp.experts.down_proj_bias.copy_(tensors["down_bias"])
    return mlp
