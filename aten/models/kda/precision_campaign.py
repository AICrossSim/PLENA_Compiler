"""Deterministic long-sequence KDA recurrent-state precision campaign.

The KDA heads are mathematically independent.  The campaign therefore runs
three full 128x128 heads in parallel, rather than allocating all 96 identical
head pipelines, while retaining Kimi K3's real key/value dimensions and decay
formula.  Update and reduction arithmetic stays FP32; only the state stored at
the requested checkpoint interval is quantized.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from compiler.aten.models.kda.state_precision import (
    StateStorage,
    quantize_state,
    storage_bytes,
)


@dataclass(frozen=True)
class CampaignConfig:
    tokens: tuple[int, ...] = (2048, 8192)
    seeds: tuple[int, ...] = (17, 42, 2026)
    key_dim: int = 128
    value_dim: int = 128
    full_model_heads: int = 96
    gate_lower_bound: float = -5.0
    checkpoint_intervals: tuple[int, ...] = (1, 128)


def _inputs(config: CampaignConfig, tokens: int) -> tuple[Tensor, ...]:
    """Generate one deterministic, long-memory trace per seed."""
    streams = []
    for seed in config.seeds:
        generator = torch.Generator().manual_seed(seed)
        q = torch.randn(tokens, config.key_dim, generator=generator)
        k = torch.randn(tokens, config.key_dim, generator=generator)
        v = 0.25 * torch.randn(tokens, config.value_dim, generator=generator)
        # A negative gate produces a slow decay, which is the precision-stress
        # regime. This is synthetic input, not a claim about language quality.
        gate = -8.0 + 0.5 * torch.randn(tokens, config.key_dim, generator=generator)
        beta = torch.randn(tokens, generator=generator)
        streams.append((q, k, v, gate, beta))
    return tuple(torch.stack(values, dim=0) for values in zip(*streams))


def _run(
    inputs: tuple[Tensor, ...],
    config: CampaignConfig,
    storage: StateStorage,
    checkpoint_interval: int,
) -> tuple[Tensor, Tensor]:
    q, k, v, gate, beta_logit = inputs
    batch, tokens, _ = q.shape
    state = torch.zeros(batch, config.value_dim, config.key_dim, dtype=torch.float32)
    outputs = torch.empty(batch, tokens, config.value_dim, dtype=torch.float32)
    scale = 1.0 / math.sqrt(config.key_dim)

    # This is activate_log_decay with a_log=0 and dt_bias=0. The campaign keeps
    # those constants explicit so the exact long-memory stress is reproducible.
    for token in range(tokens):
        q_t = q[:, token].float()
        k_t = k[:, token].float()
        q_n = q_t * torch.rsqrt(q_t.square().sum(-1, keepdim=True) + 1.0e-6)
        k_n = k_t * torch.rsqrt(k_t.square().sum(-1, keepdim=True) + 1.0e-6)
        log_decay = config.gate_lower_bound * torch.sigmoid(gate[:, token].float())
        decayed = state * torch.exp(log_decay)[:, None, :]
        prediction = torch.einsum("bvk,bk->bv", decayed, k_n)
        error = torch.sigmoid(beta_logit[:, token].float())[:, None] * (
            v[:, token].float() - prediction
        )
        state = decayed + error[:, :, None] * k_n[:, None, :]
        outputs[:, token] = scale * torch.einsum("bvk,bk->bv", state, q_n)
        if (token + 1) % checkpoint_interval == 0 or token + 1 == tokens:
            state = quantize_state(state, storage)
    return outputs, state


def _metrics(actual: Tensor, expected: Tensor) -> dict[str, float]:
    diff = (actual.float() - expected.float()).reshape(actual.shape[0], -1)
    reference = expected.float().reshape(expected.shape[0], -1)
    relative_l2 = torch.linalg.vector_norm(diff, dim=1) / torch.linalg.vector_norm(
        reference, dim=1
    ).clamp_min(1.0e-12)
    cosine = torch.nn.functional.cosine_similarity(actual.reshape(actual.shape[0], -1),
                                                    expected.reshape(expected.shape[0], -1))
    return {
        "relative_l2_mean": relative_l2.mean().item(),
        "relative_l2_max": relative_l2.max().item(),
        "max_abs_max": diff.abs().amax(dim=1).max().item(),
        "cosine_min": cosine.min().item(),
    }


def run_campaign(config: CampaignConfig = CampaignConfig()) -> dict:
    records = []
    full_state_elements = config.full_model_heads * config.key_dim * config.value_dim
    for tokens in config.tokens:
        inputs = _inputs(config, tokens)
        reference_output, reference_state = _run(
            inputs, config, StateStorage.FP32, checkpoint_interval=tokens
        )
        for checkpoint_interval in config.checkpoint_intervals:
            schedule = "token" if checkpoint_interval == 1 else f"chunk{checkpoint_interval}"
            for storage in StateStorage:
                output, state = _run(inputs, config, storage, checkpoint_interval)
                records.append(
                    {
                        "tokens": tokens,
                        "schedule": schedule,
                        "checkpoint_interval": checkpoint_interval,
                        "storage": storage.value,
                        "output": _metrics(output, reference_output),
                        "state": _metrics(state, reference_state),
                        "full_kimi_layer_bytes": storage_bytes(
                            full_state_elements, storage
                        ),
                        "nan_count": int(torch.isnan(output).sum() + torch.isnan(state).sum()),
                        "inf_count": int(torch.isinf(output).sum() + torch.isinf(state).sum()),
                    }
                )
    return {
        "schema_version": 1,
        "scope": {
            "model": "Kimi K3 KDA recurrent core",
            "real_dimensions": {
                "key_dim": config.key_dim,
                "value_dim": config.value_dim,
                "full_model_heads": config.full_model_heads,
            },
            "represented_heads": len(config.seeds),
            "seeds": list(config.seeds),
            "weights": "synthetic deterministic recurrent inputs; no checkpoint weights",
            "arithmetic": "FP32 update and reduction; storage-only quantization",
            "decay": "Kimi formula with a_log=0, dt_bias=0, long-memory gate near -8",
        },
        "config": asdict(config),
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 8192])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    config = CampaignConfig(tokens=tuple(args.tokens))
    rendered = json.dumps(run_campaign(config), indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
