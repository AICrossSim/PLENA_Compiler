from __future__ import annotations

import json

from aten.kda.scheduler import KIMI_K3_KDA_LAYERS, KdaScheduleConfig, KimiK3KdaScheduler
from aten.mamba.scheduler import (
    MambaScheduleConfig,
    NEMOTRON3_MAMBA_LAYERS,
    Nemotron3MambaScheduler,
    SchedulePhase,
)
from aten.state import (
    PrecisionCode,
    ResidencyTarget,
    apply_residency_plan,
    build_capacity_residency_plan,
    kda_resident_bytes,
    load_dse_residency_plan,
    mamba_resident_bytes,
)


MIB = 1024 * 1024


def _record(precision: str, cache_mib: int, hit_rate: float) -> dict:
    return {
        "design": {
            "name": f"decode-{precision}-b8-n4-cache{cache_mib}-pinned",
            "state_cache_bytes": cache_mib * MIB,
            "state_cache_policy": "pinned",
            "projection_buffer_banks": 8,
            "state_macs_per_cycle": 128,
        },
        "state_cache": {"hit_rate": hit_rate},
    }


def _report(tmp_path):
    path = tmp_path / "dse.json"
    path.write_text(
        json.dumps(
            {
                "status": "uncalibrated_sensitivity_not_rtl_performance",
                "model_key": "nemotron3_nano_30b_a3b",
                "decode_system": {
                    "records": [
                        _record("bf16", 16, 0.652),
                        _record("bf16", 24, 0.957),
                        _record("bf16", 32, 1.0),
                        _record("fp32", 48, 0.957),
                        _record("fp32", 64, 1.0),
                    ]
                },
            }
        )
    )
    return path


def test_capacity_knee_becomes_an_explicit_22_layer_map(tmp_path) -> None:
    plan = load_dse_residency_plan(
        _report(tmp_path),
        state_precision=PrecisionCode.BF16,
        layer_ids=NEMOTRON3_MAMBA_LAYERS,
        batch_size=1,
    )
    assert plan.capacity_bytes == 24 * MIB
    assert plan.entry_bytes == mamba_resident_bytes(PrecisionCode.BF16) == 1_097_728
    assert plan.entry_count == 22
    assert plan.resident_keys == tuple((0, layer) for layer in NEMOTRON3_MAMBA_LAYERS[:22])
    assert plan.streaming_keys == ((0, NEMOTRON3_MAMBA_LAYERS[-1]),)


def test_full_resident_target_uses_the_32_mib_dse_point(tmp_path) -> None:
    plan = load_dse_residency_plan(
        _report(tmp_path),
        state_precision=PrecisionCode.BF16,
        layer_ids=NEMOTRON3_MAMBA_LAYERS,
        batch_size=1,
        target=ResidencyTarget.FULL_RESIDENT,
    )
    assert plan.capacity_bytes == 32 * MIB
    assert plan.entry_count == 23
    assert not plan.streaming_keys


def test_scheduler_executes_the_dse_residency_map(tmp_path) -> None:
    plan = load_dse_residency_plan(
        _report(tmp_path),
        state_precision=PrecisionCode.BF16,
        layer_ids=NEMOTRON3_MAMBA_LAYERS,
        batch_size=1,
    )
    config = apply_residency_plan(
        MambaScheduleConfig(
            phase=SchedulePhase.DECODE,
            decode_tokens=2,
            state_precision=PrecisionCode.BF16,
        ),
        plan,
    )
    trace = Nemotron3MambaScheduler(config).build()
    assert trace.config.resident_state_keys == plan.resident_keys
    assert trace.count("PRELOAD") == 22
    assert trace.count("COMMIT") == 22
    assert trace.cache_hits == 22
    assert trace.cache_misses == 24


def test_kda_capacity_map_counts_fp32_state_and_bf16_conv_bytes() -> None:
    entry_bytes = kda_resident_bytes(PrecisionCode.FP32, PrecisionCode.BF16)
    assert entry_bytes == 6_586_368
    plan = build_capacity_residency_plan(
        model_key="kimi_k3_kda",
        capacity_bytes=32 * MIB,
        entry_bytes=entry_bytes,
        state_precision=PrecisionCode.FP32,
        layer_ids=KIMI_K3_KDA_LAYERS,
        batch_size=1,
        source="test:32MiB",
    )
    assert plan.entry_count == 5
    assert plan.resident_keys == tuple((0, layer) for layer in KIMI_K3_KDA_LAYERS[:5])
    assert len(plan.streaming_keys) == 64

    config = apply_residency_plan(
        KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1),
        plan,
    )
    trace = KimiK3KdaScheduler(config).build()
    assert trace.count("PRELOAD") == 5
    assert trace.count("COMMIT") == 5
    assert trace.config.residency_capacity_bytes == 32 * MIB
