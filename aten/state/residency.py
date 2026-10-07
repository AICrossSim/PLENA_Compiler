"""Convert Simulator DSE capacity results into explicit Compiler residency maps."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .contract import KdaPayload, Mamba2Payload, PrecisionCode, StateDescriptor

if TYPE_CHECKING:
    from aten.mamba.scheduler import MambaScheduleConfig


MIB = 1024 * 1024


class ResidencyTarget(StrEnum):
    CAPACITY_KNEE = "capacity-knee"
    FULL_RESIDENT = "full-resident"


@dataclass(frozen=True)
class StateResidencyPlan:
    model_key: str
    target: ResidencyTarget
    state_precision: PrecisionCode
    capacity_bytes: int
    entry_bytes: int
    resident_keys: tuple[tuple[int, int], ...]
    streaming_keys: tuple[tuple[int, int], ...]
    expected_hit_rate: float
    source: str

    @property
    def entry_count(self) -> int:
        return len(self.resident_keys)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_key": self.model_key,
            "target": self.target.value,
            "state_precision": self.state_precision.name.lower(),
            "capacity_bytes": self.capacity_bytes,
            "capacity_mib": self.capacity_bytes / MIB,
            "entry_bytes": self.entry_bytes,
            "resident_keys": [
                {"request_id": request_id, "layer_id": layer_id}
                for request_id, layer_id in self.resident_keys
            ],
            "streaming_keys": [
                {"request_id": request_id, "layer_id": layer_id}
                for request_id, layer_id in self.streaming_keys
            ],
            "expected_hit_rate": self.expected_hit_rate,
            "source": self.source,
        }


def mamba_resident_bytes(precision: PrecisionCode) -> int:
    descriptor = StateDescriptor(
        payload=Mamba2Payload(),
        num_heads=64,
        state_precision=precision,
    )
    return descriptor.resident_bytes


def kda_resident_bytes(
    precision: PrecisionCode = PrecisionCode.FP32,
    conv_precision: PrecisionCode = PrecisionCode.BF16,
) -> int:
    descriptor = StateDescriptor(
        payload=KdaPayload(),
        num_heads=96,
        state_precision=precision,
        conv_state_precision=conv_precision,
    )
    return descriptor.resident_bytes


def build_capacity_residency_plan(
    *,
    model_key: str,
    capacity_bytes: int,
    entry_bytes: int,
    state_precision: PrecisionCode,
    layer_ids: tuple[int, ...],
    batch_size: int,
    target: ResidencyTarget = ResidencyTarget.CAPACITY_KNEE,
    source: str = "explicit_capacity",
) -> StateResidencyPlan:
    """Turn a byte capacity into the exact request/layer residency map.

    This path is shared by Mamba and KDA. It intentionally takes the complete
    per-entry byte count, including convolution state and MX8 scales, rather
    than deriving a layer count from recurrent-state bytes alone.
    """

    if capacity_bytes < 0 or entry_bytes <= 0 or batch_size <= 0:
        raise ValueError("residency capacity, entry size, and batch size are invalid")
    all_keys = tuple(
        (request_id, layer_id)
        for layer_id in layer_ids
        for request_id in range(batch_size)
    )
    entry_count = min(len(all_keys), capacity_bytes // entry_bytes)
    resident_keys = all_keys[:entry_count]
    return StateResidencyPlan(
        model_key=model_key,
        target=target,
        state_precision=state_precision,
        capacity_bytes=capacity_bytes,
        entry_bytes=entry_bytes,
        resident_keys=resident_keys,
        streaming_keys=all_keys[entry_count:],
        expected_hit_rate=entry_count / len(all_keys) if all_keys else 0.0,
        source=source,
    )


def load_dse_residency_plan(
    report_path: Path,
    *,
    state_precision: PrecisionCode,
    layer_ids: tuple[int, ...],
    batch_size: int,
    target: ResidencyTarget = ResidencyTarget.CAPACITY_KNEE,
) -> StateResidencyPlan:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    document = json.loads(report_path.read_text())
    if document.get("status") != "uncalibrated_sensitivity_not_rtl_performance":
        raise ValueError("unsupported or missing Nemotron sensitivity report status")
    records = document.get("decode_system", {}).get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("DSE report has no decode_system records")

    precision_name = state_precision.name.lower()
    minimum_hit_rate = 0.95 if target == ResidencyTarget.CAPACITY_KNEE else 1.0
    candidates = [
        record
        for record in records
        if _design(record).get("name", "").startswith(f"decode-{precision_name}-")
        and _design(record).get("state_cache_policy") == "pinned"
        and float(_state_cache(record).get("hit_rate", 0.0)) >= minimum_hit_rate
    ]
    if not candidates:
        raise ValueError(
            f"DSE report has no pinned {precision_name} design for target {target.value}"
        )
    selected = min(
        candidates,
        key=lambda record: (
            int(_design(record)["state_cache_bytes"]),
            int(_design(record).get("projection_buffer_banks", 1)),
            int(_design(record).get("state_macs_per_cycle", 1)),
        ),
    )
    capacity_bytes = int(_design(selected)["state_cache_bytes"])
    entry_bytes = mamba_resident_bytes(state_precision)
    all_keys = tuple(
        (request_id, layer_id)
        for layer_id in layer_ids
        for request_id in range(batch_size)
    )
    entry_count = min(len(all_keys), capacity_bytes // entry_bytes)
    if entry_count == 0:
        raise ValueError(
            f"selected DSE capacity {capacity_bytes} cannot hold one {entry_bytes}-byte state"
        )
    resident_keys = all_keys[:entry_count]
    return StateResidencyPlan(
        model_key=str(document.get("model_key", "unknown")),
        target=target,
        state_precision=state_precision,
        capacity_bytes=capacity_bytes,
        entry_bytes=entry_bytes,
        resident_keys=resident_keys,
        streaming_keys=all_keys[entry_count:],
        expected_hit_rate=len(resident_keys) / len(all_keys),
        source=str(report_path.resolve()),
    )


def apply_residency_plan(
    config: MambaScheduleConfig,
    plan: StateResidencyPlan,
) -> MambaScheduleConfig:
    from aten.mamba.scheduler import CachePolicy

    if config.state_precision != plan.state_precision:
        raise ValueError("DSE plan precision does not match scheduler state precision")
    plan_requests = {request_id for request_id, _ in plan.resident_keys + plan.streaming_keys}
    if plan_requests != set(range(config.batch_size)):
        raise ValueError("DSE plan batch size does not match scheduler batch size")
    return replace(
        config,
        state_cache_entries=plan.entry_count,
        cache_policy=CachePolicy.PINNED,
        resident_state_keys=plan.resident_keys,
        residency_capacity_bytes=plan.capacity_bytes,
        residency_source=plan.source,
        residency_target=plan.target.value,
    )


def _design(record: dict[str, Any]) -> dict[str, Any]:
    design = record.get("design")
    if not isinstance(design, dict):
        raise ValueError("DSE record has no design object")
    return design


def _state_cache(record: dict[str, Any]) -> dict[str, Any]:
    cache = record.get("state_cache")
    if not isinstance(cache, dict):
        raise ValueError("DSE record has no state_cache object")
    return cache
