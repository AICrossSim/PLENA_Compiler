"""Canonical native-decoder schedule profiles.

Public lowering APIs retain explicit compatibility arguments for focused A/B
tests. Production callers should select one of the profiles in this module so
the compiler, CostEmitter, and DSE cannot drift onto different schedule sets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


CURRENT_DSE_PROFILE = "current-dse-v1"
RTL_VALIDATION_PROFILE = "rtl-validation-v1"
RTL_V6_CANDIDATE_PROFILE = "rtl-v6-candidate-v1"


@dataclass(frozen=True)
class CompilerScheduleOptions:
    packed_attention_schedule: str
    softmax_state_schedule: str
    packed_qk_schedule: str
    vector_scalar_schedule: str
    softmax_vector_schedule: str
    pv_accumulation_schedule: str
    softmax_row_lanes: int
    softmax_row_issue_schedule: str
    selector_schedule: str
    reduction_output_mode: str
    gqa_pipeline_schedule: str
    address_generation_mode: str
    ffn_address_schedule: str
    ffn_projection_schedule: str
    moe_lowering_schedule: str

    def as_kwargs(self) -> dict[str, Any]:
        return asdict(self)


_PROFILES = {
    CURRENT_DSE_PROFILE: CompilerScheduleOptions(
        packed_attention_schedule="direct-first-block-v1",
        softmax_state_schedule="streamed-v2",
        packed_qk_schedule="broadcast-k-major-v1",
        vector_scalar_schedule="rtl-v5",
        softmax_vector_schedule="single-row-v1",
        pv_accumulation_schedule="shift-add-v1",
        softmax_row_lanes=1,
        softmax_row_issue_schedule="group-serial-v1",
        selector_schedule="hoisted-v1",
        reduction_output_mode="overwrite-v1",
        gqa_pipeline_schedule="row-interleaved-v1",
        address_generation_mode="loop-agu-v1",
        ffn_address_schedule="live-stride-v1",
        ffn_projection_schedule="affine-loop-v2",
        moe_lowering_schedule="compact-route-v2",
    ),
    RTL_VALIDATION_PROFILE: CompilerScheduleOptions(
        packed_attention_schedule="direct-first-block-v1",
        softmax_state_schedule="streamed-v2",
        packed_qk_schedule="head-major-v1",
        vector_scalar_schedule="rtl-v5",
        softmax_vector_schedule="single-row-v1",
        pv_accumulation_schedule="shift-add-v1",
        softmax_row_lanes=1,
        softmax_row_issue_schedule="group-serial-v1",
        selector_schedule="hoisted-v1",
        reduction_output_mode="overwrite-v1",
        gqa_pipeline_schedule="row-interleaved-v1",
        address_generation_mode="loop-agu-v1",
        ffn_address_schedule="live-stride-v1",
        ffn_projection_schedule="affine-loop-v2",
        moe_lowering_schedule="compact-route-v2",
    ),
    RTL_V6_CANDIDATE_PROFILE: CompilerScheduleOptions(
        packed_attention_schedule="direct-first-block-v1",
        softmax_state_schedule="row-bank-simd-v3",
        packed_qk_schedule="broadcast-k-major-v1",
        vector_scalar_schedule="rtl-v6",
        softmax_vector_schedule="multi-row-v1",
        pv_accumulation_schedule="direct-packed-rmw-v1",
        softmax_row_lanes=4,
        softmax_row_issue_schedule="wavefront-v1",
        selector_schedule="hoisted-v1",
        reduction_output_mode="overwrite-v1",
        gqa_pipeline_schedule="row-interleaved-v1",
        address_generation_mode="loop-agu-v1",
        ffn_address_schedule="live-stride-v1",
        ffn_projection_schedule="affine-loop-v2",
        moe_lowering_schedule="compact-route-v2",
    ),
}


def compiler_schedule_profile(name: str) -> CompilerScheduleOptions:
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown compiler schedule profile {name!r}; "
            f"expected one of {sorted(_PROFILES)}"
        ) from exc


def compiler_schedule_profile_kwargs(name: str) -> dict[str, Any]:
    return compiler_schedule_profile(name).as_kwargs()


__all__ = [
    "CURRENT_DSE_PROFILE",
    "RTL_VALIDATION_PROFILE",
    "RTL_V6_CANDIDATE_PROFILE",
    "CompilerScheduleOptions",
    "compiler_schedule_profile",
    "compiler_schedule_profile_kwargs",
]
