from __future__ import annotations

from pathlib import Path

import pytest

from compiler.aten.plena.attention_pipeline_plan import (
    GQATimingProfile,
    RowPipelineOp,
    interleave_row_chains,
)
from compiler.aten.plena.compiler import PlenaCompiler


def test_rtl_v3_or_later_timing_profile_is_required_and_identified() -> None:
    profile = GQATimingProfile.load()

    assert profile.rob_depth == 8
    assert profile.fp_register_count == 16
    assert profile.vector_ii == 1
    assert len(profile.sha256) == 64
    assert profile.reduction_latency(kind="sum", segment_width=16) > 1

    with pytest.raises(
        FileNotFoundError, match="requires an RTL-v3-or-later timing artifact"
    ):
        GQATimingProfile.load(Path("/tmp/does-not-exist-rtl-v3-timing.json"))


def test_row_list_scheduler_preserves_each_row_order() -> None:
    chains = (
        (
            RowPipelineOp("r0.load", "scalar_load", 1),
            RowPipelineOp("r0.reci", "scalar_reciprocal", 8),
            RowPipelineOp("r0.mul", "vector", 11),
        ),
        (
            RowPipelineOp("r1.load", "scalar_load", 1),
            RowPipelineOp("r1.reci", "scalar_reciprocal", 8),
            RowPipelineOp("r1.mul", "vector", 11),
        ),
    )

    scheduled = interleave_row_chains(chains)

    assert scheduled.index("r0.load") < scheduled.index("r0.reci") < scheduled.index("r0.mul")
    assert scheduled.index("r1.load") < scheduled.index("r1.reci") < scheduled.index("r1.mul")
    assert scheduled[:2] == ("r0.load", "r1.load")


def test_pipeline_schedule_requires_rtl_v3() -> None:
    with pytest.raises(ValueError, match="requires vector_scalar_schedule='rtl-v3'"):
        PlenaCompiler(
            vector_scalar_schedule="rtl-v2",
            gqa_pipeline_schedule="row-interleaved-v1",
        )

    legacy = PlenaCompiler(vector_scalar_schedule="rtl-v2")
    assert legacy.gqa_pipeline_schedule == "row-serial"

    rtl_v4 = PlenaCompiler(
        vector_scalar_schedule="rtl-v4",
        gqa_pipeline_schedule="row-interleaved-v1",
    )
    assert rtl_v4.gqa_pipeline_schedule == "row-interleaved-v1"

    rtl_v5 = PlenaCompiler(
        vector_scalar_schedule="rtl-v5",
        gqa_pipeline_schedule="row-interleaved-v1",
    )
    assert rtl_v5.gqa_pipeline_schedule == "row-interleaved-v1"
