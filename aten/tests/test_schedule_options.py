from compiler.aten.plena.schedule_options import (
    CURRENT_DSE_PROFILE,
    RTL_VALIDATION_PROFILE,
    compiler_schedule_profile,
    compiler_schedule_profile_kwargs,
)


def test_current_dse_profile_selects_only_canonical_schedules() -> None:
    profile = compiler_schedule_profile(CURRENT_DSE_PROFILE)

    assert profile.vector_scalar_schedule == "rtl-v4"
    assert profile.softmax_state_schedule == "streamed-v2"
    assert profile.packed_qk_schedule == "broadcast-k-major-v1"
    assert profile.address_generation_mode == "loop-agu-v1"
    assert profile.ffn_projection_schedule == "affine-loop-v2"
    assert profile.moe_lowering_schedule == "compact-route-v2"
    assert "loop-agu-v2" not in profile.as_kwargs().values()


def test_rtl_validation_profile_uses_head_major_fallback() -> None:
    profile = compiler_schedule_profile(RTL_VALIDATION_PROFILE)

    assert profile.packed_qk_schedule == "head-major-v1"
    assert profile.vector_scalar_schedule == "rtl-v4"
    assert compiler_schedule_profile_kwargs(RTL_VALIDATION_PROFILE) == (
        profile.as_kwargs()
    )
