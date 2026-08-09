"""Guards on the MoE region terminator.

These cover a failure that produces a plausible-looking wrong answer rather than
an error, which is why it is pinned here rather than left to a numerical test.

The terminator exists because markers are sticky and a program does not end
where its MoE region does. Without one, the last MoE marker runs to the end of
the file and every instruction after it -- an lm_head, the next sublayer -- is
billed to a MoE stage, and that cost lands in the shared-vs-routed ratio. The
size of the effect is whatever the epilogue happens to be, so it is measured in
the pull request rather than pinned here, where it would rot.
"""

from __future__ import annotations

import pytest

from compiler.aten.plena.program_routed_moe import (
    MOE_END_STAGE,
    MOE_STAGES,
    moe_end_marker,
    moe_stage_marker,
)


def test_the_terminator_is_part_of_the_declared_vocabulary() -> None:
    """It has to be, or the emulator's both-directions guard rejects it."""
    assert MOE_END_STAGE in MOE_STAGES


def test_the_terminator_is_not_a_stage_a_caller_could_mistake_for_work() -> None:
    """No other declared stage may be confusable with the terminator.

    Asserting `MOE_END_STAGE == "non_moe"` and then that it does not start with
    a work prefix is two ways of restating the literal on the line above. What
    is worth pinning is the relationship to the rest of the vocabulary: the
    terminator must not be a prefix of, or prefixed by, any stage that names
    real work, or a substring match somewhere would pick the wrong one.
    """
    others = MOE_STAGES - {MOE_END_STAGE}
    assert others, "MOE_STAGES holds nothing but the terminator"
    for stage in others:
        assert not stage.startswith(MOE_END_STAGE), f"{stage!r} is prefixed by the terminator"
        assert not MOE_END_STAGE.startswith(stage), f"the terminator is prefixed by {stage!r}"


def test_moe_end_marker_emits_the_terminator() -> None:
    assert moe_end_marker() == f"@stage={MOE_END_STAGE}"
    assert moe_end_marker("after the combine") == f"@stage={MOE_END_STAGE} after the combine"


def test_a_typo_of_the_terminator_fails_like_any_other() -> None:
    """It is a marker, not a special case.

    `moe_stage_marker(MOE_END_STAGE) == moe_end_marker()` was the previous
    assertion here; it is a tautology, since `moe_end_marker` is defined as that
    call. What is checkable is that the terminator is not exempt from the
    vocabulary check -- a near miss must be rejected, not silently accepted as
    "close enough to the end marker".
    """
    for typo in ("non-moe", "nonmoe", "non_moe_", "NON_MOE"):
        with pytest.raises(ValueError, match="unknown MoE stage"):
            moe_stage_marker(typo)
