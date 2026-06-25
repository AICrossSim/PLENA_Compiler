"""Tests for GPT-OSS real-layer comparison thresholds."""

from __future__ import annotations

import pytest
import torch

from aten.gpt_oss_moe import assert_compare_within, assert_gap_in_band, compare_stats


def _reference_tensor() -> torch.Tensor:
    return torch.linspace(-3.0, 3.0, 37, dtype=torch.float32).reshape(1, 37)


def test_hf_golden_a_threshold_accepts_tiny_math_noise():
    ref = _reference_tensor()
    actual = ref + ref.std(unbiased=False) * 1e-4

    stats = assert_compare_within(
        actual,
        ref,
        name="HF_vs_GoldenA",
        max_rel_rms=1e-2,
        rtol=1e-2,
    )

    assert stats.rel_rms < 1e-3
    assert stats.allclose


def test_emulator_golden_b_threshold_accepts_two_percent_bound():
    ref = _reference_tensor()
    actual = ref * 1.004

    stats = assert_compare_within(
        actual,
        ref,
        name="emu_vs_GoldenB",
        max_rel_rms=0.02,
        rtol=0.02,
    )

    assert stats.rel_rms < 0.005
    assert stats.allclose


def test_compare_threshold_rejects_large_lowering_error():
    ref = _reference_tensor()
    actual = ref * 1.10

    with pytest.raises(AssertionError, match="emu_vs_GoldenB failed"):
        assert_compare_within(
            actual,
            ref,
            name="emu_vs_GoldenB",
            max_rel_rms=0.02,
            rtol=0.02,
        )


def test_a_b_gap_band_is_recordable_and_checked():
    golden_a = _reference_tensor()
    golden_b = golden_a * 0.98

    stats = assert_gap_in_band(
        golden_b,
        golden_a,
        name="GoldenA_vs_GoldenB",
        min_rel_rms=0.005,
        max_rel_rms=0.15,
    )

    assert 0.005 <= stats.rel_rms <= 0.15


def test_compare_stats_uses_one_percent_reference_std_as_atol():
    ref = _reference_tensor()
    stats = compare_stats(ref, ref, rtol=0.02)

    assert stats.atol == pytest.approx(float(ref.std(unbiased=False).item()) * 0.01)
    assert stats.rel_rms == 0.0
    assert stats.pass_rate == 1.0
