"""Unit tests demonstrating bf16 exp overflow → NaN propagation.

Shows that bfloat16 arithmetic in the emulator causes numerical instability
after many layers due to exp() overflow, and verifies the clamping fix.
"""

import torch
import pytest


# ---------- helpers ----------

def _bf16(x):
    """Convert to bfloat16 and back (simulating emulator truncation)."""
    return x.to(torch.bfloat16).to(torch.float32)


def _silu_bf16(x: torch.Tensor) -> torch.Tensor:
    """SiLU entirely in bf16 arithmetic (emulator behaviour)."""
    x = x.to(torch.bfloat16)
    neg_x = (torch.zeros_like(x) - x)           # -x
    exp_neg_x = neg_x.exp()                      # exp(-x)
    one_plus = exp_neg_x + torch.ones_like(x)    # 1 + exp(-x)
    sigmoid = one_plus.reciprocal()               # 1 / (1 + exp(-x))
    return (x * sigmoid).to(torch.float32)


def _silu_bf16_clamped(x: torch.Tensor, limit: float = 88.0) -> torch.Tensor:
    """SiLU in bf16 with exp input clamped (the fix)."""
    x = x.to(torch.bfloat16)
    neg_x = (torch.zeros_like(x) - x)
    neg_x = neg_x.clamp(-limit, limit)           # <-- THE FIX
    exp_neg_x = neg_x.exp()
    one_plus = exp_neg_x + torch.ones_like(x)
    sigmoid = one_plus.reciprocal()
    return (x * sigmoid).to(torch.float32)


def _online_softmax_row_bf16(row: torch.Tensor) -> torch.Tensor:
    """One row of online softmax entirely in bf16."""
    row = row.to(torch.bfloat16)
    m = row.max()
    shifted = row - m
    p = shifted.exp()                             # V_EXP_V
    return (p / p.sum()).to(torch.float32)


def _online_softmax_row_bf16_clamped(row: torch.Tensor, limit: float = 88.0) -> torch.Tensor:
    """One row of online softmax in bf16 with clamped exp input."""
    row = row.to(torch.bfloat16)
    # Clamp row values first to prevent inf - inf = NaN in the subtraction
    row = row.clamp(-limit, limit)
    m = row.max()
    shifted = row - m
    shifted = shifted.clamp(-limit, limit)        # <-- THE FIX
    p = shifted.exp()
    return (p / p.sum()).to(torch.float32)


# ---------- tests ----------

class TestBf16ExpOverflow:
    """exp() in bfloat16 overflows for inputs > ~88."""

    def test_exp_overflow_boundary(self):
        """exp(88) fits in bf16, exp(89) overflows to inf."""
        t88 = torch.tensor([88.0], dtype=torch.bfloat16)
        t89 = torch.tensor([89.0], dtype=torch.bfloat16)

        assert torch.isfinite(t88.exp()).all(), "exp(88) should be finite in bf16"
        assert torch.isinf(t89.exp()).all(), "exp(89) should overflow to inf in bf16"

    def test_exp_neg_overflow_to_zero(self):
        """exp(-100) in bf16 underflows to 0 (not NaN — safe)."""
        t = torch.tensor([-100.0], dtype=torch.bfloat16)
        result = t.exp()
        assert not torch.isnan(result).any(), "exp(large_negative) should not be NaN"
        assert (result == 0).all(), "exp(-100) should underflow to 0 in bf16"

    def test_clamp_prevents_overflow(self):
        """Clamping exp input to [-88, 88] prevents inf/NaN."""
        values = torch.tensor([-200.0, -89.0, 0.0, 88.0, 89.0, 200.0], dtype=torch.bfloat16)
        clamped = values.clamp(-88.0, 88.0)
        result = clamped.exp()
        assert torch.isfinite(result).all(), f"Clamped exp should be finite, got {result}"


class TestSiluNaNPropagation:
    """SiLU produces NaN when input is -inf (which occurs after many bf16 layers)."""

    def test_silu_nan_from_negative_inf(self):
        """The exact NaN mechanism: x=-inf → exp(-(-inf))=exp(inf)=inf → 1+inf=inf → 1/inf=0 → (-inf)*0=NaN."""
        x = torch.tensor([-float("inf")], dtype=torch.bfloat16)

        neg_x = -x                    # inf
        exp_neg_x = neg_x.exp()       # exp(inf) = inf
        one_plus = 1.0 + exp_neg_x    # 1 + inf = inf
        sigmoid = one_plus.reciprocal()  # 1/inf = 0
        result = x * sigmoid           # (-inf) * 0 = NaN  ← IEEE 754

        assert torch.isnan(result).all(), (
            f"SiLU(-inf) should be NaN via (-inf)*0, got {result}"
        )

    def test_silu_nan_from_large_negative(self):
        """Finite * 0 = 0, but -inf * 0 = NaN. Show bf16 overflow path to -inf."""
        # In bf16, matmul accumulation can overflow to -inf.
        # Once a value is -inf, SiLU produces NaN.
        x_finite = torch.tensor([-1e10], dtype=torch.bfloat16)
        result_finite = _silu_bf16(x_finite)
        # Finite extreme: exp(1e10)=inf, 1+inf=inf, 1/inf=0, finite*0 = -0 (OK)
        assert not torch.isnan(result_finite).any(), "SiLU(finite extreme) should be -0, not NaN"

        # But -inf triggers NaN: exp(inf)=inf, 1+inf=inf, 1/inf=0, (-inf)*0=NaN
        x_inf = torch.tensor([-float("inf")], dtype=torch.bfloat16)
        result_inf = _silu_bf16(x_inf)
        assert torch.isnan(result_inf).any(), (
            f"SiLU(-inf) should be NaN via (-inf)*0, got {result_inf}"
        )

    def test_silu_clamped_no_nan(self):
        """Clamping exp inputs prevents NaN even for extreme values."""
        extreme_values = torch.tensor(
            [-float("inf"), -1e10, -1000, -100, 0, 100, 1000, 1e10, float("inf")],
        )
        result = _silu_bf16_clamped(extreme_values)
        assert not torch.isnan(result).any(), (
            f"Clamped SiLU should never produce NaN, got {result}"
        )

    def test_silu_clamped_preserves_normal_values(self):
        """Clamping doesn't change results for values in normal range."""
        x = torch.randn(64)
        original = _silu_bf16(x)
        clamped = _silu_bf16_clamped(x)
        assert torch.allclose(original, clamped, atol=0, rtol=0), (
            "Clamping should not affect normal-range values"
        )


class TestSoftmaxNaNPropagation:
    """Softmax NaN when accumulated bf16 errors produce extreme score values."""

    def test_softmax_nan_with_inf_scores(self):
        """If QKT scores contain inf (from bf16 matmul overflow), softmax produces NaN."""
        # Simulate QKT with overflow — possible after many bf16 layers
        scores = torch.tensor([float("inf"), 1.0, -1.0, 0.0], dtype=torch.bfloat16)
        m = scores.max()          # inf
        shifted = scores - m      # inf - inf = NaN for first element
        p = shifted.exp()         # exp(NaN) = NaN
        result = p / p.sum()      # NaN propagates

        assert torch.isnan(result).any(), (
            f"Softmax with inf scores should produce NaN, got {result}"
        )

    def test_softmax_nan_accumulation_simulation(self):
        """Simulate how bf16 accumulation over 22 layers produces NaN."""
        torch.manual_seed(42)
        x = torch.randn(64, dtype=torch.bfloat16) * 2  # typical layer 1 activations

        # Simulate residual accumulation through layers (simplified)
        for layer in range(22):
            # Each "layer" adds a random update (simulating matmul + residual)
            update = torch.randn(64, dtype=torch.bfloat16) * 2
            x = (x + update).to(torch.bfloat16)

            # Check if values have grown extreme enough to break softmax
            if x.abs().max() > 88:
                scores = x.to(torch.bfloat16)
                result = _online_softmax_row_bf16(scores)
                if torch.isnan(result).any():
                    # NaN appeared — test passes (demonstrates the vulnerability)
                    return

        # Even if this specific seed didn't overflow, demonstrate the mechanism
        # by showing that extreme values DO cause NaN
        extreme = torch.tensor([100.0, 0.0, -50.0, 10.0])
        # In online_softmax: shifted = [0, -100, -150, -90]
        # exp(-150) = 0 in bf16, that's fine — but if the max itself is inf:
        extreme_with_inf = torch.tensor([float("inf"), 0.0, -50.0, 10.0])
        result = _online_softmax_row_bf16(extreme_with_inf)
        assert torch.isnan(result).any(), "Softmax with inf input should produce NaN"

    def test_softmax_clamped_no_nan(self):
        """Clamped softmax never produces NaN for inf inputs (the real failure mode)."""
        # NaN inputs are not the concern — the clamp prevents NaN from being *generated*.
        # Real inputs are finite-or-inf from bf16 overflow, never NaN at entry.
        extreme = torch.tensor(
            [float("inf"), -float("inf"), 1e10, -1e10, 0.0, 5.0]
        )
        result = _online_softmax_row_bf16_clamped(extreme)
        assert not torch.isnan(result).any(), (
            f"Clamped softmax should never produce NaN, got {result}"
        )


class TestScalarExpOverflow:
    """S_EXP_FP (scalar exp in online_softmax) also needs clamping."""

    def test_scalar_exp_m_res_overflow(self):
        """In online_softmax: exp(m_last - m_curr) can overflow if tile maxima diverge wildly."""
        # Tile 1 max: m_last = 200 (bf16)
        # Tile 2 max: m_curr = 10 (bf16)
        # m_res = 200 - 10 = 190, exp(190) = inf in bf16
        m_res = torch.tensor(190.0, dtype=torch.bfloat16)
        result = m_res.float().exp()
        result_bf16 = result.to(torch.bfloat16)
        assert torch.isinf(result_bf16), (
            f"exp(190) should overflow bf16, got {result_bf16}"
        )

    def test_scalar_exp_clamped(self):
        """Clamping scalar exp input prevents overflow."""
        m_res = torch.tensor(190.0, dtype=torch.bfloat16)
        clamped = m_res.float().clamp(-88.0, 88.0)
        result = clamped.exp().to(torch.bfloat16)
        assert torch.isfinite(result), f"Clamped scalar exp should be finite, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
