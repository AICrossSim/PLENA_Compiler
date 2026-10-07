"""The chunked SSD reference must agree with the plain recurrence.

This is the load-bearing test of the whole Mamba effort. The PLENA prefill
lowering implements the chunked form, so the chunked reference is what the
emulator gets compared against -- and a chunked reference that quietly disagrees
with the recurrence would make every downstream numerical result meaningless
while looking green.

The cases below deliberately include multi-chunk sequences, a non-dividing
sequence length (so the tail chunk is exercised), multiple groups, and an
initial state, because those are the four places the chunk bookkeeping can be
wrong without a single-chunk test noticing.
"""

from __future__ import annotations

import unittest

import torch

from compiler.aten.models.mamba2.reference import (
    mamba2_recurrent_reference,
    mamba2_reference,
    random_mamba2_params,
    ssd_chunk_reference,
)


def _scan_inputs(*, batch, seq, heads, head_dim, state, groups, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, seq, heads, head_dim, generator=g)
    # dt strictly positive, as softplus guarantees downstream.
    dt = torch.rand(batch, seq, heads, generator=g) * 0.2 + 0.01
    A = -torch.exp(torch.rand(heads, generator=g) * 1.0)
    B = torch.randn(batch, seq, groups, state, generator=g) * 0.3
    C = torch.randn(batch, seq, groups, state, generator=g) * 0.3
    D = torch.randn(heads, generator=g) * 0.5
    return x, dt, A, B, C, D


class TestSSDChunkedMatchesRecurrence(unittest.TestCase):
    def _assert_agrees(self, *, seq, chunk, groups=1, heads=4, initial=False, seed=0):
        head_dim, state, batch = 8, 4, 2
        x, dt, A, B, C, D = _scan_inputs(
            batch=batch, seq=seq, heads=heads, head_dim=head_dim, state=state, groups=groups, seed=seed
        )
        init = None
        if initial:
            gen = torch.Generator().manual_seed(seed + 1)
            init = torch.randn(batch, heads, state, head_dim, generator=gen) * 0.2

        y_ref, h_ref = mamba2_recurrent_reference(x, dt, A, B, C, D, initial_state=init)
        y_chunk, h_chunk = ssd_chunk_reference(x, dt, A, B, C, D, chunk, initial_state=init)

        torch.testing.assert_close(y_chunk, y_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(h_chunk, h_ref, rtol=1e-4, atol=1e-5)

    def test_single_chunk(self):
        self._assert_agrees(seq=8, chunk=8)

    def test_multiple_whole_chunks(self):
        self._assert_agrees(seq=32, chunk=8)

    def test_tail_chunk_shorter_than_chunk_size(self):
        # seq % chunk != 0 is the branch most likely to be live at real dims and
        # least likely to be covered by a synthetic single-chunk test.
        self._assert_agrees(seq=30, chunk=8)

    def test_multiple_groups(self):
        self._assert_agrees(seq=24, chunk=8, groups=2, heads=4)

    def test_carries_an_initial_state(self):
        # Cross-chunk state passing is the entire point of the chunked form; a
        # zero initial state hides an error in exactly that term.
        self._assert_agrees(seq=24, chunk=8, initial=True)

    def test_slow_decay_regime(self):
        """A close to zero makes the decay ~1 and stresses accumulation.

        This is the regime that justifies an SSM (long memory) and the regime
        where a chunked form that mishandles the inter-chunk decay drifts.
        """
        batch, seq, heads, head_dim, state, groups, chunk = 2, 32, 4, 8, 4, 1, 8
        g = torch.Generator().manual_seed(7)
        x = torch.randn(batch, seq, heads, head_dim, generator=g)
        dt = torch.rand(batch, seq, heads, generator=g) * 0.02 + 0.001
        A = -torch.full((heads,), 0.05)
        B = torch.randn(batch, seq, groups, state, generator=g) * 0.3
        C = torch.randn(batch, seq, groups, state, generator=g) * 0.3
        D = torch.randn(heads, generator=g) * 0.5

        y_ref, h_ref = mamba2_recurrent_reference(x, dt, A, B, C, D)
        y_chunk, h_chunk = ssd_chunk_reference(x, dt, A, B, C, D, chunk)
        torch.testing.assert_close(y_chunk, y_ref, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(h_chunk, h_ref, rtol=1e-4, atol=1e-5)


class TestMamba2LayerReference(unittest.TestCase):
    def test_full_layer_agrees_between_chunked_and_recurrent_scans(self):
        params = random_mamba2_params(
            hidden_size=32, num_heads=4, head_dim=8, state_size=4, n_groups=1, chunk_size=8, seed=3
        )
        hidden = torch.randn(2, 24, 32, generator=torch.Generator().manual_seed(4))

        chunked = mamba2_reference(hidden, params, use_chunked=True)
        recurrent = mamba2_reference(hidden, params, use_chunked=False)

        torch.testing.assert_close(chunked.y, recurrent.y, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(chunked.final_state, recurrent.final_state, rtol=1e-4, atol=1e-5)

    def test_intermediates_have_the_shapes_the_lowering_assumes(self):
        params = random_mamba2_params(
            hidden_size=32, num_heads=4, head_dim=8, state_size=4, n_groups=2, chunk_size=8, seed=5
        )
        out = mamba2_reference(torch.randn(2, 16, 32), params)

        self.assertEqual(out.y.shape, (2, 16, 32))
        self.assertEqual(out.x.shape, (2, 16, 4, 8))
        self.assertEqual(out.B.shape, (2, 16, 2, 4))
        self.assertEqual(out.C.shape, (2, 16, 2, 4))
        self.assertEqual(out.dt.shape, (2, 16, 4))
        self.assertEqual(out.final_state.shape, (2, 4, 4, 8))

    def test_dt_is_positive_and_respects_the_clamp(self):
        params = random_mamba2_params(
            hidden_size=32, num_heads=4, head_dim=8, state_size=4, chunk_size=8, seed=6
        )
        params.time_step_min = 0.01
        params.time_step_max = 0.1
        out = mamba2_reference(torch.randn(2, 16, 32), params)
        self.assertTrue(torch.all(out.dt >= 0.01))
        self.assertTrue(torch.all(out.dt <= 0.1))

    def test_gate_is_applied_before_the_variance(self):
        """RMSNorm(y * silu(z)), not RMSNorm(y) * silu(z).

        Both upstream implementations gate first -- HuggingFace's MambaRMSNormGated
        reduces after multiplying, and mamba_ssm defaults to norm_before_gate=False.
        RMSNorm is not per-element homogeneous, so the two orders are different
        functions; at these shapes they differ by tens of percent, and the error is
        unrecoverable once real weights are loaded.
        """
        params = random_mamba2_params(
            hidden_size=32, num_heads=4, head_dim=8, state_size=4, n_groups=1, chunk_size=8, seed=11
        )
        hidden = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(12))
        out = mamba2_reference(hidden, params)

        # Recompute the norm block both ways from the same scan output.
        d_inner = params.d_inner
        y = out.scan_out.reshape(2, 16, d_inner)
        z = out.z
        gate_then_norm = y * torch.nn.functional.silu(z)
        gate_then_norm = gate_then_norm * torch.rsqrt(
            gate_then_norm.pow(2).mean(-1, keepdim=True) + params.norm_eps
        )
        gate_then_norm = gate_then_norm * params.norm_weight
        expected = gate_then_norm @ params.out_proj_weight.T

        torch.testing.assert_close(out.y, expected, rtol=1e-4, atol=1e-5)

        norm_then_gate = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + params.norm_eps)
        norm_then_gate = norm_then_gate * params.norm_weight * torch.nn.functional.silu(z)
        self.assertFalse(
            torch.allclose(expected, norm_then_gate @ params.out_proj_weight.T, rtol=1e-2),
            "the two orders must be distinguishable, else this test proves nothing",
        )

    def test_grouped_norm_differs_from_ungrouped_when_n_groups_gt_1(self):
        """n_groups > 1 must actually change the normalisation.

        Nothing else in the suite exercised the grouping, so a lowering that
        normalised over the whole of d_inner would have passed everything.
        """
        params = random_mamba2_params(
            hidden_size=32, num_heads=4, head_dim=8, state_size=4, n_groups=2, chunk_size=8, seed=13
        )
        hidden = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(14))
        out = mamba2_reference(hidden, params)

        d_inner = params.d_inner
        gated = out.scan_out.reshape(2, 16, d_inner) * torch.nn.functional.silu(out.z)
        ungrouped = gated * torch.rsqrt(gated.pow(2).mean(-1, keepdim=True) + params.norm_eps)
        ungrouped = (ungrouped * params.norm_weight) @ params.out_proj_weight.T
        self.assertFalse(
            torch.allclose(out.y, ungrouped, rtol=1e-2),
            "grouped and ungrouped RMSNorm must differ at n_groups=2",
        )

    def test_full_layer_agrees_between_scans_with_multiple_groups(self):
        params = random_mamba2_params(
            hidden_size=32, num_heads=4, head_dim=8, state_size=4, n_groups=2, chunk_size=8, seed=15
        )
        hidden = torch.randn(2, 24, 32, generator=torch.Generator().manual_seed(16))
        chunked = mamba2_reference(hidden, params, use_chunked=True)
        recurrent = mamba2_reference(hidden, params, use_chunked=False)
        torch.testing.assert_close(chunked.y, recurrent.y, rtol=1e-4, atol=1e-5)

    def test_conv1d_is_causal(self):
        """Perturbing a future timestep must not change an earlier output.

        Catches an off-by-one in the tap indexing, which is the single easiest
        mistake to make in the conv and is invisible in an aggregate error metric.
        """
        params = random_mamba2_params(
            hidden_size=32, num_heads=4, head_dim=8, state_size=4, chunk_size=8, seed=8
        )
        hidden = torch.randn(1, 12, 32, generator=torch.Generator().manual_seed(9))
        base = mamba2_reference(hidden, params)

        perturbed = hidden.clone()
        perturbed[:, 7:] += 5.0
        after = mamba2_reference(perturbed, params)

        torch.testing.assert_close(after.y[:, :7], base.y[:, :7], rtol=1e-5, atol=1e-6)
        self.assertFalse(torch.allclose(after.y[:, 7:], base.y[:, 7:]))


if __name__ == "__main__":
    unittest.main()
