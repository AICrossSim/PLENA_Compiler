from __future__ import annotations

import math

import pytest
import torch

from compiler.aten.models.kda.reference import (
    KdaConvWeights,
    KdaShape,
    KdaState,
    KdaRecurrentState,
    activate_log_decay,
    kda_recurrent_sequence,
    kda_state_engine_prefill,
    kda_state_engine_step,
    kda_step,
)
from compiler.aten.models.kda.state_precision import StateStorage, quantize_state, storage_bytes


def _small_shape() -> KdaShape:
    return KdaShape(hidden_size=8, num_heads=2, key_dim=4, value_dim=3, conv_kernel=2)


def test_official_kimi_k3_state_geometry() -> None:
    shape = KdaShape.kimi_k3()
    assert shape.projection_size == 12_288
    assert shape.state_elements == 1_572_864
    assert shape.state_elements * 4 == 6 * 1024 * 1024
    assert shape.state_elements * 2 == 3 * 1024 * 1024
    assert shape.conv_state_elements * 2 == 288 * 1024


def test_log_decay_is_channelwise_and_bounded() -> None:
    shape = _small_shape()
    gate = torch.tensor(
        [[[-100.0, -1.0, 0.0, 1.0], [100.0, 1.0, 0.0, -1.0]]],
        dtype=torch.float32,
    )
    decay = activate_log_decay(
        gate,
        torch.zeros(shape.num_heads),
        torch.zeros(shape.num_heads, shape.key_dim),
        lower_bound=shape.gate_lower_bound,
    )
    assert torch.all(decay <= 0)
    assert torch.all(decay >= shape.gate_lower_bound)
    assert decay[0, 0, 0] > decay[0, 0, -1]


def test_log_decay_matches_its_closed_form() -> None:
    """Pin the exact Kimi K3 formula, with a_log and dt_bias actually varying.

    test_log_decay_is_channelwise_and_bounded passes zeros for both, so it
    holds for any function of `gate` alone -- dropping the dt_bias term or the
    exp(a_log) rate leaves it green. Every other test in this file that touches
    decay calls activate_log_decay as its own oracle, so nothing else can catch
    an error inside it either.
    """
    torch.manual_seed(23)
    shape = KdaShape(hidden_size=32, num_heads=3, key_dim=5, value_dim=7, conv_kernel=4)
    batch = 2
    gate = torch.randn(batch, shape.num_heads, shape.key_dim)
    a_log = torch.randn(shape.num_heads)
    dt_bias = torch.randn(shape.num_heads, shape.key_dim)

    actual = activate_log_decay(
        gate, a_log, dt_bias, lower_bound=shape.gate_lower_bound
    )
    rate = torch.exp(a_log)[None, :, None]
    expected = shape.gate_lower_bound * torch.sigmoid(
        rate * (gate + dt_bias[None, :, :])
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_step_matches_delta_rule_matrix_form() -> None:
    torch.manual_seed(7)
    shape = _small_shape()
    batch = 1
    q = torch.randn(batch, shape.num_heads, shape.key_dim)
    k = torch.randn_like(q)
    v = torch.randn(batch, shape.num_heads, shape.value_dim)
    gate = torch.randn_like(q)
    beta_logit = torch.randn(batch, shape.num_heads)
    initial = KdaState(torch.randn(batch, shape.num_heads, shape.value_dim, shape.key_dim))
    a_log = torch.randn(shape.num_heads)
    dt_bias = torch.randn(shape.num_heads, shape.key_dim)
    scale = 0.25

    output, final = kda_step(
        q,
        k,
        v,
        gate,
        beta_logit,
        initial,
        a_log,
        dt_bias,
        shape,
        scale=scale,
        state_storage=StateStorage.FP32,
    )

    qn = q.float() * torch.rsqrt(q.float().square().sum(dim=-1, keepdim=True) + 1.0e-6)
    kn = k.float() * torch.rsqrt(k.float().square().sum(dim=-1, keepdim=True) + 1.0e-6)
    log_decay = activate_log_decay(gate, a_log, dt_bias, lower_bound=shape.gate_lower_bound)
    decayed = initial.recurrent * torch.exp(log_decay)[:, :, None, :]
    beta = torch.sigmoid(beta_logit)
    expected_states = []
    expected_outputs = []
    for head in range(shape.num_heads):
        identity = torch.eye(shape.key_dim)
        transition = identity - beta[0, head] * torch.outer(kn[0, head], kn[0, head])
        expected = decayed[0, head] @ transition
        expected = expected + beta[0, head] * torch.outer(v[0, head], kn[0, head])
        expected_states.append(expected)
        expected_outputs.append(scale * (expected @ qn[0, head]))
    expected_state = torch.stack(expected_states).unsqueeze(0)
    expected_output = torch.stack(expected_outputs).unsqueeze(0)

    torch.testing.assert_close(final.recurrent, expected_state)
    torch.testing.assert_close(output, expected_output)


def test_sequence_matches_repeated_steps() -> None:
    torch.manual_seed(11)
    shape = _small_shape()
    batch, tokens = 2, 5
    q = torch.randn(batch, tokens, shape.num_heads, shape.key_dim)
    k = torch.randn_like(q)
    v = torch.randn(batch, tokens, shape.num_heads, shape.value_dim)
    gate = torch.randn_like(q)
    beta = torch.randn(batch, tokens, shape.num_heads)
    a_log = torch.randn(shape.num_heads)
    dt_bias = torch.randn(shape.num_heads, shape.key_dim)
    initial = KdaState.zeros(shape, batch)

    outputs, final = kda_recurrent_sequence(
        q,
        k,
        v,
        gate,
        beta,
        initial,
        a_log,
        dt_bias,
        shape,
        state_storage=StateStorage.FP32,
    )
    assert outputs.shape == (batch, tokens, shape.num_heads, shape.value_dim)
    assert final.recurrent.shape == (batch, shape.num_heads, shape.value_dim, shape.key_dim)
    assert torch.isfinite(outputs).all()

    # The name of this test promises this comparison; without it the shape and
    # isfinite assertions above hold even if the sequence drops the carried
    # state between tokens, or emits them in reverse order.
    carried = KdaState.zeros(shape, batch)
    expected = []
    for token in range(tokens):
        out, carried = kda_step(
            q[:, token],
            k[:, token],
            v[:, token],
            gate[:, token],
            beta[:, token],
            carried,
            a_log,
            dt_bias,
            shape,
            state_storage=StateStorage.FP32,
        )
        expected.append(out)
    torch.testing.assert_close(
        outputs, torch.stack(expected, dim=1), rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        final.recurrent, carried.recurrent, rtol=1e-6, atol=1e-7
    )


def test_prefill_includes_conv_and_matches_repeated_step() -> None:
    torch.manual_seed(17)
    shape = _small_shape()
    batch, tokens = 2, 4
    key_width = shape.num_heads * shape.key_dim
    value_width = shape.num_heads * shape.value_dim
    projection_width = 3 * key_width + value_width + shape.num_heads
    projected = torch.randn(batch, tokens, projection_width)
    conv_weights = KdaConvWeights(
        q=torch.randn(key_width, shape.conv_kernel),
        k=torch.randn(key_width, shape.conv_kernel),
        v=torch.randn(value_width, shape.conv_kernel),
    )
    a_log = torch.randn(shape.num_heads)
    dt_bias = torch.randn(shape.num_heads, shape.key_dim)
    initial = KdaRecurrentState.zeros(shape, batch)
    output, state = kda_state_engine_prefill(
        projected,
        initial,
        conv_weights,
        a_log,
        dt_bias,
        shape,
        state_storage=StateStorage.BF16,
    )
    expected_outputs = []
    expected_state = initial
    for token in projected.unbind(1):
        token_output, expected_state = kda_state_engine_step(
            token,
            expected_state,
            conv_weights,
            a_log,
            dt_bias,
            shape,
            state_storage=StateStorage.BF16,
        )
        expected_outputs.append(token_output)
    torch.testing.assert_close(output, torch.stack(expected_outputs, dim=1))
    torch.testing.assert_close(state.recurrent, expected_state.recurrent)
    torch.testing.assert_close(state.conv, expected_state.conv)


def test_default_scale_uses_key_dimension() -> None:
    shape = _small_shape()
    q = torch.ones(1, shape.num_heads, shape.key_dim)
    k = torch.ones_like(q)
    v = torch.ones(1, shape.num_heads, shape.value_dim)
    gate = torch.zeros_like(q)
    beta = torch.zeros(1, shape.num_heads)
    state = KdaState.zeros(shape, 1)
    a_log = torch.zeros(shape.num_heads)
    dt_bias = torch.zeros(shape.num_heads, shape.key_dim)

    default, _ = kda_step(q, k, v, gate, beta, state, a_log, dt_bias, shape)
    explicit, _ = kda_step(
        q,
        k,
        v,
        gate,
        beta,
        state,
        a_log,
        dt_bias,
        shape,
        scale=1.0 / math.sqrt(shape.key_dim),
    )
    torch.testing.assert_close(default, explicit)


def test_shape_validation_rejects_wrong_state_layout() -> None:
    shape = _small_shape()
    q = torch.zeros(1, shape.num_heads, shape.key_dim)
    v = torch.zeros(1, shape.num_heads, shape.value_dim)
    wrong = KdaState(torch.zeros(1, shape.num_heads, shape.key_dim, shape.value_dim))
    with pytest.raises(ValueError, match="state has shape"):
        kda_step(
            q,
            q,
            v,
            q,
            torch.zeros(1, shape.num_heads),
            wrong,
            torch.zeros(shape.num_heads),
            torch.zeros(shape.num_heads, shape.key_dim),
            shape,
        )


def test_conv_state_precision_is_independent_of_recurrent_state_precision() -> None:
    """FP32 recurrent state with BF16 conv state is what actually ships.

    The descriptor carries `state_precision` and `conv_state_precision`
    separately, but this reference used to fold both onto one parameter, so the
    shipped Kimi combination had no CPU reference at all. Two tokens are needed:
    the first token's conv rounding only becomes observable once it is read back
    as history.
    """
    torch.manual_seed(23)
    shape = _small_shape()
    batch, tokens = 1, 2
    key_width = shape.num_heads * shape.key_dim
    value_width = shape.num_heads * shape.value_dim
    projected = torch.randn(batch, tokens, 3 * key_width + value_width + shape.num_heads)
    conv_weights = KdaConvWeights(
        q=torch.randn(key_width, shape.conv_kernel),
        k=torch.randn(key_width, shape.conv_kernel),
        v=torch.randn(value_width, shape.conv_kernel),
    )
    a_log = torch.randn(shape.num_heads)
    dt_bias = torch.randn(shape.num_heads, shape.key_dim)
    initial = KdaRecurrentState.zeros(shape, batch)

    common = dict(state_storage=StateStorage.FP32)
    uniform, uniform_state = kda_state_engine_prefill(projected, initial, conv_weights, a_log, dt_bias, shape, **common)
    mixed, mixed_state = kda_state_engine_prefill(
        projected, initial, conv_weights, a_log, dt_bias, shape, conv_state_storage=StateStorage.BF16, **common
    )

    # The recurrent state stays FP32 in both runs, so any divergence can only
    # have entered through the conv history.
    assert not torch.equal(uniform_state.conv, mixed_state.conv)
    assert not torch.equal(uniform, mixed)
    torch.testing.assert_close(uniform, mixed, rtol=2e-2, atol=2e-2)


def test_decode_step_matches_the_transposed_formulation():
    """``T[k, v] == S[v, k]``. The contract between this reference and the
    PLENA lowering in ``aten/plena/program_kda_recurrent.py``.

    The lowering stores state transposed, with ``key`` as the row axis, because
    in that orientation every step of the recurrence -- decay, predict, update,
    read out -- is one broadcast-scalar operation per row. This test pins the
    two orientations together so a lowering bug cannot hide behind a plausible
    reimplementation of the maths.
    """
    torch.manual_seed(0)
    # Non-degenerate on purpose, each choice load-bearing:
    #   batch = 2       -- batch 1 lets a batch-0 broadcast leak pass
    #   key != value    -- a square state makes a transposed axis numerically
    #                      equivalent instead of a shape error, and lets the
    #                      output scale 1/sqrt(key_dim) be confused with value_dim
    #   0.05 * randn    -- at unit variance ||x||^2 is O(1) and the 1e-6 epsilon
    #                      inside rsqrt perturbs by ~2e-7, under this test's atol,
    #                      so the FlashKDA normalisation convention below would
    #                      not actually be pinned. At this magnitude it is.
    shape = KdaShape(hidden_size=32, num_heads=3, key_dim=8, value_dim=5, conv_kernel=4)
    batch = 2

    state = KdaState(
        torch.randn(batch, shape.num_heads, shape.value_dim, shape.key_dim)
    )
    q = 0.05 * torch.randn(batch, shape.num_heads, shape.key_dim)
    k = 0.05 * torch.randn(batch, shape.num_heads, shape.key_dim)
    v = torch.randn(batch, shape.num_heads, shape.value_dim)
    gate = torch.randn(batch, shape.num_heads, shape.key_dim)
    beta_logit = torch.randn(batch, shape.num_heads)
    a_log = torch.randn(shape.num_heads)
    dt_bias = torch.randn(shape.num_heads, shape.key_dim)

    expected_out, expected_state = kda_step(
        q, k, v, gate, beta_logit, state.clone(), a_log, dt_bias, shape
    )

    # --- transposed replay: T is [batch, heads, key_dim, value_dim] ---------
    t = state.recurrent.transpose(-2, -1).clone()

    # Same epsilon-inside-rsqrt normalisation the reference uses;
    # torch.nn.functional.normalize clamps instead and does NOT agree.
    def _norm(x):
        return x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1.0e-6)

    q_n, k_n = _norm(q), _norm(k)
    log_decay = activate_log_decay(
        gate, a_log, dt_bias, lower_bound=shape.gate_lower_bound
    )

    # decay: one scalar per key row
    t = t * torch.exp(log_decay)[:, :, :, None]
    # predict: accumulate a value-length vector across key rows
    pred = torch.einsum("bhkv,bhk->bhv", t, k_n)
    beta = torch.sigmoid(beta_logit.float())[:, :, None]
    error = beta * (v.float() - pred)
    # update: rank-1, one scalar per key row
    t = t + error[:, :, None, :] * k_n[:, :, :, None]
    # read out: on the UPDATED state
    scale = 1.0 / math.sqrt(shape.key_dim)
    out = scale * torch.einsum("bhkv,bhk->bhv", t, q_n)

    torch.testing.assert_close(out, expected_out, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        t.transpose(-2, -1), expected_state.recurrent, rtol=1e-5, atol=1e-6
    )


# ---------------------------------------------------------------------------
# state_precision.py coverage.
#
# These two cover a module that arrived without its tests. A second Mamba
# reference beside aten/models/mamba2/reference.py would be a drift hazard, so
# only the precision helper was kept -- and without these
# quantize_state's MX8_B128 path, _round_e4m3fn, and storage_bytes have zero
# coverage in this repo while sitting in the import graph of every KDA test.
# ---------------------------------------------------------------------------


def test_state_storage_round_trips_and_byte_counts() -> None:
    value = torch.randn(257, generator=torch.Generator().manual_seed(23)) * 3.7
    torch.testing.assert_close(quantize_state(value, StateStorage.FP32), value)
    for storage in (StateStorage.BF16, StateStorage.FP16, StateStorage.MX8_B128):
        rounded = quantize_state(value, storage)
        assert rounded.shape == value.shape
        assert torch.isfinite(rounded).all()
        assert not torch.equal(rounded, value)
    assert storage_bytes(256, StateStorage.FP32) == 1024
    assert storage_bytes(256, StateStorage.BF16) == 512
    assert storage_bytes(256, StateStorage.MX8_B128) == 258


def test_mx8_e4m3fn_rounding_carries_across_exponents() -> None:
    value = torch.zeros(128)
    value[:5] = torch.tensor([1.9375, 248.0, 432.0, 448.0, -448.0])
    restored = quantize_state(value, StateStorage.MX8_B128)
    torch.testing.assert_close(
        restored[:5],
        torch.tensor([2.0, 256.0, 448.0, 448.0, -448.0]),
        rtol=0,
        atol=0,
    )
