"""Real published weights, not `randn`.

Synthetic weights make every lowering test pass on values that no trained model
would produce, so a defect that only appears on realistic structure never
surfaces. This binds a real one: layer 0 of `AntonV/mamba2-130m-hf`, a published
Mamba-2 with 24 layers, `state_size` 128, 24 heads of `head_dim` 64.

What a real checkpoint tests that synthetic weights cannot:

* **Structure.** Layer 0's `A` is not a spread: 23 of its 24 heads sit between
  -0.73 and -0.27, and one sits at -5.06, about ten times the rest. A tight
  cluster plus a single outlier is what training produces and a symmetric draw
  does not -- and that outlier head is the one most likely to expose a decay
  that underflows.
* **Layout.** The projection packs `z`, `x`, `B`, `C` and `dt` into one 3352-wide
  tensor in a fixed order; `conv1d.weight` is `[channels, 1, kernel]`, not
  `[channels, kernel]`. Getting either wrong is a plausible wrong answer that
  synthetic weights, generated to whatever shape the test assumed, cannot catch.

Skipped when the checkpoint is not cached: this must not require network in CI.
Fetch it once with

    python -c "from huggingface_hub import snapshot_download as d; d('AntonV/mamba2-130m-hf')"
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.mamba2.reference import (  # noqa: E402
    mamba2_recurrent_reference,
)
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.program_mamba_common import Mamba2Shape  # noqa: E402
from compiler.aten.tests.isa_interpreter import Machine  # noqa: E402

REPO = "AntonV/mamba2-130m-hf"
MLEN = 64


def _checkpoint():
    """Layer 0's mixer weights and the model config, or skip."""
    try:
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
    except ImportError:  # pragma: no cover - depends on the environment
        pytest.skip("huggingface_hub / safetensors not installed")
    import json
    from pathlib import Path

    try:
        path = Path(snapshot_download(REPO, local_files_only=True))
    except Exception:  # pragma: no cover - not cached
        pytest.skip(
            f"{REPO} is not cached; fetch it with snapshot_download('{REPO}')"
        )
    config = json.loads((path / "config.json").read_text())
    weights = {}
    with safe_open(path / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            if key.startswith("backbone.layers.0.mixer."):
                weights[key.split("mixer.")[1]] = f.get_tensor(key).float()
    return config, weights


def test_the_real_decay_is_clustered_with_an_outlier():
    """The structure real weights have and `randn` does not.

    Layer 0's `A` is not a spread: 23 of its 24 heads sit between -0.73 and
    -0.27, and one is at -5.06, about ten times the rest. A tight cluster plus a
    single outlier is what training produces and what a symmetric draw does not,
    and it is the head most likely to expose a decay that underflows.

    Asserted from the checkpoint rather than from a threshold I picked: an
    earlier version of this demanded `min < -10` on the strength of nothing.
    """
    _, w = _checkpoint()
    a = -torch.exp(w["A_log"])
    outlier = a.min()
    rest = a[a != outlier]
    assert outlier < 5 * rest.min(), (
        f"the outlier at {outlier:.2f} is no longer far from the cluster "
        f"({rest.min():.2f}..{rest.max():.2f})"
    )
    assert rest.std() < 0.15, (
        f"the other heads are no longer clustered (std {rest.std():.3f})"
    )


def test_mamba2_decode_step_on_published_weights():
    """`ssm_decode_step_v0` against the textbook recurrence, on layer 0's real
    `A`, `D` and `dt_bias`, with the projection's own split order.

    The emitter takes `dA = exp(A * dt)` precomputed, because it depends only on
    a weight and a scalar; the reference takes raw `A` and computes it. Feeding
    both from the same checkpoint is the point.
    """
    config, w = _checkpoint()
    heads = config["num_heads"]
    head_dim = config["head_dim"]
    state = config["state_size"]
    groups = config["n_groups"]
    assert head_dim == MLEN, f"this test assumes head_dim == mlen, got {head_dim}"

    torch.manual_seed(0)
    # Activations, since B, C, dt and x come from the projection and not from a
    # weight. The weights under test are A_log, D and dt_bias.
    x = torch.randn(heads, head_dim) * 0.5
    b = torch.randn(groups, state) * 0.5
    c = torch.randn(groups, state) * 0.5
    dt = torch.nn.functional.softplus(torch.randn(heads) * 0.5 + w["dt_bias"])
    a = -torch.exp(w["A_log"])
    d = w["D"]

    y_ref, state_ref = mamba2_recurrent_reference(
        x=x[None, None], dt=dt[None, None], A=a,
        B=b.reshape(1, 1, groups, state), C=c.reshape(1, 1, groups, state),
        D=d, initial_state=torch.zeros(1, heads, state, head_dim),
    )

    shape = Mamba2Shape(
        hidden_size=config["hidden_size"], num_heads=heads, head_dim=head_dim,
        state_size=state, n_groups=groups, conv_kernel=config["conv_kernel"],
        chunk_size=config["chunk_size"], seq_len=1,
    )
    p = PlenaCompiler(mlen=MLEN, blen=4)
    up = lambda n: ((n + MLEN - 1) // MLEN) * MLEN  # noqa: E731
    st = p.alloc("state", up(heads * state), MLEN)
    xv = p.alloc("x", up(heads), MLEN)
    yv = p.alloc("y", up(heads), MLEN)
    scratch = p.alloc("scratch", MLEN, MLEN)
    fps = {
        "b": p.fp_var("b", size=groups * state),
        "c": p.fp_var("c", size=groups * state),
        "da": p.fp_var("da", size=heads),
        "dt": p.fp_var("dt", size=heads),
        "d": p.fp_var("d", size=heads),
    }
    consts = p.mamba_fp_constants()
    mark = len(p.get_code())
    p.ssm_decode_step_v0(
        state=st, x=xv, b_fp=fps["b"], c_fp=fps["c"], da_fp=fps["da"],
        dt_fp=fps["dt"], d_fp=fps["d"], y=yv, scratch=scratch,
        shape=shape, consts=consts,
    )
    code = p.get_code()[mark:]

    m = Machine(vlen=MLEN, vram_words=1 << 20, fpram_words=1 << 14)
    base = lambda v: p.get_vram_layout(v.name).vram_base_addr  # noqa: E731
    for h in range(heads):
        m.write_vram_row(base(xv) + h * MLEN, x[h].tolist())
        m.write_vram_row(base(yv) + h * MLEN, [7.5] * MLEN)   # garbage: must clear
        for n in range(state):
            m.write_vram_row(base(st) + (h * state + n) * MLEN, [0.0] * MLEN)
    m.write_fpram(fps["b"].address, b.flatten().tolist())
    m.write_fpram(fps["c"].address, c.flatten().tolist())
    m.write_fpram(fps["da"].address, torch.exp(a * dt).tolist())
    m.write_fpram(fps["dt"].address, dt.tolist())
    m.write_fpram(fps["d"].address, d.tolist())
    m.write_fpram(consts.zero.address, p.mamba_fp_constant_values(shape))
    m.run(code)

    y = torch.tensor([m.read_vram_row(base(yv) + h * MLEN, head_dim)
                      for h in range(heads)])
    torch.testing.assert_close(y, y_ref[0, 0], rtol=1e-4, atol=1e-5)

    got_state = torch.tensor([
        [m.read_vram_row(base(st) + (h * state + n) * MLEN, head_dim)
         for n in range(state)]
        for h in range(heads)
    ])
    torch.testing.assert_close(got_state, state_ref[0], rtol=1e-4, atol=1e-5)


def test_the_checkpoints_projection_layout_is_what_the_lowering_assumes():
    """`in_proj` packs `z`, `x`, `B`, `C`, `dt` in one tensor, in that order.

    The widths have to add up, and `conv1d.weight` is `[channels, 1, kernel]`
    rather than `[channels, kernel]` -- the reference's `_causal_conv_step`
    accepts both, and a lowering that assumed the wrong one would silently
    transpose a 4-tap filter.
    """
    config, w = _checkpoint()
    heads, head_dim = config["num_heads"], config["head_dim"]
    state, groups = config["state_size"], config["n_groups"]
    inner = heads * head_dim
    expected = 2 * inner + 2 * groups * state + heads
    assert w["in_proj.weight"].shape == (expected, config["hidden_size"]), (
        f"in_proj is {tuple(w['in_proj.weight'].shape)}; z|x|B|C|dt should be "
        f"{expected} wide"
    )
    conv_channels = inner + 2 * groups * state
    assert w["conv1d.weight"].shape == (conv_channels, 1, config["conv_kernel"])
    assert w["D"].shape == (heads,) and w["A_log"].shape == (heads,)
