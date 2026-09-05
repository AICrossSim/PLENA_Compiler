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

import json
import os
import sys
from pathlib import Path

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


def _checkpoint_path() -> Path:
    """Return the pinned local snapshot without making tests network-dependent."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover - depends on the environment
        pytest.skip("huggingface_hub is not installed")

    explicit = os.environ.get("PLENA_MAMBA2_130M_CHECKPOINT")
    if explicit:
        path = Path(explicit)
        if not (path / "config.json").exists():
            pytest.fail(f"PLENA_MAMBA2_130M_CHECKPOINT is not a snapshot: {path}")
        return path

    try:
        return Path(snapshot_download(REPO, local_files_only=True))
    except Exception:  # pragma: no cover - not cached
        pytest.skip(
            f"{REPO} is not cached; fetch it with snapshot_download('{REPO}')"
        )


def _checkpoint():
    """Layer 0's mixer weights and the model config, or skip."""
    try:
        from safetensors import safe_open
    except ImportError:  # pragma: no cover - depends on the environment
        pytest.skip("safetensors is not installed")
    path = _checkpoint_path()
    config = json.loads((path / "config.json").read_text())
    weights = {}
    with safe_open(path / "model.safetensors", framework="pt") as f:
        for key in f.keys():
            if key.startswith("backbone.layers.0.mixer."):
                weights[key.split("mixer.")[1]] = f.get_tensor(key).float()
    return config, weights


class _RealRecurrenceRuntime:
    """One reusable PLENA recurrence program for every layer in the checkpoint."""

    def __init__(self, config) -> None:
        heads = config.num_heads
        state = config.state_size
        groups = config.n_groups
        assert config.head_dim == MLEN
        self.shape = Mamba2Shape(
            hidden_size=config.hidden_size,
            num_heads=heads,
            head_dim=config.head_dim,
            state_size=state,
            n_groups=groups,
            conv_kernel=config.conv_kernel,
            chunk_size=MLEN,
            seq_len=1,
        )
        self.compiler = PlenaCompiler(mlen=MLEN, blen=4)
        p = self.compiler
        up = lambda n: ((n + MLEN - 1) // MLEN) * MLEN  # noqa: E731
        self.state = p.alloc("e2e_state", up(heads * state), MLEN)
        self.x = p.alloc("e2e_x", up(heads), MLEN)
        self.y = p.alloc("e2e_y", up(heads), MLEN)
        self.scratch = p.alloc("e2e_scratch", MLEN, MLEN)
        self.fp = {
            "b": p.fp_var("e2e_b", size=groups * state),
            "c": p.fp_var("e2e_c", size=groups * state),
            "da": p.fp_var("e2e_da", size=heads),
            "dt": p.fp_var("e2e_dt", size=heads),
            "d": p.fp_var("e2e_d", size=heads),
        }
        self.consts = p.mamba_fp_constants()
        mark = len(p.get_code())
        p.ssm_decode_step_v0(
            state=self.state,
            x=self.x,
            b_fp=self.fp["b"],
            c_fp=self.fp["c"],
            da_fp=self.fp["da"],
            dt_fp=self.fp["dt"],
            d_fp=self.fp["d"],
            y=self.y,
            scratch=self.scratch,
            shape=self.shape,
            consts=self.consts,
        )
        self.code = p.get_code()[mark:]
        self.machine = Machine(vlen=MLEN, vram_words=1 << 20, fpram_words=1 << 14)

    def _base(self, value) -> int:
        return self.compiler.get_vram_layout(value.name).vram_base_addr

    def run(self, *, x, b, c, da, dt, d, state):
        heads = self.shape.num_heads
        state_size = self.shape.state_size
        head_dim = self.shape.head_dim
        x_heads = x.view(heads, head_dim)
        for head in range(heads):
            self.machine.write_vram_row(self._base(self.x) + head * MLEN, x_heads[head].tolist())
            self.machine.write_vram_row(self._base(self.y) + head * MLEN, [7.5] * MLEN)
            for state_index in range(state_size):
                self.machine.write_vram_row(
                    self._base(self.state) + (head * state_size + state_index) * MLEN,
                    state[head, :, state_index].tolist(),
                )
        for name, value in (("b", b), ("c", c), ("da", da), ("dt", dt), ("d", d)):
            self.machine.write_fpram(self.fp[name].address, value.flatten().tolist())
        self.machine.write_fpram(
            self.consts.zero.address,
            self.compiler.mamba_fp_constant_values(self.shape),
        )
        self.machine.run(self.code)
        y = torch.tensor(
            [
                self.machine.read_vram_row(self._base(self.y) + head * MLEN, head_dim)
                for head in range(heads)
            ]
        )
        new_state = torch.tensor(
            [
                [
                    self.machine.read_vram_row(
                        self._base(self.state) + (head * state_size + state_index) * MLEN,
                        head_dim,
                    )
                    for state_index in range(state_size)
                ]
                for head in range(heads)
            ]
        ).permute(0, 2, 1)
        return y, new_state


def _load_real_model():
    try:
        from transformers import Mamba2ForCausalLM
    except ImportError:  # pragma: no cover - depends on the environment
        pytest.skip("transformers with Mamba2ForCausalLM is not installed")
    return Mamba2ForCausalLM.from_pretrained(
        _checkpoint_path(),
        dtype=torch.float32,
        local_files_only=True,
    ).eval()


def _decode_with_isa_recurrence(model, runtime, token, ssm_states, conv_states):
    """Execute one model step with only the recurrent core on PLENA ISA.

    Existing Matrix projections, convolution, dt activation, normalization and
    residual arithmetic deliberately remain in PyTorch. This is a real-weight
    recurrent-chain gate, not a claim that the complete model runs on the ISA
    interpreter.
    """
    config = model.config
    heads = config.num_heads
    state_size = config.state_size
    groups = config.n_groups
    inner = heads * config.head_dim
    hidden = model.backbone.embeddings(token.view(1, 1))[0, 0]
    for layer_index, layer in enumerate(model.backbone.layers):
        mixer = layer.mixer
        projected = layer.norm(hidden) @ mixer.in_proj.weight.T
        z, xbc, dt_raw = projected.split(
            [inner, inner + 2 * groups * state_size, heads], dim=-1
        )
        window = torch.cat([conv_states[layer_index][:, 1:], xbc[:, None]], dim=-1)
        convolved = (
            (window * mixer.conv1d.weight[:, 0, :]).sum(-1) + mixer.conv1d.bias
        )
        x, b, c = torch.nn.functional.silu(convolved).split(
            [inner, groups * state_size, groups * state_size], dim=-1
        )
        dt = torch.nn.functional.softplus(dt_raw + mixer.dt_bias)
        da = torch.exp(dt * -torch.exp(mixer.A_log.float()))
        y, ssm_states[layer_index] = runtime.run(
            x=x,
            b=b,
            c=c,
            da=da,
            dt=dt,
            d=mixer.D,
            state=ssm_states[layer_index],
        )
        conv_states[layer_index] = window
        gated = mixer.norm(y.reshape(1, inner), z.reshape(1, inner))
        hidden = (hidden + gated @ mixer.out_proj.weight.T).reshape(-1)
    return model.lm_head(model.backbone.norm_f(hidden))


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


def test_real_checkpoint_24_layer_multi_token_recurrence_chain():
    """Carry PLENA-written recurrent state through 24 layers and multiple tokens.

    The reference and ISA-assisted paths start from independent copies of the
    same prefill cache. Every subsequent ISA-assisted token consumes the state
    written by its own previous token, so a systematic update error compounds
    instead of being hidden by re-seeding from the reference.
    """
    torch.set_grad_enabled(False)
    model = _load_real_model()
    config = model.config
    assert config.num_hidden_layers == 24

    prompt = torch.tensor([[1, 17, 42, 9]], dtype=torch.long)
    with torch.inference_mode():
        prefill = model(prompt, use_cache=True)
    reference_cache = prefill.cache_params
    isa_ssm = [
        reference_cache.ssm_states[index][0].clone()
        for index in range(config.num_hidden_layers)
    ]
    isa_conv = [
        reference_cache.conv_states[index][0].clone()
        for index in range(config.num_hidden_layers)
    ]
    runtime = _RealRecurrenceRuntime(config)
    decode_tokens = int(os.environ.get("PLENA_REAL_CHECKPOINT_TOKENS", "4"))
    assert decode_tokens > 0
    reference_token = prefill.logits[0, -1].argmax().view(1)
    isa_token = reference_token.clone()
    worst_logit_error = 0.0
    worst_state_error = 0.0

    for step in range(decode_tokens):
        with torch.inference_mode():
            reference_logits = model(
                reference_token.view(1, 1),
                cache_params=reference_cache,
                use_cache=True,
                cache_position=torch.tensor([prompt.shape[1] + step]),
            ).logits[0, -1]
            isa_logits = _decode_with_isa_recurrence(
                model,
                runtime,
                isa_token,
                isa_ssm,
                isa_conv,
            )

        worst_logit_error = max(
            worst_logit_error,
            (isa_logits - reference_logits).abs().max().item(),
        )
        for layer_index in range(config.num_hidden_layers):
            worst_state_error = max(
                worst_state_error,
                (
                    isa_ssm[layer_index]
                    - reference_cache.ssm_states[layer_index][0]
                )
                .abs()
                .max()
                .item(),
            )
        assert isa_logits.argmax() == reference_logits.argmax()
        assert torch.equal(isa_logits.topk(5).indices, reference_logits.topk(5).indices)
        reference_token = reference_logits.argmax().view(1)
        isa_token = isa_logits.argmax().view(1)

    assert runtime.machine.executed > 4 * config.num_hidden_layers
    assert worst_state_error < 1e-4
    assert worst_logit_error < 5e-3
    print(
        json.dumps(
            {
                "model": REPO,
                "layers": config.num_hidden_layers,
                "decode_tokens": decode_tokens,
                "plena_recurrent_dynamic_instructions": runtime.machine.executed,
                "worst_state_max_abs_error": worst_state_error,
                "worst_logit_max_abs_error": worst_logit_error,
                "top1_agreement": f"{decode_tokens}/{decode_tokens}",
                "top5_agreement": f"{decode_tokens}/{decode_tokens}",
                "claim_boundary": (
                    "only the recurrent core executes in the PLENA ISA interpreter; "
                    "surrounding stages execute in PyTorch"
                ),
            },
            sort_keys=True,
        )
    )


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
