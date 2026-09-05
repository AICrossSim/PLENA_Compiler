"""The whole KDA mixer for one decode token, against `kda_state_engine_step`.

This is the first test that runs conv, gates and recurrence together, so it is
also the first that can catch a mismatch *between* them -- a layout convention
that each side reads consistently on its own but differently from the other.

The mixer streams per head because FPRAM is 512 slots and materialising every
head's decay / q_hat / k_hat costs 3 * num_heads * key_dim. These tests check
the streamed result equals the reference, which computes all heads at once.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.kda.reference import (  # noqa: E402
    KdaConvWeights,
    KdaState,
    KdaRecurrentState,
    kda_state_engine_step,
)
from compiler.aten.models.kda.shape import KdaShape  # noqa: E402
from compiler.aten.models.kda.state_precision import StateStorage  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.program_kda_common import (  # noqa: E402
    kda_blocks,
    kda_state_row,
    kda_state_rows,
    kda_vector_row,
    kda_vector_rows,
)
from compiler.aten.plena.program_kda_gates import (  # noqa: E402
    kda_head_blocks,
    kda_key_blocks,
    kda_key_row,
)
from compiler.aten.plena.program_kda_mixer import KdaMixerBuffers  # noqa: E402
from compiler.aten.tests.isa_interpreter import Machine  # noqa: E402

MLEN = 8


def _rows_up(n: int) -> int:
    return ((n + MLEN - 1) // MLEN) * MLEN


class _MixerHarness:
    """Post-conv activations in, mixer output and updated state out.

    The convolutions are covered by test_kda_conv.py; feeding their outputs in
    directly keeps this test on the seam between the gate and recurrence
    layouts, which is what has not been exercised before.
    """

    def __init__(self, shape: KdaShape):
        self.shape = shape
        p = self.prog = PlenaCompiler(mlen=MLEN, blen=2)
        kb = kda_key_blocks(shape, MLEN)
        vb = kda_blocks(shape, MLEN)

        key_rows = _rows_up(shape.num_heads * kb)
        vec_rows = _rows_up(kda_vector_rows(shape, MLEN))
        self.q = p.alloc("q", key_rows, MLEN)
        self.k = p.alloc("k", key_rows, MLEN)
        self.gate = p.alloc("gate", key_rows, MLEN)
        self.dt_bias = p.alloc("dt_bias", key_rows, MLEN)
        self.v = p.alloc("v", vec_rows, MLEN)
        self.out = p.alloc("out", vec_rows, MLEN)
        self.pred = p.alloc("pred", vec_rows, MLEN)
        self.err = p.alloc("err", vec_rows, MLEN)
        self.state = p.alloc("state", _rows_up(kda_state_rows(shape, MLEN)), MLEN)
        self.beta_logit = p.alloc(
            "beta_logit", _rows_up(kda_head_blocks(shape, MLEN)), MLEN
        )
        self.sq_scratch = p.alloc("sq_scratch", key_rows, MLEN)

        # The per-head window, reused every head. decay and q_hat are one
        # allocation because they are never live at the same time -- that is
        # what brings Kimi K3 from 620 slots to 492, under FP_SRAM_DEPTH.
        self.k_hat = p.fp_var("k_hat", size=shape.key_dim)
        self.decay = p.fp_var("decay_and_q_hat", size=shape.key_dim)
        self.q_hat = self.decay
        self.beta = p.fp_var("beta", size=kda_head_blocks(shape, MLEN) * MLEN)
        self.part = p.fp_var("part", size=kb)
        self.acc = p.fp_var("acc", size=1)
        self.scale = p.fp_var("scale", size=1)
        self.rate = p.fp_var("rate", size=shape.num_heads)
        self.lower = p.fp_var("lower", size=1)
        self.consts = p.kda_fp_constants()
        self.vb, self.kb = vb, kb
        self.mark = len(p.get_code())

    def fp_vars(self):
        """Every FPRAM allocation this harness makes, for the budget test."""
        return [
            self.k_hat, self.decay, self.beta, self.part, self.acc, self.scale,
            self.rate, self.lower, *self.consts.as_list(),
        ]

    def _base(self, var) -> int:
        return self.prog.get_vram_layout(var.name).vram_base_addr

    def buffers(self) -> KdaMixerBuffers:
        return KdaMixerBuffers(
            q=self.q, k=self.k, v=self.v, gate=self.gate, dt_bias=self.dt_bias,
            beta_logit=self.beta_logit, state=self.state, out=self.out,
            pred=self.pred, err=self.err, sq_scratch=self.sq_scratch, decay_fp=self.decay, q_hat_fp=self.q_hat,
            k_hat_fp=self.k_hat, beta_fp=self.beta, part_fp=self.part,
            acc_fp=self.acc, output_scale_fp=self.scale, rate_fp=self.rate,
            lower_bound_fp=self.lower, consts=self.consts,
        )

    def _seed_key_tile(self, m, tile, values) -> None:
        """values is [heads, key_dim]."""
        s = self.shape
        for h in range(s.num_heads):
            for b in range(self.kb):
                m.write_vram_row(
                    self._base(tile) + kda_key_row(s, MLEN, h, b) * MLEN,
                    values[h, b * MLEN : (b + 1) * MLEN].tolist(),
                )

    def _seed_value_tile(self, m, tile, values) -> None:
        """values is [heads, value_dim]."""
        s = self.shape
        for h in range(s.num_heads):
            for c in range(self.vb):
                m.write_vram_row(
                    self._base(tile) + kda_vector_row(s, MLEN, h, c) * MLEN,
                    values[h, c * MLEN : (c + 1) * MLEN].tolist(),
                )

    def run(self, *, q, k, v, gate, dt_bias, beta_logit, a_log, state, output_scale,
            per_head=False, head_rows_override=None):
        """``per_head`` lowers one head per ``kda_mixer_step_v0`` call.

        That is the shape a whole-layer loop takes, and it must give bit-for-bit
        the same program as one call over every head -- which it only does
        because beta is produced once, outside. It used to be folded into the
        mixer, and the second call then computed ``sigmoid(sigmoid(beta))``.
        """
        s = self.shape
        b = self.buffers()
        self.prog.kda_beta_scalars_v0(
            beta_logit=b.beta_logit, beta_fp=b.beta_fp, consts=b.consts, shape=s
        )
        if head_rows_override is not None:
            self.prog.kda_mixer_step_v0(
                buffers=b, shape=s, head_rows=head_rows_override
            )
        elif per_head:
            for h in range(s.num_heads):
                self.prog.kda_mixer_step_v0(buffers=b, shape=s, head_rows=[h])
        else:
            self.prog.kda_mixer_step_v0(buffers=b, shape=s)
        code = self.prog.get_code()[self.mark :]

        m = Machine(vlen=MLEN, vram_words=1 << 16, fpram_words=1 << 13)
        self._seed_key_tile(m, self.q, q)
        self._seed_key_tile(m, self.k, k)
        self._seed_key_tile(m, self.gate, gate)
        self._seed_key_tile(m, self.dt_bias, dt_bias)
        self._seed_value_tile(m, self.v, v)
        for h in range(s.num_heads):
            for c in range(self.vb):
                for j in range(s.key_dim):
                    m.write_vram_row(
                        self._base(self.state) + kda_state_row(s, MLEN, h, c, j) * MLEN,
                        state[h, j, c * MLEN : (c + 1) * MLEN].tolist(),
                    )
        for hb in range(kda_head_blocks(s, MLEN)):
            chunk = beta_logit[hb * MLEN : (hb + 1) * MLEN].tolist()
            m.write_vram_row(
                self._base(self.beta_logit) + hb * MLEN,
                chunk + [0.0] * (MLEN - len(chunk)),
            )

        m.write_fpram(self.consts.zero.address, PlenaCompiler.kda_fp_constant_values())
        m.write_fpram(self.rate.address, torch.exp(a_log).tolist())
        m.write_fpram(self.lower.address, [s.gate_lower_bound])
        m.write_fpram(self.scale.address, [output_scale])
        m.run(code)

        out = torch.tensor(
            [
                [
                    val
                    for c in range(self.vb)
                    for val in m.read_vram_row(
                        self._base(self.out) + kda_vector_row(s, MLEN, h, c) * MLEN, MLEN
                    )
                ]
                for h in range(s.num_heads)
            ],
            dtype=torch.float32,
        )
        new_state = torch.tensor(
            [
                [
                    [
                        val
                        for c in range(self.vb)
                        for val in m.read_vram_row(
                            self._base(self.state)
                            + kda_state_row(s, MLEN, h, c, j) * MLEN,
                            MLEN,
                        )
                    ]
                    for j in range(s.key_dim)
                ]
                for h in range(s.num_heads)
            ],
            dtype=torch.float32,
        )
        return out, new_state, m


def _case(seed: int, heads: int, key_dim: int, value_dim: int):
    """Post-conv activations plus the reference result for one token."""
    torch.manual_seed(seed)
    shape = KdaShape(
        hidden_size=heads * value_dim, num_heads=heads, key_dim=key_dim,
        value_dim=value_dim, conv_kernel=4,
    )
    q = 0.5 * torch.randn(heads, key_dim)
    k = 0.5 * torch.randn(heads, key_dim)
    v = torch.randn(heads, value_dim)
    gate = 0.8 * torch.randn(heads, key_dim)
    dt_bias = 0.8 * torch.randn(heads, key_dim)
    beta_logit = torch.randn(heads)
    a_log = 0.5 * torch.randn(heads)
    state = torch.randn(heads, value_dim, key_dim)

    # kda_step is the post-conv half of kda_state_engine_step.
    from compiler.aten.models.kda.reference import kda_step

    out, new = kda_step(
        q[None], k[None], v[None], gate[None], beta_logit[None],
        KdaState(state[None].clone()), a_log, dt_bias, shape,
    )
    output_scale = 1.0 / (key_dim**0.5)
    return shape, dict(
        q=q, k=k, v=v, gate=gate, dt_bias=dt_bias, beta_logit=beta_logit,
        a_log=a_log, output_scale=output_scale,
        state_in_T=state.transpose(-2, -1).contiguous(),   # [heads, key, value]
        expected_out=out[0],
        expected_state_T=new.recurrent[0].transpose(-2, -1).contiguous(),
    )


@pytest.mark.parametrize(
    "seed,heads,key_dim,value_dim",
    [
        (1, 1, MLEN, MLEN),          # smallest complete mixer
        (2, 3, MLEN, MLEN),          # several heads streaming
        (3, 2, MLEN * 2, MLEN),      # key spans two blocks
        (4, 2, MLEN, MLEN * 2),      # value spans two blocks
        (5, 3, MLEN * 2, MLEN * 2),  # both -- Kimi K3's shape
    ],
)
def test_mixer_matches_the_reference(seed, heads, key_dim, value_dim):
    shape, c = _case(seed, heads, key_dim, value_dim)
    out, new_state, _ = _MixerHarness(shape).run(
        q=c["q"], k=c["k"], v=c["v"], gate=c["gate"], dt_bias=c["dt_bias"],
        beta_logit=c["beta_logit"], a_log=c["a_log"], state=c["state_in_T"],
        output_scale=c["output_scale"],
    )
    torch.testing.assert_close(out, c["expected_out"], rtol=5e-5, atol=5e-6)
    torch.testing.assert_close(new_state, c["expected_state_T"], rtol=5e-5, atol=5e-6)


def test_the_scalar_window_really_is_reused():
    """The whole point of streaming. If the emitter had kept the all-at-once
    layout it would index decay_fp[h * key_dim], which for h > 0 runs past a
    key_dim-sized window -- so this shape passing at all is the evidence."""
    shape, c = _case(6, 4, MLEN * 2, MLEN)
    h = _MixerHarness(shape)
    assert h.decay.size == shape.key_dim
    assert h.q_hat.size == shape.key_dim
    items = h.prog.kda_mixer_fpram_slots(shape)
    assert items["k_hat"] == items["decay_or_q_hat"] == shape.key_dim

    out, _, _ = h.run(
        q=c["q"], k=c["k"], v=c["v"], gate=c["gate"], dt_bias=c["dt_bias"],
        beta_logit=c["beta_logit"], a_log=c["a_log"], state=c["state_in_T"],
        output_scale=c["output_scale"],
    )
    torch.testing.assert_close(out, c["expected_out"], rtol=5e-5, atol=5e-6)


def test_fpram_stays_within_the_hardware_file():
    """FP_SRAM_DEPTH is 512, and the whole allocation has to fit -- not just
    the per-head window.

    This test asserted `3 * key_dim + 1 <= 512` and passed while the real
    footprint was 620. Two things were missing: beta costs a padded row per
    S_MAP_FP_V (128 at Kimi K3, not 1) and rate is indexed by head number
    (96). Counting those, `3 * key_dim + beta` alone is 512 -- so the old
    layout could not fit however the rest was arranged.
    """
    kimi = KdaShape(
        hidden_size=7168, num_heads=96, key_dim=128, value_dim=128, conv_kernel=4
    )
    p = PlenaCompiler(mlen=64, blen=4)
    items = p.kda_mixer_fpram_slots(kimi)

    assert items["beta"] == 128, "S_MAP_FP_V writes whole rows, not one slot"
    assert items["rate"] == 96, "rate_fp is indexed by head number"
    assert items["total"] == 492
    assert items["total"] <= 512, (
        f"the mixer needs {items['total']} FPRAM slots against FP_SRAM_DEPTH 512: "
        f"{items}"
    )

    # The layout this replaced, so a regression to it is loud rather than quiet.
    unshared = items["total"] + kimi.key_dim
    assert unshared > 512, (
        "a separate decay and q_hat window is what overflowed; if this ever fits, "
        "sharing them is no longer load-bearing"
    )
    all_at_once = 3 * kimi.num_heads * kimi.key_dim + kimi.num_heads
    assert all_at_once > 512 * 70, (
        "if this ever fits, the streaming was unnecessary -- recheck the shape"
    )


def test_fpram_accounting_matches_what_is_actually_allocated():
    """The itemised count must equal the allocator's high-water mark.

    Otherwise it is a comment, not a budget: the compiler's FPRAMAllocator
    defaults to 1024 slots, so nothing else in the stack would catch a mixer
    that overflows the real 512.
    """
    shape, _ = _case(11, 3, MLEN, MLEN)
    h = _MixerHarness(shape)
    items = h.prog.kda_mixer_fpram_slots(shape)

    used = [(v.address, v.size) for v in h.fp_vars()]
    high_water = max(a + n for a, n in used)
    assert high_water <= items["total"], (
        f"the harness allocates {high_water} slots but the budget claims "
        f"{items['total']}: {items}"
    )


def test_decay_and_q_hat_may_share_one_window():
    """The two are never live at once, and the mixer must still be correct when
    a caller exploits that -- which is the only reason Kimi K3 fits."""
    shape, c = _case(12, 2, MLEN * 2, MLEN)
    h = _MixerHarness(shape)
    assert h.q_hat is h.decay, "the harness is meant to exercise the shared window"
    out, state_out, _ = h.run(
        q=c["q"], k=c["k"], v=c["v"], gate=c["gate"], dt_bias=c["dt_bias"],
        beta_logit=c["beta_logit"], a_log=c["a_log"], state=c["state_in_T"],
        output_scale=c["output_scale"],
    )
    torch.testing.assert_close(out, c["expected_out"], rtol=5e-5, atol=5e-6)
    torch.testing.assert_close(state_out, c["expected_state_T"], rtol=5e-5, atol=5e-6)


def test_heads_can_be_lowered_individually():
    """head_rows selects a subset, which is what a per-head outer loop uses.

    Three calls of one head each must equal one call of three heads. This
    failed before beta moved out of the mixer: head 0 was right and heads 1
    and 2 were off by 1.2e-2, because each call re-applied sigmoid to
    beta_logit in place.
    """
    shape, c = _case(8, 3, MLEN, MLEN)
    args = dict(
        q=c["q"], k=c["k"], v=c["v"], gate=c["gate"], dt_bias=c["dt_bias"],
        beta_logit=c["beta_logit"], a_log=c["a_log"], state=c["state_in_T"],
        output_scale=c["output_scale"],
    )
    split, state_split, _ = _MixerHarness(shape).run(per_head=True, **args)
    torch.testing.assert_close(split, c["expected_out"], rtol=5e-5, atol=5e-6)
    torch.testing.assert_close(state_split, c["expected_state_T"], rtol=5e-5, atol=5e-6)

    together, state_together, _ = _MixerHarness(shape).run(**args)
    torch.testing.assert_close(split, together, rtol=0, atol=0)
    torch.testing.assert_close(state_split, state_together, rtol=0, atol=0)


def test_head_rows_actually_restricts_the_work():
    """Lowering head 0 alone must leave the other heads' state untouched.

    Without this, an implementation that ignores head_rows and does every head
    passes every other test in this file.
    """
    shape, c = _case(9, 3, MLEN, MLEN)
    h = _MixerHarness(shape)
    _, state_out, _ = h.run(head_rows_override=[0], **dict(
        q=c["q"], k=c["k"], v=c["v"], gate=c["gate"], dt_bias=c["dt_bias"],
        beta_logit=c["beta_logit"], a_log=c["a_log"], state=c["state_in_T"],
        output_scale=c["output_scale"],
    ))
    torch.testing.assert_close(
        state_out[0], c["expected_state_T"][0], rtol=5e-5, atol=5e-6
    )
    for head in range(1, shape.num_heads):
        torch.testing.assert_close(
            state_out[head], c["state_in_T"][head], rtol=0, atol=0
        )


# ---------------------------------------------------------------------------
# The conv/mixer seam.
#
# kda_conv_step_v0 writes its output blocked by *channel*: block cb covers
# channels [cb*mlen, (cb+1)*mlen). kda_mixer_step_v0 reads q and k blocked by
# *key*: kda_key_row(h, b) = h*key_blocks + b. Those are two different helpers
# in two different modules, and the mixer feeds directly from the conv, so if
# they disagree every KDA result is wrong in a way neither module's own tests
# can see.
# ---------------------------------------------------------------------------


def test_conv_channel_blocks_and_mixer_key_blocks_are_the_same_rows():
    """For a key-width tensor the channel index is h*key_dim + k, so channel
    block cb is head cb//key_blocks's key block cb%key_blocks -- which is
    exactly kda_key_row. This holds because key_dim is required to be a whole
    number of blocks; it would not if a head's keys straddled a block."""
    from compiler.aten.plena.program_kda_common import kda_conv_blocks

    for heads, key_dim in [(1, MLEN), (3, MLEN), (2, MLEN * 2), (5, MLEN * 3)]:
        shape = KdaShape(
            hidden_size=heads * MLEN, num_heads=heads, key_dim=key_dim,
            value_dim=MLEN, conv_kernel=4,
        )
        channels = heads * key_dim
        key_blocks = kda_key_blocks(shape, MLEN)
        assert kda_conv_blocks(channels, MLEN) == heads * key_blocks

        for h in range(heads):
            for b in range(key_blocks):
                # the channel block holding head h's key block b
                first_channel = h * key_dim + b * MLEN
                channel_block = first_channel // MLEN
                assert channel_block == kda_key_row(shape, MLEN, h, b), (
                    f"conv writes head {h} key block {b} to row {channel_block} "
                    f"but the mixer reads it from {kda_key_row(shape, MLEN, h, b)}"
                )


def test_conv_channel_blocks_and_mixer_value_blocks_are_the_same_rows():
    """Same seam for v, which is value-width: channel index is h*value_dim + j,
    so channel block cb is kda_vector_row(h, cb%value_blocks)."""
    from compiler.aten.plena.program_kda_common import kda_conv_blocks

    for heads, value_dim in [(1, MLEN), (3, MLEN), (2, MLEN * 2), (4, MLEN * 3)]:
        shape = KdaShape(
            hidden_size=heads * value_dim, num_heads=heads, key_dim=MLEN,
            value_dim=value_dim, conv_kernel=4,
        )
        channels = heads * value_dim
        value_blocks = kda_blocks(shape, MLEN)
        assert kda_conv_blocks(channels, MLEN) == heads * value_blocks

        for h in range(heads):
            for c in range(value_blocks):
                channel_block = (h * value_dim + c * MLEN) // MLEN
                assert channel_block == kda_vector_row(shape, MLEN, h, c), (
                    f"conv writes head {h} value block {c} to row {channel_block} "
                    f"but the mixer reads it from {kda_vector_row(shape, MLEN, h, c)}"
                )
