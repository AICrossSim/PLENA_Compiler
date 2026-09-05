"""KDA's causal conv step, executed against the CPU reference.

Same approach as `test_kda_decode_step.py`: the emitted assembly runs on
`aten/tests/isa_interpreter.py`, so these compare numbers rather than asserting
which opcodes appear.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.kda import reference as kda_ref  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.program_kda_common import (  # noqa: E402
    kda_conv_blocks,
    kda_conv_state_row,
)
from compiler.aten.tests.isa_interpreter import Machine  # noqa: E402

MLEN = 8
#: exp() saturates well before this; the reference's silu does not, so keeping
#: inputs modest is about comparing the same function, not about hiding a bug.
SCALE = 0.6


def _rows_up(n: int) -> int:
    return ((n + MLEN - 1) // MLEN) * MLEN


class _ConvHarness:
    def __init__(self, channels: int, kernel: int):
        self.channels, self.kernel = channels, kernel
        self.blocks = kda_conv_blocks(channels, MLEN)
        p = self.prog = PlenaCompiler(mlen=MLEN, blen=2)

        rows = _rows_up(self.blocks * kernel)
        self.conv_state = p.alloc("conv_state", rows, MLEN)
        self.weight = p.alloc("weight", rows, MLEN)
        blk = _rows_up(self.blocks)
        self.bias = p.alloc("bias", blk, MLEN)
        self.x_new = p.alloc("x_new", blk, MLEN)
        self.out = p.alloc("out", blk, MLEN)
        self.scratch = p.alloc("scratch", MLEN, MLEN)
        self.consts = p.kda_fp_constants()
        self.mark = len(p.get_code())

    def _base(self, var) -> int:
        return self.prog.get_vram_layout(var.name).vram_base_addr

    def emit(self, *, use_bias: bool = True, apply_silu: bool = True) -> str:
        self.prog.kda_conv_step_v0(
            x_new=self.x_new, conv_state=self.conv_state, weight=self.weight,
            bias=self.bias if use_bias else None, out=self.out, scratch=self.scratch,
            consts=self.consts, channels=self.channels, kernel=self.kernel,
            apply_silu=apply_silu,
        )
        return self.prog.get_code()[self.mark :]

    def _wide(self, m, tile, vec) -> None:
        """Scatter a channels-wide vector across its column blocks."""
        for cb in range(self.blocks):
            m.write_vram_row(
                self._base(tile) + cb * MLEN, vec[cb * MLEN : (cb + 1) * MLEN].tolist()
            )

    def run(self, *, state, weight, bias, x_new, **kw):
        """state [channels, kernel]; weight [channels, kernel]; bias/x_new [channels]."""
        code = self.emit(**kw)
        m = Machine(vlen=MLEN)
        for cb in range(self.blocks):
            lanes = slice(cb * MLEN, (cb + 1) * MLEN)
            for t in range(self.kernel):
                row = kda_conv_state_row(self.channels, MLEN, self.kernel, cb, t)
                m.write_vram_row(
                    self._base(self.conv_state) + row * MLEN, state[lanes, t].tolist()
                )
                m.write_vram_row(
                    self._base(self.weight) + row * MLEN, weight[lanes, t].tolist()
                )
        self._wide(m, self.bias, bias)
        self._wide(m, self.x_new, x_new)
        # `out` is a persistent tile: on token n+1 it holds token n's
        # activations, and q/k/v reuse it. Seeding it with garbage makes the
        # emitter's fill-zero load-bearing -- a fresh Machine zeroes VRAM, so
        # without this an accumulator that is never cleared is indistinguishable
        # from one that happens to start at zero.
        for cb in range(self.out.shape[0]):
            m.write_vram_row(self._base(self.out) + cb * MLEN, [7.5] * MLEN)
        m.write_fpram(
            self.consts.zero.address, PlenaCompiler.kda_fp_constant_values()
        )
        m.run(code)

        out = torch.tensor(
            [
                x
                for cb in range(self.blocks)
                for x in m.read_vram_row(self._base(self.out) + cb * MLEN, MLEN)
            ],
            dtype=torch.float32,
        )
        new_state = torch.tensor(
            [
                [
                    m.read_vram_row(
                        self._base(self.conv_state)
                        + kda_conv_state_row(self.channels, MLEN, self.kernel, cb, t) * MLEN,
                        MLEN,
                    )
                    for t in range(self.kernel)
                ]
                for cb in range(self.blocks)
            ],
            dtype=torch.float32,
        )  # [blocks, kernel, mlen]
        new_state = new_state.permute(0, 2, 1).reshape(self.channels, self.kernel)
        return out, new_state, m


def _case(seed: int, channels: int, kernel: int):
    torch.manual_seed(seed)
    state = SCALE * torch.randn(channels, kernel)
    weight = SCALE * torch.randn(channels, kernel)
    bias = SCALE * torch.randn(channels)
    x_new = SCALE * torch.randn(channels)
    out, new = kda_ref._causal_conv_step(
        x_new[None, :], state[None, :, :], weight, bias
    )
    return dict(
        state=state, weight=weight, bias=bias, x_new=x_new,
        expected_out=out[0], expected_state=new[0],
    )


@pytest.mark.parametrize(
    "seed,channels,kernel",
    [
        (1, MLEN, 4),        # one channel block, the Kimi kernel width
        (2, MLEN * 2, 4),    # two blocks: the case Kimi actually needs
        (3, MLEN * 3, 2),    # three blocks, short kernel
        (4, MLEN, 1),        # kernel 1: no history, pure pointwise
    ],
)
def test_conv_step_matches_the_reference(seed, channels, kernel):
    c = _case(seed, channels, kernel)
    h = _ConvHarness(channels, kernel)
    out, new_state, _ = h.run(
        state=c["state"], weight=c["weight"], bias=c["bias"], x_new=c["x_new"]
    )
    torch.testing.assert_close(out, c["expected_out"], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(new_state, c["expected_state"], rtol=1e-5, atol=1e-6)


def test_history_shifts_rather_than_being_overwritten():
    """The oldest tap must fall off and the rest move down by one. A roll in the
    wrong direction, or an append that clobbers a live tap, still produces
    finite output of the right shape."""
    kernel = 4
    c = _case(7, MLEN, kernel)
    h = _ConvHarness(MLEN, kernel)
    _, new_state, _ = h.run(
        state=c["state"], weight=c["weight"], bias=c["bias"], x_new=c["x_new"]
    )
    # taps 0..k-2 of the new state are taps 1..k-1 of the old
    torch.testing.assert_close(new_state[:, :-1], c["state"][:, 1:], rtol=0, atol=0)
    # the newest tap is this token
    torch.testing.assert_close(new_state[:, -1], c["x_new"], rtol=1e-6, atol=1e-7)


def test_bias_is_optional_and_actually_applied():
    c = _case(8, MLEN, 4)
    with_bias, _, _ = _ConvHarness(MLEN, 4).run(
        state=c["state"], weight=c["weight"], bias=c["bias"], x_new=c["x_new"]
    )
    without, _, _ = _ConvHarness(MLEN, 4).run(
        state=c["state"], weight=c["weight"], bias=c["bias"], x_new=c["x_new"],
        use_bias=False,
    )
    assert not torch.allclose(with_bias, without, rtol=1e-3, atol=1e-3)

    ref_no_bias, _ = kda_ref._causal_conv_step(
        c["x_new"][None, :], c["state"][None, :, :], c["weight"], None
    )
    torch.testing.assert_close(without, ref_no_bias[0], rtol=2e-5, atol=2e-6)


def test_channel_blocks_do_not_leak_into_each_other():
    """Every channel block has its own history and its own taps."""
    channels, kernel = MLEN * 3, 3
    c = _case(9, channels, kernel)
    h = _ConvHarness(channels, kernel)
    out, _, _ = h.run(
        state=c["state"], weight=c["weight"], bias=c["bias"], x_new=c["x_new"]
    )
    torch.testing.assert_close(out, c["expected_out"], rtol=2e-5, atol=2e-6)

    # perturb only block 1's input; blocks 0 and 2 must not move
    perturbed = c["x_new"].clone()
    perturbed[MLEN : 2 * MLEN] += 1.0
    out2, _, _ = _ConvHarness(channels, kernel).run(
        state=c["state"], weight=c["weight"], bias=c["bias"], x_new=perturbed
    )
    torch.testing.assert_close(out2[:MLEN], out[:MLEN], rtol=0, atol=0)
    torch.testing.assert_close(out2[2 * MLEN :], out[2 * MLEN :], rtol=0, atol=0)
    assert not torch.allclose(out2[MLEN : 2 * MLEN], out[MLEN : 2 * MLEN])


def test_rejects_shapes_it_would_get_wrong():
    p = PlenaCompiler(mlen=MLEN, blen=2)
    consts = p.kda_fp_constants()
    ok = dict(
        conv_state=p.alloc("cs", MLEN, MLEN), weight=p.alloc("w", MLEN, MLEN),
        bias=None, out=p.alloc("o", MLEN, MLEN), scratch=p.alloc("sc", MLEN, MLEN),
        consts=consts, channels=MLEN, kernel=4,
    )
    with pytest.raises(ValueError, match="multiple of mlen"):
        p.kda_conv_step_v0(x_new=p.alloc("x1", MLEN, MLEN), **{**ok, "channels": MLEN + 3})
    with pytest.raises(ValueError, match="exactly mlen"):
        p.kda_conv_step_v0(x_new=p.alloc("x2", MLEN, MLEN * 2), **ok)
    with pytest.raises(ValueError, match="conv_state needs"):
        p.kda_conv_step_v0(
            x_new=p.alloc("x3", MLEN, MLEN), **{**ok, "channels": MLEN * 4, "kernel": 4}
        )
    # scratch aliased onto ANY operand: mamba_row_copy is zero-then-add, so it
    # wipes whatever it aliases mid-loop.
    for victim in ("conv_state", "weight", "out"):
        with pytest.raises(ValueError, match="distinct"):
            p.kda_conv_step_v0(
                x_new=p.alloc(f"xa_{victim}", MLEN, MLEN),
                **{**ok, "scratch": ok[victim]},
            )
    biased = {**ok, "bias": p.alloc("b1", MLEN, MLEN)}
    with pytest.raises(ValueError, match="distinct"):
        p.kda_conv_step_v0(
            x_new=p.alloc("xb", MLEN, MLEN),
            **{**biased, "scratch": biased["bias"]},
        )
    x_alias = p.alloc("xc", MLEN, MLEN)
    with pytest.raises(ValueError, match="distinct"):
        p.kda_conv_step_v0(x_new=x_alias, **{**ok, "scratch": x_alias})
