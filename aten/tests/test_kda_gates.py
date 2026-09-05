"""KDA's decay and beta scalars, executed against the CPU reference.

`activate_log_decay` is the oracle for decay; `beta = sigmoid(beta_logit)` comes
from `kda_step`. Both end up in FPRAM, so these tests read the FP memory back
rather than a VRAM tile.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.kda.reference import activate_log_decay  # noqa: E402
from compiler.aten.models.kda.shape import KdaShape  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.program_kda_gates import (  # noqa: E402
    kda_head_blocks,
    kda_key_blocks,
    kda_key_row,
)
from compiler.aten.tests.isa_interpreter import Machine  # noqa: E402

MLEN = 8


def _rows_up(n: int) -> int:
    return ((n + MLEN - 1) // MLEN) * MLEN


def _shape(num_heads: int, key_dim: int) -> KdaShape:
    return KdaShape(
        hidden_size=num_heads * MLEN,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=MLEN,
        conv_kernel=4,
    )


class _GateHarness:
    def __init__(self, shape: KdaShape):
        self.shape = shape
        p = self.prog = PlenaCompiler(mlen=MLEN, blen=2)
        kb = kda_key_blocks(shape, MLEN)
        rows = _rows_up(shape.num_heads * kb)
        self.gate = p.alloc("gate", rows, MLEN)
        self.dt_bias = p.alloc("dt_bias", rows, MLEN)
        self.beta_logit = p.alloc("beta_logit", _rows_up(kda_head_blocks(shape, MLEN)), MLEN)
        self.rate = p.fp_var("rate", size=shape.num_heads)
        self.lower = p.fp_var("lower", size=1)
        self.decay = p.fp_var("decay", size=shape.num_heads * shape.key_dim)
        # Sized to exactly what S_MAP_FP_V writes -- rows x mlen, not num_heads.
        # The sentinel after it turns an over-write into a visible failure
        # instead of a silent clobber of whatever came next.
        self.beta = p.fp_var("beta", size=kda_head_blocks(shape, MLEN) * MLEN)
        self.sentinel = p.fp_var("sentinel", size=MLEN)
        self.consts = p.kda_fp_constants()
        self.mark = len(p.get_code())

    def _base(self, var) -> int:
        return self.prog.get_vram_layout(var.name).vram_base_addr

    def run(self, *, gate, dt_bias, a_log, beta_logit):
        """gate/dt_bias [heads, key_dim]; a_log [heads]; beta_logit [heads]."""
        s = self.shape
        self.prog.kda_decay_scalars_v0(
            gate=self.gate, dt_bias=self.dt_bias, rate_fp=self.rate,
            lower_bound_fp=self.lower, decay_fp=self.decay, consts=self.consts,
            shape=s,
        )
        self.prog.kda_beta_scalars_v0(
            beta_logit=self.beta_logit, beta_fp=self.beta, consts=self.consts, shape=s,
        )
        code = self.prog.get_code()[self.mark :]

        m = Machine(vlen=MLEN)
        kb = kda_key_blocks(s, MLEN)
        for h in range(s.num_heads):
            for b in range(kb):
                lanes = slice(b * MLEN, (b + 1) * MLEN)
                row = kda_key_row(s, MLEN, h, b)
                m.write_vram_row(self._base(self.gate) + row * MLEN, gate[h, lanes].tolist())
                m.write_vram_row(
                    self._base(self.dt_bias) + row * MLEN, dt_bias[h, lanes].tolist()
                )
        for hb in range(kda_head_blocks(s, MLEN)):
            chunk = beta_logit[hb * MLEN : (hb + 1) * MLEN].tolist()
            m.write_vram_row(
                self._base(self.beta_logit) + hb * MLEN, chunk + [0.0] * (MLEN - len(chunk))
            )

        m.write_fpram(self.consts.zero.address, PlenaCompiler.kda_fp_constant_values())
        m.write_fpram(self.sentinel.address, [-99.0] * MLEN)
        # rate = exp(a_log), a host constant: it depends only on a weight.
        m.write_fpram(self.rate.address, torch.exp(a_log).tolist())
        m.write_fpram(self.lower.address, [s.gate_lower_bound])
        m.run(code)

        decay = torch.tensor(
            [
                self.decay_slot(m, h, k)
                for h in range(s.num_heads)
                for k in range(s.key_dim)
            ],
            dtype=torch.float32,
        ).reshape(s.num_heads, s.key_dim)
        beta = torch.tensor(
            [m.fpram[self.beta.address + h] for h in range(s.num_heads)],
            dtype=torch.float32,
        )
        assert all(
            m.fpram[self.sentinel.address + i] == -99.0 for i in range(MLEN)
        ), "an FPRAM write ran past its allocation into the next one"
        return decay, beta, m

    def decay_slot(self, m, head: int, key: int) -> float:
        """key-major within head, which is what the recurrence indexes."""
        return m.fpram[self.decay.address + head * self.shape.key_dim + key]


def _case(seed: int, num_heads: int, key_dim: int):
    torch.manual_seed(seed)
    shape = _shape(num_heads, key_dim)
    gate = 0.8 * torch.randn(num_heads, key_dim)
    dt_bias = 0.8 * torch.randn(num_heads, key_dim)
    a_log = 0.5 * torch.randn(num_heads)
    beta_logit = torch.randn(num_heads)

    log_decay = activate_log_decay(
        gate[None, :, :], a_log, dt_bias, lower_bound=shape.gate_lower_bound
    )
    return shape, dict(
        gate=gate, dt_bias=dt_bias, a_log=a_log, beta_logit=beta_logit,
        expected_decay=torch.exp(log_decay)[0],
        expected_beta=torch.sigmoid(beta_logit),
    )


@pytest.mark.parametrize(
    "seed,heads,key_dim",
    [
        (1, 1, MLEN),        # one head, one key block
        (2, 3, MLEN),        # several heads
        (3, 1, MLEN * 2),    # one head spanning two key blocks
        (4, 4, MLEN * 2),    # several heads, two key blocks -- Kimi's shape
        (5, 2, MLEN * 3),    # three key blocks
    ],
)
def test_decay_and_beta_match_the_reference(seed, heads, key_dim):
    shape, c = _case(seed, heads, key_dim)
    decay, beta, _ = _GateHarness(shape).run(
        gate=c["gate"], dt_bias=c["dt_bias"], a_log=c["a_log"],
        beta_logit=c["beta_logit"],
    )
    torch.testing.assert_close(decay, c["expected_decay"], rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(beta, c["expected_beta"], rtol=2e-5, atol=2e-6)


def test_heads_with_a_partial_beta_row_are_handled():
    """beta_logit holds one value per head, so its last row is partly empty
    whenever num_heads is not a multiple of mlen -- Kimi K3 is 96 against 64."""
    heads = MLEN + 3
    shape, c = _case(6, heads, MLEN)
    assert kda_head_blocks(shape, MLEN) == 2
    _, beta, _ = _GateHarness(shape).run(
        gate=c["gate"], dt_bias=c["dt_bias"], a_log=c["a_log"],
        beta_logit=c["beta_logit"],
    )
    torch.testing.assert_close(beta, c["expected_beta"], rtol=2e-5, atol=2e-6)


def test_decay_uses_each_head_own_rate():
    """`rate = exp(a_log[h])` is per head. Broadcasting one head's rate to all
    of them still yields plausible decays in (0, 1)."""
    shape, c = _case(7, 3, MLEN * 2)
    decay, _, _ = _GateHarness(shape).run(
        gate=c["gate"], dt_bias=c["dt_bias"], a_log=c["a_log"],
        beta_logit=c["beta_logit"],
    )
    torch.testing.assert_close(decay, c["expected_decay"], rtol=3e-5, atol=3e-6)

    # every head's rate genuinely differs, so a shared rate would be visible
    same_rate = torch.full_like(c["a_log"], c["a_log"][0].item())
    wrong = torch.exp(
        activate_log_decay(
            c["gate"][None, :, :], same_rate, c["dt_bias"],
            lower_bound=shape.gate_lower_bound,
        )
    )[0]
    assert not torch.allclose(decay, wrong, rtol=1e-3, atol=1e-3)


def test_decay_lands_key_major_within_each_head():
    """The recurrence indexes decay_fp[h * key_dim + k]. tile_row_to_fpram puts
    row i at base + i*mlen, so the row ordering is what makes that true."""
    shape, c = _case(8, 3, MLEN * 2)
    h = _GateHarness(shape)
    decay, _, m = h.run(
        gate=c["gate"], dt_bias=c["dt_bias"], a_log=c["a_log"],
        beta_logit=c["beta_logit"],
    )
    for head in range(shape.num_heads):
        for key in range(shape.key_dim):
            assert h.decay_slot(m, head, key) == pytest.approx(
                c["expected_decay"][head, key].item(), rel=3e-5, abs=3e-6
            )


def test_decay_is_bounded_by_the_gate_lower_bound():
    """log_decay is in [lower_bound, 0], so decay is in [exp(lower_bound), 1]."""
    shape, c = _case(9, 3, MLEN * 2)
    decay, _, _ = _GateHarness(shape).run(
        gate=c["gate"], dt_bias=c["dt_bias"], a_log=c["a_log"],
        beta_logit=c["beta_logit"],
    )
    import math

    assert decay.min().item() >= math.exp(shape.gate_lower_bound) - 1e-5
    assert decay.max().item() <= 1.0 + 1e-5


def test_rejects_shapes_and_undersized_fpram():
    p = PlenaCompiler(mlen=MLEN, blen=2)
    shape = _shape(2, MLEN)
    consts = p.kda_fp_constants()
    ok = dict(
        dt_bias=p.alloc("db", MLEN, MLEN), rate_fp=p.fp_var("r", size=2),
        lower_bound_fp=p.fp_var("l", size=1), consts=consts, shape=shape,
    )
    with pytest.raises(ValueError, match="exactly mlen"):
        p.kda_decay_scalars_v0(
            gate=p.alloc("g1", MLEN, MLEN * 2),
            decay_fp=p.fp_var("d1", size=2 * MLEN), **ok,
        )
    with pytest.raises(ValueError, match="decay_fp"):
        p.kda_decay_scalars_v0(
            gate=p.alloc("g2", MLEN, MLEN), decay_fp=p.fp_var("d2", size=3), **ok,
        )
    with pytest.raises(ValueError, match="rate_fp"):
        p.kda_decay_scalars_v0(
            gate=p.alloc("g3", MLEN, MLEN), decay_fp=p.fp_var("d3", size=2 * MLEN),
            **{**ok, "rate_fp": p.fp_var("r2", size=1)},
        )
    with pytest.raises(ValueError, match="multiple of mlen"):
        kda_key_blocks(_shape(1, MLEN + 1), MLEN)


def test_beta_fp_must_be_sized_for_the_full_row_write():
    """S_MAP_FP_V moves a whole mlen row, so sizing beta_fp to num_heads lets
    the tail land in the next allocation. At Kimi K3's 96 heads against mlen 64
    that is 32 slots of sigmoid(padding) written outside the buffer."""
    heads = MLEN + 3
    shape = _shape(heads, MLEN)
    p = PlenaCompiler(mlen=MLEN, blen=2)
    consts = p.kda_fp_constants()
    beta_logit = p.alloc("bl", _rows_up(kda_head_blocks(shape, MLEN)), MLEN)
    with pytest.raises(ValueError, match="padded row count"):
        p.kda_beta_scalars_v0(
            beta_logit=beta_logit,
            beta_fp=p.fp_var("small", size=heads),
            consts=consts,
            shape=shape,
        )
    # exactly the padded size is accepted
    p.kda_beta_scalars_v0(
        beta_logit=beta_logit,
        beta_fp=p.fp_var("right", size=kda_head_blocks(shape, MLEN) * MLEN),
        consts=consts,
        shape=shape,
    )


# ---------------------------------------------------------------------------
# Per-head streaming.
#
# FPRAM is 512 slots in RTL. Producing every head's decay at once needs
# num_heads * key_dim -- 12,288 for Kimi K3 -- so the mixer streams one head at
# a time and reuses the window. These pin that the streamed result is identical
# to the all-at-once one.
# ---------------------------------------------------------------------------


class _PerHeadHarness(_GateHarness):
    """Same tiles, but decay_fp sized for one head rather than all of them."""

    def __init__(self, shape: KdaShape):
        super().__init__(shape)
        self.one_head_decay = self.prog.fp_var("one_head_decay", size=shape.key_dim)
        self.mark = len(self.prog.get_code())

    def run_head(self, head: int, *, gate, dt_bias, a_log):
        s = self.shape
        self.prog.kda_decay_scalars_v0(
            gate=self.gate, dt_bias=self.dt_bias, rate_fp=self.rate,
            lower_bound_fp=self.lower, decay_fp=self.one_head_decay,
            consts=self.consts, shape=s, heads=[head],
        )
        code = self.prog.get_code()[self.mark :]
        self.mark = len(self.prog.get_code())

        m = Machine(vlen=MLEN)
        kb = kda_key_blocks(s, MLEN)
        for h in range(s.num_heads):
            for b in range(kb):
                lanes = slice(b * MLEN, (b + 1) * MLEN)
                row = kda_key_row(s, MLEN, h, b)
                m.write_vram_row(self._base(self.gate) + row * MLEN, gate[h, lanes].tolist())
                m.write_vram_row(
                    self._base(self.dt_bias) + row * MLEN, dt_bias[h, lanes].tolist()
                )
        m.write_fpram(self.consts.zero.address, PlenaCompiler.kda_fp_constant_values())
        m.write_fpram(self.rate.address, torch.exp(a_log).tolist())
        m.write_fpram(self.lower.address, [s.gate_lower_bound])
        m.run(code)
        return torch.tensor(
            [m.fpram[self.one_head_decay.address + k] for k in range(s.key_dim)],
            dtype=torch.float32,
        )


@pytest.mark.parametrize("heads,key_dim", [(3, MLEN), (4, MLEN * 2), (2, MLEN * 3)])
def test_one_head_at_a_time_matches_all_at_once(heads, key_dim):
    """The streamed window holds key_dim slots instead of num_heads*key_dim,
    and the scalars land at offset 0 because tile_row_to_fpram addresses by
    position in `rows`, not by row index."""
    shape, c = _case(21, heads, key_dim)
    for h in range(heads):
        got = _PerHeadHarness(shape).run_head(
            h, gate=c["gate"], dt_bias=c["dt_bias"], a_log=c["a_log"]
        )
        torch.testing.assert_close(
            got, c["expected_decay"][h], rtol=3e-5, atol=3e-6
        )


def test_streaming_uses_the_selected_head_rate_not_its_position():
    """rate_fp is indexed by head number. Using the position within `heads`
    would silently give every streamed head rate[0]."""
    shape, c = _case(22, 4, MLEN * 2)
    last = shape.num_heads - 1
    got = _PerHeadHarness(shape).run_head(
        last, gate=c["gate"], dt_bias=c["dt_bias"], a_log=c["a_log"]
    )
    torch.testing.assert_close(got, c["expected_decay"][last], rtol=3e-5, atol=3e-6)

    # head 0's rate must give a different answer, or the test proves nothing
    with_rate0 = torch.exp(
        activate_log_decay(
            c["gate"][None, last : last + 1, :],
            c["a_log"][:1],
            c["dt_bias"][last : last + 1],
            lower_bound=shape.gate_lower_bound,
        )
    )[0, 0]
    assert not torch.allclose(got, with_rate0, rtol=1e-3, atol=1e-3)


def test_streaming_rejects_a_head_out_of_range():
    shape, _ = _case(23, 2, MLEN)
    h = _PerHeadHarness(shape)
    with pytest.raises(ValueError, match="out of range"):
        h.prog.kda_decay_scalars_v0(
            gate=h.gate, dt_bias=h.dt_bias, rate_fp=h.rate, lower_bound_fp=h.lower,
            decay_fp=h.one_head_decay, consts=h.consts, shape=shape, heads=[2],
        )


def test_a_head_subset_bounds_the_tile_by_row_number_not_by_count():
    """With heads=[h] the rows are a slice out of the middle, so the row count
    is smaller than the highest row number. Checking the count accepts a tile
    the emitter then reads and writes past the end of -- into whatever VRAM
    object was allocated next, with no error anywhere in the stack."""
    shape = KdaShape(
        hidden_size=32, num_heads=16, key_dim=MLEN, value_dim=MLEN, conv_kernel=4
    )
    p = PlenaCompiler(mlen=MLEN, blen=2)
    # Room for one head's worth of rows, but head 12 lives at row 12.
    gate = p.alloc("gate", MLEN, MLEN)
    dt_bias = p.alloc("dt_bias", MLEN, MLEN)
    with pytest.raises(ValueError, match="to reach head 12"):
        p.kda_decay_scalars_v0(
            gate=gate, dt_bias=dt_bias,
            rate_fp=p.fp_var("rate", size=shape.num_heads),
            lower_bound_fp=p.fp_var("lb", size=1),
            decay_fp=p.fp_var("decay", size=MLEN),
            consts=p.kda_fp_constants(), shape=shape, heads=[12],
        )
