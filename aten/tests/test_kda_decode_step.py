"""The KDA decode step, executed and compared against the CPU reference.

`aten/tests/isa_interpreter.py` runs the emitted assembly, so these are
numerical tests of the lowering rather than assertions about which opcodes
appear. See that module's docstring for what it does and does not model.

Phase 0: the lowering uses only instructions that already exist. `V_FMA_VF`
arrives in Task 7 and Task 8 converts this kernel onto it; the counts recorded
by `test_records_the_phase0_instruction_count` are the baseline that change is
measured against.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.kda.reference import KdaState, activate_log_decay, kda_step  # noqa: E402
from compiler.aten.models.kda.shape import KdaShape  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.program_kda_common import (  # noqa: E402
    kda_blocks,
    kda_stage_marker,
    kda_state_row,
    kda_state_rows,
    kda_vector_row,
    kda_vector_rows,
)
from compiler.aten.tests.isa_interpreter import Machine, UnsupportedInstruction  # noqa: E402

#: What the recurrence emitted before Task 8. Named rather than inlined so the
#: assertion below reads as "one opcode more than this" and not as a magic list.
_PHASE0_OPCODES = frozenset(
    {
        "S_ADDI_INT",
        "S_LD_FP",
        "S_ST_FP",
        "S_ADD_FP",
        "V_MUL_VF",
        "V_ADD_VF",
        "V_SUB_VF",
        "V_ADD_VV",
        "V_SUB_VV",
        "V_MUL_VV",
        "V_RED_SUM",
        "C_LOOP_START",
        "C_LOOP_END",
    }
)

MLEN = 8


def _rows_up(n: int) -> int:
    """VRAM row counts must be a multiple of mlen."""
    return ((n + MLEN - 1) // MLEN) * MLEN


class _Harness:
    """Lower one decode step, seed VRAM/FPRAM from tensors, execute, read back."""

    def __init__(self, shape: KdaShape):
        self.shape = shape
        self.prog = PlenaCompiler(mlen=MLEN, blen=2)
        p, s = self.prog, shape

        # Every tile is exactly mlen wide; the column block lives in the row.
        self.state = p.alloc("state", _rows_up(kda_state_rows(s, MLEN)), MLEN)
        vec_rows = _rows_up(kda_vector_rows(s, MLEN))
        self.v = p.alloc("v", vec_rows, MLEN)
        self.o = p.alloc("o", vec_rows, MLEN)
        self.pred = p.alloc("pred", vec_rows, MLEN)
        self.err = p.alloc("err", vec_rows, MLEN)

        n_key = s.num_heads * s.key_dim
        self.q_fp = p.fp_var("q_hat", size=n_key)
        self.k_fp = p.fp_var("k_hat", size=n_key)
        self.decay_fp = p.fp_var("decay", size=n_key)
        self.beta_fp = p.fp_var("beta", size=s.num_heads)
        self.scale_fp = p.fp_var("out_scale", size=1)

        self.mark = len(p.get_code())

    def emit(self, **kw):
        self.prog.kda_decode_step_v0(
            state=self.state, q_fp=self.q_fp, k_fp=self.k_fp, decay_fp=self.decay_fp,
            beta_fp=self.beta_fp, v=self.v, o=self.o, pred=self.pred,
            err=self.err, shape=self.shape, output_scale_fp=self.scale_fp, **kw,
        )
        return self.prog.get_code()[self.mark :]

    def _base(self, var) -> int:
        return self.prog.get_vram_layout(var.name).vram_base_addr

    def seed_state(self, m, state) -> None:
        """state is [heads, key, value]; scatter it into the flattened tile."""
        s = self.shape
        for h in range(s.num_heads):
            for c in range(kda_blocks(s, MLEN)):
                lanes = slice(c * MLEN, (c + 1) * MLEN)
                for j in range(s.key_dim):
                    m.write_vram_row(
                        self._base(self.state) + kda_state_row(s, MLEN, h, c, j) * MLEN,
                        state[h, j, lanes].tolist(),
                    )

    def seed_token(self, m, *, v, q_hat, k_hat, decay, beta, output_scale) -> None:
        """Per-token inputs. Deliberately separate from seed_state: a decode
        loop rewrites these every token while the state stays on chip."""
        s = self.shape
        for h in range(s.num_heads):
            for c in range(kda_blocks(s, MLEN)):
                m.write_vram_row(
                    self._base(self.v) + kda_vector_row(s, MLEN, h, c) * MLEN,
                    v[h, c * MLEN : (c + 1) * MLEN].tolist(),
                )
        m.write_fpram(self.q_fp.address, q_hat.reshape(-1).tolist())
        m.write_fpram(self.k_fp.address, k_hat.reshape(-1).tolist())
        m.write_fpram(self.decay_fp.address, decay.reshape(-1).tolist())
        m.write_fpram(self.beta_fp.address, beta.tolist())
        m.write_fpram(self.scale_fp.address, [output_scale])

    def read_vector(self, m, tile):
        """Gather a flattened [heads*blocks, mlen] tile back to [heads, value]."""
        s = self.shape
        return torch.tensor(
            [
                [
                    x
                    for c in range(kda_blocks(s, MLEN))
                    for x in m.read_vram_row(
                        self._base(tile) + kda_vector_row(s, MLEN, h, c) * MLEN, MLEN
                    )
                ]
                for h in range(s.num_heads)
            ],
            dtype=torch.float32,
        )

    def run(self, *, state, q_hat, k_hat, decay, beta, v, output_scale):
        """state [heads,key,value]; q/k/decay [heads,key]; beta [heads]; v [heads,value]."""
        s = self.shape
        code = self.emit()
        m = Machine(vlen=MLEN)

        # Seed the accumulators with garbage. A fresh Machine is all zeros, so
        # without this 'the program clears it' and 'it started at zero' are
        # indistinguishable -- deleting either vram_fill_zero used to leave
        # every test here green but one.
        for _acc in (self.pred, self.o):
            for _r in range(_acc.shape[0]):
                m.write_vram_row(self._base(_acc) + _r * MLEN, [7.5] * MLEN)

        blocks = kda_blocks(s, MLEN)
        self.seed_state(m, state)
        self.seed_token(m, v=v, q_hat=q_hat, k_hat=k_hat, decay=decay,
                        beta=beta, output_scale=output_scale)
        m.run(code)

        out = torch.tensor(
            [
                [
                    x
                    for c in range(blocks)
                    for x in m.read_vram_row(
                        self._base(self.o) + kda_vector_row(s, MLEN, h, c) * MLEN, MLEN
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
                        x
                        for c in range(blocks)
                        for x in m.read_vram_row(
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


def _case(seed: int, *, num_heads: int, key_dim: int, value_dim: int = MLEN):
    """One random decode step: the shape, the reference result, and the FPRAM
    scalars the lowering expects to be handed already normalised."""
    torch.manual_seed(seed)
    shape = KdaShape(
        hidden_size=num_heads * value_dim,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        conv_kernel=4,
    )
    state = torch.randn(1, num_heads, value_dim, key_dim)
    q = 0.5 * torch.randn(1, num_heads, key_dim)
    k = 0.5 * torch.randn(1, num_heads, key_dim)
    v = torch.randn(1, num_heads, value_dim)
    gate = torch.randn(1, num_heads, key_dim)
    beta_logit = torch.randn(1, num_heads)
    a_log = torch.randn(num_heads)
    dt_bias = torch.randn(num_heads, key_dim)

    out, new = kda_step(
        q, k, v, gate, beta_logit, KdaState(state.clone()), a_log, dt_bias, shape
    )

    def _norm(x):
        # Must match reference.py exactly: epsilon inside rsqrt, not F.normalize.
        return x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1.0e-6)

    log_decay = activate_log_decay(
        gate, a_log, dt_bias, lower_bound=shape.gate_lower_bound
    )
    return shape, {
        # inputs to the lowering, all batch-0
        "state_T": state[0].transpose(-2, -1).contiguous(),  # [heads, key, value]
        "q_hat": _norm(q)[0],
        "k_hat": _norm(k)[0],
        "decay": torch.exp(log_decay)[0],
        "beta": torch.sigmoid(beta_logit.float())[0],
        "v": v[0],
        "output_scale": 1.0 / (shape.key_dim**0.5),
        # expected results
        "out": out[0],
        "state_out_T": new.recurrent[0].transpose(-2, -1).contiguous(),
    }


def _run(shape, ref):
    h = _Harness(shape)
    out, new_state, machine = h.run(
        state=ref["state_T"],
        q_hat=ref["q_hat"],
        k_hat=ref["k_hat"],
        decay=ref["decay"],
        beta=ref["beta"],
        v=ref["v"],
        output_scale=ref["output_scale"],
    )
    return out, new_state, machine


# ---------------------------------------------------------------------------
# Numerical equivalence with the CPU reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed,heads,key_dim",
    [
        (1, 1, 2),   # smallest thing that can be wrong
        (2, 1, 5),   # key_dim != value_dim: a transposed axis is not a shape error
        (3, 3, 4),   # multiple heads: catches per-head FPRAM offset bugs
        (4, 2, 8),   # key_dim == value_dim == mlen, the square case
    ],
)
def test_decode_step_matches_the_reference(seed, heads, key_dim):
    shape, ref = _case(seed, num_heads=heads, key_dim=key_dim)
    out, new_state, _ = _run(shape, ref)
    torch.testing.assert_close(out, ref["out"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(new_state, ref["state_out_T"], rtol=1e-5, atol=1e-6)


def test_read_out_uses_the_updated_state_not_the_decayed_one():
    """The delta rule reads the state *after* the rank-1 update.

    Reducing over the decayed-but-not-updated state is a plausible reordering
    that changes the answer, and the output alone would still look reasonable.
    """
    shape, ref = _case(7, num_heads=2, key_dim=4)
    out, _, _ = _run(shape, ref)

    stale = torch.einsum(
        "hkv,hk->hv", ref["state_T"] * ref["decay"][:, :, None], ref["q_hat"]
    ) * ref["output_scale"]
    assert not torch.allclose(out, stale, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(out, ref["out"], rtol=1e-5, atol=1e-6)


def test_heads_do_not_leak_into_each_other():
    """Head h must touch only its own state rows and its own FPRAM slice.
    Running one head at a time must reproduce the all-heads run."""
    shape, ref = _case(5, num_heads=3, key_dim=4)
    together, _, _ = _run(shape, ref)

    for h in range(shape.num_heads):
        harness = _Harness(shape)
        code = harness.emit(head_rows=[h])
        m = Machine(vlen=MLEN)
        harness.seed_state(m, ref["state_T"])
        harness.seed_token(
            m, v=ref["v"], q_hat=ref["q_hat"], k_hat=ref["k_hat"],
            decay=ref["decay"], beta=ref["beta"], output_scale=ref["output_scale"],
        )
        m.run(code)

        torch.testing.assert_close(
            harness.read_vector(m, harness.o)[h], together[h], rtol=1e-5, atol=1e-6
        )

        # every other head's state untouched, bit for bit
        for other in range(shape.num_heads):
            if other == h:
                continue
            for c in range(kda_blocks(shape, MLEN)):
                for j in range(shape.key_dim):
                    got = m.read_vram_row(
                        harness._base(harness.state)
                        + kda_state_row(shape, MLEN, other, c, j) * MLEN,
                        MLEN,
                    )
                    torch.testing.assert_close(
                        torch.tensor(got),
                        ref["state_T"][other, j, c * MLEN : (c + 1) * MLEN],
                        rtol=0, atol=0,
                    )


# ---------------------------------------------------------------------------
# Guards and baseline
# ---------------------------------------------------------------------------


def test_rejects_a_value_dim_that_is_not_a_whole_number_of_blocks():
    """A partial trailing block would leave lanes past value_dim holding
    whatever was there before -- and they would still be summed."""
    shape = KdaShape(
        hidden_size=64, num_heads=1, key_dim=2, value_dim=MLEN + 3, conv_kernel=4
    )
    with pytest.raises(ValueError, match="multiple of mlen"):
        _Harness(shape).emit()


def test_rejects_a_tile_that_is_not_exactly_one_block_wide():
    """Every KDA tile carries the column block in its row index. A wider tile
    would be swept by helpers that silently cover only block 0."""
    shape, _ = _case(1, num_heads=1, key_dim=2)
    h = _Harness(shape)
    wide = h.prog.alloc("wide", MLEN, MLEN * 2)
    with pytest.raises(ValueError, match="exactly mlen"):
        h.prog.kda_decode_step_v0(
            state=h.state, q_fp=h.q_fp, k_fp=h.k_fp,
            decay_fp=h.decay_fp, beta_fp=h.beta_fp, v=h.v, o=h.o, pred=wide,
            err=h.err, shape=shape, output_scale_fp=h.scale_fp,
        )


def test_rejects_pred_aliased_to_state():
    """Was `scratch` aliased to state. Task 8 removed the scratch tile -- the
    FMA accumulates in place -- so the aliasing that still matters is `pred`
    and `err`, which the recurrence writes while reading state."""
    shape, _ = _case(1, num_heads=1, key_dim=2)
    h = _Harness(shape)
    with pytest.raises(ValueError, match="must not alias state"):
        h.prog.kda_decode_step_v0(
            state=h.state, q_fp=h.q_fp, k_fp=h.k_fp,
            decay_fp=h.decay_fp, beta_fp=h.beta_fp, v=h.v, o=h.o, pred=h.state,
            err=h.err, shape=shape, output_scale_fp=h.scale_fp,
        )


def test_rejects_a_negative_fp_head_stride():
    shape, _ = _case(1, num_heads=2, key_dim=2)
    h = _Harness(shape)
    with pytest.raises(ValueError, match="must not be negative"):
        h.prog.kda_decode_step_v0(
            state=h.state, q_fp=h.q_fp, k_fp=h.k_fp,
            decay_fp=h.decay_fp, beta_fp=h.beta_fp, v=h.v, o=h.o, pred=h.pred,
            err=h.err, shape=shape, output_scale_fp=h.scale_fp,
            fp_head_stride=-1,
        )


def test_rejects_more_than_one_head_when_the_window_is_reused():
    """fp_head_stride=0 means every head reads the same FPRAM slots. Two heads
    would then silently share one head's decay, q_hat and k_hat -- a finite,
    plausible, wrong answer, which is exactly what this guard exists to stop."""
    shape, _ = _case(1, num_heads=2, key_dim=2)
    h = _Harness(shape)
    with pytest.raises(ValueError, match="only one head may be lowered"):
        h.prog.kda_decode_step_v0(
            state=h.state, q_fp=h.q_fp, k_fp=h.k_fp,
            decay_fp=h.decay_fp, beta_fp=h.beta_fp, v=h.v, o=h.o, pred=h.pred,
            err=h.err, shape=shape, output_scale_fp=h.scale_fp,
            head_rows=[0, 1], fp_head_stride=0,
        )


def test_the_two_halves_emit_the_same_program_as_the_whole_step():
    """kda_decode_step_v0 is a wrapper. If the halves ever drift from it, the
    mixer -- which calls the halves so it can share one FPRAM window between
    decay and q_hat -- silently stops matching what these tests cover."""
    shape, _ = _case(3, num_heads=2, key_dim=2)

    whole = _Harness(shape)
    whole.prog.kda_decode_step_v0(
        state=whole.state, q_fp=whole.q_fp,
        k_fp=whole.k_fp, decay_fp=whole.decay_fp, beta_fp=whole.beta_fp,
        v=whole.v, o=whole.o, pred=whole.pred, err=whole.err, shape=shape,
        output_scale_fp=whole.scale_fp,
    )

    halves = _Harness(shape)
    halves.prog.kda_decode_predict_v0(
        state=halves.state, k_fp=halves.k_fp,
        decay_fp=halves.decay_fp, beta_fp=halves.beta_fp, v=halves.v,
        pred=halves.pred, err=halves.err, shape=shape,
    )
    halves.prog.kda_decode_update_v0(
        state=halves.state, k_fp=halves.k_fp,
        q_fp=halves.q_fp, o=halves.o, err=halves.err, shape=shape,
        output_scale_fp=halves.scale_fp,
    )

    assert whole.prog.get_code()[whole.mark :] == halves.prog.get_code()[halves.mark :]


def test_emits_exactly_one_opcode_beyond_the_phase0_set():
    """This asserted `V_FMA_VF not in code` through Phase 0, when the point was
    that KDA needs no new opcode. That claim was demonstrated and now it is
    spent: Task 8 converts the sweeps onto the one instruction Phase 1 adds.

    What still has to hold is that it is exactly *one*. The interpreter raises
    on anything it does not model, so a lowering that reaches for a second new
    opcode fails here rather than being silently half-checked.

    "One" is a statement about **this kernel**, not about the branch. The
    branch adds three opcodes -- `V_SOFTPLUS_V` 0x39, `S_MAP_FP_V` 0x3A and
    `V_FMA_VF` 0x3B -- and `test_no_state_engine.py` pins that count. The
    decode step reaches for only the third of them; the other two belong to
    Mamba's `dt` and to the FPRAM window fill.
    """
    shape, ref = _case(1, num_heads=2, key_dim=4)
    code = _Harness(shape).emit()
    assert "V_FMA_VF" in code, "the sweeps are supposed to be fused now"
    ops = {
        line.strip().replace(",", " ").split()[0]
        for line in code.splitlines()
        if line.strip() and not line.strip().startswith(";")
    }
    assert ops - _PHASE0_OPCODES == {"V_FMA_VF"}, (
        f"beyond Phase 0's set the decode step may emit only V_FMA_VF, "
        f"got {sorted(ops - _PHASE0_OPCODES)}"
    )
    try:
        Machine(vlen=MLEN).run(code)
    except UnsupportedInstruction as exc:  # pragma: no cover - failure path
        pytest.fail(f"decode step emitted an unmodelled instruction: {exc}")


def test_the_sweeps_are_hardware_loops_not_unrolled():
    """The static-footprint claim. Each (head, block) contributes three sweeps
    -- predict, rank-1, read-out -- and each must be one FMA inside one loop,
    however many keys it walks.

    Without this, a change that broke the row progression would stay green: the
    numbers would still be right, and the program would just be 64x larger.
    """
    for key_dim in (4, 8, 16):
        shape, _ = _case(2, num_heads=1, key_dim=key_dim)
        code = _Harness(shape).emit()
        body = [
            line.strip()
            for line in code.splitlines()
            if line.strip() and not line.strip().startswith(";")
        ]
        blocks = kda_blocks(shape, MLEN)
        assert sum("V_FMA_VF" in line for line in body) == 3 * blocks, (
            f"key_dim={key_dim}: expected one FMA per sweep, got "
            f"{sum('V_FMA_VF' in line for line in body)}"
        )


def test_records_the_phase0_instruction_count():
    """Baseline for Task 8. Not a budget -- the number this prints, against the
    one Task 8 prints, is the measured case for spending opcode 0x3B."""
    shape, ref = _case(3, num_heads=1, key_dim=8)
    code = _Harness(shape).emit()
    static = len([l for l in code.splitlines() if l.strip() and not l.strip().startswith(";")])
    _, _, machine = _run(shape, ref)
    print(
        f"PHASE0_KDA_DECODE heads={shape.num_heads} key_dim={shape.key_dim} "
        f"static={static} dynamic={machine.executed}"
    )
    assert static > 0 and machine.executed > 0


def test_two_tokens_in_one_program_match_two_reference_steps():
    """The accumulators must be zeroed, and only a second token can prove it.

    Every other test runs one step on a freshly-zeroed Machine, so an
    accumulator that is never cleared is indistinguishable from one that happens
    to start at zero. A real decode loop reuses the same `pred` and `o` tiles
    every token: without the fill-zero, token 1 accumulates on top of token 0's
    residue. Deleting either `vram_fill_zero` left the rest of this file green.

    This also covers the state being updated in place on chip and carried into
    the next step, which is the whole point of a recurrent kernel.
    """
    torch.manual_seed(31)
    heads, key_dim, value_dim = 2, 4, MLEN * 2
    shape = KdaShape(
        hidden_size=heads * value_dim, num_heads=heads, key_dim=key_dim,
        value_dim=value_dim, conv_kernel=4,
    )

    state = torch.randn(1, heads, value_dim, key_dim)
    tokens = [
        dict(
            q=0.5 * torch.randn(1, heads, key_dim),
            k=0.5 * torch.randn(1, heads, key_dim),
            v=torch.randn(1, heads, value_dim),
            gate=torch.randn(1, heads, key_dim),
            beta_logit=torch.randn(1, heads),
        )
        for _ in range(2)
    ]
    a_log = torch.randn(heads)
    dt_bias = torch.randn(heads, key_dim)

    def _norm(x):
        return x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1.0e-6)

    # reference: two sequential steps carrying state
    carried = KdaState(state.clone())
    expected_out = []
    for t in tokens:
        out, carried = kda_step(
            t["q"], t["k"], t["v"], t["gate"], t["beta_logit"], carried,
            a_log, dt_bias, shape,
        )
        expected_out.append(out[0])

    # lowering: one program, the step emitted twice, same tiles reused
    h = _Harness(shape)
    programs = []
    for _ in range(2):
        start = len(h.prog.get_code())
        h.emit()
        programs.append(h.prog.get_code()[start:])

    m = Machine(vlen=MLEN)
    h.seed_state(m, state[0].transpose(-2, -1).contiguous())

    # Token 1's scalars are written between the halves -- what a real decode
    # loop does through S_MAP_FP_V. VRAM is never reset: `state` carries, and
    # `pred` / `o` must be cleared by the program itself.
    actual_out = []
    for t, program in zip(tokens, programs):
        log_decay = activate_log_decay(
            t["gate"], a_log, dt_bias, lower_bound=shape.gate_lower_bound
        )
        h.seed_token(
            m,
            v=t["v"][0],
            q_hat=_norm(t["q"])[0],
            k_hat=_norm(t["k"])[0],
            decay=torch.exp(log_decay)[0],
            beta=torch.sigmoid(t["beta_logit"].float())[0],
            output_scale=1.0 / (key_dim**0.5),
        )
        m.run(program)
        actual_out.append(h.read_vector(m, h.o))

    torch.testing.assert_close(actual_out[0], expected_out[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_out[1], expected_out[1], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Multiple column blocks -- the case the flattened layout exists for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed,heads,key_dim,blocks",
    [
        (11, 1, 3, 2),   # smallest multi-block case
        (12, 2, 5, 2),   # several heads, key_dim != value_dim
        (13, 1, 4, 3),   # three blocks: catches an off-by-one in the block stride
    ],
)
def test_decode_step_matches_the_reference_across_column_blocks(seed, heads, key_dim, blocks):
    """value_dim > mlen. Kimi K3 is value_dim=128 against a default mlen=64.

    With the old [key, value] layout this was impossible: the helper family is
    split between "walks every column block" and "silently block 0 only", so the
    state came out correct in its first mlen lanes and stale in the rest.
    """
    shape, ref = _case(seed, num_heads=heads, key_dim=key_dim, value_dim=MLEN * blocks)
    assert kda_blocks(shape, MLEN) == blocks
    out, new_state, _ = _run(shape, ref)
    torch.testing.assert_close(out, ref["out"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(new_state, ref["state_out_T"], rtol=1e-5, atol=1e-6)


def test_every_lane_past_the_first_block_is_actually_updated():
    """The failure the flattened layout prevents, asserted directly: lanes in
    block 1 must differ from their input, not merely be finite."""
    shape, ref = _case(14, num_heads=1, key_dim=4, value_dim=MLEN * 2)
    _, new_state, _ = _run(shape, ref)
    before = ref["state_T"]
    tail_before = before[:, :, MLEN:]
    tail_after = new_state[:, :, MLEN:]
    assert not torch.allclose(tail_after, tail_before, rtol=1e-4, atol=1e-5), (
        "lanes past the first column block were never touched"
    )
    torch.testing.assert_close(
        tail_after, ref["state_out_T"][:, :, MLEN:], rtol=1e-5, atol=1e-6
    )
