"""`ssm_decode_step_v0` against the textbook selective-scan recurrence.

Written when Task 8 converted this kernel onto `V_FMA_VF` and the whole suite
stayed green -- because **nothing tested it**. `grep -rl ssm_decode_step_v0`
found the emitter, its caller in `aten/ops/plena/mamba_ops.py`, and a docstring
mention in the KDA recurrence. No test. A rewrite of the state update and the
output contraction was therefore unverified.

The oracle is `aten/tests/isa_interpreter.py`, the same one KDA's decode step
uses, so this checks the emitted instructions rather than the Python that
emitted them.
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
from compiler.aten.tests.isa_interpreter import (  # noqa: E402
    Machine,
    UnsupportedInstruction,
)

MLEN = 8


def _rows_up(n: int) -> int:
    return ((n + MLEN - 1) // MLEN) * MLEN


class _Harness:
    """One decode step, driven through the ISA interpreter."""

    def __init__(self, shape: Mamba2Shape):
        self.shape = shape
        p = self.prog = PlenaCompiler(mlen=MLEN, blen=2)
        n = shape.state_size
        self.state = p.alloc("state", _rows_up(shape.num_heads * n), MLEN)
        self.x = p.alloc("x", _rows_up(shape.num_heads), MLEN)
        self.y = p.alloc("y", _rows_up(shape.num_heads), MLEN)
        self.scratch = p.alloc("scratch", MLEN, MLEN)
        self.b_fp = p.fp_var("b", size=shape.n_groups * n)
        self.c_fp = p.fp_var("c", size=shape.n_groups * n)
        self.da_fp = p.fp_var("da", size=shape.num_heads)
        self.dt_fp = p.fp_var("dt", size=shape.num_heads)
        self.d_fp = p.fp_var("d", size=shape.num_heads)
        self.consts = p.mamba_fp_constants()
        self.mark = len(p.get_code())

    def _base(self, var) -> int:
        return self.prog.get_vram_layout(var.name).vram_base_addr

    def emit(self) -> str:
        self.prog.ssm_decode_step_v0(
            state=self.state, x=self.x, b_fp=self.b_fp, c_fp=self.c_fp,
            da_fp=self.da_fp, dt_fp=self.dt_fp, d_fp=self.d_fp, y=self.y,
            scratch=self.scratch, shape=self.shape, consts=self.consts,
        )
        return self.prog.get_code()[self.mark :]

    def run(self, *, x, state, b, c, da, dt, d):
        s = self.shape
        code = self.emit()
        m = Machine(vlen=MLEN, vram_words=1 << 16, fpram_words=1 << 13)
        for h in range(s.num_heads):
            m.write_vram_row(self._base(self.x) + h * MLEN, x[h].tolist())
            # Seed y with garbage: the contraction must clear it, and a fresh
            # Machine is all zeros, which would hide a missing fill.
            m.write_vram_row(self._base(self.y) + h * MLEN, [7.5] * MLEN)
            for n in range(s.state_size):
                m.write_vram_row(
                    self._base(self.state) + (h * s.state_size + n) * MLEN,
                    state[h, n].tolist(),
                )
        m.write_fpram(self.b_fp.address, b.flatten().tolist())
        m.write_fpram(self.c_fp.address, c.flatten().tolist())
        m.write_fpram(self.da_fp.address, da.tolist())
        m.write_fpram(self.dt_fp.address, dt.tolist())
        m.write_fpram(self.d_fp.address, d.tolist())
        m.write_fpram(self.consts.zero.address, self.prog.mamba_fp_constant_values(self.shape))
        m.run(code)

        y = torch.tensor([
            m.read_vram_row(self._base(self.y) + h * MLEN, s.head_dim)
            for h in range(s.num_heads)
        ])
        new_state = torch.tensor([
            [
                m.read_vram_row(
                    self._base(self.state) + (h * s.state_size + n) * MLEN, s.head_dim
                )
                for n in range(s.state_size)
            ]
            for h in range(s.num_heads)
        ])
        return y, new_state, m


def _case(seed: int, *, heads: int, state_size: int, groups: int = 1):
    torch.manual_seed(seed)
    shape = Mamba2Shape(
        hidden_size=heads * MLEN, num_heads=heads, head_dim=MLEN,
        state_size=state_size, n_groups=groups, conv_kernel=4,
        chunk_size=16, seq_len=1,
    )
    x = torch.randn(heads, MLEN)
    state = torch.randn(heads, state_size, MLEN)
    b = torch.randn(groups, state_size)
    c = torch.randn(groups, state_size)
    dt = torch.rand(heads) * 0.5 + 0.1
    a = -torch.rand(heads) - 0.5
    d = torch.randn(heads)

    # The emitter takes dA = exp(A * dt) precomputed; the reference takes A.
    y_ref, state_ref = mamba2_recurrent_reference(
        x=x[None, None], dt=dt[None, None], A=a,
        B=b.reshape(1, 1, groups, state_size),
        C=c.reshape(1, 1, groups, state_size),
        D=d, initial_state=state[None],
    )
    return shape, {
        "x": x, "state": state, "b": b, "c": c, "d": d, "dt": dt,
        "da": torch.exp(a * dt),
        "y_ref": y_ref[0, 0], "state_ref": state_ref[0],
    }


@pytest.mark.parametrize(
    "seed,heads,state_size,groups",
    [(1, 1, 4, 1), (2, 2, 4, 1), (3, 4, 8, 2), (4, 2, 16, 1), (5, 6, 4, 3)],
)
def test_decode_step_matches_the_reference(seed, heads, state_size, groups):
    shape, c = _case(seed, heads=heads, state_size=state_size, groups=groups)
    y, state, _ = _Harness(shape).run(
        x=c["x"], state=c["state"], b=c["b"], c=c["c"],
        da=c["da"], dt=c["dt"], d=c["d"],
    )
    torch.testing.assert_close(y, c["y_ref"], rtol=5e-5, atol=5e-6)
    torch.testing.assert_close(state, c["state_ref"], rtol=5e-5, atol=5e-6)


def test_the_output_accumulator_is_cleared():
    """`y` is reused every token, so the contraction must zero it first. The
    harness seeds it with 7.5 -- on a fresh all-zero Machine a missing fill
    would be invisible."""
    shape, c = _case(6, heads=2, state_size=4)
    y, _, _ = _Harness(shape).run(
        x=c["x"], state=c["state"], b=c["b"], c=c["c"],
        da=c["da"], dt=c["dt"], d=c["d"],
    )
    torch.testing.assert_close(y, c["y_ref"], rtol=5e-5, atol=5e-6)


def test_the_sweeps_are_hardware_loops():
    """Two FMA sweeps per head -- the rank-1 update and the output contraction
    -- plus one broadcast FMA for the D skip. Each must be a single instruction
    inside a loop however large state_size is."""
    for state_size in (4, 8, 16):
        shape, _ = _case(7, heads=2, state_size=state_size)
        code = _Harness(shape).emit()
        body = [
            line.strip()
            for line in code.splitlines()
            if line.strip() and not line.strip().startswith(";")
        ]
        fmas = sum("V_FMA_VF" in line for line in body)
        assert fmas == 3 * shape.num_heads, (
            f"state_size={state_size}: expected 3 FMAs per head, got {fmas}"
        )


def test_emits_nothing_the_oracle_cannot_model():
    shape, c = _case(8, heads=2, state_size=4)
    code = _Harness(shape).emit()
    try:
        Machine(vlen=MLEN).run(code)
    except UnsupportedInstruction as exc:  # pragma: no cover - failure path
        pytest.fail(f"decode step emitted an unmodelled instruction: {exc}")
