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
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.mamba2.reference import (  # noqa: E402
    mamba2_recurrent_reference,
)
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.program_mamba_common import Mamba2Shape  # noqa: E402
from compiler.aten.plena.program_ssm_recurrent import MambaDecodeInvocation  # noqa: E402
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


def test_static_batch_executes_private_state_for_every_request():
    """Batched decode is explicit static repetition, not hidden batch-1 code."""

    batch = 4
    cases = [_case(100 + i, heads=2, state_size=4) for i in range(batch)]
    shape = replace(cases[0][0], batch_size=batch)
    p = PlenaCompiler(mlen=MLEN, blen=2)
    items = []
    for i in range(batch):
        n = shape.state_size
        items.append(
            MambaDecodeInvocation(
                state=p.alloc(f"state{i}", _rows_up(shape.num_heads * n), MLEN),
                x=p.alloc(f"x{i}", _rows_up(shape.num_heads), MLEN),
                b_fp=p.fp_var(f"b{i}", size=shape.n_groups * n),
                c_fp=p.fp_var(f"c{i}", size=shape.n_groups * n),
                da_fp=p.fp_var(f"da{i}", size=shape.num_heads),
                dt_fp=p.fp_var(f"dt{i}", size=shape.num_heads),
                d_fp=p.fp_var(f"d{i}", size=shape.num_heads),
                y=p.alloc(f"y{i}", _rows_up(shape.num_heads), MLEN),
                scratch=p.alloc(f"scratch{i}", MLEN, MLEN),
            )
        )
    consts = p.mamba_fp_constants()
    mark = len(p.get_code())
    p.ssm_decode_batch_v0(invocations=items, shape=shape, consts=consts)
    code = p.get_code()[mark:]
    assert code.count("static Mamba batch request=") == batch

    m = Machine(vlen=MLEN, vram_words=1 << 18, fpram_words=1 << 14)
    base = lambda var: p.get_vram_layout(var.name).vram_base_addr  # noqa: E731
    for item, (_, case) in zip(items, cases):
        for h in range(shape.num_heads):
            m.write_vram_row(base(item.x) + h * MLEN, case["x"][h].tolist())
            m.write_vram_row(base(item.y) + h * MLEN, [7.5] * MLEN)
            for n in range(shape.state_size):
                m.write_vram_row(
                    base(item.state) + (h * shape.state_size + n) * MLEN,
                    case["state"][h, n].tolist(),
                )
        m.write_fpram(item.b_fp.address, case["b"].flatten().tolist())
        m.write_fpram(item.c_fp.address, case["c"].flatten().tolist())
        m.write_fpram(item.da_fp.address, case["da"].tolist())
        m.write_fpram(item.dt_fp.address, case["dt"].tolist())
        m.write_fpram(item.d_fp.address, case["d"].tolist())
    m.write_fpram(consts.zero.address, p.mamba_fp_constant_values(shape))
    m.run(code)

    for item, (_, case) in zip(items, cases):
        got_y = torch.tensor(
            [m.read_vram_row(base(item.y) + h * MLEN, MLEN) for h in range(shape.num_heads)]
        )
        got_state = torch.tensor(
            [
                [
                    m.read_vram_row(
                        base(item.state) + (h * shape.state_size + n) * MLEN,
                        MLEN,
                    )
                    for n in range(shape.state_size)
                ]
                for h in range(shape.num_heads)
            ]
        )
        torch.testing.assert_close(got_y, case["y_ref"], rtol=5e-5, atol=5e-6)
        torch.testing.assert_close(got_state, case["state_ref"], rtol=5e-5, atol=5e-6)


def test_static_mamba_batch_rejects_shared_state_storage():
    shape, _ = _case(200, heads=1, state_size=4)
    p = PlenaCompiler(mlen=MLEN, blen=2)
    item = MambaDecodeInvocation(
        state=p.alloc("state", MLEN, MLEN),
        x=p.alloc("x", MLEN, MLEN),
        b_fp=p.fp_var("b", size=shape.state_size),
        c_fp=p.fp_var("c", size=shape.state_size),
        da_fp=p.fp_var("da", size=1),
        dt_fp=p.fp_var("dt", size=1),
        d_fp=p.fp_var("d", size=1),
        y=p.alloc("y", MLEN, MLEN),
        scratch=p.alloc("scratch", MLEN, MLEN),
    )
    consts = p.mamba_fp_constants()
    with pytest.raises(ValueError, match="state tensors must be request-private"):
        p.ssm_decode_batch_v0(
            invocations=[item, item],
            shape=replace(shape, batch_size=2),
            consts=consts,
        )
