"""KDA chunk primitives: the cumulative decay and the UT transform.

Both run through the ISA interpreter, so what is checked is the emitted
instructions rather than the Python that emitted them.

The chunked algebra these serve was verified against `kda_step` run
sequentially before any of it was lowered; `program_kda_chunk`'s module
docstring carries the derivation and the measurements. Two results from that
work are pinned here as tests, because they are the ones that are easy to
"simplify" back into a wrong answer:

* the decay weighting in `M` is not optional -- dropping it is wrong by 1.8e-1,
* the cumulative decay is a running product, not `exp` of a running sum.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.kda.shape import KdaShape  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.program_kda_chunk import (  # noqa: E402
    kda_chunk_cols,
    kda_chunk_rows,
    kda_max_chunk_for,
)
from compiler.aten.plena.program_kda_gates import kda_key_blocks  # noqa: E402
from compiler.aten.tests.isa_interpreter import (  # noqa: E402
    Machine,
    UnsupportedInstruction,
)

MLEN = 8


def _rows_up(n: int, mlen: int = MLEN) -> int:
    return ((n + mlen - 1) // mlen) * mlen


def _static(asm: str) -> int:
    return len([ln for ln in asm.splitlines() if ln.strip() and not ln.strip().startswith(";")])


def _shape(key_dim: int, mlen: int = MLEN) -> KdaShape:
    return KdaShape(
        hidden_size=8, num_heads=1, key_dim=key_dim, value_dim=mlen, conv_kernel=4
    )


# ---------------------------------------------------------------------------
# The cumulative decay.


def _run_cumprod(seed: int, key_dim: int, chunk: int):
    torch.manual_seed(seed)
    shape = _shape(key_dim)
    p = PlenaCompiler(mlen=MLEN, blen=2)
    blocks = kda_key_blocks(shape, MLEN)
    rows = kda_chunk_rows(shape, MLEN, chunk)
    cols = kda_chunk_cols(shape, MLEN)
    decay = p.alloc("decay", rows, cols)
    prev = p.alloc("prev", rows, cols)
    mark = len(p.get_code())
    p.kda_chunk_decay_cumprod_v0(decay=decay, prev=prev, chunk=chunk, shape=shape)
    code = p.get_code()[mark:]

    # Per-step decays, the range kda_decay_scalars_v0 produces: exp of something
    # in [gate_lower_bound, 0], so every factor is in [e^-5, 1].
    d = torch.exp(shape.gate_lower_bound * torch.sigmoid(torch.randn(chunk, key_dim)))
    m = Machine(vlen=MLEN, vram_words=1 << 16, fpram_words=1 << 13)
    # The key axis is on lanes: block `b` of timestep `t` is row `t` of column
    # block `b`, which the tile addressing places at `b * rows * MLEN + t * MLEN`.
    addr = lambda b, t: p.get_vram_tile_addr(decay.name, 0, b) + t * MLEN  # noqa: E731
    for t in range(chunk):
        for b in range(blocks):
            m.write_vram_row(addr(b, t), d[t, b * MLEN : (b + 1) * MLEN].tolist())
    m.run(code)
    got = torch.tensor([
        sum((m.read_vram_row(addr(b, t), MLEN) for b in range(blocks)), [])
        for t in range(chunk)
    ])
    return got, torch.cumprod(d, dim=0), code


@pytest.mark.parametrize(
    "seed,key_dim,chunk",
    [(1, MLEN, 2), (2, MLEN, 4), (3, MLEN, 16), (4, MLEN * 2, 16), (5, MLEN * 4, 8)],
)
def test_cumulative_decay_matches_a_running_product(seed, key_dim, chunk):
    got, want, _ = _run_cumprod(seed, key_dim, chunk)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)


def test_cumulative_decay_spans_every_key_block():
    """key_dim > mlen puts the key axis across column blocks. A scan that ran
    only over block 0 would leave the upper half holding the per-step decay --
    finite, plausible, wrong."""
    got, want, _ = _run_cumprod(6, MLEN * 2, 8)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-7)
    # The two halves decay independently; if they had been chained the second
    # would be the product of both.
    assert got.shape[1] == MLEN * 2


def test_cumulative_decay_is_not_a_plain_copy():
    """The scan must actually multiply. Guards against a version that emits the
    staging copy and drops the multiply, which leaves decay holding the
    per-step values -- correct at t=0 and wrong everywhere after."""
    got, want, _ = _run_cumprod(7, MLEN, 8)
    assert (got[-1] < got[0]).all(), "a cumulative product of factors < 1 must decrease"


# ---------------------------------------------------------------------------
# The UT transform.


def _run_ut(seed: int, chunk: int, mlen: int = MLEN, *, scale: float = 0.3):
    torch.manual_seed(seed)
    shape = _shape(mlen, mlen)
    p = PlenaCompiler(mlen=mlen, blen=2)
    up = _rows_up(chunk, mlen)
    m_tile = p.alloc("m", up, mlen)
    identity = p.alloc("ident", up, mlen)
    t_out = p.alloc("t_out", up, mlen)
    m_fp = p.fp_var("m_fp", size=mlen)
    beta_fp = p.fp_var("beta", size=up)
    consts = p.kda_fp_constants()
    mark = len(p.get_code())
    p.kda_ut_transform_v0(
        m=m_tile, identity=identity, beta_fp=beta_fp, t_out=t_out,
        m_fp=m_fp, consts=consts, chunk=chunk, shape=shape,
    )
    code = p.get_code()[mark:]

    m_dense = torch.tril(torch.randn(chunk, chunk) * scale, -1)
    beta = torch.sigmoid(torch.randn(chunk))
    want = torch.linalg.inv(
        torch.eye(chunk) + torch.diag(beta) @ m_dense
    ) @ torch.diag(beta)

    mac = Machine(vlen=mlen, vram_words=1 << 18, fpram_words=1 << 13)
    mb = p.get_vram_layout(m_tile.name).vram_base_addr
    ib = p.get_vram_layout(identity.name).vram_base_addr
    tb = p.get_vram_layout(t_out.name).vram_base_addr
    pad = [0.0] * (mlen - chunk)
    for i in range(chunk):
        mac.write_vram_row(mb + i * mlen, m_dense[i].tolist() + pad)
        mac.write_vram_row(
            ib + i * mlen, [1.0 if j == i else 0.0 for j in range(chunk)] + pad
        )
        # Garbage, so "the program wrote it" is distinguishable from "it was
        # already zero" -- a fresh Machine is all zeros.
        mac.write_vram_row(tb + i * mlen, [9.5] * mlen)
    mac.write_fpram(beta_fp.address, beta.tolist())
    mac.write_fpram(consts.zero.address, p.kda_fp_constant_values())
    mac.run(code)
    got = torch.tensor([mac.read_vram_row(tb + i * mlen, chunk) for i in range(chunk)])
    return got, want, code, mac


@pytest.mark.parametrize("seed,chunk,mlen", [(1, 2, 8), (2, 4, 8), (3, 8, 8), (4, 16, 16)])
def test_ut_transform_matches_a_dense_inverse(seed, chunk, mlen):
    got, want, _, _ = _run_ut(seed, chunk, mlen)
    torch.testing.assert_close(got, want, rtol=5e-5, atol=5e-6)


def test_ut_transform_ignores_m_on_and_above_the_diagonal():
    """The substitution reads only M[i, j] for j < i.

    Every other test feeds a strictly lower-triangular M, so the emitter
    could read the diagonal and still pass -- extending its sweep from
    range(i) to range(i + 1) left the whole file green. In the real
    pipeline M comes from a gram matrix whose diagonal is |k|^2, near 1,
    so a mask that is off by one row would be a real wrong answer rather than a
    harmless zero. Pin the contract instead of the accident.
    """
    torch.manual_seed(13)
    chunk, mlen = 8, 8
    shape = _shape(mlen, mlen)
    strict = torch.tril(torch.randn(chunk, chunk) * 0.3, -1)
    # Same strict lower triangle, but the diagonal and above are loud.
    polluted = strict + torch.triu(torch.randn(chunk, chunk) * 5.0 + 3.0)

    outs = []
    for m_dense in (strict, polluted):
        p = PlenaCompiler(mlen=mlen, blen=2)
        up = _rows_up(chunk, mlen)
        m_tile = p.alloc("m", up, mlen)
        identity = p.alloc("ident", up, mlen)
        t_out = p.alloc("t_out", up, mlen)
        m_fp = p.fp_var("m_fp", size=mlen)
        beta_fp = p.fp_var("beta", size=up)
        consts = p.kda_fp_constants()
        mark = len(p.get_code())
        p.kda_ut_transform_v0(
            m=m_tile, identity=identity, beta_fp=beta_fp, t_out=t_out,
            m_fp=m_fp, consts=consts, chunk=chunk, shape=shape,
        )
        code = p.get_code()[mark:]

        torch.manual_seed(13)
        beta = torch.sigmoid(torch.randn(chunk))
        mac = Machine(vlen=mlen, vram_words=1 << 18, fpram_words=1 << 13)
        mb = p.get_vram_layout(m_tile.name).vram_base_addr
        ib = p.get_vram_layout(identity.name).vram_base_addr
        tb = p.get_vram_layout(t_out.name).vram_base_addr
        pad = [0.0] * (mlen - chunk)
        for i in range(chunk):
            mac.write_vram_row(mb + i * mlen, m_dense[i].tolist() + pad)
            mac.write_vram_row(
                ib + i * mlen, [1.0 if j == i else 0.0 for j in range(chunk)] + pad
            )
            mac.write_vram_row(tb + i * mlen, [9.5] * mlen)
        mac.write_fpram(beta_fp.address, beta.tolist())
        mac.write_fpram(consts.zero.address, p.kda_fp_constant_values())
        mac.run(code)
        outs.append(torch.tensor([mac.read_vram_row(tb + i * mlen, chunk) for i in range(chunk)]))

    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)


def test_ut_transform_is_lower_triangular():
    """Forward substitution must never write above the diagonal."""
    got, _, _, _ = _run_ut(5, 8)
    assert torch.triu(got, diagonal=1).abs().max() < 1e-6


def test_ut_transform_diagonal_is_beta():
    """L is unitriangular, so T's diagonal is exactly beta. A missing identity
    row or a misapplied beta shows up here first."""
    torch.manual_seed(9)
    got, want, _, _ = _run_ut(9, 8)
    torch.testing.assert_close(torch.diagonal(got), torch.diagonal(want), rtol=5e-5, atol=5e-6)


def test_ut_transform_overwrites_its_destination():
    """t_out is seeded with 9.5. Anything left over means a row was not written."""
    got, _, _, _ = _run_ut(10, 8)
    assert (got.abs() < 5.0).all(), f"stale seed value survived: {got}"


def test_ut_transform_is_one_hardware_loop_per_row():
    """The substitution's inner sum walks j < i with the destination pinned, so
    it is a progression and must not unroll. Without this a change that broke
    the progression would still be numerically right and quadratically larger."""
    counts = {}
    for chunk in (4, 8, 16):
        _, _, code, _ = _run_ut(11, chunk, 16)
        counts[chunk] = code.count("V_FMA_VF")
    # One FMA per row that has predecessors, whatever the row length.
    assert counts == {4: 3, 8: 7, 16: 15}, counts


def test_ut_transform_emits_nothing_the_oracle_cannot_model():
    _, _, code, _ = _run_ut(12, 8)
    try:
        Machine(vlen=MLEN).run(code)
    except UnsupportedInstruction as exc:  # pragma: no cover - failure path
        pytest.fail(f"UT transform emitted an unmodelled instruction: {exc}")


# ---------------------------------------------------------------------------
# The chunk-size bound.


def test_rejects_a_chunk_whose_reciprocal_decay_overflows_bf16():
    """M is formed as a matmul of k*A against k/A, and 1/A reaches
    exp(chunk * |gate_lower_bound|). bf16 tops out at exp(88.7), so at Kimi K3's
    -5 the last chunk that works is 17. The failure is silent -- inf, then nan
    through the whole solve -- so it is refused rather than warned about."""
    p = PlenaCompiler(mlen=64, blen=2)
    shape = KdaShape(
        hidden_size=8, num_heads=1, key_dim=64, value_dim=64, conv_kernel=4
    )
    assert shape.gate_lower_bound == -5.0
    for ok in (1, 16, 17):
        p.kda_chunk_check_range(ok, shape)
    for bad in (18, 32, 64):
        with pytest.raises(ValueError, match="past bf16"):
            p.kda_chunk_check_range(bad, shape)


def test_the_chunk_bound_scales_with_the_gate_lower_bound():
    assert kda_max_chunk_for(-5.0) == 17
    assert kda_max_chunk_for(-2.5) == 35
    assert kda_max_chunk_for(-10.0) == 8
    with pytest.raises(ValueError, match="must be negative"):
        kda_max_chunk_for(0.0)


def test_rejects_a_chunk_wider_than_mlen():
    """The [chunk, chunk] tiles here are one row per timestep, so the row must
    hold the whole chunk in a single block."""
    p = PlenaCompiler(mlen=MLEN, blen=2)
    shape = _shape(MLEN)
    up = _rows_up(MLEN * 2)
    with pytest.raises(ValueError, match="exceeds mlen"):
        p.kda_ut_transform_v0(
            m=p.alloc("m", up, MLEN), identity=p.alloc("i", up, MLEN),
            beta_fp=p.fp_var("b", size=up), t_out=p.alloc("t", up, MLEN),
            m_fp=p.fp_var("mf", size=MLEN), consts=p.kda_fp_constants(),
            chunk=MLEN * 2, shape=shape,
        )


def test_rejects_an_m_fp_sized_to_chunk_rather_than_mlen():
    """S_MAP_FP_V moves a whole row, so m_fp must be mlen slots. Sizing it to
    chunk writes past the allocation -- the same trap beta_fp had."""
    p = PlenaCompiler(mlen=MLEN, blen=2)
    shape = _shape(MLEN)
    with pytest.raises(ValueError, match="whole row"):
        p.kda_ut_transform_v0(
            m=p.alloc("m", MLEN, MLEN), identity=p.alloc("i", MLEN, MLEN),
            beta_fp=p.fp_var("b", size=MLEN), t_out=p.alloc("t", MLEN, MLEN),
            m_fp=p.fp_var("mf", size=4), consts=p.kda_fp_constants(),
            chunk=4, shape=shape,
        )


def test_rejects_t_out_aliased_to_m():
    p = PlenaCompiler(mlen=MLEN, blen=2)
    shape = _shape(MLEN)
    m = p.alloc("m", MLEN, MLEN)
    with pytest.raises(ValueError, match="must not alias m"):
        p.kda_ut_transform_v0(
            m=m, identity=p.alloc("i", MLEN, MLEN),
            beta_fp=p.fp_var("b", size=MLEN), t_out=m,
            m_fp=p.fp_var("mf", size=MLEN), consts=p.kda_fp_constants(),
            chunk=4, shape=shape,
        )
