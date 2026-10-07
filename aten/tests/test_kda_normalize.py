"""KDA's L2 normalisation of q and k, executed against the reference.

`q` and `k` are key-width, and Kimi K3 is `key_dim = 128` against a default
`mlen = 64`. The norm contracts over the key axis, so the two halves of one `q`
share a single norm -- this is the one place in KDA where the column-block
folding used everywhere else does *not* apply, and normalising each row on its
own would divide each half by its own partial norm.

The reference is `reference.py`'s `x * rsqrt(sum(x^2) + 1e-6)`, matching
FlashKDA's recurrent kernel. `torch.nn.functional.normalize` clamps instead and
does not agree.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.tests.isa_interpreter import Machine  # noqa: E402

MLEN = 8
EPS = 1.0e-6


def _rows_up(n: int) -> int:
    return ((n + MLEN - 1) // MLEN) * MLEN


def _reference(x: torch.Tensor) -> torch.Tensor:
    """[vectors, width] -> normalised, exactly as reference.py does it."""
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + EPS)


class _NormHarness:
    def __init__(self, vectors: int, blocks: int):
        self.vectors, self.blocks = vectors, blocks
        p = self.prog = PlenaCompiler(mlen=MLEN, blen=2)
        rows = _rows_up(vectors * blocks)
        self.vec = p.alloc("vec", rows, MLEN)
        self.sq = p.alloc("sq", rows, MLEN)
        self.part = p.fp_var("part", size=vectors * blocks)
        self.acc = p.fp_var("acc", size=vectors)
        self.consts = p.kda_fp_constants()
        self.mark = len(p.get_code())

    def _base(self, var) -> int:
        return self.prog.get_vram_layout(var.name).vram_base_addr

    def run(self, x: torch.Tensor):
        """x is [vectors, blocks * mlen]."""
        self.prog.kda_l2_normalize_blocked_v0(
            self.vec, vectors=self.vectors, blocks=self.blocks,
            sq_scratch=self.sq, part_fp=self.part, acc_fp=self.acc,
            consts=self.consts,
        )
        code = self.prog.get_code()[self.mark :]

        m = Machine(vlen=MLEN)
        for v in range(self.vectors):
            for c in range(self.blocks):
                m.write_vram_row(
                    self._base(self.vec) + (v * self.blocks + c) * MLEN,
                    x[v, c * MLEN : (c + 1) * MLEN].tolist(),
                )
        m.write_fpram(self.consts.zero.address, PlenaCompiler.kda_fp_constant_values())
        m.run(code)

        return torch.tensor(
            [
                [
                    val
                    for c in range(self.blocks)
                    for val in m.read_vram_row(
                        self._base(self.vec) + (v * self.blocks + c) * MLEN, MLEN
                    )
                ]
                for v in range(self.vectors)
            ],
            dtype=torch.float32,
        ), m


@pytest.mark.parametrize(
    "seed,vectors,blocks",
    [
        (1, 1, 1),   # degenerate: one vector, one block
        (2, 3, 1),   # several vectors, still one block
        (3, 1, 2),   # THE case: one vector spanning two blocks
        (4, 4, 2),   # several vectors, two blocks -- Kimi K3's shape
        (5, 2, 3),   # three blocks
    ],
)
def test_blocked_normalize_matches_the_reference(seed, vectors, blocks):
    torch.manual_seed(seed)
    x = 0.7 * torch.randn(vectors, blocks * MLEN)
    got, _ = _NormHarness(vectors, blocks).run(x)
    torch.testing.assert_close(got, _reference(x), rtol=2e-5, atol=2e-6)


def test_both_halves_share_one_norm():
    """The whole point. If each row were normalised by its own partial norm the
    result would still be finite and unit-ish per half -- this pins the *joint*
    norm by comparing against a per-half normalisation and requiring they differ.
    """
    torch.manual_seed(11)
    x = 0.7 * torch.randn(1, 2 * MLEN)
    # make the halves very unequal, so a per-half norm is clearly wrong
    x[0, MLEN:] *= 6.0

    got, _ = _NormHarness(1, 2).run(x)
    torch.testing.assert_close(got, _reference(x), rtol=2e-5, atol=2e-6)

    per_half = torch.cat(
        [_reference(x[:, :MLEN]), _reference(x[:, MLEN:])], dim=-1
    )
    assert not torch.allclose(got, per_half, rtol=1e-2, atol=1e-2), (
        "each half was normalised by its own norm"
    )

    # and the joint norm really is 1 (up to eps)
    assert got.square().sum().item() == pytest.approx(1.0, abs=2e-4)


def test_vectors_do_not_share_a_norm():
    """Each vector gets its own norm; a fold that summed across vectors would
    give every one the same scale."""
    torch.manual_seed(12)
    x = 0.7 * torch.randn(3, 2 * MLEN)
    x[1] *= 5.0
    got, _ = _NormHarness(3, 2).run(x)
    for v in range(3):
        assert got[v].square().sum().item() == pytest.approx(1.0, abs=2e-4)


def test_rejects_undersized_fpram_and_aliased_scratch():
    p = PlenaCompiler(mlen=MLEN, blen=2)
    vec = p.alloc("v", MLEN, MLEN)
    sq = p.alloc("s", MLEN, MLEN)
    consts = p.kda_fp_constants()
    ok = dict(vectors=2, blocks=2, sq_scratch=sq, consts=consts)

    with pytest.raises(ValueError, match="part_fp"):
        p.kda_l2_normalize_blocked_v0(
            vec, part_fp=p.fp_var("p1", size=3), acc_fp=p.fp_var("a1", size=2), **ok
        )
    with pytest.raises(ValueError, match="acc_fp"):
        p.kda_l2_normalize_blocked_v0(
            vec, part_fp=p.fp_var("p2", size=4), acc_fp=p.fp_var("a2", size=1), **ok
        )
    with pytest.raises(ValueError, match="distinct"):
        p.kda_l2_normalize_blocked_v0(
            vec, part_fp=p.fp_var("p3", size=4), acc_fp=p.fp_var("a3", size=2),
            **{**ok, "sq_scratch": vec},
        )


# ---------------------------------------------------------------------------
# first_row: normalising one head's slice of a tile that holds every head.
#
# The mixer streams per head to keep FPRAM bounded, and q/k arrive as one tile
# covering all heads. Copying a head's rows out first would cost key_blocks row
# copies per head per tensor; first_row addresses them in place instead.
# ---------------------------------------------------------------------------


class _SliceHarness:
    def __init__(self, heads: int, blocks: int):
        self.heads, self.blocks = heads, blocks
        p = self.prog = PlenaCompiler(mlen=MLEN, blen=2)
        rows = _rows_up(heads * blocks)
        self.vec = p.alloc("vec", rows, MLEN)
        self.sq = p.alloc("sq", rows, MLEN)
        self.part = p.fp_var("part", size=blocks)
        self.acc = p.fp_var("acc", size=1)
        self.consts = p.kda_fp_constants()
        self.mark = len(p.get_code())

    def _base(self, var) -> int:
        return self.prog.get_vram_layout(var.name).vram_base_addr

    def run(self, x: torch.Tensor, head: int):
        """x is [heads, blocks*mlen]; normalise only `head`'s rows in place."""
        self.prog.kda_l2_normalize_blocked_v0(
            self.vec, vectors=1, blocks=self.blocks, sq_scratch=self.sq,
            part_fp=self.part, acc_fp=self.acc, consts=self.consts,
            first_row=head * self.blocks,
        )
        code = self.prog.get_code()[self.mark :]

        m = Machine(vlen=MLEN)
        for h in range(self.heads):
            for c in range(self.blocks):
                m.write_vram_row(
                    self._base(self.vec) + (h * self.blocks + c) * MLEN,
                    x[h, c * MLEN : (c + 1) * MLEN].tolist(),
                )
        m.write_fpram(self.consts.zero.address, PlenaCompiler.kda_fp_constant_values())
        m.run(code)

        return torch.tensor(
            [
                [
                    val
                    for c in range(self.blocks)
                    for val in m.read_vram_row(
                        self._base(self.vec) + (h * self.blocks + c) * MLEN, MLEN
                    )
                ]
                for h in range(self.heads)
            ],
            dtype=torch.float32,
        )


@pytest.mark.parametrize("heads,blocks", [(3, 1), (3, 2), (2, 3)])
def test_first_row_normalises_only_that_head(heads, blocks):
    torch.manual_seed(31)
    x = 0.7 * torch.randn(heads, blocks * MLEN)
    for head in range(heads):
        got = _SliceHarness(heads, blocks).run(x, head)
        # the selected head is normalised
        torch.testing.assert_close(
            got[head], _reference(x[head : head + 1])[0], rtol=2e-5, atol=2e-6
        )
        # every other head is untouched, bit for bit
        for other in range(heads):
            if other != head:
                torch.testing.assert_close(got[other], x[other], rtol=0, atol=0)


def test_first_row_uses_its_own_slice_not_row_zero():
    """A first_row that is accepted but ignored would normalise head 0 every
    time. Making the heads' magnitudes differ makes that visible."""
    torch.manual_seed(32)
    heads, blocks = 3, 2
    x = 0.7 * torch.randn(heads, blocks * MLEN)
    x[0] *= 8.0
    got = _SliceHarness(heads, blocks).run(x, 2)
    torch.testing.assert_close(got[0], x[0], rtol=0, atol=0)
    assert got[2].square().sum().item() == pytest.approx(1.0, abs=2e-4)


def test_sq_scratch_must_reach_past_first_row():
    """`vec`'s own bound fires first when both are short, so a too-small
    sq_scratch needs a vec that is big enough. Without this the sq_scratch
    check could be written against `vectors * blocks` alone and every test
    still passed -- while the mixer, which walks first_row across heads, wrote
    the squares of the last head past the end of the scratch."""
    p = PlenaCompiler(mlen=MLEN, blen=2)
    vec = p.alloc("v", MLEN * 2, MLEN)
    sq = p.alloc("s", MLEN, MLEN)
    with pytest.raises(ValueError, match="sq_scratch must match"):
        p.kda_l2_normalize_blocked_v0(
            vec, vectors=1, blocks=2, sq_scratch=sq,
            part_fp=p.fp_var("p", size=2), acc_fp=p.fp_var("a", size=1),
            consts=p.kda_fp_constants(), first_row=MLEN,
        )


def test_the_squares_copy_is_only_as_long_as_the_slice():
    """Copying from row 0 up to the last live row makes the cost O(first_row),
    and the mixer moves first_row across every head -- quadratic in head count.
    Two slices of the same width must cost the same however far in they sit."""
    counts = {}
    for first_row in (0, MLEN * 4, MLEN * 32):
        p = PlenaCompiler(mlen=MLEN, blen=2)
        vec = p.alloc("v", MLEN * 40, MLEN)
        sq = p.alloc("s", MLEN * 40, MLEN)
        consts = p.kda_fp_constants()
        mark = len(p.get_code())
        p.kda_l2_normalize_blocked_v0(
            vec, vectors=1, blocks=2, sq_scratch=sq,
            part_fp=p.fp_var(f"p{first_row}", size=2),
            acc_fp=p.fp_var(f"a{first_row}", size=1),
            consts=consts, first_row=first_row,
        )
        counts[first_row] = len(p.get_code()) - mark

    # A row of block_copy costs about 80 instructions, so a copy that started
    # at row 0 would put 256 * 80 between the two ends. What is left is the
    # wider immediate for a further-out address, which is a constant per call.
    spread = counts[MLEN * 32] - counts[0]
    assert spread < 100, f"cost still grows with first_row: {counts}"


def test_first_row_bounds_are_checked():
    p = PlenaCompiler(mlen=MLEN, blen=2)
    vec = p.alloc("v", MLEN, MLEN)
    sq = p.alloc("s", MLEN, MLEN)
    with pytest.raises(ValueError, match="rows"):
        p.kda_l2_normalize_blocked_v0(
            vec, vectors=1, blocks=2, sq_scratch=sq,
            part_fp=p.fp_var("p", size=2), acc_fp=p.fp_var("a", size=1),
            consts=p.kda_fp_constants(), first_row=MLEN,
        )
