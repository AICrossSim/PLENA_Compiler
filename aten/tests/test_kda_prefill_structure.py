"""Structural coverage for `kda_prefill_chunk_v0`.

The prefill layer had **no pytest coverage at all**. Its only caller anywhere
was `transactional_emulator/testbench/kda/kda_stage_test.py`, a hand-run script
in no CI job, and the ISA interpreter models no matmul so numerical coverage in
pytest is not available.

What *is* available is the emitted assembly. Every projection announces its
operands and tile indices in a comment:

    ; VRAM Sub Projection T To: k[0][:] @ kda_k_tilde_spill[0][:]^T -> gram[0][0]
    ; VRAM Sub Projection To:   err_t[0][:] @ kda_k_end_spill[:][0] -> contrib[0][0]

so operand order, the `_to` versus `_T_to` choice, the block indices, and the
spill/prefetch pairing can all be pinned without executing anything. Those are
exactly the mistakes that produce a finite, plausible, wrong answer -- a reviewer
found that swapping two block indices in one projection passed bit-identically,
because at the only shape ever tested both were 0.

Numerical correctness stays with the emulator stage test. This file is what
fails in CI when the composition changes.
"""

from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.kda.shape import KdaShape  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.instruction_stream import (  # noqa: E402
    dynamic_count,
    static_count,
)
from compiler.aten.plena.program_kda_prefill import (  # noqa: E402
    KdaPrefillBuffers,
    kda_prefill_state_transpose_shapes,
    kda_prefill_tile_shapes,
)

MLEN = 64

#: `; VRAM Sub Projection [T ]To: A[i][:] @ B[...] -> C[r][c]`
_PROJ = re.compile(
    r";\s*VRAM Sub Projection (T )?To:\s*"
    r"(?P<vram>\w+)\[(?P<vi>\d+)\]\[:\]\s*@\s*"
    r"(?P<mram>\w+)(?P<mslot>\[[^\]]*\]\[[^\]]*\])\^?T?\s*->\s*"
    r"(?P<target>\w+)\[(?P<tr>\d+)\]\[(?P<tc>\d+)\]"
)
_STORE = re.compile(r";\s*Store (?P<tile>\w+) from VRAM to HBM")


def _build(
    chunk: int = 16,
    mlen: int = MLEN,
    key_dim: int | None = None,
    value_dim: int | None = None,
    emit: bool = True,
):
    """Compile one chunk, allocating every tile at the shape the emitter demands.

    Allocation goes through `kda_prefill_tile_shapes` rather than a local
    `mlen x mlen` for everything, which is what it used to be. That divergence
    is how a spill came to prefetch past its allocation: the emitter's idea of
    the tiles and the caller's were written separately and only overlapped at
    the one shape anyone ran.
    """
    shape = KdaShape(
        hidden_size=mlen, num_heads=1,
        key_dim=key_dim or mlen, value_dim=value_dim or mlen, conv_kernel=4,
    )
    p = PlenaCompiler(mlen=mlen, blen=4)
    # The build-level KV precision check reads the active TOML and the shipped
    # one is MX; that path has its own tests in test_spill_precision_guard.py.
    p._bf16_kv_checked = True
    want = kda_prefill_tile_shapes(shape, mlen, chunk)
    tiles = {n: p.alloc(n, r, c, strict=False) for n, (r, c) in want.items()}
    buffers = KdaPrefillBuffers(
        **tiles,
        beta_fp=p.fp_var("beta", size=mlen), m_fp=p.fp_var("m", size=mlen),
        output_scale_fp=p.fp_var("os", size=1), consts=p.kda_fp_constants(),
    )
    mark = len(p.get_code())
    if emit:
        p.kda_prefill_chunk_v0(buffers=buffers, chunk=chunk, shape=shape)
    return p, buffers, p.get_code()[mark:]


def _projections(code: str):
    out = []
    for line in code.splitlines():
        m = _PROJ.search(line)
        if m:
            out.append({
                "transposed": bool(m.group(1)),
                "vram": m.group("vram"), "vi": int(m.group("vi")),
                "mram": m.group("mram"), "mslot": m.group("mslot"),
                "target": m.group("target"),
                "tr": int(m.group("tr")), "tc": int(m.group("tc")),
            })
    return out


#: The seven products, in emission order. Each entry is
#: (vram operand, mram operand, target, transposed).
#:
#: `transposed` is `_T_to`, which contracts the MRAM operand's *lanes*; the
#: plain form contracts its *rows*. The last one is the only plain form, and it
#: is the one that reads past a spill's live rows -- see
#: `kda_prefill_spill_v0` for why that matters.
EXPECTED = [
    ("k",       "kda_k_tilde_spill", "gram",            True),   # M   = k_hat  @ k_tilde^T
    ("q",       "kda_k_tilde_spill", "readout",         True),   # N   = q_hat  @ k_tilde^T
    ("state",   "kda_k_hat_spill",   "contrib",         True),   # S_0 @ k_hat^T
    ("v_t",     "kda_t_spill",       "err_t",           True),   # E^T = W^T    @ T^T
    ("q",       "kda_state_spill",   "out",             True),   # q_hat @ S_0^T
    ("readout", "kda_err_spill",     "readout_contrib", True),   # N @ E
    ("err_t",   "kda_k_end_spill",   "state_contrib",   False),  # E^T @ k_end
]


def test_the_seven_products_and_their_operand_order():
    """Operand order, target, and `_to` versus `_T_to` for every product.

    Reversing any product's operands, or swapping the two forms, is a finite
    plausible wrong answer that only the emulator would otherwise catch.
    """
    _, _, code = _build()
    got = [(p["vram"], p["mram"], p["target"], p["transposed"]) for p in _projections(code)]
    assert got == EXPECTED, f"\ngot      {got}\nexpected {EXPECTED}"


def test_the_state_tail_product_contracts_over_mram_rows():
    """`E^T @ k_end` must be the plain form, and it must be the only one.

    `_T_to` contracts the MRAM operand's lanes; the plain form contracts its
    rows. Here the contraction is over *time*, which is `k_end`'s row axis, so
    `_T_to` would contract over key instead -- shapes match, answer does not.
    """
    _, _, code = _build()
    plain = [p for p in _projections(code) if not p["transposed"]]
    assert len(plain) == 1, f"expected exactly one plain projection, got {plain}"
    assert plain[0]["mram"] == "kda_k_end_spill"


def test_every_spill_is_stored_before_the_projection_that_reads_it():
    """A prefetch of a region written later reads the previous chunk's data --
    silently, because the regions are reused."""
    _, _, code = _build()
    order, seen = [], set()
    for line in code.splitlines():
        m = _STORE.search(line)
        if m:
            order.append(m.group("tile"))
        p = _PROJ.search(line)
        if p and p.group("mram").startswith("kda_"):
            seen.add(p.group("mram"))
            spilled = {f"kda_{t}_spill" for t in order} | {
                "kda_k_tilde_spill" if "k_tilde" in order else "",
                "kda_k_hat_spill" if "k" in order else "",
                "kda_t_spill" if "t_mat" in order else "",
                "kda_state_spill" if "state" in order else "",
                "kda_err_spill" if "err_t" in order else "",
                "kda_k_end_spill" if "k_end" in order else "",
            }
            assert p.group("mram") in spilled, (
                f"{p.group('mram')} is read before it is stored; stores so far: {order}"
            )
    assert len(seen) == 6, f"expected six distinct spills, saw {sorted(seen)}"


def test_each_spill_zeroes_its_tail_before_storing():
    """The prefetch pulls a whole mlen x mlen block, more than any chunk's live
    data. Each spill owns its tail so the read is defined regardless of what the
    tile held before -- the regions are reused across chunks.

    This is not observable numerically today: every spilled tile is freshly
    allocated, so its tail is already zero, and the state projection's other
    operand happens to be zero there too. That is precisely the accident this
    pins, so that it stays an invariant rather than a coincidence.
    """
    _, _, code = _build()
    lines = code.splitlines()
    stores = [(i, _STORE.search(ln).group("tile"))
              for i, ln in enumerate(lines) if _STORE.search(ln)]
    assert len(stores) == 6, [t for _, t in stores]
    # `state` and `err_t` are value-width, so at value_dim == mlen they have no
    # tail and correctly emit no fill. The other four are chunk- or key-width.
    needs_tail = {"k_tilde", "k", "t_mat", "k_end"}
    for i, tile in stores:
        window = "\n".join(lines[max(0, i - 40) : i])
        has_fill = "VRAM Fill Zero" in window
        if tile in needs_tail:
            assert has_fill, f"{tile} is spilled without zeroing its tail first"


def test_the_broadcast_is_logarithmic_not_linear():
    """Copying one row into N used to be N copies -- 64 at mlen 64, 128 at Kimi's
    value_dim. Doubling the filled span makes it ceil(log2(N)) + 1.

    Asserted by scaling rather than by a threshold: the state's decay is
    broadcast across `value_dim` rows, so a linear form grows with `mlen` and a
    logarithmic one barely does. A fixed number would only pin today's shape.
    """
    counts = {}
    for mlen in (16, 32, 64):
        _, _, code = _build(chunk=8, mlen=mlen)
        counts[mlen] = code.count("VRAM Fill Zero")
    growth = counts[64] - counts[16]
    assert growth <= 8, (
        f"fill count grows with mlen ({counts}); a linear broadcast would add "
        f"one per row, i.e. 48 across this range"
    )


@pytest.mark.parametrize("key_dim,value_dim", [(8, 8), (16, 8), (8, 16), (16, 16), (32, 16)])
def test_the_seven_products_survive_every_block_count(key_dim, value_dim):
    """The composition must not change when an axis stops fitting one block.

    `key_dim > mlen` was refused outright until the key axis moved onto lanes;
    Kimi K3 is 128 against mlen 64 on both axes. What multi-block changes is how
    many times each product is emitted, not which products there are or which
    way round their operands go -- so the distinct product list is invariant and
    is checked as one.
    """
    _, _, code = _build(chunk=4, mlen=8, key_dim=key_dim, value_dim=value_dim)
    got = [(x["vram"], x["mram"], x["target"], x["transposed"]) for x in _projections(code)]
    seen, distinct = set(), []
    for entry in got:
        if entry not in seen:
            seen.add(entry)
            distinct.append(entry)
    assert distinct == EXPECTED, f"\ngot      {distinct}\nexpected {EXPECTED}"


def test_multi_block_axes_emit_one_projection_per_block_pair():
    """Each product is emitted once per (row block, column block) of its target.

    Pins the block *count*, which is what the invariant product list above
    cannot see. `state @ k_hat^T` writes `[value_dim, chunk]`, so it runs once
    per value block; `q_hat @ S_0^T` writes `[chunk, value_dim]`, once per value
    block too; `E^T @ k_end` writes `[value_dim, key_dim]`, so value blocks
    times key blocks. Getting these wrong leaves part of the target holding
    whatever it held before -- zeros on a fresh tile, so silent.
    """
    _, _, code = _build(chunk=4, mlen=8, key_dim=16, value_dim=16)
    counts = {}
    for x in _projections(code):
        counts[(x["vram"], x["mram"])] = counts.get((x["vram"], x["mram"]), 0) + 1
    # key_blocks = value_blocks = 2, t_blocks = 1 at chunk 4.
    assert counts[("k", "kda_k_tilde_spill")] == 1        # gram is [chunk, chunk]
    assert counts[("q", "kda_k_tilde_spill")] == 1
    assert counts[("state", "kda_k_hat_spill")] == 2      # value blocks
    assert counts[("v_t", "kda_t_spill")] == 2            # value blocks
    assert counts[("q", "kda_state_spill")] == 2          # value blocks
    assert counts[("readout", "kda_err_spill")] == 2      # value blocks
    assert counts[("err_t", "kda_k_end_spill")] == 4      # value x key blocks


def test_multi_block_state_update_covers_every_target_block():
    """`E^T @ k_end` must write all four (value, key) blocks of the state
    accumulator, each exactly once. A loop that ran only the diagonal, or that
    reused one target index, leaves half the state stale."""
    _, _, code = _build(chunk=4, mlen=8, key_dim=16, value_dim=16)
    tail = [(x["tr"], x["tc"]) for x in _projections(code)
            if x["mram"] == "kda_k_end_spill"]
    assert sorted(tail) == [(0, 0), (0, 1), (1, 0), (1, 1)], tail


def test_refuses_a_tile_whose_width_does_not_match_its_axis():
    """The width is what tells a projection how far its contraction runs, so a
    tile one block too narrow contracts over half the key axis and returns a
    finite wrong answer. Checked exactly, not as a lower bound."""
    p, buffers, _ = _build(chunk=4, mlen=8, key_dim=16, value_dim=8, emit=False)
    shape = KdaShape(hidden_size=8, num_heads=1, key_dim=16, value_dim=8, conv_kernel=4)
    buffers.k_tilde = p.alloc("narrow", 8, 8, strict=False)
    with pytest.raises(ValueError, match="columns wide"):
        p.kda_prefill_chunk_v0(buffers=buffers, chunk=4, shape=shape)


def test_refuses_a_chunk_wider_than_mlen():
    """At mlen 64 the bf16 range check fires first -- the largest chunk it
    allows is 17 -- so this needs a small mlen to reach the width check at all."""
    with pytest.raises(ValueError, match="exceeds mlen"):
        _build(chunk=16, mlen=8, key_dim=8)


def test_refuses_a_tile_that_is_not_mlen_rows():
    """Every projection writes a whole mlen x mlen block and every spill is
    prefetched as one, so a tile that is either must be mlen rows tall however
    few rows carry data. Sizing them to `chunk` is what let the prefetch run
    past the allocation."""
    p, buffers, _ = _build(emit=False)
    buffers.k_end = p.alloc("short", 16, MLEN, strict=False)
    with pytest.raises(ValueError, match="rows"):
        p.kda_prefill_chunk_v0(
            buffers=buffers, chunk=16,
            shape=KdaShape(
                hidden_size=MLEN, num_heads=1, key_dim=MLEN,
                value_dim=MLEN, conv_kernel=4,
            ),
        )


#: Measured 2026-08-26, mlen 64, one head, keyed by (chunk, key_dim, value_dim).
#: Set ~10% above measured; raising one needs a line in the plan saying why.
#:
#: The single-block row is what it was before the key axis moved onto lanes
#: (1476 against a 1500 measurement at chunk 16), so the layout change is free at
#: the shape that already worked. The multi-block rows are new: they are what
#: Kimi K3 actually costs, 2426 at its 128 x 128 head against 1476 for a 64 x 64
#: one -- 1.6x the instructions for 4x the state.
PREFILL_STATIC_MAX = {
    (4, 64, 64): 900,    (4, 128, 64): 1370,   (4, 128, 128): 1620,
    (8, 64, 64): 1150,   (8, 128, 64): 1710,   (8, 128, 128): 1980,
    (16, 64, 64): 1650,  (16, 128, 64): 2380,  (16, 128, 128): 2670,
}


@pytest.mark.parametrize("shape_key", sorted(PREFILL_STATIC_MAX))
def test_static_instruction_budget(shape_key):
    chunk, key_dim, value_dim = shape_key
    _, _, code = _build(chunk=chunk, key_dim=key_dim, value_dim=value_dim)
    static = static_count(code)
    print(f"KDA_PREFILL chunk={chunk} key={key_dim} value={value_dim} static={static}")
    assert static <= PREFILL_STATIC_MAX[shape_key], (
        f"{static} static instructions at chunk {chunk}, key_dim {key_dim}, "
        f"value_dim {value_dim} exceeds {PREFILL_STATIC_MAX[shape_key]}"
    )


#: Issued instructions, measured 2026-08-27 at `mlen` 64, one head. Set ~15%
#: above. These are the numbers that price the kernel; `PREFILL_STATIC_MAX`
#: above prices the binary and the two do not scale together.
PREFILL_DYNAMIC_MAX = {
    (4, 64, 64): 21_800,   (4, 128, 64): 29_800,   (4, 128, 128): 50_500,
    (8, 64, 64): 22_200,   (8, 128, 64): 30_400,   (8, 128, 128): 51_000,
    (16, 64, 64): 23_200,  (16, 128, 64): 31_700,  (16, 128, 128): 52_400,
}


@pytest.mark.parametrize("shape_key", sorted(PREFILL_DYNAMIC_MAX))
def test_dynamic_instruction_budget(shape_key):
    chunk, key_dim, value_dim = shape_key
    _, _, code = _build(chunk=chunk, key_dim=key_dim, value_dim=value_dim)
    dynamic = dynamic_count(code)
    print(
        f"KDA_PREFILL chunk={chunk} key={key_dim} value={value_dim} "
        f"dynamic={dynamic} static={static_count(code)}"
    )
    assert dynamic <= PREFILL_DYNAMIC_MAX[shape_key], (
        f"{dynamic} issued instructions at chunk {chunk}, key_dim {key_dim}, "
        f"value_dim {value_dim} exceeds {PREFILL_DYNAMIC_MAX[shape_key]}"
    )


def test_the_head_scales_worse_than_the_image_says():
    """`128 x 128` against `64 x 64`: 1.6x by image, 2.3x by issue stream.

    The measurement note in `static_path_measurements.md` read the image ratio
    as the cost of quadrupling the state -- "1.6x the instructions for 4x the
    state". The kernel is sublinear in the state either way, which was the
    point, but by 2.3x rather than 1.6x. Pinned so the two ratios are always
    reported together.
    """
    ratios = {}
    for chunk in (4, 8, 16):
        small = _build(chunk=chunk, key_dim=64, value_dim=64)[2]
        large = _build(chunk=chunk, key_dim=128, value_dim=128)[2]
        ratios[chunk] = (
            static_count(large) / static_count(small),
            dynamic_count(large) / dynamic_count(small),
        )
        print(
            f"KDA_PREFILL chunk={chunk} 128x128/64x64 "
            f"static={ratios[chunk][0]:.2f}x dynamic={ratios[chunk][1]:.2f}x"
        )
    for chunk, (image, work) in ratios.items():
        assert image < work, (
            f"at chunk {chunk} the image ratio {image:.2f} is not below the "
            f"work ratio {work:.2f}; the image used to understate the scaling "
            f"and if it stops doing so the reason should be written down"
        )
        assert work < 4.0, (
            f"at chunk {chunk} the kernel costs {work:.2f}x for 4x the state; "
            f"above 4.0 it is no longer sublinear and the layout move regressed"
        )


def test_chunk_is_nearly_free_in_issued_instructions():
    """Quadrupling `chunk` costs 6% of the issue stream and 80% of the image.

    A consequence of prefill being spill-dominated: the per-chunk cost is the
    fixed traffic of filling and storing `mlen`-sized tiles, which does not
    move with how many tokens the chunk holds. It is worth pinning because the
    `chunk` cap of 17 -- set by bf16 range on `1/A`, not by the lowering -- is
    therefore more expensive than the image suggests. Every token that cannot
    join a chunk pays the fixed cost again.

    Measured at `128 x 128`, `mlen` 64: chunk 4 -> 16 moves the image from
    1,470 to 2,426 and the issue stream from 43,851 to 45,515.
    """
    small = _build(chunk=4, key_dim=128, value_dim=128)[2]
    large = _build(chunk=16, key_dim=128, value_dim=128)[2]
    image = static_count(large) / static_count(small)
    work = dynamic_count(large) / dynamic_count(small)
    print(f"KDA_PREFILL chunk 4->16 static={image:.2f}x dynamic={work:.3f}x")
    assert work < 1.15, (
        f"chunk 4 -> 16 cost {work:.3f}x of the issue stream; it was 1.04x when "
        f"measured, and a rise means the chunk dimension started paying for "
        f"itself somewhere it did not before"
    )
    assert image > 1.4, (
        f"the image grew only {image:.2f}x; it was 1.65x, and a fall means a "
        f"sweep that was unrolled over `chunk` became a loop -- good news, but "
        f"it changes what this test is documenting"
    )


# ---------------------------------------------------------------------------
# The prefill -> decode state layout boundary.


def test_the_two_state_layouts_are_transposes_of_each_other():
    """Documents the mismatch by executing it.

    decode indexes the state with `kda_state_row` -- one row per key, value on
    lanes -- and prefill holds it one row per value, key on lanes. At Kimi K3
    `key_dim == value_dim == 128`, so the shapes match and handing prefill's
    state straight to decode gives a finite, plausible, wrong answer.

    Both layouts are deliberate: decode's makes each sweep a row progression and
    therefore a hardware loop, and prefill's makes all seven of its products
    land on the projection primitives without an explicit transpose. So the
    conversion belongs at the boundary, and
    `kda_prefill_state_to_decode_layout_v0` is it.
    """
    import torch

    from compiler.aten.plena.program_kda_common import kda_state_row

    key_dim = value_dim = MLEN
    shape = KdaShape(
        hidden_size=value_dim, num_heads=1, key_dim=key_dim,
        value_dim=value_dim, conv_kernel=4,
    )
    # A state whose (value, key) entry is distinguishable from (key, value).
    dense = torch.arange(value_dim * key_dim).reshape(value_dim, key_dim).float()

    # decode reads one row per key with value on lanes, so what it sees is the
    # transpose of what prefill leaves behind.
    assert kda_state_row(shape, MLEN, 0, 0, 2) == 2, "one row per key"
    as_decode_sees_it = torch.stack([dense[:, k] for k in range(key_dim)])
    torch.testing.assert_close(as_decode_sees_it, dense.T)
    assert not torch.allclose(as_decode_sees_it, dense), (
        "if these ever coincide the mismatch is gone and the converter is dead code"
    )


def _transpose(mlen: int = MLEN, key_dim: int | None = None, value_dim: int | None = None):
    """Compile the layout converter alone, tiles sized from its own shape table."""
    from compiler.aten.plena.program_ssd import SPILLED_ACTIVATION

    shape = KdaShape(
        hidden_size=mlen, num_heads=1,
        key_dim=key_dim or mlen, value_dim=value_dim or mlen, conv_kernel=4,
    )
    p = PlenaCompiler(mlen=mlen, blen=4)
    p._bf16_kv_checked = True
    want = kda_prefill_state_transpose_shapes(shape, mlen)
    tiles = {
        "state": p.alloc("state", *want["state"], strict=False),
        "identity": p.alloc("t_identity", *want["identity"], strict=False),
        "out": p.alloc("state_T", *want["out"], strict=False),
    }
    mark = len(p.get_code())
    p.kda_prefill_state_to_decode_layout_v0(
        shape=shape, precision=SPILLED_ACTIVATION, **tiles
    )
    return p, tiles, shape, _projections(p.get_code()[mark:])


def test_the_layout_converter_projects_the_state_against_the_identity():
    """A transpose is `out[i][j] = sum_k I[i][k] * state[j][k]`, so every
    projection is the `_T_to` form with the identity as the VRAM operand and the
    spilled state as MRAM."""
    _, _, _, projections = _transpose()
    assert len(projections) == 1, projections
    only = projections[0]
    assert only["transposed"], "a transpose is the _T_to form"
    assert only["vram"] == "t_identity"
    assert only["mram"] == "kda_state_transpose_spill"
    assert only["target"] == "state_T"


def test_the_layout_converter_transposes_the_block_indices_too():
    """With more than one block the block indices swap as well as the elements.

    `out[ib][jb] = I[ib][:] @ state[jb][:]^T` with `ib` over key blocks and `jb`
    over value blocks, so a converter that emitted `(jb, ib)` -- or that only
    ran the diagonal -- would move the elements correctly inside each block and
    put the blocks in the wrong place. At `key_dim == value_dim` the tile is
    square, so nothing about the shapes would object.
    """
    _, _, _, projections = _transpose(mlen=8, key_dim=16, value_dim=16)
    pairs = sorted((x["tr"], x["tc"]) for x in projections)
    assert pairs == [(0, 0), (0, 1), (1, 0), (1, 1)], pairs
    # The MRAM operand is indexed by the *value* block while the target row is
    # the *key* block, which is the swap itself. `_T_to` prints its operand as
    # `name[row][:]`, so the first bracket is the value block and it must equal
    # the target's column index, not its row index.
    for x in projections:
        value_block = int(x["mslot"].split("]")[0].lstrip("["))
        assert value_block == x["tc"], (
            f"out[{x['tr']}][{x['tc']}] reads state block {value_block}; "
            f"it must read the value block that becomes its column"
        )


def test_the_layout_converter_refuses_a_chunk_width_identity():
    """Its identity spans the whole key axis, not `chunk`. The UT transform's
    `[chunk, chunk]` identity is a different tile, and handing that one over
    contracts against part of the key axis for a finite wrong answer."""
    from compiler.aten.plena.program_ssd import SPILLED_ACTIVATION

    shape = KdaShape(hidden_size=8, num_heads=1, key_dim=16, value_dim=16, conv_kernel=4)
    p = PlenaCompiler(mlen=8, blen=4)
    p._bf16_kv_checked = True
    with pytest.raises(ValueError, match="columns"):
        p.kda_prefill_state_to_decode_layout_v0(
            state=p.alloc("state", 16, 16, strict=False),
            identity=p.alloc("chunk_identity", 8, 8, strict=False),
            out=p.alloc("state_T", 16, 16, strict=False),
            shape=shape, precision=SPILLED_ACTIVATION,
        )


def test_the_layout_converter_refuses_to_alias_its_output():
    """A projection overwrites its target, so transposing in place would read
    half-written data."""
    from compiler.aten.plena.program_ssd import SPILLED_ACTIVATION

    shape = KdaShape(
        hidden_size=MLEN, num_heads=1, key_dim=MLEN, value_dim=MLEN, conv_kernel=4
    )
    p = PlenaCompiler(mlen=MLEN, blen=4)
    p._bf16_kv_checked = True
    state = p.alloc("state", MLEN, MLEN, strict=False)
    with pytest.raises(ValueError, match="must not alias"):
        p.kda_prefill_state_to_decode_layout_v0(
            state=state, identity=p.alloc("t_identity", MLEN, MLEN, strict=False),
            out=state, shape=shape, precision=SPILLED_ACTIVATION,
        )
