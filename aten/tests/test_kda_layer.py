"""The KDA layer's projection split, which is a naming operation and not a copy.

The input projection produces one tile of `3*key_width + value_width + heads`
columns; the mixer wants per-`(head, key block)` tiles of `mlen`. Those look
like different layouts and a naive conversion is a scatter of one row per
column block -- around 700 per layer at Kimi K3's shape, each a full block
operation.

They are the same bytes. Column block `c` of any VRAM tile sits at
`base + c * mlen * mlen`, so block `c` of `[tokens, N]` *is* the `[tokens, mlen]`
tile at that address. `vram_column_block_view` names one by registering a VRAM
object at the computed address rather than allocating, and emits nothing.

These tests pin that it emits nothing, that the alias is real, and that the
section boundaries match the order `kda_state_engine_step` splits in -- an
off-by-one-block there hands the mixer `k` where it expects `q`, which is a
finite plausible wrong answer.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.kda.shape import KdaShape  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.instruction_stream import (  # noqa: E402
    dynamic_count,
    static_count,
)
from compiler.aten.plena.program_kda_conv import kda_conv_blocks  # noqa: E402
from compiler.aten.plena.program_kda_layer import (  # noqa: E402
    kda_projection_features,
    kda_projection_sections,
    kda_projection_width,
)
from compiler.aten.tests.isa_interpreter import Machine  # noqa: E402

MLEN = 8


def _shape(heads=4, key_dim=MLEN, value_dim=MLEN) -> KdaShape:
    return KdaShape(
        hidden_size=heads * value_dim, num_heads=heads, key_dim=key_dim,
        value_dim=value_dim, conv_kernel=4,
    )


def _static(asm: str) -> int:
    return len([ln for ln in asm.splitlines() if ln.strip() and not ln.strip().startswith(";")])


def test_the_split_emits_nothing():
    """The whole point. A scatter would be ~700 block operations per layer at
    Kimi K3; naming a view is free."""
    shape = _shape()
    p = PlenaCompiler(mlen=MLEN, blen=2)
    projected = p.alloc("projected", MLEN, kda_projection_width(shape, MLEN))
    mark = len(p.get_code())
    p.kda_split_projection_v0(projected=projected, shape=shape, rows=4)
    assert _static(p.get_code()[mark:]) == 0


def test_the_sections_follow_the_reference_split_order():
    """`kda_state_engine_step` splits [key, key, value, key, heads] in that
    order. Getting it wrong hands the mixer k where it expects q."""
    shape = _shape(heads=4, key_dim=MLEN * 2, value_dim=MLEN)
    sections = kda_projection_sections(shape, MLEN)
    assert [name for name, _, _ in sections] == ["q", "k", "v", "gate", "beta"]
    key_blocks = shape.projection_size // MLEN
    value_blocks = shape.num_heads * shape.value_dim // MLEN
    assert [count for _, _, count in sections] == [
        key_blocks, key_blocks, value_blocks, key_blocks, 1
    ]
    # Contiguous, no gaps and no overlaps.
    expected_first = 0
    for _, first, count in sections:
        assert first == expected_first
        expected_first += count


def test_beta_is_padded_to_a_block_and_the_width_says_so():
    """beta is one value per head -- 96 at Kimi K3 against mlen 64 -- so it is
    the one section that is not naturally a whole block. The host has to
    materialise W_in at the padded width with the padding columns zero; a
    narrower weight leaves the tail of beta's block holding whatever preceded
    it in HBM."""
    kimi = KdaShape(
        hidden_size=7168, num_heads=96, key_dim=128, value_dim=128, conv_kernel=4
    )
    logical = kda_projection_features(kimi)
    padded = kda_projection_width(kimi, 64)
    assert padded > logical, "beta's 96 heads do not fill a 64-lane block"
    assert padded % 64 == 0
    # Exactly the beta padding, nothing else.
    assert padded - logical == 64 * 2 - 96


@pytest.mark.parametrize(
    "parent_rows,strict",
    [
        (MLEN, True),        # the only shape the first version tested
        (MLEN * 2, True),    # two row blocks -- the stride is physical_rows, not mlen
        (MLEN * 4, True),    # four
        (2, False),          # padded: alloc pads rows to blen, not to mlen
    ],
)
def test_a_view_lands_on_its_parents_column_block(parent_rows, strict):
    """The view's address must equal the parent's own block address.

    Column blocks are strided by `physical_rows * mlen`, not `mlen * mlen`;
    those coincide only when a tile is exactly mlen rows tall, which is the one
    shape this was first tested at. Both the right formula and a wrong one
    passed, because the test seeded and read at the view's *own* address -- it
    was self-consistent with any address the view invented.
    """
    shape = _shape()
    p = PlenaCompiler(mlen=MLEN, blen=2)
    width = kda_projection_width(shape, MLEN)
    projected = p.alloc("projected", parent_rows, width, strict=strict)
    n_blocks = p.get_vram_layout(projected.name).physical_shape[1] // MLEN
    for block in range(n_blocks):
        view = p.vram_column_block_view(projected, block, name=f"v{block}", rows=1)
        assert p.get_vram_layout(view.name).vram_base_addr == p._tile_addr(
            projected.name, 0, block
        ), f"view of block {block} is not where the parent's block {block} is"


def test_a_view_aliases_its_parent():
    """Written through the PARENT's address, read through the view.

    Seeding at the view's own address would pass for any address at all; the
    point is that the two agree.
    """
    shape = _shape()
    p = PlenaCompiler(mlen=MLEN, blen=2)
    width = kda_projection_width(shape, MLEN)
    # Two row blocks, so the stride actually discriminates.
    projected = p.alloc("projected", MLEN * 2, width)
    views = p.kda_split_projection_v0(projected=projected, shape=shape, rows=MLEN)
    block = kda_projection_sections(shape, MLEN)[2][1] + 1     # v's second block
    target = views["v"][1]

    parent_addr = p._tile_addr(projected.name, 0, block)
    assert p.get_vram_layout(target.name).vram_base_addr == parent_addr

    consts = p.kda_fp_constants()
    two = p.fp_var("two", size=1)
    mark = len(p.get_code())
    p.tile_row_mul_fp_broadcast(target, two, rows=[0, 1])
    code = p.get_code()[mark:]

    m = Machine(vlen=MLEN, vram_words=1 << 18, fpram_words=1 << 13)
    m.write_fpram(consts.zero.address, p.kda_fp_constant_values())
    m.write_fpram(two.address, [2.0])
    seed = [float(i + 1) for i in range(MLEN)]
    # Through the parent's address, not the view's.
    m.write_vram_row(parent_addr, seed)
    m.write_vram_row(parent_addr + MLEN, seed)
    # A neighbouring block must not move -- a wrong stride scaled someone else's
    # bytes and left these alone.
    neighbour = p._tile_addr(projected.name, 0, block + 1)
    m.write_vram_row(neighbour, seed)
    m.run(code)

    assert m.read_vram_row(parent_addr, MLEN) == [v * 2 for v in seed]
    assert m.read_vram_row(neighbour, MLEN) == seed, "the view touched another block"


def test_a_view_is_bounded_by_its_parent():
    shape = _shape()
    p = PlenaCompiler(mlen=MLEN, blen=2)
    projected = p.alloc("projected", MLEN, kda_projection_width(shape, MLEN))
    n_blocks = projected.physical_shape[1] // MLEN
    with pytest.raises(ValueError, match="out of range"):
        p.vram_column_block_view(projected, n_blocks, name="past_the_end")
    phys_rows = p.get_vram_layout(projected.name).physical_shape[0]
    # One past, not far past: `rows=MLEN*2` cleared the boundary by so much that
    # a permissive off-by-one bound survived.
    with pytest.raises(ValueError, match="exceeds"):
        p.vram_column_block_view(projected, 0, name="too_tall", rows=phys_rows + 1)


def test_every_per_head_tile_is_a_block_boundary():
    """The split is only free because each `(head, key block)` starts on one.
    At Kimi K3 that is 96 heads x 2 blocks for each of q, k and gate."""
    kimi = KdaShape(
        hidden_size=7168, num_heads=96, key_dim=128, value_dim=128, conv_kernel=4
    )
    mlen = 64
    for head in range(kimi.num_heads):
        for block in range(kimi.key_dim // mlen):
            feature = head * kimi.key_dim + block * mlen
            assert feature % mlen == 0, (
                f"head {head} block {block} starts at feature {feature}, "
                f"which is not a block boundary"
            )


def test_a_duplicate_view_name_is_refused():
    """`register_vram_matrix` overwrites without complaint and views resolve by
    name at emit time, so a duplicate silently repoints every view already
    handed out. `prefix` defaults to "kda", so two splits in one program collide
    -- and the layer this is for runs 93 times."""
    shape = _shape()
    p = PlenaCompiler(mlen=MLEN, blen=2)
    width = kda_projection_width(shape, MLEN)
    a = p.alloc("proj_a", MLEN, width)
    b = p.alloc("proj_b", MLEN, width)
    p.kda_split_projection_v0(projected=a, shape=shape, rows=1)
    with pytest.raises(ValueError, match="already exists"):
        p.kda_split_projection_v0(projected=b, shape=shape, rows=1)
    # A distinct prefix is the way through.
    p.kda_split_projection_v0(projected=b, shape=shape, rows=1, prefix="layer1")


def test_the_split_needs_key_dim_on_a_block_boundary_not_just_key_width():
    """The invariant is per head. 4 heads of key_dim 6 gives key_width 24,
    divisible by mlen 8 -- while heads 1, 2 and 3 all start mid-block."""
    bad = KdaShape(hidden_size=32, num_heads=4, key_dim=6, value_dim=MLEN, conv_kernel=4)
    p = PlenaCompiler(mlen=MLEN, blen=2)
    assert (bad.num_heads * bad.key_dim) % MLEN == 0, "key_width alone looks fine"
    projected = p.alloc("projected", MLEN, MLEN * 8)
    with pytest.raises(ValueError, match="multiple of mlen"):
        p.kda_split_projection_v0(projected=projected, shape=bad, rows=1)


def test_a_view_carries_its_parents_physical_shape():
    """So a view of a view addresses correctly.

    Built by hand, a view's var carried the rows asked for while its layout
    carried the parent's -- and reading the var gave a nested view the wrong
    bounds and, past one row block, the wrong address. Going through `alloc_at`
    with an explicit physical_shape makes the two agree by construction.
    """
    p = PlenaCompiler(mlen=MLEN, blen=2)
    parent = p.alloc("parent", MLEN * 2, MLEN * 4)
    view = p.vram_column_block_view(parent, 1, name="v1", rows=MLEN)
    assert view.physical_shape == p.get_vram_layout(view.name).physical_shape

    nested = p.vram_column_block_view(view, 0, name="v2", rows=MLEN * 2)
    assert p.get_vram_layout(nested.name).vram_base_addr == p._tile_addr(
        parent.name, 0, 1
    )


def test_the_tall_view_makes_a_projection_one_dense_tile():
    """Column block `c` is at `base + c*physical_rows*mlen` and rows within a
    block are linear, so the whole tile already *is* a dense
    `[blocks*physical_rows, mlen]` tile. Feature block `c`'s token `t` is at row
    `c*stride + t` -- which is what lets a gather be one hardware loop."""
    p = PlenaCompiler(mlen=MLEN, blen=2)
    for rows, strict in ((MLEN, True), (MLEN * 2, True), (2, False)):
        parent = p.alloc(f"p{rows}{strict}", rows, MLEN * 4, strict=strict)
        tall, stride = p.vram_tall_view(parent, name=f"t{rows}{strict}")
        phys_rows = p.get_vram_layout(parent.name).physical_shape[0]
        assert stride == phys_rows
        assert tall.shape[0] == 4 * phys_rows
        for c in range(4):
            # Row c*stride of the tall view is block c's row 0.
            want = p._tile_addr(parent.name, 0, c)
            got = p.get_vram_layout(tall.name).vram_base_addr + c * stride * MLEN
            assert got == want, f"block {c}: tall view says {got}, parent says {want}"


def test_the_gather_is_one_hardware_loop_per_section():
    """14 static instructions per section, independent of block count. A copy
    per block would be one loop scaffold each -- 10 instructions per block, so
    7,674 for a Kimi K3 layer against about 70."""
    shape = _shape(heads=8, key_dim=MLEN * 2)
    p = PlenaCompiler(mlen=MLEN, blen=2)
    consts = p.kda_fp_constants()
    projected = p.alloc("projected", MLEN, kda_projection_width(shape, MLEN))
    counts = {}
    for section, (first, count) in {
        n: (f, c) for n, f, c in kda_projection_sections(shape, MLEN)
    }.items():
        dst = p.alloc(f"dst_{section}", max(count, MLEN), MLEN, strict=False)
        mark = len(p.get_code())
        p.kda_gather_projection_v0(
            projected=projected, dst=dst, section=section, shape=shape,
            consts=consts, name=f"tall_{section}",
        )
        code = p.get_code()[mark:]
        counts[section] = (count, _static(code), code.count("V_FMA_VF"))
    for section, (blocks, static, fmas) in counts.items():
        assert fmas == 1, f"{section} spans {blocks} blocks but emitted {fmas} FMAs"
        assert static < 30, f"{section}: {static} instructions for {blocks} blocks"
    # q spans 16 blocks and beta 1. They differ by the loop's two step
    # increments, which a single-block sweep skips -- not by block count.
    assert counts["q"][1] - counts["beta"][1] <= 2, counts
    assert counts["q"][0] == 16 and counts["beta"][0] == 1, counts


def test_the_gather_cost_does_not_grow_with_head_count():
    """The property, stated directly: 96 heads must cost what 2 heads cost."""
    costs = {}
    for heads in (2, 8, 96):
        shape = _shape(heads=heads, key_dim=MLEN * 2)
        p = PlenaCompiler(mlen=MLEN, blen=2)
        consts = p.kda_fp_constants()
        projected = p.alloc("projected", MLEN, kda_projection_width(shape, MLEN))
        first, count = {n: (f, c) for n, f, c in
                        kda_projection_sections(shape, MLEN)}["q"]
        dst = p.alloc("dst", max(count, MLEN), MLEN, strict=False)
        mark = len(p.get_code())
        p.kda_gather_projection_v0(
            projected=projected, dst=dst, section="q", shape=shape, consts=consts,
        )
        costs[heads] = _static(p.get_code()[mark:])
    assert len(set(costs.values())) == 1, f"gather cost grows with heads: {costs}"


def test_the_gather_moves_the_right_features():
    """Section boundaries and the tall view's stride, checked against values.

    An off-by-one block here hands the mixer `k` where it expects `q`.
    """
    import torch

    shape = _shape(heads=2, key_dim=MLEN * 2)
    p = PlenaCompiler(mlen=MLEN, blen=2)
    consts = p.kda_fp_constants()
    width = kda_projection_width(shape, MLEN)
    projected = p.alloc("projected", MLEN, width)
    sections = {n: (f, c) for n, f, c in kda_projection_sections(shape, MLEN)}
    dsts = {
        n: p.alloc(f"dst_{n}", max(c, MLEN), MLEN, strict=False)
        for n, (_, c) in sections.items()
    }
    mark = len(p.get_code())
    for section in sections:
        p.kda_gather_projection_v0(
            projected=projected, dst=dsts[section], section=section,
            shape=shape, consts=consts, name=f"tall_{section}",
        )
    code = p.get_code()[mark:]

    m = Machine(vlen=MLEN, vram_words=1 << 18, fpram_words=1 << 13)
    m.write_fpram(consts.zero.address, p.kda_fp_constant_values())
    n_blocks = p.get_vram_layout(projected.name).physical_shape[1] // MLEN
    # Block b's token 0 is marked with b, so a misrouted gather is obvious.
    for b in range(n_blocks):
        m.write_vram_row(p._tile_addr(projected.name, 0, b), [float(b)] * MLEN)
    m.run(code)

    for section, (first, count) in sections.items():
        base = p.get_vram_layout(dsts[section].name).vram_base_addr
        got = [m.read_vram_row(base + r * MLEN, 1)[0] for r in range(count)]
        want = [float(first + c) for c in range(count)]
        assert got == want, f"{section}: gathered blocks {got}, expected {want}"


# ---------------------------------------------------------------------------
# The assembled layer, against the reference's own boundary.


def _layer_case(seed: int, heads: int, key_dim: int, value_dim: int, kernel: int = 4):
    """Everything `kda_state_engine_step` needs, plus the compiled equivalent."""
    import torch

    from compiler.aten.models.kda.reference import (
        KdaConvWeights,
        KdaRecurrentState,
        kda_state_engine_step,
    )
    from compiler.aten.plena.program_kda_common import (
        kda_state_rows,
        kda_vector_row,
        kda_vector_rows,
    )
    from compiler.aten.plena.program_kda_conv import kda_conv_blocks, kda_conv_state_row
    from compiler.aten.plena.program_kda_gates import (
        kda_head_blocks,
        kda_key_blocks,
        kda_key_row,
    )
    from compiler.aten.plena.program_kda_mixer import KdaMixerBuffers

    torch.manual_seed(seed)
    shape = KdaShape(
        hidden_size=heads * value_dim, num_heads=heads, key_dim=key_dim,
        value_dim=value_dim, conv_kernel=kernel,
    )
    key_width = shape.projection_size
    value_width = heads * value_dim
    projected = torch.randn(1, kda_projection_features(shape)) * 0.5
    conv_w = {
        n: torch.randn(w, kernel) * 0.3
        for n, w in (("q", key_width), ("k", key_width), ("v", value_width))
    }
    # `kernel` taps, not kernel-1: both the reference and the emitter roll the
    # window and overwrite the last with the new token, so the state carries the
    # full window.
    conv_hist = torch.randn(1, key_width * 2 + value_width, kernel) * 0.3
    state = KdaRecurrentState(
        torch.randn(1, heads, value_dim, key_dim) * 0.1, conv_hist.clone()
    )
    a_log = torch.randn(heads) * 0.5
    dt_bias = torch.randn(heads, key_dim) * 0.5
    out_ref, state_ref = kda_state_engine_step(
        projected, state,
        KdaConvWeights(conv_w["q"], conv_w["k"], conv_w["v"], None, None, None),
        a_log, dt_bias, shape, state_storage="fp32",
    )
    return dict(
        shape=shape, projected=projected, conv_w=conv_w, conv_hist=conv_hist,
        state=state, a_log=a_log, dt_bias=dt_bias,
        out_ref=out_ref, state_ref=state_ref,
        key_width=key_width, value_width=value_width,
    )


def test_the_assembled_layer_matches_kda_state_engine_step():
    """Gather, three convolutions, the gates and the recurrence, against the
    reference's own boundary.

    This is what an earlier version of this file said could not be built without
    changing two emitters. It needed neither: one gather over a tall view, and
    everything downstream is as it was.
    """
    import torch

    from compiler.aten.plena.program_kda_common import (
        kda_state_row,
        kda_state_rows,
        kda_vector_row,
        kda_vector_rows,
    )
    from compiler.aten.plena.program_kda_conv import kda_conv_blocks, kda_conv_state_row
    from compiler.aten.plena.program_kda_gates import (
        kda_head_blocks,
        kda_key_blocks,
        kda_key_row,
    )
    from compiler.aten.plena.program_kda_mixer import KdaMixerBuffers

    c = _layer_case(3, heads=3, key_dim=MLEN * 2, value_dim=MLEN)
    shape = c["shape"]
    p = PlenaCompiler(mlen=MLEN, blen=2)
    consts = p.kda_fp_constants()
    kb = kda_key_blocks(shape, MLEN)
    up = lambda n: ((n + MLEN - 1) // MLEN) * MLEN  # noqa: E731
    a = lambda n, r: p.alloc(n, up(r), MLEN, strict=False)  # noqa: E731

    # The projection output, allocated at blen rows -- the whole point of the
    # stride being physical_rows rather than mlen.
    projected = p.alloc("projected", p.blen, kda_projection_width(shape, MLEN),
                        strict=False)
    sections = {n: (f, cnt) for n, f, cnt in kda_projection_sections(shape, MLEN)}
    gathered = {
        n: a(f"g_{n}", max(cnt, MLEN)) for n, (_, cnt) in sections.items()
    }
    widths = {"q": c["key_width"], "k": c["key_width"], "v": c["value_width"]}
    conv_state = {n: a(f"cs_{n}", kda_conv_blocks(w, MLEN) * shape.conv_kernel)
                  for n, w in widths.items()}
    conv_weight = {n: a(f"cw_{n}", kda_conv_blocks(w, MLEN) * shape.conv_kernel)
                   for n, w in widths.items()}
    conv_scratch = a("conv_scratch", kda_conv_blocks(max(widths.values()), MLEN))

    decay = p.fp_var("decay_and_q_hat", size=shape.key_dim)
    buffers = KdaMixerBuffers(
        q=a("q", shape.num_heads * kb), k=a("k", shape.num_heads * kb),
        v=a("v", kda_vector_rows(shape, MLEN)),
        gate=gathered["gate"], dt_bias=a("dt_bias", shape.num_heads * kb),
        beta_logit=gathered["beta"],
        state=a("state", kda_state_rows(shape, MLEN)),
        out=a("out", kda_vector_rows(shape, MLEN)),
        pred=a("pred", kda_vector_rows(shape, MLEN)),
        err=a("err", kda_vector_rows(shape, MLEN)),
        sq_scratch=a("sq_scratch", shape.num_heads * kb),
        decay_fp=decay, q_hat_fp=decay,
        k_hat_fp=p.fp_var("k_hat", size=shape.key_dim),
        beta_fp=p.fp_var("beta", size=kda_head_blocks(shape, MLEN) * MLEN),
        part_fp=p.fp_var("part", size=kb), acc_fp=p.fp_var("acc", size=1),
        output_scale_fp=p.fp_var("output_scale", size=1),
        rate_fp=p.fp_var("rate", size=shape.num_heads),
        lower_bound_fp=p.fp_var("lower_bound", size=1), consts=consts,
    )
    mark = len(p.get_code())
    p.kda_layer_from_projected_v0(
        projected=projected, gathered=gathered, conv_state=conv_state,
        conv_weight=conv_weight, conv_bias={}, conv_scratch=conv_scratch,
        mixer_buffers=buffers, shape=shape,
    )
    code = p.get_code()[mark:]

    m = Machine(vlen=MLEN, vram_words=1 << 20, fpram_words=1 << 14)
    m.write_fpram(consts.zero.address, p.kda_fp_constant_values())
    m.write_fpram(buffers.output_scale_fp.address, [1.0 / shape.key_dim ** 0.5])
    m.write_fpram(buffers.rate_fp.address, torch.exp(c["a_log"]).tolist())
    m.write_fpram(buffers.lower_bound_fp.address, [shape.gate_lower_bound])

    # The projection output: feature f at column block f//mlen, row 0.
    flat = c["projected"][0]
    for blk in range(-(-flat.numel() // MLEN)):
        chunk = flat[blk * MLEN : (blk + 1) * MLEN].tolist()
        chunk += [0.0] * (MLEN - len(chunk))
        m.write_vram_row(p._tile_addr(projected.name, 0, blk), chunk)

    # Conv history and weights, blocked by kda_conv_state_row.
    hist = c["conv_hist"][0]
    offsets = {"q": 0, "k": c["key_width"], "v": 2 * c["key_width"]}
    for name, width in widths.items():
        blocks = kda_conv_blocks(width, MLEN)
        cs_base = p.get_vram_layout(conv_state[name].name).vram_base_addr
        cw_base = p.get_vram_layout(conv_weight[name].name).vram_base_addr
        for cb in range(blocks):
            lo = cb * MLEN
            for tap in range(shape.conv_kernel):
                row = kda_conv_state_row(width, MLEN, shape.conv_kernel, cb, tap)
                vals = hist[offsets[name] + lo : offsets[name] + lo + MLEN, tap]
                m.write_vram_row(cs_base + row * MLEN, vals.tolist())
                m.write_vram_row(
                    cw_base + row * MLEN,
                    c["conv_w"][name][lo : lo + MLEN, tap].tolist(),
                )

    # dt_bias, and the incoming recurrent state, transposed as decode holds it.
    dt_base = p.get_vram_layout(buffers.dt_bias.name).vram_base_addr
    for h in range(shape.num_heads):
        for b in range(kb):
            m.write_vram_row(
                dt_base + kda_key_row(shape, MLEN, h, b) * MLEN,
                c["dt_bias"][h, b * MLEN : (b + 1) * MLEN].tolist(),
            )
    st_base = p.get_vram_layout(buffers.state.name).vram_base_addr
    blocks_v = kda_vector_rows(shape, MLEN) // shape.num_heads
    for h in range(shape.num_heads):
        for blk in range(blocks_v):
            for key in range(shape.key_dim):
                row = kda_state_row(shape, MLEN, h, blk, key)
                vals = c["state"].recurrent[0, h, blk * MLEN : (blk + 1) * MLEN, key]
                m.write_vram_row(st_base + row * MLEN, vals.tolist())
    m.run(code)

    out_base = p.get_vram_layout(buffers.out.name).vram_base_addr
    got = torch.tensor([
        sum((m.read_vram_row(
            out_base + kda_vector_row(shape, MLEN, h, blk) * MLEN, MLEN)
            for blk in range(blocks_v)), [])
        for h in range(shape.num_heads)
    ]).reshape(1, -1)
    torch.testing.assert_close(got, c["out_ref"], rtol=5e-4, atol=5e-5)


# ---------------------------------------------------------------------------
# Layer composition at depth: the HBM address space, and FPRAM reuse.


def _kimi_layers(layers: int):
    """Stack `layers` Kimi K3 KDA layers and return the program."""
    from compiler.aten.plena.program_kda_common import kda_state_rows, kda_vector_rows
    from compiler.aten.plena.program_kda_conv import kda_conv_blocks
    from compiler.aten.plena.program_kda_gates import kda_head_blocks, kda_key_blocks
    from compiler.aten.plena.program_kda_mixer import KdaMixerBuffers

    mlen = 64
    shape = KdaShape.kimi_k3()
    key_width = shape.projection_size
    value_width = shape.num_heads * shape.value_dim
    kernel = shape.conv_kernel
    kb = kda_key_blocks(shape, mlen)
    up = lambda n: ((n + mlen - 1) // mlen) * mlen  # noqa: E731
    counts = {n: c for n, _, c in kda_projection_sections(shape, mlen)}

    p = PlenaCompiler(mlen=mlen, blen=4)
    # Bypasses kda_require_state_precision_v0, which a real lowering must call
    # -- under an MX KV type the state decodes as e4m3 while the address stride
    # assumes 2 bytes. Acceptable for a structural measurement; it does mean the
    # program measured here could not be executed as configured.
    p._bf16_kv_checked = True
    consts = p.kda_fp_constants()
    a = lambda n, r: p.alloc(n, up(r), mlen, strict=False)  # noqa: E731
    # One FPRAM window for the whole model. It is scratch, not storage.
    shared = {
        k: p.fp_var(k, size=s)
        for k, s in (
            ("decay", shape.key_dim), ("k_hat", shape.key_dim),
            ("beta", kda_head_blocks(shape, mlen) * mlen), ("part", kb),
            ("acc", 1), ("os", 1), ("rate", shape.num_heads), ("lb", 1),
        )
    }
    widths = {"q": key_width, "k": key_width, "v": value_width}
    for i in range(layers):
        projected = p.load_batch(
            p.input(f"proj{i}", (p.blen, kda_projection_width(shape, mlen))),
            name=f"projv{i}",
        )
        gathered = {n: a(f"g{i}_{n}", max(c, mlen)) for n, c in counts.items()}
        conv_state = {
            n: p.load_batch(
                p.input(f"cs{i}_{n}", (kda_conv_blocks(w, mlen) * kernel, mlen)),
                name=f"csv{i}_{n}",
            )
            for n, w in widths.items()
        }
        conv_weight = {
            n: p.load_batch(
                p.input(f"cw{i}_{n}", (kda_conv_blocks(w, mlen) * kernel, mlen)),
                name=f"cwv{i}_{n}",
            )
            for n, w in widths.items()
        }
        # The state, dt_bias and the projection come from HBM, exactly as
        # `kda_stage_test.py::case_layer` stages them. Allocating them in VRAM
        # instead made the HBM figure count only the conv tensors -- one seventh
        # of a layer's real traffic -- and the two tests then disagreed about
        # where a layer's state lives.
        state_v = p.load_batch(
            p.input(f"st{i}", (up(kda_state_rows(shape, mlen)), mlen)), name=f"stv{i}"
        )
        dtb_v = p.load_batch(
            p.input(f"dtb{i}", (up(shape.num_heads * kb), mlen)), name=f"dtbv{i}"
        )
        buffers = KdaMixerBuffers(
            q=a(f"q{i}", shape.num_heads * kb), k=a(f"k{i}", shape.num_heads * kb),
            v=a(f"v{i}", kda_vector_rows(shape, mlen)), gate=gathered["gate"],
            dt_bias=dtb_v, beta_logit=gathered["beta"],
            state=state_v,
            out=a(f"o{i}", kda_vector_rows(shape, mlen)),
            pred=a(f"pr{i}", kda_vector_rows(shape, mlen)),
            err=a(f"er{i}", kda_vector_rows(shape, mlen)),
            sq_scratch=a(f"sq{i}", shape.num_heads * kb),
            decay_fp=shared["decay"], q_hat_fp=shared["decay"],
            k_hat_fp=shared["k_hat"], beta_fp=shared["beta"],
            part_fp=shared["part"], acc_fp=shared["acc"],
            output_scale_fp=shared["os"], rate_fp=shared["rate"],
            lower_bound_fp=shared["lb"], consts=consts,
        )
        p.kda_layer_from_projected_v0(
            projected=projected, gathered=gathered, conv_state=conv_state,
            conv_weight=conv_weight, conv_bias={},
            conv_scratch=a(f"csc{i}", kda_conv_blocks(max(widths.values()), mlen)),
            mixer_buffers=buffers, shape=shape,
        )
    return p


def test_layers_compose_without_name_collisions():
    """Stacking layers must need nothing from the caller.

    Every view the layer takes is named after its projection tile, which is
    unique per layer. With a fixed prefix the second layer collided -- and
    because `register_vram_matrix` overwrites silently and views resolve by name
    at emit time, without the duplicate guard it would have repointed layer 1's
    views at layer 2's projection instead of raising.
    """
    _kimi_layers(4)


#: Measured 2026-08-26. Everything a layer stages -- conv history and weights,
#: the recurrent state, dt_bias and the projection tile -- excluding only the
#: projection *weights*, which any implementation needs and which are
#: model-inherent.
#:
#: Gated because the emulator preloads HBM from a flat file starting at offset 0,
#: so the allocation has to span every address the program touches. A layout that
#: spreads its regions over a wide address space needs an allocation and a file
#: as large as that span, whatever the live data comes to, and stops being
#: executable long before it stops being describable. Emulator HBM_SIZE is
#: 16 GiB.
#:
#: An earlier version of this counted only the conv tensors, because the helper
#: put the state and the projection in VRAM while the emulator test staged them
#: from HBM. That reported 324 KiB per layer against a real 2,282.
KIMI_HBM_PER_LAYER_MAX = 2560 * 1024
KIMI_LAYERS_FULL = 93


def test_the_hbm_footprint_does_not_repeat_the_descriptor_blowup():
    one = _kimi_layers(1)._next_hbm_addr
    two = _kimi_layers(2)._next_hbm_addr
    per_layer = two - one
    full = one + (KIMI_LAYERS_FULL - 1) * per_layer
    print(
        f"KIMI_K3_HBM per_layer={per_layer} full_{KIMI_LAYERS_FULL}={full} "
        f"({full / (1 << 30):.4f} GiB)"
    )
    assert per_layer <= KIMI_HBM_PER_LAYER_MAX, (
        f"{per_layer} bytes per layer exceeds {KIMI_HBM_PER_LAYER_MAX}"
    )
    assert full < (1 << 30), (
        f"{full / (1 << 30):.2f} GiB at {KIMI_LAYERS_FULL} layers, against a 1 GiB "
        f"bar. The emulator is configured for 16 GiB and preloads it from a flat "
        f"file, so the whole span has to be allocated and written -- a layout that "
        f"crosses the configured size stops being executable at all"
    )


def test_fpram_fits_the_hardware_file_at_full_depth():
    """The whole model's FPRAM must fit `FP_SRAM_DEPTH`, which is **512**.

    The compiler's own `FPRAMAllocator` defaults to 1024 and so will not catch
    an overflow -- that disagreement with the SystemVerilog is why this asserts
    the hardware number directly.

    Sharing the window across layers is the *caller's* decision, and this helper
    makes it; no KDA emitter allocates FPRAM at all (`grep fp_var
    aten/plena/program_kda_*.py` is empty, and a mutation that adds a per-call
    `fp_var` turns this red). So what is pinned here is two things: that the
    emitters stay caller-allocated, and that one shared window is enough for a
    93-layer model. A caller that allocates per layer overflows 1024 by the
    fourth layer and 512 by the second; that is a mistake this test cannot make
    on their behalf, but the number it asserts is the one they have to meet.
    """
    used = {n: _kimi_layers(n).fpram_allocator.next_free for n in (1, 2, 4)}
    assert len(set(used.values())) == 1, (
        f"FPRAM use grows with depth: {used}. It is scratch, not storage."
    )
    full = _kimi_layers(KIMI_LAYERS_FULL).fpram_allocator.next_free
    print(f"KIMI_K3_FPRAM full_{KIMI_LAYERS_FULL}={full}")
    assert full == used[1], "93 layers must use exactly what one layer uses"
    assert full <= 512, (
        f"{full} FPRAM slots against FP_SRAM_DEPTH 512 "
        f"(PLENA_RTL/src/definitions/configuration.svh)"
    )


# ---------------------------------------------------------------------------
# The gather is not forced by packing, and it is not forced at all.


def _conv_reading_projection(direct: bool, mlen: int = 64):
    """Kimi K3's three convolutions, fed by gather-then-conv or by stride."""
    shape = KdaShape.kimi_k3()
    sections = {n: (f, c) for n, f, c in kda_projection_sections(shape, mlen)}
    widths = {
        "q": shape.projection_size, "k": shape.projection_size,
        "v": shape.num_heads * shape.value_dim,
    }
    p = PlenaCompiler(mlen=mlen, blen=4)
    consts = p.kda_fp_constants()
    projected = p.alloc(
        "proj", p.blen, kda_projection_width(shape, mlen), strict=False
    )
    up = lambda n: ((n + mlen - 1) // mlen) * mlen  # noqa: E731
    a = lambda n, r: p.alloc(n, up(r), mlen, strict=False)  # noqa: E731
    mark = len(p.get_code())
    for name, width in widths.items():
        first, count = sections[name]
        blocks = kda_conv_blocks(width, mlen)
        common = dict(
            conv_state=a(f"cs_{name}", blocks * shape.conv_kernel),
            weight=a(f"w_{name}", blocks * shape.conv_kernel), bias=None,
            out=a(f"o_{name}", blocks), scratch=a(f"sc_{name}", blocks),
            consts=consts, channels=width, kernel=shape.conv_kernel,
        )
        if direct:
            tall, stride = p.vram_tall_view(projected, name=f"tall_{name}")
            p.kda_conv_step_v0(
                x_new=tall, x_new_row_base=first * stride,
                x_new_row_stride=stride, **common,
            )
        else:
            dst = a(f"g_{name}", max(count, mlen))
            p.kda_gather_projection_v0(
                projected=projected, dst=dst, section=name, shape=shape,
                consts=consts, name=f"t_{name}",
            )
            p.kda_conv_step_v0(x_new=dst, **common)
    return p.get_code()[mark:]


def test_reading_the_projection_at_a_stride_removes_the_gather():
    """The gather is overhead, and the consumer can simply read across it.

    The projection cannot produce a dense tile. `M_MM_WO` writes a
    `blen x blen` sub-tile and the writeback loops cover `mlen / blen` column
    groups, so the smallest thing it can lay down is `blen` token-rows by
    `mlen` lanes: column block `c` is at row `c * blen`. The convolution wants
    one token's blocks as consecutive rows, and the mismatch is exactly `blen`.

    That mismatch is a property of the matrix writeback, not of how the weight
    matrices are packed -- which is why splitting the packed projection into
    five separate ones changes nothing (`test_separate_projections_do_not_help`
    measures that).

    Reading across it is free. `mamba_row_copy` takes a row index and the
    sweeps underneath collapse any arithmetic progression into one hardware
    loop whose step is an `S_ADDI_INT` immediate, so a step of `blen` costs
    what a step of 1 costs. Feeding the convolution the projection tile with
    `x_new_row_stride=blen` therefore does the gather's job for nothing.
    """
    gathered = _conv_reading_projection(direct=False)
    direct = _conv_reading_projection(direct=True)
    saved = dynamic_count(gathered) - dynamic_count(direct)
    print(
        f"KDA_CONV_FEED gather+conv static={static_count(gathered)} "
        f"dynamic={dynamic_count(gathered)} | stride static={static_count(direct)} "
        f"dynamic={dynamic_count(direct)} | saved={saved}"
    )
    assert saved > 3_500, (
        f"reading at a stride saved only {saved} issued instructions; the three "
        f"q/k/v gathers were 3,474 plus their zero-fills when measured"
    )
    assert dynamic_count(direct) < dynamic_count(gathered)
    assert static_count(direct) < static_count(gathered)


def test_the_stride_reads_exactly_the_rows_the_gather_would_have():
    """Equivalence, without running anything: the source rows must match.

    `kda_gather_projection_v0` copies tall row `(first + c) * stride + token`
    into dense row `c`, and the convolution then reads dense row `cb`. Reading
    the tall view at `base = first * stride`, step `stride`, visits
    `(first + cb) * stride` -- the same rows, for token 0. The emitted
    `VRAM Matrix Add` comments carry the row ranges, so this is checkable on
    the assembly rather than by argument.
    """
    mlen, blen, kernel = 8, 2, 2
    channels = 4 * mlen
    first, stride = 3, blen
    p = PlenaCompiler(mlen=mlen, blen=blen)
    consts = p.kda_fp_constants()
    a = lambda n, r: p.alloc(n, r, mlen, strict=False)  # noqa: E731
    blocks = kda_conv_blocks(channels, mlen)
    tall = a("tall", (first + blocks) * stride)
    common = dict(
        conv_state=a("cs", blocks * kernel), weight=a("w", blocks * kernel),
        bias=None, out=a("o", blocks), scratch=a("sc", blocks), consts=consts,
        channels=channels, kernel=kernel, apply_silu=False,
    )
    mark = len(p.get_code())
    p.kda_conv_step_v0(
        x_new=tall, x_new_row_base=first * stride, x_new_row_stride=stride,
        **common,
    )
    reads = [
        ln for ln in p.get_code()[mark:].splitlines() if "+= tall[" in ln
    ]
    want = [f"tall[{(first + cb) * stride}:{(first + cb) * stride + 1}]"
            for cb in range(blocks)]
    got = [ln.split("+= ")[1].split(" ")[0] for ln in reads]
    print(f"KDA_CONV_FEED source rows {got}")
    assert got == want, f"read {got}, the gather would have read {want}"


def test_separate_projections_do_not_help():
    """Splitting the packed projection into five changes neither cost.

    The proposal is that the gather exists because q, k, v, gate and beta share
    one output tile, and that projecting them separately -- each writing
    straight to where its consumer wants it -- would remove it. Measured at
    Kimi K3, `mlen` 64, `blen` 4, with the projection lowered through
    `linear_projection` both ways:

        packed    matmul 1,090,417 static / 13,876,267 issued, 21,560 M_MM
        separate  matmul 1,107,395 static / 13,893,245 issued, 21,560 M_MM
        gather    70 static / 4,650 issued, both

    Identical matmul work, 0.12% more instructions for the separate form from
    the extra per-projection setup, and the same gather. It cannot help,
    because the gather is forced by the `blen`-row writeback granularity rather
    than by the sharing -- see `kda_conv_step_v0`'s docstring.

    This test pins the part that is cheap to recompute: the section count and
    total block count are unchanged by splitting, which is what makes the two
    matmul costs equal. The full build is 14M instructions and too slow to run
    in CI; the numbers above were measured on 2026-08-28.
    """
    mlen = 64
    shape = KdaShape.kimi_k3()
    sections = kda_projection_sections(shape, mlen)
    packed_blocks = sum(c for _, _, c in sections)
    separate_blocks = sum(
        max(1, -(-w // mlen)) for w in (
            shape.projection_size, shape.projection_size,
            shape.num_heads * shape.value_dim, shape.projection_size,
            shape.num_heads,
        )
    )
    print(f"KDA_PROJ packed_blocks={packed_blocks} separate_blocks={separate_blocks}")
    assert packed_blocks == separate_blocks, (
        f"packing changed the output block count ({packed_blocks} vs "
        f"{separate_blocks}); the two lowerings would then not be equal-cost "
        f"and the measurement above would need redoing"
    )


def test_what_the_gate_and_beta_gathers_would_cost_to_remove():
    """The remaining 1,176 instructions, and why they stay.

    `kda_conv_step_v0` reading at a stride removed q, k and v. Gate and beta
    reach `kda_decay_scalars_v0` and `kda_beta_scalars_v0` instead, and those
    two are not symmetric:

    * **beta** is only single-tile work -- `_kda_sigmoid_inplace` then
      `tile_row_to_fpram`, both taking an explicit row list, and
      `tile_row_to_fpram` addresses FPRAM by *position* in that list rather
      than by row index. A stride would drop straight in. It is worth 18
      instructions.
    * **gate** is not. `kda_decay_scalars_v0` opens with
      `tile_row_add(gate, dt_bias, rows=rows)`, and underneath,
      `_emit_tile_row_vector_op` derives one address progression and steps
      both operands by it. A strided `gate` against a dense `dt_bias` needs
      that emitter to take `(dst, src)` row pairs the way `_emit_tile_row_fma`
      already takes triples. It is worth 1,158.

    Together **1,176 issued instructions, 0.24% of the state-engine path**, for
    a change to a row-op emitter that every kernel on the branch shares. The
    other way round -- laying `dt_bias` out at the same stride so one row list
    still serves both -- needs no ISA change but pads it from 192 rows to 768,
    which is 74 KiB more per layer and 6.8 MiB more HBM traffic per token
    across 93 layers, to save 1,176 instructions. Both trades are bad.

    So they stay, and this test records the price rather than the intention.
    If the row-op emitter ever grows `(dst, src)` pairs for another reason,
    gate becomes free and this should be revisited.
    """
    mlen = 64
    shape = KdaShape.kimi_k3()
    p = PlenaCompiler(mlen=mlen, blen=4)
    consts = p.kda_fp_constants()
    projected = p.alloc(
        "proj", p.blen, kda_projection_width(shape, mlen), strict=False
    )
    remaining = 0
    for name, _first, count in kda_projection_sections(shape, mlen):
        if name not in ("gate", "beta"):
            continue
        dst = p.alloc(f"d_{name}", max(count, mlen), mlen, strict=False)
        mark = len(p.get_code())
        p.kda_gather_projection_v0(
            projected=projected, dst=dst, section=name, shape=shape,
            consts=consts, name=f"t_{name}",
        )
        remaining += dynamic_count(p.get_code()[mark:])
    share = remaining / 492_681
    print(f"KDA_GATHER_REMAINING dynamic={remaining} share={share:.3%}")
    assert remaining < 1_500, (
        f"the gate and beta gathers are {remaining} issued instructions; they "
        f"were 1,176, and a rise means a section grew or the sweep unrolled"
    )
    assert share < 0.005, (
        f"at {share:.3%} of the state-engine path the trade above should be "
        f"re-examined; it was 0.24% when the decision was made"
    )
