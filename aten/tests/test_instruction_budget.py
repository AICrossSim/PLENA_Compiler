"""Static instruction-count gates for the recurrent kernels.

The only guard against a lowering change that is numerically correct but
quietly unrolls a sweep that used to be a hardware loop. Unrolling is what makes
a whole-model program image tens of megabytes, and no numeric test can see it.

What these gates assert is that the **image** is flat in the contracted
dimension. Before Task 8 both kernels emitted a `copy + multiply + add` triple
per state row, which could not become a hardware loop -- the scratch row it
staged through was the *same* row every iteration, so the destination never
formed an arithmetic progression -- and the image therefore grew linearly:

    KDA decode, one head:   key_dim  8 ->   505    Mamba decode, one head:
                            key_dim 16 ->   961        state  4 -> 200
                            key_dim 64 -> 3,697        state  8 -> 352
                            key_dim128 -> 7,345        state 16 -> 656
                                                       state 32 -> 1,384

After the conversion onto `V_FMA_VF` each sweep is one instruction inside one
hardware loop, so the image is constant: **76** per (head, block) for KDA at
every key_dim from 4 to 128, and **54** per head for Mamba at every state_size.

**That constancy is a property of the image and of nothing else, and the
`97x` this file used to quote for it was a comparison of an unrolled program's
size against a looped program's size.** The work did not become constant and
could not have: a recurrence has to touch every state row, and no encoding
changes how many rows there are. Measured with the loops expanded (`_dynamic`,
`mlen` 8, one head), both kernels are still exactly linear --

    KDA:    16 * key_dim    + 46      (174, 302, 1,070, 2,094 at 8/16/64/128)
    Mamba:  11 * state_size + 33      (77, 121, 209, 385 at 4/8/16/32)

-- against slopes of about 57 and 38 before. What the conversion bought is a
**3.6x smaller coefficient** on a line that stays a line, plus an image that
stops growing. Both are real; only the second is a factor of 97.
`test_the_conversion_reduced_the_slope_not_the_linearity` pins the fits.

Budgets are MEASURED, not derived: run the test, read the reported count, set
the constant ~10% above it, and note the date. Raising a budget needs a line in
docs/superpowers/plans/2026-08-25-static-mamba-kda.md saying why.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.kda.shape import KdaShape  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.program_kda_common import kda_blocks  # noqa: E402
from compiler.aten.plena.instruction_stream import (  # noqa: E402
    dynamic_count as _dynamic,
    static_count as _static,
)
from compiler.aten.plena.program_mamba_common import Mamba2Shape  # noqa: E402

MLEN = 8

#: Measured 2026-08-26 after Task 8: 76 per (head, block), flat in key_dim.
KDA_DECODE_STATIC_INSTR_PER_UNIT_MAX = 85

#: Measured 2026-08-26 after Task 8: 54 per head, flat in state_size.
MAMBA_DECODE_STATIC_INSTR_PER_HEAD_MAX = 60


def _rows_up(n: int) -> int:
    return ((n + MLEN - 1) // MLEN) * MLEN


def _kda_decode_asm(num_heads: int, key_dim: int) -> tuple[str, int]:
    shape = KdaShape(
        hidden_size=num_heads * MLEN, num_heads=num_heads, key_dim=key_dim,
        value_dim=MLEN, conv_kernel=4,
    )
    p = PlenaCompiler(mlen=MLEN, blen=2)
    blocks = kda_blocks(shape, MLEN)
    state = p.alloc("state", _rows_up(num_heads * blocks * key_dim), MLEN)
    vec = lambda name: p.alloc(name, _rows_up(num_heads * blocks), MLEN)  # noqa: E731
    v, o, pred, err = vec("v"), vec("o"), vec("pred"), vec("err")
    fp = lambda name, n: p.fp_var(name, size=n)  # noqa: E731
    mark = len(p.get_code())
    p.kda_decode_step_v0(
        state=state,
        q_fp=fp("q", num_heads * key_dim), k_fp=fp("k", num_heads * key_dim),
        decay_fp=fp("d", num_heads * key_dim), beta_fp=fp("b", _rows_up(num_heads)),
        v=v, o=o, pred=pred, err=err, shape=shape,
        output_scale_fp=fp("s", 1),
    )
    return p.get_code()[mark:], num_heads * blocks


def _kda_decode(num_heads: int, key_dim: int) -> tuple[int, int]:
    asm, units = _kda_decode_asm(num_heads, key_dim)
    return _static(asm), units


def _mamba_decode_asm(num_heads: int, state_size: int) -> str:
    shape = Mamba2Shape(
        hidden_size=num_heads * MLEN, num_heads=num_heads, head_dim=MLEN,
        state_size=state_size, n_groups=1, conv_kernel=4, chunk_size=16, seq_len=1,
    )
    p = PlenaCompiler(mlen=MLEN, blen=2)
    state = p.alloc("state", _rows_up(num_heads * state_size), MLEN)
    x = p.alloc("x", _rows_up(num_heads), MLEN)
    y = p.alloc("y", _rows_up(num_heads), MLEN)
    scratch = p.alloc("scratch", MLEN, MLEN)
    mark = len(p.get_code())
    p.ssm_decode_step_v0(
        state=state, x=x,
        b_fp=p.fp_var("b", size=state_size), c_fp=p.fp_var("c", size=state_size),
        da_fp=p.fp_var("da", size=num_heads), dt_fp=p.fp_var("dt", size=num_heads),
        d_fp=p.fp_var("d", size=num_heads), y=y, scratch=scratch,
        shape=shape, consts=p.mamba_fp_constants(),
    )
    return p.get_code()[mark:]


def _mamba_decode(num_heads: int, state_size: int) -> int:
    return _static(_mamba_decode_asm(num_heads, state_size))


@pytest.mark.parametrize("num_heads,key_dim", [(1, 4), (1, 8), (2, 8), (4, 16), (1, 64)])
def test_kda_decode_step_static_instruction_count(num_heads, key_dim):
    static, units = _kda_decode(num_heads, key_dim)
    per_unit = static // units
    print(
        f"KDA_DECODE heads={num_heads} key_dim={key_dim} "
        f"static={static} per_unit={per_unit}"
    )
    assert per_unit <= KDA_DECODE_STATIC_INSTR_PER_UNIT_MAX, (
        f"{per_unit} static instructions per (head, block) exceeds "
        f"{KDA_DECODE_STATIC_INSTR_PER_UNIT_MAX}; a sweep probably fell off the "
        f"hardware-loop path"
    )


@pytest.mark.parametrize("num_heads,state_size", [(1, 4), (2, 8), (4, 16), (8, 16)])
def test_mamba_decode_step_static_instruction_count(num_heads, state_size):
    static = _mamba_decode(num_heads, state_size)
    per_head = static // num_heads
    print(
        f"MAMBA_DECODE heads={num_heads} state={state_size} "
        f"static={static} per_head={per_head}"
    )
    assert per_head <= MAMBA_DECODE_STATIC_INSTR_PER_HEAD_MAX, (
        f"{per_head} static instructions per head exceeds "
        f"{MAMBA_DECODE_STATIC_INSTR_PER_HEAD_MAX}; a sweep probably fell off the "
        f"hardware-loop path"
    )


def test_the_cost_is_flat_in_the_contracted_dimension():
    """The claim the budgets encode, asserted directly.

    A budget alone would still pass if the cost grew and someone raised it.
    This fails on *any* growth, which is what the hardware loop actually buys.
    """
    kda = {key_dim: _kda_decode(1, key_dim)[0] for key_dim in (4, 8, 16, 64)}
    assert len(set(kda.values())) == 1, (
        f"KDA decode cost varies with key_dim: {kda}; before Task 8 it was "
        f"505 / 961 / 3,697 at key_dim 8 / 16 / 64"
    )

    mamba = {n: _mamba_decode(1, n) for n in (4, 8, 16, 32)}
    assert len(set(mamba.values())) == 1, (
        f"Mamba decode cost varies with state_size: {mamba}; before Task 8 it "
        f"was 200 / 352 / 656 at state_size 4 / 8 / 16"
    )


# ===========================================================================
# Whole-model scale.
#
# The static footprint of a model's recurrent path, which is the part this work
# changed. The projections, MoE blocks, norms and embeddings are shared code this
# work did not touch, and no whole-model program has been compiled here -- so what
# is gated is one layer's state-engine path times the layer count, and that is
# stated rather than presented as a whole-model total.
#
# The layer count and head count below are a projected K3 shape, not a released
# model: doc/Model_Lib/kimi-linear-48b-a3b.json is hidden 2304 with 32 KDA heads
# over 27 layers. The per-head geometry is the same (head_dim 128, conv kernel 4),
# so the per-head numbers carry across; anything multiplied by heads or layers
# does not.
#
# The layer *does* assemble end to end: gather, three convolutions, the gates and
# the recurrence, matching `kda_state_engine_step` -- see
# `test_kda_layer.py::test_the_assembled_layer_matches_kda_state_engine_step`.
# An earlier version of this comment claimed it could not, on the grounds that
# two emitters would first need a column-block index. That was wrong twice over;
# `program_kda_layer`'s module docstring has the correction.
#
# The convolutions are the larger half of the *image* and were ungated until this
# was measured. They are not the larger half of the work -- see
# `test_the_image_and_the_work_rank_the_kernels_differently`.

#: Kimi K3, from `KdaShape.kimi_k3()`: 96 heads, key_dim and value_dim 128.
KIMI_LAYERS = 93

#: Measured 2026-08-26 at mlen 64. Set ~10% above; raising it needs a line in
#: docs/superpowers/plans/2026-08-25-static-mamba-kda.md saying why.
KIMI_MIXER_STATIC_PER_LAYER_MAX = 43_500


def _kimi_mixer_asm() -> str:
    from compiler.aten.plena.program_kda_common import (
        kda_state_rows,
        kda_vector_rows,
    )
    from compiler.aten.plena.program_kda_gates import kda_head_blocks, kda_key_blocks
    from compiler.aten.plena.program_kda_mixer import KdaMixerBuffers

    mlen = 64
    shape = KdaShape.kimi_k3()
    p = PlenaCompiler(mlen=mlen, blen=4)
    kb = kda_key_blocks(shape, mlen)
    up = lambda n: ((n + mlen - 1) // mlen) * mlen  # noqa: E731
    a = lambda n, r: p.alloc(n, up(r), mlen, strict=False)  # noqa: E731
    decay = p.fp_var("decay_and_q_hat", size=shape.key_dim)
    buffers = KdaMixerBuffers(
        q=a("q", shape.num_heads * kb), k=a("k", shape.num_heads * kb),
        v=a("v", kda_vector_rows(shape, mlen)),
        gate=a("gate", shape.num_heads * kb),
        dt_bias=a("dt_bias", shape.num_heads * kb),
        beta_logit=a("beta_logit", kda_head_blocks(shape, mlen)),
        state=a("state", kda_state_rows(shape, mlen)),
        out=a("out", kda_vector_rows(shape, mlen)),
        pred=a("pred", kda_vector_rows(shape, mlen)),
        err=a("err", kda_vector_rows(shape, mlen)),
        sq_scratch=a("sq_scratch", shape.num_heads * kb),
        decay_fp=decay, q_hat_fp=decay,   # one window; see kda_decode_predict_v0
        k_hat_fp=p.fp_var("k_hat", size=shape.key_dim),
        beta_fp=p.fp_var("beta", size=kda_head_blocks(shape, mlen) * mlen),
        part_fp=p.fp_var("part", size=kb), acc_fp=p.fp_var("acc", size=1),
        output_scale_fp=p.fp_var("output_scale", size=1),
        rate_fp=p.fp_var("rate", size=shape.num_heads),
        lower_bound_fp=p.fp_var("lower_bound", size=1),
        consts=p.kda_fp_constants(),
    )
    mark = len(p.get_code())
    p.kda_beta_scalars_v0(
        beta_logit=buffers.beta_logit, beta_fp=buffers.beta_fp,
        consts=buffers.consts, shape=shape,
    )
    p.kda_mixer_step_v0(buffers=buffers, shape=shape)
    return p.get_code()[mark:]


def _kimi_mixer_static() -> int:
    return _static(_kimi_mixer_asm())


def test_kimi_k3_mixer_static_instruction_count():
    per_layer = _kimi_mixer_static()
    print(
        f"KIMI_K3_MIXER per_layer={per_layer} layers={KIMI_LAYERS} "
        f"total={per_layer * KIMI_LAYERS}"
    )
    assert per_layer <= KIMI_MIXER_STATIC_PER_LAYER_MAX, (
        f"{per_layer} static instructions per layer exceeds "
        f"{KIMI_MIXER_STATIC_PER_LAYER_MAX}; a sweep probably fell off the "
        f"hardware-loop path"
    )


def _fit_line(points: list[tuple[int, int]]) -> tuple[float, float]:
    """Exact `(slope, intercept)` for points that lie on a line, else raise."""
    (x0, y0), (x1, y1) = points[0], points[1]
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0
    for x, y in points:
        if abs(slope * x + intercept - y) > 1e-9:
            raise AssertionError(f"not collinear at x={x}: {y} vs {slope * x + intercept}")
    return slope, intercept


def test_the_conversion_reduced_the_slope_not_the_linearity():
    """What `V_FMA_VF` actually bought, measured on the issue stream.

    This file quoted `7,345 -> 76, a factor of 97` and the phrase "the cost
    went from linear in the contracted dimension to constant". Both describe
    the program image. The image did go constant, and 97 is the right number
    for it -- but it is the ratio between an unrolled program's *size* and a
    looped program's *size*, and nothing about a hardware loop makes its body
    run fewer times.

    Expanded, both kernels are still exactly linear in the contracted
    dimension, and have to be: a recurrence touches every state row and the
    encoding cannot change how many rows there are. What changed is the
    coefficient -- the `copy + multiply + add` triple became one FMA, so the
    per-row cost fell by about 3.6x. That is the honest figure for the
    conversion's effect on time, and it is 27x smaller than the one this file
    used to report.

    The budgets above stay image budgets, which is correct: they exist to
    catch a sweep falling off the hardware-loop path, and that is an image
    failure needing an image measurement.
    """
    kda = [(kd, _dynamic(_kda_decode_asm(1, kd)[0])) for kd in (8, 16, 64, 128)]
    mamba = [(ss, _dynamic(_mamba_decode_asm(1, ss))) for ss in (4, 8, 16, 32)]
    kda_slope, kda_c = _fit_line(kda)
    mamba_slope, mamba_c = _fit_line(mamba)
    print(
        f"KDA_DECODE dynamic = {kda_slope:.0f}*key_dim + {kda_c:.0f}  {kda}\n"
        f"MAMBA_DECODE dynamic = {mamba_slope:.0f}*state_size + {mamba_c:.0f}  {mamba}"
    )

    # The image is flat over the same range -- that is the budgeted property.
    assert len({_static(_kda_decode_asm(1, kd)[0]) for kd, _ in kda}) == 1
    assert len({_static(_mamba_decode_asm(1, ss)) for ss, _ in mamba}) == 1

    # The work is not. Slopes measured 2026-08-27; bounds are ~40% either side.
    assert 10 <= kda_slope <= 24, f"KDA slope {kda_slope} left its measured band"
    assert 7 <= mamba_slope <= 16, f"Mamba slope {mamba_slope} left its measured band"

    # Pre-conversion slopes, from the table in this module's docstring:
    # KDA (7,345-505)/(128-8) = 57.0, Mamba (1,384-200)/(32-4) = 42.3.
    assert (57.0 / kda_slope) < 10, (
        f"the conversion cut the KDA slope by {57.0 / kda_slope:.1f}x; if this "
        f"ever approaches the image's 97x, the two measurements have been "
        f"conflated again"
    )
    assert (42.3 / mamba_slope) < 10, (
        f"the conversion cut the Mamba slope by {42.3 / mamba_slope:.1f}x"
    )


def test_the_recurrence_is_the_bulk_of_the_layers_work():
    """The recurrence dominates the layer, and the image says the opposite.

    The old version of this test multiplied a per-unit *image* count by the
    unit count and compared it against the layer's *image*, concluding that
    without the `V_FMA_VF` conversion the recurrence would be 97% of the
    layer. Two image numbers divided cannot say what fraction of a layer
    anything is.

    Measured on the issue stream, the mixer -- which is the recurrence plus
    the gates around it -- is 87% of the layer against the three
    convolutions' 12%, while by image the convolutions are the larger half.
    So the conclusion the old test reached happens to survive; its arithmetic
    did not.
    """
    conv = sum(_dynamic(a) for a in _kimi_conv_asm())
    mixer = _dynamic(_kimi_mixer_asm())
    layer = conv + mixer + _dynamic(_kimi_gather_asm())
    print(f"KIMI_K3 mixer share of issued layer = {mixer / layer:.1%}")
    assert mixer / layer > 0.7, (
        f"the mixer is {mixer / layer:.1%} of the issued layer; it was 87% when "
        f"measured, and a large fall means work moved out of the recurrence"
    )


#: Measured 2026-08-26 at mlen 64, Kimi K3. The three convolutions, 58% of the
#: layer's *image* -- larger than the mixer, and previously ungated. Their share
#: of the issued stream is 12%; the two orderings are opposite.
KIMI_CONV_STATIC_PER_LAYER_MAX = 59_000

#: The whole state-engine path: gather + conv x3 + gates + recurrence.
KIMI_LAYER_STATIC_MAX = 103_000


def _kimi_conv_asm() -> list[str]:
    from compiler.aten.plena.program_kda_conv import kda_conv_blocks

    mlen = 64
    shape = KdaShape.kimi_k3()
    out: list[str] = []
    for channels in (
        shape.projection_size, shape.projection_size,
        shape.num_heads * shape.value_dim,
    ):
        p = PlenaCompiler(mlen=mlen, blen=4)
        consts = p.kda_fp_constants()
        blocks = kda_conv_blocks(channels, mlen)
        up = lambda n: ((n + mlen - 1) // mlen) * mlen  # noqa: E731
        a = lambda n, r: p.alloc(n, up(r), mlen, strict=False)  # noqa: E731
        mark = len(p.get_code())
        p.kda_conv_step_v0(
            x_new=a("x", blocks), conv_state=a("cs", blocks * shape.conv_kernel),
            weight=a("w", blocks * shape.conv_kernel), bias=None,
            out=a("o", blocks), scratch=a("sc", blocks), consts=consts,
            channels=channels, kernel=shape.conv_kernel,
        )
        out.append(p.get_code()[mark:])
    return out


def _kimi_conv_static() -> int:
    return sum(_static(a) for a in _kimi_conv_asm())


def test_kimi_k3_conv_static_instruction_count():
    """The convolutions, the larger half of the layer's program image.

    They were ungated while the mixer was, which had the budget covering 42% of
    the thing it was meant to protect.

    "Larger half" is a statement about the image and nothing else. The
    convolutions issue 59,409 instructions against the mixer's 428,622, so by
    work they are the smaller *eighth*. Both budgets are image budgets and are
    right to be -- they guard against unrolling -- but neither ranks the
    kernels.
    """
    total = _kimi_conv_static()
    print(f"KIMI_K3_CONV q+k+v static={total}")
    assert total <= KIMI_CONV_STATIC_PER_LAYER_MAX, (
        f"{total} static instructions for the three convolutions exceeds "
        f"{KIMI_CONV_STATIC_PER_LAYER_MAX}"
    )


def test_kimi_k3_layer_static_instruction_count():
    """Gather + conv x3 + gates + recurrence, and the 93-layer total."""
    gather = 70          # five sections, one hardware loop each; see test_kda_layer
    layer = _kimi_conv_static() + _kimi_mixer_static() + gather
    print(
        f"KIMI_K3_LAYER static={layer} layers={KIMI_LAYERS} "
        f"total={layer * KIMI_LAYERS}"
    )
    assert layer <= KIMI_LAYER_STATIC_MAX, (
        f"{layer} static instructions per layer exceeds {KIMI_LAYER_STATIC_MAX}"
    )


def _kimi_gather_asm() -> str:
    """The five-section projection gather, at Kimi K3's real block counts."""
    from compiler.aten.plena.program_kda_layer import (
        kda_projection_sections,
        kda_projection_width,
    )

    mlen = 64
    shape = KdaShape.kimi_k3()
    p = PlenaCompiler(mlen=mlen, blen=4)
    consts = p.kda_fp_constants()
    projected = p.alloc("proj", 64, kda_projection_width(shape, mlen), strict=False)
    mark = len(p.get_code())
    for name, _first, count in kda_projection_sections(shape, mlen):
        dst = p.alloc(f"dst_{name}", max(count, mlen), mlen, strict=False)
        p.kda_gather_projection_v0(
            projected=projected, dst=dst, section=name, shape=shape,
            consts=consts, name=f"tall_{name}",
        )
    return p.get_code()[mark:]


#: Measured 2026-08-27 at mlen 64, Kimi K3: 4,650 executed against 492,681 for
#: the layer, 0.94%. Set ~50% above; the ratio, not the absolute, is the claim.
KIMI_GATHER_DYNAMIC_SHARE_MAX = 0.015


def test_the_gather_is_one_percent_of_the_layer_when_the_loops_are_expanded():
    """The projection split cost, measured as issued instructions.

    This test previously divided **static** counts -- 70 against 93,353 -- and
    reported 0.07%. That number was real but it answered the wrong question:
    the gather is one hardware loop per section and its body runs once per
    feature block, 192 of them for q/k/v/gate. Expanded, the gather issues
    4,650 instructions against 492,681 for the layer, and the honest figure is
    **0.94%** -- thirteen times the static one.

    Both numbers matter and they are not interchangeable. The static count is
    what the `V_FMA_VF` conversion bought (a program image that does not grow
    with `key_dim`) and it belongs in the budget gates above. The dynamic count
    is what the gather *costs*, and it is the only one that may be compared
    against an alternative way of getting the sections into place.

    0.94% is still small enough that no amount of making the gather faster
    changes the layer, which is the conclusion the old test was reaching for.
    It is not small enough to call the gather free, which is what it said.
    """
    gather = _dynamic(_kimi_gather_asm())
    layer = (
        sum(_dynamic(a) for a in _kimi_conv_asm())
        + _dynamic(_kimi_mixer_asm())
        + gather
    )
    share = gather / layer
    print(
        f"KIMI_K3_GATHER dynamic={gather} layer_dynamic={layer} "
        f"share={share:.4%} (static share was {70 / 93_353:.4%})"
    )
    assert share <= KIMI_GATHER_DYNAMIC_SHARE_MAX, (
        f"the gather is {share:.3%} of the issued layer, over the "
        f"{KIMI_GATHER_DYNAMIC_SHARE_MAX:.1%} budget"
    )
    assert share > 10 * (70 / 93_353), (
        "the dynamic share should be an order of magnitude above the static "
        "one; if it is not, the gather stopped being a hardware loop"
    )


def test_the_image_and_the_work_rank_the_kernels_differently():
    """The convolutions are the larger half of the image and an eighth of the work.

    Pinned because the inversion is severe enough to misdirect effort, and it
    already did: the comments in this file called the convolutions "the larger
    half of the layer" on the strength of 53,757 against 39,526. Expanded, the
    convolutions issue 59,409 and the mixer 428,622 -- the mixer is 87% of the
    layer and the convolutions 12%, the opposite order.

    The reason is structural rather than incidental. The convolutions are a
    four-tap FIR over feature blocks: the image carries one instruction per tap
    per block because the taps are separate weights, and almost nothing loops.
    The mixer is the recurrence, which the `V_FMA_VF` conversion collapsed into
    hardware loops whose bodies run once per key block per head -- a small
    image over a large trip count, which is exactly the shape the image cannot
    see.

    So this file's budgets, all of which are image budgets, must never be read
    as a cost ranking. They gate one failure mode -- a sweep falling off the
    hardware-loop path and unrolling -- and they are the right instrument for
    it. Anything about where the layer's time goes has to come from `_dynamic`.
    """
    conv = sum(_dynamic(a) for a in _kimi_conv_asm())
    mixer = _dynamic(_kimi_mixer_asm())
    gather = _dynamic(_kimi_gather_asm())
    layer = conv + mixer + gather
    print(
        f"KIMI_K3_LAYER dynamic conv={conv} ({conv / layer:.1%}) "
        f"mixer={mixer} ({mixer / layer:.1%}) gather={gather} "
        f"({gather / layer:.1%}) layer={layer}"
    )
    assert _kimi_conv_static() > _kimi_mixer_static(), (
        "the convolutions should still be the larger half of the image"
    )
    assert mixer > 3 * conv, (
        f"the mixer should dominate the issued stream ({mixer} vs {conv}); if it "
        f"no longer does, either a recurrence sweep unrolled or a convolution "
        f"grew a loop, and the budgets above cannot tell you which"
    )


# ---------------------------------------------------------------------------
# The input projection, which no budget covered until 2026-08-28.

#: Measured at Kimi K3, `mlen` 64, `blen` 4, `mram_tile_capacity` 64 -- the
#: capacity the shipped transactional config implies. Set ~10% above.
KIMI_PROJECTION_STATIC_MAX = 430_000
KIMI_PROJECTION_DYNAMIC_MAX = 5_470_000

#: `PlenaCompiler`'s unconditional default until 2026-08-28.
LEGACY_MRAM_TILES = 4


def _shipped_config_mode(mlen: int) -> str | None:
    """The config section declaring `mlen`, or None if the file is not here.

    `plena_settings.toml` lives in the **simulator** repository; this one is a
    submodule of it. A checkout of the compiler alone -- which is what its own
    CI does -- has no such file, `_find_plena_settings_toml` returns None, and
    every config-derived value falls back to its hardcoded default.

    That split is worth naming rather than papering over. It means the compiler
    emits different programs standalone than it does inside the simulator
    checkout, and it is why the reader could be broken in two ways at once for
    as long as it was: the job that would have caught it never had a file to
    read.
    """
    from compiler.aten.plena.compiler import _config_mode_for_mlen, _find_plena_settings_toml

    path = _find_plena_settings_toml()
    # `_find_plena_settings_toml` returns PLENA_SETTINGS_TOML unchecked, so a
    # stale or deliberately bogus env var gives a path that is not there.
    if path is None or not path.exists():
        return None
    return _config_mode_for_mlen(mlen)

#: `MatrixSram::new(tile_size = MLEN, depth = MATRIX_SRAM_SIZE)` keeps
#: `depth / tile_size` tiles, so the shipped transactional config -- MLEN 64,
#: MATRIX_SRAM_SIZE 4096 -- is 64. This is now what `PlenaCompiler(mlen=64)`
#: derives.
CONFIGURED_MRAM_TILES = 64


def _kimi_projection_asm(mram_tile_capacity: int) -> str:
    """`X @ W_in` for one Kimi K3 KDA layer, through `linear_projection`."""
    from compiler.aten.plena.program_kda_layer import kda_projection_width

    mlen, blen = 64, 4
    shape = KdaShape.kimi_k3()
    p = PlenaCompiler(mlen=mlen, blen=blen, mram_tile_capacity=mram_tile_capacity)
    x = p.load_batch(p.input("x", (blen, shape.hidden_size)), name="xv")
    weight = p.input("W_in", (shape.hidden_size, kda_projection_width(shape, mlen)))
    mark = len(p.get_code())
    p.linear_projection(x, weight, name="projected")
    return p.get_code()[mark:]


def test_kimi_k3_projection_instruction_budget():
    """The largest kernel in the layer, and the last one to get a gate.

    The projection was never in the KDA lowering -- `test_kda_layer.py` loads
    its result rather than computing it -- so nothing here covered it. Built,
    it is 389,717 static and 4,967,367 issued instructions for one layer,
    against 93,353 and 492,681 for the whole state-engine path.

    Those two pairs must **not** be added, ranked, or turned into a
    percentage. An `M_MM` is a `64 x 64 x 64` matmul and a `V_FMA_VF` is a
    64-lane vector operation; counting them in the same unit says nothing
    about time. The gate exists because a kernel this size should not be
    outside every budget, not because the comparison means anything.
    """
    asm = _kimi_projection_asm(CONFIGURED_MRAM_TILES)
    static, dynamic = _static(asm), _dynamic(asm)
    print(f"KIMI_K3_PROJECTION static={static} dynamic={dynamic}")
    assert static <= KIMI_PROJECTION_STATIC_MAX, (
        f"{static} static instructions exceeds {KIMI_PROJECTION_STATIC_MAX}"
    )
    assert dynamic <= KIMI_PROJECTION_DYNAMIC_MAX, (
        f"{dynamic} issued instructions exceeds {KIMI_PROJECTION_DYNAMIC_MAX}"
    )


def test_the_mram_tile_capacity_now_comes_from_the_configuration():
    """It was an unconditional 4, and the machine it compiles for has 64.

    `PlenaCompiler` took `mram_tile_capacity: int = 4` and nothing derived it
    from `MATRIX_SRAM_SIZE`. The emulator's `MatrixSram::new` keeps
    `depth / tile_size` tiles, so the shipped transactional config -- MLEN 64
    over MATRIX_SRAM_SIZE 4096 -- is 64 tiles, sixteen times the default.

    Kimi K3's projection contracts over 112 k-tiles and `linear_projection`
    re-streams the weights once per capacity-sized chunk, so the gap was
    expensive:

        cap   4 (32 KiB)    1,090,417 static   13,876,267 issued
        cap  16 (128 KiB)     524,467 static    6,680,617 issued
        cap  64 (512 KiB)     389,717 static    4,967,367 issued
        cap 112 (896 KiB)     362,767 static    4,624,717 issued

    The default now derives from the config section whose `MLEN` is the one
    being compiled for -- not from `[MODE].active`, which selects the machine
    the *simulator* models and is `analytic` (MLEN 2048) while every program
    here is compiled at 64 and run by the transactional emulator. A shape no
    section declares, such as the `mlen` 8 used by most tests in this file,
    keeps the old default and is unaffected.
    """
    # An `mlen` no section declares, and an explicit override, hold with or
    # without the file -- so they are checked before the skip.
    assert PlenaCompiler(mlen=MLEN, blen=2).mram_tile_capacity == LEGACY_MRAM_TILES
    assert PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=7).mram_tile_capacity == 7
    if _shipped_config_mode(64) is None:
        pytest.skip("no plena_settings.toml declaring MLEN 64 on this checkout")
    assert PlenaCompiler(mlen=64, blen=4).mram_tile_capacity == CONFIGURED_MRAM_TILES

    default = _dynamic(_kimi_projection_asm(LEGACY_MRAM_TILES))
    configured = _dynamic(_kimi_projection_asm(CONFIGURED_MRAM_TILES))
    ratio = default / configured
    print(
        f"KIMI_K3_PROJECTION cap={LEGACY_MRAM_TILES} dynamic={default} "
        f"cap={CONFIGURED_MRAM_TILES} dynamic={configured} ratio={ratio:.2f}"
    )
    assert 2.2 <= ratio <= 3.5, (
        f"the old default cost {ratio:.2f}x the configured capacity; it was "
        f"2.79x when measured, and a move means either the config or the "
        f"K-split strategy changed"
    )


def test_the_analytic_config_cannot_hold_one_matrix_tile():
    """`256 / 2048 = 0`, and a zero-tile capacity has no sane fallback.

    `ANALYTIC` declares MLEN 2048 and MATRIX_SRAM_SIZE 256, so the emulator's
    own arithmetic gives it no whole matrix tile at all. Nothing reads it
    today -- that mode does not drive the transactional emulator -- but the
    derivation must not quietly substitute a number nobody chose, because a
    contraction split into zero-tile chunks does not terminate. It raises, and
    names both figures.
    """
    # An explicit capacity overrides whatever the config says, so the mode
    # stays compilable either way. True with or without the file.
    assert PlenaCompiler(mlen=2048, blen=4, mram_tile_capacity=1).mram_tile_capacity == 1
    if _shipped_config_mode(2048) is None:
        pytest.skip("no plena_settings.toml declaring MLEN 2048 on this checkout")
    with pytest.raises(ValueError, match="MATRIX_SRAM_SIZE 256"):
        PlenaCompiler(mlen=2048, blen=4)


def test_the_configuration_is_actually_read():
    """Two independent reasons it never was, both fixed here.

    `_behavior_config_value` asked `load_toml_config` for `[BEHAVIOR].CONFIG`
    and fell back to a caller-supplied default on any exception. It took that
    fallback every single time, for two reasons at once:

    1. `plena_settings.toml` has no `[BEHAVIOR]` table. It ships `[MODE]`,
       `[ANALYTIC.*]` and `[TRANSACTIONAL.*]`.
    2. `load_toml_config` required the third-party `toml` package and raised
       `ImportError` without it. `toml` is not a declared dependency and CI
       installs `pytest pyyaml`, so the import failed there too -- and the
       `except Exception` around the call turned "cannot read the config" into
       "the default is correct".

    So `PlenaCompiler`'s docstring promised HLEN, BROADCAST_AMOUNT and the
    prefetch amounts came from the TOML, and none of them ever did. The reader
    now goes through `tomllib`, which is stdlib from 3.11, and searches
    `BEHAVIOR` then the section matching `mlen`.

    At `mlen` 64 that moves HLEN from 64 to the transactional config's 16 and
    BROADCAST_AMOUNT from 1 to 4. Neither changes an emitted program: `hlen` is
    read only by a packed-attention precondition, whose one test sets both
    attributes by hand at `mlen` 256, and `broadcast_amount` is read nowhere
    outside the constructor. The prefetch and writeback amounts, which do reach
    the emitters, are 4 in that section and were already 4.
    """
    from compiler.aten.plena.compiler import (  # noqa: F401
        _config_mode_for_mlen,
        _config_section,
        _find_plena_settings_toml,
    )

    settings = _find_plena_settings_toml()
    if settings is None or not settings.exists():
        pytest.skip(
            "no plena_settings.toml on this checkout -- it lives in the "
            "simulator repository, and the compiler's own CI checks out only "
            "this one, which is why nothing here ever caught the reader"
        )

    assert _config_section("TRANSACTIONAL"), (
        "the transactional CONFIG table came back empty; the reader is not "
        "reading the file"
    )
    assert _config_section("BEHAVIOR") == {}, (
        "a [BEHAVIOR] table now exists; it takes priority over the machine "
        "sections and this test's expectations below need rechecking"
    )
    assert _config_mode_for_mlen(64) == "TRANSACTIONAL"
    assert _config_mode_for_mlen(2048) == "ANALYTIC"
    assert _config_mode_for_mlen(MLEN) is None, (
        f"mlen {MLEN} now matches a declared machine; the tests in this file "
        f"would start reading its numbers instead of the defaults"
    )

    p = PlenaCompiler(mlen=64, blen=4)
    print(
        f"CONFIG_AT_MLEN_64 hlen={p.hlen} broadcast={p.broadcast_amount} "
        f"prefetch={p.hbm_v_prefetch_amount} writeback={p.hbm_v_writeback_amount} "
        f"mram_tiles={p.mram_tile_capacity}"
    )
    assert (p.hlen, p.broadcast_amount) == (16, 4), (
        "these are the transactional config's values; before the fix they were "
        "64 and 1, which is what the hardcoded defaults gave"
    )
    assert (p.hbm_v_prefetch_amount, p.hbm_v_writeback_amount) == (4, 4), (
        "these reach the emitters, so a change here moves every load_batch on "
        "the branch and needs its own decision"
    )
