"""KDA lowering: state layout, data movement, and normalisation.

Numerical validation against the CPU reference lives in the emulator testbench;
what these tests can check without an emulator is that the emitted ISA has the
shape the design requires -- the transposed state tile, load/store agreeing on
precision, and normalisation using the ISA's scalar sqrt rather than a vector op
that does not exist.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.models.kda.reference import KdaShape  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.program_kda_common import (  # noqa: E402
    kda_blocks,
    kda_conv_blocks,
    kda_conv_state_row,
    kda_conv_channels,
    kda_head_base,
    kda_state_row,
    kda_state_rows,
    kda_vector_row,
    kda_vector_rows,
)
from compiler.aten.plena.state_precision import (  # noqa: E402
    STATE_ELEMENT_BYTES,
    STATE_PRECISION_SELECTOR,
)

MLEN = 8

#: PLENA stores KDA recurrent state as BF16 in its own HBM precision class.
STATE_BYTES_PER_ELEMENT = STATE_ELEMENT_BYTES


def _vram_base(prog, var) -> int:
    """Byte address of a VRAM tile, via the compiler's own symbol table."""
    return prog.get_vram_layout(var.name).vram_base_addr


def _prog() -> PlenaCompiler:
    return PlenaCompiler(mlen=MLEN, blen=2)


def _shape(**kw) -> KdaShape:
    base = dict(
        hidden_size=MLEN * 4,
        num_heads=2,
        key_dim=4,
        value_dim=MLEN,
        conv_kernel=4,
    )
    base.update(kw)
    return KdaShape(**base)


def _body(code: str) -> list[str]:
    return [
        line.strip()
        for line in code.splitlines()
        if line.strip() and not line.strip().startswith((";", "//"))
    ]


def _trace(code: str) -> list[tuple[str, list[str], dict[str, int]]]:
    """Emitted instructions with GP registers resolved to their last constant.

    Opcode-presence assertions are close to worthless here: `V_MUL_VF` appears
    because `vram_fill_zero` zeroes a tile, `V_RED_SUM` appears as long as any
    reduction survives. What distinguishes a correct emitter from a broken one
    is *which addresses* the vector ops read and write, and those arrive through
    `S_ADDI_INT gpN, gp0, <imm>`. This resolves that much of the dataflow --
    enough to pin operands, not enough to be a simulator.
    """
    regs: dict[str, int] = {"gp0": 0}
    out: list[tuple[str, list[str], dict[str, int]]] = []
    for line in _body(code):
        parts = line.replace(",", " ").split()
        op, args = parts[0], parts[1:]
        if op == "S_ADDI_INT" and len(args) == 3:
            dst, src, imm = args
            if src in regs and imm.lstrip("-").isdigit():
                regs[dst] = regs[src] + int(imm)
            else:
                regs.pop(dst, None)
        out.append((op, args, dict(regs)))
    return out


def _operands(trace, opcode: str) -> list[tuple[int, ...]]:
    """Resolved addresses of every occurrence of `opcode`, in order."""
    found = []
    for op, args, regs in trace:
        if op != opcode:
            continue
        vals = tuple(regs[a] for a in args if a in regs)
        found.append(vals)
    return found


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def test_state_rows_covers_every_head_block_and_key():
    shape = _shape(num_heads=3, key_dim=5, value_dim=MLEN)
    assert kda_blocks(shape, MLEN) == 1
    assert kda_state_rows(shape, MLEN) == 15
    assert [kda_head_base(shape, MLEN, h) for h in range(3)] == [0, 5, 10]

    wide = _shape(num_heads=3, key_dim=5, value_dim=MLEN * 2)
    assert kda_blocks(wide, MLEN) == 2
    assert kda_state_rows(wide, MLEN) == 30


def test_flattened_layout_is_a_bijection():
    """Every (head, block, key) gets its own row, and together they tile the
    whole state exactly -- no overlap, no gap. Overlap is one head's sweep
    corrupting another's; a gap is a row nothing ever writes."""
    shape = _shape(num_heads=4, key_dim=6, value_dim=MLEN * 3)
    blocks = kda_blocks(shape, MLEN)
    rows = [
        kda_state_row(shape, MLEN, h, c, j)
        for h in range(shape.num_heads)
        for c in range(blocks)
        for j in range(shape.key_dim)
    ]
    assert len(rows) == len(set(rows)), "two (head, block, key) share a row"
    assert set(rows) == set(range(kda_state_rows(shape, MLEN)))


def test_row_ordering_is_head_then_block_then_key():
    """Pin the ordering, not merely that it is *some* bijection.

    A bijection test accepts a head/block swap, and so does a unit-stride test.
    That matters because this is the HBM byte layout of the pinned state tensor
    (kda_load_state_v0 maps row r to offset r*mlen), so once a prefill kernel or
    a checkpoint loader writes the initial state it is a cross-component
    contract. At value_dim == mlen the swap is the identity, which is why this
    has to run at blocks > 1.
    """
    shape = _shape(num_heads=3, key_dim=4, value_dim=MLEN * 2)
    blocks = kda_blocks(shape, MLEN)
    expected = [
        (h * blocks + c) * shape.key_dim
        for h in range(shape.num_heads)
        for c in range(blocks)
    ]
    assert [
        kda_state_row(shape, MLEN, h, c, 0)
        for h in range(shape.num_heads)
        for c in range(blocks)
    ] == expected

    # kda_head_base must honour its block argument, not silently return block 0
    assert [
        kda_head_base(shape, MLEN, h, c)
        for h in range(shape.num_heads)
        for c in range(blocks)
    ] == expected


def test_keys_are_consecutive_within_a_head_and_block():
    """This is what makes each sweep a single hardware loop: fixing
    (head, block) must leave the keys at unit stride."""
    shape = _shape(num_heads=3, key_dim=6, value_dim=MLEN * 2)
    for h in range(shape.num_heads):
        for c in range(kda_blocks(shape, MLEN)):
            rows = [kda_state_row(shape, MLEN, h, c, j) for j in range(shape.key_dim)]
            steps = {b - a for a, b in zip(rows, rows[1:])}
            assert steps == {1}, f"head {h} block {c} is not unit stride: {rows}"


def test_vector_rows_block_the_same_way_as_the_state():
    """q, v, the output and the accumulators are all value-width, so they must
    be blocked on the same convention -- a mixture reads block 0 of one tile
    against block 1 of another."""
    shape = _shape(num_heads=3, key_dim=4, value_dim=MLEN * 2)
    blocks = kda_blocks(shape, MLEN)
    assert kda_vector_rows(shape, MLEN) == shape.num_heads * blocks
    rows = [
        kda_vector_row(shape, MLEN, h, c)
        for h in range(shape.num_heads)
        for c in range(blocks)
    ]
    assert rows == sorted(rows) and len(set(rows)) == len(rows)


def test_layout_helpers_reject_out_of_range():
    shape = _shape(num_heads=2, key_dim=3, value_dim=MLEN * 2)
    for bad in (dict(head=2), dict(head=-1), dict(block=2), dict(key=3)):
        args = dict(head=0, block=0, key=0) | bad
        with pytest.raises(ValueError, match="out of range"):
            kda_state_row(shape, MLEN, **args)


def test_value_dim_must_be_a_whole_number_of_blocks():
    """A partial trailing block leaves lanes past value_dim holding stale data
    that the reduction would still sum."""
    shape = _shape(num_heads=1, key_dim=2, value_dim=MLEN + 3)
    with pytest.raises(ValueError, match="multiple of mlen"):
        kda_blocks(shape, MLEN)


def test_conv_channels_matches_the_reference_geometry():
    """2*key + value per head. Cross-checked against KdaShape's own accounting
    so the lowering and the reference cannot disagree about conv state size."""
    shape = _shape(num_heads=3, key_dim=5, value_dim=MLEN)
    assert kda_conv_channels(shape) == 3 * (2 * 5 + MLEN)
    assert kda_conv_channels(shape) * shape.conv_kernel == shape.conv_state_elements


# ---------------------------------------------------------------------------
# State residency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blocks", [1, 2, 3])
def test_state_loads_in_the_transposed_layout(blocks):
    prog, shape = _prog(), _shape(value_dim=MLEN * blocks)
    addr = prog.kda_pin_state_v0("kda_state", shape)
    state = prog.kda_load_state_v0("kda_state", shape, addr)
    assert state.shape == (kda_state_rows(shape, MLEN), MLEN)
    assert prog._inputs["kda_state"].hbm_size == (
        shape.num_heads * shape.key_dim * shape.value_dim * STATE_BYTES_PER_ELEMENT
    )


def _scale_reg_immediates(code: str) -> list[int]:
    """Value written to C_SET_SCALE_REG, which encodes rows*cols*bytes_per_elem."""
    return [regs[args[0]] for op, args, regs in _trace(code) if op == "C_SET_SCALE_REG"]


@pytest.mark.parametrize("blocks", [1, 2, 3])
def test_state_store_mirrors_the_load_precision(blocks):
    """A load/store precision mismatch changes the row stride, so the state read
    back is not the state written -- a wrong answer, not an error.

    Asserting `stored.hbm_addr == addr` would be a tautology: `store()` builds
    the InputVar from the address it was handed. The precision has to be read
    out of the emitted ISA, and C_SET_SCALE_REG carries it: its immediate is
    rows * value_dim * bytes_per_element on both sides.
    """
    prog, shape = _prog(), _shape(value_dim=MLEN * blocks)
    addr = prog.kda_pin_state_v0("kda_state", shape)

    mark = len(prog.get_code())
    state = prog.kda_load_state_v0("kda_state", shape, addr)
    load_code = prog.get_code()[mark:]

    mark = len(prog.get_code())
    stored = prog.kda_store_state_v0(state, "kda_state", addr)
    store_code = prog.get_code()[mark:]

    assert stored.hbm_addr == addr
    expected = kda_state_rows(shape, MLEN) * MLEN * STATE_BYTES_PER_ELEMENT
    assert _scale_reg_immediates(load_code) == [expected]
    assert _scale_reg_immediates(store_code) == [expected]

    # The precision-class operand of the transfer itself: 2 == recurrent State.
    (prefetch,) = [
        line for line in _body(load_code) if line.startswith("H_PREFETCH_V")
    ]
    (store_v,) = [
        line for line in _body(store_code) if line.startswith("H_STORE_V")
    ]
    assert prefetch.replace(",", " ").split()[-1] == str(STATE_PRECISION_SELECTOR)
    assert store_v.replace(",", " ").split()[-1] == str(STATE_PRECISION_SELECTOR)


def test_plena_state_precision_guard_accepts_only_plain_bf16():
    prog = _prog()
    bf16 = {
        "HBM_STATE_TYPE": {
            "format": "Plain",
            "DATA_TYPE": {
                "type": "Fp",
                "sign": True,
                "exponent": 8,
                "mantissa": 7,
            },
        }
    }
    prog.kda_require_state_precision_v0(bf16)
    with pytest.raises(ValueError, match="Plain BF16"):
        prog.kda_require_state_precision_v0(
            {
                "HBM_STATE_TYPE": {
                    "format": "Plain",
                    "DATA_TYPE": {
                        "type": "Fp",
                        "sign": True,
                        "exponent": 8,
                        "mantissa": 23,
                    },
                }
            }
        )


@pytest.mark.parametrize("blocks", [1, 2, 3])
def test_state_round_trips_to_the_same_address(blocks):
    """Decode reads the state at the top of a step and writes it back at the
    bottom; if the two addresses differ the next token reads stale state."""
    prog, shape = _prog(), _shape(value_dim=MLEN * blocks)
    addr = prog.kda_pin_state_v0("kda_state", shape)
    state = prog.kda_load_state_v0("kda_state", shape, addr)
    prog.kda_store_state_v0(state, "kda_state", addr)
    reloaded = prog.kda_load_state_v0("kda_state", shape, addr)
    assert reloaded.shape == state.shape
    base, size = prog.pinned_hbm_region("kda_state")
    assert base == addr
    assert size == kda_state_rows(shape, MLEN) * MLEN


@pytest.mark.parametrize("blocks", [1, 2, 3])
def test_pinned_state_region_does_not_collide_with_later_tensors(blocks):
    """pin_hbm_region must keep the auto-allocator off the state range, or a
    later input()/store() silently overwrites the carried state."""
    prog, shape = _prog(), _shape(value_dim=MLEN * blocks)
    addr = prog.kda_pin_state_v0("kda_state", shape)
    _base, size_elements = prog.pinned_hbm_region("kda_state")
    # pinned_hbm_region reports ELEMENTS; hbm_addr is BYTES. Comparing the two
    # directly is how an under-reserved region passes this test.
    size_bytes = prog.hbm_tensor_size(
        size_elements, hbm_element_bytes=STATE_BYTES_PER_ELEMENT
    )
    other = prog.input("unrelated", (MLEN, MLEN))
    assert other.hbm_addr >= addr + size_bytes


# ---------------------------------------------------------------------------
# Convolution history
# ---------------------------------------------------------------------------


def test_conv_roll_shifts_history_and_appends():
    prog, shape = _prog(), _shape(conv_kernel=4)
    # VRAM row counts must be a multiple of mlen (memory_state.py:309), so the
    # conv_kernel-1 rows of history live in the first rows of an mlen-row tile.
    conv = prog.alloc("conv", MLEN, MLEN)
    new = prog.alloc("new", MLEN, MLEN)
    mark = len(prog.get_code())
    prog.kda_conv_state_roll_v0(conv, new, 0, shape)
    code = prog.get_code()[mark:]
    assert _body(code), "roll emitted nothing"

    # Counting V_ADD_VV alone cannot tell a correct roll from one that shifts the
    # wrong way or appends from the wrong tile -- all three emit 3 copies. Pin
    # the (dst, src) row addresses instead.
    conv_base = _vram_base(prog, conv)
    new_base = _vram_base(prog, new)
    history = shape.conv_kernel - 1
    expected = [
        (conv_base + i * MLEN, conv_base + (i + 1) * MLEN) for i in range(history - 1)
    ] + [(conv_base + (history - 1) * MLEN, new_base)]
    assert _operands(_trace(code), "V_ADD_VV") == [
        (dst, dst, src) for dst, src in expected
    ]


def test_conv_roll_is_a_noop_for_kernel_one():
    prog, shape = _prog(), _shape(conv_kernel=1)
    conv = prog.alloc("conv", MLEN, MLEN)
    new = prog.alloc("new", MLEN, MLEN)
    before = prog.get_code()
    prog.kda_conv_state_roll_v0(conv, new, 0, shape)
    assert prog.get_code() == before


def test_conv_roll_rejects_an_undersized_history():
    # conv_kernel=10 needs 9 rows of history; the tile below holds mlen=8.
    prog, shape = _prog(), _shape(conv_kernel=10)
    conv = prog.alloc("conv", MLEN, MLEN)
    new = prog.alloc("new", MLEN, MLEN)
    with pytest.raises(ValueError, match="conv_state needs"):
        prog.kda_conv_state_roll_v0(conv, new, 0, shape)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_l2_normalize_uses_the_scalar_sqrt_path():
    """There is no vector square root -- S_FP_OP carries SQRT_FP, V_ELEMENT_OP
    does not -- so the reciprocal norm must come out of the scalar unit."""
    prog = _prog()
    vec = prog.alloc("qk", MLEN, MLEN)
    scratch = prog.alloc("sq", MLEN, MLEN)
    consts = prog.kda_fp_constants()
    acc = prog.fp_var("kda_norm_acc", size=2)
    rows = [0, 1]
    mark = len(prog.get_code())
    prog.kda_l2_normalize_v0(vec, rows, scratch, acc, consts)
    code = prog.get_code()[mark:]
    trace = _trace(code)
    emitted = _body(code)

    vec_base = _vram_base(prog, vec)
    scratch_base = _vram_base(prog, scratch)

    # Scalar rsqrt, one pair per row: there is no vector square root.
    assert sum("S_SQRT_FP" in line for line in emitted) == len(rows)
    assert sum("S_RECI_FP" in line for line in emitted) == len(rows)

    # The block copy that stages the vector into the scratch. mamba_block_copy
    # is zero-then-add, so its V_ADD_VV carries the direction; swapping src and
    # dst zeroes the vector and leaves every other operand below unchanged --
    # the one mutation that survived the first round of this test.
    copies = _operands(trace, "V_ADD_VV")
    assert copies, "no block copy emitted"
    assert all(
        dst == scratch_base and src == vec_base for dst, _, src in copies
    ), f"block copy runs the wrong way: {copies}"

    # The square must write the scratch and read the vector -- not the reverse,
    # which would zero the vector, and not vector-times-vector in place.
    muls = _operands(trace, "V_MUL_VV")
    assert muls, "no elementwise square emitted"
    assert all(dst == scratch_base and src == vec_base for dst, _, src in muls)

    # The reduction must read the squared scratch, not the raw vector.
    reds = _operands(trace, "V_RED_SUM")
    assert reds, "no reduction emitted"
    assert all(src == scratch_base for *_, src in reds)

    # The final scaling must write the vector itself.
    scale_targets = {dst for dst, *_ in _operands(trace, "V_MUL_VF")}
    assert vec_base in scale_targets


def test_kda_constant_values_use_l2_not_rms_and_the_flashkda_epsilon():
    """reci_group must be 1.0: mamba_rsqrt_fpram computes
    1/sqrt(acc*reci_group + eps), and L2 is sqrt(sum), not sqrt(mean).
    eps must be 1e-6 to match the CPU reference's rsqrt(sum + 1e-6)."""
    values = PlenaCompiler.kda_fp_constant_values()
    zero, one, neg_one, dt_min, dt_max, reci_group, eps = values
    assert (zero, one, neg_one) == (0.0, 1.0, -1.0)
    assert reci_group == 1.0
    assert eps == pytest.approx(1.0e-6)
    assert torch.isfinite(torch.tensor([dt_min, dt_max])).all()


def test_kda_constants_allocate_seven_distinct_slots():
    prog = _prog()
    consts = prog.kda_fp_constants()
    addresses = [v.address for v in consts.as_list()]
    assert len(set(addresses)) == 7
    assert len(PlenaCompiler.kda_fp_constant_values()) == 7


# ---------------------------------------------------------------------------
# Conv state layout
# ---------------------------------------------------------------------------


def test_conv_state_rows_are_block_major_with_consecutive_taps():
    """Pin the formula, not merely that it is a bijection.

    The emitter, the test scatter and the test gather all funnel through this
    helper, so any bijection cancels out and the numerics stay green. But the
    layout claim -- one block's taps consecutive, so the history shift is a run
    of adjacent row copies -- rests on this exact ordering.
    """
    channels, kernel = MLEN * 3, 4
    blocks = kda_conv_blocks(channels, MLEN)
    assert blocks == 3
    rows = [
        kda_conv_state_row(channels, MLEN, kernel, cb, t)
        for cb in range(blocks)
        for t in range(kernel)
    ]
    assert rows == list(range(blocks * kernel))

    for cb in range(blocks):
        taps = [kda_conv_state_row(channels, MLEN, kernel, cb, t) for t in range(kernel)]
        assert taps == list(range(cb * kernel, (cb + 1) * kernel))


def test_conv_state_row_rejects_out_of_range():
    channels, kernel = MLEN * 2, 4
    with pytest.raises(ValueError, match="out of range"):
        kda_conv_state_row(channels, MLEN, kernel, 2, 0)
    with pytest.raises(ValueError, match="out of range"):
        kda_conv_state_row(channels, MLEN, kernel, 0, kernel)


def test_conv_blocks_rejects_a_partial_trailing_block():
    with pytest.raises(ValueError, match="multiple of mlen"):
        kda_conv_blocks(MLEN + 1, MLEN)
