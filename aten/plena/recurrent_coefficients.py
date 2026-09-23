"""Executable software candidates for recurrent coefficient production.

These programs use ordinary BF16 Vector instructions. Their approximation
error is a separate contract from the ideal GPU-prepared delta fixture.
"""

from dataclasses import dataclass

from compiler.aten.plena.prepared_vector_recurrence import _Emitter


class CompactCoefficientLoader:
    """Ordinary-ISA broadcast with finite, explicitly loaded Vector storage.

    mappings[(field,row)] lists (HBM row, source lane, destination first, count).
    The supplied static masks contain ones over each destination interval and
    zero elsewhere. Dynamic coefficients are cached once per recurrent group.
    Rows 0..15 belong to the recurrence, 16..57 cache data/masks, 58 holds a
    one-hot constant, 59..61 are scratch. No Matrix view or fused arithmetic.
    """

    def __init__(self, mappings, mask_addresses, one_hot):
        self.mappings = mappings
        self.masks = dict(mask_addresses)
        sources = sorted({a for entries in mappings.values() for a, _, _, _ in entries})
        intervals = sorted({(start, count) for entries in mappings.values() for _, _, start, count in entries})
        if set(intervals) != set(self.masks) or len(sources) + len(intervals) > 42:
            raise ValueError("compact coefficient cache exceeds 42 existing Vector rows or masks missing")
        self.source_rows = {a: 16 + i for i, a in enumerate(sources)}
        self.mask_rows = {key: 16 + len(sources) + i for i, key in enumerate(intervals)}
        self.one_hot = one_hot
        for entries in mappings.values():
            occupied = set()
            for a, lane, start, count in entries:
                if a % 64 or not 0 <= lane < 2048 or not 0 <= start < start + count <= 2048:
                    raise ValueError("invalid compact coefficient range")
                target = set(range(start, start + count))
                if occupied & target:
                    raise ValueError("coefficient destinations overlap")
                occupied |= target
            if len(occupied) != 2048:
                raise ValueError("expanded coefficient must cover the complete row")

    def prepare(self, out):
        for address, row in self.source_rows.items():
            out.transfer(row, address)
        for key, row in self.mask_rows.items():
            out.transfer(row, self.masks[key])
        out.transfer(58, self.one_hot)

    def load(self, out, name, index, target):
        if not 0 <= target < 16:
            raise ValueError("coefficient destination aliases cache")
        out.binary("ADD", target, 7, 7)
        for address, lane, start, count in self.mappings[name, index]:
            out.address(1, 59 * 2048)
            out.address(2, 58 * 2048)
            out.address(3, lane)
            out.lines.append("V_SHFT_V gp1, gp2, gp3")
            out.binary("MUL", 60, self.source_rows[address], 59)
            out.address(1, 60 * 2048)
            out.lines.extend(["S_SUB_FP f1, f0, f0", "V_RED_SUM f1, gp1, 0"])
            out.address(1, 61 * 2048)
            out.address(2, self.mask_rows[start, count] * 2048)
            out.lines.append("V_MUL_VF gp1, gp2, f1, 0")
            out.binary("ADD", target, target, 61)


def lower_softmax_rows(source, temporary, destination, values, tail_mask, *, minimum_slot=0):
    """Bounded three-pass softmax with existing Vector/scalar instructions.

    FP SRAM minimum_slot contains a finite BF16 lower bound on live logits.
    Input padding is at or below that bound. tail_mask contains ones on the
    live lanes of the last row and zeros elsewhere. No unbounded SRAM vector;
    intermediate exponentials spill to owned HBM rows. BF16 after each op.
    """
    if type(values) is not int or values < 1:
        raise ValueError("positive softmax extent")
    rows = (values + 2047) // 2048
    regions = [(a, rows * 4096) for a in (source, temporary, destination)] + [(tail_mask, 4096)]
    for i, (a, n) in enumerate(regions):
        if a % 64 or a < 0 or a + n > 2**32 or any(a < b + m and b < a + n for b, m in regions[:i]):
            raise ValueError("softmax buffers overlap or exceed ABI")
    e = _Emitter(2048, True)
    e.address(4, 0)
    e.lines.append(f"S_LD_FP f1, gp4, {minimum_slot}")
    for r in range(rows):
        e.transfer(0, source + r * 4096)
        e.address(1, 0)
        e.lines.append("V_RED_MAX f1, gp1, 0")
    e.lines.append("S_SUB_FP f2, f0, f0")
    e.transfer(2, tail_mask)
    for r in range(rows):
        e.transfer(0, source + r * 4096)
        e.address(1, 0)
        e.address(2, 0)
        e.lines.append("V_SUB_VF gp1, gp2, f1, 0, 0")
        _unary(e, "EXP", 0, 0)
        if r == rows - 1:
            e.binary("MUL", 0, 0, 2)
        e.address(1, 0)
        e.lines.append("V_RED_SUM f2, gp1, 0")
        e.transfer(0, temporary + r * 4096, store=True)
    e.lines.append("S_RECI_FP f2, f2, 0")
    for r in range(rows):
        e.transfer(0, temporary + r * 4096)
        e.address(1, 0)
        e.address(2, 0)
        e.lines.append("V_MUL_VF gp1, gp2, f2, 0")
        e.transfer(0, destination + r * 4096, store=True)
    return "\n".join(e.lines) + "\n"


def lower_positive_normalize(source, destination, values):
    """Normalize positive, zero-padded rows with BF16 tree and reciprocal.

    The caller guarantees a nonzero sum. Used after router selection; unlike
    softmax this does not exponentiate already-positive sigmoid scores.
    """
    if type(values) is not int or values < 1:
        raise ValueError("positive normalization extent required")
    rows = (values + 2047) // 2048
    size = rows * 4096
    if any(a < 0 or a % 64 or a + size > 2**32 for a in (source, destination)) or (
        source < destination + size and destination < source + size
    ):
        raise ValueError("normalization buffers overlap or exceed ABI")
    e = _Emitter(2048, True)
    e.lines.append("S_SUB_FP f1, f0, f0")
    for r in range(rows):
        e.transfer(0, source + r * 4096)
        e.address(1, 0)
        e.lines.append("V_RED_SUM f1, gp1, 0")
    e.lines.append("S_RECI_FP f1, f1, 0")
    for r in range(rows):
        e.transfer(0, source + r * 4096)
        e.address(1, 0)
        e.address(2, 0)
        e.lines.append("V_MUL_VF gp1, gp2, f1, 0")
        e.transfer(0, destination + r * 4096, store=True)
    return "\n".join(e.lines) + "\n"


def lower_dot_rows(source, weights, destination, one_hot, values):
    """Padded BF16 multiply/tree, scalar row sum, one live scalar output lane."""
    if values < 1:
        raise ValueError("empty dot")
    rows = (values + 2047) // 2048
    spans = [
        (source, rows * 4096),
        (weights, rows * 4096),
        (destination, 4096),
        (one_hot, 4096),
    ]
    for i, (a, n) in enumerate(spans):
        if a < 0 or a % 64 or a + n > 2**32 or any(a < b + m and b < a + n for b, m in spans[:i]):
            raise ValueError("dot regions overlap or exceed ABI")
    e = _Emitter(2048, True)
    e.lines.append("S_SUB_FP f1, f0, f0")
    for r in range(rows):
        e.transfer(0, source + r * 4096)
        e.transfer(1, weights + r * 4096)
        e.binary("MUL", 2, 0, 1)
        e.address(1, 4096)
        e.lines.append("V_RED_SUM f1, gp1, 0")
    e.transfer(0, one_hot)
    e.address(1, 0)
    e.address(2, 0)
    e.lines.append("V_MUL_VF gp1, gp2, f1, 0")
    e.transfer(0, destination, store=True)
    return "\n".join(e.lines) + "\n"


DELTA_CONSTANTS = (-1 / 16, 1 / 24, 1 / 6, 1 / 2, 1, 2)


def lower_delta_from_log(input_bases, output_bases, constants_base, *, mlen=2048):
    """Compute -expm1(log_decay) for log_decay in [-5,0], BF16 at each op.

    Range reduction t=-log_decay/16; fourth-order Taylor at t<=5/16;
    four doubling identities delta(2t)=delta(t)*(2-delta(t)). The input
    range is a caller obligation, not an unimplemented hardware comparison.
    Six constant rows 16..21 survive every input tile, scratch 22..24.
    Each input/output is a complete padded row; tails and padding cost bytes.
    No lookup, expm1 opcode, FP32 storage, or hidden grouped broadcast.
    """
    if len(input_bases) != len(output_bases) or not input_bases:
        raise ValueError("one output row per nonempty input row is required")
    out = _Emitter(mlen, True)
    out.lines.append("; @stage=recurrent_delta_producer")
    for i in range(len(DELTA_CONSTANTS)):
        out.transfer(16 + i, constants_base + i * mlen * 2)
    for source, destination in zip(input_bases, output_bases):
        out.transfer(22, source)
        out.binary("MUL", 22, 22, 16)
        out.binary("MUL", 23, 22, 17)
        out.binary("SUB", 23, 18, 23)
        out.binary("MUL", 23, 22, 23)
        out.binary("SUB", 23, 19, 23)
        out.binary("MUL", 23, 22, 23)
        out.binary("SUB", 23, 20, 23)
        out.binary("MUL", 23, 22, 23)
        for _ in range(4):
            out.binary("SUB", 24, 21, 23)
            out.binary("MUL", 23, 23, 24)
        out.transfer(23, destination, store=True)
    return "\n".join(out.lines) + "\n"


DELTA_RATIONAL_CONSTANTS = (-1 / 16, 1 / 6, 1 / 2, 1, 2)

# Rows 16..20: rational approximation; row 24: -1; row 25: lower bound.
# Each constant occupies a full Vector row. No implicit scalar broadcast.
GATE_CONSTANTS = (*DELTA_RATIONAL_CONSTANTS, -1, -5)


@dataclass(frozen=True)
class MambaGateRow:
    """A padded row of independent heads; addresses are HBM byte addresses.

    Inputs are post-projection dt, dt_bias, and the static negative A (not
    A_log). Folding exp(A_log) into model constants is outside token execution.
    The caller owns packing, conv, b/c and x production. This is deliberately
    a gate producer, not a complete Mamba layer.
    """

    raw_dt: int
    dt_bias: int
    negative_a: int
    dt: int
    delta: int


@dataclass(frozen=True)
class KdaGateRow:
    """Padded independent channels for the bounded KDA decay formula.

    Produces delta from lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
    and beta from sigmoid(raw_beta). A_log replication and all input packing
    must be accounted for by the caller; no hidden gather/broadcast occurs.
    Unbounded/softplus decay is a different formula and is not accepted here.
    """

    gate: int
    dt_bias: int
    a_log: int
    raw_beta: int
    delta: int
    beta: int


def _unary(out, operation, destination, source):
    out.address(1, destination * out.mlen)
    out.address(2, source * out.mlen)
    out.lines.append(f"V_{operation}_V gp1, gp2, 0")


def _delta_in_registers(out):
    """BF16 log in row 21 -> candidate delta in row 22, clobbers 21..23."""
    out.binary("MUL", 21, 21, 16)
    out.binary("MUL", 22, 21, 17)
    out.binary("ADD", 22, 18, 22)
    out.binary("MUL", 22, 21, 22)
    out.binary("ADD", 22, 19, 22)
    out.binary("MUL", 22, 21, 22)
    out.binary("ADD", 23, 19, 22)
    _unary(out, "RECI", 23, 23)
    out.binary("MUL", 22, 22, 23)
    for _ in range(4):
        out.binary("SUB", 23, 20, 22)
        out.binary("MUL", 22, 22, 23)


def _sigmoid(out, row):
    # exp(-softplus(-x)) avoids exp(+large), reciprocal overflow, and 1-p
    # cancellation. Rounding after each existing instruction is intentional.
    out.binary("MUL", row, row, 24)
    _unary(out, "SOFTPLUS", row, row)
    out.binary("MUL", row, row, 24)
    _unary(out, "EXP", row, row)


def _gate_emitter(rows, constants_base, mlen, vector_sram_rows, row_type, output_fields):
    if type(mlen) is not int or type(vector_sram_rows) is not int or mlen < 32 or mlen % 32 or vector_sram_rows < 26:
        raise ValueError("gate producer needs whole 32-element words and 26 Vector rows")
    if not rows or any(not isinstance(row, row_type) for row in rows):
        raise ValueError("nonempty, uniformly typed gate rows required")
    span = mlen * 2
    if type(constants_base) is not int or constants_base < 0 or constants_base % 64:
        raise ValueError("invalid constants HBM range")
    inputs = [(constants_base, constants_base + len(GATE_CONSTANTS) * span)]
    outputs = []
    for row in rows:
        for name, address in vars(row).items():
            if type(address) is not int or address < 0 or address % 64 or address + span > 2**32:
                raise ValueError("HBM rows require aligned, nonnegative 32-bit byte addresses")
            (outputs if name in output_fields else inputs).append((address, address + span))
    if not isinstance(constants_base, int) or constants_base < 0 or constants_base % 64 or inputs[0][1] > 2**32:
        raise ValueError("invalid constants HBM range")
    # Reject aliases across the ENTIRE program, including a future row's input.
    # In-place operation needs a separate lifetime analysis, not a local check.
    for index, (start, end) in enumerate(outputs):
        if any(start < b and a < end for a, b in inputs + outputs[:index]):
            raise ValueError("gate output aliases an input, constants, or another output")
    out = _Emitter(mlen, True)
    out.lines.append("; @stage=raw_gate_producer_bf16_rational_candidate")
    for index, row in enumerate((16, 17, 18, 19, 20, 24, 25)):
        out.transfer(row, constants_base + index * span)
    return out


def lower_mamba_gate_rows(rows, constants_base, *, mlen=2048, vector_sram_rows=64):
    """Real instructions from raw dt to dt/delta; BF16 at EVERY boundary.

    log=A*softplus(raw_dt+dt_bias), then the rational delta candidate. Caller
    must validate finite inputs, A<=0 and rounded log in [-32768,0]. The output
    is not numerically equivalent to FP32-prepared coefficients by definition.
    Workspace rows 0..2 and 16..25; no dynamic allocation or added ISA.
    """
    rows = tuple(rows)
    out = _gate_emitter(rows, constants_base, mlen, vector_sram_rows, MambaGateRow, {"dt", "delta"})
    for row in rows:
        out.transfer(0, row.raw_dt)
        out.transfer(1, row.dt_bias)
        out.binary("ADD", 0, 0, 1)
        _unary(out, "SOFTPLUS", 0, 0)
        out.transfer(0, row.dt, store=True)
        out.transfer(2, row.negative_a)
        out.binary("MUL", 21, 0, 2)
        _delta_in_registers(out)
        out.transfer(22, row.delta, store=True)
    return "\n".join(out.lines) + "\n"


def lower_kda_gate_rows(rows, constants_base, *, mlen=2048, vector_sram_rows=64):
    """Bounded KDA gate and beta; lower bound is constants row 25 (default -5).

    q/k normalization, convolution and coefficient packing are separate
    producers. Their outputs must not be silently substituted by this API.
    """
    rows = tuple(rows)
    out = _gate_emitter(rows, constants_base, mlen, vector_sram_rows, KdaGateRow, {"delta", "beta"})
    for row in rows:
        out.transfer(0, row.gate)
        out.transfer(1, row.dt_bias)
        out.binary("ADD", 0, 0, 1)
        out.transfer(2, row.a_log)
        _unary(out, "EXP", 2, 2)
        out.binary("MUL", 0, 0, 2)
        _sigmoid(out, 0)
        out.binary("MUL", 21, 0, 25)
        _delta_in_registers(out)
        out.transfer(22, row.delta, store=True)
        out.transfer(0, row.raw_beta)
        _sigmoid(out, 0)
        out.transfer(0, row.beta, store=True)
    return "\n".join(out.lines) + "\n"


def lower_delta_rational(input_bases, output_bases, constants_base, *, mlen=2048, addend_bases=None):
    """Candidate for BF16 log-decay in [-32768,0], caller-checked domain.

    For t=-log/16, approximate exp(-t) by 1/(1+t+t*t/2+t*t*t/6).
    Compute its complement as u/(1+u) to avoid near-one subtraction, then
    apply four doubling identities. This uses the existing reciprocal unit.
    It is not the accepted ideal-expm1 precision contract. Measure quality
    before selecting this producer. Constants own rows 16..20, scratch 21..23.

    Optional addends form prepacked (delta, b) rows: input logs occupy even
    positions with zero in odd positions; addends occupy odd positions with
    zero in even positions. The caller must establish that layout. The addend
    read and BF16 add are actual instructions, not a host-side coefficient
    insertion. Input row 21 is dead when reused for this addend. Packing log
    and b at the input boundary remains outside this producer.
    """
    if len(input_bases) != len(output_bases) or not input_bases:
        raise ValueError("one output row per nonempty input row is required")
    if addend_bases is not None and len(addend_bases) != len(input_bases):
        raise ValueError("one packed addend row per input row is required")
    out = _Emitter(mlen, True)
    out.lines.append("; @stage=recurrent_delta_rational_candidate")
    for i in range(len(DELTA_RATIONAL_CONSTANTS)):
        out.transfer(16 + i, constants_base + i * mlen * 2)
    for index, (source, destination) in enumerate(zip(input_bases, output_bases)):
        out.transfer(21, source)
        _delta_in_registers(out)
        if addend_bases is not None:
            out.transfer(21, addend_bases[index])
            out.binary("ADD", 22, 22, 21)
        out.transfer(22, destination, store=True)
    return "\n".join(out.lines) + "\n"


@dataclass(frozen=True)
class ConvStep:
    """Four-tap depthwise convolution, tap-major BF16 history/weights in HBM.

    Each channel block is a complete 2048-value row. History survives tokens;
    newest projected input is read from its producer's output allocation.
    """

    input: int
    history: int
    weights: int
    output: int
    channels: int
    bias: int | None = None


def lower_conv_steps(steps, constants_base, *, mlen=2048):
    """Four BF16 products, balanced BF16 adds, optional bias and SiLU.

    This explicit per-instruction contract differs from the native FP32
    convolution accumulation. No precomputed convolved inputs or free shifts.
    Workspace: Vector rows 0..5, 24. History shifts are actual HBM stores.
    """
    steps = tuple(steps)
    if not steps or mlen != 2048:
        raise ValueError("nonempty conv steps at VLEN=2048 required")
    regions = [(constants_base, constants_base + len(GATE_CONSTANTS) * mlen * 2)]
    for step in steps:
        if not isinstance(step, ConvStep) or type(step.channels) is not int or step.channels < 1:
            raise ValueError("positive channel count required")
        size = (step.channels + mlen - 1) // mlen * mlen * 2
        for name, address, span in (
            ("input", step.input, size),
            ("history", step.history, 4 * size),
            ("weights", step.weights, 4 * size),
            ("output", step.output, size),
        ):
            if type(address) is not int or address < 0 or address % 64 or address + span > 2**32:
                raise ValueError("invalid convolution HBM region")
            regions.append((address, address + span))
        if step.bias is not None:
            if type(step.bias) is not int or step.bias < 0 or step.bias % 64 or step.bias + size > 2**32:
                raise ValueError("invalid bias region")
            regions.append((step.bias, step.bias + size))
        local = [
            (step.input, step.input + size),
            (step.weights, step.weights + 4 * size),
            (constants_base, constants_base + len(GATE_CONSTANTS) * mlen * 2),
        ]
        if step.bias is not None:
            local.append((step.bias, step.bias + size))
        if any(a < step.history + 4 * size and step.history < b for a, b in local):
            raise ValueError("history aliases immutable operands")
        if any(
            a < step.output + size and step.output < b for a, b in local + [(step.history, step.history + 4 * size)]
        ):
            raise ValueError("output aliases live convolution operands")
    if type(constants_base) is not int or constants_base < 0 or constants_base % 64 or regions[0][1] > 2**32:
        raise ValueError("invalid constant region")
    out = _Emitter(mlen, True)
    out.lines.append("; @stage=causal_conv4_bf16_tree_silu")
    out.transfer(24, constants_base + 5 * mlen * 2)  # -1 for stable sigmoid
    for step in steps:
        blocks = (step.channels + mlen - 1) // mlen
        span = blocks * mlen * 2
        for block in range(blocks):
            offset = block * mlen * 2
            for tap in range(4):
                source = step.input + offset if tap == 3 else step.history + (tap + 1) * span + offset
                out.transfer(0, source)
                out.transfer(0, step.history + tap * span + offset, store=True)
                out.transfer(1, step.weights + tap * span + offset)
                out.binary("MUL", 2 + tap, 0, 1)
            out.binary("ADD", 2, 2, 3)
            out.binary("ADD", 4, 4, 5)
            out.binary("ADD", 2, 2, 4)
            if step.bias is not None:
                out.transfer(1, step.bias + offset)
                out.binary("ADD", 2, 2, 1)
            # Preserve the preactivation in row 2 while sigmoid runs in row 3.
            out.binary("MUL", 3, 2, 24)
            _unary(out, "SOFTPLUS", 3, 3)
            out.binary("MUL", 3, 3, 24)
            _unary(out, "EXP", 3, 3)
            out.binary("MUL", 2, 2, 3)
            out.transfer(2, step.output + offset, store=True)
    return "\n".join(out.lines) + "\n"


@dataclass(frozen=True)
class L2NormRows:
    input: int
    output: int
    channels: int
    masks: int
    zero: int
    # FP SRAM slots: epsilon, output scale (1 for k, 1/sqrt(width) for q).
    epsilon_slot: int = 0
    scale_slot: int = 1
    width: int = 128


def lower_l2norm_rows(p: L2NormRows, *, mlen=2048):
    """Head-local BF16 square/tree/scalar sqrt/reciprocal and scale.

    All masks are static constants, not prepared norms. Each logical head is
    reduced separately using existing Vector and scalar instructions. The
    conservative schedule recomputes no values on the host and needs six rows.
    """
    if mlen != 2048 or p.width not in (32, 64, 128, 256, 512, 1024, 2048) or p.channels < 1 or p.channels % p.width:
        raise ValueError("normalization needs complete power-of-two groups")
    span = (p.channels + mlen - 1) // mlen * mlen * 2
    regions = [
        (p.input, span),
        (p.output, span),
        (p.masks, (mlen // p.width) * mlen * 2),
        (p.zero, mlen * 2),
    ]
    for i, (a, n) in enumerate(regions):
        if type(a) is not int or a < 0 or a % 64 or a + n > 2**32:
            raise ValueError("invalid norm allocation")
        if any(a < b + m and b < a + n for b, m in regions[:i]):
            raise ValueError("norm allocations overlap")
    if not 0 <= p.epsilon_slot < 512 or not 0 <= p.scale_slot < 512:
        raise ValueError("norm constants exceed existing FP SRAM")
    e = _Emitter(mlen, True)
    e.lines.append("; @stage=qk_l2norm_bf16_tree_candidate")
    e.address(4, 0)
    e.lines.extend([f"S_LD_FP f2, gp4, {p.epsilon_slot}", f"S_LD_FP f3, gp4, {p.scale_slot}"])
    for start in range(0, p.channels, mlen):
        e.transfer(0, p.input + start * 2)
        e.transfer(5, p.zero)
        for head in range(min(mlen // p.width, (p.channels - start) // p.width)):
            e.transfer(1, p.masks + head * mlen * 2)
            e.binary("MUL", 2, 0, 1)
            e.binary("MUL", 3, 2, 2)
            e.address(1, 3 * mlen)
            e.lines.extend(
                [
                    "S_SUB_FP f1, f0, f0",
                    "V_RED_SUM f1, gp1, 0",
                    "S_ADD_FP f1, f1, f2",
                    "S_SQRT_FP f1, f1, 0",
                    "S_RECI_FP f1, f1, 0",
                    "S_MUL_FP f1, f1, f3",
                ]
            )
            e.address(1, 4 * mlen)
            e.address(2, 2 * mlen)
            e.lines.append("V_MUL_VF gp1, gp2, f1, 0")
            e.binary("ADD", 5, 5, 4)
        e.transfer(5, p.output + start * 2, store=True)
    return "\n".join(e.lines) + "\n"


def lower_bf16_gather(sources, destination, zero, one_hot, *, mlen=2048, strategy="reference"):
    """Correctness-first software packing using existing scalar/Vector ISA.

    sources[i] is None (zero) or (aligned HBM row base, lane). A one-hot mask,
    BF16 tree reduction, and scalar multiply copy each live element exactly.
    It uses no FP SRAM staging, arbitrary SRAM gather port, or host-produced
    intermediate. Expensive; not a claim of the optimal software baseline.
    Full destination rows are owned by caller. Reference/grouped workspace is
    rows 0..6; pattern additionally owns rows 7..63 for reusable static masks.
    Cached also reuses source rows and identical contributions across output
    rows within the same 64-row budget. All construction uses ordinary ISA.
    """
    sources = tuple(sources)
    if strategy not in ("reference", "grouped", "pattern", "cached"):
        raise ValueError("unknown software gather strategy")
    if mlen != 2048 or not sources:
        raise ValueError("nonempty VLEN=2048 gather required")
    span = (len(sources) + mlen - 1) // mlen * mlen * 2
    for a, n in ((destination, span), (zero, mlen * 2), (one_hot, mlen * 2)):
        if type(a) is not int or a < 0 or a % 64 or a + n > 2**32:
            raise ValueError("invalid gather allocation")
    reads = [(zero, zero + mlen * 2), (one_hot, one_hot + mlen * 2)]
    for src in sources:
        if src is None:
            continue
        if len(src) != 2 or any(type(v) is not int for v in src):
            raise ValueError("source must be an HBM row/lane pair")
        a, lane = src
        if a < 0 or a % 64 or a + mlen * 2 > 2**32 or not 0 <= lane < mlen:
            raise ValueError("gather source out of range")
        reads.append((a, a + mlen * 2))
    if any(a < destination + span and destination < b for a, b in reads):
        raise ValueError("gather destination aliases a source")
    if strategy == "cached":
        return _lower_cached_gather(sources, destination, zero, one_hot, mlen)
    e = _Emitter(mlen, True)
    e.lines.append("; @stage=software_coefficient_gather_reference")
    e.transfer(6, one_hot)
    loaded = None

    def shift(dst, index, source=6):
        e.address(1, dst * mlen)
        e.address(2, source * mlen)
        e.address(3, index)
        e.lines.append("V_SHFT_V gp1, gp2, gp3")

    patterns = {}
    if strategy == "pattern":
        for first in range(0, len(sources), mlen):
            groups = {}
            for i, src in enumerate(sources[first : first + mlen]):
                if src is not None:
                    groups.setdefault(src, []).append(i)
            for indices in groups.values():
                pattern = tuple(i - indices[0] for i in indices)
                if len(pattern) > 1:
                    patterns.setdefault(pattern, 7 + len(patterns))
        if len(patterns) > 57:
            strategy = "grouped"
            patterns = {}
        for pattern, row in patterns.items():
            e.transfer(row, zero)
            for index in pattern:
                shift(3, index)
                e.binary("ADD", row, row, 3)

    for first in range(0, len(sources), mlen):
        e.transfer(5, zero)
        if strategy == "pattern":
            groups = {}
            for i, src in enumerate(sources[first : first + mlen]):
                if src is not None:
                    groups.setdefault(src, []).append(i)
            for (address, lane), indices in sorted(groups.items()):
                if loaded != address:
                    e.transfer(0, address)
                    loaded = address
                shift(1, lane)
                e.binary("MUL", 2, 0, 1)
                e.address(1, 2 * mlen)
                e.lines.extend(["S_SUB_FP f1, f0, f0", "V_RED_SUM f1, gp1, 0"])
                pattern = tuple(i - indices[0] for i in indices)
                shift(3, indices[0], patterns.get(pattern, 6))
                e.address(1, 4 * mlen)
                e.address(2, 3 * mlen)
                e.lines.append("V_MUL_VF gp1, gp2, f1, 0")
                e.binary("ADD", 5, 5, 4)
            e.transfer(5, destination + first * 2, store=True)
            continue
        entries = [(i, s) for i, s in enumerate(sources[first : first + mlen]) if s is not None]
        if strategy == "grouped":
            entries.sort(key=lambda pair: pair[1])
        previous_source = None
        for index, src in entries:
            address, lane = src
            if loaded != address:
                e.transfer(0, address)
                loaded = address
            if strategy == "reference" or src != previous_source:
                shift(1, lane)
                e.binary("MUL", 2, 0, 1)
                e.address(1, 2 * mlen)
                e.lines.extend(["S_SUB_FP f1, f0, f0", "V_RED_SUM f1, gp1, 0"])
            previous_source = src
            shift(3, index)
            e.address(1, 4 * mlen)
            e.address(2, 3 * mlen)
            e.lines.append("V_MUL_VF gp1, gp2, f1, 0")
            e.binary("ADD", 5, 5, 4)
        e.transfer(5, destination + first * 2, store=True)
    return "\n".join(e.lines) + "\n"


def _lower_cached_gather(sources, destination, zero, one_hot, mlen):
    """Reuse identical (source, destination-mask) contributions, not values.

    Compiler analysis only inspects addresses. In particular a repeated decay
    layout is constructed once on the accelerator and retained in an existing
    Vector row; coefficients are never folded from a host numerical capture.
    A contribution may be shared only by the output rows that actually contain
    it. Unique pieces still use the reference scalar extraction arithmetic.
    """
    from collections import defaultdict

    rows, uses, patterns = [], defaultdict(list), {}
    for first in range(0, len(sources), mlen):
        groups = {}
        for index, source in enumerate(sources[first : first + mlen]):
            if source is not None:
                groups.setdefault(source, []).append(index)
        keys = [(source, tuple(indices)) for source, indices in sorted(groups.items())]
        for source, indices in keys:
            uses[source, indices].append(len(rows))
            pattern = tuple(i - indices[0] for i in indices)
            if len(pattern) > 1:
                patterns.setdefault(pattern, 7 + len(patterns))
        rows.append(keys)
    if len(patterns) > 57:
        return lower_bf16_gather(sources, destination, zero, one_hot, strategy="grouped")

    # Group common contributions by identical live output-row sets. This uses
    # one SRAM row for an entire shared part (e.g. all heads' decay fields).
    common = defaultdict(list)
    for key, output_rows in uses.items():
        if len(output_rows) > 1:
            common[tuple(output_rows)].append(key)
    next_row = 7 + len(patterns)
    ranked = sorted(common.items(), key=lambda item: (-(len(item[0]) - 1) * len(item[1]), item[0]))
    retained = {}
    for output_rows, keys in ranked:
        if next_row == 64:
            break
        retained[output_rows] = (next_row, keys)
        next_row += 1
    addresses = sorted({source[0] for keys in rows for source, _ in keys})
    source_cache = {a: next_row + i for i, a in enumerate(addresses[: 64 - next_row])}
    cached_keys = {key for _, keys in retained.values() for key in keys}
    e = _Emitter(mlen, True)
    e.lines.append("; @stage=software_coefficient_gather_cached workspace=64")
    e.transfer(6, one_hot)

    def shift(dst, index, source=6):
        e.address(1, dst * mlen)
        e.address(2, source * mlen)
        e.address(3, index)
        e.lines.append("V_SHFT_V gp1, gp2, gp3")

    for pattern, row in patterns.items():
        e.transfer(row, zero)
        for index in pattern:
            shift(3, index)
            e.binary("ADD", row, row, 3)
    for address, row in source_cache.items():
        e.transfer(row, address)
    loaded = None

    def contribute(destination_row, key):
        nonlocal loaded
        (address, lane), indices = key
        source_row = source_cache.get(address, 0)
        if source_row == 0 and loaded != address:
            e.transfer(0, address)
            loaded = address
        shift(1, lane)
        e.binary("MUL", 2, source_row, 1)
        e.address(1, 2 * mlen)
        e.lines.extend(["S_SUB_FP f1, f0, f0", "V_RED_SUM f1, gp1, 0"])
        pattern = tuple(i - indices[0] for i in indices)
        shift(3, indices[0], patterns.get(pattern, 6))
        e.address(1, 4 * mlen)
        e.address(2, 3 * mlen)
        e.lines.append("V_MUL_VF gp1, gp2, f1, 0")
        e.binary("ADD", destination_row, destination_row, 4)

    for row, keys in retained.values():
        e.transfer(row, zero)
        for key in keys:
            contribute(row, key)
    for index, keys in enumerate(rows):
        e.transfer(5, zero)
        for output_rows, (row, _) in retained.items():
            if index in output_rows:
                e.binary("ADD", 5, 5, row)
        for key in keys:
            if key not in cached_keys:
                contribute(5, key)
        e.transfer(5, destination + index * mlen * 2, store=True)
    return "\n".join(e.lines) + "\n"


def lower_global_rms(
    input_base,
    weight_base,
    output_base,
    channels,
    *,
    epsilon_slot=0,
    scale_slot=1,
    mlen=2048,
):
    """Whole hidden vector RMS: BF16 row trees and scalar inter-row sum.

    FP slots hold channels*epsilon and sqrt(channels). Input and weight tail
    lanes MUST be zero, and allocations include complete Vector rows. No
    source/output overlap is allowed. Existing scalar precision stays BF16.
    """
    if type(channels) is not int or channels < 1 or mlen != 2048:
        raise ValueError("invalid RMS shape")
    span = (channels + mlen - 1) // mlen * mlen * 2
    regions = [(input_base, span), (weight_base, span), (output_base, span)]
    for i, (a, n) in enumerate(regions):
        if type(a) is not int or a < 0 or a % 64 or a + n > 2**32:
            raise ValueError("invalid RMS allocation")
        if any(a < b + m and b < a + n for b, m in regions[:i]):
            raise ValueError("RMS allocations overlap")
    if not 0 <= epsilon_slot < 512 or not 0 <= scale_slot < 512:
        raise ValueError("invalid FP constant slot")
    e = _Emitter(mlen, True)
    e.lines.append("; @stage=global_rms_bf16_tree_candidate")
    e.address(4, 0)
    e.lines.extend(
        [
            f"S_LD_FP f2, gp4, {epsilon_slot}",
            f"S_LD_FP f3, gp4, {scale_slot}",
            "S_SUB_FP f1, f0, f0",
        ]
    )
    for first in range(0, channels, mlen):
        e.transfer(0, input_base + first * 2)
        e.binary("MUL", 1, 0, 0)
        e.address(1, mlen)
        e.lines.append("V_RED_SUM f1, gp1, 0")
    e.lines.extend(
        [
            "S_ADD_FP f1, f1, f2",
            "S_SQRT_FP f1, f1, 0",
            "S_RECI_FP f1, f1, 0",
            "S_MUL_FP f1, f1, f3",
        ]
    )
    for first in range(0, channels, mlen):
        e.transfer(0, input_base + first * 2)
        e.address(1, mlen)
        e.address(2, 0)
        e.lines.append("V_MUL_VF gp1, gp2, f1, 0")
        e.transfer(2, weight_base + first * 2)
        e.binary("MUL", 1, 1, 2)
        e.transfer(1, output_base + first * 2, store=True)
    return "\n".join(e.lines) + "\n"


def lower_pointwise_rows(
    input_base,
    other_base,
    output_base,
    rows,
    *,
    sigmoid_input=False,
    constants_base=None,
    mlen=2048,
):
    """Elementwise product, optionally sigmoid(input)*other, using real rows."""
    if type(rows) is not int or rows < 1 or mlen != 2048:
        raise ValueError("invalid row count")
    span = rows * mlen * 2
    regions = [(input_base, span), (other_base, span), (output_base, span)]
    if sigmoid_input:
        if constants_base is None:
            raise ValueError("sigmoid requires static constants")
        regions.append((constants_base, len(GATE_CONSTANTS) * mlen * 2))
    if any(type(a) is not int or a < 0 or a % 64 or a + n > 2**32 for a, n in regions):
        raise ValueError("invalid pointwise allocation")
    # Exact in-place rows are safe: both operands are read before the store.
    # Partial overlap can overwrite a future input row and must be rejected.
    for source in (input_base, other_base):
        if source != output_base and source < output_base + span and output_base < source + span:
            raise ValueError("pointwise output partially overlaps a live input")
    if (
        sigmoid_input
        and constants_base < output_base + span
        and output_base < constants_base + len(GATE_CONSTANTS) * mlen * 2
    ):
        raise ValueError("pointwise output overlaps static constants")
    e = _Emitter(mlen, True)
    e.lines.append("; @stage=pointwise_gate_product")
    if sigmoid_input:
        e.transfer(24, constants_base + 5 * mlen * 2)
    for r in range(rows):
        e.transfer(0, input_base + r * mlen * 2)
        if sigmoid_input:
            _sigmoid(e, 0)
        e.transfer(1, other_base + r * mlen * 2)
        e.binary("MUL", 0, 0, 1)
        e.transfer(0, output_base + r * mlen * 2, store=True)
    return "\n".join(e.lines) + "\n"
