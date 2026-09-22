"""Executable software candidates for recurrent coefficient production.

These programs use ordinary BF16 Vector instructions. Their approximation
error is a separate contract from the ideal GPU-prepared delta fixture.
"""

from dataclasses import dataclass

from compiler.aten.plena.prepared_vector_recurrence import _Emitter

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
