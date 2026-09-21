"""Executable software candidates for recurrent coefficient production.

These programs use ordinary BF16 Vector instructions. Their approximation
error is a separate contract from the ideal GPU-prepared delta fixture.
"""

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


def lower_delta_rational(input_bases, output_bases, constants_base, *, mlen=2048):
    """Candidate for BF16 log-decay in [-32768,0], caller-checked domain.

    For t=-log/16, approximate exp(-t) by 1/(1+t+t*t/2+t*t*t/6).
    Compute its complement as u/(1+u) to avoid near-one subtraction, then
    apply four doubling identities. This uses the existing reciprocal unit.
    It is not the accepted ideal-expm1 precision contract. Measure quality
    before selecting this producer. Constants own rows 16..20, scratch 21..23.
    """
    if len(input_bases) != len(output_bases) or not input_bases:
        raise ValueError("one output row per nonempty input row is required")
    out = _Emitter(mlen, True)
    out.lines.append("; @stage=recurrent_delta_rational_candidate")
    for i in range(len(DELTA_RATIONAL_CONSTANTS)):
        out.transfer(16 + i, constants_base + i * mlen * 2)
    for source, destination in zip(input_bases, output_bases):
        out.transfer(21, source)
        out.binary("MUL", 21, 21, 16)
        out.binary("MUL", 22, 21, 17)
        out.binary("ADD", 22, 18, 22)
        out.binary("MUL", 22, 21, 22)
        out.binary("ADD", 22, 19, 22)
        out.binary("MUL", 22, 21, 22)
        out.binary("ADD", 23, 19, 22)
        out.address(1, 23 * mlen)
        out.address(2, 23 * mlen)
        out.lines.append("V_RECI_V gp1, gp2, 0")
        out.binary("MUL", 22, 22, 23)
        for _ in range(4):
            out.binary("SUB", 23, 20, 22)
            out.binary("MUL", 22, 22, 23)
        out.transfer(22, destination, store=True)
    return "\n".join(out.lines) + "\n"
