"""Controlled ordinary-Vector recurrence baselines at the Matrix test geometry.

These are new executable controls, not the historical analytic A/B instruction
census. A emits explicit addresses; B applies static register-value reuse.
Both use existing VV operations, BF16 SRAM, and identical prepared operands.
Coefficients are explicitly expanded across lanes in HBM (no hidden gather).
"""

from dataclasses import dataclass
from collections.abc import Mapping

from compiler.asm_templates._imm import load_large_int
from compiler.aten.plena.matrix_recurrence_lowering import (
    MatrixRecurrenceSpec,
    RecurrenceKind,
)


@dataclass(frozen=True)
class PreparedVectorGroup:
    state_base: int
    fields: Mapping[str, int]


class _Emitter:
    def __init__(self, mlen: int, optimize: bool):
        self.mlen = mlen
        self.optimize = optimize
        self.lines: list[str] = []
        self.known: dict[int, int] = {}

    def address(self, register: int, value: int):
        if value < 0 or value >= 2**32:
            raise ValueError("address exceeds unsigned 32-bit GP ABI")
        if self.optimize and self.known.get(register) == value:
            return
        self.lines.extend(load_large_int(register, value))
        self.known[register] = value

    def transfer(self, row: int, address: int, *, store: bool = False):
        if address % 64:
            raise ValueError("HBM row must be 64-byte aligned")
        self.address(1, row * self.mlen)
        self.address(2, address)
        self.lines.append(f"H_{'STORE' if store else 'PREFETCH'}_V gp1, gp2, a0, 0, 2")

    def binary(self, op: str, destination: int, source1: int, source2: int):
        for register, row in ((1, destination), (2, source1), (3, source2)):
            self.address(register, row * self.mlen)
        self.lines.append(f"V_{op}_VV gp1, gp2, gp3, 0")


def lower_prepared_vector_recurrence(
    spec: MatrixRecurrenceSpec,
    groups: tuple[PreparedVectorGroup, ...],
    *,
    mlen: int = 2048,
    static_address_reuse: bool = False,
) -> str:
    """Emit BF16 row operations; peak live footprint is eight VLEN rows.

    State HBM order is [group][recurrence row][head][lane]. The caller must
    reserve private persistent state and token fields for every request.
    All arithmetic boundaries are explicitly BF16, including each reduction
    addition. This differs from L_TILE's local FP32 reduction and is audited
    numerically by the executable comparison, never silently equated to it.
    """
    if mlen % spec.row_elements or spec.heads % (mlen // spec.row_elements):
        raise ValueError("controlled Vector baseline requires full packed head groups")
    if len(groups) != spec.heads // (mlen // spec.row_elements):
        raise ValueError("prepared Vector group count differs from model shape")
    out = _Emitter(mlen, static_address_reuse)
    out.lines.append("; @stage=prepared_vector_recurrence")
    row_bytes = mlen * 2
    for group in groups:
        f = group.fields
        # VRAM: state=0, coefficients=1, x/value=2, scratch/error=3,
        # output=4, temporary=5, prediction=6, zero=7.
        out.transfer(7, f["zero"])
        out.transfer(2, f["x" if spec.kind is RecurrenceKind.MAMBA else "value"])
        if spec.kind is RecurrenceKind.MAMBA:
            out.transfer(1, f["dt"])
            out.binary("MUL", 3, 2, 1)
            out.binary("ADD", 4, 7, 7)
            for row in range(spec.recurrence_rows):
                offset = row * row_bytes
                out.transfer(0, group.state_base + offset)
                out.transfer(1, f["a"] + offset)
                out.binary("MUL", 0, 0, 1)
                out.transfer(1, f["b"] + offset)
                out.binary("MUL", 5, 3, 1)
                out.binary("ADD", 0, 0, 5)
                out.transfer(0, group.state_base + offset, store=True)
                out.transfer(1, f["c"] + offset)
                out.binary("MUL", 5, 0, 1)
                out.binary("ADD", 4, 4, 5)
            out.transfer(1, f["d"])
            out.binary("MUL", 5, 2, 1)
            out.binary("ADD", 4, 4, 5)
        else:
            out.binary("ADD", 6, 7, 7)
            for row in range(spec.recurrence_rows):
                offset = row * row_bytes
                out.transfer(0, group.state_base + offset)
                out.transfer(1, f["decay"] + offset)
                out.binary("MUL", 0, 0, 1)
                out.transfer(0, group.state_base + offset, store=True)
                out.transfer(1, f["key"] + offset)
                out.binary("MUL", 5, 0, 1)
                out.binary("ADD", 6, 6, 5)
            out.binary("SUB", 3, 2, 6)
            out.transfer(1, f["beta"])
            out.binary("MUL", 3, 3, 1)
            out.binary("ADD", 4, 7, 7)
            for row in range(spec.recurrence_rows):
                offset = row * row_bytes
                out.transfer(0, group.state_base + offset)
                out.transfer(1, f["key"] + offset)
                out.binary("MUL", 5, 3, 1)
                out.binary("ADD", 0, 0, 5)
                out.transfer(0, group.state_base + offset, store=True)
                out.transfer(1, f["query"] + offset)
                out.binary("MUL", 5, 0, 1)
                out.binary("ADD", 4, 4, 5)
        out.transfer(4, f["output"], store=True)
    return "\n".join(out.lines) + "\n"
