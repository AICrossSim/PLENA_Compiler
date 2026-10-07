"""Count assembly image words and dynamically issued instructions.

Static counts describe binary size; dynamic counts expand hardware loops.
Neither count is a cycle estimate: execution cost also depends on the unit,
memory traffic, arithmetic service and dependencies.
"""

from __future__ import annotations

import re

__all__ = [
    "static_count",
    "dynamic_count",
    "opcode_census",
    "arithmetic_share",
    "self_advance_counts",
    "ARITHMETIC",
]


def _instructions(asm: str) -> list[str]:
    return [
        line.strip()
        for line in asm.splitlines()
        if line.strip() and not line.strip().startswith(";")
    ]


def static_count(asm: str) -> int:
    """Instructions in the program image: one per emitted line, comments aside."""
    return len(_instructions(asm))


def dynamic_count(asm: str) -> int:
    """Instructions issued, with every ``C_LOOP_START`` expanded by its trip count.

    Convention: ``C_LOOP_START`` issues once, the body issues once per trip,
    and ``C_LOOP_END`` is **not counted at all** -- it is treated as a
    zero-overhead loop boundary that the sequencer resolves without an issue
    slot, which is what a hardware loop construct with a trip-count register
    normally is. Nesting is handled.

    Counting the loop end instead would add about 20% to the recurrent kernels
    here. It moves any numerator and denominator together, so the ratios this
    is used for do not depend on the choice -- but the absolute figures do, and
    they are reported under this convention. (An earlier version of this
    docstring claimed the loop end *was* counted, which the code never did.)

    Trip counts are the immediate in the ``C_LOOP_START`` word, which is what
    the emitters put there; a loop whose count came from a register at runtime
    would not be countable this way, and the static path does not have one.
    """
    lines = _instructions(asm)

    def walk(i: int) -> tuple[int, int]:
        total = 0
        while i < len(lines):
            op = lines[i].split()[0].rstrip(",")
            if op == "C_LOOP_START":
                trips = int(re.findall(r"(-?\d+)", lines[i])[-1])
                body, i = walk(i + 1)
                total += 1 + body * trips
                continue
            if op == "C_LOOP_END":
                return total, i + 1
            total += 1
            i += 1
        return total, i

    return walk(0)[0]


#: Opcodes that compute. Everything else is scaffolding: pointer arithmetic,
#: loop control, and moving scalars between FPRAM and the register file.
ARITHMETIC = frozenset({
    "V_FMA_VF", "V_MUL_VV", "V_ADD_VV", "V_SUB_VV", "V_MUL_VF", "V_ADD_VF",
    "V_MUL_VV.MV", "V_ADD_VV.MV", "V_SUB_VV.MV",
    "V_SUB_VF", "V_MAX_VF", "V_EXP_V", "V_RECI_V", "V_SOFTPLUS_V", "V_RED_SUM",
    "V_RED_MAX", "V_MOV_VF", "V_SHIFT_V", "V_CLR_V",
    "M_MM", "M_TMM", "M_MM_WO", "M_BTMM",
    "S_ADD_FP", "S_MUL_FP", "S_RECI_FP", "S_SQRT_FP", "S_EXP_FP",
})

_SELF_ADVANCE = re.compile(r"^S_ADDI_INT\s+gp(\d+)\s*,\s*gp(\d+)\s*,\s*(-?\d+)")
_LOOP_START = re.compile(r"^C_LOOP_START\s+gp(\d+)\s*,\s*(\d+)")
_GP = re.compile(r"\bgp(\d+)\b")


def opcode_census(asm: str) -> dict[str, int]:
    """Issued instructions per opcode, hardware loops expanded.

    Same expansion as :func:`dynamic_count`, but keyed by mnemonic, which is
    what turns "the kernel issues N instructions" into "and here is what they
    are". The recurrent kernels spend three quarters of their issue slots on
    scaffolding, and that is only visible per opcode.
    """
    lines = _instructions(asm)
    counts: dict[str, int] = {}

    def walk(i: int) -> tuple[dict[str, int], int]:
        local: dict[str, int] = {}
        while i < len(lines):
            op = lines[i].split()[0].rstrip(",")
            if op == "C_LOOP_START":
                trips = int(re.findall(r"(-?\d+)", lines[i])[-1])
                body, i = walk(i + 1)
                local[op] = local.get(op, 0) + 1
                for k, v in body.items():
                    local[k] = local.get(k, 0) + v * trips
                continue
            if op == "C_LOOP_END":
                return local, i + 1
            local[op] = local.get(op, 0) + 1
            i += 1
        return local, i

    counts, _ = walk(0)
    return counts


def arithmetic_share(asm: str) -> float:
    """Fraction of issued instructions that compute something."""
    census = opcode_census(asm)
    total = sum(census.values())
    if not total:
        return 0.0
    return sum(n for op, n in census.items() if op in ARITHMETIC) / total


def _loop_bodies(lines: list[str]) -> list[tuple[list[str], int]]:
    """Every loop body paired with how many times it runs, nesting included."""
    out: list[tuple[list[str], int]] = []

    def scan(block: list[str], trips: int) -> None:
        i = 0
        while i < len(block):
            m = _LOOP_START.match(block[i])
            if not m:
                i += 1
                continue
            reg, count = m.group(1), int(m.group(2))
            j, depth = i + 1, 1
            while j < len(block) and depth:
                if _LOOP_START.match(block[j]):
                    depth += 1
                elif block[j].startswith(f"C_LOOP_END gp{reg}") and depth == 1:
                    break
                elif block[j].startswith("C_LOOP_END"):
                    depth -= 1
                j += 1
            body = block[i + 1 : j]
            out.append((body, trips * count))
            scan(body, trips * count)
            i = j + 1

    scan(lines, 1)
    return out


def self_advance_counts(asm: str) -> tuple[int, int]:
    """``(foldable, unfoldable)`` issued pointer self-advances inside loops.

    A sweep advances its pointers with ``S_ADDI_INT gpN, gpN, step`` once per
    trip. A post-increment addressing mode -- the operand carrying its own
    stride, the way ``_emit_tile_row_fma`` already carries independent row
    progressions -- would fold those into the instruction that consumes the
    pointer, and they would stop occupying an issue slot.

    "Would" only holds where the register actually has a consumer in the same
    body; an advance with nothing to fold into has to stay. This reports the
    split so the saving is priced against instructions that can really
    disappear rather than against every advance that happens to sit in a loop.

    Loop *setup* (``S_ADDI_INT gpN, gp0, addr``) is excluded: it runs once and
    survives any addressing mode.
    """
    lines = _instructions(asm)
    foldable = unfoldable = 0
    for body, trips in _loop_bodies(lines):
        consumers: dict[str, int] = {}
        for line in body:
            if _SELF_ADVANCE.match(line) or line.startswith("C_LOOP"):
                continue
            for reg in _GP.findall(line):
                consumers[reg] = consumers.get(reg, 0) + 1
        for line in body:
            m = _SELF_ADVANCE.match(line)
            if not m or m.group(1) != m.group(2):
                continue
            if consumers.get(m.group(1), 0):
                foldable += trips
            else:
                unfoldable += trips
    return foldable, unfoldable
