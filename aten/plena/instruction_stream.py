"""Counting an emitted program two ways: as an image, and as an issue stream.

These are different measurements and the difference is not small. A sweep that
the `V_FMA_VF` conversion collapsed into a hardware loop occupies a handful of
words in the program image and issues its body once per trip -- 192 times for a
Kimi K3 projection section, 128 times for a `key_dim` 128 recurrence.

Which one to use follows from the question:

* **Image** (:func:`static_count`) answers "how large is the binary" and
  nothing else. It is the right instrument for the budget gates in
  `test_instruction_budget.py`, whose whole purpose is to catch a sweep falling
  off the hardware-loop path and unrolling -- an image failure that no numeric
  test can see.
* **Issue stream** (:func:`dynamic_count`) answers "how much work is there".
  Every claim about cost, about which kernel dominates, or about what an
  alternative lowering would save has to be made against this one.

Conflating them is not a hypothetical. Three claims on this branch were made
from image counts and were wrong by 13x, 7x and 27x respectively: the
projection gather's share of a layer, the conv-versus-mixer ordering, and the
`V_FMA_VF` conversion's effect on time.
"""

from __future__ import annotations

import re

__all__ = ["static_count", "dynamic_count"]


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

    Convention: ``C_LOOP_START`` issues once, the body and the ``C_LOOP_END``
    branch issue once per trip. Nesting is handled. A different convention for
    the loop-end branch moves any numerator and denominator together, so the
    ratios this is used for do not depend on the choice.

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
