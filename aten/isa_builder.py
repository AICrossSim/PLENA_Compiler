"""Typed ISA builder for the ATen PLENA compiler path.

This is intentionally small: it models the physical instruction stream and
prints the same assembly syntax the existing compiler emits.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol


class Renderable(Protocol):
    def render(self) -> str:
        """Render this object as assembly text."""


@dataclass(frozen=True)
class Register:
    prefix: str
    index: int

    def render(self) -> str:
        return f"{self.prefix}{self.index}"


def gp(index: int) -> Register:
    return Register("gp", index)


def fp(index: int) -> Register:
    return Register("f", index)


def addr(index: int) -> Register:
    return Register("a", index)


AsmArg = str | int | Register


def render_arg(arg: AsmArg) -> str:
    if isinstance(arg, Register):
        return arg.render()
    return str(arg)


@dataclass(frozen=True)
class Instr:
    opcode: str
    args: tuple[AsmArg, ...] = ()

    def render(self) -> str:
        if not self.args:
            return self.opcode
        return f"{self.opcode} {', '.join(render_arg(arg) for arg in self.args)}"


@dataclass(frozen=True)
class Comment:
    text: str

    def render(self) -> str:
        text = self.text.rstrip()
        if text.startswith(";"):
            return text
        return f"; {text}"


AsmItem = str | Instr | Comment
IMM2_BOUND = 1 << 18


@dataclass
class IsaBuilder:
    items: list[AsmItem] = field(default_factory=list)

    def comment(self, text: str) -> IsaBuilder:
        self.items.append(Comment(text))
        return self

    def instr(self, opcode: str, *args: AsmArg) -> IsaBuilder:
        self.items.append(Instr(opcode, args))
        return self

    def raw(self, line: str) -> IsaBuilder:
        self.items.append(line.rstrip("\n"))
        return self

    def extend(self, items: Iterable[AsmItem]) -> IsaBuilder:
        self.items.extend(items)
        return self

    def render(self) -> str:
        if not self.items:
            return ""
        return "\n".join(render_item(item) for item in legalize_large_immediates(self.items)) + "\n"


AsmInput = str | Renderable


def render_item(item: AsmItem) -> str:
    if isinstance(item, str):
        return item.rstrip("\n")
    return item.render()


def render_asm(value: AsmInput) -> str:
    if isinstance(value, str):
        return value
    return value.render()


def is_gp_zero(arg: AsmArg) -> bool:
    return isinstance(arg, Register) and arg.prefix == "gp" and arg.index == 0


def legalize_large_immediates(items: Iterable[AsmItem]) -> list[AsmItem]:
    """Split typed absolute S_ADDI_INT loads that exceed the immediate field.

    This is the typed equivalent of plena_frontend._fix_large_immediates.
    Raw string items are intentionally left alone until those call sites move
    onto typed instructions.
    """
    legalized: list[AsmItem] = []
    for item in items:
        if isinstance(item, Instr) and item.opcode == "S_ADDI_INT" and len(item.args) == 3:
            rd, rs, imm = item.args
            if isinstance(rd, Register) and is_gp_zero(rs) and isinstance(imm, int) and imm >= IMM2_BOUND:
                upper = imm >> 12
                lower = imm & 0xFFF
                legalized.append(Instr("S_LUI_INT", (rd, upper)))
                if lower:
                    legalized.append(Instr("S_ADDI_INT", (rd, rd, lower)))
                continue
        legalized.append(item)
    return legalized
