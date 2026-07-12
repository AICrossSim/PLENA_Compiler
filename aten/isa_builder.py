"""Typed ISA builder for the ATen PLENA compiler path.

This is intentionally small: it models the physical instruction stream and
prints the same assembly syntax the existing compiler emits.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from asm_templates._imm import add_large_int as _add_large_int_lines
from asm_templates._imm import load_large_int as _load_large_int_lines


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
    dma: DmaTransfer | None = None

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


@dataclass(frozen=True)
class RepeatAxis:
    """One compressed repetition axis for a symbolic DMA event."""

    name: str
    count: int
    element_base_delta: int = 0
    scale_base_delta: int = 0
    logical_element_delta: int | None = None
    logical_scale_delta: int | None = None


@dataclass(frozen=True)
class DmaTransfer:
    """Semantic description of one HBM instruction.

    Addresses and sizes use bytes/elements exactly as the transactional DMA
    entry points do.  Rendering ignores this metadata; CostEmitter consumes it
    without having to reconstruct register state from assembly text.
    """

    opcode: str
    direction: str
    precision: str
    element_base: int
    scale_base: int
    dim: int
    amount: int
    stride: int
    rstride: int = 0
    write_amount: int = 1
    axes: tuple[RepeatAxis, ...] = ()
    geometry_fidelity: str = "exact"
    source: str | None = None
    memory_object: str | None = None
    logical_object_elements: int | None = None
    precision_role: str | None = None
    logical_element_offset: int | None = None
    logical_scale_offset: int | None = None
    logical_stride: int | None = None


@dataclass(frozen=True)
class Sequence:
    items: tuple[AsmItem, ...]


@dataclass(frozen=True)
class CompileTimeRepeat:
    count: int
    body: Sequence
    axis: RepeatAxis | None = None


@dataclass(frozen=True)
class HardwareLoop:
    loop_register: AsmArg
    count: int
    body: Sequence
    effective_count: int | None = None
    axis: RepeatAxis | None = None


@dataclass(frozen=True)
class Stage:
    name: str
    body: Sequence


AsmItem = str | Instr | Comment | Sequence | CompileTimeRepeat | HardwareLoop | Stage
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

    def dma_instr(self, opcode: str, *args: AsmArg, dma: DmaTransfer) -> IsaBuilder:
        if dma.opcode != opcode:
            raise ValueError(f"DMA metadata opcode {dma.opcode!r} does not match instruction {opcode!r}")
        self.items.append(Instr(opcode, args, dma=dma))
        return self

    def raw(self, line: str) -> IsaBuilder:
        self.items.append(line.rstrip("\n"))
        return self

    def extend(self, items: Iterable[AsmItem]) -> IsaBuilder:
        self.items.extend(items)
        return self

    def sequence(self, items: Iterable[AsmItem]) -> IsaBuilder:
        self.items.append(Sequence(tuple(items)))
        return self

    def repeat(
        self,
        count: int,
        body: IsaBuilder | Sequence | Iterable[AsmItem],
        *,
        axis: RepeatAxis | None = None,
    ) -> IsaBuilder:
        self.items.append(CompileTimeRepeat(count, as_sequence(body), axis=axis))
        return self

    def hardware_loop(
        self,
        loop_register: AsmArg,
        count: int,
        body: IsaBuilder | Sequence | Iterable[AsmItem],
        *,
        effective_count: int | None = None,
        axis: RepeatAxis | None = None,
    ) -> IsaBuilder:
        self.items.append(
            HardwareLoop(
                loop_register=loop_register,
                count=count,
                body=as_sequence(body),
                effective_count=effective_count,
                axis=axis,
            )
        )
        return self

    def stage(self, name: str, body: IsaBuilder | Sequence | Iterable[AsmItem]) -> IsaBuilder:
        self.items.append(Stage(name=name, body=as_sequence(body)))
        return self

    def render(self) -> str:
        if not self.items:
            return ""
        rendered = list(render_items(legalize_large_immediates(self.items)))
        return "\n".join(rendered) + ("\n" if rendered else "")


AsmInput = str | Renderable


def render_item(item: AsmItem) -> str:
    if isinstance(item, str):
        return item.rstrip("\n")
    if isinstance(item, (Instr, Comment)):
        return item.render()
    return "\n".join(render_items((item,)))


def render_items(items: Iterable[AsmItem]) -> Iterable[str]:
    for item in items:
        if isinstance(item, str):
            yield item.rstrip("\n")
        elif isinstance(item, (Instr, Comment)):
            yield item.render()
        elif isinstance(item, Sequence):
            yield from render_items(item.items)
        elif isinstance(item, CompileTimeRepeat):
            if item.count < 0:
                raise ValueError(f"CompileTimeRepeat count must be >= 0, got {item.count}")
            for _ in range(item.count):
                yield from render_items(item.body.items)
        elif isinstance(item, HardwareLoop):
            if item.count <= 0:
                raise ValueError(f"HardwareLoop count must be > 0, got {item.count}")
            yield Instr("C_LOOP_START", (item.loop_register, item.count)).render()
            yield from render_items(item.body.items)
            yield Instr("C_LOOP_END", (item.loop_register,)).render()
        elif isinstance(item, Stage):
            yield from render_items(item.body.items)
        else:
            raise TypeError(f"Unsupported ASM item: {type(item).__name__}")


def render_asm(value: AsmInput) -> str:
    if isinstance(value, str):
        return value
    return value.render()


def instr_from_rendered_line(line: str) -> Instr:
    """Wrap one already-formatted instruction line without changing spelling."""
    stripped = line.strip()
    opcode, separator, tail = stripped.partition(" ")
    if not separator:
        return Instr(opcode)
    args = []
    for value in tail.split(","):
        value = value.strip()
        if value.startswith("gp") and value[2:].isdigit():
            args.append(gp(int(value[2:])))
        elif value.startswith("f") and value[1:].isdigit():
            args.append(fp(int(value[1:])))
        elif value.startswith("a") and value[1:].isdigit():
            args.append(addr(int(value[1:])))
        else:
            try:
                args.append(int(value, 0))
            except ValueError:
                args.append(value)
    return Instr(opcode, tuple(args))


@lru_cache(maxsize=4096)
def parse_legacy_asm(text: str) -> Sequence:
    """Convert legacy template text into typed instructions for CostSink."""
    items: list[AsmItem] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(";"):
            items.append(Comment(line))
            continue
        opcode = line.split(maxsplit=1)[0]
        if not opcode.startswith(("S_", "C_", "H_", "V_", "M_")):
            raise ValueError(f"unsupported raw ASM line in symbolic cost lowering: {line}")
        items.append(instr_from_rendered_line(line))
    return Sequence(tuple(items))


def as_sequence(value: IsaBuilder | Sequence | Iterable[AsmItem]) -> Sequence:
    if isinstance(value, IsaBuilder):
        return Sequence(tuple(value.items))
    if isinstance(value, Sequence):
        return value
    return Sequence(tuple(value))


def is_gp_zero(arg: AsmArg) -> bool:
    return isinstance(arg, Register) and arg.prefix == "gp" and arg.index == 0


def legalize_large_immediates(items: Iterable[AsmItem]) -> list[AsmItem]:
    """Split typed S_ADDI_INT instructions that exceed the immediate field.

    This is the typed equivalent of plena_frontend._fix_large_immediates. Raw
    string items are intentionally left alone until those call sites move onto
    typed instructions.
    """
    legalized: list[AsmItem] = []
    for item in items:
        if isinstance(item, Sequence):
            legalized.append(Sequence(tuple(legalize_large_immediates(item.items))))
            continue
        if isinstance(item, CompileTimeRepeat):
            legalized.append(
                CompileTimeRepeat(
                    item.count,
                    Sequence(tuple(legalize_large_immediates(item.body.items))),
                    axis=item.axis,
                )
            )
            continue
        if isinstance(item, HardwareLoop):
            legalized.append(
                HardwareLoop(
                    loop_register=item.loop_register,
                    count=item.count,
                    body=Sequence(tuple(legalize_large_immediates(item.body.items))),
                    effective_count=item.effective_count,
                    axis=item.axis,
                )
            )
            continue
        if isinstance(item, Stage):
            legalized.append(
                Stage(item.name, Sequence(tuple(legalize_large_immediates(item.body.items))))
            )
            continue
        if isinstance(item, Instr) and item.opcode == "S_ADDI_INT" and len(item.args) == 3:
            rd, rs, imm = item.args
            if (
                isinstance(rd, Register)
                and isinstance(rs, Register)
                and rd.prefix == "gp"
                and rs.prefix == "gp"
                and isinstance(imm, int)
                and imm >= IMM2_BOUND
            ):
                replacement = (
                    _load_large_int_lines(rd.index, imm)
                    if is_gp_zero(rs)
                    else _add_large_int_lines(rd.index, rs.index, imm, temp_reg=None)
                )
                legalized.extend(instr_from_rendered_line(line) for line in replacement)
                continue
        legalized.append(item)
    return legalized
