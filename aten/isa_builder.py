"""Typed final-schedule IR for the ATen PLENA compiler path.

The production compiler still has a mixture of typed builders and legacy ASM
templates.  This module gives both paths one final-schedule representation:
rendering and analytical accounting traverse the same nodes *after* immediate
legalization and loop formation.  The metadata is deliberately hardware and
model agnostic; consumers may interpret stages and variants without teaching
the compiler a latency formula.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol, Union

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


AsmArg = Union[str, int, Register]


def render_arg(arg: AsmArg) -> str:
    if isinstance(arg, Register):
        return arg.render()
    return str(arg)


@dataclass(frozen=True)
class RepeatAxis:
    """Affine address movement attached to one symbolic repeat axis."""

    name: str
    count: int
    deltas: tuple[tuple[str, int], ...] = ()

    @classmethod
    def from_mapping(cls, name: str, count: int, deltas: Mapping[str, int]) -> "RepeatAxis":
        return cls(name=name, count=count, deltas=tuple(sorted((str(k), int(v)) for k, v in deltas.items())))

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError(f"RepeatAxis count must be nonnegative, got {self.count}")


@dataclass(frozen=True)
class DmaTransfer:
    """Compiler-owned physical DMA geometry.

    Units are explicit.  Base addresses and strides are bytes; ``amount`` and
    ``write_amount`` retain the ISA transfer-count semantics.  ``role`` is a
    semantic ownership label such as ``weight``, ``activation`` or ``kv``.
    """

    opcode: str
    direction: str
    role: str
    element_base_bytes: int
    scale_base_bytes: int | None
    dim: int
    amount: int
    stride_bytes: int
    rstride: int = 0
    write_amount: int = 1
    precision: str = "runtime"
    element_bytes: int = 1
    axes: tuple[RepeatAxis, ...] = ()
    geometry_fidelity: str = "exact"
    memory_object: str | None = None

    def __post_init__(self) -> None:
        if not self.opcode.startswith("H_"):
            raise ValueError(f"DMA opcode must start with H_, got {self.opcode!r}")
        if self.direction not in {"read", "write"}:
            raise ValueError(f"DMA direction must be read/write, got {self.direction!r}")
        if not self.role:
            raise ValueError("DMA role must be non-empty")
        for name in ("element_base_bytes", "dim", "amount", "stride_bytes", "rstride", "write_amount"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"DmaTransfer.{name} must be nonnegative")
        if self.scale_base_bytes is not None and self.scale_base_bytes < 0:
            raise ValueError("DmaTransfer.scale_base_bytes must be nonnegative")
        if self.element_bytes <= 0:
            raise ValueError("DmaTransfer.element_bytes must be positive")


@dataclass(frozen=True)
class ActiveDimensions:
    """Logical activity carried by a physical instruction."""

    rows: int | None = None
    cols: int | None = None
    lanes: int | None = None
    total_lanes: int | None = None
    bits: int | None = None

    def __post_init__(self) -> None:
        for name in ("rows", "cols", "lanes", "total_lanes", "bits"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"ActiveDimensions.{name} must be nonnegative")
        if self.lanes is not None and self.total_lanes is not None and self.lanes > self.total_lanes:
            raise ValueError("active lanes cannot exceed total lanes")


@dataclass(frozen=True)
class SramActivity:
    """One compiler-known SRAM access made by an instruction."""

    memory: str
    direction: str
    accesses: int = 1
    bytes_per_access: int | None = None

    def __post_init__(self) -> None:
        if self.direction not in {"read", "write"}:
            raise ValueError(f"SRAM direction must be read/write, got {self.direction!r}")
        if self.accesses < 0:
            raise ValueError("SramActivity.accesses must be nonnegative")
        if self.bytes_per_access is not None and self.bytes_per_access < 0:
            raise ValueError("SramActivity.bytes_per_access must be nonnegative")


@dataclass(frozen=True)
class Instr:
    opcode: str
    args: tuple[AsmArg, ...] = ()
    variant: tuple[tuple[str, str], ...] = ()
    active: ActiveDimensions | None = None
    dma: DmaTransfer | None = None
    sram: tuple[SramActivity, ...] = ()

    def render(self) -> str:
        if not self.args:
            return self.opcode
        return f"{self.opcode} {', '.join(render_arg(arg) for arg in self.args)}"

    @classmethod
    def with_metadata(
        cls,
        opcode: str,
        args: Iterable[AsmArg] = (),
        *,
        variant: Mapping[str, Any] | None = None,
        active: ActiveDimensions | None = None,
        dma: DmaTransfer | None = None,
        sram: Iterable[SramActivity] = (),
    ) -> "Instr":
        normalized_variant = tuple(
            sorted((str(key), str(value)) for key, value in (variant or {}).items())
        )
        return cls(
            opcode=opcode,
            args=tuple(args),
            variant=normalized_variant,
            active=active,
            dma=dma,
            sram=tuple(sram),
        )


@dataclass(frozen=True)
class Comment:
    text: str

    def render(self) -> str:
        text = self.text.rstrip()
        if text.startswith(";"):
            return text
        return f"; {text}"


@dataclass(frozen=True)
class Sequence:
    items: tuple["AsmItem", ...]


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
    """Opaque hierarchical ownership path for a final schedule subtree."""

    path: str
    body: Sequence

    def __post_init__(self) -> None:
        if not self.path or self.path.startswith("/") or self.path.endswith("/"):
            raise ValueError(f"stage path must be a non-empty relative path, got {self.path!r}")


AsmItem = Union[str, Instr, Comment, Sequence, CompileTimeRepeat, HardwareLoop, Stage]
IMM2_BOUND = 1 << 18


@dataclass
class IsaBuilder:
    items: list[AsmItem] = field(default_factory=list)

    def comment(self, text: str) -> "IsaBuilder":
        self.items.append(Comment(text))
        return self

    def instr(
        self,
        opcode: str,
        *args: AsmArg,
        variant: Mapping[str, Any] | None = None,
        active: ActiveDimensions | None = None,
        dma: DmaTransfer | None = None,
        sram: Iterable[SramActivity] = (),
    ) -> "IsaBuilder":
        if dma is not None and dma.opcode != opcode:
            raise ValueError(f"DMA metadata opcode {dma.opcode!r} does not match {opcode!r}")
        self.items.append(
            Instr.with_metadata(
                opcode,
                args,
                variant=variant,
                active=active,
                dma=dma,
                sram=sram,
            )
        )
        return self

    def raw(self, line: str) -> "IsaBuilder":
        self.items.append(line.rstrip("\n"))
        return self

    def extend(self, items: Iterable[AsmItem]) -> "IsaBuilder":
        self.items.extend(items)
        return self

    def sequence(self, items: Iterable[AsmItem]) -> "IsaBuilder":
        self.items.append(Sequence(tuple(items)))
        return self

    def repeat(
        self,
        count: int,
        body: "IsaBuilder | Sequence | Iterable[AsmItem]",
        *,
        axis: RepeatAxis | None = None,
    ) -> "IsaBuilder":
        self.items.append(CompileTimeRepeat(count=count, body=as_sequence(body), axis=axis))
        return self

    def hardware_loop(
        self,
        loop_register: AsmArg,
        count: int,
        body: "IsaBuilder | Sequence | Iterable[AsmItem]",
        *,
        effective_count: int | None = None,
        axis: RepeatAxis | None = None,
    ) -> "IsaBuilder":
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

    def stage(self, path: str, body: "IsaBuilder | Sequence | Iterable[AsmItem]") -> "IsaBuilder":
        self.items.append(Stage(path=path, body=as_sequence(body)))
        return self

    def finalized(self) -> Sequence:
        """Return the immutable schedule after compiler-wide legalization."""
        return Sequence(tuple(legalize_large_immediates(self.items)))

    def render(self) -> str:
        rendered = list(render_items(self.finalized().items))
        return "\n".join(rendered) + ("\n" if rendered else "")


AsmInput = Union[str, Renderable]


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
    """Parse one already-legal instruction without changing its spelling."""
    stripped = line.strip()
    opcode, separator, tail = stripped.partition(" ")
    if not separator:
        return Instr(opcode)
    args: list[AsmArg] = []
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
    """Convert finalized legacy-template text into typed schedule leaves."""
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
            raise ValueError(f"unsupported raw ASM line in final schedule: {line}")
        items.append(instr_from_rendered_line(line))
    return Sequence(tuple(items))


def as_sequence(value: "IsaBuilder | Sequence | Iterable[AsmItem]") -> Sequence:
    if isinstance(value, IsaBuilder):
        return Sequence(tuple(value.items))
    if isinstance(value, Sequence):
        return value
    return Sequence(tuple(value))


def is_gp_zero(arg: AsmArg) -> bool:
    return isinstance(arg, Register) and arg.prefix == "gp" and arg.index == 0


def legalize_large_immediates(items: Iterable[AsmItem]) -> list[AsmItem]:
    """Legalize typed immediates recursively without changing schedule metadata."""
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
            legalized.append(Stage(item.path, Sequence(tuple(legalize_large_immediates(item.body.items)))))
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


__all__ = [
    "ActiveDimensions",
    "AsmArg",
    "AsmInput",
    "AsmItem",
    "Comment",
    "CompileTimeRepeat",
    "DmaTransfer",
    "HardwareLoop",
    "Instr",
    "IsaBuilder",
    "Register",
    "RepeatAxis",
    "Sequence",
    "SramActivity",
    "Stage",
    "addr",
    "as_sequence",
    "fp",
    "gp",
    "legalize_large_immediates",
    "parse_legacy_asm",
    "render_arg",
    "render_asm",
    "render_items",
]
