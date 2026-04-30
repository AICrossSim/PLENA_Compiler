"""HLIR -- the small, pass-friendly IR that flows between PLENA passes.

Pipeline overview:

    TIR PrimFunc
        |
        v   PASS 1  PlenaCodegen.lower_to_hlir()
    HLIR (Buffer + Op stream, no addresses)
        |
        v   PASS 2  AddressAllocationPass
    HLIR (each Buffer now has hbm/vram/mram address; Op args reference
          buffers, not raw addresses; stride/scale defaults filled in)
        |
        v   PASS 3  ISAEmitterPass
    Real ISA text

Why a tiny custom IR (rather than re-walking TIR each pass):
    - we need to attach pass-specific state (resolved addresses, register
      hints, scheduling info) to ops and buffers
    - TIR doesn't easily carry that without dialect machinery
    - keeping the IR small (fewer than ten op kinds for now) means each
      pass is a single-file Python function, easy to read and test

Op kinds intentionally mirror our `intrinsics` registry one-to-one:
each `plena.<x>` extern call in TIR becomes one HLIR Op with kind=<x>.
That keeps the codegen pass mechanical -- no clever rewrites yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import scope as _scope


@dataclass
class Buffer:
    """One tile/tensor with shape, scope, dtype, and (after Pass 2) address."""

    name: str
    scope: str                       # one of scope.{HBM,VRAM,MRAM,FPRAM}
    shape: Tuple[int, ...]
    dtype: str

    # Filled by AddressAllocationPass. None until then.
    address: Optional[int] = None     # base address in the buffer's scope
    hbm_offset: int = 0               # for HBM tiles that are sub-regions
    hbm_stride: Optional[int] = None  # row stride in HBM (defaults to mlen)
    hbm_scale_size: Optional[int] = None  # tile elem count (defaults to tile_elems)

    # Pass-attached scratch (e.g. logical_rows, logical_cols, row_blocks,
    # col_blocks for HBM buffers). Free-form so passes can stash hints
    # without growing this dataclass.
    annotations: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_elements(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n

    @property
    def byte_size(self) -> int:
        bits = {"float16": 16, "bfloat16": 16, "float32": 32, "int8": 8, "int32": 32}.get(
            self.dtype, 32
        )
        return self.num_elements * bits // 8


@dataclass
class BufferSlice:
    """A logical sub-region of a parent HBM buffer.

    Conventions (BSHD-aware, mirroring the layout rules in the package
    docstring):
        * `parent`: name of the parent Buffer in the same HLIRModule.
                    Must be an HBM buffer (slicing VRAM/MRAM is not a
                    thing PLENA exposes natively).
        * `starts`: per-dim start indices in the parent's logical shape.
                    Each entry is either:
                       - a Python int (compile-time-known)
                       - a tir.PrimExpr (loop-derived; lowered by
                         ExprMaterializer at ISA emit time)
        * `extents`: per-dim element counts of the slice in each parent
                     dim. Currently restricted to compile-time ints.

    Address resolution conventions:
        * The slice inherits the parent's `hbm_addr`, `hbm_stride`,
          `hbm_scale_size`. Pass 3 computes the additional `hbm_offset`
          from `starts` and adds it to the parent's `hbm_offset`.
        * Pass 3 iterates tiles within the slice using `extents` to
          decide row_blocks / col_blocks (BSHD-aware H*D merge same as
          for whole buffers).
    """
    parent: str                               # name of parent Buffer
    starts: Tuple[Any, ...]                   # int | tir.PrimExpr per dim
    extents: Tuple[int, ...]                  # int per dim


@dataclass
class Op:
    """One HLIR op.

    Two flavours:
        * Leaf op (most ops): `kind` matches a `plena.<x>` intrinsic.
          `buffer_args` are buffer names; `scalar_args` holds Python ints
          or `tir.PrimExpr` (compound expressions involving loop vars).
        * Structured op (only `for` for now): `kind == "for"`, `body` is
          a non-empty list of nested ops, and `annotations` holds the
          loop metadata (`loop_var`, `extent`, `init`). Pass 3 recurses
          on `body` while binding `loop_var` to a GP register.

    After Pass 2 every buffer arg has a resolved address on its Buffer.
    Pass 3 reads those addresses (and any PrimExpr scalar args via
    ExprMaterializer) and emits ISA.
    """

    kind: str
    # Each entry is either a buffer name (whole-buffer reference) or a
    # BufferSlice (a sub-region of a parent buffer).
    buffer_args: List[Any]            # List[str | BufferSlice]
    scalar_args: List[Any] = field(default_factory=list)  # int | float | str | tir.PrimExpr
    annotations: Dict[str, Any] = field(default_factory=dict)  # debug/passes
    # Only set for structured ops (currently just "for"). Leaves leave it None.
    body: Optional[List["Op"]] = None


def make_for_op(
    loop_var,
    extent,
    body: List[Op],
    init: int = 0,
) -> Op:
    """Helper: build a structured For op."""
    return Op(
        kind="for",
        buffer_args=[],
        scalar_args=[],
        annotations={"loop_var": loop_var, "extent": extent, "init": init},
        body=body,
    )


@dataclass
class HLIRModule:
    """One PrimFunc lowered to HLIR. Buffers + linear op stream."""

    name: str
    buffers: Dict[str, Buffer]        # name -> Buffer
    ops: List[Op]
    # PrimFunc parameter names in their declaration order. Useful so
    # downstream passes know which buffers come in as inputs/outputs vs
    # internally-allocated scratch.
    param_names: List[str] = field(default_factory=list)

    def get_buffer(self, name: str) -> Buffer:
        if name not in self.buffers:
            raise KeyError(
                f"buffer {name!r} not found in HLIR. "
                f"Known: {sorted(self.buffers.keys())}"
            )
        return self.buffers[name]

    def buffers_in_scope(self, scope: str) -> List[Buffer]:
        return [b for b in self.buffers.values() if b.scope == scope]

    def __repr__(self) -> str:
        bs = ", ".join(f"{b.name}<{b.scope}>" for b in self.buffers.values())
        return f"HLIRModule({self.name!r}, buffers=[{bs}], ops={len(self.ops)})"


def format_hlir(mod: HLIRModule) -> str:
    """Pretty-print HLIR. Used for `--dump-hlir`."""
    lines = [f"HLIRModule(name={mod.name!r})", ""]
    lines.append(f"Params (in declaration order):")
    for p in mod.param_names:
        lines.append(f"  - {p}")
    lines.append("")

    lines.append("Buffers:")
    name_w = max((len(b.name) for b in mod.buffers.values()), default=4)
    for b in mod.buffers.values():
        addr = "<unalloc>" if b.address is None else str(b.address)
        shape_s = "x".join(str(s) for s in b.shape)
        extras = ""
        if b.scope == "hbm":
            extras = (
                f"  stride={b.hbm_stride}"
                f"  scale={b.hbm_scale_size}"
                f"  hbm_offset={b.hbm_offset}"
            )
        lines.append(
            f"  {b.name:<{name_w}}  scope={b.scope:<5}  addr={addr:<8}  "
            f"shape={shape_s}  dtype={b.dtype}{extras}"
        )
    lines.append("")

    lines.append("Ops:")
    _format_ops(mod.ops, lines, indent=2, prefix=[0])
    return "\n".join(lines) + "\n"


def _format_ops(ops: List[Op], lines: List[str], indent: int, prefix: List[int]) -> None:
    """Recursive op pretty-printer; handles structured ops (for) with nesting."""
    for op in ops:
        idx = prefix[0]
        prefix[0] += 1
        ind = " " * indent
        if op.kind == "for":
            lv = op.annotations.get("loop_var")
            ext = op.annotations.get("extent")
            init = op.annotations.get("init", 0)
            lines.append(
                f"{ind}[{idx:2d}]  for  {getattr(lv, 'name', lv)} "
                f"in [{init}, {ext}):"
            )
            _format_ops(op.body or [], lines, indent + 4, prefix)
        else:
            bs = ", ".join(_fmt_buf_arg(a) for a in op.buffer_args) if op.buffer_args else "-"
            ss = ", ".join(_fmt_scalar(a) for a in op.scalar_args) if op.scalar_args else "-"
            lines.append(f"{ind}[{idx:2d}]  {op.kind:<14}  bufs=({bs})  scalars=({ss})")


def _fmt_buf_arg(a) -> str:
    """Render a buffer ref or slice for HLIR dump."""
    if isinstance(a, str):
        return a
    if isinstance(a, BufferSlice):
        starts = ",".join(str(s) if isinstance(s, (int, float)) else f"<{type(s).__name__}>"
                          for s in a.starts)
        extents = ",".join(str(e) for e in a.extents)
        return f"{a.parent}[starts=({starts}), ext=({extents})]"
    return str(a)


def _fmt_scalar(x) -> str:
    """Compact display for ints / strs / PrimExprs."""
    if isinstance(x, (int, float, str)):
        return str(x)
    return f"<{type(x).__name__} {x}>"


# Sanity helper used by passes to assert progress.
def assert_addresses_resolved(mod: HLIRModule) -> None:
    missing = [b.name for b in mod.buffers.values() if b.address is None]
    if missing:
        raise RuntimeError(
            f"address allocation incomplete; buffers without address: {missing}"
        )


__all__ = [
    "Buffer", "BufferSlice", "Op", "HLIRModule",
    "make_for_op",
    "assert_addresses_resolved", "format_hlir",
]
