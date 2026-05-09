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


@dataclass(frozen=True)
class TileLayout:
    """Physical multi-tile VRAM/MRAM layout for a 4D BSHD-shaped buffer.

    A buffer with logical shape ``(B, S, H, D)`` whose ``S`` and/or ``D``
    overflow MLEN gets stored physically as a 7D layout:

        (D_TILES, S_TILES, H_GROUPS, B,   MLEN, LANE_COUNT, D_INNER)
        └─────── outer tile index ───────┘ └──── inner per-tile ──┘

    where:
        S_TILES  = ceildiv(S, MLEN)
        D_TILES  = ceildiv(D, MLEN)              if D > MLEN else 1
        D_INNER  = MLEN if D > MLEN else min(D, MLEN)
        H_GROUPS = ceildiv(H, LANE_COUNT)
        LANE_COUNT determined by D_INNER:
            * D_INNER == MLEN  → LANE_COUNT = 1
            * D_INNER  < MLEN  → LANE_COUNT = MLEN // D_INNER
                                  (typically MLEN // HLEN, hardware-dependent)

    Each per-tile inner block ``(MLEN, LANE_COUNT, D_INNER)`` is exactly
    one ``H_LOAD_V`` worth of data. The outer ``(D_TILES, S_TILES,
    H_GROUPS, B)`` is the tile grid; multi-tile DMAs walk it in
    row-major (D_TILE outermost so D_TILES > 1 cases stay contiguous in
    the natural way).

    Logical (b, s, h, d) → physical 7D:
        d_tile  = d // MLEN          d_inner = d %  MLEN
        s_tile  = s // MLEN          s_inner = s %  MLEN
        h_grp   = h // LANE_COUNT    lane    = h %  LANE_COUNT
    Physical flat offset:
        d_tile  * (S_TILES * H_GROUPS * B * MLEN * LANE_COUNT * D_INNER)
      + s_tile  * (H_GROUPS * B * MLEN * LANE_COUNT * D_INNER)
      + h_grp   * (B * MLEN * LANE_COUNT * D_INNER)
      + b       * (MLEN * LANE_COUNT * D_INNER)
      + s_inner * (LANE_COUNT * D_INNER)
      + lane    * D_INNER
      + d_inner

    Total physical element count equals logical numel — the layout is
    just a permutation; AddressAllocationPass uses ``numel`` regardless.
    """
    # Logical 4D shape (B, S, H, D).
    logical_b: int
    logical_s: int
    logical_h: int
    logical_d: int
    # Tile grid sizes.
    d_tiles: int
    s_tiles: int
    h_groups: int
    # Inner tile dims.
    mlen: int
    lane_count: int
    d_inner: int

    @property
    def tile_elems(self) -> int:
        """Element count of one inner tile = MLEN * LANE_COUNT * D_INNER."""
        return self.mlen * self.lane_count * self.d_inner

    @property
    def num_tiles(self) -> int:
        """Total number of inner tiles in the buffer."""
        return self.d_tiles * self.s_tiles * self.h_groups * self.logical_b


# Layout name -> (batch_idx, row_idx, channel_idx, col_idx) into a 4D shape.
# The TileLayout / 7D physical layout / multi-tile DMA logic is all written in
# canonical BSHD terms (batch / row-tiled / channel-grouped / col-tiled).
# A buffer declared in another layout (NCHW etc.) just has its axes permuted
# at the boundary — every downstream pass keeps thinking BSHD.
LAYOUT_AXES = {
    "BSHD": (0, 1, 2, 3),  # B=axes[0], S=axes[1], H=axes[2], D=axes[3]
    "NCHW": (0, 2, 1, 3),  # N=axes[0], H=axes[2], C=axes[1], W=axes[3]
}

DEFAULT_LAYOUT = "BSHD"


def _select_axes(values, layout: str):
    """Pick (batch, row, channel, col) values from a 4D ``values`` sequence
    according to ``layout``. ``values`` is either a shape tuple or a starts
    PrimExpr tuple — same indexing rule applies to both."""
    if layout not in LAYOUT_AXES:
        raise ValueError(
            f"unknown layout {layout!r}; known: {sorted(LAYOUT_AXES)}"
        )
    bi, ri, ci, di = LAYOUT_AXES[layout]
    return values[bi], values[ri], values[ci], values[di]


def logical_2d_extents(shape_or_extents, layout: str = DEFAULT_LAYOUT):
    """Project a 4D shape (or slice extents) to ``(rows, cols)``.

    ``rows`` = batch * row-dim (the dims that get s-tiled / batched);
    ``cols`` = channel * col-dim (the dims that fit inside one MLEN-wide
    chunk per logical row).

    For BSHD the row-dim is axes[1] and channel-dim is axes[2], so the
    legacy "merge last two as cols, fold the rest into rows" heuristic
    happens to match. For NCHW the row-dim is axes[2] and the channel
    is axes[1], so we have to look up axes per ``LAYOUT_AXES`` instead
    of going by position.

    Lower-rank shapes (3D / 2D / 1D) keep the old per-rank heuristic
    since they're not multi-tile-eligible anyway.
    """
    n = len(shape_or_extents)
    if n == 0:
        return (1, 1)
    if n == 1:
        return (1, int(shape_or_extents[0]))
    if n == 2:
        return (int(shape_or_extents[0]), int(shape_or_extents[1]))
    if n == 3:
        # Keep the legacy "fold leading dims into rows, cols = D" rule.
        return (int(shape_or_extents[0]), int(shape_or_extents[1]) * int(shape_or_extents[2]))
    if n != 4:
        raise ValueError(
            f"logical_2d_extents only handles up to 4D; got {n}-D"
        )
    bi, ri, ci, di = LAYOUT_AXES[layout]
    rows = int(shape_or_extents[bi]) * int(shape_or_extents[ri])
    cols = int(shape_or_extents[ci]) * int(shape_or_extents[di])
    return (rows, cols)


def hbm_strides_for_layout(shape, layout: str = DEFAULT_LAYOUT):
    """Return ``(b_stride, s_stride, h_stride, d_stride)`` in *canonical*
    (B, S, H, D) order for a 4D shape laid out row-major in HBM under
    ``layout``.

    Each stride is in element units (HBM is row-major in source-layout
    order). For BSHD the strides come out in trivial order; for NCHW
    they get permuted because the row-dim (H) and channel-dim (C) swap
    relative to canonical positions.

    Used by ``_emit_dma_h2v_slice_multi_tile`` and friends to compute
    the per-tile HBM offset.
    """
    if len(shape) != 4:
        raise ValueError(f"hbm_strides_for_layout needs 4D shape; got {tuple(shape)}")
    src_strides = [1, 1, 1, 1]
    for i in range(2, -1, -1):
        src_strides[i] = src_strides[i + 1] * int(shape[i + 1])
    bi, ri, ci, di = LAYOUT_AXES[layout]
    return (src_strides[bi], src_strides[ri], src_strides[ci], src_strides[di])


def make_tile_layout(
    *, shape=None, layout: str = DEFAULT_LAYOUT, mlen: int, hlen: int,
    # Legacy keyword form (b/s/h/d) kept for back-compat with any older
    # caller. New callers should pass ``shape=...`` plus ``layout=...``.
    b: Optional[int] = None, s: Optional[int] = None,
    h: Optional[int] = None, d: Optional[int] = None,
) -> Optional["TileLayout"]:
    """Build a TileLayout for a 4D buffer if (and only if) it needs
    multi-tile storage.

    Two calling forms (kept compatible during the layout migration):
        make_tile_layout(shape=(...), layout="NCHW", mlen=..., hlen=...)
        make_tile_layout(b=..., s=..., h=..., d=..., mlen=..., hlen=...)

    The ``shape``/``layout`` form picks (b, s, h, d) per ``LAYOUT_AXES``;
    everything downstream still works in canonical BSHD terms. Returns
    None for buffers that fit a single inner tile (caller treats them
    as plain row-major just like before).

    Tiling is required when any of:
        * S > MLEN (need S-direction tiling)
        * D > MLEN (need D-direction tiling, the new "outer D_TILES" dim)
        * H > MLEN // HLEN (need H-direction lane grouping past one group)

    For now we only handle ``b == 1`` cleanly; multi-batch tiling can be
    added later by a caller wanting it.
    """
    if shape is not None:
        if any(v is not None for v in (b, s, h, d)):
            raise ValueError(
                "make_tile_layout: pass either ``shape``/``layout`` OR the "
                "legacy ``b``/``s``/``h``/``d`` kwargs, not both"
            )
        if len(shape) != 4:
            raise ValueError(
                f"make_tile_layout: shape must be 4D; got {tuple(shape)}"
            )
        b, s, h, d = (int(x) for x in _select_axes(shape, layout))
    else:
        if any(v is None for v in (b, s, h, d)):
            raise ValueError(
                "make_tile_layout: legacy form requires all four of "
                "b/s/h/d"
            )
    if mlen <= 0 or hlen <= 0 or hlen > mlen or mlen % hlen != 0:
        raise ValueError(
            f"invalid mlen={mlen}, hlen={hlen}: require 0 < hlen <= mlen and "
            f"mlen % hlen == 0"
        )

    # Inner-tile shape derivation. D == mlen → LANE_COUNT = 1;
    # D < mlen (typically D == hlen) → LANE_COUNT = mlen // D.
    if d >= mlen:
        d_inner = mlen
        lane_count = 1
    else:
        if mlen % d != 0:
            raise ValueError(
                f"narrow-D tile requires mlen % d == 0; got mlen={mlen}, d={d}"
            )
        d_inner = d
        lane_count = mlen // d

    d_tiles = (d + mlen - 1) // mlen if d > mlen else 1
    s_tiles = (s + mlen - 1) // mlen
    if h % lane_count != 0:
        raise ValueError(
            f"H ({h}) must be a multiple of LANE_COUNT ({lane_count})"
        )
    h_groups = h // lane_count

    # Single-tile fast path. Layout-conditional so we can preserve
    # both BSHD's "row-major scratch fragment" convention and NCHW's
    # "per-channel tile" semantics:
    #
    # * BSHD (legacy default) — return None whenever s ≤ mlen AND
    #   d ≤ mlen, regardless of h_groups. Kernels like
    #   flash_attention_min and tiled_conv2d allocate VRAM-only
    #   fragments like S_loc (1, H, mlen, mlen) that get expanded
    #   to 4D by ``allocate_group_memory`` but conceptually live as
    #   a 2D (rows, mlen) tile in row-major. Forcing the 7D
    #   physical layout here permutes the offsets and breaks every
    #   internal access (since these buffers never see HBM, the
    #   logical-vs-physical layout difference matters).
    #
    # * Anything else (NCHW for now) — require ALL tile-grid dims to
    #   collapse to 1 (d_tiles = s_tiles = h_groups = b = 1). NCHW's
    #   channel axis sits outer of (H, W) in HBM, so a multi-channel
    #   buffer with h_groups > 1 genuinely needs multi-tile staging
    #   even when each per-channel block fits a single MLEN×MLEN
    #   inner tile — otherwise the stage_output / v2h_slice fast
    #   paths would compute the wrong cross-channel HBM offset.
    if layout == "BSHD":
        if s <= mlen and d <= mlen:
            return None
    else:
        if d_tiles == 1 and s_tiles == 1 and h_groups == 1 and b == 1:
            return None

    return TileLayout(
        logical_b=b, logical_s=s, logical_h=h, logical_d=d,
        d_tiles=d_tiles, s_tiles=s_tiles, h_groups=h_groups,
        mlen=mlen, lane_count=lane_count, d_inner=d_inner,
    )


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

    # Multi-tile physical layout descriptor for VRAM/MRAM buffers whose
    # logical shape overflows one inner tile. None means "single tile,
    # plain row-major" — the existing case all kernels written before
    # this change relied on. See ``TileLayout`` docstring for the
    # logical→physical mapping.
    tile_layout: Optional[TileLayout] = None

    # 4D-buffer layout hint, used to resolve which axis is the row /
    # channel / col dim. ``BSHD`` (the default) means axes are already
    # in canonical batch-row-channel-col order; ``NCHW`` means
    # axes[1] is the channel and axes[2] is the row, so callers must
    # permute before computing tile offsets / lane groups. Ignored for
    # non-4D shapes. See ``LAYOUT_AXES`` for the mapping.
    layout: str = "BSHD"

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


@dataclass(frozen=True)
class BufferElement:
    """One scalar element reference within a buffer.

    Used for FPRAM-backed `_at` operands where the frontend keeps
    tilelang-style indexing (`buf[row]`, later expanded to `buf[by,row]`)
    but the ISA ultimately expects a flat scalar address.
    """

    buffer: str
    indices: Tuple[Any, ...]


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
    if isinstance(x, BufferElement):
        idx = ", ".join(str(i) if isinstance(i, (int, float, str)) else f"<{type(i).__name__}>"
                        for i in x.indices)
        return f"{x.buffer}[{idx}]"
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
    "Buffer", "BufferSlice", "BufferElement", "Op", "HLIRModule",
    "make_for_op",
    "assert_addresses_resolved", "format_hlir",
]
