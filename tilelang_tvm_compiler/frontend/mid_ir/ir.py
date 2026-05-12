"""Mid-IR node definitions.

A small dataclass IR sitting between raw TIR and PLENA HLIR. Designed
to make the **lane-fusion rewrite a sequence of mechanical, one-thing-each
passes** instead of the layered Graph-IR-everything approach.

Pipeline (each step is one pass; the IR shape only changes in the
specific ways noted):

  raw tir.PrimFunc
    │
    │  pass_1_fold:  nested for + BufferStore → Elementwise / Reduce / Broadcast
    │  pass_2_mark:  tag dma / btmm gemm / elementwise sites with .marker
    │                (blockIdx still alive throughout)
    │  pass_3_split: pick a blockIdx axis; split it into (number, phase);
    │                grow every non-global buffer by one outer `cluster` dim
    │                — only *add*, never permute.
    │  pass_4_async: wrap each marked op in Async(...)
    │  pass_5_loop:  introduce `for phase in [0, cluster_count)` but break
    │                it into multiple fors at every Async boundary
    │  pass_6_fuse:  collapse `for phase: <single async op>` into
    │                MultiLaneOp(op, dim_map = {cluster_axis: (buf, dim)})
    │  pass_7_perm:  use dim_map to permute the cluster axis into the
    │                physical-layout slot per buffer
    │
  mid_ir ready
    │
    │  pass_8_to_plena: lower to HLIR
    │
  HLIRModule

This file ONLY defines the nodes + a tiny printer. No pass logic here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Op kinds
# ---------------------------------------------------------------------------


class BinOp(Enum):
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    MAX = "max"
    MIN = "min"


class UnaryOp(Enum):
    EXP = "exp"
    RECI = "reci"
    SQRT = "sqrt"
    COPY = "copy"
    NEG = "neg"


class ReduceOp(Enum):
    SUM = "sum"
    MAX = "max"
    MIN = "min"


class Marker(Enum):
    """Tag attached to op sites by pass_2_mark. Drives async/cluster handling
    downstream. Untagged ops stay sequential, never get lane-fused."""
    DMA = "dma"        # HBM ↔ on-chip transfer
    BTMM = "btmm"      # head-fused matmul
    LANE_OP = "lane_op"  # an elementwise/reduce that lives inside the cluster


# ---------------------------------------------------------------------------
# Buffer references
# ---------------------------------------------------------------------------


@dataclass
class BufferDef:
    """A buffer the kernel allocates or receives.

    ``shape`` is the *logical* shape (what the kernel author wrote).
    Passes that grow the buffer (e.g. pass_3) update this in place by
    prepending dims; pass_7 may permute. ``scope`` is one of
    ``"global"`` / ``"shared"`` / ``"fragment"`` etc — same string the
    kernel uses with T.alloc_shared / T.alloc_fragment.

    ``cluster_dim`` is the index into ``shape`` of the lane/cluster
    axis (the dim pass_3_split prepended when growing the buffer for
    cluster fusion). pass_4b_view / pass_5b_burn_view track it through
    any axis permutation. ``None`` for buffers that don't have a
    cluster axis (HBM params, user-declared ``global.*`` caches, etc).
    Downstream addressing reads this directly instead of guessing the
    lane position from shape values.
    """
    name: str
    shape: List[int]           # logical extents (int-only; symbolic later)
    dtype: str
    scope: str = "global"
    cluster_dim: Optional[int] = None

    def with_outer_dim(self, extent: int) -> "BufferDef":
        # Prepending a dim shifts every existing axis by +1, including
        # the cluster_dim marker if set.
        new_cluster = None if self.cluster_dim is None else self.cluster_dim + 1
        return BufferDef(
            name=self.name,
            shape=[extent] + list(self.shape),
            dtype=self.dtype,
            scope=self.scope,
            cluster_dim=new_cluster,
        )


@dataclass
class BufferRef:
    """A read/write of a buffer at a given index tuple.

    ``indices`` lists one IndexExpr per dim of ``buffer.shape``. Each
    is either an int (static), a string (symbolic var name — e.g.
    ``"by"``, ``"by_phase"``, ``"row"``), or a Slice (``":"``-style
    whole-axis access).

    By convention pass_1_fold produces refs where every dim that the
    fused loop spanned is a ``Slice``, and remaining dims are concrete
    indices.

    ``view_perm`` is set by pass_4b_view to express the op-local view
    permutation: ``view_perm[i]`` says which physical dim the op sees
    as logical dim ``i``. Identity (None) means logical = physical.
    Example: physical buffer shape ``[lane=4, S=64, D=16]``;
    ``view_perm=[1, 0, 2]`` means the op sees ``[S=64, lane=4, D=16]``
    (BSHD view of a BHSD-shell buffer).
    """
    buffer: BufferDef
    indices: List["IndexExpr"]
    view_perm: Optional[List[int]] = None


@dataclass
class Slice:
    """``:`` — whole-axis access. May carry an explicit range later
    if needed; for now whole-axis is the only kind we care about."""
    pass


# An IndexExpr is one of: int (concrete index / extent literal),
# str (variable name), Slice (whole axis), or a compound dict
# {"op": "add", "args": [...]} for things like ``by_phase + by_number*C``.
# We keep the compound form opaque to start with — passes that need to
# manipulate the arithmetic can parse the dict.
IndexExpr = Union[int, str, Slice, dict]


# ---------------------------------------------------------------------------
# Op nodes — the three fold targets
# ---------------------------------------------------------------------------


@dataclass
class Elementwise:
    """``dst[idx] = op(src_0[idx], src_1[idx], ...)`` over matching axes.

    ``op`` is BinOp for 2+ srcs, UnaryOp for 1 src. All srcs and dst
    must have matching shapes on the axes participating in the
    operation (a Broadcast wraps a src whose shape is smaller).

    ``axis`` is None for full-shape elementwise. When set, only the
    given axis (or list of axes) is "active" — other axes are
    independent and the op fires once per element along them. This
    covers the "row op" family (axis=-1 means "act on last dim,
    broadcast over the others").

    ``size`` is the per-issue element count: how many elements ONE
    invocation of the op processes. Critical signal for downstream
    lowering — the fold pass merges some forms of element loop into
    an Elementwise, and ``size`` is what tells the lowering whether
    that fold represents a vector (``size == MLEN``, one
    ``V_*_VV/V_*_VF`` instruction per call) or a scalar
    (``size == 1``, one ``S_*_FP``). Without it, SIMD and SISD
    elementwise dst patterns collapse to the same mid_ir node and
    the lowering can't tell which ISA op family applies.

    ``can_async`` is True when the HW lowering is a single multi-lane
    vector instruction (``v_add`` / ``v_exp_v`` / ``v_reci_v`` etc.).
    False when the lowering is per-row (``row_sub_fp_at`` and friends —
    typically the case when one src is a Broadcast wrapping a smaller-
    rank fp scalar). pass_2_mark sets this; pass_4_async only wraps
    ops with can_async=True in Async regions.
    """
    dst: BufferRef
    srcs: List[Union[BufferRef, "Broadcast"]]
    op: Union[BinOp, UnaryOp]
    axis: Optional[Union[int, List[int]]] = None
    size: int = 1
    marker: Optional[Marker] = None
    can_async: bool = False


@dataclass
class Broadcast:
    """Wrap a smaller-rank src to match a larger-rank dst.

    ``broadcast_dims`` is the list of dst-dim indices along which
    ``src`` repeats. E.g. dst is rank 3, src is rank 1 with values per
    last-axis position → broadcast_dims = [0, 1].
    """
    src: BufferRef
    broadcast_dims: List[int]


@dataclass
class Reduce:
    """``dst[idx_without_axis] = reduce(src[idx], op, axis)``.

    ``axis`` is the single axis being collapsed (we don't fold
    multi-axis reductions at the mid-IR level). Use ``axis=-1`` for
    "reduce along the last dim", which is how row-reduce maps in.

    ``can_async`` is always False — reduce on PLENA is per-row
    (``row_reduce_max_at`` / ``row_reduce_sum_at``), one row at a time
    into a per-row fp scalar slot. No multi-lane reduce HW op exists.
    """
    dst: BufferRef
    src: BufferRef
    op: ReduceOp
    axis: int
    marker: Optional[Marker] = None
    can_async: bool = False


# ---------------------------------------------------------------------------
# Op nodes — gemm and dma stay as their own kinds (we don't decompose
# them into elementwise/reduce; the HW has dedicated instructions).
# ---------------------------------------------------------------------------


@dataclass
class Gemm:
    """``c = a @ b`` (transpose flags carried).

    The ``kind`` matches the kernel's ``T.attr(0, KIND, ...)`` — most
    importantly ``"btmm"`` for head-fused vs the default per-head form.
    Pass_2_mark sets the marker to BTMM when kind == "btmm".

    ``can_async`` is True for kind=="btmm" (one multi-lane M_BTMM
    instruction); False for kind=="overwrite" (per-head matmul that
    runs inside the lane loop, one matmul per lane).
    """
    a: BufferRef
    b: BufferRef
    c: BufferRef
    transpose_a: bool = False
    transpose_b: bool = False
    kind: str = "overwrite"
    marker: Optional[Marker] = None
    can_async: bool = False


@dataclass
class Dma:
    """``dst = src`` across a memory scope boundary (HBM ↔ on-chip).

    Both src and dst are BufferRefs whose ``indices`` describe the
    slice being transferred. Direction is implicit from src.scope /
    dst.scope.

    ``can_async`` is always True — DMA is always a single multi-lane
    HW instruction (``H_LOAD_V`` / ``H_STORE_V`` etc.).
    """
    src: BufferRef
    dst: BufferRef
    marker: Optional[Marker] = None
    can_async: bool = False


@dataclass
class RawStore:
    """A BufferStore that the fold pass couldn't recognize as one of
    Elementwise / Broadcast / Reduce.

    Lives inside an enclosing ``For`` loop (or a chain of them) and
    represents an opaque per-iteration scalar update. Examples that
    end up as RawStore today:

      * ``in_FP_padded[MLEN + k] = 0``        (compound dst index)
      * ``shift_FP[m] = in_FP_padded[m + kw]`` (shifted copy)
      * any RHS the fold pass doesn't decompose

    Downstream passes treat RawStore as opaque: they don't peek at
    ``value``, don't apply cluster/permute rewrites to its indices.
    The lowering pass (mid_ir → plena_ir / HLIR) is responsible for
    pattern-matching specific RawStore shapes (e.g. conv2d's shift
    copy → ``plena.row_load_v_to_fp`` etc.) and erroring on
    unrecognized ones.

    ``value`` is held as an opaque object (typically a tir.PrimExpr
    captured from raw TIR). Mid-IR walkers don't touch it.
    """
    dst: BufferRef
    value: object   # tir.PrimExpr or similar — opaque to mid_ir passes


# ---------------------------------------------------------------------------
# Structure nodes
# ---------------------------------------------------------------------------


class ParallelKind(Enum):
    """Kinds of parallel axes mid-IR represents. Three concepts, never
    interchangeable:

    * ``BLOCK_IDX`` — a HW grid axis. N independent program instances
      run; no lockstep guarantee. Comes from ``T.Kernel(...)``'s grid
      bindings. Has a ``thread_tag`` ("blockIdx.x" / .y / .z).
    * ``LOGICAL_GRID`` — a parallel axis the kernel author marked with
      ``T.Parallel(...)`` that fold couldn't collapse into an
      Elementwise/Reduce. Semantically still N independent instances
      (kernel author asserts iteration order doesn't matter), but
      not bound to a HW grid dim — it's an inner / kernel-body axis.
      No ``thread_tag``. Pass_3 may split a LOGICAL_GRID just like a
      BLOCK_IDX.
    * ``CLUSTER`` — a lockstep multi-lane axis. ``cluster_count`` lanes
      execute the same instruction stream in lockstep, one HW
      instruction = one operation across all lanes. Created by pass_3
      when it splits a (BLOCK_IDX | LOGICAL_GRID) lane axis. Carries
      ``parent_grid_axis_name`` pointing at the matching grid number
      axis.

    Never converted to a For during mid-IR. pass_8_to_plena flattens
    every kind into the appropriate HLIR form.
    """
    BLOCK_IDX = "blockIdx"
    LOGICAL_GRID = "logical_grid"
    CLUSTER = "cluster"


@dataclass
class ParallelAxis:
    """SPMD parallel axis. ``extent`` independent threads execute
    ``body`` concurrently. NOT a sequential loop — pass_8 is the only
    place where parallelism collapses into a for.

    ``axis_name`` is the user-visible name (e.g. ``"by"``,
    ``"by_phase"``, ``"q_block"``).

    ``thread_tag`` carries the underlying ``blockIdx.*`` tag for
    BLOCK_IDX axes only. None for LOGICAL_GRID and CLUSTER.

    ``parent_grid_axis_name`` is the cluster→grid back-link set by
    pass_3 when it splits a lane axis. A CLUSTER axis carries the
    name of the grid-number axis it was split out of (e.g. cluster
    ``by_phase`` has parent ``by_number``). None for grid kinds.
    """
    axis_name: str
    extent: int
    body: List["Stmt"]
    kind: ParallelKind
    thread_tag: Optional[str] = None
    parent_grid_axis_name: Optional[str] = None


@dataclass
class For:
    """A truly-sequential loop. Each iteration runs after the previous.

    Comes from the kernel's ``T.serial(...)`` / ``T.unroll(...)`` —
    e.g. flash_attention's ``for kv_block`` or conv2d's ``for oc / for
    oh``. NEVER carries a ``thread_tag`` and NEVER represents a
    parallel axis (that's ``ParallelAxis``).

    ``kind`` is one of ``"serial"`` (default) or ``"unroll"`` (pass_8
    asks the lowering to fully unroll the loop body — used for tiny
    KW/KH loops in conv2d).
    """
    loop_var: str
    extent: int
    body: List["Stmt"]
    kind: str = "serial"                # "serial" | "unroll"


@dataclass
class Async:
    """Wrap one or more ops in an async region. Pass_5_loop uses Async
    boundaries to *split* an enclosing ``for phase`` into multiple fors
    (each Async ends up in its own for, fused into a multi-lane HW op
    by pass_6_fuse).
    """
    body: List["Stmt"]
    scope_id: int                       # unique per async region


@dataclass
class MultiLaneOp:
    """Output of pass_6_fuse: a single op that the HW executes across
    all cluster lanes in one instruction.

    Multi-axis clusters are supported from day one: a kernel that
    cluster-fuses both ``by`` and ``q_block`` would produce a
    MultiLaneOp with two entries in ``cluster_axis_names`` and
    matching-length lists in ``dim_map`` per buffer.

    Fields:
      * ``inner``               the underlying op (Dma / Gemm /
                                Elementwise / Reduce) with
                                cluster-dim indices pre-resolved.
      * ``cluster_axis_names``  the list of axis names this op fuses
                                across, e.g. ``["by_phase"]`` or
                                ``["by_phase", "qb_phase"]``. Order
                                matches the entries in ``dim_map``.
      * ``dim_map``             ``buf_name -> [dim_idx_for_axis_0,
                                dim_idx_for_axis_1, ...]``. Length of
                                each list matches ``len(cluster_axis_names)``.
                                Pass_7_perm reads this when deciding
                                where each cluster axis sits physically.
    """
    inner: "Op"
    cluster_axis_names: List[str]
    dim_map: Dict[str, List[int]]


# A "statement" is anything that appears in a body list.
Stmt = Union[ParallelAxis, For, Async, Dma, Gemm, Elementwise, Broadcast,
             Reduce, RawStore, MultiLaneOp]

# An "Op" is the leaf-level op kinds (no structure).
Op = Union[Dma, Gemm, Elementwise, Reduce]


# ---------------------------------------------------------------------------
# Function
# ---------------------------------------------------------------------------


@dataclass
class MidFunc:
    """Top-level: a kernel function in mid-IR form.

    ``params`` are param buffers (always global / HBM). ``allocs`` are
    on-chip buffers the kernel allocates. ``body`` is the sequence of
    statements at the kernel's grid scope.

    ``lane_axes`` records the kernel author's
    ``T.func_attr({"plena.lane_axis": ["by"]})`` declaration so pass_3
    knows which blockIdx(es) to split. Multi-axis cluster is supported
    from day one: a kernel can declare ``["by", "q_block"]`` to
    cluster-fuse both. Each axis gets its own cluster_count entry, in
    the same order. ``cluster_counts`` is filled in by pass_3
    (typically = ``[lane_count]`` = ``[MLEN / btmm_hlen]``).

    Mid-IR output preserves blockIdx — see ``For`` docstring. The
    body's outermost layer is typically a chain of ``For(thread_tag=
    "blockIdx.*")`` for both untouched grid axes and the *_number
    halves of split lane axes.
    """
    name: str
    params: List[BufferDef]
    allocs: List[BufferDef]
    body: List[Stmt]
    lane_axes: List[str] = field(default_factory=list)
    cluster_counts: List[int] = field(default_factory=list)
    attrs: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tiny printer (debug only — not stable, format may change freely)
# ---------------------------------------------------------------------------


def _fmt_idx(i: IndexExpr) -> str:
    if isinstance(i, Slice):
        return ":"
    if isinstance(i, dict):
        op = i.get("op", "?")
        args = i.get("args", [])
        return f"({op} {' '.join(_fmt_idx(a) for a in args)})"
    return str(i)


def _fmt_ref(r: BufferRef) -> str:
    body = r.buffer.name
    if r.view_perm is not None and list(r.view_perm) != list(range(len(r.indices))):
        body += f"<view={list(r.view_perm)}>"
    if not r.indices:
        return body
    return f"{body}[{', '.join(_fmt_idx(i) for i in r.indices)}]"


def _fmt_src(s) -> str:
    if isinstance(s, Broadcast):
        return f"bcast({_fmt_ref(s.src)}, dims={s.broadcast_dims})"
    return _fmt_ref(s)


def _fmt_marker(m: Optional[Marker], can_async: bool = False) -> str:
    if m is None:
        return ""
    suffix = " async" if can_async else ""
    return f" #{m.value}{suffix}"


def _print_stmt(s: Stmt, indent: int, out: List[str]) -> None:
    pad = "  " * indent
    if isinstance(s, ParallelAxis):
        # Display kind directly as the keyword:
        #   BLOCK_IDX    → "grid"          (HW grid axis: blockIdx.* binding)
        #   LOGICAL_GRID → "logical_grid"  (kernel-body parallel, no HW binding)
        #   CLUSTER      → "cluster"       (lockstep lane axis)
        # Suffixes:
        #   * BLOCK_IDX shows ``[blockIdx.y]`` (its thread_tag)
        #   * CLUSTER   shows ``← <parent_grid_axis_name>`` so the
        #                cluster→grid back-link is readable at a glance
        if s.kind == ParallelKind.BLOCK_IDX:
            keyword = "grid"
            suffix = f" [{s.thread_tag}]" if s.thread_tag else ""
        elif s.kind == ParallelKind.LOGICAL_GRID:
            keyword = "logical_grid"
            suffix = ""
        else:
            keyword = "cluster"
            suffix = (f" ← {s.parent_grid_axis_name}"
                      if s.parent_grid_axis_name else "")
        out.append(
            f"{pad}{keyword} {s.axis_name} in 0..{s.extent}{suffix}:"
        )
        for b in s.body:
            _print_stmt(b, indent + 1, out)
        return
    if isinstance(s, For):
        kind = "" if s.kind == "serial" else f" ({s.kind})"
        out.append(f"{pad}for {s.loop_var} in 0..{s.extent}{kind}:")
        for b in s.body:
            _print_stmt(b, indent + 1, out)
        return
    if isinstance(s, Async):
        out.append(f"{pad}async #{s.scope_id} {{")
        for b in s.body:
            _print_stmt(b, indent + 1, out)
        out.append(f"{pad}}}")
        return
    if isinstance(s, Dma):
        out.append(
            f"{pad}dma {_fmt_ref(s.src)} -> {_fmt_ref(s.dst)}"
            f"{_fmt_marker(s.marker, s.can_async)}"
        )
        return
    if isinstance(s, Gemm):
        ta = "ᵀ" if s.transpose_a else ""
        tb = "ᵀ" if s.transpose_b else ""
        out.append(
            f"{pad}gemm[{s.kind}] {_fmt_ref(s.c)} = "
            f"{_fmt_ref(s.a)}{ta} @ {_fmt_ref(s.b)}{tb}"
            f"{_fmt_marker(s.marker, s.can_async)}"
        )
        return
    if isinstance(s, Elementwise):
        srcs = ", ".join(_fmt_src(x) for x in s.srcs)
        axis = f" axis={s.axis}" if s.axis is not None else ""
        out.append(
            f"{pad}elementwise[{s.op.value}] {_fmt_ref(s.dst)} = "
            f"f({srcs}){axis}{_fmt_marker(s.marker, s.can_async)}"
        )
        return
    if isinstance(s, Reduce):
        out.append(
            f"{pad}reduce[{s.op.value} axis={s.axis}] "
            f"{_fmt_ref(s.dst)} = R({_fmt_ref(s.src)})"
            f"{_fmt_marker(s.marker, s.can_async)}"
        )
        return
    if isinstance(s, RawStore):
        out.append(f"{pad}raw_store {_fmt_ref(s.dst)} = <opaque>")
        return
    if isinstance(s, MultiLaneOp):
        out.append(
            f"{pad}multi_lane (cluster_axes={s.cluster_axis_names}, "
            f"dim_map={s.dim_map}) {{"
        )
        _print_stmt(s.inner, indent + 1, out)
        out.append(f"{pad}}}")
        return
    out.append(f"{pad}<unknown stmt: {type(s).__name__}>")


def format_func(fn: MidFunc) -> str:
    """Return a multi-line text dump of ``fn`` for eyeballing."""
    out: List[str] = []
    params = ", ".join(
        f"{p.name}: {p.dtype}{tuple(p.shape)} @{p.scope}" for p in fn.params
    )
    out.append(f"func @{fn.name}({params})")
    if fn.lane_axes:
        out.append(f"  // lane_axes = {fn.lane_axes}, "
                   f"cluster_counts = {fn.cluster_counts}")
    if fn.allocs:
        out.append("  allocs:")
        for a in fn.allocs:
            out.append(f"    {a.name}: {a.dtype}{tuple(a.shape)} @{a.scope}")
    out.append("  body:")
    for s in fn.body:
        _print_stmt(s, 2, out)
    return "\n".join(out)


__all__ = [
    "BinOp", "UnaryOp", "ReduceOp", "Marker",
    "BufferDef", "BufferRef", "Slice", "IndexExpr",
    "Elementwise", "Broadcast", "Reduce",
    "Gemm", "Dma", "RawStore",
    "ParallelKind", "ParallelAxis",
    "For", "Async", "MultiLaneOp",
    "MidFunc",
    "format_func",
]
