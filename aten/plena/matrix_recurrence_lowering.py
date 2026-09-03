"""Lower complete linear recurrences through compiler-managed Matrix SRAM.

This module is the model-facing edge of the generic ``L_TILE`` ISA.  It maps
the complete Mamba-2 and KDA decode recurrences to the same three algebraic
primitives while keeping every live recurrent operand in an explicit Matrix
SRAM view.  There is no cache, tag, replacement policy, private state store, or
runtime scheduler.

Three physical controls are emitted from the same formulas:

``fixed_row_pitch``
    A diagnostic reproduction of the old row-pitch-only control.  It is kept
    out of headline comparisons because it unnecessarily forbids the compiler
    from using the column phase already available in a fixed diagonal SRAM.

``fixed``
    The executable single-base fixed-diagonal descriptor.  The compiler chooses
    one allocation base, tile pitch and row chunking.  It needs more chunks to
    describe the complete recurrence and is deliberately not called the
    strongest fixed-hardware bank control.

``fixed-phased D'`` (evaluation control, not another executable layout enum)
    Existing fixed-diagonal wiring with one ordinary compiler-selected column
    base per logical head tile.  It reaches the same conflict-free physical
    bank coordinates as ``affine`` without programmable row/tile skew, but
    needs many base/view bindings instead of one compact descriptor.  D' is the
    fair control for pure bank-conflict claims; ``fixed`` remains the executable
    control for descriptor/chunk/issue comparisons.

``affine``
    The compiler additionally chooses row/tile skew.  Several logical head
    rows then share a physical SRAM row while occupying disjoint banks.  Lane
    order is restored by the Matrix-SRAM view on read.

The default point preserves the paper's approximately 1 MiB Matrix-SRAM bit
budget with a uniform BF16 data path.  It contains 256 MLEN-wide rows and uses
full-width head groups (32 Mamba heads or 16 KDA heads).  The profiled GPU
implementations use FP32 recurrent state; BF16 is therefore an explicit PLENA
mixed-precision design choice whose error must be reported separately.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from enum import StrEnum

from compiler.asm_templates._imm import load_large_int
from compiler.aten.plena.instruction_stream import (
    dynamic_count,
    opcode_census,
    static_count,
)
from compiler.aten.plena.mview import (
    LTilePrimitive,
    MatrixViewAllocation,
    MatrixViewAxis,
    MatrixViewDescriptor,
    MatrixViewFlags,
    MatrixViewMap,
    MatrixViewShape,
    validate_disjoint_matrix_views,
)


PAPER_MLEN = 2048
PAPER_BANKS = 64
PAPER_BANK_WIDTH = 32
BF16_BYTES = 2
ONE_MIB = 1024 * 1024
STATE_DMA_VIEW = 3
STATE_PRECISION_SELECTOR = 2


class RecurrenceLayout(StrEnum):
    FIXED_ROW_PITCH = "fixed_row_pitch"
    FIXED = "fixed"
    AFFINE = "affine"


class RecurrenceKind(StrEnum):
    """Algorithm-independent field contract selected by one recurrence."""

    MAMBA = "mamba"
    KDA = "kda"


@dataclass(frozen=True)
class LoweringRegisters:
    """Five caller-owned GP registers used while emitting ``L_TILE``."""

    destination: int = 9
    source: int = 10
    scale: int = 11
    shape: int = 12
    mapping: int = 13

    def validate(self) -> None:
        values = (
            self.destination,
            self.source,
            self.scale,
            self.shape,
            self.mapping,
        )
        if len(set(values)) != len(values):
            raise ValueError("L_TILE lowering registers must be distinct")
        if any(register < 1 or register > 15 for register in values):
            raise ValueError("L_TILE lowering registers must be GP1..GP15")


@dataclass(frozen=True)
class MatrixSramPoint:
    """Physical Matrix-SRAM capacity used by one recurrence tile."""

    mlen: int = PAPER_MLEN
    banks: int = PAPER_BANKS
    bank_width: int = PAPER_BANK_WIDTH
    capacity_bytes: int = ONE_MIB
    element_bytes: int = BF16_BYTES

    def validate(self) -> None:
        if self.mlen != self.banks * self.bank_width:
            raise ValueError("MLEN must equal banks * bank_width")
        if self.banks < 1 or self.banks & (self.banks - 1):
            raise ValueError("Matrix-SRAM bank count must be a power of two")
        row_bytes = self.mlen * self.element_bytes
        if self.capacity_bytes <= 0 or self.capacity_bytes % row_bytes:
            raise ValueError("Matrix-SRAM capacity must contain whole physical rows")

    @property
    def depth_rows(self) -> int:
        self.validate()
        return self.capacity_bytes // (self.mlen * self.element_bytes)

    @property
    def capacity_bank_words(self) -> int:
        return self.depth_rows * self.banks


@dataclass(frozen=True)
class MatrixRecurrenceSpec:
    name: str
    kind: RecurrenceKind
    heads: int
    row_elements: int
    recurrence_rows: int
    primitives: tuple[LTilePrimitive, ...]

    @property
    def state_bytes_per_head(self) -> int:
        return self.recurrence_rows * self.row_elements * BF16_BYTES

    @property
    def state_bytes_per_layer(self) -> int:
        return self.heads * self.state_bytes_per_head

    def group_heads(self, point: MatrixSramPoint) -> int:
        """Choose the largest group that fits capacity and one MLEN packet."""

        capacity_heads = point.capacity_bytes // (2 * self.state_bytes_per_head)
        packet_heads = point.mlen // self.row_elements
        heads = min(capacity_heads, packet_heads)
        if heads < 1:
            raise ValueError(f"{self.name}: one state head does not fit Matrix SRAM")
        return min(self.heads, heads)

    def validate(self, point: MatrixSramPoint) -> None:
        point.validate()
        if self.heads % self.group_heads(point):
            raise ValueError(
                f"{self.name}: {self.heads} heads do not divide into "
                f"groups of {self.group_heads(point)}"
            )
        if self.row_elements % point.bank_width:
            raise ValueError("one recurrent row must contain whole bank words")


NEMOTRON_MAMBA = MatrixRecurrenceSpec(
    name="nemotron3_mamba2",
    kind=RecurrenceKind.MAMBA,
    heads=64,
    row_elements=64,
    recurrence_rows=128,
    primitives=(
        LTilePrimitive.SCALE_ACCUM,
        LTilePrimitive.SCALE_ACCUM,
        LTilePrimitive.DOT_REDUCE,
        LTilePrimitive.SCALE_ACCUM,
    ),
)

KIMI_KDA = MatrixRecurrenceSpec(
    name="kimi_k3_kda",
    kind=RecurrenceKind.KDA,
    heads=96,
    row_elements=128,
    recurrence_rows=128,
    primitives=(
        LTilePrimitive.SCALE_ACCUM,
        LTilePrimitive.DOT_REDUCE,
        LTilePrimitive.SCALE_ACCUM,
        LTilePrimitive.OUTER_UPDATE,
        LTilePrimitive.DOT_REDUCE,
    ),
)


@dataclass(frozen=True)
class RecurrenceWorkingSet:
    spec: MatrixRecurrenceSpec
    point: MatrixSramPoint
    layout: RecurrenceLayout
    group_heads: int
    state_rows_per_chunk: int
    allocations: tuple[MatrixViewAllocation, ...]
    capacity_facts: dict[str, int]

    @property
    def chunks(self) -> int:
        return self.spec.recurrence_rows // self.state_rows_per_chunk

    @property
    def groups(self) -> int:
        return self.spec.heads // self.group_heads

    @property
    def packet_values(self) -> int:
        return self.group_heads * self.spec.row_elements

    def allocation(self, name: str) -> MatrixViewAllocation:
        for allocation in self.allocations:
            if allocation.name == name:
                return allocation
        raise KeyError(name)

    def to_dict(self) -> dict[str, object]:
        return {
            "layout": self.layout,
            "group_heads": self.group_heads,
            "groups": self.groups,
            "state_rows_per_chunk": self.state_rows_per_chunk,
            "chunks": self.chunks,
            "packet_values": self.packet_values,
            "vector_lane_utilization": self.packet_values / self.point.mlen,
            "capacity_bytes": self.point.capacity_bytes,
            "depth_rows": self.point.depth_rows,
            "capacity_facts": self.capacity_facts,
            "allocations": [
                {
                    "name": allocation.name,
                    "base": allocation.base,
                    "shape": asdict(allocation.descriptor.shape),
                    "mapping": {
                        "tile_pitch_rows": allocation.descriptor.mapping.tile_pitch_rows,
                        "row_skew": allocation.descriptor.mapping.row_skew,
                        "tile_skew": allocation.descriptor.mapping.tile_skew,
                        "flags": int(allocation.descriptor.mapping.flags),
                    },
                }
                for allocation in self.allocations
            ],
        }


@dataclass(frozen=True)
class RecurrenceFieldPacket:
    """One compiler-visible prepared recurrence operand in HBM.

    This is a functional interface for the standalone Matrix-SRAM recurrence
    program.  It is deliberately not a hidden descriptor fetch: the compiler
    emits an ordinary viewed DMA for every packet.  ``logical_values`` is the
    tensor payload consumed by the view; ``transfer_values`` includes the VLEN
    tail that the existing HBM interface reads for a short packet.
    """

    field: str
    target: str
    group: int
    chunk: int | None
    hbm_byte_offset: int
    logical_values: int
    transfer_values: int

    @property
    def storage_bytes(self) -> int:
        return self.transfer_values * BF16_BYTES

    @property
    def key(self) -> tuple[str, int, int | None]:
        return (self.field, self.group, self.chunk)

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {"storage_bytes": self.storage_bytes}


@dataclass(frozen=True)
class RecurrenceFieldManifest:
    """Static HBM placement for all non-state operands of one recurrence."""

    base: int
    end: int
    packets: tuple[RecurrenceFieldPacket, ...]

    def packet(
        self,
        field: str,
        *,
        group: int,
        chunk: int | None = None,
    ) -> RecurrenceFieldPacket:
        key = (field, group, chunk)
        matches = [packet for packet in self.packets if packet.key == key]
        if len(matches) != 1:
            raise KeyError(f"expected one recurrence field packet {key}, found {len(matches)}")
        return matches[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "base": self.base,
            "end": self.end,
            "storage_bytes": self.end - self.base,
            "packets": [packet.to_dict() for packet in self.packets],
        }


def _addr(point: MatrixSramPoint, row: int, bank_phase: int) -> int:
    if not 0 <= bank_phase < point.banks:
        raise ValueError(f"bank phase {bank_phase} is outside the Matrix SRAM")
    return row * point.mlen + bank_phase * point.bank_width


def _descriptor(
    *,
    rows: int,
    cols: int,
    tiles: int,
    pitch: int,
    affine: bool,
    row_skew: int = 0,
    tile_skew: int = 0,
    broadcast: bool = False,
) -> MatrixViewDescriptor:
    flags = MatrixViewFlags.STRICT_BOUNDS
    if affine:
        flags |= MatrixViewFlags.AFFINE
    if broadcast:
        flags |= MatrixViewFlags.BROADCAST_MINOR
    return MatrixViewDescriptor(
        MatrixViewShape(rows=rows, cols=cols, tile_count=tiles),
        MatrixViewMap(
            tile_pitch_rows=pitch,
            row_skew=row_skew,
            tile_skew=tile_skew,
            flags=flags,
        ),
    )


def build_recurrence_working_set(
    spec: MatrixRecurrenceSpec,
    *,
    layout: RecurrenceLayout | str,
    point: MatrixSramPoint | None = None,
) -> RecurrenceWorkingSet:
    """Build and statically prove one same-capacity recurrence allocation."""

    point = point or MatrixSramPoint()
    layout = RecurrenceLayout(layout)
    spec.validate(point)
    group_heads = spec.group_heads(point)
    # Both controls start from the same maximal head group.  Fixed wiring may
    # need shorter state-row chunks, but silently shrinking only its head group
    # would give the affine treatment an extra scheduling freedom and make the
    # comparison unfair.
    words_per_row = spec.row_elements // point.bank_width
    # One scale packet carries either one scalar per head (DOT/OUTER) or one
    # [a,b] pair per head (SCALE_ACCUM).  It is read one cycle before the state
    # packet and segment-broadcast across each logical state row.  Replicating
    # one scalar into an entire bank word per head would waste 16-32x capacity
    # and model traffic that the architecture never needs.
    scalar_cols = max(
        point.bank_width,
        ((2 * group_heads + point.bank_width - 1) // point.bank_width)
        * point.bank_width,
    )
    affine = layout is RecurrenceLayout.AFFINE
    if affine:
        state_rows = spec.recurrence_rows
        state = _descriptor(
            rows=state_rows,
            cols=spec.row_elements,
            tiles=group_heads,
            pitch=0,
            affine=True,
            row_skew=1,
            tile_skew=words_per_row,
        )
        scalar = _descriptor(
            rows=state_rows,
            cols=scalar_cols,
            tiles=1,
            pitch=state_rows,
            affine=True,
            row_skew=1,
            tile_skew=1,
            broadcast=True,
        )
        vector = _descriptor(
            rows=1,
            cols=spec.row_elements,
            tiles=group_heads,
            # A one-row temporary does not need state-style row sharing.  The
            # existing diagonal map reaches the packet floor when consecutive
            # head tiles are spaced by one logical row width.  Pitch 1 would
            # overlap adjacent heads by all but one bank word (2--4x service).
            pitch=words_per_row,
            affine=True,
            row_skew=1,
        )
        occupied_state_banks = group_heads * words_per_row
        # At the 1 MiB point the head-group state occupies half of every
        # physical row, so the live fields use the remaining bank phases.  At
        # the 2 MiB point the larger group occupies all 64 banks; place fields
        # in the second 128-row half instead of inventing a non-existent bank
        # phase 64.  Both are compiler-owned scratchpad allocations.
        fields_base_row = state_rows if occupied_state_banks == point.banks else 0
        fields_first_bank = 0 if fields_base_row else occupied_state_banks
        if spec.kind is RecurrenceKind.KDA:
            head_pair = _descriptor(
                rows=2,
                cols=state_rows,
                tiles=group_heads,
                # Decay stores two logical rows (a/b) per head.  Physical rows
                # remain eight rows apart so all 2x4 words are disjoint.  A
                # four-bank tile phase makes both the complete viewed DMA
                # (128 words, two per bank) and each 32-word column packet
                # conflict-free for the BF16 16-head group.
                pitch=2 * words_per_row,
                affine=True,
                row_skew=1,
                tile_skew=words_per_row,
                broadcast=True,
            )
            head_scalar = _descriptor(
                rows=1,
                cols=state_rows,
                tiles=group_heads,
                # One 128-value row occupies four bank words.  The matching
                # four-row tile pitch makes adjacent heads occupy disjoint
                # four-bank slices in a full viewed DMA.
                pitch=words_per_row,
                affine=True,
                row_skew=1,
                broadcast=True,
            )
            phases = {
                "state": 0,
                # Decay occupies both logical scalar rows and therefore uses
                # the opposite half-bank phase at both supported capacities.
                # Combined with head_pair's affine coefficients this reaches
                # the theoretical one-port service floor without aliasing the
                # resident state or the other live fields.
                "decay": point.banks // 2,
                "key": fields_first_bank + words_per_row,
                "query": fields_first_bank + 2 * words_per_row,
                "beta": fields_first_bank + 3 * words_per_row,
                "error": fields_first_bank + 3 * words_per_row + 1,
                "prediction": fields_first_bank + 4 * words_per_row + 1,
                "output": fields_first_bank + 5 * words_per_row + 1,
            }
            base_rows = dict.fromkeys(phases, fields_base_row)
            base_rows["state"] = 0
            descriptors = {
                "state": state,
                "decay": head_pair,
                "key": head_scalar,
                "query": head_scalar,
                "beta": _shape_rows(scalar, 1),
                "error": vector,
                "prediction": vector,
                "output": vector,
            }
        else:
            scalar_words = scalar_cols // point.bank_width
            phases = {
                "state": 0,
                "dt": fields_first_bank,
                "update": fields_first_bank + scalar_words,
                "c": fields_first_bank + 2 * scalar_words,
                "d": fields_first_bank + 3 * scalar_words,
                "x": fields_first_bank + 4 * scalar_words,
                "scratch": fields_first_bank + 4 * scalar_words + words_per_row,
                "output": fields_first_bank + 4 * scalar_words + 2 * words_per_row,
            }
            base_rows = dict.fromkeys(phases, fields_base_row)
            base_rows["state"] = 0
            descriptors = {
                "state": state,
                "dt": _shape_rows(scalar, 1),
                "update": scalar,
                "c": scalar,
                "d": _shape_rows(scalar, 1),
                "x": vector,
                "scratch": vector,
                "output": vector,
            }
    else:
        # This executable fixed control has one base and one descriptor. A tile
        # spans ``words_per_row`` physical rows, so more recurrence chunks are
        # required because this descriptor cannot express a distinct ordinary
        # column base for every head tile. The fair bank-only D' control gives
        # each tile that base freedom and is evaluated separately; it must not
        # be confused with this compact-ISA control.
        state_rows = (
            words_per_row
            if layout is RecurrenceLayout.FIXED
            else point.depth_rows // group_heads
        )
        if state_rows < 1 or spec.recurrence_rows % state_rows:
            raise ValueError(f"{spec.name}: fixed state chunks do not divide recurrence rows")
        state = _descriptor(
            rows=state_rows,
            cols=spec.row_elements,
            tiles=group_heads,
            pitch=state_rows,
            affine=False,
        )
        scalar = _descriptor(
            rows=state_rows,
            cols=scalar_cols,
            tiles=1,
            pitch=state_rows,
            affine=False,
            broadcast=True,
        )
        vector = _descriptor(
            rows=1,
            cols=spec.row_elements,
            tiles=group_heads,
            pitch=state_rows,
            affine=False,
        )
        if spec.kind is RecurrenceKind.KDA:
            padded_key_cols = max(point.bank_width, state_rows)
            head_pair = _descriptor(
                rows=2,
                cols=padded_key_cols,
                tiles=group_heads,
                pitch=2,
                affine=False,
                broadcast=True,
            )
            head_scalar = _descriptor(
                rows=1,
                cols=padded_key_cols,
                tiles=group_heads,
                pitch=1,
                affine=False,
                broadcast=True,
            )
            phases = {
                "state": 0,
                "decay": words_per_row,
                "key": 2 * words_per_row,
                "query": 3 * words_per_row,
                "beta": 4 * words_per_row,
                "error": 4 * words_per_row + 1,
                "prediction": 5 * words_per_row + 1,
                "output": 6 * words_per_row + 1,
            }
            descriptors = {
                "state": state,
                "decay": head_pair,
                "key": head_scalar,
                "query": head_scalar,
                "beta": _shape_rows(scalar, 1),
                "error": vector,
                "prediction": vector,
                "output": vector,
            }
        else:
            scalar_words = scalar_cols // point.bank_width
            phases = {
                "state": 0,
                "dt": words_per_row,
                "update": words_per_row + scalar_words,
                "c": words_per_row + 2 * scalar_words,
                "d": words_per_row + 3 * scalar_words,
                "x": words_per_row + 4 * scalar_words,
                "scratch": 2 * words_per_row + 4 * scalar_words,
                "output": 3 * words_per_row + 4 * scalar_words,
            }
            descriptors = {
                "state": state,
                "dt": _shape_rows(scalar, 1),
                "update": scalar,
                "c": scalar,
                "d": _shape_rows(scalar, 1),
                "x": vector,
                "scratch": vector,
                "output": vector,
            }
        base_rows = dict.fromkeys(descriptors, 0)

    allocations = tuple(
        MatrixViewAllocation(
            name,
            _addr(point, base_rows[name], phases[name]),
            descriptor,
        )
        for name, descriptor in descriptors.items()
    )
    facts = validate_disjoint_matrix_views(
        allocations,
        mlen=point.mlen,
        banks=point.banks,
        bank_width=point.bank_width,
        depth_rows=point.depth_rows,
    )
    return RecurrenceWorkingSet(
        spec=spec,
        point=point,
        layout=layout,
        group_heads=group_heads,
        state_rows_per_chunk=state_rows,
        allocations=allocations,
        capacity_facts=facts,
    )


def _shape_rows(descriptor: MatrixViewDescriptor, rows: int) -> MatrixViewDescriptor:
    return MatrixViewDescriptor(
        MatrixViewShape(
            rows=rows,
            cols=descriptor.shape.cols,
            tile_count=descriptor.shape.tile_count,
        ),
        descriptor.mapping,
    )


def _round_up(value: int, multiple: int) -> int:
    if multiple <= 0:
        raise ValueError("rounding multiple must be positive")
    return ((value + multiple - 1) // multiple) * multiple


def build_recurrence_field_manifest(
    working_set: RecurrenceWorkingSet,
    *,
    field_hbm_base: int,
) -> RecurrenceFieldManifest:
    """Place every prepared non-state operand in a disjoint BF16 HBM packet.

    The standalone recurrence consumes post-projection/post-convolution values,
    so decay/exponential and compact scale pairs are already prepared.  The
    integrated model lowering may replace these DMAs with direct Matrix/Vector
    producer writeback, but it must preserve this exact logical field contract.
    """

    if field_hbm_base < 0 or field_hbm_base % 64:
        raise ValueError("field_hbm_base must be a non-negative 64-byte address")
    packets: list[RecurrenceFieldPacket] = []
    cursor = field_hbm_base

    def add(field: str, target: str, group: int, chunk: int | None = None) -> None:
        nonlocal cursor
        descriptor = working_set.allocation(target).descriptor
        logical_values = (
            descriptor.shape.rows
            * descriptor.shape.cols
            * descriptor.shape.tile_count
        )
        transfer_values = _round_up(logical_values, working_set.point.mlen)
        packets.append(
            RecurrenceFieldPacket(
                field=field,
                target=target,
                group=group,
                chunk=chunk,
                hbm_byte_offset=cursor,
                logical_values=logical_values,
                transfer_values=transfer_values,
            )
        )
        cursor += transfer_values * BF16_BYTES

    for group in range(working_set.groups):
        if working_set.spec.kind is RecurrenceKind.MAMBA:
            for field, target in (
                ("x", "x"),
                ("scratch_zero", "scratch"),
                ("dt", "dt"),
                ("output_zero", "output"),
                ("d", "d"),
            ):
                add(field, target, group)
            for chunk in range(working_set.chunks):
                add("update", "update", group, chunk)
                add("c", "c", group, chunk)
            add("output_result", "output", group)
        elif working_set.spec.kind is RecurrenceKind.KDA:
            for field, target in (
                ("prediction_zero", "prediction"),
                ("value", "error"),
                ("beta", "beta"),
                ("output_zero", "output"),
            ):
                add(field, target, group)
            for chunk in range(working_set.chunks):
                add("decay", "decay", group, chunk)
                add("key", "key", group, chunk)
                add("query", "query", group, chunk)
            add("output_result", "output", group)
        else:  # pragma: no cover - all public specs are enumerated above
            raise ValueError(f"no recurrence-field ABI for {working_set.spec.name}")

    return RecurrenceFieldManifest(
        base=field_hbm_base,
        end=cursor,
        packets=tuple(packets),
    )


def _configure_view(
    *,
    slot: int,
    descriptor: MatrixViewDescriptor,
    shape_register: int,
    map_register: int,
) -> list[str]:
    return [
        *load_large_int(shape_register, descriptor.shape.pack()),
        *load_large_int(map_register, descriptor.mapping.pack()),
        f"L_TILE_CFG {slot}, gp{shape_register}, gp{map_register}",
    ]


def _emit_exec(
    lines: list[str],
    *,
    dst: MatrixViewAllocation,
    src: MatrixViewAllocation,
    scales: MatrixViewAllocation,
    primitive: LTilePrimitive,
    rows: int | None = None,
    source_axis: MatrixViewAxis = MatrixViewAxis.ROW,
    scale_axis: MatrixViewAxis = MatrixViewAxis.ROW,
    label: str,
    registers: LoweringRegisters,
) -> None:
    dst_descriptor = dst.descriptor if rows is None else _shape_rows(dst.descriptor, rows)
    src_rows = rows if source_axis is MatrixViewAxis.ROW else None
    if source_axis is MatrixViewAxis.ROW and src.descriptor.shape.rows == 1:
        src_rows = 1
    src_descriptor = (
        src.descriptor if src_rows is None else _shape_rows(src.descriptor, src_rows)
    )
    scale_rows = rows if scale_axis is MatrixViewAxis.ROW else None
    scale_descriptor = (
        scales.descriptor
        if scale_rows is None
        else _shape_rows(scales.descriptor, scale_rows)
    )
    lines.append(f"; @l_tile_step={label}")
    lines.extend(
        _configure_view(
            slot=0,
            descriptor=dst_descriptor,
            shape_register=registers.shape,
            map_register=registers.mapping,
        )
    )
    lines.extend(
        _configure_view(
            slot=1,
            descriptor=src_descriptor,
            shape_register=registers.shape,
            map_register=registers.mapping,
        )
    )
    lines.extend(
        _configure_view(
            slot=2,
            descriptor=scale_descriptor,
            shape_register=registers.shape,
            map_register=registers.mapping,
        )
    )
    lines.extend(load_large_int(registers.destination, dst.base))
    lines.extend(load_large_int(registers.source, src.base))
    lines.extend(load_large_int(registers.scale, scales.base))
    axis_mask = int(source_axis) | int(scale_axis) << 1
    suffix = "" if axis_mask == 0 else f", {axis_mask}"
    lines.append(
        f"L_TILE_EXEC gp{registers.destination}, gp{registers.source}, "
        f"gp{registers.scale}, {int(primitive)}{suffix}"
    )


def _emit_state_transfer(
    lines: list[str],
    *,
    working_set: RecurrenceWorkingSet,
    state: MatrixViewAllocation,
    direction: str,
    group: int,
    chunk: int,
    rows: int,
    values: int,
    state_hbm_base: int,
    hbm_address_register: int,
    registers: LoweringRegisters,
) -> None:
    if direction not in {"load", "reload_intermediate", "store", "store_intermediate"}:
        raise ValueError(f"unsupported Matrix state transfer direction {direction!r}")
    if not 0 <= hbm_address_register <= 15:
        raise ValueError("HBM address register must be a0..a15")
    expected_values = working_set.group_heads * rows * working_set.spec.row_elements
    if values != expected_values:
        raise ValueError(
            f"state packet has {values} values, expected {expected_values}"
        )

    # Persistent state is laid out packet-major in HBM:
    # [head-group][state-row chunk][head-in-group][row][lane].  This makes one
    # explicit DMA packet contiguous without a hidden gather engine.  Fixed and
    # affine variants move the same number of BF16 values; only their on-chip
    # bank placement differs.
    packet = group * working_set.chunks + chunk
    hbm_byte_offset = state_hbm_base + packet * values * BF16_BYTES
    descriptor = _shape_rows(state.descriptor, rows)
    lines.append(
        f"; @matrix_state_{direction} group={group} chunk={chunk} "
        f"rows={rows} values={values} precision=bf16 hbm_byte_offset={hbm_byte_offset}"
    )
    lines.extend(
        _configure_view(
            slot=STATE_DMA_VIEW,
            descriptor=descriptor,
            shape_register=registers.shape,
            map_register=registers.mapping,
        )
    )
    lines.extend(load_large_int(registers.destination, state.base))
    lines.extend(load_large_int(registers.source, hbm_byte_offset))
    opcode = (
        "H_PREFETCH_V.MV"
        if direction in {"load", "reload_intermediate"}
        else "H_STORE_V.MV"
    )
    lines.append(
        f"{opcode} gp{registers.destination}, gp{registers.source}, "
        f"a{hbm_address_register}, 0, {STATE_PRECISION_SELECTOR}, {STATE_DMA_VIEW}"
    )


def _emit_field_load(
    lines: list[str],
    *,
    working_set: RecurrenceWorkingSet,
    manifest: RecurrenceFieldManifest,
    field: str,
    group: int,
    chunk: int | None,
    hbm_address_register: int,
    registers: LoweringRegisters,
) -> None:
    """Emit a real viewed DMA for one prepared recurrence operand."""

    packet = manifest.packet(field, group=group, chunk=chunk)
    target = working_set.allocation(packet.target)
    descriptor = target.descriptor
    logical_values = (
        descriptor.shape.rows
        * descriptor.shape.cols
        * descriptor.shape.tile_count
    )
    if logical_values != packet.logical_values:
        raise AssertionError(
            f"field {packet.key} changed shape after its HBM manifest was built"
        )
    chunk_text = "none" if chunk is None else str(chunk)
    lines.append(
        f"; @matrix_field_load field={field} target={packet.target} "
        f"group={group} chunk={chunk_text} logical_values={packet.logical_values} "
        f"transfer_values={packet.transfer_values} precision=bf16 "
        f"hbm_byte_offset={packet.hbm_byte_offset}"
    )
    lines.extend(
        _configure_view(
            slot=STATE_DMA_VIEW,
            descriptor=descriptor,
            shape_register=registers.shape,
            map_register=registers.mapping,
        )
    )
    lines.extend(load_large_int(registers.destination, target.base))
    lines.extend(load_large_int(registers.source, packet.hbm_byte_offset))
    lines.append(
        f"H_PREFETCH_V.MV gp{registers.destination}, gp{registers.source}, "
        f"a{hbm_address_register}, 0, {STATE_PRECISION_SELECTOR}, {STATE_DMA_VIEW}"
    )


def _emit_field_store(
    lines: list[str],
    *,
    working_set: RecurrenceWorkingSet,
    manifest: RecurrenceFieldManifest,
    field: str,
    group: int,
    chunk: int | None,
    hbm_address_register: int,
    registers: LoweringRegisters,
) -> None:
    """Store one produced Matrix view before its scratch allocation is reused."""

    packet = manifest.packet(field, group=group, chunk=chunk)
    source = working_set.allocation(packet.target)
    descriptor = source.descriptor
    chunk_text = "none" if chunk is None else str(chunk)
    lines.append(
        f"; @matrix_field_store field={field} source={packet.target} "
        f"group={group} chunk={chunk_text} logical_values={packet.logical_values} "
        f"transfer_values={packet.transfer_values} precision=bf16 "
        f"hbm_byte_offset={packet.hbm_byte_offset}"
    )
    lines.extend(
        _configure_view(
            slot=STATE_DMA_VIEW,
            descriptor=descriptor,
            shape_register=registers.shape,
            map_register=registers.mapping,
        )
    )
    lines.extend(load_large_int(registers.destination, source.base))
    lines.extend(load_large_int(registers.source, packet.hbm_byte_offset))
    lines.append(
        f"H_STORE_V.MV gp{registers.destination}, gp{registers.source}, "
        f"a{hbm_address_register}, 0, {STATE_PRECISION_SELECTOR}, {STATE_DMA_VIEW}"
    )


def _lower_mamba(
    working_set: RecurrenceWorkingSet,
    field_manifest: RecurrenceFieldManifest,
    registers: LoweringRegisters,
    *,
    state_hbm_base: int,
    hbm_address_register: int,
) -> list[str]:
    lines: list[str] = []
    state = working_set.allocation("state")
    dt = working_set.allocation("dt")
    update = working_set.allocation("update")
    c = working_set.allocation("c")
    d = working_set.allocation("d")
    x = working_set.allocation("x")
    scratch = working_set.allocation("scratch")
    output = working_set.allocation("output")

    for group in range(working_set.groups):
        lines.append(f"; @head_group={group}/{working_set.groups}")
        for field in ("x", "scratch_zero", "dt", "output_zero"):
            _emit_field_load(
                lines,
                working_set=working_set,
                manifest=field_manifest,
                field=field,
                group=group,
                chunk=None,
                hbm_address_register=hbm_address_register,
                registers=registers,
            )
        _emit_exec(
            lines,
            dst=scratch,
            src=x,
            scales=dt,
            primitive=LTilePrimitive.SCALE_ACCUM,
            rows=1,
            label="mamba_dt_times_x",
            registers=registers,
        )
        for chunk in range(working_set.chunks):
            values = (
                working_set.group_heads
                * working_set.state_rows_per_chunk
                * working_set.spec.row_elements
            )
            _emit_state_transfer(
                lines,
                working_set=working_set,
                state=state,
                direction="load",
                group=group,
                chunk=chunk,
                rows=working_set.state_rows_per_chunk,
                values=values,
                state_hbm_base=state_hbm_base,
                hbm_address_register=hbm_address_register,
                registers=registers,
            )
            for field in ("update", "c"):
                _emit_field_load(
                    lines,
                    working_set=working_set,
                    manifest=field_manifest,
                    field=field,
                    group=group,
                    chunk=chunk,
                    hbm_address_register=hbm_address_register,
                    registers=registers,
                )
            _emit_exec(
                lines,
                dst=state,
                src=scratch,
                scales=update,
                primitive=LTilePrimitive.SCALE_ACCUM,
                label="mamba_state_decay_rank1_update",
                registers=registers,
            )
            _emit_exec(
                lines,
                dst=output,
                src=state,
                scales=c,
                primitive=LTilePrimitive.DOT_REDUCE,
                label="mamba_c_readout",
                registers=registers,
            )
            _emit_state_transfer(
                lines,
                working_set=working_set,
                state=state,
                direction="store",
                group=group,
                chunk=chunk,
                rows=working_set.state_rows_per_chunk,
                values=values,
                state_hbm_base=state_hbm_base,
                hbm_address_register=hbm_address_register,
                registers=registers,
            )
        _emit_field_load(
            lines,
            working_set=working_set,
            manifest=field_manifest,
            field="d",
            group=group,
            chunk=None,
            hbm_address_register=hbm_address_register,
            registers=registers,
        )
        _emit_exec(
            lines,
            dst=output,
            src=x,
            scales=d,
            primitive=LTilePrimitive.SCALE_ACCUM,
            rows=1,
            label="mamba_skip",
            registers=registers,
        )
        _emit_field_store(
            lines,
            working_set=working_set,
            manifest=field_manifest,
            field="output_result",
            group=group,
            chunk=None,
            hbm_address_register=hbm_address_register,
            registers=registers,
        )
    return lines


def _lower_kda(
    working_set: RecurrenceWorkingSet,
    field_manifest: RecurrenceFieldManifest,
    registers: LoweringRegisters,
    *,
    state_hbm_base: int,
    hbm_address_register: int,
) -> list[str]:
    lines: list[str] = []
    state = working_set.allocation("state")
    decay = working_set.allocation("decay")
    key = working_set.allocation("key")
    query = working_set.allocation("query")
    beta = working_set.allocation("beta")
    error = working_set.allocation("error")
    prediction = working_set.allocation("prediction")
    output = working_set.allocation("output")

    for group in range(working_set.groups):
        lines.append(f"; @head_group={group}/{working_set.groups}")
        _emit_field_load(
            lines,
            working_set=working_set,
            manifest=field_manifest,
            field="prediction_zero",
            group=group,
            chunk=None,
            hbm_address_register=hbm_address_register,
            registers=registers,
        )
        # Fixed wiring cannot hold a complete head-group state and its fields;
        # retain decayed chunks explicitly between the predict and update passes.
        for chunk in range(working_set.chunks):
            values = (
                working_set.group_heads
                * working_set.state_rows_per_chunk
                * working_set.spec.row_elements
            )
            _emit_state_transfer(
                lines,
                working_set=working_set,
                state=state,
                direction="load",
                group=group,
                chunk=chunk,
                rows=working_set.state_rows_per_chunk,
                values=values,
                state_hbm_base=state_hbm_base,
                hbm_address_register=hbm_address_register,
                registers=registers,
            )
            for field in ("decay", "key"):
                _emit_field_load(
                    lines,
                    working_set=working_set,
                    manifest=field_manifest,
                    field=field,
                    group=group,
                    chunk=chunk,
                    hbm_address_register=hbm_address_register,
                    registers=registers,
                )
            _emit_exec(
                lines,
                dst=state,
                src=prediction,
                scales=decay,
                primitive=LTilePrimitive.SCALE_ACCUM,
                scale_axis=MatrixViewAxis.COLUMN,
                label="kda_decay",
                registers=registers,
            )
            _emit_exec(
                lines,
                dst=prediction,
                src=state,
                scales=key,
                primitive=LTilePrimitive.DOT_REDUCE,
                scale_axis=MatrixViewAxis.COLUMN,
                label="kda_prediction",
                registers=registers,
            )
            if working_set.chunks > 1:
                _emit_state_transfer(
                    lines,
                    working_set=working_set,
                    state=state,
                    direction="store_intermediate",
                    group=group,
                    chunk=chunk,
                    rows=working_set.state_rows_per_chunk,
                    values=values,
                    state_hbm_base=state_hbm_base,
                    hbm_address_register=hbm_address_register,
                    registers=registers,
                )
        for field in ("value", "beta", "output_zero"):
            _emit_field_load(
                lines,
                working_set=working_set,
                manifest=field_manifest,
                field=field,
                group=group,
                chunk=None,
                hbm_address_register=hbm_address_register,
                registers=registers,
            )
        _emit_exec(
            lines,
            dst=error,
            src=prediction,
            scales=beta,
            primitive=LTilePrimitive.SCALE_ACCUM,
            rows=1,
            label="kda_beta_error",
            registers=registers,
        )
        for chunk in range(working_set.chunks):
            values = (
                working_set.group_heads
                * working_set.state_rows_per_chunk
                * working_set.spec.row_elements
            )
            if working_set.chunks > 1:
                _emit_state_transfer(
                    lines,
                    working_set=working_set,
                    state=state,
                    direction="reload_intermediate",
                    group=group,
                    chunk=chunk,
                    rows=working_set.state_rows_per_chunk,
                    values=values,
                    state_hbm_base=state_hbm_base,
                    hbm_address_register=hbm_address_register,
                    registers=registers,
                )
            if working_set.chunks > 1:
                _emit_field_load(
                    lines,
                    working_set=working_set,
                    manifest=field_manifest,
                    field="key",
                    group=group,
                    chunk=chunk,
                    hbm_address_register=hbm_address_register,
                    registers=registers,
                )
            _emit_field_load(
                lines,
                working_set=working_set,
                manifest=field_manifest,
                field="query",
                group=group,
                chunk=chunk,
                hbm_address_register=hbm_address_register,
                registers=registers,
            )
            _emit_exec(
                lines,
                dst=state,
                src=error,
                scales=key,
                primitive=LTilePrimitive.OUTER_UPDATE,
                scale_axis=MatrixViewAxis.COLUMN,
                label="kda_rank1_update",
                registers=registers,
            )
            _emit_exec(
                lines,
                dst=output,
                src=state,
                scales=query,
                primitive=LTilePrimitive.DOT_REDUCE,
                scale_axis=MatrixViewAxis.COLUMN,
                label="kda_readout",
                registers=registers,
            )
            _emit_state_transfer(
                lines,
                working_set=working_set,
                state=state,
                direction="store",
                group=group,
                chunk=chunk,
                rows=working_set.state_rows_per_chunk,
                values=values,
                state_hbm_base=state_hbm_base,
                hbm_address_register=hbm_address_register,
                registers=registers,
            )
        _emit_field_store(
            lines,
            working_set=working_set,
            manifest=field_manifest,
            field="output_result",
            group=group,
            chunk=None,
            hbm_address_register=hbm_address_register,
            registers=registers,
        )
    return lines


def lower_matrix_recurrence(
    spec: MatrixRecurrenceSpec,
    *,
    co_layout: bool | None = None,
    layout: RecurrenceLayout | str | None = None,
    point: MatrixSramPoint | None = None,
    mlen: int | None = None,
    blen: int | None = None,
    registers: LoweringRegisters | None = None,
    state_hbm_base: int = 0,
    field_hbm_base: int | None = None,
    hbm_address_register: int = 0,
) -> str:
    """Emit one official-shape layer's complete recurrent Matrix-SRAM path.

    ``co_layout`` is retained as a compatibility spelling for existing callers;
    new code should pass ``layout``.  ``mlen``/``blen`` may only restate the
    physical point and cannot silently change its capacity.
    """

    if layout is None:
        layout = RecurrenceLayout.AFFINE if co_layout else RecurrenceLayout.FIXED
    elif co_layout is not None and (RecurrenceLayout(layout) is RecurrenceLayout.AFFINE) != co_layout:
        raise ValueError("co_layout and layout select different mechanisms")
    point = point or MatrixSramPoint()
    if mlen is not None and mlen != point.mlen:
        raise ValueError("mlen must match the explicit Matrix-SRAM point")
    if blen is not None and blen != point.bank_width:
        raise ValueError("blen must match the explicit Matrix-SRAM bank width")
    registers = registers or LoweringRegisters()
    registers.validate()
    if state_hbm_base < 0:
        raise ValueError("state_hbm_base must be a non-negative byte address")
    if state_hbm_base % 64:
        raise ValueError("state_hbm_base must be 64-byte aligned")
    if not 0 <= hbm_address_register <= 15:
        raise ValueError("hbm_address_register must be in [0, 15]")
    working_set = build_recurrence_working_set(spec, layout=layout, point=point)
    if field_hbm_base is None:
        field_hbm_base = _round_up(
            state_hbm_base + spec.state_bytes_per_layer,
            64,
        )
    field_manifest = build_recurrence_field_manifest(
        working_set,
        field_hbm_base=field_hbm_base,
    )
    lines = [
        f"; @stage={spec.name}_matrix_recurrence",
        f"; @layout={working_set.layout}",
        f"; @matrix_sram_bytes={point.capacity_bytes}",
        f"; @group_heads={working_set.group_heads}",
        f"; @state_rows_per_chunk={working_set.state_rows_per_chunk}",
        "; @state_precision=bf16",
        "; @state_storage=compiler_managed_matrix_sram_no_cache",
        "; @field_contract=prepared_bf16_post_projection_post_conv",
        f"; @field_hbm_base={field_manifest.base}",
        f"; @field_hbm_end={field_manifest.end}",
    ]
    lines.extend(
        _lower_mamba(
            working_set,
            field_manifest,
            registers,
            state_hbm_base=state_hbm_base,
            hbm_address_register=hbm_address_register,
        )
        if spec.kind is RecurrenceKind.MAMBA
        else _lower_kda(
            working_set,
            field_manifest,
            registers,
            state_hbm_base=state_hbm_base,
            hbm_address_register=hbm_address_register,
        )
    )
    return "\n".join(lines) + "\n"


def lowering_metrics(assembly: str) -> dict[str, object]:
    census = opcode_census(assembly)
    primitive_census: Counter[str] = Counter()
    transfers: Counter[str] = Counter()
    transfer_values_by_direction: Counter[str] = Counter()
    transfer_values = 0
    field_transfers: Counter[str] = Counter()
    field_stores: Counter[str] = Counter()
    field_logical_values = 0
    field_transfer_values = 0
    for raw in assembly.splitlines():
        line = raw.strip()
        if line.startswith("L_TILE_EXEC"):
            operands = [operand.strip() for operand in line.split(None, 1)[1].split(",")]
            primitive = int(operands[3], 0)
            primitive_census[LTilePrimitive(primitive).name] += 1
        if line.startswith("; @matrix_state_"):
            direction = line.split()[1].removeprefix("@matrix_state_")
            transfers[direction] += 1
            for token in line.split():
                if token.startswith("values="):
                    values = int(token.split("=", 1)[1])
                    transfer_values += values
                    transfer_values_by_direction[direction] += values
        if line.startswith("; @matrix_field_load"):
            tokens = {
                token.split("=", 1)[0]: token.split("=", 1)[1]
                for token in line.split()
                if "=" in token
            }
            field_transfers[tokens["field"]] += 1
            field_logical_values += int(tokens["logical_values"])
            field_transfer_values += int(tokens["transfer_values"])
        if line.startswith("; @matrix_field_store"):
            tokens = {
                token.split("=", 1)[0]: token.split("=", 1)[1]
                for token in line.split()
                if "=" in token
            }
            field_stores[tokens["field"]] += 1
    return {
        "static_instructions": static_count(assembly),
        "dynamic_issued_instructions": dynamic_count(assembly),
        "opcode_census": census,
        "l_tile_exec_count": sum(primitive_census.values()),
        "primitive_census": dict(sorted(primitive_census.items())),
        "state_transfer_census": dict(sorted(transfers.items())),
        "state_transfer_values": transfer_values,
        "state_transfer_values_by_direction": dict(
            sorted(transfer_values_by_direction.items())
        ),
        "field_transfer_census": dict(sorted(field_transfers.items())),
        "field_store_census": dict(sorted(field_stores.items())),
        "field_logical_values": field_logical_values,
        "field_transfer_values": field_transfer_values,
        "contains_l_tile": bool(primitive_census),
    }


def validate_recurrence_field_loads(
    assembly: str,
    *,
    expected: set[tuple[str, int, int | None]] | None = None,
) -> set[tuple[str, int, int | None]]:
    """Require every field marker to own one executable viewed prefetch.

    This intentionally rejects the old ``@matrix_field_write`` comments.  A
    comment is documentation, not evidence that a producer populated Matrix
    SRAM.
    """

    if "@matrix_field_write=" in assembly:
        raise ValueError("legacy comment-only Matrix field write is not executable")
    lines = assembly.splitlines()
    observed: set[tuple[str, int, int | None]] = set()
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("; @matrix_field_load"):
            continue
        tokens = {
            token.split("=", 1)[0]: token.split("=", 1)[1]
            for token in line.split()
            if "=" in token
        }
        required = {
            "field",
            "target",
            "group",
            "chunk",
            "logical_values",
            "transfer_values",
            "precision",
            "hbm_byte_offset",
        }
        missing = required - tokens.keys()
        if missing:
            raise ValueError(f"malformed Matrix field marker missing {sorted(missing)}")
        chunk = None if tokens["chunk"] == "none" else int(tokens["chunk"])
        key = (tokens["field"], int(tokens["group"]), chunk)
        if key in observed and tokens["field"] not in {"key"}:
            raise ValueError(f"duplicate Matrix field load marker {key}")
        observed.add(key)
        end = next(
            (
                cursor
                for cursor in range(index + 1, len(lines))
                if lines[cursor].strip().startswith("; @matrix_field_load")
                or lines[cursor].strip().startswith("; @matrix_field_store")
                or lines[cursor].strip().startswith("; @matrix_state_")
                or lines[cursor].strip().startswith("; @l_tile_step=")
            ),
            len(lines),
        )
        block = "\n".join(lines[index:end])
        if "L_TILE_CFG 3" not in block or "H_PREFETCH_V.MV" not in block:
            raise ValueError(f"Matrix field marker {key} has no executable viewed DMA")
        if tokens["precision"] != "bf16" or ", 0, 2, 3" not in block:
            raise ValueError(f"Matrix field marker {key} does not use BF16 state traffic")
        logical = int(tokens["logical_values"])
        transferred = int(tokens["transfer_values"])
        if transferred < logical or transferred % PAPER_MLEN:
            raise ValueError(f"Matrix field marker {key} has an invalid DMA extent")
    if expected is not None and observed != expected:
        raise ValueError(
            f"Matrix field coverage differs: missing={sorted(expected - observed)!r}, "
            f"extra={sorted(observed - expected)!r}"
        )
    return observed


def validate_recurrence_output_stores(
    assembly: str,
    *,
    expected_groups: int,
) -> dict[int, int]:
    """Prove that every head-group output leaves Matrix SRAM exactly once."""

    lines = assembly.splitlines()
    stores: dict[int, int] = {}
    store_indices: dict[int, int] = {}
    current_group: int | None = None
    last_exec_by_group: dict[int, int] = {}
    store_offsets: set[int] = set()
    for index, raw in enumerate(lines):
        line = raw.strip()
        if line.startswith("; @head_group="):
            current_group = int(line.split("=", 1)[1].split("/", 1)[0])
        elif line.startswith("L_TILE_EXEC"):
            if current_group is None:
                raise ValueError("L_TILE_EXEC appears before a head-group marker")
            last_exec_by_group[current_group] = index
        elif line.startswith("; @matrix_field_store"):
            tokens = {
                token.split("=", 1)[0]: token.split("=", 1)[1]
                for token in line.split()
                if "=" in token
            }
            if tokens.get("field") != "output_result":
                raise ValueError(f"unexpected recurrence output field {tokens.get('field')}")
            group = int(tokens["group"])
            if group in stores:
                raise ValueError(f"head group {group} stores its output more than once")
            if group != current_group or group not in last_exec_by_group:
                raise ValueError(f"head group {group} output is stored before any EXEC")
            offset = int(tokens["hbm_byte_offset"])
            if offset in store_offsets:
                raise ValueError(f"head groups alias output HBM offset {offset}")
            store_offsets.add(offset)
            end = next(
                (
                    cursor
                    for cursor in range(index + 1, len(lines))
                    if lines[cursor].strip().startswith("; @head_group=")
                    or lines[cursor].strip().startswith("; @matrix_field_")
                    or lines[cursor].strip().startswith("; @matrix_state_")
                    or lines[cursor].strip().startswith("; @l_tile_step=")
                ),
                len(lines),
            )
            block = "\n".join(lines[index:end])
            if "L_TILE_CFG 3" not in block or "H_STORE_V.MV" not in block:
                raise ValueError(f"head group {group} output marker has no viewed store")
            stores[group] = offset
            store_indices[group] = index
    expected = set(range(expected_groups))
    if set(stores) != expected:
        raise ValueError(
            "recurrence output coverage differs: "
            f"missing={sorted(expected - set(stores))}, "
            f"extra={sorted(set(stores) - expected)}"
        )
    for group, store_index in store_indices.items():
        if store_index <= last_exec_by_group[group]:
            raise ValueError(f"head group {group} output is not stored after its final EXEC")
    return stores


def build_matrix_recurrence_report() -> dict[str, object]:
    models: dict[str, object] = {}
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        capacity_points: dict[str, object] = {}
        for capacity_bytes in (ONE_MIB, 2 * ONE_MIB):
            point = MatrixSramPoint(capacity_bytes=capacity_bytes)
            variants: dict[str, object] = {}
            for layout in RecurrenceLayout:
                working_set = build_recurrence_working_set(spec, layout=layout, point=point)
                assembly = lower_matrix_recurrence(spec, layout=layout, point=point)
                field_manifest = build_recurrence_field_manifest(
                    working_set,
                    field_hbm_base=_round_up(spec.state_bytes_per_layer, 64),
                )
                validate_recurrence_field_loads(
                    assembly,
                    expected={
                        packet.key
                        for packet in field_manifest.packets
                        if packet.field != "output_result"
                    },
                )
                validate_recurrence_output_stores(
                    assembly,
                    expected_groups=working_set.groups,
                )
                variants[layout] = {
                    "working_set": working_set.to_dict(),
                    "field_manifest": field_manifest.to_dict(),
                    "metrics": lowering_metrics(assembly),
                    "assembly": assembly,
                }
            capacity_points[str(capacity_bytes)] = variants
        models[spec.name] = {
            "spec": {
                "name": spec.name,
                "kind": spec.kind,
                "heads": spec.heads,
                "row_elements": spec.row_elements,
                "recurrence_rows": spec.recurrence_rows,
                "state_bytes_per_head": spec.state_bytes_per_head,
                "state_bytes_per_layer": spec.state_bytes_per_layer,
                "primitives": [primitive.name for primitive in spec.primitives],
            },
            "capacity_points": capacity_points,
        }
    return {
        "schema_version": 2,
        "geometry": {
            "mlen": PAPER_MLEN,
            "banks": PAPER_BANKS,
            "bank_width": PAPER_BANK_WIDTH,
            "state_element_bytes": BF16_BYTES,
        },
        "models": models,
        "architectural_boundary": {
            "storage": "existing compiler-managed Matrix SRAM scratchpad",
            "math": "existing Vector arithmetic through generic L_TILE primitives",
            "cache": False,
            "private_state_sram": False,
            "runtime_scheduler": False,
            "new_mac_array": False,
        },
    }


__all__ = [
    "BF16_BYTES",
    "KIMI_KDA",
    "LoweringRegisters",
    "NEMOTRON_MAMBA",
    "ONE_MIB",
    "MatrixRecurrenceSpec",
    "MatrixSramPoint",
    "RecurrenceLayout",
    "RecurrenceKind",
    "RecurrenceFieldManifest",
    "RecurrenceFieldPacket",
    "RecurrenceWorkingSet",
    "build_matrix_recurrence_report",
    "build_recurrence_field_manifest",
    "build_recurrence_working_set",
    "lower_matrix_recurrence",
    "lowering_metrics",
    "validate_recurrence_field_loads",
    "validate_recurrence_output_stores",
]
