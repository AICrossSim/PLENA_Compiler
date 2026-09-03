"""Extract Matrix-SRAM access packets from emitted PLENA assembly.

The extractor operates on the executable instruction stream, not a model-name
table.  It records the logical Matrix cells needed in one service cycle and
keeps loop multiplicity separate from the packet shape.  This makes a crucial
distinction visible: today's ``M_*`` instructions name exactly one Matrix tile;
an affine per-tile skew cannot help until a view-tagged lowering explicitly
co-issues more than one tile.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

from compiler.aten.plena.mview import (
    LTilePrimitive,
    MatrixViewAxis,
    MatrixViewDescriptor,
    MatrixViewMap,
    MatrixViewShape,
)


_MATRIX_READ_OPS = frozenset(
    {"M_MM", "M_TMM", "M_BMM", "M_BTMM", "M_MV", "M_TMV", "M_BMV", "M_BTMV"}
)
_MATRIX_WRITE_OPS = frozenset({"H_PREFETCH_M"})
_MATRIX_VIEW_VECTOR_OPS = frozenset({"V_ADD_VV.MV", "V_SUB_VV.MV", "V_MUL_VV.MV"})
_MATRIX_VIEW_DMA_OPS = frozenset({"H_PREFETCH_V.MV", "H_STORE_V.MV"})
_STAGE = re.compile(r"@stage=([A-Za-z0-9_\-]+)")
_AXIS = re.compile(r"@axis=([A-Za-z0-9_\-]+)")
_GP = re.compile(r"^gp(\d+)$")
_S_ADDI = re.compile(r"^S_ADDI_INT\s+gp(\d+)\s*,\s*gp(\d+)\s*,\s*(-?\d+)$")
_S_BINARY = re.compile(r"^S_(ADD|SUB|MUL)_INT\s+gp(\d+)\s*,\s*gp(\d+)\s*,\s*gp(\d+)$")
_S_LUI = re.compile(r"^S_LUI_INT\s+gp(\d+)\s*,\s*(-?\d+)$")
_LOOP_START = re.compile(r"^C_LOOP_START\s+gp\d+\s*,\s*(\d+)$")


@dataclass(frozen=True)
class LogicalCell:
    tile: int | str
    row: int | str
    col: int | str


@dataclass(frozen=True)
class MatrixAccessPacket:
    instruction_index: int
    opcode: str
    stage: str
    direction: str
    axis: str
    matrix_address: int | str
    tile_count: int
    elements_per_tile: int
    repeats: int
    sample_cells: tuple[LogicalCell, ...]
    view_slot: int | None = None
    operand: str = "matrix"
    view_rows: int | None = None
    view_cols: int | None = None
    tile_pitch_rows: int | None = None
    view_alpha: int | None = None
    view_tile_skew: int | None = None
    view_affine: bool | None = None
    view_tile_start: int | None = None
    view_line_axis: str | None = None
    view_line_period: int | None = None
    view_broadcast_tile: bool | None = None
    l_tile_primitive: str | None = None
    # Compile-time address delta between successive executions of the
    # innermost hardware loop containing this instruction.  ``None`` means the
    # scalar update was not affine and therefore cannot be replayed exactly.
    address_stride_elements: int | None = 0

    @property
    def values_per_packet(self) -> int:
        return self.tile_count * self.elements_per_tile

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["values_per_packet"] = self.values_per_packet
        return result


def _view_mapping_metadata(
    descriptor: MatrixViewDescriptor | None,
) -> tuple[int | None, int | None, bool | None]:
    """Return the physical coefficients selected by one configured view."""

    if descriptor is None:
        return None, None, None
    mapping = descriptor.mapping
    affine = mapping.affine_enabled
    return (
        mapping.row_skew if affine else 1,
        mapping.tile_skew if affine else 0,
        affine,
    )


@dataclass(frozen=True)
class PacketGeometry:
    mlen: int
    blen: int
    hlen: int

    def __post_init__(self) -> None:
        if min(self.mlen, self.blen, self.hlen) <= 0:
            raise ValueError("Matrix packet geometry must be positive")
        if self.mlen % self.hlen:
            raise ValueError("MLEN must be divisible by HLEN")


def extract_matrix_access_packets(
    assembly: str,
    geometry: PacketGeometry,
) -> tuple[MatrixAccessPacket, ...]:
    """Extract every Matrix-SRAM read/write in ``assembly``.

    GP values are propagated through the scalar integer instructions used by
    PLENA's emitters.  A value that depends on runtime data stays symbolic as
    ``gpN`` rather than being guessed.  Hardware-loop trip counts multiply
    ``repeats`` but packet coordinates describe one service cycle only.
    """

    registers: list[int | None] = [0] + [None] * 15
    loop_address_strides = _loop_address_strides(assembly)
    stage = "unattributed"
    packet_axis: str | None = None
    views: dict[int, MatrixViewDescriptor] = {}
    loop_trips: list[int] = []
    packets: list[MatrixAccessPacket] = []
    covered_matrix_instructions = 0
    instruction_index = 0

    for raw in assembly.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(";") or line.startswith("//"):
            match = _STAGE.search(line)
            if match:
                stage = match.group(1)
            match = _AXIS.search(line)
            if match:
                packet_axis = match.group(1)
            continue
        line = line.split(";", 1)[0].split("//", 1)[0].strip()
        if not line:
            continue

        loop = _LOOP_START.match(line)
        if loop:
            loop_trips.append(int(loop.group(1)))
            instruction_index += 1
            continue
        if line.startswith("C_LOOP_END"):
            if not loop_trips:
                raise ValueError("C_LOOP_END without a matching C_LOOP_START")
            loop_trips.pop()
            instruction_index += 1
            continue

        _update_gp_state(line, registers)
        opcode, operands = _split_instruction(line)
        multiplier = _product(loop_trips)
        if opcode == "L_TILE_CFG":
            _update_matrix_views(
                operands,
                registers,
                views,
            )
        elif opcode in _MATRIX_READ_OPS:
            packets.append(
                _matrix_read_packet(
                    instruction_index,
                    opcode,
                    operands,
                    stage,
                    multiplier,
                    registers,
                    geometry,
                    views,
                    loop_address_strides.get(instruction_index, {}),
                )
            )
            covered_matrix_instructions += 1
        elif opcode in _MATRIX_WRITE_OPS:
            packets.append(
                _matrix_dma_packet(
                    instruction_index,
                    opcode,
                    operands,
                    stage,
                    multiplier,
                    registers,
                    geometry,
                    loop_address_strides.get(instruction_index, {}),
                )
            )
            covered_matrix_instructions += 1
        elif opcode == "M_MM_WO" and len(operands) == 4:
            packets.append(
                _matrix_view_writeback_packet(
                    instruction_index,
                    opcode,
                    operands,
                    stage,
                    multiplier,
                    registers,
                    views,
                    geometry,
                    loop_address_strides.get(instruction_index, {}),
                )
            )
            covered_matrix_instructions += 1
        elif opcode in _MATRIX_VIEW_VECTOR_OPS:
            packets.extend(
                _matrix_view_vector_packets(
                    instruction_index,
                    opcode,
                    operands,
                    stage,
                    packet_axis,
                    multiplier,
                    registers,
                    views,
                    loop_address_strides.get(instruction_index, {}),
                )
            )
            covered_matrix_instructions += 1
        elif opcode in _MATRIX_VIEW_DMA_OPS:
            packets.append(
                _matrix_view_dma_packet(
                    instruction_index,
                    opcode,
                    operands,
                    stage,
                    multiplier,
                    registers,
                    views,
                    loop_address_strides.get(instruction_index, {}),
                )
            )
            covered_matrix_instructions += 1
        elif opcode == "L_TILE_EXEC":
            packets.extend(
                _l_tile_exec_packets(
                    instruction_index,
                    operands,
                    stage,
                    multiplier,
                    registers,
                    geometry,
                    views,
                    loop_address_strides.get(instruction_index, {}),
                )
            )
            covered_matrix_instructions += 1
        instruction_index += 1

    if loop_trips:
        raise ValueError("unterminated C_LOOP_START in assembly")
    matrix_lines = matrix_access_instruction_count(assembly)
    if covered_matrix_instructions != matrix_lines:
        raise AssertionError(
            "Matrix packet extraction covered "
            f"{covered_matrix_instructions} of {matrix_lines} Matrix instructions"
        )
    return tuple(packets)


def matrix_access_instruction_count(assembly: str) -> int:
    """Count executable instructions that the packet extractor must cover.

    This intentionally counts instructions, not operand packets: one
    view-qualified Vector instruction can issue two Matrix reads and one Matrix
    write in the same service group.  Reporters use this public helper together
    with the unique ``instruction_index`` values in the extracted packets, so
    extraction coverage cannot silently diverge from the parser's definition.
    """

    return sum(_is_matrix_access_instruction(raw) for raw in assembly.splitlines())


def packet_histogram(packets: Iterable[MatrixAccessPacket]) -> list[dict[str, object]]:
    """Aggregate by the shape that determines bank-conflict opportunity."""

    counts: Counter[tuple[str, str, int, int]] = Counter()
    dynamic: Counter[tuple[str, str, int, int]] = Counter()
    for packet in packets:
        key = (packet.stage, packet.axis, packet.tile_count, packet.elements_per_tile)
        counts[key] += 1
        dynamic[key] += packet.repeats
    return [
        {
            "stage": stage,
            "axis": axis,
            "tiles": tiles,
            "elements_per_tile": elements,
            "static_packets": counts[(stage, axis, tiles, elements)],
            "dynamic_packets": dynamic[(stage, axis, tiles, elements)],
            "values_per_packet": tiles * elements,
            "per_tile_skew_can_help": tiles > 1,
        }
        for stage, axis, tiles, elements in sorted(counts)
    ]


def coissued_packet_histogram(
    packets: Iterable[MatrixAccessPacket],
) -> list[dict[str, object]]:
    """Aggregate Matrix operands that one instruction services together.

    A view-qualified Vector instruction may read two Matrix operands in one
    issue slot.  Counting those operands as unrelated packets hides the actual
    bank load seen by the SRAM.  Reads are therefore grouped by instruction;
    the later destination write remains a separate service group.
    """

    groups: dict[tuple[int, str, str], list[MatrixAccessPacket]] = {}
    for packet in packets:
        groups.setdefault(
            (packet.instruction_index, packet.direction, packet.axis), []
        ).append(packet)

    static: Counter[tuple[str, str, str, str, tuple[tuple[str, int, int], ...]]] = (
        Counter()
    )
    dynamic: Counter[tuple[str, str, str, str, tuple[tuple[str, int, int], ...]]] = (
        Counter()
    )
    for group in groups.values():
        repeats = {packet.repeats for packet in group}
        if len(repeats) != 1:
            raise ValueError(
                "co-issued Matrix operands have different loop multiplicities"
            )
        stages = {packet.stage for packet in group}
        axes = {packet.axis for packet in group}
        opcodes = {packet.opcode for packet in group}
        if len(stages) != 1 or len(axes) != 1 or len(opcodes) != 1:
            raise ValueError(
                "co-issued Matrix operands disagree on instruction identity"
            )
        shapes = tuple(
            sorted(
                (packet.operand, packet.tile_count, packet.elements_per_tile)
                for packet in group
            )
        )
        key = (
            next(iter(stages)),
            next(iter(axes)),
            next(iter(opcodes)),
            group[0].direction,
            shapes,
        )
        static[key] += 1
        dynamic[key] += repeats.pop()

    result = []
    for key in sorted(static):
        stage, axis, opcode, direction, shapes = key
        operands = [
            {
                "name": operand,
                "tiles": tiles,
                "elements_per_tile": elements,
                "values": tiles * elements,
            }
            for operand, tiles, elements in shapes
        ]
        result.append(
            {
                "stage": stage,
                "axis": axis,
                "opcode": opcode,
                "direction": direction,
                "same_cycle_operands": len(operands),
                "operands": operands,
                "tiles_total": sum(item["tiles"] for item in operands),
                "values_per_service": sum(item["values"] for item in operands),
                "static_service_groups": static[key],
                "dynamic_service_groups": dynamic[key],
                "per_tile_skew_can_help": any(item["tiles"] > 1 for item in operands),
            }
        )
    return result


def coissued_packet_groups(
    packets: Iterable[MatrixAccessPacket],
) -> list[dict[str, object]]:
    """Return each static Matrix service group without dropping addresses.

    The histogram is convenient for presentation, but physical bank service
    depends on the allocation base as well as the packet shape.  This compact
    form preserves every operand that one instruction services together while
    omitting the potentially large cell list.
    """

    groups: dict[tuple[int, str, str], list[MatrixAccessPacket]] = {}
    for packet in packets:
        groups.setdefault(
            (packet.instruction_index, packet.direction, packet.axis), []
        ).append(packet)

    result: list[dict[str, object]] = []
    for (instruction_index, direction, _phase), group in sorted(groups.items()):
        repeats = {packet.repeats for packet in group}
        stages = {packet.stage for packet in group}
        axes = {packet.axis for packet in group}
        opcodes = {packet.opcode for packet in group}
        if len(repeats) != 1 or len(stages) != 1 or len(axes) != 1 or len(opcodes) != 1:
            raise ValueError(
                "co-issued Matrix operands disagree on instruction metadata"
            )
        result.append(
            {
                "instruction_index": instruction_index,
                "stage": next(iter(stages)),
                "axis": next(iter(axes)),
                "opcode": next(iter(opcodes)),
                "direction": direction,
                "repeats": repeats.pop(),
                "operands": [
                    {
                        "name": packet.operand,
                        "matrix_address": packet.matrix_address,
                        "tiles": packet.tile_count,
                        "elements_per_tile": packet.elements_per_tile,
                        "view_slot": packet.view_slot,
                        "view_rows": packet.view_rows,
                        "view_cols": packet.view_cols,
                        "tile_pitch_rows": packet.tile_pitch_rows,
                        "view_alpha": packet.view_alpha,
                        "view_tile_skew": packet.view_tile_skew,
                        "view_affine": packet.view_affine,
                        "view_tile_start": packet.view_tile_start,
                        "view_line_axis": packet.view_line_axis,
                        "view_line_period": packet.view_line_period,
                        "view_broadcast_tile": packet.view_broadcast_tile,
                        "l_tile_primitive": packet.l_tile_primitive,
                        "address_stride_elements": packet.address_stride_elements,
                    }
                    for packet in group
                ],
            }
        )
    return result


def _matrix_read_packet(
    instruction_index: int,
    opcode: str,
    operands: list[str],
    stage: str,
    multiplier: int,
    registers: list[int | None],
    geometry: PacketGeometry,
    views: dict[int, MatrixViewDescriptor],
    loop_strides: dict[int, int | None],
) -> MatrixAccessPacket:
    if len(operands) not in (3, 4):
        raise ValueError(
            f"{opcode} requires 3 operands plus an optional view, got {operands}"
        )
    matrix_operand = operands[1]
    matrix_address = _resolve_operand(matrix_operand, registers)
    view_slot = int(operands[3]) if len(operands) == 4 else None
    descriptor = None
    if view_slot is not None:
        try:
            descriptor = views[view_slot]
        except KeyError as error:
            raise ValueError(
                f"{opcode} uses unconfigured Matrix-view slot {view_slot}"
            ) from error
    tile_count = descriptor.shape.tile_count if descriptor is not None else 1
    tile: int | str
    if isinstance(matrix_address, int):
        tile = matrix_address // (geometry.mlen * geometry.mlen)
    else:
        tile = matrix_address

    transposed = opcode in {"M_TMM", "M_BTMM", "M_TMV", "M_BTMV"}
    broadcast = opcode in {"M_BMM", "M_BTMM", "M_BMV", "M_BTMV"}
    width = geometry.hlen if broadcast else geometry.blen
    if descriptor is not None and transposed:
        axis = "column"
        elements_per_tile = descriptor.shape.rows
        cells = tuple(
            LogicalCell(tile_index, row, "cycle")
            for tile_index in range(descriptor.shape.tile_count)
            for row in range(descriptor.shape.rows)
        )
    elif descriptor is not None:
        axis = "row"
        elements_per_tile = descriptor.shape.cols
        cells = tuple(
            LogicalCell(tile_index, "cycle", col)
            for tile_index in range(descriptor.shape.tile_count)
            for col in range(descriptor.shape.cols)
        )
    elif transposed:
        axis = "column"
        elements_per_tile = width
        cells = tuple(LogicalCell(tile, row, "cycle") for row in range(width))
    else:
        axis = "row"
        elements_per_tile = width
        cells = tuple(LogicalCell(tile, "cycle", col) for col in range(width))
    view_alpha, view_tile_skew, view_affine = _view_mapping_metadata(descriptor)
    return MatrixAccessPacket(
        instruction_index=instruction_index,
        opcode=opcode,
        stage=stage,
        direction="read",
        axis=axis,
        matrix_address=matrix_address,
        tile_count=tile_count,
        elements_per_tile=elements_per_tile,
        repeats=multiplier * geometry.mlen,
        sample_cells=cells,
        view_slot=view_slot,
        view_rows=descriptor.shape.rows if descriptor is not None else None,
        view_cols=descriptor.shape.cols if descriptor is not None else None,
        tile_pitch_rows=(
            descriptor.mapping.tile_pitch_rows if descriptor is not None else None
        ),
        view_alpha=view_alpha,
        view_tile_skew=view_tile_skew,
        view_affine=view_affine,
        address_stride_elements=_operand_loop_stride(matrix_operand, loop_strides),
    )


def _matrix_dma_packet(
    instruction_index: int,
    opcode: str,
    operands: list[str],
    stage: str,
    multiplier: int,
    registers: list[int | None],
    geometry: PacketGeometry,
    loop_strides: dict[int, int | None],
) -> MatrixAccessPacket:
    if len(operands) < 1:
        raise ValueError(f"{opcode} requires a Matrix destination")
    address = _resolve_operand(operands[0], registers)
    tile = (
        address // (geometry.mlen * geometry.mlen)
        if isinstance(address, int)
        else address
    )
    return MatrixAccessPacket(
        instruction_index=instruction_index,
        opcode=opcode,
        stage=stage,
        direction="write",
        axis="dma",
        matrix_address=address,
        tile_count=1,
        elements_per_tile=geometry.mlen,
        repeats=multiplier * geometry.mlen,
        sample_cells=tuple(
            LogicalCell(tile, 0, col) for col in range(min(geometry.mlen, 8))
        ),
        address_stride_elements=_operand_loop_stride(operands[0], loop_strides),
    )


def _update_matrix_views(
    operands: list[str],
    registers: list[int | None],
    views: dict[int, MatrixViewDescriptor],
) -> None:
    if len(operands) != 3:
        raise ValueError(f"L_TILE_CFG requires three operands, got {operands}")
    slot = int(operands[0], 0)
    if not 0 <= slot < 4:
        raise ValueError(f"Matrix-view slot must be in [0, 4), got {slot}")
    shape_word = _require_resolved(operands[1], registers, "Matrix-view shape")
    map_word = _require_resolved(operands[2], registers, "Matrix-view mapping")
    descriptor = MatrixViewDescriptor(
        MatrixViewShape.unpack(shape_word),
        MatrixViewMap.unpack(map_word),
    )
    descriptor.shape.validate()
    descriptor.mapping.validate()
    views[slot] = descriptor


def _matrix_view_vector_packets(
    instruction_index: int,
    opcode: str,
    operands: list[str],
    stage: str,
    packet_axis: str | None,
    multiplier: int,
    registers: list[int | None],
    views: dict[int, MatrixViewDescriptor],
    loop_strides: dict[int, int | None],
) -> tuple[MatrixAccessPacket, ...]:
    if len(operands) != 5:
        raise ValueError(f"{opcode} requires rd, rs1, rs2, rmask, view-mask")
    view_mask = int(operands[4], 0)
    if not 1 <= view_mask <= 0b111:
        raise ValueError(f"{opcode} Matrix-view mask must be in [1, 7]")
    result = []
    for slot, operand_index, name, direction in (
        (0, 0, "destination", "write"),
        (1, 1, "source1", "read"),
        (2, 2, "source2", "read"),
    ):
        if not view_mask & (1 << slot):
            continue
        result.append(
            _matrix_view_operand_packet(
                instruction_index=instruction_index,
                opcode=opcode,
                stage=stage,
                axis=packet_axis or "view_packet",
                direction=direction,
                matrix_address=_resolve_operand(operands[operand_index], registers),
                address_stride_elements=_operand_loop_stride(
                    operands[operand_index], loop_strides
                ),
                multiplier=multiplier,
                slot=slot,
                operand=name,
                views=views,
            )
        )
    return tuple(result)


def _matrix_view_dma_packet(
    instruction_index: int,
    opcode: str,
    operands: list[str],
    stage: str,
    multiplier: int,
    registers: list[int | None],
    views: dict[int, MatrixViewDescriptor],
    loop_strides: dict[int, int | None],
) -> MatrixAccessPacket:
    """Describe an ordinary Vector DMA redirected through a Matrix view."""

    if len(operands) != 6:
        raise ValueError(
            f"{opcode} requires destination, source, address register, offset, "
            "precision, and Matrix-view slot"
        )
    slot = int(operands[5], 0)
    try:
        descriptor = views[slot]
    except KeyError as error:
        raise ValueError(
            f"{opcode} uses unconfigured Matrix-view slot {slot}"
        ) from error
    matrix_operand = operands[0]
    shape = descriptor.shape
    view_alpha, view_tile_skew, view_affine = _view_mapping_metadata(descriptor)
    sample = tuple(
        LogicalCell(tile, row, col)
        for tile in range(shape.tile_count)
        for row in range(shape.rows)
        for col in range(shape.cols)
    )[:32]
    return MatrixAccessPacket(
        instruction_index=instruction_index,
        opcode=opcode,
        stage=stage,
        direction="write" if opcode == "H_PREFETCH_V.MV" else "read",
        axis="view_dma",
        matrix_address=_resolve_operand(matrix_operand, registers),
        tile_count=shape.tile_count,
        elements_per_tile=shape.rows * shape.cols,
        repeats=multiplier,
        sample_cells=sample,
        view_slot=slot,
        operand="dma_destination" if opcode == "H_PREFETCH_V.MV" else "dma_source",
        view_rows=shape.rows,
        view_cols=shape.cols,
        tile_pitch_rows=descriptor.mapping.tile_pitch_rows,
        view_alpha=view_alpha,
        view_tile_skew=view_tile_skew,
        view_affine=view_affine,
        address_stride_elements=_operand_loop_stride(matrix_operand, loop_strides),
    )


def _axis_extent(
    descriptor: MatrixViewDescriptor, axis: MatrixViewAxis
) -> tuple[int, int]:
    """Return (logical lines, values per line) for one view axis."""

    if axis is MatrixViewAxis.ROW:
        return descriptor.shape.rows, descriptor.shape.cols
    return descriptor.shape.cols, descriptor.shape.rows


def _line_cells(
    *,
    descriptor: MatrixViewDescriptor,
    axis: MatrixViewAxis,
    tile_start: int,
    tile_count: int,
    line: int,
    broadcast_tile: bool = False,
) -> tuple[LogicalCell, ...]:
    cells: list[LogicalCell] = []
    for packet_tile in range(tile_start, tile_start + tile_count):
        tile = 0 if broadcast_tile else packet_tile
        if axis is MatrixViewAxis.ROW:
            cells.extend(
                LogicalCell(tile, line, col) for col in range(descriptor.shape.cols)
            )
        else:
            cells.extend(
                LogicalCell(tile, row, line) for row in range(descriptor.shape.rows)
            )
    return tuple(cells)


def _l_tile_packet(
    *,
    instruction_index: int,
    stage: str,
    direction: str,
    phase: str,
    matrix_address: int | str,
    address_stride_elements: int | None,
    repeats: int,
    slot: int,
    operand: str,
    descriptor: MatrixViewDescriptor,
    axis: MatrixViewAxis,
    tile_start: int,
    tile_count: int,
    line: int,
    line_period: int,
    primitive: LTilePrimitive,
    broadcast_tile: bool = False,
) -> MatrixAccessPacket:
    _, line_width = _axis_extent(descriptor, axis)
    view_alpha, view_tile_skew, view_affine = _view_mapping_metadata(descriptor)
    return MatrixAccessPacket(
        instruction_index=instruction_index,
        opcode="L_TILE_EXEC",
        stage=stage,
        direction=direction,
        axis=phase,
        matrix_address=matrix_address,
        tile_count=tile_count,
        elements_per_tile=line_width,
        repeats=repeats,
        sample_cells=_line_cells(
            descriptor=descriptor,
            axis=axis,
            tile_start=tile_start,
            tile_count=tile_count,
            line=line,
            broadcast_tile=broadcast_tile,
        ),
        view_slot=slot,
        operand=operand,
        view_rows=descriptor.shape.rows,
        view_cols=descriptor.shape.cols,
        tile_pitch_rows=descriptor.mapping.tile_pitch_rows,
        view_alpha=view_alpha,
        view_tile_skew=view_tile_skew,
        view_affine=view_affine,
        view_tile_start=tile_start,
        view_line_axis=axis.name.lower(),
        view_line_period=line_period,
        view_broadcast_tile=broadcast_tile,
        l_tile_primitive=primitive.name,
        address_stride_elements=address_stride_elements,
    )


def _l_tile_exec_packets(
    instruction_index: int,
    operands: list[str],
    stage: str,
    multiplier: int,
    registers: list[int | None],
    geometry: PacketGeometry,
    views: dict[int, MatrixViewDescriptor],
    loop_strides: dict[int, int | None],
) -> tuple[MatrixAccessPacket, ...]:
    """Expand one deterministic L_TILE macro into physical SRAM phases.

    The expansion mirrors ``Accelerator::execute_l_tile``.  Destination,
    source, and compact scalar packets use the existing one-read-port Matrix
    SRAM sequentially; they are deliberately assigned distinct ``axis`` phase
    names so a report cannot mislabel them as same-cycle operands.
    """

    if len(operands) not in (4, 5):
        raise ValueError(
            "L_TILE_EXEC requires destination/source/scale bases, primitive, "
            "and an optional axis mask"
        )
    try:
        descriptors = [views[slot] for slot in range(3)]
    except KeyError as error:
        raise ValueError(
            f"L_TILE_EXEC uses unconfigured Matrix-view slot {error.args[0]}"
        ) from error
    dst, source, scales = descriptors
    primitive = LTilePrimitive(int(operands[3], 0))
    axis_mask = int(operands[4], 0) if len(operands) == 5 else 0
    if not 0 <= axis_mask <= 3:
        raise ValueError("L_TILE_EXEC axis mask must be in [0, 3]")
    source_axis = MatrixViewAxis(axis_mask & 1)
    scale_axis = MatrixViewAxis((axis_mask >> 1) & 1)
    bases = [_resolve_operand(operand, registers) for operand in operands[:3]]
    strides = [_operand_loop_stride(operand, loop_strides) for operand in operands[:3]]
    result: list[MatrixAccessPacket] = []

    if primitive in {LTilePrimitive.SCALE_ACCUM, LTilePrimitive.OUTER_UPDATE}:
        source_lines, source_width = _axis_extent(source, source_axis)
        scale_lines, _ = _axis_extent(scales, scale_axis)
        if source_width != dst.shape.cols:
            raise ValueError("row-wise L_TILE source/destination widths differ")
        if source_lines not in {1, dst.shape.rows}:
            raise ValueError(
                "row-wise L_TILE source rows must be one or match destination"
            )
        if scale_lines < dst.shape.rows:
            raise ValueError(
                "L_TILE scale view has fewer logical lines than destination"
            )
        tiles_per_packet = max(1, geometry.mlen // dst.shape.cols)
        for tile_start in range(0, dst.shape.tile_count, tiles_per_packet):
            tile_count = min(tiles_per_packet, dst.shape.tile_count - tile_start)
            row_repeats = multiplier * dst.shape.rows
            for (
                direction,
                phase,
                slot,
                operand,
                descriptor,
                axis,
                line_period,
                base,
                stride,
            ) in (
                (
                    "read",
                    "l_tile_dst_read",
                    0,
                    "destination",
                    dst,
                    MatrixViewAxis.ROW,
                    dst.shape.rows,
                    bases[0],
                    strides[0],
                ),
                (
                    "read",
                    "l_tile_source_read",
                    1,
                    "source",
                    source,
                    source_axis,
                    source_lines,
                    bases[1],
                    strides[1],
                ),
                (
                    "write",
                    "l_tile_dst_write",
                    0,
                    "destination",
                    dst,
                    MatrixViewAxis.ROW,
                    dst.shape.rows,
                    bases[0],
                    strides[0],
                ),
            ):
                result.append(
                    _l_tile_packet(
                        instruction_index=instruction_index,
                        stage=stage,
                        direction=direction,
                        phase=phase,
                        matrix_address=base,
                        address_stride_elements=stride,
                        repeats=row_repeats,
                        slot=slot,
                        operand=operand,
                        descriptor=descriptor,
                        axis=axis,
                        tile_start=tile_start,
                        tile_count=tile_count,
                        line=0,
                        line_period=line_period,
                        primitive=primitive,
                        broadcast_tile=(descriptor.shape.tile_count == 1),
                    )
                )
            scale_tile_count = 1 if scales.shape.tile_count == 1 else tile_count
            result.append(
                _l_tile_packet(
                    instruction_index=instruction_index,
                    stage=stage,
                    direction="read",
                    phase="l_tile_scale_read",
                    matrix_address=bases[2],
                    address_stride_elements=strides[2],
                    repeats=row_repeats,
                    slot=2,
                    operand="scale",
                    descriptor=scales,
                    axis=scale_axis,
                    tile_start=0 if scales.shape.tile_count == 1 else tile_start,
                    tile_count=scale_tile_count,
                    line=0,
                    line_period=scale_lines,
                    primitive=primitive,
                    broadcast_tile=False,
                )
            )
    else:
        source_lines, source_width = _axis_extent(source, source_axis)
        scale_lines, _ = _axis_extent(scales, scale_axis)
        if dst.shape.rows != 1 or dst.shape.cols != source_width:
            raise ValueError("DOT_REDUCE destination must be one row per source tile")
        if dst.shape.tile_count != source.shape.tile_count:
            raise ValueError("DOT_REDUCE destination/source tile counts differ")
        if scale_lines < source_lines:
            raise ValueError("DOT_REDUCE scale view has fewer lines than reduction rows")
        tiles_per_packet = max(1, geometry.mlen // source_width)
        for tile_start in range(0, source.shape.tile_count, tiles_per_packet):
            tile_count = min(tiles_per_packet, source.shape.tile_count - tile_start)
            packet_repeats = multiplier
            for direction, phase in (
                ("read", "l_tile_dst_read"),
                ("write", "l_tile_dst_write"),
            ):
                result.append(
                    _l_tile_packet(
                        instruction_index=instruction_index,
                        stage=stage,
                        direction=direction,
                        phase=phase,
                        matrix_address=bases[0],
                        address_stride_elements=strides[0],
                        repeats=packet_repeats,
                        slot=0,
                        operand="destination",
                        descriptor=dst,
                        axis=MatrixViewAxis.ROW,
                        tile_start=tile_start,
                        tile_count=tile_count,
                        line=0,
                        line_period=1,
                        primitive=primitive,
                    )
                )
            result.append(
                _l_tile_packet(
                    instruction_index=instruction_index,
                    stage=stage,
                    direction="read",
                    phase="l_tile_source_read",
                    matrix_address=bases[1],
                    address_stride_elements=strides[1],
                    repeats=packet_repeats * source_lines,
                    slot=1,
                    operand="source",
                    descriptor=source,
                    axis=source_axis,
                    tile_start=tile_start,
                    tile_count=tile_count,
                    line=0,
                    line_period=source_lines,
                    primitive=primitive,
                )
            )
            scale_tile_count = 1 if scales.shape.tile_count == 1 else tile_count
            result.append(
                _l_tile_packet(
                    instruction_index=instruction_index,
                    stage=stage,
                    direction="read",
                    phase="l_tile_scale_read",
                    matrix_address=bases[2],
                    address_stride_elements=strides[2],
                    repeats=packet_repeats * source_lines,
                    slot=2,
                    operand="scale",
                    descriptor=scales,
                    axis=scale_axis,
                    tile_start=0 if scales.shape.tile_count == 1 else tile_start,
                    tile_count=scale_tile_count,
                    line=0,
                    line_period=scale_lines,
                    primitive=primitive,
                )
            )
    return tuple(result)


def _matrix_view_operand_packet(
    *,
    instruction_index: int,
    opcode: str,
    stage: str,
    axis: str,
    direction: str,
    matrix_address: int | str,
    address_stride_elements: int | None,
    multiplier: int,
    slot: int,
    operand: str,
    views: dict[int, MatrixViewDescriptor],
) -> MatrixAccessPacket:
    try:
        descriptor = views[slot]
    except KeyError as error:
        raise ValueError(
            f"{opcode} uses unconfigured Matrix-view slot {slot}"
        ) from error
    shape = descriptor.shape
    view_alpha, view_tile_skew, view_affine = _view_mapping_metadata(descriptor)
    cells = tuple(
        LogicalCell(tile, row, col)
        for tile in range(shape.tile_count)
        for row in range(shape.rows)
        for col in range(shape.cols)
    )
    return MatrixAccessPacket(
        instruction_index=instruction_index,
        opcode=opcode,
        stage=stage,
        direction=direction,
        axis=axis,
        matrix_address=matrix_address,
        tile_count=shape.tile_count,
        elements_per_tile=shape.rows * shape.cols,
        repeats=multiplier,
        sample_cells=cells,
        view_slot=slot,
        operand=operand,
        view_rows=shape.rows,
        view_cols=shape.cols,
        tile_pitch_rows=descriptor.mapping.tile_pitch_rows,
        view_alpha=view_alpha,
        view_tile_skew=view_tile_skew,
        view_affine=view_affine,
        address_stride_elements=address_stride_elements,
    )


def _matrix_view_writeback_packet(
    instruction_index: int,
    opcode: str,
    operands: list[str],
    stage: str,
    multiplier: int,
    registers: list[int | None],
    views: dict[int, MatrixViewDescriptor],
    geometry: PacketGeometry,
    loop_strides: dict[int, int | None],
) -> MatrixAccessPacket:
    slot = int(operands[3], 0)
    try:
        descriptor = views[slot]
    except KeyError as error:
        raise ValueError(
            f"{opcode} uses unconfigured Matrix-view slot {slot}"
        ) from error
    logical_offset = _require_resolved(
        operands[2], registers, "Matrix accumulator logical offset"
    )
    shape = descriptor.shape
    view_alpha, view_tile_skew, view_affine = _view_mapping_metadata(descriptor)
    values_per_tile = shape.rows * shape.cols
    if logical_offset % geometry.blen:
        raise ValueError("M_MM_WO logical offset must select one BLEN-wide fragment")
    tile = logical_offset // values_per_tile
    within = logical_offset % values_per_tile
    start_col = within % shape.cols
    if tile >= shape.tile_count or start_col + geometry.blen > shape.cols:
        raise ValueError("M_MM_WO fragment exceeds its configured Matrix view")
    cells = []
    for micro_row in range(shape.rows):
        flat = logical_offset + micro_row * shape.cols
        micro_tile = flat // values_per_tile
        micro_within = flat % values_per_tile
        row = micro_within // shape.cols
        col = micro_within % shape.cols
        if micro_tile >= shape.tile_count or col != start_col:
            raise ValueError("M_MM_WO microtile wraps a configured Matrix-view row")
        cells.extend(
            LogicalCell(micro_tile, row, col + lane) for lane in range(geometry.blen)
        )
    return MatrixAccessPacket(
        instruction_index=instruction_index,
        opcode=opcode,
        stage=stage,
        direction="write",
        axis="producer_writeback",
        matrix_address=_resolve_operand(operands[0], registers),
        tile_count=1,
        elements_per_tile=shape.rows * geometry.blen,
        repeats=multiplier,
        sample_cells=tuple(cells),
        view_slot=slot,
        operand="accumulator",
        view_rows=shape.rows,
        view_cols=shape.cols,
        tile_pitch_rows=descriptor.mapping.tile_pitch_rows,
        view_alpha=view_alpha,
        view_tile_skew=view_tile_skew,
        view_affine=view_affine,
        address_stride_elements=_operand_loop_stride(operands[0], loop_strides),
    )


def _is_matrix_access_instruction(raw: str) -> bool:
    clean = raw.split(";", 1)[0].split("//", 1)[0].strip()
    if not clean:
        return False
    opcode, operands = _split_instruction(clean)
    return (
        opcode in _MATRIX_READ_OPS
        or opcode in _MATRIX_WRITE_OPS
        or opcode in _MATRIX_VIEW_VECTOR_OPS
        or opcode in _MATRIX_VIEW_DMA_OPS
        or opcode == "L_TILE_EXEC"
        or (opcode == "M_MM_WO" and len(operands) == 4)
    )


def _operand_loop_stride(
    operand: str, loop_strides: dict[int, int | None]
) -> int | None:
    """Return the affine loop delta for one GP address operand.

    Matrix addresses encoded as immediates are invariant.  A GP that is not
    modified by the innermost loop is likewise invariant.  ``None`` is
    reserved for a GP whose loop update is not a constant self-increment.
    """

    match = _GP.match(operand)
    if match is None:
        return 0
    return loop_strides.get(int(match.group(1)), 0)


def _loop_address_strides(assembly: str) -> dict[int, dict[int, int | None]]:
    """Recover per-iteration GP deltas for every instruction in a hardware loop.

    The emitted recurrent kernels use constant ``S_ADDI_INT gpN, gpN, imm``
    updates.  Recording that delta lets the downstream bank model replay every
    dynamic packet address.  If a loop writes a GP in any other way, the value
    is marked unresolved instead of being guessed.
    """

    lines: list[str] = []
    for raw in assembly.splitlines():
        clean = raw.split(";", 1)[0].split("//", 1)[0].strip()
        if clean:
            lines.append(clean)

    stack: list[int] = []
    loops: list[tuple[int, int, dict[int, int | None]]] = []
    for index, line in enumerate(lines):
        if _LOOP_START.match(line):
            stack.append(index)
            continue
        if not line.startswith("C_LOOP_END"):
            continue
        if not stack:
            raise ValueError("C_LOOP_END without a matching C_LOOP_START")
        start = stack.pop()
        deltas: Counter[int] = Counter()
        unresolved: set[int] = set()
        for body_line in lines[start + 1 : index]:
            if match := _S_ADDI.match(body_line):
                dst, src, immediate = map(int, match.groups())
                if dst == src:
                    deltas[dst] += immediate
                else:
                    unresolved.add(dst)
                continue
            if match := _S_BINARY.match(body_line):
                unresolved.add(int(match.group(2)))
                continue
            if match := _S_LUI.match(body_line):
                unresolved.add(int(match.group(1)))
        strides: dict[int, int | None] = dict(deltas)
        strides.update({register: None for register in unresolved})
        loops.append((start, index, strides))
    if stack:
        raise ValueError("unterminated C_LOOP_START in assembly")

    result: dict[int, dict[int, int | None]] = {}
    # Outer loops are installed first; an inner loop then overrides them for
    # instructions in its body, which is the required per-issue address delta.
    for start, end, strides in sorted(
        loops, key=lambda loop: loop[1] - loop[0], reverse=True
    ):
        for index in range(start + 1, end):
            result[index] = strides
    return result


def _split_instruction(line: str) -> tuple[str, list[str]]:
    parts = line.split(maxsplit=1)
    return parts[0], [] if len(parts) == 1 else [
        part.strip() for part in parts[1].split(",")
    ]


def _resolve_operand(operand: str, registers: list[int | None]) -> int | str:
    match = _GP.match(operand)
    if match:
        register = int(match.group(1))
        return registers[register] if registers[register] is not None else operand
    try:
        return int(operand, 0)
    except ValueError:
        return operand


def _require_resolved(
    operand: str,
    registers: list[int | None],
    label: str,
) -> int:
    value = _resolve_operand(operand, registers)
    if not isinstance(value, int):
        raise ValueError(f"{label} is not compile-time-resolved: {operand}")
    return value


def _update_gp_state(line: str, registers: list[int | None]) -> None:
    if match := _S_ADDI.match(line):
        dst, src, immediate = map(int, match.groups())
        registers[dst] = (
            None
            if registers[src] is None
            else (registers[src] + immediate) & 0xFFFF_FFFF
        )
        return
    if match := _S_BINARY.match(line):
        operation, dst_raw, lhs_raw, rhs_raw = match.groups()
        dst, lhs, rhs = int(dst_raw), int(lhs_raw), int(rhs_raw)
        a, b = registers[lhs], registers[rhs]
        if a is None or b is None:
            registers[dst] = None
        elif operation == "ADD":
            registers[dst] = (a + b) & 0xFFFF_FFFF
        elif operation == "SUB":
            registers[dst] = (a - b) & 0xFFFF_FFFF
        else:
            registers[dst] = (a * b) & 0xFFFF_FFFF
        return
    if match := _S_LUI.match(line):
        dst, immediate = map(int, match.groups())
        registers[dst] = (immediate << 12) & 0xFFFF_FFFF


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


__all__ = [
    "coissued_packet_groups",
    "coissued_packet_histogram",
    "LogicalCell",
    "MatrixAccessPacket",
    "PacketGeometry",
    "extract_matrix_access_packets",
    "packet_histogram",
]
