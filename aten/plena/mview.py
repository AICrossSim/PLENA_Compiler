"""Packed ISA contract for compiler-programmable Matrix-SRAM views.

The view describes physical placement only.  It does not encode an arithmetic
operation, a model, or a runtime traversal.  Existing Matrix/Vector opcodes
select a configured slot explicitly and retain their original mathematics.

Two GP values are sufficient for the hot form:

``shape``
    ``rows_minus_one[11:0] | cols_minus_one[23:12] |
    tile_count_minus_one[31:24]``

``mapping``
    ``tile_pitch_rows[15:0] | reserved[21:16] |
    tile_phase_stride[27:22] | flags[31:28]``

``tile_pitch_rows`` is measured in full physical Matrix-SRAM rows (one bank
word from every bank), not elements.  A zero pitch is legal only when the
inter-tile phase keeps every logical bank word distinct.  That compact form
is what places several logical head rows in the same physical row but in
different banks.  The injectivity check below rejects an unsafe zero pitch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum, IntFlag


L_MVIEW_OPCODE = 0x3F
L_MVIEW_MAX_SLOTS = 4
# Version 3 removes the programmable row coefficient and optional storage
# precision. Bits [21:16] and flag bits [2:0] are now reserved and must trap.
L_MVIEW_CONTRACT_VERSION = 3
MATRIX_VIEW_DMA_MARKER = 1 << 31
MATRIX_VIEW_DMA_SLOT_SHIFT = 29
MATRIX_VIEW_DMA_HIGH_RESERVED_MASK = 0b111 << 26

_DIM_BITS = 12
_DIM_MASK = (1 << _DIM_BITS) - 1
_TILE_COUNT_BITS = 8
_TILE_COUNT_MASK = (1 << _TILE_COUNT_BITS) - 1
_PITCH_BITS = 16
_PITCH_MASK = (1 << _PITCH_BITS) - 1
_PHASE_BITS = 6
_PHASE_MASK = (1 << _PHASE_BITS) - 1


class MatrixViewForm(IntEnum):
    """Function selector carried by the shared ``L_TILE`` opcode."""

    CONFIG = 1
    EXEC = 3


class LTilePrimitive(IntEnum):
    """Model-independent tensor primitives executed over Matrix-SRAM views.

    Shapes, broadcasting and reduction axes come from the three configured
    views.  No primitive names a model or owns persistent state.
    """

    # dst[row, :] = scale[row, 0] * dst[row, :]
    #             + scale[row, 1] * src[row_or_broadcast, :]
    # The decoder walks logical rows; the scale view is segment-broadcast.
    SCALE_ACCUM = 0

    # dst[0, col] = sum_row(src[row, col] * scale[row, 0]).  The source is
    # consumed as a group of logical columns, so this form exercises the
    # transposable/diagonal Matrix-SRAM path rather than materialising a
    # transpose or reducing one row at a time.
    DOT_REDUCE = 1

    # dst[row, :] += vector[row_or_broadcast, :] * scale[row, 0].  This is the
    # generic outer/rank-1 update used by linear recurrences.
    OUTER_UPDATE = 2


class MatrixViewAxis(IntEnum):
    """Logical line direction selected by an ``L_TILE.EXEC`` operand.

    Placement remains a property of the configured view.  Axis is a use-site
    property: the same physical cells may be consumed as logical rows during
    decode or logical columns at a transpose boundary without creating a
    second copy.
    """

    ROW = 0
    COLUMN = 1


class MatrixViewFlags(IntFlag):
    """Placement and element properties for one explicit scratchpad view."""

    # Bits 0..2 are reserved. Bounds are always strict and storage is BF16.
    BROADCAST_MINOR = 1 << 3


@dataclass(frozen=True)
class MatrixViewShape:
    rows: int
    cols: int
    tile_count: int = 1

    def validate(self) -> None:
        for name, value, maximum in (
            ("rows", self.rows, 1 << _DIM_BITS),
            ("cols", self.cols, 1 << _DIM_BITS),
            ("tile_count", self.tile_count, 1 << _TILE_COUNT_BITS),
        ):
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be in [1, {maximum}], got {value}")

    def pack(self) -> int:
        self.validate()
        return (
            (self.rows - 1)
            | (self.cols - 1) << _DIM_BITS
            | (self.tile_count - 1) << (2 * _DIM_BITS)
        )

    @classmethod
    def unpack(cls, word: int) -> "MatrixViewShape":
        _require_u32(word, "shape word")
        return cls(
            rows=(word & _DIM_MASK) + 1,
            cols=((word >> _DIM_BITS) & _DIM_MASK) + 1,
            tile_count=((word >> (2 * _DIM_BITS)) & _TILE_COUNT_MASK) + 1,
        )


@dataclass(frozen=True)
class MatrixViewMap:
    """Compiler-selected physical placement for one Matrix tensor view.

    PLENA's published diagonal row mapping remains fixed. The only programmable
    map term is ``tile_phase_stride``: a six-bit phase applied between logical
    tiles. A tensor's allocation base supplies the constant bank phase, so no
    model-specific field or group identifier is needed.

    This restriction follows the fair D' control: a programmable row
    coefficient provided no bank-service gain for either official workload.
    Freezing it to the existing diagonal value removes a bank-wide multiplier
    or adder candidate before RTL while retaining compact multi-tile placement.
    """

    tile_pitch_rows: int
    tile_phase_stride: int = 0
    flags: MatrixViewFlags = MatrixViewFlags(0)

    @property
    def phased_enabled(self) -> bool:
        """Whether the descriptor applies an inter-tile bank phase."""

        return self.tile_phase_stride != 0

    def validate(self) -> None:
        if not 0 <= self.tile_pitch_rows <= _PITCH_MASK:
            raise ValueError(
                f"tile_pitch_rows must be in [0, {_PITCH_MASK}], got {self.tile_pitch_rows}"
            )
        if not 0 <= self.tile_phase_stride <= _PHASE_MASK:
            raise ValueError(
                "tile_phase_stride must fit "
                f"{_PHASE_BITS} bits, got {self.tile_phase_stride}"
            )
        unknown = int(self.flags) & ~int(MatrixViewFlags.BROADCAST_MINOR)
        if unknown:
            raise ValueError(f"unknown Matrix-view flags 0x{unknown:x}")

    def pack(self) -> int:
        self.validate()
        return (
            self.tile_pitch_rows
            | self.tile_phase_stride << 22
            | int(self.flags) << 28
        )

    @classmethod
    def unpack(cls, word: int) -> "MatrixViewMap":
        _require_u32(word, "mapping word")
        if (word >> 16) & _PHASE_MASK:
            raise ValueError("Matrix-view mapping bits [21:16] are reserved")
        mapping = cls(
            tile_pitch_rows=word & _PITCH_MASK,
            tile_phase_stride=(word >> 22) & _PHASE_MASK,
            flags=MatrixViewFlags((word >> 28) & 0xF),
        )
        mapping.validate()
        return mapping


@dataclass(frozen=True)
class MatrixViewDescriptor:
    shape: MatrixViewShape
    mapping: MatrixViewMap

    def validate_for_machine(self, *, banks: int, bank_width: int) -> None:
        self.shape.validate()
        self.mapping.validate()
        if banks < 1 or banks & (banks - 1):
            raise ValueError(f"banks must be a positive power of two, got {banks}")
        if bank_width < 1:
            raise ValueError(f"bank_width must be positive, got {bank_width}")
        if self.shape.cols % bank_width:
            raise ValueError(
                f"cols {self.shape.cols} must contain whole {bank_width}-element bank words"
            )
        words_per_row = self.shape.cols // bank_width
        row_groups = (words_per_row + banks - 1) // banks
        alpha = 1
        tile_phase_stride = self.mapping.tile_phase_stride
        occupied: dict[tuple[int, int], tuple[int, int, int]] = {}
        for tile in range(self.shape.tile_count):
            for row in range(self.shape.rows):
                for word in range(words_per_row):
                    bank_row = (
                        tile * self.mapping.tile_pitch_rows
                        + row * row_groups
                        + word // banks
                    )
                    bank = (
                        alpha * bank_row + tile_phase_stride * tile + word
                    ) % banks
                    key = (bank, bank_row)
                    previous = occupied.setdefault(key, (tile, row, word))
                    if previous != (tile, row, word):
                        raise ValueError(
                            "Matrix view aliases logical bank words: "
                            f"{previous} and {(tile, row, word)} both map to "
                            f"bank={bank}, row={bank_row}"
                        )


@dataclass(frozen=True)
class MatrixViewAllocation:
    """One compiler-owned tensor view placed at an explicit Matrix-SRAM base."""

    name: str
    base: int
    descriptor: MatrixViewDescriptor


def validate_disjoint_matrix_views(
    allocations: list[MatrixViewAllocation] | tuple[MatrixViewAllocation, ...],
    *,
    mlen: int,
    banks: int,
    bank_width: int,
    depth_rows: int,
    fixed_alpha: int = 1,
    fixed_gamma: int = 0,
) -> dict[str, int]:
    """Prove that all simultaneously-live views occupy distinct bank words.

    This is a static scratchpad allocation check, not a cache simulation.  It
    uses the same coordinate equation as the Rust Matrix SRAM and returns a few
    capacity facts for reports.  A bad placement fails during compilation.
    """

    if mlen != banks * bank_width:
        raise ValueError("mlen must equal banks * bank_width")
    if depth_rows <= 0:
        raise ValueError("depth_rows must be positive")
    if not 0 <= fixed_alpha < banks or not 0 <= fixed_gamma < banks:
        raise ValueError("fixed Matrix-SRAM coefficients must fit the bank count")

    occupied: dict[tuple[int, int], tuple[str, int, int, int]] = {}
    max_row = 0
    for allocation in allocations:
        descriptor = allocation.descriptor
        descriptor.validate_for_machine(banks=banks, bank_width=bank_width)
        if allocation.base < 0 or allocation.base % bank_width:
            raise ValueError(
                f"{allocation.name}: base {allocation.base} is not bank-word aligned"
            )
        base_row, base_offset = divmod(allocation.base, mlen)
        base_bank = base_offset // bank_width
        words_per_row = descriptor.shape.cols // bank_width
        row_groups = (words_per_row + banks - 1) // banks
        alpha = fixed_alpha
        tile_phase_stride = descriptor.mapping.tile_phase_stride
        for tile in range(descriptor.shape.tile_count):
            for row in range(descriptor.shape.rows):
                for word in range(words_per_row):
                    bank_row = (
                        base_row
                        + tile * descriptor.mapping.tile_pitch_rows
                        + row * row_groups
                        + word // banks
                    )
                    if bank_row >= depth_rows:
                        raise ValueError(
                            f"{allocation.name}: bank row {bank_row} exceeds "
                            f"Matrix-SRAM depth {depth_rows}"
                        )
                    bank = (
                        base_bank
                        + alpha * bank_row
                        + tile_phase_stride * tile
                        + fixed_gamma * (bank_row // banks)
                        + word
                    ) % banks
                    key = (bank, bank_row)
                    current = (allocation.name, tile, row, word)
                    previous = occupied.setdefault(key, current)
                    if previous != current:
                        raise ValueError(
                            "Matrix views alias physical bank word "
                            f"bank={bank}, row={bank_row}: {previous} and {current}"
                        )
                    max_row = max(max_row, bank_row + 1)
    return {
        "bank_words": len(occupied),
        "capacity_bank_words": banks * depth_rows,
        "max_bank_row": max_row,
    }


def encode_l_tile_cfg(*, slot: int, shape_register: int, map_register: int) -> int:
    """Encode ``L_TILE_CFG slot, shape_reg, map_reg`` canonically."""

    _require_slot(slot)
    _require_register(shape_register, "shape_register")
    _require_register(map_register, "map_register")
    return (
        L_MVIEW_OPCODE
        | shape_register << 6
        | map_register << 10
        | slot << 14
        | int(MatrixViewForm.CONFIG) << 22
    )


def encode_l_tile_exec(
    *,
    dst_register: int,
    src1_register: int,
    src2_register: int,
    primitive: LTilePrimitive | int,
    source_axis: MatrixViewAxis | int = MatrixViewAxis.ROW,
    scale_axis: MatrixViewAxis | int = MatrixViewAxis.ROW,
) -> int:
    """Encode ``L_TILE_EXEC dst, src1, src2, primitive`` canonically.

    Slots 0/1/2 are the destination/source-1/source-2 descriptors.  Keeping the
    slot assignment positional leaves all operand bases explicit in the
    instruction and avoids an implicit SELECT state.
    """

    for name, register in (
        ("dst_register", dst_register),
        ("src1_register", src1_register),
        ("src2_register", src2_register),
    ):
        _require_register(register, name)
    try:
        selected = LTilePrimitive(int(primitive))
    except ValueError as error:
        raise ValueError(f"reserved L_TILE primitive {primitive}") from error
    try:
        selected_source_axis = MatrixViewAxis(int(source_axis))
        selected_scale_axis = MatrixViewAxis(int(scale_axis))
    except ValueError as error:
        raise ValueError("reserved L_TILE operand axis") from error
    return (
        L_MVIEW_OPCODE
        | dst_register << 6
        | src1_register << 10
        | src2_register << 14
        | int(selected) << 18
        | int(MatrixViewForm.EXEC) << 22
        | int(selected_source_axis) << 26
        | int(selected_scale_axis) << 27
    )


def decode_l_tile_exec_word(
    word: int,
) -> tuple[int, int, int, LTilePrimitive, MatrixViewAxis, MatrixViewAxis]:
    """Decode one canonical ``L_TILE_EXEC`` word."""

    _require_u32(word, "instruction word")
    if word & 0x3F != L_MVIEW_OPCODE or ((word >> 22) & 0xF) != MatrixViewForm.EXEC:
        raise ValueError("word is not an L_TILE_EXEC encoding")
    if word >> 28:
        raise ValueError("reserved L_TILE_EXEC bits [31:28] must be zero")
    try:
        primitive = LTilePrimitive((word >> 18) & 0xF)
    except ValueError as error:
        raise ValueError("reserved L_TILE primitive") from error
    return (
        (word >> 6) & 0xF,
        (word >> 10) & 0xF,
        (word >> 14) & 0xF,
        primitive,
        MatrixViewAxis((word >> 26) & 0x1),
        MatrixViewAxis((word >> 27) & 0x1),
    )


def encode_matrix_view_dma_word(legacy_word: int, *, slot: int) -> int:
    """Qualify an existing vector DMA word with an explicit Matrix view.

    The physical opcode and every legacy operand field remain unchanged. Bit
    31 marks the addressing form, bits 30:29 name a configured view, and bits
    28:26 remain canonical zero. This is not a new transfer operation.
    """

    _require_u32(legacy_word, "legacy DMA word")
    _require_slot(slot)
    if legacy_word >> 26:
        raise ValueError("legacy DMA word uses reserved bits [31:26]")
    if legacy_word & 0x3F not in {0x29, 0x2A}:
        raise ValueError("Matrix-view DMA requires H_PREFETCH_V or H_STORE_V")
    return legacy_word | MATRIX_VIEW_DMA_MARKER | (slot << MATRIX_VIEW_DMA_SLOT_SHIFT)


def decode_matrix_view_dma_slot(word: int) -> int | None:
    """Return the explicit Matrix-view slot, or ``None`` for a legacy DMA."""

    _require_u32(word, "DMA word")
    if word >> 26 == 0:
        return None
    if not word & MATRIX_VIEW_DMA_MARKER or word & MATRIX_VIEW_DMA_HIGH_RESERVED_MASK:
        raise ValueError("non-canonical Matrix-view DMA bits [31:26]")
    slot = (word >> MATRIX_VIEW_DMA_SLOT_SHIFT) & 0b11
    _require_slot(slot)
    return slot


def decode_l_mview_word(word: int) -> tuple[MatrixViewForm, int, int, int]:
    """Return ``(form, slot, operand_a, operand_b)`` for a canonical word."""

    _require_u32(word, "instruction word")
    if word & 0x3F != L_MVIEW_OPCODE or word >> 26:
        raise ValueError("word is not a canonical L_MVIEW encoding")
    try:
        form = MatrixViewForm((word >> 22) & 0xF)
    except ValueError as error:
        raise ValueError("reserved L_MVIEW form") from error
    slot = (word >> 14) & 0xF
    _require_slot(slot)
    operand_a = (word >> 6) & 0xF
    operand_b = (word >> 10) & 0xF
    if (word >> 18) & 0xF:
        raise ValueError("reserved L_MVIEW bits [21:18] must be zero")
    if form is MatrixViewForm.EXEC:
        raise ValueError("use decode_l_tile_exec_word for EXEC form")
    return form, slot, operand_a, operand_b


def validate_matrix_view_dominance(assembly: str) -> None:
    """Reject a Matrix consumer whose explicit view is not configured first.

    This is a syntactic must-dataflow property over the static loop control-flow
    graph.  There is no implicit SELECT state. Intersections at loop back-edges
    prevent a first-iteration-only configuration from being
    accepted as dominating every dynamic use. Inside a loop, C_BREAK has both a
    fallthrough edge and an edge to the instruction after the matching loop end.
    Intersecting those paths is conservative for both the public debug-exception
    wording and the transactional emulator's loop-break behavior.
    """

    consumers = {
        "M_MM",
        "M_TMM",
        "M_BMM",
        "M_BTMM",
        "M_MV",
        "M_TMV",
        "M_BMV",
        "M_BTMV",
    }
    instructions: list[tuple[int, str, list[str]]] = []
    for line_number, raw in enumerate(assembly.splitlines(), start=1):
        line = raw.split(";", 1)[0].split("//", 1)[0].strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        opcode = parts[0]
        operands = [] if len(parts) == 1 else [part.strip() for part in parts[1].split(",")]
        instructions.append((line_number, opcode, operands))

    loop_stack: list[int] = []
    end_to_start: dict[int, int] = {}
    start_to_end: dict[int, int] = {}
    break_to_start: dict[int, int] = {}
    for index, (line_number, opcode, operands) in enumerate(instructions):
        if opcode == "L_TILE_CFG":
            if len(operands) != 3:
                raise ValueError(f"line {line_number}: malformed {opcode}")
            _require_slot(int(operands[0], 0))
        if opcode == "C_LOOP_START":
            loop_stack.append(index)
        elif opcode == "C_BREAK" and loop_stack:
            break_to_start[index] = loop_stack[-1]
        elif opcode == "C_LOOP_END":
            if not loop_stack:
                raise ValueError(f"line {line_number}: C_LOOP_END without C_LOOP_START")
            start = loop_stack.pop()
            end_to_start[index] = start
            start_to_end[start] = index
    if loop_stack:
        line_number = instructions[loop_stack[-1]][0]
        raise ValueError(f"line {line_number}: C_LOOP_START without C_LOOP_END")

    def successors(index: int) -> tuple[int, ...]:
        _, opcode, _ = instructions[index]
        fallthrough = index + 1
        if opcode == "C_LOOP_END":
            start = end_to_start[index]
            result = [start + 1]
            if fallthrough < len(instructions):
                result.append(fallthrough)
            return tuple(result)
        if opcode == "C_BREAK" and index in break_to_start:
            result = []
            if fallthrough < len(instructions):
                result.append(fallthrough)
            after_loop = start_to_end[break_to_start[index]] + 1
            if after_loop < len(instructions):
                result.append(after_loop)
            return tuple(dict.fromkeys(result))
        return (fallthrough,) if fallthrough < len(instructions) else ()

    State = frozenset[tuple[str, int]]

    def transfer(state: State, opcode: str, operands: list[str]) -> State:
        result = set(state)
        if opcode == "L_TILE_CFG":
            slot = int(operands[0], 0)
            result.difference_update({token for token in result if token[1] == slot})
            result.update({("shape", slot), ("mapping", slot), ("configured", slot)})
        return frozenset(result)

    if not instructions:
        return
    in_states: list[State | None] = [None] * len(instructions)
    in_states[0] = frozenset()
    work = deque([0])
    while work:
        index = work.popleft()
        state = in_states[index]
        assert state is not None
        _, opcode, operands = instructions[index]
        output = transfer(state, opcode, operands)
        for target in successors(index):
            previous = in_states[target]
            merged = output if previous is None else previous & output
            if merged != previous:
                in_states[target] = merged
                work.append(target)

    for (line_number, opcode, operands), state in zip(instructions, in_states, strict=True):
        if state is None:
            continue
        if opcode == "L_TILE_CFG":
            if len(operands) != 3:
                raise ValueError(f"line {line_number}: malformed L_TILE_CFG")
            slot = int(operands[0], 0)
            _require_slot(slot)
            continue
        if opcode == "L_TILE_EXEC":
            if len(operands) not in {4, 5}:
                raise ValueError(f"line {line_number}: malformed L_TILE_EXEC")
            LTilePrimitive(int(operands[3], 0))
            if len(operands) == 5:
                axis_mask = int(operands[4], 0)
                if not 0 <= axis_mask <= 0b11:
                    raise ValueError(
                        f"line {line_number}: L_TILE_EXEC axis mask must be in [0, 3]"
                    )
            for slot in range(3):
                if ("configured", slot) not in state:
                    raise ValueError(
                        f"line {line_number}: L_TILE_EXEC consumes Matrix view {slot} "
                        "before a dominating configuration"
                    )
            continue
        if opcode == "M_MM_WO" and len(operands) == 4:
            slot = int(operands[3], 0)
            _require_slot(slot)
            if ("configured", slot) not in state:
                raise ValueError(
                    f"line {line_number}: M_MM_WO consumes Matrix view {slot} "
                    "before a dominating configuration"
                )
            continue
        if opcode in {"H_PREFETCH_V.MV", "H_STORE_V.MV"}:
            if len(operands) != 6:
                raise ValueError(f"line {line_number}: malformed {opcode}")
            slot = int(operands[5], 0)
            _require_slot(slot)
            if ("configured", slot) not in state:
                raise ValueError(
                    f"line {line_number}: {opcode} consumes Matrix view {slot} "
                    "before a dominating configuration"
                )
            continue
        if opcode.endswith(".MV"):
            if len(operands) != 5:
                raise ValueError(f"line {line_number}: malformed {opcode}")
            mask = int(operands[4], 0)
            if not 1 <= mask <= 0b111:
                raise ValueError(
                    f"line {line_number}: Matrix-view operand mask must be in [1, 7]"
                )
            for slot in range(3):
                if mask & (1 << slot) and ("configured", slot) not in state:
                    raise ValueError(
                        f"line {line_number}: {opcode} consumes Matrix view {slot} "
                        "before a dominating configuration"
                    )
            continue
        if opcode in consumers and len(operands) == 4:
            slot = int(operands[3], 0)
            _require_slot(slot)
            if ("configured", slot) not in state:
                raise ValueError(
                    f"line {line_number}: {opcode} consumes Matrix view {slot} "
                    "before a dominating configuration"
                )


def _require_slot(slot: int) -> None:
    if not 0 <= slot < L_MVIEW_MAX_SLOTS:
        raise ValueError(f"slot must be in [0, {L_MVIEW_MAX_SLOTS}), got {slot}")


def _require_register(register: int, name: str) -> None:
    if not 0 <= register < 16:
        raise ValueError(f"{name} must fit four bits, got {register}")


def _require_u32(word: int, name: str) -> None:
    if not 0 <= word <= 0xFFFF_FFFF:
        raise ValueError(f"{name} must be an unsigned 32-bit integer")


__all__ = [
    "LTilePrimitive",
    "L_MVIEW_CONTRACT_VERSION",
    "L_MVIEW_MAX_SLOTS",
    "L_MVIEW_OPCODE",
    "MATRIX_VIEW_DMA_HIGH_RESERVED_MASK",
    "MATRIX_VIEW_DMA_MARKER",
    "MATRIX_VIEW_DMA_SLOT_SHIFT",
    "MatrixViewDescriptor",
    "MatrixViewAllocation",
    "MatrixViewAxis",
    "MatrixViewFlags",
    "MatrixViewForm",
    "MatrixViewMap",
    "MatrixViewShape",
    "decode_l_tile_exec_word",
    "decode_matrix_view_dma_slot",
    "decode_l_mview_word",
    "encode_l_tile_exec",
    "encode_matrix_view_dma_word",
    "encode_l_tile_cfg",
    "validate_matrix_view_dominance",
    "validate_disjoint_matrix_views",
]
