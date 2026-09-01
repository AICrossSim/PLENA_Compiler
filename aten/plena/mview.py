"""Packed ISA contract for compiler-programmable Matrix-SRAM views.

The view describes physical placement only.  It does not encode an arithmetic
operation, a model, or a runtime traversal.  Existing Matrix/Vector opcodes
select a configured slot explicitly and retain their original mathematics.

Two GP values are sufficient for the hot form:

``shape``
    ``rows_minus_one[11:0] | cols_minus_one[23:12] |
    tile_count_minus_one[31:24]``

``mapping``
    ``tile_pitch_rows[15:0] | alpha[21:16] | reserved_zero[27:22] |
    flags[31:28]``

``tile_pitch_rows`` is measured in full physical Matrix-SRAM rows (one bank
word from every bank), not elements.  This keeps a 2048x2048 tile representable
without putting the machine's bank count or bank width in the ISA.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag


L_MVIEW_OPCODE = 0x3F
L_MVIEW_MAX_SLOTS = 4
L_MVIEW_CONTRACT_VERSION = 1

_DIM_BITS = 12
_DIM_MASK = (1 << _DIM_BITS) - 1
_TILE_COUNT_BITS = 8
_TILE_COUNT_MASK = (1 << _TILE_COUNT_BITS) - 1
_PITCH_BITS = 16
_PITCH_MASK = (1 << _PITCH_BITS) - 1
_ALPHA_BITS = 6
_ALPHA_MASK = (1 << _ALPHA_BITS) - 1


class MatrixViewForm(IntEnum):
    """Function selector carried by the shared ``L_MVIEW`` opcode."""

    FULL = 1
    FIELD = 2


class MatrixViewField(IntEnum):
    """Fields that the cold partial-update form may replace."""

    RESET = 0
    SHAPE = 1
    MAPPING = 2


class MatrixViewFlags(IntFlag):
    """Placement properties; ``STRICT_BOUNDS`` is the canonical hot path."""

    STRICT_BOUNDS = 1 << 0


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
    """Compiler-selected skew and physical pitch for one Matrix tensor view.

    ``alpha`` multiplies the *physical* row address.  Bits [27:22] remain
    reserved until a real compiler-emitted co-access proves that another affine
    term is needed; a synthetic per-tile offset is not part of this contract.
    """

    tile_pitch_rows: int
    alpha: int = 1
    flags: MatrixViewFlags = MatrixViewFlags.STRICT_BOUNDS

    def validate(self) -> None:
        if not 1 <= self.tile_pitch_rows <= _PITCH_MASK:
            raise ValueError(
                f"tile_pitch_rows must be in [1, {_PITCH_MASK}], got {self.tile_pitch_rows}"
            )
        if not 0 <= self.alpha <= _ALPHA_MASK:
            raise ValueError(f"alpha must fit {_ALPHA_BITS} bits, got {self.alpha}")
        unknown = int(self.flags) & ~int(MatrixViewFlags.STRICT_BOUNDS)
        if unknown:
            raise ValueError(f"unknown Matrix-view flags 0x{unknown:x}")

    def pack(self) -> int:
        self.validate()
        return (
            self.tile_pitch_rows
            | self.alpha << _PITCH_BITS
            | int(self.flags) << 28
        )

    @classmethod
    def unpack(cls, word: int) -> "MatrixViewMap":
        _require_u32(word, "mapping word")
        mapping = cls(
            tile_pitch_rows=word & _PITCH_MASK,
            alpha=(word >> _PITCH_BITS) & _ALPHA_MASK,
            flags=MatrixViewFlags((word >> 28) & 0xF),
        )
        if word & (_ALPHA_MASK << (_PITCH_BITS + _ALPHA_BITS)):
            raise ValueError("mapping bits [27:22] are reserved and must be zero")
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
        words_per_row = (self.shape.cols // bank_width + banks - 1) // banks
        required_pitch = self.shape.rows * words_per_row
        if self.mapping.tile_pitch_rows < required_pitch:
            raise ValueError(
                "tile_pitch_rows aliases consecutive tiles: "
                f"need at least {required_pitch}, got {self.mapping.tile_pitch_rows}"
            )
        if self.mapping.alpha >= banks:
            raise ValueError(
                f"alpha must be below the machine's {banks} banks, got "
                f"{self.mapping.alpha}"
            )


def encode_l_mview_full(*, slot: int, shape_register: int, map_register: int) -> int:
    """Encode ``L_MVIEW.FULL slot, shape_reg, map_reg`` canonically."""

    _require_slot(slot)
    _require_register(shape_register, "shape_register")
    _require_register(map_register, "map_register")
    return (
        L_MVIEW_OPCODE
        | shape_register << 6
        | map_register << 10
        | slot << 14
        | int(MatrixViewForm.FULL) << 22
    )


def encode_l_mview_field(
    *, slot: int, field: MatrixViewField | int, value_register: int
) -> int:
    """Encode ``L_MVIEW.FIELD slot, field, value_reg`` canonically."""

    _require_slot(slot)
    _require_register(value_register, "value_register")
    try:
        selected = MatrixViewField(int(field))
    except ValueError as error:
        raise ValueError(f"reserved Matrix-view field {field}") from error
    return (
        L_MVIEW_OPCODE
        | value_register << 6
        | int(selected) << 10
        | slot << 14
        | int(MatrixViewForm.FIELD) << 22
    )


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
    if form is MatrixViewForm.FIELD:
        try:
            MatrixViewField(operand_b)
        except ValueError as error:
            raise ValueError("reserved Matrix-view field") from error
    return form, slot, operand_a, operand_b


def validate_matrix_view_dominance(assembly: str) -> None:
    """Reject a Matrix consumer whose explicit view is not configured first.

    This is deliberately a syntactic, single-pass property. There is no
    implicit SELECT state. A SHAPE+MAPPING field pair is the cold equivalent of
    one atomic FULL; RESET removes the slot.
    """

    configured: set[int] = set()
    partial: dict[int, set[MatrixViewField]] = {}
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
    for line_number, raw in enumerate(assembly.splitlines(), start=1):
        line = raw.split(";", 1)[0].split("//", 1)[0].strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        opcode = parts[0]
        operands = [] if len(parts) == 1 else [part.strip() for part in parts[1].split(",")]
        if opcode == "L_MVIEW_FULL":
            if len(operands) != 3:
                raise ValueError(f"line {line_number}: malformed L_MVIEW_FULL")
            slot = int(operands[0], 0)
            _require_slot(slot)
            configured.add(slot)
            partial.pop(slot, None)
            continue
        if opcode == "L_MVIEW_FIELD":
            if len(operands) != 3:
                raise ValueError(f"line {line_number}: malformed L_MVIEW_FIELD")
            slot = int(operands[0], 0)
            _require_slot(slot)
            field = MatrixViewField(int(operands[1], 0))
            if field is MatrixViewField.RESET:
                configured.discard(slot)
                partial.pop(slot, None)
            else:
                fields = partial.setdefault(slot, set())
                fields.add(field)
                if fields >= {MatrixViewField.SHAPE, MatrixViewField.MAPPING}:
                    configured.add(slot)
            continue
        if opcode == "M_MM_WO" and len(operands) == 4:
            slot = int(operands[3], 0)
            _require_slot(slot)
            if slot not in configured:
                raise ValueError(
                    f"line {line_number}: M_MM_WO consumes Matrix view {slot} "
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
                if mask & (1 << slot) and slot not in configured:
                    raise ValueError(
                        f"line {line_number}: {opcode} consumes Matrix view {slot} "
                        "before a dominating configuration"
                    )
            continue
        if opcode in consumers and len(operands) == 4:
            slot = int(operands[3], 0)
            _require_slot(slot)
            if slot not in configured:
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
    "L_MVIEW_CONTRACT_VERSION",
    "L_MVIEW_MAX_SLOTS",
    "L_MVIEW_OPCODE",
    "MatrixViewDescriptor",
    "MatrixViewField",
    "MatrixViewFlags",
    "MatrixViewForm",
    "MatrixViewMap",
    "MatrixViewShape",
    "decode_l_mview_word",
    "encode_l_mview_field",
    "encode_l_mview_full",
    "validate_matrix_view_dominance",
]
