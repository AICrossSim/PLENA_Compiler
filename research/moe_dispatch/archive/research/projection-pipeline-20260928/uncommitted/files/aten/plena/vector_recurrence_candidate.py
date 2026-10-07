"""Experimental fused recurrence through *Vector SRAM only*.

This interface is an explicit hardware candidate, not an unchanged-ISA or
area-free Vector optimization. Forms 4/5/6 of opcode 0x3f configure and execute
it; Matrix L_TILE forms 1/2/3 retain their existing meaning. Arithmetic reuses
the proposed fused FP32 contract while partial sums stay in the existing BF16
Vector SRAM. Levels 1..4 isolate expanded supply, compact supply, an invariant
latch and bounded traversal. They do not establish physical ALU sharing.
"""

from dataclasses import dataclass
from enum import IntEnum

from compiler.aten.plena.ltile_native import CoefficientView
from compiler.aten.plena.prepared_vector_recurrence import _Emitter


class VectorRecurrencePrimitive(IntEnum):
    UPDATE = 0
    PREDICTION_PRODUCT = 1
    READOUT_PRODUCT = 2
    RESIDUAL = 3
    HOLD = 4
    TREE_RESET = 5
    TREE_ACC = 6
    TREE_WRITE = 7
    SCALE_INPUT = 8
    SKIP_OUTPUT = 9


@dataclass(frozen=True)
class VectorRecurrenceConfig:
    rows: int
    width: int
    level: int
    row_origin: int = 0

    def pack(self):
        if any(type(v) is not int for v in (self.rows, self.width, self.level, self.row_origin)):
            raise ValueError("Vector recurrence fields must be integers")
        if not 1 <= self.rows <= 32 or self.width not in (64, 128):
            raise ValueError("Vector recurrence geometry exceeds bounded SRAM workspace")
        if not 1 <= self.level <= 4 or not 0 <= self.row_origin < 128:
            raise ValueError("invalid Vector recurrence level or row origin")
        if self.row_origin + self.rows > 128 or (self.level != 4 and self.rows != 1):
            raise ValueError("only the FSM arm may traverse more than one row")
        return (self.rows - 1) | (int(self.width == 128) << 5), self.level | (self.row_origin << 3)


def encode_vector_recurrence(*, form, dst_register, src1_register, src2_register, primitive=0):
    """Canonical 32-bit experimental Vector-only words; no Matrix view mask."""
    if form not in (4, 5, 6):
        raise ValueError("reserved Vector recurrence form")
    for register in (dst_register, src1_register, src2_register):
        if type(register) is not int or not 0 <= register < 16:
            raise ValueError("Vector recurrence register must be GP0..15")
    if form == 4 and dst_register != 0:
        raise ValueError("Vector recurrence has one control descriptor")
    if form == 5 and dst_register > 2:
        raise ValueError("Vector recurrence coefficient slot must be 0..2")
    if form == 6:
        primitive = VectorRecurrencePrimitive(primitive)
    elif primitive:
        raise ValueError("configuration words have no primitive")
    if form != 6:
        # Like L_TILE_CFG/CCFG: low/shape in bits 9:6, high/control in
        # bits 13:10 and the small slot number in bits 17:14.
        dst_register, src1_register, src2_register = src1_register, src2_register, dst_register
    return (0x3f | dst_register << 6 | src1_register << 10 | src2_register << 14
            | int(primitive) << 18 | form << 22)


class CandidateEmitter(_Emitter):
    def __init__(self, kind, level):
        if kind not in ("mamba", "kda"):
            raise ValueError("expected Mamba or KDA")
        super().__init__(2048, True)
        self.kind, self.level = kind, level
        self.width = 64 if kind == "mamba" else 128
        self.heads = 2048 // self.width
        self._config = None
        self._coefficients = {}

    def configure(self, origin=0, rows=1):
        words = VectorRecurrenceConfig(rows, self.width, self.level, origin).pack()
        if words != self._config:
            self.address(12, words[0]); self.address(13, words[1])
            self.lines.append("V_REC_CFG 0, gp12, gp13")
            self._config = words

    def coefficient(self, slot, descriptor):
        if descriptor.base + descriptor.extent > 131072:
            raise ValueError("Vector coefficient view exceeds 256 KiB SRAM")
        word = descriptor.pack()
        if self._coefficients.get(slot) != word:
            self.address(12, word & 0xffffffff); self.address(13, word >> 32)
            self.lines.append(f"V_REC_CCFG {slot}, gp12, gp13")
            self._coefficients[slot] = word

    def expanded(self, slot, row):
        self.coefficient(slot, CoefficientView(row * 2048, self.width, 0, 0, self.heads, 2048))

    def execute(self, primitive, destination=0, source=0, invariant=0):
        for register, row in ((9, destination), (10, source), (11, invariant)):
            self.address(register, row * 2048)
        self.lines.append(f"V_REC_EXEC gp9, gp10, gp11, {int(primitive)}")

    def merge_leaf(self, index, destination):
        """Existing Vector adds, BF16 after each merge; seven SRAM levels."""
        merges = 0
        while index & (1 << merges):
            merges += 1
        for level in range(merges):
            target = 5
            if level == merges - 1:
                target = destination if index == 127 else 8 + merges
            self.binary("ADD", target, 8 + level, 5)


def lower_vector_candidate_group(kind, *, level, state_base, input_base, output_base,
                                 zero_base, scalar_base=None, skip_base=None,
                                 native=None, coefficient_loader=None):
    """One private HBM group, all transfers explicit and capacity bounded.

    Row mode owns state row 0. FSM mode owns rows 20..51, so 32 state rows,
    existing tree/workspace rows 0..15, compact fields 16..18 and scalar row
    19 fit in 208 KiB. KDA prediction/update reload the ORIGINAL HBM state:
    no decayed BF16 state or hidden resident 512 KiB tile is introduced.
    Level 1's ordinary software loader expands into rows 1/15 and retains its
    documented rows 16..61 cache. Levels 2..4 load compact producer rows once.
    """
    e = CandidateEmitter(kind, level)
    e.configure()
    e.lines.append(f"; @vector_recurrence_candidate={level}; arithmetic=fused_fp32_bf16_tree_rn")
    e.transfer(7, zero_base)
    e.transfer(2, input_base)
    if level == 1:
        if coefficient_loader is None or native is not None:
            raise ValueError("expanded candidate requires an ordinary software coefficient loader")
        coefficient_loader.prepare(e)
        if kind == "mamba":
            coefficient_loader.load(e, "dt", 0, 1)
            e.binary("MUL", 3, 2, 1)
            coefficient_loader.load(e, "a", 0, 15)
        e.expanded(0, 15); e.expanded(1, 1); e.expanded(2, 1)
    else:
        if native is None or len(native) != 3 or scalar_base is None:
            raise ValueError("compact candidate requires three producer views and a scalar row")
        loaded = {}
        for slot, (hbm, offset, hs, rs, repeat) in enumerate(native):
            if hbm not in loaded:
                loaded[hbm] = 16 + len(loaded)
                e.transfer(loaded[hbm], hbm)
            descriptor = CoefficientView(loaded[hbm] * 2048 + offset, hs, rs, repeat, e.heads, 2048 - offset)
            descriptor.address(127, e.heads - 1)
            e.coefficient(slot, descriptor)
        e.transfer(19, scalar_base)
        if kind == "mamba":
            saved_delta = e._coefficients[0]
            e.coefficient(0, CoefficientView(19 * 2048 + 1, 2, 0, 0, e.heads, 2 * e.heads - 1))
            e.execute(VectorRecurrencePrimitive.SCALE_INPUT, 3, 2, 19)
            e._coefficients.pop(0)
            hbm, offset, hs, rs, repeat = native[0]
            e.coefficient(0, CoefficientView(loaded[hbm] * 2048 + offset, hs, rs, repeat, e.heads, 2048 - offset))
            assert e._coefficients[0] == saved_delta

    def hold(row):
        if level >= 3:
            e.configure()
            e.execute(VectorRecurrencePrimitive.HOLD, 0, row)

    def scan(prediction=False):
        output = 6 if prediction else 4
        if level == 4:
            e.configure()
            e.execute(VectorRecurrencePrimitive.TREE_RESET)
        chunks = range(0, 128, 32) if level == 4 else range(128)
        for origin in chunks:
            count = 32 if level == 4 else 1
            first_state = 20 if level == 4 else 0
            for r in range(count):
                e.transfer(first_state + r, state_base + (origin + r) * 4096)
            e.configure(origin, count)
            if level == 1:
                if kind == "kda":
                    coefficient_loader.load(e, "decay", origin, 15)
                coefficient_loader.load(e, "key" if kind == "kda" else "b", origin, 1)
            leaf = 5 if level == 4 or origin % 2 else 8
            if prediction:
                e.execute(VectorRecurrencePrimitive.PREDICTION_PRODUCT, leaf, first_state)
            else:
                e.execute(VectorRecurrencePrimitive.UPDATE, first_state, first_state, 3)
                for r in range(count):
                    e.transfer(first_state + r, state_base + (origin + r) * 4096, store=True)
                if level == 1:
                    coefficient_loader.load(e, "query" if kind == "kda" else "c", origin, 1)
                e.execute(VectorRecurrencePrimitive.READOUT_PRODUCT, leaf, first_state)
            if level != 4:
                e.merge_leaf(origin, output)
        if level == 4:
            e.configure()
            e.execute(VectorRecurrencePrimitive.TREE_WRITE, output)

    if kind == "kda":
        scan(prediction=True)
        e.configure()
        if level == 1:
            coefficient_loader.load(e, "beta", 0, 1)
            e.expanded(0, 1)
        else:
            e.coefficient(0, CoefficientView(19 * 2048, 2, 0, 0, e.heads, 2 * e.heads))
        e.execute(VectorRecurrencePrimitive.RESIDUAL, 3, 2, 6)
        if level == 1:
            e.expanded(0, 15)
        else:
            hbm, offset, hs, rs, repeat = native[0]
            e.coefficient(0, CoefficientView(loaded[hbm] * 2048 + offset, hs, rs, repeat, e.heads, 2048 - offset))
    hold(3)
    scan()
    if kind == "mamba":
        e.configure()
        if level == 1:
            coefficient_loader.load(e, "d", 0, 1)
            e.expanded(0, 1)
            e.execute(VectorRecurrencePrimitive.SKIP_OUTPUT, 4, 2)
        else:
            if skip_base is None:
                raise ValueError("Mamba compact candidate requires skip producer row")
            e.transfer(19, skip_base)
            e.coefficient(0, CoefficientView(19 * 2048 + 1, 2, 0, 0, e.heads, 2 * e.heads - 1))
            e.execute(VectorRecurrencePrimitive.SKIP_OUTPUT, 4, 2, 19)
    e.transfer(4, output_base, store=True)
    return "\n".join(e.lines) + "\n"
