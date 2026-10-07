"""Native BF16 coefficient views over existing fixed-diagonal Matrix SRAM.

CCFG uses L_TILE form 2. Three 64-bit descriptors are loaded from pairs of
32-bit GP registers; this is an explicit ISA extension, not implicit MMIO.
Addresses and strides are BF16 elements. DMA still moves real producer data.
"""
from dataclasses import dataclass

from compiler.aten.plena.mview import (LTilePrimitive as Op, MatrixViewAllocation,
    MatrixViewFlags, validate_disjoint_matrix_views)
from compiler.aten.plena.ltile_v2 import Emitter, View, Descriptor, Shape, Map, views


@dataclass(frozen=True)
class CoefficientView:
    base: int
    head_stride: int
    row_stride: int
    repeat_log2: int
    heads: int
    extent: int

    def __post_init__(self):
        fields = ((self.base, 19), (self.head_stride, 13), (self.row_stride, 9),
                  (self.repeat_log2, 3), (self.heads - 1, 6), (self.extent - 1, 12))
        if any(type(v) is not int or not 0 <= v < 1 << n for v, n in fields):
            raise ValueError("coefficient view exceeds its 64-bit encoding")
        if self.base + self.extent > 524288:
            raise ValueError("coefficient view exceeds 1 MiB SRAM")

    def address(self, row, head):
        if row < 0 or not 0 <= head < self.heads:
            raise ValueError("coefficient index outside view")
        offset = (head >> self.repeat_log2) * self.head_stride + row * self.row_stride
        if offset >= self.extent:
            raise ValueError("coefficient offset exceeds extent")
        return self.base + offset

    def pack(self):
        return (self.base | self.head_stride << 19 | self.row_stride << 32
                | self.repeat_log2 << 41 | (self.heads - 1) << 44 | (self.extent - 1) << 50)


def lower_native_group(o, memory, emitter=None):
    """One full resident group; native entries: HBM row, offset, H/R strides, repeat.

    Three 2048-element producer rows are copied without permutation. The
    descriptor selects a possibly offset/shared subset. Duplicate HBM rows
    share one allocation/transfer. Original state and scalar ABI is retained.
    """
    if o.control != "fsm" or not (o.phased and o.broadcast and o.resident):
        raise ValueError("native prototype requires phased resident FSM")
    if len(memory["native"]) != 3 or len(memory["states"]) != 1:
        raise ValueError("native group requires three coefficient views and one full state tile")
    e = emitter or Emitter()
    state = views(o)["state"]
    # Packed legacy vectors use a tall, sparse placement interleaved with the
    # coefficient view. Native fields need separate rows; keep vectors compact
    # across head phases so a full-row coefficient DMA cannot overwrite them.
    def vector(row):
        return View(row*2048, Descriptor(Shape(1, o.width, o.heads), Map(0, o.width//32)))
    x, out, scratch = vector(128), vector(129), vector(130)
    scalar = View(131*2048, Descriptor(Shape(1, ((2*o.heads+31)//32)*32),
        Map(1, flags=MatrixViewFlags.BROADCAST_MINOR)))
    coeff = scalar  # native primitives use CCFG instead of this operand
    allocated = [MatrixViewAllocation(n, v.base, v.descriptor) for n,v in
                 (("state",state),("input",x),("out",out),("scratch",scratch),("scalar",scalar))]
    loaded = {}
    for slot, (hbm, offset, hs, rs, repeat) in enumerate(memory["native"]):
        if not ((rs == 0 and hs in (0, 1)) or (rs == 1 and hs in (0, 128) and offset % 32 == 0)):
            raise ValueError("native sector engine supports contiguous row sectors or row-invariant scalars")
        if hbm not in loaded:
            base = (144 + 2 * len(loaded)) * 2048
            loaded[hbm] = base
            field = View(base, Descriptor(Shape(1, 2048), Map(1)))
            allocated.append(MatrixViewAllocation(f"native{slot}", base, field.descriptor))
            e.dma(field, hbm)
        descriptor = CoefficientView(loaded[hbm] + offset, hs, rs, repeat, o.heads, 2048-offset)
        descriptor.address(127, o.heads-1)  # fail before encoding an invalid walk
        word = descriptor.pack()
        e.address(12, word & 0xffffffff)
        e.address(13, word >> 32)
        e.lines.append(f"L_TILE_CCFG {slot}, gp12, gp13")

    validate_disjoint_matrix_views(allocated, mlen=2048, banks=64, bank_width=32, depth_rows=256)

    def run(op, d, s, c=coeff):
        e.execute(op, d, s, c)

    e.dma(x, memory["input"])
    e.dma(scalar, memory["scalar"])
    e.dma(state, memory["states"][0])
    if o.kind == "mamba":
        run(Op.SCALE_ACCUM, scratch, x, scalar)
    else:
        run(Op.REDUCE_BEGIN, out, out, scalar)
        run(Op.NATIVE_DECAY_REDUCE_ACC, out, state)
        run(Op.RESIDUAL_WRITE, scratch, x, scalar)
    run(Op.NATIVE_DELTA_UPDATE, state, scratch)
    run(Op.REDUCE_BEGIN, out, out, scalar)
    run(Op.NATIVE_REDUCE_ACC, out, state)
    run(Op.REDUCE_WRITE, out, out, scalar)
    e.dma(state, memory["states"][0], True)
    if "snapshots" in memory:
        e.dma(state, memory["snapshots"][0], True)
    if o.kind == "mamba":
        e.dma(scalar, memory["skip"])
        run(Op.SCALE_ACCUM, out, x, scalar)
    e.dma(out, memory["output"], True)
    return e
