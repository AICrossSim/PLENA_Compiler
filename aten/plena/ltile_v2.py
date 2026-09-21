"""Executable v2 lowering; no analytic packet timing is used here.

The ordinary-control arm shares new arithmetic interfaces with the walker.
It is explicitly NOT an unchanged-old-ISA hardware baseline. Address reuse,
hardware scalar loops, placement and coefficient preparation are shared.
"""
from dataclasses import dataclass, replace
from compiler.asm_templates._imm import load_large_int
from compiler.aten.plena.mview import (
    LTilePrimitive as Op, MatrixViewDescriptor as Descriptor,
    MatrixViewShape as Shape, MatrixViewMap as Map, MatrixViewFlags as Flag,
    MatrixViewAllocation, validate_disjoint_matrix_views,
)

@dataclass(frozen=True)
class Options:
    kind: str
    control: str = "fsm"
    phased: bool = True
    broadcast: bool = True
    resident: bool = True
    group_values: int = 2048

    @property
    def width(self): return 64 if self.kind == "mamba" else 128
    @property
    def heads(self): return self.group_values // self.width
    @property
    def chunk_rows(self):
        if not self.phased: return 2 if self.kind == "mamba" else 4
        return 128 if self.broadcast else 32

@dataclass(frozen=True)
class View:
    base: int
    descriptor: Descriptor

def views(o: Options) -> dict[str, View]:
    assert o.kind in ("mamba", "kda") and o.control in ("row", "fsm")
    assert o.group_values == 2048
    h, w, r = o.heads, o.width, o.chunk_rows
    state = View(0, Descriptor(Shape(r,w,h), Map(0, w//32) if o.phased else Map(r)))
    state_rows = r if o.phased else r*h
    cb = state_rows*2048
    scalar_cols = ((2*h+31)//32)*32
    cols = scalar_cols if o.broadcast else 2*h*w
    coeff = View(cb, Descriptor(Shape(r,cols), Map(r, flags=Flag.BROADCAST_MINOR if o.broadcast else Flag(0))))
    vb = cb+scalar_cols if o.broadcast else cb+2*r*2048
    vector = lambda index: View(vb+index*w, Descriptor(Shape(1,w,h), Map(w//32)))
    beta = View(vb+3*w, Descriptor(Shape(1,scalar_cols), Map(1, flags=Flag.BROADCAST_MINOR)))
    result = {"state": state, "coeff": coeff, "input": vector(0), "output": vector(1), "scratch": vector(2), "scalar": beta}
    # Capacity and aliasing are checked against the exact fixed-diagonal map.
    allocations = [MatrixViewAllocation(name, v.base, v.descriptor) for name,v in result.items()]
    validate_disjoint_matrix_views(allocations, mlen=2048, banks=64, bank_width=32, depth_rows=256)
    return result

class Emitter:
    def __init__(self):
        self.lines = []
        self.gp = {}
        self.config = {}

    def address(self, reg, value):
        if self.gp.get(reg) != value:
            self.lines.extend(load_large_int(reg, value))
            self.gp[reg] = value

    def view(self, slot, descriptor):
        words = descriptor.shape.pack(), descriptor.mapping.pack()
        if self.config.get(slot) != words:
            self.address(12, words[0]); self.address(13, words[1])
            self.lines.append(f"L_TILE_CFG {slot}, gp12, gp13")
            self.config[slot] = words

    def dma(self, v, hbm, store=False):
        self.view(3, v.descriptor)
        self.address(9,v.base); self.address(10,hbm)
        self.lines.append(f"H_{'STORE' if store else 'PREFETCH'}_V.MV gp9, gp10, a0, 0, 2, 3")

    def execute(self, primitive, dst, src, coeff, *, row_control=False):
        operands = [dst,src,coeff]
        rows = src.descriptor.shape.rows if primitive in (Op.REDUCE_ACC, Op.DECAY_REDUCE_ACC) else dst.descriptor.shape.rows
        loop = row_control and rows > 1
        strides = []
        for slot, (reg, v) in enumerate(zip((9,10,11), operands)):
            d = v.descriptor
            stride = 2048*((d.shape.cols+2047)//2048) if d.shape.rows > 1 else 0
            if loop and stride:
                d = replace(d, shape=replace(d.shape, rows=1))
            self.view(slot,d); self.address(reg,v.base)
            strides.append(stride)
        if loop: self.lines.append(f"C_LOOP_START gp15, {rows}")
        self.lines.append(f"L_TILE_EXEC gp9, gp10, gp11, {int(primitive)}")
        if loop:
            for reg, stride in zip((9,10,11),strides):
                if stride:
                    self.lines.append(f"S_ADDI_INT gp{reg}, gp{reg}, {stride}")
                    self.gp.pop(reg,None)
            self.lines.append("C_LOOP_END gp15")

def lower_group(o: Options, memory: dict, emitter: Emitter | None = None) -> Emitter:
    """Lower one head group/token using caller-owned, private HBM addresses.

    memory: states/update/dot lists (one entry per row chunk), input/scalar/
    skip/output addresses, optional snapshots list. All data is BF16 and all
    arrays are prepacked to the selected views. No dynamic queues or models
    are encoded in the hardware instruction.
    """
    e = emitter or Emitter()
    v = views(o)
    state, coeff, x, out, scratch, scalar = (v[n] for n in ("state","coeff","input","output","scratch","scalar"))
    chunks = 128//o.chunk_rows
    assert len(memory["states"]) == len(memory["update"]) == len(memory["dot"]) == chunks
    row = o.control == "row"
    def run(op,d,s,c): e.execute(op,d,s,c,row_control=row)
    def saved(chunk):
        e.dma(state,memory["states"][chunk],True)
        if "snapshots" in memory: e.dma(state,memory["snapshots"][chunk],True)

    e.dma(x,memory["input"])
    e.dma(scalar,memory["scalar"])
    if o.kind == "mamba":
        # BF16(dt*x), then fused state update, BF16 final output and skip.
        run(Op.SCALE_ACCUM,scratch,x,scalar)
        run(Op.REDUCE_BEGIN,out,out,scalar)
        for chunk in range(chunks):
            e.dma(state,memory["states"][chunk])
            e.dma(coeff,memory["update"][chunk])
            run(Op.DELTA_UPDATE,state,scratch,coeff)
            if not o.resident:
                saved(chunk); e.dma(state,memory["states"][chunk])
            e.dma(coeff,memory["dot"][chunk])
            run(Op.REDUCE_ACC,out,state,coeff)
            if o.resident: saved(chunk)
        run(Op.REDUCE_WRITE,out,out,scalar)
        e.dma(scalar,memory["skip"])
        run(Op.SCALE_ACCUM,out,x,scalar)
    else:
        # Prediction consumes the original BF16 tile without a decayed BF16
        # intermediate; residual is BF16 before the second state scan.
        run(Op.REDUCE_BEGIN,out,out,scalar)
        for chunk in range(chunks):
            e.dma(state,memory["states"][chunk])
            e.dma(coeff,memory["update"][chunk])
            run(Op.DECAY_REDUCE_ACC,out,state,coeff)
        run(Op.RESIDUAL_WRITE,scratch,x,scalar)
        run(Op.REDUCE_BEGIN,out,out,scalar)
        for chunk in range(chunks):
            if not (o.resident and chunks == 1):
                e.dma(state,memory["states"][chunk])
                e.dma(coeff,memory["update"][chunk])
            run(Op.DELTA_UPDATE,state,scratch,coeff)
            if not o.resident:
                saved(chunk); e.dma(state,memory["states"][chunk])
            e.dma(coeff,memory["dot"][chunk])
            run(Op.REDUCE_ACC,out,state,coeff)
            if o.resident: saved(chunk)
        run(Op.REDUCE_WRITE,out,out,scalar)
    e.dma(out,memory["output"],True)
    return e
