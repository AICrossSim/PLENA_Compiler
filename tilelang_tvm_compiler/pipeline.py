"""End-to-end driver: TIR PrimFunc -> real PLENA ISA text.

Orchestrates the three passes:
    1. PlenaCodegen.lower_to_hlir   (TIR -> HLIR)
    2. AddressAllocationPass         (HLIR + addresses)
    3. IsaEmitterPass                (HLIR -> ISA text)

Hardware constants for the program shim are passed in via PlenaTarget,
which we keep deliberately small for now -- mlen/blen/btmm shape are
fixed per chip variant.
"""

from __future__ import annotations

from dataclasses import dataclass

import tvm
from tvm import tir

from .address_alloc import AddressAllocationPass, AddressAllocConfig
from .codegen import PlenaCodegen
from .hlir import HLIRModule
from .isa_pass import IsaEmitterPass
from .program_shim import make_shim
from .register_alloc import RegisterAllocator


@dataclass
class PlenaTarget:
    """Hardware-shape constants. Equivalent to TileTensorProgram() ctor."""

    mlen: int = 64
    blen: int = 4
    btmm_lane_count: int = 4   # group_heads
    btmm_hlen: int = 16        # head dim per BTMM lane


@dataclass
class CompiledKernel:
    name: str
    hlir: HLIRModule
    isa_text: str

    def __repr__(self) -> str:
        return (
            f"CompiledKernel(name={self.name!r}, "
            f"buffers={len(self.hlir.buffers)}, "
            f"ops={len(self.hlir.ops)}, "
            f"isa_lines={self.isa_text.count(chr(10))})"
        )


def compile_kernel(
    prim_func: tir.PrimFunc,
    *,
    target: PlenaTarget,
    name: str = "kernel",
) -> CompiledKernel:
    # Pass 1
    cg = PlenaCodegen(prim_func, name=name)
    mod = cg.lower_to_hlir()

    # Pass 2
    addr_pass = AddressAllocationPass(AddressAllocConfig(
        mlen=target.mlen,
        blen=target.blen,
    ))
    addr_pass.run(mod)

    # Pass 3
    shim = make_shim(
        mlen=target.mlen,
        blen=target.blen,
        btmm_lane_count=target.btmm_lane_count,
        btmm_hlen=target.btmm_hlen,
        register_allocator=RegisterAllocator(),
    )
    isa_pass = IsaEmitterPass(shim)
    isa_text = isa_pass.run(mod)

    return CompiledKernel(name=name, hlir=mod, isa_text=isa_text)


def compile_module(
    mod: tvm.IRModule,
    *,
    target: PlenaTarget,
) -> dict:
    out = {}
    for gv, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue
        out[gv.name_hint] = compile_kernel(func, target=target, name=gv.name_hint)
    return out


__all__ = ["PlenaTarget", "CompiledKernel", "compile_kernel", "compile_module"]
