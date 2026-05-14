"""End-to-end driver: raw TIR PrimFunc -> real PLENA ISA text.

Orchestrates:
    0. inline_let_stmts + lower_compound_fp_stores  (stmt prep)
    1. mid_ir pipeline (10 passes, see frontend/mid_ir/passes/)
    2. AddressAllocationPass                         (HLIR + addresses)
    3. IsaEmitterPass                                (HLIR -> ISA text)

The legacy ``frontend/`` graph-IR pipeline + ``codegen.PlenaCodegen``
are no longer in the call path. They're still on disk for reference
but aren't imported here.

Hardware constants for the program shim are passed in via PlenaTarget,
which we keep deliberately small for now -- mlen/blen/btmm shape are
fixed per chip variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tvm
from tvm import tir

from .address_alloc import AddressAllocationPass, AddressAllocConfig
from . import dead_buffer_elim as _dead_buffer_elim
# Direct submodule imports to avoid the legacy frontend package's
# __init__ (which imports compile_func → frontend/pipeline.py →
# ..pipeline.PlenaTarget, a circular import once we land here).
from .frontend.passes import inline_let_stmts as _stmt_inline_let
from .frontend.passes import lower_compound_fp_stores as _stmt_lower_compound
from .frontend.mid_ir.passes import infer_lane_axis as _mid_infer_lane_axis
from .frontend.mid_ir.passes import fold as _mid_fold
from .frontend.mid_ir.passes import mark as _mid_mark
from .frontend.mid_ir.passes import split as _mid_split
from .frontend.mid_ir.passes import distribute_cluster as _mid_distribute
from .frontend.mid_ir.passes import async_wrap as _mid_async
from .frontend.mid_ir.passes import view as _mid_view
from .frontend.mid_ir.passes import fuse as _mid_fuse
from .frontend.mid_ir.passes import burn_view as _mid_burn
from .frontend.mid_ir.passes import to_plena as _mid_to_plena
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
    # GP allocator trace captured during ISA emit. ``None`` if the
    # compile path didn't expose it. Each entry is a dict with keys
    # ``asm_line``/``site``/``event``/``free``/``in_use``/``pinned``
    # plus event-specific fields (regs, slot, addr, n, ...).
    gp_trace: list = None

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
    midir_dump_dir: Optional[Path] = None,
) -> CompiledKernel:
    """Lower a raw TIR PrimFunc through the mid_ir pipeline + downstream
    address-alloc + ISA-emit passes.

    ``midir_dump_dir`` (when set): pass_6_to_plena will write a
    human-readable ``<name>.midir.txt`` snapshot there for debugging.
    """
    # ---------- 0. stmt prep ----------
    func = _stmt_inline_let.run(prim_func)
    func = _stmt_lower_compound.run(func)

    # ---------- 1. mid_ir pipeline ----------
    func = _mid_infer_lane_axis.run(func)
    midfn = _mid_fold.run(func, name=name)
    midfn = _mid_mark.run(midfn)
    midfn = _mid_split.run(midfn)
    midfn = _mid_distribute.run(midfn)
    midfn = _mid_async.run(midfn)
    midfn = _mid_view.run(midfn)
    midfn = _mid_fuse.run(midfn)
    midfn = _mid_burn.run(midfn)
    mod = _mid_to_plena.run(midfn, build_dir=midir_dump_dir, mlen=target.mlen)

    # DEBUG: dump HLIR immediately after to_plena so we can inspect it
    # even when later passes fail.
    if midir_dump_dir is not None:
        from .hlir import format_hlir as _fmt
        (midir_dump_dir / "post_to_plena.hlir.txt").write_text(_fmt(mod))

    # ---------- 1.5. drop unreachable buffers ----------
    # Buffers declared in the kernel but not referenced by any HLIR op
    # (e.g. softmax-state fragments in a stub kernel that bypasses
    # softmax) would otherwise waste FPRAM/VRAM and can also crash
    # downstream shape checks if their post-expansion layout doesn't
    # match the lane mode that was never inferred.
    _dead_buffer_elim.run(mod)

    # ---------- 2. address alloc ----------
    addr_pass = AddressAllocationPass(AddressAllocConfig(
        mlen=target.mlen,
        blen=target.blen,
        hlen=target.btmm_hlen,
    ))
    addr_pass.run(mod)

    # ---------- 3. ISA emit ----------
    allocator = RegisterAllocator()
    shim = make_shim(
        mlen=target.mlen,
        blen=target.blen,
        btmm_lane_count=target.btmm_lane_count,
        btmm_hlen=target.btmm_hlen,
        register_allocator=allocator,
    )
    isa_pass = IsaEmitterPass(shim)
    isa_text = isa_pass.run(mod)

    return CompiledKernel(
        name=name, hlir=mod, isa_text=isa_text,
        gp_trace=allocator.trace_rows(),
    )


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
