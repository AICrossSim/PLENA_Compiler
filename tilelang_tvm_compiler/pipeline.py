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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tvm
from tvm import tir

from .address_alloc import AddressAllocationPass, AddressAllocConfig
from . import dead_buffer_elim as _dead_buffer_elim
from . import fuse_adjacent_loops as _fuse_adjacent_loops
from . import loop_interchange as _loop_interchange
from . import loop_register_alloc as _loop_register_alloc
from . import plena_settings as _plena_settings
# Direct submodule imports to avoid the legacy frontend package's
# __init__ (which imports compile_func → frontend/pipeline.py →
# ..pipeline.PlenaTarget, a circular import once we land here).
from .frontend.passes import inline_let_stmts as _stmt_inline_let
from .frontend.passes import fission_vector_chains as _stmt_fission_vec
from .frontend.passes import lower_compound_fp_stores as _stmt_lower_compound
from .frontend.passes import hoist_float_constants as _stmt_hoist_consts
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
    """Hardware-shape constants. Equivalent to TileTensorProgram() ctor.

    Defaults are read from ``plena_settings.toml`` (the active mode's
    MLEN / HLEN / BLEN) so the compiler and the simulator never drift.
    Pass explicit values to override for a non-default target / test.
    """

    mlen: int = field(default_factory=_plena_settings.mlen)
    blen: int = field(default_factory=_plena_settings.blen)
    # group_heads — how many narrow heads pack into one MLEN vector.
    btmm_lane_count: int = field(
        default_factory=lambda: _plena_settings.load_sizes().hardware_lane_count
    )
    btmm_hlen: int = field(default_factory=_plena_settings.hlen)


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
    # lowir recording: ``(op_idx, expr_str)`` pairs captured at the
    # var->gp materialization chokepoint during ISA emit. Feeds the
    # ``<kernel>.lowir.txt`` report — the symbolic "last variable-form"
    # of every address expression the ISA actually consumes. Empty list
    # if not recorded.
    lowir_log: list = None

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
    addr_config_override: Optional[AddressAllocConfig] = None,
    use_v2: bool = True,
) -> CompiledKernel:
    """Lower a raw TIR PrimFunc through the mid_ir pipeline + downstream
    address-alloc + ISA-emit passes.

    ``midir_dump_dir`` (when set): pass_6_to_plena will write a
    human-readable ``<name>.midir.txt`` snapshot there for debugging.

    ``addr_config_override`` (when set): use this AddressAllocConfig
    verbatim for the address-alloc pass instead of building a default
    one from ``target``. Used by multi-kernel drivers that stitch
    several kernels into one continuous ASM run and need to control
    the FPRAM / HBM bases per kernel.

    ``use_v2`` (default True): when True, route the post-address HLIR
    through the PreIsaPassV2 → MIR → ISA pipeline instead of the legacy
    single-pass ``IsaEmitterPass``. midir is now the default compile
    path; pass ``use_v2=False`` to fall back to the legacy emitter. The v2 path is fully op-coverage-
    complete (all 38 HLIR op kinds) and produces structurally
    identical HW-op streams (same M_MM/M_BTMM/H_PREFETCH_V/etc.
    count and order); GP numbers can differ. Set this when you want
    the v2 path's tighter register allocation (~9 GPs on full matmul
    vs 14+ in legacy) and the MIR-level dump for debugging.

    Unroll loops in v2: the MIR is kept compact (no physical
    expansion); ``mir_to_isa._emit_loop_unroll`` clones each iter's
    body into a throwaway scratch block, substitutes the loop_var
    to a plain int, runs scratch-local constant folding, then emits
    the folded result. This collapses per-iter address arithmetic
    without inflating the MIR, and is consistently shorter ISA than
    the previous "physically unroll then re-fold" strategy because
    LICM's hoisted invariants survive (they aren't duplicated per
    unroll iter).
    """
    # ---------- 0. stmt prep ----------
    func = _stmt_inline_let.run(prim_func)
    # Fission compound vector (VRAM) chains into one-op-per-loop form
    # before anything else looks at the IR. Activation kernels compute
    # directly on 2D shared buffers; this rewrite is what makes their
    # nested RHS legal for the single-op mid_ir fold. No-op for the
    # rank-1 FPRAM path that lower_compound_fp_stores handles next.
    func = _stmt_fission_vec.run(func)
    # DEBUG: dump the TIR right after fission (one-op-per-loop form) so we
    # can inspect the pre-mid_ir loop structure.
    if midir_dump_dir is not None:
        (midir_dump_dir / "post_fission.tir.txt").write_text(str(func))
    func = _stmt_lower_compound.run(func)
    # Hoist FP literals (T.float16(c) etc.) into auto-synthesised
    # ``global.fpram`` 1-slot buffers so the kernel author doesn't have
    # to declare a SCALE / NEG_INF / etc. fragment + a testbench
    # preload by hand. See hoist_float_constants.py for the contract.
    func = _stmt_hoist_consts.run(func)

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

    # ---------- 1.25. loop interchange + fusion to a fixed point ----------
    # to_plena lowers each per-lane op into its own for-loop. Two
    # structural passes alternate until the IR stops changing:
    #   * loop_interchange — lifts a cluster ``for`` out of an enclosing
    #     loop so it becomes a sibling of other cluster loops;
    #   * fuse_adjacent_loops — merges adjacent same-shape loops.
    # Alternating both to a fixed point lets interchange expose a fusion
    # opportunity, fusion expose a further interchange, and so on.
    # Structural-only — runs before address allocation. The iteration
    # cap is a safety net; convergence is monotone (each step strictly
    # reduces loop count or nesting) so it terminates well before it.
    for _ in range(64):
        mod, _ic_changed = _loop_interchange.run(mod)
        mod, _fu_changed = _fuse_adjacent_loops.run(mod)
        if not (_ic_changed or _fu_changed):
            break

    # ---------- 1.5. drop unreachable buffers ----------
    # Buffers declared in the kernel but not referenced by any HLIR op
    # (e.g. softmax-state fragments in a stub kernel that bypasses
    # softmax) would otherwise waste FPRAM/VRAM and can also crash
    # downstream shape checks if their post-expansion layout doesn't
    # match the lane mode that was never inferred.
    _dead_buffer_elim.run(mod)

    # ---------- 2. address alloc ----------
    if addr_config_override is not None:
        addr_cfg = addr_config_override
    else:
        addr_cfg = AddressAllocConfig(
            mlen=target.mlen,
            blen=target.blen,
            hlen=target.btmm_hlen,
        )
    addr_pass = AddressAllocationPass(addr_cfg)
    addr_pass.run(mod)

    # ---------- 2.5. loop-register allocation ----------
    # Assign each serial ``for`` loop's C_LOOP counter (gp_loop) a GP by
    # HLIR liveness, stamping it on the op. The returned set is reserved
    # away from the emit-stage allocator so per-op temporaries can never
    # collide with a loop counter. See doc/LOOP_REGISTER_ALLOC.md.
    loop_reserved_gp = _loop_register_alloc.run(mod)

    # ---------- 3. ISA emit ----------
    allocator = RegisterAllocator(
        gp_reserved=(0, *sorted(loop_reserved_gp)),
    )
    shim = make_shim(
        mlen=target.mlen,
        blen=target.blen,
        btmm_lane_count=target.btmm_lane_count,
        btmm_hlen=target.btmm_hlen,
        v_prefetch_amount=_plena_settings.v_prefetch_amount(),
        v_writeback_amount=_plena_settings.v_writeback_amount(),
        register_allocator=allocator,
    )
    if use_v2:
        # v2 path: HLIR → PreIsaIR v2 → MIR → opt pipeline → ISA.
        # PreIsaPassV2 still delegates layout/offset helpers to a
        # legacy IsaEmitterPass instance, but the *visible* ISA
        # output here comes from the v2 MIR emit.
        from .pre_isa_pass_v2 import PreIsaPassV2
        from . import pre_isa_to_mir as _p2m
        from . import mir as _mir
        from . import mir_to_isa as _m2i
        from . import mir_passes as _mp
        pre = PreIsaPassV2(shim).run(mod)
        mir_fn = _p2m.convert(pre, shim)
        _mir.verify(mir_fn)
        # Per-pass MIR dump (debugging): one .mir file per pass under
        # ``<midir_dump_dir>/mir_passes/`` so each address-PrimExpr fold
        # is visible step by step.
        _mir_dump_dir = (
            (midir_dump_dir / "mir_passes") if midir_dump_dir else None
        )
        # DLE + const-fold + DCE + CSE to a fixed point. Reduces
        # GP pressure (peels extent-1 loops, folds static
        # address arithmetic, deduplicates repeated bases).
        _mp.run_default_pipeline(
            mir_fn, enable_licm=True, dump_dir=_mir_dump_dir,
        )
        _mir.verify(mir_fn)
        isa_text = _m2i.emit(mir_fn, shim)
        return CompiledKernel(
            name=name, hlir=mod, isa_text=isa_text,
            gp_trace=allocator.trace_rows(),
            lowir_log=[],
        )

    isa_pass = IsaEmitterPass(shim)
    # Record symbolic address expressions for the lowir report. Enabled
    # before run() so the recorder captures the real emit pass — no
    # second codegen pass, no drift from the actual ISA.
    isa_pass.materializer.enable_lowir_log()
    isa_text = isa_pass.run(mod)

    return CompiledKernel(
        name=name, hlir=mod, isa_text=isa_text,
        gp_trace=allocator.trace_rows(),
        lowir_log=list(isa_pass.materializer.lowir_log()),
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
