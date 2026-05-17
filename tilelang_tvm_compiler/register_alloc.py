"""Tiny free-list register allocator with optional GP spill to IntRAM.

ISAEmitter calls into us mid-emit to get scratch registers and returns
them when the instruction sequence is finished:

    gp_regs = compiler.register_allocator.allocate_gp(5)   # list[int]
    ... emit ISA using gp{gp_regs[0]}, gp{gp_regs[1]}, ...
    compiler.register_allocator.free_gp(gp_regs)

When the free GP pool can't satisfy a request, the runtime can fall
back to a *borrow scope*: temporarily move the contents of some
currently-allocated GPs to IntRAM (``S_ST_INT``), reuse those slots
for the request, then restore (``S_LD_INT``) them when the borrow
ends. Use this around a single leaf emit that doesn't read any of the
spilled GPs in its body:

    borrowed, token = ra.spill_borrow(
        6, compiler=program.compiler, protect=[gp_addr_offset_reg]
    )
    # ... emit ISA using borrowed GPs ...
    ra.spill_return(token, compiler=program.compiler)

Pool sizes match the PLENA spec (16 GP, 8 addr) minus a few reserved:
    - gp0 reserved as the constant-zero register
    - addr0..7 all available; runtime convention reserves none

IntRAM spill region:
    - The IntRAM (1024 u32 words) is shared with the user. We reserve
      slots at ``[SPILL_BASE, SPILL_BASE + SPILL_SLOTS)`` for GP
      saves. ``SPILL_BASE`` defaults to 256 to leave headroom for any
      user preload at the start of IntRAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


class RegisterExhausted(RuntimeError):
    pass


# IntRAM regions (units = u32 words; emulator's intsram is sized 1024).
#  [0, 256)            user / preload scratch
#  [SPILL_BASE, ...)   GP auto-spill backing store
#  [IDX_BASE, ...)     loop idx backing store
# Keeping the regions disjoint means a loop's idx in IntRAM can never
# be clobbered by a GP spill done by the loop body.
SPILL_BASE = 256
SPILL_SLOTS = 256
IDX_BASE = 512
IDX_SLOTS = 256


@dataclass
class _SpillRecord:
    orig_reg: int
    slot: int


@dataclass
class BorrowToken:
    """Opaque handle returned from ``spill_borrow``. Pass back to
    ``spill_return`` to release the borrow and reload spilled GPs."""
    borrowed: List[int]
    spilled: List[_SpillRecord] = field(default_factory=list)


class RegisterAllocator:
    def __init__(
        self,
        *,
        gp_total: int = 16,
        addr_total: int = 8,
        gp_reserved: Iterable[int] = (0,),  # gp0 = constant zero
        addr_reserved: Iterable[int] = (),
    ) -> None:
        gp_reserved_set = set(gp_reserved)
        addr_reserved_set = set(addr_reserved)
        self._gp_total = gp_total
        self._gp_free: List[int] = [i for i in range(gp_total) if i not in gp_reserved_set]
        # In-use GPs in allocation order (LIFO spill candidates pulled
        # from the end). Always disjoint from ``_gp_free``.
        self._gp_in_use: List[int] = []
        self._addr_free: List[int] = [i for i in range(addr_total) if i not in addr_reserved_set]
        # Spill slot bitmap.
        self._spill_slots_in_use: List[bool] = [False] * SPILL_SLOTS
        # Idx slot bitmap (loop idx values stored in IntRAM instead of
        # GPs so deep nests don't pin the whole GP file).
        self._idx_slots_in_use: List[bool] = [False] * IDX_SLOTS
        # Late-bound by the shim/compiler so allocate_gp can auto-spill
        # on demand. Stays None for tests that don't wire it up.
        self.compiler = None
        # Each successful auto-spill records (orig_reg, slot) so the
        # matching ``free_gp`` reload-restores from IntRAM.
        self._auto_spills_by_borrow: Dict[int, _SpillRecord] = {}
        # Set of GPs that hold long-lived bindings (loop indices etc.)
        # — never picked as spill candidates. The ISA pass populates
        # this via ``pin_gp`` / ``unpin_gp`` whenever it binds /
        # unbinds a loop var, keeping the symbol_table contents safe
        # from being trashed by auto-spill.
        self._pinned_gp: set = set()
        # GP register event trace, one row per state-changing call.
        # Filled by ``_record(...)`` from every public mutator below.
        # Dumped at end of compile to ``<build_dir>/<asm>.gp_trace.tsv``
        # so the ASM can be cross-referenced line-by-line with the
        # allocation events that produced it.
        self._trace: List[Dict[str, str]] = []
        # Source-site stack pushed by ISA pass / materializer when they
        # want events grouped under a logical operation (e.g. ``op[12]
        # row_reduce_sum_at``, ``materialize Add``). Whatever sits on
        # top here annotates every event until popped.
        self._site_stack: List[str] = []

    # ------------------------------------------------------------------
    # Event trace
    # ------------------------------------------------------------------
    def push_site(self, label: str) -> None:
        """Annotate every subsequent event until the matching ``pop_site``
        with ``label``. Used by ISA pass / materializer to tag events
        with the logical op or expression they came from."""
        self._site_stack.append(label)

    def pop_site(self) -> None:
        if self._site_stack:
            self._site_stack.pop()

    def _site(self) -> str:
        return " > ".join(self._site_stack)

    def _asm_line(self) -> int:
        """Current line count of ``generated_code`` — used as a coarse
        cursor so trace rows can be aligned with the ASM dump."""
        if self.compiler is None:
            return 0
        return self.compiler.generated_code.count("\n") + 1

    def _record(self, event: str, **fields: object) -> None:
        """Append one row to ``self._trace``. Captures the post-mutation
        state (free pool / in-use list / pinned set) so a reader can
        replay the timeline without reconstructing it from deltas."""
        row: Dict[str, object] = {
            "asm_line": self._asm_line(),
            "site":     self._site(),
            "event":    event,
        }
        row.update(fields)
        # Snapshot pool state AFTER the event, so a reader can compare
        # consecutive rows to spot leaks / double-frees / pinning bugs.
        row["free"]   = ",".join(str(r) for r in self._gp_free)
        row["in_use"] = ",".join(str(r) for r in self._gp_in_use)
        row["pinned"] = ",".join(str(r) for r in sorted(self._pinned_gp))
        self._trace.append(row)

    def trace_rows(self) -> List[Dict[str, object]]:
        return list(self._trace)

    # ------------------------------------------------------------------
    # GP register pool
    # ------------------------------------------------------------------
    def allocate_gp(self, n: int) -> List[int]:
        # Auto-spill is allowed only against UNPINNED in-use regs. Loop
        # hw-counters and idx regs are always pinned by `_emit_for`, so
        # they can never be picked as spill victims -- which was the
        # earlier bug (spilled hw counter reloaded too late, after
        # C_LOOP_END had already read garbage). If every in-use reg is
        # pinned, `_auto_spill` itself raises RegisterExhausted -- that
        # is the kernel-author's signal to convert one of the outer
        # `for` loops to `T.unroll(...)` (which doesn't pin
        # gp_loop+gp_idx) so non-loop work has room to spill.
        if n > len(self._gp_free):
            self._auto_spill(n - len(self._gp_free))
        out = self._gp_free[:n]
        self._gp_free = self._gp_free[n:]
        self._gp_in_use.extend(out)
        self._record("allocate_gp", n=n, regs=",".join(str(r) for r in out))
        return out

    def pin_gp(self, reg: int) -> None:
        """Mark ``reg`` as carrying a long-lived value (loop index,
        symbol-table binding etc.) so ``_auto_spill`` never picks it
        as a spill candidate. Spilling such a register would silently
        corrupt the binding because the materializer reads it via
        the symbol table without going through ``free_gp``."""
        self._pinned_gp.add(reg)
        self._record("pin_gp", regs=str(reg))

    def unpin_gp(self, reg: int) -> None:
        self._pinned_gp.discard(reg)
        self._record("unpin_gp", regs=str(reg))

    def _auto_spill(self, need: int) -> None:
        """Free up ``need`` more GPs by spilling the most-recently
        allocated in-use ones to IntRAM. Each spilled GP is recorded
        keyed by its (new) register number so the matching ``free_gp``
        triggers a reload.

        Pinned GPs (loop indices etc.) are never spilled because the
        symbol table refers to them by register number without going
        through ``free_gp`` — spilling them would silently corrupt
        the value the materializer reads back."""
        if self.compiler is None:
            raise RegisterExhausted(
                f"auto-spill required ({need} more GP) but no compiler "
                f"bound on the allocator; call shim.compiler.bind_allocator() "
                f"or pass `compiler=` to the RegisterAllocator constructor"
            )
        candidates: List[int] = []
        for r in reversed(self._gp_in_use):
            if r in self._pinned_gp:
                continue
            candidates.append(r)
            if len(candidates) == need:
                break
        if len(candidates) < need:
            raise RegisterExhausted(
                f"auto-spill: need {need} GPs but only {len(candidates)} "
                f"unpinned in-use to spill; in_use={self._gp_in_use!r} "
                f"pinned={sorted(self._pinned_gp)!r}. "
                f"Hint: every in-use GP is pinned (typically by nested "
                f"`for` loops reserving gp_loop+gp_idx each). Convert one "
                f"of the outer loops to `T.unroll(...)` so it doesn't "
                f"pin two regs, leaving room for non-loop work to spill."
            )
        for r in candidates:
            slot = self._claim_spill_slot()
            addr = SPILL_BASE + slot
            self.compiler.generated_code += (
                f"; auto-spill gp{r} -> intram[{addr}]\n"
                f"S_ST_INT gp{r}, gp0, {addr}\n"
            )
            self._gp_in_use.remove(r)
            self._gp_free.insert(0, r)
            # Record the spill keyed by the register number — when the
            # caller frees this register later we use the same key to
            # reload its contents into the same physical GP.
            self._auto_spills_by_borrow[r] = _SpillRecord(orig_reg=r, slot=slot)
            self._record(
                "auto_spill", regs=str(r), slot=slot, addr=addr,
            )

    def free_gp(self, regs: Iterable[int]) -> None:
        # Push back at the front to maximise locality (next alloc reuses
        # the same register, keeping the live range short and dump
        # human-readable). If this reg was auto-spilled earlier, emit a
        # reload first so the original outer-scope content is restored
        # before anyone re-allocates the register.
        for r in regs:
            if r in self._gp_free:
                # Idempotent free: some callers (e.g. ExprMaterializer
                # tracking both ``register`` and ``intermediates``) may
                # release the same register twice when constant-folding
                # collapsed an intermediate onto the final result reg.
                # Tolerate it instead of crashing.
                self._record("free_gp_noop", regs=str(r))
                continue
            if r in self._gp_in_use:
                self._gp_in_use.remove(r)
            rec = self._auto_spills_by_borrow.pop(r, None)
            if rec is not None:
                addr = SPILL_BASE + rec.slot
                if self.compiler is not None:
                    self.compiler.generated_code += (
                        f"; auto-reload gp{r} <- intram[{addr}]\n"
                        f"S_LD_INT gp{r}, gp0, {addr}\n"
                    )
                self._release_spill_slot(rec.slot)
                # Reg goes back to in-use (its outer-scope owner still
                # holds it). Don't push to free pool.
                self._gp_in_use.append(r)
                self._record(
                    "auto_reload", regs=str(r), slot=rec.slot, addr=addr,
                )
                continue
            self._gp_free.insert(0, r)
            self._record("free_gp", regs=str(r))

    # ------------------------------------------------------------------
    # Spill-borrow API
    # ------------------------------------------------------------------
    def spill_borrow(
        self,
        n: int,
        *,
        compiler,
        protect: Optional[Iterable[int]] = None,
    ) -> Tuple[List[int], BorrowToken]:
        """Borrow ``n`` GP registers, spilling currently-allocated ones
        to IntRAM if necessary. Emits ``S_ST_INT`` lines into
        ``compiler.generated_code`` for every spilled GP. Returns
        ``(borrowed, token)`` — pass ``token`` back to ``spill_return``
        to restore the spilled state.

        ``protect`` is a set of currently-in-use GPs the caller still
        needs to read inside the borrow scope — they are excluded from
        spill candidates and will trigger ``RegisterExhausted`` if
        spilling them is the only way to satisfy the request.
        """
        protect_set = set(protect or ())
        protect_set.discard(0)  # gp0 reserved-zero is never spillable anyway

        need = n - len(self._gp_free)
        spilled: List[_SpillRecord] = []
        if need > 0:
            candidates: List[int] = []
            for r in reversed(self._gp_in_use):
                if r in protect_set:
                    continue
                # Pinned GPs (loop hw counters, long-lived symbol-table
                # bindings) are referenced by register number without
                # going through free_gp, so spilling them would silently
                # corrupt the value. Matches the _auto_spill filter.
                if r in self._pinned_gp:
                    continue
                candidates.append(r)
                if len(candidates) == need:
                    break
            if len(candidates) < need:
                raise RegisterExhausted(
                    f"spill_borrow: need to spill {need} GP(s) but only "
                    f"{len(candidates)} are unprotected; in_use="
                    f"{self._gp_in_use!r} protect={sorted(protect_set)!r} "
                    f"pinned={sorted(self._pinned_gp)!r}"
                )
            for r in candidates:
                slot = self._claim_spill_slot()
                addr = SPILL_BASE + slot
                compiler.generated_code += (
                    f"; spill gp{r} -> intram[{addr}]\n"
                    f"S_ST_INT gp{r}, gp0, {addr}\n"
                )
                spilled.append(_SpillRecord(orig_reg=r, slot=slot))
                self._gp_in_use.remove(r)
                self._gp_free.insert(0, r)
                self._record(
                    "borrow_spill", regs=str(r), slot=slot, addr=addr,
                )

        borrowed = self.allocate_gp(n)
        self._record("spill_borrow", n=n,
                     regs=",".join(str(r) for r in borrowed),
                     spilled=",".join(str(s.orig_reg) for s in spilled))
        return borrowed, BorrowToken(borrowed=borrowed, spilled=spilled)

    def spill_return(self, token: BorrowToken, *, compiler) -> None:
        """End a borrow scope: free the borrowed GPs, re-allocate the
        previously spilled GPs at their original register numbers, and
        emit ``S_LD_INT`` to restore their contents from IntRAM."""
        self.free_gp(token.borrowed)
        for rec in token.spilled:
            if rec.orig_reg in self._gp_free:
                self._gp_free.remove(rec.orig_reg)
            else:
                raise RuntimeError(
                    f"spill_return: expected gp{rec.orig_reg} to be free "
                    f"(no one else may take it during a borrow scope), but "
                    f"free pool is {self._gp_free!r}"
                )
            self._gp_in_use.append(rec.orig_reg)
            addr = SPILL_BASE + rec.slot
            compiler.generated_code += (
                f"; reload gp{rec.orig_reg} <- intram[{addr}]\n"
                f"S_LD_INT gp{rec.orig_reg}, gp0, {addr}\n"
            )
            self._release_spill_slot(rec.slot)
            self._record(
                "borrow_reload", regs=str(rec.orig_reg),
                slot=rec.slot, addr=addr,
            )

    def _claim_spill_slot(self) -> int:
        for i, used in enumerate(self._spill_slots_in_use):
            if not used:
                self._spill_slots_in_use[i] = True
                return i
        raise RegisterExhausted(
            f"spill slots exhausted ({SPILL_SLOTS} used). Bump SPILL_SLOTS "
            f"or reduce simultaneous register pressure."
        )

    def _release_spill_slot(self, slot: int) -> None:
        if not self._spill_slots_in_use[slot]:
            raise RuntimeError(f"double-release of spill slot {slot}")
        self._spill_slots_in_use[slot] = False

    # ------------------------------------------------------------------
    # Loop idx slot pool (IntRAM-backed loop indices). Disjoint from
    # spill slots so a body's GP spill can't clobber an outer loop's idx.
    # ------------------------------------------------------------------
    def claim_idx_slot(self) -> int:
        """Allocate an IntRAM word for a loop's idx. Returns the
        absolute IntRAM address (suitable for `S_LD_INT gp, gp0, addr`).
        """
        for i, used in enumerate(self._idx_slots_in_use):
            if not used:
                self._idx_slots_in_use[i] = True
                self._record("claim_idx_slot", slot=i, addr=IDX_BASE + i)
                return IDX_BASE + i
        raise RegisterExhausted(
            f"idx slots exhausted ({IDX_SLOTS} used). Bump IDX_SLOTS or "
            f"reduce simultaneous loop nesting depth."
        )

    def release_idx_slot(self, addr: int) -> None:
        slot = addr - IDX_BASE
        if not (0 <= slot < IDX_SLOTS):
            raise RuntimeError(f"release_idx_slot: addr {addr} out of range")
        if not self._idx_slots_in_use[slot]:
            raise RuntimeError(f"double-release of idx slot {slot}")
        self._idx_slots_in_use[slot] = False
        self._record("release_idx_slot", slot=slot, addr=addr)

    # ------------------------------------------------------------------
    # Address register pool
    # ------------------------------------------------------------------
    def allocate_addr(self, n: int) -> List[int]:
        if n > len(self._addr_free):
            raise RegisterExhausted(
                f"requested {n} addr registers but only {len(self._addr_free)} free"
            )
        out = self._addr_free[:n]
        self._addr_free = self._addr_free[n:]
        self._record(
            "allocate_addr", n=n, regs=",".join(f"a{r}" for r in out),
        )
        return out

    def free_addr(self, regs: Iterable[int]) -> None:
        for r in regs:
            if r in self._addr_free:
                raise RuntimeError(f"double-free of a{r}")
            self._addr_free.insert(0, r)
            self._record("free_addr", regs=f"a{r}")


__all__ = [
    "RegisterAllocator",
    "RegisterExhausted",
    "BorrowToken",
    "SPILL_BASE",
    "SPILL_SLOTS",
]
