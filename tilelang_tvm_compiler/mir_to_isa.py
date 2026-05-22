"""MIR → ISA text emitter.

A mechanical lowering pass: walk the MirFunction, allocate physical
GP / addr registers to each SSA MirValue, then emit one line of
PLENA ISA text per non-meta MirInstr.

This is the FIRST version — it uses a TRIVIAL register allocator:
every i32 MirValue gets a fresh GP from the free pool; addr_reg
values get a fresh ``aN`` slot. Values are never released → if a
kernel uses more than 16 GP-resident values the emit fails. That's
fine for the POC; the real register allocator (with live-range
analysis + IntRAM spill) is a follow-up pass that decorates each
MirValue with a "physical home" annotation before this emit runs.

The emit dispatches on ``mir.OPCODES[opcode].isa_mnemonic`` and
formats the operands per opcode-specific rules.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

from . import mir


class MirToIsaError(RuntimeError):
    pass


# ---------------------------------------------------------------------
# Physical register state
# ---------------------------------------------------------------------

# GP file: gp0 is the constant-zero source (hardware-fixed); gp15 is
# reserved for the serial loop counter (matching the legacy
# convention); gp1..gp15 are user-allocatable. Serial loop counters
# borrow from this same pool — the emit layer mints a fresh MirValue
# per serial loop and pins its GP for the loop's lifetime, then
# releases it on exit. Nested serial loops therefore consume one
# extra GP per nesting level (the loop counter itself is dead inside
# the body — only the IntRAM-backed lvar gets read/written there).
GP_TOTAL = 16
GP_USER_FIRST = 1
GP_USER_LAST = 15


def compute_emit_order(fn: "mir.MirFunction") -> Dict[int, int]:
    """Assign every ``MirInstr`` a stable emit index by a DFS walk that
    visits items in the SAME order ``MirToIsa`` emits them.

    Returns ``{id(MirInstr): emit_idx}``. The emit walk
    (:meth:`MirToIsa._emit_instr`) reads each instr's pre-assigned index
    from this map instead of a running counter, so the interval table
    (keyed off these indices) and the release driver can never drift —
    that drift was the root of the register-corruption bugs.
    """
    order: Dict[int, int] = {}
    counter = [0]

    def _walk(items):
        for item in items:
            if isinstance(item, mir.MirInstr):
                order[id(item)] = counter[0]
                counter[0] += 1
            elif isinstance(item, mir.MirLoop):
                for blk in item.body:
                    _walk(blk.items)

    for blk in fn.blocks:
        _walk(blk.items)
    return order


def _check_no_iter_args(fn: "mir.MirFunction") -> None:
    """Guard: this allocator handles only the induction variable as a
    loop-carried SSA value. A loop body block with MORE THAN ONE block
    argument means real ``iter_args`` (loop-carried accumulators kept in
    SSA/GP form), which need phi-congruence register allocation (the
    block-arg and the corresponding yield value must share a register,
    pinned across the loop). That is NOT implemented — and today's MIR
    can't even express it (MirLoop has no yield). Fail loud if it ever
    appears, so the unsupported case can never silently miscompile.
    """
    def _walk(items):
        for item in items:
            if isinstance(item, mir.MirLoop):
                for blk in item.body:
                    if len(blk.arguments) > 1:
                        raise MirToIsaError(
                            f"loop {item.name!r} body block "
                            f"{blk.name!r} has {len(blk.arguments)} block "
                            f"arguments {[a.name for a in blk.arguments]}; "
                            f"only the induction variable is supported. "
                            f"Extra args are iter_args (loop-carried SSA "
                            f"values) requiring phi-congruence register "
                            f"allocation, which is not implemented."
                        )
                    _walk(blk.items)
    for blk in fn.blocks:
        _walk(blk.items)


def _compute_live_intervals(
    fn: "mir.MirFunction", emit_order: Dict[int, int],
    loop_last_idx_out: Optional[Dict[int, int]] = None,
    carried_by_loop_out: Optional[Dict[int, set]] = None,
) -> Dict[int, int]:
    """Structured (SCF-aware) live-interval ends for every i32 MirValue.

    A value's interval is ``[def_point, end]`` in ``emit_order`` index
    space. ``end`` is the point past which the value is provably dead,
    so the allocator may recycle its GP once ``cur_idx > end``.

    SCF rule — the whole point of this pass:

      ``end[v] = max over each use u of v:
                   emit_idx[u], lifted OUTWARD through every loop that
                   encloses u but NOT v's definition, up to that loop's
                   LAST body emit index.``

    Because a serial loop's single emitted body runs N times at runtime,
    any value defined outside a loop but read inside it must stay live
    across the whole loop region — hence the outward lift. A value
    defined and used entirely within one scope gets the plain textual
    last use. The loop_var / loop counter need no special-casing: their
    def sits at the loop head and their uses are inside, so the rule
    extends them to the loop's end automatically.

    ``gp0`` (function const) is omitted — the allocator pins it to
    register 0 and never recycles it.

    ``carried_by_loop_out`` (when given): filled with
    ``{id(MirLoop): {id(MirValue) live-in to that loop}}`` — values
    defined OUTSIDE a loop but used INSIDE it. These are LOOP-CARRIED:
    they must hold the same GP across every runtime re-entry, so the
    allocator must never spill them while that loop is open (a spill
    emitted in the single-pass body is not replayed on the back-edge,
    so the next iteration's head would read a clobbered GP). This is
    the structural fact that the region-recursive model expresses as
    "live-in GPs are not lent into the child region".

    Returns ``{id(MirValue): end_idx}``.
    """
    # Pass 1: record each instr's enclosing-loop id stack, each loop's
    # last body emit index, and each value's defining-loop stack.
    instr_loops: Dict[int, Tuple[int, ...]] = {}
    def_loops: Dict[int, Tuple[int, ...]] = {}
    loop_last_idx: Dict[int, int] = {}

    def _walk(items, loop_stack: Tuple[int, ...]):
        for item in items:
            if isinstance(item, mir.MirInstr):
                idx = emit_order[id(item)]
                instr_loops[id(item)] = loop_stack
                for lid in loop_stack:
                    cur = loop_last_idx.get(lid, idx)
                    loop_last_idx[lid] = max(cur, idx)
                if item.result is not None:
                    def_loops[id(item.result)] = loop_stack
            elif isinstance(item, mir.MirLoop):
                lid = id(item)
                inner = loop_stack + (lid,)
                for blk in item.body:
                    # Block arguments (loop_var) are defined inside.
                    for arg in blk.arguments:
                        def_loops[id(arg)] = inner
                    _walk(blk.items, inner)

    for blk in fn.blocks:
        for arg in blk.arguments:
            def_loops[id(arg)] = ()
        _walk(blk.items, ())

    # Pass 2: for every use, extend the value's end through enclosing
    # loops the def is not in, and record those loops as carriers.
    end: Dict[int, int] = {}
    carried: Dict[int, set] = {}

    def _visit_instr(instr: "mir.MirInstr"):
        use_idx = emit_order[id(instr)]
        use_loops = instr_loops[id(instr)]
        for op in instr.operands:
            if not isinstance(op, mir.MirValue) or op.is_function_const:
                continue
            def_set = def_loops.get(id(op), ())
            candidate = use_idx
            for lid in use_loops:
                if lid in def_set:
                    continue
                # ``op`` is defined outside ``lid`` but used inside it →
                # loop-carried into ``lid``.
                candidate = max(candidate, loop_last_idx.get(lid, use_idx))
                carried.setdefault(lid, set()).add(id(op))
            if candidate > end.get(id(op), -1):
                end[id(op)] = candidate

    def _walk2(items):
        for item in items:
            if isinstance(item, mir.MirInstr):
                _visit_instr(item)
            elif isinstance(item, mir.MirLoop):
                for blk in item.body:
                    _walk2(blk.items)

    for blk in fn.blocks:
        _walk2(blk.items)
    if loop_last_idx_out is not None:
        loop_last_idx_out.update(loop_last_idx)
    if carried_by_loop_out is not None:
        carried_by_loop_out.update(carried)
    return end


class _LinearScanAllocator:
    """Linear-scan register allocator over STRUCTURED live intervals,
    with IntRAM spill-on-pressure.

    Two pre-passes (``compute_emit_order`` + ``_compute_live_intervals``)
    assign every instr a stable emit index and every value an interval
    END in that index space. The end is SCF-aware: a value defined
    outside a loop but used inside it has its end lifted to the loop's
    last body index, so it stays live across every runtime re-entry of
    the serially-emitted body. The emit walker reads each instr's
    pre-assigned index (NOT a running counter — that drifted whenever
    emit appended ISA the pre-pass never walked, which was the root of
    the register-corruption bugs) and calls ``release_dead_at(idx)``;
    values whose end ≤ idx have their GP returned to the free pool.

    No data-value pinning: loop-carried values survive purely via their
    extended interval. The ONLY pins are the serial-loop counter and
    lvar, whose GPs are named raw in the C_LOOP prologue/epilogue and so
    must not move while the loop is open (``pin`` / ``unpin``).

    When ``assign_i32`` is called with the free pool empty, the
    allocator spills the live value with the farthest interval end to
    a fresh IntRAM slot. The spilled value's GP is freed and handed
    to the new request; emit-layer ``_fmt_operand`` calls into the
    allocator to reload the spilled value back into a GP on demand.

    The allocator emits ``S_ST_INT`` (spill) / ``S_LD_INT`` (reload)
    instrs by appending to a caller-owned list passed at
    construction time (``isa_lines``), so the surrounding emit
    layer doesn't need to know spill happened — except that we
    warn via ``warnings.warn`` once per function so the user knows
    register pressure was high.

    IntRAM slots: we use slot 0..MAX_INTRAM_SLOTS-1, allocated
    monotonically as values get spilled. The first
    :class:`MAX_LOOP_INTRAM_SLOTS` are reserved by emit for
    serial-loop idx slots — see ``_emit_loop_serial``.
    """

    def __init__(self, fn: "mir.MirFunction") -> None:
        self.fn = fn
        self.emit_order = compute_emit_order(fn)
        # Structured live-interval ENDS: ``{id(MirValue): end_idx}``.
        # A value is dead (its GP recyclable) once ``cur_idx > end``.
        # SCF-aware: loop-carried values already extend to their loop's
        # end, so NO manual pinning is needed.
        # ``loop_last_idx`` maps id(MirLoop) → its last body emit index;
        # emit uses it to give serial-loop counters a correct interval.
        # Reject loop-carried SSA values (iter_args) we can't handle.
        _check_no_iter_args(fn)
        self.loop_last_idx: Dict[int, int] = {}
        # ``carried_by_loop[id(MirLoop)]`` = set of value-ids live-in to
        # that loop (defined outside, used inside). While that loop is
        # open during emit, these must not be spilled — see
        # ``_pick_spill_victim`` / ``enter_loop`` / ``exit_loop``.
        self.carried_by_loop: Dict[int, set] = {}
        self.end = _compute_live_intervals(
            fn, self.emit_order, self.loop_last_idx,
            self.carried_by_loop,
        )
        # id(MirValue) -> the MirValue, for human-readable diagnostics
        # (the gp maps are keyed by id only).
        self._vid_to_value: Dict[int, "mir.MirValue"] = {}
        # Free GP pool — LIFO so freshly-freed GPs come back first
        # (best for live-locality + dump readability).
        self.free_gp: List[int] = list(
            range(GP_USER_FIRST, GP_USER_LAST + 1),
        )
        self.value_to_gp: Dict[int, int] = {}
        self.value_to_areg: Dict[int, int] = {}
        self._next_areg: int = 0
        # Manual interval ends for values minted DURING emit (the serial
        # loop counter), which the pre-pass never walked. emit calls
        # ``set_end`` for these. Merged into ``end`` lookups.
        self._manual_end: Dict[int, int] = {}
        # Values whose physical GP is referenced by RAW NUMBER in emitted
        # ISA outside the allocator's operand path (serial-loop counter +
        # lvar, used directly in the C_LOOP_START/END + idx prologue/
        # epilogue). These must never be spilled or recycled while the
        # loop is open, or the raw GP number would point at stale data.
        # This is the ONLY pin concept left; loop-carried *data* values
        # need no pin — their structured interval keeps them live.
        self._pinned: set = set()
        # GPs holding the CURRENT instruction's input operands. While set,
        # ``_pick_spill_victim`` will not steal them: their tokens have
        # already been emitted into this instruction's ISA text, so
        # spilling them (e.g. when assign_i32 allocates the result under
        # register pressure) would silently invalidate the operand the
        # instruction is about to read. Set by emit before assigning the
        # result GP, cleared after. Holds id(MirValue)s.
        self._operand_lock: set = set()
        # Stack of id(MirLoop) for serial loops currently OPEN during
        # emit (pushed in enter_loop, popped in exit_loop). The UNION of
        # their ``carried_by_loop`` sets is the set of values that must
        # not be spilled right now: each is live across a loop body whose
        # single-pass emission won't replay a spill on the back-edge, so
        # spilling it corrupts the next iteration. Maintained incrementally
        # in ``_carried_now`` for O(1) victim checks.
        self._open_loops: List[int] = []
        self._carried_now: set = set()
        # IntRAM spill state. ``value_to_slot`` maps id(MirValue) →
        # IntRAM slot index for values whose GP was reclaimed via
        # spill. They're not in ``value_to_gp`` while spilled; a
        # subsequent operand reference reloads them via
        # ``ensure_in_gp``. ``intram_next_slot`` is the next free
        # IntRAM slot — emit reserves slots 0..N-1 for serial-loop
        # counters (see ``reserve_intram_slot`` / serial loop
        # epilogue logic), so this counter starts above the
        # reservation.
        # IntRAM spill state — the SINGLE spill mechanism. ``value_to_slot``
        # maps id(MirValue) → its IntRAM slot. A spilled value lives in
        # IntRAM; on each use it is reloaded into a throwaway GP and then
        # promptly returned to IntRAM (its slot entry is RETAINED across
        # the reload). This "reload-per-use, never resident" discipline is
        # what makes spilling correct across a loop back-edge — every use
        # carries its own S_LD_INT, which re-executes each iteration — and
        # lets many spilled values time-share a few GPs. The store to a
        # value's slot happens once at spill; the value is loop-invariant
        # OR dies within the body, so the slot stays valid for re-reads.
        self.value_to_slot: Dict[int, int] = {}
        self._intram_next_slot: int = 0
        # Hook for emit to receive spill/reload ISA lines + to
        # capture cur_idx for reload sequencing. emit binds these
        # at construction time.
        self._isa_lines: Optional[List[str]] = None
        # Track whether we've already warned for this function
        # (warn at most once per kernel; per-spill noise is
        # excessive).
        self._spill_warned: bool = False
        # Total spill / reload counts for diagnostics.
        self.spill_count: int = 0
        self.reload_count: int = 0

    # ---- emit hookups (set by MirToIsa.__init__) ----
    def bind_emit(self, isa_lines: List[str], get_cur_idx) -> None:
        """Hand the allocator the live ISA-line list + a callback
        to fetch the current instr idx (used when picking spill
        victims — we want the one whose interval end is farthest from
        ``cur_idx``)."""
        self._isa_lines = isa_lines
        self._get_cur_idx = get_cur_idx

    def reserve_intram_slot(self) -> int:
        """Reserve and return the next IntRAM slot. Used by emit
        for serial-loop idx slots (see ``_emit_loop_serial``)."""
        slot = self._intram_next_slot
        self._intram_next_slot += 1
        return slot

    # ---- free-pool invariants ----
    # Two invariants the allocator must never violate:
    #   (1) a GP appears in ``free_gp`` AT MOST ONCE — otherwise two
    #       distinct ``pop`` calls hand the same physical register to two
    #       live values, and one silently clobbers the other;
    #   (2) a GP in ``free_gp`` is owned by NO value in ``value_to_gp``.
    # Every return-to-pool routes through ``_free_gp`` and every take
    # through ``_take_gp`` so these hold by construction. A violation is a
    # real allocator bug; we fail loud rather than emit corrupt ISA.
    def _free_gp(self, gp: int) -> None:
        if gp == 0:
            return  # gp0 is never pooled
        if gp in self.free_gp:
            raise MirToIsaError(
                f"alloc invariant: gp{gp} freed twice (already in pool "
                f"{self.free_gp}). Double-free corrupts allocation."
            )
        owner = next((v for v, g in self.value_to_gp.items() if g == gp),
                     None)
        if owner is not None:
            raise MirToIsaError(
                f"alloc invariant: gp{gp} freed while still owned by "
                f"value@{owner}."
            )
        self.free_gp.insert(0, gp)

    def _take_gp(self) -> int:
        gp = self.free_gp.pop(0)
        owner = next((v for v, g in self.value_to_gp.items() if g == gp),
                     None)
        if owner is not None:
            raise MirToIsaError(
                f"alloc invariant: gp{gp} taken from pool but still owned "
                f"by value@{owner} — pool/owner state diverged."
            )
        return gp

    def set_end(self, v: "mir.MirValue", end_idx: int) -> None:
        """Register an interval end for a value minted during emit (the
        serial loop counter). The pre-pass never saw it, so emit must
        declare how long it lives — typically the loop's last body
        emit index."""
        self._manual_end[id(v)] = end_idx

    def _make_room(self) -> bool:
        """Free exactly one GP so a subsequent ``_take_gp`` succeeds, by
        spilling one value to IntRAM. Returns True if a GP was freed,
        False if nothing is spillable (genuine, unrecoverable pressure).

        There is ONE spill mechanism now: store to IntRAM, reload per use.
        Loop-carried values are spillable too — correctness across the
        loop back-edge comes from reloading at every use (each S_LD_INT
        re-executes each iteration), not from keeping the value resident."""
        victim = self._pick_spill_victim()
        if victim is None:
            return False
        self._spill_value(victim)
        return True

    def spill_carried_at_entry(self, lp: "mir.MirLoop") -> None:
        """SCOPE-ENTRY spill (design §3). Called BEFORE ``C_LOOP_START``
        — i.e. outside the loop body — to demote loop-carried values to
        IntRAM whose ``S_ST_INT`` must therefore land OUTSIDE the body.

        We spill every carried value of ``lp`` that is currently resident
        in a GP. The store is emitted here (outside the body, once). The
        body then reads each via reload-from-slot (loads only, which
        re-execute correctly every iteration because the slot was written
        outside the body and the value is loop-invariant). This is the
        structural fix for the "spill-store landed in the body and got
        overwritten across iterations" bug: stores never sit in a body.

        After this, ``_pick_spill_victim`` forbids spilling carried
        values mid-body, so no new carried store can land in the body.
        Spilling ALL carried values here is conservative (may add reload
        traffic) but always correct; keeping some in GP is a later
        optimisation."""
        carried = self.carried_by_loop.get(id(lp), set())
        for vid in sorted(carried):
            if vid in self.value_to_gp and self.value_to_gp[vid] != 0 \
                    and vid not in self._pinned:
                self._spill_value(vid)

    def enter_loop(self, lp: "mir.MirLoop") -> None:
        """Mark a serial loop as open during emit: its loop-carried
        values become unspillable mid-body until ``exit_loop`` (their
        stores were already done at scope entry — see
        ``spill_carried_at_entry``). Must be called right before emitting
        the loop body, paired with exit_loop."""
        lid = id(lp)
        self._open_loops.append(lid)
        self._carried_now |= self.carried_by_loop.get(lid, set())

    def exit_loop(self, lp: "mir.MirLoop") -> None:
        """End the open-loop scope started by ``enter_loop``. Recompute
        the carried set from the loops still open (a value carried by an
        outer loop must stay protected even after the inner one closes)."""
        lid = id(lp)
        assert self._open_loops and self._open_loops[-1] == lid, (
            "enter_loop/exit_loop mismatch"
        )
        self._open_loops.pop()
        self._carried_now = set()
        for open_lid in self._open_loops:
            self._carried_now |= self.carried_by_loop.get(open_lid, set())

    def _end_of(self, vid: int) -> Optional[int]:
        """Interval end for ``vid`` (manual override wins). None means
        the value has no recorded use at all."""
        if vid in self._manual_end:
            return self._manual_end[vid]
        return self.end.get(vid)

    def pin(self, v: "mir.MirValue") -> None:
        """Pin ``v``'s GP against spill AND recycle for the duration of
        an open serial loop. Only for emit-managed values whose GP is
        named raw in the prologue/epilogue (counter, lvar)."""
        self._pinned.add(id(v))

    def unpin(self, v: "mir.MirValue") -> None:
        self._pinned.discard(id(v))
        # Release now if dead: at loop exit the counter/lvar are done.
        gp = self.value_to_gp.get(id(v))
        if gp is not None and gp != 0:
            del self.value_to_gp[id(v)]
            self._free_gp(gp)

    def lock_operands(self, vids) -> None:
        """Protect these values' GPs from being chosen as spill victims.
        Emit calls this with the current instruction's input-operand
        ids before allocating the result GP, so a result allocation
        under register pressure can't spill an operand whose token was
        already emitted for this instruction."""
        self._operand_lock = set(vids)

    def clear_operand_lock(self) -> None:
        self._operand_lock = set()

    # ---- spill / reload ----
    def _pick_spill_victim(self) -> Optional[int]:
        """Return id(MirValue) of the value whose GP we'll steal, or None
        if no candidate. Among SPILLABLE values, picks the one with the
        FARTHEST interval end (least likely to be referenced soon).

        A value is NOT spillable MID-BODY if it is:
          * gp0 (the const-zero fixture),
          * pinned (serial-loop counter/lvar, named raw in ISA),
          * an operand of the current instruction (token already emitted),
          * loop-carried by any currently-open loop (``_carried_now``):
            its store must stay OUTSIDE the body (see
            ``spill_carried_at_entry``); spilling it here would emit an
            ``S_ST_INT`` inside the body that re-executes every iteration
            with a clobbered GP. Carried values are demoted at scope
            ENTRY instead, where the store is outside the body.
        So mid-body spill only ever targets ``local`` temporaries whose
        store+reload sit in the same straight-line iteration — safe."""
        cur = self._get_cur_idx() if self._get_cur_idx else -1
        best_vid = None
        best_end = -1
        for vid, gp in self.value_to_gp.items():
            if gp == 0:
                continue
            if vid in self._pinned:
                continue  # raw-GP-named (counter/lvar) — never move it
            if vid in self._operand_lock:
                continue  # current instr's live operand — would corrupt it
            if vid in self._carried_now:
                continue  # carried — demoted at scope entry, not mid-body
            e = self._end_of(vid)
            e = cur if e is None else e
            if e > best_end:
                best_end = e
                best_vid = vid
        return best_vid

    def _spill_value(self, vid: int) -> int:
        """Spill the value with id ``vid``: free its GP back to the pool,
        leaving the value in IntRAM (``value_to_slot``). Returns the freed
        GP number.

        If the value ALREADY has a slot (it was spilled before and is
        only transiently reloaded), we DON'T store again — the slot's
        contents are still valid (the value is loop-invariant or unchanged
        since the original store). We just free the GP. This is what makes
        reload-per-use cheap: one store ever, many reloads."""
        gp = self.value_to_gp.pop(vid)
        existing = self.value_to_slot.get(vid)
        if existing is not None:
            self._isa_lines.append(
                f"; RE-EVICT value@{vid} (slot {existing} still valid) "
                f"freeing gp{gp}"
            )
            self._free_gp(gp)
            return gp
        slot = self.reserve_intram_slot()
        self.value_to_slot[vid] = slot
        _nm = self._vid_to_value.get(vid)
        _nm = f"%{_nm.name}" if _nm is not None else f"id{vid}"
        self._isa_lines.append(
            f"; SPILL {_nm} (value@{vid}) -> intram[{slot}] (freeing gp{gp})"
        )
        self._isa_lines.append(f"S_ST_INT gp{gp}, gp0, {slot}")
        # Return the GP to the free pool so the caller (assign_i32 /
        # ensure_in_gp) can grab it.
        self._free_gp(gp)
        self.spill_count += 1
        if not self._spill_warned:
            self._spill_warned = True
            warnings.warn(
                f"mir_to_isa: IntRAM spill triggered in function "
                f"{self.fn.name!r} — register pressure exceeded "
                f"{GP_USER_LAST - GP_USER_FIRST + 1} GPs. "
                f"Each spill incurs an S_ST_INT + S_LD_INT per use. "
                f"Consider reducing LICM aggressiveness, hoisting "
                f"fewer invariants, or accepting the IntRAM round-trip.",
                RuntimeWarning,
                stacklevel=3,
            )
        return gp

    def ensure_in_gp(self, v: "mir.MirValue") -> int:
        """Like ``assign_i32`` but for an OPERAND read. If ``v`` is
        currently spilled, reload it from its IntRAM slot into a
        GP (possibly triggering another spill to free that GP).
        Returns the GP holding ``v``."""
        if v.is_function_const:
            return 0
        if id(v) in self.value_to_gp:
            return self.value_to_gp[id(v)]
        if id(v) in self.value_to_slot:
            # Reload from IntRAM into a GP. The slot entry is RETAINED
            # (not popped): the value stays spill-resident, so after this
            # use ``release_dead_at`` frees the GP again and the next use
            # reloads afresh. Reload-per-use keeps it correct across the
            # loop back-edge and lets spilled values time-share GPs.
            slot = self.value_to_slot[id(v)]
            if not self.free_gp and not self._make_room():
                raise MirToIsaError(
                    f"reload: no free GP and nothing to spill "
                    f"for value {v!r}"
                )
            gp = self._take_gp()
            self.value_to_gp[id(v)] = gp
            self._isa_lines.append(
                f"; RELOAD %{v.name} (value@{id(v)}) from intram[{slot}] "
                f"-> gp{gp}"
            )
            self._isa_lines.append(f"S_LD_INT gp{gp}, gp0, {slot}")
            self.reload_count += 1
            return gp
        # Reading an operand that is in NEITHER a GP nor a spill slot
        # means its value was dropped (release_dead_at deleted it from
        # value_to_gp) before this use — i.e. its computed last_use is
        # earlier than this actual use. Allocating a fresh GP here would
        # silently hand back garbage (no reload), corrupting the operand.
        # This is always a live-range bug, never legitimate. Fail loud.
        cur = self._get_cur_idx() if self._get_cur_idx else -1
        raise MirToIsaError(
            f"ensure_in_gp: operand value@{id(v)} ({v!r}) read at "
            f"cur_idx={cur} is in neither a GP nor a spill slot — it was "
            f"released before this use. Its interval end="
            f"{self._end_of(id(v))!r}. This is a live-interval bug: the "
            f"value's recorded end is earlier than this actual use."
        )

    def assign_i32(self, v: "mir.MirValue") -> int:
        """Return the gp number for ``v``, allocating a new one if
        first sight. The function-level gp0 constant is fixed at
        register 0. On free-pool exhaustion, spills a live value."""
        self._vid_to_value[id(v)] = v
        if v.is_function_const:
            self.value_to_gp[id(v)] = 0
            return 0
        if id(v) in self.value_to_gp:
            return self.value_to_gp[id(v)]
        if id(v) in self.value_to_slot:
            # Defining a value that we've spilled? Shouldn't
            # happen — spill is on USES, but the def itself sets
            # the initial value. Reload via ensure_in_gp instead.
            return self.ensure_in_gp(v)
        if not self.free_gp:
            # Free one GP by spilling a value to IntRAM. Only fires when
            # the pool is empty, so the common path is untouched.
            if not self._make_room():
                # Nothing spillable. Break down WHICH value holds each GP
                # and why, so we can see what LICM hoisted / what's
                # carried across the open loops.
                def _nm(vid):
                    val = self._vid_to_value.get(vid)
                    return val.name if val is not None else f"id{vid}"
                lines = []
                for vid, g in sorted(self.value_to_gp.items(),
                                     key=lambda kv: kv[1]):
                    why = []
                    if vid in self._pinned:
                        why.append("pinned(ctr/lvar)")
                    if vid in self._carried_now:
                        why.append("carried")
                    if vid in self._operand_lock:
                        why.append("operand")
                    lines.append(
                        f"    gp{g} = %{_nm(vid)}  end={self._end_of(vid)}"
                        f"  [{','.join(why) or 'free?'}]"
                    )
                raise MirToIsaError(
                    f"GP file exhausted ({len(self.value_to_gp)} of "
                    f"{GP_USER_LAST - GP_USER_FIRST + 1} GPs held, none "
                    f"spillable) while defining %{v.name}.\n"
                    f"  open loops: {len(self._open_loops)}\n"
                    + "\n".join(lines)
                )
        gp = self._take_gp()
        self.value_to_gp[id(v)] = gp
        return gp

    def assign_addr_reg(self, v: "mir.MirValue") -> int:
        if id(v) in self.value_to_areg:
            return self.value_to_areg[id(v)]
        areg = self._next_areg
        self._next_areg += 1
        self.value_to_areg[id(v)] = areg
        return areg

    def release_dead_at(self, cur_idx: int) -> None:
        """Return GPs to the free pool. Two cases:

        * A SPILLED value (has a ``value_to_slot`` entry) that is
          currently reloaded into a GP is freed PROMPTLY — its slot keeps
          the value, the next use reloads again. This "never resident"
          discipline is the time-sharing + back-edge correctness of the
          single spill mechanism.
        * A non-spilled value is freed once its structured interval end
          is ``<= cur_idx`` (provably dead, including across serial-loop
          re-entries — the interval extends carried values to the loop
          end). Values with no recorded end are dead on definition.

        Convention: ``_emit_instr`` calls with ``cur-1`` before operand
        formatting and ``cur`` after emitting (so ``end <= cur_idx``)."""
        for vid in list(self.value_to_gp.keys()):
            gp = self.value_to_gp[vid]
            if gp == 0:
                continue  # gp0 is a permanent hardware fixture
            if vid in self._operand_lock:
                continue  # token in the current line — don't reclaim yet
            if vid in self._pinned:
                continue  # emit owns the counter/lvar lifecycle
            if vid in self.value_to_slot:
                # Spilled value, transiently reloaded: free its GP now;
                # the slot keeps the value for the next use's reload.
                del self.value_to_gp[vid]
                self._free_gp(gp)
                continue
            if vid in self._carried_now:
                continue  # live across an open loop — keep its GP put
            e = self._end_of(vid)
            if e is None or e <= cur_idx:
                # Drop ownership first, then return to pool (the pool
                # invariant check requires the GP be unowned).
                del self.value_to_gp[vid]
                self._free_gp(gp)

    def store_result(self) -> None:
        """No-op for the linear-scan allocator (results live in GPs and
        are spilled lazily on pressure). The stable allocator overrides
        this to write each result back to its fixed slot. Present here so
        ``_emit_instr`` can call it unconditionally for both allocators."""
        return


# ---------------------------------------------------------------------
# Stable allocator — correctness-first, NO reuse optimisation.
# ---------------------------------------------------------------------
#
# The linear-scan allocator above mixes 4 mechanisms (end-interval
# release, carried-spill protection, scope-entry spill, pin) whose
# interaction has缝 — bugs that only show up across loop re-entries
# (NQ=2) and are nearly impossible to debug. This allocator trades ALL
# optimisation for absolute robustness + trivial debuggability:
#
#   ONE rule: every i32 SSA value lives in its OWN permanent IntRAM slot
#   (never reused). GPs are pure intra-instruction scratch.
#     * define a value : compute into a scratch GP, then S_ST_INT it to
#                        the value's fixed slot. The GP is then free.
#     * use a value    : S_LD_INT its fixed slot into a fresh scratch GP.
#     * NOTHING survives an instruction in a GP (except the serial-loop
#       counter, which the hardware itself owns, and gp0=const-zero).
#
# Because no GP state crosses an instruction boundary, a loop body that
# is emitted once but runs N times is correct for every N: each
# instruction is self-contained (reload → compute → store). There is no
# spill decision, no carried set, no live interval, no cross-iteration
# state — therefore no缝 to leak through. It is slow (a load per use, a
# store per def) but provably correct and dead-simple to read: the slot
# number IS the value's identity.
#
# Hardware-forced exceptions (the only non-uniform parts):
#   * gp0 — hardware const-zero, never allocated.
#   * serial-loop counter — C_LOOP_START/END read/decrement it; it must
#     occupy a fixed GP for the loop's lifetime (``pin``).
#   * loop_var index — already IntRAM-backed (idx slot); fits the model.
class _StableAllocator:
    """Every value -> a permanent IntRAM slot; GPs are intra-instr scratch.

    Same public interface as ``_LinearScanAllocator`` so ``MirToIsa`` /
    ``_emit_instr`` use it unchanged, plus ``store_result`` (emit calls it
    after the instruction line to write the result GP back to its slot).
    Switch via ``MirToIsa(... )`` constructing this instead of the
    linear-scan one (see USE_STABLE_ALLOC)."""

    def __init__(self, fn: "mir.MirFunction") -> None:
        self.fn = fn
        self.emit_order = compute_emit_order(fn)
        _check_no_iter_args(fn)
        # Permanent slot per value-id. Allocated on first sight (define or
        # use), never reused. Counters/lvars also get pinned GPs.
        self.value_to_slot: Dict[int, int] = {}
        self._intram_next_slot: int = 0
        # Scratch GP pool. gp0 reserved (const-zero). The rest are handed
        # out per-instruction and reclaimed at every release_dead_at.
        self._all_scratch = list(range(GP_USER_FIRST, GP_USER_LAST + 1))
        self.free_gp: List[int] = list(self._all_scratch)
        # gp currently holding each value THIS instruction (transient).
        self.value_to_gp: Dict[int, int] = {}
        # Pinned values (serial-loop counter + lvar) hold a GP for the
        # whole loop; never reclaimed by release_dead_at.
        self._pinned: set = set()
        self._pin_gp: Dict[int, int] = {}
        # The value whose result-GP must be stored back after this instr.
        self._pending_store: Optional[Tuple[int, int]] = None  # (vid, gp)
        self._vid_to_value: Dict[int, "mir.MirValue"] = {}
        # addr-reg file (same as linear-scan).
        self.value_to_areg: Dict[int, int] = {}
        self._next_areg: int = 0
        self._isa_lines: Optional[List[str]] = None
        self._get_cur_idx = None
        self.spill_count = 0
        self.reload_count = 0
        # Unused-but-interface-compatible bits:
        self.loop_last_idx: Dict[int, int] = {}
        self.carried_by_loop: Dict[int, set] = {}
        self.end: Dict[int, int] = {}

    # ---- emit hookups ----
    def bind_emit(self, isa_lines, get_cur_idx) -> None:
        self._isa_lines = isa_lines
        self._get_cur_idx = get_cur_idx

    def reserve_intram_slot(self) -> int:
        slot = self._intram_next_slot
        self._intram_next_slot += 1
        return slot

    def _slot_of(self, v: "mir.MirValue") -> int:
        """Permanent slot for value ``v`` (allocate on first sight)."""
        if id(v) not in self.value_to_slot:
            self.value_to_slot[id(v)] = self.reserve_intram_slot()
        return self.value_to_slot[id(v)]

    def _take_scratch(self) -> int:
        if not self.free_gp:
            raise MirToIsaError(
                "stable alloc: out of scratch GPs in a single instruction "
                f"(have {len(self._all_scratch)} usable, all in use). A "
                "single MIR instr referenced more distinct i32 values than "
                "the GP file holds — should not happen for PLENA opcodes."
            )
        return self.free_gp.pop(0)

    def _free_scratch(self, gp: int) -> None:
        if gp == 0:
            return
        if gp not in self.free_gp:
            self.free_gp.insert(0, gp)

    # ---- value access ----
    def assign_i32(self, v: "mir.MirValue") -> int:
        """Define ``v``: hand out a scratch GP for the result and schedule
        a store-back to v's permanent slot after the instruction line."""
        self._vid_to_value[id(v)] = v
        if v.is_function_const:
            return 0
        if id(v) in self._pinned:               # counter/lvar: fixed GP
            return self._pin_gp[id(v)]
        gp = self._take_scratch()
        self.value_to_gp[id(v)] = gp
        # ensure the slot exists; schedule store-back.
        slot = self._slot_of(v)
        self._pending_store = (id(v), gp, slot)
        return gp

    def ensure_in_gp(self, v: "mir.MirValue") -> int:
        """Use ``v``: reload its permanent slot into a fresh scratch GP."""
        self._vid_to_value[id(v)] = v
        if v.is_function_const:
            return 0
        if id(v) in self._pinned:               # counter/lvar: fixed GP
            return self._pin_gp[id(v)]
        if id(v) in self.value_to_gp:           # already loaded this instr
            return self.value_to_gp[id(v)]
        if id(v) not in self.value_to_slot:
            raise MirToIsaError(
                f"stable alloc: use of value@{id(v)} (%{v.name}) before "
                f"any definition (no slot). live-range / emit-order bug."
            )
        slot = self.value_to_slot[id(v)]
        gp = self._take_scratch()
        self.value_to_gp[id(v)] = gp
        self._isa_lines.append(
            f"; LD %{v.name} (slot {slot}) -> gp{gp}"
        )
        self._isa_lines.append(f"S_LD_INT gp{gp}, gp0, {slot}")
        self.reload_count += 1
        return gp

    def store_result(self) -> None:
        """Emit the store-back for the value defined this instruction
        (called by emit right AFTER the instruction line)."""
        if self._pending_store is None:
            return
        vid, gp, slot = self._pending_store
        self._pending_store = None
        val = self._vid_to_value.get(vid)
        nm = f"%{val.name}" if val is not None else f"id{vid}"
        self._isa_lines.append(f"; ST {nm} gp{gp} -> slot {slot}")
        self._isa_lines.append(f"S_ST_INT gp{gp}, gp0, {slot}")
        self.spill_count += 1

    def assign_addr_reg(self, v: "mir.MirValue") -> int:
        if id(v) in self.value_to_areg:
            return self.value_to_areg[id(v)]
        areg = self._next_areg
        self._next_areg += 1
        self.value_to_areg[id(v)] = areg
        return areg

    def release_dead_at(self, cur_idx: int) -> None:
        """Reclaim ALL scratch GPs (nothing crosses an instruction).
        Pinned counter/lvar GPs are kept."""
        for vid in list(self.value_to_gp.keys()):
            if vid in self._pinned:
                continue
            gp = self.value_to_gp.pop(vid)
            self._free_scratch(gp)

    # ---- pins (counter / lvar) ----
    def pin(self, v: "mir.MirValue") -> None:
        # Give the pinned value its own dedicated GP for the loop.
        # ``_emit_loop_serial`` calls ``assign_i32(v)`` immediately before
        # ``pin(v)`` to mint the GP the prologue text already references —
        # so ADOPT that GP rather than taking a fresh one (otherwise the
        # prologue and body would disagree on which GP holds the counter/
        # lvar). Also drop any store-back scheduled by that assign_i32:
        # the counter/lvar live in a dedicated GP (and the lvar in its idx
        # slot), NOT in a data slot — a stray S_ST_INT would clobber a
        # data slot on the next instruction's store_result.
        if id(v) not in self._pin_gp:
            if id(v) in self.value_to_gp:
                gp = self.value_to_gp[id(v)]
            else:
                gp = self._take_scratch()
                self.value_to_gp[id(v)] = gp
            self._pin_gp[id(v)] = gp
        if self._pending_store is not None and self._pending_store[0] == id(v):
            self._pending_store = None
        self._pinned.add(id(v))

    def unpin(self, v: "mir.MirValue") -> None:
        self._pinned.discard(id(v))
        gp = self._pin_gp.pop(id(v), None)
        if gp is not None:
            self.value_to_gp.pop(id(v), None)
            self._free_scratch(gp)

    # ---- no-op / trivial interface compatibility ----
    def set_end(self, v, end_idx) -> None:
        pass

    def enter_loop(self, lp) -> None:
        pass

    def exit_loop(self, lp) -> None:
        pass

    def spill_carried_at_entry(self, lp) -> None:
        pass

    def lock_operands(self, vids) -> None:
        pass

    def clear_operand_lock(self) -> None:
        pass

    def assign_i32_pin_gp(self, v):  # helper for pin GP lookup
        return self._pin_gp.get(id(v))


# Switch: True = robust stable allocator (every value -> fixed IntRAM
# slot, GPs intra-instruction only). False = linear-scan optimiser above.
USE_STABLE_ALLOC = True


# ---------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------

class MirToIsa:
    """Walk a MirFunction; produce ISA text."""

    def __init__(self, fn: mir.MirFunction, shim) -> None:
        self.fn = fn
        self.shim = shim
        self.alloc = (
            _StableAllocator(fn) if USE_STABLE_ALLOC
            else _LinearScanAllocator(fn)
        )
        self.lines: List[str] = []
        # For serial loops we need to claim an IntRAM idx slot and a
        # GP-loop counter; pin/release like legacy ``_emit_for``.
        # Track open serial loops to emit the matching epilogue.
        self._serial_loop_stack: List[Dict] = []
        # Current emit index, set per instr from the pre-assigned
        # ``compute_emit_order`` map (NOT a running counter). Drives the
        # allocator's release_dead_at / spill-victim choice.
        self._cur_idx: int = -1
        # Wire allocator so spill/reload can emit S_ST_INT/S_LD_INT
        # into our ``lines`` list and ask us for the current idx
        # when picking a spill victim.
        self.alloc.bind_emit(self.lines, lambda: self._cur_idx)

    def run(self) -> str:
        # Header.
        self.lines.append(f"; PLENA ISA  --  kernel: {self.fn.name}")
        self.lines.append(
            "; generated by tilelang_tvm_compiler (PreIsaIR v2 → MIR path)"
        )
        self.lines.append("; " + "=" * 60)
        for blk in self.fn.blocks:
            self._emit_block(blk)
        return "\n".join(self.lines) + "\n"

    def _emit_block(self, blk: mir.MirBlock) -> None:
        for item in blk.items:
            if isinstance(item, mir.MirInstr):
                self._emit_instr(item)
            elif isinstance(item, mir.MirLoop):
                self._emit_loop(item)
            else:
                raise MirToIsaError(
                    f"unexpected block item: {type(item).__name__}"
                )

    def _emit_loop(self, lp: mir.MirLoop) -> None:
        if lp.loop_kind == "serial":
            self._emit_loop_serial(lp)
            return
        if lp.loop_kind == "unroll":
            # Emit-time unrolling was removed: it cloned the body into a
            # scratch block (minting MirValues the precomputed last_use
            # table never saw) and corrupted register allocation. All
            # loops now lower to hardware C_LOOPs — pre_isa_ir_v2's
            # FORCE_SERIAL_LOOPS downgrades unroll→serial at construction,
            # so this branch should be unreachable. If it fires, a loop
            # was built after that switch was bypassed.
            raise MirToIsaError(
                f"loop {lp.name!r} reached emit with loop_kind='unroll'; "
                f"emit-time unrolling is removed (see "
                f"pre_isa_ir_v2.FORCE_SERIAL_LOOPS). All loops must be "
                f"serial by the time MIR is emitted."
            )
        raise MirToIsaError(
            f"unknown loop_kind {lp.loop_kind!r} on loop {lp.name}"
        )

    def _emit_loop_serial(self, lp: mir.MirLoop) -> None:
        """Emit a hardware-backed serial loop.

        The loop_var is materialised the way the PLENA ISA spec
        mandates: a SEPARATE software-maintained index in an IntRAM
        slot, NOT the hardware loop counter register. Per the spec
        (``doc/plena_isa_spec.md`` C_LOOP_START):

            "The loop counter register ``rd`` does NOT contain the
            current iteration index. You must maintain your own
            index variable and increment it manually inside the
            loop."

        So:
          * a **counter GP** holds ``C_LOOP_START``'s remaining-iter
            count (hardware-managed; we never read it as data),
          * a **lvar GP** + **IntRAM idx slot** hold the real index;
            the body reads it at entry, the epilogue increments and
            stores it back.

        A prior experiment derived the loop_var as ``counter - 1`` to
        save the idx slot + 3 instrs/iter. It passed in the simulator
        (whose ``gp[rd]`` happens to expose ``extent..1``) but the
        hardware spec explicitly forbids reading ``rd`` as an index,
        so it was undefined behaviour on real silicon and has been
        removed. ``order_independent`` annotations still flow through
        the IR (kernel → HLIR → MIR) for any future, spec-safe
        order-independence optimisation, but the backend treats every
        serial loop identically here.

        Resources are emit-time-only physical concerns (no upper
        layer should know about them):

        * **counter GP** — ``C_LOOP_START gpN, K`` operand. Minted
          fresh per loop.
        * **lvar GP** — holds the current index in the body.
        * **IntRAM idx slot** — software index backing store.

        Both the counter and lvar GPs are named RAW (by number) in the
        prologue/epilogue ISA, outside the allocator's operand path, so
        they are ``pin``-ned: never spilled or recycled while the loop is
        open. They also get an interval end at the loop's last body index
        so the structured allocator sees them as live across the body
        (the body is emitted once but runs N times). Data values carried
        from outside the loop need NO pin — their structured interval
        already extends across the loop.

        Non-zero ``lp.init`` is handled in the prelude by storing the
        init value into the idx slot instead of zero.
        """
        loop_end = self.alloc.loop_last_idx.get(id(lp))

        # 0) SCOPE ENTRY: demote this loop's carried values to IntRAM
        # NOW — before C_LOOP_START — so their S_ST_INT lands OUTSIDE the
        # body (design §3). Inside the body they are reloaded on use
        # (loads only). This is the structural fix for spill-stores
        # landing in a loop body and getting clobbered across iterations.
        # It also frees the GPs the counter/lvar need below.
        self.alloc.spill_carried_at_entry(lp)

        # 1) Counter GP. Pinned: C_LOOP_START/END name its GP raw.
        counter_val = self.fn.mint_value("i32", hint=f"loop_ctr_{lp.name}")
        counter_gp = self.alloc.assign_i32(counter_val)
        self.alloc.pin(counter_val)
        if loop_end is not None:
            self.alloc.set_end(counter_val, loop_end)

        # 2) lvar GP — body uses it; pinned because the idx prologue/
        # epilogue (S_LD/S_ADDI/S_ST) name its GP raw too.
        lvar_gp = self.alloc.assign_i32(lp.loop_var)
        self.alloc.pin(lp.loop_var)
        if loop_end is not None:
            self.alloc.set_end(lp.loop_var, loop_end)

        # 3) IntRAM idx slot + LD entry + LD/ADDI/ST exit.
        idx_addr = self.alloc.reserve_intram_slot()
        self.lines.append(
            f"; for {lp.loop_var.name} in [{lp.init}, "
            f"{lp.init + lp.extent}) -- hw counter gp{counter_gp}, "
            f"idx ram[{idx_addr}]"
        )
        if lp.init == 0:
            self.lines.append(f"S_ST_INT gp0, gp0, {idx_addr}")
        else:
            init_imm = int(lp.init)
            if 0 <= init_imm <= 262143:
                self.lines.append(
                    f"S_ADDI_INT gp{lvar_gp}, gp0, {init_imm}"
                )
            else:
                raise MirToIsaError(
                    f"serial loop init {init_imm} exceeds S_ADDI_INT "
                    f"immediate range; S_LUI fallback not yet wired"
                )
            self.lines.append(
                f"S_ST_INT gp{lvar_gp}, gp0, {idx_addr}"
            )
        self.lines.append(
            f"C_LOOP_START gp{counter_gp}, {lp.extent}"
        )
        self.lines.append(
            f"S_LD_INT gp{lvar_gp}, gp0, {idx_addr}"
        )
        self._serial_loop_stack.append({
            "counter_gp": counter_gp,
            "counter_val": counter_val,
            "idx_addr": idx_addr,
            "lvar_gp": lvar_gp,
            "lvar_name": lp.loop_var.name,
            "loop_var": lp.loop_var,
        })
        # Open the loop scope: its loop-carried values become unspillable
        # for the whole body (a spill here wouldn't replay on the
        # back-edge — see _pick_spill_victim).
        self.alloc.enter_loop(lp)
        try:
            for body_blk in lp.body:
                self._emit_block(body_blk)
        finally:
            self.alloc.exit_loop(lp)
            # Emit owns counter/lvar lifecycle: unpin AFTER the body so
            # their raw GPs survive the whole region, then the epilogue
            # below still references them by the same number.
            self.alloc.unpin(lp.loop_var)
            self.alloc.unpin(counter_val)
        st = self._serial_loop_stack.pop()
        self.lines.append(
            f"; idx {st['lvar_name']} += 1 (ram[{st['idx_addr']}])"
        )
        self.lines.append(
            f"S_LD_INT gp{st['lvar_gp']}, gp0, {st['idx_addr']}"
        )
        self.lines.append(
            f"S_ADDI_INT gp{st['lvar_gp']}, gp{st['lvar_gp']}, 1"
        )
        self.lines.append(
            f"S_ST_INT gp{st['lvar_gp']}, gp0, {st['idx_addr']}"
        )
        self.lines.append(f"C_LOOP_END gp{st['counter_gp']}")

    def _emit_instr(self, instr: mir.MirInstr) -> None:
        # Use the PRE-ASSIGNED emit index (compute_emit_order), not a
        # running counter — this is the same index space the interval
        # table is keyed on, so cur_idx and interval ends can never
        # drift (the old running-counter approach drifted whenever emit
        # appended ISA lines the interval pre-pass never walked).
        cur = self.alloc.emit_order[id(instr)]
        self._cur_idx = cur
        op = instr.opcode
        if op == "_COMMENT":
            text = instr.operands[0] if instr.operands else ""
            self.lines.append(f"; {text}")
            self.alloc.release_dead_at(cur)
            return

        spec = mir.OPCODES.get(op)
        if spec is None:
            raise MirToIsaError(f"unknown opcode {op!r}")

        # FIRST: release any GPs whose interval END was before this
        # instruction (end <= cur-1). This lets ``assign_i32(instr.result)``
        # (below) see a free pool that already excludes values that died
        # at end == cur-1. Without it, assign_i32 grabs a fresh GP for the
        # result while the previous instr's now-dead operand GPs are still
        # parked — peak GP usage inflates by one slot at every link in an
        # address-arithmetic chain.
        self.alloc.release_dead_at(cur - 1)

        # Format operands per opcode spec. Lock each i32 operand's GP
        # INCREMENTALLY — as soon as its token is emitted, it must not be
        # spilled/remat-evicted to make room for a LATER operand or the
        # result (its gpN is already in this line's text). Locking only
        # after the whole loop would leave earlier operands stealable
        # while formatting later ones.
        tokens: List[str] = []
        operand_value_ids = []
        self.alloc.lock_operands(operand_value_ids)
        try:
            for i, (val, kind) in enumerate(
                zip(instr.operands, spec.operand_kinds),
            ):
                tokens.append(self._fmt_operand(val, kind))
                if kind == "i32" and isinstance(val, mir.MirValue) \
                        and not val.is_function_const:
                    operand_value_ids.append(id(val))
                    self.alloc.lock_operands(operand_value_ids)

            # Format result prefix (if non-void). The dst GP goes FIRST
            # in the ISA arg list. Operand locks (above) keep the inputs
            # from being evicted to make room for the result.
            result_tok: Optional[str] = None
            if instr.result is not None:
                if spec.result_type == "i32":
                    gp = self.alloc.assign_i32(instr.result)
                    result_tok = f"gp{gp}"
                elif spec.result_type == "addr_reg":
                    areg = self.alloc.assign_addr_reg(instr.result)
                    result_tok = f"a{areg}"
                else:
                    raise MirToIsaError(
                        f"{op}: don't know how to assign physical reg for "
                        f"result_type {spec.result_type!r}"
                    )

            # PLENA ISA convention: the destination GP goes FIRST in the
            # operand list (just like RISC-V). The MIR operand list
            # carries SOURCES only; we prepend dst here.
            if result_tok is not None:
                arg_list = ", ".join([result_tok] + tokens)
            else:
                arg_list = ", ".join(tokens)
            # Inline MIR-trace tag: map this ISA line back to its MIR
            # SSA value(s), so a reader can follow which %N each GP holds.
            def _opname(o):
                if isinstance(o, mir.MirValue):
                    return f"%{o.name}"
                return str(o)
            res_nm = f"%{instr.result.name}=" if instr.result is not None else ""
            ops_nm = ",".join(_opname(o) for o in instr.operands)
            self.lines.append(
                f"{spec.isa_mnemonic} {arg_list}"
                f"   ; {res_nm}{op}({ops_nm})"
            )
        finally:
            self.alloc.clear_operand_lock()
        # Store-back the result to its permanent slot (stable allocator).
        # Must run while the result GP is still owned (before the release
        # below frees it). No-op for the linear-scan allocator.
        self.alloc.store_result()
        # Post-emit: release any value whose interval END == cur. These
        # are operands whose final use was this very instruction. They
        # can't be released before assign_i32 above (their GP was needed
        # to format the operand text), but they're dead now and the next
        # instr can reuse them.
        self.alloc.release_dead_at(cur)

    def _fmt_operand(self, val, kind: str) -> str:
        if kind == "i32":
            if isinstance(val, mir.MirValue):
                # ensure_in_gp reloads from IntRAM if val was
                # spilled; transparent to the caller.
                return f"gp{self.alloc.ensure_in_gp(val)}"
            if isinstance(val, int):
                return str(val)
            raise MirToIsaError(
                f"i32 operand expects MirValue or int; got {val!r}"
            )
        if kind == "literal_int":
            return str(int(val))
        if kind == "fp_reg":
            if not isinstance(val, str):
                raise MirToIsaError(
                    f"fp_reg operand expects str; got {val!r}"
                )
            return val
        if kind == "verbatim_str":
            if not isinstance(val, str):
                raise MirToIsaError(
                    f"verbatim_str expects str; got {val!r}"
                )
            return val
        if kind == "addr_reg":
            if isinstance(val, mir.MirValue):
                return f"a{self.alloc.assign_addr_reg(val)}"
            raise MirToIsaError(
                f"addr_reg operand expects MirValue; got {val!r}"
            )
        raise MirToIsaError(f"unknown operand kind {kind!r}")


def emit(fn: mir.MirFunction, shim) -> str:
    """Convert a MirFunction to ISA text."""
    return MirToIsa(fn, shim).run()


__all__ = ["emit", "MirToIsa", "MirToIsaError"]
