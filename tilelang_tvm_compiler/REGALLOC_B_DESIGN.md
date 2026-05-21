# Region-recursive GP allocation (design B)

Replaces the linear-scan-on-flattened-IR allocator in `mir_to_isa.py`.
The MIR is already clean SCF (blocks of interleaved instrs + nested
`MirLoop` regions). This design allocates *over that tree* instead of
flattening it, so loop-carried values are handled by construction — no
`_open_loop_ends` / `_operand_lock` / `_no_release` / pin patches.

## The one idea

> A region's **live-in** values (used inside, defined outside) occupy
> fixed GPs for the WHOLE region. Only values *born and killed inside*
> the region are spill candidates within it.

Why this kills the current bug: emit is single-pass but a loop body runs
N times. A spill emitted in the body's second half is not replayed when
control jumps back to the head, so spilling a value the head re-reads
corrupts iteration 2. Live-in values are exactly the values the head
re-reads. If they are reserved before the body and never offered to the
in-body spill picker, the corruption is structurally impossible.

A value born and killed inside the body is safe to spill: its spill and
reload both lie within one iteration's emitted code, so the round-trip
re-executes intact every iteration.

## Liveness the SCF way (no flat index)

For each value `v`, `def_region(v)` = the innermost region containing its
def. `v` is **live-in** to region `R` iff it has a use inside `R` and
`def_region(v)` is a strict ancestor of `R` (i.e. `v` is defined outside
`R`).

Per region `R` we need:
- `live_in(R)`  — values defined outside `R`, used inside `R`.
- `local(R)`    — values defined inside `R` (directly, not in a child
                  region). These are the spillable ones for `R`.

Both are computable in one post-order walk:
```
live_in(R)  = (union over children C of live_in(C)
               ∪ {operands of R's own instrs})
              minus {values defined in R}
local(R)    = {results of R's own instrs}     # children's locals stay theirs
```
(`live_in` of the function's top region is just the function consts /
block args — typically only gp0.)

## Allocation walk

`assign_region(R, free)`:
- `free` = set of GP numbers usable inside `R` (caller already subtracted
  what it reserved for `R`'s live-ins + the loop counter/lvar).
- Walk `R`'s items in order. For each:
  - **MirInstr**: operands already hold GPs (live-in reserved, or a local
    assigned earlier in this walk). Assign the result a GP from `free`
    (or spill a *local* whose last in-region use has passed). Free locals
    whose last use is this instr.
  - **MirLoop child `L`**:
      1. `carried = live_in(L) ∩ (currently-held values)` — these must
         stay put across `L`. They are *already* in GPs; we simply do NOT
         lend those GPs into `L`.
      2. Reserve a GP for `L`'s counter + lvar.
      3. `inner_free = free − {carried GPs} − {counter,lvar GPs} −
         {GPs held by R-locals still live across L}`.
      4. `assign_region(L, inner_free)`.
      5. On return, reclaim counter/lvar GPs.

The spill picker, when invoked inside `R`, can only see `local(R)` values
not yet dead — never a live-in (those GPs were never in `inner_free`).

## What stays identical
- `compute_emit_order` — still used, only to order last-use *within a
  region* (a region is straight-line in emit order, so a plain index
  inside the region is exact; no cross-loop extension needed anymore).
- `assign_addr_reg` (a-regs), fp_reg tokens, gp0 = const-zero, IntRAM
  slot reservation, the S_ST_INT/S_LD_INT spill/reload emission, the
  serial-loop prologue/epilogue, `_free_gp`/`_take_gp` invariants.
- The `_emit_instr` operand-formatting + result-prepend logic.

## What is deleted
- `_compute_live_intervals` (the flat `end` table) → replaced by
  `live_in`/`local` per region.
- `release_dead_at(cur_idx)` global sweep → per-region last-use frees.
- `_open_loop_ends`, `_operand_lock`, `_no_release`, `pin`/`unpin` for
  data values. (Counter/lvar still get a GP reserved for the region,
  but via reservation, not a pin set.)

## Spill correctness (now provable)
- A spilled value is always a `local(R)` of the region currently being
  walked. Its spill point and every reload point are emitted inside the
  same straight-line region body. A loop re-entry re-executes that whole
  body, so the spill/reload pair re-runs together each iteration. No
  cross-iteration leak.
- Live-ins are never spilled inside a child loop (their GPs aren't lent
  in), so the head's re-read always sees the correct GP.

## Register-pressure note
Reserving all live-ins of a deep loop nest could exhaust 15 GPs. If
`inner_free` is empty when a child needs registers, that's genuine
pressure: spill an *outer* `local` (safe — it's straight-line at that
level) before descending, or, as a later refinement, allow spilling a
live-in to a **dedicated per-region IntRAM slot reloaded at the loop
head** (the "loop-head fixup" that makes spilling a carried value safe).
Start without that; the min kernels fit.

## Migration
Single file (`mir_to_isa.py`). `MirToIsa.run()` calls
`assign_region(top, all_user_gps)` once; `_emit_instr` and the serial
loop emitter call into the region allocator instead of the linear-scan
one. Keep the old class around behind a flag for one commit to A/B the
ISA, then delete.
