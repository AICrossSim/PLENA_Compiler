# Scope-recursive GP allocation (the real fix)

Replaces the linear-scan + patch pile in `mir_to_isa.py`. The MIR is
clean SCF (blocks of instrs + nested `MirLoop` regions). Allocate **over
that tree, with spill/reload happening only at scope boundaries** — never
inside a loop body. That boundary discipline is what every previous bug
violated (a spill/reload instruction landing in a loop body re-executes
every iteration with stale register state).

## 0. Why all the previous bugs were the same bug

A `S_ST_INT`/`S_LD_INT` for a value that lives ACROSS a loop must not sit
in the loop body. The body is emitted once but runs N times:
- store-in-body → each iteration overwrites the slot with whatever the GP
  now holds (the `intram[7]=2048` bug);
- reload-in-body-before-its-store → reads last iteration's clobbered GP
  (the `%74=800` bug).

Fix structurally: **spill at scope entry, reload at scope exit, both
OUTSIDE the body.** Then a body never contains a cross-boundary
spill/reload, so it can't be corrupted by re-execution.

## 1. Pre-pass: classify every value (one walk)

Walk the SCF tree once. For each `MirValue` record:

- **kind**:
  - `counter`  — a serial loop's hardware counter (minted by emit). Needs
    a GP pinned for the whole loop; HW reads/decrements it. Never spill.
  - `loop_idx` — a loop_var (block argument). Has an IntRAM idx slot;
    reloaded at each iteration header, ++ stored at footer. Never needs a
    persistent GP.
  - `normal`   — everything else (address arithmetic, temporaries).
- **def_scope**: the innermost region (loop) containing its def. Top-level
  = the function scope.
- **last_use / live scopes**: the set of scopes in which it is read.

From this derive, per scope S (a loop region or the top region):
- `live_in(S)`  — values defined OUTSIDE S, read INSIDE S (carried).
- `local(S)`    — values defined directly in S.

(`gp0` is the const-zero fixture; never allocated/spilled.)

## 2. Per-scope register budget (decide BEFORE entering)

Recurse the tree. Entering scope S we know:
- `regs_free` = GPs available from the parent.
- S needs: 1 GP for its counter (if serial loop), 1 for its loop_idx
  while live, plus enough for `local(S)` peak + the `live_in(S)` values
  it actually touches.

If `regs_free` can't cover S's needs, we **demote the lowest-priority
carried values to IntRAM at S's entry** (see §3). This is the decision
point — made structurally per scope, not reactively mid-body.

## 3. Scope entry / exit protocol (the heart of it)

Entering scope S (right after `C_LOOP_START`, before the body):
1. Counter: assign + pin its GP.
2. loop_idx: header `S_LD_INT idx_gp, slot` (already the case).
3. For each `live_in(S)` value V the body will read:
   - if a GP is available → keep V in its GP (no memory traffic);
   - else → it is **resident in IntRAM**: ensure V's value is in its slot
     (stored at S's ENTRY, outside the body), and the body reads V by
     reloading from the slot **at each use** — but those reloads are
     body-local and re-execute fine because the SLOT WAS WRITTEN AT ENTRY
     (outside the body) and V is loop-invariant.
   Key: the STORE is emitted at entry (outside body). Only LOADs are in
   the body. Loads-in-body are safe; stores-in-body are not.

Exiting scope S:
4. Any value that S's body left in IntRAM but the PARENT scope still needs
   in a GP → reload it once here (outside S, after `C_LOOP_END`).
5. Free S's counter/idx GPs.

So the invariant holds by construction: **every `S_ST_INT` is at a scope
entry, every cross-scope `S_LD_INT` is at a scope entry/exit — none in a
body that loops.** Body-internal spill/reload only ever touches `local(S)`
temporaries whose store+load are in the same straight-line body (one
iteration), which is correct.

## 4. Inside a scope body (straight-line)

Within S's body, between child loops, it's straight-line: ordinary
linear assignment from `regs_free`, and `local(S)` temporaries may
spill/reload freely (same-iteration, safe). Child `MirLoop` → recurse
(§2/§3) with the GPs not reserved by S's carried set + counter/idx.

## 5. What this kills / keeps
- Kills: `_carried_now` spill-prohibition hacks, volatile/remat tiers,
  the "spill landed in body" corruption, the four-dict state churn.
- Keeps: `compute_emit_order`, `_free_gp`/`_take_gp` invariants, IntRAM
  monotonic slot allocator, operand-lock, addr_reg/fp/gp0 handling, the
  serial-loop prologue/epilogue ISA, `_check_no_iter_args` guard.

## 6. Register pressure / failure
If even after demoting all `live_in(S)` to IntRAM a scope still can't fit
`local(S)` peak + counter + idx in the GP file, that's genuine pressure →
fail loud (or spill a `local` to IntRAM within the body, which is safe).
For fa_min: counter+idx per level (≤2/level) + a few locals; the carried
constants/address-terms go to IntRAM, reloaded per use. Fits.

## 7. Migration
Single file. New `ScopeAllocator` driven by a recursive `emit_scope`
replacing the flat `_emit_block`/`_emit_loop_serial` walk's allocation
decisions. Keep old class behind a flag for one commit to A/B the ISA,
then delete. Build the classification pre-pass first; assert it matches
the `_check_no_iter_args` expectations (only loop_idx as block arg).
