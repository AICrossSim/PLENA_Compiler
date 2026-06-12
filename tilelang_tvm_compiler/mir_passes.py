"""MIR optimisation passes.

Each pass takes a :class:`mir.MirFunction` and mutates it in place;
returns ``True`` if it changed anything (so a driver can iterate to
a fixed point). Passes are independent; ``run_default_pipeline`` ties
them together in a sensible order.

Implemented passes
------------------

* :func:`dead_loop_elim` — eliminate ``extent <= 1`` loops. ``extent
  == 0`` deletes the whole loop; ``extent == 1`` peels the body up
  one level, replacing ``loop_var`` uses with ``IntImm(init)``.

* :func:`const_fold` — fold instructions whose i32 inputs are all
  compile-time constants (``IntImm``). Substitutes the resulting
  constant value into every use, removes the now-dead instruction.
  Handles the same set the runtime ALU ops cover (S_ADDI/SLLI/SRLI/
  ADD/SUB/MUL).

* :func:`dce` — dead-code elimination. Drops any non-side-effecting
  MirInstr whose result has no users. Side-effecting opcodes
  (memory writes, control-register sets, HW ops) are kept regardless.

* :func:`cse` — common subexpression elimination within a block.
  Two MirInstrs with the same opcode and operand identities collapse
  to one; the duplicate's result has its uses redirected to the first.

* :func:`reassociate` — flatten chains of ``S_ADD_INT`` /
  ``S_ADDI_INT`` into multi-term sums, canonicalise the term lists,
  fold IntImm terms, then rebuild as left-associative chains that
  share the longest common prefix with any already-existing chain.
  This is what lets two address PrimExprs like
  ``mat = head*hlen + oc*blen + base`` and
  ``result = orow*S + head*hlen + oc*blen + base`` collapse so that
  ``result`` literally becomes ``S_ADD_INT %mat, %orow_term``
  instead of recomputing four operands from scratch.

* :func:`run_default_pipeline` — runs DLE → const_fold → DCE →
  reassociate → CSE → LICM(optional) to a fixed point (or until
  ``max_iters`` is reached).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from tvm import tir

from . import mir


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _replace_all_uses(old: mir.MirValue, new: "mir.MirOperand") -> None:
    """Replace every use of ``old`` with ``new`` across the function.

    ``new`` can be any MirOperand (MirValue, int, IntImm, str). We
    walk ``old.used_by`` (a snapshot to avoid mutation-during-iter)
    and ask each using instr to swap. Each ``set_operand`` call
    updates ``new.used_by`` (if new is a MirValue) and removes the
    using instr from ``old.used_by`` automatically.
    """
    for user in list(old.used_by):
        for i, op in enumerate(list(user.operands)):
            if op is old:
                user.set_operand(i, new)


def _is_int_const(op: "mir.MirOperand") -> bool:
    """True iff ``op`` represents a compile-time integer constant.

    Three forms count:
      * Python ``int``
      * ``tir.IntImm``
      * function-level constants (``gp0`` = 0)
      * a MirValue produced by ``S_ADDI_INT gp0, K`` (the canonical
        "materialise constant" idiom). This makes const-prop
        transitive: the result of a previous fold becomes a const
        input for the next.
    """
    if isinstance(op, int):
        return True
    if isinstance(op, tir.IntImm):
        return True
    if isinstance(op, mir.MirValue):
        if op.is_function_const:
            return True
        d = op.defined_by
        if (d is not None and d.opcode == "S_ADDI_INT"
                and isinstance(d.operands[0], mir.MirValue)
                and d.operands[0].is_function_const
                and isinstance(d.operands[1], int)):
            return True
    return False


def _int_value(op: "mir.MirOperand") -> int:
    if isinstance(op, int):
        return int(op)
    if isinstance(op, tir.IntImm):
        return int(op.value)
    if isinstance(op, mir.MirValue):
        if op.is_function_const:
            return 0
        d = op.defined_by
        if (d is not None and d.opcode == "S_ADDI_INT"
                and isinstance(d.operands[0], mir.MirValue)
                and d.operands[0].is_function_const
                and isinstance(d.operands[1], int)):
            return int(d.operands[1])
    raise TypeError(f"_int_value: not a constant: {op!r}")


def _is_side_effecting(instr: mir.MirInstr) -> bool:
    """True iff this instr has effects beyond its SSA result.

    Anything writing to memory (S_ST_*, M_*_WO, H_STORE_V), binding a
    control register (C_SET_*), launching a HW kernel (M_MM, M_BTMM,
    H_PREFETCH_*, V_RED_*, M_TMM …) — none of those are safe to drop
    by DCE no matter how unused their (often void) result is. We
    err on the side of caution: keep anything NOT in the known-pure
    set.

    Pure: scalar ALU ops that compute one i32 from inputs (no I/O,
    no register-file bind). Their result is the *only* observable
    effect.
    """
    pure = {
        "S_ADDI_INT", "S_SLLI_INT", "S_SRLI_INT", "S_LUI_INT",
        "S_SLL_INT", "S_SRL_INT",
        "S_ADD_INT", "S_SUB_INT", "S_MUL_INT",
        "S_DIV_INT", "S_REM_INT",
    }
    return instr.opcode not in pure


def _walk_blocks(fn: mir.MirFunction):
    """Yield every MirBlock in the function, recursing into loop
    bodies. Order: outer block first, then nested bodies in source
    order. Sufficient for passes that just need "every block"."""
    def _recurse(blk: mir.MirBlock):
        yield blk
        for item in blk.items:
            if isinstance(item, mir.MirLoop):
                for b in item.body:
                    yield from _recurse(b)
    for blk in fn.blocks:
        yield from _recurse(blk)


def _walk_blocks_with_parent_item_lists(fn: mir.MirFunction):
    """Yield every (block, items_list_owning_the_block, block_index)
    triple. Used by DLE to splice a loop's body items back into the
    parent items list. For top-level blocks, ``items_list_owning_the_block``
    is ``fn.blocks`` and ``block_index`` is its index there."""
    # Top-level blocks live in fn.blocks.
    for i, blk in enumerate(fn.blocks):
        yield blk, fn.blocks, i
        # Recurse into loops inside this block.
        yield from _recurse_loops_in_block(blk)


def _recurse_loops_in_block(blk: mir.MirBlock):
    for item in blk.items:
        if isinstance(item, mir.MirLoop):
            for i, body_blk in enumerate(item.body):
                yield body_blk, item.body, i
                yield from _recurse_loops_in_block(body_blk)


# ---------------------------------------------------------------------
# Pass 1: dead loop elimination
# ---------------------------------------------------------------------

def dead_loop_elim(fn: mir.MirFunction) -> bool:
    """Eliminate ``extent <= 1`` loops.

    Strategy: walk every block; for each loop child whose extent is
    0 (delete) or 1 (peel), rewrite in place. Peeling moves the
    loop body's items into the parent block at the loop's old
    position, substituting ``IntImm(init)`` for every use of
    ``loop_var``.

    Caveat: the body block's own ``arguments`` list (which contained
    ``loop_var``) is discarded as the items splice up — the parent
    block is the new owner, and the loop_var has been RAUW'd away
    so nobody references it any more.

    Returns True if any loop was removed/peeled.
    """
    changed = False

    def _process_block(parent_blk: mir.MirBlock) -> None:
        """Rewrite ``parent_blk.items`` in place. We pass the OWNING
        block (not just the items list) so we can re-wire spliced
        items' ``parent``/``parent_block`` to the new home."""
        nonlocal changed
        items = parent_blk.items
        i = 0
        while i < len(items):
            it = items[i]
            if not isinstance(it, mir.MirLoop):
                i += 1
                continue
            lp = it
            # Recurse inside-out — peeling an inner extent-1 loop
            # exposes the items in its parent (= this lp's body),
            # which may itself become peelable in a later iteration.
            for body_blk in lp.body:
                _process_block(body_blk)

            if lp.extent <= 0:
                for use in list(lp.loop_var.used_by):
                    raise mir.MirVerifyError(
                        f"DLE: cannot delete loop {lp.name!r} "
                        f"(extent=0): loop_var still has user "
                        f"{use!r} outside the body"
                    )
                del items[i]
                changed = True
                continue

            if lp.extent == 1:
                if len(lp.body) != 1:
                    i += 1
                    continue
                body = lp.body[0]
                # Use a plain Python int (not tir.IntImm): the emit layer's
                # _fmt_operand accepts MirValue / int but NOT tir.IntImm, and
                # const_fold's _is_int_const treats int as constant too. Passing
                # IntImm here leaked into S_MUL_INT operands -> "got 0" at emit.
                init_const = int(lp.init)
                _replace_all_uses(lp.loop_var, init_const)
                lp.loop_var.block_arg_of = None
                # Splice body items into parent_blk at position
                # ``i`` (replacing the loop). Set each spliced
                # item's parent pointer to ``parent_blk`` — they
                # now live there, not in the discarded body
                # block. Without this rewire, a child loop's
                # ``parent_block`` would point at the dead body
                # block and any later scope walk would see a
                # broken chain (the dominance check would fail
                # to recognise valid uses).
                spliced = list(body.items)
                for sub in spliced:
                    if isinstance(sub, mir.MirInstr):
                        sub.parent = parent_blk
                    elif isinstance(sub, mir.MirLoop):
                        sub.parent_block = parent_blk
                items[i:i + 1] = spliced
                changed = True
                continue

            i += 1

    for blk in fn.blocks:
        _process_block(blk)
    return changed


# ---------------------------------------------------------------------
# Pass 2: constant folding
# ---------------------------------------------------------------------

def const_fold(fn: mir.MirFunction) -> bool:
    """Fold pure ALU instructions whose i32 inputs are all integer
    constants. The instruction's operands are rewritten in place:
    when both inputs are const, the instr is collapsed into the
    canonical "materialise constant" form ``S_ADDI_INT gp0, K``
    (or rewritten to be a ``S_LUI_INT`` + ``S_ADDI_INT`` chain when
    K exceeds the 18-bit immediate range — but for now we just emit
    a single S_ADDI_INT and rely on a later pass to handle wide
    immediates).

    Why rewrite the SAME instr instead of RAUW'ing an IntImm? The
    MIR i32 operand kind requires a MirValue at every use site
    (the emit layer turns it into ``gp{N}``). A bare IntImm in an
    i32 slot has no GP and cannot be emitted. By rewriting the
    folding instr to ``S_ADDI_INT gp0, K``, the value's identity
    is preserved (same MirValue, same uses), but its definition
    becomes a cheap "load constant into GP" — and any number of
    folded ADDIs producing the same K collapse to one via CSE.

    Recognised opcodes (the pure set):

      * ``S_ADDI_INT (x, k)``      → x + k        (k is literal_int)
      * ``S_SLLI_INT (x, k)``      → x << k
      * ``S_SRLI_INT (x, k)``      → x >> k
      * ``S_LUI_INT (k)``          → k << 12
      * ``S_SLL_INT (x, y)``       → x << y
      * ``S_SRL_INT (x, y)``       → x >> y
      * ``S_ADD_INT (x, y)``       → x + y
      * ``S_SUB_INT (x, y)``       → x - y
      * ``S_MUL_INT (x, y)``       → x * y

    Returns True if any instruction was rewritten.
    """
    changed = False

    foldable = {"S_ADDI_INT", "S_SLLI_INT", "S_SRLI_INT", "S_LUI_INT",
                "S_SLL_INT", "S_SRL_INT",
                "S_ADD_INT", "S_SUB_INT", "S_MUL_INT"}

    def _try_fold(instr: mir.MirInstr) -> Optional[int]:
        op = instr.opcode
        if op not in foldable:
            return None
        ops = instr.operands
        if op == "S_LUI_INT":
            if not _is_int_const(ops[0]):
                return None
            return _int_value(ops[0]) << 12
        # All others: 2 operands. The first is always i32, the
        # second varies (literal_int for immediates, i32 for reg).
        if not (_is_int_const(ops[0]) and _is_int_const(ops[1])):
            return None
        a, b = _int_value(ops[0]), _int_value(ops[1])
        if op == "S_ADDI_INT" or op == "S_ADD_INT":
            return a + b
        if op == "S_SUB_INT":
            return a - b
        if op == "S_MUL_INT":
            return a * b
        if op == "S_SLLI_INT" or op == "S_SLL_INT":
            return a << b
        if op == "S_SRLI_INT" or op == "S_SRL_INT":
            return a >> b
        return None

    # Iterate to a local fixed point. Each round we try BOTH
    # constant folding and algebraic identity peepholes:
    #
    #   * ``S_ADD x, gp0`` / ``S_ADD gp0, x`` → x
    #   * ``S_SUB x, gp0`` → x
    #   * ``S_MUL gp0, _`` / ``S_MUL _, gp0`` → 0 → gp0
    #   * ``S_SLLI gp0, _`` / ``S_SRLI gp0, _`` → gp0
    #   * folded result == 0 → RAUW to gp0 (don't materialise zero)
    #
    # These identity hits are common after the per-PreIsaOp address
    # PrimExpr lowering: many sub-expressions reduce to additions
    # against gp0 (e.g. ``0 * stride + offset``).
    def _identity_rewrite(instr: mir.MirInstr) -> bool:
        """Return True if we RAUW'd the result to a simpler operand
        (caller marks ``changed`` and continues — the instr's now
        dead; DCE sweeps)."""
        op = instr.opcode
        ops = instr.operands
        gp0 = fn.gp0_value

        def _is_gp0(o):
            return isinstance(o, mir.MirValue) and o.is_function_const

        if op == "S_ADD_INT":
            if _is_gp0(ops[0]):
                _replace_all_uses(instr.result, ops[1])
                return True
            if _is_gp0(ops[1]):
                _replace_all_uses(instr.result, ops[0])
                return True
        elif op == "S_SUB_INT":
            if _is_gp0(ops[1]):
                _replace_all_uses(instr.result, ops[0])
                return True
        elif op == "S_MUL_INT":
            if _is_gp0(ops[0]) or _is_gp0(ops[1]):
                _replace_all_uses(instr.result, gp0)
                return True
        elif op in ("S_SLLI_INT", "S_SRLI_INT", "S_SLL_INT", "S_SRL_INT"):
            if _is_gp0(ops[0]):
                _replace_all_uses(instr.result, gp0)
                return True
        elif op == "S_ADDI_INT":
            # ``ADDI gp0, 0`` is gp0 itself.
            if _is_gp0(ops[0]) and isinstance(ops[1], int) and ops[1] == 0:
                _replace_all_uses(instr.result, gp0)
                return True
            # ``ADDI x, 0`` is x.
            if isinstance(ops[1], int) and ops[1] == 0:
                _replace_all_uses(instr.result, ops[0])
                return True
        return False

    for _iteration in range(1000):
        any_this_round = False
        for instr in fn.walk_instrs():
            if instr.result is None:
                continue
            # Identity peephole first — cheaper, and tends to
            # expose more folding opportunities (a gp0 propagating
            # through ADD chains).
            if _identity_rewrite(instr):
                any_this_round = True
                changed = True
                continue
            folded = _try_fold(instr)
            if folded is None:
                continue
            # If folded value is 0, prefer RAUW to gp0 over
            # materialising an ``ADDI gp0, 0``.
            if folded == 0:
                _replace_all_uses(instr.result, fn.gp0_value)
                # detach operands
                for op in list(instr.operands):
                    if isinstance(op, mir.MirValue):
                        try:
                            op.used_by.remove(instr)
                        except ValueError:
                            pass
                any_this_round = True
                changed = True
                continue
            # Wide immediates (>= 2^17 abs) need S_LUI+ADDI; leave
            # them as-is for now (the original PrimExpr lowering
            # already handled wide imms; if we hit one here it
            # came from a fold and we don't want to over-eagerly
            # rewrite).
            if not (-(1 << 16) <= folded < (1 << 16)):
                continue
            # Detach old operands.
            for op in list(instr.operands):
                if isinstance(op, mir.MirValue):
                    try:
                        op.used_by.remove(instr)
                    except ValueError:
                        pass
            # Rewrite this instr to ``S_ADDI_INT gp0, folded``.
            instr.opcode = "S_ADDI_INT"
            instr.operands = [fn.gp0_value, int(folded)]
            fn.gp0_value.used_by.append(instr)
            any_this_round = True
            changed = True
        if not any_this_round:
            break
    return changed


# ---------------------------------------------------------------------
# Pass 3: dead code elimination
# ---------------------------------------------------------------------

def dce(fn: mir.MirFunction) -> bool:
    """Drop pure MirInstrs whose result has no users.

    A "pure" instr is one whose only effect is producing its SSA
    result (no memory write, no control-register bind, no HW kernel
    issue). See ``_is_side_effecting`` for the keep-set.

    Iterates to fixed point — removing one instr may make its
    operands' producers dead too.

    Returns True if any instruction was removed.
    """
    changed = False

    for _iteration in range(1000):
        any_this_round = False
        for blk in _walk_blocks(fn):
            i = 0
            while i < len(blk.items):
                it = blk.items[i]
                if not isinstance(it, mir.MirInstr):
                    i += 1
                    continue
                if it.result is None:
                    i += 1
                    continue
                if _is_side_effecting(it):
                    i += 1
                    continue
                if it.result.used_by:
                    i += 1
                    continue
                # Drop. First sever this instr's uses on its
                # operands (so their used_by lists don't hold
                # dangling pointers).
                for op in list(it.operands):
                    if isinstance(op, mir.MirValue):
                        try:
                            op.used_by.remove(it)
                        except ValueError:
                            pass
                # Drop the result's defined_by binding too —
                # the MirValue becomes orphaned and unreachable.
                it.result.defined_by = None
                del blk.items[i]
                any_this_round = True
                changed = True
            # don't advance i — items shifted left
        if not any_this_round:
            break
    return changed


# ---------------------------------------------------------------------
# Pass 4: common subexpression elimination (intra-block)
# ---------------------------------------------------------------------

def cse(fn: mir.MirFunction) -> bool:
    """Within each block, collapse duplicate pure expressions.

    Two MirInstrs are duplicates when they have:
      * the same opcode,
      * the same number of operands, and
      * each operand pair is either identical Python int / IntImm
        (by value), the same str token, or the same MirValue (by
        identity).

    The FIRST occurrence wins; later occurrences have their result's
    uses redirected to the first occurrence's result, then the
    duplicate is dropped.

    Only pure instructions are eligible (CSE'ing a memory write or
    control-register set would change program behaviour).

    Block-local only: a value defined in an outer block can be CSE'd
    against another defined in an inner block, but we don't yet
    move definitions across block boundaries (that's LICM). The
    common case — repeated address-base computation inside a single
    loop body — is fully handled here.

    Returns True if any instruction was eliminated.
    """
    changed = False

    def _operand_key(op):
        if isinstance(op, mir.MirValue):
            return ("v", id(op))
        if isinstance(op, tir.IntImm):
            return ("i", int(op.value))
        if isinstance(op, int):
            return ("i", int(op))
        if isinstance(op, str):
            return ("s", op)
        return ("o", repr(op))

    def _process_block(blk: mir.MirBlock):
        nonlocal changed
        # opcode + operand-keys-tuple → first MirInstr that
        # produced it. Reset per block (no cross-block CSE
        # without dominator info).
        seen = {}
        i = 0
        while i < len(blk.items):
            it = blk.items[i]
            if not isinstance(it, mir.MirInstr):
                i += 1
                # Loops are entered separately below.
                continue
            if it.result is None or _is_side_effecting(it):
                i += 1
                continue
            key = (
                it.opcode,
                tuple(_operand_key(o) for o in it.operands),
            )
            if key in seen:
                first = seen[key]
                # Redirect uses of this duplicate to the first
                # producer's result.
                _replace_all_uses(it.result, first.result)
                # Drop this instr.
                for op in list(it.operands):
                    if isinstance(op, mir.MirValue):
                        try:
                            op.used_by.remove(it)
                        except ValueError:
                            pass
                it.result.defined_by = None
                del blk.items[i]
                changed = True
                continue
            seen[key] = it
            i += 1
        # Recurse into nested loops.
        for it in blk.items:
            if isinstance(it, mir.MirLoop):
                for body_blk in it.body:
                    _process_block(body_blk)

    for blk in fn.blocks:
        _process_block(blk)
    return changed


# ---------------------------------------------------------------------
# Pass 5: loop-invariant code motion (LICM)
# ---------------------------------------------------------------------

def licm(fn: mir.MirFunction) -> bool:
    """Hoist loop-invariant pure instructions out of their enclosing
    loop.

    An instruction is invariant in loop L if every MirValue operand
    is defined OUTSIDE L's body (i.e. in an ancestor block) or is a
    block argument of an ancestor block / function-level constant.
    Only pure instructions (see ``_is_side_effecting``) are eligible;
    hoisting a HW kernel issue or a control-register set would
    change program semantics.

    Hoisting strategy: move the invariant instr from its body
    position to the parent block, immediately before the loop itself.
    The MIR scope rules guarantee this is safe — the parent block is
    an ancestor of every body block, so all SSA uses still see a
    valid def. (PLENA MIR has no MirIf, so there are no branch
    paths to confuse this.)

    Iterates to fixed point — hoisting an inner-loop instr can
    expose an outer-loop invariant.
    """
    changed = False

    def _is_invariant(instr: mir.MirInstr,
                      forbidden_defs: set) -> bool:
        """``instr`` is invariant w.r.t. some loop iff none of its
        operand MirValues were defined inside that loop (i.e.
        ``forbidden_defs`` contains every MirValue defined inside
        the loop's body, including the loop_var)."""
        if _is_side_effecting(instr):
            return False
        for op in instr.operands:
            if not isinstance(op, mir.MirValue):
                continue  # literals are always invariant
            if id(op) in forbidden_defs:
                return False
        return True

    def _defs_inside(blk: mir.MirBlock, out: set) -> None:
        """Collect ``id()`` of every MirValue defined inside this
        block subtree (block args + instr results + nested loop
        body args + nested instr results)."""
        for arg in blk.arguments:
            out.add(id(arg))
        for item in blk.items:
            if isinstance(item, mir.MirInstr):
                if item.result is not None:
                    out.add(id(item.result))
            elif isinstance(item, mir.MirLoop):
                for b in item.body:
                    _defs_inside(b, out)

    def _process_loop_in(parent_items: List, loop_idx: int) -> None:
        """Hoist any invariant instrs from ``parent_items[loop_idx]``
        (a MirLoop) up into ``parent_items`` just before it."""
        nonlocal changed
        lp = parent_items[loop_idx]
        # Collect defs inside the loop. We re-collect after each
        # hoist (a hoisted instr's result no longer counts as
        # inside).
        while True:
            forbidden = set()
            for b in lp.body:
                _defs_inside(b, forbidden)
            # Try to find ONE invariant instr in this iteration;
            # if found, hoist it and recompute forbidden.
            hoisted = False
            for b in lp.body:
                for i, it in enumerate(b.items):
                    if not isinstance(it, mir.MirInstr):
                        continue
                    if _is_invariant(it, forbidden):
                        # Remove from body.
                        del b.items[i]
                        # Insert in parent_items just before lp.
                        # parent_blk is the block that owns ``lp``;
                        # the hoisted instr now lives there, so wire
                        # its ``parent`` accordingly. Without this,
                        # verify's dominance check (which looks at
                        # ``defined_by.parent``) silently sees None
                        # and skips the check — a false-negative
                        # rather than a real OK.
                        parent_items.insert(loop_idx, it)
                        it.parent = lp.parent_block
                        loop_idx += 1  # lp shifted right by 1
                        hoisted = True
                        changed = True
                        break
                if hoisted:
                    break
            if not hoisted:
                break
        # Recurse into nested loops AFTER the outer is stable —
        # an inner-loop hoist that puts a def into THIS loop's
        # body doesn't make it invariant for THIS loop (the inner
        # is still inside), so order doesn't matter much; we
        # process inside-out by recursing here.
        for b in lp.body:
            i = 0
            while i < len(b.items):
                it = b.items[i]
                if isinstance(it, mir.MirLoop):
                    _process_loop_in(b.items, i)
                i += 1

    def _process_block(blk: mir.MirBlock) -> None:
        i = 0
        while i < len(blk.items):
            it = blk.items[i]
            if isinstance(it, mir.MirLoop):
                _process_loop_in(blk.items, i)
            i += 1

    for blk in fn.blocks:
        _process_block(blk)
    return changed


# ---------------------------------------------------------------------
# Pass 6: reassociate ADD chains for sharing
# ---------------------------------------------------------------------

def reassociate(fn: mir.MirFunction) -> bool:
    """Canonicalise ``S_ADD_INT`` / ``S_ADDI_INT`` chains so that
    chains with overlapping term sets share their common prefix.

    A "chain" is the transitive closure of ``S_ADD_INT`` /
    ``S_ADDI_INT`` instrs whose operands are themselves chain
    results. A *leaf* is any operand that is NOT a chain result: a
    block argument, an instruction with a non-additive opcode, an
    IntImm, gp0, etc. By recursively unfolding chain operands we
    rewrite each chain instr's "view" as a multiset of leaf terms
    + one folded IntImm constant.

    Sharing rule: any two chains with the same canonical leaf list
    collapse to the same MirValue (RAUW on duplicate result). Two
    chains where one's leaf list is a **prefix** of the other's
    share the partial-sum value: the longer chain's result becomes
    ``S_ADD_INT shorter_result, extra_leaf``.

    Caveats:
      * An interior chain instr with multiple users is treated as
        an opaque leaf — splitting it would force its other users
        to also restructure (and would lose CSE if they were a
        chain too). It can still ANCHOR other chains as a leaf,
        which is what enables the "result = mat + orow_term" case
        without breaking ``mat`` itself.
      * IntImm folding here covers the same arithmetic
        ``const_fold`` does on a per-instr basis, but with the
        full multiset visible — so ``(((x+0)+3)+5)+x`` reduces to
        ``2*x+8`` (well, we don't synthesise multiplication, but
        we DO collapse the two ``x`` and the ``0+3+5`` into a
        canonical form that CSE then dedups across chains).

    This is one outer iteration: rebuild chains greedily,
    largest-first (so longer chains get a chance to anchor the
    "shared prefix" slot before shorter chains decide what to
    point at). Outer driver re-runs the rest of the pipeline +
    this pass to a fixed point.

    Returns True if any chain was rewritten.
    """
    changed = False

    def _is_add_instr(instr) -> bool:
        return (instr is not None
                and instr.opcode in (
                    "S_ADD_INT", "S_ADDI_INT", "S_SUB_INT",
                ))

    def _is_chain_result(op) -> bool:
        """``op`` is a MirValue defined by an ADD/ADDI/SUB instr
        with EXACTLY ONE user. Multi-user chain instrs are
        barriers — absorbing them would orphan the other users."""
        if not isinstance(op, mir.MirValue):
            return False
        d = op.defined_by
        if d is None:
            return False
        if not _is_add_instr(d):
            return False
        if len(op.used_by) != 1:
            return False
        return True

    # ---- Phase 1: flatten ----
    #
    # Each chain instr collapses into ``(signed_leaves, const)`` where
    # ``signed_leaves`` is a list of ``(sign, MirValue)`` and
    # ``const`` is the (signed) integer sum. ``S_SUB_INT a, b`` is
    # treated as ``+a + (-b)`` so its leaves get split into a
    # positive group (from a) and a negative group (from b). This
    # turns SUB into just-another-ADD with negated terms, letting
    # the rest of the algorithm (canonical sort + prefix cache)
    # work unchanged. Reconstruction (Phase 2) re-emits SUBs only
    # if any leaf has sign = -1.
    #
    # When absorbing a sub-chain whose head sign is ``s``, we visit
    # its operands with sign ``s`` for the first operand and ``s``
    # for the second (ADD/ADDI) or ``-s`` (SUB second operand).

    def _flatten(instr) -> Tuple[List[Tuple[int, "mir.MirValue"]], int]:
        """Return ``(signed_leaves, signed_const)``. ``signed_leaves``
        is a list of ``(sign ∈ {+1,-1}, MirValue)`` sorted by
        (sign, name) for a stable canonical form."""
        leaves: List[Tuple[int, "mir.MirValue"]] = []
        const = 0
        # Worklist of (sign, operand, opcode-of-parent, operand-index)
        # — opcode + index lets us flip the sign on SUB's second
        # operand. We seed with ``instr``'s operands directly.
        work: List[Tuple[int, "mir.MirOperand"]] = []
        # For the top-level instr: each operand inherits sign +1,
        # except the second operand of S_SUB_INT flips to -1.
        for i, op in enumerate(instr.operands):
            sign = 1
            if instr.opcode == "S_SUB_INT" and i == 1:
                sign = -1
            work.append((sign, op))

        while work:
            sign, op = work.pop()
            if isinstance(op, int):
                const += sign * op
                continue
            if isinstance(op, tir.IntImm):
                const += sign * int(op.value)
                continue
            if isinstance(op, mir.MirValue):
                if op.is_function_const:
                    continue
                if _is_chain_result(op):
                    d = op.defined_by
                    # Absorb: push d's operands with the right
                    # signs given the outer ``sign``.
                    for j, sub_op in enumerate(d.operands):
                        sub_sign = sign
                        if d.opcode == "S_SUB_INT" and j == 1:
                            sub_sign = -sign
                        work.append((sub_sign, sub_op))
                    continue
                leaves.append((sign, op))
                continue
            # Unknown — drop silently.
        # Cancel matching +/- pairs of the SAME MirValue.
        # ``%x + %x - %x`` reduces to ``%x``.
        canon: Dict[int, int] = {}        # id(v) -> net sign sum
        canon_val: Dict[int, "mir.MirValue"] = {}
        for s, v in leaves:
            canon[id(v)] = canon.get(id(v), 0) + s
            canon_val[id(v)] = v
        out: List[Tuple[int, "mir.MirValue"]] = []
        for vid, net in canon.items():
            v = canon_val[vid]
            # net of +2 / -2 / etc. could be expressed as
            # multiplication; we don't synth muls. Keep |net| as
            # repeated additions/subtractions if the magnitude is
            # small (1 is by far the common case). For >|1| we
            # just expand into |net| copies — rare in address code.
            if net == 0:
                continue
            mag = abs(net)
            sub_sign = 1 if net > 0 else -1
            for _ in range(mag):
                out.append((sub_sign, v))
        out.sort(key=lambda sv: (sv[0], sv[1].name))
        return out, const

    # ---- Phase 2: per-block rebuild ----
    def _process_block(blk: mir.MirBlock) -> None:
        nonlocal changed
        # Cache: (tuple of (sign, leaf-id), const) → MirValue.
        prefix_cache: Dict[Tuple, "mir.MirValue"] = {}

        chain_entries = []
        for idx, it in enumerate(blk.items):
            if isinstance(it, mir.MirInstr) and _is_add_instr(it):
                leaves, const = _flatten(it)
                chain_entries.append((idx, it, leaves, const))

        if not chain_entries:
            for it in blk.items:
                if isinstance(it, mir.MirLoop):
                    for b in it.body:
                        _process_block(b)
            return

        def _key(signed_leaves, const):
            return (tuple((s, id(v)) for s, v in signed_leaves), const)

        for _, instr, leaves, const in chain_entries:
            if instr not in blk.items:
                continue

            k = _key(leaves, const)
            if k in prefix_cache:
                cached = prefix_cache[k]
                if cached is instr.result:
                    continue
                _replace_all_uses(instr.result, cached)
                changed = True
                continue

            # Reconstruction strategy.
            #
            # signed_leaves is sorted (sign, name). Sign +1 first,
            # then -1. We:
            #   1. Build the positive sub-sum first using the same
            #      prefix-cache trick (longest matching pos prefix
            #      reuses an existing partial sum).
            #   2. For each negative leaf, emit S_SUB_INT acc, leaf.
            #   3. Finally fold ``const`` into the accumulator via
            #      S_ADDI_INT acc, const (PLENA ADDI takes a
            #      signed 18-bit immediate, so negative const works
            #      too).
            pos_leaves = [v for s, v in leaves if s > 0]
            neg_leaves = [v for s, v in leaves if s < 0]

            # ---- find prefix cache hit for pos_leaves ----
            best_prefix = None
            best_prefix_value = None
            best_prefix_const_in = False
            for n in range(len(pos_leaves), 0, -1):
                # Pos prefix means signed leaves of the prefix are
                # all (+1, v). The cache key for a positive-only
                # prefix uses sign tag +1.
                pref_signed = [(1, v) for v in pos_leaves[:n]]
                sub_key_full = _key(pref_signed, const)
                if sub_key_full in prefix_cache:
                    best_prefix = pos_leaves[:n]
                    best_prefix_value = prefix_cache[sub_key_full]
                    best_prefix_const_in = True
                    break
                sub_key_no_c = _key(pref_signed, 0)
                if sub_key_no_c in prefix_cache:
                    best_prefix = pos_leaves[:n]
                    best_prefix_value = prefix_cache[sub_key_no_c]
                    best_prefix_const_in = False
                    break

            if best_prefix is not None:
                start_value = best_prefix_value
                remaining_pos = pos_leaves[len(best_prefix):]
                pending_const = 0 if best_prefix_const_in else const
            else:
                # No pos-prefix match; start from scratch.
                if not pos_leaves:
                    if not neg_leaves:
                        # All-const sum — const_fold should have
                        # handled it; skip.
                        continue
                    # Pure negative sum: ``0 - n1 - n2 ...``. Start
                    # with the negative of the first neg_leaf? PLENA
                    # has no NEG; emit as ``S_SUB_INT gp0, neg[0]``,
                    # then SUB the rest, then ADDI const.
                    start_value = fn.gp0_value
                    remaining_pos = []
                    pending_const = const
                else:
                    start_value = pos_leaves[0]
                    remaining_pos = pos_leaves[1:]
                    pending_const = const

            # ---- build tail ----
            # Each tail element is (opcode, second_operand) where:
            #   ("S_ADD_INT", MirValue) — pos leaf
            #   ("S_SUB_INT", MirValue) — neg leaf
            #   ("S_ADDI_INT", int)     — pending const fold
            tail = []
            for v in remaining_pos:
                tail.append(("S_ADD_INT", v))
            for v in neg_leaves:
                tail.append(("S_SUB_INT", v))
            if pending_const != 0:
                # Check range; out of range → don't fold here, drop.
                if -(1 << 16) <= pending_const < (1 << 16):
                    tail.append(("S_ADDI_INT", int(pending_const)))
                else:
                    # Wide const — leave the original instr alone.
                    # (Rare; const_fold handles ordinary 18-bit
                    # range. Pathological case skip.)
                    continue

            insert_idx = blk.items.index(instr)
            accum = start_value

            def _emit_partial(opcode, src, second):
                nonlocal insert_idx
                dst = fn.mint_value("i32")
                new_instr = mir.MirInstr(
                    opcode=opcode, operands=[src, second], result=dst,
                )
                new_instr.parent = blk
                blk.items.insert(insert_idx, new_instr)
                insert_idx += 1
                return dst

            if not tail:
                _replace_all_uses(instr.result, start_value)
                prefix_cache[k] = start_value
                changed = True
                continue

            # Track running canonical key for caching partial sums.
            # We mirror the same canonical form _flatten produces:
            # sorted by (sign, name). So the partial-sum at each
            # tail step is the input prefix + leaves consumed by
            # tail so far. Easiest: just rebuild the canonical
            # signed-leaf list for the current accum.
            cur_signed = [(1, v) for v in (best_prefix
                                           if best_prefix is not None
                                           else (
                                               [pos_leaves[0]]
                                               if pos_leaves else []
                                           ))]
            cur_const = const if best_prefix_const_in else 0

            for step_i, (step_op, step_arg) in enumerate(tail):
                last = (step_i == len(tail) - 1)
                if last:
                    # Reuse instr.
                    for op in list(instr.operands):
                        if isinstance(op, mir.MirValue):
                            try:
                                op.used_by.remove(instr)
                            except ValueError:
                                pass
                    instr.opcode = step_op
                    instr.operands = [accum, step_arg]
                    if isinstance(accum, mir.MirValue):
                        accum.used_by.append(instr)
                    if isinstance(step_arg, mir.MirValue):
                        step_arg.used_by.append(instr)
                    new_val = instr.result
                else:
                    new_val = _emit_partial(step_op, accum, step_arg)
                accum = new_val

                # Update the running canonical state + cache the
                # partial sum so later chains find it.
                if step_op == "S_ADD_INT":
                    cur_signed = sorted(
                        cur_signed + [(1, step_arg)],
                        key=lambda sv: (sv[0], sv[1].name),
                    )
                elif step_op == "S_SUB_INT":
                    cur_signed = sorted(
                        cur_signed + [(-1, step_arg)],
                        key=lambda sv: (sv[0], sv[1].name),
                    )
                elif step_op == "S_ADDI_INT":
                    cur_const += int(step_arg)
                # Cache this partial sum's key for downstream reuse.
                prefix_cache[(tuple((s, id(v)) for s, v in cur_signed),
                              cur_const)] = accum

            # Final partial-sum cache entry already wrote the full
            # canonical key on the last loop iteration. Belt-and-
            # braces: re-cache the original (k) form too. By the
            # construction above (last tail step reuses ``instr``),
            # ``accum is instr.result``.
            assert accum is instr.result, (
                f"reassoc: tail's last step should reuse instr; "
                f"got accum={accum!r} vs instr.result={instr.result!r}"
            )
            prefix_cache[k] = accum
            changed = True

        # Recurse into nested loops.
        for it in blk.items:
            if isinstance(it, mir.MirLoop):
                for b in it.body:
                    _process_block(b)

    for blk in fn.blocks:
        _process_block(blk)
    return changed


# ---------------------------------------------------------------------
# Pass: unroll
# ---------------------------------------------------------------------

def _clone_operand(
    op: "mir.MirOperand",
    value_map: Dict["mir.MirValue", "mir.MirOperand"],
) -> "mir.MirOperand":
    """Translate one operand under a SSA-value mapping.

    Plain ints / IntImms / verbatim strings are pass-through. MirValue
    operands look up ``value_map``; values not in the map (defined
    outside the cloned region — outer loop_vars, gp0, function args,
    buffer-base SSA values, …) reuse the original.
    """
    if isinstance(op, mir.MirValue):
        return value_map.get(op, op)
    return op


def _clone_instr(
    instr: mir.MirInstr,
    fn: mir.MirFunction,
    value_map: Dict["mir.MirValue", "mir.MirOperand"],
) -> mir.MirInstr:
    """Clone a MirInstr. The clone gets fresh result MirValues; the
    ``value_map`` is updated so later instrs in the same cloned region
    pick up the new defs."""
    new_operands = [_clone_operand(o, value_map) for o in instr.operands]
    new_result: Optional[mir.MirValue] = None
    if instr.result is not None:
        old = instr.result
        # Mint a fresh SSA value of the same dtype. Preserve a hint
        # so the dump stays human-readable across the unroll.
        hint = ""
        if "_" in old.name:
            hint = old.name.split("_", 1)[1]
        new_result = fn.mint_value(old.dtype, hint=hint)
        value_map[old] = new_result
    new_instr = mir.MirInstr(instr.opcode, new_operands, result=new_result)
    # Carry over any optimisation-hint annotations (cheap shallow copy).
    if instr.annotations:
        new_instr.annotations = dict(instr.annotations)
    return new_instr


def _clone_block(
    blk: mir.MirBlock,
    fn: mir.MirFunction,
    value_map: Dict["mir.MirValue", "mir.MirOperand"],
    name_suffix: str,
) -> mir.MirBlock:
    """Deep-clone a MirBlock, including any nested MirLoop regions.
    Block arguments get fresh MirValues mapped in ``value_map``."""
    new_blk = mir.MirBlock(name=f"{blk.name}{name_suffix}")
    for arg in blk.arguments:
        # If the caller (e.g. _clone_loop) already minted a fresh
        # value for this argument and put it in the map, reuse it —
        # the loop_var's MirValue identity must be the same one that
        # the enclosing MirLoop holds in ``loop_var``.
        if arg in value_map:
            existing = value_map[arg]
            if not isinstance(existing, mir.MirValue):
                raise mir.MirVerifyError(
                    f"_clone_block: block-arg {arg!r} was pre-mapped "
                    f"to a non-MirValue {existing!r}"
                )
            new_blk.add_argument(existing)
            continue
        hint = arg.name.split("_", 1)[1] if "_" in arg.name else ""
        new_arg = fn.mint_value(arg.dtype, hint=hint)
        value_map[arg] = new_arg
        new_blk.add_argument(new_arg)
    for item in blk.items:
        if isinstance(item, mir.MirInstr):
            new_blk.append(_clone_instr(item, fn, value_map))
        elif isinstance(item, mir.MirLoop):
            new_blk.append(_clone_loop(item, fn, value_map, name_suffix))
        else:
            raise TypeError(
                f"_clone_block: unexpected item {type(item).__name__}"
            )
    return new_blk


def _clone_loop(
    lp: mir.MirLoop,
    fn: mir.MirFunction,
    value_map: Dict["mir.MirValue", "mir.MirOperand"],
    name_suffix: str,
) -> mir.MirLoop:
    """Deep-clone a MirLoop with a fresh loop_var. Used when we clone
    an outer unroll body whose body contains a nested loop (typically
    a serial inner loop — nested unroll loops are already flattened
    by the innermost-first walk in :func:`unroll_loops`)."""
    hint = lp.loop_var.name.split("_", 1)[1] if "_" in lp.loop_var.name else ""
    new_lvar = fn.mint_value("i32", hint=hint)
    value_map[lp.loop_var] = new_lvar
    new_lp = mir.MirLoop(
        name=f"{lp.name}{name_suffix}",
        loop_var=new_lvar,
        init=lp.init,
        extent=lp.extent,
        body=[],
        loop_kind=lp.loop_kind,
        annotations=dict(lp.annotations),
    )
    for body_blk in lp.body:
        cloned = _clone_block(body_blk, fn, value_map, name_suffix)
        new_lp.add_body_block(cloned)
    return new_lp


def unroll_loops(fn: mir.MirFunction) -> bool:
    """Physically unroll every ``loop_kind == "unroll"`` MirLoop.

    For each unroll loop, the body is cloned ``extent`` times and the
    clones are spliced into the parent items list at the loop's old
    position. In each clone, every reference to the loop_var is
    replaced by the integer iteration value, and every body-local SSA
    def is replaced by a fresh MirValue (preserving the SSA single-def
    invariant). The loop region itself is then removed.

    Walks innermost-first so that when an outer unroll is processed,
    its body is already flat (any inner unroll has been expanded and
    constant-folded by the surrounding pipeline run). Nested serial
    loops inside an unroll body are deep-cloned per iteration.

    Why integers, not IntImms, for the iter value: a plain ``int``
    travels safely through every i32 operand slot — :func:`const_fold`
    recognises it, :func:`mir.verify` accepts it, and the
    ``mir_to_isa`` emit layer formats it directly when it lands in an
    operand slot. IntImm would force the emit layer to special-case.

    Returns True iff any loop was unrolled.
    """
    changed = False

    def _process_block(parent_blk: mir.MirBlock) -> None:
        """Innermost-first rewrite of ``parent_blk.items``. Recurses
        into every nested loop body before considering whether to
        unroll the current loop itself."""
        nonlocal changed
        items = parent_blk.items
        i = 0
        while i < len(items):
            it = items[i]
            if not isinstance(it, mir.MirLoop):
                i += 1
                continue
            lp = it
            # Recurse first so any inner unroll loops are flattened
            # before we clone the current body.
            for body_blk in lp.body:
                _process_block(body_blk)

            if lp.loop_kind != "unroll":
                i += 1
                continue

            if lp.extent < 0:
                raise mir.MirVerifyError(
                    f"unroll_loops: loop {lp.name!r} has negative "
                    f"extent {lp.extent}"
                )
            if lp.extent == 0:
                # Drop the whole loop. Its loop_var must be
                # unreferenced outside the body (block argument).
                del items[i]
                changed = True
                continue
            if len(lp.body) != 1:
                raise mir.MirVerifyError(
                    f"unroll_loops: loop {lp.name!r} has "
                    f"{len(lp.body)} body blocks; expected exactly 1"
                )
            body = lp.body[0]
            spliced: List[mir.MirInstr | mir.MirLoop] = []
            for k in range(lp.extent):
                iter_val = int(lp.init) + k
                # Fresh per-iter value map. The loop_var maps to a
                # plain int; every body-local def will be remapped to
                # a fresh MirValue as instrs are cloned.
                vmap: Dict[mir.MirValue, mir.MirOperand] = {
                    lp.loop_var: iter_val,
                }
                suffix = f"__u{k}"
                for sub in body.items:
                    if isinstance(sub, mir.MirInstr):
                        new_instr = _clone_instr(sub, fn, vmap)
                        new_instr.parent = parent_blk
                        spliced.append(new_instr)
                    elif isinstance(sub, mir.MirLoop):
                        new_lp = _clone_loop(sub, fn, vmap, suffix)
                        new_lp.parent_block = parent_blk
                        spliced.append(new_lp)
                    else:
                        raise TypeError(
                            f"unroll_loops: unexpected body item "
                            f"{type(sub).__name__}"
                        )
            # Splice in place of the loop region.
            items[i:i + 1] = spliced
            # The old loop_var is no longer referenced (every clone
            # used a plain int). Detach the block_arg pointer so
            # downstream verify doesn't see a stale arg.
            lp.loop_var.block_arg_of = None
            changed = True
            # Skip over the freshly-spliced items — they're already
            # flat and don't contain any further unroll loops to
            # process (inner unrolls were handled by the recursion
            # at the top of this iteration).
            i += len(spliced)

    for blk in fn.blocks:
        _process_block(blk)
    return changed


# ---------------------------------------------------------------------
# Default pipeline
# ---------------------------------------------------------------------

def run_default_pipeline(
    fn: mir.MirFunction, *, max_iters: int = 16,
    enable_licm: bool = False,
    dump_dir=None,
) -> List[Tuple[str, int]]:
    """Run DLE → const_fold → DCE → CSE (→ LICM) to a fixed point.

    LICM is disabled by default. It reduces instruction count but
    can increase register pressure beyond what the current
    linear-scan allocator can serve (no IntRAM spill yet). Enable
    explicitly once spill support lands.

    ``dump_dir`` (when set): after every pass that reports a change,
    write the full ``format_mir`` snapshot to
    ``<dump_dir>/NN_iterM_passname.mir`` (NN is a global step counter).
    The pre-opt and final states are dumped too. Lets you watch each
    address PrimExpr fold step by step.

    Returns a list of ``(pass_name, run_count)`` tuples for
    diagnostics — the number of outer iterations in which each
    pass returned True.
    """
    step = [0]

    def _dump(tag: str) -> None:
        if dump_dir is None:
            return
        from pathlib import Path
        d = Path(dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{step[0]:02d}_{tag}.mir"
        path.write_text(mir.format_mir(fn))
        step[0] += 1

    _dump("input")
    passes = [
        ("dead_loop_elim", dead_loop_elim),
        ("const_fold", const_fold),
        ("dce", dce),
        ("reassociate", reassociate),
        ("cse", cse),
    ]
    if enable_licm:
        passes.append(("licm", licm))

    counts = {name: 0 for name, _ in passes}
    for it in range(max_iters):
        any_change = False
        for name, fn_pass in passes:
            if fn_pass(fn):
                counts[name] += 1
                any_change = True
                _dump(f"iter{it}_{name}")
        if not any_change:
            break
    _dump("final")
    return list(counts.items())
