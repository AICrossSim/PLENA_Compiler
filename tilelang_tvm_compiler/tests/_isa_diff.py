"""Semantic ISA diff utility for the matmul / DMA family migration.

Used by byte-equal tests where strict register-number equality is
intractable — legacy ``emit_matmul_*`` helpers pre-allocate
``allocate_gp(6)`` blocks and use only a subset, producing scrambled
GP-number choices (``gp2``/``gp1``/``gp4``) that the PreIsaIR per-iter
materialiser can't reproduce without abandoning the var-ref operand
model entirely.

``semantic_isa_equal(legacy, new)`` returns True iff:
  * Both decode to the same sequence of ``(mnemonic, [operand,...])``
    tuples (comments stripped, blank lines stripped).
  * Each operand token is either:
      - identical literal (e.g. ``"0"``, ``"f0"``, ``"f1"``, ``"a3"``)
      - or a ``gp<N>`` reference, consistently renamed:
        the FIRST time ``gpN`` appears on the legacy side at position
        (instr_i, operand_j), it gets bijected to whatever ``gpM``
        the new side has at that same position; subsequent appearances
        of legacy ``gpN`` MUST map to the same ``gpM``, and the
        bijection must be one-to-one (no two legacy GPs map to the
        same new GP).

This catches:
  * any opcode mismatch
  * any literal-immediate mismatch (e.g. wrong stride / offset)
  * any GP-aliasing bug (e.g. using gp_dst where gp_src expected)

while tolerating:
  * GP renumbering (gp2 ↔ gp1 etc.) from different allocation orders
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


_GP_RE = re.compile(r"^gp(\d+)$")


def _instr_decode(text: str) -> List[Tuple[str, List[str]]]:
    """``[(mnemonic, [tok, tok, ...]), ...]`` — comments/blanks dropped."""
    out: List[Tuple[str, List[str]]] = []
    for raw in text.split("\n"):
        ln = raw.strip()
        if not ln or ln.startswith(";"):
            continue
        head, _, tail = ln.partition(" ")
        mnem = head.strip().rstrip(",")
        operands = [t.strip() for t in tail.split(",") if t.strip()]
        out.append((mnem, operands))
    return out


def _is_gp(tok: str) -> bool:
    return _GP_RE.match(tok) is not None


def semantic_isa_equal(
    legacy_text: str, new_text: str,
) -> Tuple[bool, Optional[str]]:
    """Return (equal, error_message).

    ``equal`` is True iff the two ISA streams differ only by GP
    renumbering. The bijection is reset on each ``M_*_WO`` (matmul
    write-back) — those are natural iteration boundaries in unrolled
    matmul / mv code where legacy keeps the same GPs across iters but
    the PreIsaIR per-iter scope allocates fresh ones each time. We
    require GP consistency WITHIN a "block" (between WO boundaries)
    but allow re-mapping ACROSS blocks.

    For DMA-style streams with no WO marker, the bijection is whole-
    stream (no boundary triggers a reset).

    ``error_message`` is None on success or a human-readable
    explanation of the first mismatch.
    """
    legacy = _instr_decode(legacy_text)
    new = _instr_decode(new_text)
    if len(legacy) != len(new):
        return False, (
            f"instruction count differs: legacy={len(legacy)} new={len(new)}\n"
            f"legacy:\n  " + "\n  ".join(
                f"{m} {', '.join(ops)}" for m, ops in legacy
            ) + "\nnew:\n  " + "\n  ".join(
                f"{m} {', '.join(ops)}" for m, ops in new
            )
        )
    gp_map_l2n: Dict[str, str] = {}
    gp_seen_new: Dict[str, str] = {}
    for i, ((lm, lops), (nm, nops)) in enumerate(zip(legacy, new)):
        if lm != nm:
            return False, (
                f"instr [{i}] mnemonic mismatch: legacy={lm!r} new={nm!r}\n"
                f"legacy line: {lm} {', '.join(lops)}\n"
                f"new line:    {nm} {', '.join(nops)}"
            )
        if len(lops) != len(nops):
            return False, (
                f"instr [{i}] operand-count mismatch: "
                f"legacy={len(lops)} new={len(nops)}\n"
                f"legacy: {lm} {', '.join(lops)}\n"
                f"new:    {nm} {', '.join(nops)}"
            )
        for j, (lt, nt) in enumerate(zip(lops, nops)):
            lt_is_gp = _is_gp(lt)
            nt_is_gp = _is_gp(nt)
            if lt_is_gp != nt_is_gp:
                return False, (
                    f"instr [{i}] operand[{j}]: one is GP, other isn't\n"
                    f"legacy={lt!r} new={nt!r}"
                )
            if not lt_is_gp:
                if lt != nt:
                    return False, (
                        f"instr [{i}] operand[{j}] literal mismatch: "
                        f"legacy={lt!r} new={nt!r}\n"
                        f"legacy: {lm} {', '.join(lops)}\n"
                        f"new:    {nm} {', '.join(nops)}"
                    )
                continue
            prev_mapped = gp_map_l2n.get(lt)
            if prev_mapped is None:
                if nt in gp_seen_new and gp_seen_new[nt] != lt:
                    return False, (
                        f"instr [{i}] operand[{j}]: GP bijection "
                        f"broken — new {nt!r} already mapped to "
                        f"legacy {gp_seen_new[nt]!r}, now needs to "
                        f"also represent legacy {lt!r}"
                    )
                gp_map_l2n[lt] = nt
                gp_seen_new[nt] = lt
            else:
                if prev_mapped != nt:
                    return False, (
                        f"instr [{i}] operand[{j}]: legacy {lt!r} "
                        f"previously mapped to new {prev_mapped!r}, "
                        f"now appears as {nt!r}\n"
                        f"legacy: {lm} {', '.join(lops)}\n"
                        f"new:    {nm} {', '.join(nops)}"
                    )
        # WO instructions are natural iteration boundaries — legacy
        # reuses GPs across iters while PreIsaIR per_iter scope
        # allocates fresh ones. Reset the bijection here.
        if lm in ("M_MM_WO", "M_MV_WO", "M_BMV_WO", "M_BMM_WO"):
            gp_map_l2n = {}
            gp_seen_new = {}
    return True, None


def assert_semantic_isa_equal(legacy_text: str, new_text: str) -> None:
    """Convenience assertion for pytest tests."""
    ok, err = semantic_isa_equal(legacy_text, new_text)
    if not ok:
        raise AssertionError(
            f"semantic ISA equality failed:\n{err}\n\n"
            f"=== legacy ===\n{legacy_text}\n\n"
            f"=== new ===\n{new_text}"
        )
