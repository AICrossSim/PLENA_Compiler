"""Tiny free-list register allocator.

ISAEmitter calls into us mid-emit to get scratch registers and returns
them when the instruction sequence is finished:

    gp_regs = compiler.register_allocator.allocate_gp(5)   # list[int]
    ... emit ISA using gp{gp_regs[0]}, gp{gp_regs[1]}, ...
    compiler.register_allocator.free_gp(gp_regs)

The runtime version is more elaborate (lifetime tracking, conflict
detection, conservative reuse). Ours is the minimum that satisfies
the API contract: a free-list initialised from a fixed pool, allocate
pops from the front, free pushes back.

Pool sizes match the PLENA spec (16 GP, 8 addr) minus a few reserved:
    - gp0 reserved as the constant-zero register
    - addr0..7 all available; runtime convention reserves none
"""

from __future__ import annotations

from typing import Iterable, List


class RegisterExhausted(RuntimeError):
    pass


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
        self._gp_free: List[int] = [i for i in range(gp_total) if i not in gp_reserved_set]
        self._addr_free: List[int] = [i for i in range(addr_total) if i not in addr_reserved_set]

    # ------------------------------------------------------------------
    # GP register pool
    # ------------------------------------------------------------------
    def allocate_gp(self, n: int) -> List[int]:
        if n > len(self._gp_free):
            raise RegisterExhausted(
                f"requested {n} GP registers but only {len(self._gp_free)} free"
            )
        out = self._gp_free[:n]
        self._gp_free = self._gp_free[n:]
        return out

    def free_gp(self, regs: Iterable[int]) -> None:
        # Push back at the front to maximise locality (next alloc reuses
        # the same register, keeping the live range short and dump
        # human-readable).
        for r in regs:
            if r in self._gp_free:
                raise RuntimeError(f"double-free of gp{r}")
            self._gp_free.insert(0, r)

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
        return out

    def free_addr(self, regs: Iterable[int]) -> None:
        for r in regs:
            if r in self._addr_free:
                raise RuntimeError(f"double-free of a{r}")
            self._addr_free.insert(0, r)


__all__ = ["RegisterAllocator", "RegisterExhausted"]
