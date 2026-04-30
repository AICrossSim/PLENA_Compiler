"""Minimal stand-in for the runtime TileTensorProgram + Compiler objects
that ISAEmitter pokes into.

ISAEmitter (the file we copied wholesale) reads:
    self.program.mlen
    self.program.blen
    self.program.tile_elems
    self.program.btmm_lane_count
    self.program.btmm_hlen
    self.program.compiler.register_allocator
    self.program.compiler.generated_code   (string, accumulated by += )

For methods we don't use yet (emit_matmul, emit_fp_kernel, ...) it also
touches `self.program._arith_progression` and various tile/_helpers/_types
symbols. Those will fail at call time if invoked. We document the
contract here and add fields lazily as we enable more emitter methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .register_alloc import RegisterAllocator


@dataclass
class CompilerShim:
    """Holds the pieces ISAEmitter expects under `program.compiler`."""

    register_allocator: RegisterAllocator = field(default_factory=RegisterAllocator)
    generated_code: str = ""


@dataclass
class ProgramShim:
    """Holds the hardware constants ISAEmitter expects under `program`."""

    mlen: int
    blen: int
    btmm_lane_count: int
    btmm_hlen: int
    compiler: CompilerShim = field(default_factory=CompilerShim)

    @property
    def tile_elems(self) -> int:
        return self.mlen * self.mlen

    @staticmethod
    def _arith_progression(values):
        """Detect an arithmetic progression in a list of ints.

        Returns (start, count, step) when the input is a non-empty AP,
        else None. Single-element inputs are treated as a degenerate AP
        with step=0 so emit_matmul can use its hardware-loop fast path
        with pair_count=1 instead of falling through to explicit
        unrolling.
        """
        if not values:
            return None
        if len(values) == 1:
            return (int(values[0]), 1, 0)
        step = int(values[1]) - int(values[0])
        for i in range(2, len(values)):
            if int(values[i]) - int(values[i - 1]) != step:
                return None
        return (int(values[0]), len(values), step)


def make_shim(
    *,
    mlen: int,
    blen: int,
    btmm_lane_count: int,
    btmm_hlen: int,
    register_allocator: Optional[RegisterAllocator] = None,
) -> ProgramShim:
    compiler = CompilerShim(register_allocator=register_allocator or RegisterAllocator())
    return ProgramShim(
        mlen=mlen,
        blen=blen,
        btmm_lane_count=btmm_lane_count,
        btmm_hlen=btmm_hlen,
        compiler=compiler,
    )


__all__ = ["ProgramShim", "CompilerShim", "make_shim"]
