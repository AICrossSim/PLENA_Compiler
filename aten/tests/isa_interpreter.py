"""A tiny interpreter for the ISA subset the KDA decode lowering emits.

Why this exists
---------------
The transactional emulator is the authority on ISA semantics, but running it
needs ``libramulator`` built, which is a C++ dependency the compiler repo's own
test suite does not have. Without something in between, a lowering change can
only be checked for *shape* -- "an S_SQRT_FP appears somewhere" -- and the review
of Task 2 showed exactly how far that gets you: thirteen mutations that change
the arithmetic left every such assertion green.

So this executes the emitted assembly against the CPU reference. It is a test
oracle, deliberately narrow:

* Seven opcodes, listed in ``_OPS``. Anything else raises rather than being
  skipped, so a lowering that starts emitting something new fails loudly here
  instead of being silently half-checked.
* No timing, no precision model, no memory hierarchy. Values are float64,
  while the emulator's FP registers are bf16 and every VRAM write is quantised.
  "Validated numerically" here means the *arithmetic and addressing* are right,
  not that the result holds at hardware precision.
* Batch is not modelled, because the lowering has no batch concept.

It is a second implementation of ISA semantics and could therefore drift from
the emulator. That is a real risk and the reason it is scoped to seven opcodes
whose meaning is one line each in ``doc/plena_isa_spec.md``. The whole-model
emulator run is what finally cross-checks it; until then this catches the class
of bug that actually occurs in lowering work -- wrong row, wrong FPRAM slot,
wrong order.

Semantics, from ``doc/plena_isa_spec.md`` and
``transactional_emulator/src/accelerator/loop_state.rs``:

===================================  ======================================
``S_ADDI_INT rd, rs1, imm``          ``gp[rd] = gp[rs1] + imm``
``S_LD_FP fd, rs1, imm``             ``f[fd] = FPRAM[gp[rs1] + imm]``
``V_MUL_VF rd, rs1, f2, rmask``      ``V[rd] = V[rs1] * f[f2]``
``V_ADD_VF rd, rs1, f2, rmask``      ``V[rd] = V[rs1] + f[f2]``
``V_SUB_VF rd, rs1, f2, rmask, ord`` ``ord=0: V[rs1] - f[f2]``; ``ord=1: f[f2] - V[rs1]``
``V_ADD_VV rd, rs1, rs2, rmask``     ``V[rd] = V[rs1] + V[rs2]``
``V_SUB_VV rd, rs1, rs2, rmask``     ``V[rd] = V[rs1] - V[rs2]``
``V_MUL_VV rd, rs1, rs2, rmask``     ``V[rd] = V[rs1] * V[rs2]``
``V_EXP_V rd, rs1, rmask``           ``V[rd] = exp(clamp(V[rs1], -88, 88))``
``V_RECI_V rd, rs1, rmask``          ``V[rd] = 1 / V[rs1]``; ``1/0 -> inf``
``V_RED_SUM fd, rs1, ...``           ``f[fd] += sum(V[rs1])`` -- **accumulates**
``S_ST_FP fd, rs1, imm``             ``FPRAM[gp[rs1] + imm] = f[fd]`` (fd is the source)
``S_MAP_FP_V rd, rs1, imm``          ``FPRAM[gp[rd]+imm ..+vlen] = V[gp[rs1]]``
``S_ADD_FP fd, fs1, fs2``            ``f[fd] = f[fs1] + f[fs2]``
``S_SUB_FP fd, fs1, fs2``            ``f[fd] = f[fs1] - f[fs2]``
``S_MUL_FP fd, fs1, fs2``            ``f[fd] = f[fs1] * f[fs2]``
``S_RECI_FP fd, fs1``                ``f[fd] = 1 / f[fs1]``
``S_SQRT_FP fd, fs1``                ``f[fd] = sqrt(f[fs1])``

A write to ``f0`` by any of those is **discarded**, matching
``dispatch.rs:415-424``.
``C_LOOP_START rd, imm``             ``gp[rd] = imm``; remember this pc
``C_LOOP_END rd``                    if ``gp[rd] > 1``: ``gp[rd] -= 1``, jump
                                     to ``start_pc + 1``; else fall through
===================================  ======================================

``V_SUB_VV`` is ``rs1 - rs2``. Take that from the emulator
(``vector_machine.rs::sub``), not from the operation line in
``doc/plena_isa_spec.md``, which read ``rs2 - rs1`` until this was found --
"correcting" this interpreter to match that wording inverts the sign of KDA's
error term.

``gp0`` and ``f0`` are **ordinary writable registers that initialise to zero**,
exactly as in the emulator (``accelerator/registers.rs`` reads and writes them
through the same plain arrays as any other). They are not hardwired. What keeps
them zero is ``aten/plena/registers.py``'s allocator, which hands out gp1+/f1+
and never gives either away -- so the zeroing idiom ``V_MUL_VF gpX, gpX, f0``
rests on a compiler convention, not on hardware.

Modelling that faithfully rather than hardwiring zero here is deliberate: a
hardwired oracle would keep passing an emitter that started writing ``gp0``,
which on real hardware would corrupt every address computed from it.
``test_isa_interpreter.py`` and ``test_kda_decode_step.py`` guard the
convention directly instead.

Vector operands are element addresses; each op touches ``vlen`` consecutive
elements. ``rmask`` is ignored: every KDA emission passes 0 (all lanes), and
asserting that is cheaper than modelling the mask register.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = ["Machine", "UnsupportedInstruction"]

_OPS = frozenset(
    {
        "S_ADDI_INT",
        "S_LD_FP",
        "V_MUL_VF",
        "V_FMA_VF",
        "V_ADD_VF",
        "V_SUB_VF",
        "V_ADD_VV",
        "V_SUB_VV",
        "V_MUL_VV",
        "V_EXP_V",
        "V_RECI_V",
        "V_RED_SUM",
        "S_ST_FP",
        "S_MAP_FP_V",
        "S_ADD_FP",
        "S_SUB_FP",
        "S_MUL_FP",
        "S_RECI_FP",
        "S_SQRT_FP",
        "C_LOOP_START",
        "C_LOOP_END",
    }
)

#: Scalar FP ops whose destination is an FP register. ``dispatch.rs:415-424``
#: makes every one of these a no-op when ``rd == 0``, so ``f0`` cannot be
#: clobbered by them -- which is what the ``V_MUL_VF gpX, gpX, f0`` zeroing
#: idiom rests on. ``S_LD_FP`` is deliberately absent from that guard list, and
#: therefore from this one.
_FP_DEST_OPS = frozenset(
    {"S_ADD_FP", "S_SUB_FP", "S_MUL_FP", "S_RECI_FP", "S_SQRT_FP"}
)

#: ``exp`` saturates rather than overflowing, matching
#: ``vector_machine.rs::exp``, which clamps to keep bf16 from going infinite.
_EXP_CLAMP = 88.0

#: Guards against a runaway loop turning a failing test into a hang.
_MAX_STEPS = 5_000_000


class UnsupportedInstruction(RuntimeError):
    """Raised for any opcode outside `_OPS`, rather than skipping it."""


def _reg(token: str, prefix: str) -> int:
    if not token.startswith(prefix):
        raise UnsupportedInstruction(f"expected {prefix} register, got {token!r}")
    return int(token[len(prefix) :])


@dataclass
class Machine:
    vlen: int
    vram_words: int = 1 << 16
    fpram_words: int = 1 << 12
    vram: list[float] = field(init=False)
    fpram: list[float] = field(init=False)
    gp: list[int] = field(init=False)
    fp: list[float] = field(init=False)
    executed: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.vram = [0.0] * self.vram_words
        self.fpram = [0.0] * self.fpram_words
        self.gp = [0] * 16
        self.fp = [0.0] * 8

    # -- memory helpers ---------------------------------------------------

    def write_vram_row(self, addr: int, values) -> None:
        vals = list(values)
        if len(vals) > self.vlen:
            raise ValueError(f"row is {len(vals)} wide, vlen is {self.vlen}")
        self.vram[addr : addr + len(vals)] = [float(x) for x in vals]

    def read_vram_row(self, addr: int, count: int | None = None) -> list[float]:
        n = self.vlen if count is None else count
        return list(self.vram[addr : addr + n])

    def write_fpram(self, addr: int, values) -> None:
        vals = [float(x) for x in values]
        self.fpram[addr : addr + len(vals)] = vals

    # -- execution --------------------------------------------------------

    def run(self, code: str) -> "Machine":
        program = _parse(code)
        loop_starts: dict[int, int] = {}
        pc = 0
        steps = 0
        while pc < len(program):
            steps += 1
            if steps > _MAX_STEPS:
                raise RuntimeError(f"exceeded {_MAX_STEPS} steps; runaway loop at pc {pc}")
            op, args = program[pc]
            if op == "S_ADDI_INT":
                rd, rs1, imm = _reg(args[0], "gp"), _reg(args[1], "gp"), int(args[2])
                self._set_gp(rd, self._gp(rs1) + imm)
            elif op == "S_LD_FP":
                fd, rs1, imm = _reg(args[0], "f"), _reg(args[1], "gp"), int(args[2])
                self.fp[fd] = self.fpram[self._gp(rs1) + imm]
            elif op == "V_RED_SUM":
                # The masked reduce_sum in vector_machine.rs:404-418 is a
                # materially different computation -- it broadcasts each head's
                # sum back over its slice before summing -- so refuse rather
                # than model it wrongly.
                # Every emitter writes zeros here; rather than pin which slot
                # carries rmask, refuse any non-zero trailing operand.
                if any(a != "0" for a in args[2:]):
                    raise UnsupportedInstruction(
                        f"V_RED_SUM with non-zero mask operands {args[2:]}; "
                        f"only the unmasked form is modelled"
                    )
                fd, rs1 = _reg(args[0], "f"), _reg(args[1], "gp")
                src = self._gp(rs1)
                if src % self.vlen:
                    raise UnsupportedInstruction(
                        f"V_RED_SUM address {src} is not a multiple of vlen {self.vlen}"
                    )
                # Accumulates: dispatch.rs seeds reduce_sum with the current f[rd].
                self._set_fp(fd, self.fp[fd] + sum(self.vram[src : src + self.vlen]))
            elif op.startswith("V_"):
                self._vector(op, args)
            elif op == "S_MAP_FP_V":
                # Mirror of S_MAP_V_FP, and the operand roles mirror too: rs1 is
                # the VRAM source row, rd the FP_MEM base, so that in both
                # instructions "rd names the destination memory".
                rd, rs1, imm = _reg(args[0], "gp"), _reg(args[1], "gp"), int(args[2])
                src = self._gp(rs1)
                if src % self.vlen:
                    raise UnsupportedInstruction(
                        f"S_MAP_FP_V source {src} is not a multiple of vlen {self.vlen}"
                    )
                start = self._gp(rd) + imm
                # Python slice assignment past the end *extends* the list rather
                # than failing, so an out-of-bounds write would silently land at
                # the wrong address and stay green. The emulator asserts on the
                # FP SRAM bound (scalar_sram.rs), so refuse here too.
                if start < 0 or start + self.vlen > len(self.fpram):
                    raise UnsupportedInstruction(
                        f"S_MAP_FP_V writes FPRAM[{start}:{start + self.vlen}] past "
                        f"the modelled {len(self.fpram)} slots"
                    )
                self.fpram[start : start + self.vlen] = self.vram[src : src + self.vlen]
            elif op == "S_ST_FP":
                fd, rs1, imm = _reg(args[0], "f"), _reg(args[1], "gp"), int(args[2])
                self.fpram[self._gp(rs1) + imm] = self.fp[fd]
            elif op in _FP_DEST_OPS:
                self._scalar_fp(op, args)
            elif op == "C_LOOP_START":
                rd, imm = _reg(args[0], "gp"), int(args[1])
                if imm <= 0:
                    raise UnsupportedInstruction(f"C_LOOP_START with count {imm}")
                self._set_gp(rd, imm)
                loop_starts[rd] = pc
            elif op == "C_LOOP_END":
                rd = _reg(args[0], "gp")
                if rd not in loop_starts:
                    raise UnsupportedInstruction(f"C_LOOP_END gp{rd} with no matching start")
                if self._gp(rd) > 1:
                    self._set_gp(rd, self._gp(rd) - 1)
                    pc = loop_starts[rd] + 1
                    self.executed += 1
                    continue
            else:  # pragma: no cover - _parse already rejects these
                raise UnsupportedInstruction(op)
            self.executed += 1
            pc += 1
        return self

    def _scalar_fp(self, op: str, args: list[str]) -> None:
        fd = _reg(args[0], "f")
        a = self.fp[_reg(args[1], "f")]
        if op == "S_RECI_FP":
            # bf16::ONE / a. Python's `a == 0.0` is true for -0.0 too, so the
            # sign has to come from copysign or -0.0 would yield +inf.
            out = math.copysign(math.inf, a) if a == 0.0 else 1.0 / a
        elif op == "S_SQRT_FP":
            out = math.sqrt(a) if a >= 0.0 else math.nan
        else:
            b = self.fp[_reg(args[2], "f")]
            out = {"S_ADD_FP": a + b, "S_SUB_FP": a - b, "S_MUL_FP": a * b}[op]
        self._set_fp(fd, out)

    def _set_fp(self, index: int, value: float) -> None:
        # dispatch.rs discards these writes when rd == 0.
        if index != 0:
            self.fp[index] = value

    def _vector(self, op: str, args: list[str]) -> None:
        # V_SUB_VF carries rorder after rmask; every other V-type ends on rmask.
        rmask = args[3] if op == "V_SUB_VF" else args[-1]
        if rmask != "0":
            raise UnsupportedInstruction(f"{op} with rmask={rmask}; only 0 is modelled")
        dst = self._gp(_reg(args[0], "gp"))
        src = self._gp(_reg(args[1], "gp"))
        # The emulator's vector SRAM asserts VLEN alignment (lib/sram addr_to_cell);
        # an unaligned vector address panics there rather than reading across a
        # row boundary. Model that, so a misaligned emission fails here too.
        for name, addr in (("dst", dst), ("src", src)):
            if addr % self.vlen:
                raise UnsupportedInstruction(
                    f"{op} {name} address {addr} is not a multiple of vlen {self.vlen}"
                )
        n = self.vlen
        if op in ("V_EXP_V", "V_RECI_V"):
            if op == "V_EXP_V":
                out = [
                    math.exp(min(max(self.vram[src + i], -_EXP_CLAMP), _EXP_CLAMP))
                    for i in range(n)
                ]
            else:
                # vector_machine.rs uses tensor.reciprocal(), which yields
                # +/-inf on zero rather than trapping. Model that: a lowering
                # that reciprocates an unused lane must behave the same here.
                out = [
                    (
                        math.copysign(math.inf, self.vram[src + i])
                        if self.vram[src + i] == 0.0
                        else 1.0 / self.vram[src + i]
                    )
                    for i in range(n)
                ]
        elif op == "V_FMA_VF":
            # The only V-type op that reads its destination. Modelled as one
            # expression rather than a multiply followed by an add, matching
            # vector_machine.rs::fma_scalar, which quantises once on the sum.
            f = self.fp[_reg(args[2], "f")]
            out = [self.vram[dst + i] + self.vram[src + i] * f for i in range(n)]
        elif op in ("V_MUL_VF", "V_ADD_VF", "V_SUB_VF"):
            f = self.fp[_reg(args[2], "f")]
            if op == "V_MUL_VF":
                out = [self.vram[src + i] * f for i in range(n)]
            elif op == "V_ADD_VF":
                out = [self.vram[src + i] + f for i in range(n)]
            else:
                reverse = len(args) > 4 and args[4] == "1"
                out = [
                    (f - self.vram[src + i]) if reverse else (self.vram[src + i] - f)
                    for i in range(n)
                ]
        else:
            src2 = self._gp(_reg(args[2], "gp"))
            if src2 % self.vlen:
                raise UnsupportedInstruction(
                    f"{op} src2 address {src2} is not a multiple of vlen {self.vlen}"
                )
            if op == "V_ADD_VV":
                out = [self.vram[src + i] + self.vram[src2 + i] for i in range(n)]
            elif op == "V_SUB_VV":
                out = [self.vram[src + i] - self.vram[src2 + i] for i in range(n)]
            elif op == "V_MUL_VV":
                out = [self.vram[src + i] * self.vram[src2 + i] for i in range(n)]
            else:  # pragma: no cover - _parse rejects anything else
                raise UnsupportedInstruction(op)
        self.vram[dst : dst + n] = out

    def _gp(self, index: int) -> int:
        return self.gp[index]

    def _set_gp(self, index: int, value: int) -> None:
        self.gp[index] = value


def _parse(code: str) -> list[tuple[str, list[str]]]:
    program: list[tuple[str, list[str]]] = []
    for raw in code.splitlines():
        line = raw.split(";", 1)[0].split("//", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split()
        op, args = parts[0], parts[1:]
        if op not in _OPS:
            raise UnsupportedInstruction(
                f"{op!r} is outside the modelled subset {sorted(_OPS)}. "
                f"The lowering emitted something this oracle does not understand; "
                f"extend it deliberately rather than letting the check pass."
            )
        program.append((op, args))
    return program
