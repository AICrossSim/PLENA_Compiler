# Hybrid L-Compute Compiler Contract

## Scope

The implementation targets one static PLENA instruction stream for
Attention/MLA, MoE, Mamba-2, and KDA. It adds no cache and no model-specific
state instruction. Recurrent state is an ordinary tensor whose address and
transfers remain visible to the Compiler.

## Architectural boundary

Matrix operands remain in Matrix SRAM. Matrix projections normally write their
results to Vector/output SRAM, so the producer-consumer layout decision is made
on that writeback path. The current RTL is not claimed to contain the proposed
banked output SRAM; the physical bank model is an architecture candidate that
must earn its cost in Simulator DSE before any RTL work.

## ISA

`L_STREAM_CFG value, target, slot, field` configures one of four affine operand
streams. It contains no model name, head count, recurrence equation, or fused
Mamba/KDA operation. Existing vector instructions execute the arithmetic and
existing hardware-loop instructions execute repetition.

A configured stream can supply a GP address or an FP-memory address, advance it
after each consuming Matrix/Vector operation, and carry affine placement
metadata. Unsupported, short, reverse, or unprofitable walks keep the original
static lowering.

This is the review justification for one opcode: the static baseline can
already compute the formulas, but regular loops spend issue slots on pointer
updates and scalar loads that are fully known at compile time. The same stream
semantics is tested on Nemotron Mamba, Kimi KDA, and a model-independent SAXPY
sweep.

## Layout planning

The planner consumes logical producer and consumer packets, not a model name.
For every candidate it checks:

- one-to-one logical-to-physical placement;
- producer write service and consumer read service;
- bandwidth floor versus true bank-conflict stalls;
- explicit gather/reorder cost;
- cyclic lane-restore cost.

The physical map is:

```text
bank = (stripe + alpha*major + beta*field + gamma*group) mod banks
bank_row = base + outer*pitch + floor(stripe/banks)
sublane = minor mod bank_width
```

`consumer-major` is a producer schedule that can remove a gather with existing
strides. It receives no invented bank-conflict benefit. `affine-skewed` is a
separate architecture candidate and is selected only on a strict total-cost
win.

## Current verified status

- Official Nemotron and Kimi layer censuses and dimensions are checked.
- `L_STREAM_CFG` has a canonical 32-bit encoding, golden-word test, bounds
  checks, and assembler support.
- Baseline output is byte-stable when stream addressing is disabled.
- Stream lowering removes address/scalar issue from regular FMA, unary, binary,
  reduction, and map loops; short and reverse walks fall back.
- Affine physical mappings round-trip without aliasing, and deliberately bad
  mappings fail.
- The Compiler report separately prices projection packets and candidate
  multirow state packets.

The multirow state layout is not counted as executable speedup until a matching
packetized consumer lowering executes it. Its current result is an architecture
upper bound, not an end-to-end claim.

## Remaining evidence gates

The Simulator must place Compiler issue counts, bank/FIFO service, Matrix and
Vector work, HBM traffic, MoE routing, and producer-consumer overlap on one
shared timeline. Only that A-G comparison may be used for stage, layer, or
model speedups. Real dimensions with symbolic weights and real-checkpoint
numerical execution remain separate completion levels.
