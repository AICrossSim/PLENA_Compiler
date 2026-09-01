# Hybrid L-Compute Compiler Contract

> **Historical Vector-SRAM experiment.** This document describes the earlier
> `L_CFG` stream-addressing path. Its issue reductions and speedups are not
> evidence for the Matrix-SRAM affine co-layout contribution. The current
> Matrix ISA decision, fixed-map control and state-precision boundary are in
> [`matrix_lcompute_isa_review.md`](matrix_lcompute_isa_review.md).

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

`L_CFG value, target, slot, field` configures one of four affine operand
views. It contains no model name, head count, recurrence equation, or fused
Mamba/KDA operation. Existing vector instructions execute the arithmetic and
existing hardware-loop instructions execute repetition.

A configured stream is inert until a consuming Vector instruction explicitly
selects it with the three-bit consumer mask in `funct1[2:0]`. Selected slots
0..2 can supply GP or FP-memory operands and advance exactly
once after the operation completes; unselected slots and architectural GP
registers do not change. Slot 3 is reserved for Matrix projection writeback.
A zero mask is byte-for-byte the legacy encoding.

```text
L_CFG      value, target, slot, field
V_MUL_VF   dst, src, scalar, rmask, lmask
V_FMA_VF   dst, src, scalar, rmask, lmask
```

`V_FMA_VF` is not a physical opcode. The assembler emits `V_MUL_VF=0x12`
with `funct1[3]=1`; plain multiply uses `funct1[3]=0`. This keeps the ISA
delta to one model-independent layout opcode while preserving readable code.

In packet mode, one existing Vector operation consumes several logical rows
whose atoms are restored into one VLEN-wide operand. `V_SUB_VF` retains its
pre-existing operand-order meaning in `funct1`, and unsupported, short,
reverse, or unprofitable walks keep the original static lowering. Matrix final
writeback has no spare mode bits in its current encoding, so the prototype
uses a WRITE-qualified slot 3 as its compiler-owned producer convention; this
is documented separately from the explicit Vector-consumer contract.

This is the review justification for one layout opcode: the static baseline can
already compute the formulas, but regular loops spend issue slots on pointer
updates and scalar loads that are fully known at compile time. The same stream
semantics is tested on Nemotron Mamba, Kimi KDA, and a model-independent SAXPY
sweep.

Configuration is fail-closed. Reserved fields, unsupported flags, out-of-range
slots/registers, zero extents, aliased bank rows, duplicate live targets,
address overflow, an unconfigured selected slot, use of producer slot 3 as a
Vector operand, a selected slot targeting a
non-operand, and a packet outside its declared extent are rejected. Contract-v2
`AUTO_ADVANCE` is reserved and rejected. A failed update is atomic: it cannot
leave a partially changed live slot. The Compiler emits `ENABLE` last and emits
`RESET` before a slot is reused.

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
- `L_CFG` has a canonical 32-bit encoding, golden-word test, bounds
  checks, and assembler support.
- Existing Vector machine words remain byte-stable at `lmask=0`; report
  artifacts record the exact nonzero masks consumed by each packet operation.
- Baseline output is byte-stable when stream addressing is disabled.
- Stream lowering removes address/scalar issue from regular FMA, unary, binary,
  reduction, and map loops; short and reverse walks fall back.
- Mamba state decay/rank-one update and KDA state decay/rank-one update emit
  executable multi-row packets. Cross-row prediction/readout reductions retain
  the ordinary-row fallback.
- Affine physical mappings round-trip without aliasing, and deliberately bad
  mappings fail.
- The Compiler report separately identifies ordinary stream operations and the
  exact `V_MUL_VF`/`V_FMA_VF` operations issued by recurrent packets.
- A Compiler-to-Rust Matrix projection test executes affine final writeback and
  explicit Vector lane restoration with zero BF16 error against PyTorch.
- A published 24-layer Mamba-2 checkpoint carries PLENA-interpreted recurrence
  state through four decode tokens with top-1/top-5 agreement on every token;
  the surrounding Matrix and PyTorch stages are explicitly outside that gate.

At the official decode shapes and the PLENA paper's 2048-element packet point,
dynamic issue counts are:

| Workload | Static baseline | Arlo post-increment | Ordinary stream | Affine packet | Baseline / ordinary stream |
|---|---:|---:|---:|---:|---:|
| Nemotron Mamba recurrence | 92,399 | 51,311 | 33,257 | 19,049 | 2.778x |
| Kimi K3 KDA mixer | 215,387 | 116,219 | 81,659 | 61,115 | 2.638x |
| Model-independent SAXPY | 1,284 | 516 | 301 | 301 | 4.266x |

Nemotron uses 64-element semantic state rows. Kimi uses its natural
128-element state rows; using the old 64-wide KDA split at a 2048-wide system
point would unfairly weaken the ordinary baseline. Packet arithmetic still
uses 64-element bank words and preserves exact element work.

The affine packet base is aligned and expressed in packet rows. One 2048-value
packet occupies one 32-bank physical row; the row-major comparison retains 32
padded short-row locations. This capacity check is part of the executable
round-trip test, not an assumed zero-cost layout.

These are issued-instruction reductions, not hardware-cycle or full-model
speedups. The Simulator shared-resource campaign provides those separately.

The packet counts above include setup and arithmetic issue, but not physical
bank stalls. The Simulator executes the same packets against row-major and
affine physical placement to price those stalls on the full model schedule.

## Freeze result

The shared-resource Simulator campaign now includes Compiler issue counts,
bank/FIFO service, Matrix, Vector, HBM, MoE routing pressure, and the full
52/93-layer schedules. It supports two different decisions:

- stream addressing earns the one general opcode: it improves decode on
  Mamba, KDA, and generic affine loops without changing their arithmetic;
- affine co-layout removes every measured conflict from the executable
  multi-row Mamba/KDA packet path;
- the 64-lane negative control remains slower than the best ordinary-row
  stream, so conflict removal alone is not enough to justify the mechanism;
- at the paper's 2048-wide point, affine packet beats ordinary stream by
  1.13473x for Nemotron and 1.01497x for Kimi; packet plus writeback overlap
  beats the Arlo post-increment full-model baseline by 1.30910x and 1.04025x.

The exact Compiler/Simulator lane sweep places the crossover near 128 lanes
for Mamba and 256 lanes for KDA. This is the ISA freeze condition: the same
general stream semantics is retained, but the affine packet performance claim
is limited to widths above the measured crossover.

Real dimensions with symbolic weights and real-checkpoint numerical execution
remain separate completion levels. Full Nemotron/Kimi checkpoint execution is
not claimed.
