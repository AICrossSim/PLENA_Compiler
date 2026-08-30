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
metadata. In packet mode, one existing Vector operation consumes several
logical rows whose atoms are restored into one VLEN-wide operand. Unsupported,
short, reverse, reduction, or unprofitable walks keep the original static
lowering.

This is the review justification for one opcode: the static baseline can
already compute the formulas, but regular loops spend issue slots on pointer
updates and scalar loads that are fully known at compile time. The same stream
semantics is tested on Nemotron Mamba, Kimi KDA, and a model-independent SAXPY
sweep.

Configuration is fail-closed. Reserved fields, unsupported flags, out-of-range
slots/registers, zero extents, aliased bank rows, duplicate live targets,
address overflow, and a packet outside its declared extent are rejected. A
failed update is atomic: it cannot leave a partially changed live slot. The
Compiler emits `ENABLE` last and emits `RESET` before a slot is reused.

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
- Mamba state decay/rank-one update and KDA state decay/rank-one update emit
  executable multi-row packets. Cross-row prediction/readout reductions retain
  the ordinary-row fallback.
- Affine physical mappings round-trip without aliasing, and deliberately bad
  mappings fail.
- The Compiler report separately identifies ordinary stream operations and the
  exact `V_MUL_VF`/`V_FMA_VF` operations issued by recurrent packets.

At the official decode shapes, dynamic issue counts are:

| Workload | Static baseline | Arlo post-increment | Ordinary stream | Affine packet | Baseline / ordinary stream |
|---|---:|---:|---:|---:|---:|
| Nemotron Mamba recurrence | 92,399 | 51,311 | 33,257 | 35,177 | 2.778x |
| Kimi K3 KDA mixer | 428,238 | 226,242 | 160,782 | 165,774 | 2.663x |
| Model-independent SAXPY | 1,284 | 516 | 301 | 301 | 4.266x |

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
- on the current 64-lane datapath, affine packet execution is still 0.46%
  slower than the best ordinary-row stream for Nemotron and 0.056% slower for
  Kimi. Conflict removal is therefore validated, but superiority over the
  ordinary-row path is not claimed.

Real dimensions with symbolic weights and real-checkpoint numerical execution
remain separate completion levels. Full Nemotron/Kimi checkpoint execution is
not claimed.
