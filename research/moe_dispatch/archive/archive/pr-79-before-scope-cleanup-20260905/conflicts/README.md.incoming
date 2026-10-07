# PLENA Compiler

> **Draft review only — not for merge.** This branch collects the Mamba/KDA
> extensions and their historical experiments for review. Start with
> [review scope and precision](doc/REVIEW_SCOPE_20260905.md). Neither L_TILE nor
> either optional dot experiment is a selected replacement architecture.

## Matrix SRAM L-Compute

Original PLENA already has Matrix accumulators. This work adds recurrent-layer
lowering, Matrix-view/L_TILE forms and supporting operations; it does not
establish that original PLENA lacked accumulation. Reusing the original Matrix
accumulators for the new recurrence mapping remains a separate review question.

This branch evaluates linear recurrences on PLENA's existing Matrix SRAM. The
Compiler explicitly loads prepared fields and recurrent state, configures a
logical Matrix view, executes bank-parallel packets, and stores state and output
back to HBM. State, prepared fields and SRAM writeback use BF16. L_TILE's Rust
model retains FP32 partial sums across rows and rounds update primitives on
writeback; its physical arithmetic/storage reuse is not yet proven. The official
GPU implementations' FP32 state is a separate accuracy reference. NVFP4/MX8
weight sensitivity tables count traffic; these recurrence kernels do not execute
checkpoint-weight decoding or a complete model.

```text
prepared fields + BF16 HBM state
                 |
                 v
       existing Matrix SRAM
 fixed diagonal + compiler tile phase
                 |
      bank packet + lane restore
                 |
                 v
   modeled recurrence arithmetic
     (RTL reuse to be reviewed)
                 |
                 v
        BF16 state/output in HBM
```

There is no state cache, private state SRAM, `X_STATE`, command queue, runtime
scheduler or new MAC array. Matrix SRAM is an explicitly addressed scratchpad:
the Compiler owns every address, transfer and lifetime.

### ISA

One physical opcode (`0x3f`, named `L_TILE`) carries two model-independent
Matrix-SRAM forms. The older `L_CFG` encoding at `funct1=0` remains only so the
historical software baseline can be reproduced; it is not part of the frozen
Matrix-SRAM RTL candidate:

```text
L_TILE_CFG     slot, shape_reg, map_reg
L_TILE_EXEC    dst, src, scale, primitive[, axis_mask]
```

The view descriptor contains tensor shape, row pitch, a compiler-selected
inter-tile bank phase, and one broadcast flag; bounds checking is always
strict. The row term is fixed to PLENA's published diagonal mapping; arbitrary
programmable row skew is rejected.
`L_TILE_EXEC` accepts only three algebraic
primitives: scale-accumulate, dot-reduce and outer-update. The decoder walks a
statically known view; reuse of PLENA arithmetic is the design goal, with
precision, feedback and routing still requiring a physical mapping. Neither
the encoding nor decoder contains `Mamba`, `KDA`, a model-specific head count,
or a cache policy. Configuration is atomic; there is no partial-update form. A
loop-aware dominance check rejects use of an unconfigured
view.

Matrix-view DMA reuses `H_PREFETCH_V` and `H_STORE_V`: bit 31 marks the viewed
form and bits 30:29 select the view slot. Ordinary and viewed transfers both
decode Activation=0, KV=1 and State=2. Old binaries intentionally using selector
2 as a KV alias must use selector 1; selector 2 now selects independent BF16 state.

Mamba-2 composes scale-accumulate and dot-reduce. KDA composes all three
primitives. Attention/MLA/MoE keep their existing PLENA instructions. Arlo's
row-by-row static lowering remains the software fallback and historical
analytic `B` ablation. The separately executable prepared A/B controls are new
programs, not instruction-for-instruction replays of that census or the paper.
It is separate from L-Compute: Arlo reduces address/loop issue; `L_TILE` walks a
configured multi-row view. Coefficient generation (`softplus`/`exp`) remains an
upstream Vector stage and is not silently credited to `L_TILE`.

### What is wired into the Compiler

- Nemotron 3: all 23 Mamba layers in the official 52-layer order emit the
  prepared-coefficient recurrence core through `L_TILE`.
- Kimi K3: all 69 KDA layers in the official 93-layer order emit the
  prepared-coefficient recurrence core through `L_TILE`.
- Recurrent state is BF16: 1 MiB per Nemotron Mamba layer and 3 MiB per Kimi
  KDA layer at batch 1.
- Each layer owns disjoint, 64-byte-aligned HBM ranges. The 1 MiB Matrix SRAM is
  reused sequentially and never treated as a cache.
- Every head group's output has a distinct HBM destination and is checked after
  Rust execution; a later group cannot overwrite an earlier result.

The matching storage study reports BF16 output relative-L2 error of 0.000312
for Nemotron at 32K tokens and 0.017061 for Kimi at 2K tokens versus FP32 state.
These are synthetic recurrence errors, not checkpoint-level quality results.

The compact phased schedules assemble to legal 32-bit words. Their ordinary
Attention/MLA/MoE entries are schedule markers linked to the existing analytic
paths, not a claim that checkpoint weights have run numerically from the first
to final layer in Rust.

### Historical pre-RTL results

The table below preserves the earlier formula campaign without changing its
numbers. It predates subsequent accounting fixes and is not the current review
branch's fresh result. See the paired Simulator PR's versioned artifacts and
fresh validation; do not combine historical and executable A/B labels.

At `MLEN=2048`, `BLEN=32`, 64 banks, a 1 MiB BF16 Matrix SRAM and 1560 HBM
bytes/cycle, the historical formula-based B1 decode timeline is:

| Model | Unoptimized extension A | Arlo static B | Fixed single-base C | Compact phased D | D/A | D/B |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron 3 | 4,055,091 | 3,110,067 | 2,192,850 | 2,014,094 | 2.0134x | 1.5442x |
| Kimi K3 | 103,816,704 | 97,013,856 | 93,124,740 | 91,173,903 | 1.1387x | 1.0641x |

`A` and `B` are one-cycle-per-issued-instruction proxies for this work's
unoptimized and Arlo static recurrence streams; original PLENA did not publish
these Mamba/KDA workloads. They are not Rust cycle measurements. `C` and `D` add
explicit service, arithmetic and HBM terms, so `D/A` and `D/B` primarily
measure multi-row utilization plus issue compression, not programmable-skew
speedup.

KDA decay/beta preparation remains visible upstream: the B1 timeline charges
the same 5,107,104 ordinary elementwise operations and 1,702,368 exponentials,
or 4,485 Vector cycles, to every variant. The preparation follows the official
decay and beta formulas before `L_TILE` consumes those coefficients.

`C` is an executable descriptor constrained to one base phase; it is not the
fair bank-only control. The fair `D'` control gives the original fixed diagonal
wiring an ordinary per-tile base phase. It maps the same physical cells as `D`,
has zero bank stalls, and gives `D/D' = 1.00x` on both official BF16 state
packets. Therefore this branch does **not** claim a programmable-skew bank
speedup. `C -> D` includes descriptor compactness, fewer chunks/instructions,
lower ideal service, and KDA spill removal.

The connected test is stronger than the analytic replay: Compiler assembly is
assembled into canonical 32-bit words and executed by Rust for four consecutive
tokens at official recurrence geometry. It compares 524,288 Nemotron and
1,572,864 Kimi state values plus every head-group output. All four fixed/phased
cases pass; the largest relative-L2 error is 0.0071 with BF16 stored operands
and the variant-specific intermediate rounding described above.

Whole-model cycles use official dimensions, pinned GPU calibration, measured
Nemotron routing where available, and symbolic PLENA weights. Only the 23/69
recurrent layers are executable in this schedule; ordinary layers are analytic
markers. This is not silicon measurement or first-to-last real-checkpoint Rust
execution. Prefill receives no L-Compute speedup; previous prefill speedup
claims remain withdrawn. There is no overlap credit at the 1 MiB point.

See [the ISA review](doc/matrix_lcompute_isa_review.md) and the Simulator's
machine-readable `matrix_lcompute_e2e_v5` campaign for the complete fairness,
capacity, bandwidth, port-width and evidence boundaries.

The historical implementation contract is recorded in
[the pre-RTL handoff](doc/matrix_lcompute_pre_rtl_freeze.md). Current acceptance
questions and optional experiments are listed in the
[review scope](doc/REVIEW_SCOPE_20260905.md).

## MoE code organization

- `aten/plena/program_routed_moe.py` contains reusable routed-MoE lowering
  helpers: router logits, V_TOPK selection, dynamic expert-weight addressing,
  routed gather/scatter, expert activation, and combine.
- `aten/models/gpt_oss/` contains GPT-OSS-specific reference semantics and
  real-checkpoint loading utilities used to validate that substrate.
- ISA, assembler, and hardware documentation remain in `assembler/` and `doc/`.
