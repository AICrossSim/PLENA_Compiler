# Matrix SRAM L-Compute ISA Review

## Decision

Keep one atomic Matrix-SRAM configuration form and one deterministic execution
form:

```text
L_TILE_CFG     slot, shape_reg, map_reg
L_TILE_EXEC    dst, src, scale, primitive[, axis_mask]
```

Both forms share the physical `L_TILE` opcode `0x3f`. `funct1` selects atomic
configuration or execution. The historical `L_CFG` form at `funct1=0` is kept
only to reproduce the software experiment and is excluded from the frozen RTL
candidate. The ISA has no partial-update form, `MAMBA_STEP`, `KDA_STEP`,
cache operation, model-specific head count, or runtime work queue.

The design adds no cache, private recurrent SRAM, new MAC array, or runtime
scheduler. Recurrent state is explicitly transferred between HBM and the
existing Matrix SRAM under Compiler ownership.

## Architectural Problem

PLENA's prior Matrix SRAM supports one logical row or one logical column. A
linear recurrence needs several narrow logical rows at once:

- Nemotron Mamba-2 consumes rows from `x`, `dt`, `DeltaA`, `B`, `C` and state;
- Kimi KDA consumes rows from `q`, `k`, `v`, decay, beta and state.

With a fixed physical map, words requested in the same packet can target one
single-port bank and serialize. A normal transpose only changes row versus
column orientation; it does not solve conflicts among several simultaneous
logical rows or fields.

That observation alone does not justify programmable skew. For the two official
BF16 state packets evaluated here, PLENA's existing fixed diagonal mapping plus
an ordinary Compiler-selected base-bank phase per tile is already
conflict-free. The arbitrary programmable row coefficient is therefore removed.
Only a compact inter-tile phase remains, as an encoding compression rather than
a separately credited bank-performance mechanism.

The Compiler instead assigns each live tensor a view. For each logical bank
word it selects a physical row and bank:

```text
bank_row = base_row
         + tile * tile_pitch_rows
         + logical_row * row_groups
         + word / bank_count

bank = (base_bank
      + bank_row
      + tile_phase_stride * tile
      + fixed_group_phase * (bank_row / bank_count)
      + word) mod bank_count
```

The same descriptor is used on write and read. A cyclic lane restore returns
the original logical order after a row or column packet is read. Numbered-value
round trips and alias checks make a wrong map observable.

## ISA Contract

`L_TILE_CFG` atomically installs one of four view records. The two 32-bit
descriptor words are:

```text
shape:
  rows_minus_one[11:0]
  cols_minus_one[23:12]
  tile_count_minus_one[31:24]

mapping:
  tile_pitch_rows[15:0]
  reserved_zero[21:16]
  tile_phase_stride[27:22]
  flags[31:28]
```

Bits `[21:16]` and flag bits 0 through 2 are reserved and must be zero. Flag bit
3 encodes minor-dimension broadcast; bounds are always strict. The
effective row coefficient is always one and is not encoded. The fair D' result
showed no bank-service reason to carry a general row multiplier into RTL. Only
the inter-tile phase remains configurable, which lets one compact descriptor
express compiler-known base phases for many tiles.
Matrix SRAM, prepared fields and recurrent state are uniformly BF16. Reserved
encodings are rejected by the Compiler, assembler and Rust decoder.

`L_TILE_EXEC` explicitly names three base registers. Slots 0, 1 and 2 provide
their view descriptors, so there is no hidden `SELECT` state. The source and
scale operands may independently use a row or column axis. The configured view
shape is the static loop bound expanded by the decoder.

Viewed transfers reuse the existing `H_PREFETCH_V` and `H_STORE_V` opcodes.
Bit 31 marks a Matrix-view transfer and bits 30:29 name its slot. This extension
does not reinterpret legacy words: for a legacy transfer, every nonzero
precision selector still means KV; the viewed form alone admits the explicit
BF16 state selector.

The primitive field has a closed, model-independent set:

| Primitive | Tensor meaning |
|---|---|
| `SCALE_ACCUM` | `dst = a * dst + b * src` with segment scalars |
| `DOT_REDUCE` | multiply followed by a segmented dot reduction |
| `OUTER_UPDATE` | rank-1 update of a Matrix-SRAM tile |

These are algebraic forms shared by linear recurrent layers. Mamba-2 composes
scale-accumulate and dot-reduce; KDA composes all three. The decoder never
selects a model.

## Why `EXEC` Is Justified

Arlo's static lowering proves that existing Vector operations can express the
formulas. It also exposes the bottleneck: the row-by-row path spends most issue
slots on pointer updates, scalar loads and loop control. Existing `.MV` forms
consume one Matrix packet per issued instruction; they do not walk the whole
view, provide one scalar per logical row, or fuse update/reduction forms.

`L_TILE_EXEC` does not add arithmetic. It moves a deterministic Matrix-view walk
from the instruction stream into a small sequencer and sends restored packets
to existing Vector multiply/add/reduce hardware. This is analogous to a Matrix
instruction describing a tiled GEMM rather than listing every MAC.

The alternative model-specific fused instructions were rejected because they
would duplicate arithmetic and require decoder branches for each recurrence.
The alternative of only adding an addressing suffix was rejected because it
still issues once per packet and does not remove the measured loop/scalar
overhead.

## Generalization Boundary

The physical view is used by ordinary row access, transposed column access,
projection writeback, Mamba packets and KDA packets. Attention/MLA/MoE continue
to use their existing arithmetic instructions and are checked for unchanged row
and column service.

`L_TILE_EXEC` is general across linear recurrences, not a claim to
accelerate every operator. That narrower statement is deliberate and testable:
another recurrence is supported when it decomposes into the three primitives
and provides legal view shapes; adding it does not change the ISA or decoder.

## Compiler Integration

The official manifests are pinned to:

- Nemotron 3: 52 layers, including 23 Mamba, 23 MoE and 6 GQA layers;
- Kimi K3: 93 layers, including 69 KDA and 24 MLA layers, with 92 latent-MoE
  blocks and one dense FFN.

Every official Mamba/KDA layer emits a complete `L_TILE` recurrence. HBM state
and field arenas are disjoint and 64-byte aligned. The evaluated PLENA state is
BF16: 1 MiB per Nemotron Mamba layer and 3 MiB per Kimi KDA layer. The official
GPU implementations retain FP32 state; that is an accuracy reference, not the
PLENA storage geometry. The 1 MiB Matrix SRAM is reused by statically streamed
head groups.

All compact phased schedules pass view-dominance validation and assemble into canonical
32-bit words. Ordinary layers are supplied by the existing analytic timeline;
they are not duplicated as fake instructions in the recurrence schedule.

## Fair Ablation

| Variant | Mechanism | Credit |
|---|---|---|
| A | original PLENA row-by-row recurrence | baseline |
| B | A plus Arlo static lowering | Compiler-only |
| C | executable single-base fixed descriptor | descriptor/issue control |
| D' | fixed diagonal wiring plus one legal base phase per head tile | fair bank-only control |
| D | compact Compiler-selected phased descriptor | executable L-Compute treatment |
| E | D plus capacity-legal static overlap | overlap only |

C, D' and D keep model formulas, BF16 state, Matrix capacity, bank count, ports
and Vector arithmetic identical. D' gives the fixed control every ordinary
base-placement freedom available to D. It occupies the same physical bank
words as D and reaches zero stalls, so the programmable mapping receives 1.00x
pure bank credit. C-to-D changes descriptor compactness, chunking, issue count and
KDA spill traffic; it is not labelled a bank-conflict speedup.

At 1 MiB, E cannot yet hold a second disjoint state group: it needs at least
45,312 more bytes for Nemotron or 28,736 for Kimi. E therefore equals D and
receives no fabricated overlap credit.

## Current Results

Paper-point model: `MLEN=2048`, `BLEN=32`, 64 banks, BF16 state, 1 MiB Matrix
SRAM and 1560 HBM bytes/cycle.

### Compiler-address recurrence replay

This table is a Python replay of the complete Compiler service groups in Rust
phase order, not a Rust cycle measurement:

| Model | C local cycles | D local cycles | C bank stall | D bank stall |
|---|---:|---:|---:|---:|
| Nemotron Mamba-2 | 12,216 | 3,680 | 256 | 0 |
| Kimi K3 KDA | 45,300 | 18,780 | 0 | 0 |

The fair fixed-wiring `D'` state packet takes one cycle with zero stalls for
both models and occupies the same cells as D. Thus D/D' pure-bank speedup is
`1.00x`; C-to-D differences are descriptor, chunk, issue, ideal-service and
KDA-spill effects.

### Full formula-based B1 decode timeline

`A` and `B` are one-cycle-per-issued-instruction proxies for the original and
Arlo streams; neither is a Rust cycle measurement. `C` and `D` add explicit
Matrix service, arithmetic and HBM terms. The `D/A` and `D/B` ratios therefore
measure multi-row utilization plus issue compression, not programmable skew.

| Model | A | B | C | D | D/A | D/B | D/C |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nemotron 3 | 4,055,091 | 3,110,067 | 2,192,850 | 2,014,094 | 2.0134x | 1.5442x | 1.0888x |
| Kimi K3 | 103,816,704 | 97,013,856 | 93,124,740 | 91,173,903 | 1.1387x | 1.0641x | 1.0214x |

KDA decay/beta preparation is identical in every variant: B1 charges 5,107,104
ordinary elementwise operations and 1,702,368 exponentials across 69 layers as
4,485 Vector cycles. The model implements the official decay and beta formulas;
`L_TILE` receives only prepared coefficients. Mamba dt/exp remains its separate
Vector stage.

The C-to-D savings break down as follows:

| Model | Total | HBM | issue | ideal service | bank stall | arithmetic |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron 3 | 178,756 | 0 | 161,552 | 11,316 | 5,888 | 0 |
| Kimi K3 | 1,950,837 | 278,277 | 1,463,904 | 208,656 | 0 | 0 |

The BF16 point uses 32 values in one 512-bit bank word per cycle; no hidden port
widening is credited. A separate 256-bit, two-beat sensitivity point is kept in
the generated campaign and is explicitly a Python timing model rather than a
Rust port-timing result.

The current timing scoreboard tracks Matrix views with conservative logical
extents rather than exact physical bank-word footprints. Physical pending cells
preserve functional ordering, and the evaluated E variant receives no overlap
credit; exact view-overlap timing is therefore outside the present claim.

### Compiler-to-Rust numerical execution

Four-token tests at official recurrence geometry pass through Compiler
lowering, assembly, canonical machine words, Rust decoding, physical BF16 banks
and explicit HBM state/output readback. They compare 524,288 Nemotron and
1,572,864 Kimi state values plus all head-group outputs. The largest relative-L2
error is 0.0071. Fixed/phased cycle differences in these tests include program
expansion and transfers and are not credited to programmable skew.

## Resource Contract

Pre-RTL structural proxies, not PPA:

- extra SRAM payload: 0 bytes;
- cache/tag/replacement state: 0 bits;
- new MAC lanes: 0;
- extra read/write ports per bank: 0/0;
- four view records: 256 bits;
- sequencer state upper bound: 256 bits plus three loop counters;
- segment-scalar broadcast: at most 32 x 16 bits (32 Nemotron 64-value
  segments, or 16 Kimi 128-value segments, at MLEN=2048);
- programmable row-bank coefficient: removed before RTL (0 adders);
- inter-tile phase generation: one six-bit accumulator per active view;
- cyclic restore: 64 bank words over six mux stages;
- incremental bank-port width over the 512-bit reference: 0 bits.

The row term is frozen to PLENA's existing diagonal wiring because D' does not
show a bank-speed benefit for a programmable coefficient. Synthesis is required
before quoting area, frequency, power, energy or PPA.

## Evidence Boundary

Demonstrated:

- canonical Compiler/assembler encoding and loop-aware view dominance;
- physical Rust banks, row/column reads, lane restoration and viewed writes;
- four-token official-geometry numerical Mamba and KDA recurrences compiled to
  machine words and executed by Rust with BF16 state/output readback;
- official-shape address replay and zero compact-phased bank stalls;
- every official recurrent layer emitting legal `L_TILE` machine words;
- official 52/93-layer formula-based timelines and ordinary-layer no-regression.

Not demonstrated:

- real checkpoint weights numerically executed from first to final layer in
  Rust for Nemotron or Kimi;
- a transactional `L_TILE` prefill speedup;
- capacity-legal producer/consumer overlap at the one-MiB point;
- RTL timing, synthesis, PPA or Token/J.
