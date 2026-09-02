# Matrix SRAM L-Compute ISA Review

## Decision

The fair Round-3 control changes the ISA decision:

1. PLENA keeps one fixed diagonal Matrix-SRAM bank mapping.
2. The Compiler selects only `tile_pitch_rows` for each tensor view.
3. Nemotron uses pitch 2 and Kimi uses pitch 4 at `MLEN=2048, BLEN=32`.
4. Both reach the same bank floor as a counterfactual programmable-skew search.
5. Therefore per-view `alpha` and `gamma` are not architectural fields.

This is a useful negative result. It removes an unnecessary degree of freedom
from the ISA while retaining conflict-free multi-row access.

## Problem Being Expressed

The original Matrix SRAM can expose one logical row or one logical column. A
Mamba or KDA recurrence instead consumes a packet containing narrow rows from
several heads. With pitch 1, those rows can request the same single-port bank in
the same service step.

The Compiler already knows the producer and consumer shapes. It lays successive
head tiles at a pitch that makes the fixed diagonal mapping distribute their
bank words:

```text
physical_row = base_row + tile * tile_pitch_rows
             + logical_row * row_groups + word_group

bank = (physical_row + bank_word) mod bank_count
```

This is placement metadata. It does not encode a recurrence formula, fused
Mamba/KDA operation, cache policy or runtime traversal.

## Public ISA

One opcode, `0x3F`, has two canonical forms selected by `funct1`:

```text
L_MVIEW.FULL   slot, shape_reg, map_reg     # funct1 = 1
L_MVIEW.FIELD  slot, field, value_reg       # funct1 = 2
```

`FULL` configures the hot path atomically. `FIELD` supports reset or a cold
shape/mapping update. Existing consumers explicitly name a view:

```text
M_MM_WO     ..., view=slot
M_MM/M_TMM  ..., view=slot
<vector-op>.MV ..., operand_view_mask
```

The arithmetic opcode is unchanged. `.MV` is an operand addressing mode, not a
new arithmetic primitive. Its three mask bits qualify destination, source 1 and
source 2 with slots 0, 1 and 2. Slot 3 remains available to explicit Matrix
producer/consumer encodings; arbitrary Vector slot routing is not claimed.

The packed descriptor is:

```text
shape:
  rows_minus_one[11:0]
  cols_minus_one[23:12]
  tile_count_minus_one[31:24]

mapping:
  tile_pitch_rows[15:0]
  reserved_zero[27:16]
  flags[31:28]
```

The mapping word deliberately carries no skew coefficient, bank count, bank
width, model name, head count or operation. `STRICT_BOUNDS` is currently the
only valid flag. Nonzero reserved bits are rejected by the Compiler, assembler
and Rust decoder.

## Why This ISA Is General Enough

The instruction describes a tensor view over Matrix SRAM. The same contract is
used for:

- ordinary row access;
- transposed column access;
- Matrix projection writeback into a consumer-shaped view;
- Mamba cross-head packets;
- KDA cross-head packets.

Attention and MoE use the ordinary row/column forms and do not need model
specific encodings. Mamba and KDA differ only in shape and pitch. No opcode is
named after either model and no decoder branch selects either formula.

This also explains why a fused `MAMBA_STEP` or `KDA_STEP` was rejected. Those
instructions would duplicate existing Matrix/Vector arithmetic and would make
the ISA's users model specific. `L_MVIEW` exposes only the physical placement
property that ordinary operations cannot otherwise name.

## Static Semantics

There is no implicit `SELECT` state. Every consumer carries its view operand.
The assembler runs a must-dataflow analysis over static loop back-edges and
rejects a consumer unless its configuration dominates every dynamic use.
Inside a loop, `C_BREAK` contributes both a fallthrough edge and an edge to the
matching loop exit; their intersection is conservative for the public
debug-exception wording and the emulator's legacy loop-break behavior.

The canonical encoder is shared by the assembler tests and the contract module.
`L_MVIEW.FULL` seeds both descriptor words; `FIELD RESET` invalidates the slot.
View-qualified `M_MM_WO` ignores stale legacy `L_CFG` auto-advance state.

## Fair Control and Measured Outcome

The control and treatment have the same freedoms except the one being tested:

| Path | Fixed bank wiring | Per-view pitch | Per-view skew |
|---|---|---|---|
| C, pitch-1 packet | `alpha=1, gamma=0` | fixed at 1 | no |
| Implemented co-layout | `alpha=1, gamma=0` | yes | no |
| Counterfactual upper bound | searched | yes | yes |

Every path uses the same dynamic operation stream and moves the same numbered
values. At 64 banks with 32 BF16 values per bank word:

| Packet | Pitch-1 service | Implemented service | Skew upper bound |
|---|---:|---:|---:|
| Nemotron, 32 heads x 64 values | 2 cycles | 1 | 1 |
| Kimi, 16 heads x 128 values | 4 cycles | 1 | 1 |

The implemented pitches are 2 and 4. Across every recurrence row, each model
places and restores 262,144 values with no physical alias. The interleaving
fills the apparent gaps, so capacity overhead is zero. Programmable skew has a
measured upper-bound speedup of exactly `1.0x` over the implementation.

Ordinary Attention/MoE row and column access is replayed at every one of the 64
allocation-base phases and retains the same values and service cycles.

## What L-Compute Does and Does Not Get Credit For

Credit boundaries are strict:

- Arlo's address/loop compression is `A -> B` Compiler credit.
- Matrix L-Compute is pitch-1 `C -> implemented co-layout` bank credit.
- Projection/consumer overlap is measured separately from the co-layout.
- The programmable-skew upper bound gets no architectural credit.
- The old KDA prefill `3.387x/1.713x` claims are withdrawn; the two complete
  paths were not measured under the same timeline.

For official FP32 B1 decode, the formula-based serial full-model gains of the
implemented co-layout over pitch 1 are `1.00562x` for Nemotron and `1.00648x`
for Kimi. The local packet gains are larger (`2x` and `4x`) but HBM, MoE and
other stages dominate the full timeline.

The local result is numbered Python physical-cell replay of real Compiler
addresses. The full-model result uses official shapes, GPU calibration and
symbolic PLENA weights. It is not a real-checkpoint first-to-last Rust run.

## Resource Contract

This pre-RTL design adds no cache, tags, replacement state, private recurrent
SRAM, new MAC lanes, SRAM payload or SRAM ports. It reuses:

- the existing fixed diagonal Matrix-SRAM bank mapping;
- existing Matrix row/column data paths;
- four small view records;
- lane selection/restoration already required by column access;
- existing Matrix and Vector arithmetic.

These are structural proxies, not PPA. No frequency, area, power, energy or
Token/J claim is valid until a later RTL implementation and synthesis.

## Explicit Boundaries

- Handwritten assembly can deliberately write and read a tensor through
  mismatched descriptors. Generated recurrence paths use one descriptor for the
  producer and all consumers; a future typed ownership pass can enforce this
  above raw assembly.
- Matrix SRAM is an explicitly addressed scratchpad. Tensor-region ownership
  and non-overlap remain Compiler allocation responsibilities.
- Opcode `0x3F, funct1=0` remains the legacy `L_CFG` compatibility form but is
  outside the frozen Matrix-view contract.
- Official recurrent state remains explicit FP32 HBM traffic. There is no
  hidden state residency or cache claim.
- Nemotron 52-layer and Kimi 93-layer numbers are analytic timelines with
  symbolic weights, not full transactional real-weight execution.
