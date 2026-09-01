# Matrix SRAM L-Compute ISA Review

## Decision

The architectural feature is a compiler-programmable **Matrix-SRAM view**, not
a Mamba or KDA instruction. The arithmetic remains on existing PLENA Matrix and
Vector operations. The view changes only physical bank placement and restores
logical lane order before an operation observes its operand.

The frozen public shape is:

```text
L_MVIEW.FULL   slot, shape_reg, map_reg
L_MVIEW.FIELD  slot, field, value_reg
<consumer>     ..., view=slot
<Vector op>.MV ..., operand_view_mask
```

`FULL` and `FIELD` share opcode `0x3f` and use different `funct` values. `FULL`
is the hot form; `FIELD` exists for a cold partial update or reset. There is no
implicit `SELECT`: each consumer names the slot in its own machine word.
For the three-operand Vector family, the explicit mask maps destination,
source-1 and source-2 to slots 0, 1 and 2. This keeps the arithmetic opcode and
its mathematical semantics unchanged; `.MV` is an operand-addressing mode.

## What a view contains

`shape_reg` packs:

```text
rows_minus_one[11:0]
cols_minus_one[23:12]
tile_count_minus_one[31:24]
```

`map_reg` packs:

```text
tile_pitch_rows[15:0]
alpha[21:16]
reserved_zero[27:22]
flags[31:28]
```

`gamma`, bank count, bank width and row/column direction are deliberately
absent. `gamma` is one machine constant. Bank geometry is a machine property,
not architectural state. Row versus column is already expressed by `M_MM`
versus `M_TMM`; a second field would duplicate existing semantics.

For physical bank-row address `r_phys` and bank word `w`:

```text
bank = alpha*r_phys + gamma*floor(r_phys/banks) + w  (mod banks)
```

`r_phys` already includes allocation base, tile pitch, logical row and any
wide-row word group. Computing the bank from a tile-local row discards this
information and creates a false need for a per-tile phase; that earlier model
and all speedups based on it have been withdrawn.

The decoder expansion is deterministic:

```text
AFFINE_ADDRESS -> BANK_READ -> LANE_RESTORE
               -> EXISTING_OPERATION -> BANK_WRITE
```

It is not a runtime scheduler and it does not interpret a model formula.

## Why one ISA-visible value is necessary

The conflict-free skew equals the number of physical bank words in one logical
tensor row. Kimi's 128-value row needs `alpha=4` at `BLEN=32`; Nemotron's
64-value row needs `alpha=2`. That number is a tensor property. The hardware
sees only physical addresses and cannot infer where the Compiler's logical row
ends.

`D'` exhaustively searches all 4096 global `(alpha,gamma)` wirings, preserving
the ordinary 128-value column-read floor, and scores them on the dynamic
addresses emitted by the real recurrence lowerings. The selected mapping is
then replayed through numbered physical words:

| Official FP32 field traffic | Original fixed `C` | Best global `D'` | Per-view `D` |
|---|---:|---:|---:|
| Nemotron Mamba | 1536 cycles, 768 stalls | 768, 0 | 768, 0 |
| Kimi KDA | 12288 cycles, 9216 stalls | 6144, 3072 | 3072, 0 |

Nemotron honestly reports `D' == D`; it does not justify programmability by
itself. Kimi is the counterexample that does: the globally fixed map cannot
simultaneously preserve ordinary column service and reach Kimi's packet floor,
while the Compiler-selected row-width skew does. No measured real lowering
requires `beta`, so it remains reserved rather than becoming ISA state.

## Why this is general rather than a renamed model opcode

The instruction describes an affine placement family. It contains no Mamba,
KDA, attention, head count, state dimension, update equation or loop bound.
The same closure covers:

- existing row and transposed Matrix reads;
- cross-head packets in Mamba and KDA;
- cross-field projection packets;
- grouped B/C operands;
- ordinary Attention/MLA/MoE tiles with the identity view.

Generality is a property of the address family, not the number of benchmark
names. A future lowering is supported when its simultaneous-access packet can
be represented by the same affine coefficients; it does not require an ISA
change.

## Why not a fixed layout?

The `D'` experiment is mandatory. It exhaustively chooses the best global
fixed `(alpha,gamma)` on the physical row before comparing it with per-view
`alpha`. Nemotron already demonstrates the required negative: when `D == D'`,
the Compiler should keep the fixed map and emit no view configuration. Kimi
demonstrates the positive case. This control prevents a programmable feature
from taking credit for a change that one hardwired constant can reproduce.

## Why no implicit configuration selection?

An implicit active-view register makes correctness dependent on hidden state.
With an explicit slot on every consumer, “configuration dominates use” is a
single-pass syntactic property. The Compiler and assembler test it, and the
Rust decoder fails closed on an unconfigured slot.

This choice follows the useful part of established ISA practice:

- RISC-V Vector documents that `vtype` primarily exists to fit a 32-bit
  encoding, while vector memory operations explicitly encode their addressing
  properties.
- Arm SME tile moves explicitly identify tile/slice direction and index.
- Intel AMX demonstrates the software and context-management cost of a large
  implicit tile configuration; L-MVIEW keeps only 256 bits of ordinary,
  compiler-written placement state and adds no architectural tile payload.

## Why not a fused Mamba/KDA instruction?

A fused state-step instruction would duplicate `M_MM`, vector multiply/add,
reduction and hardware-loop semantics, then require a decoder path for every
new recurrence. L-MVIEW instead modifies operand addressing while preserving
the existing operation's mathematics. This makes the decoder expansion
composable and keeps the fallback path byte-identical when no view is selected.

## Why this is not SSR or DataMaestro

Stream Semantic Registers remove explicit scalar load/store issue by mapping
streams onto registers. They do not select the physical bank skew of each
producer tile. DataMaestro supports programmable streams and mitigates
conflicts at access time. L-MVIEW's distinct operation is compiler-guided
placement: the Matrix producer writes a tensor using the skew implied by its
logical row width, so the later packet has no conflict to arbitrate.

Arlo's static loop/address lowering remains a separate, compatible optimization:

```text
Arlo lowering: fewer pointer, scalar and loop instructions
L-MVIEW:       more Matrix bank words served in the same cycle
```

The same tensor must not be gathered by the first path and then scattered by
the second. The Compiler chooses one physical layout at the producer.

## State and memory boundary

No cache, tag, replacement policy, private state SRAM, `X_STATE`, queue or
runtime state machine is part of this design. State is an explicitly addressed
tensor and all HBM transfers remain visible in the program and cost model.

Official Kimi state is FP32 and 6 MiB per KDA layer; official Nemotron state is
FP32 and 2 MiB per Mamba layer. The current Matrix SRAM is BF16, so the official
path cannot silently claim full state residence. The unconditional L-MVIEW
scope is BF16 projection/temporary data. BF16/FP16/MX8 state is a separately
labelled mixed-precision design point and requires an accuracy gate.

## Resource contract before RTL

At the evaluated 64-bank point the candidate requires:

- four 64-bit view records: 256 configuration bits;
- 64 six-bit bank-select adders;
- a 64-word cyclic lane-restore network, not an arbitrary crossbar;
- no new operand SRAM: a two-source operation on a one-read-port bank reuses
  the existing Vector operand buffer to hold the first 2048-element BF16
  packet (4096 bytes of required existing capacity) while source 2 arrives;
- a 32768-bit-per-cycle Matrix-to-Vector operand path and matching
  Vector-to-Matrix writeback path, which may reuse the existing wide row buses
  but require input/output selection muxes;
- no extra SRAM payload, cache metadata, MAC lanes or per-bank ports.

These are structural proxies. LUTs, gates, area, frequency and power are not
reported until synthesis exists.

## Implementation status

Implemented and tested on both sides:

- canonical `FULL`/`FIELD` encoding and bounds checks;
- explicit view slots on existing Matrix read consumers;
- dominance checking;
- physically banked Rust Matrix SRAM;
- inverse lane restore and wrong-skew negative tests;
- skewed Matrix-accumulator writeback;
- direct `M_MM_WO` fragments into the consumer's true logical head shape;
- existing `V_ADD_VV/V_SUB_VV/V_MUL_VV` consuming Matrix packets through an
  explicit `.MV` addressing mode, with no transfer instruction;
- real Nemotron/Kimi 2048-value packet roundtrips.
- row and column reads from the same physical cells, including a non-symmetric
  KDA prefill `[value,key]` to decode `[key,value]` handoff;
- an emitted-code census of the legacy Kimi identity transpose and a zero-MAC
  Matrix-view alternative, kept outside the official FP32 timing claim.

The producer and consumer now share the same descriptor. A Matrix accumulator
writes one `BLEN`-wide fragment at a time into a wider logical consumer tile;
the Compiler advances the logical offset by `BLEN`, not by a producer-only tile
size. Tests cover the paper shapes `32 x 64` and `16 x 128`, require complete
value/lane recovery, and prove that reading those bytes through the old
`64 x 32` producer-only descriptor returns the wrong result. The remaining
pre-RTL limitation is physical cost: the bank selector, cyclic restore and wide
bypass muxes above are structural proxies until synthesis exists. The official
52/93-layer packet core is a structural lowering with symbolic addresses; the
current Rust numerical recurrence uses reduced outer dimensions and therefore
does not constitute a real-checkpoint first-to-last-layer execution.

The prefill measurement exposes a second reason for a view rather than an
identity GEMM. The mathematical transpose costs `128^3` MACs per head, or
13.89 G across 96 heads and 69 KDA layers. The current MLEN=2048 lowering pads
all three dimensions and actually emits 56.90 T MACs. A BF16/MX8 view reads the
same per-head cells by columns and explicitly streams them out in decode order,
executing zero transpose MACs. The later cross-head decode packet is a separate
compact view (`alpha=4`), so no direct-residence benefit is claimed across that
boundary. Official FP32 state is still streamed and receives none of this
credit.

## References

- P. Budnik and D. J. Kuck, “The Organization and Use of Parallel Memories,”
  IEEE Transactions on Computers, 1971, DOI 10.1109/T-C.1971.223171.
- RISC-V “V” Vector Extension, Version 1.0, sections 3.4 and 7.2.
- Arm A-profile SME `MOVA` / tile-slice access specification.
- F. Schuiki et al., “Stream Semantic Registers,” arXiv:1911.08356.
- X. Yi et al., “DataMaestro,” arXiv:2504.14091.
