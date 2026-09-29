# Compiler front-end and real-input provenance

This document describes `compiler.py` and `workloads.json`. These are an independent
candidate compiler front-end and explicit execution/storage plans. They do not
emit the main PLENA Compiler ISA, synthesize hardware, run a pretrained model, or
measure timing. Performance reported by another runner must identify its own
timing assumptions and validation separately.

## Captured workload, without synthetic routes

The source is
`capture://outputs/moe_spatial_batch_real_20260921/real_inputs/manifest.json`,
SHA-256 `b43aa9daac7d07c3b2ad003caef56bbb9a00637c69778e872f460b8ebe78dd73`.
It contains original expert IDs, ordered top-k slots, unmodified route scores,
sample identifiers, tensor dimensions, input and weight tensor hashes.

The capture script executed embeddings, full decoder layer 0, and attention plus
normalization of decoder layer 1 on the first 16 BFCL_v3_simple prompts. It retained
each prompt's final input token and applied the actual layer-1 router. This is a
**last-token prefill capture**, not an autoregressive decode trajectory. Router
selection is an input to this study; router and preceding-model latency are not
included.

The compiler takes the first B captured tokens for B = 2, 4, 8, 16. It preserves
route-rank order and original route scores, without re-normalization. Routed
expert IDs remain the original model IDs; the shared branch uses logical ID -1.
For this model H = 2048, routed intermediate dimension F = 1408, shared F = 2816,
and top-k = 6. Dimensions are checked against all three captured weight shapes
and the local model configuration. During original capture import the supplied input payload is hash-verified; bundled
replay checks only the metadata checksums and retains historical source hashes;
the large weight payloads are not loaded by this compiler, and their hashes are
attributed to the capture manifest rather than claimed freshly verified.

| Batch | Routed expert IDs and token counts | Shared Me | BF16 selected weight bytes, MiB |
|---:|---|---:|---:|
| 2 | 25:2, 26:1, 43:2, 52:2, 55:2, 56:1, 57:2 | 2 | 148.5 |
| 4 | 5:1, 25:4, 26:2, 43:3, 52:4, 55:4, 56:2, 57:4 | 4 | 165.0 |
| 8 | 5:2, 9:2, 25:8, 26:2, 43:7, 52:8, 55:8, 56:3, 57:8 | 8 | 181.5 |
| 16 | 5:3, 9:3, 25:16, 26:7, 43:15, 52:16, 55:16, 56:4, 57:16 | 16 | 181.5 |

Weight byte counts are **derived from the captured tensor shapes**: each selected
expert reads gate, up and down once. They are not new Ramulator measurements.
Each workload includes exact token indices and route slots for restoring the
original ordering after expert grouping.

## Explicit expert DAG and two ownership modes

Each expert declares `input -> gate/up -> SiLU/product -> Z -> down -> retirement`.
Gate and up consume X[Me,H], produce BF16 matrices [Me,F], and down consumes
Z[Me,F] and produces [Me,H]. Each GEMM uses BF16 source values, FP32 partials, and
the declared BF16 inter-stage rounding. Before route combination, down outputs
must receive the model's BF16 rounding; route weighting/accumulation uses the
captured order and scores. The compiler does not execute these operations.

Physical dimensions are M x N x K: 6x4x512, two 3x4x512 cores, or 4x4x512 plus
2x4x512. Every organization has 12,288 physical multipliers at parallel K=512.

* **whole**: one candidate per eligible core. A current expert instance keeps its
  owner through gate, up, vector operations and down. It streams weights in
  finite slots; the entire expert weight tensor is never required to reside.
* **paired_n**: gate and up use the same contiguous F-column partition. Each core
  creates its local Z columns. Explicit source-read/bus/destination-write copies
  exchange these pieces so each core has full Z before down. Down splits H
  output columns. This mode requires joint admission and reserves both workspaces;
  it is not free opportunistic migration. Two-core Z exchange payload is 2*Me*F
  bytes, while each core gathers a complete X input.

The partition uses 4-column boundaries at
`floor(cumulative_M/total_M * ceil(N/4))*4`, clipped at the true N. For H=2048,
4+2 owns columns [0,1364) and [1364,2048); 3+3 owns 1024 columns each. Whole and
paired modes have identical useful MACs and total weight bytes for an expert.
Additional X copies, Z exchange and retirement traffic remain explicit costs.

## Address and storage contract

Addresses are bytes. Proposed HBM storage uses BF16 W[N,K] with rows aligned to
32 bytes. Each `(expert slot, phase)` has a fixed 16 MiB slot; original routed IDs
are slots 0..63 and shared is slot 64. The phase bases are distinct. This is a
**new full-DAG logical layout**, not the prior isolated-phase native-HBM address
sequence. It must not be advertised as a replay of that unchanged address trace.

All organizations retain these aggregate capacities:

* Accumulator/intermediate/control arena: 2 MiB, physically partitioned by M.
* Control: 4 KiB inside that 2 MiB, not additional storage.
* W: 48 KiB total = 8 KiB shared ingress + 40 KiB private tiles (10 or 5+5 slots).
* X operand SRAM: 12 KiB total, two slots per core (12 / 6+6 / 8+4 KiB).

Private accumulator and control partitions use 32-byte boundaries and sum exactly
to the aggregate budget. For 4+2 the arena sizes are 1,398,080 and 699,072 bytes;
control reservations are 2,720 and 1,376 bytes. This granule rounding differs by
at most a few dozen bytes from an unaligned 2:1 division.

Before dispatch, fixed H-column owners reserve:

1. Original layer input X: 2*B*H bytes distributed by those columns.
2. FP32 inbox payload for every expert output: 4*H*sum(Me) bytes in total.
3. FP32 final combined output: 4*B*H bytes in total.

In addition, each core explicitly stores a copy of the finite route table:
`B*top_k*16 + E*64` bytes, for 16-byte token/expert/rank/score entries and 64-byte
expert records. This is data in the 2 MiB arena, separate from the 4 KiB controller
state. The result/input region begins at `align32(control_share + route_bytes)`;
workspace begins after that region. These route records, like original X, are
already supplied at layer entry and do not imply timing the upstream router.

Each execution core gathers only its expert's captured token rows from these
finite input regions into its workspace. Local and remote gathers both charge
source reads, shared-bus bytes, and destination writes. There is no unlimited
on-chip X producer. Upstream production into the original-input regions is
outside this layer's measured scope.

Retirement similarly copies producer Y to the pre-reserved inbox column owners.
Local copies are charged, too. A completed expert can release its workspace
without waiting for other experts to reach ordered combination, because its
inbox has already been reserved. This removes the circular dependency in which
finished experts occupy both cores while an earlier route-rank expert cannot
start. Result inboxes contain FP32 payload only: the old permanent 16-byte metadata
per four-output row is **not retained**. Metadata exists only for bounded active
groups and is released at group retirement, consistently for all organizations.

An active whole-expert workspace reserves:

```
align(2*Me*H)       # gathered X
+ align(2*Me*F)     # gate -> Z, in-place
+ align(2*Me*F)     # up
+ align(4*Me*H)     # producer Y
+ 2*align(Me*G*32)  # finite FP32/metadata scratch, G resident bands
```

For paired N, the gate/Z region still covers full F, up covers only local F,
and producer Y covers only local H. The compiler emits explicit gate and Z
views of their shared allocation: vector reads of each gate/up chunk must finish
before Z overwrites the gate chunk. Remote Z fragments fill the other F columns;
down waits for the entire required Z input. The alias saves capacity, not vector
read/write service.

The current G=4 group fits four W slots and leaves one lookahead slot on a dual
core. Invalid group sizes fail before a plan is produced. Every candidate has its
own absolute workspace addresses, peak, headroom and feasibility flag; aggregate
capacity never substitutes for an individual core capacity check. For example,
B16/4+2 whole mode permits a routed Me16 expert on the small core (peak 666,464 B
within 699,072 B) but rejects the shared Me16 expert there (756,576 B). The shared
expert is feasible on the large core; the planner does not silently spill it or
borrow the other core's SRAM.

## Finite control accounting

The 2026-09-30 Current/Next repair replaces the reserved but unused cost LUT with
explicit successor, progress and return ownership state. All state remains inside
the original **4 KiB arena reservation**, not W/X SRAM. Registers/tags are charged
as bytes; their physical placement and synthesized implementation are unproven.

| Item | Single | Dual |
|---|---:|---:|
| Existing controller records | 1,200 B | 1,872 B |
| Pending FIFO, 8 × 64 B | 512 B | 512 B |
| Current, 128 B/core | 128 B | 256 B |
| Next, 128 B/core | 128 B | 256 B |
| Progress/service estimate state, 64 B/core | 64 B | 128 B |
| Return tags, 256 × 16 bits | 512 B | 512 B |
| W slot owner records, 10 × 8 B | 80 B | 80 B |
| Global credit/aging/cursor state | 128 B | 128 B |
| **Used / remaining reserve** | **2,752 / 1,344 B** | **3,744 / 352 B** |

Previous totals were 2,160/3,152 B: this revision consumes 592 more bytes, with no
new physical capacity. Shared control records are placed in available reserved
partitions and accessed through one charged control port. Per-core totals are
2,048/1,696 B for 3+3 and 2,448/1,296 B for 4+2, within the respective reserves.
Return tags encode core/slot/sector offset; task/phase/tile lives in the slot owner.
An outstanding response prevents slot reuse, so those tags cannot alias a later
owner. Software event IDs and reporting histories are simulator instrumentation,
not extra hardware queues. No synthesis area/timing claim is implied.

The plan does not grant zero-cost selection: its consuming simulator must charge
descriptor reads, bounded candidate comparisons, state updates and owner commit.
An owner commits atomically with FIFO-to-Next binding after standalone workspace
feasibility is checked. The workspace itself is acquired only at promotion after
Current retires. Next can reserve **one** existing W slot, not another X/accumulator
workspace. W destination reservation precedes the first request; each **32 B**
credit is consumed only on DMA acceptance and returned after safe SRAM landing.
Already-prefetched work does not migrate. Four Current bands plus one Next tile
fit each dual core's five slots. `runtime_protocol` documents these interface and
release rules; descriptor identity is the stable expert-array index plus the
explicit expert ID, Me, shapes, token map, phase addresses and result-plan index.

## Verification performed here

`python3 -m unittest discover -s research/moe_dispatch -p 'test_*.py' -v` checks the captured IDs/scores/ranks,
bundled checksums and supplied capture-input hashes, original model dimensions, HBM tensor non-aliasing, exact output
column coverage, weight-byte/MAC conservation, charged local/remote transfer
accounting, full-Z alias views, finite slots, aggregate and per-core storage
limits, and a deliberate over-capacity rejection. These are compiler contract
checks, not timing results or full-model numerical validation.

Reproduction:

```
python3 research/moe_dispatch/compiler.py \
  --plans-dir /tmp/plena-dispatch-v1-compiler-plans
```

This writes four verified captured workloads and 24 explicit organization/mode
plans. Temporary plans belong under `/tmp`; no old repository, frozen report,
native HBM setting, default simulator configuration or weight payload is edited.

The timing runner can call `engine_layout(workload, lanes, group=4)` for compact
absolute addresses. `cores[c]` contains its capacity, reserved workspace base and
all source/result region addresses; `whole[e][c]` and `paired_n[e][c]` contain
allocation bases, row strides, sizes, feasibility and peak. Gate and Z explicitly
alias one physical region; scratch0 and scratch1 are separate arrays of 32-byte
records (16-byte FP32 data, 16-byte metadata). Thus the timing engine can charge
the banks touched by actual planned addresses rather than one placeholder address
for every transfer. `--engine-layouts-dir /tmp/plena-dispatch-v1-engine-layouts`
exports these twelve compact layouts.
