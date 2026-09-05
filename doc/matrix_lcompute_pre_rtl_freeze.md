# Matrix SRAM L-Compute: Pre-RTL Freeze

> **Draft review only — not for merge.** The title describes the historical
> Compiler/Simulator contract, not hardware approval or a newly selected
> architecture. Read [review scope](REVIEW_SCOPE_20260905.md) first.

This document preserves implementation behavior and historical evidence. RTL,
synthesis, frequency, area, power and energy are outside this revision. Test
counts, timing tables, commit IDs and hashes below belong to their archived
runs; for this review checkout, **see paired PR fresh validation**. Old numbers
are retained rather than presented as regenerated results.

## 1. Frozen architectural boundary

The historical candidate uses existing Matrix SRAM capacity and aims to reuse
arithmetic. Original PLENA already has Matrix accumulators; neither their reuse
for this recurrence mapping nor L_TILE's Vector feedback/routing is established
by the following functional model:

```text
HBM -- explicit viewed DMA --> banked Matrix SRAM
                                  |
                           L_TILE row walk
                                  |
                     fixed diagonal bank mapping
                     + compiler tile phase
                                  |
                         cyclic lane restore
                                  |
                   modeled recurrence arithmetic
                                  |
                Matrix SRAM -- explicit DMA --> HBM
```

The Compiler owns every address, lifetime, load and store. The design contains:

- no cache, tags, replacement policy or hit/miss behavior;
- no private recurrent-state SRAM;
- no `X_STATE`, model-specific step instruction or HBM descriptor fetch;
- no runtime queue, work scheduler or completion record;
- no new MAC lane.

Storage is BF16 for Matrix SRAM, prepared fields, state and output. The Rust
model uses FP32 accumulation. DOT_REDUCE retains one FP32 accumulator per lane
across rows: 65,536 bits / 8 KiB at VLEN=2048. Register reuse, feedback muxes,
rounding points and throughput still need an explicit RTL mapping; zero extra
SRAM payload is not a proof of zero extra arithmetic storage.

NVFP4/MX8 weight tables are traffic sensitivity models, not execution of those
weight formats in the Rust recurrence kernel. The prepared A/B execution
controls are new programs and must not be called original-paper baselines.
Optional FP32-dot and BF16-pairwise controls default off and do not select a
new architecture. The six precision paths are separated in the review scope.

## 2. Frozen ISA

The packed Matrix-view descriptor is contract version 3. Version 3 retires the
programmable row coefficient and optional storage-precision flags; decoders
must reject those formerly occupied bits rather than silently reinterpret them.

The Matrix mechanism consumes one physical opcode:

```text
L_TILE_CFG   slot, shape_reg, map_reg
L_TILE_EXEC  dst_base, src_base, scale_base, primitive[, axis_mask]
```

Both use `L_TILE=0x3F`; `funct1=1` selects CFG and `funct1=3` selects EXEC.
The instruction does not contain a model name or head count.

`L_TILE_CFG` installs one of four atomic view records. The descriptor contains
rows, columns, tile count, tile row pitch, tile bank-phase stride and flags.
The Matrix row coefficient is not programmable: it is fixed to the diagonal
mapping already used by PLENA and is not encoded. Mapping bits `[21:16]` and
flag bits 0 through 2 are reserved and trap when non-zero. A zero tile phase selects the
ordinary fixed form; a non-zero phase compactly describes per-tile base phases.

```text
shape   = rows_minus_one[11:0] | cols_minus_one[23:12]
        | tile_count_minus_one[31:24]
mapping = tile_pitch_rows[15:0] | reserved_zero[21:16]
        | tile_phase_stride[27:22] | flags[31:28]
```

Flag bits 0 through 2 are reserved. Flag bit 3 is minor-dimension broadcast;
bounds are always strict. Matrix-view storage is uniformly BF16.

`L_TILE_EXEC` walks slots 0/1/2 as destination/source/scale and applies one of:

| Primitive | Algebraic operation |
|---|---|
| `SCALE_ACCUM` | segment-wise `dst = a*dst + b*src` |
| `DOT_REDUCE` | segmented multiply and dot reduction |
| `OUTER_UPDATE` | rank-1 destination update |

The decoder expands a finite static loop over the configured shape. It does not
choose an algorithm at runtime. Mamba-2 and KDA are compiler-generated sequences
of these generic operations.

Viewed state transfer is an addressing form of existing `H_PREFETCH_V` and
`H_STORE_V`, not another opcode. Existing Vector FMA is an encoding mode of
`V_MUL_VF`, not another opcode.

Two other physical opcodes are required for complete static coefficient
preparation, independently of Matrix layout:

| Opcode | Purpose |
|---|---|
| `V_SOFTPLUS_V=0x3D` | Mamba/KDA elementwise coefficient preparation |
| `S_MAP_FP_V=0x3E` | inverse whole-row Vector-to-FP-register-file transfer |

Therefore the narrow Mamba/KDA functional path requires three physical
opcodes (`0x3D..0x3F`), while Matrix L-Compute itself owns only `0x3F`.

The older `L_CFG` form at `0x3F/funct1=0` remains executable only to reproduce
the historical Vector-stream experiment. It is not part of this RTL freeze and
official `L_TILE` recurrence schedules must not emit it.

### 2.1 Full opcode delta against main

The narrow functional count above is not the branch-merge count. Relative to
Compiler main `d89ad59`, this branch contains all seven previously free 6-bit
opcode values:

| Opcode | Mnemonic | Introducing commit | Used by recurrence lowering |
|---|---|---|---|
| `0x39` | `C_ROUTE_BEGIN` | `aec6dcb` | No |
| `0x3A` | `C_ROUTE_LOOP_START` | `aec6dcb` | No |
| `0x3B` | `C_ROUTE_LOOP_END` | `aec6dcb` | No |
| `0x3C` | `V_ROUTE_MUL` | `aec6dcb` | No |
| `0x3D` | `V_SOFTPLUS_V` | `56fd25a` | Yes |
| `0x3E` | `S_MAP_FP_V` | `56fd25a` | Yes |
| `0x3F` | `L_TILE` | `01f37ea` | Yes |

Matrix L-Compute adds one opcode (`0x3F`), the complete static recurrent path
uses three (`0x3D..0x3F`), and merging this branch as-is would add seven. The
four routed-MoE opcodes at `0x39..0x3C` are orthogonal to Matrix L-Compute and
are not emitted by `matrix_recurrence_lowering.py`. **They are excluded from
this Matrix L-Compute RTL handoff** and must be split into a routed-MoE branch
or a separately reviewed RTL phase before implementation.

Merging all seven values would reduce the free 6-bit opcode count from seven
to zero. Any later extension would therefore have to use a `funct1` subform or
an extended encoding. The handoff review must include the output of
`git diff main..HEAD -- doc/operation.svh`; the narrow three-opcode statement
must never be substituted for that complete branch diff.

## 3. Frozen physical mapping

For one logical bank word:

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

Read and write use the same equation. A cyclic restore returns bank words to
logical lane order. The official recurrence builder proves disjoint physical
view allocations. The direct-view projection entry points enforce the owned
scratch boundary in section 4.1. Bare CFG/store/EXEC emitters do not provide an
automatic lifetime proof for arbitrary caller-supplied addresses.

The final map deliberately removes an arbitrary programmable row coefficient.
The fair D' experiment showed that fixed diagonal wiring plus ordinary
per-tile base phases reaches the same zero-stall bank coordinates as the
programmable treatment for both official workloads. Keeping only a tile-phase
stride preserves the compact descriptor without carrying unjustified bank-wide
arithmetic into RTL.

## 4. Compiler behavior

Arlo's row-by-row static lowering remains the software fallback and B baseline.
It is not executed before L-Compute on the same tensor. Callers select an
ordinary lowering or an explicit L-Tile entry point. There is no general
automatic selector that rewrites unsupported L-Tile shapes to the ordinary
path:

```text
ordinary entry point       -> Arlo row-by-row Vector lowering
supported L-Tile entry     -> L_TILE multi-row lowering
unsupported L-Tile input  -> compile-time error; caller selects another path
```

The official schedules emit L-Tile recurrence programs for all 23 Nemotron
Mamba layers and all 69 Kimi KDA layers. Attention, MLA and MoE remain ordinary
PLENA paths and schedule markers in the formula timeline; they are not fake
L-Tile operations.

State is streamed through the same 1 MiB Matrix SRAM:

- Nemotron BF16 state: 1 MiB/layer, processed as 32-head groups;
- Kimi BF16 state: 3 MiB/layer, processed as 16-head groups.

There is no residency or overlap credit at the 1 MiB point. A second live group
does not fit after recurrence operands are included.

### 4.1 Direct-view projection and scratch ownership

`linear_projection(..., matrix_view_descriptor=...)` and
`vram_sub_projection_stream_k_accum_to(...)` implement a narrow producer ABI:

- The descriptor must describe one complete MLEN-wide consumer packet:
  `cols * tile_count == MLEN`, with `rows` equal to the live decode rows and
  `1 <= rows <= BLEN`. Column widths contain whole BLEN bank words.
- Its full physical-row footprint must fit inside **one reserved
  `MLEN * MLEN` scratch tile**, including gaps introduced by tile pitch. Fitting
  the overall Matrix SRAM alone is insufficient. A larger footprint is
  rejected before output allocation, view configuration or projection emit.
- An explicit `matrix_view_base` must be a nonnegative, tile-aligned base of a
  persistent MRAM reservation. `base + footprint` must fit both that owner and
  physical capacity; a transient weight allocation is not an owner because a
  weight reset releases it. Nonzero reserved bases are supported and tested.
- Weight K chunks use the real contiguous capacity after all reservations.
  The allocator preserves reservation addresses across resets and does not
  reclaim holes below its final reserved block. A required weight tile is
  checked before creating scratch; reservations cannot be freed as transient
  weights or confused with an existing allocation of the same name.
- The high-level direct-view projection accepts exactly **one physical output
  row/column tile**. Multiple output blocks, including extra physical padding
  blocks, are rejected because they would reuse the same view and overwrite
  earlier results. Streaming multiple K chunks into that one output packet
  remains supported. Multi-output producer lowering needs separate owned
  views or an explicit consumer/store before scratch reuse and is not yet
  implemented.

These constraints govern the producer helper; they do not reduce the generic
ISA descriptor's supported shapes or the standalone recurrence builder's
multi-tile allocations. Nor do they turn the returned legacy output tensor
into a general typed Matrix-view result with automatically tracked lifetime.

### 4.2 Official-shape recurrent wrappers

`ssm_decode_step_l_tile_v0` consumes **one request** and rejects
`Mamba2Shape.batch_size != 1` before allocating registers or emitting code.
The Mamba and KDA wrappers both require **MLEN=2048, BLEN=32** to match their
Matrix-SRAM point. The selected capacity must contain whole physical rows and
must not exceed the Compiler's configured SRAM capacity.

The Mamba wrapper retains its official `64 heads x 128 state rows x 64 values`
and `seq_len=1` contract; the KDA wrapper retains `96 x 128 x 128`. Both use
the standalone lowering's default state/field HBM ABI. They do not allocate
request-private HBM arenas and cannot be treated as a completed batched or
arbitrarily composable whole-model interface. The recorded B16 GPU baseline
and analytic timelines are separate evidence from this B1 numerical wrapper.

### 4.3 Projection packet-report scope

`matrix_packet_report` preserves the official full weight shape and HBM stride
for direct-view projection examples, but emits only the first output packet.
For the Nemotron and Kimi examples it records:

```text
emitted_output_columns = [0, 2048]
full_output_tile_count = 6
full_projection_emitted = false
```

Each contains 64 BLEN writeback fragments. The previous 384-fragment report
included six output blocks writing the same scratch and must not be used as
evidence of a valid full projection. Complete packet-extraction coverage means
coverage of the stated emitted fixture, not complete model numerical execution.

## 5. Demonstrated evidence

The previously archived connected gate supplies the following evidence; the
2026-09-05 Compiler regression result is recorded separately in section 9:

- canonical assembly and 32-bit decoding;
- loop-aware configuration dominance and fail-closed reserved encodings;
- physical `banks x rows x bank_width` storage, row/column reads and lane restore;
- single-packet Matrix projection writeback into a configured view with zero BF16 error;
- four consecutive official-geometry decode tokens for Nemotron Mamba-2 and
  Kimi KDA through Compiler -> assembler -> Rust -> HBM readback;
- final state and every token output compared, with maximum relative-L2 error
  0.0071; intermediate state snapshots are not part of the four-token count;
- zero phased-layout bank stalls for both official recurrences;
- a published 24-layer Mamba-2 checkpoint with every recurrence executed by
  Rust L-Tile and a continuous host-BF16 surrounding data path;
- synthetic transactional S128 prefill for Mamba-2 and KDA;
- official 52/93-layer formula timelines and real Nemotron routing replay;
- no ordinary Attention/MLA/MoE row/column service regression in the model.

The following **historical, pre-fix 2026-09-04** B1 table used the paper point
(`MLEN=2048`, `BLEN=32`, 64 banks, 1 MiB BF16 Matrix SRAM, 1560 HBM bytes/cycle).
It has not been regenerated here for the corrected Vector-cycle accounting and
final-normalization work. Use the newly archived Simulator campaign for current
timing claims. The Kimi weight-density label is corrected to MXFP4:

| Model | Weight density | Endpoint | A | B | C | D | D/A | D/B | D/C |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Nemotron 3 | mixed NVFP4 | strict serial | 4,055,091 | 3,110,067 | 2,192,850 | 2,014,094 | 2.0134x | 1.5442x | 1.0888x |
| Nemotron 3 | mixed NVFP4 | ideal overlap | 2,127,686 | 1,876,583 | 1,876,583 | 1,876,583 | 1.1338x | 1.0000x | 1.0000x |
| Nemotron 3 | uniform BF16 | strict serial | 6,360,486 | 5,415,462 | 4,498,245 | 4,319,489 | 1.4725x | 1.2537x | 1.0414x |
| Nemotron 3 | uniform BF16 | ideal overlap | 4,181,978 | 4,181,978 | 4,181,978 | 4,181,978 | 1.0000x | 1.0000x | 1.0000x |
| Kimi K3 | mixed MXFP4 | strict serial | 103,816,704 | 97,013,856 | 93,124,740 | 91,173,903 | 1.1387x | 1.0641x | 1.0214x |
| Kimi K3 | mixed MXFP4 | ideal overlap | 88,142,659 | 88,142,659 | 88,420,867 | 88,142,590 | 1.0000x | 1.0000x | 1.0032x |
| Kimi K3 | uniform BF16 | strict serial | 149,593,151 | 142,790,303 | 138,901,187 | 136,950,350 | 1.0923x | 1.0426x | 1.0142x |
| Kimi K3 | uniform BF16 | ideal overlap | 133,919,106 | 133,919,106 | 134,197,314 | 133,919,037 | 1.0000x | 1.0000x | 1.0021x |

Strict serial is `HBM + Matrix + Vector + L-Compute`, the dependency-safe
endpoint evaluated in that historical table. Ideal overlap is
`max(HBM, Matrix, Vector + L-Compute)`; it assumes complete resource overlap
and ignores dependencies, SRAM capacity and arbitration. It is a lower bound,
not a schedule emitted by the Compiler.

Matrix cycles are identical across the five variants. HBM cycles are also
identical for the Nemotron rows above, but **not universally**: Kimi C incurs
an intermediate fixed-layout spill, and C/D use exact state-DMA accounting.
For example, mixed Kimi B1 has 88,420,867 HBM cycles in C versus 88,142,590
in D. Consequently the base Kimi ideal `D/C` is 1.0032x rather than 1.0000x.
This modeled exception disproves the stronger all-variants-equal premise.

A and B charge one cycle per dynamically issued recurrence instruction while
setting Matrix service and Vector arithmetic costs to zero. C, D and E use the
service model from actual lowering. `D/A` and `D/B` therefore compare two
different evidence classes. These ratios combine multi-row execution,
issue/descriptor compression and, where present, spill changes; they are not
silicon speedups or programmable-skew speedups.

### 5.1 Historical agentic workload envelope (2026-09-04)

The following pre-fix medians cover 93 length-sorted, disjoint workload groups, each
with 32 decode steps and strict replay of measured Nemotron eager-routing
expert unions. `N` is the number of groups. P95 values for B4, B8 and B16 are
exploratory because `N < 20`.

| B | N | D/A ideal | D/C ideal | D/B ideal | D/C serial | D/B serial |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 1.1361x | 1.0000x | 1.0000x | 1.0890x | 1.5456x |
| 2 | 24 | 1.8423x | 1.0000x | 1.0239x | 1.1385x | 1.8490x |
| 4 | 12 | 2.8151x | 1.0000x | 1.5642x | 1.2003x | 2.2282x |
| 8 | 6 | 4.0843x | 1.0000x | 2.2699x | 1.2717x | 2.6661x |
| 16 | 3 | 5.8904x | 1.0000x | 3.2743x | 1.3579x | 3.1943x |

| B | N | mixed NVFP4 D/B serial / ideal | MXFP8 D/B serial / ideal | BF16 D/B serial / ideal |
|---:|---:|---:|---:|---:|
| 1 | 48 | 1.546 / 1.000 | 1.485 / 1.000 | 1.254 / 1.000 |
| 2 | 24 | 1.849 / 1.024 | 1.697 / 1.000 | 1.371 / 1.000 |
| 4 | 12 | 2.228 / 1.564 | 1.945 / 1.154 | 1.515 / 1.000 |
| 8 | 6 | 2.666 / 2.270 | 2.236 / 1.576 | 1.692 / 1.000 |
| 16 | 3 | 3.194 / 3.274 | 2.628 / 2.211 | 1.950 / 1.165 |

The agentic `D/C ideal = 1.0000x` result is a Nemotron result: routing changes
the MoE timeline but not recurrent-state packet coordinates. It must not be
generalized to Kimi C, whose fixed-layout spill changes its HBM endpoint.

## 6. Fair bank conclusion

The fair bank-only comparison is D versus D':

- D uses one compact descriptor with `tile_phase_stride`;
- D' uses fixed diagonal wiring and one ordinary compiler-selected base phase
  per tile;
- both map every official state word to the same physical cell;
- both reach zero bank stalls;
- D/D' pure bank-service speedup is 1.00x for Nemotron and Kimi.

The defensible architecture claim is therefore:

> A compiler-phased diagonal Matrix view makes full multi-row Mamba/KDA
> recurrence conflict-free and lets one deterministic L-Tile operation replace
> row-by-row issue. Programmable row skew is not required.

## 7. Pre-RTL resource contract

These are structural bounds, not PPA:

| Item | Frozen requirement |
|---|---:|
| New SRAM payload | 0 bytes |
| Cache/tag/replacement state | 0 bits |
| New MAC lanes | 0 |
| Extra Matrix bank ports | 0 |
| View records | 4 x 64 bits |
| Programmable row-skew arithmetic | 0 |
| Tile-phase generation | one 6-bit accumulator per active view |
| Sequencer | three bounded loop counters plus primitive/axis state |
| Segment broadcast | up to 32 BF16 scalars at MLEN=2048 |
| Lane restore | cyclic bank-word rotation, not an arbitrary crossbar |

The bank word remains 512 bits (32 BF16 values), matching the evaluated
reference port. No wider port is credited.

## 8. Not demonstrated

This freeze does not claim:

- all-operation, real-weight Nemotron 52-layer or Kimi 93-layer Rust execution;
- transactional full-model prefill or TTFT;
- real producer/consumer overlap at the one-MiB point;
- PPA, maximum frequency, power, Token/J or speedup over a GPU;
- an executable schedule that reaches the ideal resource-overlap lower bound;
  in the historical agentic envelope `D/C ideal` is 1.0000x at every batch and B1
  `D/B ideal` is 1.000x, so strict-serial issue savings cannot be presented as
  a guaranteed gain on a fully overlapping machine;
- weight/dequantization behavior beyond the stated traffic model: mixed
  NVFP4 uses about 0.5625 byte/value including one FP8 block scale per 16
  values, excludes dequantization compute, tensor-global scales and physical
  padding, while Matrix-SRAM elements remain BF16. In the historical agentic B16 row,
  changing weights from mixed NVFP4 to uniform BF16 changes D/B from
  3.194/3.274 (serial/ideal) to 1.950/1.165.

Those are next-phase validation items. None may be inferred from formula cycles.

## 9. Handoff gate

Before starting RTL, the following command must exit zero against the exact
Compiler commit used by the Simulator:

```bash
nix develop --no-write-lock-file --command \
  just test-matrix-lcompute /absolute/path/to/PLENA_Compiler
```

The handoff must archive the two commit IDs, the gate exit code, test counts,
the connected recurrence summary, the generated campaign hashes, and the
output of `git diff main..HEAD -- doc/operation.svh`.

The **2026-09-05 Compiler** verification ran the 17 files in the canonical
Hybrid L-Compute CI step: **211 passed in 179.47 seconds, exit 0** (188 existing
cases plus 23 new regression cases). It covers footprint/ownership rejection,
nonzero bases, wide-output rejection without partial emit, B1/geometry guards,
and the packet-report scope. The valid B1 Mamba wrapper's assembly SHA256
remains `cf86e2b89e86c72b1e0884317e901f85a1cae2f605aa3d0ef20ff2d2e56962cd`.
This Compiler-only result must not be substituted for the final connected gate.

The following count record is **historical**, from before the 2026-09-05 fixes:

- Simulator Python: 108 passed;
- Compiler: 188 passed;
- Rust workspace: 298 passed across 7 unit-test executables and 6 empty doctest
  suites, including 180 in the `transactional_emulator` unit-test executable;
- full gate: exit 0;
- historical Compiler mechanism commit: `c2e7d03e14b4c43350fd3d232cb2ee6058a494c4`.

The reviewed pre-fix Compiler HEAD was
`330e93da425eee107a0f3299f5f039fad1d74cd4`; it is the implementation baseline,
not the revision containing these corrections. Record the corrected Compiler
and Simulator commit IDs after the final connected gate and submodule pin.

Historical machine-readable evidence (regenerate before a new result freeze):

```text
01a8965c58c9203c05272edab50459b64fe66fb5f4340166d57218c6d5b180c6  artifacts/matrix_lcompute_connected_bf16/summary.json
cfd26f07ce7c81b36f11532c31bd6435f8e8d24a138029fba7ab467bd60dd6c1  artifacts/matrix_lcompute_e2e_v5/campaign.json
2ee3fa0d15f65e276f71b6763c5b55de078efef816be81ddc7f143242f135aed  artifacts/matrix_lcompute_e2e_v5/headline.csv
3f0f015c2dc420b3ee13c827a61b086ec78904a5005efa72e6a303864af7534b  artifacts/matrix_lcompute_agentic_v1/campaign.json
11c549ad31da440fe8973af98eca5e2234b4d99bdb4a061cd27a019e5bab41c5  artifacts/matrix_lcompute_agentic_v1/summary.csv
```

## 2026-09-05: executable precision and recurrence controls

The main weight-traffic experiment is the measured Nemotron checkpoint's
NVFP4/BF16 exclusion policy. Recurrent state and prepared coefficients remain
BF16; reproducing the original paper's precision point is not a goal of this
revision. Simulator `PrecisionContract` separately declares W/A/KV/state and
uses MX8 block8 only for the weight sensitivity table.

`lower_matrix_recurrence(..., snapshot_hbm_base=...)` emits explicit diagnostic
HBM copies of each token's completed state. The snapshot range must be aligned
and disjoint from live state/fields. Snapshot DMA is diagnostic overhead and
must not enter performance comparisons.

`prepared_vector_recurrence.py` supplies new executable packed VV controls:
explicit addressing (A) and static address reuse (B), using only existing
ordinary Vector operations. These are not the historical analytic original/Arlo
instruction census. Their prepared scalar values are expanded across HBM lanes;
L_TILE keeps compact coefficient packets. Each ordinary VV operation rounds to
BF16; L_TILE has local FP32 reduction. The Simulator reports exact checks against
each operation sequence and a separate shared 1% relative-L2 budget. A failed
shared budget disqualifies a performance claim.

Ordinary DMA selector **2 now means State**, matching `STATE_PRECISION_SELECTOR`.
The previous Simulator decoder interpreted it as KV even though this Compiler
used it for independent BF16 state. Canonical Activation=0 and KV=1 are unchanged;
old binaries deliberately using selector 2 as a KV alias must be reassembled
with selector 1. Viewed DMA already decoded selector 2 correctly. No opcode or
SRAM capacity is added by this correction.
