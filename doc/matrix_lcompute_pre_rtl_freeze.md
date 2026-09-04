# Matrix SRAM L-Compute: Pre-RTL Freeze

This document is the canonical handoff for the Matrix-SRAM L-Compute branch.
It freezes only behavior supported by Compiler and Simulator evidence. RTL,
synthesis, frequency, area, power and energy are outside this revision.

## 1. Frozen architectural boundary

The final candidate reuses PLENA's existing Matrix SRAM and Vector arithmetic:

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
                  existing Vector mul/add/reduce
                                  |
                Matrix SRAM -- explicit DMA --> HBM
```

The Compiler owns every address, lifetime, load and store. The design contains:

- no cache, tags, replacement policy or hit/miss behavior;
- no private recurrent-state SRAM;
- no `X_STATE`, model-specific step instruction or HBM descriptor fetch;
- no runtime queue, work scheduler or completion record;
- no new MAC lane.

Storage is BF16 for Matrix SRAM, prepared fields, state and output. Multiply and
reduction accumulation remains FP32 inside the existing arithmetic datapath.

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

Therefore the complete Mamba/KDA functional ISA delta is three physical
opcodes (`0x3D..0x3F`), while Matrix L-Compute itself owns only `0x3F`.

The older `L_CFG` form at `0x3F/funct1=0` remains executable only to reproduce
the historical Vector-stream experiment. It is not part of this RTL freeze and
official `L_TILE` recurrence schedules must not emit it.

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
logical lane order. The Compiler rejects aliasing and out-of-capacity views.

The final map deliberately removes an arbitrary programmable row coefficient.
The fair D' experiment showed that fixed diagonal wiring plus ordinary
per-tile base phases reaches the same zero-stall bank coordinates as the
programmable treatment for both official workloads. Keeping only a tile-phase
stride preserves the compact descriptor without carrying unjustified bank-wide
arithmetic into RTL.

## 4. Compiler behavior

Arlo's row-by-row static lowering remains the software fallback and B baseline.
It is not executed before L-Compute on the same tensor. For each recurrent
region the Compiler chooses exactly one path:

```text
unsupported/small/tail shape -> Arlo row-by-row Vector lowering
supported regular view       -> L_TILE multi-row lowering
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

## 5. Demonstrated evidence

The complete gate currently proves:

- canonical assembly and 32-bit decoding;
- loop-aware configuration dominance and fail-closed reserved encodings;
- physical `banks x rows x bank_width` storage, row/column reads and lane restore;
- direct Matrix projection writeback into a configured view with zero BF16 error;
- four consecutive official-geometry decode tokens for Nemotron Mamba-2 and
  Kimi KDA through Compiler -> assembler -> Rust -> HBM readback;
- complete state and output comparisons, with maximum relative-L2 error 0.0071;
- zero phased-layout bank stalls for both official recurrences;
- a published 24-layer Mamba-2 checkpoint with every recurrence executed by
  Rust L-Tile and a continuous host-BF16 surrounding data path;
- synthetic transactional S128 prefill for Mamba-2 and KDA;
- official 52/93-layer formula timelines and real Nemotron routing replay;
- no ordinary Attention/MLA/MoE row/column service regression in the model.

At the paper point (`MLEN=2048`, `BLEN=32`, 64 banks, 1 MiB BF16 Matrix SRAM,
1560 HBM bytes/cycle), the formula-based B1 decode timeline is:

| Model | Original A | Arlo B | L-Tile D | D/A | D/B |
|---|---:|---:|---:|---:|---:|
| Nemotron 3 | 4,055,091 | 3,110,067 | 2,014,094 | 2.0134x | 1.5442x |
| Kimi K3 | 103,816,704 | 97,013,856 | 91,173,903 | 1.1387x | 1.0641x |

These ratios combine multi-row execution and issue/descriptor compression.
They are not silicon speedups and not programmable-skew speedups.

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
- PPA, maximum frequency, power, Token/J or speedup over a GPU.

Those are next-phase validation items. None may be inferred from formula cycles.

## 9. Handoff gate

Before starting RTL, the following command must exit zero against the exact
Compiler commit used by the Simulator:

```bash
nix develop --no-write-lock-file --command \
  just test-matrix-lcompute /absolute/path/to/PLENA_Compiler
```

The handoff must archive the two commit IDs, the gate exit code, test counts,
the connected recurrence summary and the generated campaign hashes.
