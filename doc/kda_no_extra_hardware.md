# KDA without additional datapath hardware

> **Draft review only — not for merge.** This is an optional numerical control,
> not the selected architecture or a replacement for original PLENA mapping.
> See [review scope](REVIEW_SCOPE_20260905.md).

Status: executable compiler-only prototype, not a proof of model quality or
optimal end-to-end performance. The reference hardware point is the existing
BF16-capable Vector configuration (VLEN=2048, 64 SRAM rows/256 KiB). No RTL,
opcode, SRAM capacity/port, lane accumulator, queue, or cache is added.
This does not claim that an arbitrary original-paper FP12/FP16 netlist supports
BF16 without configuration verification.

## Optional implementation

Keep weights on the ordinary Matrix projection/MoE path. Recurrent state and
prepared coefficients remain BF16. Compiler packs 16 heads ×128 value lanes
into a VLEN row, uses existing H_PREFETCH_V/H_STORE_V and ordinary Vector ALU
instructions, and changes the two KDA dot reductions to a balanced BF16 tree.

For 128 products, form adjacent pairs, pairs of pairs, and so on. Products and
every addition round to BF16. The compiler schedules the tree as inputs arrive;
no hardware tree controller or runtime work queue is needed. A term with index
`i` merges the partial levels corresponding to trailing one bits of `i`.
Even terms write directly to level0; odd terms use temporary row5. Each final
merge writes its destination level directly, so no copy adds are required.

| Resource | Allocation / work |
|---|---|
| Existing working rows | 0..7 |
| Existing SRAM partial-sum rows | 8..14 |
| Total reserved Vector SRAM | 15 rows =60 KiB, within existing256 KiB |
| Incremental allocation in existing SRAM | 7 rows =28 KiB; not added capacity |
| New persistent FP32 state | 0 |
| New instruction encodings / RTL modifications | 0 |
| Per dot | 128 BF16 MUL +127 BF16 ADD |
| Rounding-path depth | 7 additions, not7 hardware cycles |

The output and state-update boundaries otherwise remain those of ordinary
Vector instructions. The independent oracle builds the whole tree tensor;
the generated program streams through seven SRAM partial rows. Their different
implementations must produce bitwise-identical outputs and states.

## What this experiment establishes

It passes the standard tested inputs on which ordinary sequential reduction
failed the shared error budget, using only ordinary BF16 instructions. Stronger
synthetic stress still fails; this is not a general numerical fix. The 128-term
binary sum needs 127 additions, which the implementation reaches without copy
adds. It does not establish better end-to-end performance or model quality.

Original PLENA already includes Matrix accumulators; the presence of FP32 host
temporaries does not establish a need to add new accumulators. The original
Matrix unit remains a candidate for these dots. Current Rust reads a
full MLEN-square tile (8 MiB in BF16), but this is not a hardware capacity lower
bound: local RTL M_TMM reads BLEN contiguous MLEN-wide rows. A transposed
32x2048 panel occupies 128 KiB. Packing 16 private heads along K and zero-masking
16 activation rows can express the two KDA dots with 48 M_TMM instructions per
token across all 96 heads. This is a mapping candidate, not an executed result.

The current fixed Matrix prefetch length is MLEN, and Rust clamps shorter
prefetches to MLEN; a panel path therefore needs a validated static configuration
and simulator read-granularity correction. M_MM_WO writes 32 full VLEN rows:
activation plus a separate output buffer plus the 15-row recurrence scratch
would exceed 256 KiB. Reusing consumed activation storage for output could fit,
but requires a checked lifetime schedule and extra activation reloads. Padding,
packing, quantization and DMA cannot be free. Rust f32 host accumulators are not
evidence of original RTL BF16-by-BF16 FP32 hardware. The MX input and accumulator
formats must be established before selecting this mapping; changing a static
RTL parameter is also different from keeping a frozen netlist unchanged.

## Qualification boundaries

There are two separate tests:

1. Execution contract: exact match to the specified BF16 instruction DAG,
   including every product, sum, state update and store boundary.
2. Algorithm/model quality: compare with the appropriate training/inference
   reference on actual layer operands, long-memory states and model outputs.

Passing the first or matching the old L_TILE reference does not prove the
second. Original deterministic decay0.84..0.96 forgets quickly; a512-token run
is not automatically a difficult long-memory test. Synthetic near-unit-decay
stress already produces failures even for the experimental FP32-dot extension.
Do not loosen error thresholds merely to publish a speedup.

L_TILE.DOT_REDUCE also keeps per-lane FP32 state in the functional model and
its other primitives fuse rounding boundaries. Its physical resource reuse is
unproven. Strict zero-hardware scope excludes L_TILE/view-control/routing changes.
If control compression is studied later, both treatments must execute the same
BF16 arithmetic DAG; descriptors, muxes and sequencers must be counted as added
logic rather than described as zero hardware.

## Reproduction

From the Simulator repository with this Compiler selected:

```bash
PLENA_COMPILER_ROOT=/path/to/Compiler .venv/bin/python -m transactional_emulator.testbench.aten.matrix_lcompute_execution_compare --model kda --variants A B --batches 1 --tokens 2 --pairwise-bf16-dot --keep-build --output-dir /path/to/new-results
```

For request isolation and every intermediate state, use `--variants B --batches
2 --tokens 4 --seed 17 --snapshot-states`. Diagnostic
snapshots add real DMA instructions and do not qualify performance ratios.
`--pairwise-bf16-dot` rejects D and the experimental FP32 flag. The generated
instruction whitelist is checked before assembly. SRAM capacity is checked by
the lowering. Old/default and FP32-experimental paths remain for historical
reproduction. Both experimental flags default to false; enabling this explicit
flag selects a test control, not an accepted architectural change.
