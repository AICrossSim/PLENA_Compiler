# KDA without additional datapath hardware

Status: executable compiler-only prototype, not a proof of model quality or
optimal end-to-end performance. The reference hardware point is the existing
BF16-capable Vector configuration (VLEN=2048, 64 SRAM rows/256 KiB). No RTL,
opcode, SRAM capacity/port, lane accumulator, queue, or cache is added.
This does not claim that an arbitrary original-paper FP12/FP16 netlist supports
BF16 without configuration verification.

## Selected implementation

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

## Why this is the first choice

It fixes the numerical failure of the tested ordinary sequential reduction
without changing the arithmetic hardware or adding FP32 feedback storage.
The128-term binary sum already needs127 additions; the implementation reaches
that count without copy adds. This is an economical starting point, not a
claim of global optimality. Compensated sums need more ALU work and SRAM traffic.

The original Matrix unit remains a candidate for dots, but current M_MV consumes
an MLEN2048 tile: a fully padded BF16 tile needs8 MiB, beyond the1 MiB Matrix
SRAM point. Compact-view support is itself an extension. KDA has private state
per head, whereas M_BMV does not simply provide independent grouped GEMV.
Moreover, Rust f32 host accumulators are not evidence of original RTL BF16×BF16
FP32 hardware. Input format conversion, instruction support, capacity and
utilization must be established before selecting a Matrix mapping.

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
reproduction; the recommended no-extra-hardware path is this explicit flag.
