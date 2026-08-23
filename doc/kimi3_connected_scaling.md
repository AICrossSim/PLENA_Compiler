# Kimi K3 Connected Lowering: Current Boundary

The connected builder now emits and assembles all 93 Kimi K3 layers for one
decode token: 69 KDA mixers, 24 MLA mixers, 92 LatentMoE blocks, one dense FFN,
and the AttnRes paths between them.

## What changed

The old emitter repeated the full output-column body for every wide Matrix
projection. A one-head diagnostic produced 100,221,916 instructions, 3.74 GB of
assembly, took 7m10s, and peaked at 24.1 GiB RSS.

The compact lowering keeps K chunks explicit but moves the repeated output-tile
and micro-column traversal into hardware `C_LOOP`s. The assembler also streams
instructions directly to the `.mem` file instead of retaining a second list of
millions of parsed instructions.

The full 96-head result is:

| Metric | Result |
|---|---:|
| Layers | 69 KDA + 24 MLA |
| Instructions | 11,662,716 |
| Raw 32-bit machine code | 44.490 MiB |
| Assembly text | 292.29 MB |
| Build + streaming assembly | about 2m10s |
| Peak host RSS | about 4.7 GiB |
| Symbolic HBM bindings | 2,713 |
| Symbolic HBM address span | 3.21 TB (2.92 TiB) |

Every instruction is encoded as one legal 32-bit word. The artifact builder
also rejects duplicate, overlapping, or out-of-range parameter regions.

## Numerical evidence

Two compact Matrix paths execute in the Rust transactional emulator:

- MXFP8 `1x320 @ 320x384`, two K chunks and six N tiles: 384/384 exact.
- Plain-BF16 stream-K `1x320 @ 320x128`, five K tiles: 128/128 exact.

The second test found and fixed a real address bug: the Matrix micro-column
offset must advance by `BLEN*MLEN` elements, not `BLEN` elements. Assembly-only
validation could not detect that error.

The S128 campaign found a second boundary bug: an 80-row MLA history was backed
by 80 physical rows even though attention reads 64-row Matrix tiles. History
scratch is now rounded to 128 rows, and a five-chunk test pins both the shifted
causal diagonals and the physical K/V allocation.

The matching Simulator branch also runs a compact whole-backbone program in one
Rust invocation: 69 KDA, 24 MLA, 92 LatentMoE, and one dense FFN. The base S16
prefill plus four-token decode case executes 4,646,465 machine instructions in
80,522,239 simulator cycles, and all 3,740 hidden/residual checkpoints match the
CPU reference. The long S16 plus D128 gate executes 66,016,808 instructions in
1,492,322,041 cycles with 100% allclose and 0.2163% relative-L2 error. All 69
KDA state lifetimes and 24 compressed MLA caches pass; the persistent HBM
manifest contains no expanded all-head K/V object. Recomputing each cache from
the actual Rust producer hidden is bit exact, so cache correctness is checked
independently from accumulated BF16 whole-model drift.

## What this does not prove

The real-shape artifact still uses symbolic HBM ranges; no Kimi checkpoint has
been packed or executed. Its 3.21 TB address span requires a sparse or streamed
checkpoint backend rather than a dense Rust HBM image. Multi-token cache/prefill
and 93-layer replay are implemented only in compact synthetic fixtures, so their
cycles are not a Kimi performance estimate. Checkpoint packing, real-width whole-model replay,
instruction-memory provisioning, RTL timing, and PPA remain future work. The 96
MLA head bodies are still emitted statically; looping them would reduce code
size but is no longer a blocker for producing the bounded machine-code artifact.
