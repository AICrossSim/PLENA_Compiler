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
| Instructions | 11,502,370 |
| Raw 32-bit machine code | 43.88 MiB |
| Assembly text | 292.29 MB |
| Build + streaming assembly | about 2m10s |
| Peak host RSS | about 4.7 GiB |
| Symbolic HBM bindings | 2,713 |

Every instruction is encoded as one legal 32-bit word. The artifact builder
also rejects duplicate, overlapping, or out-of-range parameter regions.

## Numerical evidence

Two compact Matrix paths execute in the Rust transactional emulator:

- MXFP8 `1x320 @ 320x384`, two K chunks and six N tiles: 384/384 exact.
- Plain-BF16 stream-K `1x320 @ 320x128`, five K tiles: 128/128 exact.

The second test found and fixed a real address bug: the Matrix micro-column
offset must advance by `BLEN*MLEN` elements, not `BLEN` elements. Assembly-only
validation could not detect that error.

The matching Simulator branch also runs a compact whole-backbone program in one
Rust invocation: 69 KDA, 24 MLA, 92 LatentMoE, and one dense FFN across S16
causal prefill plus four decode tokens. It executes 4,646,741 instructions in
80,526,139 simulator cycles, and all 3,740 hidden/residual checkpoints match the
CPU reference. All 69 KDA state lifetimes and 24 compressed MLA caches pass;
the persistent HBM manifest contains no expanded all-head K/V object.

## What this does not prove

The real-shape artifact still uses symbolic HBM ranges; no Kimi checkpoint has
been packed or executed. Multi-token cache/prefill and 93-layer replay are
implemented only in compact synthetic fixtures, so their cycles are not a Kimi
performance estimate. Checkpoint packing, real-width whole-model replay,
instruction-memory provisioning, RTL timing, and PPA remain future work. The 96
MLA head bodies are still emitted statically; looping them would reduce code
size but is no longer a blocker for producing the bounded machine-code artifact.
