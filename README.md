# PLENA Compiler

## Qwen3-MoE decode boundary

The Qwen3-30B-A3B substrate pins the exact Transformers 5.5 routing order:
BF16 router logits, FP32 softmax/top-8/renormalization, FP32 route-score
storage and multiplication, then one BF16 cast of each weighted expert output.
`V_TOPK` writes the selected scores to the dedicated FP32 route SRAM and
`V_MUL_ROUTE_F32` consumes them. The emulator emits
`route_f32_sram_dump.bin` for provenance.

The 4,096-byte route SRAM holds 1,024 FP32 scores. The current lowering keeps
all top-8 scores for a batch resident, so batch size is capped at 128. Larger
batches fail before code emission; token-serial score reuse is not implemented.

The compiler can emit a hashed contract for all 48 ordered
attention-to-runtime-MoE layers. Each layer requires all 128 expert banks to
remain addressable and forbids dense fallback. This contract is an audit
artifact, not an executable full-model claim: the native frontend rejects the
target until complete layer transaction parity and full-layer lowering are
implemented. `V_ROUTER_LINEAR_BF16` reconstructs BF16 inputs and weights and
executes the installed libtorch `F.linear` operation with one BF16 logits cast.
Hidden-64 and target hidden-2048 fixtures pin all 128 output logits against
Transformers 5.5. Route-op timing is structural and uncalibrated, and the new
opcodes do not have validated RTL.

The content-addressed hidden-64 validation transaction now executes a complete
compiler-generated binary: attention RMSNorm, MXFP8 Q/K/V projections, Q/K
RMSNorm, RoPE, PackedKV cache append/read, q_len=1 attention, output projection,
and the exact routed-MoE tail. The harness compares every retained BF16 boundary,
FP32 route scores, runtime expert IDs, and final BF16 bytes with the compatible
PyTorch/Transformers-5.5 oracle. It also verifies eight-assignment conservation,
the exact post-append MX cache image, and that all untargeted HBM bytes remain
unchanged. Program, inputs, outputs, source identity, invocation, and op trace
are hash-bound; altered or partial evidence fails closed. Run
`transactional_emulator/testbench/aten/qwen3_moe_single_layer_transaction.py`
with the release emulator, or enable the gated pytest using
`PLENA_RUN_QWEN3_SINGLE_LAYER_TRANSACTION=1`.

This proof is deliberately scoped to batch-1, q_len=1, hidden/head-dim 64,
one KV head, intermediate 64, and cache position 3. It does not validate the
target hidden-2048/GQA geometry, all 48 layers, narrow expert-vector profiles,
timing, or RTL. Those full frontend, emulator-target, timing, RTL, and
publication flags therefore remain false.

## Decode-local LM-head boundary

`asm_templates.lm_head.local_lm_head_lowering_receipt` exposes the native
`(vocab, hidden)` weight layout, MLEN-padded physical rows and columns,
BLEN-padded batch rows, exact MX data/scale bytes, matrix event counts, and the
profile-ID-plus-MLEN numerical identity. The projection now initializes a
dedicated LM-head HBM base and the test stager preserves native row order while
zero-padding the physical tensor.

The serving path remains fail-closed. The current projection materializes all
logits and does not zero-fill padded activation rows, mask padded vocabulary
rows, stream running top-20/argmax state, or merge tensor-parallel candidates.
The required distributed order is local top-20 per rank followed by a global
score-descending/token-ID-ascending merge at the sample owner and a selected
`uint32` token broadcast. Until those operations, profile-aware staging, and
calibrated event timing exist, serving compiler, emulator parity, RTL, and
publication validity remain false. Full BF16 logits are permitted only for the
offline NLL evaluator.
