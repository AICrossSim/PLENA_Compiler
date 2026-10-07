# Compiler round2 merge log

Frozen source HEAD: `480d558c72f8fce19572b2408eb02a1ae52c40eb`. No rebase, squash, branch deletion, or worktree deletion.


## Merge `feat/routed-moe-compiler-substrate` (`54a68e6969651eb5c1c3397ac59bed6585c08806`)

Three-stage conflicting blobs preserved under `research/moe_dispatch/archive/feat/routed-moe-compiler-substrate/conflicts`. Non-conflicting changes merged normally.

- `README.md`: Retain current v3 overview; full incoming README is preserved in archive.
- `assembler/tests/test_vector_rmask_handling.py`: Retain stronger non-zero-mask encoding assertions.
- `aten/models/gpt_oss/real_layer_utils.py`: Retain explicit HuggingFace vs PLENA router-weight orientation documentation.
- `aten/plena/compiler.py`: Retain current shared-expert mixin; incoming July substrate predates shared expert support.
- `aten/plena/isa_attention.py`: Retain inline normalization and attention-sink denominator handling; incoming removes later correctness fixes.
- `aten/plena/isa_compiler.py`: Retain row-aligned HBM sizing via hbm_tensor_size; incoming unaligned size formula is older.
- `aten/plena/isa_matrix.py`: Retain corrected mat_col_stride=blen*mlen; incoming bare blen repeats MRAM output columns.
- `aten/plena/program_attention.py`: Retain partial Q/KV dimensions, unrolled pack and inline normalization fixes.
- `aten/plena/program_routed_moe.py`: Retain generalized moe_* API, deprecated aliases, sticky stage terminator, runtime TOPK policy and K-split handling already superseding the early gpt_oss wrappers.
- `aten/plena/program_tensors.py`: Retain row-aligned allocation and scale-row storage accounting.
- `aten/tests/test_gpt_oss_moe_reference.py`: Retain current Transformers router triple-result compatibility.
- `aten/tests/test_plena_compiler.py`: Retain added dynamic single-K and split-K regression tests.
- `doc/operation.svh`: Retain C_SET_TOPK_REG opcode added after initial V_TOPK substrate.
- `doc/plena_isa_spec.md`: Retain newer runtime TOPK/row-aligned ISA documentation; incoming version preserved in archive.

## Merge `codex/moe-e2e-sync` (`1add2606985e5962df3a0e7c2ea90b8b302a2ec8`)

All conflicting blobs archived under `research/moe_dispatch/archive/codex/moe-e2e-sync/conflicts`. Non-conflicting edits are retained.

- `aten/plena/program_routed_moe.py`: 2 conflict block(s), chose current. Retain newer sticky router stage markers and arbitrary runtime top-k policy; keep non-conflicting Qwen label and vector-format documentation updates.
