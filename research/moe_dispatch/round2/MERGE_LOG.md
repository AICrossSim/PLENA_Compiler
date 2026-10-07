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
- `fix/portable-developer-paths` (`29fdbc17a8877d44bf54ad58bc2d538449a263c3`): merged with `--no-ff`, no conflicts.
- `agent/portable-compiler-tooling` (`1a26215f831c38e1bd92e83eab7e87f2a4ef47d0`): merged with `--no-ff`, no conflicts.

## Merge `local/archive-e2e-compiler-expert-20260727` (`ea9e8c7a1f4ae3a06ee8896ceaf3453977179840`)

All conflicting blobs archived under `research/moe_dispatch/archive/local/archive-e2e-compiler-expert-20260727/conflicts`. Non-conflicting edits are retained.

- `aten/plena/program_routed_moe.py`: 0 conflict block(s), chose current. Retain current stage-correct grouped route broadcasting and modern top-k APIs instead of incompatible older scalar broadcast refactor; older separated expert-FFN sources preserved for compatibility review. Retain HF-version compatibility comments in reference test.
- `aten/tests/test_gpt_oss_moe_reference.py`: 2 conflict block(s), chose current. Retain current stage-correct grouped route broadcasting and modern top-k APIs instead of incompatible older scalar broadcast refactor; older separated expert-FFN sources preserved for compatibility review. Retain HF-version compatibility comments in reference test.

## Merge `local/archive-old-topk-compiler-20260727` (`1468974f29828a7dea898943ca5fffaff4976ad4`)

All conflicting blobs archived under `research/moe_dispatch/archive/local/archive-old-topk-compiler-20260727/conflicts`. Non-conflicting edits are retained.

- `assembler/assembly_to_binary.py`: 1 conflict block(s), chose current. Retain C_SET_TOPK_REG opcode and masked-vector TOPK encoding; incoming archive predates programmable routing and would restore the obsolete funct/rstride encoding.
- `doc/operation.svh`: 1 conflict block(s), chose current. Retain C_SET_TOPK_REG opcode and masked-vector TOPK encoding; incoming archive predates programmable routing and would restore the obsolete funct/rstride encoding.

## Merge `local/candidate-compiler-topk-contract-20260726` (`71ba7ff4f1d697d5a444f85da6bc8e4484090772`)

All conflicting blobs archived under `research/moe_dispatch/archive/local/candidate-compiler-topk-contract-20260726/conflicts`. Non-conflicting edits are retained.

- `aten/plena/program_routed_moe.py`: 2 conflict block(s), chose current. Retain stage attribution and programmable TOPK policies supporting DeepSeek 64/top-6; old contract documentation lacks those later extensions.
- `doc/plena_isa_spec.md`: 1 conflict block(s), chose current. Retain stage attribution and programmable TOPK policies supporting DeepSeek 64/top-6; old contract documentation lacks those later extensions.

## Merge `local/compiler-expert-ffn-20260727` (`8cfb0d071524d18c5a3bc98d2ef498509e123389`)

All conflicting blobs archived under `research/moe_dispatch/archive/local/compiler-expert-ffn-20260727/conflicts`. Non-conflicting edits are retained.

- `aten/plena/program_routed_moe.py`: 0 conflict block(s), chose current. Keep newer stage-correct routed implementation and version-aware reference tests; preserve new historical standalone expert-FFN, scalar route-weight and dynamic-combine methods alongside it, adding required modern stage arguments without changing current methods.
- `aten/tests/test_gpt_oss_moe_reference.py`: 2 conflict block(s), chose current. Keep newer stage-correct routed implementation and version-aware reference tests; preserve new historical standalone expert-FFN, scalar route-weight and dynamic-combine methods alongside it, adding required modern stage arguments without changing current methods.
