# PLENA Compiler

## Hybrid Mamba/KDA support

This branch adds a static, model-independent path for Nemotron 3 and Kimi K3.
It does not add a Mamba/KDA coprocessor, private state cache, or `X_STATE`.

The Compiler performs two independent optimizations:

1. It replaces regular pointer/scalar traffic inside existing hardware loops
   with `L_STREAM_CFG`. Existing Matrix/Vector opcodes still define the math,
   and `C_LOOP_START/END` still define repetition.
2. It evaluates row-major, transpose, consumer-major, and affine-skewed
   placements at the Matrix-writeback-to-output-SRAM boundary. A layout is
   selected only after checking bijection, producer writes, consumer reads,
   bank stalls, and lane-restore cost.

For recurrent decay and rank-one updates, one existing Vector operation can
consume a 64-value packet assembled from sixteen logical rows. The same
lowering supports Mamba-2 and KDA; cross-row reductions stay on the ordinary
row fallback. The Rust simulator executes these packets and shows that affine
placement removes their bank conflicts, while remaining slightly slower than
the best ordinary-row stream at the current 64-lane design point.

The official manifests are pinned to 52 Nemotron layers (23 Mamba, 23 MoE,
6 GQA) and 93 Kimi layers (69 KDA, 24 MLA). The checked report uses their real
dimensions, but projection weights remain symbolic; this is a performance and
code-generation result, not a real-checkpoint end-to-end numerical claim.

Run the reproducible Compiler report:

```bash
PYTHONPATH=.. python -m compiler.aten.plena.hybrid_compile_report \
  --model-lib doc/Model_Lib --json-out /tmp/hybrid-compiler-report.json
```

The report separates dynamic issue reduction from local SRAM service cycles.
Neither number is presented as a layer or full-model speedup. See
[`doc/hybrid_lcompute.md`](doc/hybrid_lcompute.md) for the ISA boundary,
fallback policy, validated status, and remaining gates.

## MoE code organization

- `aten/plena/program_routed_moe.py` contains reusable routed-MoE lowering
  helpers: router logits, V_TOPK selection, dynamic expert-weight addressing,
  routed gather/scatter, expert activation, and combine.
- `aten/models/gpt_oss/` contains GPT-OSS-specific reference semantics and
  real-checkpoint loading utilities used to validate that substrate.
- ISA, assembler, and hardware documentation remain in `assembler/` and `doc/`.
