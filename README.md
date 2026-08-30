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

For recurrent decay and rank-one updates, packet width is independent from the
semantic state-row width. At the PLENA paper's `VLEN=2048` point, Nemotron
retains 64-element Mamba rows and Kimi retains natural 128-element KDA rows;
one existing Vector operation combines their 64-element bank-word atoms into
a 2048-element segmented-scalar packet. Cross-row reductions stay on the
ordinary fallback.

Affine packets use a packet-aligned physical base. The Compiler therefore
packs the 32 bank words of one packet into one physical SRAM row instead of
leaving 32 mostly empty VLEN rows; unaligned compact packet bases are rejected
rather than silently aliasing another tensor.

This distinction keeps the comparison fair. Reusing the old 64-wide KDA
lowering would split every natural KDA row in half and overstate the new path.
With exact rows, one official-size Mamba recurrence compiles from 92,399
baseline dynamic instructions to 19,049 affine-packet instructions; one KDA
mixer compiles from 215,387 to 61,115. These are issue counts, not full-model
cycles.

The official manifests are pinned to 52 Nemotron layers (23 Mamba, 23 MoE,
6 GQA) and 93 Kimi layers (69 KDA, 24 MLA). The checked report uses their real
dimensions, but projection weights remain symbolic; this is a performance and
code-generation result, not a real-checkpoint end-to-end numerical claim.

Run the reproducible Compiler report:

```bash
PYTHONPATH=.. python -m compiler.aten.plena.hybrid_compile_report \
  --model-lib doc/Model_Lib \
  --packet-elements 2048 --storage-atom 64 \
  --banks 32 --bank-width 64 --blen 32 \
  --mamba-row-elements 64 --kda-row-elements 128 \
  --json-out /tmp/hybrid-compiler-report.json
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
