# PLENA Compiler

## Review setup

The default environment excludes the optional TileLang/CUDA toolchain. A clean
CPU checkout can run the common-state Compiler guards with:

```bash
uv sync --frozen
uv run --frozen pytest -q -m "not slow" \
  assembler/tests/test_c_set_topk_reg_encoding.py \
  assembler/tests/test_l_scatter_m_encoding.py \
  assembler/tests/test_x_state_encoding.py \
  aten/tests/test_kda_scheduler.py \
  aten/tests/test_kimi_k3_full_program.py \
  aten/tests/test_kimi_k3_hybrid.py \
  aten/tests/test_layout_contract.py \
  aten/tests/test_mamba_scheduler.py \
  aten/tests/test_mla_attention.py \
  aten/tests/test_mram_binding_tracking.py \
  aten/tests/test_nemotron3_blocks.py \
  aten/tests/test_nemotron3_full_program.py \
  aten/tests/test_nemotron3_hybrid.py \
  aten/tests/test_projection_scatter.py \
  aten/tests/test_state_contract.py \
  aten/tests/test_state_isa_lowering.py \
  aten/tests/test_state_lowering.py \
  aten/tests/test_state_memory_image.py \
  aten/tests/test_state_residency.py
```

Use `uv sync --frozen --group tvm` only for TileLang/TVM development.

## MoE code organization

- `aten/plena/program_routed_moe.py` contains reusable routed-MoE lowering
  helpers: router logits, V_TOPK selection, dynamic expert-weight addressing,
  routed gather/scatter, expert activation, and combine.
- `aten/models/gpt_oss/` contains GPT-OSS-specific reference semantics and
  real-checkpoint loading utilities used to validate that substrate.
- ISA, assembler, and hardware documentation remain in `assembler/` and `doc/`.


## Heterogeneous MoE dispatch research

This branch includes an isolated, reproducible [MoE dispatch research prototype](research/moe_dispatch/README.md). It compares 6 / 3+3 / 4+2 cores with explicit private-memory plans and analytical execution; it does not replace the production ISA or native HBM backend.
