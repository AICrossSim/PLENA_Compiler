# PLENA Compiler

## MoE code organization

- `aten/plena/program_routed_moe.py` contains reusable routed-MoE lowering
  helpers: router logits, V_TOPK selection, dynamic expert-weight addressing,
  routed gather/scatter, expert activation, and combine.
- `aten/models/gpt_oss/` contains GPT-OSS-specific reference semantics and
  real-checkpoint loading utilities used to validate that substrate.
- ISA, assembler, and hardware documentation remain in `assembler/` and `doc/`.


## Heterogeneous MoE dispatch research

This branch includes an isolated, reproducible [MoE dispatch research prototype](research/moe_dispatch/README.md). It compares 6 / 3+3 / 4+2 cores with explicit private-memory plans and analytical execution; it does not replace the production ISA or native HBM backend.
