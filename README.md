# PLENA Compiler

## MoE code organization

- `aten/plena/program_routed_moe.py` contains reusable routed-MoE lowering
  helpers: router logits, V_TOPK selection, dynamic expert-weight addressing,
  routed gather/scatter, expert activation, and combine.
- `aten/models/gpt_oss/` contains GPT-OSS-specific reference semantics and
  real-checkpoint loading utilities used to validate that substrate.
- ISA, assembler, and hardware documentation remain in `assembler/` and `doc/`.

## Recurrent compilation

Mamba-2, KDA and Matrix-SRAM recurrence support is described in
[recurrent compute](doc/recurrent_compute.md), including ISA, precision and
execution boundaries.
