# PLENA Compiler

## Matrix SRAM recurrent sublayers

The maintained research branch is `feat/matrix-sram-lcompute` in both Compiler
and Simulator. It contains Mamba/KDA coefficient production, native recurrent
supply and bounded projection mapping. The Simulator pins its matching Compiler
commit; initialize that submodule for a reproducible pair.

- [Implementation, ISA changes and tests](doc/l_tile_projection.md)
- [Simulator progress, results and reproduction](https://github.com/AICrossSim/PLENA_Simulator/blob/feat/matrix-sram-lcompute/doc/l_tile_projection.md)

The current candidate uses BF16 state, FP32 update intermediates and a BF16 tree.
Projection replay/slicing/batching requires explicit additional interfaces and
finite storage. This branch does not establish integrated RTL/PPA, final model
quality or a speedup over the best unmodified PLENA implementation. The separate
`review/matrix-lcompute-20260905` Draft PR retains the older minimal review scope.

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
