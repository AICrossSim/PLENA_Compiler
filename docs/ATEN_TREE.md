# ATen Compiler Tree

This summarizes the current `aten/` package layout. Generated `__pycache__/`
directories are intentionally omitted.

```text
aten/
|-- __init__.py                 # Public ATen package exports
|-- native_ops.yaml             # Operator registry spec: signatures and dispatch targets
|-- isa_builder.py              # Typed ISA instruction/register builder and legalization
|-- model_extract.py            # HuggingFace model config/layer/embedding extraction helpers
|-- plena_frontend.py           # HF decoder model -> PLENA program -> ISA text
|-- e2e_runner.py               # HF model -> ATen compiler -> emulator -> golden check
|-- reference.py                # CPU golden/reference math and MXFP/BF16 helpers
|-- vram_stage_compare.py       # Debug tooling for VRAM stage comparisons
|
|-- ops/                        # ATen-style operator dispatch layer
|   |-- __init__.py             # User-facing ops.* dispatch functions
|   |-- registry.py             # Backend registry: CPU vs PLENA implementation lookup
|   |
|   |-- cpu/                    # PyTorch reference backend
|   |   |-- __init__.py
|   |   |-- attention_ops.py    # CPU flash attention reference
|   |   |-- conv_ops.py         # CPU conv reference
|   |   |-- embedding_ops.py    # CPU embedding/RoPE reference
|   |   |-- ffn_ops.py          # CPU FFN reference
|   |   |-- linear_ops.py       # CPU linear reference
|   |   |-- norm_ops.py         # CPU RMS/layer norm reference
|   |   +-- softmax_ops.py      # CPU softmax reference
|   |
|   +-- plena/                  # PLENA backend operator wrappers
|       |-- __init__.py
|       |-- attention_ops.py    # ops.flash_attention -> prog.flash_attention
|       |-- conv_ops.py         # conv lowering / PLENA conv codegen helper
|       |-- embedding_ops.py    # embedding_add / rope wrappers
|       |-- ffn_ops.py          # ops.ffn -> prog.ffn
|       |-- linear_ops.py       # ops.linear -> prog.linear_projection
|       |-- norm_ops.py         # rms_norm / layer_norm wrappers
|       +-- softmax_ops.py      # PLENA softmax lowering
|
|-- plena/                      # Canonical PLENA compiler implementation package
|   |-- __init__.py             # Canonical exports: PlenaCompiler, IsaCompiler, vars, constants
|   |-- compiler.py             # Top-level PlenaCompiler composition class
|   |-- constants.py            # BLEN/MLEN/immediate constants
|   |-- vars.py                 # Tensor/Input/VRAM/FP variable descriptors
|   |-- registers.py            # GP/ADDR/FP register allocation helpers
|   |-- memory.py               # Memory layout/address helpers
|   |-- memory_state.py         # Compiler-owned tensor/input/fp memory state
|   |
|   |-- program_tensors.py      # Program-level tensor allocation/load/store helpers
|   |-- program_fp_tile_ops.py  # Program-level FP/tile scalar-vector operations
|   |-- program_matrix_ops.py   # Program-level matrix/projection/FFN operations
|   |-- program_attention.py    # Program-level flash attention operation
|   |
|   |-- isa_compiler.py         # Low-level ISA emitter base and typed emit path
|   |-- isa_emit.py             # Generic emit helpers
|   |-- isa_fp_ops.py           # Scalar FP/FPRAM/tile FP ISA helpers
|   |-- isa_tile_rows.py        # Tile-row unary/binary loop emitters
|   |-- isa_matrix.py           # Matrix/projection/load/store ISA emitters
|   +-- isa_attention.py        # Attention-specific ISA helpers
|
+-- tests/
    |-- __init__.py
    |-- test_bf16_numerical_stability.py
    |-- test_plena_compiler.py
    +-- test_quantization_ablation.py
```

Key points:

- `aten/plena/` is the canonical compiler implementation package.
- `aten/ops/` is the ATen-style dispatcher surface.
- `aten/plena_frontend.py` is the HuggingFace/ATen frontend that drives model
  compilation.
- `aten/e2e_runner.py` runs the ATen compiler path through the emulator and
  golden comparison.
- The old `aten/plena_compiler.py` compatibility facade has been removed.
