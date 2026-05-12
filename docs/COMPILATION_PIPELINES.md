# PLENA Compiler -- Compilation Pipelines

## Overview

The PLENA compiler has **two compilation pipelines** that share low-level
ISA instruction templates (`asm_templates/`) but differ in their frontends,
backends, and weight-handling strategies.

```
                    +------------------+
                    |  HuggingFace     |
                    |  Model / Config  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
     Pipeline 1 (ATen)            Pipeline 2 (Generator)
              |                             |
    torch.export -> FX graph      HF config -> LLMModelParser
              |                             |
    ATen op dispatch              symbolic graph -> code_gen_pass
              |                             |
    PlenaCompiler + ops.*         direct asm_template dispatch
    (VRAM/MRAM/FPRAM mgmt)       (scheduler-driven registers)
              |                             |
              +----------- v ---------------+
                           |
                   asm_templates/*
                   (shared ISA emitters)
                           |
                     assembler/
                  (ASM -> binary .mem)
```

---

## Pipeline 1: ATen Path

**Status**: Production-ready, numerically verified (98-100% allclose)

### How it works

1. **Frontend**: A PyTorch `nn.Module` is traced via `torch.export`, producing
   an FX graph of ATen operations (`aten.linear`, `aten.rms_norm`,
   `aten.scaled_dot_product_attention`, etc.).

2. **Op dispatch**: Each ATen node is matched against registered PLENA ops
   in `aten/ops/plena/` (linear, attention, FFN, norm, conv, softmax,
   embedding). A CPU fallback registry (`aten/ops/cpu/`) provides reference
   implementations for ops not yet hardware-mapped.

3. **Backend**: `PlenaCompiler` (`aten/plena/`) manages all
   hardware state -- VRAM allocation, MRAM tile scheduling, FPRAM slot
   assignment, HBM weight layout, and address register initialization
   (`C_SET_ADDR_REG`). It calls into `asm_templates/` to emit ISA strings.

4. **Weight loading**: `sim_env_utils/build_env.py` and
   `transactional_emulator/tools/create_sim_env.py` build the simulation
   environment, applying MXFP8 quantization when writing weights to HBM.

5. **Verification**: The emulator runs the assembled binary and output
   activations are compared against a PyTorch golden reference.

### Key files

| File | Role |
|------|------|
| `aten/plena/` | Canonical PlenaCompiler implementation package |
| `aten/plena_frontend.py` | HuggingFace model frontend that drives ATen compilation |
| `aten/ops/plena/*.py` | Registered ATen op implementations (linear, attention, ffn, norm, conv, softmax, embedding) |
| `aten/ops/cpu/*.py` | CPU reference fallbacks |
| `aten/ops/registry.py` | Op dispatch registry |
| `generator/aten_runner.py` | E2E harness: model load -> compile -> emulate -> verify |
| `sim_env_utils/build_env.py` | Simulation environment builder |

### Entry points

- **Single-layer tests**: `model_layer_test_builder.py::build_and_run_decoder_test`
- **Full-model E2E**: `generator/aten_runner.py::run_aten_e2e`
- **CLI**: `python -m generator.runner aten <model> --seq-len 32 --num-layers 1`

### Test suite

Tests live in `transactional_emulator/testbench/aten/` and
`transactional_emulator/testbench/models/`. As of this writing, 18/18
pass with 98-100% allclose.

---

## Pipeline 2: Generator Path

**Status**: Generates valid ISA for structural analysis and smoke tests.

### How it works

1. **Frontend**: `LLMModelParser` (`generator/parser/llm_parser.py`) reads a
   HuggingFace model config (local JSON or remote `config.json`) and builds
   a **symbolic graph** from dimensions alone -- no weights are loaded.

2. **Scheduling**: `gen_scheduler` (`generator/scheduler/scheduler.py`)
   assigns memory layout and register mappings using library files
   (`mem_layout_lib.json`, `reg_assignment_lib.json`).

3. **Backend**: `code_gen_pass` (`generator/passes/code_gen.py`) walks the
   symbolic graph and dispatches each node to the appropriate
   `asm_templates/` function, passing scheduler-derived register and
   address parameters. It emits address-register initialization for HBM-backed
   weights before the generated compute body.

4. **Weight loading**: For E2E smoke tests, `test_generator_e2e.py` has a
   `_build_hbm_from_hf_weights` helper that loads real weights. The
   standard codegen path does not touch weights at all.

5. **Output**: A `.asm` file that assembles cleanly and runs on the emulator.
   The generator path is still primarily used for structural codegen and
   utilization work; the ATen path remains the numerically verified flow.

### Key files

| File | Role |
|------|------|
| `generator/parser/llm_parser.py` | HF config -> symbolic graph |
| `generator/parser/hardware_parser.py` | `configuration.svh` / `precision.svh` reader |
| `generator/scheduler/scheduler.py` | Memory layout + register assignment |
| `generator/passes/code_gen.py` | Symbolic graph -> ASM via asm_template dispatch |
| `generator/passes/utilization_report.py` | PE utilization analysis |
| `generator/runner.py` | CLI entry point for codegen / utilization modes |

### Entry points

- **CLI**: `python -m generator.runner codegen <model> output.asm --seq-len 512`
- **Utilization**: `python -m generator.runner utilization <model> dummy.asm`

### Test suite

Tests live in `generator/tests/`:
- `test_llm_parser.py` -- parser unit tests
- `test_embedding_code_gen.py`, `test_attention_code_gen.py` -- codegen unit tests
- `test_generator_e2e.py` -- full-pipeline smoke tests
- `test_vlm_parser.py`, `test_vlm_code_gen.py` -- vision/multimodal tests

CI workflow: `.github/workflows/` runs these on push.

---

## Shared Infrastructure

### `asm_templates/` -- ISA instruction emitters

Stateless functions that take dimensions, register indices, and addresses
as parameters and return PLENA ISA instruction strings. Both pipelines
call into these.

| Template | Operation |
|----------|-----------|
| `projection_asm.py` | Linear projections (Q/K/V/O), transposed variant |
| `ffn_asm.py` | FFN (gate/up/down with K-split support) |
| `flashattn/` | Flash attention (qkt, pv, online_softmax, output, reset) |
| `normalization_asm.py` | RMSNorm, LayerNorm |
| `embedding_asm.py` | Token embedding DMA from HBM |
| `rope_asm.py` | Rotary position embedding |
| `im2col_asm.py`, `im2col_asm_no_shift.py` | Conv2d im2col (shift and no-shift variants) |
| `batched_matmul_asm.py` | Batched matrix multiply |
| `silu_asm.py`, `gelu_asm.py` | Activation functions |
| `lm_head.py` | LM head projection |
| `gemv_asm.py` | General matrix-vector multiply |
| `preload_act.py`, `preload_addr_reg.py` | VRAM/register preloading |
| `store_act_asm.py` | Activation store-back to HBM |
| `reset_reg_asm.py` | Register reset helpers |
| `_imm.py`, `_k_split.py` | Large immediate + K-split utilities |

### `assembler/` -- ASM text to binary

- `parser.py` -- tokenizes `.asm` text
- `assembly_to_binary.py` -- encodes instructions into `.mem` binary

### `doc/` -- ISA and hardware definitions

- `operation.svh` -- opcode definitions (the ISA)
- `configuration.svh` -- hardware parameters (MLEN, VLEN, BLEN, SRAM sizes)
- `precision.svh` -- floating-point format widths
- `plena_isa_spec.md` -- full ISA specification
- `memory_layout.md` -- HBM/VRAM/MRAM memory map
- `Model_Lib/` -- pre-characterized model configs (JSON)

---

## When to Use Which

| Use case | Recommended pipeline |
|----------|---------------------|
| Numerical verification of a layer/model | ATen (`runner.py aten`) |
| ASM generation for profiling/analysis | Generator (`runner.py codegen`) |
| New model bring-up | ATen first (verified), then Generator (analysis) |
| RTL co-simulation | ATen (address registers set correctly) |
| PE utilization estimation | Generator (`runner.py utilization`) |
| CI smoke tests (assembly validity) | Generator (fast, no weights needed) |

---

## Known Gaps in Generator Path

1. **HBM address registers**: `code_gen_pass` does not emit
   `C_SET_ADDR_REG` instructions. The ATen path's `PlenaCompiler` handles
   this via `preload_addr_reg_asm`.

2. **Weight layout**: The generator does not manage HBM weight placement.
   Weight offsets are symbolic, not physical addresses.

3. **FPRAM slot conventions**: The generator does not initialize FPRAM
   precious slots (attn_scale, -inf, eps, 1/hidden, 1.0). The ATen path
   seeds these via `PlenaCompiler.fp_sram` and preserves them across ops.

4. **K-split partial sums**: The generator does not handle K-dimension
   splitting for large linear layers (K_col > 4*MLEN). The ATen path
   implements this in `linear_ops.py`.

5. **MXFP8 quantization**: Generator tests that do load weights
   (`_build_hbm_from_hf_weights`) apply quantization, but the standard
   codegen path has no weight awareness at all.
