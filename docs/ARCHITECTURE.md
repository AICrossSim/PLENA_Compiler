# PLENA Compiler Architecture

## Directory Structure

```
PLENA_Compiler/
|-- asm_templates/           # Shared ISA instruction emitters (used by BOTH pipelines)
|   |-- flashattn/           #   Flash attention (qkt, pv, online_softmax, output, reset)
|   |-- ffn_asm.py           #   FFN (gate/up/down with K-split)
|   |-- projection_asm.py    #   Linear projections (Q/K/V/O)
|   |-- normalization_asm.py #   RMSNorm / LayerNorm
|   |-- embedding_asm.py     #   Token embedding DMA
|   |-- rope_asm.py          #   Rotary position embedding
|   |-- im2col_asm.py        #   Conv2d im2col (shift variant)
|   |-- im2col_asm_no_shift.py # Conv2d im2col (no-shift variant)
|   |-- batched_matmul_asm.py  # Batched matrix multiply
|   |-- silu_asm.py          #   SiLU activation
|   |-- gelu_asm.py          #   GELU activation
|   |-- lm_head.py           #   LM head projection
|   |-- gemv_asm.py          #   General matrix-vector multiply
|   |-- _imm.py              #   Large immediate helpers
|   |-- _k_split.py          #   K-split utilities
|   |-- preload_act.py       #   VRAM preloading
|   |-- preload_addr_reg.py  #   Address register preloading
|   |-- store_act_asm.py     #   Activation store-back
|   +-- reset_reg_asm.py     #   Register reset helpers
|
|-- aten/                    # Pipeline 1: ATen compilation backend
|   |-- plena_frontend.py    #   native HF decoder -> PLENA program -> ISA text
|   |-- sliced_emulator_runner.py #   sliced HF weights -> emulator -> golden
|   |-- plena/               #   Canonical PlenaCompiler implementation package
|   |   |-- compiler.py      #     PlenaCompiler composition class
|   |   |-- memory_state.py  #     Tensor/input/FP memory state
|   |   |-- program_*.py     #     High-level program operations
|   |   +-- isa_*.py         #     Low-level ISA emitters
|   +-- ops/                 #   Registered ATen op implementations
|       |-- registry.py      #     Op dispatch registry
|       |-- plena/           #     PLENA backend wrappers
|       +-- cpu/             #     CPU reference implementations
|
|-- generator/               # Pipeline 2: Config-driven code generation
|   |-- runner.py            #   CLI entry point for codegen/utilization
|   |-- parser/              #   HF config -> symbolic graph
|   |   |-- llm_parser.py    #     LLMModelParser (text decoder graphs)
|   |   +-- hardware_parser.py #   configuration.svh / precision.svh reader
|   |-- passes/              #   Compilation passes
|   |   |-- code_gen.py      #     Symbolic graph -> ASM (generator backend)
|   |   +-- utilization_report.py # PE utilization analysis
|   |-- scheduler/           #   Memory layout + register assignment
|   |   |-- scheduler.py     #     gen_scheduler entry point
|   |   |-- mem_layout_lib.json  # Memory layout library
|   |   +-- reg_assignment_lib.json # Register assignment library
|   +-- tests/               #   Unit tests + E2E harness
|       |-- test_llm_parser.py
|       |-- test_generator_e2e.py
|       |-- test_embedding_code_gen.py
|       |-- test_attention_code_gen.py
|       |-- test_vlm_parser.py
|       +-- test_vlm_code_gen.py
|
|-- assembler/               # ASM text -> binary .mem
|   |-- parser.py            #   Tokenizer
|   +-- assembly_to_binary.py #  Instruction encoder
|
|-- sim_env_utils/           # Simulation environment builders
|   |-- build_env.py         #   Build env for ATen pipeline
|   +-- build_sys_tools.py   #   System-level build helpers
|
|-- doc/                     # ISA + hardware parameter definitions
|   |-- operation.svh        #   Opcode definitions (the ISA)
|   |-- configuration.svh    #   Hardware parameters (MLEN, VLEN, BLEN, SRAM sizes)
|   |-- precision.svh        #   Floating-point format widths
|   |-- plena_isa_spec.md    #   Full ISA specification
|   |-- memory_layout.md     #   HBM/VRAM/MRAM memory map
|   +-- Model_Lib/           #   Pre-characterized model configs
|
+-- docs/                    # This directory -- architecture documentation
    |-- ARCHITECTURE.md      #   This file
    +-- COMPILATION_PIPELINES.md # Detailed pipeline comparison
```

## Compilation Flow Summary

### Pipeline 1 (ATen) -- Numerically Verified

```
nn.Module
  -> torch.export (FX graph of ATen ops)
  -> op dispatch (aten/ops/registry.py)
  -> PlenaCompiler (aten/plena/)
     - VRAM/MRAM/FPRAM allocation
     - HBM weight layout + address register init
     - calls asm_templates/* for ISA emission
  -> assembler/ (ASM -> .mem binary)
  -> emulator (run + compare against PyTorch golden)
```

### Pipeline 2 (Generator) -- ASM Analysis

```
HF config.json
  -> LLMModelParser (generator/parser/llm_parser.py)
  -> symbolic graph (nodes with dimensions, no weights)
  -> gen_scheduler (memory layout + register assignment)
  -> code_gen_pass (generator/passes/code_gen.py)
     - dispatches each node to asm_templates/*
  -> assembler/ (ASM -> .mem binary)
  -> emulator smoke / utilization analysis
```

See `docs/COMPILATION_PIPELINES.md` for detailed comparison, known gaps,
and guidance on when to use which pipeline.
