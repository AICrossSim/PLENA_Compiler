# PLENA Multi-Model Compiler Guide

The goal of `generator/multi_model` is to provide an extensible development framework for compiling AI models into PLENA hardware assembly. The workflow is centered around one pseudo-execution of a model: record module calls and runtime data relationships, turn that information into call records that can be analyzed and lowered, and then generate PLENA-oriented asm.

This directory currently provides the following core pieces: model loading and input preparation, call-record capture and export, model metadata extraction, a handler-based codegen framework, two asm generation paths, and utilization analysis with test examples. The current focus is still on multimodal models and Transformer-like models.

Typical work in this directory looks like this: bring in a model or a module type, confirm that pseudo-execution produces the right call records, write a handler for the corresponding `module type`, connect it to either the `asm_templates` path or the `asm_lib` path, and then inspect the generated asm and utilization report.

## Current Scope

This project is still in progress. It should not be treated as a finished model compiler.

- The current focus is the prototype pipeline: pseudo-execution capture -> call-record organization -> asm generation -> utilization analysis.
- Support for most real-world models is still incomplete.
- Even for model families that already have parsing paths, such as Qwen, many modules still cannot be fully lowered.
- In most cases, the blocker is not model loading in the parser, but missing or incomplete handlers for the corresponding `module type`.
- The current codegen keeps two generation modes, but they should not be mixed. That scenario has not been specifically tested.
- In practice, this directory is better treated as a development and extension framework than as a stable end-user compiler tool.

## 1. What This Directory Does

The overall flow can be summarized in four steps:

1. Use `VLMModelParser` to load a model and inputs, then run one pseudo-execution.
2. During that run, record module calls, input/output information, and symbol relationships, store them as a tree of call records, and also export a flattened list in execution order.
3. Use `vlm_codegen()` or `VLMAssemblyGenerator` to map those call records into PLENA asm.
4. Optionally use `utilization_report.py` to analyze compute and activation-memory utilization from the recorded run.

In this document, a "node" is not a static graph op. It means one recorded module invocation. If the same module is called multiple times during execution, it will correspond to multiple nodes.

Typical outputs include:

- Call-record JSON
- Model info JSON
- `.asm` files
- Utilization reports in Markdown or JSON

## 2. Recommended Entry Points

If you are new to this directory, read the following in order:

1. [run.py](run.py)
   The most direct CLI entry point. It connects model loading, pseudo-execution capture, codegen, and reporting.
2. [vlm_parser.py](vlm_parser.py)
   Handles model loading, input templates, hook / FX capture, call-record flattening, and model metadata extraction.
3. [vlm_codegen.py](vlm_codegen.py)
   A simplified external wrapper around codegen.
4. [vlm_codegen_generator.py](vlm_codegen_generator.py)
   The main generator. It manages runtime state, shared context, handler dispatch, and asm assembly.
5. [vlm_codegen_handlers.py](vlm_codegen_handlers.py)
   The default lowering handlers. This is one of the main files you will edit when adding support.
6. [doc/handlerAPI.md](doc/handlerAPI.md)
   Required reading if you want to add new handlers.
7. [asm_lib/README.md](asm_lib/README.md)
   Recommended if you need to work on lower-level PLENAProgram or compiler-side kernels.

## 3. Project Structure

This section is organized by responsibility rather than only by file name.

### 3.1 Entrypoints and Demos

- [run.py](run.py)
  CLI entry point. Supports both local demo models and Hugging Face models, and exports asm, call records, and reports.
- [playground.py](playground.py)
  Lightweight debugging entry point for trying individual helpers.
- [utilization_report_run.py](utilization_report_run.py)
  Standalone example for utilization reporting.

### 3.2 Model Parsing and Runtime Record Capture

- [vlm_parser.py](vlm_parser.py)
  The core parser. It is responsible for:
  - loading local `nn.Module` instances or Hugging Face models
  - preparing processor-backed inputs for models such as Qwen3-VL
  - recording leaf-module / FX call information during one pseudo-execution
  - exporting flattened call lists
  - extracting the `model_info` required by codegen
- [plena_backend.py](plena_backend.py)
  Backend matching and graph-processing logic used during capture and compilation.
- [model/TRANSFOMER_ENCODER.py](model/TRANSFOMER_ENCODER.py)
  A local demo model for running the flow without external model dependencies.

### 3.3 Codegen Core

- [vlm_codegen.py](vlm_codegen.py)
  External wrapper layer.
- [vlm_codegen_env.py](vlm_codegen_env.py)
  Manages hardware parameters, scheduling configuration, template loading, and the type-to-operation registry.
- [vlm_codegen_generator.py](vlm_codegen_generator.py)
  Manages:
  - runtime parameter and activation bindings
  - shared context
  - handler registration and dispatch
  - reusable type bodies
  - output organization for two lowering paths
- [vlm_codegen_handlers.py](vlm_codegen_handlers.py)
  Default handlers, including:
  - embedding
  - linear
  - layer norm / RMS norm
  - text attention / vision attention
  - FFN / MLP
  - `plena_shared` lowering for Qwen3 vision MLP
- [vlm_codegen_context.py](vlm_codegen_context.py)
  Shared codegen context and symbol-management helpers.

### 3.3.1 Two Codegen Generation Modes

The current codegen combines two generation modes:

1. `asm_templates` mode
   This is the template-based path built on the older compiler flow. It depends on a pre-arranged memory layout, and that layout must be registered in `VLMCodegenEnvironment`.
2. `asm_lib` kernel compiler mode
   This path is based on the compiler components in [asm_lib/](asm_lib/). The corresponding kernel compiler is responsible for lower-level generation details.

What they have in common:

- Both require you to write the corresponding handler first.
- Both require binding that handler to a `module type`, which is then resolved through `module type -> operation_key -> handler`.

Important limitation:

- For a given development target, choose one generation mode.
- Do not mix the `asm_templates` path and the `asm_lib` kernel compiler path in the same development flow.
- The current codegen has not been specifically tested for that mixed-use scenario. These two paths should be understood as two available options, not a recommended combination.

### 3.4 Lower-Level Compiler Support in `asm_lib`

- [asm_lib/plena_program.py](asm_lib/plena_program.py)
  The high-level DSL for people describing programs.
- [asm_lib/developer_compiler.py](asm_lib/developer_compiler.py)
  Lower-level ISA generation and resource management.
- [asm_lib/sub_matrix_manager.py](asm_lib/sub_matrix_manager.py)
  Core logic for HBM / VRAM / FPRAM layout and sub-block addressing.

### 3.5 Analysis, Documentation, and Tests

- [utilization_report.py](utilization_report.py)
  Compute and memory-utilization analysis based on recorded execution.
- [doc/handlerAPI.md](doc/handlerAPI.md)
  Handler extension API documentation.
- [test/test_vlm_parser.py](test/test_vlm_parser.py)
  Parser tests.
- [test/test_vlm_codegen.py](test/test_vlm_codegen.py)
  Codegen and shared-context lowering tests.
- [test/test_utilization_report.py](test/test_utilization_report.py)
  Utilization-report tests.

### 3.6 Input and Output Directories

- [inputs/](inputs/)
  Example inputs, currently including an image sample.
- [outputs/](outputs/)
  Temporary outputs and preview results.

## 4. Core Data Flow

You can think about the system like this:

```text
model + inputs
  -> VLMModelParser
  -> call tree / flattened call list / model_info
  -> VLMCodegenEnvironment(type_registry, hw, sched)
  -> VLMAssemblyGenerator(operation_handlers)
  -> handler(node, model_info)
  -> choose one generation mode
  -> PLENA asm
```

There are two key mappings:

1. `node["type"] -> operation_key`
   Managed by `VLMCodegenEnvironment.type_registry`.
2. `operation_key -> handler`
   Managed by `VLMAssemblyGenerator.operation_handlers`.

`node` is still the name used in code, but conceptually it means one recorded module invocation.

These two mappings are also the main extension points when adding support for new operations.

## 5. How to Run the Existing Flow

The Python project configuration lives in [../pyproject.toml](../pyproject.toml). Current dependencies include at least:

- `torch`
- `transformers`
- `huggingface-hub`
- `pillow`
- `qwen-vl-utils`
- `torchvision`

It is recommended to run from [../](../) with `uv`, and explicitly set `PYTHONPATH=..` so that the top-level [../../asm_templates/](../../asm_templates/) package can be imported correctly. Otherwise codegen will fall back to stub templates.

### 5.1 Run the Local Demo Model

```bash
cd generator
PYTHONPATH=.. uv run python multi_model/run.py \
  --output-prefix ./multi_model/outputs/demo/run
```

This command will:

- build the local `MultiLayerEncoder` demo model
- generate call-record JSON
- generate model info JSON
- generate `.asm`
- generate a utilization report

### 5.2 Run a Hugging Face Model

```bash
cd generator
PYTHONPATH=.. uv run python multi_model/run.py \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --image ./multi_model/inputs/img/image.png \
  --text "Describe this image." \
  --output-prefix ./multi_model/outputs/qwen3_vl/run
```

Notes:

- If the model uses a processor-backed input flow, you usually must provide `--image`.
- The first run of a Hugging Face model may download model artifacts.
- This path is mainly for pseudo-execution capture and codegen prototyping. It does not mean all real model modules are already supported.

### 5.3 Generate Only the Utilization Report

```bash
cd generator
PYTHONPATH=.. uv run python multi_model/utilization_report_run.py
```

## 6. How to Read the Outputs

`run.py` usually produces the following files:

- `*_trace.json`
  A flattened list of recorded calls. Useful for inspecting input/output shapes, weights, and symbols for each module invocation.
- `*_model_info.json`
  Model-level metadata required by codegen, such as `hidden_size`, `num_layers`, and `head_dim`.
- `*.asm`
  Final asm output. It usually contains:
  - template / partial lowering sections
  - covered-node binding notes
  - shared-context lowering summaries
  - reusable type bodies
  - shared PLENA program sections
- `*_report.md` / `*_report.json`
  Compute and activation-memory utilization results.

When debugging lowering issues, inspect these together:

1. `type`, `name`, `in_syms`, and `out_syms` in the call records
2. Size-related fields in `model_info`
3. asm comments such as `lowering=...`, `weight hbm=...`, and `covered_by=...`

## 7. Common Development Tasks

### 7.1 Add Support for a New Module Call Type

This is the most common case. A typical sequence is:

1. Confirm that the parser can already record this module type during pseudo-execution.
2. Confirm the `type` value used by those call records.
3. Register that `type` to an `operation_key` in [vlm_codegen_env.py](vlm_codegen_env.py).
4. Implement the handler in [vlm_codegen_handlers.py](vlm_codegen_handlers.py).
5. Register the handler under that `operation_key`.
6. Add tests that cover at least:
   - handler dispatch works
   - expected lowering markers appear in asm
   - input/output symbol bindings are correct

If you only need a custom mapping, the setup looks like this:

```python
from vlm_codegen_env import VLMCodegenEnvironment
from vlm_codegen_generator import VLMAssemblyGenerator

env = VLMCodegenEnvironment()
env.register_type("CustomKernel", "custom_kernel")

generator = VLMAssemblyGenerator(env)
generator.register_handler("custom_kernel", my_handler)
```

### 7.2 Write a New Handler

Read this together with:

- [doc/handlerAPI.md](doc/handlerAPI.md)

The job of a handler is to lower one call record into a codegen result. Its signature is roughly:

```python
def my_handler(generator, node, model_info) -> LoweringResult:
    ...
```

Common `LoweringResult` modes include:

- `template`
- `plena_shared`
- `partial`
- `unsupported`

Development guidance:

- Prefer reusing helper methods exposed by `VLMAssemblyGenerator` instead of editing internal state directly.
- For template-style lowering, prefer `generator.env.template(...)` plus `generator.wrap_template(...)`.
- If you need cross-node shared symbols or parameters, prefer `plena_shared`.
- Within one development path, pick one generation mode. Do not mix `asm_templates`-based handlers with `asm_lib` kernel compiler handlers.

### 7.3 Connect a Handler to `asm_lib` Compiler Logic

If template lowering is not expressive enough for a complex kernel, call into `asm_lib` from the handler.

The recommended layering is:

1. `handler`
   Decides when lowering should trigger and translates metadata from the call record into compiler inputs.
2. `asm_lib/plena_program.py`
   Good for describing algorithms with a higher-level DSL.
3. `asm_lib/developer_compiler.py`
   Good for lower-level ISA generation flows.
4. `asm_lib/sub_matrix_manager.py`
   Good for layout, addressing, and allocator work.

If you want a reference implementation for shared lowering, start with the existing Qwen3 vision MLP path.

One additional engineering rule:

- Once you decide to use the `asm_lib` kernel compiler path for a class of modules, keep development and testing on that path.
- Do not assume that some handlers can use the old `asm_templates` path while others switch to `asm_lib` and still work together reliably.
- The repository does not currently treat that mixed setup as a supported, validated scenario.

### 7.4 Extend the Parser

If model execution is not being recorded correctly, or metadata extraction is not sufficient, fix the parser before touching codegen.

Typical places to modify:

- model-loading branches in [vlm_parser.py](vlm_parser.py)
- `extract_model_info(...)`
- `trace_leaf_modules(...)`
- mappings from FX ops to logical operations
- Qwen / VLM input-template logic such as `template_qwen3_vl_inputs(...)`

A practical rule of thumb:

- If a module call is not recorded, fix the parser.
- If the call record exists but no asm is generated, fix type registration or the handler.
- If the handler is selected but the generated asm is too weak or too low-level, move work into `asm_lib`.

## 8. Recommended Development Flow

Here is a reliable development sequence:

1. Start with a minimal reproducible model, ideally a simple `nn.Module` that you wrote locally so it is easy to see what each layer is doing.
2. Run the parser once and export the resulting call records to JSON.
3. Identify the exact call-record shape you want to support, including its `type`, input/output shapes, weight shapes, and symbol information.
4. Write the smallest handler that can at least be called correctly. If full asm generation is not ready yet, return `partial` or `unsupported` explicitly.
5. Then extend that handler step by step until it can generate the asm you need in a stable way.
6. Add tests that confirm the correct handler is selected and the generated result matches expectations for that input shape.
7. Only after that, validate the change again with a real model.

The reason for this order is simple: the hard part is usually not "writing the asm string". The hard part is making sure that the information recorded during model execution actually matches the assumptions used by your lowering logic. If you start directly with a real model, it becomes much harder to tell whether a problem comes from parsing, handler selection, or asm generation.

## 9. Testing

Tests live under [test/](test/). They are written in `unittest` style. Since `pyproject.toml` does not currently declare `pytest` by default, the safer default is to run each file directly with `python`. If your local environment already has `pytest`, you can also use it.

From [../](../):

```bash
PYTHONPATH=.. uv run python multi_model/test/test_vlm_parser.py
PYTHONPATH=.. uv run python multi_model/test/test_vlm_codegen.py
PYTHONPATH=.. uv run python multi_model/test/test_utilization_report.py
```

If `pytest` is installed in your environment, you can also run:

```bash
cd generator
PYTHONPATH=.. uv run python -m pytest multi_model/test/test_vlm_codegen.py -v
```

At minimum, new features should usually add:

- parser / runtime-record tests
- codegen / handler-dispatch tests

## 10. Key Design Constraints

Keep these constraints in mind while developing:

- `VLMAssemblyGenerator` is mainly responsible for scheduling, binding, reuse, and dispatch. Do not keep pushing all lowering details back into the generator itself.
- Specific lowering logic should live in handlers as much as possible.
- More complex kernel logic should move further down into `asm_lib`.
- The `type` in recorded module calls and the codegen `operation_key` are two different layers. Do not conflate them.
- The project currently keeps both `asm_templates` and `asm_lib` generation paths, but development should explicitly choose one of them instead of mixing both.

## 11. Reading Recommendations

If your goal is:

- "Run the flow once"
  Start with [run.py](run.py).
- "Add lowering for a new module"
  Start with [doc/handlerAPI.md](doc/handlerAPI.md) and [vlm_codegen_handlers.py](vlm_codegen_handlers.py).
- "Implement lower-level PLENA kernel compiler logic"
  Start with [asm_lib/README.md](asm_lib/README.md).
- "Debug why a model cannot be compiled"
  Start with the call-record JSON, `model_info`, and handler dispatch.
