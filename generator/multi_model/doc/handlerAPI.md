# Handler API

## Purpose

`VLMAssemblyGenerator` is now responsible only for:

- runtime state management
- shared context / symbol binding
- output rendering and type body reuse
- handler registration and dispatch

Concrete lowering handlers have been split into:

- `/Users/yuemingwang/Doc/IC/FYP/PLENA_Compiler/generator/multi_model/vlm_codegen_handlers.py`

This lets developers implement handlers independently and register them into `VLMAssemblyGenerator` once they are ready.

## Dispatch Flow

The call chain during generation is now:

```text
node["type"]
  -> env.register_type(...) / env.type_registry
  -> operation_key
  -> generator.operation_handlers[operation_key]
  -> handler(generator, node, model_info)
```

Notes:

- `node["type"]` is the module type from the traced graph
- `operation_key` is the logical operation name at the codegen layer
- one `operation_key` can map to a handler from any source, such as `asm_templates`, `asm_lib`, or a custom kernel compiler

## Handler Signature

```python
from vlm_codegen_generator import VLMAssemblyGenerator
from vlm_codegen_handlers import LoweringResult


def my_handler(
    generator: VLMAssemblyGenerator,
    node: dict,
    model_info: dict,
) -> LoweringResult:
    ...
```

The return value uses `LoweringResult`:

```python
LoweringResult(
    mode="template",      # or plena_shared / partial / unsupported
    asm="...",
    setup_asm="...",
    reuse_label="Linear",
    comments=["..."],
)
```

Common `mode` values:

- `template`: standard template-style lowering
- `plena_shared`: shared-context lowering using `env.shared_context` or an `asm_lib` compiler
- `partial`: some code can be generated, but the lowering is not complete
- `unsupported`: the current handler does not support this node configuration

## Default Behavior

`VLMAssemblyGenerator` automatically registers the default handlers from `vlm_codegen_handlers.py` by default:

```python
env = VLMCodegenEnvironment()
generator = VLMAssemblyGenerator(env)
```

If you want full control over the registry:

```python
env = VLMCodegenEnvironment()
generator = VLMAssemblyGenerator(env, auto_register_default_handlers=False)
```

## Registration API

`VLMAssemblyGenerator` exposes the following registration APIs:

```python
generator.register_handler("custom_kernel", my_handler)
generator.register_handlers({"custom_kernel": my_handler})
generator.clear_handlers()
```

If the traced node type has not yet been mapped to a codegen op, register the type first:

```python
env.register_type("CustomKernel", "custom_kernel")
```

## Recommended Generator APIs

When implementing handlers, prefer the methods exposed by `VLMAssemblyGenerator` instead of mutating internal state directly:

- `batch_from_node`
- `seq_len_from_node`
- `hidden_from_node`
- `in_shape`
- `out_shape`
- `matrix_shape`
- `shape_tuple`
- `resolve_input_activation`
- `resolve_symbol_binding`
- `bind_template_outputs`
- `choose_activation_address`
- `emit_addr_reg_preload`
- `ensure_runtime_parameter`
- `shared_parameter_handle`
- `linear_weight_binding`
- `linear_weight_reg_binding`
- `vision_linear_child`
- `weight_shape`
- `canonical_param_name`
- `wrap_template`
- `render_node_section`

The lower-level environment object is still exposed through `generator.env`:

- `generator.env.template("projection")`
- `generator.env.shared_context`
- `generator.env.reg(...)`
- `generator.env.mem(...)`
- `generator.env.fp_mem(...)`

## Example 1: Handler Based on `asm_templates`

This is the minimal pattern for a template-based handler:

```python
from vlm_codegen_handlers import LoweringResult


def custom_template_handler(generator, node, model_info) -> LoweringResult:
    hidden = generator.hidden_from_node(node, model_info)
    batch = generator.batch_from_node(node, model_info)
    activation_addr, _ = generator.resolve_input_activation(node, default_block="block1")
    result_addr = generator.choose_activation_address(
        generator.env.mem("block2", 0),
        avoid={activation_addr},
    )

    asm = generator.env.template("projection")(
        mlen=generator.env.hw_value("mlen", 64),
        blen=generator.env.hw_value("blen", 4),
        batch=batch,
        hidden_size=hidden,
        alive_registers=generator.env.hw_value("alive_registers", [1, 2, 3, 4]),
        out_features=hidden,
        w_base_hbm_offset_reg=generator.env.reg("q_weight_offset", 2),
        rope_hbm_offset_reg=generator.env.reg("rope_params_offset", 6),
        rope_on_chip_address=generator.env.mem("block3", 0),
        activation_base_address=activation_addr,
        result_base_address=result_addr,
        rope_enabled=False,
    )

    generator.bind_template_outputs(node, result_addr)
    return generator.wrap_template(
        asm,
        reuse_label=node.get("type"),
        comments=[f"activation={activation_addr}", f"result={result_addr}"],
    )
```

Register it as:

```python
env.register_type("CustomKernel", "custom_kernel")
generator.register_handler("custom_kernel", custom_template_handler)
```

## Example 2: Handler Based on an `asm_lib` Kernel Compiler

If the handler directly invokes a compiler inside `asm_lib`, the recommended mode is `plena_shared`:

```python
from vlm_codegen_handlers import LoweringResult


def custom_shared_handler(generator, node, model_info) -> LoweringResult:
    context = generator.env.shared_context

    input_meta = (node.get("in") or [None])[0]
    matrix_shape = generator.matrix_shape(input_meta)
    if matrix_shape is None:
        return LoweringResult(mode="unsupported", comments=["missing input matrix shape"])

    in_sym = ((node.get("in_syms") or [None])[:1] or [None])[0] or "custom.input"
    input_tensor = context.ensure_symbol_tensor(
        symbol=in_sym,
        matrix_shape=matrix_shape,
        producer="model_input",
        hbm_addr=generator.ensure_runtime_symbol_input(in_sym, matrix_shape),
    )

    weight_handle, weight_binding = generator.shared_parameter_handle(
        canonical_name=generator.canonical_param_name(node["name"], "weight"),
        logical_shape=(matrix_shape[1], matrix_shape[1]),
        source_shape=generator.weight_shape(node, "weight"),
        layout="linear_weight_in_out",
    )

    # Call your own asm_lib compiler here
    # emission = my_compiler.emit(...)

    context.mark_shared_lowering_used()
    return LoweringResult(
        mode="plena_shared",
        comments=[f"weight_hbm={weight_binding.hbm_addr}"],
    )
```

This pattern is suitable when:

- symbols or parameters need to be shared across nodes
- you want to reuse the emit flow of a compiler already inside `asm_lib`
- you want the final program to be compiled uniformly by the shared program pipeline

## Recommended Extension Structure

It is recommended to separate handlers by source:

- `vlm_codegen_handlers.py`
  - default built-in handlers
- `asm_lib/..._handlers.py`
  - shared lowerings bound to a specific kernel compiler
- `asm_templates/..._handlers.py`
  - lowerings bound to a specific template group

Then register them as needed after generator initialization:

```python
generator = VLMAssemblyGenerator(env)
generator.register_handlers(MY_ASM_LIB_HANDLERS)
generator.register_handlers(MY_TEMPLATE_HANDLERS)
```

To override a default handler, simply re-register the same `operation_key`.

## Current Default Handler Registry

The default registry lives in:

- `/Users/yuemingwang/Doc/IC/FYP/PLENA_Compiler/generator/multi_model/vlm_codegen_handlers.py`

You can reuse it directly:

```python
from vlm_codegen_handlers import DEFAULT_OPERATION_HANDLERS, register_default_handlers
```

## Development Conventions

- write new handlers as pure functions when possible; do not move lowering logic back into `vlm_codegen_generator.py`
- keep node-type mappings in `VLMCodegenEnvironment.register_type/register_types`
- let handlers cooperate through symbol binding and shared context instead of hard-coding call order between handlers
- if a handler needs reusable common logic, add a helper to `VLMAssemblyGenerator` first and call it from handlers

This keeps:

- the generator core dispatch stable
- handler files independently maintainable
- both `asm_templates` and `asm_lib` integrated through the same `VLMAssemblyGenerator`
