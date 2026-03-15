# Handler API

## 目的

`VLMAssemblyGenerator` 现在只负责：

- runtime state 管理
- shared context / symbol binding
- 输出渲染与 type body 复用
- handler 注册与分发

具体 lowering handlers 被拆到：

- `/Users/yuemingwang/Doc/IC/FYP/PLENA_Compiler/generator/multi_model/vlm_codegen_handlers.py`

这样开发者可以独立开发 handler，完成后再注册到 `VLMAssemblyGenerator`。

## 分发流程

生成阶段的调用链现在是：

```text
node["type"]
  -> env.register_type(...) / env.type_registry
  -> operation_key
  -> generator.operation_handlers[operation_key]
  -> handler(generator, node, model_info)
```

注意：

- `node["type"]` 是 traced graph 里的模块类型
- `operation_key` 是 codegen 层的逻辑操作名
- 一个 `operation_key` 可以映射到任意来源的 handler，例如 `asm_templates`、`asm_lib`、自定义 kernel compiler

## Handler 签名

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

返回值使用 `LoweringResult`：

```python
LoweringResult(
    mode="template",      # 或 plena_shared / partial / unsupported
    asm="...",
    setup_asm="...",
    reuse_label="Linear",
    comments=["..."],
)
```

常见 `mode`：

- `template`: 普通模板式 lowering
- `plena_shared`: 使用 `env.shared_context` / `asm_lib` 编译器做共享上下文 lowering
- `partial`: 局部可生成，但不是完整 lowering
- `unsupported`: 当前 handler 不支持该节点配置

## 默认行为

`VLMAssemblyGenerator` 默认会自动注册 `vlm_codegen_handlers.py` 中的默认 handler：

```python
env = VLMCodegenEnvironment()
generator = VLMAssemblyGenerator(env)
```

如果你想完全自己控制注册表：

```python
env = VLMCodegenEnvironment()
generator = VLMAssemblyGenerator(env, auto_register_default_handlers=False)
```

## 注册 API

`VLMAssemblyGenerator` 公开了以下注册接口：

```python
generator.register_handler("custom_kernel", my_handler)
generator.register_handlers({"custom_kernel": my_handler})
generator.clear_handlers()
```

如果 traced node type 还没有映射到 codegen op，需要先注册 type：

```python
env.register_type("CustomKernel", "custom_kernel")
```

## 推荐使用的 Generator API

handler 开发时，优先使用 `VLMAssemblyGenerator` 暴露出来的这些方法，而不是直接改内部状态：

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

底层环境对象仍然通过 `generator.env` 暴露：

- `generator.env.template("projection")`
- `generator.env.shared_context`
- `generator.env.reg(...)`
- `generator.env.mem(...)`
- `generator.env.fp_mem(...)`

## 示例 1：基于 asm_templates 的 handler

这是模板式 handler 的最小模式：

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

注册方式：

```python
env.register_type("CustomKernel", "custom_kernel")
generator.register_handler("custom_kernel", custom_template_handler)
```

## 示例 2：基于 asm_lib kernel compiler 的 handler

如果 handler 直接调用 `asm_lib` 里的 compiler，推荐走 `plena_shared` 模式：

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

    # 这里调用你自己的 asm_lib compiler
    # emission = my_compiler.emit(...)

    context.mark_shared_lowering_used()
    return LoweringResult(
        mode="plena_shared",
        comments=[f"weight_hbm={weight_binding.hbm_addr}"],
    )
```

这种模式适合：

- 需要跨节点共享 symbol / parameter
- 需要复用 `asm_lib` 内 compiler 的 emit 流程
- 希望最终由 shared program 统一编译

## 推荐的扩展方式

推荐把不同来源的 handler 按来源拆分：

- `vlm_codegen_handlers.py`
  - 默认内建 handlers
- `asm_lib/..._handlers.py`
  - 绑定某个 kernel compiler 的 shared lowering
- `asm_templates/..._handlers.py`
  - 绑定某组模板 lowering

然后在生成器初始化后按需注册：

```python
generator = VLMAssemblyGenerator(env)
generator.register_handlers(MY_ASM_LIB_HANDLERS)
generator.register_handlers(MY_TEMPLATE_HANDLERS)
```

如果要覆盖默认 handler，直接使用同一个 `operation_key` 重新注册即可。

## 当前默认 handler 注册表

默认注册表在：

- `/Users/yuemingwang/Doc/IC/FYP/PLENA_Compiler/generator/multi_model/vlm_codegen_handlers.py`

可直接复用：

```python
from vlm_codegen_handlers import DEFAULT_OPERATION_HANDLERS, register_default_handlers
```

## 开发约定

- 新 handler 优先写成纯函数，不要把 lowering 逻辑继续塞回 `vlm_codegen_generator.py`
- 节点类型映射放在 `VLMCodegenEnvironment.register_type/register_types`
- handler 之间通过 symbol binding / shared context 协作，不直接硬编码彼此调用顺序
- 如果 handler 需要复用已有公共逻辑，优先给 `VLMAssemblyGenerator` 增加公共 helper，再在 handler 中调用

这样可以保持：

- generator 核心调度稳定
- handler 文件独立开发
- `asm_templates` 和 `asm_lib` 两种路径都能接入同一套 `VLMAssemblyGenerator`
