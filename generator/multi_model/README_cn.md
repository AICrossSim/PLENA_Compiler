# PLENA Multi-Model 编译生成器

`generator/multi_model` 的目标，是为“把 AI 模型编译到 PLENA 硬件 asm”这件事提供一套可扩展的开发框架。它围绕一次模型伪运行展开：先抓取模块调用过程和运行时数据关系，再把这些信息整理成可分析、可 lower 的调用记录，最后生成面向 PLENA 的 asm。

这个目录当前提供的核心能力包括：模型加载与输入准备、调用记录抓取与导出、模型元信息提取、基于 handler 的 codegen 框架、两类 asm 生成路径，以及利用率分析与测试样例。当前重点仍然是多模态模型和类 Transformer 模型的编译支持。

开发者在这个目录里通常做的事情是：接入一个模型或一个模块类型，先确认伪运行能否抓到正确的调用记录，再为对应 `module type` 编写 handler，把它接到 `asm_templates` 或 `asm_lib` 的一条生成路径上，最后检查 asm 输出和利用率分析结果。

## 当前实现边界

这个工程目前仍处于“进行中”的状态，不能把它理解成一个已经完成的大模型编译器。

- 当前重点在于打通“模型伪运行抓取 -> 调用记录整理 -> asm 生成 -> 利用率分析”这条原型链路。
- 对大多数真实模型的支持还不完整。
- 即使已经有解析路径的模型系列，例如 Qwen 系列，很多模块也还不能完整 lower。
- 主要原因通常不是 parser 无法加载模型，而是对应 `module type` 的 handler 还没有实现，或者实现仍是 `partial` / `unsupported`。
- 当前 codegen 同时保留了两种生成模式，但不要把两种模式混合使用；这一场景目前没有经过专门测试。
- 因此，当前目录更适合作为开发和扩展框架，而不是直接面向最终用户的稳定编译工具。

## 1. 这个目录在做什么

整体工作流可以概括为四步：

1. 使用 `VLMModelParser` 加载模型与输入，并对模型做一次伪运行。
2. 在伪运行过程中抓取模块调用关系、输入输出信息和符号依赖，保存成树状调用记录；同时也会导出按执行顺序展开后的扁平调用列表。
3. 使用 `vlm_codegen()` / `VLMAssemblyGenerator` 把这些调用记录映射为 PLENA asm。
4. 选做：使用 `utilization_report.py` 对这次运行记录做算力与激活内存利用率分析。

这里的“节点”不是静态图里的抽象 op 节点，而是“一次模块被调用时生成的一条调用记录”。同一个模块如果在运行中被调用多次，就会对应多个节点。

输出物通常包括：

- 调用记录 JSON
- model info JSON
- `.asm` 文件
- 利用率报告（Markdown / JSON）

## 2. 推荐先理解的主路径

如果你第一次进入这个目录，建议按下面顺序阅读：

1. [run.py](run.py)
   这是最直接的命令行入口，串起“加载模型 -> 伪运行抓取 -> codegen -> report”完整流程。
2. [vlm_parser.py](vlm_parser.py)
   负责模型加载、输入模板、hook / FX 抓取、调用记录扁平化、模型元信息提取。
3. [vlm_codegen.py](vlm_codegen.py)
   对外提供简化的 codegen 包装接口。
4. [vlm_codegen_generator.py](vlm_codegen_generator.py)
   真正的生成器主体，管理 runtime state、共享上下文、handler 分发与 asm 拼装。
5. [vlm_codegen_handlers.py](vlm_codegen_handlers.py)
   默认 lowering handler 集合，是新增支持时最常改动的文件。
6. [doc/handlerAPI_cn.md](doc/handlerAPI_cn.md)
   如果你要新增 handler，这是必须读的文档。
7. [asm_lib/README_cn.md](asm_lib/README_cn.md)
   如果你要继续往底层写 PLENAProgram / compiler 级别 kernel，也建议一起读。

## 3. 项目结构

下面按“职责”而不是按文件名简单罗列：

### 3.1 入口与演示

- [run.py](run.py)
  命令行入口。支持本地 demo 模型，也支持加载 Hugging Face 模型，并导出 asm、调用记录和 report。
- [playground.py](playground.py)
  轻量调试入口，适合验证单个工具函数。
- [utilization_report_run.py](utilization_report_run.py)
  单独演示利用率报告生成。

### 3.2 模型解析与运行记录抓取

- [vlm_parser.py](vlm_parser.py)
  核心解析器。负责：
  - 加载本地 `nn.Module` 或 Hugging Face 模型
  - 为 Qwen3-VL 之类模型准备 processor 输入
  - 在一次伪运行中抓取 leaf module / FX 调用信息
  - 导出扁平调用列表
  - 提取 codegen 所需 `model_info`
- [plena_backend.py](plena_backend.py)
  backend 匹配与 graph 处理相关逻辑，服务于抓取 / compile 阶段。
- [model/TRANSFOMER_ENCODER.py](model/TRANSFOMER_ENCODER.py)
  本地 demo 模型，方便不依赖外部模型时直接跑通流程。

### 3.3 Codegen 主体

- [vlm_codegen.py](vlm_codegen.py)
  对外包装层。
- [vlm_codegen_env.py](vlm_codegen_env.py)
  管理硬件参数、调度配置、模板加载、type 到 operation 的注册表。
- [vlm_codegen_generator.py](vlm_codegen_generator.py)
  管理：
  - runtime parameter / activation binding
  - shared context
  - handler 注册与分发
  - type body 复用
  - 两类 lowering 路径的输出组织
- [vlm_codegen_handlers.py](vlm_codegen_handlers.py)
  默认 handler 集，包括：
  - embedding
  - linear
  - layer norm / RMS norm
  - text attention / vision attention
  - FFN / MLP
  - Qwen3 vision MLP 的 `plena_shared` lowering
- [vlm_codegen_context.py](vlm_codegen_context.py)
  shared codegen 上下文与 symbol 管理辅助逻辑。

### 3.3.1 两种 codegen 生成模式

当前 codegen 部分同时融合了两种生成模式：

1. `asm_templates` 模式
   这是基于旧 compiler 的模板式生成路径。它依赖预先编排好的内存布局，并需要在 `VLMCodegenEnvironment` 中完成相应注册。
2. `asm_lib` kernel compiler 模式
   这是基于 [asm_lib/](asm_lib/) 的 compiler 路径，由对应 kernel compiler 负责更底层的生成逻辑。

两种模式的共同点是：

- 都需要先编写对应的 handler。
- 都需要把 handler 绑定到某个 `module type`，再通过 `module type -> operation_key -> handler` 的映射命中。

使用时的限制是：

- 对一个开发目标，建议只选择一种生成模式。
- 不要在同一套开发路径里混合使用 `asm_templates` 模式和 `asm_lib` kernel compiler 模式。
- 当前 codegen 没有针对这种混合使用场景做专门测试，因此 README 中提到的两套能力，应理解为“都存在”，而不是“推荐混搭”。

### 3.4 asm_lib 底层编译能力

- [asm_lib/plena_program.py](asm_lib/plena_program.py)
  面向“写程序的人”的高层 DSL。
- [asm_lib/developer_compiler.py](asm_lib/developer_compiler.py)
  更底层的 ISA 生成与资源管理实现。
- [asm_lib/sub_matrix_manager.py](asm_lib/sub_matrix_manager.py)
  HBM / VRAM / FPRAM 布局与子块寻址核心。


### 3.5 分析、文档与测试

- [utilization_report.py](utilization_report.py)
  基于运行记录的算力与内存利用率分析。
- [doc/handlerAPI.md](doc/handlerAPI.md)
- [doc/handlerAPI_cn.md](doc/handlerAPI_cn.md)
  handler 扩展接口说明。
- [test/test_vlm_parser.py](test/test_vlm_parser.py)
  parser 行为测试。
- [test/test_vlm_codegen.py](test/test_vlm_codegen.py)
  codegen 与 shared-context lowering 测试。
- [test/test_utilization_report.py](test/test_utilization_report.py)
  利用率报告测试。

### 3.6 输入输出目录

- [inputs/](inputs/)
  示例输入，目前含图片样例。
- [outputs/](outputs/)
  临时输出与预览结果目录。

## 4. 核心数据流

开发时可以把整个系统理解成下面这条链路：

```text
model + inputs
  -> VLMModelParser
  -> 调用树 / 扁平调用列表 / model_info
  -> VLMCodegenEnvironment(type_registry, hw, sched)
  -> VLMAssemblyGenerator(operation_handlers)
  -> handler(node, model_info)
  -> 选择一种生成模式
  -> PLENA asm
```

其中有两个关键映射：

1. `node["type"] -> operation_key`
   由 `VLMCodegenEnvironment.type_registry` 管理。
2. `operation_key -> handler`
   由 `VLMAssemblyGenerator.operation_handlers` 管理。

这里的 `node` 仍然是代码里的字段命名，但语义上应理解为“一次模块调用记录”。

这也是后续扩展新算子时最重要的两个插入点。

## 5. 如何运行现有流程

本项目的 Python 配置在 [../pyproject.toml](../pyproject.toml) 中，当前依赖至少包括：

- `torch`
- `transformers`
- `huggingface-hub`
- `pillow`
- `qwen-vl-utils`
- `torchvision`

建议在 [../](../) 目录下使用 `uv` 运行，并显式补上 `PYTHONPATH=..`，这样顶层的 [../../asm_templates/](../../asm_templates/) 才能被正确导入。否则 codegen 会退回到 stub template。

### 5.1 跑本地 demo 模型

```bash
cd generator
PYTHONPATH=.. uv run python multi_model/run.py \
  --output-prefix ./multi_model/outputs/demo/run
```

这个命令会：

- 构造本地 `MultiLayerEncoder` demo 模型
- 生成调用记录 JSON
- 生成 model info JSON
- 生成 `.asm`
- 生成利用率报告

### 5.2 跑 Hugging Face 模型

```bash
cd generator
PYTHONPATH=.. uv run python multi_model/run.py \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --image ./multi_model/inputs/img/image.png \
  --text "Describe this image." \
  --output-prefix ./multi_model/outputs/qwen3_vl/run
```

注意：

- 如果模型需要 processor 输入，通常必须提供 `--image`。
- 首次运行 Hugging Face 模型会触发模型下载。
- 这条路径主要用于伪运行抓取与 codegen 原型验证，不代表已经覆盖了所有真实模型模块。

### 5.3 单独生成利用率报告

```bash
cd generator
PYTHONPATH=.. uv run python multi_model/utilization_report_run.py
```

## 6. 生成结果怎么看

`run.py` 的输出一般包括以下几类文件：

- `*_trace.json`
  扁平化后的调用记录列表，适合定位每个模块调用的输入输出形状、weight 和 symbol。
- `*_model_info.json`
  codegen 所需的模型级元信息，例如 `hidden_size`、`num_layers`、`head_dim`。
- `*.asm`
  最终 asm 输出。通常会同时包含：
  - template / partial lowering 区段
  - covered node 绑定说明
  - shared-context lowering 摘要
  - reusable type bodies
  - shared PLENA program
- `*_report.md` / `*_report.json`
  硬件利用率与激活内存分析结果。

如果你在排查 lowering 问题，优先同时看：

1. 调用记录里的 `type`、`name`、`in_syms`、`out_syms`
2. `model_info` 中的尺寸字段
3. asm 注释里标出的 `lowering=...`、`weight hbm=...`、`covered_by=...`

## 7. 开发者最常做的几类工作

### 7.1 新增一个模块调用类型的支持

这是最常见场景。标准步骤如下：

1. 先确认 parser 是否已经能把目标模块在伪运行中抓取成调用记录。
2. 确认这类调用记录的 `type` 是什么。
3. 在 [vlm_codegen_env.py](vlm_codegen_env.py) 中把这个 `type` 注册到一个 `operation_key`。
4. 在 [vlm_codegen_handlers.py](vlm_codegen_handlers.py) 中实现 handler。
5. 把 handler 注册到 `operation_key`。
6. 增加测试，至少覆盖：
   - handler 正常命中
   - asm 中出现预期 lowering 标记
   - 输入输出 symbol 绑定不出错

如果只是新增一个自定义映射，也可以这样做：

```python
from vlm_codegen_env import VLMCodegenEnvironment
from vlm_codegen_generator import VLMAssemblyGenerator

env = VLMCodegenEnvironment()
env.register_type("CustomKernel", "custom_kernel")

generator = VLMAssemblyGenerator(env)
generator.register_handler("custom_kernel", my_handler)
```

### 7.2 写一个新的 handler

这一部分请直接配合阅读：

- [doc/handlerAPI_cn.md](doc/handlerAPI_cn.md)

handler 的职责是：把一条调用记录 lower 成某种 codegen 结果。它的签名大致是：

```python
def my_handler(generator, node, model_info) -> LoweringResult:
    ...
```

`LoweringResult` 目前常见模式有：

- `template`
- `plena_shared`
- `partial`
- `unsupported`

开发建议：

- 优先复用 `VLMAssemblyGenerator` 暴露的 helper，而不是直接改内部状态。
- 如果是模板型 lowering，优先走 `generator.env.template(...)` + `generator.wrap_template(...)`。
- 如果需要跨节点共享 symbol / parameter，优先走 `plena_shared` 模式。
- 但在同一条开发路径里，应尽量只选一种生成模式，不要把 `asm_templates` 路径和 `asm_lib` kernel compiler 路径混合使用。

### 7.3 把 handler 接到 asm_lib compiler

当模板式 lowering 不够表达复杂 kernel 时，可以在 handler 里调用 `asm_lib`。

这里的推荐分层是：

1. `handler`
   决定什么时候命中 lowering，并把调用记录中的元信息转换为 compiler 需要的输入。
2. `asm_lib/plena_program.py`
   适合用高层 DSL 描述算法。
3. `asm_lib/developer_compiler.py`
   适合封装更低层的 ISA 生成流程。
4. `asm_lib/sub_matrix_manager.py`
   适合补块布局、地址计算或 allocator 能力。

如果你要做新的共享 lowering，建议先看现有的 `Qwen3VLVisionMLP` 实现，它已经是一个比较完整的样板。

这里还要注意一条工程约束：

- 既然你已经决定走 `asm_lib` kernel compiler 路线，就尽量让这一类模块都沿着这条路线开发和测试。
- 不要一部分 handler 走旧 `asm_templates` 路线，另一部分再切到 `asm_lib` 路线并假设它们可以稳定混合工作。
- 当前仓库没有把这种混合使用作为受支持场景来验证。

### 7.4 扩展 parser

当模型调用过程抓不出来，或者元信息抽取不够时，需要先改 parser，而不是直接改 codegen。

典型修改点：

- [vlm_parser.py](vlm_parser.py) 中的模型加载分支
- `extract_model_info(...)`
- `trace_leaf_modules(...)`
- FX op 到逻辑操作的映射
- Qwen / VLM 输入模板逻辑，例如 `template_qwen3_vl_inputs(...)`

一个实用判断原则：

- 如果问题是“模块调用没有被抓到”，改 parser。
- 如果问题是“调用记录已经有了，但没有生成 asm”，改 type 注册或 handler。
- 如果问题是“handler 命中了，但 asm 太弱或太底层”，改 asm_lib。

## 8. 推荐开发流程

下面是一套比较稳妥的迭代方式：

1. 先准备一个最小可复现模型，最好是你自己在本地写的简单 `nn.Module`，这样更容易看清每一层实际发生了什么。
2. 用 parser 跑一遍模型，把这次运行产生的调用记录导出成 JSON。
3. 看清楚你要处理的那一类调用记录，包括它的 `type`、输入输出形状、权重形状和符号信息。
4. 先写一个最基本的 handler，让它至少能被正确调用；如果暂时还不能完整生成 asm，也可以先明确返回 `partial` 或 `unsupported`。
5. 再一步一步把这类模块的生成逻辑写完整，直到可以稳定输出你需要的 asm。
6. 写测试，确认这类输入下会进入正确的 handler，并且输出结果符合预期。
7. 最后再拿真实模型做集成验证。

这样做的原因很简单：这里真正容易出问题的地方，通常不是“把 asm 写出来”，而是“模型运行时记录下来的信息，和你写生成逻辑时假设的信息，到底是不是同一回事”。如果一开始就直接在真实模型上改，很难分清问题是出在模型解析、handler 选择，还是 asm 生成本身。

## 9. 测试方式

当前测试都在 [test/](test/) 下，代码是 `unittest` 风格。当前 `pyproject.toml` 没有默认声明 `pytest` 依赖，因此更稳妥的方式是直接用 `python` 执行测试文件；如果你的本地环境额外安装了 `pytest`，也可以用 `pytest` 跑。

在 [../](../) 目录下：

```bash
PYTHONPATH=.. uv run python multi_model/test/test_vlm_parser.py
PYTHONPATH=.. uv run python multi_model/test/test_vlm_codegen.py
PYTHONPATH=.. uv run python multi_model/test/test_utilization_report.py
```

如果你的环境里已经安装了 `pytest`，也可以使用：

```bash
cd generator
PYTHONPATH=.. uv run python -m pytest multi_model/test/test_vlm_codegen.py -v
```

推荐新增功能时至少补这两类测试：

- parser / 运行记录抓取测试
- codegen / handler 命中测试

## 10. 当前目录最重要的设计约束

开发时最好始终记住下面几点：

- `VLMAssemblyGenerator` 现在主要负责“调度、绑定、复用、分发”，不应该把所有 lowering 细节都继续塞回生成器里。
- 具体 lowering 逻辑应尽量下沉到 handler。
- 更复杂的 kernel 逻辑应继续下沉到 `asm_lib`。
- 模块调用记录里的 `type` 和 codegen operation key 是两层概念，不要混用。
- 当前工程同时保留 `asm_templates` 和 `asm_lib` 两条生成路径，但开发时应明确选择其一，不要把两条路径混合使用。

## 11. 阅读建议

如果你的目标是：

- “先跑通一遍”
  先看 [run.py](run.py)。
- “新增一个模块的 lowering”
  先看 [doc/handlerAPI_cn.md](doc/handlerAPI_cn.md) 和 [vlm_codegen_handlers.py](vlm_codegen_handlers.py)。
- “实现更底层的 PLENA kernel compiler”
  先看 [asm_lib/README_cn.md](asm_lib/README_cn.md)。
- “排查模型为什么不能编译”
  先看调用记录 JSON、`model_info` 和 handler 命中情况。
