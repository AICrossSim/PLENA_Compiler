# PLENA ASM Lib 使用说明

本文档说明 `plena_program.py`、`developer_compiler.py` 和 `sub_matrix_manager.py` 的分工、适用场景和推荐使用方式。

## 1. 三个文件的角色分层

### `plena_program.py`
- 这是上层 API，面向“写程序的人”。
- 它提供一个 Python 风格的 DSL，把 HBM 输入、VRAM tensor、FPRAM 标量和常见算子封装成对象与方法。
- 你通常应该从 `PLENAProgram` 开始，而不是直接手写 `DeveloperCompiler`。
- 典型职责：
  - 声明输入：`input()`
  - 从 HBM 加载到 VRAM：`load_batch()`
  - 分配中间结果：`alloc()`, `fp_var()`
  - 组织算子：`norm()`, `vram_sub_projection_to()`, `vram_add()`, `tile_row_*()`
  - 管理作用域与复用函数：`@prog.function`
  - 导出 ISA：`compile()`

### `developer_compiler.py`
- 这是下层编译后端，面向“写编译器/写底层 kernel 的人”。
- 它直接管理：
  - `SubMatrixManager` / symbol table
  - VRAM / HBM / FPRAM 地址
  - GP / Address / FP 寄存器分配
  - ISA 模板拼接和代码累计
- 典型职责：
  - 把上层操作变成具体 ISA
  - 保证 load/store/projection/softmax/FPRAM 运算的地址和寄存器合法
  - 提供更底层的 address-based helper，比如 `fpvar_*_asm()`、`tile_row_*_asm()`

### `sub_matrix_manager.py`
- 这是布局和地址计算中枢，位于 `PLENAProgram` 与 `DeveloperCompiler` 的下方。
- 它不负责“写程序流程”，而负责“定义对象如何在 HBM / VRAM / MRAM / FPRAM 中组织和寻址”。
- 典型职责：
  - 管理 HBM / VRAM / FPRAM 对象注册
  - 把大矩阵切成 `64 x 64` 子块
  - 预计算每个子块的 HBM offset、VRAM 地址、MRAM 装载位置
  - 管理 `VRAMAllocator`、`MRAMAllocator`、`FPRAMAllocator`
  - 生成块加载和块乘法相关的底层 ISA，如：
    - `load_sub_matrix_asm()`
    - `load_row_sub_matrices_asm()`
    - `load_col_sub_matrices_asm()`
    - `vram_sub_projection_asm()`
    - `vram_sub_projection_T_asm()`
    - `vram_block_add_asm()`

### 推荐理解方式
- `PLENAProgram` = 编程接口 / 调度层
- `DeveloperCompiler` = ISA 生成器 / 资源管理层
- `SubMatrixManager` = 布局、分块和地址后端

## 2. 你应该如何使用这三个文件

### 首选用法：只用 `PLENAProgram`
如果你的目标是“描述一个算子流程并生成 ISA”，优先使用 `PLENAProgram`。

推荐工作流：
1. 创建程序对象：`prog = PLENAProgram(mlen=64, blen=4)`
2. 声明 HBM 输入：`prog.input(...)`
3. 把需要参与计算的输入 load 到 VRAM：`prog.load_batch(...)`
4. 为中间结果分配 VRAM/FPRAM：`prog.alloc(...)`, `prog.fp_var(...)`
5. 调用上层算子 API 组织计算
6. 调用 `prog.compile()` 取出最终 ISA 字符串

这样做的好处：
- 少碰地址细节
- 不需要自己管理寄存器释放
- 命名空间和临时 tensor 生命周期更安全

### 进阶用法：通过 `prog.compiler` 下钻到底层
如果上层 API 缺一个操作，或者你要验证某段 ISA 模板的行为，可以从 `prog.compiler` 进入 `DeveloperCompiler`。

适合这种情况：
- 你要新增一个上层 API，但目前只有底层 kernel
- 你要调试 symbol table / VRAM 地址 / FPRAM 地址
- 你要直接拼一段 `*_asm` helper

建议模式：
- 仍然先用 `PLENAProgram` 管输入和对象生命周期
- 只把缺失的那一小段逻辑下放到 `prog.compiler`
- 不要整段程序都绕过 `PLENAProgram`，否则你要自己承担对象注册、命名和资源释放问题

### 更底层用法：直接使用 `SubMatrixManager`
只有在以下情况，才建议直接碰 `SubMatrixManager`：
- 你要调试块地址是否正确
- 你要验证 HBM 行主序和 VRAM 列块优先之间的布局差异
- 你要新增新的块级 ISA 生成 helper
- 你要检查 allocator 的复用、碎片回收和对齐行为

通常不建议把它当成主入口，因为它不负责上层程序语义，只负责内存对象和块布局。

## 3. 这三个文件的核心接口

### `PLENAProgram` 常用入口
- `input(name, shape, hbm_addr=None)`
  - 在 HBM 中声明输入张量。
- `load_batch(input_var, name=None)`
  - 生成从 HBM 到 VRAM 的 load ISA。
- `alloc(name, rows, cols)`
  - 分配 VRAM 中间矩阵。
- `store(tensor_var, name=None, hbm_addr=None)`
  - 把 VRAM 结果写回 HBM。
- `fp_var(name, size=1)`
  - 在 FPRAM 中声明标量或标量数组。
- `norm()`, `rms_norm()`, `layer_norm()`
  - 对 VRAM tensor 原地归一化。
- `tile_row_max/sum/exp/reci/sub_fp/mul_fp/...`
  - 面向 tile 行的规约或逐行标量运算。
- `vram_sub_projection_to()`
  - `A[row_block] @ W[:, col_block]`
- `vram_sub_projection_T_to()`
  - `A[row_block] @ W[row_block]^T`
- `vram_add()`, `vram_block_add_to()`
  - 做矩阵或块级累加。
- `init_online_softmax()`, `online_softmax_block()`, `compute_pv()`, `scale_o_row()`, `final_scale_o()`
  - 为 flash attention 提供专门流程。
- `compile()`
  - 返回累积生成的 ISA。

### `DeveloperCompiler` 常用入口
- `add_hbm_object()`
- `load_batch()`
- `store_to_hbm()`
- `allocate_vram_matrix()`
- `allocate_fpram()`
- `tile_row_*()`
- `fpram_*()`
- `vram_sub_projection_to()`
- `vram_sub_projection_T_to()`
- `init_online_softmax()`
- `online_softmax_block()`
- `compute_pv()`
- `get_code()`

这些接口更偏底层，参数多是内部名字、地址或 block 索引。

### `SubMatrixManager` 常用入口
- `add_hbm_object()`
- `add_vram_object()`
- `add_fpram_object()`
- `free_hbm_object()`
- `free_vram_object()`
- `free_fpram_object()`
- `register_matrix()`
- `register_vram_matrix()`
- `get_sub_block()`
- `get_row_blocks()`
- `get_col_blocks()`
- `get_vram_sub_block()`
- `compute_hbm_offset()`
- `compute_absolute_hbm_addr()`
- `load_sub_matrix_asm()`
- `load_row_sub_matrices_asm()`
- `load_col_sub_matrices_asm()`
- `vram_sub_projection_asm()`
- `vram_sub_projection_T_asm()`
- `vram_block_add_asm()`
- `generate_address_table()`
- `print_table()`
- `print_layout()`

这些接口的核心价值不是“表达算法”，而是“表达布局和地址”。

## 4. 实际推荐的开发边界

### 只做模型/算子编排时
用 `PLENAProgram`。

### 新增一个新算子时
推荐顺序：
1. 先看 `SubMatrixManager` 是否已经有合适的块布局和块级 ISA helper
2. 如果没有，就先在 `SubMatrixManager` 增加地址/块级能力
3. 在 `DeveloperCompiler` 里把这部分能力封装成更完整的编译步骤
4. 在 `PLENAProgram` 里包一层用户友好的 API
5. 在测试文件里用 `PLENAProgram` 调用验证

这是这个目录当前最自然的扩展方式。

## 5. `sub_matrix_manager.py` 的关键能力

### 1. 统一对象模型
`SubMatrixManager.__getitem__()` 会把 HBM / VRAM / FPRAM 信息合并成统一的 `MemoryObjectInfo` 视图。

这意味着：
- 同一个名字可以同时有 HBM 布局和 VRAM 布局
- `DeveloperCompiler` 不需要自己维护多套表
- `PLENAProgram.symbol_table` 本质上也是在看这里

### 2. 三类 allocator
- `VRAMAllocator`
  - 管理 VRAM 连续空间，按 `MLEN=64` 对齐。
- `MRAMAllocator`
  - 管理 MRAM 子块空间，按 `MLEN * MLEN = 4096` 对齐。
- `FPRAMAllocator`
  - 管理 FPRAM 标量空间，支持 `save_state()` / `restore_state()`。

它们都建立在 `VirtualMemoryManager` 的 best-fit + free-list + coalesce 模型上。

### 3. HBM 和 VRAM 的布局差异被显式编码
- HBM：二维矩阵按 row-major 连续存储
- VRAM：按 `[batch, mlen, hidden/mlen]` 的列块优先格式组织

这正是为什么：
- HBM 子块地址主要靠 `hbm_base + hbm_offset`
- VRAM 子块地址主要靠 `vram_base + col_block * batch * mlen + row_block * mlen * mlen`

### 4. 子块级寻址能力
`MatrixBlockLayout` 和 `VRAMMatrixBlockLayout` 会预先把整个大矩阵拆成多个 `64 x 64` 子块，并记录：
- 属于哪个父矩阵
- 行块 / 列块索引
- HBM offset 或 VRAM address
- 是否已经装到 MRAM

这使 `DeveloperCompiler` 可以直接拿块信息生成 ISA，而不是运行时手算地址。

### 5. 块级 ISA 生成能力
`SubMatrixManager` 已经直接提供底层块操作代码生成：
- 从 HBM 到 MRAM 的块加载
- 一整行块 / 一整列块加载
- `VRAM row-block @ MRAM col-block`
- `VRAM row-block @ MRAM row-block^T`
- `VRAM block + VRAM block`

所以很多 `DeveloperCompiler` 方法本质上是在做：
1. 调 allocator / 注册布局
2. 检查块已加载
3. 调 `SubMatrixManager` 的 `*_asm()`
4. 释放寄存器并累计代码

## 6. 最小示例

```python
from plena_program import PLENAProgram

prog = PLENAProgram(mlen=64, blen=4)

# 1. HBM 输入
x = prog.input("X", shape=(128, 128))
w = prog.input("W", shape=(128, 128))

# 2. 把需要的输入加载到 VRAM
x_vram = prog.load_batch(x, name="Xv")

# 3. 分配结果矩阵
y = prog.alloc("Y", 128, 128)

# 4. 做一个子块投影：Y[0][0] = Xv[0][:] @ W[:][0]
prog.vram_sub_projection_to(
    vram_matrix=x_vram,
    vram_row_idx=0,
    mram_input=w,
    mram_col_idx=0,
    target=y,
    target_row_idx=0,
    target_col_idx=0,
)

# 5. 导出 ISA
asm = prog.compile()
print(asm)
```

## 7. 调试布局与地址时的建议用法

如果你不是在写完整程序，而是在查地址问题，直接看 `SubMatrixManager` 会更有效。

例如你可以重点用：
- `print_table()`
- `print_layout(name)`
- `generate_address_table(name)`
- `get_sub_block(name, r, c)`
- `get_vram_sub_block(name, r, c)`

这类操作适合验证：
- 某个矩阵是否真的注册了
- 某个子块应该落在 HBM / VRAM 的哪个地址
- 某个块是否已经被加载到 MRAM

## 8. 一个更贴近仓库现状的使用模式

当前目录下的 `flash_attention_plena_test.py` 展示了最完整的参考路径：
- 用 `PLENAProgram` 声明 `Q/K/V`
- `load_batch(Q)` 把 Q 放进 VRAM
- 用 `alloc()` 创建 `S/PV/O`
- 循环调用：
  - `vram_sub_projection_T_to()`
  - `online_softmax_block()`
  - `compute_pv()`
  - `scale_o_row()`
  - `vram_add()`
  - `final_scale_o()`
- 最后 `compile()`

如果你要写新的 attention / linear / norm 流程，建议直接仿照这个文件的组织方式。

## 9. 需要特别注意的点

### 1. `PLENAProgram` 是 eager 的
- 每调一次 API，就会立刻把对应 ISA 追加到内部字符串。
- `compile()` 不是“开始编译”，而是“取出已经累计好的代码”。

### 2. `@` 运算符当前不能用
- `TensorVar.__matmul__` 最终会走 `_dispatch_matmul()`
- 但 `_dispatch_matmul()` 明确直接抛错
- 所以你应该用显式 API，如：
  - `vram_sub_projection_to()`
  - `vram_sub_projection_T_to()`

### 3. 名字分成 display name 和 internal name
- 尤其在 `@prog.function` 作用域里，内部名字会自动加前缀
- 这是为了避免重复调用同一函数时名字冲突

### 4. HBM 大小不是简单的 `h * w`
- 当前默认考虑 MXFP 的 `real_data_ratio = 1.125`
- 所以 HBM 分配和 VRAM 逻辑大小并不完全一样

### 5. 直接使用 `DeveloperCompiler` 时，必须自己保证对象已注册
- 比如 `load_batch()` 之前，需要先有 HBM object
- 比如做 projection 前，需要目标 VRAM matrix 已分配

### 6. 绝大多数 shape 必须是 `mlen` 的整数倍
- `SubMatrixManager.register_matrix()` 要求 HBM 矩阵行列都能被 `mlen` 整除
- `register_vram_matrix()` 也要求 batch 和 hidden 都能被 `mlen` 整除
- 当前这套块级编译假设非常强，不是任意 shape 都能直接喂进去

### 7. MRAM 里的块必须先 load，后 compute
- `vram_sub_projection_asm()` 和 `vram_sub_projection_T_asm()` 都依赖对应的 MRAM 子块已经装载
- 如果块没有先通过 `load_row_sub_matrices_asm()` / `load_col_sub_matrices_asm()` 或对应编译器接口装进去，会直接报错

### 8. allocator 支持复用，但名字管理仍然重要
- `free_*()` 之后空间会回收到 free-list
- 后续分配可能复用旧地址
- 所以调试地址时，必须结合对象名字和当前生命周期一起看，不能只看“某个地址以前是谁”

## 10. 如果你要扩展这套系统

最稳妥的做法是：
1. 先看 `SubMatrixManager` 是否已有足够接近的布局/块级 helper
2. 若没有，先补 `sub_matrix_manager.py`
3. 再在 `developer_compiler.py` 里组装成完整的编译步骤
4. 最后在 `plena_program.py` 提供用户接口
5. 参考 `flash_attention_plena_test.py` 写验证样例

## 11. 一句话建议

日常写 PLENA 程序时，用 `PLENAProgram`。
只有在“新增底层能力”或“调试地址/寄存器/块布局/ISA 模板”时，才下钻到 `DeveloperCompiler` 和 `SubMatrixManager`。

## 12. TODO: Qwen3-VL 需要支持的模块

以下 TODO 根据 `generator/.venv/lib/python3.13/site-packages/transformers/models/qwen3_vl/modeling_qwen3_vl.py` 中的模块定义和前向路径整理，目的是反推 PLENA 编译器后续需要覆盖的能力。

### P0: 文本主干必须支持
- `[ ]` `Qwen3VLTextRMSNorm`
  - 文本模型和 attention 内部都依赖 RMSNorm。
- `[ ]` `Qwen3VLTextAttention`
  - 需要支持 `q_proj / k_proj / v_proj / o_proj`
  - 需要支持 causal self-attention
  - 需要支持 GQA/MQA 风格的 `num_attention_heads / num_key_value_heads`
  - 需要支持 `q_norm` 和 `k_norm` 这两个按 head_dim 做的 RMSNorm
- `[ ]` Text RoPE
  - 需要支持文本分支使用的 rotary position embedding
  - 最低要求是支持 Q/K 上的 rotary 应用
- `[ ]` `Qwen3VLTextMLP`
  - 这是 gated MLP：`down_proj(act(gate_proj(x)) * up_proj(x))`
  - 不等价于普通两层 FFN，必须支持逐元素乘
- `[ ]` `Qwen3VLTextDecoderLayer`
  - 结构是：
    - input RMSNorm
    - self-attention
    - residual add
    - post-attention RMSNorm
    - gated MLP
    - residual add
- `[ ]` `embed_tokens`
  - 文本 token embedding 查表
- `[ ]` final `norm`
  - 文本模型最后一层 RMSNorm
- `[ ]` `lm_head`
  - 线性输出到 vocab logits

### P0: 视觉主干必须支持
- `[ ]` `Qwen3VLVisionPatchEmbed`
  - 本质是带 stride 的 `Conv3d`
  - 对视频/图像 patch 做时空分块嵌入
- `[ ]` 视觉位置编码
  - 需要支持 learned 2D position embedding 插值
  - 需要支持 vision rotary embedding
- `[ ]` `Qwen3VLVisionAttention`
  - 需要支持 vision self-attention
  - 需要支持 `qkv` 融合线性层和 `proj`
  - 需要支持 non-causal attention
  - 需要支持 variable-length 序列块，至少在编译视角要支持分块 attention
- `[ ]` `Qwen3VLVisionMLP`
  - 标准两层 MLP：`fc1 -> act -> fc2`
- `[ ]` `Qwen3VLVisionBlock`
  - 结构是：
    - LayerNorm
    - self-attention
    - residual add
    - LayerNorm
    - MLP
    - residual add
- `[ ]` `Qwen3VLVisionPatchMerger`
  - 视觉塔输出后需要 patch merge
  - 包含 LayerNorm、Linear、GELU、Linear
  - deepstack 分支也复用同类 merger

### P1: 多模态桥接必须支持
- `[ ]` image/video placeholder 替换
  - 需要把视觉 encoder 输出写回到文本 `inputs_embeds` 的占位符位置
- `[ ]` DeepStack visual feature injection
  - 视觉塔会额外输出若干 deepstack features
  - 文本侧 early hidden states 会整合这些视觉特征
- `[ ]` 3D / M-RoPE position ids
  - Qwen3-VL 文本序列不是纯 1D 位置
  - 需要支持文本 token + 视觉 token 的 3D position ids 组织
- `[ ]` image/video feature splitting and merge
  - 视觉输出会按 `grid_thw` 重新切分，再映射回多模态 token 序列

### P1: 编译器算子级能力 TODO
- `[ ]` Linear family
  - 普通 `Linear`
  - 融合 `QKV` 线性层
  - gated MLP 所需的双分支线性层
- `[ ]` Norm family
  - `RMSNorm`
  - `LayerNorm`
- `[ ]` Activation family
  - `SiLU` 或模型配置对应文本激活
  - `GELU` 用于 vision patch merger
  - vision/text MLP 需要的通用激活算子
- `[ ]` Elementwise family
  - add
  - mul
  - residual add
  - gated mul
- `[ ]` Attention family
  - causal self-attention
  - non-causal self-attention
  - GQA/MQA
  - QK RoPE
  - softmax
  - attention @ V
- `[ ]` Embedding family
  - token embedding
  - learned position embedding gather
- `[ ]` Vision-specific family
  - patchify / patch embedding
  - patch merge / reshape / shuffle
  - 2D/3D positional encoding支持

### P1: 图编排级模块 TODO
- `[ ]` Vision encoder stack
  - `PatchEmbed -> PosEmbed -> N x VisionBlock -> PatchMerger`
- `[ ]` Text decoder stack
  - `Embedding -> N x TextDecoderLayer -> RMSNorm -> LMHead`
- `[ ]` Multimodal fusion path
  - vision features 编码
  - placeholder 替换到文本 embedding
  - DeepStack feature 注入
  - unified decoder forward

### P2: 推理机制与系统能力 TODO
- `[ ]` KV cache
  - `Qwen3VLTextAttention` 前向明确依赖 `past_key_values.update(...)`
- `[ ]` generation path
  - 至少需要支持 prefill / decode 两阶段
- `[ ]` 动态 position update
  - 生成时需要基于 `rope_deltas` 和 cache position 更新 position ids
- `[ ]` variable-length visual sequence handling
  - 视觉 attention 使用 `cu_seqlens`
  - 编译器侧至少需要能表达“变长块分段 attention”

### 当前建议的落地优先级
1. 先把文本主干做完整：
   - `Embedding + RMSNorm + TextAttention + TextMLP + DecoderLayer + LMHead`
2. 再补 attention 细节：
   - GQA
   - RoPE
   - KV cache
3. 再接视觉主干：
   - PatchEmbed
   - VisionBlock
   - PatchMerger
4. 最后做多模态桥接：
   - placeholder 替换
   - DeepStack
   - 3D/M-RoPE
