# PLENA Compiler：Mamba / KDA 支持

这个分支让 PLENA Compiler 能描述并生成两类混合模型的执行程序：

- **Nemotron 3 Nano**：52 层，包括 23 层 Mamba、23 层 MoE、6 层 Attention。
- **Kimi K3**：93 层，包括 69 层 KDA、24 层 MLA，以及对应的 MoE。

Compiler 负责生成指令、内存地址和 descriptor；真正执行这些指令的是
[PLENA Simulator](https://github.com/AICrossSim/PLENA_Simulator/tree/feature/mamba-kda-support)。

## 数据怎么走

```text
hidden state
  -> Matrix projection
  -> L_SCATTER_M：按下一步的读取方式把数据放进不同 SRAM bank
  -> X_STATE：执行 Mamba-2 或 KDA 状态更新
  -> Vector / Matrix：门控、归一化、输出投影
  -> Attention / MLA / MoE / residual
```

`L_SCATTER_M` 只改变数据在 SRAM bank 中的物理位置，不改变 tensor 数值；
`X_STATE` 用同一套指令格式执行 Mamba-2 和 KDA，两者通过 descriptor 区分。

## 当前进度

| 内容 | 状态 | 说明 |
|---|:---:|---|
| Mamba-2 / KDA 指令和 descriptor | 完成 | Compiler 与 Simulator 使用同一份二进制格式 |
| `L_SCATTER_M` bank 排布 | 完成 | 支持 row-major、transpose、Mamba、KDA 和自定义模式 |
| Nemotron 3 层结构 | 完成 | 23 Mamba + 23 MoE + 6 Attention，完整程序可汇编 |
| Kimi K3 层结构 | 完成 | 69 KDA + 24 MLA 的结构 trace 已生成 |
| Matrix、Attention、MoE、residual 连接 | 完成 | 缩小外围尺寸的程序已在 Rust Simulator 数值对拍 |
| 完整 Kimi 93 层机器码 | 未完成 | 大 GEMM 和 96-head MLA 还需要硬件循环，当前会明确拒绝生成 |
| Prefill 和多 token MLA/GQA cache | 未完成 | 当前 connected 路径只验证单 token decode |
| 真实权重整模执行 | 未完成 | 不能声称 Nemotron/Kimi 已在 PLENA 从第一层跑到最后一层 |
| RTL、频率、面积和端到端加速比 | 未开始 | 属于下一阶段，不是本仓库当前结论 |

## 快速开始

```bash
git clone --branch feature/mamba-kda-support \
  https://github.com/AICrossSim/PLENA_Compiler.git
cd PLENA_Compiler
uv sync --frozen
```

运行核心测试：

```bash
uv run pytest -q -m "not slow" \
  assembler/tests/test_x_state_encoding.py \
  assembler/tests/test_l_scatter_m_encoding.py \
  aten/tests/test_hybrid_substrate.py \
  aten/tests/test_nemotron3_hybrid.py \
  aten/tests/test_kimi_k3_hybrid.py
```

完整的分支测试列表见 [CI 配置](.github/workflows/ci.yml)。TileLang/TVM 不是
默认依赖；只有开发 TileLang 路径时才运行 `uv sync --frozen --group tvm`。

## 生成模型 trace

Nemotron 3 decode：

```bash
uv run python -m aten.nemotron3.trace \
  --phase decode --decode-tokens 1 \
  --output build/nemotron3.json \
  --mamba-physical-output build/nemotron3-mamba.json \
  --mamba-physical-asm-output build/nemotron3-mamba.s
```

Kimi K3 decode：

```bash
uv run python -m aten.kimi3.trace \
  --phase decode --batch-size 1 --state-cache-mib 32 \
  --output build/kimi-k3.json
```

这些命令不下载模型权重。输出用于检查层顺序、指令、descriptor 和内存布局，
不代表完整模型已经执行。

## 代码位置

| 目录 | 内容 |
|---|---|
| `aten/state/` | 共用的 `X_STATE`、`L_SCATTER_M`、内存和 residency 规则 |
| `aten/mamba/`、`aten/kda/` | 两种状态模型的调度和 projection lowering |
| `aten/nemotron3/`、`aten/kimi3/` | 两个真实模型的层结构和 connected program |
| `spec/` | Compiler/Simulator 共用的二进制 contract 和 golden 数据 |
| `doc/` | ISA、地址布局和未完成项的详细设计说明 |

## 结果应该怎么理解

本分支证明的是：Compiler 能生成一致的指令和数据布局，并把相邻计算阶段接起来。
它还没有证明真实模型精度、RTL 时序、FPGA PPA 或相对 GPU 的端到端加速。
