# PLENA Compiler：Mamba / KDA 支持

这个分支让 PLENA Compiler 能描述并生成两类混合模型的执行程序：

- **Nemotron 3 Nano**：52 层，包括 23 层 Mamba、23 层 MoE、6 层 Attention。
- **Kimi K3**：93 层，包括 69 层 KDA、24 层 MLA，以及对应的 MoE。

Compiler 负责生成指令、内存地址和 descriptor；真正执行这些指令的是
[PLENA Simulator](https://github.com/AICrossSim/PLENA_Simulator/tree/feature/mamba-kda-support)。

## 架构图

三张图分别回答三个问题：PLENA 整体增加了什么、一个 Mamba token 怎么执行，以及
`L_SCATTER_M` 为什么能减少 Matrix SRAM bank conflict。

### Figure 1：Hybrid PLENA 总体架构

![Hybrid PLENA architecture for Mamba, Attention, and MoE](docs/architecture/hybrid_plena_mamba.svg)

[SVG](docs/architecture/hybrid_plena_mamba.svg) · [论文用 PDF](docs/architecture/hybrid_plena_mamba.pdf)

这张图保留原 PLENA 的 Scalar、Matrix、Vector、SRAM 和 HBM 主体，只在真实的
Matrix 写回与状态数据路径上加入 Mamba 扩展。同一套 Matrix/Vector 单元继续执行
Attention、MoE 和 Mamba 的矩阵计算。输入投影完成后，`L_SCATTER_M` 根据 `X_STATE`
下一拍的读取方式重新安排 SRAM bank；`X_STATE` 再从独立的 persistent state cache
读取并更新 recurrent state。图中虚线模块已在 Compiler 和 Rust Simulator 中实现，
但尚未进入 RTL/PPA。

### Figure 2：Nemotron 3 Mamba-2 decode 数据流

![Nemotron 3 Mamba-2 decode step and L-Compute co-layout](docs/architecture/mamba2_lcompute_flow.svg)

[SVG](docs/architecture/mamba2_lcompute_flow.svg) · [论文用 PDF](docs/architecture/mamba2_lcompute_flow.pdf)

这张图展示一个真实尺寸的 Nemotron 3 Mamba decode token。`gate` 留在 Vector SRAM，
`x/B/C/dt` 经过 16-bank group-major skew 排布后交给 state engine；该排布只改变物理
bank 地址，不改变 tensor 数值，也不产生 HBM transpose/repack 流量。图中的 3.568x
是 projection-buffer 的局部读取服务提升，不是整层或整模加速比。

### Figure 3：L-Compute 如何消除热点 bank

![Compiler-guided bank co-layout for Mamba state packets](docs/architecture/lcompute_bank_colayout.svg)

[SVG](docs/architecture/lcompute_bank_colayout.svg) · [论文用 PDF](docs/architecture/lcompute_bank_colayout.pdf)

这张图只解释 bank mapping 机制。`X_STATE` 同一拍需要 8 个 head 的 4 个 P-lane；
row-major 会让这些值反复落到 bank 0–3。`L_SCATTER_M` 在 Matrix 写回时按 head
将地址平移 4 个 bank，使同一个 32-value packet 均匀分布到 16 个 single-port bank。
逻辑 tensor 和
消费顺序都不变，也不需要经过 HBM 做 transpose。

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
| Compact Matrix 循环 | 完成 | MXFP8 GEMM 与 BF16 stream-K 均在 Rust 中逐元素一致 |
| Nemotron 52 层机器码 | 完成 | 6,202,663 条、23.66 MiB；参数由 symbolic HBM manifest 描述 |
| Kimi 93 层机器码 | 完成 | 96-head 共 11,502,370 条，原始机器码 43.88 MiB |
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
  aten/tests/test_compact_matrix_loops.py \
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

生成完整单 token decode 机器码和 symbolic-weight 地址清单：

```bash
uv run python -m aten.nemotron3.artifact build/nemotron3-decode
uv run python -m aten.kimi3.artifact build/kimi-k3-decode
```

每个目录包含 `.mem`、state/layout descriptors、FPRAM 常量、symbolic HBM
manifest、SHA256 和一份 summary。Kimi 的全量构建约需 2 分钟和 5 GiB 主存；它不会
下载 checkpoint。

这些命令不下载模型权重。`.mem` 中的指令已经逐条编码，但 manifest 中的参数仍是
待填充地址，所以不代表真实 checkpoint 已经从第一层执行到最后一层。

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
