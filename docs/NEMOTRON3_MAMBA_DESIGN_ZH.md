# PLENA 支持 Nemotron 3 Mamba：从代码到硬件的完整设计

## 先用一句人话说明

PLENA 原来擅长做大矩阵乘法；Nemotron 3 的 Mamba 层除了两个大矩阵乘法，
还要保存一份“读完这个 token 后的记忆”。我们保留原 Matrix/Vector 硬件做
大矩阵和普通算子，只增加一个小型 state engine 负责卷积、更新记忆和产生
Mamba 输出，再由 Compiler 管理这份记忆什么时候留在片上、什么时候搬到 HBM。

## 真实工作负载，不是 toy Mamba

NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 的公开配置是：

| 项目 | 真实值 | 对硬件的含义 |
|---|---:|---|
| 总层数 | 52 | 不能假设每层都是 Attention。 |
| Mamba / MoE / Attention | 23 / 23 / 6 | 三类层必须共用 PLENA，而不是做一个独立纯 Mamba 芯片。 |
| hidden size | 2688 | Mamba 输入、输出都是 2688。 |
| Mamba heads / head dim | 64 / 64 | 中间 x 一共 4096 个值。 |
| groups / heads per group | 8 / 8 | 每个 B/C 只生成一次，供同组 8 个 head 共享。 |
| state dim | 128 | 每个 `(head, p)` 要保存 128 个状态。 |
| conv kernel | 4 | 还要保存最近 4 个 x/B/C 输入。 |
| projection width | 10304 | `gate 4096 + x/B/C 6144 + dt 64`。 |

单个 Mamba 层的 FP32 SSM state 是 2 MiB，conv state 是 96 KiB。23 层一
个 request 合计 48.16 MiB。如果每个 token 都从 HBM 读回来再写回去，仅
state 就产生约 96.31 MiB/token 的流量。

公开配置和实现：

- [NVIDIA Nemotron 3 config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json)
- [NVIDIA Nemotron 3 model implementation](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/modeling_nemotron_h.py)
- [Mamba-2 / SSD paper](https://arxiv.org/abs/2405.21060)

## 一个 decode token 到底怎么走

以 request 0、某个 Mamba layer 为例：

1. Compiler 检查这个 layer 的 state 是否在片上。命中就直接用；没命中就
   发 `STATE_PREFETCH`，把 SSM state 和 conv state 搬入一个 slot。
2. 现有 Matrix machine 做 `2688 -> 10304` 的 in-projection，产生
   `gate/x/B/C/dt`。
3. Matrix 结果写回时，L-compute 同时把结果从线性顺序 scatter 到一个小型
   Mamba projection buffer 的 group-major bank 顺序。它只搬位置，不做近似、
   不改变数值。
4. `STEP` 启动 state engine：
   `conv1d -> softplus(dt) -> exp(dt*A) -> state = state*dA + dt*B*x`。
5. 同一条流水线马上做 `y = state*C + D*x`。更新后的 state 不需要先写回
   SRAM 再读一次，所以叫 fused state-update/output。
6. Mamba sequencer 借用现有 Vector service 做 `SiLU(gate) + group RMSNorm`，
   产生 4096 个输出值，不在 state engine 里复制一套 RMSNorm reduction。
7. 现有 Matrix machine 做 `4096 -> 2688` 的 out-projection，然后进入
   residual 和下一个 Nemotron 层。
8. state 留在 cache 时只标记 dirty；需要换出时先 `STATE_COMMIT`，再
   `STATE_EVICT`。Compiler 禁止直接丢掉 dirty state。

## state engine 应该多大

建议第一版参数是：

```text
8 head lanes x 4 head-dim lanes x 8 state-dim lanes
= 256 state elements/cycle
```

一个 Mamba 层有 `64 x 64 x 128 = 524,288` 个 state elements，所以理想下限
约 2,048 cycle/layer。这里的“8 个 head”正好对应一个 group；B/C 每拍各读
8 个 state-dim 值，然后广播给 8 个 head，不应该复制读八遍。

这个数字不是最终 RTL 参数。DSE 必须扫描 128/256/512 elements/cycle，并把
DSP、频率、state-cache 带宽和 projection 时间一起比较。

### state cache 具体怎样供出 256 个值

建议把 256 lanes 组织成：

```text
32 logical banks = 8 head lanes x 4 p lanes
each bank word    = 8 consecutive N values
each bank         = 1 read + 1 write per cycle
```

地址可以写成：

```text
bank = local_head * 4 + p_lane
row  = (cache_slot, group, p_tile, n_tile)
```

每拍 32 个 bank 各读一个 8-value word，得到 256 个旧 state；完成 update 和
C multiply 后同拍写回另一个 1R1W port。一个 group 有 16 个 `p_tile` 和 16 个
`n_tile`，所以是 256 cycle；8 group 合计 2,048 cycle。

调度顺序用 `n_tile` 外层、`p_tile` 内层。每个 group 的 512 个 x 保存在约
1 KiB local buffer，512 个 FP32 C-reduction accumulator 约 2 KiB；这样每个
B/C n-tile 只读一次，然后向 8 head 广播，不会为了省 B/C 又反复读 x。

## “斜着存”到底是什么

SRAM 有 16 个 bank，可以把它理解成 16 个独立抽屉。一个 bank 一拍通常只能
取一个地址。普通 row-major 下，8 个 head 的相同 4 个 `p` 很容易反复落到
同几个 bank，state engine 就会等。

我们采用可逆的循环偏移：

```text
x/gate bank = (p_in_row + local_head * 4) mod 16
B bank      = n mod 16
C bank      = (n + 8) mod 16
dt bank     = local_head
```

每个 group 和每个字段都从新的 16-bank row 开始；8 个 dt 后补 8 个空位置，
防止下一个 group 与它共用一行。真实 10,304 个 projection 值只增加 64 个
padding，物理容量是 10,368，开销约 0.62%。

结果是：

- 32 个 x（8 head x 4 p）均匀分到 16 bank，两拍完成，达到单端口理论下限。
- 8 个 B 和 8 个 C 正好覆盖 16 bank，一拍完成。
- 映射对全部 10,304 个逻辑值是 one-to-one，不存在两个值写到同一物理地址。

这里的 L-compute 是**新增的布局单元定义**，不是 main RTL 已经存在的模块。
它应接在 `matrix_machine -> Vector SRAM` 的结果 drain 旁边，把 Mamba
in-projection 同步写入一个 16-bank projection buffer。一个 token 的物理布局
是 10,368 个值：BF16 约 20.25 KiB，decode ping-pong 约 40.5 KiB。Prefill 不
保存整个 128-token chunk，而是按 token/tile 流过这个 buffer。

Matrix drain 可能一次 burst 出 64 个值，而 16 个单写口 bank 一拍只能 scatter
16 个值。L-compute 前面必须有至少 256-value 的 write FIFO，用 4 拍消化一个
64-value burst；DSE/RTL 要记录 `projection_fifo_stall_cycles`。如果真实 drain
平均速度超过 16 value/cycle，就要改成 32 bank、双写口或每 bank 更宽，不能
在 analytic model 里把 layout write 当成零成本。

### 为什么不直接改现有 Matrix SRAM

最新 RTL 的真实数据路是：Matrix SRAM 从 HBM 收 MX 权重，Matrix machine 的
结果写入 Vector SRAM。Matrix SRAM 没有 Matrix-result/Vector 写口；硬塞进去
还要增加 FP/BF16 到 weight-MX 格式转换和新的仲裁。它已有普通 row/column
gather 与 `transposed_read`，所以“再做一次 transpose”本身也不是新贡献。

可选方案的取舍是：

| 方案 | 改动 | 结论 |
|---|---|---|
| 新增小型 banked projection buffer | 在 Matrix writeback 旁路 scatter；state engine 独占读口 | **推荐**，数据路短、容易做 ablation。 |
| 改 Vector SRAM | 给现有双口宽 SRAM增加 skew write 和多-bank gather | 可行但侵入大，会影响 Matrix/Vector 正常调度。 |
| 强行复用 Matrix SRAM | 新增结果写口、格式转换、权重/activation 仲裁 | 不推荐，成本高且与现有职责冲突。 |

因此论文可以叫 “group-major skewed projection SRAM/buffer”，不应在 RTL 没有
对应数据路时声称“Matrix SRAM 已经支持斜存”。

## 为什么 state 不放 Matrix SRAM

Matrix SRAM 还要给 Attention、MoE 和两个 projection 使用。完整 Mamba state
每层超过 2 MiB，塞进去会挤掉权重/activation，并争用同一读口。正确拆分是：

| 存储 | 放什么 | 原因 |
|---|---|---|
| Matrix SRAM | projection/Attention/MoE 的 MX 权重 tile | 保留现有职责和 row/column gather。 |
| Mamba projection buffer | 当前 token/tile 的 gate、x、B、C、dt | 生命周期短，接 Matrix result bypass 并按 group skew。 |
| banked state cache | persistent SSM + conv state | 每个 token 都复用，需要专用高带宽端口。 |
| HBM | 放不下的 request/layer state、全部权重 | 容量大但延迟和流量高。 |

对顺序访问 23 个 Mamba layer，容量小于 23-layer working set 的普通 LRU 会
发生循环抖动，第二个 token 仍可能 0 hit。Compiler 应优先固定 pin 一部分
`(request, layer)`，其余 state 走 transient stream slot，而不是盲目用 LRU。

## Prefill 和 decode 不是同一条快路径

- decode 一次只有一个 token，必须做低延迟 recurrent `STEP`。
- prefill 有一长串 token，`PREFILL` 应按 128-token chunk 做 SSD affine scan：
  先算 chunk 内部，再组合 chunk 边界 state，最后产生 chunk 输出。
- 第一版 RTL 可以先让 `PREFILL` 顺序执行来验证正确性；论文版本必须实现并
  比较 chunked SSD，否则长 prompt 性能不成立。

CPU golden reference 已提供逐 token prefill；通用 chunked affine scan 会与
逐步 recurrence 逐值比较，给 Simulator/RTL 一个不依赖 GPU 的正确答案。

## Mixed precision 应该怎么定义

第一阶段只量化“存下来的 state”，不量化 FP32 更新过程：

| 模式 | 存储 | 更新/exp/reduction | 目的 |
|---|---|---|---|
| FP32 | FP32 | FP32 | golden baseline |
| BF16 | BF16 round-trip | FP32 | 2x state 容量 |
| FP16 | FP16 round-trip | FP32 | 对比动态范围问题 |
| MX8-B128 | E4M3FN + 每 128 state 一个 PoT scale | FP32 | 约 3.97x state 容量 |

MX8-B128 是本项目候选格式。OCP MXFP8 的标准 block 是 32，不能在论文里把
B128 写成完全标准的 OCP MXFP8。参考规范：
[OCP Microscaling Formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)。

最终精度结论必须在真实权重和真实 validation prompts 上报告 perplexity/task
accuracy；目前 CPU synthetic recurrence 只能筛掉明显不稳定的格式，不能代替
模型精度实验。

当前 CPU 初步结果如下。长实验使用 `H=8, P=8, N=128, 512 tokens` 的代表性
tile；另一个实验使用完整 `H=64, P=64, N=128` 跑 32 tokens。输入是固定 seed
的稳定随机 recurrence，不是 Nemotron 权重：

| 存储 | 压缩率 | 512-token output relative-L2 | real-shape 32-token output relative-L2 |
|---|---:|---:|---:|
| BF16 | 2.00x | 0.000795 | 0.000505 |
| FP16 | 2.00x | 0.0000987 | 0.0000632 |
| MX8-B128 | 3.97x | 0.0148 | 0.00823 |

这说明 BF16/FP16 可以继续做真实模型验证；MX8-B128 目前误差明显更大，不能
直接冻结进 RTL。下一轮应比较 block-32/64/128、scale 选取方法，以及是否只对
部分 layer/head 使用 MX8。

## Compiler 调度和 ISA

只新增 `X_MAMBA=0x39`，sub-op 为：

`STATE_PREFETCH / STATE_RESET / PREFILL / STEP / STATE_COMMIT / STATE_EVICT / WAIT`

shape、地址、precision、layout 和 cache slot 放在 256-byte descriptor。斜存
不是独立 opcode。这样同一份 trace 能运行 row-major 与 skewed 设计，也避免
6-bit opcode 空间被很快用完。完整 wire contract 见
[`doc/nemotron3_mamba_isa.md`](../doc/nemotron3_mamba_isa.md)。

Compiler trace 当前会显式输出：state miss/hit、in-projection、layout scatter、
STEP/PREFILL、Vector gated group RMSNorm、out-projection、commit/evict 和最终
WAIT，并用状态机检查数据生命周期。

## 现有加速器能借什么，不能照搬什么

| 工作 | 它真正做了什么 | 我们借什么 | 它没有解决什么 |
|---|---|---|---|
| [MARCA](https://arxiv.org/abs/2409.11440) | 可重构 PE 同时做 linear/elementwise，复用 nonlinear 单元和 buffer。 | Matrix/Vector/state engine 的资源复用和 bypass 思路。 | 不是 Nemotron hybrid，也没有 48 MiB/request 的跨层 state 调度。 |
| [LightMamba](https://arxiv.org/abs/2502.15260) | rotation-assisted 4-bit、PoT SSM quant、tiling/fusion。 | 权重量化和 PoT state 的实验方法。 | 目标是纯 Mamba FPGA；不能证明 Nemotron MoE/Attention 端到端效果。 |
| [FastMamba](https://arxiv.org/abs/2505.18975) | Hadamard 后 8-bit linear、PoT SSM/conv、非线性近似和向量流水线。 | PLENA 已有 Hadamard opcode，可测试 projection weight quantization。 | 主要评估 Mamba2-130M/2.7B；没有 group-shared Nemotron 调度。 |
| [SpecMamba](https://arxiv.org/abs/2509.19873) | speculative decoding、state rollback/FIFO verification。 | 将来若做 speculative，需要借它的 state checkpoint/rollback。 | 当前项目不做 speculation，直接照搬会扩大范围。 |
| [LowRank-SSM](https://arxiv.org/abs/2608.02954) | projection 做 mixed-rank SVD，fused scan，多 AXI 通道。 | 它提醒我们 projection 常是主要时间，不能只优化 recurrence。 | 改权重且需要 accuracy/rank search；应作为后续选项，不是第一版。 |

HCFMaNet 是多模态医学图像融合网络，不是 Mamba 硬件加速器，与本项目的 RTL
设计没有直接关系。

## 论文 novelty 应该怎样写才站得住

不能写成“我们给 PLENA 加了一条 Mamba 指令”。更完整的主线应是：

1. **真实 hybrid workload contract**：同一 Matrix/Vector 系统运行 23 Mamba、
   23 MoE、6 Attention，而不是只跑小型纯 Mamba。
2. **group-aware dataflow**：in-projection 旁路写成 group-major skewed 布局，
   B/C 每组只读一次并广播，state update 与 C reduction 融合。
3. **capacity-aware state virtualization**：Compiler 用 pin/stream 调度解决每个
   request 48.16 MiB state；报告 HBM bytes/token 和 hit rate。
4. **state-aware mixed precision**：权重/activation 与 persistent state 分开选
   precision，FP32 保留更新和 reduction。

其中任何一个点单独都不够强；贡献是它们在真实 Nemotron hybrid 系统中的
联合设计和可测量端到端结果。

## 最大风险和必须做的 ablation

单个 Mamba layer 每 token 的 in/out projection 约 38.7M MAC，而 persistent
state 只有 524,288 elements。只把 STEP 做快，不代表整层或整模型快。至少要
报告：

1. row-major、无 B/C broadcast、state 全走 HBM；
2. 只加 B/C broadcast；
3. 再加 group-major skew；
4. 再加 pin/stream state cache；
5. 再加 BF16/MX8 state；
6. 最后加 projection weight quantization 或 projection/state overlap。

每一级都报告 Mamba stage latency、完整 token latency、HBM bytes/token、bank
stall、cache hit rate、精度和 FPGA PPA。若完整模型提升被 MoE 权重流量淹没，
论文应诚实转向“hybrid system efficiency / state virtualization”，不能只展示
局部 kernel speedup。

当前**未校准** DSE 已经显示这个风险：body-only decode baseline 约
5,596 MiB HBM/token，其中无 cache 的 Mamba state 读写约 96.31 MiB/token。
64 MiB full state cache 把 steady-state state read 降为 0，但完整 HBM 仍约
5,512 MiB/token，估算 token latency 只从约 91.7 ms 降到 90.3 ms。绝对时间
不能用于论文，因为还没有 GPU/RTL 校准；但比例说明只做 state cache 不够，
还必须处理 projection/weight traffic 或与 MoE/Attention service overlap。

## 实施顺序

1. 已完成：CPU reference、state precision 实验、workload/DSE、ISA contract、
   capacity-aware trace、skew 地址映射。
2. 下一步：把 `X_MAMBA` semantic handler 接入 transactional simulator，用 CPU
   reference 对 STEP/PREFILL/RESET/COMMIT 做 differential test。
3. 再下一步：Compiler 把抽象 trace lower 到现有 Matrix/Vector assembly，并
   生成 descriptor buffer；不要先写 RTL。
4. George 的 GPU profiling 到后，校准 projection/conv/scan 和端到端 baseline。
5. 参数冻结后再开 RTL branch：先同步 `0x35..0x38`，然后实现 descriptor
   fetch、L-compute、state engine/cache、仲裁和性能计数器。

RTL 至少要增加这些 counter：`projection_fifo_stall`、`projection_bank_stall`、
`state_engine_active`、`state_cache_hit/miss/evict`、`state_hbm_read/write_bytes`、
`bc_buffer_reads`、`matrix_service_wait`、`vector_service_wait`。没有这些 counter，
Simulator 与 RTL 的 cycle 差异无法定位，paper 的 ablation 也不可验证。

## 现在怎样复现

Simulator worktree：

```bash
uv run pytest -q analytic_models/reference analytic_models/performance
uv run python -m analytic_models.performance.nemotron3_precision \
  --tokens 512 --heads 8 --head-dim 8 --state-dim 128 --groups 1
uv run python -m analytic_models.performance.nemotron3_precision \
  --tokens 32 --real-shape
```

Compiler worktree：

```bash
uv run --with pyyaml pytest -q \
  aten/tests/test_mamba_contract.py aten/tests/test_mamba_scheduler.py
uv run --with pyyaml python -m aten.mamba.trace \
  --phase decode --decode-tokens 2 \
  --state-cache-entries 4 --cache-policy pinned \
  --state-precision mx8_b128
uv run --with pyyaml python -m aten.mamba.trace \
  --phase prefill --sequence-length 257
```

Compiler 的 `pyproject.toml` 目前漏列了旧 registry 已经依赖的 PyYAML，所以
本地命令暂时显式使用 `--with pyyaml`；新 Mamba 模块本身没有这个额外依赖。
