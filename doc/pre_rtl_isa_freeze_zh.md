# PLENA Hybrid ISA: RTL 前冻结清单

## 结论

Kimi/Nemotron 分支曾把 `0x39` 用作 `C_SET_TOPK_BIAS_REG`，而尚未合入 main
的 batch-4 Shared-MoE route branch 把 `0x39-0x3C` 用作四条 route 指令。最终
方案不移动 Shared-MoE：policy 和 correction-bias 地址共用 `0x38`，由 bits
`[13:10]` 的 `target` 选择。

冻结编码如下：

| Opcode | 指令 | 作用 |
|---:|---|---|
| `0x35` | `V_MAX_VF` | Vector 与 FP scalar 做 max |
| `0x36` | `V_MIN_VF` | Vector 与 FP scalar 做 min |
| `0x37` | `V_TOPK` | 生成 routed expert id 和 route weight |
| `0x38` | `C_SET_TOPK_REG rd, target` | target 0 设置 policy；target 1 设置 correction-bias VRAM 地址 |
| `0x39` | `C_ROUTE_BEGIN` | 建立 batch route context |
| `0x3A` | `C_ROUTE_LOOP_START` | 按 unique expert 开始动态循环 |
| `0x3B` | `C_ROUTE_LOOP_END` | 切到下一个 unique expert 或退出 |
| `0x3C` | `V_ROUTE_MUL` | 应用当前 expert 对某 token 的 route weight/mask |
| `0x3D` | `X_STATE` | Mamba-2/KDA state engine，功能由 subop 区分 |
| `0x3E` | 未分配 | 保留一格扩展空间 |
| `0x3F` | `L_SCATTER_M` | 按 descriptor 把 Matrix writeback 排成 row/transpose/Mamba/KDA/custom bank layout |

旧的 `C_SET_TOPK_REG rd` 等价于 `target=0`，机器码逐字节不变。Kimi/Nemotron
的 bias 写入改成 `C_SET_TOPK_REG rd, 1`；旧 `C_SET_TOPK_BIAS_REG` 助记符不再是
正式 ISA，也不保留隐式 alias。`target=2..15` 与 bits `[31:14]` 非零均为非法编码。
Shared-MoE branch 的 `0x39-0x3C` 因此可原样合入。Compiler、Simulator 和 RTL
必须由同一份 machine-readable contract 生成或检查常量和 golden words。

## Kimi Top-K 当前实现

单 token decode 已经不再复制 Top-16 expert body。Compiler 使用现有
`C_LOOP_START/END` 只发出一份 body，循环中的 GP index 动态读取：

```text
IntSRAM[topk_base + rank] -> expert id
FPRAM[weight_base + rank] -> route weight
tile-major HBM table + expert id -> gate/up/down tiles
expert output * route weight -> routed accumulator
```

compact Rust 对拍读取两个不同 expert，最终误差为 0；当前动态版本为
24,297 cycles。这个结果证明了动态 index、64-bit tile-group
地址和 route weight，不只是证明汇编器接受了循环。

batch route extension 仍需扩展到 Kimi 的 `896 experts / top-16`。不能保留旧分支
写死的 `<=256 experts / top-8 / exactly four tokens` 限制；合法性应是：

```text
num_experts <= 1024
top_k <= 16
batch * top_k <= configured IntSRAM/FPRAM route capacity
```

## 不增加的指令

Matrix output-column、K-tile 和 MLA head traversal 不需要新 opcode。它们应该用
现有 `C_LOOP`，循环携带以下地址：

- activation VRAM tile base；
- output/scratch VRAM tile base；
- Matrix SRAM tile base；
- 64-bit HBM tile-group base；
- expert/head index。

HBM low 32-bit 地址跨过 4 GiB 时，Compiler 在静态 high-address window 边界拆分
loop segment。不要在 GP 上做没有 carry 的 32-bit 加法。Matrix/head traversal 不增加
opcode；但 producer-consumer layout 由 `L_SCATTER_M=0x3F` 显式选择，256-byte layout
descriptor 保存 bank 数、端口数、FIFO、field rotation 和 mapping CRC。`X_STATE`
descriptor 仍不携带 layout，因此 state 数学语义不随排布变化。

`L_SCATTER_M` 自身的第一版微架构参数已经可以冻结：16 个 single-port banks、
64-value Matrix burst、64-value FIFO。Simulator 的 FIFO sweep 中 64 与 256 entries
周期相同，full-shape Mamba/KDA roundtrip、alias 负向测试以及 compact Rust 连续数值链
都已通过。这里冻结的是 Compiler/Simulator contract；Matrix writeback stream tap、
bank mux PPA 和频率仍必须由 RTL 验证。

## 完整 Kimi binary 的当前状态

旧的静态 Matrix emitter 在 `heads=1` 时产生：

```text
100,221,916 instructions
3,739,264,558 assembly bytes
7m10s compile time
24.1 GiB peak RSS
```

其中 LatentMoE 占 91,162,388 条。加入 compact Matrix output-column 循环和 streaming
assembler 后，真实 96-head、93 层程序现在是：

```text
11,502,370 instructions
43.88 MiB raw 32-bit machine code
2m10s build + assemble
4.7 GiB peak RSS
```

全部指令可编码，MXFP8 与 BF16 stream-K 两条 compact Matrix 路径都已在 Rust 中
逐元素对拍。MLA 的 96 个 head body 仍静态发射，是后续 code-size 优化，不再阻塞
完整机器码产物。RTL 前剩余条件是：

1. multi-token compressed MLA cache append/read 有独立数值测试；
2. symbolic HBM manifest 绑定真实 checkpoint 后完成整模数值 replay；
3. Compiler/Simulator 的 `X_STATE`/`L_SCATTER_M` contract 与 golden words 一致；RTL
   开始前再把同一 machine-readable contract 接入第三仓。
