# Compiler draft review scope — 2026-09-05

**Draft review only — not for merge.** This checkout makes the accumulated
Mamba/KDA work reviewable against main. It does not select L_TILE, FP32-dot or
BF16-pairwise as the final architecture. Historical experiment results remain
evidence with their original scope; a passing simulator is not RTL acceptance.

## What original PLENA already provides

Original PLENA has Matrix accumulators, output-stationary partial sums and
cross-array reduction. Its Compiler maps tiled computation to Matrix, Vector,
Scalar and memory instructions; its Simulator models those instructions.
The [paper, Figure 6 and section III-B](https://arxiv.org/html/2509.09505v3)
describes MX operands, INT accumulation and conversion before activation
writeback. Configurable Vector precision is a separate choice.

This branch adds Mamba/KDA references and lowering, recurrent-state transfers,
Matrix-view/L_TILE encodings and supporting operations. It does not establish
that original PLENA lacked accumulators. Whether a particular recurrence can
use existing accumulators depends on its data mapping, precision, feedback and
writeback schedule. A Rust `f32` temporary alone proves neither a hardware gap
nor successful reuse. The original paper, local RTL revisions and this branch's
BF16 experiment configuration must not be treated as one frozen implementation.

## Precision: six different execution paths

| Path | Weight treatment | State and operands | Arithmetic / writeback boundary |
|---|---|---|---|
| Full-model analytic timeline | NVFP4/MX8/BF16 logical byte accounting; no Rust weight decode | Separate W/A/KV/state contracts; this campaign uses BF16 A/KV/state | Work, traffic and cycle model; no complete-model numerical execution |
| Prepared ordinary A/B control | No checkpoint weights in kernel | Actual BF16 state and prepared coefficients | Ordinary VV results round when written to BF16 SRAM |
| L_TILE C/D recurrence | No checkpoint weights in kernel | BF16 Matrix SRAM, state and prepared fields | FP32 dot partial sums across rows; update primitives round at writeback; physical reuse remains unproven |
| Optional FP32-dot control | No checkpoint weights | BF16 operands/state | Only two KDA dots retain FP32 products/sums; other VV boundaries remain BF16 |
| Optional pairwise control | No checkpoint weights | BF16 operands/state and partial rows in existing Vector SRAM | Balanced tree; every MUL and ADD still rounds to BF16 |
| GPU baseline | Actual Nemotron NVFP4 checkpoint execution | GPU implementation's precision, including FP32 recurrent state | Independent GPU reference and timing, not execution of PLENA instructions |

The paired Simulator's `PrecisionContract` separates W, A, KV and state. NVFP4
weight accounting uses E2M1/block16 plus one E4M3 scale byte per block and retains
the checkpoint's BF16 exclusions. MX8 weight sensitivity explicitly uses
E4M3/block8 plus one E8M0 scale byte. Tensor-global scales and unknown physical
checkpoint padding are excluded and disclosed; DMA burst rounding is separate.
Old block128 reports remain historical. These weight tables do not validate a
PLENA NVFP4/MX8 numerical decoder or quantized-model quality.

This work does not change the global W/A/KV formats. Independent
`HBM_STATE_TYPE=Plain BF16` prevents recurrent state from inheriting the KV
format. DMA selectors are Activation=0, KV=1, State=2. The ordinary Simulator
decoder's former nonzero-selector-as-KV behavior was corrected; binaries that
intentionally used selector 2 as a KV alias must use selector 1.

Both optional controls default off in
[`prepared_vector_recurrence.py`](../aten/plena/prepared_vector_recurrence.py).
`experimental_fp32_dot=True` emits V_DOT_RESET/ACC/WRITE and requires the paired
Simulator's `PLENA_EXPERIMENTAL_FP32_DOT=1`. It explicitly models an additional
4×VLEN-byte FP32 accumulator plus validity and assumed throughput.
`pairwise_bf16_dot=True` uses ordinary instructions and 15 existing Vector SRAM
rows at VLEN=2048, including seven partial-sum rows. The flags are exclusive;
neither changes the original Matrix accumulator by default.

中文：这里把“权重有多大”“state 怎么存”“中间结果何时舍入”分开记录。
NVFP4/MX8 表改的是权重流量计数；实际 Rust 递推使用 BF16 输入和 state。
L_TILE 的部分中间计算保持 FP32，所以“BF16 存储”不等于“全程 BF16 算术”。
两种额外 dot 实验默认关闭，也没有被选为替代原 PLENA 的主架构。

## Baselines and claims

Historical analytic A/B are this work's unoptimized/Arlo static recurrence
instruction censuses. They use an issue proxy and are not original-paper
Mamba/KDA measurements. Prepared executable A/B are different control programs:
A reloads addresses; B reuses known addresses. Their coefficient expansion and
rounding differ from L_TILE. Keep these identities in every table and PR claim.

L_TILE retains up to 8 KiB of FP32 dot state at VLEN=2048 in the functional model.
Its update primitives also combine arithmetic before storage rounding. Existing
SRAM capacity does not prove that registers, feedback and routing already cover
these operations. Review that mapping before claiming zero additional hardware.
The new executable A/B timing cannot be substituted into the old whole-model
timeline or multiplied into its published speedups.

## Source navigation

| Review question | Compiler entry points |
|---|---|
| Recurrence formulas and state precision | [`aten/models/mamba2/reference.py`](../aten/models/mamba2/reference.py), [`aten/models/kda/reference.py`](../aten/models/kda/reference.py), [`state_precision.py`](../aten/plena/state_precision.py) |
| Original-style tiled Matrix lowering and producer ownership | [`program_matrix_ops.py`](../aten/plena/program_matrix_ops.py), [`isa_matrix.py`](../aten/plena/isa_matrix.py), [`memory.py`](../aten/plena/memory.py) |
| View validation, lifetime and packet construction | [`mview.py`](../aten/plena/mview.py), [`matrix_access_packets.py`](../aten/plena/matrix_access_packets.py), [`matrix_recurrence_lowering.py`](../aten/plena/matrix_recurrence_lowering.py) |
| Layer schedule versus full numerical execution | [`hybrid_l_tile_schedule.py`](../aten/plena/hybrid_l_tile_schedule.py), [`hybrid_compile_report.py`](../aten/plena/hybrid_compile_report.py), [`matrix_prefill_handoff.py`](../aten/plena/matrix_prefill_handoff.py) |
| Prepared controls and optional dot experiments | [`prepared_vector_recurrence.py`](../aten/plena/prepared_vector_recurrence.py), [`test_prepared_vector_recurrence.py`](../aten/tests/test_prepared_vector_recurrence.py), [`EXPERIMENTAL_FP32_DOT.md`](EXPERIMENTAL_FP32_DOT.md) |
| Exact encodings and branch-wide opcode scope | [`assembly_to_binary.py`](../assembler/assembly_to_binary.py), [`matrix_lcompute_pre_rtl_freeze.md`](matrix_lcompute_pre_rtl_freeze.md#21-full-opcode-delta-against-main), [`matrix_lcompute_isa_review.md`](matrix_lcompute_isa_review.md) |

The narrow L_TILE functional path uses L_TILE plus V_SOFTPLUS_V and S_MAP_FP_V.
The complete branch has a larger ISA delta, documented in the linked opcode
table; it must not be summarized as a one-opcode change against main.

## Evidence already produced; acceptance still pending

The review fixes reject state/field HBM overlap and any arena outside the GP32
offset window, and preserve Python 3.10 enum string semantics without changing
the declared minimum Python version. Four official default recurrence assembly
hashes remain unchanged. These are input/compatibility fixes, not new algorithms.

One encoding decision remains open: Matrix-view `M_MM_WO` consumes bit17 of the
old immediate. Old binaries using that bit can be reinterpreted as viewed
writeback; rejecting newly assembled high immediates does not preserve those
binaries. The paired Simulator review documents `0x80000046` as a concrete
example. Do not merge before reviewing this ABI boundary. Other routed-MoE
encodings in the broader Compiler are also not covered by the narrow recurrence
execution tests.

- Historical Compiler→assembler→Rust runs execute BF16 Mamba/KDA recurrence
  cores and compare actual HBM state/output. Multi-seed and token-chain campaigns
  also check intermediate state; exact phased-reference agreement is a contract
  check, not a claim of exact agreement with FP32 mathematics on all inputs.
- Prepared Nemotron A/B/D ran B1–B16, two tokens each, with private request state.
  Prepared KDA ordinary B1 fails the shared output budget, so its qualified
  speedup is blank. Its own rounding-reference match still passes.
- Optional FP32-dot and pairwise runs provide separate diagnostics. The pairwise
  prototype passed actual B1 A/B and B2 four-token checks and standard CPU long
  chains; stronger synthetic long-memory stress still fails. These are not
  quantized checkpoint quality results or a newly accepted performance baseline.
- Real Nemotron GPU timing/routing and component GPU evidence are external
  calibration inputs. Prompt workloads are not evidence of solved agent tasks
  or SWE-bench benchmark scores. Raw GPU evidence and historical energy caveats
  remain owned by the paired Simulator campaign.
- The 52/93-layer schedules and analytic timelines are not first-to-last real
  Nemotron/Kimi checkpoint execution in Rust. The smaller Mamba checkpoint
  experiment also uses host execution for surrounding operators.

For this review checkout, see [fresh Compiler validation](REVIEW_VALIDATION_20260905.md)
and the paired Simulator PR's connected execution evidence. Historical test
counts in older documents apply only to their recorded runs. Still outstanding:
physical precision/resource mapping, the chosen baseline and rounding contract,
real-model quality under long-memory inputs, complete-model Rust execution, and
RTL/PPA validation. No final architectural selection follows from this draft.
