# PLENA Compiler

## Matrix SRAM L-Compute

This branch keeps two independent optimizations for Nemotron 3 and Kimi K3:

1. Arlo's static lowering reduces pointer, scalar and loop instructions.
2. Matrix L-Compute uses PLENA's fixed diagonal Matrix-SRAM bank wiring and a
   Compiler-selected physical-row pitch. Matrix writeback and its consumers use
   the same view, so row, column and multi-row packets read the same cells with
   logical lane order restored.

The second item is the architecture mechanism evaluated here. It is not the
older Vector-SRAM `L_CFG` experiment and it adds no state cache, private SRAM,
`X_STATE`, MAC array, queue or runtime scheduler.

The implemented ISA uses one model-independent configuration opcode with two forms:

```text
L_MVIEW.FULL   slot, shape_reg, map_reg
L_MVIEW.FIELD  slot, field, value_reg
<Matrix op>    ..., view=slot
<Vector op>.MV ..., operand_view_mask
```

The descriptor contains shape, physical-row pitch and bounds flags. Mapping
bits `[27:16]` are reserved and must be zero: a fair search found that
per-view programmable skew provides no benefit once the control may also choose
pitch. It contains no model name, recurrence equation, head count, bank count
or traversal. Existing `M_MM/M_TMM/M_MV` words explicitly name the view.
Existing binary Vector operations use `.MV` as an addressing-mode suffix: mask
bits select whether destination, source 1 and source 2 use slots 0, 1 and 2. A
must-dataflow check over loop back-edges rejects use before configuration.

At `MLEN=2048`, one packet uses 32 Mamba heads x 64 values or 16 KDA heads x
128 values. The pitch-1 control and implemented co-layout have exactly the same
dynamic operation stream. Compiler pitches 2 (Mamba) and 4 (KDA) reduce local
packet service from 2 to 1 cycles and 4 to 1 cycles, respectively, with zero
bank-conflict stalls. An exhaustive counterfactual search over programmable
skew cannot improve either result. All recurrence rows fill the apparent pitch
gaps: 262,144 values per model were placed and read back with zero alias and
zero capacity overhead.

The corresponding official-shape, B1 decode serial analytic timelines are:

| Model | Pitch-1 | Implemented co-layout | Whole-model gain | Programmable-skew upper bound |
|---|---:|---:|---:|---:|
| Nemotron 3 (52 layers) | 3,160,138 | 3,142,474 | 1.00562x | 1.00000x over implemented |
| Kimi K3 (93 layers) | 98,804,544 | 98,168,640 | 1.00648x | 1.00000x over implemented |

The local bank result comes from numbered physical-cell replay of real Compiler
addresses. The full-model result is a formula-based serial analytic timeline
with official dimensions, GPU calibration and symbolic PLENA weights; it is not
a first-to-last transactional checkpoint execution.

The official manifests are pinned to 52 Nemotron layers (23 Mamba, 23 MoE,
6 GQA) and 93 Kimi layers (69 KDA, 24 MLA). Dimensions and GPU calibration are
real, but the full PLENA programs still use symbolic weights. A separate
published `mamba2-130m-hf` gate validates 24 layers and carried recurrent state;
its surrounding Matrix stages run in PyTorch and are not described as a full
PLENA checkpoint execution.

Official recurrent state remains explicit FP32 traffic: 2 MiB per Nemotron
Mamba layer and 6 MiB per Kimi KDA layer. The BF16 Matrix SRAM cannot silently
hold either format. Low-precision state results are reported separately with
their accuracy error.

There are no transfer-only `L_MVIEW` instructions. Matrix projection fragments
use the existing `M_MM_WO` with a view-qualified destination and advance by one
`BLEN` fragment until the real consumer shape is full: `32 x 64` for the
Nemotron packet and `16 x 128` for Kimi at `MLEN=2048`. Existing Vector
arithmetic reads those packets directly through `.MV`; no intermediate gather
or copy is inserted. See [the ISA review](doc/matrix_lcompute_isa_review.md) for
the encoding argument, physical data path and resource contract.

KDA prefill is a separate boundary. The legacy Compiler emits an identity GEMM
to convert `[value,key]` into decode's `[key,value]`; a BF16/MX8 Matrix view
instead reads 16,384 numbered values through the column axis with zero
transpose MACs. The old `3.387x/1.713x` prefill claims are withdrawn because
the two complete paths were not measured under the same timeline. Official
FP32 state receives no Matrix-residency or prefill speedup credit.

## MoE code organization

- `aten/plena/program_routed_moe.py` contains reusable routed-MoE lowering
  helpers: router logits, V_TOPK selection, dynamic expert-weight addressing,
  routed gather/scatter, expert activation, and combine.
- `aten/models/gpt_oss/` contains GPT-OSS-specific reference semantics and
  real-checkpoint loading utilities used to validate that substrate.
- ISA, assembler, and hardware documentation remain in `assembler/` and `doc/`.
