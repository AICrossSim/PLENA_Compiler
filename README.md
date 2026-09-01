# PLENA Compiler

## Matrix SRAM L-Compute

This branch has two independent optimizations for Nemotron 3 and Kimi K3:

1. Arlo's static lowering reduces pointer, scalar and loop instructions.
2. Matrix L-Compute assigns a Compiler-selected affine skew to each Matrix
   tensor view. Matrix writeback places values diagonally across physical banks;
   row, column, cross-head and cross-field consumers read the same cells and
   restore logical lane order.

The second item is the architecture contribution. It is not the older
Vector-SRAM `L_CFG` experiment and it does not add a state cache, private SRAM,
`X_STATE`, MAC array, queue or runtime scheduler.

The implemented ISA uses one model-independent configuration opcode with two forms:

```text
L_MVIEW.FULL   slot, shape_reg, map_reg
L_MVIEW.FIELD  slot, field, value_reg
<Matrix op>    ..., view=slot
<Vector op>.MV ..., operand_view_mask
```

The descriptor contains shape, physical-row pitch and `alpha`, the skew selected
from the tensor's logical row width. It contains no model
name, recurrence equation, head count, bank count or traversal. Existing
`M_MM/M_TMM/M_MV` words explicitly name the view. Existing binary Vector
operations use `.MV` as an addressing-mode suffix: mask bits select whether the
destination, source 1 and source 2 use configured slots 0, 1 and 2. A
single-pass dominance check rejects use before configuration.

At `MLEN=2048`, one packet uses 32 Mamba heads x 64 values or 16 KDA heads x
128 values. `D'` searches all 4096 global `(alpha,gamma)` wirings on physical
rows taken from the emitted Compiler addresses. Against that strongest fixed
control, per-view `alpha` gives no further local gain for Nemotron, while Kimi
service falls from 6144 to 3072 cycles across the real decode lowering (2.0x)
and bank-conflict stalls fall from 3072 to zero. The same numbered values are
written, read and restored in every case; arithmetic and issue counts are held
constant.

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

KDA prefill has a separate layout boundary. The legacy Compiler emits an
identity GEMM to convert `[value,key]` into decode's `[key,value]`. At Kimi's
real 96-head, 69-layer shape this is 13.89 G logical MACs, while the current
MLEN padding actually emits 56.90 T MACs. A BF16/MX8 Matrix-view candidate
reads the same per-head cells by the column axis and explicitly streams them
out in decode order: 0 transpose MACs and all 16,384 non-symmetric values
checked per head. Decode's later 16-head packet is a separate compact view, not
the same resident allocation. This result is reported on its own and is not
credited to the official FP32 path.

## MoE code organization

- `aten/plena/program_routed_moe.py` contains reusable routed-MoE lowering
  helpers: router logits, V_TOPK selection, dynamic expert-weight addressing,
  routed gather/scatter, expert activation, and combine.
- `aten/models/gpt_oss/` contains GPT-OSS-specific reference semantics and
  real-checkpoint loading utilities used to validate that substrate.
- ISA, assembler, and hardware documentation remain in `assembler/` and `doc/`.
