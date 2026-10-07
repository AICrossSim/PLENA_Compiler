# Mamba/KDA compilation and Matrix SRAM L-Compute

This branch adds static Mamba-2 and KDA lowering and an explicit Matrix-SRAM
recurrence path. It is a Compiler/Simulator implementation for review; hardware
implementation and resource mapping remain separate work.

## Supported paths

| Path | Entry points and scope |
|---|---|
| Static Mamba-2 | `program_ssm_recurrent.py`, `program_ssd.py`, and `program_mamba_common.py`: decode recurrence, chunked SSD, convolution and state transfers |
| Static KDA | `program_kda_*.py`: convolution, gates, normalization, recurrence, chunked prefill and layer composition |
| Matrix SRAM recurrence | `lower_matrix_recurrence`: explicit prepared-field/state DMA, views, recurrence primitives and output/state stores |
| Hybrid schedule | `hybrid_l_tile_schedule.py`: recurrent programs in the Nemotron/Kimi layer order; ordinary layers are schedule markers |
| Projection | Existing Matrix accumulation and writeback, with optional affine/Matrix-view output layout and persistent scratch ownership |

The Matrix-SRAM wrappers require MLEN=2048, BLEN=32, one request, and the
official recurrent shapes: Mamba 64 heads ×128 state rows ×64 values; KDA
96 heads ×128 keys ×128 values. Unsupported wrapper shapes are rejected.
Callers choose the ordinary or Matrix-SRAM entry point explicitly.

Prepared coefficients are an explicit input boundary, after projection,
convolution and coefficient generation. Executing that recurrence does not
execute a whole checkpoint. The separate KDA layer emitter includes projections
and surrounding stages; its structural tests are distinct from Rust numerical
execution of the prepared recurrence.

## Precision and storage

Weight, activation, KV and recurrent-state formats are separate choices. NVFP4
weights do not imply NVFP4 recurrent state. Matrix-SRAM state, prepared fields
and output use BF16; state DMA uses precision selector 2 independently of KV.
Legacy ordinary selector 2 therefore no longer aliases KV; use selector 1 for KV.

BF16 storage is distinct from intermediate arithmetic. L_TILE dot reduction
retains FP32 partial sums before BF16 writeback. Ordinary Vector instructions
have their own rounding boundaries. Numerical comparisons must state both
contracts and must pass a shared error budget before reporting a speed ratio.
Existing Matrix operations already accumulate before writeback; the Simulator's
FP32 representation does not establish the physical precision of every unit.

There is no private recurrent-state SRAM or cache in this lowering. State and
fields occupy disjoint, aligned HBM arenas and explicitly allocated Matrix SRAM.
GP DMA offsets must fit 32 bits; architectural address registers provide the
HBM base. Direct-view projection additionally requires a persistent scratch
owner and one complete output packet before scratch reuse.

## ISA interface

`L_TILE_CFG` and `L_TILE_EXEC` share opcode `0x3f` (subforms 1 and 3).
Four view slots describe bounded shapes, pitch and tile phase over the fixed
diagonal bank mapping. EXEC implements scale-accumulate, dot-reduce and
outer-update; it does not encode a model name. Matrix-view DMA is an explicit
form of `H_PREFETCH_V`/`H_STORE_V`; Matrix-view writeback qualifies `M_MM_WO`.
`M_MM_WO` now uses immediate bit 17 as a view marker: an old word such as
`0x80000046` is reinterpreted as view 0. Existing binaries using that bit are
not backward compatible; the assembler rejects such legacy offsets.

Static coefficient preparation also uses `V_SOFTPLUS_V` (`0x3d`) and
`S_MAP_FP_V` (`0x3e`). `V_FMA_VF` is an arithmetic subform of `V_MUL_VF`.
The branch also reserves routed-MoE opcodes `0x39..0x3c`, outside this recurrent
path. Encoding definitions are in `doc/operation.svh` and `aten/plena/mview.py`.

The historical `L_CFG` Vector-stream form remains supported. Prepared Vector
controls also retain opt-in FP32-dot and pairwise-BF16 options; both default to
false. Their instruction encodings and storage contracts are tested, but they
do not select the main recurrent lowering automatically.

L_TILE control, routing and live FP32 partial sums still require hardware
mapping. This implementation does not establish zero additional hardware,
RTL timing, area or power.

## Validation

CI runs CPU formula and emitted-instruction numerical tests, parser/codegen,
assembler encodings, affine addressing, SRAM ownership, HBM arena bounds,
precision guards and layer composition. Real-checkpoint tests skip when their
checkpoint is absent. Connected machine-code execution and cycle accounting
are tested in the matching Simulator repository.
