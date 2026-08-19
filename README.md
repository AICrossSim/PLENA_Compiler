# PLENA Compiler

## Review setup

The default environment excludes the optional TileLang/CUDA toolchain. A clean
CPU checkout can run the common-state Compiler guards with:

```bash
uv sync --frozen
uv run --frozen pytest -q -m "not slow" \
  asm_templates/tests/test_large_immediate.py \
  assembler/tests/test_c_set_topk_reg_encoding.py \
  assembler/tests/test_l_scatter_m_encoding.py \
  assembler/tests/test_x_state_encoding.py \
  aten/tests/test_kda_scheduler.py \
  aten/tests/test_kimi_k3_full_program.py \
  aten/tests/test_kimi_k3_hybrid.py \
  aten/tests/test_hybrid_substrate.py \
  aten/tests/test_layout_contract.py \
  aten/tests/test_mamba_scheduler.py \
  aten/tests/test_mla_attention.py \
  aten/tests/test_mram_binding_tracking.py \
  aten/tests/test_nemotron3_blocks.py \
  aten/tests/test_nemotron3_full_program.py \
  aten/tests/test_nemotron3_hybrid.py \
  aten/tests/test_projection_scatter.py \
  aten/tests/test_state_contract.py \
  aten/tests/test_state_isa_lowering.py \
  aten/tests/test_state_lowering.py \
  aten/tests/test_state_memory_image.py \
  aten/tests/test_state_residency.py
```

Use `uv sync --frozen --group tvm` only for TileLang/TVM development.

## MoE code organization

- `aten/plena/program_routed_moe.py` contains reusable routed-MoE lowering
  helpers: router logits, V_TOPK selection, dynamic expert-weight addressing,
  routed gather/scatter, expert activation, and combine.
- `aten/models/gpt_oss/` contains GPT-OSS-specific reference semantics and
  real-checkpoint loading utilities used to validate that substrate.
- ISA, assembler, and hardware documentation remain in `assembler/` and `doc/`.

## Common recurrent-state extension

`aten/state` defines the descriptor-driven `X_STATE=0x3D` contract shared by
Nemotron 3 Mamba-2 and Kimi K3 KDA. `aten/mamba` and `aten/kda` generate
Matrix -> X_STATE -> FENCE -> Vector -> Matrix semantic traces, while
`aten.state.lower_state_trace` materializes aligned descriptor images,
register writes, and X_STATE instruction words. The exact ABI and current
precision limits are documented in
[`doc/nemotron3_mamba_isa.md`](doc/nemotron3_mamba_isa.md).
The provisional first-RTL parameter scope is machine-readable in
[`spec/x_state_v2_rtl_candidate.json`](spec/x_state_v2_rtl_candidate.json); it
is a synthesis input, not a PPA result or a final optimality claim.
The trace CLIs accept `--async-pipeline` for a two-request streaming schedule
that overlaps the second Matrix projection with the first State command.

`aten.state.isa_lowering` also lowers the complete real-shape KDA mixer path
(independent q/k/v, low-rank decay, padded beta, output gate, per-head
RMSNorm, output projection) to existing Matrix/Vector ISA plus `X_STATE`.
`aten.kimi3.trace` emits the full 93-layer text-backbone schedule and an
optional physical KDA program:

```bash
uv run python -m aten.kimi3.trace --phase decode --batch-size 1 \
  --state-cache-mib 32 --output build/kimi-k3-full.json \
  --kda-physical-output build/kimi-k3-kda-physical.json \
  --kda-physical-asm-output build/kimi-k3-kda.s
```

`aten.kimi3.connected_program` and `aten.nemotron3.connected_program` connect
MLA/GQA projections, AttnRes/residual ownership, routed and shared experts,
expert combine, and KDA/Mamba fixed-address handoffs with real producer-consumer
tensors. Compact Kimi blocks and the Nemotron whole-model program assemble to
32-bit machine words. Kimi's routed Top-16 path now uses one dynamic expert body
inside `C_LOOP` and is numerically verified in Rust. Full Kimi emission still
fails fast because Matrix output-column/K-tile traversal and MLA's 24 x 96 head
bodies are not looped. Even a post-Top-K `heads=1` diagnostic emitted 100.2
million instructions, took 7m10s, and peaked at 24.1 GiB RSS. Prefill and
persistent multi-token MLA/GQA cache append are also rejected until their
physical contracts exist.

The cross-branch opcode conflict and the pre-RTL freeze proposal are recorded
in [`doc/pre_rtl_isa_freeze_zh.md`](doc/pre_rtl_isa_freeze_zh.md).

Routed-expert weights use a tile-major HBM table. Each runtime expert id adds
only one MX tile stride (4,608 bytes at MLEN=64); the compiler supplies the
static tile-group high/low address halves to `C_SET_ADDR_REG`. This avoids
overflowing a 32-bit GP when Nemotron/Kimi symbolic weight arenas exceed 4 GiB,
while keeping each tile's element and scale streams together. Compact
Simulator tests execute this layout with Rust `V_TOPK` output and compare the
selected expert result against CPU golden values.

`PROJECTION_SCATTER` now lowers to executable `L_SCATTER_M=0x3F` plus a
256-byte layout descriptor. The lowered JSON retains a versioned debug view of
the ping-pong buffer, source Vector-SRAM address, field producer/consumer, bank
mapping, FIFO capacity, burst width, and spill policy. Mamba defaults to
group-major/skewed placement. Both defaults use 16 single-port banks, one
64-value producer burst, and a 64-value FIFO; the 64-entry size is the minimum
zero-stall point from the Simulator sweep. KDA uses an 8-bank `k` rotation, stripes scalar
`beta` values across banks by head, and materializes the five independently
produced projection tensors before recurrence consumption. See
[`spec/l_scatter_m_v1.json`](spec/l_scatter_m_v1.json) for the binary contract
and the matching Simulator Rust parser/tests for executable semantics.
