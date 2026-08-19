# X_STATE v2 Contract

`X_STATE` is the common recurrent-state command for Nemotron 3 Mamba-2 and
Kimi K3 KDA. The machine-readable source is
[`spec/x_state_v2.json`](../spec/x_state_v2.json). Run
`uv run python tools/state_contract.py --check` to detect stale generated code
or opcode collisions.

The provisional first-RTL resource and precision scope is frozen separately in
[`spec/x_state_v2_rtl_candidate.json`](../spec/x_state_v2_rtl_candidate.json).
Keeping this out of the descriptor lets the same binary compare future lane,
bank, cache, and layout implementations.

To verify the current Compiler and Simulator opcode maps before Simulator work,
run `uv run python tools/state_contract.py --check --simulator-root <path>`.

## Ownership boundary

The existing Matrix service performs input and output projections. The existing
Vector service performs output gating and normalization. `X_STATE` owns only
short convolution, recurrent state update, and the reduction that produces the
pre-gate output. It never fetches projection weights.

Projection layout is owned by `L_SCATTER_M`, not by `X_STATE`. Its separate
descriptor makes row-major, transpose and workload-specific skew explicit while
keeping the recurrent math descriptor unchanged. State-cache banking remains a
microarchitecture parameter.

## Instruction

`X_STATE` uses opcode `0x3D` and the existing `R_FUNCT` bit layout:

```text
31       26 25       22 21       18 17       14 13       10 9         6 5      0
+----------+-----------+-----------+-----------+-----------+-----------+--------+
| reserved | subop     | queue_id  | desc_hbm | desc_off  | context   | 0x3D   |
+----------+-----------+-----------+-----------+-----------+-----------+--------+
```

`C_SET_TOPK_REG=0x38` uses target 0 for policy and target 1 for the no-aux
correction-bias address. `0x39-0x3C` are reserved for the routed-MoE control
extension, leaving `0x3E` free. Descriptor
commands use `context_gp`, `descriptor_offset_gp`, and `descriptor_hbm_reg`.
`FENCE` does not fetch a descriptor and requires all three fields to encode zero.

| Value | Subop | State behavior |
|---:|---|---|
| 0 | `PRELOAD` | Copy clean HBM state into an explicit state-SRAM region. |
| 1 | `RESET` | Establish zero recurrent and convolution state. |
| 2 | `PREFILL` | Process `valid_tokens` sequentially and update state. |
| 3 | `STEP` | Process exactly one decode token. |
| 4 | `COMMIT` | Write dirty resident state to HBM and keep it resident. |
| 5 | `EVICT` | Release clean resident state. Dirty eviction is illegal. |
| 6 | `FENCE` | Wait for one State Engine queue. No descriptor is fetched. |

The transactional Simulator defaults to blocking execution. Its optional
event-driven mode implements 16 in-order queues: issue returns after descriptor
validation, commands on one queue remain ordered, and `FENCE` waits for the
selected queue. A non-`NO_EVENT` dependency must name a producer that has
already been issued. Duplicate completion-event producers and self-dependencies
are state hazards.

## Descriptor

The descriptor is 256 bytes, little-endian, and 64-byte aligned. Bytes 0-127
form the common header; bytes 128-255 are a Mamba-2 or KDA payload selected by
the `algorithm` field.

Common state identity is `(context_id, request_id, layer_id, state_id)`. HBM
addresses are byte addresses. Vector SRAM addresses and strides are element
addresses. Accumulation is fixed to FP32 in v2.

`state_sram_offset=0xFFFFFFFF` selects direct streaming. In this mode `STEP` and
`PREFILL` read and write HBM state internally through transient head-tile
buffers; the compiler must not emit fake `PRELOAD`, `COMMIT`, or `EVICT`
commands. Any other offset selects an explicitly compiler-managed resident SRAM
region, where the normal preload/dirty/commit/evict lifecycle applies.

The descriptor stores exact recurrent-state and convolution-state byte counts.
The decoder recomputes both from algorithm, shape, batch, and state precision;
a mismatch is an invalid descriptor.

Mamba-2 payload fields describe `head_dim`, `state_dim`, `groups`, convolution,
projected X/B/C and dt offsets, and A/dt/D parameters. KDA payload fields
describe `key_dim`, `value_dim`, Q/K/V/decay/beta offsets, three convolution
parameter sets, decay parameters, and output scaling. Output gate and norm
parameters stay outside `X_STATE` because the Vector service owns those stages.

## Vector SRAM records

`input_vram_addr` and `output_vram_addr` point to token-major records. The
record for local token `t` and batch element `b` starts at:

```text
base + (t * batch_size + b) * token_stride
```

Offsets in a payload are relative to the start of one input record. Mamba uses
the non-overlapping segments below, where `H=num_heads`, `P=head_dim`,
`G=groups`, and `N=state_dim`:

```text
gate: [0, H*P)
x/B/C packet: [xbc_offset, xbc_offset + H*P + 2*G*N)
dt: [dt_offset, dt_offset + H)
```

KDA uses Q, K, V, decay and beta segments of lengths `H*K`, `H*K`, `H*V`,
`H*K`, and `H`. The decoder rejects overlapping segments and strides smaller
than the logical record. Output records contain only the recurrent core result;
the gate remains in a separate compiler-owned live range.

For correctness, Compiler traces use this order:

```text
Matrix projection -> projection scatter -> X_STATE issue -> FENCE(queue)
-> Vector gate/norm -> Matrix output projection
```

The `FENCE` immediately before Vector is required in asynchronous mode. A final
queue fence drains lifecycle commands before the program completes.
With `--async-pipeline` and at least two streaming requests, the trace generator
uses two projection/output buffers and queues 0/1. It may place request 1's
Matrix projection and X_STATE issue between request 0's issue and fence, so
Matrix work and the independent State command overlap without violating either
request's consumer dependency. Resident-cache pipelining remains disabled until
cross-queue cache lifecycle events are defined.

## Precision and scales

`activation_precision` describes input/output Vector SRAM values,
`parameter_precision` describes parameters fetched by `X_STATE`, and
`state_precision` describes the large recurrent matrix. `conv_state_precision`
independently describes the short convolution history; this is required for the
profiled KDA path, which uses FP32 recurrent state and BF16 convolution state.
Every update and reduction accumulates in FP32. BF16 and FP16 state is
requantized after each token, not only at the end of a chunk.

`MX8_B128` stores E4M3FN values plus one E8M0 scale byte per 128 values along
the innermost tensor dimension. Recurrent-state scale bytes precede
convolution-state scale bytes at `state_scale_addr`. Parameter scales follow
the parameter fetch order defined by the selected payload. The current
transactional Vector SRAM is not scale-aware, so MX8 state and parameters are
executable but MX8 activation VRAM is rejected rather than silently decoded
without scales.

## Reproducible contract checks

`tools/generate_state_golden.py` creates real-shape and tiny numerical
descriptors with the Compiler packer. The Simulator consumes the same JSON in
its Rust parser and numerical execution tests. `aten.state.lower_state_trace`
then lowers semantic Nemotron and Kimi schedules into an aligned descriptor
image plus X_STATE register writes and instruction words. General Matrix and
Vector events remain semantic until the existing PLENA physical allocator maps
their buffers and emits assembly.

## Executable projection layout

`PROJECTION_SCATTER` lowers to `L_SCATTER_M=0x3F` before the matching
`X_STATE`. It has a separate 256-byte, 64-byte-aligned descriptor with modes
`ROW_MAJOR`, `TRANSPOSE`, `MAMBA_SKEW`, `KDA_SKEW`, and `CUSTOM`. The descriptor
fixes the physical ping-pong buffer row, source Vector-SRAM address and stride,
Matrix burst width, FIFO capacity, spill policy, and every field's
logical-to-`(row,bank)` mapping. CRC32 protects the executable mapping; the
lowered JSON retains a SHA-256 debug view for DSE and review.

The frozen pre-RTL default is 16 single-port banks, a 64-value Matrix burst,
and a 64-value FIFO. The Simulator sweep observed a 64-value high-watermark and
identical cycles for 64 and 256 entries, so the larger default was removed.

Mamba fields are `gate/x/B/C/dt`. `gate` names the Vector service as its
consumer and must remain available after the state update; `x/B/C/dt` name
X_STATE. The flow field distinguishes fully buffered operation from a
FIFO-with-spill candidate, and all write/read/spill/backpressure cycles are
reported. The profiled KDA wrapper produces independent
`q/k/v/decay/beta` tensors rather than one packed projection. Its default
layout therefore materializes all five fields, uses group-major placement,
rotates `k` by eight banks so each PLENA `q[8]/k[8]/decay[8]` consumer packet
reaches the 16-bank, single-port service lower bound, and stripes scalar
`beta` values across banks by head so producer bursts do not serialize on bank
zero. Row-major and unrotated group-major mappings remain available as
ablations.

The layout mode is visible in the `L_SCATTER_M` word; detailed rotations and
buffer geometry live in its descriptor. Ablations therefore keep Matrix,
Vector, and `X_STATE` instructions identical and change only the layout command
and descriptor. The current transactional implementation sources the completed
Matrix result from Vector SRAM and never performs an HBM transpose/repack. A
future RTL can fuse the same architectural command into `M_MM_WO` writeback;
that fusion and its PPA are not claimed before RTL exists.
