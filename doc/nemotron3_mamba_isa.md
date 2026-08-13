# Proposed PLENA Nemotron 3 Mamba ISA Contract

Status: executable compiler contract; **not yet implemented by RTL or the
transactional simulator**. Opcode allocation is provisional until Compiler,
Simulator, and RTL first agree on existing opcodes `0x35` through `0x38`.

## Design rule

Use one coarse-grained asynchronous command, `X_MAMBA = 0x39`. The instruction
points to a 256-byte descriptor. State precision, projection layout, shapes,
and addresses are descriptor fields, not separate opcodes. This preserves the
remaining six 6-bit opcode values and lets one binary compare row-major and
skewed hardware.

## Instruction encoding

```text
31       26 25       22 21       18 17       14 13       10 9         6 5       0
+-----------+-----------+-----------+-----------+-----------+-----------+---------+
| reserved  | subop     | queue_id  | desc_hbm  | desc_off  | context   | 0x39    |
+-----------+-----------+-----------+-----------+-----------+-----------+---------+
```

- `context`: GP register holding runtime/context ID.
- `desc_off`: GP register holding the descriptor byte offset.
- `desc_hbm`: address register `a0..a7` holding the descriptor base.
- `queue_id`: asynchronous Mamba command queue.
- bits `26..31` must be zero in version 1.

## Sub-operations

| Value | Name | Required precondition | Architectural effect |
|---:|---|---|---|
| 0 | `STATE_PREFETCH` | state is clean in HBM | Load SSM + conv state into a cache or transient stream slot. |
| 1 | `STATE_RESET` | state is absent or clean | Zero SSM + conv state and mark it dirty. Used for a fresh prefill. |
| 2 | `PREFILL` | state is resident | Consume `valid_tokens <= chunk_size`; causal conv and SSD recurrence continue from the resident state. |
| 3 | `STEP` | state is resident, `valid_tokens=1` | Consume one projected token and produce one post-gate/norm vector. |
| 4 | `STATE_COMMIT` | state is resident and dirty | Write SSM + conv state and scales to HBM; keep a clean resident copy. |
| 5 | `STATE_EVICT` | state is resident and clean | Release the cache slot. Dirty eviction is illegal. |
| 6 | `WAIT` | none | Wait for prior commands on the selected queue and surface completion faults. |

`PREFILL` and `STEP` have identical numerical semantics. `PREFILL` permits a
chunked SSD implementation; `STEP` selects the lower-latency recurrent path.

## Descriptor v1

All values are little-endian. The descriptor is 256 bytes and 64-byte aligned.
Unused bytes are zero and reserved for future versions.

| Offset | Type | Field |
|---:|---|---|
| 0 | `u32` | magic `0x4D324D42` |
| 4 | `u16` | version `1` |
| 6 | `u16` | size `256` |
| 8 | `u32` | flags: B/C group-shared, continue-state, last-chunk |
| 12 | `u32` | batch size |
| 16 | `u32` | sequence length; supports contexts beyond 64K |
| 20 | `u32` | request ID |
| 24 | `u32` | token/chunk offset |
| 28 | `u32` | completion ID |
| 32..48 | `u16` | heads, head dim, state dim, groups, conv kernel, chunk size, valid tokens, layer ID, cache slot |
| 50..52 | `u8` | state precision, activation precision, projection layout |
| 64..160 | `u64` | projection/output/state/conv/parameter/scale/completion addresses |

The exact pack/unpack implementation is in
`aten/mamba/contract.py`. Version 1 fixes Nemotron's group-shared B/C semantic:
B and C are stored once per group and broadcast to eight heads. Software must
not expand them to 64 heads before transfer.

## Precision semantics

- State update, exponential, C reduction, and accumulation execute in FP32.
- `state_precision` controls only the persistent SSM/conv state store.
- `MX8_B128` means E4M3FN values and one power-of-two scale per 128 values along
  the state dimension. It is a PLENA candidate, not strict OCP MXFP8 block-32.
- Activation and weight quantization remain controlled by existing Matrix and
  Vector services. `X_MAMBA` does not silently change projection weights.

## Layout semantics

`ROW_MAJOR` and `GROUP_MAJOR_SKEWED` are architecturally equivalent. The latter
asks L-compute to scatter the linear projection into the physical bank layout
of a dedicated Mamba projection buffer on the Matrix-result writeback path.
The layout is not a new instruction and does not change numerical results.

## Faults and completion

The engine must report at least: invalid descriptor/version, illegal shape,
misaligned address, dirty eviction, state-cache miss on `STEP/PREFILL`, queue
fault, and unsupported precision/layout. Performance counters are not part of
the completion record and should be exposed as CSRs.
