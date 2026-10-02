# Legacy token chunks

`legacy_chunks.prepare_legacy_workload(workload, lanes, group=4, resources=None)`
extends capacity for a legacy `joint_v1` workload whose original whole-expert
plan has an expert with no feasible core. If every expert already has a feasible
whole placement, the returned deep copy contains exactly the original compiler
layout and no outer execution plan. `compiler.py` and its supported kernels are
unchanged. This helper does not change `supply_v3`.

The extension retains, on every physical core, the declared control reservation,
the complete original route metadata, that core's column partition of the full
layer's BF16 input X, and the same columns of the full FP32 final Y. These
persistent regions must fit even on a core that receives no expert. The wrapper
occupies 32 bytes of existing controller headroom; insufficient headroom or an
oversized persistent arena produces an explicit `ValueError`.

Candidate token chunk sizes are tested in descending order. The selected size is
the largest for which **every actual contiguous chunk** gives every active
expert, including Shared, at least one feasible whole-core placement. This uses
the original compiler's exact workspace requirements and the captured route
distribution. All active expert FP32 inboxes are placed after the full persistent
arena. Whole and paired workspace allocations and their aliases are uniformly
rebased after those inboxes; capacity and peak checks use the resulting physical
addresses. Hardware parameters and capacities are unchanged.

The original workload and engine layout remain intact. The added
`legacy_batch_execution` object has schema `plena_legacy_token_chunks_v1`, a
`chunk_size`, `persistent_cores`, a `control_record`, and ordered
`chunks: [{token_range: [start, end], workload: child}]`. Child token indices are
local rows; `original_token_indices` retains their original positions. Expert
IDs, order, weights, route slots, and scores are preserved. Shared retains every
token, and every original routed row appears exactly once across the children.

This is a distinct large-batch timing protocol. Each child runs and completely
drains the original legacy kernel, including its charged input gather, expert
retirement, and ordered combine. Child X and final-Y views alias their global
rows, so the wrapper adds no payload copies. Descriptor setup uses existing
controller and accumulator ports. Kernel state, DMA streams, and weight storage
restart for each chunk; there is no overlap or weight cache across boundaries.
Consequently `weight_read_bytes` includes actual repeated HBM weight transfers,
while `unique_weight_bytes` describes the original unique expert weights.

The helper plans finite storage for analytical timing. It does not execute tensor
payloads, claim equivalence to one infeasible full-window run, or claim full-model
inference. Numerical validation belongs to the separately checked legacy kernels
and outer route/row mapping.

Run the capacity, conservation, alias, control, and import-isolation checks from
this directory with `python3.12 -m unittest -q test_legacy_chunks test_compiler`.
