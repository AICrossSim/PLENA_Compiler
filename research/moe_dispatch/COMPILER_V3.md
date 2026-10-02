# Supply-first v3 compiler

`compiler_v3.py` is a separate front end; it does not modify `compiler.py` or the
previous ISA. It emits compact K-segment descriptors, real MX code/scale/factor
payloads and byte accounting. Runtime routing counts remain dynamic inputs.
Timing comes from the Rust simulator, not from Python packing or an analytic
peak-throughput estimate.

The canonical numerical and budget specifications live in the simulator
checkout. Set their path explicitly for a standalone compiler checkout:

```sh
PLENA_V3_REFERENCE_DIR=/path/to/simulator/research/moe_dispatch/v3_reference \
python research/moe_dispatch/compiler_v3.py \
  --workload-json /path/to/development.json --output /path/to/v3_plan.json \
  --lanes 4,2 --precision P2 --main-bits 4 --factor-a mxint4 \
  --factor-b bf16 --rank-lanes 8 --comp-mode lanes
```

Dimensions use `W[N,K]`, `X[Me,K]`, output `[Me,N]`. M×N×K hardware sizes are
`6×4×512`, `3×4×512` twice, and `4×4×512 + 2×4×512`. All have12,288 main
multipliers; the extra rank lanes are charged separately.

The default routed expert has H2048 and F1408; Shared has F2816. The exact
packed main-plus-factor bytes are4,970,112 and9,889,920 respectively. Main
weight tile counts are4,352 and8,704. Routed A prepasses use82 tiles per M wave;
A8-split doubles their issue count without doubling fetched bytes. Down's final
K segment has384 live columns and its W4-plus-B tile is896B.

`pack_projection` emits a byte string and physical offsets, with signed code
packing (including3-bit codes and INT4-8), E8M0 scales, BF16 B fragments and
32B transaction alignment. `unpack_projection` reconstructs operands solely
from that byte string. Tests compare actual decoded operands against quantized
references for nine A/B format combinations and check default full expert
payload length. No original float tensor is used by the decoder.

`projection_layout` keeps an expert's compact N-loop descriptors instead of
materializing thousands of runtime tasks. A `main` descriptor identifies K
segment, valid K, deterministic N expansion, payload size and embedded rank
indices; `a_prepass` and `b_tail` descriptors explicitly charge their traffic
and issues. Streamed Down puts fused ranks in the last K segment and emits
lane-only tails for the remaining ranks. P2 legality checks reject BF16 A,
K extension, slack packing, and separate/offload with non-INT4 B.

WS A prepasses visit groups of at most32 rank columns, then ascend K segments,
and drain each completed group into BF16 U before advancing. This fits the fixed
FP32 accumulator for L16/T96. The physical A file remains K-segment-major;
the AGU derives each rank group's offsets. IS may use K-first traversal. These
orders change neither unique A bytes nor MAC count.

Streamed WS Down visits K segments outside output N groups. It atomically drains
each group's FP32 delta directly to the shared Combine buffer after every main
K segment and lane-only tail group. It never retains a private Me-by-H output
matrix. This charges extra Combine RMW traffic and follows actual event order;
interleaved experts can therefore have different FP32 rounding from merging
whole expert outputs. The numerical oracle replays the ordered drain events.

BF16 U is physically packed by the K segment that consumes each rank lane:
flatten the segment rank lists, then retain the logical order of any tail ranks.
`u_bf16_layout` gives both permutation directions, contiguous segment slices and
the tail offset. UStore performs this permutation from logical FP32 U_d partial
state without allocating extra bytes. Streamed Down has identity order because
its fused ranks all reside in the last segment. Numerical gold keeps logical
rank indexing; only the physical BF16 addressing changes.

`storage` reports **frozen hardware capacity**, while `task_live_storage` gives
the current workload's live bound. Tcapacity is128 for P2/L8,96 for P1/L16
and the enlarged512GB/s landing pool. Hardware cannot resize for each batch.
Single and homogeneous cores use switchable WS/IS backing stores; Current
and Next share operand, accumulator and WOR resources. Eighteen WOR slots and
aggregate pool/X/decoder limits are identical in iso-port comparisons. Global
Z/U pools require runtime admission credits. Formats or organizations exceeding
2,158,592B fail compilation. The specialized stream core has two whole-K-slice
X buffers, each up to4×512 BF16 values:8KiB total, independent of whether the
physical row width is2 or3. Me4 therefore uses two or more row waves from the
same resident slice, without creating separate X copies per wave.

The single switchable array has two compute contexts. Interleaving is legal only
when the sum of both live accumulator footprints fits that shared fixed arena,
both Z/U reservations fit, and both operand groups fit WOR. WS uses Me×32×4B;
IS uses Me×max(2F,H)×4B. Contexts have disjoint addresses and a charged switch;
an inadmissible second context remains prefetch-only. This does not allocate a
second full accumulator or let either context overwrite the other. The
`inline_silu=false` ablation uses a finite deferred SiLU actor: WS spills and
reads its group through the second half of the existing double-buffered arena,
so each live context needs twice the WS footprint; IS uses its existing whole
Gate/Up backing. Spill/read bandwidth and completion dependencies remain charged.

Switchable IS supports the full physical M for routed experts. Its fixed
accumulator is `max(WS group capacity, M×max(2Ir,d)×4B)`. A particular descriptor
uses IS only if Me≤M **and** its real `Me×max(2F,H)×4B` fits this arena. Shared's
larger F can require WS even at small Me; this is a capacity check used by every
switchable organization. Provisioning worst-case full-M Shared IS instead would
exceed the fixed T128 budget by16,704B for single6 and65,856B for single8.

The compiler's JSON adds `v3_engine_layout`, `byte_stats`, `storage`,
`task_live_storage` and `v3_config` without removing the captured routing inputs.
The input provenance is copied; a synthetic fixture remains labeled synthetic.
