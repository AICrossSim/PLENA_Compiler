# Recurrent sublayer compiler: current research implementation

Both repositories publish the complete implementation on
`feat/matrix-sram-lcompute`. The Simulator pins the matching Compiler commit as
its `PLENA_Compiler` submodule. The older `review/matrix-lcompute-20260905` PR
branch is a separate, smaller mechanism review; it is not this implementation.

## What is implemented

The main candidate as of 2026-10-01 uses **software-only projection mapping**.
`aten/plena/isa_projection_software.py` selects bounded request groups and
existing Vector SRAM allocations using resident M_MV. Its transposed schedule
uses the existing M_TMV opcode with offline N32-by-K weight packets and
K256/512/1024. Four output rows are read serially into the same finite Matrix
operand registers; larger K changes the BF16 reduction grouping and is tested
as a separate numerical contract. It introduces no new projection opcode or
payload storage relative to the common Matrix-view platform. M_MM.P, replay/slice
buffers and segmented reduction below remain historical hardware ablations,
not requirements of that candidate. The original packed M_MM address stride
has also been corrected to BLEN*MLEN and checked by machine execution with
partial sums surviving weight reloads. Full-size M_MM cycle calibration is
still required before calling this the best original Matrix mapping.

The compiler emits input/output projections, convolution, normalization, gate
and delta producers, coefficient placement, recurrence, and output processing
for Mamba/KDA sublayers. Requests have private inputs, outputs and state;
projection weight panels are shared. The executable boundary is input norm
through output projection, excluding the outer residual and FFN/MoE block.

| Module | Responsibility |
| --- | --- |
| `aten/plena/isa_matrix_projection.py` | Resident M_MV and compact/batched M_MM.P schedules; bounded input rows, K/N tails and private output merges |
| `aten/plena/isa_projection_software.py` | Request grouping, existing-row allocation and static-transposed M_TMV; explicit reloads and K boundaries, no new projection datapath |
| `aten/plena/recurrent_coefficients.py` | Executed BF16 gate/delta producers, gather and compact software coefficient reuse |
| `aten/plena/ltile_v2.py` | Fused delta update and explicit reduction lifetime |
| `aten/plena/ltile_native.py` | Compact coefficient descriptors and native Mamba/KDA recurrence lowering |
| `aten/plena/projection_handoff.py` | Optional static Vector-row retention; preserves HBM stores and charges copies |
| `aten/plena/mview.py` | View ownership, bounds, descriptor encoding and dominance checks |

The Simulator's `analytic_models/performance/ltile_layers.py` composes these
same lowerings into concrete sublayer programs. Python accounting consumes the
emitted instructions; it does not replace them with a measured speedup factor.

## Interfaces and arithmetic

- `L_TILE_CFG` retains four Matrix view slots. `L_TILE_CCFG slot, low, high`
  supplies three compact coefficient descriptors. Native update/reduction
  modes share the existing L_TILE physical opcode; there is no model-name ISA.
- `L_TILE_EXEC` has additional v2 delta/reduction modes. See `LTilePrimitive`
  in `mview.py` and `CoefficientView` in `ltile_native.py` for the checked encoding. Historical
  forms remain available; they have different arithmetic and evidence.
- `M_MM.P dst, weights, inputs, config, view` is an explicit M_MM function
  encoding. It accepts one to four requests, K<=256 and N=32 packets. The
  configuration holds rows-1 in bits 0..1, input stride/256 in bits 2..9, and
  output stride/32 in bits 10..17; the remaining bits are zero. Strides count
  BF16 elements. Legacy M_MM words remain unchanged. The compiler selects
  one-request or four-request tiles, including partial final tiles.
- The current projection candidate uses BF16 packet reductions and cross-K
  partial sums. Native recurrence uses BF16 operands/state, separate FP32
  update intermediates, BF16 RN commit and a BF16 pairwise tree. Ordinary
  Vector recurrence is a different arithmetic contract.

The historical M_MM.P projection candidate needs a finite 16 KiB weight replay buffer, 4 KiB
row-transfer buffer, 2 KiB useful-input storage and 256 B output hold, plus
selection/control. These are modeled hardware requirements, not free compiler
optimizations or synthesized area. No concurrent Matrix/recurrence execution
is claimed. The static handoff pass is not a generic Matrix-to-Vector copy.

## Validation

From this repository (Python 3.12, CPU torch, numpy, pytest and pyyaml):

```sh
PYTHONPATH=. python -m pytest \
  assembler/tests/test_l_mview.py assembler/tests/test_projection_mode.py \
  aten/tests/test_compact_projection.py aten/tests/test_projection_handoff.py \
  aten/tests/test_connected_layer_lowering.py aten/tests/test_ltile_v2.py \
  aten/tests/test_recurrent_gate_producer.py aten/tests/test_mview_contract.py
```

See the matching Simulator's `doc/l_tile_projection.md` for machine execution,
analytical reproduction, checked-in result tables and remaining limitations.
Full-shape B1/2/4/8/16 fixtures repeat a captured B1 request at private addresses;
separate synthetic tests use distinct requests and tails. One-token exact
same-arithmetic checks do not certify long-chain model quality.

## Still research work

Best legal original M_MM/M_MV mapping, generic SRAM-to-SRAM handoff, integrated
Matrix/recurrent overlap, complete-model numerical execution, and RTL/PPA are
not completed by this checkpoint. The `old_isa` recurrence control still uses
shared extended services; it is not an untouched original-PLENA baseline.
Proposed joint residency and WS/IS/OS selection are not reported as implemented.

## Publication checks

The focused assembler, projection, recurrence, ownership and CI-registration
suite passed 157 tests. A broader `assembler/tests aten/tests` run also exposed
two unrelated failures reproduced on the pre-upload remote commit `9fd6a59`:
the Qwen packed-router test expects an old assembly comment, and the cached
CLM-60M quantization diagnostic reaches 89.8% versus its >95% assertion in this
environment. These checks were not disabled or relabeled as passing. The
missing CI registration of the new projection tests was fixed and rechecked.
