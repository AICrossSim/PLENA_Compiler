# MoE normal-buffer exporter: review guide

This change supplies the input and independent numerical reference for the paired
Simulator MoE experiment. It exports fixed routes; it does not add a runtime
router or lower dual-core execution into the existing instruction stream.

## What to review

1. `aten/plena/moe_normal_export.py`: validate inputs/routes, group routes by
   expert while retaining token/slot/weight, encode weights, and emit a versioned
   workload plus an independently computed golden output.
2. `aten/plena/moe_full_shape_export.py`: generate deterministic nonzero arrays
   at specified model dimensions and vectorize independent oracle outputs. Each
   dot product still accumulates in ascending K with separate FP32 multiply/add.
3. The two corresponding tests in `aten/tests`: codec bytes, row padding,
   malformed routes, duplicate expert selections, shared experts and numerical
   agreement between the scalar and vectorized paths.

## Data contract

`X[T,D]` enters an expert with gate/up weights `[F,D]` and down weights `[D,F]`.
Those are physical output-major weight rows; logical computation is
`gate=X×Wgateᵀ`, `up=X×Wupᵀ`, `Z=SiLU(gate)×up`, `Y=Z×Wdownᵀ`.
The transposed mathematical notation does not require a runtime transpose buffer.
Each projection and SwiGLU output rounds to BF16; route-weighted combine uses
stable token/slot order with FP32 accumulation and final BF16 rounding.

Weights use the actual local PLENA E4M3/E8M0 codec, with 8 elements per scale.
Elements and scales occupy separate address ranges, with explicit row strides
and tail padding. The oracle decodes the exported bytes before computing.
This local codec is not a claim of interchangeability with every MX format.

Files: `workload.json` (shapes/routes/addresses), `weights.bin` (encoded bytes),
`golden.json` (expected output), and `source_arrays.json` for the small exporter.
Hashes bind the workload to the weight image and record its provenance.

## Run

Use Python 3.10+ with torch, NumPy, bitstring, PyYAML and pytest installed. Set `PLENA_TOOLS` to a
checkout of the Simulator's pinned PLENA_Tools submodule. From this repository:

```sh
PYTHONPATH="$PWD:$PLENA_TOOLS" python -m pytest \
  aten/tests/test_moe_normal_export.py aten/tests/test_moe_full_shape_export.py -q
PYTHONPATH="$PLENA_TOOLS" python aten/plena/moe_normal_export.py \
  --output-dir /tmp/plena-moe-fixture --shared
```

The paired Simulator PR contains the single/dual configurations and a smoke
command that invokes this exporter, runs Rust twice per configuration and checks
exact BF16 output, resource limits, HBM counters and repeat determinism.

## Scope and result

The review checkout passes **21 exporter tests**. The previous performance
campaign used D=2048 with F=512 (Qwen) or F=1408 (DeepSeek), token windows 8/32,
archived routes and synthetic nonzero weights/activations. The best tested dual
configuration took 8.88% more simulated time than the optimized single baseline;
this exporter makes the comparison reproducible, not faster by itself.
See the paired Simulator review for core/SRAM dimensions and timing boundaries.
