# Heterogeneous MoE dispatch planner

Research branch: `research/moe-heterogeneous-dispatch`.

This frontend compiles captured MoE routes into **explicit task and private-memory
plans** for one `6x4x512`, two `3x4x512`, or `4x4x512 + 2x4x512` engines. Dimensions
are **M x N x K**; all organizations have 12,288 multipliers. Runtime ownership is
chosen by the companion Simulator, before weight requests begin.

The planner describes complete gate/up/activation/down experts, storage lifetimes,
addresses, capacity checks, and two execution templates:

- **whole:** keep an expert instance on one core through retirement.
- **paired_n:** split output columns, explicitly exchange intermediate Z, then
  synchronize before down. Partial sums never migrate for free.

This is an independent research frontend, **not production PLENA ISA lowering**.
It does not run the model or estimate latency. Online learned dispatch prediction
is future work; the current Simulator uses an explicit shape/service heuristic.

## Run from the Compiler repository root

Python 3.10+ and the standard library are sufficient. No model weights, Torch,
private workspace paths, or original capture files are needed for bundled replay.

```bash
python3 -m unittest discover -s research/moe_dispatch -p 'test_*.py' -v
python3 research/moe_dispatch/compiler.py \
  --output /tmp/moe-dispatch/workloads.json \
  --plans-dir /tmp/moe-dispatch/plans \
  --engine-layouts-dir /tmp/moe-dispatch/layouts
```

Outputs: a workload bundle, 24 detailed plans (4 batches x 3 organizations x 2
templates), and 12 compact runtime layouts. Files declare schema versions, byte
addresses, per-core capacity, useful MACs, physical tile counts and copy traffic.

To import a different original capture, provide `--manifest capture/manifest.json
--config model/config.json`. The importer verifies the supplied input payload's
hash; input/dataset paths may be relative to the manifest. Large weight hashes
remain capture metadata, not a fresh verification of model weights.

## Input and evidence scope

`fixtures/workloads.json` preserves DeepSeek-V2-Lite-Chat first-MoE-layer BFCL
last-token prefill routes for B2/B4/B8/B16: H=2048, routed F=1408, shared F=2816,
top-k=6. These are nested prefixes of one 16-token capture, not four independent
benchmarks or a decode trajectory. Router latency is excluded.

`fixtures/captured_routes.json` independently retains the capture's route/score
matrix for contract checks. `SHA256SUMS` checks the bundled metadata. Historical
source hashes are retained, but `capture://` references are provenance identifiers,
not files to open; original inputs, weights and prompts are not bundled.

## Companion Simulator

[PLENA_Simulator, same branch](https://github.com/AICrossSim/PLENA_Simulator/tree/research/moe-heterogeneous-dispatch/research/moe_dispatch)
pins this Compiler commit through its existing `PLENA_Compiler` submodule. It
imports `engine_layout()` directly and hashes the selected planner in every
experiment. There is no duplicate editable planner in the Simulator.

See [COMPILER_METHODS.md](COMPILER_METHODS.md) for layouts, copies and accounting.
The Simulator contains the analytical assumptions, numerical audit and result
tables; none of those are native HBM or full-model inference measurements.
