# Fresh Compiler review validation — 2026-09-05

Draft review only; not a merge or RTL acceptance gate.

| Check | Result |
|---|---|
| Paired Simulator's `test-matrix-lcompute-compiler` file list, including the new compatibility regression | 231 passed, exit 0 |
| Focused enum/address/recurrence/layout checks | 112 passed, exit 0; overlaps the larger gate above |
| Default generated assembly before/after review fixes | Nemotron/Kimi x fixed/affine: all four SHA256 values unchanged |
| Forced Python 3.10 enum fallback on the available Python 3.11 interpreter | Imports, string/format/JSON/auto semantics and KDA assembly smoke passed |
| Whitespace/conflict check | `git diff --check` passed |

The available environment does not contain Python 3.10; this is not a claim
that the complete suite ran under that interpreter. The 231-test run emitted
one warning about NumPy being absent from this environment; no test failed or
was skipped in that gate. No real GPU or new checkpoint download was used.

Review fixes are limited to Python compatibility and explicit HBM arena
validation. They do not change recurrence formulas, rounding rules, default
experiment flags, or the four default recurrence programs.

The gate list is maintained in the paired Simulator's `justfile` under
`test-matrix-lcompute-compiler`; it covers assembler/view encoding, dominance,
layouts, hybrid schedules, precision contracts, recurrence lowering, optional
controls and projection writeback. The Compiler CI list now includes the
optional-control and Python compatibility guards too.

Run against the paired checked-out Simulator settings, for example:

```bash
PLENA_SETTINGS_TOML=/path/to/Simulator/plena_settings.toml \
PYTHONPATH=. python -m pytest -q \
  aten/tests/test_python_310_enum_compat.py \
  aten/tests/test_matrix_recurrence_lowering.py \
  aten/tests/test_hybrid_l_tile_schedule.py \
  aten/tests/test_mview_contract.py \
  aten/tests/test_affine_layout.py
```

The paired Simulator PR records Rust workspace, projection and connected
numerical execution separately. This file does not claim full real-checkpoint
layer/model execution, all repository test suites, physical hardware reuse,
RTL timing, PPA, or a new workload-level speedup.
