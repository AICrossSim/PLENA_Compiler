# PLENA Generator

The generator path is the symbolic codegen and utilization-analysis pipeline:

```text
HF config -> symbolic graph -> scheduler -> ASM
```

It is separate from the ATen compiler/emulator path. For quick numerically verified
sim-sliced ATen checks, use:

```bash
python -m compiler.aten.sliced_emulator_runner AICrossSim/clm-60m --seq-len 64 --num-layers 1
```

For native hidden/intermediate dimensions, use
`compiler.aten.plena_frontend.compile_native_hf_decoder` from Python.

Run symbolic codegen:

```bash
python -m generator.runner codegen AICrossSim/clm-60m output.asm --seq-len 512
```

Run utilization analysis:

```bash
python -m generator.runner utilization AICrossSim/clm-60m dummy.asm --seq-len 512
```

Project dependencies and tooling live in the repository-root `pyproject.toml`.
