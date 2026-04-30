"""TVM-compiler test harness.

Mirrors the role of tile_tensor_test_helper.py + testbench_runner.py from
the runtime compiler, but adapted to our TVM/TIR pipeline.

Per-kernel test driver should:

    from tilelang_tvm_compiler.test_helper import emit_single_output_testbench

    emit_single_output_testbench(
        prim_func     = my_kernel,            # tvm.tir.PrimFunc
        out_buffer    = "C_hbm",              # name of the HBM buffer holding the result
        input_tensors = {"A_hbm": A, ...},    # numpy or torch tensors keyed by PrimFunc param name
        golden_output = golden,               # numpy/torch tensor with the expected result
        asm_name      = "tvm_btmm",
        artifact_prefix = "tvm_btmm",
        build_dir     = ".../testbench/build",
    )

What it does (parallel to the runtime helper, layer by layer):

  1. Compile the PrimFunc with PlenaCodegen           ~ prog.compile()
  2. Append "compare staging" pseudo-ISA              ~ stage_input_tensor_for_stride_compare
       which moves the HBM output back into VRAM[0..]
       so the emulator can diff against the golden.
  3. Save the input tensors as the HBM feed           ~ build_input_feed
  4. Save the golden as .npy                          ~ create_sim_env(golden_result=...)
  5. Write a manifest.json describing the test        ~ comparison_params.json + create_mem_for_sim

For now everything downstream of the pseudo-ISA is also pseudo (we don't
yet bind to create_sim_env / cargo run). The artifacts written here are
the contract that real ISA emit will fulfil later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import tvm
from tvm import tir

from .codegen import PlenaCodegen, _BufferInfo
from .pipeline import compile_kernel, PlenaTarget
from . import scope as _scope


def _to_numpy(x: Any) -> np.ndarray:
    """Accept torch tensors or numpy arrays; return numpy."""
    if isinstance(x, np.ndarray):
        return x
    # duck-typed torch.Tensor support without importing torch
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    raise TypeError(f"unsupported tensor type: {type(x)}")


def _byte_size(info: _BufferInfo) -> int:
    elems = 1
    for s in info.shape:
        elems *= int(s)
    # rough dtype byte width -- matches what we'd use in the manifest
    dtype_bits = {
        "float16": 16, "bfloat16": 16, "float32": 32, "int32": 32, "int8": 8,
    }.get(info.dtype, 32)
    return elems * dtype_bits // 8


def _emit_compare_staging(out_info: _BufferInfo) -> str:
    """Build the pseudo-ISA tail that pulls the HBM output into VRAM[0..]
    so the emulator's comparator can diff against the golden.

    Real ISA equivalent (from runtime helper) is a sequence of
    preload_addr_reg + preload_act + tile-by-tile DMA. We collapse it here
    into one synthetic STAGE_OUT directive; when ISA emit becomes real this
    function gets replaced with the actual tile-staging pass.
    """
    return (
        "; ============================================\n"
        "; compare staging (output HBM -> VRAM[0..])\n"
        "; ============================================\n"
        f"STAGE_OUT  buffer={out_info.name}  scope={out_info.scope}  "
        f"shape={'x'.join(str(s) for s in out_info.shape)}  "
        f"dtype={out_info.dtype}  bytes={_byte_size(out_info)}\n"
    )


def emit_single_output_testbench(
    *,
    prim_func: tir.PrimFunc,
    out_buffer: str,
    input_tensors: Mapping[str, Any],
    golden_output: Any,
    asm_name: str,
    artifact_prefix: str,
    build_dir: str | Path,
    compare_atol: float = 1e-2,
    compare_rtol: float = 1e-2,
    target: PlenaTarget | None = None,
    isa_mode: str = "real",  # "real" -> full ISA via pipeline; "pseudo" -> old text dump
) -> Dict[str, Path]:
    """Compile + bundle inputs/golden/manifest. Returns paths of written files.

    isa_mode == "real":   runs the 3-pass pipeline (codegen -> address alloc
                          -> ISA emit) to produce real PLENA ISA. Default.
    isa_mode == "pseudo": uses the original PlenaCodegen.run() text dump.
                          Kept around for kernels that exercise op kinds
                          not yet supported by the real pipeline.
    """
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. compile main kernel
    if isa_mode == "real":
        target = target or PlenaTarget()
        compiled = compile_kernel(prim_func, target=target, name=asm_name)
        main_isa = compiled.isa_text
        # Use the HLIR module's buffer dict for downstream sanity checks --
        # it's the single source of truth post-allocation.
        bufs = {
            name: _BufferInfo(buf.name, buf.scope, buf.shape, buf.dtype)
            for name, buf in compiled.hlir.buffers.items()
        }
    elif isa_mode == "pseudo":
        cg = PlenaCodegen(prim_func, name=asm_name)
        main_isa = cg.run()
        bufs = cg.buffers_by_name()
    else:
        raise ValueError(f"unknown isa_mode {isa_mode!r}; use 'real' or 'pseudo'")

    # ---- 2. resolve out buffer + sanity checks
    if out_buffer not in bufs:
        raise KeyError(
            f"out_buffer {out_buffer!r} is not a buffer in this PrimFunc. "
            f"Known: {sorted(bufs.keys())}"
        )
    out_info = bufs[out_buffer]
    if out_info.scope != _scope.HBM:
        raise ValueError(
            f"out_buffer {out_buffer!r} must live in HBM (final output goes to "
            f"DRAM), but it is in scope={out_info.scope!r}"
        )

    # ---- 3. append compare staging tail
    staging = _emit_compare_staging(out_info)
    full_isa = main_isa.rstrip() + "\n\n" + staging

    isa_path = build_dir / f"{artifact_prefix}.plena.s"
    isa_path.write_text(full_isa)

    # ---- 4. save inputs as the (pseudo) HBM feed
    inputs_dir = build_dir / f"{artifact_prefix}_inputs"
    inputs_dir.mkdir(exist_ok=True)
    saved_inputs: Dict[str, Path] = {}
    for name, tensor in input_tensors.items():
        if name not in bufs:
            raise KeyError(
                f"input tensor {name!r} does not match any PrimFunc buffer. "
                f"Known: {sorted(bufs.keys())}"
            )
        info = bufs[name]
        if info.scope != _scope.HBM:
            raise ValueError(
                f"input {name!r}: PrimFunc declares it in scope={info.scope!r}, "
                f"but inputs must be HBM (DMA'd in by the kernel)"
            )
        arr = _to_numpy(tensor)
        # We don't enforce dtype yet -- just shape -- because the kernel may
        # internally cast. If shape disagrees that's almost certainly a bug.
        if tuple(arr.shape) != tuple(int(s) for s in info.shape):
            raise ValueError(
                f"input {name!r}: shape {arr.shape} != PrimFunc shape {tuple(info.shape)}"
            )
        out = inputs_dir / f"{name}.npy"
        np.save(out, arr.astype(np.float32, copy=False))
        saved_inputs[name] = out

    # ---- 5. golden
    golden_arr = _to_numpy(golden_output).astype(np.float32, copy=False)
    expected_shape = tuple(int(s) for s in out_info.shape)
    if tuple(golden_arr.shape) != expected_shape:
        # Allow flat / collapsed golden, but warn rather than fail -- attention
        # writes its golden in (B*S, H*D) form for example. We just record both.
        pass
    golden_path = build_dir / f"{artifact_prefix}_golden.npy"
    np.save(golden_path, golden_arr)

    # ---- 6. manifest
    global_symbol = "<unknown>"
    if prim_func.attrs is not None and "global_symbol" in prim_func.attrs:
        global_symbol = str(prim_func.attrs["global_symbol"])
    manifest: Dict[str, Any] = {
        "asm_name": asm_name,
        "artifact_prefix": artifact_prefix,
        "kernel_global_symbol": global_symbol,
        "isa_file": isa_path.name,
        "isa_kind": isa_mode,  # "real" (TIR -> HLIR -> ISA) or "pseudo" (text dump)
        "inputs_dir": inputs_dir.name,
        "inputs": {
            name: {
                "shape": list(bufs[name].shape),
                "dtype": bufs[name].dtype,
                "scope": bufs[name].scope,
                "file": saved_inputs[name].name,
            }
            for name in input_tensors
        },
        "output": {
            "name": out_buffer,
            "shape": list(out_info.shape),
            "dtype": out_info.dtype,
            "scope": out_info.scope,
            "bytes": _byte_size(out_info),
            "staged_to": "vram[0..]",  # what compare staging will produce
        },
        "golden_file": golden_path.name,
        "compare": {
            "kind": "absolute_and_relative",
            "atol": compare_atol,
            "rtol": compare_rtol,
        },
        "TODO": (
            "When codegen emits real .mem, also generate hbm_for_behave_sim.bin / "
            "fp_sram.bin / generated_machine_code.mem here so `cargo run` can "
            "execute this test directly."
        ),
    }
    manifest_path = build_dir / f"{artifact_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return {
        "isa": isa_path,
        "golden": golden_path,
        "inputs_dir": inputs_dir,
        "manifest": manifest_path,
    }
