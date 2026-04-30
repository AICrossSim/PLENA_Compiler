"""CLI entry for cross-venv invocation.

The TVM compiler runs in a Python 3.11 venv (.venv-tvm) because no
apache-tvm wheel exists for 3.12. The rest of the project (torch, sim
env utilities, golden compute) lives in the main 3.12 venv.

Driver scripts in the main venv subprocess into here to get the ISA
text without dragging TVM into their own interpreter:

    out = subprocess.check_output([
        ".venv-tvm/bin/python", "-m", "tilelang_tvm_compiler",
        "compile",
        "--kernel", "tilelang_tvm_compiler.kernels.minimal_btmm:minimal_btmm",
        "--asm-name", "tvm_btmm_kernel",
    ], env={"LD_LIBRARY_PATH": "", ...}).decode()

`out` is the full ISA text, ready to drop into create_sim_env's
`generated_code` argument.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from .pipeline import compile_kernel, PlenaTarget
from .program_shim import make_shim
from .register_alloc import RegisterAllocator
from .isa_emitter import ISAEmitter
from .hlir import format_hlir
from . import scope as _scope


def _parse_kernel_kwargs(spec: str | None) -> dict:
    """Parse a comma-separated `k=v,k=v` string into a kwargs dict.
    Values are coerced to int if possible (the only case we currently
    need for shape parameters)."""
    if not spec:
        return {}
    out: dict = {}
    for pair in spec.split(","):
        if not pair.strip():
            continue
        if "=" not in pair:
            raise SystemExit(
                f"--kernel-kwargs entry must be key=value, got {pair!r}"
            )
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            out[k] = v
    return out


def _resolve_kernel(spec: str, kwargs: dict | None = None):
    """Resolve a `module.path:symbol` string into a TIR PrimFunc.

    `symbol` can be either:
      * a `tir.PrimFunc` directly (no kwargs accepted)
      * a callable factory; we call it with `kwargs` and accept either a
        PrimFunc or a `(PrimFunc, ...)` tuple as the return value
    """
    from tvm import tir
    if ":" not in spec:
        raise SystemExit(
            f"--kernel must be of the form module:funcname, got {spec!r}"
        )
    mod_path, func_name = spec.split(":", 1)
    mod = importlib.import_module(mod_path)
    if not hasattr(mod, func_name):
        raise SystemExit(f"{mod_path!r} has no attribute {func_name!r}")
    obj = getattr(mod, func_name)
    if isinstance(obj, tir.PrimFunc):
        if kwargs:
            raise SystemExit(
                f"{func_name!r} is already a PrimFunc; --kernel-kwargs not allowed"
            )
        return obj
    if callable(obj):
        result = obj(**(kwargs or {}))
        if isinstance(result, tuple):
            # Factories like make_tiled_btmm return (PrimFunc, constants).
            result = result[0]
        if not isinstance(result, tir.PrimFunc):
            raise SystemExit(
                f"factory {func_name!r} returned {type(result).__name__}, "
                f"expected tir.PrimFunc"
            )
        return result
    raise SystemExit(
        f"{func_name!r} is neither PrimFunc nor callable: {type(obj).__name__}"
    )


def _emit_output_staging(
    compiled,
    target: PlenaTarget,
    out_buffer_name: str,
) -> str:
    """Append "load output back to VRAM[0..]" ISA so view_mem can compare.

    The output buffer ends up in HBM after the main kernel. To check it
    against the golden, the runtime convention is to drop tile-by-tile
    DMAs at the end of the program that re-load HBM into VRAM[0..],
    laid out tile-major. view_mem.py then reads that VRAM region and
    compares against golden_result.txt.

    For a 2D logical view (rows, cols) of the output, we walk col-blocks
    first, then row-blocks (matches the runtime helper's "stage_order:
    col_major"), and emit one emit_load_tile_from_hbm per tile.
    """
    buf = compiled.hlir.get_buffer(out_buffer_name)
    if buf.scope != _scope.HBM:
        raise SystemExit(
            f"--stage-output buffer {out_buffer_name!r} must be in HBM, "
            f"got {buf.scope!r}"
        )
    rows, cols = _logical_2d(buf.shape)
    mlen = target.mlen
    if rows % mlen or cols % mlen:
        raise SystemExit(
            f"staging only supports mlen-aligned shapes for now, got "
            f"rows={rows} cols={cols} mlen={mlen}"
        )
    row_blocks = rows // mlen
    col_blocks = cols // mlen
    tile_elems = mlen * mlen

    shim = make_shim(
        mlen=target.mlen,
        blen=target.blen,
        btmm_lane_count=target.btmm_lane_count,
        btmm_hlen=target.btmm_hlen,
        register_allocator=RegisterAllocator(),
    )
    emitter = ISAEmitter(shim)

    shim.compiler.generated_code = (
        "\n; ============================================================\n"
        f"; compare staging: {out_buffer_name} (HBM @ {buf.address}) -> VRAM[0..]\n"
        f"; layout: rows={rows} cols={cols} -> {row_blocks}x{col_blocks} tiles "
        f"({mlen}x{mlen} each), col-block-major\n"
        "; ============================================================\n"
    )

    # SCALE register must be set to the FULL HBM tensor size (rows*cols),
    # not to a single tile. This matches the spec: "scale offset specifies
    # the distance between data blocks and their scale factors in HBM",
    # which is keyed off the tensor's total element count.
    full_tensor_size = rows * cols
    vram_addr = 0
    for j in range(col_blocks):
        for i in range(row_blocks):
            hbm_offset_elems = i * mlen * cols + j * mlen
            shim.compiler.generated_code += (
                f"; stage tile [{i},{j}]  hbm_offset(elems)={hbm_offset_elems}  "
                f"-> vram[{vram_addr}]\n"
            )
            emitter.emit_load_tile_from_hbm(
                hbm_addr=buf.address,
                vram_addr=vram_addr,
                hbm_stride=cols,                 # full row stride
                hbm_scale_size=full_tensor_size,  # full tensor, NOT one tile
                hbm_start_offset=hbm_offset_elems,
            )
            vram_addr += tile_elems

    return shim.compiler.generated_code


def _logical_2d(shape) -> tuple[int, int]:
    """Same BSHD-aware collapse as address_alloc._logical_2d. Kept inline
    here so the CLI doesn't take a hard dep on the address pass module."""
    if len(shape) == 0:
        return (1, 1)
    if len(shape) == 1:
        return (1, int(shape[0]))
    if len(shape) == 2:
        return (int(shape[0]), int(shape[1]))
    rows = 1
    for s in shape[:-2]:
        rows *= int(s)
    cols = int(shape[-2]) * int(shape[-1])
    return (rows, cols)


def _cmd_compile(args: argparse.Namespace) -> int:
    kernel_kwargs = _parse_kernel_kwargs(args.kernel_kwargs)
    func = _resolve_kernel(args.kernel, kernel_kwargs)
    target = PlenaTarget(
        mlen=args.mlen,
        blen=args.blen,
        btmm_lane_count=args.btmm_lane_count,
        btmm_hlen=args.btmm_hlen,
    )
    compiled = compile_kernel(func, target=target, name=args.asm_name)
    isa_text = compiled.isa_text
    if args.stage_output:
        isa_text = isa_text.rstrip() + _emit_output_staging(
            compiled, target, args.stage_output,
        )

    if args.dump_hlir:
        Path(args.dump_hlir).write_text(format_hlir(compiled.hlir))

    if args.output:
        Path(args.output).write_text(isa_text)
    else:
        sys.stdout.write(isa_text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tilelang_tvm_compiler")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compile = sub.add_parser(
        "compile",
        help="Compile a TIR PrimFunc to PLENA ISA text.",
    )
    p_compile.add_argument(
        "--kernel",
        required=True,
        help='Kernel spec, e.g. "tilelang_tvm_compiler.kernels.minimal_btmm:minimal_btmm"; '
             'may also point at a factory function used together with --kernel-kwargs',
    )
    p_compile.add_argument(
        "--kernel-kwargs",
        default=None,
        help="Comma-separated k=v pairs to pass when --kernel resolves to a "
             "factory (e.g. `seq_q=128,seq_k=128`). Values are coerced to int "
             "when possible.",
    )
    p_compile.add_argument("--asm-name", default="kernel")
    p_compile.add_argument("--output", default=None,
                           help="If given, write ISA to this path; else stdout.")
    p_compile.add_argument("--mlen", type=int, default=64)
    p_compile.add_argument("--blen", type=int, default=4)
    p_compile.add_argument("--btmm-lane-count", type=int, default=4)
    p_compile.add_argument("--btmm-hlen", type=int, default=16)
    p_compile.add_argument(
        "--stage-output",
        default=None,
        help="If given, append per-tile DMA HBM->VRAM[0..] for this output "
             "buffer so view_mem.py can compare against golden.",
    )
    p_compile.add_argument(
        "--dump-hlir",
        default=None,
        help="If given, write a human-readable HLIR dump to this path "
             "(after address allocation; final form fed into ISA emit).",
    )
    p_compile.set_defaults(func=_cmd_compile)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
