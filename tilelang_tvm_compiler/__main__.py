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
import json
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
    print(f"[kernel] {mod_path} -> {mod.__file__}", file=sys.stderr)
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
    from tilelang_tvm_compiler.hlir import (
        hbm_strides_for_layout,
        make_tile_layout,
    )

    buf = compiled.hlir.get_buffer(out_buffer_name)
    if buf.scope != _scope.HBM:
        raise SystemExit(
            f"--stage-output buffer {out_buffer_name!r} must be in HBM, "
            f"got {buf.scope!r}"
        )
    rows, cols = _logical_2d(buf.shape, buf.layout)
    mlen = target.mlen
    if rows % mlen or cols % mlen:
        raise SystemExit(
            f"staging only supports mlen-aligned shapes for now, got "
            f"rows={rows} cols={cols} mlen={mlen}"
        )
    tile_elems = mlen * mlen
    full_tensor_size = rows * cols

    shim = make_shim(
        mlen=target.mlen,
        blen=target.blen,
        btmm_lane_count=target.btmm_lane_count,
        btmm_hlen=target.btmm_hlen,
        register_allocator=RegisterAllocator(),
    )
    emitter = ISAEmitter(shim)

    # Per-tile DMA stride between successive rows of one inner tile.
    # For BSHD this equals cols (legacy behaviour); for NCHW it is
    # the row-axis HBM stride (= W), NOT the cross-channel cols.
    # ``buf.hbm_stride`` was already set to the right value by
    # AddressAllocationPass (via _row_stride_for_layout).
    inner_tile_stride = buf.hbm_stride if buf.hbm_stride is not None else cols

    # ----- Multi-tile path: when the buffer's 4D logical shape needs
    # the 7D physical layout (e.g. NCHW with C_OUT > 1), iterate the
    # tile grid in canonical (D_TILES, S_TILES, H_GROUPS, B) order
    # and emit one DMA per inner tile. HBM offsets per tile come from
    # ``hbm_strides_for_layout`` so they correctly account for
    # NCHW's channel-major HBM layout.
    if len(buf.shape) == 4:
        layout = make_tile_layout(
            shape=tuple(int(x) for x in buf.shape), layout=buf.layout,
            mlen=mlen, hlen=target.btmm_hlen,
        )
    else:
        layout = None

    shim.compiler.generated_code = (
        "\n; ============================================================\n"
        f"; compare staging: {out_buffer_name} (HBM @ {buf.address}) -> VRAM[0..]\n"
    )

    if layout is not None:
        hbm_b, hbm_s, hbm_h, _hbm_d = hbm_strides_for_layout(
            buf.shape, buf.layout,
        )
        shim.compiler.generated_code += (
            f"; tile_layout: d_tiles={layout.d_tiles} s_tiles={layout.s_tiles} "
            f"h_groups={layout.h_groups} b={layout.logical_b}\n"
            f"; ({mlen}x{mlen} per inner tile, layout={buf.layout})\n"
            "; ============================================================\n"
        )
        # Iteration order matches the legacy col-major-block-major
        # ``for j in col_blocks: for i in row_blocks`` walk: outer is
        # the col-axis tile (d_tile, then h_grp), inner is the
        # row-axis tile (s_tile, then b). This keeps the per-tile
        # VRAM landing position byte-identical to what the
        # comparator's stride-mode reassembler assumes.
        vram_addr = 0
        for d_tile in range(layout.d_tiles):
            for h_grp in range(layout.h_groups):
                for s_tile in range(layout.s_tiles):
                    for b in range(layout.logical_b):
                        hbm_off = (
                            b * hbm_b
                            + s_tile * mlen * hbm_s
                            + h_grp * layout.lane_count * hbm_h
                            + d_tile * mlen
                        )
                        shim.compiler.generated_code += (
                            f"; stage tile (d={d_tile}, h={h_grp}, "
                            f"s={s_tile}, b={b}) hbm_off={hbm_off}  "
                            f"-> vram[{vram_addr}]\n"
                        )
                        emitter.emit_load_tile_from_hbm(
                            hbm_addr=buf.address,
                            vram_addr=vram_addr,
                            hbm_stride=inner_tile_stride,
                            hbm_scale_size=full_tensor_size,
                            hbm_start_offset=hbm_off,
                        )
                        vram_addr += tile_elems
        return shim.compiler.generated_code

    # ----- Single-tile fast path (non-4D, or 4D buffers that fit
    # exactly one MLEN×MLEN tile) — same iteration as before.
    row_blocks = rows // mlen
    col_blocks = cols // mlen
    shim.compiler.generated_code += (
        f"; layout: rows={rows} cols={cols} -> {row_blocks}x{col_blocks} tiles "
        f"({mlen}x{mlen} each), col-block-major\n"
        "; ============================================================\n"
    )
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
                hbm_stride=inner_tile_stride,
                hbm_scale_size=full_tensor_size,
                hbm_start_offset=hbm_offset_elems,
            )
            vram_addr += tile_elems

    return shim.compiler.generated_code


def _logical_2d(shape, layout: str = "BSHD") -> tuple[int, int]:
    """Layout-aware (rows, cols) projection. Delegates to the shared
    helper so __main__'s stage-output staging picks the same axes as
    address_alloc / isa_pass."""
    from tilelang_tvm_compiler.hlir import logical_2d_extents
    return logical_2d_extents(shape, layout)


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

    if args.dump_buffer_addrs:
        # Single source of truth for buffer addresses: dump the post
        # AddressAllocationPass HLIR addresses as JSON for testbenches /
        # external tooling to consume. Avoids the constants-dict-vs-actual
        # drift that bit us in flash_decode_min (the FPRAM SCALE/M_INIT/
        # L_INIT addresses the testbench used were a hand-rolled mirror
        # of `_slot_addresses`, off by 64 words from what TVM actually
        # allocated, leading to head-1/2 numerical drift).
        addr_table = {
            buf.name: {
                "scope": buf.scope,
                "address": buf.address,
                "shape": [int(s) for s in buf.shape],
                "dtype": str(buf.dtype),
            }
            for buf in compiled.hlir.buffers.values()
        }
        Path(args.dump_buffer_addrs).write_text(
            json.dumps(addr_table, indent=2)
        )

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
    p_compile.add_argument(
        "--dump-buffer-addrs",
        default=None,
        help="If given, write a JSON dict {buffer_name: {scope, address, "
             "shape, dtype}} so testbenches can read the *actual* allocated "
             "addresses instead of mirroring them by hand. Single source of "
             "truth for FPRAM / VRAM / MRAM / HBM offsets.",
    )
    p_compile.set_defaults(func=_cmd_compile)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
