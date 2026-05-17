"""Run flash_attention_min through the new mid_ir pipeline end-to-end
and print the resulting HLIR.

Usage:
    nix develop --command bash -c '
        PYTHONPATH=compiler .venv/bin/python -m \\
            tilelang_tvm_compiler.scripts.run_flash_attention_midir
    '

Or with the .venv-tvm Python — but that one doesn't have tilelang
installed today (env-split issue documented elsewhere). Pick whichever
venv has tilelang + tvm both available.

Output:
  * <build_dir>/<kernel_name>.midir.txt    (mid_ir snapshot before lowering)
  * stdout:                                  formatted HLIR module
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Local helper: build a raw PrimFunc straight from the kernel source,
# bypassing the legacy compile_func wrapper (which would run the old
# frontend pipeline). We monkey-import the kernel module and grab its
# T.prim_func before make_*_min calls compile_func on it.

import tilelang.language as T

# Pull KIND constant + recreate the kernel literally so we get the raw
# PrimFunc (instead of the post-frontend lowered one make_flash_attention_min
# returns).
from tilelang_tvm_compiler.frontend.gemm_macros import KIND


def build_raw_flash_attention_min(*,
                                   rows: int = 64,
                                   hlen: int = 16,
                                   head_count: int = 4,
                                   num_kv_blocks: int = 2,
                                   num_q_blocks: int = 2):
    """Mirror of make_flash_attention_min in kernels/flash_attention_min.py
    but stops *before* compile_func — returns the raw tir.PrimFunc."""

    MLEN = 64
    if rows != MLEN:
        raise ValueError(f"rows must == MLEN={MLEN}, got {rows}")
    if MLEN % hlen != 0:
        raise ValueError(f"hlen must divide MLEN={MLEN}, got {hlen}")
    hardware_lane_count = MLEN // hlen
    if head_count % hardware_lane_count != 0:
        raise ValueError(
            f"head_count must be multiple of MLEN/hlen={hardware_lane_count}"
        )

    kv_seq = num_kv_blocks * rows
    q_seq = num_q_blocks * rows

    @T.prim_func
    def flash_attention_min(
        Q_hbm: T.Tensor((1, q_seq, head_count, hlen), "float16"),
        K_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        V_hbm: T.Tensor((1, kv_seq, head_count, hlen), "float16"),
        O_hbm: T.Tensor((1, q_seq, head_count, hlen), "float16"),
    ):
        with T.Kernel(num_q_blocks, head_count, threads=128) as (q_block, by):
            Q_sh = T.alloc_shared((rows, hlen), "float16")
            K_sh = T.alloc_shared((rows, hlen), "float16")
            V_sh = T.alloc_shared((rows, hlen), "float16")
            PV_loc = T.alloc_fragment((rows, hlen), "float16")
            O_loc = T.alloc_fragment((rows, hlen), "float16")
            S_loc = T.alloc_fragment((rows, MLEN), "float16")
            M_OLD = T.alloc_fragment((rows,), "float16")
            M_CURR = T.alloc_fragment((rows,), "float16")
            M_RES = T.alloc_fragment((rows,), "float16")
            L_OLD = T.alloc_fragment((rows,), "float16")
            L_NEW = T.alloc_fragment((rows,), "float16")
            P_SUM = T.alloc_fragment((rows,), "float16")
            SCALE = T.alloc_fragment((rows,), "float16")
            L_INV = T.alloc_fragment((rows,), "float16")
            M_INIT = T.alloc_fragment((rows,), "float16")
            L_INIT = T.alloc_fragment((rows,), "float16")

            T.copy(Q_hbm[0, q_block * rows, by, 0], Q_sh)

            for row in T.serial(rows):
                for col in T.Parallel(hlen):
                    O_loc[row, col] = T.float16(0)

            for row in T.serial(rows):
                M_OLD[row] = M_INIT[row]
                L_OLD[row] = L_INIT[row]

            for kv_block in T.unroll(num_kv_blocks):
                T.copy(K_hbm[0, kv_block * rows, by, 0], K_sh)
                T.copy(V_hbm[0, kv_block * rows, by, 0], V_sh)

                with T.attr(0, KIND, "btmm"):
                    T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)

                for row in T.serial(rows):
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = S_loc[row, col] * SCALE[row]
                    M_CURR[row] = M_OLD[row]

                T.reduce_max(S_loc, M_CURR, dim=1, clear=False)

                for row in T.serial(rows):
                    M_RES[row] = M_OLD[row] - M_CURR[row]
                    M_RES[row] = T.exp(M_RES[row])
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = S_loc[row, col] - M_CURR[row]
                    for col in T.Parallel(MLEN):
                        S_loc[row, col] = T.exp(S_loc[row, col])
                    P_SUM[row] = L_INIT[row]

                T.reduce_sum(S_loc, P_SUM, dim=1, clear=False)

                for row in T.serial(rows):
                    L_NEW[row] = L_OLD[row] * M_RES[row]
                    L_NEW[row] = L_NEW[row] + P_SUM[row]
                    for col in T.Parallel(hlen):
                        O_loc[row, col] = O_loc[row, col] * M_RES[row]
                    M_OLD[row] = M_CURR[row]
                    L_OLD[row] = L_NEW[row]

                T.gemm(S_loc, V_sh, PV_loc)

                for row in T.serial(rows):
                    for col in T.Parallel(hlen):
                        O_loc[row, col] = O_loc[row, col] + PV_loc[row, col]

            for row in T.serial(rows):
                L_INV[row] = 1.0 / L_NEW[row]
                for col in T.Parallel(hlen):
                    O_loc[row, col] = O_loc[row, col] * L_INV[row]

            T.copy(O_loc, O_hbm[0, q_block * rows, by, 0])

    return flash_attention_min


def main(argv) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=None,
                        help="Where to dump <name>.midir.txt (default: skip)")
    parser.add_argument("--num-q-blocks", type=int, default=2)
    parser.add_argument("--num-kv-blocks", type=int, default=2)
    parser.add_argument("--head-count", type=int, default=4)
    args = parser.parse_args(argv)

    raw = build_raw_flash_attention_min(
        num_q_blocks=args.num_q_blocks,
        num_kv_blocks=args.num_kv_blocks,
        head_count=args.head_count,
    )

    # Run the legacy stmt-prep (inline_let_stmts + lower_compound_fp_stores)
    # — these ARE pre-fold steps, not part of the new mid_ir pipeline.
    from tilelang_tvm_compiler.frontend.passes import inline_let_stmts
    from tilelang_tvm_compiler.frontend.passes import lower_compound_fp_stores
    raw = inline_let_stmts.run(raw)
    raw = lower_compound_fp_stores.run(raw)

    # Mid_ir pipeline.
    from tilelang_tvm_compiler.frontend.mid_ir.passes.infer_lane_axis import run as infer_run
    from tilelang_tvm_compiler.frontend.mid_ir.passes.fold import run as fold_run
    from tilelang_tvm_compiler.frontend.mid_ir.passes.mark import run as mark_run
    from tilelang_tvm_compiler.frontend.mid_ir.passes.split import run as split_run
    from tilelang_tvm_compiler.frontend.mid_ir.passes.distribute_cluster import run as dist_run
    from tilelang_tvm_compiler.frontend.mid_ir.passes.async_wrap import run as async_run
    from tilelang_tvm_compiler.frontend.mid_ir.passes.view import run as view_run
    from tilelang_tvm_compiler.frontend.mid_ir.passes.fuse import run as fuse_run
    from tilelang_tvm_compiler.frontend.mid_ir.passes.burn_view import run as burn_run
    from tilelang_tvm_compiler.frontend.mid_ir.passes.to_plena import run as to_plena_run

    raw = infer_run(raw)
    print(f"[infer_lane_axis] picked: "
          f"{raw.attrs['plena.lane_axis'] if raw.attrs and 'plena.lane_axis' in raw.attrs else None}",
          file=sys.stderr)

    midfn = fold_run(raw, name="flash_attention_min")
    midfn = mark_run(midfn)
    midfn = split_run(midfn)
    midfn = dist_run(midfn)
    midfn = async_run(midfn)
    midfn = view_run(midfn)
    midfn = fuse_run(midfn)
    midfn = burn_run(midfn)

    hlir = to_plena_run(midfn, build_dir=args.build_dir)

    from tilelang_tvm_compiler.hlir import format_hlir
    print(format_hlir(hlir))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
