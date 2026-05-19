"""TVM testbench harness — single entry point for tvm_*_test.py drivers.

Each ``transactional_emulator/testbench/tvm_<kernel>_test.py`` had grown
into 200-370 lines of mostly-identical boilerplate (subprocess into the
TVM venv, write .pt files, call create_sim_env / create_mem_for_sim,
write comparison_params.json). This helper collapses the shared flow
into a single ``run(spec)`` call. Per-kernel code shrinks to:

    * a few constants (MLEN, HLEN, ...)
    * a ``build_inputs_and_golden(seed)`` function
    * an optional ``build_fp_preload`` and/or ``build_pre_kernel_stub``
    * a ``build_comparison_params`` function (or static dict)
    * a single ``TvmTestbenchSpec(...)`` and ``run(spec)`` call

The helper itself does not import torch eagerly because this file lives
in the compiler tree, which is loaded under multiple Python venvs.
Torch is imported inside ``run()`` where the testbench's own venv has
already set ``sys.path`` for it.

Pipeline (in order):

    1. Subprocess into the TVM venv to compile TIR -> PLENA ISA text,
       optionally dumping HLIR and the buffer-address JSON.
    2. If ``parse_buffer_addrs`` was given, parse the JSON into the
       address dict the per-kernel hooks expect.
    3. If ``build_pre_kernel_stub`` was given, prepend its output to the
       kernel ISA. Used by conv2d_min / flash_decode_min for the FPRAM
       staging -> VRAM cache copy that has to happen before the kernel
       proper.
    4. ``build_inputs_and_golden(seed)`` produces:
         - ``hbm_inputs``: dict[name -> torch.Tensor] for HBM staging
         - ``golden_flat``: 2D flat golden the comparator diffs against
         - any extras (e.g. ``q_token`` for flash_decode) consumed by
           ``build_fp_preload``
    5. If ``build_fp_preload`` was given, it returns the FP-preload
       tensor positioned at the right FPRAM addresses.
    6. ``create_sim_env`` writes .pt / .asm / fp_sram.bin / int_sram.bin
       and the golden file. ``create_mem_for_sim`` assembles + packs HBM.
    7. Write ``comparison_params.json`` so view_mem.py knows where to
       read the staged output.

The helper does not call cargo / view_mem itself — those are still
driven by `just build-emulator-debug <name>`. This file only produces
the artefact set in ``build/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


# ---------------------------------------------------------------------------
# Repo discovery. The helper lives at:
#   <repo>/compiler/tilelang_tvm_compiler/test_helper.py
# so the repo root is two ``parent`` hops up.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
REPO_ROOT = _THIS_FILE.parent.parent.parent
TESTBENCH_DIR = REPO_ROOT / "transactional_emulator" / "testbench"

# Default LD_LIBRARY_PATH for the ``.venv`` (Python 3.12) flow — torch
# loaded from that venv requires this Nix-provided libstdc++. The older
# ``.venv-tvm`` (Python 3.11, TVM-only) flow needs LD_LIBRARY_PATH="".
DEFAULT_LD_LIBRARY_PATH = "/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib"


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

# Input-build hook contract:
#   def build_inputs_and_golden(seed: int) -> dict
# Required keys in the returned dict:
#   - "hbm_inputs":  dict[str, torch.Tensor]   # buffers to stage into HBM
#   - "golden_flat": torch.Tensor              # 2D flat golden (rows, MLEN-aligned cols)
# Any other keys are kernel-specific extras (e.g. ``q_token`` for
# flash_decode_min) and are forwarded to ``build_fp_preload`` unchanged.
InputsBuilder = Callable[[int], dict]

# Buffer-addresses parser:
#   def parse_buffer_addrs(raw_json_dict: dict) -> dict
# Receives the raw output of ``--dump-buffer-addrs`` (each entry has
# ``scope``, ``address``, ``shape``, ``dtype``). Returns whatever shape
# the per-kernel hooks find convenient.
BufferAddrsParser = Callable[[dict], dict]

# Pre-kernel ASM stub (concatenated BEFORE the compiled kernel ISA):
#   def build_pre_kernel_stub(addrs: dict) -> str
PreKernelStubBuilder = Callable[[dict], str]

# FP preload builder:
#   def build_fp_preload(io: dict, addrs: dict) -> torch.Tensor
# ``io`` is the dict returned by ``build_inputs_and_golden``.
FpPreloadBuilder = Callable[[dict, dict], Any]   # Any to avoid eager torch import

# Comparison-params builder:
#   def build_comparison_params(io: dict, addrs: dict) -> dict
# Receives the same ``io`` (so it can read shapes off ``hbm_inputs`` /
# ``golden_flat``) and the parsed addrs (some kernels need an O_CACHE
# address for ``start_row_idx``).
ComparisonParamsBuilder = Callable[[dict, dict], dict]


@dataclass
class TvmTestbenchSpec:
    """Everything one ``tvm_<kernel>_test.py`` needs to declare."""

    # ---- identity ----
    asm_name: str
    """Used for the .asm filename, log messages, and (after the helper
    runs) ``{asm_name}_generated_asm_code.asm`` in build/."""

    kernel: str
    """Kernel spec passed to ``tilelang_tvm_compiler compile --kernel``,
    e.g. ``"tilelang_tvm_compiler.kernels.conv2d_min:make_conv2d_min"``."""

    build_inputs_and_golden: InputsBuilder
    """See ``InputsBuilder`` above."""

    build_comparison_params: ComparisonParamsBuilder
    """See ``ComparisonParamsBuilder`` above."""

    # ---- compile-time tuneables ----
    kernel_kwargs: Mapping[str, Any] = field(default_factory=dict)
    """k=v pairs forwarded as ``--kernel-kwargs k1=v1,k2=v2,...``."""

    mlen: int = 64
    btmm_hlen: Optional[int] = None
    btmm_lane_count: Optional[int] = None

    stage_output: Optional[str] = None
    """Buffer name to re-stage from HBM -> VRAM at the end of the
    kernel for view_mem comparison (passed via ``--stage-output``)."""

    artifact_prefix: Optional[str] = None
    """Prefix for ancillary build artefacts. Defaults to ``asm_name``."""

    # ---- venv / subprocess env ----
    venv_name: str = ".venv"
    """Subdir of the repo root containing the Python venv used to invoke
    the compiler. ``.venv`` (Python 3.12, the new default) or the legacy
    ``.venv-tvm`` (Python 3.11, TVM-wheel-only)."""

    ld_library_path: Optional[str] = DEFAULT_LD_LIBRARY_PATH
    """Forwarded as the subprocess's ``LD_LIBRARY_PATH``. Pass ``""`` to
    explicitly clear it (the ``.venv-tvm`` convention) or ``None`` to
    inherit from the parent process unchanged."""

    # ---- buffer-addrs JSON ----
    parse_buffer_addrs: Optional[BufferAddrsParser] = None
    """If given, the helper passes ``--dump-buffer-addrs`` to the
    compiler, then calls this function with the parsed JSON. The result
    is forwarded to ``build_pre_kernel_stub`` /
    ``build_fp_preload`` / ``build_comparison_params``. If omitted, the
    helper still passes an empty dict to the hooks, so kernels that
    don't need address introspection don't pay for it."""

    # ---- optional kernel hooks ----
    build_pre_kernel_stub: Optional[PreKernelStubBuilder] = None
    build_fp_preload: Optional[FpPreloadBuilder] = None
    patch_isa: Optional[Any] = None
    """Optional last-mile rewrite hook over the assembled ASM text.

    Signature: ``(isa_text: str) -> str``. Called after the compile
    subprocess emits the kernel ASM and (if any) the pre-kernel stub
    is prepended, but before ``create_sim_env`` writes it to disk.
    Used by debug step kernels to patch a single instruction's
    operand (e.g. flip ``V_RED_SUM ..., 1`` to ``..., 0`` to see
    what the unmasked path produces) without touching the compiler.
    """
    int_preload: Any = None
    """Static int-preload tensor (sim_env_utils takes it through). None
    for everything we have today."""

    # ---- misc ----
    seed: int = 0
    """Forwarded to ``build_inputs_and_golden``."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_kwargs(kwargs: Mapping[str, Any]) -> str:
    return ",".join(f"{k}={v}" for k, v in kwargs.items())


def _compile_via_subprocess(
    spec: TvmTestbenchSpec,
    *,
    hlir_path: Path,
    addrs_path: Optional[Path],
) -> str:
    """Subprocess into the TVM venv to compile the kernel.

    Returns the kernel's ISA text (stdout). Raises ``RuntimeError``
    with the captured stderr on failure.
    """
    venv_python = REPO_ROOT / spec.venv_name / "bin" / "python"
    if not venv_python.exists():
        raise RuntimeError(
            f"venv python not found: {venv_python}. Set "
            f"TvmTestbenchSpec.venv_name to a venv that exists."
        )
    cmd = [
        str(venv_python), "-m", "tilelang_tvm_compiler", "compile",
        "--kernel", spec.kernel,
        "--asm-name", spec.asm_name,
        "--mlen", str(spec.mlen),
    ]
    if spec.kernel_kwargs:
        cmd += ["--kernel-kwargs", _format_kwargs(spec.kernel_kwargs)]
    if spec.btmm_lane_count is not None:
        cmd += ["--btmm-lane-count", str(spec.btmm_lane_count)]
    if spec.btmm_hlen is not None:
        cmd += ["--btmm-hlen", str(spec.btmm_hlen)]
    if spec.stage_output is not None:
        cmd += ["--stage-output", spec.stage_output]
    cmd += ["--dump-hlir", str(hlir_path)]
    if addrs_path is not None:
        cmd += ["--dump-buffer-addrs", str(addrs_path)]

    env = os.environ.copy()
    if spec.ld_library_path is not None:
        env["LD_LIBRARY_PATH"] = spec.ld_library_path
    env["PYTHONPATH"] = str(REPO_ROOT / "compiler")

    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"TVM compile subprocess failed (returncode={res.returncode}). "
            f"See stderr above. Command: {' '.join(cmd)}"
        )
    return res.stdout


def _validate_io(io: dict) -> None:
    if not isinstance(io, dict):
        raise TypeError(
            f"build_inputs_and_golden must return a dict; got "
            f"{type(io).__name__}"
        )
    missing = {"hbm_inputs", "golden_flat"} - set(io)
    if missing:
        raise KeyError(
            f"build_inputs_and_golden return dict is missing required keys: "
            f"{sorted(missing)} (must include 'hbm_inputs' and 'golden_flat')"
        )
    if not isinstance(io["hbm_inputs"], dict):
        raise TypeError(
            f"build_inputs_and_golden['hbm_inputs'] must be a dict, got "
            f"{type(io['hbm_inputs']).__name__}"
        )


# ---------------------------------------------------------------------------
# Canonical output layout — the SINGLE source of truth for how a kernel's
# logical (B, S, H, D) output maps to its physical VRAM staging layout,
# and therefore how golden must be flattened and how the comparator must
# reorder the staged VRAM dump before diffing.
#
# Why this exists: golden-flatten order, --stage-output VRAM layout, and
# check_mem's reorder were each hand-coded per kernel, each carrying its
# own (often wrong) layout assumption. Every disagreement showed up as a
# "compare fails" mystery. Route ALL THREE through this one function so
# they cannot drift.
#
# The layout facts (must match the compiler's TileLayout / --stage-output
# and check_mem.reorder_stride_mode):
#
#   * One mlen-wide VRAM tile packs LANE_COUNT = mlen // hlen heads
#     side by side (each head occupies hlen columns).
#   * A logical output with H heads therefore spans
#     H_GROUPS = ceil(H / LANE_COUNT) head-groups.
#   * --stage-output loads O_hbm back into VRAM head-group-major:
#     head-group 0 for ALL S rows first, then head-group 1, ...
#     i.e. physical order  [h_group][s][lane][d].
#   * golden_flat is batch-major  [s][h][d]  (one row's H heads
#     contiguous). These two are the same data in different
#     permutations.
#   * check_mem.reorder_stride_mode converts head-group-major chunks
#     back to batch-major exactly when chunks_per_batch > 1 (i.e. when
#     H_GROUPS > 1). For H_GROUPS == 1 there is no interleaving and the
#     reorder must stay off.
# ---------------------------------------------------------------------------

# Every kernel output, whatever its logical rank, ultimately stages
# into VRAM as a 2D grid:  num_batches logical rows, each
# elements_per_batch wide. A row wider than MLEN occupies
# CHUNKS_PER_BATCH = ceil(elements_per_batch / mlen) mlen-wide physical
# chunks, and --stage-output writes those chunk-group-major (chunk 0 of
# every row first, then chunk 1 of every row, ...). golden_flat is
# batch-major (each row's full width contiguous). When CHUNKS_PER_BATCH
# > 1 the two orders differ and the comparator must reorder; that is
# precisely what check_mem.reorder_stride_mode does, so use_stride_mode
# is just CHUNKS_PER_BATCH > 1.
#
# BSHD (B,S,H,D) outputs are the common special case: rows = B*S,
# cols = H*D, and chunk groups coincide with head-groups
# (LANE_COUNT = mlen//hlen heads per mlen tile). conv NCHW outputs and
# plain (M,N) matmul outputs are the same 2D grid with a different
# logical-shape story — :func:`resolve_output_layout` accepts any of
# them and they all funnel through the identical 2D math here.

@dataclass(frozen=True)
class OutputLayout:
    """Resolved 2D staging layout for one kernel output.

    ``num_batches`` logical rows, ``elements_per_batch`` values each,
    physically chunked into ``mlen``-wide pieces."""
    num_batches: int
    elements_per_batch: int
    mlen: int

    @property
    def chunks_per_batch(self) -> int:
        """mlen-wide physical chunks one logical row spans."""
        return (self.elements_per_batch + self.mlen - 1) // self.mlen

    @property
    def use_stride_mode(self) -> bool:
        """True iff the staged VRAM is chunk-group-major and must be
        reordered back to batch-major before diffing against golden."""
        return self.chunks_per_batch > 1

    def comparison_params(self) -> dict:
        """The geometry block check_mem / view_mem need. Merge this into
        whatever kernel-specific keys (check_hbm, start_row_idx, ...)
        the SPEC still wants to set."""
        return {
            "num_rows": self.num_batches * self.chunks_per_batch,
            "num_batches": self.num_batches,
            "elements_per_batch": self.elements_per_batch,
            "row_dim": self.mlen,
            "use_stride_mode": self.use_stride_mode,
        }

    def flatten_golden(self, out):
        """Flatten a golden tensor of any rank into the canonical 2D
        ``golden_flat``: ``(num_batches, elements_per_batch)``,
        batch-major. The comparator reorders the VRAM side to match
        this — never the reverse. Raises if the tensor's element count
        doesn't match ``num_batches * elements_per_batch``."""
        want = self.num_batches * self.elements_per_batch
        got = 1
        for dim in out.shape:
            got *= int(dim)
        if got != want:
            raise ValueError(
                f"flatten_golden: golden has {got} elements but layout "
                f"expects num_batches*elements_per_batch = "
                f"{self.num_batches}*{self.elements_per_batch} = {want} "
                f"(golden shape {tuple(out.shape)})"
            )
        return out.reshape(self.num_batches, self.elements_per_batch)


def resolve_output_layout(
    *,
    mlen: int,
    # --- form 1: BSHD output ---
    b: Optional[int] = None,
    s: Optional[int] = None,
    h: Optional[int] = None,
    d: Optional[int] = None,
    hlen: Optional[int] = None,
    # --- form 2: explicit 2D output ---
    num_batches: Optional[int] = None,
    elements_per_batch: Optional[int] = None,
) -> OutputLayout:
    """Build the canonical :class:`OutputLayout`.

    Two calling forms — pick the one matching the kernel's output:

      * BSHD:  ``resolve_output_layout(b=, s=, h=, d=, mlen=, hlen=)``
        rows = b*s, cols = h*d. ``hlen`` is checked against ``mlen``
        for the lane-packing invariant (``h`` must be a multiple of
        ``mlen//hlen``) so a mis-shaped head count fails loudly here.

      * explicit 2D: ``resolve_output_layout(num_batches=,
        elements_per_batch=, mlen=)`` — for matmul ``(M,N)``, conv
        ``(C_OUT*H, W)`` and anything else that isn't naturally BSHD.

    Whichever form, the result drives BOTH ``flatten_golden`` and
    ``comparison_params`` so the golden order and the comparator's
    reorder agree by construction.
    """
    if mlen <= 0:
        raise ValueError(f"resolve_output_layout: mlen must be > 0; got {mlen}")

    bshd_given = any(v is not None for v in (b, s, h, d, hlen))
    twod_given = any(v is not None for v in (num_batches, elements_per_batch))
    if bshd_given and twod_given:
        raise ValueError(
            "resolve_output_layout: pass EITHER the BSHD form "
            "(b/s/h/d/hlen) OR the explicit 2D form "
            "(num_batches/elements_per_batch), not both"
        )
    if twod_given:
        if num_batches is None or elements_per_batch is None:
            raise ValueError(
                "resolve_output_layout 2D form needs both num_batches "
                "and elements_per_batch"
            )
        return OutputLayout(
            num_batches=int(num_batches),
            elements_per_batch=int(elements_per_batch),
            mlen=int(mlen),
        )

    # BSHD form.
    if any(v is None for v in (b, s, h, d, hlen)):
        raise ValueError(
            "resolve_output_layout BSHD form needs all of b/s/h/d/hlen"
        )
    if hlen <= 0 or mlen % hlen != 0:
        raise ValueError(
            f"resolve_output_layout: need mlen % hlen == 0; "
            f"got mlen={mlen}, hlen={hlen}"
        )
    lane_count = mlen // hlen
    if h % lane_count != 0:
        raise ValueError(
            f"resolve_output_layout: H ({h}) must be a multiple of "
            f"LANE_COUNT ({lane_count} = mlen//hlen)"
        )
    return OutputLayout(
        num_batches=int(b) * int(s),
        elements_per_batch=int(h) * int(d),
        mlen=int(mlen),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(spec: TvmTestbenchSpec) -> int:
    """Drive the full TVM testbench pipeline for ``spec``.

    Writes everything ``just build-emulator-debug`` expects under
    ``transactional_emulator/testbench/build/``. Returns 0 on success.
    """
    # Lazy imports — these need the testbench's venv site-packages to be
    # on sys.path, which the testbench itself sets up before importing us.
    from compiler.sim_env_utils import create_mem_for_sim
    from transactional_emulator.tools.create_sim_env import create_sim_env

    artifact_prefix = spec.artifact_prefix or spec.asm_name

    build_dir = TESTBENCH_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. compile ----------
    print(f"[1/4] Compiling TVM {spec.asm_name} kernel ...")
    hlir_path = build_dir / f"{spec.asm_name}.hlir.txt"
    addrs_path: Optional[Path] = (
        build_dir / f"{spec.asm_name}.buffer_addrs.json"
        if spec.parse_buffer_addrs is not None else None
    )
    kernel_isa = _compile_via_subprocess(
        spec, hlir_path=hlir_path, addrs_path=addrs_path,
    )

    addrs: dict = {}
    if addrs_path is not None:
        raw_addrs = json.loads(addrs_path.read_text())
        addrs = spec.parse_buffer_addrs(raw_addrs)  # type: ignore[misc]

    # Optional pre-kernel stub (FPRAM staging -> VRAM cache, etc.).
    stub_isa = ""
    if spec.build_pre_kernel_stub is not None:
        stub_isa = spec.build_pre_kernel_stub(addrs)
    isa_text = stub_isa + kernel_isa
    if spec.patch_isa is not None:
        patched = spec.patch_isa(isa_text)
        if patched != isa_text:
            print(
                f"      NB  patch_isa hook rewrote ASM "
                f"({isa_text.count(chr(10))} -> "
                f"{patched.count(chr(10))} lines)"
            )
        isa_text = patched

    # Large-immediate normalisation, on the FULL assembled text.
    # ``isa_pass`` runs this on the kernel body, but the pre-kernel
    # stub (``build_pre_kernel_stub``) and any ``patch_isa`` hook are
    # assembled outside that path — their ``S_ADDI_INT`` immediates
    # never got normalised. With MLEN-512 addresses those overflow the
    # 18-bit immediate slot. Re-run the pass over the concatenation so
    # every section is covered.
    from .isa_pass import _normalize_large_addi_immediates
    _before = isa_text.count(chr(10))
    isa_text = _normalize_large_addi_immediates(isa_text)
    _after = isa_text.count(chr(10))
    if _after != _before:
        print(
            f"      NB  large-imm normalise expanded ASM "
            f"({_before} -> {_after} lines)"
        )
    print(
        f"      OK  ({kernel_isa.count(chr(10))} kernel lines"
        + (f" + {stub_isa.count(chr(10))} stub lines" if stub_isa else "")
        + f", HLIR -> {hlir_path.name})"
    )

    # ---------- 2. inputs + golden + (optional) FP preload ----------
    print(f"[2/4] Generating inputs + golden{' + FP preload' if spec.build_fp_preload else ''} ...")
    io = spec.build_inputs_and_golden(spec.seed)
    _validate_io(io)
    hbm_inputs: dict = io["hbm_inputs"]
    golden_flat = io["golden_flat"]

    fp_preload = None
    if spec.build_fp_preload is not None:
        fp_preload = spec.build_fp_preload(io, addrs)

    # Auto-preload hoisted FP constants. The compiler's
    # ``hoist_float_constants`` pre-pass turns every ``T.float16(c)``
    # use into a 1-slot ``global.fpram`` buffer; the buffer-addrs dump
    # carries each one's slot address and value. Write those slots
    # here so per-kernel testbenches don't have to enumerate them.
    if addrs_path is not None and raw_addrs:
        import torch  # local — testbench venv has torch on sys.path here
        const_entries = [
            (int(entry["address"]), float(entry["value"]))
            for entry in raw_addrs.values()
            if isinstance(entry, dict) and "value" in entry
        ]
        if const_entries:
            max_const_addr = max(addr for addr, _ in const_entries)
            needed = max_const_addr + 1
            if fp_preload is None:
                fp_preload = torch.zeros(needed, dtype=torch.float16)
            elif fp_preload.numel() < needed:
                grown = torch.zeros(needed, dtype=fp_preload.dtype)
                grown[: fp_preload.numel()] = fp_preload
                fp_preload = grown
            for addr, value in const_entries:
                fp_preload[addr] = value
            print(f"           auto-preloaded {len(const_entries)} FP constant(s)")

    input_feed = {
        name: t.contiguous().reshape(1, -1) for name, t in hbm_inputs.items()
    }
    input_order = list(input_feed)
    summary = ", ".join(
        f"{n}={tuple(t.shape)}" for n, t in hbm_inputs.items()
    )
    print(f"      OK  hbm_inputs: {summary}")
    print(f"           golden flat: {tuple(golden_flat.shape)}")
    if fp_preload is not None:
        print(f"           fp_preload: {tuple(fp_preload.shape)}")

    # ---------- 3. create_sim_env (.pt + .asm + fp/int sram bins) ----------
    print(f"[3/4] create_sim_env -> .pt + .asm + fp/int sram bins ...")
    create_sim_env(
        input_tensor=input_feed,
        generated_code=isa_text,
        golden_result={"original_output": golden_flat},
        fp_preload=fp_preload,
        int_preload=spec.int_preload,
        build_dir=str(build_dir),
    )
    print(f"      OK  -> {build_dir}")

    # ---------- 4. create_mem_for_sim (assemble + pack HBM) ----------
    print(f"[4/4] create_mem_for_sim -> assemble .asm + pack HBM bin ...")
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=spec.asm_name,
        data=None,
        specified_data_order=input_order,
        build_path=build_dir,
    )
    print(f"      OK  -> generated_machine_code.mem + hbm_for_behave_sim.bin")

    # ---------- comparison_params + asm snapshot ----------
    comparison_params = spec.build_comparison_params(io, addrs)
    cmp_path = build_dir / "comparison_params.json"
    cmp_path.write_text(json.dumps(comparison_params, indent=2))
    print(f"      wrote comparison_params.json -> {cmp_path}")

    (build_dir / f"{artifact_prefix}_generated_asm_code.asm").write_text(isa_text)

    print()
    print("=" * 60)
    print(f"build/ ready for: just build-emulator-debug {artifact_prefix}")
    print("=" * 60)
    return 0


__all__ = [
    "TvmTestbenchSpec",
    "run",
    "OutputLayout",
    "resolve_output_layout",
    "REPO_ROOT",
    "TESTBENCH_DIR",
    "DEFAULT_LD_LIBRARY_PATH",
]
