# tilelang_tvm_compiler — AI Agent Knowledge Base

This file is intentionally written for AI agents (and humans) starting a new
session. It captures the non-obvious lessons accumulated while building the
TVM-frontend path from TIR → HLIR → PLENA ISA. **Read this before editing
kernels under `kernels/`, intrinsics, the lowering pass, or testbenches under
`transactional_emulator/testbench/`.**

If you discover something the next agent will trip on, append it here.

---

## 1. Pipeline at a glance

> **Architecture note.** The old graph-IR frontend
> (`graph_ir.Graph`, `lift_from_raw_primfunc`, `split_lane_groups`,
> `materialize_to_primfunc`, `PlenaCodegen`) **has been deleted**.
> `frontend/pipeline.py` is now a stub that raises if called. The only
> active lowering chain is the **mid_ir pipeline**. Don't trust any
> doc/comment that still mentions `graph_ir`.

The real driver is **`tilelang_tvm_compiler/pipeline.py : compile_kernel`**:

```
  @T.prim_func (tilelang DSL)
     │  0. stmt prep
     │     inline_let_stmts → lower_compound_fp_stores → hoist_float_constants
     ▼
  raw tir.PrimFunc (FP literals hoisted to global.fpram 1-slot buffers)
     │  1. mid_ir pipeline  (frontend/mid_ir/passes/, 9 passes)
     │     infer_lane_axis  — pick the lane axis (see §11)
     │     fold             — raw TIR → mid_ir.MidFunc dataclass tree
     │     mark             — tag each op with its lane-fusion role
     │     split            — lane blockIdx → (number, phase); grow
     │                        non-global buffers by a cluster outer dim
     │     distribute_cluster — push CLUSTER axes inside unroll/pipeline loops
     │     async_wrap       — wrap can-async ops in Async regions
     │     view             — assign view_perm to every BufferRef;
     │                        substitute lane var on HBM refs
     │     fuse             — collapse each Async region → one MultiLaneOp
     │     burn_view        — bake view_perm into physical shape + indices
     │     to_plena         — MidFunc → HLIRModule  (exits mid_ir domain)
     ▼
  HLIRModule                 ← buffers + Op stream, no addresses
     │  1.5  dead_buffer_elim — drop buffers no HLIR op references
     │  2.   AddressAllocationPass (address_alloc.py)
     ▼
  HLIRModule + addresses     ← per-buffer base address resolved
     │  3.   IsaEmitterPass.run (isa_pass.py)  — RegisterAllocator + shim
     ▼
  ISA text  (`*_generated_asm_code.asm`)
```

- **The mid_ir is a typed dataclass tree** (`mid_ir/ir.py : MidFunc`),
  not TIR. Passes 1–8 are `MidFunc → MidFunc`; `to_plena` (pass 9) is the
  single bridge out to `HLIRModule`. The only TIR-level walkers left are
  the pass-0 stmt-prep steps and `infer_lane_axis` (which still inspects
  raw TIR — see §11).
- `compile_kernel(prim_func, *, target, name, midir_dump_dir=...,
  addr_config_override=...)` is the entry point. `midir_dump_dir` makes
  `to_plena` write `<name>.midir.txt` **and** `compile_kernel` write
  `post_to_plena.hlir.txt` right after `to_plena` — both survive a later
  pass failure, so they are the go-to debugging artefacts.
- `addr_config_override` lets a multi-kernel driver pin FPRAM/HBM bases
  per kernel — used by `tvm_single_stream_block_test` to stitch the
  MMDiT chain into one continuous ASM run.

---

## 2. Hardware mental model (essential)

### Memories

| Storage | Layout | dtype (per `plena_settings.toml` ANALYTIC mode) |
|---|---|---|
| HBM    | DRAM, large | **MX-FP8** (block=8, elem e4m3, scale e8m0) |
| MRAM   | matrix-side SRAM | **bf16** (Plain Fp, exp=8, mantissa=7) |
| VRAM   | vector-side SRAM | **bf16** |
| FPRAM  | scalar FP scratch | bf16 |

The MX-FP8 quantization on HBM is real and intentional. Outputs from a
fp32 reference will have ~5–15% relative error vs. simulator output once
they've round-tripped through HBM at magnitudes like 5–17. **This is not
a kernel bug.** If you suspect quantization noise, switch the
`plena_settings.toml` HBM types from `format = "Mx"` to `format = "Plain"`
bf16 and rerun — error should drop to ~0.

### Tile sizes

- `MLEN = 64` — full vector tile width. A "VRAM row" is MLEN elements wide.
- `HLEN = 16` (typical) — narrow head dim. Used for BTMM and per-head MM.
- `BLEN = 4` — systolic-array tile size (BLEN × BLEN per `M_MM_WO`).
- `LANE_COUNT = MLEN / HLEN = 4` — number of hardware lanes packed into
  one VRAM row.

### Matrix engines

| Op | Hardware | Reduction dim | Output shape per head | Use |
|---|---|---|---|---|
| `plena.btmm` | `M_BTMM`/`M_BMM_WO` | **HLEN** | (MLEN, MLEN) | Q @ K^T (lhs (B,S,H,D), rhs in MRAM) |
| `plena.mm` | `M_MM`/`M_MM_WO` | **MLEN** | (MLEN, MLEN) | regular MM, single head |
| `plena.mm_slot` | `M_MM`/`M_MM_WO` (column-slot loop) | **MLEN** | (MLEN, hlen) | per-head narrow MM (P @ V, etc.) |

**Crucial rule**: `M_MM 0, rs1, rs2` reads vector tile from VRAM (rs2) and
matrix tile from MRAM (rs1). So **A @ B → A in VRAM, B in MRAM** regardless
of "narrow vs wide". The runtime compiler enforces this in
`_compute_manager.py:54-55` (`ensure_value_tile_in_place(rhs, "mram")`).

### Buffer layouts you will see

| Layout name | Shape pattern | Where |
|---|---|---|
| BSHD | `(B, S, H, D)`  | HBM tensors (canonical), DMA preserves it |
| BHSD | `(B, H, S, D)`  | BTMM #1 output (head-major: head h's tile starts at `h * S * D`) |

`BHSD` and `BSHD` differ in **whether heads are interleaved within a row
(BSHD packed-narrow)** or **stacked as separate row groups (BHSD)**. The
`_logical_2d` helper flattens shapes to (rows, cols) — for BHSD `(1, 4,
64, 64)` you get `(4*64, 64*64/64) = (256, 64)`. Picking head h's tile
out of a BHSD VRAM buffer means addressing at `base + h * MLEN * MLEN`.

### Where quantization actually happens (verified in the sim)

Common misconception: that the sim quantizes everywhere. It does not.

- **`QuantTensor::quantize` is a TODO no-op** — it does not quantize.
  Any VRAM tile written via `QuantTensor::quantize(t, ty)` keeps its
  fp32 values; the `ty` is just a label.
- The real MX-E4M3 quantization happens in **`into_bytes`** (the
  VRAM→HBM serialise path, `H_STORE_V`). HBM stores MX-E4M3.
- The HBM→VRAM path (`H_PREFETCH_V` / `transfer_mx_from_hbm`) *decodes*
  MX back to fp32; it does not add a second rounding.
- Net effect: an intermediate tensor that a kernel writes to an HBM
  scratch and the next kernel reads back is **MX-quantized exactly once**
  (at the store).
- On-chip matmul / vector ops run in **fp32** in the sim (`to_kind(Float)`).
- The Rust `into_bytes` MX quantizer is line-for-line equivalent to the
  Python `_mx_fp_quantize_hardware` (same `floor(log2(max)+1e-9)`, same
  bias, same block grouping over the last dim). So a golden modelled with
  `_mx_fp_quantize_hardware` matches the sim's HBM bytes — *provided* the
  golden quantizes the tensor in its real staged 4D shape (block grouping
  is over the last dim; quantizing a reshaped 2D view changes the block
  boundaries).

---

## 3. Intrinsics convention

Every `plena.*` intrinsic has a fixed `operand_scopes` tuple in
`intrinsics.py`. The trailing `None` slots are *scalar* slots (length must
match exactly at codegen time, see `_verify_scopes` in `codegen.py:474`).

### The `_at` family — per-row scalar tasks

The "row scalar" ops come in two physical shapes:

**FP-touching, signature `(vram_row, lane, mask)`** (3 scalars):
- `plena.row_reduce_max_at` / `plena.row_reduce_sum_at`
- `plena.row_sub_fp_at` / `plena.row_add_fp_at` / `plena.row_mul_fp_at`

**VRAM-only, signature `(vram_row, mask)`** (2 scalars):
- `plena.row_exp_at`

**Mask semantics (KEY):**
- `mask = 0`  → unmasked path: lowering emits `<opcode> ..., 0` (no V_MASK
  setup, no tail clear). VRAM addressing uses `vram_row`. FP addressing
  uses `lane * fp_buf.shape[-1] + vram_row`.
- `mask != 0` (literal or PrimExpr) → masked: emits `C_SET_V_MASK_REG mask`,
  uses `..., 1` flag, **always emits a tail clear `C_SET_V_MASK_REG 0`**
  so subsequent ops see V_MASK=0.

The unification is implemented by `mask_static_zero` in
`isa_pass.py:_emit_row_scalar_op_at`. When mask is the literal 0, V_MASK
emission is skipped — but it MUST be a literal int / IntImm. A PrimExpr
that happens to evaluate to zero will not trigger the optimisation.

**When to use mask=0 vs mask=1<<lane:**
- BHSD layout where each VRAM row is one head's full mlen-wide data:
  `mask=0`, scalar = `lane * rows + row` (used for both VRAM and FP).
- BSHD packed-narrow where heads occupy column slots within a row:
  `mask = 1 << lane`, scalars = `(row, lane, 1<<lane)`.

### `plena.fp_*_at` — FP scalar ops at offset

Signature: `(buffers..., row)`. The single scalar is the FP element offset
within the FPRAM buffer. For `_emit_fp_scalar_op_at` in `isa_pass.py:197`,
the row expression is materialised **once** and reused via `S_ADDI_INT
gp_dst, gp_row, buf.address` per buffer (a CSE optimisation that saves
many redundant SLLI/ADDs and is what keeps the row-loop body under the
emulator's 10 000-instr cap).

Supported `kernel_op` values: `copy, add, sub, mul, max, exp, reci, sqrt`.
Unary ones (copy/exp/reci/sqrt) take 2 buffers; binary (add/sub/mul/max)
take 3. The `_emit_fp_scalar_op_at` dispatches on `len(op.buffer_args)`.

### `plena.mm_slot` — narrow output MM

Signature: `(A_v, B_m, C_v, lhs_row_offset, rhs_col_offset, dst_col_offset, col_count)`.

- `lhs_row_offset` (in elements): pick a head's mlen×mlen tile out of a
  multi-head VRAM LHS buffer (e.g. BHSD-laid-out S_v from BTMM). Static int
  → folded into `lhs_vram_addr` literal. **Dynamic PrimExpr** → materialised
  to a register, passed as `lhs_vram_addr_reg` to `emit_slot_matmul`.
- `rhs_col_offset` / `dst_col_offset`: column slot within the mlen-wide
  RHS/DST tile. Both can be PrimExpr.
- `col_count`: must be a compile-time int divisible by BLEN.

The output is mlen × col_count, written to `C_v` at columns
`[dst_col_offset, dst_col_offset + col_count)`.

### BTMM caveats

- `plena.btmm` semantics: `(B, S, H, D) @ (B, S, H, D) → (B, H, S, MLEN)`,
  per-head reduce over D=HLEN. Only correct for **Q @ K^T-style** patterns.
- **It is wrong for P @ V** because P @ V reduces over MLEN (not HLEN). Use
  `plena.mm_slot` per head for that.

---

## 4. TVMScript pitfalls

These break the kernel parse before codegen even runs:

- **`from __future__ import annotations` BREAKS TVMScript**. PEP 563
  defers all annotations to strings; TVMScript's parser falls back to
  `func.__annotations__` for some args, gets a string, hands it to a C++
  TVM API expecting an Object, and crashes with
  `expected Object but got str`. Just don't use it in kernel files.

- **Function-body Python locals get auto-promoted to LetStmt**. Writing
  `lane_mask = 1 << lane` inside a `@T.prim_func` body becomes a
  `lane_mask: T.int32 = T.shift_left(1, lane)` at TIR level — i.e. a
  bound `tir.Var`. The `ExprMaterializer` does **not** see through Lets,
  so it errors with "unbound tir.Var 'lane_mask'". **Always inline the
  expression at the call site** instead of binding to a local.

- **`for x in range(N)` and `for x in T.serial(N)` and `for x in T.unroll(N)`
  all create `tir.For` loops** — they do NOT Python-iterate. The loop
  variable is a `tir.Var`. Mul/Add involving it become `tir.Mul/Add`
  PrimExprs, not Python ints.

  - `T.unroll(N)` annotates `kind=ForKind.UNROLLED`; isa_pass's `_emit_for`
    detects this and emits the body N times with `gp_idx` pre-set to each
    iter value. **The body's tir.Var references still resolve via the
    materializer's symbol_table to that gp register** — so scalar args
    like `kv_block * mlen` still appear as PrimExpr Muls, not constants.
    If you need a literal for an op that demands compile-time int (rare,
    e.g. an op that bakes the value into asm-text), Python-unroll the
    call site (write N explicit copies).

- **TIR Mul / Add of constants doesn't fold automatically** until you
  hit the materializer. `Add(IntImm(c1), Add(IntImm(c2), x))` is now
  flattened to `Add(c1+c2, x)` at materialise time
  (`expr_materializer.py:113`), and `Add(IntImm, Var)` lowers to a single
  `S_ADDI_INT` when the immediate fits in 12 bits (`_S_ADDI_MAX = 4095`).
  Don't manually pre-compute things in Python expecting tir folding —
  the materializer does it.

---

## 5. The 10 000-instr cap

The transactional emulator panics if any **hardware loop body** runs more
than `MAX_LOOP_INSTRUCTIONS` (default 10 000) **dynamic** instructions in
a single iteration. The check is per-loop on the loop stack
(`main.rs:1572`).

- This is **per loop, per iteration**. So for a row loop with 64 iters and
  body of 145 instr, the cap is `145 ≤ 10 000` (per row iter), and the
  outer loop sees `64 × 145 = 9 280` per its own iter — also under.
- An outer kv/q loop wrapped around a row loop would see `64 × <row
  body>` per its own iter; if that exceeds 10 000, **unroll the outer
  loop** with `T.unroll(N)`. Unrolled iters don't appear as a hardware
  loop, so they don't accumulate.
- Optimisations that lowered the row body include
  `_emit_fp_scalar_op_at`'s row-expr CSE (one materialise, S_ADDI per
  buffer) and the materializer's Add/Mul folds.

---

## 6. FP buffer layout convention (FlashAttention kernel)

FP buffers in `flash_attention_min.py` are declared as 1D per-lane
fragments `(rows,)` and the compiler expands each to `(lane_count, rows)`
inside the lane group. The address allocator places them sequentially
starting at `FPRAM_USER_BASE = 32`, each slot `lane_count * rows` wide
(= `4 * 64 = 256` for the typical config).

Current FP buffers (per the actual HLIR, in declaration order):

```
M_OLD   addr = 32 + 0 * 256 = 32
M_CURR  addr = 32 + 1 * 256 = 288
M_RES   addr = 32 + 2 * 256 = 544
L_OLD   addr = 32 + 3 * 256 = 800
L_NEW   addr = 32 + 4 * 256 = 1056
P_SUM   addr = 32 + 5 * 256 = 1312
L_INV   addr = 32 + 6 * 256 = 1568
```

Per-lane addressing within an FP buffer: element `[lane, row]` is at
offset `base + lane * rows + row`.

**Scale / M_init / L_init are no longer declared FP buffers.** The
kernel embeds the literals directly as `T.float16(...)`
(`scale_val = 1/sqrt(d_k)`, `-1.0e4` for the M init, `0` for L). The
`hoist_float_constants` pre-pass synthesises a 1-slot `global.fpram`
buffer per unique value (e.g. `__const_f16_0p25`,
`__const_f16_neg10000`), and `test_helper` auto-preloads them from the
`--dump-buffer-addrs` JSON. So the kernel no longer needs the testbench
to preload an `active_lane` segment of any user FP buffer — every FP
buffer is written before it is read (M_OLD/L_OLD reset from the hoisted
consts at the top of each q_block).

---

## 7. FlashAttention kernel structure (current state)

`flash_attention_min.py` produces this op nest (HLIR view, with
`head_count=8, num_q_blocks=2, num_kv_blocks=2`). All heads are run —
the per-`by_phase` loops below cover the full `lane_count` (= MLEN/HLEN):

```
for q_block in [0, 2):                    ; outer Q loop
    dma Q[q_block] -> Q_sh
    for row in [0, 64):                    ; zero running output
        v_zero O_loc[row]
    for row in [0, 64):                    ; reset FP state (ALL lanes)
        for by_phase in [0, 4):
            fp_copy_at __const_f16_neg10000 -> M_OLD
        for by_phase in [0, 4):
            fp_zero_at L_OLD
    for kv_block in [0, 2):                ; KV loop
        dma K[kv_block] -> K_sh
        dma V[kv_block] -> V_sh
        btmm Q_sh @ K_sh -> S_loc          ; per-head Q @ K^T
        for row in [0, 64):
            for by_phase in [0, 4):
                row_mul_fp S_loc *= __const_f16_0p25   ; 1/sqrt(d_k)
            for by_phase in [0, 4):
                fp_copy_at M_OLD -> M_CURR
        for row in [0, 64):
            for by_phase in [0, 4):
                row_reduce_max_at S_loc -> M_CURR
        for row in [0, 64):
            for by_phase in [0, 4):
                fp_sub_at M_OLD - M_CURR -> M_RES
            for by_phase in [0, 4):
                fp_exp_at M_RES -> M_RES
            for by_phase in [0, 4):
                row_sub_fp S_loc -= M_CURR
            for by_phase in [0, 4):
                row_exp S_loc = exp(S_loc)
            for by_phase in [0, 4):
                fp_zero_at P_SUM
        for row in [0, 64):
            for by_phase in [0, 4):
                row_reduce_sum_at S_loc -> P_SUM
        for row in [0, 64):
            for by_phase in [0, 4):
                fp_mul_at L_NEW = L_OLD * M_RES
            for by_phase in [0, 4):
                fp_add_at L_NEW += P_SUM
            for by_phase in [0, 4):
                row_mul_fp O_loc *= M_RES
            for by_phase in [0, 4):
                fp_copy_at M_CURR -> M_OLD
            for by_phase in [0, 4):
                fp_copy_at L_NEW -> L_OLD
        for by_phase in [0, 4):            ; per-head P @ V
            matmul S_loc[by_phase] @ V_sh[..by_phase..] -> PV_loc[..by_phase..]
        for row in [0, 64):
            v_add O_loc += PV_loc
    for row in [0, 64):                    ; finalize: O /= L_new
        for by_phase in [0, 4):
            fp_reci_at L_NEW -> L_INV
        for by_phase in [0, 4):
            row_mul_fp O_loc *= L_INV
    dma O_loc -> O_hbm[q_block]
```

- **Softmax is run on every head** — each `for by_phase in [0, 4)` walks
  the full lane group. (An earlier version ran only a single
  `active_lane`; that is no longer the case.)
- **P @ V uses `plena.matmul`** (`M_MM` / `M_MM_WO`), one issuance per
  head — *not* `mm_slot`.
- `M_OLD` / `L_OLD` are **re-initialised inside the q_block loop** from
  the hoisted consts `__const_f16_neg10000` / a zero store, so a
  multi-q-block run resets cleanly per tile.

### Layouts

- `S_loc` is **BHSD** (BTMM #1's natural output): each VRAM row is one
  head's full mlen-wide score row.
- `O_loc` / `PV_loc` are **BSHD**: heads occupy column slots within a
  row. `matmul` writes head h's hlen columns at the matching slot.

### What's intentionally NOT done yet

- **Causal mask** — needs a preloaded VRAM `mask` buffer + `v_add`
  before softmax. Mirror `attention.py`'s approach.
- **Batch > 1**.

---

## 8. Testbench conventions
(`transactional_emulator/testbench/tvm_flash_attention_min_test.py` and
similar `tvm_*_test.py` files at the testbench root.)

- The build recipe `just build-emulator-debug <name>` looks for the script
  at `transactional_emulator/testbench/<name>_test.py` (top level), or for
  a hard-coded list of names like `tvm_online_softmax_min` it routes to
  `transactional_emulator/testbench/tile_tensor_kernel_programs/<name>.py`.
- Use a robust repo-root finder (walk up `_THIS_FILE.parents` for
  `.venv-tvm` and `compiler/`) — depth depends on which subdir the script
  ends up in.
- The compile happens via `subprocess.run(VENV_TVM_PYTHON, "-m",
  "tilelang_tvm_compiler", "compile", ...)` because TVM only lives in
  `.venv-tvm`. The test script itself runs in the main `.venv` (Python 3.12).
- `create_sim_env` lays down `Q_hbm.pt`, `K_hbm.pt`, `V_hbm.pt`, `O_hbm.pt`,
  `golden_result.txt`, `fp_sram.bin`, `int_sram.bin`. `create_mem_for_sim`
  produces `generated_machine_code.mem` and `hbm_for_behave_sim.bin`.
- `comparison_params.json` controls the post-run diff — set `num_rows`,
  `num_batches`, `elements_per_batch`, `row_dim` so they tile the flat
  golden correctly.

### Golden gotchas

- The kernel runs softmax on **all heads** now, so the golden is plain
  per-head `softmax(scaled_score) @ V` for every head. (An earlier
  version ran a single `active_lane` and the golden had to mirror that
  with `score @ V` for non-active heads — that is no longer needed.)
- `torch.softmax(x, dim=-1)` is mathematically equivalent to the kernel's
  online `max → sub → exp → sum → divide` chain. Either works for the
  golden; the online form is only needed if you want to model the f16
  truncation of each FPRAM scalar step.

### Golden comparison — the biggest time sink (read this)

The golden *comparison* path has bitten us harder than any kernel bug.
Two distinct bugs, both in how the golden reaches `check_mem.py`:

1. **`golden_result.txt` was written with `%.2f`** (2 decimal places).
   `check_mem.parse_golden_output` then parsed that text back as the
   golden — so a true golden of `-0.0083` became `-0.01`, and small
   values showed a fake ~0.8 relative error. Fix: `create_sim_env.py`
   now also writes a lossless `golden_output.pt`; `parse_golden_output`
   prefers the `.pt` and only falls back to the text.

2. **`check_mem.py` down-cast the golden to `bfloat16`** before
   comparing ("for fair comparison with hardware"). bf16 has 7 mantissa
   bits — rounding the golden first inflates `|err|/|golden|` for small
   values. Fix: keep the golden `.float()`; only the *simulated* side
   stays bf16 (that IS the VRAM storage).

If a comparison shows a wall of fake error on small magnitudes, suspect
the comparison path before the kernel. Verify `golden_output.pt` exists
in `build/` and that `parse_golden_output` is reading it.

### Diagnosing a chained-kernel (SSB / MMDiT) failure

When a multi-kernel chain's match rate collapses but each kernel passes
its own standalone test, the bug is almost always a **kernel-to-kernel
hand-off**, not a kernel. The proven isolation technique:

- Give the suspect kernel's input an **independent HBM tensor** (its own
  address, role `"input"`, preloaded) instead of aliasing the upstream
  kernel's output `"scratch"` buffer. The upstream kernel still runs but
  can no longer overwrite the isolated input.
- Feed that independent input either a clean random tensor *or* the
  upstream kernel's own golden output.
- If the kernel now passes → the bug is the upstream→this hand-off (the
  sim writing the shared HBM wrong, or a layout mismatch). If it still
  fails → the kernel itself, in the chain environment, is wrong.
- **Pitfall**: do NOT just preload the shared `scratch` buffer — the
  upstream kernel will overwrite it at runtime. The input must be a
  *separate* address the upstream kernel never writes.

---

## 9. Lessons from previous failure modes

These are real bugs that cost time during development. If you reintroduce
any of these, the test will fail in confusing ways:

- **Don't reuse one scalar for two different addressings**. The pre-fix
  `_at` ops took a single `row` scalar and used it as both VRAM row index
  and FP element offset. For multi-lane FP state with single-lane VRAM
  data (BSHD), these need to differ. Hence the current `(vram_row, lane,
  mask)` triple.

- **Don't put all kv_blocks in one hw loop body**. With softmax body ~145
  instr × 64 row iters = 9 280 per kv iter, an outer `T.serial(num_kv)`
  loop would multiply that into one hw-loop iter and hit the 10 000 cap.
  Use `T.unroll`. Same applies to q_block.

- **`M_OLD` / `L_OLD` must be reset *inside* the q_block loop**. After
  the first q_block runs, `M_OLD` is overwritten by `fp_copy(M_curr →
  M_old)` at the end of every row, so the next q_block would start from
  stale state. The current kernel handles this correctly: it re-inits
  `M_OLD` / `L_OLD` from the hoisted consts (`__const_f16_neg10000` /
  zero) at the top of each q_block. Do not move that reset outside the
  loop or back to a one-time preload.

- **Don't use `from __future__ import annotations`** in kernel files. (See
  §4.)

- **`mm_slot` LHS check used to require single-tile**. Was relaxed to allow
  multi-head LHS via `lhs_row_offset` scalar. Existing callers (tiled_mm)
  pass `0` as the new first scalar.

---

## 10. Useful one-liners

Recompile a kernel + dump HLIR (from repo root):

```
PYTHONPATH=compiler LD_LIBRARY_PATH= .venv-tvm/bin/python -m tilelang_tvm_compiler \
  compile \
  --kernel "tilelang_tvm_compiler.kernels.flash_attention_min:make_flash_attention_min" \
  --kernel-kwargs "rows=64,hlen=16,lane_count=4,active_lane=2,num_kv_blocks=2,num_q_blocks=2" \
  --asm-name flash_attention_min --mlen 64 --btmm-hlen 16 --stage-output O_hbm \
  --dump-hlir transactional_emulator/testbench/build/flash_attention_min.hlir.txt \
  > transactional_emulator/testbench/build/flash_attention_min_generated_asm_code.asm
```

Run all relevant unit tests:

```
cd compiler && PYTHONPATH=. LD_LIBRARY_PATH= ../.venv-tvm/bin/python \
  tilelang_tvm_compiler/tests/test_expr_materializer.py
# then test_narrow_mm_emitter.py, test_fpram_ops.py, test_loop_dma.py, ...
```

Measure inner-loop body size (in lines, ≈ static instr) to verify the
10 000-cap budget:

```
awk '/; for row in/{f++} f==1 && /C_LOOP_START.*64/{n=NR+1; next} \
     f==1 && /C_LOOP_END/ && n {print "row body:", NR-n; exit}' \
  transactional_emulator/testbench/build/<asm_file>.asm
```

Show the high-level ISA structure (loops + matmul ops) without flooding
the terminal:

```
grep -nE '^; for |^C_LOOP_(START|END)|^M_BTMM|^M_BMM_WO|^M_MM\b|^V_ADD_VV' \
  transactional_emulator/testbench/build/<asm_file>.asm
```

---

## 11. Lane fusion — the core mechanism

Multi-lane fusion is the heart of the frontend: `MLEN / HLEN` hardware
lanes are packed into one VRAM row, and one multi-lane HW op fires once
for all lanes instead of looping. Getting the **lane axis** right is
what makes this work.

### Lane axis is picked by IR-node analysis, NOT string matching

`infer_lane_axis.py` decides which `blockIdx.*` grid var is the lane
axis. The judgment is done on the **TIR AST**, not on text:

```python
# infer_lane_axis._collect_bare_index_var_names
def visit(node):
    if isinstance(node, tir.BufferLoad):
        for idx in node.indices:
            if isinstance(idx, tir.Var):      # ← the actual test
                found.add(idx.name)
stmt_functor.post_order_visit(func.body, visit)
```

The rule: a grid var is a **lane candidate** iff it appears as a
**bare index slot** — `BufferLoad.indices[i]` is *exactly* a `tir.Var`
node — somewhere in the body, AND its extent is divisible by LANE.

- `Q_hbm[0, q_block*rows, by, 0]` — the `by` slot is a naked `tir.Var`
  node → `by` is a lane candidate.
- `q_block * rows` — that's a `tir.Mul` node; `q_block` is *inside* it,
  not bare → `q_block` is an outer control loop, **not** a lane axis.

A plain string search for `"by"` could never tell those apart — both
texts contain `by`/`q_block`. Only inspecting the IR node *type* at the
index position (`isinstance(idx, tir.Var)`) distinguishes a per-lane
index from an arithmetic offset. This is the precise sense in which
lane fusion is **value/ref-based, not string-based**.

Resolution: 0 candidates → no lane axis (cluster pipeline skipped);
1 → picked; 2+ → `InferLaneAxisError`, author must set
`T.func_attr({"plena.lane_axis": "<name>"})`. A manual attr always wins.

### How the lane axis flows through the rest of the pipeline

- **split** turns the lane blockIdx into `(number, phase)` and grows
  every non-global buffer by a cluster outer dim, so per-lane data has
  somewhere to live.
- **mark** tags each op with its lane-fusion role; **async_wrap** groups
  can-async ops; **fuse** collapses each Async region into a single
  `MultiLaneOp` — that is the actual "fire once for all lanes" step.
- **view** assigns a `view_perm` to every `BufferRef` and substitutes
  the lane var on HBM refs; **burn_view** bakes that permutation into
  the physical shape + index tuples.

### Bare-Var detection vs. lane-var substitution — two different things

`infer_lane_axis`'s **bare-`tir.Var`** test only *picks which axis is
the lane*. It is deliberately strict: `by + 8` (a `tir.Add`) is NOT a
lane candidate, only a naked `by` is. This keeps the axis choice
unambiguous.

Once the axis is chosen, **`view._subst_lane_var` is recursive** and
handles the lane var wherever it sits inside an index expression — not
just bare:

```python
def _subst_lane_var(idx, ctx):
    if isinstance(idx, VarRef) and idx == ctx.original_var:
        return phase + number * cluster_count        # the substitution
    if isinstance(idx, dict):                        # compound node (add/mul/…)
        return {"op": idx["op"],
                "args": [_subst_lane_var(a, ctx) for a in idx["args"]]}
    return idx
```

So an HBM index like `O_hbm[..., by + o_head_offset, ...]` works: the
recursion descends into the `add`, finds the `VarRef(by)` inside, and
rewrites just that leaf — the `+ o_head_offset` is preserved. This is
what `flash_attention_min.py` relies on to write its output into a
head-slice of a wider tensor.

Summary: **lane-axis selection** = strict bare-Var only; **lane-var
substitution** = recursive, accepts the lane var combined with offsets
(`by + c`, `by * c`, …).

So "multi-lane fuse" is not one pass — it's the chain
`infer_lane_axis → split → mark → async_wrap → fuse → view → burn_view`,
all operating on typed mid_ir nodes.
