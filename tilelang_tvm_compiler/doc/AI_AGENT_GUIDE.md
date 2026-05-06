# tilelang_tvm_compiler — AI Agent Knowledge Base

This file is intentionally written for AI agents (and humans) starting a new
session. It captures the non-obvious lessons accumulated while building the
TVM-frontend path from TIR → HLIR → PLENA ISA. **Read this before editing
kernels under `kernels/`, intrinsics, the lowering pass, or testbenches under
`transactional_emulator/testbench/`.**

If you discover something the next agent will trip on, append it here.

---

## 1. Pipeline at a glance

```
  TIR PrimFunc
     │  (PlenaCodegen.lower_to_hlir, codegen.py)
     ▼
  HLIR Module                 ← buffers + Op stream, no addresses
     │  (AddressAllocationPass, address_alloc.py)
     ▼
  HLIR Module + addresses     ← per-buffer base address resolved
     │  (IsaEmitterPass.run, isa_pass.py)
     ▼
  ISA text  (printed to stdout / `*_generated_asm_code.asm`)
```

- The compiler is invoked as a subprocess (`python -m tilelang_tvm_compiler
  compile ...`) from a Python 3.11 venv (`.venv-tvm`) because TVM is only
  installed there. The main project venv (`.venv`, 3.12) is for testbench
  inputs/golden via PyTorch.
- `--dump-hlir <path>` writes the post-pass-2 HLIR — extremely useful for
  debugging op ordering and scalar-expression rendering. **It is only
  written if compile_kernel returns successfully**; on a pass-3 failure the
  HLIR file you see may be stale from a previous run.

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

FP buffers in `flash_attention_min.py` are all `(lane_count, rows)` shape.
The address allocator places them sequentially starting at `FPRAM_USER_BASE
= 32`. Declaration order **matters** — the testbench preload depends on it:

```
M_old   addr = 32 + 0 * 256 = 32
M_curr  addr = 32 + 1 * 256 = 288
M_res   addr = 32 + 2 * 256 = 544
L_old   addr = 32 + 3 * 256 = 800
L_new   addr = 32 + 4 * 256 = 1056
P_sum   addr = 32 + 5 * 256 = 1312
Scale   addr = 32 + 6 * 256 = 1568
L_inv   addr = 32 + 7 * 256 = 1824
M_init  addr = 32 + 8 * 256 = 2080
L_init  addr = 32 + 9 * 256 = 2336
```

Per-lane addressing within an FP buffer: element `[lane, row]` is at offset
`base + lane * rows + row`. For `active_lane=2, rows=64`, that's
`base + 128 + row`. **The active_lane segment must be preloaded by the
testbench** for buffers the kernel reads before writing
(`Scale`, `M_init`, `L_init`).

---

## 7. FlashAttention kernel structure (current state)

`flash_attention_min.py` produces this op nest (HLIR view, with
`active_lane=2, num_q_blocks=2, num_kv_blocks=2`):

```
for q_block in [0, 2):              ; outer Q loop, T.unroll
    dma Q[q_block] -> Q_v
    zero_v O_v
    for row in [0, 64):              ; reset active_lane FP state
        fp_copy_at M_init -> M_old
        fp_copy_at L_init -> L_old
    for kv_block in [0, 2):          ; KV loop, T.unroll
        dma K[kv_block] -> K_m
        dma V[kv_block] -> V_m
        btmm Q_v @ K_m -> S_v        ; per-head Q @ K^T
        for row in [0, 64):           ; online softmax body, active_lane only
            row_mul_fp_at S_v *= Scale          ; 1/sqrt(d_k)
            fp_copy_at M_old -> M_curr
            row_reduce_max_at S_v -> M_curr     ; m = max(m_old, row_max)
            fp_sub_at M_old - M_curr -> M_res   ; m_old - m_curr
            fp_exp_at M_res -> M_res            ; exp(m_old - m_curr)
            row_sub_fp_at S_v -= M_curr
            row_exp_at S_v = exp(S_v)           ; P_block (un-normalised)
            row_reduce_sum_at S_v -> P_sum
            fp_mul_at L_new = L_old * M_res
            fp_add_at L_new += P_sum
            row_mul_fp_at O_v *= M_res          ; rescale prev O (BSHD, masked)
            fp_copy_at M_curr -> M_old
            fp_copy_at L_new -> L_old
        for h in [0, 4):              ; per-head P @ V via mm_slot
            mm_slot S_v[h] @ V_m[..h..] -> PV_v[..h..]
        v_add O_v += PV_v
    for row in [0, 64):              ; finalize: O /= L_new
        fp_reci_at L_new -> L_inv
        row_mul_fp_at O_v *= L_inv          ; BSHD, masked
    dma O_v -> O_hbm[q_block]
```

### Two layouts collide here

- `S_v` is **BHSD** (BTMM #1's natural output). Each VRAM row is one head's
  full mlen-wide score row. → `row_*_at` ops use `mask=0` and scalar
  `active_lane * rows + row` for both VRAM row & FP offset.
- `O_v` is **BSHD**. Heads occupy column slots within a row.
  → `row_mul_fp_at` for the rescale uses `mask = 1 << active_lane`,
  scalars `(row, active_lane, mask)`.

`PV_v` mirrors `O_v` (BSHD) so `v_add` and the BSHD layout match. `mm_slot`
writes head h's hlen columns at `dst_col_offset = h * hlen`.

### What's intentionally NOT done yet

- **Multi-head softmax**: only `active_lane` is run through softmax. The
  other 3 lanes' `S_v` rows stay as raw `Q @ K^T`, BTMM #2 (mm_slot) still
  runs per-head and writes `score @ V` for them. The testbench's golden
  mirrors this exactly (active_lane: full softmax(QK^T/√d) @ V; others:
  raw `score @ V`). To make it real multi-head, the easiest path is a
  software `for active_lane in T.unroll(lane_count)` around the softmax
  body (4× cost for correctness on all heads).
- **Causal mask** — needs a preloaded VRAM `mask` buffer + `v_add` before
  softmax. Mirror `attention.py`'s approach.
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

- The kernel currently runs softmax on `active_lane` only. So the golden
  for that head is `softmax(scaled_score) @ V`, but for **non-active heads
  the golden must be `score @ V` (no softmax)** to match what the kernel
  actually produces. Don't lazily run softmax on all heads in the golden.
- `torch.softmax(x, dim=-1)` is mathematically equivalent to the kernel's
  online `max → sub → exp → sum → divide` chain. We previously wrote it
  out manually for debugging; either works.

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

- **Don't preload `M_old` directly in a multi-q-block kernel**. After the
  first q_block runs, `M_old` is overwritten by `fp_copy(M_curr → M_old)`
  at the end of every row. The next q_block must reset from a separate
  `M_init` constant buffer. Same for L.

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
