# PLENA Simulator Cost & Concurrency Model

What every PLENA opcode actually costs when run through
[`transactional_emulator`](../../transactional_emulator/src/main.rs),
and what (does / doesn't) run in parallel. Everything below was
verified from the Rust source on 2026-05-21. The active config is
analytic mode, dc_lib_en = 1, `mlen=1024`, `blen=8`, `hlen=128`,
`VLEN=1024`.

## TL;DR for compiler authors

1. The simulator runs **one sequential main task**. Every opcode
   handler `.await`s its `cycle!(N)` cost before returning. There is
   **no instruction-level parallelism**: while `M_MM` blocks for 1024
   cycles, no scalar / vector / scalar-FP instr makes progress.
2. The **only background task** is HBM DMA. Inside that task there
   are **no `cycle!()` calls** — HBM transfers are modelled as
   instantaneous, infinite-bandwidth, zero-latency.
3. **IntRAM access is 1 cycle**, identical to a scalar ADD. Spill
   to IntRAM is not a 5–6× penalty (that was an unjustified
   assumption); it's a 1× penalty. Register allocation should treat
   spill as essentially free.
4. **Belady's algorithm (farthest last_use) is the optimal spill
   picker** under this model — reload is a flat 1 cycle, so
   minimising the count of reloads equals minimising spill cost.
5. **LICM should be aggressive always** — hoisting one invariant
   out of an `extent=N` loop saves N×ALU cycles; the worst case
   (the hoisted value gets spilled and reloaded N times) costs
   N×LD_INT cycles = same total. Net is always non-negative.
6. **Strength reduction and rematerialisation give 0 benefit.**
   `S_SLLI_INT` and `S_MUL_INT` are both 1 cycle; `S_LUI_INT` and
   `S_LD_INT` are both 1 cycle. Trading one for the other never
   wins.
7. **C_LOOP is good for heavy bodies, bad for trivial bodies.** The
   `C_LOOP_END` adds 1 cycle per iteration on top of the body, so
   for a 1-cycle body the loop is 2× the unrolled cost. For a
   1024-cycle `M_MM` body the overhead is 0.1%.

## Concurrency: one task, sequential dispatch

`transactional_emulator/src/main.rs:2515-2521`:

```rust
#[tokio::main]
async fn main() {
    let executor = Executor::new();
    executor.spawn(start());
    executor.enter(Instant::ETERNITY).await;
}
```

`start()` builds the M/V/HBM machines and calls `do_ops` exactly
once. `do_ops` is a flat `while pc < ops.len()` loop:

```rust
async fn do_ops(&mut self, ops: &[Opcode]) {
    let mut pc = 0;
    while pc < ops.len() {
        match ops[pc] {
            M_MM { .. } => self.m_machine.mm(...).await,        // cycle!(1024)
            V_ADD_VV { .. } => self.v_machine.add(...).await,   // cycle!(1)
            S_ADD_INT { .. } => { ...; cycle!(1); }
            // ...
        }
        pc += 1;
    }
}
```

`cycle!(N)` expands to `Executor::current().resolve_at(now + N).await`.
That schedules a timer and suspends the current task until the
executor advances simulated time to `now + N`. Because only one
task ever issues opcodes, suspending it means *no further opcode
can issue* until the timer fires.

### Time only moves forward in `executor.enter()`

The scheduler loop (`lib/runtime/src/executor.rs:185-252`) is:

1. Run every ready task to completion (or until it `.await`s).
2. Pop the earliest pending timer. Set `now = timer.resolve_at`.
3. Wake that timer. Go to 1.

Crucially: **simulated time never advances except in step 2**.
Anything in step 1 — including HBM reads, channel sends, mutex
locks — completes "instantly" from the simulated-clock POV.

### `tokio::join!` is in-task, not cross-instr

The V_*_VV handlers use `tokio::join!(self.vram.read(vs1),
self.vram.read(vs2))` to read both operands "concurrently". This
join is *inside* one opcode handler in the main task; it does not
overlap with any other instruction.

### The lone exception: HBM DMA

The H_PREFETCH_V / H_PREFETCH_M / H_STORE_V handlers
(`main.rs:2039-2165`) all delegate to
`transfer_mx_from_hbm` (or its store counterpart). Inside,
`Executor::current().spawn(async move { ... })` fires off a
background task that:

* reads HBM 64 bytes at a time via `hbm_clone.read(addr).await`,
* sends the assembled tensor through a oneshot channel,
* the main task's `continous_write_delayed(...).await` waits on
  that channel and writes the destination tile.

But `hbm.read()` itself contains *no* `cycle!()` calls. The
background task therefore runs to completion in step 1 of the
scheduler loop — instantaneously, in simulated time. The main
task's `.await` on the channel completes at the same `now`.

This means: the HBM model is **infinite bandwidth, zero latency,
unbounded outstanding transactions**. Issuing an `H_PREFETCH_V`
costs zero cycles on the main timeline, and the prefetched data is
available immediately to any downstream consumer.

## Per-opcode cycle cost

All costs are for analytic mode, `dc_lib_en=1`. dc_lib_dis varies
slightly (mostly 2× scalar FP exp/sqrt/reci) but `SCALAR_INT_BASIC`
is 1 in both modes.

### Scalar integer (`S_*_INT`)

| Opcode | Cycles |
|--------|-------:|
| `S_ADD_INT` | 1 |
| `S_ADDI_INT` | 1 |
| `S_SUB_INT` | 1 |
| `S_MUL_INT` | 1 |
| `S_LUI_INT` | 1 |
| `S_SLL_INT` / `S_SLLI_INT` | 1 |
| `S_SRL_INT` / `S_SRLI_INT` | 1 |
| **`S_LD_INT`** | **1** |
| **`S_ST_INT`** | **1** |

Every entry calls `cycle!(*SCALAR_INT_BASIC_CYCLES)` and that
constant is `1` (see `load_config.rs:339`). IntRAM access is the
same cost as ADD.

### Scalar FP (`S_*_FP`)

| Opcode | Cycles (analytic) |
|--------|-------:|
| `S_ADD_FP` / `S_SUB_FP` / `S_MUL_FP` / `S_MAX_FP` | 1 |
| `S_EXP_FP` | 1 |
| `S_RECI_FP` | 1 |
| `S_SQRT_FP` | 1 |
| `S_LD_FP` / `S_ST_FP` | 1 |
| **`S_MAP_V_FP`** | **1024 (`VLEN`)** |
| **`S_MAP_FP_V`** | **1024 (`VLEN`)** |

`S_MAP_V_FP` / `S_MAP_FP_V` broadcast a scalar across a vector;
they cost a full vector pass. Avoid them in inner loops.

### Vector (`V_*`, 1024-wide)

| Opcode | Cycles |
|--------|-------:|
| `V_ADD_VV` / `V_ADD_VF` | 1 |
| `V_SUB_VV` / `V_SUB_VF` | 1 |
| `V_MUL_VV` / `V_MUL_VF` | 1 |
| `V_EXP_V` | 1 |
| `V_RECI_V` | 2 |
| `V_RED_MAX` | 4 |
| `V_RED_SUM` | 8 |

`V_RED_SUM` is the most expensive scalar-to-from-vector op. Used
once per softmax row, so flash-attention pays
`8 × num_softmax_rows` for it.

### Systolic (`M_*`)

| Opcode | Cycles |
|--------|-------:|
| `M_MM` / `M_TMM` | **mlen = 1024** |
| `M_BMM` / `M_BTMM` | 1024 |
| `M_MV` / `M_TMV` | 1024 |
| `M_BMV` / `M_BTMV` | **1** |
| `M_MM_WO` / `M_BMM_WO` / `M_MV_WO` / `M_BMV_WO` | 1 |

`M_BMV` / `M_BTMV` cost just 1 cycle — they're the broadcast
matrix-vector op, the lane-parallel sibling of `M_MV`. flash-decode
relies on this.

`M_MM` dominates everything in flash-attention: at 1024 cycles
each, 2048 of them = 2,097,152 cycles, which is ~99.5% of the
kernel's simulated runtime. Optimising the surrounding scalar
arithmetic to save dozens of cycles is statistical noise.

### HBM (`H_*`)

| Opcode | Cycles |
|--------|-------:|
| `H_PREFETCH_V` | **0** |
| `H_PREFETCH_M` | **0** |
| `H_STORE_V` | **0** |
| `H_LOAD_V` | **0** |

All HBM ops dispatch via `Executor::spawn` and rely on tile-future
machinery; the main task isn't charged. See "Concurrency" above
for why.

### Control (`C_*`)

| Opcode | Cycles |
|--------|-------:|
| `C_SET_ADDR_REG` | 1 |
| `C_SET_SCALE_REG` | 1 |
| `C_SET_STRIDE_REG` | 1 |
| `C_SET_V_MASK_REG` | 1 |
| `C_LOOP_START` | 1 |
| `C_LOOP_END` | 1 *per iteration that loops back* |

`C_LOOP_END` runs N times for an `extent=N` loop (once at the end
of each iteration's body), each costing 1 cycle. `C_LOOP_START`
runs only once.

## C_LOOP vs unroll

A loop `for i in [0, N): body` costs:

* Unrolled: `N × body_cycles`
* C_LOOP: `1 + N × (body_cycles + 1)`

Per-iteration overhead is 2 extra cycles (the START at the top,
the END at the bottom — but START is amortised over N iters so
effectively just 1 extra per iter from C_LOOP_END).

Break-even point:

| body_cycles | unroll | C_LOOP | C_LOOP / unroll |
|---:|---:|---:|---:|
| 1 (scalar) | N | 1 + 2N | ~2× |
| 5 (small vec chain) | 5N | 1 + 6N | ~1.2× |
| 50 | 50N | 1 + 51N | 1.02× |
| 1024 (M_MM) | 1024N | 1 + 1025N | 1.001× |

* **Trivial bodies**: unroll is ~2× faster. ISA bloat per iter is
  small enough that unrolling is also acceptable for size.
* **Heavy bodies (M_MM)**: C_LOOP overhead is negligible. Use
  `loop_kind="serial"` to keep ISA compact.

## How this maps to compiler decisions

The v2 compiler is documented at
[`plena_v2_pipeline.md`](./plena_v2_pipeline.md). Specific
choices justified by this cost model:

- **Spill picker uses Belady (farthest last_use).** At a uniform
  1-cycle reload, the total reload count = the total spill cost,
  and Belady provably minimises miss/reload count.
- **LICM is enabled by default.** Hoist always non-negative even
  under worst-case spill thrashing.
- **`const_fold` peephole keeps `S_ADDI_INT %x, 0 → %x`**.
  Eliminating an ADDI saves 1 cycle.
- **`reassociate` canonicalises ADD chains** and folds matching
  ±terms. Lets CSE find shared partial sums (e.g.
  `result_addr = mat_addr + orow_term`).
- **No strength reduction pass.** SLLI / MUL / LUI / LD_INT are
  all 1 cycle.
- **No rematerialisation pass.** Same reason.
- **PreIsaPassV2 emits `loop_kind="unroll"` for small extents and
  should switch to `"serial"` for M_MM-dominated bodies.** (Open
  item — see v2 pipeline doc.)
- **DMA placement is "issue early, no scheduling needed".** Any
  H_PREFETCH that hoists out of an inner loop is free. There is no
  reason to limit prefetches by count.

## Where this model deviates from real hardware

This is a *behavioural* simulator. Real PLENA almost certainly has:

* finite HBM bandwidth (model says 0)
* separate M, V, S issue ports that can overlap (model has 0
  parallelism)
* a non-trivial pipeline depth before issued instructions retire

A compiler tuned aggressively against this simulator may make
choices that look bad on silicon. Concretely:

* aggressive prefetch may saturate HBM bandwidth on real hardware
* aggressive LICM increases register pressure that the simulator
  resolves for free via IntRAM spill at 1 cycle, but real spill
  costs depend on the IntRAM port latency
* there's no benefit modelled for issuing scalar arithmetic in
  parallel with M_MM, so the compiler won't schedule for it

If/when the simulator gains a more realistic timing model, revisit
this doc and the spill/LICM thresholds.
