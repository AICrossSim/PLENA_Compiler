# Plan: RoPE via shuffle-matrix matmul (kill the FPRAM pair-swap)

## Why rope_min is slow (~53 ms, ~57% of the SSB layer)

The pair-swap (OUT[d] depends on X[d] AND its pair partner X[d^1]) is a
cross-element shuffle. The current kernel handles it by mapping each VRAM
row into FPRAM (`T.copy(XQ_sh[row,0], X_FP)` = v2f) and doing per-element
scalar FMA addressing `X_FP[e]` / `X_FP[o]` as scalars
(rope_min.py:118-130). Per row: 4× v2f MAP + a scalar chain over half_dim +
1× f2v. On 1024×1024 the V↔FPRAM MAPs + scalar ops dominate → 53 ms.

## The math (turn the shuffle into a matmul + vector ops)

Current per-element form (rope_min.py:127-128, C=COS, S=SIN, NS=-SIN):

    OUT[e] = X[e]*C[e] + X[o]*NS[e]      e = 2i   (even)
    OUT[o] = X[o]*C[o] + X[e]*S[o]       o = 2i+1 (odd)

Every output is `OUT[d] = X[d]*COS[d] + X[d^1]*SGN_SIN[d]` where:
- `d^1` = pair partner (even↔odd swap),
- `SGN_SIN[d]` = -sin at even d, +sin at odd d (i.e. the existing
  NEG_SIN/SIN interleaved by position — host can pre-combine into ONE
  `SGN_SIN` tensor, or keep two and select; see Open Q1).

So, fully vectorized:

    OUT = X ⊙ COS  +  shuffle(X) ⊙ SGN_SIN

where ⊙ is elementwise (whole-tile V_MUL/V_ADD) and `shuffle(X)` swaps
each adjacent (even,odd) pair.

### shuffle as a matmul

`shuffle(X) = X @ P`, P is the (hlen × hlen) **pair-swap permutation
matrix**: block-diagonal, each 2×2 block = [[0,1],[1,0]]. I.e.
`P[2i, 2i+1] = P[2i+1, 2i] = 1`, else 0.

P is symmetric (P = P^T), so transpose_B doesn't matter — convenient for
the BTMM path (which transposes the MRAM tile).

`X @ P`: X is (rows, hlen), P is (hlen, hlen) → K-dim = hlen = 128 (same
as linear's K; BTMM splits it by HLEN into lane_count partials that get
summed — handled by the existing btmm_mm + bmm_wo accumulate, §2.2.1).

## Kernel structure (new rope kernel, no FPRAM)

```
# load X, COS, SGN_SIN tiles (HBM -> VRAM), P (HBM -> MRAM auto)
XS = X @ P                      # btmm_mm: shuffle(X), into XS_loc (VRAM)
                                #   (K-loop trivial: 1 k_block, K=hlen)
T1 = X  ⊙ COS                   # whole-tile V_MUL  -> T1 (VRAM)
T2 = XS ⊙ SGN_SIN               # whole-tile V_MUL  -> T2 (VRAM)
OUT = T1 + T2                   # whole-tile V_ADD  -> Q_OUT (VRAM)
# OUT -> HBM
```

All four compute steps are whole-tile vector / BTMM ops — no per-element
FPRAM scalar, no V↔FPRAM MAP. This is the same shape as the activation
kernels after the VRAM rewrite, so it folds/fissions cleanly.

## Decisions (locked)

- **P source**: host pre-generates the pair-swap permutation matrix as an
  HBM input tensor (testbench builds it). Kernel just loads it; B-operand
  auto-routes to MRAM via `_infer_scope_overrides`.
- **X@P path**: reuse the new BTMM `btmm_mm` path (M_BTMM + deferred
  bmm_wo drain + lane-tile accumulate). P as B → MRAM.

## Resolved decisions

- **Q1 SGN_SIN — RESOLVED (pre-combine)**: host pre-combines into ONE
  `SGN_SIN[d]` tensor (even d = -sin, odd d = +sin). Second term is a
  single `XS ⊙ SGN_SIN`. One V_MUL, one fewer HBM input than the
  NEG_SIN+SIN split.

- **Q2 layout — RESOLVED (BSHD → BS·1·(H*D))**: collapse head+dim into one
  `H*D` axis so the tile is a big linear-style block, NOT per-head. Tensors
  become `(1, SEQ, 1, H*D)` (H*D = 8*128 = 1024 = 2 MLEN blocks). The
  shuffle is then a plain **(SEQ × H*D) @ (H*D × H*D)** GEMM, exactly like
  linear_min: M=SEQ, N=K=H*D=1024, grid over (n_block, m_block),
  k_blocks = H*D/MLEN = 2. P is the **(H*D × H*D)=(1024×1024) pair-swap**
  permutation = block-diagonal 2×2 swaps `[[0,1],[1,0]]`
  (P[2i,2i+1]=P[2i+1,2i]=1) across the whole H*D row. Head boundaries are
  multiples of 128 (even), so no 2×2 straddles a head → the global
  pair-swap == per-head pair-swap. Runs over the verified btmm_mm path
  (multi-k_block accumulate). **lane-split concern gone — it's just linear.**

- **Q3 operand placement — RESOLVED**: X (rows×512 activation tile) is the
  matmul LHS, stays VRAM; P is the B operand → auto MRAM via
  `_infer_scope_overrides`. Per BTMM contract.

- **Q4 numerics — low risk**: P is a 0/1 permutation; 0 and 1 are exact in
  MX-E4M3, so X@P adds no quant error beyond X's own. Confirm P's MX
  round-trip keeps exact 0/1 during impl.

## Expected payoff

rope from ~53 ms (FPRAM scalar) → a handful of whole-tile BTMM+V_MUL+V_ADD
ops, expected 1-2 orders of magnitude faster (cf. gelu 164→2.3 ms, linear
70→0.38 ms after the analogous vector/BTMM rewrites). Would drop rope from
~57% of the SSB layer to a small fraction, shifting the bottleneck to
flash_attention.

## Same structure applies to K-side RoPE

K-side RoPE is identical with SIN↔NEG_SIN role swap — same shuffle-matmul
+ vector form.
