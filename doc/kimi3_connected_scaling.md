# Kimi K3 Connected Lowering: Scaling Boundary

The compact connected MLA, LatentMoE, AttnRes, KDA, and consecutive-block tests
execute in the Rust transactional emulator and compare against CPU references.
They prove dataflow and arithmetic for those blocks. They do not prove that the
current compiler can emit one practical 93-layer Kimi binary.

Two measured full-model attempts exposed the static emitter boundary:

- Before MRAM binding tracking was fixed, a 96-head build ran for 2 h 19 min
  without completing its first test. `reset_mram()` scanned every registered
  HBM tile for every projection tile.
- After reset became O(resident MRAM tiles), a one-head build reached routed MoE
  quickly but still exceeded a 10 min limit and 8 GiB RSS. The remaining cause
  is the static expansion of `92 layers x Top-16 = 1,472` expert bodies.

The routed path now emits one dynamic expert body inside a 16-iteration
`C_LOOP`. A compact Rust numerical run reads distinct expert IDs and route
weights on each iteration, remains bit-exact against the CPU reference, and
produces a 3,191-line compact test program in 24,297 cycles. Earlier static and
dynamic measurements were taken before later address/lifetime fixes, so they
are retained only in development history and are not presented as a controlled
speedup comparison. The public full-scale builder now fails fast on the
remaining Matrix and MLA expansion unless
`allow_unbounded_static_expansion=True` is explicitly supplied for diagnostics.
That override is not a deployable-binary claim.

After looped Top-K landed, a 93-layer `heads=1` diagnostic did finish, but it
still emitted 100,221,916 instructions and 3,739,264,558 bytes of assembly,
took 7m10s, and peaked at 24.1 GiB RSS. Latent MoE alone accounted for
91,162,388 instructions. This proves that the next boundary is the static
Matrix output-column/K-tile emitter, not merely MLA's head loop.

The required compiler work is:

1. Roll Matrix output-column and K-tile traversal into address-carrying hardware
   loops, including tile-major 64-bit expert-weight addressing.
2. Emit MLA's per-head projection/attention body as a dynamic 96-iteration loop.
3. Add Rust numerical tests for both loop bodies, then retry the 93-layer build
   under an explicit compile-time and machine-code-size budget.
