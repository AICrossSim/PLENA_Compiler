# Experimental Vector FP32 dot accumulator

This is an opt-in numeric/timing control, outside the frozen Matrix-SRAM
L-Compute ISA/hardware claims. Ordinary VV words and default lowering remain
unchanged. It does not represent the historical Arlo instruction census.

## Storage and arithmetic

One VLEN-element FP32 accumulator plus a valid bit is added (8192 bytes at
VLEN=2048). It is separate from the BF16 Vector/Matrix SRAM arrays; their
capacity and ports remain as configured. This additional state is **not free**.
A physical implementation also needs a datapath that retains FP32 products and
partial sums and supports conversion on writeback. No RTL, area or frequency
result is supplied.

`V_DOT_RESET` zeros and activates the accumulator. `V_DOT_ACC` reads two whole
VLEN SRAM rows, multiplies their elements as FP32, then adds sequentially to the
FP32 accumulator (separate multiply and add, no fused FMA). `V_DOT_WRITE` rounds
the accumulator to the SRAM format, writes one full row, and invalidates it.
The KDA control uses BF16 SRAM. ACC/WRITE without RESET must fail.

## Encoding

| Assembly | Physical opcode | funct1 | Canonical unused fields |
|---|---|---|---|
| `V_DOT_RESET gp0, gp0, gp0, 0` | 0x0D | 8 | rd, rs1, rs2, rmask = 0 |
| `V_DOT_ACC gp0, gpX, gpY, 0` | 0x11 | 8 | rd, rmask = 0 |
| `V_DOT_WRITE gpD, gp0, gp0, 0` | 0x0F | 8 | rs1, rs2, rmask = 0 |

These reserve three previously invalid subfunctions, not three new physical
opcodes. They are explicit aliases; an ordinary VV mnemonic cannot silently
select them. Matrix-view forms still require a nonzero operand mask. Simulator
execution requires `PLENA_EXPERIMENTAL_FP32_DOT=1`; default execution rejects
RESET. Dependency tracking includes the accumulator as an architectural resource.

## KDA contract and timing assumptions

`lower_prepared_vector_recurrence(..., experimental_fp32_dot=True)` selects
this extension for KDA's prediction and output dots only. Decayed state remains
BF16; subtraction, beta multiply, rank-1 product and state addition keep the
ordinary VV BF16 boundaries. This matches the GPU diagnostic `promote_dot`,
not the common L_TILE rounding at every other arithmetic boundary.

Unified serial timing charges one issue cycle per instruction, two ordinary
single-port bank reads per ACC, and one write per WRITE. ACC assumes the
configured vector multiply plus add latency applies to FP32; RESET and the
writeback conversion each assume one cycle. DMA and address instructions remain
explicit and are included. The FP32 throughput is an assumption requiring RTL
validation; it is not an empirical silicon speedup.

A/B still expand coefficients in HBM whereas D uses compact fields. Therefore
the measured comparison includes DMA/layout differences. Per-token diagnostic
state snapshots add DMA and must never qualify a performance ratio.
