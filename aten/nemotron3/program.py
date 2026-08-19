"""Emit one physical program covering all 52 Nemotron 3 layers.

Until now the Mamba mixers lowered to real assembly through ``IsaBuilder`` while
attention and MoE stayed semantic events, because the two halves are built by
different frameworks: ``aten.state.isa_lowering`` hand-picks registers and writes
assembly text, and ``aten.plena.PlenaCompiler`` owns a register allocator and a
tiled memory map. Concatenating their text would look like it compiled and then
clobber registers across the seam.

Two facts make one program possible:

* ``IsaEmitMixin.emit`` already accepts an ``IsaBuilder``, so a hand-written block
  can be spliced into a compiler-built program.
* ``flash_attention`` and the routed-MoE emitters allocate and free in balanced
  pairs, so the register file is empty between layers.

So the seam is placed at layer boundaries: each layer reloads the addresses it
needs and leaves nothing live in a register, and only the hidden activation in
Vector SRAM crosses the boundary. ``assert_registers_are_free`` enforces that
after every layer instead of trusting it -- a leak there is exactly the failure
that would be invisible in the assembly text and wrong in the machine words.
"""

from __future__ import annotations

from collections import defaultdict

from aten.mamba.scheduler import (
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    SchedulePhase,
)
from aten.nemotron3.scheduler import (
    HybridLayerType,
    Nemotron3Architecture,
    SYMBOL_TO_LAYER,
)
from aten.plena import FullModelProgram, assert_registers_are_free
from aten.state.isa_lowering import lower_mamba_trace_to_existing_isa

def _mamba_assembly_by_layer(
    config: MambaScheduleConfig, *, mlen: int, blen: int
) -> tuple[dict[int, str], object]:
    """Per-layer assembly for the Mamba mixers, keyed by layer id.

    ``mlen``/``blen`` must match the compiler the blocks are spliced into: the
    Mamba lowering bakes the tile geometry into its addresses, so lowering at 64
    and splicing into a 128-wide compiler would read the wrong rows.
    """
    trace = Nemotron3MambaScheduler(config).build()
    program = lower_mamba_trace_to_existing_isa(trace, mlen=mlen, blen=blen, vlen=mlen)
    chunks: dict[int, list[str]] = defaultdict(list)
    for event in program.events:
        memory = event.memory
        if memory is None:
            # Control events (fences) carry no layer; they belong to the layer
            # whose block they close, which is the one most recently appended.
            if chunks:
                chunks[max(chunks)].append(event.assembly)
            continue
        chunks[memory.layer_id].append(event.assembly)
    return {layer_id: "".join(parts) for layer_id, parts in chunks.items()}, program


def build_nemotron3_program(
    config: MambaScheduleConfig,
    *,
    architecture: Nemotron3Architecture | None = None,
    context_length: int | None = None,
    mlen: int | None = None,
    blen: int = 4,
) -> FullModelProgram:
    """Emit every layer of Nemotron 3 into one register-safe program."""
    from aten.plena import PlenaCompiler

    if config.phase != SchedulePhase.DECODE:
        raise ValueError("full-model lowering currently emits the decode program")
    arch = architecture or Nemotron3Architecture()
    # One tile geometry has to serve both halves, and their constraints do not
    # intersect: the Mamba projection requires mlen to divide its 10,304-wide
    # output (so mlen <= 64 in practice), while `_flash_attention_gqa_fused`
    # derives its broadcast factor as `mlen // h_qkv` and therefore needs mlen to
    # be a multiple of the 128-wide head. 64 is not a multiple of 128, so the
    # fused GQA path cannot be used here; attention is emitted through the MHA
    # path per KV group instead. Fusing GQA is an optimisation, not a
    # correctness requirement, and unblocking it needs the projection lowering to
    # accept a padded output width.
    if mlen is None:
        mlen = 64
    if arch.hidden_size % mlen or (arch.attention_heads * arch.attention_head_dim) % mlen:
        raise ValueError(f"mlen {mlen} does not tile the Nemotron projections")
    # The staged K/V tiles are mlen rows tall, so the modelled context has to fit
    # inside one tile per batch element.
    if context_length is None:
        context_length = mlen // config.batch_size
    if config.batch_size * context_length > mlen:
        raise ValueError(
            f"batch_size*context_length ({config.batch_size}*{context_length}) exceeds "
            f"the {mlen}-row K/V tile this program stages"
        )
    mamba_assembly, mamba_program = _mamba_assembly_by_layer(config, mlen=mlen, blen=blen)

    compiler = PlenaCompiler(mlen=mlen, blen=blen)
    compiler.hlen = 16

    hidden = arch.hidden_size
    stage_counts: dict[str, int] = defaultdict(int)

    def measure(stage: str, emit) -> None:
        before = len(compiler.generated_code)
        emit()
        after = compiler.generated_code
        added = after[before:]
        stage_counts[stage] += sum(
            1
            for line in added.splitlines()
            if line.strip() and not line.strip().startswith(";")
        )

    hidden_input = compiler.input(
        "hidden", shape=(mlen, hidden), physical_shape=(mlen, hidden), prestaged_vram_addr=0
    )
    hidden_state = compiler.load_batch(hidden_input, name="hidden")

    for layer_id, symbol in enumerate(arch.pattern):
        layer_type = SYMBOL_TO_LAYER[symbol]
        compiler.emit_comment(f"; ==== layer {layer_id} ({layer_type.value}) ====")

        # In place, matching the reference block order: norm feeds the mixer and
        # the residual is added back to the unnormalised activation.
        measure("input_rms_norm", lambda: compiler.rms_norm(hidden_state))

        if layer_type is HybridLayerType.MAMBA:
            assembly = mamba_assembly.get(layer_id)
            if assembly is None:
                raise ValueError(f"no Mamba assembly was lowered for layer {layer_id}")
            # Spliced verbatim: the block reloads every address it uses and holds
            # nothing across its own boundary.
            measure("mamba_mixer", lambda a=assembly: compiler.emit(a))
        elif layer_type is HybridLayerType.ATTENTION:
            # One call per query head. The projection emitter matches VRAM row
            # blocks against MRAM row blocks, so Q and K have to be head-width
            # (128), not the concatenated 4096-wide projection -- and the fused
            # GQA path is unavailable here because its broadcast factor needs mlen
            # to be a multiple of that head width, which the 10,304-wide Mamba
            # projection forbids.
            head = arch.attention_head_dim
            for q_head in range(arch.attention_heads):
                kv_group = q_head * arch.kv_heads // arch.attention_heads
                q = compiler.load_batch(
                    compiler.input(
                        f"q_{layer_id}_{q_head}",
                        shape=(mlen, head),
                        physical_shape=(mlen, head),
                        prestaged_vram_addr=0,
                    ),
                    name=f"q_{layer_id}_{q_head}",
                )
                k = compiler.input(
                    f"k_{layer_id}_{kv_group}_{q_head}",
                    shape=(mlen, head),
                    physical_shape=(mlen, head),
                )
                v = compiler.input(
                    f"v_{layer_id}_{kv_group}_{q_head}",
                    shape=(mlen, head),
                    physical_shape=(mlen, head),
                )
                measure(
                    "attention",
                    lambda q=q, k=k, v=v: compiler.flash_attention(
                        q,
                        k,
                        v,
                        scale=head**-0.5,
                        batch_size=config.batch_size,
                        seq_len=mlen,
                        kv_seq_len=context_length,
                    ),
                )
        else:
            logits = compiler.load_batch(
                compiler.input(
                    f"router_logits_{layer_id}",
                    shape=(mlen, arch.routed_experts),
                    physical_shape=(mlen, arch.routed_experts),
                    prestaged_vram_addr=0,
                ),
                name=f"router_logits_{layer_id}",
            )
            measure(
                "moe_router_top_k",
                lambda logits=logits: compiler.moe_router_select_v0(
                    logits,
                    token_idx=0,
                    weights_fp_base=0,
                    indices_int_base=0,
                    num_experts=arch.routed_experts,
                    top_k=arch.experts_per_token,
                    policy_name="gpt_oss",
                ),
            )

        assert_registers_are_free(compiler, f"layer {layer_id} ({layer_type.value})")

    assembly = compiler.compile() if hasattr(compiler, "compile") else compiler.generated_code
    instruction_count = sum(
        1
        for line in assembly.splitlines()
        if line.strip() and not line.strip().startswith(";")
    )
    layer_types = [SYMBOL_TO_LAYER[symbol].value for symbol in arch.pattern]
    return FullModelProgram(
        model="nemotron3",
        phase=config.phase.value,
        layer_counts={
            layer_type.value: layer_types.count(layer_type.value)
            for layer_type in HybridLayerType
        },
        assembly=assembly,
        instruction_count=instruction_count,
        descriptor_base=mamba_program.descriptor_base,
        descriptor_image=mamba_program.descriptor_image,
        layout_descriptor_base=mamba_program.layout_descriptor_base,
        layout_descriptor_image=mamba_program.layout_descriptor_image,
        stage_instruction_counts=dict(stage_counts),
    )


# Preserve the former instruction-coverage builder for comparison.  The public
# entry point now uses the connected producer-consumer implementation.
build_nemotron3_instruction_coverage_program = build_nemotron3_program


def build_nemotron3_program(
    config: MambaScheduleConfig,
    *,
    architecture: Nemotron3Architecture | None = None,
    context_length: int | None = None,
    mlen: int | None = None,
    blen: int = 4,
) -> FullModelProgram:
    from aten.nemotron3.connected_program import build_connected_nemotron3_program

    return build_connected_nemotron3_program(
        config,
        architecture=architecture,
        context_length=context_length,
        mlen=mlen,
        blen=blen,
    )
