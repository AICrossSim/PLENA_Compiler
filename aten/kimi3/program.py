"""Emit one physical program covering all 93 Kimi K3 layers, MLA included.

MLA is the one mixer with no existing emitter. It is not standard attention with
different numbers: the query and key/value paths both go through a low-rank
bottleneck, and only the compressed latent is cached.

    hidden 7168 ─ q_a ─→ 1536 ─ RMSNorm ─ q_b ─→ 96 x (128 nope + 64 rope)
                └ kv_a ─→ 512 latent + 64 shared rope ─ RMSNorm ─ kv_b ─→ 96 x (128 nope + 128 v)

The attention core is then ordinary: scores come from a 192-wide key (128 nope
concatenated with 64 rope) and values are 128 wide. So MLA needs no new attention
kernel -- it needs the low-rank chain in front of one, which is why this is
emitted from `linear`/`rms_norm`/`rope` rather than a bespoke mixin.

Tile geometry works out here in a way it did not for Nemotron's fused GQA path:
every MLA width (7168, 1536, 512, 576, 192, 128, 18432, 24576) is a multiple of
64, so MLA shares one tile size with the KDA mixers and the seam needs no
compromise.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from aten.kda.scheduler import (
    KIMI_K3_KDA_LAYERS,
    KdaScheduleConfig,
    KimiK3KdaScheduler,
)
from aten.kimi3.scheduler import KimiK3Architecture
from aten.mamba.scheduler import SchedulePhase
from aten.plena import FullModelProgram, assert_registers_are_free
from aten.state.isa_lowering import lower_kda_trace_to_existing_isa


def mla_layer_ids(num_layers: int) -> tuple[int, ...]:
    """Every layer the KDA schedule does not claim.

    Derived as the complement rather than from its own "every fourth layer"
    formula: ``KIMI_K3_KDA_LAYERS`` enumerates 1..92, so the 93rd layer is not a
    multiple of four and a separate formula silently drops it -- 23 MLA layers
    instead of 24, and a KDA lookup for a layer the schedule never emitted.
    """
    kda = set(KIMI_K3_KDA_LAYERS)
    return tuple(layer for layer in range(num_layers) if layer not in kda)


@dataclass(frozen=True)
class MlaWidths:
    """Every width one MLA layer projects through."""

    hidden: int
    q_lora: int
    kv_lora: int
    qk_nope: int
    qk_rope: int
    v_head: int
    heads: int

    @property
    def qk_head(self) -> int:
        return self.qk_nope + self.qk_rope

    @property
    def q_b_out(self) -> int:
        return self.heads * self.qk_head

    @property
    def kv_a_out(self) -> int:
        return self.kv_lora + self.qk_rope

    @property
    def kv_b_out(self) -> int:
        return self.heads * (self.qk_nope + self.v_head)

    @property
    def attn_out(self) -> int:
        return self.heads * self.v_head

    def unaligned(self, mlen: int) -> list[tuple[str, int]]:
        return [
            (name, value)
            for name, value in (
                ("hidden", self.hidden),
                ("q_lora", self.q_lora),
                ("kv_lora", self.kv_lora),
                ("kv_a_out", self.kv_a_out),
                ("qk_head", self.qk_head),
                ("v_head", self.v_head),
                ("q_b_out", self.q_b_out),
                ("kv_b_out", self.kv_b_out),
                ("attn_out", self.attn_out),
            )
            if value % mlen
        ]

    @classmethod
    def from_architecture(cls, arch: KimiK3Architecture) -> MlaWidths:
        return cls(
            hidden=arch.hidden_size,
            q_lora=arch.q_lora_rank,
            kv_lora=arch.kv_lora_rank,
            qk_nope=arch.qk_nope_head_dim,
            qk_rope=arch.qk_rope_head_dim,
            v_head=arch.v_head_dim,
            heads=arch.attention_heads,
        )


def _kda_assembly_by_layer(
    config: KdaScheduleConfig, *, mlen: int, blen: int
) -> tuple[dict[int, str], object]:
    trace = KimiK3KdaScheduler(config).build()
    program = lower_kda_trace_to_existing_isa(trace, mlen=mlen, blen=blen, vlen=mlen)
    chunks: dict[int, list[str]] = defaultdict(list)
    for event in program.events:
        memory = event.memory
        if memory is None:
            if chunks:
                chunks[max(chunks)].append(event.assembly)
            continue
        chunks[memory.layer_id].append(event.assembly)
    return {layer: "".join(parts) for layer, parts in chunks.items()}, program


def build_kimi_k3_program(
    config: KdaScheduleConfig,
    *,
    architecture: KimiK3Architecture | None = None,
    mlen: int = 64,
    blen: int = 4,
    context_length: int | None = None,
    heads: int | None = None,
) -> FullModelProgram:
    """Emit every Kimi K3 layer -- KDA, MLA, LatentMoE, dense FFN -- as one program."""
    from aten.plena import PlenaCompiler

    if config.phase != SchedulePhase.DECODE:
        raise ValueError("full-model lowering currently emits the decode program")
    arch = architecture or KimiK3Architecture()
    widths = MlaWidths.from_architecture(arch)
    if heads is not None:
        widths = MlaWidths(**{**widths.__dict__, "heads": heads})
    unaligned = widths.unaligned(mlen)
    if unaligned:
        raise ValueError(
            f"mlen {mlen} does not tile these MLA widths: {unaligned}. Unlike the "
            "fused GQA path, MLA has no width that forces a larger tile, so this "
            "means the tile was chosen wrong rather than the shapes conflicting."
        )
    if context_length is None:
        context_length = mlen // config.batch_size
    if config.batch_size * context_length > mlen:
        raise ValueError(
            f"batch_size*context_length exceeds the {mlen}-row K/V tile staged here"
        )

    kda_assembly, kda_program = _kda_assembly_by_layer(config, mlen=mlen, blen=blen)
    mla_layers = set(mla_layer_ids(arch.num_layers))

    compiler = PlenaCompiler(mlen=mlen, blen=blen)
    compiler.hlen = 16
    stage_counts: dict[str, int] = defaultdict(int)

    def measure(stage: str, emit) -> None:
        before = len(compiler.generated_code)
        emit()
        added = compiler.generated_code[before:]
        stage_counts[stage] += sum(
            1
            for line in added.splitlines()
            if line.strip() and not line.strip().startswith(";")
        )

    def staged(name: str, width: int):
        return compiler.load_batch(
            compiler.input(
                name,
                shape=(mlen, width),
                physical_shape=(mlen, width),
                prestaged_vram_addr=0,
            ),
            name=name,
        )

    def weight(name: str, rows: int, cols: int):
        return compiler.input(name, shape=(rows, cols), physical_shape=(rows, cols))

    hidden_state = staged("hidden", widths.hidden)

    def emit_mla(layer_id: int) -> None:
        w = widths
        # Query side: hidden -> low-rank -> per-head nope+rope.
        measure(
            "mla_q_low_rank_projection",
            lambda: compiler.linear(
                hidden_state, weight(f"w_q_a_{layer_id}", w.hidden, w.q_lora)
            ),
        )
        q_latent = staged(f"q_latent_{layer_id}", w.q_lora)
        measure("mla_q_latent_norm", lambda: compiler.rms_norm(q_latent))
        measure(
            "mla_q_head_projection",
            lambda: compiler.linear(
                q_latent, weight(f"w_q_b_{layer_id}", w.q_lora, w.q_b_out)
            ),
        )
        # Key/value side: only the compressed latent is cached, which is the whole
        # point of MLA and why kv_a is projected before the norm.
        measure(
            "mla_kv_low_rank_projection",
            lambda: compiler.linear(
                hidden_state, weight(f"w_kv_a_{layer_id}", w.hidden, w.kv_a_out)
            ),
        )
        kv_latent = staged(f"kv_latent_{layer_id}", w.kv_lora)
        measure("mla_kv_latent_norm", lambda: compiler.rms_norm(kv_latent))
        measure(
            "mla_kv_head_projection",
            lambda: compiler.linear(
                kv_latent, weight(f"w_kv_b_{layer_id}", w.kv_lora, w.kv_b_out)
            ),
        )
        # RoPE rides on the 64-wide tail of the query heads and on the single
        # shared key rope vector.
        for tag, width in (("q", w.heads * w.qk_rope), ("k", w.qk_rope)):
            target = staged(f"{tag}_rope_{layer_id}", width)
            rotated = staged(f"{tag}_rope_rot_{layer_id}", width)
            cos = staged(f"{tag}_cos_{layer_id}", width)
            sin = staged(f"{tag}_sin_{layer_id}", width)
            measure(
                "mla_rope",
                lambda t=target, r=rotated, c=cos, s=sin: compiler.rope(t, r, c, s),
            )
        # Attention core: a 192-wide key and a 128-wide value, per head.
        for head in range(w.heads):
            q = staged(f"mla_q_{layer_id}_{head}", w.qk_head)
            k = compiler.input(
                f"mla_k_{layer_id}_{head}",
                shape=(mlen, w.qk_head),
                physical_shape=(mlen, w.qk_head),
            )
            v = compiler.input(
                f"mla_v_{layer_id}_{head}",
                shape=(mlen, w.qk_head),
                physical_shape=(mlen, w.qk_head),
            )
            measure(
                "mla_rope_kv_cache_attention",
                lambda q=q, k=k, v=v: compiler.flash_attention(
                    q,
                    k,
                    v,
                    scale=w.qk_head**-0.5,
                    batch_size=config.batch_size,
                    seq_len=mlen,
                    kv_seq_len=context_length,
                ),
            )
        attn = staged(f"mla_attn_{layer_id}", w.attn_out)
        measure(
            "mla_out_projection",
            lambda: compiler.linear(
                attn, weight(f"w_o_{layer_id}", w.attn_out, w.hidden)
            ),
        )

    for layer_id in range(arch.num_layers):
        is_mla = layer_id in mla_layers
        kind = "mla" if is_mla else "kda"
        compiler.emit_comment(f"; ==== layer {layer_id} ({kind}) ====")
        measure("input_rms_norm", lambda: compiler.rms_norm(hidden_state))

        if is_mla:
            emit_mla(layer_id)
        else:
            assembly = kda_assembly.get(layer_id)
            if assembly is None:
                raise ValueError(f"no KDA assembly was lowered for layer {layer_id}")
            measure("kda_mixer", lambda a=assembly: compiler.emit(a))

        # Layer 0 is the dense FFN; every other layer routes.
        if layer_id == 0:
            measure(
                "dense_situ_ffn",
                lambda: compiler.linear(
                    hidden_state,
                    weight("w_dense_ffn", widths.hidden, arch.dense_intermediate_size),
                ),
            )
        else:
            logits = staged(f"router_logits_{layer_id}", arch.num_experts)
            measure(
                "latent_moe_router_top_k",
                lambda logits=logits: compiler.moe_router_select_v0(
                    logits,
                    token_idx=0,
                    weights_fp_base=0,
                    indices_int_base=0,
                    num_experts=arch.num_experts,
                    top_k=arch.experts_per_token,
                    policy_name="gpt_oss",
                ),
            )

        assert_registers_are_free(compiler, f"layer {layer_id} ({kind})")

    assembly = compiler.compile()
    instruction_count = sum(
        1
        for line in assembly.splitlines()
        if line.strip() and not line.strip().startswith(";")
    )
    return FullModelProgram(
        model="kimi_k3",
        phase=config.phase.value,
        layer_counts={
            "kda": arch.num_layers - len(mla_layers),
            "mla": len(mla_layers),
            "latent_moe": arch.num_layers - 1,
            "dense_ffn": 1,
        },
        assembly=assembly,
        instruction_count=instruction_count,
        descriptor_base=kda_program.descriptor_base,
        descriptor_image=kda_program.descriptor_image,
        layout_descriptor_base=kda_program.layout_descriptor_base,
        layout_descriptor_image=kda_program.layout_descriptor_image,
        stage_instruction_counts=dict(stage_counts),
    )


# Keep the old instruction-coverage builder available for archaeology, but make
# the public entry point use the connected dataflow implementation.
build_kimi_k3_instruction_coverage_program = build_kimi_k3_program


def build_kimi_k3_program(
    config: KdaScheduleConfig,
    *,
    architecture: KimiK3Architecture | None = None,
    mlen: int = 64,
    blen: int = 4,
    context_length: int | None = None,
    heads: int | None = None,
    allow_unbounded_static_expansion: bool = False,
) -> FullModelProgram:
    from aten.kimi3.connected_program import build_connected_kimi_k3_program

    return build_connected_kimi_k3_program(
        config,
        architecture=architecture,
        mlen=mlen,
        blen=blen,
        context_length=context_length,
        heads=heads,
        allow_unbounded_static_expansion=allow_unbounded_static_expansion,
    )
