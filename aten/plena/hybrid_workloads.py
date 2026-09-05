"""Pinned real-shape workload contracts for hybrid L-Compute evaluation.

This module is the model-specific edge of the design.  It verifies the official
layer schedules and turns selected consumer topologies into generic
``LayoutRequest`` objects.  Neither the ISA nor ``AffineLayoutPlanner`` imports
this module, which keeps model names out of the architectural mechanism.

The requests describe real dimensions and repeated packet topology.  They are
performance contracts, not claims that a full checkpoint has been numerically
executed by the transactional emulator.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from compiler.aten.plena.affine_layout import BankGeometry, LogicalCoord
from compiler.aten.plena.layout_planner import AccessPacket, LayoutRequest


NEMOTRON_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
NEMOTRON_REVISION = "ce1b118ae66ec705d02c241525192832eb045fd3"
KIMI_MODEL = "moonshotai/Kimi-K3"
KIMI_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"


@dataclass(frozen=True)
class HybridLayer:
    number: int
    mixer: str | None
    ffn: str | None


@dataclass(frozen=True)
class HybridWorkloadManifest:
    name: str
    source_model: str
    source_revision: str
    hidden_size: int
    vocab_size: int
    layers: tuple[HybridLayer, ...]
    dimensions: dict[str, int]
    precisions: dict[str, str]

    def layer_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for layer in self.layers:
            for kind in (layer.mixer, layer.ffn):
                if kind is not None:
                    counts[kind] = counts.get(kind, 0) + 1
        return counts

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_model": self.source_model,
            "source_revision": self.source_revision,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "layers": [asdict(layer) for layer in self.layers],
            "layer_counts": self.layer_counts(),
            "dimensions": self.dimensions,
            "precisions": self.precisions,
            "execution_claim": "official_shape_performance_contract",
        }


def _load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def nemotron3_manifest(config_path: str | Path) -> HybridWorkloadManifest:
    config = _load_config(config_path)
    pattern = str(config["hybrid_override_pattern"])
    if len(pattern) != int(config["num_hidden_layers"]):
        raise ValueError("Nemotron layer pattern length does not match num_hidden_layers")
    mapping = {
        "M": ("mamba", None),
        "*": ("gqa", None),
        "E": (None, "moe"),
        "-": (None, "mlp"),
    }
    unknown = set(pattern) - set(mapping)
    if unknown:
        raise ValueError(f"unknown Nemotron layer markers: {sorted(unknown)}")
    layers = tuple(
        HybridLayer(number=index + 1, mixer=mapping[marker][0], ffn=mapping[marker][1])
        for index, marker in enumerate(pattern)
    )
    manifest = HybridWorkloadManifest(
        name="nemotron3_nano_30b_a3b",
        source_model=NEMOTRON_MODEL,
        source_revision=NEMOTRON_REVISION,
        hidden_size=int(config["hidden_size"]),
        vocab_size=int(config["vocab_size"]),
        layers=layers,
        dimensions={
            "mamba_heads": int(config["mamba_num_heads"]),
            "mamba_head_dim": int(config["mamba_head_dim"]),
            "mamba_state_dim": int(config["ssm_state_size"]),
            "mamba_groups": int(config["n_groups"]),
            "mamba_chunk": int(config["chunk_size"]),
            "mamba_projection_width": (
                2 * int(config["mamba_num_heads"]) * int(config["mamba_head_dim"])
                + 2 * int(config["n_groups"]) * int(config["ssm_state_size"])
                + int(config["mamba_num_heads"])
            ),
            "gqa_heads": int(config["num_attention_heads"]),
            "gqa_kv_heads": int(config["num_key_value_heads"]),
            "gqa_head_dim": int(config["head_dim"]),
            "experts": int(config["n_routed_experts"]),
            "experts_per_token": int(config["num_experts_per_tok"]),
            "shared_experts": int(config["n_shared_experts"]),
        },
        precisions={
            "activation": "bf16",
            "recurrent_state": str(config["mamba_ssm_cache_dtype"]).lower(),
            "weight_checkpoint": "nvfp4_mixed_exclusions",
        },
    )
    if manifest.layer_counts() != {"mamba": 23, "moe": 23, "gqa": 6}:
        raise ValueError(f"unexpected Nemotron layer census: {manifest.layer_counts()}")
    if manifest.dimensions["mamba_projection_width"] != 10_304:
        raise ValueError("Nemotron projection width must be 10,304")
    return manifest


def kimi_k3_manifest(config_path: str | Path) -> HybridWorkloadManifest:
    config = _load_config(config_path)
    linear = dict(config["linear_attn_config"])
    num_layers = int(config["num_hidden_layers"])
    kda = {int(layer) for layer in linear["kda_layers"]}
    mla = {int(layer) for layer in linear["full_attn_layers"]}
    expected = set(range(1, num_layers + 1))
    if kda & mla or kda | mla != expected:
        raise ValueError("Kimi KDA/MLA schedules must be a disjoint complete partition")
    layers = tuple(
        HybridLayer(
            number=number,
            mixer="kda" if number in kda else "mla",
            ffn="dense_ffn" if number == 1 else "latent_moe",
        )
        for number in range(1, num_layers + 1)
    )
    manifest = HybridWorkloadManifest(
        name="kimi_k3_text",
        source_model=str(config["source_model"]),
        source_revision=str(config["source_revision"]),
        hidden_size=int(config["hidden_size"]),
        vocab_size=int(config["vocab_size"]),
        layers=layers,
        dimensions={
            "kda_heads": int(linear["num_heads"]),
            "kda_key_dim": int(linear["head_dim"]),
            "kda_value_dim": int(linear["head_dim"]),
            "kda_conv_kernel": int(linear["short_conv_kernel_size"]),
            "q_lora_rank": int(config["q_lora_rank"]),
            "kv_lora_rank": int(config["kv_lora_rank"]),
            "qk_nope_head_dim": int(config["qk_nope_head_dim"]),
            "qk_rope_head_dim": int(config["qk_rope_head_dim"]),
            "v_head_dim": int(config["v_head_dim"]),
            "mla_cache_elements_per_token": int(config["kv_lora_rank"])
            + int(config["qk_rope_head_dim"]),
            "experts": int(config["num_experts"]),
            "experts_per_token": int(config["num_experts_per_token"]),
            "shared_experts": int(config["num_shared_experts"]),
        },
        precisions={
            "activation": "bf16",
            "recurrent_state": "fp32",
            "conv_state": "bf16",
            "weight_checkpoint": "mxfp4_mixed_exclusions",
        },
    )
    expected_counts = {"kda": 69, "mla": 24, "dense_ffn": 1, "latent_moe": 92}
    if manifest.layer_counts() != expected_counts:
        raise ValueError(f"unexpected Kimi layer census: {manifest.layer_counts()}")
    if manifest.dimensions["mla_cache_elements_per_token"] != 576:
        raise ValueError("Kimi MLA cache must remain compressed to 512+64 elements/token")
    return manifest


def _full_rows_for_ragged_fields(
    *, groups: int, field_widths: tuple[int, ...], field_majors: tuple[int, ...]
) -> tuple[AccessPacket, ...]:
    if len(field_widths) != len(field_majors):
        raise ValueError("field widths and major counts must have equal length")
    # Bank service is translation invariant in group/major: changing either
    # rotates the selected bank but does not change the number of cycles.  Keep
    # one representative row and carry the real multiplicity in `repeats` so
    # official 100k-element tensors do not make coefficient DSE quadratic.
    return tuple(
        AccessPacket(
            f"producer_f{field}",
            tuple(LogicalCoord(0, field, 0, minor) for minor in range(width)),
            repeats=groups * major_count,
        )
        for field, (width, major_count) in enumerate(zip(field_widths, field_majors, strict=True))
    )


def nemotron_projection_layout_request(
    manifest: HybridWorkloadManifest,
    geometry: BankGeometry,
    *,
    parallel_heads: int = 8,
) -> LayoutRequest:
    """One real-shape grouped Mamba consumer packet.

    The 64 existing lanes are partitioned into 8 heads x 4 x-values, 8 B
    scalars, 8 C scalars, and 8 dt scalars (56 live values).  This is a candidate
    segmented consumer topology; the manifest never claims current RTL can
    execute it in one cycle.
    """

    dims = manifest.dimensions
    heads = dims["mamba_heads"]
    groups = dims["mamba_groups"]
    heads_per_group = heads // groups
    if parallel_heads > heads_per_group or parallel_heads <= 0:
        raise ValueError("parallel_heads must fit one Mamba group")
    x_per_head = geometry.bank_width
    fields = ("x", "B", "C", "dt")
    coords = []
    for head in range(parallel_heads):
        coords.extend(LogicalCoord(0, 0, head, lane) for lane in range(x_per_head))
        coords.append(LogicalCoord(0, 3, head, 0))
    for state_lane in range(parallel_heads):
        coords.append(LogicalCoord(0, 1, state_lane, 0))
        coords.append(LogicalCoord(0, 2, state_lane, 0))
    repeats = (
        groups
        * math.ceil(heads_per_group / parallel_heads)
        * math.ceil(dims["mamba_state_dim"] / parallel_heads)
        * math.ceil(dims["mamba_head_dim"] / x_per_head)
    )
    majors = max(heads_per_group, dims["mamba_state_dim"])
    minors = max(dims["mamba_head_dim"], dims["mamba_state_dim"])
    return LayoutRequest(
        name="nemotron_mamba_projection_packet",
        groups=groups,
        fields=len(fields),
        majors=majors,
        minors=minors,
        producer_packets=_full_rows_for_ragged_fields(
            groups=groups,
            field_widths=(
                dims["mamba_head_dim"],
                dims["mamba_state_dim"],
                dims["mamba_state_dim"],
                1,
            ),
            field_majors=(heads_per_group, 1, 1, heads_per_group),
        ),
        consumer_packets=(AccessPacket("mamba_group_packet", tuple(coords), repeats=repeats),),
        baseline_reorder_cycles=repeats,
        consumer_major_supported=True,
        lane_restore_cycles_per_packet=1,
    )


def kimi_k3_projection_layout_request(
    manifest: HybridWorkloadManifest,
    geometry: BankGeometry,
    *,
    parallel_heads: int = 4,
) -> LayoutRequest:
    """Official KDA independent-projection packet, never a packed-QKV view."""

    dims = manifest.dimensions
    heads = dims["kda_heads"]
    key_dim = dims["kda_key_dim"]
    if not 0 < parallel_heads <= heads:
        raise ValueError("parallel_heads must fit KDA heads")
    # Official projection order: q, k, v, decay_low_rank, decay_g, beta,
    # output_gate, output.  Only q/k/decay_g/beta are co-consumed by this packet.
    q, k, _v, _f_a, decay_g, beta, _gate, _out = range(8)
    coords = []
    for head in range(parallel_heads):
        for field in (q, k, decay_g):
            coords.extend(
                LogicalCoord(0, field, head, lane) for lane in range(geometry.bank_width)
            )
        coords.append(LogicalCoord(0, beta, head, 0))
    repeats = math.ceil(heads / parallel_heads) * math.ceil(key_dim / geometry.bank_width)
    producer = _full_rows_for_ragged_fields(
        groups=1,
        field_widths=(key_dim, key_dim, key_dim, key_dim, key_dim, 1, key_dim, key_dim),
        field_majors=(heads, heads, heads, 1, heads, heads, heads, heads),
    )
    return LayoutRequest(
        name="kimi_k3_kda_projection_packet",
        groups=1,
        fields=8,
        majors=heads,
        minors=key_dim,
        producer_packets=producer,
        consumer_packets=(AccessPacket("kda_q_k_decay_beta", tuple(coords), repeats=repeats),),
        baseline_reorder_cycles=repeats,
        consumer_major_supported=True,
        lane_restore_cycles_per_packet=1,
    )


def state_multirow_layout_request(
    *,
    name: str,
    groups: int,
    rows_per_group: int,
    row_elements: int,
    geometry: BankGeometry,
    parallel_rows: int,
    repeats: int,
) -> LayoutRequest:
    if not 0 < parallel_rows <= rows_per_group:
        raise ValueError("parallel_rows must fit the state-row extent")
    packet = tuple(
        LogicalCoord(0, 0, major, minor)
        for major in range(parallel_rows)
        for minor in range(geometry.bank_width)
    )
    # Every producer row has the same bank-service multiplicity: changing the
    # group or major only translates/rotates the bank IDs. Represent the exact
    # real-shape count as one row with a repeat count so coefficient DSE does
    # not re-enumerate hundreds of thousands of equivalent scalar placements.
    producer = AccessPacket(
        "state_producer_row",
        tuple(LogicalCoord(0, 0, 0, minor) for minor in range(row_elements)),
        repeats=groups * rows_per_group,
    )
    return LayoutRequest(
        name=name,
        groups=groups,
        fields=1,
        majors=rows_per_group,
        minors=row_elements,
        producer_packets=(producer,),
        consumer_packets=(AccessPacket("state_multirow", packet, repeats=repeats),),
        baseline_reorder_cycles=0,
        consumer_major_supported=False,
        lane_restore_cycles_per_packet=1,
    )


__all__ = [
    "HybridLayer",
    "HybridWorkloadManifest",
    "KIMI_MODEL",
    "KIMI_REVISION",
    "NEMOTRON_MODEL",
    "NEMOTRON_REVISION",
    "kimi_k3_manifest",
    "kimi_k3_projection_layout_request",
    "nemotron3_manifest",
    "nemotron_projection_layout_request",
    "state_multirow_layout_request",
]
