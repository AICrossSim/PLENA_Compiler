"""Bind the physical ``L_TILE`` recurrence to official hybrid layer schedules.

The single-layer lowering in :mod:`matrix_recurrence_lowering` is executable,
but by itself it does not prove that a full hybrid schedule selects it at every
recurrent layer.  This module closes that compiler integration gap.  It walks
the pinned Nemotron 3 and Kimi K3 layer manifests, assigns disjoint HBM regions
to every recurrent layer, and emits the complete physical recurrence whenever
the layer is Mamba-2 or KDA.

Ordinary Attention/MLA/MoE stages are deliberately represented by schedule
markers here.  Their cycle models and existing PLENA lowerings remain outside
this recurrence-specific program.  Consequently this is evidence that every
official recurrent layer emits ``L_TILE`` in the right layer order; it is not a
claim that checkpoint weights have executed numerically from layer 1 to the
final layer.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from compiler.aten.plena.hybrid_workloads import (
    HybridWorkloadManifest,
    kimi_k3_manifest,
    nemotron3_manifest,
)
from compiler.aten.plena.instruction_stream import (
    dynamic_count,
    opcode_census,
    static_count,
)
from compiler.aten.plena.matrix_recurrence_lowering import (
    KIMI_KDA,
    NEMOTRON_MAMBA,
    MatrixRecurrenceSpec,
    MatrixSramPoint,
    RecurrenceLayout,
    build_recurrence_field_manifest,
    build_recurrence_working_set,
    lower_matrix_recurrence,
    lowering_metrics,
)


@dataclass(frozen=True)
class HybridLTileLayerRecord:
    """Physical HBM ownership and instruction census for one recurrent layer."""

    layer: int
    mixer: str
    state_hbm_begin: int
    state_hbm_end: int
    field_hbm_begin: int
    field_hbm_end: int
    static_instructions: int
    dynamic_issued_instructions: int
    l_tile_exec_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {
            "state_hbm_bytes": self.state_hbm_end - self.state_hbm_begin,
            "field_hbm_bytes": self.field_hbm_end - self.field_hbm_begin,
        }


@dataclass(frozen=True)
class HybridLTileSchedule:
    """One official hybrid layer order with executable recurrent regions."""

    model: str
    source_model: str
    source_revision: str
    layout: RecurrenceLayout
    layer_count: int
    layer_counts: dict[str, int]
    recurrent_mixer: str
    recurrent_layers: tuple[int, ...]
    state_hbm_begin: int
    state_hbm_end: int
    field_hbm_begin: int
    field_hbm_end: int
    records: tuple[HybridLTileLayerRecord, ...]
    assembly: str

    def to_report(self) -> dict[str, object]:
        census = opcode_census(self.assembly)
        return {
            "model": self.model,
            "source_model": self.source_model,
            "source_revision": self.source_revision,
            "layout": self.layout,
            "layer_count": self.layer_count,
            "layer_counts": self.layer_counts,
            "recurrent_mixer": self.recurrent_mixer,
            "recurrent_layer_count": len(self.recurrent_layers),
            "recurrent_layers": list(self.recurrent_layers),
            "all_recurrent_layers_emit_l_tile": (
                len(self.records) == len(self.recurrent_layers)
                and all(record.l_tile_exec_count > 0 for record in self.records)
            ),
            "state_hbm_arena": {
                "begin": self.state_hbm_begin,
                "end": self.state_hbm_end,
                "bytes": self.state_hbm_end - self.state_hbm_begin,
                "precision": "bf16",
                "allocation": "compiler_static_disjoint_per_layer",
            },
            "field_hbm_arena": {
                "begin": self.field_hbm_begin,
                "end": self.field_hbm_end,
                "bytes": self.field_hbm_end - self.field_hbm_begin,
                "allocation": "compiler_static_disjoint_per_layer",
            },
            "static_instructions": static_count(self.assembly),
            "dynamic_issued_instructions": dynamic_count(self.assembly),
            "l_tile_exec_count": census.get("L_TILE_EXEC", 0),
            "opcode_census": census,
            "assembly_sha256": hashlib.sha256(self.assembly.encode()).hexdigest(),
            "layers": [record.to_dict() for record in self.records],
            "architectural_boundary": {
                "recurrent_layer_lowering": "executable PLENA instructions",
                "ordinary_layer_markers": "schedule only; existing lowerings are not duplicated",
                "weights": "symbolic HBM addresses",
                "full_model_numerical_rust_execution": False,
                "cache": False,
                "private_state_sram": False,
                "runtime_scheduler": False,
                "new_mac_array": False,
            },
        }


def _align(value: int, alignment: int = 64) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _assert_disjoint(records: tuple[HybridLTileLayerRecord, ...]) -> None:
    ranges = sorted(
        [
            (record.state_hbm_begin, record.state_hbm_end, record.layer, "state")
            for record in records
        ]
        + [
            (record.field_hbm_begin, record.field_hbm_end, record.layer, "field")
            for record in records
        ]
    )
    for left, right in zip(ranges, ranges[1:], strict=False):
        if left[1] > right[0]:
            raise ValueError(
                "hybrid L_TILE HBM regions overlap: "
                f"layer {left[2]} {left[3]} [{left[0]}, {left[1]}) and "
                f"layer {right[2]} {right[3]} [{right[0]}, {right[1]})"
            )


def lower_hybrid_l_tile_schedule(
    manifest: HybridWorkloadManifest,
    *,
    recurrent_mixer: str,
    spec: MatrixRecurrenceSpec,
    layout: RecurrenceLayout | str,
    point: MatrixSramPoint | None = None,
    hbm_base: int = 0,
    hbm_address_register: int = 0,
) -> HybridLTileSchedule:
    """Emit every recurrent layer in an official hybrid schedule.

    State and prepared-field ranges are unique per layer.  Matrix SRAM itself
    is reused because layers execute sequentially and the compiler explicitly
    prefetches/stores each state group.
    """

    layout = RecurrenceLayout(layout)
    point = point or MatrixSramPoint()
    if hbm_base < 0 or hbm_base % 64:
        raise ValueError("hbm_base must be a non-negative 64-byte address")
    recurrent_layers = tuple(
        layer.number for layer in manifest.layers if layer.mixer == recurrent_mixer
    )
    if not recurrent_layers:
        raise ValueError(f"{manifest.name} contains no {recurrent_mixer} layers")

    state_begin = hbm_base
    state_end = state_begin + len(recurrent_layers) * spec.state_bytes_per_layer
    field_begin = _align(state_end)
    field_cursor = field_begin
    state_index = {layer: index for index, layer in enumerate(recurrent_layers)}
    working_set = build_recurrence_working_set(spec, layout=layout, point=point)
    lines = [
        f"; @hybrid_schedule_begin model={manifest.name} layout={layout}",
        f"; @official_layers={len(manifest.layers)}",
        f"; @recurrent_mixer={recurrent_mixer}",
        "; @ordinary_layers=markers_only_existing_plena_path",
        "; @state_storage=compiler_managed_matrix_sram_no_cache",
    ]
    records: list[HybridLTileLayerRecord] = []

    for layer in manifest.layers:
        mixer = layer.mixer or "none"
        ffn = layer.ffn or "none"
        lines.append(
            f"; @hybrid_layer_begin layer={layer.number} mixer={mixer} ffn={ffn}"
        )
        if layer.mixer == recurrent_mixer:
            index = state_index[layer.number]
            layer_state_begin = state_begin + index * spec.state_bytes_per_layer
            layer_state_end = layer_state_begin + spec.state_bytes_per_layer
            manifest_fields = build_recurrence_field_manifest(
                working_set,
                field_hbm_base=field_cursor,
            )
            layer_assembly = lower_matrix_recurrence(
                spec,
                layout=layout,
                point=point,
                state_hbm_base=layer_state_begin,
                field_hbm_base=field_cursor,
                hbm_address_register=hbm_address_register,
            )
            metrics = lowering_metrics(layer_assembly)
            lines.append(
                f"; @hybrid_recurrent_binding layer={layer.number} "
                f"state=[{layer_state_begin},{layer_state_end}) "
                f"fields=[{field_cursor},{manifest_fields.end})"
            )
            lines.extend(layer_assembly.rstrip().splitlines())
            records.append(
                HybridLTileLayerRecord(
                    layer=layer.number,
                    mixer=recurrent_mixer,
                    state_hbm_begin=layer_state_begin,
                    state_hbm_end=layer_state_end,
                    field_hbm_begin=field_cursor,
                    field_hbm_end=manifest_fields.end,
                    static_instructions=int(metrics["static_instructions"]),
                    dynamic_issued_instructions=int(
                        metrics["dynamic_issued_instructions"]
                    ),
                    l_tile_exec_count=int(metrics["l_tile_exec_count"]),
                )
            )
            field_cursor = _align(manifest_fields.end)
        else:
            lines.append(
                f"; @ordinary_mixer layer={layer.number} kind={mixer} "
                "lowering=existing_plena_path"
            )
        if layer.ffn is not None:
            lines.append(
                f"; @ordinary_ffn layer={layer.number} kind={layer.ffn} "
                "lowering=existing_plena_path"
            )
        lines.append(f"; @hybrid_layer_end layer={layer.number}")

    lines.append(f"; @hybrid_schedule_end model={manifest.name}")
    frozen_records = tuple(records)
    _assert_disjoint(frozen_records)
    return HybridLTileSchedule(
        model=manifest.name,
        source_model=manifest.source_model,
        source_revision=manifest.source_revision,
        layout=layout,
        layer_count=len(manifest.layers),
        layer_counts=manifest.layer_counts(),
        recurrent_mixer=recurrent_mixer,
        recurrent_layers=recurrent_layers,
        state_hbm_begin=state_begin,
        state_hbm_end=state_end,
        field_hbm_begin=field_begin,
        field_hbm_end=field_cursor,
        records=frozen_records,
        assembly="\n".join(lines) + "\n",
    )


def build_official_hybrid_l_tile_schedules(
    model_lib: str | Path,
    *,
    layout: RecurrenceLayout | str,
    point: MatrixSramPoint | None = None,
) -> dict[str, HybridLTileSchedule]:
    model_lib = Path(model_lib)
    return {
        "nemotron3": lower_hybrid_l_tile_schedule(
            nemotron3_manifest(model_lib / "nemotron-3-nano-30b-a3b.json"),
            recurrent_mixer="mamba",
            spec=NEMOTRON_MAMBA,
            layout=layout,
            point=point,
        ),
        "kimi_k3": lower_hybrid_l_tile_schedule(
            kimi_k3_manifest(model_lib / "kimi-k3-text.json"),
            recurrent_mixer="kda",
            spec=KIMI_KDA,
            layout=layout,
            point=point,
        ),
    }


def build_official_hybrid_l_tile_report(
    model_lib: str | Path,
    *,
    point: MatrixSramPoint | None = None,
) -> dict[str, object]:
    variants: dict[str, dict[str, object]] = {}
    for layout in RecurrenceLayout:
        schedules = build_official_hybrid_l_tile_schedules(
            model_lib,
            layout=layout,
            point=point,
        )
        variants[layout] = {
            model: schedule.to_report() for model, schedule in schedules.items()
        }
    return {
        "schema_version": 1,
        "variants": variants,
        "claim": (
            "official full layer order with executable L_TILE recurrence at every "
            "Mamba/KDA layer; ordinary layers remain existing-path schedule markers"
        ),
    }


__all__ = [
    "HybridLTileLayerRecord",
    "HybridLTileSchedule",
    "build_official_hybrid_l_tile_report",
    "build_official_hybrid_l_tile_schedules",
    "lower_hybrid_l_tile_schedule",
]
