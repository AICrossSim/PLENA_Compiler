"""Official-shape recurrent packets lowered through Matrix-SRAM views.

This module is deliberately narrower than a model program and broader than a
synthetic stride probe.  It emits the packet core that both full hybrid model
programs invoke after projection/normalisation has produced the operands.  The
math remains ordinary Vector ISA; ``L_MVIEW`` only controls where Matrix-SRAM
words are placed and how they are restored on a packet read.

The two layout variants differ only in the packed mapping word.  Consequently
the row-major/fixed and affine binaries have identical instruction and
arithmetic counts.  That property is the basis of the physical-layout
ablation: a speedup cannot be attributed to fewer instructions or different
math.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from compiler.asm_templates._imm import load_large_int
from compiler.aten.plena.instruction_stream import (
    arithmetic_share,
    dynamic_count,
    opcode_census,
    static_count,
)
from compiler.aten.plena.mview import MatrixViewDescriptor, MatrixViewMap, MatrixViewShape
from compiler.aten.plena.matrix_access_packets import (
    PacketGeometry,
    extract_matrix_access_packets,
)


@dataclass(frozen=True)
class MatrixRecurrenceSpec:
    name: str
    heads: int
    packet_heads: int
    row_elements: int
    recurrence_rows: int
    passes: int
    field_loads_per_row: int
    arithmetic_ops_per_row: int

    @property
    def packet_groups(self) -> int:
        if self.heads % self.packet_heads:
            raise ValueError(f"{self.name}: heads must divide into packet groups")
        return self.heads // self.packet_heads

    @property
    def packet_values(self) -> int:
        return self.packet_heads * self.row_elements

    def affine_alpha(self, *, bank_width: int) -> int:
        """Return the physical-row skew for one machine geometry.

        A logical head row occupies ``row_elements / bank_width`` bank words.
        Rotating consecutive heads by exactly that many banks packs a
        cross-head packet without overlap.  Keeping this derived from the
        machine's bank width is important: the ISA carries a skew, not an
        assumption that BLEN is always 32.
        """

        if bank_width <= 0 or self.row_elements % bank_width:
            raise ValueError(
                f"{self.name}: row width {self.row_elements} is not divisible "
                f"by bank width {bank_width}"
            )
        return self.row_elements // bank_width

    def validate(self, *, mlen: int, blen: int) -> None:
        if self.packet_values != mlen:
            raise ValueError(
                f"{self.name}: packet has {self.packet_values} values, MLEN is {mlen}"
            )
        if self.row_elements % blen:
            raise ValueError(f"{self.name}: one logical row must contain whole bank words")
        if self.affine_alpha(bank_width=blen) * self.packet_heads != mlen // blen:
            raise ValueError(
                f"{self.name}: affine skew does not cover every Matrix SRAM bank"
            )


NEMOTRON_MAMBA = MatrixRecurrenceSpec(
    name="nemotron3_mamba2",
    heads=64,
    packet_heads=32,
    row_elements=64,
    recurrence_rows=128,
    passes=1,
    # dA, input update and C/readout packets.
    field_loads_per_row=3,
    # state*=dA; state+=input; tmp=state*C; out+=tmp.
    arithmetic_ops_per_row=4,
)

KIMI_KDA = MatrixRecurrenceSpec(
    name="kimi_k3_kda",
    heads=96,
    packet_heads=16,
    row_elements=128,
    recurrence_rows=128,
    passes=2,
    # Sweep 1: decay and k. Sweep 2: rank-1 update and q.
    field_loads_per_row=2,
    # Each sweep performs one state transform and one accumulated contraction.
    arithmetic_ops_per_row=3,
)


def _configure_view(
    *,
    slot: int,
    descriptor: MatrixViewDescriptor,
    shape_register: int,
    map_register: int,
) -> list[str]:
    return [
        *load_large_int(shape_register, descriptor.shape.pack()),
        *load_large_int(map_register, descriptor.mapping.pack()),
        f"L_MVIEW_FULL {slot}, gp{shape_register}, gp{map_register}",
    ]


def _packet_body(spec: MatrixRecurrenceSpec, pass_index: int) -> list[str]:
    # gp1 is the state packet and gp2..gp4 are distinct field packets. Slots
    # 0/1/2 qualify destination, source-1 and source-2 respectively; gp5..gp7
    # are ordinary Vector rows. Keeping field pointers distinct is required for
    # this to describe the real recurrence rather than a packet-shape probe.
    if spec.name == NEMOTRON_MAMBA.name:
        return [
            "; @axis=cross_head_field state_decay",
            "V_MUL_VV.MV gp1, gp1, gp2, 0, 7",
            "; @axis=cross_head_field state_update",
            "V_ADD_VV.MV gp1, gp1, gp3, 0, 7",
            "; @axis=cross_head_field state_readout",
            "V_MUL_VV.MV gp5, gp1, gp4, 0, 6",
            "V_ADD_VV gp6, gp6, gp5, 0, 0",
        ]

    if spec.name == KIMI_KDA.name and pass_index == 0:
        return [
            "; @axis=cross_head_field state_decay",
            "V_MUL_VV.MV gp1, gp1, gp2, 0, 7",
            "; @axis=cross_head_field key_prediction",
            "V_MUL_VV.MV gp5, gp1, gp3, 0, 6",
            "V_ADD_VV gp6, gp6, gp5, 0, 0",
        ]
    if spec.name == KIMI_KDA.name and pass_index == 1:
        return [
            "; @axis=cross_head_field delta_update",
            "V_ADD_VV.MV gp1, gp1, gp2, 0, 7",
            "; @axis=cross_head_field query_readout",
            "V_MUL_VV.MV gp5, gp1, gp3, 0, 6",
            "V_ADD_VV gp7, gp7, gp5, 0, 0",
        ]
    raise AssertionError(f"unsupported packet body {spec.name} pass={pass_index}")


def lower_matrix_recurrence(
    spec: MatrixRecurrenceSpec,
    *,
    affine: bool,
    mlen: int = 2048,
    blen: int = 32,
) -> str:
    """Emit one official-shape recurrent layer's Matrix packet core.

    State is explicitly tiled through an existing Matrix-SRAM scratch window;
    this function does not allocate a cache or assume layer state residency.
    HBM prefetch/writeback is intentionally outside the core and is charged by
    the system model with the official FP32 state bytes.
    """

    spec.validate(mlen=mlen, blen=blen)
    alpha = spec.affine_alpha(bank_width=blen) if affine else 0
    descriptor = MatrixViewDescriptor(
        shape=MatrixViewShape(rows=1, cols=spec.row_elements, tile_count=spec.packet_heads),
        mapping=MatrixViewMap(tile_pitch_rows=1, alpha=alpha),
    )
    descriptor.validate_for_machine(banks=mlen // blen, bank_width=blen)

    lines = [
        f"; @stage={spec.name}_matrix_recurrence",
        f"; layout={'affine_per_tile' if affine else 'global_fixed'}",
        *_configure_view(slot=0, descriptor=descriptor, shape_register=14, map_register=15),
        *_configure_view(slot=1, descriptor=descriptor, shape_register=14, map_register=15),
        *_configure_view(slot=2, descriptor=descriptor, shape_register=14, map_register=15),
        *load_large_int(5, 0),
        *load_large_int(6, mlen),
        *load_large_int(7, 2 * mlen),
    ]
    loop_register = 13
    packet_span = spec.packet_heads * mlen
    state_region_span = (
        spec.packet_groups * spec.recurrence_rows * packet_span
    )
    for group in range(spec.packet_groups):
        lines.append(f"; packet_group={group}/{spec.packet_groups}")
        for pass_index in range(spec.passes):
            state_base = group * spec.recurrence_rows * packet_span
            field_set = group * spec.passes + pass_index
            lines.extend(load_large_int(1, state_base))
            field_registers = tuple(range(2, 2 + spec.field_loads_per_row))
            for field_index, register in enumerate(field_registers):
                field_base = state_region_span + (
                    field_set * spec.field_loads_per_row + field_index
                ) * spec.recurrence_rows * packet_span
                lines.extend(load_large_int(register, field_base))
            lines.append(f"C_LOOP_START gp{loop_register}, {spec.recurrence_rows}")
            lines.extend(_packet_body(spec, pass_index))
            for register in (1, *field_registers):
                lines.append(
                    f"S_ADDI_INT gp{register}, gp{register}, {packet_span}"
                )
            lines.append(f"C_LOOP_END gp{loop_register}")
    return "\n".join(lines) + "\n"


def lowering_metrics(assembly: str) -> dict[str, object]:
    census = opcode_census(assembly)
    packets = extract_matrix_access_packets(
        assembly,
        PacketGeometry(mlen=2048, blen=32, hlen=128),
    )
    packet_reads = sum(packet.repeats for packet in packets if packet.direction == "read")
    packet_writes = sum(packet.repeats for packet in packets if packet.direction == "write")
    return {
        "static_instructions": static_count(assembly),
        "dynamic_issued_instructions": dynamic_count(assembly),
        "opcode_census": census,
        "arithmetic_share": arithmetic_share(assembly),
        "packet_reads": packet_reads,
        "packet_writes": packet_writes,
    }


def build_matrix_recurrence_report() -> dict[str, object]:
    models: dict[str, object] = {}
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        fixed = lower_matrix_recurrence(spec, affine=False)
        affine = lower_matrix_recurrence(spec, affine=True)
        fixed_metrics = lowering_metrics(fixed)
        affine_metrics = lowering_metrics(affine)
        if fixed_metrics != affine_metrics:
            raise AssertionError(
                f"{spec.name}: physical layout changed the issued operation stream"
            )
        models[spec.name] = {
            "spec": asdict(spec),
            "packet_groups": spec.packet_groups,
            "packet_values": spec.packet_values,
            "fixed_alpha": 0,
            "affine_alpha": spec.affine_alpha(bank_width=32),
            "metrics": fixed_metrics,
            "assembly_fixed": fixed,
            "assembly_affine": affine,
            "only_mapping_word_differs": True,
        }
    return {
        "schema_version": 1,
        "geometry": {"mlen": 2048, "blen": 32, "banks": 64, "bank_width": 32},
        "models": models,
        "scope": {
            "state": "explicitly tiled; no cache or residency assumption",
            "math": "existing V_MUL_VV and V_ADD_VV only",
            "layout": "L_MVIEW contains no model name or update formula",
            "precision": "packet SRAM is BF16; official FP32 state traffic remains explicit",
        },
    }


__all__ = [
    "KIMI_KDA",
    "NEMOTRON_MAMBA",
    "MatrixRecurrenceSpec",
    "build_matrix_recurrence_report",
    "lower_matrix_recurrence",
    "lowering_metrics",
]
