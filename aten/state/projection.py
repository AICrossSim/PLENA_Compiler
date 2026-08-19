"""Physical sidecar contract for Matrix-result projection scatter."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from functools import cached_property, lru_cache

from .contract import KdaPayload, Mamba2Payload, StateDescriptor


PROJECTION_SCATTER_CONTRACT = "plena.projection_scatter"
PROJECTION_SCATTER_VERSION = 1


class ProjectionLayout(StrEnum):
    ROW_MAJOR = "row_major"
    GROUP_MAJOR_SKEWED = "group_major_skewed"


class ProjectionFlow(StrEnum):
    BUFFERED = "buffered"
    FIFO_WITH_SPILL = "fifo_with_spill"


@dataclass(frozen=True)
class ProjectionFieldPlan:
    name: str
    producer: str
    consumer: str
    source_offset: int
    values_per_group: int
    physical_offset: int
    physical_span: int
    local_rows: int
    local_lanes: int
    group_shared: bool = False
    skew_kind: str = "none"
    skew_stride: int = 0

    def __post_init__(self) -> None:
        for name in (
            "source_offset",
            "values_per_group",
            "physical_offset",
            "physical_span",
            "local_rows",
            "local_lanes",
            "skew_stride",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.values_per_group <= 0 or self.physical_span < self.values_per_group:
            raise ValueError("projection field span must contain its logical values")
        if self.local_rows <= 0 or self.local_lanes <= 0:
            raise ValueError("projection field shape must be positive")
        if self.local_rows * self.local_lanes != self.values_per_group:
            raise ValueError("projection field shape does not match values_per_group")
        if self.consumer not in {"state", "vector"}:
            raise ValueError("projection field consumer must be state or vector")


@dataclass(frozen=True)
class ProjectionScatterPlan:
    algorithm: str
    phase: str
    context_id: int
    request_id: int
    layer_id: int
    token_offset: int
    valid_tokens: int
    activation_bytes: int
    source_input_features: int
    source_values_per_token: int
    source_projections: int
    layout: ProjectionLayout
    banks: int
    ports_per_bank: int
    groups: int
    group_span_values: int
    physical_values_per_token: int
    physical_buffer_index: int
    physical_buffer_base_row: int
    physical_token_stride_rows: int
    physical_buffer_rows: int
    fallback_vram_addr: int
    fallback_token_stride: int
    flow: ProjectionFlow
    fifo_capacity_values: int
    producer_burst_values: int
    spill_write_values_per_cycle: int
    spill_policy: str
    head_lanes: int
    head_dim_lanes: int
    state_dim_lanes: int
    fields: tuple[ProjectionFieldPlan, ...]

    def __post_init__(self) -> None:
        positive = (
            "valid_tokens",
            "activation_bytes",
            "source_input_features",
            "source_values_per_token",
            "source_projections",
            "banks",
            "ports_per_bank",
            "groups",
            "group_span_values",
            "physical_values_per_token",
            "physical_token_stride_rows",
            "physical_buffer_rows",
            "fallback_token_stride",
            "fifo_capacity_values",
            "producer_burst_values",
            "spill_write_values_per_cycle",
            "head_lanes",
            "head_dim_lanes",
            "state_dim_lanes",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.physical_values_per_token != self.groups * self.group_span_values:
            raise ValueError("physical token size must equal groups * group span")
        if self.physical_values_per_token % self.banks:
            raise ValueError("physical token size must contain complete bank rows")
        if (
            self.physical_token_stride_rows
            != self.physical_values_per_token // self.banks
        ):
            raise ValueError("physical token row stride is inconsistent")
        if self.fallback_token_stride < self.source_values_per_token:
            raise ValueError(
                "fallback stride is smaller than the logical projection packet"
            )
        if self.physical_buffer_index not in (0, 1):
            raise ValueError("projection scatter requires one of two physical buffers")
        if self.flow == ProjectionFlow.BUFFERED and self.spill_policy != "always":
            raise ValueError("buffered flow must materialize every value")
        if (
            self.flow == ProjectionFlow.FIFO_WITH_SPILL
            and self.spill_policy != "overflow_to_vector_sram"
        ):
            raise ValueError("FIFO flow requires the Vector-SRAM overflow policy")
        if self.fifo_capacity_values < self.producer_burst_values:
            raise ValueError(
                "projection FIFO must hold at least one Matrix result burst"
            )
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("projection field names must be unique")
        if (
            sum(field.values_per_group for field in self.fields) * self.groups
            != self.source_values_per_token
        ):
            raise ValueError("projection fields do not cover the logical source packet")
        _validate_projection_shape(
            self.layout,
            self.banks,
            self.groups,
            self.group_span_values,
            self.source_values_per_token,
            self.fields,
        )

    @cached_property
    def mapping_sha256(self) -> str:
        digest = hashlib.sha256()
        for field in self.fields:
            for group in range(self.groups):
                for local_row in range(field.local_rows):
                    for lane in range(field.local_lanes):
                        source, row, bank = self.address(
                            field.name, group, local_row, lane
                        )
                        digest.update(
                            f"{field.name}:{group}:{local_row}:{lane}:{source}:{row}:{bank}\n".encode()
                        )
        return digest.hexdigest()

    @property
    def padding_values_per_token(self) -> int:
        return self.physical_values_per_token - self.source_values_per_token

    def address(
        self, field_name: str, group: int, local_row: int, lane: int
    ) -> tuple[int, int, int]:
        field = next((item for item in self.fields if item.name == field_name), None)
        if field is None:
            raise ValueError(f"unknown projection field {field_name!r}")
        if not 0 <= group < self.groups:
            raise ValueError("projection group is out of range")
        if not 0 <= local_row < field.local_rows or not 0 <= lane < field.local_lanes:
            raise ValueError("projection field coordinate is out of range")
        local = local_row * field.local_lanes + lane
        source = field.source_offset + group * field.values_per_group + local
        if self.layout == ProjectionLayout.ROW_MAJOR:
            return (
                source,
                self.physical_buffer_base_row + source // self.banks,
                source % self.banks,
            )
        physical = group * self.group_span_values + field.physical_offset + local
        skew = 0
        if field.skew_kind == "local_row_stride":
            skew = local_row * field.skew_stride
        elif field.skew_kind == "field_constant":
            skew = field.skew_stride
        elif field.skew_kind == "group_stride":
            skew = group * field.skew_stride
        elif field.skew_kind != "none":
            raise ValueError(f"unknown projection skew kind {field.skew_kind!r}")
        row = self.physical_buffer_base_row + physical // self.banks
        bank = (local % self.banks + skew) % self.banks
        return source, row, bank

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["contract"] = PROJECTION_SCATTER_CONTRACT
        result["version"] = PROJECTION_SCATTER_VERSION
        result["layout"] = self.layout.value
        result["flow"] = self.flow.value
        result["padding_values_per_token"] = self.padding_values_per_token
        result["mapping_sha256"] = self.mapping_sha256
        result["fields"] = [asdict(field) for field in self.fields]
        return result

    def _validate_source_coverage(self) -> None:
        occupied: set[int] = set()
        for field in self.fields:
            for group in range(self.groups):
                start = field.source_offset + group * field.values_per_group
                occupied.update(range(start, start + field.values_per_group))
        expected = set(range(self.source_values_per_token))
        if occupied != expected:
            missing = len(expected - occupied)
            extra = len(occupied - expected)
            raise ValueError(
                f"projection source coverage is invalid: missing={missing}, extra={extra}"
            )

    def _validate_physical_coverage(self) -> None:
        occupied: set[tuple[int, int]] = set()
        for field in self.fields:
            for group in range(self.groups):
                for local_row in range(field.local_rows):
                    for lane in range(field.local_lanes):
                        _, row, bank = self.address(field.name, group, local_row, lane)
                        coordinate = (row, bank)
                        if coordinate in occupied:
                            raise ValueError(
                                f"projection mapping aliases physical coordinate {coordinate}"
                            )
                        occupied.add(coordinate)
        if len(occupied) != self.source_values_per_token:
            raise ValueError("projection physical mapping is not bijective")


@lru_cache(maxsize=32)
def _validate_projection_shape(
    layout: ProjectionLayout,
    banks: int,
    groups: int,
    group_span_values: int,
    source_values_per_token: int,
    fields: tuple[ProjectionFieldPlan, ...],
) -> None:
    """Validate each distinct geometry once instead of once per model layer."""
    sources: set[int] = set()
    physical: set[tuple[int, int]] = set()
    for field in fields:
        for group in range(groups):
            for local_row in range(field.local_rows):
                for lane in range(field.local_lanes):
                    local = local_row * field.local_lanes + lane
                    source = (
                        field.source_offset + group * field.values_per_group + local
                    )
                    if layout == ProjectionLayout.ROW_MAJOR:
                        coordinate = (source // banks, source % banks)
                    else:
                        offset = (
                            group * group_span_values + field.physical_offset + local
                        )
                        skew = 0
                        if field.skew_kind == "local_row_stride":
                            skew = local_row * field.skew_stride
                        elif field.skew_kind == "field_constant":
                            skew = field.skew_stride
                        elif field.skew_kind == "group_stride":
                            skew = group * field.skew_stride
                        elif field.skew_kind != "none":
                            raise ValueError(
                                f"unknown projection skew kind {field.skew_kind!r}"
                            )
                        coordinate = (offset // banks, (local % banks + skew) % banks)
                    if coordinate in physical:
                        raise ValueError(
                            f"projection mapping aliases physical coordinate {coordinate}"
                        )
                    sources.add(source)
                    physical.add(coordinate)
    expected = set(range(source_values_per_token))
    if sources != expected:
        raise ValueError(
            "projection source coverage is invalid: "
            f"missing={len(expected - sources)}, extra={len(sources - expected)}"
        )
    if len(physical) != source_values_per_token:
        raise ValueError("projection physical mapping is not bijective")


@dataclass(frozen=True)
class _FieldPlacement:
    """One algorithm's physical field placement and the layout that produced it."""

    fields: tuple[ProjectionFieldPlan, ...]
    groups: int
    group_span: int
    source_projections: int
    layout: ProjectionLayout


@dataclass(frozen=True)
class ProjectionScatterConfig:
    layout: ProjectionLayout = ProjectionLayout.GROUP_MAJOR_SKEWED
    banks: int = 16
    ports_per_bank: int = 1
    fifo_capacity_values: int = 64
    producer_burst_values: int = 64
    spill_write_values_per_cycle: int = 16
    direct_bypass: bool = True
    head_lanes: int = 8
    head_dim_lanes: int = 4
    state_dim_lanes: int = 8
    matrix_input_features: int = 2688
    kda_q_bank_rotation: int = 0
    kda_k_bank_rotation: int = 0
    kda_v_bank_rotation: int = 0
    kda_decay_bank_rotation: int = 0
    kda_beta_bank_rotation: int = 0
    kda_beta_group_stride: int = 0

    def __post_init__(self) -> None:
        for name in (
            "banks",
            "ports_per_bank",
            "fifo_capacity_values",
            "producer_burst_values",
            "spill_write_values_per_cycle",
            "head_lanes",
            "head_dim_lanes",
            "state_dim_lanes",
            "matrix_input_features",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "kda_q_bank_rotation",
            "kda_k_bank_rotation",
            "kda_v_bank_rotation",
            "kda_decay_bank_rotation",
            "kda_beta_bank_rotation",
            "kda_beta_group_stride",
        ):
            value = getattr(self, name)
            if not 0 <= value < self.banks:
                raise ValueError(f"{name} must be in [0, banks)")


def build_projection_scatter_plan(
    descriptor: StateDescriptor,
    *,
    phase: str,
    config: ProjectionScatterConfig,
) -> ProjectionScatterPlan:
    payload = descriptor.payload
    if isinstance(payload, Mamba2Payload):
        placement = _mamba_fields(descriptor, config)
        algorithm = "mamba2"
    elif isinstance(payload, KdaPayload):
        placement = _kda_fields(descriptor, config)
        algorithm = "kda"
    else:
        raise TypeError(f"unsupported X_STATE payload {type(payload).__name__}")

    fields = placement.fields
    groups = placement.groups
    group_span = placement.group_span
    logical_values = payload.input_elements(descriptor.num_heads)
    physical_values = groups * group_span
    token_rows = physical_values // config.banks
    logical_buffer_values = descriptor.input_token_stride * descriptor.chunk_size
    buffer_index = descriptor.input_vram_addr // logical_buffer_values
    flow = (
        ProjectionFlow.FIFO_WITH_SPILL
        if config.direct_bypass
        else ProjectionFlow.BUFFERED
    )
    return ProjectionScatterPlan(
        algorithm=algorithm,
        phase=phase,
        context_id=descriptor.context_id,
        request_id=descriptor.request_id,
        layer_id=descriptor.layer_id,
        token_offset=descriptor.token_offset,
        valid_tokens=descriptor.valid_tokens,
        activation_bytes=descriptor.activation_precision.element_bytes,
        source_input_features=config.matrix_input_features,
        source_values_per_token=logical_values,
        source_projections=placement.source_projections,
        layout=placement.layout,
        banks=config.banks,
        ports_per_bank=config.ports_per_bank,
        groups=groups,
        group_span_values=group_span,
        physical_values_per_token=physical_values,
        physical_buffer_index=buffer_index,
        physical_buffer_base_row=buffer_index * descriptor.chunk_size * token_rows,
        physical_token_stride_rows=token_rows,
        physical_buffer_rows=descriptor.chunk_size * token_rows,
        fallback_vram_addr=descriptor.input_vram_addr,
        fallback_token_stride=descriptor.input_token_stride,
        flow=flow,
        fifo_capacity_values=config.fifo_capacity_values,
        producer_burst_values=config.producer_burst_values,
        spill_write_values_per_cycle=config.spill_write_values_per_cycle,
        spill_policy="overflow_to_vector_sram" if config.direct_bypass else "always",
        head_lanes=config.head_lanes,
        head_dim_lanes=config.head_dim_lanes,
        state_dim_lanes=config.state_dim_lanes,
        fields=fields,
    )


def projection_contract_document() -> dict[str, object]:
    return {
        "contract": PROJECTION_SCATTER_CONTRACT,
        "version": PROJECTION_SCATTER_VERSION,
        "transport": "l-scatter-m-v1-plus-lowered-trace-debug-view",
        "isa_opcode": 0x3F,
        "semantics": {
            "fifo": "L_SCATTER_M applies the descriptor FIFO and buffering policy to Matrix writeback values.",
            "spill": "Materialized values retain fallback_vram_addr while the physical bank mapping is explicit.",
            "replay": "X_STATE consumes the staged physical layout only after the matching L_SCATTER_M command.",
            "layout": "The bank mapping is encoded by an L_SCATTER_M descriptor and remains absent from X_STATE descriptors.",
            "skew_kinds": "none, local_row_stride, field_constant, group_stride",
        },
    }


def projection_contract_sha256() -> str:
    encoded = json.dumps(
        projection_contract_document(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mamba_fields(
    descriptor: StateDescriptor,
    config: ProjectionScatterConfig,
) -> _FieldPlacement:
    payload = descriptor.payload
    assert isinstance(payload, Mamba2Payload)
    heads_per_group = descriptor.num_heads // payload.groups
    d_inner = descriptor.num_heads * payload.head_dim
    group_values = payload.state_dim
    source = {
        "gate": 0,
        "x": payload.xbc_offset,
        "b": payload.xbc_offset + d_inner,
        "c": payload.xbc_offset + d_inner + payload.groups * payload.state_dim,
        "dt": payload.dt_offset,
    }
    specs = (
        (
            "x",
            "in_projection",
            "state",
            source["x"],
            heads_per_group,
            payload.head_dim,
            False,
            "local_row_stride",
            config.head_dim_lanes,
        ),
        (
            "gate",
            "in_projection",
            "vector",
            source["gate"],
            heads_per_group,
            payload.head_dim,
            False,
            "local_row_stride",
            config.head_dim_lanes,
        ),
        ("b", "in_projection", "state", source["b"], 1, group_values, True, "none", 0),
        (
            "c",
            "in_projection",
            "state",
            source["c"],
            1,
            group_values,
            True,
            "field_constant",
            config.state_dim_lanes,
        ),
        (
            "dt",
            "in_projection",
            "state",
            source["dt"],
            heads_per_group,
            1,
            False,
            "none",
            0,
        ),
    )
    return _FieldPlacement(
        fields=_place_fields(specs, payload.groups, config.banks, config.layout),
        groups=payload.groups,
        group_span=_group_span(specs, config.banks, config.layout),
        source_projections=1,
        layout=config.layout,
    )


def _kda_fields(
    descriptor: StateDescriptor,
    config: ProjectionScatterConfig,
) -> _FieldPlacement:
    payload = descriptor.payload
    assert isinstance(payload, KdaPayload)

    def rotation(value: int) -> tuple[str, int]:
        return ("field_constant", value) if value else ("none", 0)

    if config.kda_beta_bank_rotation and config.kda_beta_group_stride:
        raise ValueError(
            "kda_beta_bank_rotation and kda_beta_group_stride are mutually exclusive"
        )
    beta_skew = (
        ("group_stride", config.kda_beta_group_stride)
        if config.kda_beta_group_stride
        else rotation(config.kda_beta_bank_rotation)
    )

    specs = (
        (
            "q",
            "qkv_projection",
            "state",
            payload.q_offset,
            1,
            payload.key_dim,
            False,
            *rotation(config.kda_q_bank_rotation),
        ),
        (
            "k",
            "qkv_projection",
            "state",
            payload.k_offset,
            1,
            payload.key_dim,
            False,
            *rotation(config.kda_k_bank_rotation),
        ),
        (
            "v",
            "qkv_projection",
            "state",
            payload.v_offset,
            1,
            payload.value_dim,
            False,
            *rotation(config.kda_v_bank_rotation),
        ),
        (
            "decay",
            "decay_beta_projection",
            "state",
            payload.decay_offset,
            1,
            payload.key_dim,
            False,
            *rotation(config.kda_decay_bank_rotation),
        ),
        (
            "beta",
            "decay_beta_projection",
            "state",
            payload.beta_offset,
            1,
            1,
            False,
            *beta_skew,
        ),
    )
    # Official Kimi uses independent projection tensors. This sidecar defines
    # PLENA's physical merge: fields are grouped per head and independently
    # rotated. Rotation remains transparent to X_STATE and does not change ISA.
    return _FieldPlacement(
        fields=_place_fields(specs, descriptor.num_heads, config.banks, config.layout),
        groups=descriptor.num_heads,
        group_span=_group_span(specs, config.banks, config.layout),
        source_projections=5,
        layout=config.layout,
    )


def _place_fields(
    specs: tuple[tuple[str, str, str, int, int, int, bool, str, int], ...],
    groups: int,
    banks: int,
    layout: ProjectionLayout,
) -> tuple[ProjectionFieldPlan, ...]:
    del groups
    offset = 0
    fields = []
    for (
        name,
        producer,
        consumer,
        source_offset,
        local_rows,
        local_lanes,
        shared,
        skew_kind,
        skew_stride,
    ) in specs:
        values = local_rows * local_lanes
        if layout == ProjectionLayout.GROUP_MAJOR_SKEWED:
            offset = _align_up(offset, banks)
            span = _align_up(values, banks)
        else:
            span = values
            skew_kind = "none"
            skew_stride = 0
        fields.append(
            ProjectionFieldPlan(
                name=name,
                producer=producer,
                consumer=consumer,
                source_offset=source_offset,
                values_per_group=values,
                physical_offset=offset,
                physical_span=span,
                local_rows=local_rows,
                local_lanes=local_lanes,
                group_shared=shared,
                skew_kind=skew_kind,
                skew_stride=skew_stride,
            )
        )
        offset += span
    return tuple(fields)


def _group_span(
    specs: tuple[tuple[str, str, str, int, int, int, bool, str, int], ...],
    banks: int,
    layout: ProjectionLayout,
) -> int:
    fields = _place_fields(specs, 1, banks, layout)
    end = max(field.physical_offset + field.physical_span for field in fields)
    if layout == ProjectionLayout.ROW_MAJOR:
        return end
    return _align_up(end, banks)


def _align_up(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment
