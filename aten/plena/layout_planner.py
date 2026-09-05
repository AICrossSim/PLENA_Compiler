"""Consumer-driven planning for affine producer/consumer SRAM co-layout.

The planner is intentionally workload agnostic.  A frontend supplies the
logical values written together by a producer and read together by a consumer;
the planner checks row-major, direct consumer-major, and affine-skewed physical
placements using one bank/port model.  It never selects a layout from a model
name.

Costs in this module are *layout-buffer service cycles*.  They are not layer or
model speedups.  A system timing model must place the selected service on the
shared Matrix/Vector/HBM timeline before making an end-to-end claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable

from compiler.aten.plena.affine_layout import (
    AffineLayout,
    BankGeometry,
    LayoutKind,
    LogicalCoord,
)


@dataclass(frozen=True)
class AccessPacket:
    """A logical packet serviced as one producer or consumer request."""

    name: str
    coords: tuple[LogicalCoord, ...]
    repeats: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("packet name must not be empty")
        if not self.coords:
            raise ValueError(f"packet {self.name!r} must contain at least one value")
        if self.repeats <= 0:
            raise ValueError(f"packet {self.name!r} repeats must be positive")
        if len(set(self.coords)) != len(self.coords):
            raise ValueError(f"packet {self.name!r} contains a duplicate logical value")


@dataclass(frozen=True)
class LayoutRequest:
    """Everything the planner may use for one tensor's layout decision.

    ``baseline_reorder_cycles`` is the explicit gather/pack cost paid by the
    ordinary producer layout.  ``consumer_major_supported`` says the Matrix
    writeback can directly use the consumer's logical row order with existing
    strides.  It does not imply affine bank rotation support.

    ``lane_restore_cycles_per_packet`` prices a cyclic inverse rotation.  It is
    deliberately explicit: setting it to zero is an architectural assumption,
    not a free operation hidden by the planner.
    """

    name: str
    groups: int
    fields: int
    majors: int
    minors: int
    producer_packets: tuple[AccessPacket, ...]
    consumer_packets: tuple[AccessPacket, ...]
    baseline_reorder_cycles: int = 0
    consumer_major_supported: bool = False
    transpose_supported: bool = True
    lane_restore_cycles_per_packet: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("layout request name must not be empty")
        for name in ("groups", "fields", "majors", "minors"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.baseline_reorder_cycles < 0 or self.lane_restore_cycles_per_packet < 0:
            raise ValueError("layout costs must be non-negative")
        for packet in (*self.producer_packets, *self.consumer_packets):
            for coord in packet.coords:
                if not (
                    0 <= coord.group < self.groups
                    and 0 <= coord.field < self.fields
                    and 0 <= coord.major < self.majors
                    and 0 <= coord.minor < self.minors
                ):
                    raise ValueError(
                        f"packet {packet.name!r} coordinate {coord} is outside "
                        f"[{self.groups}, {self.fields}, {self.majors}, {self.minors}]"
                    )


@dataclass(frozen=True)
class LayoutScore:
    name: str
    layout: AffineLayout
    write_cycles: int
    write_floor_cycles: int
    read_cycles: int
    read_floor_cycles: int
    conflict_stall_cycles: int
    reorder_cycles: int
    lane_restore_cycles: int
    total_cycles: int
    mapping_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "layout": {**asdict(self.layout), "kind": self.layout.kind.value},
            "write_cycles": self.write_cycles,
            "write_floor_cycles": self.write_floor_cycles,
            "read_cycles": self.read_cycles,
            "read_floor_cycles": self.read_floor_cycles,
            "conflict_stall_cycles": self.conflict_stall_cycles,
            "reorder_cycles": self.reorder_cycles,
            "lane_restore_cycles": self.lane_restore_cycles,
            "total_cycles": self.total_cycles,
            "mapping_sha256": self.mapping_sha256,
        }


@dataclass(frozen=True)
class LayoutPlan:
    request: str
    selected: LayoutScore
    candidates: tuple[LayoutScore, ...]

    @property
    def baseline(self) -> LayoutScore:
        return next(candidate for candidate in self.candidates if candidate.name == "row_major")

    @property
    def speedup(self) -> float:
        return self.baseline.total_cycles / max(1, self.selected.total_cycles)

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request,
            "selected": self.selected.name,
            "baseline_cycles": self.baseline.total_cycles,
            "selected_cycles": self.selected.total_cycles,
            "layout_service_speedup": self.speedup,
            "scope": "layout_buffer_service_only",
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class AffineLayoutPlanner:
    """Enumerate legal affine placements and select the lowest total cost."""

    def __init__(self, geometry: BankGeometry) -> None:
        geometry.validate()
        self.geometry = geometry

    def _layout(
        self,
        request: LayoutRequest,
        *,
        kind: LayoutKind,
        alpha: int = 0,
        beta: int = 0,
        gamma: int = 0,
    ) -> AffineLayout:
        return AffineLayout(
            kind=kind,
            groups=request.groups,
            fields=request.fields,
            majors=request.majors,
            minors=request.minors,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    @staticmethod
    def _packet_cycles(
        layout: AffineLayout,
        geometry: BankGeometry,
        packets: Iterable[AccessPacket],
        *,
        write: bool,
    ) -> tuple[int, int, int]:
        service = floor = stalls = 0
        for packet in packets:
            stats = layout.packet_service(packet.coords, geometry, write=write)
            service += packet.repeats * stats.service_cycles
            floor += packet.repeats * stats.bandwidth_floor_cycles
            stalls += packet.repeats * stats.conflict_stall_cycles
        return service, floor, stalls

    def _score(
        self,
        request: LayoutRequest,
        *,
        name: str,
        layout: AffineLayout,
        reorder_cycles: int,
    ) -> LayoutScore:
        # The affine formula is bijective by construction once its pitch is
        # valid: each outer row owns a disjoint bank-row range, bank rotation is
        # a permutation, and sublane is unchanged.  Validate the finite-domain
        # contract without re-enumerating a 100k-element real tensor for every
        # point in the B^3 coefficient search.
        layout.validate(self.geometry)
        write, write_floor, write_stalls = self._packet_cycles(
            layout, self.geometry, request.producer_packets, write=True
        )
        read, read_floor, read_stalls = self._packet_cycles(
            layout, self.geometry, request.consumer_packets, write=False
        )
        rotated = bool(layout.alpha or layout.beta or layout.gamma)
        restore = (
            sum(packet.repeats for packet in request.consumer_packets)
            * request.lane_restore_cycles_per_packet
            if rotated
            else 0
        )
        contract = json.dumps(
            {
                "layout": {**asdict(layout), "kind": layout.kind.value},
                "geometry": asdict(self.geometry),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return LayoutScore(
            name=name,
            layout=layout,
            write_cycles=write,
            write_floor_cycles=write_floor,
            read_cycles=read,
            read_floor_cycles=read_floor,
            conflict_stall_cycles=write_stalls + read_stalls,
            reorder_cycles=reorder_cycles,
            lane_restore_cycles=restore,
            total_cycles=write + read + reorder_cycles + restore,
            mapping_sha256=hashlib.sha256(contract).hexdigest(),
        )

    def plan(self, request: LayoutRequest) -> LayoutPlan:
        candidates: list[LayoutScore] = []
        row = self._layout(request, kind=LayoutKind.ROW_MAJOR)
        candidates.append(
            self._score(
                request,
                name="row_major",
                layout=row,
                reorder_cycles=request.baseline_reorder_cycles,
            )
        )
        if request.consumer_major_supported:
            # Consumer-major direct write is a producer schedule: it removes
            # an explicit gather but does not, by itself, rotate physical
            # banks. Keep row placement so it receives no affine benefit.
            candidates.append(
                self._score(
                    request,
                    name="consumer_major",
                    layout=self._layout(request, kind=LayoutKind.CONSUMER_MAJOR),
                    reorder_cycles=0,
                )
            )
        if request.transpose_supported:
            candidates.append(
                self._score(
                    request,
                    name="transpose",
                    layout=self._layout(request, kind=LayoutKind.TRANSPOSE),
                    reorder_cycles=0,
                )
            )

        # Exhaustive enumeration is small (B^3; 4096 points for B=16) and avoids
        # a workload-name lookup table.  Equivalent all-zero placement is already
        # represented by row_major/consumer_major and is skipped here.
        consumer_coords = [coord for packet in request.consumer_packets for coord in packet.coords]
        alpha_values = range(self.geometry.banks) if len({c.major for c in consumer_coords}) > 1 else (0,)
        beta_values = range(self.geometry.banks) if len({c.field for c in consumer_coords}) > 1 else (0,)
        gamma_values = range(self.geometry.banks) if len({c.group for c in consumer_coords}) > 1 else (0,)
        for alpha in alpha_values:
            for beta in beta_values:
                for gamma in gamma_values:
                    if not (alpha or beta or gamma):
                        continue
                    layout = self._layout(
                        request,
                        kind=LayoutKind.AFFINE_SKEW,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                    )
                    candidates.append(
                        self._score(
                            request,
                            name=f"affine_a{alpha}_b{beta}_g{gamma}",
                            layout=layout,
                            reorder_cycles=0,
                        )
                    )

        # Prefer the simpler mapping on a tie.  This ensures affine hardware is
        # selected only when it earns a strictly lower cost.
        simplicity = {
            LayoutKind.ROW_MAJOR: 0,
            LayoutKind.CONSUMER_MAJOR: 1,
            LayoutKind.TRANSPOSE: 2,
            LayoutKind.AFFINE_SKEW: 3,
        }
        selected = min(
            candidates,
            key=lambda candidate: (
                candidate.total_cycles,
                simplicity[candidate.layout.kind],
                bool(candidate.layout.gamma),
                bool(candidate.layout.beta),
                bool(candidate.layout.alpha),
                candidate.layout.gamma,
                candidate.layout.beta,
                candidate.layout.alpha,
            ),
        )
        return LayoutPlan(request.name, selected, tuple(candidates))


def full_row_packets(
    *, groups: int, fields: int, majors: int, minors: int, name: str = "producer_row"
) -> tuple[AccessPacket, ...]:
    """Conventional wide-row producer packets for a finite logical tensor."""

    return tuple(
        AccessPacket(
            name=f"{name}_g{group}_f{field}_m{major}",
            coords=tuple(LogicalCoord(group, field, major, minor) for minor in range(minors)),
        )
        for group in range(groups)
        for field in range(fields)
        for major in range(majors)
    )


__all__ = [
    "AccessPacket",
    "AffineLayoutPlanner",
    "LayoutPlan",
    "LayoutRequest",
    "LayoutScore",
    "full_row_packets",
]
