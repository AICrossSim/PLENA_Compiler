"""Compiler contract for producer/consumer affine SRAM co-layout.

The contract is deliberately independent of Mamba, KDA, attention, and MoE.
Those workloads provide logical packets; this module only answers where each
logical scalar lives in a banked output SRAM and how many port cycles a packet
requires.

The physical SRAM is a banked view of a conventional wide row.  One bank word
contains ``bank_width`` adjacent scalars.  Reading several sublanes from the
same ``(bank, bank_row)`` therefore costs one bank access, not one access per
scalar.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Iterator

from compiler.utils.enum_compat import StrEnum


class LayoutKind(StrEnum):
    ROW_MAJOR = "row_major"
    TRANSPOSE = "transpose"
    CONSUMER_MAJOR = "consumer_major"
    AFFINE_SKEW = "affine_skew"


@dataclass(frozen=True, order=True)
class LogicalCoord:
    """One scalar in logical ``[group, field, major, minor]`` order."""

    group: int
    field: int
    major: int
    minor: int


@dataclass(frozen=True, order=True)
class PhysicalCoord:
    bank: int
    bank_row: int
    sublane: int


@dataclass(frozen=True)
class BankGeometry:
    banks: int = 16
    bank_width: int = 4
    read_ports: int = 1
    write_ports: int = 1

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

    @property
    def row_elements(self) -> int:
        return self.banks * self.bank_width


@dataclass(frozen=True)
class PacketService:
    values: int
    bank_words: int
    bandwidth_floor_cycles: int
    service_cycles: int

    @property
    def conflict_stall_cycles(self) -> int:
        return self.service_cycles - self.bandwidth_floor_cycles


@dataclass(frozen=True)
class AffineLayout:
    """A bijective affine placement over a finite four-dimensional tensor.

    ``alpha``, ``beta`` and ``gamma`` rotate successive major rows, fields and
    groups across banks.  They change only placement, never logical value
    order.  ``major_packed`` coalesces one bank word from consecutive major
    rows into the same physical wide row; this is the generic short-row layout
    used by multi-row packets. ``bank_row_pitch`` may reserve padding between
    physical rows; zero asks the contract to derive the smallest non-aliasing
    pitch.
    """

    kind: LayoutKind
    groups: int
    fields: int
    majors: int
    minors: int
    alpha: int = 0
    beta: int = 0
    gamma: int = 0
    major_packed: bool = False
    bank_row_base: int = 0
    bank_row_pitch: int = 0

    def validate(self, geometry: BankGeometry) -> None:
        geometry.validate()
        for name in ("groups", "fields", "majors", "minors"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.bank_row_base < 0 or self.bank_row_pitch < 0:
            raise ValueError("bank row base and pitch must be non-negative")
        if self.major_packed:
            if self.kind == LayoutKind.TRANSPOSE:
                raise ValueError("major-packed placement does not support TRANSPOSE")
            if math.gcd(self.alpha % geometry.banks, geometry.banks) != 1:
                raise ValueError(
                    "major-packed alpha must permute every physical bank"
                )
        if self.pitch(geometry) < self.minimum_pitch(geometry):
            raise ValueError(
                f"bank_row_pitch {self.pitch(geometry)} is smaller than the "
                f"non-aliasing minimum {self.minimum_pitch(geometry)}"
            )

    def minimum_pitch(self, geometry: BankGeometry) -> int:
        if self.major_packed:
            return 1
        inner = self.majors if self.kind == LayoutKind.TRANSPOSE else self.minors
        stripes = math.ceil(inner / geometry.bank_width)
        return math.ceil(stripes / geometry.banks)

    def pitch(self, geometry: BankGeometry) -> int:
        return self.bank_row_pitch or self.minimum_pitch(geometry)

    def iter_coords(self) -> Iterator[LogicalCoord]:
        for group in range(self.groups):
            for field in range(self.fields):
                for major in range(self.majors):
                    for minor in range(self.minors):
                        yield LogicalCoord(group, field, major, minor)

    def logical_index(self, coord: LogicalCoord) -> int:
        self._check_coord(coord)
        return (
            ((coord.group * self.fields + coord.field) * self.majors + coord.major)
            * self.minors
            + coord.minor
        )

    def place(self, coord: LogicalCoord, geometry: BankGeometry) -> PhysicalCoord:
        self.validate(geometry)
        self._check_coord(coord)
        if self.kind == LayoutKind.TRANSPOSE:
            # A logical column becomes one physical wide row. This makes a
            # column packet cheap only by paying the corresponding scatter
            # cost when a conventional row-producing Matrix operation writes.
            inner = coord.major
            outer = (coord.group * self.fields + coord.field) * self.minors + coord.minor
        else:
            inner = coord.minor
            outer = (coord.group * self.fields + coord.field) * self.majors + coord.major
        stripe, sublane = divmod(inner, geometry.bank_width)
        phase = (
            self.alpha * coord.major
            + self.beta * coord.field
            + self.gamma * coord.group
        ) % geometry.banks
        bank = (stripe + phase) % geometry.banks
        if self.major_packed:
            minor_steps = math.ceil(self.minors / geometry.bank_width)
            major_blocks = math.ceil(self.majors / geometry.banks)
            field_group = coord.group * self.fields + coord.field
            packed_row = (
                (field_group * major_blocks + coord.major // geometry.banks)
                * minor_steps
                + stripe
            )
            bank_row = self.bank_row_base + packed_row * self.pitch(geometry)
        else:
            bank_row = (
                self.bank_row_base
                + outer * self.pitch(geometry)
                + stripe // geometry.banks
            )
        return PhysicalCoord(bank=bank, bank_row=bank_row, sublane=sublane)

    def major_start_row_offset(
        self, major_start: int, geometry: BankGeometry
    ) -> int:
        """Physical-row offset for a tensor-relative major-aligned subview."""

        if major_start < 0 or major_start >= self.majors:
            raise ValueError(
                f"major_start {major_start} is outside [0, {self.majors})"
            )
        if self.major_packed:
            if major_start % geometry.banks:
                raise ValueError(
                    "major-packed subview must begin on a complete bank group"
                )
            minor_steps = math.ceil(self.minors / geometry.bank_width)
            return (
                major_start // geometry.banks
                * minor_steps
                * self.pitch(geometry)
            )
        return major_start * self.pitch(geometry)

    def assert_bijective(self, geometry: BankGeometry) -> None:
        occupied: dict[PhysicalCoord, LogicalCoord] = {}
        for logical in self.iter_coords():
            physical = self.place(logical, geometry)
            previous = occupied.setdefault(physical, logical)
            if previous != logical:
                raise ValueError(
                    f"affine layout aliases {previous} and {logical} at {physical}"
                )

    def mapping_sha256(self, geometry: BankGeometry) -> str:
        self.assert_bijective(geometry)
        digest = hashlib.sha256()
        for logical in self.iter_coords():
            physical = self.place(logical, geometry)
            digest.update(
                f"{logical.group}:{logical.field}:{logical.major}:{logical.minor}:"
                f"{physical.bank}:{physical.bank_row}:{physical.sublane}\n".encode()
            )
        return digest.hexdigest()

    def packet_service(
        self,
        packet: Iterable[LogicalCoord],
        geometry: BankGeometry,
        *,
        write: bool = False,
    ) -> PacketService:
        ports = geometry.write_ports if write else geometry.read_ports
        coords = list(packet)
        physical = [self.place(coord, geometry) for coord in coords]
        # One bank word supplies all requested sublanes at that row.
        words = {(coord.bank, coord.bank_row) for coord in physical}
        per_bank = [0] * geometry.banks
        for bank, _row in words:
            per_bank[bank] += 1
        service = max((math.ceil(count / ports) for count in per_bank), default=0)
        floor = math.ceil(len(words) / (geometry.banks * ports)) if words else 0
        return PacketService(
            values=len(coords),
            bank_words=len(words),
            bandwidth_floor_cycles=floor,
            service_cycles=service,
        )

    def to_contract_dict(self, geometry: BankGeometry) -> dict[str, object]:
        self.assert_bijective(geometry)
        return {
            "contract": "plena.affine_layout",
            "version": 1,
            "layout": {**asdict(self), "kind": self.kind.value},
            "geometry": asdict(geometry),
            "mapping_sha256": self.mapping_sha256(geometry),
        }

    def _check_coord(self, coord: LogicalCoord) -> None:
        bounds = (
            ("group", coord.group, self.groups),
            ("field", coord.field, self.fields),
            ("major", coord.major, self.majors),
            ("minor", coord.minor, self.minors),
        )
        for name, value, extent in bounds:
            if not 0 <= value < extent:
                raise ValueError(f"{name} coordinate {value} is outside [0, {extent})")


__all__ = [
    "AffineLayout",
    "BankGeometry",
    "LayoutKind",
    "LogicalCoord",
    "PacketService",
    "PhysicalCoord",
]
