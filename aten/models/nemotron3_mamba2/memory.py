"""Byte-addressed HBM allocation for persistent Mamba-2 state.

The legacy PLENA allocators use tensor elements, SRAM rows, and MX-packed HBM
rows in different places. Mamba descriptors define every address and stride in
bytes, so this module deliberately has no dependency on the legacy manager.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from .reference import Mamba2Config


U64_LIMIT = 1 << 64
U32_LIMIT = 1 << 32
STATE_ELEMENT_BYTES = 4


def _require_power_of_two(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value!r}")


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


@dataclass(frozen=True)
class MemoryRegion:
    """One non-overlapping byte range owned by a :class:`ByteAddressArena`."""

    name: str
    address: int
    size_bytes: int
    alignment: int
    kind: str

    @property
    def end_address(self) -> int:
        return self.address + self.size_bytes


class ByteAddressArena:
    """Monotonic, overlap-checking allocator for one HBM address space."""

    def __init__(
        self,
        base_address: int,
        size_bytes: int | None = None,
        *,
        default_alignment: int = 64,
    ) -> None:
        _require_power_of_two("default_alignment", default_alignment)
        if not isinstance(base_address, int) or not 0 <= base_address < U64_LIMIT:
            raise ValueError("base_address must be an unsigned 64-bit byte address")
        if size_bytes is not None and (
            not isinstance(size_bytes, int)
            or size_bytes <= 0
            or base_address + size_bytes > U64_LIMIT
        ):
            raise ValueError("arena size must define a non-empty unsigned 64-bit range")
        self.base_address = base_address
        self.limit_address = None if size_bytes is None else base_address + size_bytes
        self.default_alignment = default_alignment
        self._cursor = base_address
        self._regions: dict[str, MemoryRegion] = {}
        self._unique_counters: dict[str, int] = {}
        self._lock = RLock()

    @property
    def regions(self) -> tuple[MemoryRegion, ...]:
        with self._lock:
            return tuple(
                sorted(self._regions.values(), key=lambda region: region.address)
            )

    def owns(self, region: MemoryRegion) -> bool:
        with self._lock:
            return self._regions.get(region.name) == region

    def reserve(
        self,
        name: str,
        address: int,
        size_bytes: int,
        *,
        alignment: int | None = None,
        kind: str = "external",
    ) -> MemoryRegion:
        """Reserve an existing HBM range and reject aliases or duplicate names."""
        alignment = self.default_alignment if alignment is None else alignment
        _require_power_of_two("alignment", alignment)
        if not name:
            raise ValueError("memory region name must not be empty")
        if not isinstance(address, int) or address < self.base_address:
            raise ValueError(f"{name} starts below the arena base")
        if address % alignment:
            raise ValueError(
                f"{name} address 0x{address:x} is not {alignment}-byte aligned"
            )
        if not isinstance(size_bytes, int) or size_bytes <= 0:
            raise ValueError(f"{name} size must be positive")
        end_address = address + size_bytes
        if end_address > U64_LIMIT or (
            self.limit_address is not None and end_address > self.limit_address
        ):
            raise MemoryError(f"{name} exceeds the HBM arena")

        with self._lock:
            if name in self._regions:
                raise ValueError(f"memory region {name!r} already exists")
            for existing in self._regions.values():
                if address < existing.end_address and existing.address < end_address:
                    raise ValueError(
                        f"memory region {name!r} overlaps {existing.name!r}: "
                        f"[0x{address:x}, 0x{end_address:x}) vs "
                        f"[0x{existing.address:x}, 0x{existing.end_address:x})"
                    )
            region = MemoryRegion(name, address, size_bytes, alignment, kind)
            self._regions[name] = region
            self._cursor = max(self._cursor, end_address)
            return region

    def allocate(
        self,
        name: str,
        size_bytes: int,
        *,
        alignment: int | None = None,
        kind: str = "transient",
    ) -> MemoryRegion:
        alignment = self.default_alignment if alignment is None else alignment
        _require_power_of_two("alignment", alignment)
        with self._lock:
            address = _align_up(self._cursor, alignment)
            return self.reserve(
                name,
                address,
                size_bytes,
                alignment=alignment,
                kind=kind,
            )

    def allocate_unique(
        self,
        prefix: str,
        size_bytes: int,
        *,
        alignment: int | None = None,
        kind: str = "transient",
    ) -> MemoryRegion:
        """Allocate a range with a suffix shared by separate compiler objects."""
        with self._lock:
            index = self._unique_counters.get(prefix, 0)
            while f"{prefix}.{index}" in self._regions:
                index += 1
            self._unique_counters[prefix] = index + 1
            return self.allocate(
                f"{prefix}.{index}",
                size_bytes,
                alignment=alignment,
                kind=kind,
            )


@dataclass(frozen=True)
class MambaStateAllocation:
    context_id: int
    layer_id: int
    batch_capacity: int
    shape_signature: tuple[int, ...]
    ssm: MemoryRegion
    conv: MemoryRegion
    state_request_stride: int
    state_head_stride: int
    conv_request_stride: int


def _state_shape_signature(config: Mamba2Config) -> tuple[int, ...]:
    return (
        config.d_inner,
        config.num_heads,
        config.head_dim,
        config.state_dim,
        config.groups,
        config.conv_kernel,
    )


class MambaPersistentStateAllocator:
    """Keep one stable state mapping for every ``(context_id, layer_id)`` key."""

    def __init__(self, arena: ByteAddressArena) -> None:
        self.arena = arena
        self._allocations: dict[tuple[int, int], MambaStateAllocation] = {}
        self._lock = RLock()

    @property
    def allocations(self) -> tuple[MambaStateAllocation, ...]:
        with self._lock:
            return tuple(self._allocations[key] for key in sorted(self._allocations))

    def allocate(
        self,
        context_id: int,
        layer_id: int,
        config: Mamba2Config,
        batch_capacity: int,
    ) -> MambaStateAllocation:
        for name, value in (("context_id", context_id), ("layer_id", layer_id)):
            if not isinstance(value, int) or not 0 <= value < U32_LIMIT:
                raise ValueError(f"{name} must be an unsigned 32-bit integer")
        if not isinstance(batch_capacity, int) or batch_capacity <= 0:
            raise ValueError("batch_capacity must be positive")

        head_stride = config.head_dim * config.state_dim * STATE_ELEMENT_BYTES
        request_stride = config.num_heads * head_stride
        conv_request_stride = (
            config.conv_channels * config.conv_kernel * STATE_ELEMENT_BYTES
        )
        for name, value in (
            ("state_head_stride", head_stride),
            ("state_request_stride", request_stride),
            ("conv_request_stride", conv_request_stride),
        ):
            if value >= U32_LIMIT:
                raise ValueError(f"{name} does not fit the descriptor u32 field")

        key = (context_id, layer_id)
        signature = _state_shape_signature(config)
        with self._lock:
            existing = self._allocations.get(key)
            if existing is not None:
                if existing.shape_signature != signature:
                    raise ValueError(
                        f"state key {key} is already allocated for a different Mamba shape"
                    )
                if batch_capacity > existing.batch_capacity:
                    raise ValueError(
                        f"state key {key} has batch capacity {existing.batch_capacity}; "
                        f"growing it would invalidate previously emitted descriptors"
                    )
                return existing

            prefix = f"mamba.state.ctx{context_id}.layer{layer_id}"
            ssm = self.arena.allocate(
                f"{prefix}.ssm",
                request_stride * batch_capacity,
                kind="persistent_state",
            )
            conv = self.arena.allocate(
                f"{prefix}.conv",
                conv_request_stride * batch_capacity,
                kind="persistent_state",
            )
            allocation = MambaStateAllocation(
                context_id=context_id,
                layer_id=layer_id,
                batch_capacity=batch_capacity,
                shape_signature=signature,
                ssm=ssm,
                conv=conv,
                state_request_stride=request_stride,
                state_head_stride=head_stride,
                conv_request_stride=conv_request_stride,
            )
            self._allocations[key] = allocation
            return allocation
