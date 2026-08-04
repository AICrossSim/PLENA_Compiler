"""Versioned PackedKV storage contract shared by lowering and validation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Sequence

from compiler.aten.plena.memory import MatrixBlockLayout


_PACKED_HEAD_SELECTOR_BITS = 4


class PackedKVAblation(str, Enum):
    PADDED_PER_HEAD = "padded_per_head"
    DENSE_COMPILER = "dense_compiler"
    DENSE_SELECTOR = "dense_selector"
    IDEAL_TRAFFIC = "ideal_traffic"


@dataclass(frozen=True)
class PackedKVLayout:
    """One token stores all KV heads in a single MLEN-wide row."""

    kv_heads: int
    head_dim: int
    mlen: int
    block_size: int = 8
    element_bits: int = 4
    scale_bits: int = 8
    alignment_bytes: int = 64
    schema_version: int = 1

    def __post_init__(self) -> None:
        values = (
            self.kv_heads,
            self.head_dim,
            self.mlen,
            self.block_size,
            self.element_bits,
            self.scale_bits,
            self.alignment_bytes,
        )
        if any(value <= 0 for value in values):
            raise ValueError("PackedKV dimensions and widths must be positive")
        if self.kv_heads * self.head_dim > self.mlen:
            raise ValueError("KV heads do not fit in one MLEN row")
        if self.head_dim % self.block_size:
            raise ValueError("head_dim must be divisible by the MX block size")
        if self.mlen % self.block_size:
            raise ValueError("MLEN must be divisible by the MX block size")
        if self.mlen % self.head_dim:
            raise ValueError("MLEN must contain an integral number of head slots")
        if self.schema_version != 1:
            raise ValueError(f"unsupported PackedKV schema {self.schema_version}")

    @property
    def active_elements(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def selector_count(self) -> int:
        return self.mlen // self.head_dim

    @property
    def logical_element_plane_bytes(self) -> int:
        return math.ceil(self.mlen * self.element_bits / 8)

    @property
    def logical_scale_plane_bytes(self) -> int:
        return math.ceil((self.mlen // self.block_size) * self.scale_bits / 8)

    @property
    def element_plane_bytes(self) -> int:
        return _align_up(self.logical_element_plane_bytes, self.alignment_bytes)

    @property
    def scale_plane_bytes(self) -> int:
        return _align_up(self.logical_scale_plane_bytes, self.alignment_bytes)

    @property
    def packed_row_bytes(self) -> int:
        return self.element_plane_bytes + self.scale_plane_bytes

    @property
    def padded_row_bytes(self) -> int:
        return self.kv_heads * self.packed_row_bytes

    @property
    def byte_reduction(self) -> float:
        return self.padded_row_bytes / self.packed_row_bytes

    @property
    def layout_id(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        return f"PACKED_KV-{hashlib.sha256(payload).hexdigest()[:16]}"

    def selector_offset_elements(self, selector: int) -> int:
        if selector < 0 or selector >= self.kv_heads:
            raise ValueError(
                f"KV selector {selector} outside active range [0,{self.kv_heads})"
            )
        return selector * self.head_dim

    def value_row_offset_elements(self, token_index: int, selector: int = 0) -> int:
        if token_index < 0:
            raise ValueError("token index must be non-negative")
        return token_index * self.mlen + self.selector_offset_elements(selector)

    def pack_token(self, heads: Sequence[Sequence[float]]) -> tuple[float, ...]:
        if len(heads) != self.kv_heads:
            raise ValueError(f"expected {self.kv_heads} KV heads")
        row: list[float] = []
        for head in heads:
            if len(head) != self.head_dim:
                raise ValueError(f"each KV head must contain {self.head_dim} elements")
            row.extend(float(value) for value in head)
        row.extend(0.0 for _ in range(self.mlen - len(row)))
        return tuple(row)

    def unpack_token(self, row: Sequence[float]) -> tuple[tuple[float, ...], ...]:
        if len(row) != self.mlen:
            raise ValueError(f"packed row must contain exactly {self.mlen} elements")
        return tuple(
            tuple(float(value) for value in row[start : start + self.head_dim])
            for start in range(0, self.active_elements, self.head_dim)
        )

    def physical_bytes(
        self,
        *,
        tokens: int,
        tensors: int = 2,
        mode: PackedKVAblation = PackedKVAblation.DENSE_SELECTOR,
    ) -> int:
        if tokens < 0 or tensors <= 0:
            raise ValueError("tokens must be non-negative and tensors positive")
        row_bytes = (
            self.padded_row_bytes
            if mode == PackedKVAblation.PADDED_PER_HEAD
            else self.packed_row_bytes
        )
        return tokens * tensors * row_bytes

    def ablation_metrics(self, *, tokens: int, tensors: int = 2) -> dict[str, dict]:
        packed = self.physical_bytes(tokens=tokens, tensors=tensors)
        padded = self.physical_bytes(
            tokens=tokens,
            tensors=tensors,
            mode=PackedKVAblation.PADDED_PER_HEAD,
        )
        return {
            PackedKVAblation.PADDED_PER_HEAD.value: {
                "physical_bytes": padded,
                "selector_enabled": False,
                "repack_required": False,
            },
            PackedKVAblation.DENSE_COMPILER.value: {
                "physical_bytes": packed,
                "selector_enabled": False,
                "repack_required": True,
            },
            PackedKVAblation.DENSE_SELECTOR.value: {
                "physical_bytes": packed,
                "selector_enabled": True,
                "repack_required": False,
            },
            PackedKVAblation.IDEAL_TRAFFIC.value: {
                "physical_bytes": packed,
                "selector_enabled": True,
                "repack_required": False,
            },
        }


@dataclass(frozen=True)
class PackedKVAppendAddress:
    """Exact element and scale-plane addresses for one tail append."""

    token_index: int
    transfer_rows: int
    element_offset_bytes: int
    scale_offset_bytes: int
    element_plane_bytes: int
    element_address: int
    scale_address: int
    element_transfer_bytes: int
    scale_transfer_bytes: int


def resolve_packed_kv_append(
    cache_layout: MatrixBlockLayout,
    packed_layout: PackedKVLayout,
    *,
    token_index: int,
    transfer_rows: int,
) -> PackedKVAppendAddress:
    """Resolve a tail write without rebasing its independent scale plane."""

    if not isinstance(cache_layout, MatrixBlockLayout):
        raise TypeError("cache_layout must be a MatrixBlockLayout")
    if not isinstance(packed_layout, PackedKVLayout):
        raise TypeError("packed_layout must be a PackedKVLayout")
    if isinstance(token_index, bool) or not isinstance(token_index, int):
        raise TypeError("token_index must be an integer")
    if isinstance(transfer_rows, bool) or not isinstance(transfer_rows, int):
        raise TypeError("transfer_rows must be an integer")
    if token_index < 0 or transfer_rows <= 0:
        raise ValueError("append position and transfer size are invalid")

    physical_rows, physical_columns = (
        cache_layout.physical_shape or cache_layout.full_shape
    )
    if physical_columns != packed_layout.mlen:
        raise ValueError("cache physical width differs from PackedKV MLEN")
    if cache_layout.hbm_element_width != packed_layout.element_bits:
        raise ValueError("cache element width differs from PackedKV precision")
    if cache_layout.hbm_block_size != packed_layout.block_size:
        raise ValueError("cache block size differs from PackedKV precision")
    if cache_layout.hbm_scale_width != packed_layout.scale_bits:
        raise ValueError("cache scale width differs from PackedKV precision")
    if packed_layout.element_bits not in {2, 4, 8}:
        raise ValueError("PackedKV append requires 2-, 4-, or 8-bit elements")
    if token_index + transfer_rows > physical_rows:
        raise ValueError("append transfer exceeds the cache allocation")

    logical_element_offset = token_index * physical_columns
    transfer_elements = transfer_rows * physical_columns
    element_offset = cache_layout.element_offset_bytes(
        logical_element_offset
    )
    scale_offset = cache_layout.scale_offset_bytes(
        logical_element_offset
    )
    element_scale_ratio = (
        cache_layout.hbm_element_width
        * cache_layout.hbm_block_size
        // cache_layout.hbm_scale_width
    )
    if (
        element_scale_ratio <= 0
        or cache_layout.hbm_element_width * cache_layout.hbm_block_size
        != element_scale_ratio * cache_layout.hbm_scale_width
        or element_offset % element_scale_ratio
        or element_offset // element_scale_ratio != scale_offset
    ):
        raise ValueError("PackedKV element and scale offsets are incompatible")
    element_bytes = cache_layout.element_span_bytes(transfer_elements)
    scale_bits = (
        transfer_elements
        // cache_layout.hbm_block_size
        * cache_layout.hbm_scale_width
    )
    if scale_bits % 8:
        raise ValueError("append scale span must be byte aligned")
    scale_bytes = scale_bits // 8
    element_plane_bytes = cache_layout.element_plane_bytes
    scale_plane_bytes = cache_layout.scale_plane_bytes
    if element_offset + element_bytes > element_plane_bytes:
        raise ValueError("append element span exceeds its cache plane")
    if scale_offset + scale_bytes > scale_plane_bytes:
        raise ValueError("append scale span exceeds its cache plane")
    return PackedKVAppendAddress(
        token_index=token_index,
        transfer_rows=transfer_rows,
        element_offset_bytes=element_offset,
        scale_offset_bytes=scale_offset,
        element_plane_bytes=element_plane_bytes,
        element_address=cache_layout.hbm_base_addr + element_offset,
        scale_address=(
            cache_layout.hbm_base_addr
            + element_plane_bytes
            + scale_offset
        ),
        element_transfer_bytes=element_bytes,
        scale_transfer_bytes=scale_bytes,
    )


def validate_selector_lowering(
    layout: PackedKVLayout,
    *,
    mlen: int,
    kv_heads: int,
    head_dim: int,
    batch_size: int,
) -> None:
    """Validate the dense-selector compiler contract."""
    if not isinstance(layout, PackedKVLayout):
        raise TypeError("layout must be a PackedKVLayout")
    if layout.block_size != 8:
        raise ValueError("PackedKV deployment lowering requires block_size=8")
    if layout.element_bits not in {2, 4, 8}:
        raise ValueError("PackedKV deployment lowering requires 2-, 4-, or 8-bit elements")
    if layout.mlen != mlen:
        raise ValueError(f"PackedKV MLEN={layout.mlen} does not match compiler MLEN={mlen}")
    if layout.kv_heads != kv_heads:
        raise ValueError(f"PackedKV has {layout.kv_heads} heads, expected {kv_heads}")
    if layout.head_dim != head_dim:
        raise ValueError(f"PackedKV head_dim={layout.head_dim} does not match HLEN={head_dim}")
    selector_limit = 1 << _PACKED_HEAD_SELECTOR_BITS
    if layout.kv_heads > selector_limit:
        raise ValueError(
            "local PackedKV selector count exceeds the 4-bit M_BTMM field: "
            f"local_kv_heads={layout.kv_heads}, selector_limit={selector_limit}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment
