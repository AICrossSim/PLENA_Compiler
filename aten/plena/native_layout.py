"""Shared physical layout planning for native decoder compilation.

The native compiler and CostEmitter must agree on physical activation rows and
packed-GQA columns.  Keeping that arithmetic in one module prevents a cost
trace from silently modelling a layout that the emitted ISA cannot execute.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


NATIVE_LAYOUT_SCHEMA_VERSION = 3
NATIVE_LAYOUT_MODES = frozenset({"compact", "legacy"})


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if value <= 0 or multiple <= 0:
        raise ValueError(f"value and multiple must be positive, got {value}, {multiple}")
    return ((value + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class SequencePackingPlan:
    """Map logical ``[batch, sequence]`` rows into attention tile groups.

    Compact mode co-locates several short, independent sequences in one MLEN
    row slab.  A block-diagonal score mask preserves batch isolation.  The last
    group is completed with all-zero dummy batches, which keeps every group
    structurally identical and therefore loopable in ISA.
    """

    mode: str
    batch_size: int
    seq_len: int
    mlen: int
    batch_pack_factor: int
    padded_batch_size: int
    attention_group_count: int
    attention_group_seq_len: int
    rows_per_attention_group: int
    compile_seq_rows: int

    @classmethod
    def build(
        cls,
        *,
        batch_size: int,
        seq_len: int,
        mlen: int,
        mode: str = "compact",
    ) -> "SequencePackingPlan":
        if batch_size <= 0 or seq_len <= 0 or mlen <= 0:
            raise ValueError(
                "batch_size, seq_len, and mlen must be positive, got "
                f"{batch_size}, {seq_len}, {mlen}"
            )
        if mode not in NATIVE_LAYOUT_MODES:
            raise ValueError(
                f"native layout mode must be one of {sorted(NATIVE_LAYOUT_MODES)}, got {mode!r}"
            )

        if mode == "compact" and seq_len <= mlen:
            pack_factor = max(1, min(batch_size, mlen // seq_len))
        else:
            pack_factor = 1
        padded_batch_size = math.ceil(batch_size / pack_factor) * pack_factor
        group_count = padded_batch_size // pack_factor
        group_seq_len = pack_factor * seq_len
        rows_per_group = _ceil_to_multiple(max(mlen, group_seq_len), mlen)
        return cls(
            mode=mode,
            batch_size=batch_size,
            seq_len=seq_len,
            mlen=mlen,
            batch_pack_factor=pack_factor,
            padded_batch_size=padded_batch_size,
            attention_group_count=group_count,
            attention_group_seq_len=group_seq_len,
            rows_per_attention_group=rows_per_group,
            compile_seq_rows=group_count * rows_per_group,
        )

    @property
    def dummy_batch_count(self) -> int:
        return self.padded_batch_size - self.batch_size

    @property
    def logical_active_rows(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def packed_active_rows(self) -> int:
        return self.padded_batch_size * self.seq_len

    @property
    def row_utilization(self) -> float:
        return self.logical_active_rows / self.compile_seq_rows

    @property
    def mask_kind(self) -> str:
        return "block_diagonal_causal" if self.batch_pack_factor > 1 else "causal"

    def physical_row(self, batch_idx: int, token_idx: int) -> int:
        if batch_idx < 0 or batch_idx >= self.batch_size:
            raise IndexError(f"batch_idx={batch_idx} outside [0, {self.batch_size})")
        if token_idx < 0 or token_idx >= self.seq_len:
            raise IndexError(f"token_idx={token_idx} outside [0, {self.seq_len})")
        group_idx, slot_idx = divmod(batch_idx, self.batch_pack_factor)
        return (
            group_idx * self.rows_per_attention_group
            + slot_idx * self.seq_len
            + token_idx
        )

    def active_physical_rows(self) -> tuple[int, ...]:
        return tuple(
            self.physical_row(batch_idx, token_idx)
            for batch_idx in range(self.batch_size)
            for token_idx in range(self.seq_len)
        )

    def active_row_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return maximal half-open ranges containing real token rows.

        Compact short-sequence layouts place all real batch slots at the
        beginning of each attention slab, so one range covers each populated
        slab.  Legacy and long-sequence layouts naturally reduce to one range
        per logical batch.  Consumers must retain ``compile_seq_rows`` as the
        column-block stride; these ranges only select rows on which arithmetic
        is required.
        """

        ranges: list[tuple[int, int]] = []
        remaining_batches = self.batch_size
        for group_idx in range(self.attention_group_count):
            real_slots = min(self.batch_pack_factor, remaining_batches)
            if real_slots <= 0:
                break
            start = group_idx * self.rows_per_attention_group
            end = start + real_slots * self.seq_len
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], end)
            else:
                ranges.append((start, end))
            remaining_batches -= real_slots
        return tuple(ranges)

    def metadata(self) -> dict[str, int | float | str]:
        return {
            "schema_version": NATIVE_LAYOUT_SCHEMA_VERSION,
            "mode": self.mode,
            "logical_batch_size": self.batch_size,
            "padded_batch_size": self.padded_batch_size,
            "dummy_batch_count": self.dummy_batch_count,
            "seq_len": self.seq_len,
            "batch_pack_factor": self.batch_pack_factor,
            "attention_group_count": self.attention_group_count,
            "attention_group_seq_len": self.attention_group_seq_len,
            "rows_per_attention_group": self.rows_per_attention_group,
            "logical_active_rows": self.logical_active_rows,
            "physical_rows": self.compile_seq_rows,
            "active_row_ranges": [list(row_range) for row_range in self.active_row_ranges()],
            "row_utilization": self.row_utilization,
            "mask_kind": self.mask_kind,
        }


@dataclass(frozen=True)
class AttentionHeadPacking:
    """Physical packed-GQA storage and execution mapping.

    A logical KV/chunk group consumes only
    ``physical_broadcast * head_slot_dim`` active columns. Compact mode places
    several groups in one MLEN storage block. Matrix SRAM addresses are MLEN
    aligned, so an attention operation broadcasts its one KV head across every
    lane in that block and consumes only the score lanes assigned to the target
    logical group. It never combines different KV heads in one operation.
    """

    enabled: bool
    hlen: int
    broadcast_amount: int
    head_slot_dim: int
    group_width: int
    total_q_dim: int
    active_head_dim: int | None = None
    chunks_per_kv: int = 1
    logical_broadcast_amount: int | None = None
    logical_group_count: int = 1
    attention_group_width: int | None = None
    groups_per_storage_block: int = 1
    storage_block_count: int = 1
    compact: bool = False

    @property
    def heads_per_storage_block(self) -> int:
        """Number of Q/O head slots stored in one physical MLEN block."""
        return self.groups_per_storage_block * self.broadcast_amount

    @property
    def hardware_broadcast_amount(self) -> int:
        """Broadcast lanes used by one attention operation.

        Vector SRAM reads are VLEN-aligned, so a later group slot cannot be
        selected through an unaligned Q base.  The current KV head is therefore
        replicated across every stored lane and only the target group's result
        lanes are retained.  Different KV heads are never mixed.
        """
        return self.heads_per_storage_block

    @property
    def head_lane_utilization(self) -> float:
        """Fraction of allocated Q/O storage columns carrying logical heads."""
        active = self.logical_group_count * (self.attention_group_width or self.group_width)
        return active / self.total_q_dim

    @property
    def execution_head_lane_utilization(self) -> float:
        """Useful lanes in an aligned storage-block broadcast operation."""
        return self.broadcast_amount / self.hardware_broadcast_amount

    def group_location(self, group_idx: int) -> tuple[int, int]:
        if group_idx < 0 or group_idx >= self.logical_group_count:
            raise IndexError(
                f"group_idx={group_idx} outside [0, {self.logical_group_count})"
            )
        block_idx, slot_idx = divmod(group_idx, self.groups_per_storage_block)
        slot_width = self.attention_group_width or self.group_width
        return block_idx, slot_idx * slot_width

    def group_start_col(self, group_idx: int) -> int:
        block_idx, slot_offset = self.group_location(group_idx)
        return block_idx * self.group_width + slot_offset

    def head_start_col(self, *, kv_head: int, local_head: int) -> int:
        if local_head < 0:
            raise IndexError(f"local_head must be nonnegative, got {local_head}")
        chunk, lane = divmod(local_head, self.broadcast_amount)
        group_idx = kv_head * self.chunks_per_kv + chunk
        return self.group_start_col(group_idx) + lane * self.head_slot_dim

    def metadata(self) -> dict[str, int | float | bool | None]:
        return {
            "logical_broadcast_amount": self.logical_broadcast_amount,
            "physical_broadcast_amount": self.broadcast_amount,
            "hardware_broadcast_amount": self.hardware_broadcast_amount,
            "stored_heads_per_block": self.heads_per_storage_block,
            "logical_group_count": self.logical_group_count,
            "attention_group_width": self.attention_group_width,
            "groups_per_storage_block": self.groups_per_storage_block,
            "storage_block_count": self.storage_block_count,
            "storage_block_width": self.group_width,
            "total_q_dim": self.total_q_dim,
            "head_lane_utilization": self.head_lane_utilization,
            "storage_head_lane_utilization": self.head_lane_utilization,
            "execution_head_lane_utilization": self.execution_head_lane_utilization,
            "compact": self.compact,
        }


def build_attention_head_packing(
    *,
    mlen: int,
    hlen: int,
    head_dim: int,
    logical_broadcast_amount: int,
    gqa_ratio: int,
    num_kv_heads: int,
    mode: str = "compact",
) -> AttentionHeadPacking:
    if mode not in NATIVE_LAYOUT_MODES:
        raise ValueError(
            f"native layout mode must be one of {sorted(NATIVE_LAYOUT_MODES)}, got {mode!r}"
        )
    values = {
        "mlen": mlen,
        "hlen": hlen,
        "head_dim": head_dim,
        "logical_broadcast_amount": logical_broadcast_amount,
        "gqa_ratio": gqa_ratio,
        "num_kv_heads": num_kv_heads,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"attention packing values must be positive, got {values}")
    if hlen < head_dim:
        raise ValueError(f"HLEN={hlen} is smaller than head_dim={head_dim}")
    if mlen < hlen:
        raise ValueError(f"MLEN={mlen} is smaller than HLEN={hlen}")

    physical_broadcast = min(logical_broadcast_amount, mlen // hlen)
    chunks_per_kv = math.ceil(gqa_ratio / physical_broadcast)
    logical_group_count = num_kv_heads * chunks_per_kv
    active_group_width = physical_broadcast * hlen
    groups_per_block = mlen // active_group_width if mode == "compact" else 1
    groups_per_block = max(1, groups_per_block)
    storage_blocks = math.ceil(logical_group_count / groups_per_block)
    return AttentionHeadPacking(
        enabled=True,
        hlen=hlen,
        logical_broadcast_amount=logical_broadcast_amount,
        broadcast_amount=physical_broadcast,
        head_slot_dim=hlen,
        group_width=mlen,
        total_q_dim=storage_blocks * mlen,
        active_head_dim=head_dim,
        chunks_per_kv=chunks_per_kv,
        logical_group_count=logical_group_count,
        attention_group_width=active_group_width,
        groups_per_storage_block=groups_per_block,
        storage_block_count=storage_blocks,
        compact=mode == "compact",
    )
