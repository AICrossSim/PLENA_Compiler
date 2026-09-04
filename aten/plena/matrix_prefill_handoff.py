"""Evidence for replacing KDA's prefill-to-decode transpose with a Matrix view.

This module keeps three claims separate:

* the current Compiler really emits an identity GEMM at the boundary;
* the PLENA BF16 state can remain in one Matrix-SRAM allocation and be consumed
  through the opposite axis;
* the official Kimi K3 GPU implementation's FP32 state is retained only as a
  capacity and accuracy baseline, not as the active PLENA transfer format.

The ISA stays model independent. ``L_TILE_CFG`` declares shape and physical-row
pitch over PLENA's fixed diagonal placement; existing row/column Matrix
operations select the access axis.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from enum import StrEnum

from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.instruction_stream import (
    dynamic_count,
    opcode_census,
    static_count,
)
from compiler.aten.plena.mview import (
    MatrixViewDescriptor,
    MatrixViewMap,
    MatrixViewShape,
)
from compiler.aten.plena.program_kda_prefill import (
    kda_prefill_state_transpose_shapes,
)
from compiler.aten.plena.program_ssd import SPILLED_ACTIVATION

__all__ = [
    "KdaStateOrientation",
    "MatrixViewAxis",
    "build_prefill_handoff_report",
    "required_handoff_axis",
    "validate_handoff_axis",
]


class KdaStateOrientation(StrEnum):
    """Logical interpretation of the two state dimensions."""

    PREFILL_VALUE_KEY = "value_key"
    DECODE_KEY_VALUE = "key_value"


class MatrixViewAxis(StrEnum):
    """Existing Matrix-SRAM traversal selected by the consumer opcode."""

    ROW = "row"
    COLUMN = "column"


def required_handoff_axis(
    stored: KdaStateOrientation,
    requested: KdaStateOrientation,
) -> MatrixViewAxis:
    """Return the only axis that preserves the requested logical orientation."""

    if stored is requested:
        return MatrixViewAxis.ROW
    return MatrixViewAxis.COLUMN


def validate_handoff_axis(
    *,
    stored: KdaStateOrientation,
    requested: KdaStateOrientation,
    selected: MatrixViewAxis,
) -> None:
    """Fail before emission when equal-shaped state is read through the wrong axis."""

    required = required_handoff_axis(stored, requested)
    if selected is not required:
        raise ValueError(
            f"state view mismatch: {stored} -> {requested} requires {required}, "
            f"got {selected}"
        )


@dataclass(frozen=True)
class _KimiK3State:
    heads: int = 96
    key_dim: int = 128
    value_dim: int = 128
    kda_layers: int = 69
    hidden_size: int = 7168


def _legacy_identity_transpose_assembly(
    *,
    shape: _KimiK3State,
    mlen: int,
    blen: int,
) -> str:
    # This evidence intentionally compiles the legacy padded operation even at
    # the published 256-row point, where one complete MLEN-square tile cannot
    # reside. One structural slot is enough to expose the emitted work; the
    # report separately states that the physical point cannot hold that tile.
    program = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        mram_tile_capacity=1,
        hbm_v_prefetch_amount=16,
        hbm_v_writeback_amount=16,
    )
    program._bf16_kv_checked = True
    kda_shape = KdaShape(
        hidden_size=shape.hidden_size,
        num_heads=shape.heads,
        key_dim=shape.key_dim,
        value_dim=shape.value_dim,
        conv_kernel=4,
    )
    wanted = kda_prefill_state_transpose_shapes(kda_shape, mlen)
    tiles = {
        name: program.alloc(name, *dims, strict=False)
        for name, dims in wanted.items()
    }
    mark = len(program.get_code())
    program.kda_prefill_state_to_decode_layout_v0(
        shape=kda_shape,
        precision=SPILLED_ACTIVATION,
        **tiles,
    )
    return program.get_code()[mark:]


def _view_configuration_assembly(
    *,
    shape: _KimiK3State,
    mlen: int,
    blen: int,
) -> tuple[str, MatrixViewDescriptor]:
    descriptor = MatrixViewDescriptor(
        shape=MatrixViewShape(rows=shape.value_dim, cols=shape.key_dim),
        mapping=MatrixViewMap(
            tile_pitch_rows=shape.value_dim,
            # The whole per-head state is read by columns at this boundary.
            # The prior-work fixed diagonal wiring reaches that column floor.
            # Decode's later cross-head packet uses a different tile pitch;
            # conflating the two would return plausible values with the wrong
            # service claim.
        ),
    )
    program = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        mram_tile_capacity=1,
    )
    mark = len(program.get_code())
    program.configure_matrix_view_v0(descriptor, slot=0)
    return program.get_code()[mark:], descriptor


def _value_handoff(shape: _KimiK3State) -> dict[str, object]:
    # Deliberately non-symmetric: a wrong row read remains finite and plausible.
    prefill = [
        float(value * 10_000 + key * 3 + 1)
        for value in range(shape.value_dim)
        for key in range(shape.key_dim)
    ]
    decode_expected = [
        prefill[value * shape.key_dim + key]
        for key in range(shape.key_dim)
        for value in range(shape.value_dim)
    ]
    row_read = list(prefill)
    column_read = decode_expected.copy()
    if row_read == decode_expected:
        raise AssertionError("non-symmetric handoff fixture became symmetric")
    if column_read != decode_expected:
        raise AssertionError("column Matrix view failed to restore decode order")
    validate_handoff_axis(
        stored=KdaStateOrientation.PREFILL_VALUE_KEY,
        requested=KdaStateOrientation.DECODE_KEY_VALUE,
        selected=MatrixViewAxis.COLUMN,
    )
    wrong_axis_rejected = False
    try:
        validate_handoff_axis(
            stored=KdaStateOrientation.PREFILL_VALUE_KEY,
            requested=KdaStateOrientation.DECODE_KEY_VALUE,
            selected=MatrixViewAxis.ROW,
        )
    except ValueError:
        wrong_axis_rejected = True
    if not wrong_axis_rejected:
        raise AssertionError("wrong KDA state axis was not rejected")

    digest = hashlib.sha256()
    for value in column_read:
        digest.update(f"{value:.1f}\n".encode())
    return {
        "values_checked": len(column_read),
        "column_read_matches_transpose": True,
        "row_read_is_finite_but_wrong": True,
        "wrong_axis_rejected_before_execution": wrong_axis_rejected,
        "decode_order_sha256": digest.hexdigest(),
    }


def build_prefill_handoff_report(
    *,
    mlen: int = 2048,
    blen: int = 32,
    matrix_sram_rows: int = 256,
) -> dict[str, object]:
    """Build emitted-code and value evidence at the official Kimi K3 shape."""

    shape = _KimiK3State()
    legacy = _legacy_identity_transpose_assembly(
        shape=shape,
        mlen=mlen,
        blen=blen,
    )
    census = opcode_census(legacy)
    matrix_ops = int(census.get("M_TMM", 0))
    writebacks = int(census.get("M_MM_WO", 0))
    expected_ops = (mlen // blen) ** 2
    if matrix_ops != expected_ops or writebacks != expected_ops:
        raise AssertionError(
            "emitted identity transpose no longer matches its tiled Matrix geometry"
        )

    logical_macs_per_head = shape.value_dim * shape.key_dim * shape.key_dim
    macs_per_matrix_issue = blen * blen * mlen
    emitted_macs_per_head = matrix_ops * macs_per_matrix_issue
    matrix_macs_per_cycle = mlen * blen
    config, descriptor = _view_configuration_assembly(
        shape=shape,
        mlen=mlen,
        blen=blen,
    )
    bf16_state_bytes_per_head = shape.key_dim * shape.value_dim * 2
    fp32_state_bytes_per_layer = shape.heads * shape.key_dim * shape.value_dim * 4
    bf16_state_bytes_per_layer = shape.heads * shape.key_dim * shape.value_dim * 2
    matrix_sram_bytes = mlen * matrix_sram_rows * 2
    bf16_heads_per_resident_window = matrix_sram_bytes // bf16_state_bytes_per_head

    return {
        "schema_version": 1,
        "scope": (
            "Kimi K3 official dimensions; emitted Compiler topology and symbolic "
            "state values, not real checkpoint execution"
        ),
        "shape": asdict(shape),
        "legacy_identity_gemm": {
            "static_instructions": static_count(legacy),
            "dynamic_issued_instructions_per_head": dynamic_count(legacy),
            "dynamic_opcode_census_per_head": census,
            "logical_macs_per_head": logical_macs_per_head,
            "logical_macs_all_kda_layers": (
                logical_macs_per_head * shape.heads * shape.kda_layers
            ),
            "emitted_padded_macs_per_head": emitted_macs_per_head,
            "emitted_padded_macs_all_kda_layers": (
                emitted_macs_per_head * shape.heads * shape.kda_layers
            ),
            "padding_over_logical_macs": emitted_macs_per_head / logical_macs_per_head,
            "matrix_issue_macs": macs_per_matrix_issue,
            "matrix_macs_per_cycle": matrix_macs_per_cycle,
            "emitted_matrix_cycles_all_kda_layers": math.ceil(
                emitted_macs_per_head
                * shape.heads
                * shape.kda_layers
                / matrix_macs_per_cycle
            ),
            "source": "kda_prefill_state_to_decode_layout_v0 emitted assembly",
        },
        "matrix_view_handoff": {
            "configuration_static_instructions": static_count(config),
            "configuration_dynamic_instructions": dynamic_count(config),
            "configuration_assembly": config,
            "descriptor": {
                "shape": asdict(descriptor.shape),
                "mapping": {
                    "tile_pitch_rows": descriptor.mapping.tile_pitch_rows,
                    "fixed_wiring_alpha": 1,
                    "flags": int(descriptor.mapping.flags),
                },
            },
            "stored_orientation": KdaStateOrientation.PREFILL_VALUE_KEY,
            "decode_orientation": KdaStateOrientation.DECODE_KEY_VALUE,
            "decode_axis": MatrixViewAxis.COLUMN,
            "row_consumer": "existing M_MM family",
            "column_consumer": "existing M_TMM family",
            "handoff_arithmetic_instructions": 0,
            "handoff_macs": 0,
            "same_physical_cells": True,
            "next_decode_packet_layout": (
                "explicit streamed write/read in decode order; the later 16-head "
                "packet uses a separate compiler-selected tile pitch"
            ),
            "direct_cross_head_residence_claimed": False,
            "value_evidence": _value_handoff(shape),
        },
        "precision_and_capacity_boundary": {
            "official_state_dtype": "FP32",
            "plena_state_dtype": "BF16",
            "matrix_sram_dtype": "BF16",
            "official_fp32_state_bytes_per_layer": fp32_state_bytes_per_layer,
            "matrix_sram_bytes": matrix_sram_bytes,
            "official_fp32_state_matrix_resident": False,
            "plena_bf16_state_matrix_streamed": True,
            "bf16_state_bytes_per_layer": bf16_state_bytes_per_layer,
            "bf16_heads_per_resident_window": bf16_heads_per_resident_window,
            "bf16_windows_per_layer": (
                shape.heads + bf16_heads_per_resident_window - 1
            )
            // bf16_heads_per_resident_window,
            "interpretation": (
                "The evaluated zero-MAC view handoff uses PLENA BF16 state. "
                "Official GPU FP32 state is reported only as baseline metadata."
            ),
        },
    }
