"""Lowering for the final hidden-to-vocabulary projection (LM head)."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from collections.abc import Mapping

from .projection_asm import projection_T_asm

# The LM head reuses the checkpoint's native ``lm_head.weight`` layout: row-major
# ``(vocab_size, hidden_size)``, so row ``v`` holds the full hidden-dimension
# vector for vocabulary entry ``v`` and the HBM row stride is ``hidden_size``.
# ``logits = hidden @ lm_head.weight.T`` therefore lowers to the transposed
# projection, which streams one weight-row group at a time and needs no
# transpose pass over the 151k-row weight matrix.
HBM_WEIGHT_LAYOUT = "row_major_vocab_by_hidden"
LOCAL_LM_HEAD_RECEIPT_SCHEMA = "plena-local-lm-head-lowering/v1"
LOCAL_LM_HEAD_SELECTION = {
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "score_format": "FP32",
    "token_id_format": "UINT32",
    "argmax_tie_rule": "lowest_token_id",
}

_MATRIX_FORMAT_BITS = {
    "MXINT2": 2,
    "MXINT4": 4,
    "MXINT8": 8,
    "E1M2": 4,
    "E2M1": 4,
    "E3M4": 8,
    "E4M3": 8,
    "E5M2": 8,
}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def lm_head_physical_batch(batch: int, blen: int) -> int:
    """Return the zero-padded batch rows consumed by the matrix unit."""

    if batch <= 0 or blen <= 0:
        raise ValueError("batch and blen must be positive")
    return _ceil_div(batch, blen) * blen


def lm_head_hidden_padding(hidden_size: int, mlen: int) -> int:
    """Return the zero-padded reduction width for one MLEN partition."""

    if hidden_size <= 0 or mlen <= 0:
        raise ValueError("hidden_size and mlen must be positive")
    return _ceil_div(hidden_size, mlen) * mlen


def lm_head_vocab_padding(vocab_size: int, blen: int, mlen: int) -> int:
    """Return the physical vocabulary rows required by matrix prefetch.

    ``projection_T_asm`` prefetches an entire ``MLEN x MLEN`` weight tile for
    each output-row group.  Padding only to ``BLEN`` lets the final prefetch
    cross the LM-head allocation whenever the vocabulary is not MLEN-aligned.
    MLEN is therefore the physical allocation boundary; it is also a BLEN
    boundary because the matrix geometry requires ``MLEN % BLEN == 0``.
    """

    if min(vocab_size, blen, mlen) <= 0:
        raise ValueError("vocab_size, blen, and mlen must be positive")
    if mlen % blen:
        raise ValueError("LM head requires MLEN divisible by BLEN")
    return _ceil_div(vocab_size, mlen) * mlen


def lm_head_native_row_element_offset(token_id: int, hidden_size: int, physical_vocab: int) -> int:
    """Return a token row offset using the physical hidden-row stride."""

    if hidden_size <= 0 or physical_vocab <= 0:
        raise ValueError("hidden_size and physical_vocab must be positive")
    if not 0 <= token_id < physical_vocab:
        raise ValueError(
            f"token_id {token_id} is outside physical vocabulary [0, {physical_vocab})"
        )
    return token_id * hidden_size


def _format_bits(format_name: str) -> int:
    try:
        return _MATRIX_FORMAT_BITS[str(format_name).upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported local LM-head matrix format {format_name!r}") from exc


def _plane_bytes(bits: int, hbm_row_bits: int) -> int:
    return _ceil_div(bits, hbm_row_bits) * (hbm_row_bits // 8)


def local_lm_head_lowering_receipt(
    *,
    mlen: int,
    blen: int,
    batch: int,
    hidden_size: int,
    vocab_size: int,
    profile: Mapping[str, Any],
    profile_id: str,
    hbm_row_bits: int = 512,
    tensor_parallel_ranks: int = 1,
) -> dict[str, Any]:
    """Describe the exact local-head work and the still-missing serving path.

    This is deliberately a fail-closed compiler receipt.  It gives the hardware
    and analytic models exact tensor bytes and structural event counts, but does
    not turn those counts into calibrated cycles and does not claim that the
    current full-logit lowering implements bounded streaming selection.
    """

    if min(
        mlen,
        blen,
        batch,
        hidden_size,
        vocab_size,
        hbm_row_bits,
        tensor_parallel_ranks,
    ) <= 0:
        raise ValueError("local LM-head geometry must be positive")
    if mlen % blen:
        raise ValueError("local LM head requires MLEN divisible by BLEN")
    if hbm_row_bits % 8:
        raise ValueError("hbm_row_bits must be byte aligned")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("local LM-head receipt requires a non-empty profile_id")

    required_profile_fields = {
        "weight_format",
        "activation_format",
        "vector_format",
        "matrix_mlen",
        "block_size",
        "scale_format",
        "scale_bits",
    }
    missing = sorted(required_profile_fields - set(profile))
    if missing:
        raise ValueError(f"local LM-head profile is missing fields: {missing}")
    block_size = int(profile["block_size"])
    scale_bits = int(profile["scale_bits"])
    profile_matrix_mlen = int(profile["matrix_mlen"])
    if profile_matrix_mlen != mlen:
        raise ValueError(
            "local LM-head receipt MLEN differs from the numerical profile"
        )
    if block_size != 8:
        raise ValueError("local LM head currently requires MX block_size=8")
    if str(profile["scale_format"]).upper() != "E8M0" or scale_bits != 8:
        raise ValueError("local LM head currently requires an 8-bit E8M0 scale")

    weight_format = str(profile["weight_format"]).upper()
    activation_format = str(profile["activation_format"]).upper()
    vector_format = str(profile["vector_format"]).upper()
    weight_bits = _format_bits(weight_format)
    _format_bits(activation_format)

    physical_vocab = lm_head_vocab_padding(vocab_size, blen, mlen)
    physical_hidden = lm_head_hidden_padding(hidden_size, mlen)
    physical_batch = lm_head_physical_batch(batch, blen)
    physical_weight_elements = physical_vocab * physical_hidden
    if physical_weight_elements % block_size:
        raise ValueError("physical LM-head weight elements must fill MX blocks")
    scale_count = physical_weight_elements // block_size
    data_bytes = _plane_bytes(physical_weight_elements * weight_bits, hbm_row_bits)
    scale_bytes = _plane_bytes(scale_count * scale_bits, hbm_row_bits)

    output_tiles = physical_vocab // blen
    batch_tiles = physical_batch // blen
    reduction_tiles = physical_hidden // mlen
    prefetches = (physical_vocab // mlen) * reduction_tiles
    matrix_issues = output_tiles * batch_tiles * reduction_tiles
    matrix_writeouts = output_tiles * batch_tiles

    topk_state_bytes_per_row = LOCAL_LM_HEAD_SELECTION["top_k"] * (4 + 4)
    argmax_state_bytes_per_row = 4 + 4
    full_logits_bf16_bytes = batch * vocab_size * 2
    physical_full_logits_bf16_bytes = physical_batch * physical_vocab * 2
    streamed_matrix_tile_bf16_bytes = blen * mlen * 2
    numerical_identity = {
        "profile_id": profile_id,
        "mlen": mlen,
        "blen": blen,
        "active_hidden_size": hidden_size,
        "physical_hidden_size": physical_hidden,
        "weight_format": weight_format,
        "activation_format": activation_format,
        "vector_format": vector_format,
        "block_size": block_size,
        "scale_format": "E8M0",
        "scale_bits": scale_bits,
        "partial_conversion": "round_each_mlen_partial_to_profile.vector_format",
        "cross_instruction_accumulator": "signed_fixed16_16_wraparound",
    }

    receipt: dict[str, Any] = {
        "schema_version": LOCAL_LM_HEAD_RECEIPT_SCHEMA,
        "operation": "decode_lm_head",
        "profile_id": profile_id,
        "profile_sha256": hashlib.sha256(_canonical_json_bytes(profile)).hexdigest(),
        "matrix_geometry": {"mlen": mlen, "blen": blen},
        "numerical_identity": numerical_identity,
        "numerical_identity_sha256": hashlib.sha256(
            _canonical_json_bytes(numerical_identity)
        ).hexdigest(),
        "model_geometry": {
            "hidden_size": hidden_size,
            "physical_hidden_size": physical_hidden,
            "zero_padded_hidden_columns": physical_hidden - hidden_size,
            "active_vocab": vocab_size,
            "physical_vocab": physical_vocab,
            "masked_vocab_rows": physical_vocab - vocab_size,
            "active_batch": batch,
            "physical_batch": physical_batch,
            "zero_padded_batch_rows": physical_batch - batch,
            "zero_padded_activation_rows_required": True,
        },
        "weight_layout": {
            "semantic_layout": HBM_WEIGHT_LAYOUT,
            "logical_shape": [vocab_size, hidden_size],
            "physical_shape": [physical_vocab, physical_hidden],
            "valid_token_ids": [0, vocab_size - 1],
            "row_element_offset": "token_id * physical_hidden_size",
            "padded_rows_are_zero": True,
            "padded_hidden_columns_are_zero": True,
            "hbm_data_then_scale_planes": True,
        },
        "profile": {
            "weight_format": weight_format,
            "activation_format": activation_format,
            "vector_format": vector_format,
            "block_size": block_size,
            "scale_format": "E8M0",
            "scale_bits": scale_bits,
        },
        "numeric_semantics": {
            "matrix_k_partition": "MLEN",
            "partial_conversion": "round_each_mlen_partial_to_profile.vector_format",
            "partial_rounding": "round_to_nearest_even_to_profile.vector_format",
            "cross_instruction_accumulator": "signed_fixed16_16_wraparound",
            "matrix_numeric_format": "profile.vector_format",
            "matrix_storage_format": "profile.vector_format",
            "matrix_writeout_numeric_format": "profile.vector_format",
            "logit_container_format": "BF16",
            "container_conversion": (
                "identity" if vector_format == "BF16" else
                "exact_widen_profile.vector_format_value_to_BF16_container"
            ),
            "bf16_reference_writeout_rounding": "mantissa_truncation",
            "no_per_stage_bf16_matrix_switch": True,
            "precision_recovery": False,
            "compiler_numerical_parity_requires_same_profile_id_and_mlen": True,
        },
        "hbm_footprint": {
            "logical_weight_elements": vocab_size * hidden_size,
            "physical_weight_elements": physical_weight_elements,
            "weight_element_bits": weight_bits,
            "weight_data_bytes": data_bytes,
            "scale_count": scale_count,
            "scale_bytes": scale_bytes,
            "total_weight_hbm_bytes": data_bytes + scale_bytes,
            "hbm_row_bits": hbm_row_bits,
        },
        "structural_events": {
            "weight_prefetches": prefetches,
            "prefetched_weight_elements": prefetches * mlen * mlen,
            "matrix_issues": matrix_issues,
            "matrix_writeouts": matrix_writeouts,
            "output_tiles": output_tiles,
            "batch_tiles": batch_tiles,
            "reduction_tiles": reduction_tiles,
            "calibrated_matrix_cycles": None,
            "calibrated_vector_cycles": None,
            "calibrated_argmax_cycles": None,
        },
        "selection": {
            **LOCAL_LM_HEAD_SELECTION,
            "full_logit_materialization_allowed": False,
            "offline_nll_full_logit_materialization_only": True,
            "full_logits_bf16_bytes": full_logits_bf16_bytes,
            "physical_full_logits_bf16_bytes": physical_full_logits_bf16_bytes,
            "streamed_matrix_tile_bf16_bytes": streamed_matrix_tile_bf16_bytes,
            "topk_state_bytes_per_active_row": topk_state_bytes_per_row,
            "topk_state_bytes_active_batch": batch * topk_state_bytes_per_row,
            "topk_state_bytes_physical_batch": physical_batch * topk_state_bytes_per_row,
            "argmax_state_bytes_per_active_row": argmax_state_bytes_per_row,
            "argmax_state_bytes_active_batch": batch * argmax_state_bytes_per_row,
            "argmax_state_bytes_physical_batch": physical_batch * argmax_state_bytes_per_row,
            "padded_vocab_score": "negative_infinity",
            "tensor_parallel": {
                "ranks": tensor_parallel_ranks,
                "current_projection_layout": "unsharded_single_rank",
                "required_local_candidates_per_rank_per_row": 20,
                "required_global_merge_candidates_per_row": (
                    tensor_parallel_ranks * LOCAL_LM_HEAD_SELECTION["top_k"]
                ),
                "required_global_merge_candidate_bytes_active_batch": (
                    batch
                    * tensor_parallel_ranks
                    * LOCAL_LM_HEAD_SELECTION["top_k"]
                    * (4 + 4)
                ),
                "merge_order": "score_descending_then_global_token_id_ascending",
                "sample_owner": "one_designated_rank_after_global_merge",
                "selected_token_broadcast": True,
                "global_merge_lowered": False,
                "token_broadcast_lowered": False,
            },
        },
        "validity": {
            "native_weight_layout_valid": True,
            "mlen_vocab_padding_valid": True,
            "batch_padding_contract_valid": True,
            "batch_padding_zero_fill_lowered": False,
            "hidden_padding_contract_valid": True,
            "hidden_padding_zero_fill_lowered": False,
            "padded_vocab_mask_lowered": False,
            "projection_structure_lowered": True,
            "profile_aware_weight_staging_valid": False,
            "legacy_global_precision_address_register_init_structural": True,
            "lm_head_address_register_init_valid": False,
            "streaming_selector_lowered": False,
            "multi_tp_serving_valid": False,
            "serving_compiler_valid": False,
            "compiler_generated_binary_emulator_parity_valid": False,
            "cycle_calibrated": False,
            "rtl_validated": False,
            "publication_valid": False,
        },
        "blockers": [
            "lm_head address init is still sized from the global hardware config rather than the selected profile",
            "generator stager is bound to global TOML precision rather than profile W/A",
            "no explicit zero-fill lowering exists for physical batch rows beyond active_batch",
            "no explicit zero-fill lowering exists for physical hidden columns beyond hidden_size",
            "no serving selector currently masks padded vocabulary rows to negative infinity",
            "projection_T_asm currently materializes batch_by_vocab logits instead of consuming each tile",
            "no bounded streaming top-k20/top-p/argmax selector opcode or lowering exists",
            "no deterministic TP-local-top20 global merge/sample-owner/token-broadcast lowering exists",
            "decode-precision-profile/v2 local_head binding is not yet accepted by the compiler hook",
            "matrix/vector/argmax event counts have no calibrated cycle mapping in this receipt",
        ],
    }
    receipt["contract_sha256"] = hashlib.sha256(_canonical_json_bytes(receipt)).hexdigest()
    return receipt


def lm_head_asm(
    mlen: int,
    blen: int,
    batch: int,
    hidden_size: int,
    vocab_size: int,
    alive_registers: list[int],
    lm_head_weight_hbm_offset_reg: int,
    activation_base_address: int,
    result_base_address: int,
) -> str:
    """Generate assembly for the hidden-to-vocabulary projection.

    Computes ``logits = hidden_states @ lm_head.weight.T``, mapping
    ``(batch, hidden_size) @ (vocab_size, hidden_size).T -> (batch, vocab_size)``.

    Args:
        mlen: Matrix tile length; must divide ``hidden_size``.
        blen: Batch/block tile length.
        batch: Decode batch size.
        hidden_size: Model hidden dimension, the reduction dimension K.
        vocab_size: Active vocabulary size N. The physical HBM allocation is
            padded up to a multiple of ``mlen``.
        alive_registers: At least six free general-purpose register indices.
        lm_head_weight_hbm_offset_reg: Address register holding the HBM base of
            ``lm_head.weight``.
        activation_base_address: Vector SRAM base of the final hidden states.
        result_base_address: Vector SRAM base for the emitted logits.

    Returns:
        Assembly text for the projection.
    """
    if len(alive_registers) < 6:
        raise ValueError(
            "lm_head_asm requires at least 6 alive registers "
            f"(got {len(alive_registers)})"
        )
    padded_vocab = lm_head_vocab_padding(vocab_size, blen, mlen)
    padded_hidden = lm_head_hidden_padding(hidden_size, mlen)

    header = [
        "; === LM head: hidden -> vocab projection ===",
        f"; layout: {HBM_WEIGHT_LAYOUT} (vocab_size={vocab_size}, hidden_size={hidden_size})",
        f"; padded vocab: {padded_vocab} ({padded_vocab - vocab_size} masked entries)",
        f"; padded hidden: {padded_hidden} ({padded_hidden - hidden_size} zero columns)",
    ]
    body = projection_T_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=padded_hidden,
        alive_registers=alive_registers,
        w_base_hbm_offset_reg=lm_head_weight_hbm_offset_reg,
        activation_base_address=activation_base_address,
        result_base_address=result_base_address,
        out_features=padded_vocab,
    )
    return "\n".join(header) + "\n" + body
