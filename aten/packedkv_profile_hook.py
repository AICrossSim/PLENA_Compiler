"""Dependency-light profile validation hook for PackedKV compiler evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from compiler.assembler.assembly_to_binary import AssemblyToBinary
from compiler.assembler.parser import parse_asm_file
from compiler.aten.execution_trace import HBM_READ, iter_loop_scoped_lines
from compiler.aten.plena import PackedKVLayout, PlenaCompiler

REQUEST_SCHEMA = "decode-stage-hook-request"
RESULT_SCHEMA = "decode-stage-hook-result"
BINDING_SCHEMA = "plena-compiler-precision-binding"
METRICS_SCHEMA = "plena-compiler-packedkv-metrics"
REJECTION_SCHEMA = "plena-compiler-profile-rejection"
INPUT_RECIPE_SCHEMA = "plena-compiler-trace-input-recipe"
TRACE_CONTRACT_SCHEMA = "plena-compiler-packedkv-trace-contract"
LONG_CONTEXT_CONTRACT_SCHEMA = (
    "plena-compiler-packedkv-long-context-contract"
)
CONTEXT_SCOPE_SCHEMA = "plena-compiler-packedkv-context-scope"
PROFILE_SCHEMA = "decode-precision-profile"
RUNTIME_PRECISION_SCHEMA = "plena-runtime-precision-contract"
MATRIX_SEMANTICS_SCHEMA = "plena-matrix-semantics"
MX_PHYSICAL_SEMANTICS_SCHEMA = "plena-mx-physical-semantics"
MXINT_MATRIX_RULE = "block8_range_safe_scale_widened_mac_max_shift16_rne_vector"
MXFP_MATRIX_RULE = "product_cast_to_m_fp_then_fixed16_16_bank"
MIXED_MATRIX_RULE = "deployment_unsupported_without_trace_evidence"
ACCUMULATOR_RULE = "plena_fixed16_16_accumulate_truncate"
OUTPUT_RULE = "truncate_to_vector_format"

MATRIX_FORMATS = (
    "MXINT2",
    "MXINT4",
    "MXINT8",
    "E1M2",
    "E2M1",
    "E3M4",
    "E4M3",
    "E5M2",
)
VECTOR_FORMATS = (
    "FP_E3M2",
    "FP_E2M3",
    "FP_E6M5",
    "FP_E5M6",
    "FP_E4M7",
    "FP_E8M5",
    "BF16",
)
MXINT_WEIGHT_FORMATS = ("MXINT4", "MXINT8")
MXINT_OPERAND_FORMATS = ("MXINT2", "MXINT4", "MXINT8")
MXINT_ACTIVATION_FORMATS = ("MXINT4", "MXINT8")
MXFP_HARDWARE_FORMATS = ("E1M2", "E2M1", "E4M3", "E5M2")
EVIDENCE_TARGET_SCHEMA = "plena-evidence-target"
COMPILER_TARGET_MODE = "rtl_compiler_deployment"
COMPILER_CAPABILITY_SCOPE = "rtl_structural"
MXINT2_ACTIVATION_SCOPE = "unsupported"
TARGET = {
    "mlen": 1024,
    "blen": 8,
    "hlen": 128,
    "batch": 1,
    "kv_heads": 8,
    "head_dim": 128,
    "block_size": 8,
    "selector_bits": 4,
    "packed_kv": True,
    "batched_attention": True,
}
TRACE_BATCHES = (1, 2, 4)
TRACE_CACHE_TOKENS = 16
TRACE_Q_LEN = 1
LONG_CONTEXT_BATCH = 2
LONG_CONTEXT_CACHE_TOKENS = 33
LONG_CONTEXT_CACHE_ROWS_PER_BATCH = 48
LONG_CONTEXT_GEOMETRY = {
    **TARGET,
    "mlen": 16,
    "blen": 4,
    "hlen": 8,
    "batch": LONG_CONTEXT_BATCH,
    "kv_heads": 2,
    "head_dim": 8,
    "selector_bits": 1,
}
EMULATOR_EXECUTABLE_OPCODES = frozenset(
    {
        "M_MM", "M_TMM", "M_BMM", "M_BTMM", "M_BMM_WO", "M_MM_WO",
        "M_MV", "M_TMV", "M_BMV", "M_BTMV", "M_MV_WO", "M_BMV_WO",
        "V_ADD_VV", "V_ADD_VF", "V_SUB_VV", "V_SUB_VF", "V_MUL_VV",
        "V_MUL_VF", "V_EXP_V", "V_RECI_V", "V_RED_SUM", "V_RED_MAX",
        "S_ADD_FP", "S_SUB_FP", "S_MAX_FP", "S_MUL_FP", "S_EXP_FP",
        "S_RECI_FP", "S_SQRT_FP", "S_LD_FP", "S_ST_FP", "S_MAP_V_FP",
        "S_ADD_INT", "S_ADDI_INT", "S_SUB_INT", "S_MUL_INT", "S_LUI_INT",
        "S_LD_INT", "S_ST_INT", "H_PREFETCH_M", "H_PREFETCH_V",
        "H_STORE_V", "C_SET_ADDR_REG", "C_SET_SCALE_REG",
        "C_SET_STRIDE_REG", "C_SET_V_MASK_REG", "C_LOOP_START",
        "C_LOOP_END", "V_SHIFT_V", "C_BREAK",
    }
)
RTL_DECODER_EXECUTABLE_OPCODES = frozenset(
    {
        "M_MM", "M_TMM", "M_BMM", "M_BTMM", "M_MM_WO", "M_BMM_WO",
        "M_MV", "M_MV_WO", "V_ADD_VV", "V_ADD_VF", "V_SUB_VV",
        "V_SUB_VF", "V_MUL_VV", "V_MUL_VF", "V_EXP_V", "V_RECI_V",
        "V_RED_SUM", "V_RED_MAX", "V_PS_V", "V_SHIFT_V",
        "C_HADAMARD_TRANSFORM", "S_ADD_INT", "S_ADDI_INT", "S_SUB_INT",
        "S_MUL_INT", "S_LUI_INT", "S_LD_INT", "S_ST_INT", "S_ADD_FP",
        "S_SUB_FP", "S_MAX_FP", "S_MUL_FP", "S_EXP_FP", "S_RECI_FP",
        "S_SQRT_FP", "S_LD_FP", "S_ST_FP", "S_MAP_V_FP", "H_PREFETCH_M",
        "H_PREFETCH_V", "H_STORE_V", "C_SET_ADDR_REG", "C_SET_SCALE_REG",
        "C_SET_STRIDE_REG", "C_SET_V_MASK_REG", "C_LOOP_START",
        "C_LOOP_END", "C_BREAK",
    }
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_PROFILE_FIELDS = {
    "schema_version",
    "kind",
    "weight_format",
    "activation_format",
    "key_format",
    "value_format",
    "vector_format",
    "block_size",
    "scale_format",
    "scale_bits",
    "accumulator_rule",
    "output_rule",
    "matrix_semantics",
    "method",
    "operator_coverage",
}
_MATRIX_SEMANTICS = {
    "schema_version": MATRIX_SEMANTICS_SCHEMA,
    "block_size": 8,
    "mxint_rule": MXINT_MATRIX_RULE,
    "mxint_max_shift": 16,
    "mxint_vector_rounding": "round_to_nearest_even",
    "mxint_partial_conversion": (
        "per_mm_ic_integer_reduction_to_vector_storage_fp"
    ),
    "mxint_cross_instruction_accumulation": (
        "signed_fixed16_16_wraparound"
    ),
    "mxfp_rule": MXFP_MATRIX_RULE,
    "m_fp_format_binding": "profile.vector_format",
    "matrix_storage_fp_binding": "profile.vector_format",
    "matrix_instruction_k_partition": "MLEN",
    "qk_logical_k_partition": "HLEN",
    "fixed_accumulator_integer_bits": 16,
    "fixed_accumulator_fraction_bits": 16,
    "accumulator_rule": ACCUMULATOR_RULE,
    "output_rule": OUTPUT_RULE,
    "mixed_family_rule": MIXED_MATRIX_RULE,
    "mixed_family_deployment_supported": False,
}
_OPERATOR_COVERAGE_FIELDS = {
    "weight",
    "activation",
    "kv",
    "vector",
    "bf16",
}
_COMPILER_ROOT = Path(__file__).resolve().parents[1]


class HookError(ValueError):
    """Raised when a hook request or immutable output is malformed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("content_hash", None)
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _with_content_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_hash"] = _content_hash(result)
    return result


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HookError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                HookError(f"non-finite JSON token {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise HookError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HookError(f"{path} must contain a JSON object")
    return value


def _require_hash(value: Any, field: str) -> str:
    token = str(value)
    if not _HASH_RE.fullmatch(token):
        raise HookError(f"{field} must be a lowercase SHA-256 digest")
    return token


def _validate_request(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "stage",
        "manifest_hash",
        "profile_id",
        "profile",
        "target",
        "source_tree_sha256",
        "hook_template_hash",
        "environment_sha256",
        "content_hash",
    }
    missing = required - set(value)
    if missing:
        raise HookError(f"request is missing fields {sorted(missing)}")
    if value["schema_version"] != REQUEST_SCHEMA:
        raise HookError(f"unsupported request schema {value['schema_version']!r}")
    if value["stage"] != "compiler":
        raise HookError(f"compiler hook cannot execute stage {value['stage']!r}")
    for field in (
        "manifest_hash",
        "source_tree_sha256",
        "hook_template_hash",
        "environment_sha256",
        "content_hash",
    ):
        _require_hash(value[field], field)
    if value["content_hash"] != _content_hash(value):
        raise HookError("request content_hash does not match its canonical body")
    profile = value["profile"]
    if not isinstance(profile, dict):
        raise HookError("profile must be an object")
    expected_profile_id = "dqp-" + hashlib.sha256(
        _canonical_bytes(profile)
    ).hexdigest()
    if value["profile_id"] != expected_profile_id:
        raise HookError("profile_id does not match the canonical profile")
    target = value["target"]
    if not isinstance(target, dict):
        raise HookError("target must be an object")
    if not isinstance(
        value.get("run_long_context_capability", False),
        bool,
    ):
        raise HookError("run_long_context_capability must be a boolean")
    return dict(value)


def _matrix_descriptor(token: str) -> dict[str, Any]:
    if token.startswith("MXINT"):
        width = int(token.removeprefix("MXINT"))
        return {
            "token": token,
            "family": "mxint",
            "element_bits": width,
            "exponent_bits": None,
            "mantissa_bits": None,
            "signed": True,
            "block_scaled": True,
        }
    match = re.fullmatch(r"E(\d+)M(\d+)", token)
    if match is None:
        raise HookError(f"invalid matrix format {token!r}")
    exponent, mantissa = (int(part) for part in match.groups())
    return {
        "token": token,
        "family": "mxfp",
        "element_bits": 1 + exponent + mantissa,
        "exponent_bits": exponent,
        "mantissa_bits": mantissa,
        "signed": True,
        "block_scaled": True,
    }


def _vector_descriptor(token: str) -> dict[str, Any]:
    if token == "BF16":
        return {
            "token": token,
            "family": "bf16",
            "element_bits": 16,
            "exponent_bits": 8,
            "mantissa_bits": 7,
            "signed": True,
            "block_scaled": False,
        }
    match = re.fullmatch(r"FP_E(\d+)M(\d+)", token)
    if match is None:
        raise HookError(f"invalid vector format {token!r}")
    exponent, mantissa = (int(part) for part in match.groups())
    return {
        "token": token,
        "family": "fp",
        "element_bits": 1 + exponent + mantissa,
        "exponent_bits": exponent,
        "mantissa_bits": mantissa,
        "signed": True,
        "block_scaled": False,
    }


def _profile_support(
    profile: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[bool, str, dict[str, dict[str, Any]]]:
    if set(profile) != _PROFILE_FIELDS:
        raise HookError(f"profile fields differ from {PROFILE_SCHEMA}")
    if profile["schema_version"] != PROFILE_SCHEMA:
        raise HookError(f"unsupported profile schema {profile['schema_version']!r}")
    matrix_semantics = profile["matrix_semantics"]
    if not isinstance(matrix_semantics, dict):
        raise HookError("matrix_semantics must be an object")
    if _canonical_bytes(matrix_semantics) != _canonical_bytes(_MATRIX_SEMANTICS):
        raise HookError("matrix_semantics differs from the PLENA contract")
    coverage = profile["operator_coverage"]
    if not isinstance(coverage, dict) or set(coverage) != _OPERATOR_COVERAGE_FIELDS:
        raise HookError("profile operator_coverage fields differ from the schema")
    if any(
        not isinstance(coverage[field], list)
        or not coverage[field]
        or any(not isinstance(item, str) or not item for item in coverage[field])
        for field in _OPERATOR_COVERAGE_FIELDS
    ):
        raise HookError("profile operator coverage must contain nonempty string lists")
    if profile["accumulator_rule"] != ACCUMULATOR_RULE:
        return False, "unsupported_accumulator_rule", {}
    if profile["output_rule"] != OUTPUT_RULE:
        return False, "unsupported_output_rule", {}
    if matrix_semantics["accumulator_rule"] != profile["accumulator_rule"]:
        raise HookError("profile and matrix-semantics accumulator rules differ")
    if matrix_semantics["output_rule"] != profile["output_rule"]:
        raise HookError("profile and matrix-semantics output rules differ")
    if int(matrix_semantics["block_size"]) != int(profile["block_size"]):
        raise HookError("profile and matrix-semantics block sizes differ")
    matrix_tokens = {
        role: str(profile[field])
        for role, field in (
            ("weight", "weight_format"),
            ("activation", "activation_format"),
            ("key", "key_format"),
            ("value", "value_format"),
        )
    }
    vector_token = str(profile["vector_format"])
    unknown = set(matrix_tokens.values()) - set(MATRIX_FORMATS)
    if unknown:
        raise HookError(f"profile contains unknown matrix formats {sorted(unknown)}")
    if vector_token not in VECTOR_FORMATS:
        raise HookError(f"profile contains unknown vector format {vector_token!r}")
    if matrix_tokens["key"] != matrix_tokens["value"]:
        raise HookError("canonical decode profiles require matching K and V formats")
    descriptors = {
        role: _matrix_descriptor(token)
        for role, token in matrix_tokens.items()
    }
    descriptors["vector"] = _vector_descriptor(vector_token)

    if dict(target) != TARGET:
        return False, "unsupported_target", descriptors
    if profile["kind"] not in {"quantized", "vector_bf16_control"}:
        return False, "reference_profile_not_compilable", descriptors
    if int(profile["block_size"]) != 8:
        return False, "unsupported_block_size", descriptors
    if profile["scale_format"] != "E8M0" or int(profile["scale_bits"]) != 8:
        return False, "unsupported_scale_format", descriptors
    if profile["method"] != "rtn":
        return False, "unsupported_quantization_method", descriptors
    if profile["kind"] == "quantized" and vector_token == "BF16":
        return False, "quantized_profile_requires_vector_fp", descriptors
    if profile["kind"] == "vector_bf16_control" and vector_token != "BF16":
        return False, "vector_control_requires_bf16", descriptors

    families = {
        descriptors[role]["family"]
        for role in ("weight", "activation", "key", "value")
    }
    if len(families) != 1:
        return False, "mixed_matrix_families", descriptors
    if families == {"mxint"}:
        if matrix_tokens["weight"] not in MXINT_WEIGHT_FORMATS:
            return False, "unsupported_mxint_weight", descriptors
        if matrix_tokens["activation"] not in MXINT_ACTIVATION_FORMATS:
            return False, "unsupported_mxint_activation", descriptors
        if any(
            matrix_tokens[role] not in MXINT_OPERAND_FORMATS
            for role in ("key", "value")
        ):
            return False, "unsupported_mxint_operand", descriptors
    elif families == {"mxfp"}:
        if any(
            matrix_tokens[role] not in MXFP_HARDWARE_FORMATS
            for role in ("weight", "activation", "key", "value")
        ):
            return False, "unsupported_mxfp_operand", descriptors
    else:
        return False, "unsupported_matrix_family", descriptors
    return True, "supported", descriptors


def _format_binding_id(descriptor: Mapping[str, Any]) -> str:
    return "fmt-" + hashlib.sha256(_canonical_bytes(descriptor)).hexdigest()


def _mx_config(
    descriptor: Mapping[str, Any],
    *,
    block_size: int,
    scale_bits: int,
) -> dict[str, Any]:
    if descriptor["family"] == "mxint":
        element = {
            "type": "Int",
            "width": descriptor["element_bits"],
        }
    elif descriptor["family"] == "mxfp":
        element = {
            "type": "Fp",
            "sign": True,
            "exponent": descriptor["exponent_bits"],
            "mantissa": descriptor["mantissa_bits"],
        }
    else:
        raise HookError(
            f"matrix role cannot bind family {descriptor['family']!r}"
        )
    return {
        "format": "Mx",
        "block": block_size,
        "ELEM": element,
        "SCALE": {
            "type": "Fp",
            "sign": False,
            "exponent": scale_bits,
            "mantissa": 0,
        },
    }


def _rtl_matrix_parameters(
    prefix: str,
    descriptor: Mapping[str, Any],
) -> dict[str, int]:
    if descriptor["family"] == "mxint":
        return {
            f"{prefix}_MX_INT_ENABLE": 1,
            f"{prefix}_MX_INT_WIDTH": int(descriptor["element_bits"]),
        }
    return {
        f"{prefix}_MX_INT_ENABLE": 0,
        f"{prefix}_MX_EXP_WIDTH": int(descriptor["exponent_bits"]),
        f"{prefix}_MX_MANT_WIDTH": int(descriptor["mantissa_bits"]),
    }


def _matrix_semantics_descriptor(
    profile: Mapping[str, Any],
    descriptors: Mapping[str, Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    families = {
        descriptors[role]["family"]
        for role in ("weight", "activation", "key", "value")
    }
    if len(families) != 1:
        raise HookError("mixed-family semantics cannot bind to the current datapath")
    family = next(iter(families))
    if family == "mxint":
        active_rule = MXINT_MATRIX_RULE
    elif family == "mxfp":
        active_rule = MXFP_MATRIX_RULE
    else:
        raise HookError(f"unsupported matrix family {family!r}")
    operation_bindings = [
        {
            "operation": operation,
            "left_role": left_role,
            "right_role": right_role,
            "family": family,
            "rule": active_rule,
            "structurally_supported": True,
            "numerical_trace_conformance": "not_run",
        }
        for operation, left_role, right_role in (
            ("linear", "activation", "weight"),
            ("qk", "activation", "key"),
            ("pv", "activation", "value"),
        )
    ]
    return _with_content_hash(
        {
            "schema_version": MATRIX_SEMANTICS_SCHEMA,
            "source_profile_schema": PROFILE_SCHEMA,
            "profile_contract": dict(profile["matrix_semantics"]),
            "active_family": family,
            "active_rule": active_rule,
            "operation_bindings": operation_bindings,
            "fixed_accumulator_bank": {
                "integer_bits": 16,
                "fraction_bits": 16,
                "accumulator_rule": ACCUMULATOR_RULE,
                "writeout_rule": OUTPUT_RULE,
            },
            "instruction_reduction": {
                "physical_k_width": int(target["mlen"]),
                "qk_logical_k_width": int(target["hlen"]),
                "linear_pv_logical_k_width": int(target["mlen"]),
                "partial_conversion": (
                    "per_mm_ic_to_vector_storage_fp"
                ),
                "cross_instruction_accumulation": (
                    "signed_fixed16_16_wraparound"
                ),
            },
            "matrix_storage_fp": {
                "format": profile["vector_format"],
                "exponent_bits": int(descriptors["vector"]["exponent_bits"]),
                "mantissa_bits": int(descriptors["vector"]["mantissa_bits"]),
                "rtl_parameter_source": "V_FP",
            },
            "mxint_pipeline": {
                "block_size": 8,
                "block_mac": "exact_signed_widened_integer",
                "alignment": "bounded_exponent_alignment",
                "max_shift": 16,
                "matrix_conversion": "round_to_nearest_even_to_vector",
            },
            "mxfp_pipeline": {
                "product_conversion": "cast_each_product_to_m_fp",
                "m_fp_format_binding": "profile.vector_format",
                "bank_conversion": "m_fp_to_fixed16_16",
            },
            "mixed_family": {
                "rule": MIXED_MATRIX_RULE,
                "deployment_supported": False,
            },
            "packedkv_selector_rtl_capability": {
                "supported": family == "mxint",
                "reason": (
                    "supported"
                    if family == "mxint"
                    else "selector_is_mxint_only"
                ),
            },
            "structural_binding_valid": True,
            "numerical_trace_conformance": {
                "status": "not_run",
                "required_for_emulator_valid": True,
                "required_for_rtl_valid": True,
            },
        }
    )


def _physical_semantics_descriptor() -> dict[str, Any]:
    return _with_content_hash(
        {
            "schema_version": MX_PHYSICAL_SEMANTICS_SCHEMA,
            "block_size": 8,
            "scale_format": "E8M0",
            "scale_code_bias": 127,
            "scale_code_min": 0,
            "scale_code_max": 255,
            "scale_exponent_min": -127,
            "scale_exponent_max": 128,
            "zero_scale_code": 127,
            "element_bit_order": "little_endian_lsb_first",
            "plane_order": ["element", "scale"],
            "plane_alignment_bytes": 32,
            "mxint_encoding": "sign_magnitude",
            "mxint_canonical_zero": "positive_zero",
            "mxint_rounding": "round_to_nearest_ties_to_even",
            "mxint_scale_rule": "ceil_log2_max_abs_over_qmax_fraction",
            "mxfp_scale_rule": "floor_log2_max_abs",
        }
    )


def _runtime_precision_contract(
    profile: Mapping[str, Any],
    descriptors: Mapping[str, Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    block_size = int(profile["block_size"])
    scale_bits = int(profile["scale_bits"])
    vector = descriptors["vector"]
    rtl_parameters = {
        "BLOCK_DIM": block_size,
        "MX_SCALE_WIDTH": scale_bits,
        **_rtl_matrix_parameters("WT", descriptors["weight"]),
        **_rtl_matrix_parameters("ACT", descriptors["activation"]),
        **_rtl_matrix_parameters("KV", descriptors["key"]),
        "V_FP_EXP_WIDTH": int(vector["exponent_bits"]),
        "V_FP_MANT_WIDTH": int(vector["mantissa_bits"]),
        "M_FP_EXP_WIDTH": int(vector["exponent_bits"]),
        "M_FP_MANT_WIDTH": int(vector["mantissa_bits"]),
        "S_FP_EXP_WIDTH": int(vector["exponent_bits"]),
        "S_FP_MANT_WIDTH": int(vector["mantissa_bits"]),
    }
    bf16_plain = {
        "format": "Plain",
        "DATA_TYPE": {
            "type": "Fp",
            "sign": True,
            "exponent": 8,
            "mantissa": 7,
        },
    }
    vector_plain = {
        "format": "Plain",
        "DATA_TYPE": {
            "type": "Fp",
            "sign": True,
            "exponent": vector["exponent_bits"],
            "mantissa": vector["mantissa_bits"],
        },
    }
    semantics = _matrix_semantics_descriptor(profile, descriptors, target)
    physical_semantics = _physical_semantics_descriptor()
    return _with_content_hash(
        {
            "schema_version": RUNTIME_PRECISION_SCHEMA,
            "hbm_matrix_role_selector": {
                "weight": {"h_prefetch_m_funct1": 0},
                "key": {"h_prefetch_m_funct1": 1},
                "value": {"h_prefetch_m_funct1": 1},
            },
            "emulator_precision": {
                "MATRIX_SRAM_TYPE": bf16_plain,
                "VECTOR_SRAM_TYPE": vector_plain,
                "HBM_M_WEIGHT_TYPE": _mx_config(
                    descriptors["weight"],
                    block_size=block_size,
                    scale_bits=scale_bits,
                ),
                "HBM_M_KV_TYPE": _mx_config(
                    descriptors["key"],
                    block_size=block_size,
                    scale_bits=scale_bits,
                ),
                "HBM_V_ACT_TYPE": _mx_config(
                    descriptors["activation"],
                    block_size=block_size,
                    scale_bits=scale_bits,
                ),
                "HBM_V_KV_TYPE": _mx_config(
                    descriptors["value"],
                    block_size=block_size,
                    scale_bits=scale_bits,
                ),
                "SCALAR_FP": {
                    "type": "Fp",
                    "sign": True,
                    "exponent": vector["exponent_bits"],
                    "mantissa": vector["mantissa_bits"],
                },
                "MATRIX_SEMANTICS": semantics,
            },
            "accumulator_storage_policy": {
                "rule": profile["accumulator_rule"],
                "hardware_binding": "fixed16_16",
                "ACC_INT_WIDTH": 16,
                "ACC_FRAC_WIDTH": 16,
                "matrix_sram_transport": "BF16",
                "output_rounding": profile["output_rule"],
                "output_format": profile["vector_format"],
                "family_semantics_sha256": semantics["content_hash"],
                "structural_binding_valid": True,
                "numerical_trace_conformance": "not_run",
            },
            "matrix_semantics": semantics,
            "physical_semantics": physical_semantics,
            "emulator_required_semantics": {
                "activation_matrix_port_conversion": {
                    "source": "VECTOR_SRAM_TYPE",
                    "target_format": profile["activation_format"],
                    "target_binding": _format_binding_id(
                        descriptors["activation"]
                    ),
                    "implemented_by_compiler_hook": False,
                    "requires_emulator_validation": True,
                }
            },
            "rtl_precision_parameters": dict(sorted(rtl_parameters.items())),
            "binding_modes": {
                "weight": "compile_time_format_plus_h_prefetch_m_funct1_0",
                "activation": "compile_time_activation_datapath",
                "key": "compile_time_format_plus_h_prefetch_m_funct1_1",
                "value": "compile_time_format_plus_h_prefetch_m_funct1_1",
                "vector": "compile_time_vector_and_scalar_datapath",
            },
        }
    )


def _build_binding(
    request: Mapping[str, Any],
    descriptors: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    body = {
        "schema_version": BINDING_SCHEMA,
        "profile_id": request["profile_id"],
        "profile": request["profile"],
        "target": request["target"],
        "evidence_target": {
            "schema_version": EVIDENCE_TARGET_SCHEMA,
            "target_mode": COMPILER_TARGET_MODE,
            "capability_scope": COMPILER_CAPABILITY_SCOPE,
            "source_tree_sha256": request["source_tree_sha256"],
            "mxint2_activation_scope": MXINT2_ACTIVATION_SCOPE,
            "rtl_deployment_supports_mxint2_activation": False,
            "common_deployment_valid": False,
        },
        "matrix_binding_mode": "static_hardware_signature",
        "format_descriptors": dict(descriptors),
        "format_binding_ids": {
            role: _format_binding_id(descriptor)
            for role, descriptor in descriptors.items()
        },
        "runtime_precision_contract": _runtime_precision_contract(
            request["profile"],
            descriptors,
            request["target"],
        ),
    }
    body["binding_id"] = "cpb-" + hashlib.sha256(
        _canonical_bytes(body)
    ).hexdigest()
    return _with_content_hash(body)


def _instruction_count(assembly: str) -> int:
    prefixes = ("S_", "C_", "H_", "V_", "M_")
    return sum(
        line.strip().startswith(prefixes)
        for line in assembly.splitlines()
    )


def _cross_target_operand_violations(
    instructions: Sequence[Any],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    zero_immediate_writeouts = {
        "M_MM_WO",
        "M_BMM_WO",
        "M_MV_WO",
    }
    zero_rd_matrix_ops = {"M_MM", "M_TMM", "M_MV"}
    for pc, instruction in enumerate(instructions):
        opcode = instruction.opcode
        if (
            opcode in zero_immediate_writeouts
            and int(instruction.imm or 0) != 0
        ):
            violations.append(
                {
                    "pc": pc,
                    "opcode": opcode,
                    "field": "imm",
                    "value": int(instruction.imm),
                    "requirement": "zero_for_rtl_emulator_equivalence",
                }
            )
        if opcode in zero_rd_matrix_ops and int(instruction.rd or 0) != 0:
            violations.append(
                {
                    "pc": pc,
                    "opcode": opcode,
                    "field": "rd",
                    "value": int(instruction.rd),
                    "requirement": "zero_unused_destination_field",
                }
            )
        if opcode in {"M_BMM", "M_BTMM"} and not (
            0 <= int(instruction.rd) < TARGET["mlen"] // TARGET["hlen"]
        ):
            violations.append(
                {
                    "pc": pc,
                    "opcode": opcode,
                    "field": "rd",
                    "value": int(instruction.rd),
                    "requirement": "packed_head_selector_in_range",
                }
            )
        if opcode in {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}:
            for field, value in (
                ("rstride", instruction.rstride),
                ("funct1", instruction.funct1),
            ):
                if value not in {0, 1}:
                    violations.append(
                        {
                            "pc": pc,
                            "opcode": opcode,
                            "field": field,
                            "value": value,
                            "requirement": "binary_mode_field",
                        }
                    )
        if opcode == "V_SUB_VF" and instruction.funct1 not in {0, 1}:
            violations.append(
                {
                    "pc": pc,
                    "opcode": opcode,
                    "field": "funct1",
                    "value": instruction.funct1,
                    "requirement": "normal_or_reverse_order",
                }
            )
    return violations


def _assemble_trace(assembly_path: Path) -> tuple[bytes, dict[str, Any]]:
    assembler = AssemblyToBinary(
        str(_COMPILER_ROOT / "doc" / "operation.svh"),
        str(_COMPILER_ROOT / "doc" / "configuration.svh"),
    )
    instructions = parse_asm_file(str(assembly_path))
    words = tuple(
        assembler._convert_to_binary(instruction)
        for instruction in instructions
    )
    payload = "".join(f"0x{word:08X}\n" for word in words).encode("ascii")
    opcodes = Counter(instruction.opcode for instruction in instructions)
    emitted = frozenset(opcodes)
    unsupported_emulator = sorted(emitted - EMULATOR_EXECUTABLE_OPCODES)
    unsupported_rtl = sorted(emitted - RTL_DECODER_EXECUTABLE_OPCODES)
    operand_violations = _cross_target_operand_violations(instructions)
    matrix_prefetch_precisions = tuple(
        int(instruction.funct1)
        for instruction in instructions
        if instruction.opcode == "H_PREFETCH_M"
    )
    return payload, {
        "machine_word_count": len(words),
        "machine_code_byte_count": len(payload),
        "machine_code_sha256": hashlib.sha256(payload).hexdigest(),
        "opcode_histogram": dict(sorted(opcodes.items())),
        "emulator_unsupported_opcodes": unsupported_emulator,
        "rtl_decoder_unsupported_opcodes": unsupported_rtl,
        "emulator_opcode_coverage_valid": not unsupported_emulator,
        "rtl_decoder_opcode_coverage_valid": not unsupported_rtl,
        "execution_opcode_coverage_valid": (
            not unsupported_emulator and not unsupported_rtl
        ),
        "cross_target_operand_violations": operand_violations,
        "execution_operand_coverage_valid": not operand_violations,
        "execution_contract_valid": (
            not unsupported_emulator
            and not unsupported_rtl
            and not operand_violations
        ),
        "decoded_h_prefetch_m_precision_funct1": list(
            matrix_prefetch_precisions
        ),
        "kv_prefetch_precision_valid": bool(matrix_prefetch_precisions)
        and all(value == 1 for value in matrix_prefetch_precisions),
    }


_MATRIX_WRITEOUT_OPCODES = frozenset({"M_MM_WO", "M_BMM_WO"})
_VRAM_REGION_RE = re.compile(
    r"^; (?:Allocate VRAM Matrix|VRAM View) (?P<name>\S+?): "
    r"(?:logical=\(\d+, \d+\) physical|)\(?(?P<rows>\d+), \d+\)"
    r" at VRAM\[(?P<address>\d+)\]$"
)


def _vram_regions(assembly: str, mlen: int) -> dict[str, tuple[int, int]]:
    """Map each allocated VRAM matrix to its half-open address extent."""

    regions: dict[str, tuple[int, int]] = {}
    for line in assembly.splitlines():
        match = _VRAM_REGION_RE.fullmatch(line.strip())
        if match is None:
            continue
        address = int(match.group("address"))
        regions[match.group("name")] = (
            address,
            address + int(match.group("rows")) * mlen,
        )
    return regions


def _resolved_matrix_writeouts(
    assembly: str,
) -> tuple[tuple[int, int, int | None], ...]:
    """Return (line, loop depth, destination address) per matrix write-out.

    The destination address is resolved by replaying the integer adds that set
    it; an unresolvable destination is reported as ``None`` so callers fail
    closed rather than treating it as safe.
    """

    gp_values: dict[int, int | None] = {0: 0}
    writeouts: list[tuple[int, int, int | None]] = []
    for scoped in iter_loop_scoped_lines(assembly):
        if scoped.is_comment:
            continue
        if scoped.opcode in _MATRIX_WRITEOUT_OPCODES and scoped.args:
            register = _parse_gp(scoped.args[0])
            writeouts.append(
                (
                    scoped.line_number,
                    scoped.loop_depth,
                    None if register is None else gp_values.get(register),
                )
            )
        _track_integer_register(scoped.opcode, scoped.args, gp_values)
    return tuple(writeouts)


def _parse_gp(operand: str) -> int | None:
    operand = operand.strip()
    if not operand.startswith("gp") or not operand[2:].isdigit():
        return None
    return int(operand[2:])


def _track_integer_register(
    opcode: str,
    args: Sequence[str],
    gp_values: dict[int, int | None],
) -> None:
    """Replay the immediate integer adds the lowering uses for addressing."""

    if not args:
        return
    destination = _parse_gp(args[0])
    if destination is None or destination == 0:
        return
    if opcode == "S_ADDI_INT" and len(args) == 3:
        source = _parse_gp(args[1])
        base = gp_values.get(source) if source is not None else None
        try:
            immediate = int(args[2], 0)
        except ValueError:
            base = None
            immediate = 0
        gp_values[destination] = None if base is None else base + immediate
    elif opcode.startswith(("S_", "C_", "V_", "M_", "H_")):
        gp_values[destination] = None


def _loop_clobbered_accumulators(
    assembly: str,
    *,
    mlen: int,
) -> tuple[int, ...]:
    """Matrix write-outs inside loops that land in a softmax accumulator.

    Compact multi-block attention keeps the online-softmax output live across
    the hardware loop, so a matrix write-out into that region would destroy the
    running accumulation.  Write-outs into per-iteration scratch are correct and
    are what the compaction relies on.
    """

    accumulators = [
        extent
        for name, extent in _vram_regions(assembly, mlen).items()
        if name.startswith("_packed_O_head")
    ]
    violations: list[int] = []
    for line_number, loop_depth, address in _resolved_matrix_writeouts(assembly):
        if not loop_depth:
            continue
        if address is None or any(
            start <= address < stop for start, stop in accumulators
        ):
            violations.append(line_number)
    return tuple(violations)


def _executed_cache_reads(
    artifact,
    *,
    key_tensor: str,
    value_tensor: str,
    key_base: int,
    key_block_bytes: int,
) -> list[dict[str, int]]:
    """Pair the executed K and V cache prefetch addresses in issue order.

    The addresses come from replaying the emitted integer and control ISA, so
    hardware loops contribute every iteration they actually run rather than the
    single static instruction that encodes them.
    """

    bindings = {
        binding.trace_entry_index: binding
        for binding in artifact.request_memory.bindings
    }
    streams: dict[str, list[int]] = {key_tensor: [], value_tensor: []}
    for index, entry in enumerate(artifact.execution_trace.entries):
        if entry.tensor not in streams or entry.dma_direction != HBM_READ:
            continue
        binding = bindings.get(index)
        if binding is None:
            continue
        streams[entry.tensor].extend(
            request.address for request in binding.iter_requests()
        )
    if key_block_bytes <= 0:
        raise HookError("packed cache blocks must occupy a positive byte extent")
    if len(streams[key_tensor]) != len(streams[value_tensor]):
        raise HookError("packed cache key and value prefetches are unpaired")
    return [
        {
            "physical_k_row_block": (key_address - key_base) // key_block_bytes,
            "key_address": key_address,
            "value_address": value_address,
        }
        for key_address, value_address in zip(
            streams[key_tensor],
            streams[value_tensor],
        )
    ]


def _context_scope(
    *,
    trace_scope: str,
    batch_size: int,
    cache_tokens: int,
    cache_rows_per_batch: int,
    mlen: int,
    production_source_tree_sha256: str,
) -> dict[str, Any]:
    sequence_blocks = math.ceil(cache_tokens / mlen)
    return _with_content_hash(
        {
            "schema_version": CONTEXT_SCOPE_SCHEMA,
            "trace_scope": trace_scope,
            "batch_size": batch_size,
            "q_len": TRACE_Q_LEN,
            "cache_tokens": cache_tokens,
            "cache_position": cache_tokens - 1,
            "cache_rows_per_batch": cache_rows_per_batch,
            "sequence_blocks_per_selector": sequence_blocks,
            "final_sequence_block_tokens": (
                cache_tokens - (sequence_blocks - 1) * mlen
            ),
            "mlen": mlen,
            "production_source_tree_sha256": _require_hash(
                production_source_tree_sha256,
                "production_source_tree_sha256",
            ),
        }
    )


def _stream_seed(binding_id: str, batch_size: int, name: str) -> int:
    payload = f"{binding_id}:{batch_size}:{name}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _trace_recipe(
    binding: Mapping[str, Any],
    *,
    batch_size: int,
    cache_tokens: int,
    cache_rows_per_batch: int,
    context_scope: Mapping[str, Any],
    query_base: int,
    output_base: int,
    scratch_base: int,
    weight_base: int,
    weight_bytes: int,
    activation_base: int,
    activation_bytes: int,
    key_base: int,
    value_base: int,
) -> dict[str, Any]:
    target = binding["target"]
    mlen = int(target["mlen"])
    kv_heads = int(target["kv_heads"])
    group_stride = batch_size * mlen * mlen
    batch_stride = mlen * mlen
    q_regions = [
        {
            "batch": batch,
            "selector": selector,
            "base_address_elements": (
                query_base
                + selector * group_stride
                + batch * batch_stride
            ),
            "logical_shape": [TRACE_Q_LEN, mlen],
            "physical_shape": [mlen, mlen],
        }
        for batch in range(batch_size)
        for selector in range(kv_heads)
    ]
    kv_regions = [
        {
            "batch": batch,
            "row_start": batch * cache_rows_per_batch,
            "active_rows": cache_tokens,
            "physical_rows": cache_rows_per_batch,
            "row_width_elements": mlen,
        }
        for batch in range(batch_size)
    ]
    return _with_content_hash(
        {
            "schema_version": INPUT_RECIPE_SCHEMA,
            "profile_id": binding["profile_id"],
            "binding_id": binding["binding_id"],
            "batch_size": batch_size,
            "q_len": TRACE_Q_LEN,
            "cache_tokens": cache_tokens,
            "cache_position": cache_tokens - 1,
            "cache_rows_per_batch": cache_rows_per_batch,
            "context_scope": dict(context_scope),
            "materialized_numerical_payloads": False,
            "evidence_scope": (
                "The recipe makes compiler traces reproducible but does not "
                "constitute emulator or RTL numerical validation."
            ),
            "value_generator": {
                "schema_version": "plena-lcg32-real-input",
                "formula": (
                    "u32=(1664525*((linear_index XOR seed) AND 0xffffffff)"
                    "+1013904223) AND 0xffffffff"
                ),
                "normalization": "real=((u32+0.5)/4294967296.0)*2.0-1.0",
                "zero_padding": True,
            },
            "precision_binding": {
                "weight": binding["format_binding_ids"]["weight"],
                "activation": binding["format_binding_ids"]["activation"],
                "key": binding["format_binding_ids"]["key"],
                "value": binding["format_binding_ids"]["value"],
                "vector": binding["format_binding_ids"]["vector"],
                "block_size": binding["profile"]["block_size"],
                "scale_format": binding["profile"]["scale_format"],
                "scale_bits": binding["profile"]["scale_bits"],
                "matrix_semantics_sha256": binding[
                    "runtime_precision_contract"
                ]["matrix_semantics"]["content_hash"],
                "numerical_trace_conformance": "not_run",
            },
            "hbm_address_reservations": [
                {
                    "name": "W_bank_binding",
                    "role": "weight",
                    "base_address": weight_base,
                    "size_bytes": weight_bytes,
                    "payload_required": False,
                },
                {
                    "name": "A_ingress_binding",
                    "role": "activation",
                    "base_address": activation_base,
                    "size_bytes": activation_bytes,
                    "payload_required": False,
                },
            ],
            "memory_images": [
                {
                    "name": "Q",
                    "precision_role": "activation",
                    "space": "vram",
                    "address_unit": "elements",
                    "base_address": query_base,
                    "physical_shape": [batch_size * mlen, kv_heads * mlen],
                    "layout": "column_block_major",
                    "active_regions": q_regions,
                    "seed": _stream_seed(binding["binding_id"], batch_size, "Q"),
                    "padding_value": 0.0,
                },
                {
                    "name": "O",
                    "precision_role": "vector",
                    "space": "vram",
                    "address_unit": "elements",
                    "base_address": output_base,
                    "physical_shape": [batch_size * mlen, kv_heads * mlen],
                    "layout": "column_block_major",
                    "initial_value": 0.0,
                },
                {
                    "name": "S",
                    "precision_role": "vector",
                    "space": "vram",
                    "address_unit": "elements",
                    "base_address": scratch_base,
                    "initial_value": 0.0,
                },
                {
                    "name": "K_packed",
                    "precision_role": "key",
                    "space": "hbm",
                    "address_unit": "allocator_bytes",
                    "base_address": key_base,
                    "physical_shape": [
                        batch_size * cache_rows_per_batch,
                        mlen,
                    ],
                    "layout": "row_major_packedkv_element_and_scale_planes",
                    "active_regions": kv_regions,
                    "seed": _stream_seed(
                        binding["binding_id"], batch_size, "K_packed"
                    ),
                    "padding_value": 0.0,
                },
                {
                    "name": "V_packed",
                    "precision_role": "value",
                    "space": "hbm",
                    "address_unit": "allocator_bytes",
                    "base_address": value_base,
                    "physical_shape": [
                        batch_size * cache_rows_per_batch,
                        mlen,
                    ],
                    "layout": "row_major_packedkv_element_and_scale_planes",
                    "active_regions": kv_regions,
                    "seed": _stream_seed(
                        binding["binding_id"], batch_size, "V_packed"
                    ),
                    "padding_value": 0.0,
                },
            ],
            "fp_sram_preload": [
                {"address": 0, "value": 0.0, "role": "zero"},
                {
                    "address": 1,
                    "value": 1.0 / math.sqrt(int(target["head_dim"])),
                    "role": "softmax_scale",
                },
                {"address": 2, "value": -60000.0, "role": "finite_negative"},
            ],
        }
    )


def _compile_trace(
    binding: Mapping[str, Any],
    *,
    batch_size: int,
    production_source_tree_sha256: str,
    cache_tokens: int = TRACE_CACHE_TOKENS,
    cache_rows_per_batch: int | None = None,
    trace_scope: str = "routine",
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    target = binding["target"]
    descriptors = binding["format_descriptors"]
    mlen = int(target["mlen"])
    hlen = int(target["hlen"])
    kv_heads = int(target["kv_heads"])
    block_size = int(target["block_size"])
    weight_bits = int(descriptors["weight"]["element_bits"])
    activation_bits = int(descriptors["activation"]["element_bits"])
    kv_bits = int(descriptors["key"]["element_bits"])
    scale_bits = int(binding["profile"]["scale_bits"])
    rows_per_batch = mlen
    physical_rows = batch_size * rows_per_batch
    if cache_tokens <= 0:
        raise HookError("cache_tokens must be positive")
    if cache_rows_per_batch is None:
        cache_rows_per_batch = math.ceil(cache_tokens / mlen) * mlen
    if (
        cache_rows_per_batch < cache_tokens
        or cache_rows_per_batch % mlen
    ):
        raise HookError(
            "cache_rows_per_batch must cover the context in whole MLEN slabs"
        )
    cache_physical_rows = batch_size * cache_rows_per_batch
    q_width = kv_heads * mlen

    compiler = PlenaCompiler(
        mlen=mlen,
        blen=int(target["blen"]),
        hbm_element_width=weight_bits,
        hbm_block_size=block_size,
        hbm_scale_width=scale_bits,
    )
    compiler.hlen = hlen
    compiler.broadcast_amount = mlen // hlen
    compiler.emit(f"; PROFILE_BINDING {binding['binding_id']}\n")
    compiler.emit(
        "; FORMATS "
        f"W={binding['profile']['weight_format']} "
        f"A={binding['profile']['activation_format']} "
        f"KV={binding['profile']['key_format']} "
        f"VECTOR={binding['profile']['vector_format']} "
        f"BLOCK={block_size}\n"
    )
    query = compiler.alloc(
        "Q",
        batch_size * TRACE_Q_LEN,
        q_width,
        strict=False,
        physical_shape=(physical_rows, q_width),
    )
    output = compiler.alloc(
        "O",
        batch_size * TRACE_Q_LEN,
        q_width,
        strict=False,
        physical_shape=(physical_rows, q_width),
    )
    scratch = compiler.alloc(
        "S",
        mlen * (compiler.broadcast_amount * 2),
        mlen,
        strict=True,
    )
    weight_probe = compiler.input(
        "W_bank_binding",
        shape=(mlen, mlen),
        physical_shape=(mlen, mlen),
        precision_role="weight",
    )
    activation_probe = compiler.input(
        "A_ingress_binding",
        shape=(mlen, mlen),
        physical_shape=(mlen, mlen),
        hbm_element_width=activation_bits,
        hbm_block_size=block_size,
        hbm_scale_width=scale_bits,
        precision_role="activation",
    )
    key = compiler.input(
        "K_packed",
        shape=(batch_size * cache_tokens, mlen),
        physical_shape=(cache_physical_rows, mlen),
        hbm_element_width=kv_bits,
        hbm_block_size=block_size,
        hbm_scale_width=scale_bits,
        precision_role="key",
    )
    value = compiler.input(
        "V_packed",
        shape=(batch_size * cache_tokens, mlen),
        physical_shape=(cache_physical_rows, mlen),
        hbm_element_width=kv_bits,
        hbm_block_size=block_size,
        hbm_scale_width=scale_bits,
        precision_role="value",
    )
    compiler.flash_attention_packed_cache(
        query,
        key,
        value,
        num_kv_heads=kv_heads,
        group_heads=compiler.broadcast_amount,
        head_slot_dim=hlen,
        output_base_address=compiler.get_vram_addr(output.name),
        scratch_base_address=compiler.get_vram_addr(scratch.name),
        broadcast_amount=compiler.broadcast_amount,
        causal_mask=False,
        valid_cols=cache_tokens,
        cache_position=cache_tokens - 1,
        batch_size=batch_size,
        rows_per_batch=rows_per_batch,
        query_rows_per_batch=TRACE_Q_LEN,
        cache_rows_per_batch=cache_rows_per_batch,
    )
    artifact = compiler.compile_with_trace()
    assembly = artifact.assembly
    selector_ids = tuple(
        int(scoped.args[0])
        for scoped in iter_loop_scoped_lines(assembly)
        if scoped.opcode == "M_BTMM"
        for _ in range(scoped.multiplicity)
    )
    sequence_blocks = math.ceil(cache_tokens / mlen)
    block_tokens = tuple(
        min(mlen, cache_tokens - block * mlen)
        for block in range(sequence_blocks)
    )
    kv_tiles_by_block = tuple(
        math.ceil(tokens / int(target["blen"]))
        for tokens in block_tokens
    )
    kv_tiles = sum(kv_tiles_by_block)
    expected_selectors = batch_size * kv_heads * kv_tiles
    expected_selector_sequence = tuple(
        selector
        for _batch in range(batch_size)
        for selector in range(kv_heads)
        for _tile in range(kv_tiles)
    )
    selector_histogram = {
        str(selector): selector_ids.count(selector)
        for selector in range(kv_heads)
    }
    expected_per_selector = batch_size * kv_tiles
    selector_ok = (
        len(selector_ids) == expected_selectors
        and all(count == expected_per_selector for count in selector_histogram.values())
    )
    cache_blocks_per_batch = cache_rows_per_batch // mlen
    slab_ok = all(
        assembly.count(
            f"; PackedKV batch {batch}, selector {selector}, "
            f"K block {batch * cache_blocks_per_batch}"
        )
        == 1
        for batch in range(batch_size)
        for selector in range(kv_heads)
    )
    layout = PackedKVLayout(
        kv_heads=kv_heads,
        head_dim=hlen,
        mlen=mlen,
        block_size=block_size,
        element_bits=kv_bits,
        scale_bits=scale_bits,
    )
    query_base = compiler.get_vram_addr(query.name)
    output_base = compiler.get_vram_addr(output.name)
    scratch_base = compiler.get_vram_addr(scratch.name)
    group_stride = physical_rows * mlen
    batch_stride = rows_per_batch * mlen
    context_scope_contract = _context_scope(
        trace_scope=trace_scope,
        batch_size=batch_size,
        cache_tokens=cache_tokens,
        cache_rows_per_batch=cache_rows_per_batch,
        mlen=mlen,
        production_source_tree_sha256=production_source_tree_sha256,
    )
    expected_sequence_blocks = [
        {
            "batch": batch,
            "selector": selector,
            "sequence_block": block,
            "physical_k_row_block": (
                batch * cache_blocks_per_batch + block
            ),
            "valid_columns": block_tokens[block],
        }
        for batch in range(batch_size)
        for selector in range(kv_heads)
        for block in range(sequence_blocks)
    ]
    key_layout = compiler.get_hbm_layout(key.name)
    value_layout = compiler.get_hbm_layout(value.name)
    expected_cache_reads = [
        {
            "physical_k_row_block": item["physical_k_row_block"],
            "key_address": key_layout.hbm_base_addr
            + key_layout.element_offset_bytes(
                item["physical_k_row_block"] * mlen * mlen
            ),
            "value_address": value_layout.hbm_base_addr
            + value_layout.element_offset_bytes(
                item["physical_k_row_block"] * mlen * mlen
                + item["selector"] * hlen
            ),
        }
        for item in expected_sequence_blocks
    ]
    observed_cache_reads = _executed_cache_reads(
        artifact,
        key_tensor=key.name,
        value_tensor=value.name,
        key_base=key_layout.hbm_base_addr,
        key_block_bytes=key_layout.element_offset_bytes(mlen * mlen),
    )
    sequence_block_count = len(observed_cache_reads)
    logical_sequence_block_count = len(expected_sequence_blocks)
    expected_sequence_block_count = logical_sequence_block_count
    sequence_block_count_valid = (
        sequence_block_count == expected_sequence_block_count
    )
    physical_cache_slab_offsets_valid = (
        slab_ok and observed_cache_reads == expected_cache_reads
    )
    final_block_marker = (
        "; PackedKV compact masked-tail block, "
        f"valid columns {block_tokens[-1]}"
    )
    partial_final_sequence_block = (
        sequence_blocks > 1 and block_tokens[-1] < mlen
    )
    partial_final_sequence_block_valid = (
        partial_final_sequence_block
        and assembly.count(final_block_marker) == batch_size * kv_heads
    )
    observed_state = tuple(
        (int(head), int(base), int(stride))
        for head, base, stride in re.findall(
            r"^; PackedKV softmax state head (\d+), "
            r"base (\d+), stride (\d+)$",
            assembly,
            flags=re.MULTILINE,
        )
    )
    expected_state = tuple(
        (head, 10 + head * 3, 1)
        for _batch in range(batch_size)
        for _selector in range(kv_heads)
        for head in range(compiler.broadcast_amount)
    )
    compact_state_addresses_valid = (
        sequence_blocks > 1 and observed_state == expected_state
    )
    clobbered_accumulator_lines = _loop_clobbered_accumulators(
        assembly,
        mlen=mlen,
    )
    no_loop_clobbered_accumulator = not clobbered_accumulator_lines
    dynamic_opcodes = Counter()
    for scoped in iter_loop_scoped_lines(assembly):
        if not scoped.is_comment:
            dynamic_opcodes[scoped.opcode] += scoped.multiplicity
    pv_matrix_compute_count = dynamic_opcodes["M_MM"]
    pv_matrix_writeout_count = dynamic_opcodes["M_MM_WO"]
    qk_matrix_writeout_count = dynamic_opcodes["M_BMM_WO"]
    matrix_compute_writeout_pairing_valid = (
        pv_matrix_compute_count > 0
        and pv_matrix_compute_count == pv_matrix_writeout_count
        and len(selector_ids) == qk_matrix_writeout_count
    )
    no_dynamic_kv_group_loop = (
        "KV-looped" not in assembly
        and "attention core loop over KV groups" not in assembly
    )
    slab_bytes = key.hbm_size // batch_size
    slab_mappings = [
        {
            "batch": batch,
            "selector": selector,
            "k_row_block": batch * cache_blocks_per_batch,
            "k_selector_immediate": selector,
            "v_element_offset": (
                batch * cache_rows_per_batch * mlen + selector * hlen
            ),
            "cache_slab_row_start": batch * cache_rows_per_batch,
            "cache_slab_element_offset": (
                batch * cache_rows_per_batch * mlen
            ),
            "key_hbm_allocator_offset": batch * slab_bytes,
            "value_hbm_allocator_offset": batch * slab_bytes,
            "q_vram_address": (
                query_base
                + selector * group_stride
                + batch * batch_stride
            ),
            "o_vram_address": (
                output_base
                + selector * group_stride
                + batch * batch_stride
            ),
        }
        for batch in range(batch_size)
        for selector in range(kv_heads)
    ]
    trace_contract = _with_content_hash(
        {
            "schema_version": (
                LONG_CONTEXT_CONTRACT_SCHEMA
                if trace_scope == "scaled_long_context_structural"
                else TRACE_CONTRACT_SCHEMA
            ),
            "profile_id": binding["profile_id"],
            "binding_id": binding["binding_id"],
            "context_scope": context_scope_contract,
            "batch_size": batch_size,
            "q_len": TRACE_Q_LEN,
            "cache_tokens": cache_tokens,
            "cache_position": cache_tokens - 1,
            "cache_rows_per_batch": cache_rows_per_batch,
            "expected_selector_sequence": list(expected_selector_sequence),
            "observed_selector_sequence": list(selector_ids),
            "selector_sequence_valid": selector_ids == expected_selector_sequence,
            "slab_mappings": slab_mappings,
            "slab_mapping_markers_valid": slab_ok,
            "expected_sequence_blocks": expected_sequence_blocks,
            "expected_executed_cache_reads": expected_cache_reads,
            "observed_executed_cache_reads": observed_cache_reads,
            "physical_cache_slab_offsets_valid": (
                physical_cache_slab_offsets_valid
            ),
            "compact_state_addresses": [
                {"head": head, "base": base, "stride": stride}
                for head, base, stride in observed_state
            ],
            "compact_state_addresses_valid": compact_state_addresses_valid,
            "loop_clobbered_accumulator_lines": list(
                clobbered_accumulator_lines
            ),
            "no_loop_clobbered_accumulator": no_loop_clobbered_accumulator,
            "pv_matrix_compute_count": pv_matrix_compute_count,
            "pv_matrix_writeout_count": pv_matrix_writeout_count,
            "qk_matrix_compute_count": len(selector_ids),
            "qk_matrix_writeout_count": qk_matrix_writeout_count,
            "matrix_compute_writeout_pairing_valid": (
                matrix_compute_writeout_pairing_valid
            ),
            "no_dynamic_kv_group_loop": no_dynamic_kv_group_loop,
        }
    )
    recipe = _trace_recipe(
        binding,
        batch_size=batch_size,
        cache_tokens=cache_tokens,
        cache_rows_per_batch=cache_rows_per_batch,
        context_scope=context_scope_contract,
        query_base=query_base,
        output_base=output_base,
        scratch_base=scratch_base,
        weight_base=weight_probe.hbm_addr,
        weight_bytes=weight_probe.hbm_size,
        activation_base=activation_probe.hbm_addr,
        activation_bytes=activation_probe.hbm_size,
        key_base=key.hbm_addr,
        value_base=value.hbm_addr,
    )
    role_hbm_layouts = {
        role: {
            "precision_role": layout.precision_role,
            "element_bits": layout.hbm_element_width,
            "block_size": layout.hbm_block_size,
            "scale_bits": layout.hbm_scale_width,
            "address_unit": "physical_bytes",
            "element_plane_bytes": layout.element_plane_bytes,
            "scale_plane_bytes": layout.scale_plane_bytes,
            "total_bytes": layout.hbm_size,
        }
        for role, layout in (
            ("weight", compiler.get_hbm_layout(weight_probe.name)),
            ("activation", compiler.get_hbm_layout(activation_probe.name)),
            ("key", compiler.get_hbm_layout(key.name)),
            ("value", compiler.get_hbm_layout(value.name)),
        )
    }
    role_precision_binding_valid = all(
        role_hbm_layouts[role]
        == {
            "precision_role": role,
            "element_bits": int(descriptors[role]["element_bits"]),
            "block_size": block_size,
            "scale_bits": scale_bits,
            "address_unit": "physical_bytes",
            "element_plane_bytes": role_hbm_layouts[role][
                "element_plane_bytes"
            ],
            "scale_plane_bytes": role_hbm_layouts[role][
                "scale_plane_bytes"
            ],
            "total_bytes": role_hbm_layouts[role]["total_bytes"],
        }
        for role in ("weight", "activation", "key", "value")
    ) and all(
        item["total_bytes"]
        == item["element_plane_bytes"] + item["scale_plane_bytes"]
        for item in role_hbm_layouts.values()
    )
    accumulator_contract = binding["runtime_precision_contract"][
        "accumulator_storage_policy"
    ]
    semantics_contract = binding["runtime_precision_contract"]["matrix_semantics"]
    matrix_semantics_binding_valid = (
        semantics_contract["schema_version"] == MATRIX_SEMANTICS_SCHEMA
        and semantics_contract["source_profile_schema"] == PROFILE_SCHEMA
        and semantics_contract["profile_contract"]
        == binding["profile"]["matrix_semantics"]
        and semantics_contract["structural_binding_valid"] is True
        and semantics_contract["numerical_trace_conformance"]["status"]
        == "not_run"
        and semantics_contract["mixed_family"]["deployment_supported"] is False
        and all(
            item["structurally_supported"] is True
            and item["numerical_trace_conformance"] == "not_run"
            for item in semantics_contract["operation_bindings"]
        )
    )
    accumulator_binding_valid = (
        accumulator_contract["rule"]
        == binding["profile"]["accumulator_rule"]
        and accumulator_contract["ACC_INT_WIDTH"] == 16
        and accumulator_contract["ACC_FRAC_WIDTH"] == 16
        and accumulator_contract["output_rounding"]
        == binding["profile"]["output_rule"]
        and accumulator_contract["family_semantics_sha256"]
        == semantics_contract["content_hash"]
        and accumulator_contract["structural_binding_valid"] is True
        and accumulator_contract["numerical_trace_conformance"] == "not_run"
        and binding["runtime_precision_contract"]["rtl_precision_parameters"][
            "M_FP_EXP_WIDTH"
        ]
        == binding["runtime_precision_contract"]["rtl_precision_parameters"][
            "V_FP_EXP_WIDTH"
        ]
        and binding["runtime_precision_contract"]["rtl_precision_parameters"][
            "M_FP_MANT_WIDTH"
        ]
        == binding["runtime_precision_contract"]["rtl_precision_parameters"][
            "V_FP_MANT_WIDTH"
        ]
    )
    metrics = {
        "evidence_target": binding["evidence_target"],
        "trace_scope": trace_scope,
        "mlen": mlen,
        "blen": int(target["blen"]),
        "hlen": hlen,
        "kv_heads": kv_heads,
        "production_source_tree_sha256": production_source_tree_sha256,
        "batch_size": batch_size,
        "q_len": TRACE_Q_LEN,
        "cache_tokens": cache_tokens,
        "cache_position": cache_tokens - 1,
        "cache_rows_per_batch": cache_rows_per_batch,
        "sequence_blocks_per_selector": sequence_blocks,
        "logical_sequence_block_count": logical_sequence_block_count,
        "sequence_block_count": sequence_block_count,
        "expected_sequence_block_count": expected_sequence_block_count,
        "sequence_block_count_valid": sequence_block_count_valid,
        "final_sequence_block_tokens": block_tokens[-1],
        "partial_final_sequence_block": partial_final_sequence_block,
        "partial_final_sequence_block_valid": (
            partial_final_sequence_block_valid
        ),
        "physical_cache_slab_offsets_valid": (
            physical_cache_slab_offsets_valid
        ),
        "compact_state_addresses_valid": compact_state_addresses_valid,
        "loop_clobbered_accumulator_count": len(clobbered_accumulator_lines),
        "no_loop_clobbered_accumulator": no_loop_clobbered_accumulator,
        "pv_matrix_compute_count": pv_matrix_compute_count,
        "pv_matrix_writeout_count": pv_matrix_writeout_count,
        "qk_matrix_compute_count": len(selector_ids),
        "qk_matrix_writeout_count": qk_matrix_writeout_count,
        "matrix_compute_writeout_pairing_valid": (
            matrix_compute_writeout_pairing_valid
        ),
        "no_dynamic_kv_group_loop": no_dynamic_kv_group_loop,
        "block_size": block_size,
        "binding_id": binding["binding_id"],
        "format_binding_ids": binding["format_binding_ids"],
        "selector_instruction_count": len(selector_ids),
        "expected_selector_instruction_count": expected_selectors,
        "selector_histogram": selector_histogram,
        "selector_count_valid": selector_ok,
        "selector_sequence_valid": selector_ids == expected_selector_sequence,
        "batch_slab_mapping_valid": slab_ok,
        "assembly_line_count": len(assembly.splitlines()),
        "assembly_instruction_count": _instruction_count(assembly),
        "assembly_byte_count": len(assembly.encode("utf-8")),
        "assembly_sha256": hashlib.sha256(assembly.encode("utf-8")).hexdigest(),
        "packed_row_bytes": layout.packed_row_bytes,
        "padded_row_bytes": layout.padded_row_bytes,
        "packed_byte_reduction": layout.byte_reduction,
        "active_kv_bytes": layout.physical_bytes(
            tokens=batch_size * cache_tokens,
            tensors=2,
        ),
        "allocated_kv_hbm_bytes": key.hbm_size + value.hbm_size,
        "weight_binding_hbm_base": weight_probe.hbm_addr,
        "weight_binding_hbm_bytes": weight_probe.hbm_size,
        "activation_binding_hbm_base": activation_probe.hbm_addr,
        "activation_binding_hbm_bytes": activation_probe.hbm_size,
        "key_hbm_base": key.hbm_addr,
        "value_hbm_base": value.hbm_addr,
        "role_hbm_layouts": role_hbm_layouts,
        "physical_byte_addressing_valid": role_precision_binding_valid,
        "role_precision_binding_valid": role_precision_binding_valid,
        "accumulator_binding_valid": accumulator_binding_valid,
        "matrix_semantics_binding_valid": matrix_semantics_binding_valid,
        "structural_precision_binding_valid": (
            role_precision_binding_valid
            and accumulator_binding_valid
            and matrix_semantics_binding_valid
        ),
        "numerical_trace_conformance": "not_run",
        "packedkv_selector_rtl_capable": semantics_contract[
            "packedkv_selector_rtl_capability"
        ]["supported"],
        "packedkv_selector_rtl_capability_reason": semantics_contract[
            "packedkv_selector_rtl_capability"
        ]["reason"],
        "matrix_semantics_sha256": semantics_contract["content_hash"],
        "physical_semantics_sha256": binding[
            "runtime_precision_contract"
        ]["physical_semantics"]["content_hash"],
        "runtime_precision_contract_sha256": binding[
            "runtime_precision_contract"
        ]["content_hash"],
        "context_scope_sha256": context_scope_contract["content_hash"],
        "trace_contract_sha256": trace_contract["content_hash"],
        "input_recipe_sha256": recipe["content_hash"],
        "materialized_numerical_payloads": False,
        "compiler_evidence_scope": "structural_lowering_and_machine_assembly",
        "emulator_numerical_validation": False,
        "emulator_activation_matrix_port_validation": False,
        "rtl_numerical_validation": False,
    }
    return assembly, metrics, trace_contract, recipe


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise HookError(f"immutable output already exists with different content: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, _canonical_bytes(value) + b"\n")


def _artifact(path: Path, kind: str) -> dict[str, str]:
    payload = path.read_bytes()
    if not payload:
        raise HookError(f"artifact is empty: {path}")
    return {
        "artifact_id": "sha256-" + hashlib.sha256(payload).hexdigest(),
        "kind": kind,
        "path": str(path.resolve()),
    }


def _ensure_confined(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise HookError(f"artifact path escapes artifact-dir: {path}") from exc


def _existing_result(
    path: Path,
    request_hash: str,
    artifact_root: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = _load_json(path)
    if set(result) != {
        "schema_version",
        "stage",
        "manifest_hash",
        "profile_id",
        "request_content_hash",
        "observed_at_utc",
        "tests",
        "artifacts",
        "content_hash",
    }:
        raise HookError("existing result fields differ from the schema")
    if result.get("schema_version") != RESULT_SCHEMA:
        raise HookError("existing result has an unsupported schema")
    if result.get("stage") != "compiler":
        raise HookError("existing result has an invalid stage")
    if result.get("content_hash") != _content_hash(result):
        raise HookError("existing result content_hash is invalid")
    if result.get("request_content_hash") != request_hash:
        raise HookError("existing result belongs to a different request")
    tests = result.get("tests")
    if not isinstance(tests, list) or not tests:
        raise HookError("existing result has no tests")
    names: list[str] = []
    for test in tests:
        if not isinstance(test, dict) or set(test) != {
            "name",
            "passed",
            "metrics",
        }:
            raise HookError("existing test fields differ from the schema")
        if (
            not isinstance(test["name"], str)
            or not test["name"]
            or not isinstance(test["passed"], bool)
            or not isinstance(test["metrics"], dict)
        ):
            raise HookError("existing test fields have invalid types")
        names.append(test["name"])
    if len(names) != len(set(names)):
        raise HookError("existing result repeats a test name")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise HookError("existing result has no artifacts")
    artifact_ids: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != {
            "artifact_id",
            "kind",
            "path",
        }:
            raise HookError("existing artifact fields differ from the schema")
        artifact_path = Path(artifact["path"])
        _ensure_confined(artifact_path, artifact_root)
        if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
            raise HookError(f"existing artifact is missing or empty: {artifact_path}")
        expected_id = "sha256-" + hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()
        if artifact["artifact_id"] != expected_id:
            raise HookError(f"existing artifact hash is invalid: {artifact_path}")
        artifact_ids.append(artifact["artifact_id"])
    if len(artifact_ids) != len(set(artifact_ids)):
        raise HookError("existing result repeats an artifact ID")
    return result


def run_hook(
    request_path: Path,
    result_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    request = _validate_request(_load_json(request_path))
    artifact_root = artifact_dir.resolve()
    existing = _existing_result(
        result_path,
        request["content_hash"],
        artifact_root,
    )
    if existing is not None:
        return existing

    artifact_root.mkdir(parents=True, exist_ok=True)
    profile_root = artifact_root / request["profile_id"]
    profile_root.mkdir(parents=True, exist_ok=True)
    supported, reason, descriptors = _profile_support(
        request["profile"],
        request["target"],
    )
    tests: list[dict[str, Any]] = []
    artifacts: list[dict[str, str]] = []

    if not supported:
        rejection = _with_content_hash(
            {
                "schema_version": REJECTION_SCHEMA,
                "profile_id": request["profile_id"],
                "reason_code": reason,
                "profile": request["profile"],
                "target": request["target"],
                "evidence_target": {
                    "schema_version": EVIDENCE_TARGET_SCHEMA,
                    "target_mode": COMPILER_TARGET_MODE,
                    "capability_scope": COMPILER_CAPABILITY_SCOPE,
                    "source_tree_sha256": request["source_tree_sha256"],
                    "mxint2_activation_scope": MXINT2_ACTIVATION_SCOPE,
                    "rtl_deployment_supports_mxint2_activation": False,
                    "common_deployment_valid": False,
                },
            }
        )
        rejection_path = profile_root / "rejection.json"
        _ensure_confined(rejection_path, artifact_root)
        _write_json(rejection_path, rejection)
        artifacts.append(_artifact(rejection_path, "compiler_profile_rejection"))
        tests.append(
            {
                "name": "profile_support",
                "passed": False,
                "metrics": {
                    "reason_code": reason,
                    "weight_format": request["profile"]["weight_format"],
                    "activation_format": request["profile"]["activation_format"],
                    "kv_format": request["profile"]["key_format"],
                    "vector_format": request["profile"]["vector_format"],
                    "block_size": request["profile"]["block_size"],
                    "evidence_target": rejection["evidence_target"],
                },
            }
        )
    else:
        binding = _build_binding(request, descriptors)
        binding_path = profile_root / "binding.json"
        _ensure_confined(binding_path, artifact_root)
        _write_json(binding_path, binding)
        artifacts.append(_artifact(binding_path, "compiler_precision_binding"))
        tests.append(
            {
                "name": "profile_support",
                "passed": True,
                "metrics": {
                    "binding_id": binding["binding_id"],
                    "weight_format": request["profile"]["weight_format"],
                    "activation_format": request["profile"]["activation_format"],
                    "kv_format": request["profile"]["key_format"],
                    "vector_format": request["profile"]["vector_format"],
                    "block_size": request["profile"]["block_size"],
                    "evidence_target": binding["evidence_target"],
                },
            }
        )
        trace_metrics: dict[str, Any] = {}
        for batch_size in TRACE_BATCHES:
            assembly, metrics, trace_contract, recipe = _compile_trace(
                binding,
                batch_size=batch_size,
                production_source_tree_sha256=request[
                    "source_tree_sha256"
                ],
            )
            assembly_path = profile_root / f"packedkv-q1-b{batch_size}.asm"
            _ensure_confined(assembly_path, artifact_root)
            _write_immutable(assembly_path, assembly.encode("utf-8"))
            artifacts.append(_artifact(assembly_path, "packedkv_q1_assembly"))
            machine_code, assembler_metrics = _assemble_trace(assembly_path)
            machine_path = profile_root / f"packedkv-q1-b{batch_size}.mem"
            _ensure_confined(machine_path, artifact_root)
            _write_immutable(machine_path, machine_code)
            artifacts.append(_artifact(machine_path, "packedkv_q1_machine_code"))
            metrics.update(assembler_metrics)
            metrics["machine_word_count_valid"] = (
                metrics["machine_word_count"]
                == metrics["assembly_instruction_count"]
            )
            contract_path = profile_root / f"packedkv-q1-b{batch_size}-contract.json"
            _ensure_confined(contract_path, artifact_root)
            _write_json(contract_path, trace_contract)
            artifacts.append(
                _artifact(contract_path, "packedkv_q1_trace_contract")
            )
            recipe_path = profile_root / f"packedkv-q1-b{batch_size}-inputs.json"
            _ensure_confined(recipe_path, artifact_root)
            _write_json(recipe_path, recipe)
            artifacts.append(
                _artifact(recipe_path, "packedkv_q1_input_recipe")
            )
            trace_metrics[str(batch_size)] = metrics
            tests.append(
                {
                    "name": f"packedkv_q1_batch_{batch_size}",
                    "passed": bool(
                        metrics["selector_count_valid"]
                        and metrics["selector_sequence_valid"]
                        and metrics["batch_slab_mapping_valid"]
                        and metrics["sequence_block_count_valid"]
                        and metrics["physical_cache_slab_offsets_valid"]
                        and metrics["no_loop_clobbered_accumulator"]
                        and metrics[
                            "matrix_compute_writeout_pairing_valid"
                        ]
                        and metrics["no_dynamic_kv_group_loop"]
                        and metrics["machine_word_count_valid"]
                        and metrics["execution_contract_valid"]
                        and metrics["kv_prefetch_precision_valid"]
                        and metrics["role_precision_binding_valid"]
                        and metrics["accumulator_binding_valid"]
                        and metrics["matrix_semantics_binding_valid"]
                        and metrics["structural_precision_binding_valid"]
                        and metrics["q_len"] == 1
                        and metrics["cache_position"]
                        == metrics["cache_tokens"] - 1
                        and metrics["block_size"] == 8
                    ),
                    "metrics": metrics,
                }
            )
        if request.get("run_long_context_capability", False):
            scaled_binding = {
                **binding,
                "target": dict(LONG_CONTEXT_GEOMETRY),
            }
            assembly, metrics, trace_contract, recipe = _compile_trace(
                scaled_binding,
                batch_size=LONG_CONTEXT_BATCH,
                production_source_tree_sha256=request[
                    "source_tree_sha256"
                ],
                cache_tokens=LONG_CONTEXT_CACHE_TOKENS,
                cache_rows_per_batch=LONG_CONTEXT_CACHE_ROWS_PER_BATCH,
                trace_scope="scaled_long_context_structural",
            )
            prefix = "packedkv-q1-long-context-scaled"
            assembly_path = profile_root / f"{prefix}.asm"
            _ensure_confined(assembly_path, artifact_root)
            _write_immutable(assembly_path, assembly.encode("utf-8"))
            artifacts.append(
                _artifact(
                    assembly_path,
                    "packedkv_q1_long_context_assembly",
                )
            )
            machine_code, assembler_metrics = _assemble_trace(assembly_path)
            machine_path = profile_root / f"{prefix}.mem"
            _ensure_confined(machine_path, artifact_root)
            _write_immutable(machine_path, machine_code)
            artifacts.append(
                _artifact(
                    machine_path,
                    "packedkv_q1_long_context_machine_code",
                )
            )
            metrics.update(assembler_metrics)
            metrics["machine_word_count_valid"] = (
                metrics["machine_word_count"]
                == metrics["assembly_instruction_count"]
            )
            metrics["compiler_evidence_scope"] = (
                "scaled_structural_multitile_lowering"
            )
            metrics["full_geometry_timing_evidence"] = False
            contract_path = profile_root / f"{prefix}-contract.json"
            _ensure_confined(contract_path, artifact_root)
            _write_json(contract_path, trace_contract)
            artifacts.append(
                _artifact(
                    contract_path,
                    "packedkv_q1_long_context_trace_contract",
                )
            )
            recipe_path = profile_root / f"{prefix}-inputs.json"
            _ensure_confined(recipe_path, artifact_root)
            _write_json(recipe_path, recipe)
            artifacts.append(
                _artifact(
                    recipe_path,
                    "packedkv_q1_long_context_input_recipe",
                )
            )
            long_metrics_artifact = _with_content_hash(
                {
                    "schema_version": METRICS_SCHEMA,
                    "profile_id": request["profile_id"],
                    "binding_id": binding["binding_id"],
                    "trace": metrics,
                }
            )
            long_metrics_path = profile_root / f"{prefix}-metrics.json"
            _ensure_confined(long_metrics_path, artifact_root)
            _write_json(long_metrics_path, long_metrics_artifact)
            artifacts.append(
                _artifact(
                    long_metrics_path,
                    "packedkv_q1_long_context_metrics",
                )
            )
            tests.append(
                {
                    "name": "packedkv_q1_long_context_scaled",
                    "passed": bool(
                        metrics["batch_size"] == LONG_CONTEXT_BATCH
                        and metrics["q_len"] == 1
                        and metrics["cache_position"]
                        == metrics["cache_tokens"] - 1
                        and metrics["cache_tokens"]
                        > metrics["mlen"]
                        and metrics["sequence_blocks_per_selector"] == 3
                        and metrics["final_sequence_block_tokens"] == 1
                        and metrics["selector_count_valid"]
                        and metrics["selector_sequence_valid"]
                        and metrics["batch_slab_mapping_valid"]
                        and metrics["sequence_block_count_valid"]
                        and metrics["partial_final_sequence_block_valid"]
                        and metrics["physical_cache_slab_offsets_valid"]
                        and metrics["compact_state_addresses_valid"]
                        and metrics["no_loop_clobbered_accumulator"]
                        and metrics[
                            "matrix_compute_writeout_pairing_valid"
                        ]
                        and metrics["no_dynamic_kv_group_loop"]
                        and metrics["machine_word_count_valid"]
                        and metrics["execution_contract_valid"]
                        and metrics["kv_prefetch_precision_valid"]
                        and metrics["structural_precision_binding_valid"]
                    ),
                    "metrics": metrics,
                }
            )
        metrics_artifact = _with_content_hash(
            {
                "schema_version": METRICS_SCHEMA,
                "profile_id": request["profile_id"],
                "binding_id": binding["binding_id"],
                "traces": trace_metrics,
            }
        )
        metrics_path = profile_root / "metrics.json"
        _ensure_confined(metrics_path, artifact_root)
        _write_json(metrics_path, metrics_artifact)
        artifacts.append(_artifact(metrics_path, "compiler_trace_metrics"))

    result = _with_content_hash(
        {
            "schema_version": RESULT_SCHEMA,
            "stage": "compiler",
            "manifest_hash": request["manifest_hash"],
            "profile_id": request["profile_id"],
            "request_content_hash": request["content_hash"],
            "observed_at_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "tests": tests,
            "artifacts": artifacts,
        }
    )
    _write_json(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_hook(args.request, args.result, args.artifact_dir)
    except Exception as exc:
        print(f"compiler PackedKV hook failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
