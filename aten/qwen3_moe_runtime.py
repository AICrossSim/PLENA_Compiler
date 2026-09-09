"""Fail-closed runtime-routing contract for Qwen3-30B-A3B decode.

This module deliberately separates three claims: target structure can be
audited, the Transformers router mathematics has a CPU oracle, and the
emulator/compiler route-score path retains FP32 through the weighted output
cast. Router-linear fixtures are exact for the validation and target shapes.
End-to-end parity remains blocked at the complete layer transaction, the
per-stage BF16 switch, and full-frontend/RTL validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import struct
from typing import Any
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F

from compiler.aten.model_extract import ModelConfig, extract_model_config, find_model_root


QWEN3_MOE_MODEL_ID = "Qwen/Qwen3-30B-A3B-Thinking-2507"
QWEN3_MOE_MODEL_REVISION = "3ca25493489e939d65b4161677cc24154138d127"
QWEN3_MOE_TRANSFORMERS_ABI = "transformers==5.5.0:Qwen3MoeTopKRouter"
QWEN3_MOE_ROUTE_ISA_ABI = "V_TOPK@0x37+V_MUL_ROUTE_F32@0x38"
QWEN3_ROUTER_LINEAR_ISA_ABI = "V_ROUTER_LINEAR_BF16@0x36"
QWEN3_EXPERT_COMBINE_ISA_ABI = "V_QWEN3_EXPERT_COMBINE_BF16@0x39"
QWEN3_RMSNORM_ISA_ABI = "V_QWEN3_RMSNORM_BF16@0x3a"
QWEN3_RAW_BF16_EXPERT_DESCRIPTOR_MAGIC = b"Q3MOEBF1"
QWEN3_RAW_BF16_EXPERT_DESCRIPTOR_BYTES = 64
QWEN3_ROUTE_F32_SRAM_ENTRIES = 1024
QWEN3_ROUTE_SCORES_PER_TOKEN = 8
QWEN3_MAX_RESIDENT_ROUTED_TOKENS = (
    QWEN3_ROUTE_F32_SRAM_ENTRIES // QWEN3_ROUTE_SCORES_PER_TOKEN
)
QWEN3_MOE_TRANSACTION_BLOCKER = (
    "complete_runtime_moe_layer_emulator_transaction_parity_not_verified"
)


@dataclass(frozen=True)
class Qwen3RawBf16ExpertBankLayout:
    """Canonical raw-BF16 fused-expert HBM layout for exact validation."""

    descriptor_base: int
    hidden: int
    intermediate: int
    expert_count: int
    gate_up_base: int
    down_base: int
    gate_up_stride_bytes: int
    down_stride_bytes: int
    end_address: int

    @classmethod
    def canonical(
        cls,
        descriptor_base: int,
        *,
        hidden: int,
        intermediate: int,
        expert_count: int = 128,
    ) -> Qwen3RawBf16ExpertBankLayout:
        if descriptor_base < 0 or descriptor_base % 64:
            raise ValueError("Qwen3 expert descriptor base must be 64-byte aligned")
        if (hidden, intermediate) not in ((64, 64), (2048, 768)):
            raise ValueError(
                "exact Qwen3 expert layout supports (64,64) validation or "
                "(2048,768) target geometry"
            )
        if expert_count != 128:
            raise ValueError("Qwen3 expert layout requires exactly 128 experts")
        gate_up_stride = 2 * intermediate * hidden * 2
        down_stride = hidden * intermediate * 2
        gate_up_base = descriptor_base + QWEN3_RAW_BF16_EXPERT_DESCRIPTOR_BYTES
        down_base = gate_up_base + expert_count * gate_up_stride
        end_address = down_base + expert_count * down_stride
        if any(
            value % 64
            for value in (
                gate_up_base,
                down_base,
                gate_up_stride,
                down_stride,
                end_address,
            )
        ):
            raise AssertionError("canonical Qwen3 raw-BF16 layout lost alignment")
        return cls(
            descriptor_base=descriptor_base,
            hidden=hidden,
            intermediate=intermediate,
            expert_count=expert_count,
            gate_up_base=gate_up_base,
            down_base=down_base,
            gate_up_stride_bytes=gate_up_stride,
            down_stride_bytes=down_stride,
            end_address=end_address,
        )

    @property
    def total_bytes(self) -> int:
        return self.end_address - self.descriptor_base

    def descriptor_bytes(self) -> bytes:
        descriptor = struct.pack(
            "<8sIIIIQQQQQ",
            QWEN3_RAW_BF16_EXPERT_DESCRIPTOR_MAGIC,
            1,
            self.hidden,
            self.intermediate,
            self.expert_count,
            self.gate_up_base,
            self.down_base,
            self.gate_up_stride_bytes,
            self.down_stride_bytes,
            self.end_address,
        )
        if len(descriptor) != QWEN3_RAW_BF16_EXPERT_DESCRIPTOR_BYTES:
            raise AssertionError("Qwen3 expert descriptor ABI is not 64 bytes")
        return descriptor

    def as_dict(self) -> dict[str, Any]:
        contract = {
            "schema_version": 1,
            "descriptor_magic": QWEN3_RAW_BF16_EXPERT_DESCRIPTOR_MAGIC.decode(),
            "descriptor_version": 1,
            "descriptor_base": self.descriptor_base,
            "descriptor_bytes": QWEN3_RAW_BF16_EXPERT_DESCRIPTOR_BYTES,
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "expert_count": self.expert_count,
            "gate_up": {
                "base": self.gate_up_base,
                "shape": [self.expert_count, 2 * self.intermediate, self.hidden],
                "expert_stride_bytes": self.gate_up_stride_bytes,
            },
            "down": {
                "base": self.down_base,
                "shape": [self.expert_count, self.hidden, self.intermediate],
                "expert_stride_bytes": self.down_stride_bytes,
            },
            "end_address_exclusive": self.end_address,
            "total_bytes": self.total_bytes,
            "dtype": "BF16 little-endian",
            "matrix_order": "expert-major row-major output_by_input",
            "scale_plane": None,
            "canonical_contiguous": True,
        }
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        return {**contract, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def pack_qwen3_raw_bf16_expert_hbm(
    layout: Qwen3RawBf16ExpertBankLayout,
    fused_gate_up: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    max_materialized_bytes: int = 64 * 1024 * 1024,
) -> bytes:
    """Pack the exact tiny validation layout; target layout stays structural."""

    if layout.total_bytes > max_materialized_bytes:
        raise ValueError(
            f"raw BF16 expert image is {layout.total_bytes} bytes, above the "
            f"explicit materialization limit {max_materialized_bytes}"
        )
    expected_gate_up = (
        layout.expert_count,
        2 * layout.intermediate,
        layout.hidden,
    )
    expected_down = (
        layout.expert_count,
        layout.hidden,
        layout.intermediate,
    )
    if tuple(fused_gate_up.shape) != expected_gate_up:
        raise ValueError(
            f"fused_gate_up shape {tuple(fused_gate_up.shape)} != {expected_gate_up}"
        )
    if tuple(down_proj.shape) != expected_down:
        raise ValueError(f"down_proj shape {tuple(down_proj.shape)} != {expected_down}")
    if not fused_gate_up.is_floating_point() or not down_proj.is_floating_point():
        raise TypeError("raw BF16 expert tensors must be floating point")
    gate_up = fused_gate_up.to(dtype=torch.bfloat16, device="cpu").contiguous()
    down = down_proj.to(dtype=torch.bfloat16, device="cpu").contiguous()
    if not torch.isfinite(gate_up.float()).all() or not torch.isfinite(down.float()).all():
        raise ValueError("raw BF16 expert tensors must be finite")
    gate_up_bytes = gate_up.view(torch.uint8).numpy().tobytes()
    down_bytes = down.view(torch.uint8).numpy().tobytes()
    if len(gate_up_bytes) != layout.expert_count * layout.gate_up_stride_bytes:
        raise AssertionError("fused gate/up raw byte length does not match layout")
    if len(down_bytes) != layout.expert_count * layout.down_stride_bytes:
        raise AssertionError("down raw byte length does not match layout")
    image = bytearray(layout.total_bytes)
    image[:QWEN3_RAW_BF16_EXPERT_DESCRIPTOR_BYTES] = layout.descriptor_bytes()
    gate_offset = layout.gate_up_base - layout.descriptor_base
    down_offset = layout.down_base - layout.descriptor_base
    image[gate_offset : gate_offset + len(gate_up_bytes)] = gate_up_bytes
    image[down_offset : down_offset + len(down_bytes)] = down_bytes
    return bytes(image)


QWEN3_MOE_RUNTIME_STAGES = (
    "router_bf16",
    "topk8_runtime",
    "dispatch_runtime_expert_id",
    "expert_gate",
    "expert_up",
    "expert_swiglu",
    "expert_down",
    "route_weight",
    "scatter_combine",
)

QWEN3_ATTENTION_DECODE_STAGES = (
    "input_rmsnorm",
    "qkv_projection",
    "qk_rmsnorm_rope",
    "kv_cache_update",
    "gqa_decode_attention",
    "attention_output_projection",
    "attention_residual",
    "post_attention_rmsnorm",
)

QWEN3_FULL_LAYER_DATAFLOW_STAGES = (
    *QWEN3_ATTENTION_DECODE_STAGES,
    *QWEN3_MOE_RUNTIME_STAGES,
    "moe_residual",
)

QWEN3_MOE_STUDY_BLOCKERS = (
    "full_model_moe_frontend_not_wired",
    QWEN3_MOE_TRANSACTION_BLOCKER,
    "independent_bf16_router_precision_switch_missing",
    "router_and_fp32_route_isa_timing_uncalibrated_and_rtl_unsupported",
)


def qwen3_route_isa_provenance_receipt() -> dict[str, Any]:
    """Describe the exact route-score data path and its claim boundary."""

    contract = {
        "schema_version": 1,
        "abi": QWEN3_MOE_ROUTE_ISA_ABI,
        "topk_opcode": "V_TOPK",
        "route_multiply_opcode": "V_MUL_ROUTE_F32",
        "input_logits_dtype": "BF16",
        "softmax_dtype": "FP32",
        "topk_dtype": "FP32",
        "renormalization_dtype": "FP32",
        "route_sram_dtype": "FP32",
        "route_sram_entries": QWEN3_ROUTE_F32_SRAM_ENTRIES,
        "route_sram_bytes": QWEN3_ROUTE_F32_SRAM_ENTRIES * 4,
        "route_scores_per_token": QWEN3_ROUTE_SCORES_PER_TOKEN,
        "resident_route_schedule": "whole_batch",
        "max_resident_routed_tokens": QWEN3_MAX_RESIDENT_ROUTED_TOKENS,
        "emulator_route_dump": "route_f32_sram_dump.bin",
        "route_multiply_dtype": "FP32",
        "output_cast_dtype": "BF16",
        "functional_emulator_implementation_available": True,
        "timing_model": "structural_existing_vector_primitive_composition",
        "topk_timing_basis": (
            "ceil(experts/VLEN)*V_RED_MAX+V_ADD+V_EXP+V_RED_SUM+V_RECI+V_MUL"
        ),
        "route_multiply_timing_basis": "V_MUL",
        "timing_calibrated": False,
        "rtl_valid": False,
        "publication_rankable": False,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {**contract, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def qwen3_router_linear_provenance_receipt() -> dict[str, Any]:
    """Describe the exact BF16 router-linear emulator path."""

    contract = {
        "schema_version": 1,
        "abi": QWEN3_ROUTER_LINEAR_ISA_ABI,
        "opcode": "V_ROUTER_LINEAR_BF16",
        "transformers_abi": QWEN3_MOE_TRANSFORMERS_ABI,
        "emulator_binding": "tch==0.20.0",
        "reference_operation": "BF16 F.linear with one BF16 logits output cast",
        "input_dtype": "BF16",
        "weight_dtype": "BF16",
        "output_dtype": "BF16",
        "output_cast_count": 1,
        "expert_count": 128,
        "policies": {"0": {"hidden_size": 64}, "1": {"hidden_size": 2048}},
        "functional_emulator_implementation_available": True,
        "transformers_5_5_cpu_fixture_parity_valid": True,
        "router_to_topk_transformers_5_5_fixture_parity_valid": True,
        "fixture_hash_algorithm": "FNV-1a-64 over little-endian BF16 logits",
        "hidden64_fixture_hash": "b678a0d7fe63550a",
        "hidden2048_fixture_hash": "81e382ff292ffd8e",
        "emulator_logits_dump": "vram_dump.bin",
        "timing_model": "structural_existing_vector_primitive_composition",
        "timing_basis": "128*ceil(hidden/VLEN)*(V_MUL+V_RED_SUM)",
        "timing_calibrated": False,
        "rtl_valid": False,
        "publication_rankable": False,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {**contract, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def qwen3_moe_tail_isa_provenance_receipt() -> dict[str, Any]:
    """Pin the executable exact-BF16 post-attention MoE-tail fixture."""

    tiny_layout = Qwen3RawBf16ExpertBankLayout.canonical(
        0, hidden=64, intermediate=64
    )
    contract = {
        "schema_version": 1,
        "abi": "+".join(
            (
                QWEN3_RMSNORM_ISA_ABI,
                QWEN3_ROUTER_LINEAR_ISA_ABI,
                QWEN3_MOE_ROUTE_ISA_ABI,
                QWEN3_EXPERT_COMBINE_ISA_ABI,
            )
        ),
        "transformers_abi": QWEN3_MOE_TRANSFORMERS_ABI,
        "scope": "post_attention_moe_tail",
        "packedkv_attention_prefix_executed": False,
        "input_contract": (
            "pre-staged BF16 attention output projection plus BF16 layer residual"
        ),
        "rmsnorm": (
            "FP32 variance/rsqrt -> BF16 normalized cast -> BF16 affine multiply"
        ),
        "router": "BF16 linear -> FP32 softmax/top8/renormalization",
        "expert_hbm": "raw BF16 fused gate_up/down selected banks",
        "expert_order": "ascending expert id",
        "route_multiply": "FP32 then one BF16 contribution cast",
        "scatter_accumulator": "BF16 index_add per ascending expert",
        "final_residual": "BF16 add",
        "tiny_geometry": {"batch": 1, "hidden": 64, "intermediate": 64},
        "tiny_expert_layout": tiny_layout.as_dict(),
        "tiny_selected_expert_ids_topk_order": [29, 99, 33, 95, 8, 70, 91, 50],
        "tiny_normalized_bf16_fnv1a64": "07ebbab06edc1369",
        "tiny_output_bf16_fnv1a64": "910c569b727d0efc",
        "tiny_hbm_bytes_read": 64 + 8 * (16_384 + 8_192),
        "transformers_5_5_cpu_fixture_parity_valid": True,
        "emulator_dispatch_transaction_parity_valid": True,
        "compiler_lowering_and_assembler_valid": True,
        "compiler_generated_binary_emulator_parity_valid": False,
        "functional_emulator_implementation_available": True,
        "timing_model": "structural_uncalibrated",
        "timing_calibrated": False,
        "rtl_valid": False,
        "publication_rankable": False,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return {**contract, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def qwen3_tiny_single_layer_transaction_provenance_receipt() -> dict[str, Any]:
    """Describe the executable hidden-64 attention-to-MoE validation proof."""

    contract = {
        "schema_version": 1,
        "receipt_schema": "plena-qwen3-tiny-single-layer-decode-transaction-v1",
        "scope": "hidden64_q_len1_validation_only",
        "geometry": {
            "batch_size": 1,
            "q_len": 1,
            "hidden": 64,
            "head_dim": 64,
            "kv_heads": 1,
            "intermediate": 64,
            "experts": 128,
            "top_k": 8,
            "cache_position": 3,
        },
        "ordered_prefix": list(QWEN3_ATTENTION_DECODE_STAGES),
        "ordered_tail": [*QWEN3_MOE_RUNTIME_STAGES, "moe_residual"],
        "packedkv_attention_prefix_executed": True,
        "qk_rmsnorm_executed": True,
        "kv_cache_append_and_read_executed": True,
        "compiler_generated_binary_emulator_parity_valid": True,
        "all_bf16_boundaries_byte_exact": True,
        "route_scores_fp32_byte_exact": True,
        "route_assignment_conservation_verified": True,
        "kv_append_and_untouched_byte_conservation_verified": True,
        "content_addressed_inputs_outputs_trace": True,
        "pytorch_transformers_5_5_compatible_oracle": True,
        "transformers_abi": QWEN3_MOE_TRANSFORMERS_ABI,
        "harness": (
            "transactional_emulator/testbench/aten/"
            "qwen3_moe_single_layer_transaction.py"
        ),
        "target_geometry_valid": False,
        "full_model_compiler_valid": False,
        "emulator_target_valid": False,
        "rtl_valid": False,
        "timing_calibrated": False,
        "publication_rankable": False,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return {**contract, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


@dataclass(frozen=True)
class RouterPrecisionContract:
    """Numerical contract of the official Qwen3-MoE router.

    Expert precision is intentionally absent: it may be swept independently
    while router arithmetic stays sealed. The emulator executes this contract
    when the global vector format is BF16. The current ISA cannot enforce it
    independently when expert arithmetic uses a narrow global format.
    """

    hidden_state_dtype: str = "BF16"
    router_weight_dtype: str = "BF16"
    router_linear_output_dtype: str = "BF16"
    softmax_dtype: str = "FP32"
    topk_input_dtype: str = "FP32"
    renormalization_dtype: str = "FP32"
    route_weight_dtype: str = "FP32"
    top_k: int = 8
    num_experts: int = 128
    norm_topk_prob: bool = True
    transformers_abi: str = QWEN3_MOE_TRANSFORMERS_ABI

    @property
    def semantic_hash(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_target(self) -> None:
        expected = RouterPrecisionContract()
        if self != expected:
            differences = {
                name: (getattr(expected, name), getattr(self, name))
                for name in asdict(expected)
                if getattr(self, name) != getattr(expected, name)
            }
            raise ValueError(
                "not the sealed Qwen3-MoE router precision contract: "
                f"{differences}"
            )

    def bind_expert_vector_precision(self, value: Any) -> dict[str, Any]:
        """Return a receipt proving expert format does not alter router math."""

        if value is None or str(value).strip() == "":
            raise ValueError("expert vector precision must be explicit")
        return {
            "router_precision_contract": asdict(self),
            "router_precision_contract_hash": self.semantic_hash,
            "expert_vector_precision": value,
            "semantically_independent": True,
            "cpu_reference_valid": True,
            "emulator_router_linear_fixture_valid": True,
            "emulator_topk_opcode_available": True,
            "emulator_fp32_route_path_available": True,
            "plena_per_stage_switch_valid": False,
            "publication_rankable": False,
            "blocker": "independent_bf16_router_precision_switch_missing",
        }


QWEN3_MOE_ROUTER_PRECISION = RouterPrecisionContract()


@dataclass(frozen=True)
class RouterReferenceResult:
    """Outputs and conservation receipt from the CPU router oracle."""

    raw_logits: torch.Tensor
    probabilities: torch.Tensor
    route_scores: torch.Tensor
    expert_indices: torch.Tensor
    contract: RouterPrecisionContract = QWEN3_MOE_ROUTER_PRECISION

    def transaction_receipt(self) -> dict[str, Any]:
        tokens = int(self.expert_indices.shape[0])
        actual = int(self.expert_indices.numel())
        expected = tokens * self.contract.top_k
        if actual != expected:
            raise ValueError(
                f"router assignment conservation failed: {actual} != {expected}"
            )
        if self.expert_indices.numel() and (
            int(self.expert_indices.min()) < 0
            or int(self.expert_indices.max()) >= self.contract.num_experts
        ):
            raise ValueError("router oracle produced an out-of-range expert id")
        row_sums = self.route_scores.float().sum(dim=-1)
        return {
            "scope": "cpu_router_reference_only",
            "transformers_abi": self.contract.transformers_abi,
            "precision_contract_hash": self.contract.semantic_hash,
            "tokens": tokens,
            "expected_assignments": expected,
            "actual_assignments": actual,
            "assignment_conserved": actual == expected,
            "raw_logits_dtype": str(self.raw_logits.dtype).removeprefix("torch."),
            "probabilities_dtype": str(self.probabilities.dtype).removeprefix(
                "torch."
            ),
            "route_scores_dtype": str(self.route_scores.dtype).removeprefix(
                "torch."
            ),
            "expert_indices_dtype": str(self.expert_indices.dtype).removeprefix(
                "torch."
            ),
            "renormalized_row_sum_min": float(row_sums.min()) if tokens else None,
            "renormalized_row_sum_max": float(row_sums.max()) if tokens else None,
            "cpu_reference_valid": True,
            "emulator_router_linear_fixture_valid": True,
            "emulator_topk_opcode_available": True,
            "emulator_fp32_route_path_available": True,
            "route_isa": qwen3_route_isa_provenance_receipt(),
            "router_linear": qwen3_router_linear_provenance_receipt(),
            "emulator_equivalence_valid": False,
            "publication_rankable": False,
            "blocker": QWEN3_MOE_TRANSACTION_BLOCKER,
        }


@dataclass(frozen=True)
class RoutedMoeReferenceResult:
    """CPU result for router, expert dispatch and weighted scatter/combine."""

    output: torch.Tensor
    router: RouterReferenceResult
    assignments_per_expert: tuple[int, ...]

    def transaction_receipt(self) -> dict[str, Any]:
        router_receipt = self.router.transaction_receipt()
        actual = sum(self.assignments_per_expert)
        expected = router_receipt["expected_assignments"]
        if actual != expected:
            raise ValueError(
                f"expert dispatch conservation failed: {actual} != {expected}"
            )
        return {
            "scope": "cpu_routed_moe_reference_only",
            "router": router_receipt,
            "expected_assignments": expected,
            "actual_assignments": actual,
            "assignment_conserved": actual == expected,
            "active_expert_count": sum(
                count > 0 for count in self.assignments_per_expert
            ),
            "assignments_per_expert": list(self.assignments_per_expert),
            "output_dtype": str(self.output.dtype).removeprefix("torch."),
            "cpu_routed_moe_reference_valid": True,
            "emulator_router_linear_fixture_valid": True,
            "emulator_topk_opcode_available": True,
            "emulator_fp32_route_path_available": True,
            "route_isa": qwen3_route_isa_provenance_receipt(),
            "emulator_equivalence_valid": False,
            "publication_rankable": False,
            "blocker": QWEN3_MOE_TRANSACTION_BLOCKER,
        }


@dataclass(frozen=True)
class Qwen3PostAttentionMoeReferenceResult:
    """Exact BF16 result for the executable post-attention MoE tail."""

    output: torch.Tensor
    attention_residual: torch.Tensor
    normalized: torch.Tensor
    moe: RoutedMoeReferenceResult

    def transaction_receipt(self) -> dict[str, Any]:
        moe_receipt = self.moe.transaction_receipt()
        return {
            "scope": "post_attention_moe_tail_cpu_reference",
            "input_contract": (
                "BF16 PackedKV attention output projection plus BF16 layer residual"
            ),
            "rmsnorm": (
                "FP32 variance/rsqrt -> BF16 normalized cast -> BF16 affine multiply"
            ),
            "router": "BF16 linear -> FP32 softmax/top8/renorm",
            "expert": (
                "BF16 fused gate/up linear -> BF16 SiLU/multiply -> BF16 down linear"
            ),
            "combine": (
                "FP32 route multiply -> BF16 cast -> ascending-expert BF16 index_add"
            ),
            "final_residual": "BF16 add",
            "output_dtype": str(self.output.dtype).removeprefix("torch."),
            "moe": moe_receipt,
            "packedkv_attention_prefix_executed": False,
            "cpu_reference_valid": True,
            "emulator_transaction_parity": False,
            "publication_rankable": False,
            "blocker": QWEN3_MOE_TRANSACTION_BLOCKER,
        }


def qwen3_moe_router_cpu_reference(
    hidden_states: torch.Tensor,
    router_weight_rows: torch.Tensor,
    *,
    contract: RouterPrecisionContract = QWEN3_MOE_ROUTER_PRECISION,
) -> RouterReferenceResult:
    """Execute the Transformers 5.5 BF16/FP32 Qwen3-MoE router on CPU.

    Operation order mirrors ``Qwen3MoeTopKRouter.forward``: BF16 linear, FP32
    softmax, FP32 top-k and FP32 selected-probability renormalization. This is a
    numerical oracle, not an emulator implementation.
    """

    contract.validate_target()
    if not isinstance(hidden_states, torch.Tensor) or not isinstance(
        router_weight_rows, torch.Tensor
    ):
        raise TypeError("router inputs must be torch.Tensor instances")
    if hidden_states.ndim < 2:
        raise ValueError(
            f"hidden_states must have rank >= 2, got {tuple(hidden_states.shape)}"
        )
    hidden = int(hidden_states.shape[-1])
    if tuple(router_weight_rows.shape) != (contract.num_experts, hidden):
        raise ValueError(
            "router_weight_rows must have shape "
            f"({contract.num_experts}, {hidden}), got "
            f"{tuple(router_weight_rows.shape)}"
        )
    if (
        not hidden_states.is_floating_point()
        or not router_weight_rows.is_floating_point()
    ):
        raise TypeError("router inputs must be floating-point tensors")

    flat_hidden = hidden_states.reshape(-1, hidden).to(dtype=torch.bfloat16)
    weight = router_weight_rows.to(
        device=flat_hidden.device, dtype=torch.bfloat16
    )
    raw_logits = F.linear(flat_hidden, weight)
    if raw_logits.dtype != torch.bfloat16:
        raise RuntimeError(
            f"BF16 router linear returned unexpected dtype {raw_logits.dtype}"
        )
    probabilities = F.softmax(raw_logits, dtype=torch.float32, dim=-1)
    route_scores, expert_indices = torch.topk(
        probabilities, contract.top_k, dim=-1
    )
    if contract.norm_topk_prob:
        denominator = route_scores.sum(dim=-1, keepdim=True)
        if not torch.isfinite(denominator).all() or not (denominator > 0).all():
            raise RuntimeError(
                "router top-k denominator is non-finite or non-positive"
            )
        route_scores = route_scores / denominator
    if not (
        torch.isfinite(probabilities).all()
        and torch.isfinite(route_scores).all()
    ):
        raise RuntimeError("router oracle produced non-finite probabilities")
    return RouterReferenceResult(
        raw_logits=raw_logits,
        probabilities=probabilities,
        route_scores=route_scores,
        expert_indices=expert_indices,
        contract=contract,
    )


def qwen3_moe_cpu_reference(
    hidden_states: torch.Tensor,
    router_weight_rows: torch.Tensor,
    fused_gate_up: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    contract: RouterPrecisionContract = QWEN3_MOE_ROUTER_PRECISION,
) -> RoutedMoeReferenceResult:
    """Execute the fused routed-MLP part of the Transformers 5.5 ABI on CPU.

    Expert arithmetic and scatter accumulation are BF16, while route scores
    retain the router's FP32 contract until each weighted expert output is cast
    back to the BF16 combine buffer. Attention and residual addition are outside
    this narrowly scoped oracle.
    """

    router = qwen3_moe_router_cpu_reference(
        hidden_states, router_weight_rows, contract=contract
    )
    hidden = int(hidden_states.shape[-1])
    if fused_gate_up.ndim != 3 or down_proj.ndim != 3:
        raise ValueError("fused expert weights must be rank-3 tensors")
    if fused_gate_up.shape[0] != contract.num_experts:
        raise ValueError(
            f"fused_gate_up must contain {contract.num_experts} experts"
        )
    if fused_gate_up.shape[2] != hidden or fused_gate_up.shape[1] % 2:
        raise ValueError(
            "fused_gate_up must have shape (experts, 2*intermediate, hidden)"
        )
    intermediate = int(fused_gate_up.shape[1] // 2)
    if tuple(down_proj.shape) != (
        contract.num_experts,
        hidden,
        intermediate,
    ):
        raise ValueError(
            "down_proj must have shape "
            f"({contract.num_experts}, {hidden}, {intermediate}), got "
            f"{tuple(down_proj.shape)}"
        )
    if not fused_gate_up.is_floating_point() or not down_proj.is_floating_point():
        raise TypeError("expert weights must be floating-point tensors")

    flat_hidden = hidden_states.reshape(-1, hidden).to(dtype=torch.bfloat16)
    final = torch.zeros_like(flat_hidden)
    counts = [0] * contract.num_experts
    for expert_id in range(contract.num_experts):
        token_indices, topk_positions = torch.where(
            router.expert_indices == expert_id
        )
        if not token_indices.numel():
            continue
        counts[expert_id] = int(token_indices.numel())
        current = flat_hidden[token_indices]
        gate_up_weight = fused_gate_up[expert_id].to(
            device=current.device, dtype=torch.bfloat16
        )
        gate, up = F.linear(current, gate_up_weight).chunk(2, dim=-1)
        activated = F.silu(gate) * up
        expert_output = F.linear(
            activated,
            down_proj[expert_id].to(
                device=current.device, dtype=torch.bfloat16
            ),
        )
        weighted = expert_output * router.route_scores[
            token_indices, topk_positions, None
        ]
        final.index_add_(0, token_indices, weighted.to(dtype=final.dtype))
    return RoutedMoeReferenceResult(
        output=final.reshape(hidden_states.shape),
        router=router,
        assignments_per_expert=tuple(counts),
    )


def qwen3_post_attention_moe_cpu_reference(
    attention_output: torch.Tensor,
    layer_residual: torch.Tensor,
    post_attention_norm_weight: torch.Tensor,
    router_weight_rows: torch.Tensor,
    fused_gate_up: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    contract: RouterPrecisionContract = QWEN3_MOE_ROUTER_PRECISION,
) -> Qwen3PostAttentionMoeReferenceResult:
    """Execute the exact Transformers 5.5 post-attention MoE tail."""

    if tuple(attention_output.shape) != tuple(layer_residual.shape):
        raise ValueError("attention output and layer residual shapes must match")
    if attention_output.ndim < 2:
        raise ValueError("post-attention tensors must have rank >= 2")
    hidden = int(attention_output.shape[-1])
    if tuple(post_attention_norm_weight.shape) != (hidden,):
        raise ValueError(
            f"post-attention norm weight must have shape ({hidden},)"
        )
    if not all(
        tensor.is_floating_point()
        for tensor in (
            attention_output,
            layer_residual,
            post_attention_norm_weight,
        )
    ):
        raise TypeError("post-attention reference tensors must be floating point")
    attention_residual = (
        attention_output.to(dtype=torch.bfloat16)
        + layer_residual.to(dtype=torch.bfloat16)
    )
    fp32 = attention_residual.float()
    variance = fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = fp32 * torch.rsqrt(variance + float(eps))
    normalized = (
        normalized.to(dtype=torch.bfloat16)
        * post_attention_norm_weight.to(
            device=normalized.device, dtype=torch.bfloat16
        )
    )
    moe = qwen3_moe_cpu_reference(
        normalized,
        router_weight_rows,
        fused_gate_up,
        down_proj,
        contract=contract,
    )
    output = attention_residual + moe.output
    return Qwen3PostAttentionMoeReferenceResult(
        output=output,
        attention_residual=attention_residual,
        normalized=normalized,
        moe=moe,
    )


@dataclass(frozen=True)
class FullModelFrontendAuditReceipt:
    """Shape/ABI audit for a loaded or meta-initialized exact target model."""

    model_id: str
    revision: str
    transformers_abi: str
    audited_layer_count: int
    fused_expert_layer_count: int
    all_experts_addressable_per_layer: int
    top_k: int
    full_layer_dataflow_contract_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "scope": "full_model_frontend_structure_audit_only",
            "frontend_structure_audit_valid": True,
            "emulator_router_linear_fixture_valid": True,
            "emulator_fp32_route_path_available": True,
            "route_isa": qwen3_route_isa_provenance_receipt(),
            "router_linear": qwen3_router_linear_provenance_receipt(),
            "compiler_pipeline_valid": False,
            "runtime_routing_dataflow_wired": False,
            "weights_materialized_by_audit": False,
            "dense_fallback_forbidden": True,
            "timing_calibrated": False,
            "publication_rankable": False,
            "blockers": list(QWEN3_MOE_STUDY_BLOCKERS),
        }


def _shape(value: Any, label: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ValueError(f"{label} has no tensor shape")
    return tuple(int(dimension) for dimension in shape)


def _expect_shape(value: Any, expected: tuple[int, ...], label: str) -> None:
    actual = _shape(value, label)
    if actual != expected:
        raise ValueError(f"{label} has shape {actual}, expected {expected}")


def audit_qwen3_30b_a3b_full_model_frontend(
    model: Any,
    *,
    model_id: str,
    revision: str,
) -> FullModelFrontendAuditReceipt:
    """Audit all 48 fused layers without claiming executable lowering.

    Only module metadata and tensor shapes are inspected. This works for a
    meta-initialized model and does not materialize or transpose expert tables.
    """

    if model_id != QWEN3_MOE_MODEL_ID:
        raise ValueError(
            f"model_id must be {QWEN3_MOE_MODEL_ID!r}, got {model_id!r}"
        )
    if revision != QWEN3_MOE_MODEL_REVISION:
        raise ValueError(
            "model revision does not match the sealed target revision "
            f"{QWEN3_MOE_MODEL_REVISION}"
        )
    config = extract_model_config(model)
    # Reuse the exact target validator. Precision strings are irrelevant to a
    # structure audit, but tied K/V must still pass the canonical invariant.
    plan = build_qwen3_30b_a3b_decode_plan(
        config,
        layer_count=48,
        batch_size=1,
        key_precision="AUDIT",
        value_precision="AUDIT",
    )
    raw_config = getattr(getattr(model, "config", None), "text_config", None)
    if raw_config is None:
        raw_config = getattr(model, "config", None)
    if raw_config is None:
        raise ValueError("model has no HuggingFace config")
    if getattr(raw_config, "hidden_act", "silu") != "silu":
        raise ValueError("exact Qwen3-MoE target requires hidden_act='silu'")
    for name in (
        "shared_expert_intermediate_size",
        "num_shared_experts",
        "shared_expert_count",
    ):
        value = getattr(raw_config, name, None)
        if value not in (None, 0):
            raise ValueError(f"exact target forbids shared experts: {name}={value}")

    layers = find_model_root(model).layers
    if len(layers) != 48:
        raise ValueError(f"full target frontend requires 48 layers, got {len(layers)}")
    expected = {
        "q_proj": (4096, 2048),
        "o_proj": (2048, 4096),
        "k_proj": (512, 2048),
        "v_proj": (512, 2048),
        "router": (128, 2048),
        "gate_up_proj": (128, 1536, 2048),
        "down_proj": (128, 2048, 768),
        "input_norm": (2048,),
        "post_attn_norm": (2048,),
        "q_norm": (128,),
        "k_norm": (128,),
    }
    for layer_index, layer in enumerate(layers):
        attention = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        if attention is None or mlp is None or experts is None:
            raise ValueError(f"layer {layer_index} is not a fused Qwen3-MoE layer")
        values = {
            "q_proj": getattr(getattr(attention, "q_proj", None), "weight", None),
            "o_proj": getattr(getattr(attention, "o_proj", None), "weight", None),
            "k_proj": getattr(getattr(attention, "k_proj", None), "weight", None),
            "v_proj": getattr(getattr(attention, "v_proj", None), "weight", None),
            "router": getattr(getattr(mlp, "gate", None), "weight", None),
            "gate_up_proj": getattr(experts, "gate_up_proj", None),
            "down_proj": getattr(experts, "down_proj", None),
            "input_norm": getattr(
                getattr(layer, "input_layernorm", None), "weight", None
            ),
            "post_attn_norm": getattr(
                getattr(layer, "post_attention_layernorm", None), "weight", None
            ),
            "q_norm": getattr(getattr(attention, "q_norm", None), "weight", None),
            "k_norm": getattr(getattr(attention, "k_norm", None), "weight", None),
        }
        for name, expected_shape in expected.items():
            _expect_shape(
                values[name], expected_shape, f"layer {layer_index} {name}"
            )
        if any(
            hasattr(mlp, name)
            for name in ("shared_expert", "shared_expert_gate")
        ):
            raise ValueError(f"layer {layer_index} unexpectedly has a shared expert")
    return FullModelFrontendAuditReceipt(
        model_id=model_id,
        revision=revision,
        transformers_abi=QWEN3_MOE_TRANSFORMERS_ABI,
        audited_layer_count=48,
        fused_expert_layer_count=48,
        all_experts_addressable_per_layer=128,
        top_k=8,
        full_layer_dataflow_contract_sha256=(
            qwen3_full_frontend_contract_receipt(plan)["contract_sha256"]
        ),
    )


@dataclass(frozen=True)
class KvPrecisionContract:
    """Canonical tied K/V storage precision."""

    value: Any


@dataclass(frozen=True)
class RoutedMoeLayerPlan:
    layer_index: int
    batch_size: int
    num_experts: int = 128
    top_k: int = 8
    stages: tuple[str, ...] = QWEN3_MOE_RUNTIME_STAGES

    @property
    def expected_assignments(self) -> int:
        return self.batch_size * self.top_k

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "routing_mode": "runtime_topk",
            "router_precision": "BF16",
            "router_softmax_precision": "FP32",
            "router_execution_contract": {
                "semantic_precision_contract": asdict(
                    QWEN3_MOE_ROUTER_PRECISION
                ),
                "semantic_precision_contract_hash": (
                    QWEN3_MOE_ROUTER_PRECISION.semantic_hash
                ),
                "independent_from_expert_vector_precision": True,
                "required_global_vector_format": "BF16",
                "independent_precision_switch": False,
                "narrow_vector_profile": "analytic_only",
            },
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "expected_assignments": self.expected_assignments,
            "stages": list(self.stages),
            "shared_experts": 0,
        }


@dataclass(frozen=True)
class RoutedMoeDecodePlan:
    layers: tuple[RoutedMoeLayerPlan, ...]
    kv_precision: KvPrecisionContract
    hidden_size: int
    expert_intermediate_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    router_precision_contract: RouterPrecisionContract = (
        QWEN3_MOE_ROUTER_PRECISION
    )
    expert_vector_precision: Any | None = None

    @property
    def batch_size(self) -> int:
        return self.layers[0].batch_size

    @property
    def expected_assignments_per_layer(self) -> int:
        return self.layers[0].expected_assignments

    @property
    def expected_assignments_total(self) -> int:
        return sum(layer.expected_assignments for layer in self.layers)

    def validate_runtime_assignment_counts(
        self, counts: Mapping[int, int] | Sequence[int]
    ) -> dict[str, Any]:
        if isinstance(counts, Mapping):
            actual = {int(index): int(count) for index, count in counts.items()}
        else:
            actual = {index: int(count) for index, count in enumerate(counts)}
        expected_layers = {layer.layer_index for layer in self.layers}
        if set(actual) != expected_layers:
            raise ValueError(
                f"assignment counters cover layers {sorted(actual)}, expected "
                f"{sorted(expected_layers)}"
            )
        expected = self.expected_assignments_per_layer
        mismatches = {
            index: count for index, count in actual.items() if count != expected
        }
        if mismatches:
            raise ValueError(
                "runtime route conservation failed: each layer must dispatch "
                f"top_k*batch={expected} assignments, got {mismatches}"
            )
        return {
            "equation": "sum(expert_token_assignments) == top_k * batch",
            "expected_per_layer": expected,
            "actual_per_layer": actual,
            "expected_total": self.expected_assignments_total,
            "actual_total": sum(actual.values()),
            "conserved": True,
        }

    def study_validity_receipt(self) -> dict[str, Any]:
        """Report the boundary between the ISA substrate and study execution."""

        return {
            "compiler_substrate_valid": True,
            "frontend_structure_audit_available": True,
            "cpu_router_reference_valid": True,
            "emulator_router_linear_fixture_valid": True,
            "emulator_fp32_route_path_available": True,
            "emulator_post_attention_moe_tail_dispatch_fixture_valid": True,
            "single_layer_runtime_moe_lowering_available": True,
            "tiny_hidden64_single_layer_emulator_transaction_parity": True,
            "tiny_single_layer_transaction": (
                qwen3_tiny_single_layer_transaction_provenance_receipt()
            ),
            "single_layer_emulator_transaction_parity_scope": (
                "target_hidden2048_not_verified"
            ),
            "single_layer_emulator_transaction_parity": False,
            "route_isa": qwen3_route_isa_provenance_receipt(),
            "router_linear": qwen3_router_linear_provenance_receipt(),
            "post_attention_moe_tail": qwen3_moe_tail_isa_provenance_receipt(),
            "compiler_pipeline_valid": False,
            "emulator_valid": False,
            "rtl_valid": False,
            "timing_calibrated": False,
            "publication_rankable": False,
            "scope": "compiler_substrate_only",
            "blockers": list(QWEN3_MOE_STUDY_BLOCKERS),
        }

    def full_frontend_contract_receipt(self) -> dict[str, Any]:
        return qwen3_full_frontend_contract_receipt(self)

    def require_study_executable(self) -> None:
        blockers = ", ".join(QWEN3_MOE_STUDY_BLOCKERS)
        raise RuntimeError(f"Qwen3-MoE study execution is blocked: {blockers}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model": QWEN3_MOE_MODEL_ID,
            "model_revision": QWEN3_MOE_MODEL_REVISION,
            "transformers_router_abi": QWEN3_MOE_TRANSFORMERS_ABI,
            "routing_mode": "runtime_topk",
            "batch_size": self.batch_size,
            "compiled_layer_count": len(self.layers),
            "hidden_size": self.hidden_size,
            "expert_intermediate_size": self.expert_intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "kv_precision": self.kv_precision.value,
            "layers": [layer.as_dict() for layer in self.layers],
            "assignment_conservation": {
                "equation": "sum(expert_token_assignments) == top_k * batch",
                "expected_per_layer": self.expected_assignments_per_layer,
                "expected_total": self.expected_assignments_total,
                "runtime_counter_required": True,
            },
            "route_sram_capacity": {
                "schedule": "whole_batch",
                "entries": QWEN3_ROUTE_F32_SRAM_ENTRIES,
                "bytes": QWEN3_ROUTE_F32_SRAM_ENTRIES * 4,
                "scores_per_token": QWEN3_ROUTE_SCORES_PER_TOKEN,
                "max_resident_routed_tokens": QWEN3_MAX_RESIDENT_ROUTED_TOKENS,
                "required_entries": (
                    self.batch_size * QWEN3_ROUTE_SCORES_PER_TOKEN
                ),
                "capacity_valid": True,
                "token_serial_reuse_implemented": False,
            },
            "router_execution_contract": {
                "semantic_precision_contract": asdict(
                    self.router_precision_contract
                ),
                "semantic_precision_contract_hash": (
                    self.router_precision_contract.semantic_hash
                ),
                "expert_vector_precision": self.expert_vector_precision,
                "independent_from_expert_vector_precision": True,
                "cpu_reference_valid": True,
                "required_global_vector_format": "BF16",
                "independent_precision_switch": False,
                "narrow_vector_profile": "analytic_only",
                "reason": "the current ISA has no per-stage vector-precision switch",
            },
            "forbidden_shortcuts": [
                "dense_fallback",
                "static_expert_indices",
                "active_expert_only_residency",
                "single_layer_compile",
                "shared_expert",
            ],
            "full_frontend_contract": self.full_frontend_contract_receipt(),
            "study_validity": self.study_validity_receipt(),
        }


def qwen3_full_frontend_contract_receipt(
    plan: RoutedMoeDecodePlan,
) -> dict[str, Any]:
    """Pin the required 48-layer dataflow without claiming it is executable."""

    plan.router_precision_contract.validate_target()
    expected_dimensions = {
        "hidden_size": 2048,
        "expert_intermediate_size": 768,
        "num_attention_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 128,
    }
    actual_dimensions = {
        name: getattr(plan, name) for name in expected_dimensions
    }
    if actual_dimensions != expected_dimensions:
        raise ValueError(
            "full frontend contract requires exact target dimensions: "
            f"{actual_dimensions}"
        )
    if len(plan.layers) != 48:
        raise ValueError("full frontend contract requires all 48 layers")
    if [layer.layer_index for layer in plan.layers] != list(range(48)):
        raise ValueError("full frontend layers must be contiguous 0..47")
    if len({layer.batch_size for layer in plan.layers}) != 1:
        raise ValueError("all frontend layers must use the same batch size")
    for layer in plan.layers:
        if (
            layer.num_experts != 128
            or layer.top_k != 8
            or layer.stages != QWEN3_MOE_RUNTIME_STAGES
        ):
            raise ValueError(
                f"layer {layer.layer_index} violates the runtime MoE contract"
            )

    contract = {
        "schema_version": 1,
        "scope": "full_48_layer_attention_to_runtime_moe_contract_only",
        "model": QWEN3_MOE_MODEL_ID,
        "model_revision": QWEN3_MOE_MODEL_REVISION,
        "transformers_router_abi": QWEN3_MOE_TRANSFORMERS_ABI,
        "router_linear_isa_abi": QWEN3_ROUTER_LINEAR_ISA_ABI,
        "route_isa_abi": QWEN3_MOE_ROUTE_ISA_ABI,
        "layer_count": 48,
        "batch_size": plan.batch_size,
        "ordered_layer_dataflow": list(QWEN3_FULL_LAYER_DATAFLOW_STAGES),
        "attention": {
            "num_attention_heads": 32,
            "num_kv_heads": 4,
            "head_dim": 128,
            "qk_rmsnorm": True,
            "decode_kv_cache": True,
        },
        "routed_moe": {
            "routing_mode": "runtime_topk",
            "num_experts": 128,
            "top_k": 8,
            "all_expert_banks_addressable_per_layer": 128,
            "shared_experts": 0,
            "dense_fallback_forbidden": True,
            "router_linear_transformers_5_5_fixture_parity": True,
            "route_scores_retained_fp32_until_weighted_output_cast": True,
            "route_sram_schedule": "whole_batch",
            "route_sram_entries": QWEN3_ROUTE_F32_SRAM_ENTRIES,
            "max_resident_routed_tokens": QWEN3_MAX_RESIDENT_ROUTED_TOKENS,
            "single_layer_lowering_available": True,
            "tiny_hidden64_single_layer_emulator_transaction_parity": True,
            "single_layer_emulator_transaction_parity": False,
        },
        "layers": [
            {
                "layer_index": layer.layer_index,
                "ordered_dataflow": list(QWEN3_FULL_LAYER_DATAFLOW_STAGES),
                "expected_runtime_assignments": layer.expected_assignments,
                "all_expert_banks_addressable": 128,
            }
            for layer in plan.layers
        ],
        "attention_contract_available": True,
        "runtime_route_substrate_available": True,
        "end_to_end_lowering_available": False,
        "weights_materialized": False,
        "compiler_pipeline_valid": False,
        "emulator_valid": False,
        "rtl_valid": False,
        "timing_calibrated": False,
        "publication_rankable": False,
        "blockers": list(QWEN3_MOE_STUDY_BLOCKERS),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {**contract, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def _canonical_tied_kv(key_precision: Any, value_precision: Any) -> KvPrecisionContract:
    if key_precision != value_precision:
        raise ValueError(
            f"Qwen3-MoE requires canonical K=V precision, got "
            f"K={key_precision!r}, V={value_precision!r}"
        )
    return KvPrecisionContract(key_precision)


def build_qwen3_30b_a3b_decode_plan(
    config: ModelConfig,
    *,
    layer_count: int,
    batch_size: int,
    key_precision: Any,
    value_precision: Any,
    routing_mode: str = "runtime_topk",
    dense_fallback: bool = False,
    active_expert_ids: Sequence[int] | None = None,
    shared_expert_count: int = 0,
    router_precision_contract: RouterPrecisionContract = (
        QWEN3_MOE_ROUTER_PRECISION
    ),
    expert_vector_precision: Any | None = None,
) -> RoutedMoeDecodePlan:
    """Validate the exact target and build all 48 runtime-routed layer plans."""

    expected = {
        "model_type": "qwen3_moe",
        "num_hidden_layers": 48,
        "hidden_size": 2048,
        "num_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 128,
        "moe_inter_dim": 768,
        "num_experts": 128,
        "experts_per_token": 8,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": (),
    }
    actual = {name: getattr(config, name) for name in expected}
    mismatches = {
        name: (expected[name], actual[name])
        for name in expected
        if actual[name] != expected[name]
    }
    if mismatches:
        raise ValueError(f"not the exact Qwen3-30B-A3B target: {mismatches}")
    if layer_count != 48:
        raise ValueError(
            f"full target lowering requires all 48 layers, got {layer_count}"
        )
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if batch_size > QWEN3_MAX_RESIDENT_ROUTED_TOKENS:
        raise ValueError(
            "whole-batch Qwen3 routing exceeds FP32 route SRAM: "
            f"batch={batch_size} needs {batch_size * QWEN3_ROUTE_SCORES_PER_TOKEN} "
            f"scores, capacity={QWEN3_ROUTE_F32_SRAM_ENTRIES}; "
            "token-serial reuse is not implemented"
        )
    if routing_mode != "runtime_topk":
        raise ValueError(
            "Qwen3-30B-A3B lowering rejects host/static expert indices"
        )
    if dense_fallback:
        raise ValueError("dense fallback is forbidden for routed Qwen3-MoE layers")
    if active_expert_ids is not None:
        raise ValueError(
            "active-expert-only residency is invalid for runtime routing; "
            "all 128 experts must be addressable"
        )
    if shared_expert_count != 0:
        raise ValueError("this Qwen3 checkpoint has no shared experts")
    if not config.qk_norm:
        raise ValueError("Qwen3-MoE requires Q/K RMSNorm")
    router_precision_contract.validate_target()

    kv_precision = _canonical_tied_kv(key_precision, value_precision)
    layers = tuple(
        RoutedMoeLayerPlan(layer_index=index, batch_size=batch_size)
        for index in range(48)
    )
    return RoutedMoeDecodePlan(
        layers=layers,
        kv_precision=kv_precision,
        hidden_size=config.hidden_size,
        expert_intermediate_size=config.moe_inter_dim,
        num_attention_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        router_precision_contract=router_precision_contract,
        expert_vector_precision=expert_vector_precision,
    )


__all__ = [
    "QWEN3_ATTENTION_DECODE_STAGES",
    "QWEN3_EXPERT_COMBINE_ISA_ABI",
    "QWEN3_FULL_LAYER_DATAFLOW_STAGES",
    "QWEN3_MAX_RESIDENT_ROUTED_TOKENS",
    "QWEN3_MOE_MODEL_ID",
    "QWEN3_MOE_MODEL_REVISION",
    "QWEN3_MOE_ROUTER_PRECISION",
    "QWEN3_MOE_ROUTE_ISA_ABI",
    "QWEN3_MOE_RUNTIME_STAGES",
    "QWEN3_MOE_STUDY_BLOCKERS",
    "QWEN3_MOE_TRANSACTION_BLOCKER",
    "QWEN3_MOE_TRANSFORMERS_ABI",
    "QWEN3_RMSNORM_ISA_ABI",
    "QWEN3_ROUTER_LINEAR_ISA_ABI",
    "QWEN3_ROUTE_F32_SRAM_ENTRIES",
    "QWEN3_ROUTE_SCORES_PER_TOKEN",
    "FullModelFrontendAuditReceipt",
    "KvPrecisionContract",
    "Qwen3PostAttentionMoeReferenceResult",
    "Qwen3RawBf16ExpertBankLayout",
    "RoutedMoeDecodePlan",
    "RoutedMoeLayerPlan",
    "RoutedMoeReferenceResult",
    "RouterPrecisionContract",
    "RouterReferenceResult",
    "audit_qwen3_30b_a3b_full_model_frontend",
    "build_qwen3_30b_a3b_decode_plan",
    "pack_qwen3_raw_bf16_expert_hbm",
    "qwen3_full_frontend_contract_receipt",
    "qwen3_moe_cpu_reference",
    "qwen3_moe_router_cpu_reference",
    "qwen3_moe_tail_isa_provenance_receipt",
    "qwen3_post_attention_moe_cpu_reference",
    "qwen3_route_isa_provenance_receipt",
    "qwen3_router_linear_provenance_receipt",
]
