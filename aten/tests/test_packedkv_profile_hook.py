from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from compiler.aten.packedkv_profile_hook import (
    CONTEXT_SCOPE_SCHEMA,
    LONG_CONTEXT_CONTRACT_SCHEMA,
    MATRIX_SEMANTICS_SCHEMA,
    PROFILE_SCHEMA,
    REQUEST_SCHEMA,
    RESULT_SCHEMA,
    TARGET,
    _MATRIX_SEMANTICS,
    _assemble_trace,
    _canonical_bytes,
    _content_hash,
    run_hook,
)


def _profile(
    weight: str = "MXINT4",
    activation: str = "MXINT4",
    kv: str = "MXINT4",
) -> dict:
    profile = {
        "schema_version": PROFILE_SCHEMA,
        "kind": "quantized",
        "weight_format": weight,
        "activation_format": activation,
        "key_format": kv,
        "value_format": kv,
        "vector_format": "FP_E3M2",
        "block_size": 8,
        "scale_format": "E8M0",
        "scale_bits": 8,
        "accumulator_rule": "plena_fixed16_16_accumulate_truncate",
        "output_rule": "truncate_to_vector_format",
        "matrix_semantics": dict(_MATRIX_SEMANTICS),
        "method": "rtn",
        "operator_coverage": {
            "weight": ["attention_linear", "ffn_linear"],
            "activation": [
                "attention_linear",
                "ffn_linear",
                "qk_matmul",
                "pv_matmul",
            ],
            "kv": ["kv_cache", "qk_matmul", "pv_matmul"],
            "vector": [
                "input_rmsnorm",
                "post_attention_rmsnorm",
                "q_norm",
                "k_norm",
                "rope",
                "softmax",
                "silu_gate",
                "residual",
                "final_rmsnorm",
            ],
            "bf16": ["embedding", "lm_head"],
        },
    }
    return profile


def _request(
    profile: dict,
    *,
    run_long_context_capability: bool = False,
) -> dict:
    profile_id = "dqp-" + hashlib.sha256(_canonical_bytes(profile)).hexdigest()
    value = {
        "schema_version": REQUEST_SCHEMA,
        "stage": "compiler",
        "manifest_hash": "1" * 64,
        "profile_id": profile_id,
        "profile": profile,
        "target": dict(TARGET),
        "source_tree_sha256": "2" * 64,
        "hook_template_hash": "3" * 64,
        "environment_sha256": "4" * 64,
        "run_long_context_capability": run_long_context_capability,
    }
    value["content_hash"] = _content_hash(value)
    return value


class PackedKVProfileHookTests(unittest.TestCase):
    def _write_request(self, root: Path, request: dict) -> Path:
        path = root / "request.json"
        path.write_bytes(_canonical_bytes(request) + b"\n")
        return path

    def test_shift_mnemonic_alias_has_one_canonical_opcode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            canonical = root / "canonical.asm"
            rtl_alias = root / "rtl-alias.asm"
            canonical.write_text(
                "V_SHIFT_V gp1, gp2, gp3\n",
                encoding="utf-8",
            )
            rtl_alias.write_text(
                "V_SHFT_V gp1, gp2, gp3\n",
                encoding="utf-8",
            )
            canonical_binary, canonical_metrics = _assemble_trace(canonical)
            alias_binary, alias_metrics = _assemble_trace(rtl_alias)
            self.assertEqual(alias_binary, canonical_binary)
            self.assertEqual(
                alias_metrics["opcode_histogram"],
                {"V_SHIFT_V": 1},
            )
            self.assertTrue(
                canonical_metrics["execution_opcode_coverage_valid"]
            )
            self.assertTrue(
                alias_metrics["execution_opcode_coverage_valid"]
            )
            masked = root / "masked-reduction.asm"
            masked.write_text(
                "V_RED_MAX f1, gp2, 1\n",
                encoding="utf-8",
            )
            masked_binary, masked_metrics = _assemble_trace(masked)
            word = int(masked_binary.decode("ascii").strip(), 16)
            self.assertEqual((word >> 18) & 0xF, 1)
            self.assertEqual((word >> 14) & 0xF, 0)
            self.assertTrue(masked_metrics["execution_contract_valid"])

    def test_cross_target_operand_fields_encode_identically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assembly = Path(temporary) / "operands.asm"
            assembly.write_text(
                "\n".join(
                    (
                        "M_BTMM 1, gp2, gp3",
                        "M_MM_WO gp1, gp2, 0",
                        "H_PREFETCH_M gp1, gp2, a3, 1, 1",
                        "V_SUB_VF gp1, gp2, f3, 0, 1",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            payload, metrics = _assemble_trace(assembly)
            words = [
                int(line, 16)
                for line in payload.decode("ascii").splitlines()
            ]
            self.assertEqual(
                [word & 0x3F for word in words],
                [0x04, 0x06, 0x28, 0x10],
            )
            for word in words:
                self.assertEqual((word >> 6) & 0xF, 1)
                self.assertEqual((word >> 10) & 0xF, 2)
            self.assertEqual((words[0] >> 14) & 0xF, 3)
            self.assertEqual((words[2] >> 14) & 0xF, 3)
            self.assertEqual((words[2] >> 18) & 0xF, 1)
            self.assertEqual((words[2] >> 22) & 0xF, 1)
            self.assertEqual((words[3] >> 14) & 0xF, 3)
            self.assertEqual((words[3] >> 18) & 0xF, 0)
            self.assertEqual((words[3] >> 22) & 0xF, 1)
            self.assertTrue(metrics["execution_contract_valid"])

    def test_supported_profile_emits_q1_batch_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(
                _profile(),
                run_long_context_capability=True,
            )
            request_path = self._write_request(root, request)
            result_path = root / "result.json"
            artifact_dir = root / "artifacts"
            result = run_hook(request_path, result_path, artifact_dir)

            self.assertEqual(result["schema_version"], RESULT_SCHEMA)
            self.assertEqual(result["content_hash"], _content_hash(result))
            self.assertTrue(all(test["passed"] for test in result["tests"]))
            self.assertEqual(
                {
                    test["name"]
                    for test in result["tests"]
                    if test["name"].startswith("packedkv_q1_batch_")
                },
                {
                    "packedkv_q1_batch_1",
                    "packedkv_q1_batch_2",
                    "packedkv_q1_batch_4",
                },
            )
            artifact_kinds = [artifact["kind"] for artifact in result["artifacts"]]
            self.assertEqual(artifact_kinds.count("packedkv_q1_machine_code"), 3)
            self.assertEqual(artifact_kinds.count("packedkv_q1_trace_contract"), 3)
            self.assertEqual(artifact_kinds.count("packedkv_q1_input_recipe"), 3)
            self.assertEqual(
                artifact_kinds.count(
                    "packedkv_q1_long_context_assembly"
                ),
                1,
            )
            self.assertEqual(
                artifact_kinds.count(
                    "packedkv_q1_long_context_machine_code"
                ),
                1,
            )
            self.assertEqual(
                artifact_kinds.count(
                    "packedkv_q1_long_context_trace_contract"
                ),
                1,
            )
            self.assertEqual(
                artifact_kinds.count(
                    "packedkv_q1_long_context_input_recipe"
                ),
                1,
            )
            self.assertEqual(
                artifact_kinds.count(
                    "packedkv_q1_long_context_metrics"
                ),
                1,
            )
            binding_artifact = next(
                artifact
                for artifact in result["artifacts"]
                if artifact["kind"] == "compiler_precision_binding"
            )
            binding = json.loads(
                Path(binding_artifact["path"]).read_text(encoding="utf-8")
            )
            evidence_target = binding["evidence_target"]
            self.assertEqual(
                evidence_target["target_mode"],
                "rtl_compiler_deployment",
            )
            self.assertEqual(
                evidence_target["source_tree_sha256"],
                request["source_tree_sha256"],
            )
            self.assertEqual(
                evidence_target["mxint2_activation_scope"],
                "unsupported",
            )
            self.assertFalse(
                evidence_target["rtl_deployment_supports_mxint2_activation"]
            )
            self.assertFalse(evidence_target["common_deployment_valid"])
            runtime = binding["runtime_precision_contract"]
            self.assertEqual(
                runtime["matrix_semantics"]["schema_version"],
                MATRIX_SEMANTICS_SCHEMA,
            )
            self.assertEqual(
                runtime["matrix_semantics"]["profile_contract"],
                request["profile"]["matrix_semantics"],
            )
            self.assertTrue(
                runtime["matrix_semantics"]["structural_binding_valid"]
            )
            self.assertEqual(
                runtime["matrix_semantics"]["numerical_trace_conformance"][
                    "status"
                ],
                "not_run",
            )
            self.assertFalse(
                runtime["matrix_semantics"]["mixed_family"][
                    "deployment_supported"
                ]
            )
            self.assertEqual(
                runtime["matrix_semantics"]["mxint_pipeline"]["max_shift"],
                16,
            )
            self.assertEqual(
                runtime["matrix_semantics"]["mxint_pipeline"][
                    "matrix_conversion"
                ],
                "round_to_nearest_even_to_vector",
            )
            self.assertEqual(
                runtime["matrix_semantics"]["mxfp_pipeline"][
                    "product_conversion"
                ],
                "cast_each_product_to_m_fp",
            )
            self.assertEqual(
                runtime["matrix_semantics"]["instruction_reduction"],
                {
                    "physical_k_width": 1024,
                    "qk_logical_k_width": 128,
                    "linear_pv_logical_k_width": 1024,
                    "partial_conversion": "per_mm_ic_to_vector_storage_fp",
                    "cross_instruction_accumulation": (
                        "signed_fixed16_16_wraparound"
                    ),
                },
            )
            self.assertEqual(
                runtime["emulator_precision"]["HBM_M_WEIGHT_TYPE"]["ELEM"],
                {"type": "Int", "width": 4},
            )
            self.assertEqual(
                runtime["emulator_precision"]["HBM_M_KV_TYPE"]["ELEM"],
                {"type": "Int", "width": 4},
            )
            self.assertEqual(
                runtime["emulator_precision"]["HBM_V_ACT_TYPE"]["ELEM"],
                {"type": "Int", "width": 4},
            )
            self.assertEqual(
                runtime["emulator_precision"]["VECTOR_SRAM_TYPE"]["DATA_TYPE"],
                {
                    "type": "Fp",
                    "sign": True,
                    "exponent": 3,
                    "mantissa": 2,
                },
            )
            self.assertEqual(
                runtime["rtl_precision_parameters"]["M_FP_EXP_WIDTH"],
                3,
            )
            self.assertEqual(
                runtime["rtl_precision_parameters"]["M_FP_MANT_WIDTH"],
                2,
            )
            self.assertEqual(
                runtime["rtl_precision_parameters"]["V_FP_EXP_WIDTH"],
                3,
            )
            self.assertEqual(
                runtime["rtl_precision_parameters"]["V_FP_MANT_WIDTH"],
                2,
            )
            for test in result["tests"]:
                if not test["name"].startswith("packedkv_q1_batch_"):
                    continue
                metrics = test["metrics"]
                self.assertEqual(metrics["q_len"], 1)
                self.assertEqual(
                    metrics["cache_position"],
                    metrics["cache_tokens"] - 1,
                )
                self.assertEqual(metrics["block_size"], 8)
                self.assertTrue(metrics["selector_count_valid"])
                self.assertTrue(metrics["selector_sequence_valid"])
                self.assertTrue(metrics["batch_slab_mapping_valid"])
                self.assertTrue(metrics["sequence_block_count_valid"])
                self.assertTrue(metrics["physical_cache_slab_offsets_valid"])
                self.assertTrue(metrics["no_loop_clobbered_accumulator"])
                self.assertTrue(
                    metrics["matrix_compute_writeout_pairing_valid"]
                )
                self.assertTrue(metrics["no_dynamic_kv_group_loop"])
                self.assertTrue(metrics["machine_word_count_valid"])
                self.assertTrue(metrics["execution_opcode_coverage_valid"])
                self.assertTrue(metrics["execution_operand_coverage_valid"])
                self.assertTrue(metrics["execution_contract_valid"])
                self.assertEqual(
                    metrics["cross_target_operand_violations"],
                    [],
                )
                self.assertTrue(metrics["kv_prefetch_precision_valid"])
                self.assertTrue(metrics["role_precision_binding_valid"])
                self.assertEqual(
                    {
                        role: layout["precision_role"]
                        for role, layout in metrics["role_hbm_layouts"].items()
                    },
                    {
                        "weight": "weight",
                        "activation": "activation",
                        "key": "key",
                        "value": "value",
                    },
                )
                self.assertTrue(metrics["accumulator_binding_valid"])
                self.assertTrue(metrics["matrix_semantics_binding_valid"])
                self.assertTrue(metrics["structural_precision_binding_valid"])
                self.assertEqual(
                    metrics["numerical_trace_conformance"],
                    "not_run",
                )
                self.assertTrue(metrics["packedkv_selector_rtl_capable"])
                self.assertEqual(
                    set(metrics["decoded_h_prefetch_m_precision_funct1"]),
                    {1},
                )
                self.assertEqual(metrics["packed_byte_reduction"], 8.0)
                self.assertFalse(metrics["materialized_numerical_payloads"])
                self.assertFalse(metrics["emulator_numerical_validation"])
                self.assertFalse(
                    metrics["emulator_activation_matrix_port_validation"]
                )
                self.assertFalse(metrics["rtl_numerical_validation"])
                self.assertRegex(metrics["assembly_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(
                    metrics["machine_code_sha256"],
                    r"^[0-9a-f]{64}$",
                )
            long_context = next(
                test["metrics"]
                for test in result["tests"]
                if test["name"] == "packedkv_q1_long_context_scaled"
            )
            self.assertEqual(long_context["mlen"], 16)
            self.assertEqual(long_context["blen"], 4)
            self.assertEqual(long_context["hlen"], 8)
            self.assertEqual(long_context["batch_size"], 2)
            self.assertEqual(long_context["cache_tokens"], 33)
            self.assertEqual(long_context["cache_position"], 32)
            self.assertEqual(long_context["cache_rows_per_batch"], 48)
            self.assertEqual(
                long_context["sequence_blocks_per_selector"],
                3,
            )
            self.assertEqual(long_context["final_sequence_block_tokens"], 1)
            self.assertTrue(long_context["partial_final_sequence_block_valid"])
            self.assertTrue(long_context["compact_state_addresses_valid"])
            self.assertTrue(long_context["physical_cache_slab_offsets_valid"])
            self.assertTrue(long_context["no_loop_clobbered_accumulator"])
            self.assertTrue(
                long_context["matrix_compute_writeout_pairing_valid"]
            )
            self.assertEqual(
                long_context["pv_matrix_compute_count"],
                long_context["pv_matrix_writeout_count"],
            )
            self.assertEqual(
                long_context["qk_matrix_compute_count"],
                long_context["qk_matrix_writeout_count"],
            )
            self.assertTrue(long_context["no_dynamic_kv_group_loop"])
            self.assertEqual(
                long_context["compiler_evidence_scope"],
                "scaled_structural_multitile_lowering",
            )
            self.assertFalse(long_context["full_geometry_timing_evidence"])
            self.assertEqual(
                long_context["production_source_tree_sha256"],
                request["source_tree_sha256"],
            )
            contract_artifact = next(
                artifact
                for artifact in result["artifacts"]
                if artifact["kind"]
                == "packedkv_q1_long_context_trace_contract"
            )
            recipe_artifact = next(
                artifact
                for artifact in result["artifacts"]
                if artifact["kind"]
                == "packedkv_q1_long_context_input_recipe"
            )
            contract = json.loads(
                Path(contract_artifact["path"]).read_text(encoding="utf-8")
            )
            recipe = json.loads(
                Path(recipe_artifact["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                contract["schema_version"],
                LONG_CONTEXT_CONTRACT_SCHEMA,
            )
            self.assertEqual(
                contract["context_scope"]["schema_version"],
                CONTEXT_SCOPE_SCHEMA,
            )
            self.assertEqual(
                contract["context_scope"]["content_hash"],
                _content_hash(contract["context_scope"]),
            )
            self.assertEqual(
                contract["context_scope"],
                recipe["context_scope"],
            )
            self.assertEqual(
                contract["context_scope"][
                    "production_source_tree_sha256"
                ],
                request["source_tree_sha256"],
            )
            self.assertEqual(
                long_context["context_scope_sha256"],
                contract["context_scope"]["content_hash"],
            )
            self.assertEqual(
                long_context["trace_contract_sha256"],
                contract["content_hash"],
            )
            self.assertEqual(
                long_context["input_recipe_sha256"],
                recipe["content_hash"],
            )
            for artifact in result["artifacts"]:
                path = Path(artifact["path"]).resolve()
                path.relative_to(artifact_dir.resolve())
                self.assertGreater(path.stat().st_size, 0)

            original = result_path.read_bytes()
            rerun = run_hook(request_path, result_path, artifact_dir)
            self.assertEqual(rerun, result)
            self.assertEqual(result_path.read_bytes(), original)

    def test_unsupported_profile_returns_terminal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(_profile(weight="MXINT2"))
            result = run_hook(
                self._write_request(root, request),
                root / "result.json",
                root / "artifacts",
            )
            self.assertEqual(len(result["tests"]), 1)
            self.assertFalse(result["tests"][0]["passed"])
            self.assertEqual(
                result["tests"][0]["metrics"]["reason_code"],
                "unsupported_mxint_weight",
            )
            self.assertEqual(
                result["artifacts"][0]["kind"],
                "compiler_profile_rejection",
            )

    def test_mxint2_activation_returns_explicit_terminal_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(_profile(activation="MXINT2"))
            result = run_hook(
                self._write_request(root, request),
                root / "result.json",
                root / "artifacts",
            )

            self.assertEqual(len(result["tests"]), 1)
            self.assertFalse(result["tests"][0]["passed"])
            self.assertEqual(
                result["tests"][0]["metrics"]["reason_code"],
                "unsupported_mxint_activation",
            )
            self.assertEqual(
                result["artifacts"][0]["kind"],
                "compiler_profile_rejection",
            )

    def test_mxfp_structural_binding_does_not_claim_selector_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_hook(
                self._write_request(
                    root,
                    _request(_profile("E4M3", "E4M3", "E4M3")),
                ),
                root / "result.json",
                root / "artifacts",
            )
            self.assertTrue(all(test["passed"] for test in result["tests"]))
            traces = (
                test["metrics"]
                for test in result["tests"]
                if test["name"].startswith("packedkv_q1_batch_")
            )
            for metrics in traces:
                self.assertTrue(metrics["structural_precision_binding_valid"])
                self.assertFalse(metrics["packedkv_selector_rtl_capable"])
                self.assertEqual(
                    metrics["packedkv_selector_rtl_capability_reason"],
                    "selector_is_mxint_only",
                )
                self.assertFalse(metrics["rtl_numerical_validation"])

    def test_tampered_request_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = _request(_profile())
            request["manifest_hash"] = "5" * 64
            with self.assertRaisesRegex(ValueError, "content_hash"):
                run_hook(
                    self._write_request(root, request),
                    root / "result.json",
                    root / "artifacts",
                )

    def test_old_or_unbound_profile_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile()
            profile["schema_version"] = "decode-precision-profile-unbound"
            with self.assertRaisesRegex(ValueError, "unsupported profile schema"):
                run_hook(
                    self._write_request(root, _request(profile)),
                    root / "result.json",
                    root / "artifacts",
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile()
            profile.pop("matrix_semantics")
            with self.assertRaisesRegex(ValueError, "profile fields differ"):
                run_hook(
                    self._write_request(root, _request(profile)),
                    root / "result.json",
                    root / "artifacts",
                )

    def test_semantics_drift_is_rejected_before_lowering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = _profile()
            profile["matrix_semantics"]["mxint_max_shift"] = 15
            with self.assertRaisesRegex(ValueError, "matrix_semantics differs"):
                run_hook(
                    self._write_request(root, _request(profile)),
                    root / "result.json",
                    root / "artifacts",
                )


if __name__ == "__main__":
    unittest.main()
