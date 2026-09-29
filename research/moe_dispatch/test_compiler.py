"""Compiler contract tests: true routes, address coverage, bytes, capacity."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import compiler


class CompilerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = compiler.load_workloads()
        cls.manifest = json.loads((compiler.HERE / "fixtures/captured_routes.json").read_text())

    def test_real_routes_scores_and_dimensions_are_preserved(self):
        for w in self.data["workloads"]:
            self.assertEqual(w["hidden"], 2048)
            self.assertEqual(w["top_k"], 6)
            self.assertFalse(w["route_scores_renormalized"])
            seen = {}
            for e in w["experts"]:
                self.assertIsInstance(e["id"], int)
                self.assertEqual(e["Me"], len(e["token_indices"]))
                self.assertEqual(e["F"], 2816 if e["is_shared"] else 1408)
                for token, slot, score in zip(e["token_indices"], e["route_slots"], e["route_scores"]):
                    if e["is_shared"]:
                        self.assertEqual((slot, score), (-1, 1.0))
                    else:
                        self.assertEqual(self.manifest["routes"][token][slot], e["id"])
                        self.assertEqual(self.manifest["route_weights"][token][slot], score)
                        self.assertNotIn((token, slot), seen)
                        seen[token, slot] = e["id"]
            self.assertEqual(set(seen), {(t, s) for t in range(w["batch"]) for s in range(6)})
        b2 = self.data["workloads"][0]
        self.assertEqual({e["id"]: e["Me"] for e in b2["experts"]},
                         {25: 2, 26: 1, 43: 2, 52: 2, 55: 2, 56: 1, 57: 2, -1: 2})

    def test_bundled_checksums_and_historical_provenance(self):
        for line in (compiler.HERE / "fixtures/SHA256SUMS").read_text().splitlines():
            expected, name = line.split()
            self.assertEqual(compiler.sha256(compiler.HERE / "fixtures" / name), expected)
        for field in ("manifest", "model_config", "dataset", "input"):
            src = self.data["provenance"][field]
            self.assertTrue(src["path"].startswith("capture://"))
            self.assertEqual(len(src["sha256"]), 64)
        self.assertEqual(self.manifest["source_manifest_sha256"],
                         self.data["provenance"]["manifest"]["sha256"])
        self.assertFalse(self.data["provenance"]["complete_model_inference"])

    def test_capture_import_resolves_relative_sources_and_rejects_bad_hash(self):
        """Synthetic importer fixture; not additional real-model validation."""
        workload = self.data["workloads"][-1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.bf16").write_bytes(b"synthetic importer payload")
            (root / "dataset.json").write_text("[]")
            (root / "config.json").write_text(json.dumps({
                "n_routed_experts": 64, "hidden_size": 2048,
            }))
            manifest = {
                **self.manifest, "model": "synthetic-import-test", "model_layer": 1,
                "phase": "test", "prefix": "synthetic", "source_dataset": "dataset.json",
                "examples": [{"sample_id": token["sample_id"]} for token in workload["tokens"]],
                "x": {"path": "input.bf16", "sha256": compiler.sha256(root / "input.bf16")},
                "tensor_provenance": {
                    weight["tensor_name"]: {
                        "shape": weight["shape_nk"], "dtype": weight["dtype"],
                        "sha256": weight["source_sha256"], "shard": weight["source_shard"],
                    }
                    for expert in workload["experts"] for weight in expert["weights"].values()
                },
            }
            capture = root / "manifest.json"
            capture.write_text(json.dumps(manifest))
            imported = compiler.extract_workloads(capture, root / "config.json")
            for field in ("manifest", "model_config", "dataset", "input"):
                source = imported["provenance"][field]
                self.assertEqual(compiler.sha256(Path(source["path"])), source["sha256"])
            self.assertEqual([e["Me"] for e in imported["workloads"][-1]["experts"]],
                             [e["Me"] for e in workload["experts"]])
            (root / "input.bf16").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                compiler.extract_workloads(capture, root / "config.json")

    def test_hbm_tensor_slots_do_not_alias_between_phases_or_experts(self):
        ranges = []
        for e in self.data["workloads"][-1]["experts"]:
            for phase, w in e["weights"].items():
                ranges.append((w["hbm_base"], w["hbm_base"] + w["physical_bytes"], e["id"], phase))
        ranges.sort()
        for a, b in zip(ranges, ranges[1:]):
            self.assertLessEqual(a[1], b[0])
        self.assertEqual(len({a[0] for a in ranges}), len(ranges))

    def test_column_partitions_cover_exactly_and_are_tile_aligned(self):
        for n in (1408, 2048, 2816, 17):
            for shape in compiler.ORGANIZATIONS:
                parts = compiler.column_ranges(n, shape)
                covered = []
                for lo, hi in parts:
                    self.assertEqual(lo % 4, 0)
                    covered.extend(range(lo, hi))
                self.assertEqual(covered, list(range(n)))

    def test_completed_outputs_input_and_control_fit_exact_aggregate_budget(self):
        for w in self.data["workloads"]:
            for shape in compiler.ORGANIZATIONS:
                p = compiler.compile_workload(w, shape)
                budget = p["budget"]
                self.assertEqual(sum(budget["private_accumulator_bytes"]), 2 * 1024**2)
                self.assertEqual(sum(budget["control_bytes_per_core"]), 4096)
                self.assertEqual(sum(budget["x_bytes_per_core"]), 12 * 1024)
                self.assertEqual(budget["private_weight_slots_per_core"] * len(shape) * 4096
                                 + budget["weight_ingress_bytes"], 48 * 1024)
                rows = sum(e["Me"] for e in w["experts"])
                expected = 4 * w["hidden"] * (rows + w["batch"]) + 2 * w["batch"] * w["hidden"]
                self.assertEqual(sum(r["reserve_bytes"] for r in p["result_layout"]), expected)
                self.assertEqual(p["controller"]["total_state_bytes"]
                                 + p["controller"]["reserve_headroom_bytes"], 4096)
                for region in p["result_layout"]:
                    objects = [region["original_x"], *region["inboxes"], region["combined_output"]]
                    for first, second in zip(objects, objects[1:]):
                        self.assertEqual(first["address_bytes"] + first["bytes"], second["address_bytes"])

    def test_whole_and_split_do_identical_useful_math_and_weight_reads(self):
        for w in self.data["workloads"]:
            for shape in compiler.ORGANIZATIONS:
                for mode in ("whole", "paired_n"):
                    p = compiler.compile_workload(w, shape, mode)
                    for e, planned in zip(w["experts"], p["experts"]):
                        expected_macs = 3 * e["Me"] * e["H"] * e["F"]
                        expected_weights = sum(v["physical_bytes"] for v in e["weights"].values())
                        for candidate in planned["candidates"]:
                            self.assertEqual(candidate["total_useful_macs"], expected_macs)
                            self.assertEqual(candidate["total_weight_bytes"], expected_weights)
                            columns = []
                            for c in candidate["cores"]:
                                columns.extend(range(*c["h_columns"]))
                                self.assertEqual(sum(v["payload_bytes"] for v in c["input_copies"]), 2 * e["Me"] * e["H"])
                                for cp in c["input_copies"] + c["retire_copies"]:
                                    self.assertEqual(cp["payload_bytes"], cp["source_read_bytes"])
                                    self.assertEqual(cp["payload_bytes"], cp["destination_write_bytes"])
                                    self.assertEqual(cp["payload_bytes"], cp["shared_bus_payload_bytes"])
                            self.assertEqual(sorted(columns), list(range(e["H"])))
                            copies = [v for c in candidate["cores"] for v in c["retire_copies"]]
                            self.assertEqual(sum(v["payload_bytes"] for v in copies), 4 * e["Me"] * e["H"])
                            expected_z = 2 * e["Me"] * e["F"] if mode == "paired_n" and len(shape) == 2 else 0
                            self.assertEqual(candidate["z_cross_core_bytes"], expected_z)

    def test_workspace_ranges_do_not_overlap_and_are_checked_per_core(self):
        for w in self.data["workloads"]:
            for shape in compiler.ORGANIZATIONS:
                for mode in ("whole", "paired_n"):
                    p = compiler.compile_workload(w, shape, mode)
                    for e in p["experts"]:
                        for cand in e["candidates"]:
                            for c in cand["cores"]:
                                a = c["storage"]
                                alloc = a["allocations"]
                                self.assertGreaterEqual(alloc[0]["offset_bytes"], a["control_reserve_bytes"] + a["result_reserve_bytes"])
                                for first, second in zip(alloc, alloc[1:]):
                                    self.assertLessEqual(first["offset_bytes"] + first["bytes"], second["offset_bytes"])
                                self.assertEqual(a["feasible"], a["peak_private_bytes"] <= a["budget_bytes"])
        w = copy.deepcopy(self.data["workloads"][-1])
        w["experts"][0]["Me"] = 1000  # Deliberately invalid capacity fixture, not a reported workload.
        p = compiler.compile_workload(w, [4, 2])
        self.assertTrue(all(not c["feasible"] for c in p["experts"][0]["candidates"]))

    def test_gate_z_alias_is_explicit_and_preserves_full_down_input(self):
        w = self.data["workloads"][-1]
        for mode in ("whole", "paired_n"):
            p = compiler.compile_workload(w, [4, 2], mode)
            for e in p["experts"]:
                for candidate in e["candidates"]:
                    for c in candidate["cores"]:
                        arena = c["storage"]
                        physical = next(a for a in arena["allocations"] if a["name"] == "gate_z")
                        alias = arena["aliases"][0]
                        self.assertEqual(physical["bytes"], 2 * e["Me"] * e["F"])
                        self.assertEqual(alias["z_view"]["shape"], [e["Me"], e["F"]])
                        self.assertEqual(alias["gate_view"]["address_bytes"], physical["offset_bytes"] + 2 * c["f_columns"][0])
                        self.assertTrue(alias["vector_reads_and_writes_are_charged"])
        whole = compiler.compile_workload(w, [4, 2], "whole")
        shared = next(e for e in whole["experts"] if e["is_shared"])
        self.assertTrue(shared["candidates"][0]["feasible"])
        self.assertFalse(shared["candidates"][1]["feasible"])
        routed_m16 = next(e for e in whole["experts"] if e["expert_id"] == 25)
        self.assertTrue(all(c["feasible"] for c in routed_m16["candidates"]))

    def test_engine_addresses_match_plans_and_charge_route_state(self):
        w = self.data["workloads"][-1]
        layout = compiler.engine_layout(w, [4, 2])
        for c, core in enumerate(layout["cores"]):
            r = core["result_layout"]
            self.assertEqual(r["route_state"]["bytes"], 16 * 6 * 16 + len(w["experts"]) * 64)
            self.assertGreaterEqual(r["arena_base_bytes"], core["control"] + r["route_state"]["bytes"])
            for ei, e in enumerate(w["experts"]):
                for mode in ("whole", "paired_n"):
                    local = layout[mode][ei][c]
                    a = local["allocations"]
                    self.assertEqual(a["x"]["base"], core["reserved"])
                    self.assertEqual(a["z"]["base"], a["gate_z"]["base"])
                    self.assertEqual(a["z"]["stride"], 2 * e["F"])
                    self.assertEqual(a["gate"]["base"], a["z"]["base"] + 2 * local["f_columns"][0])
                    self.assertEqual(a["y"]["base"], a["producer_y"]["base"])
                    self.assertEqual(a["scratch0"]["stride"], 32)
                    self.assertEqual(a["scratch1"]["base"], a["scratch0"]["base"] + a["scratch0"]["bytes"])
                    self.assertEqual(local["peak"], core["reserved"] + local["workspace_bytes"])

    def test_invalid_layout_requests_fail_before_running(self):
        w = self.data["workloads"][0]
        with self.assertRaises(ValueError):
            compiler.compile_workload(w, [4, 2], resident_bands=5)
        with self.assertRaises(ValueError):
            compiler.compile_workload(w, [4, 4])
        with self.assertRaises(ValueError):
            compiler.column_ranges(4, [4, 2])

    def test_current_next_return_tags_fit_existing_control_reserve(self):
        w = self.data["workloads"][-1]
        for lanes, expected in (([6], 2752), ([3, 3], 3744), ([4, 2], 3744)):
            plan = compiler.compile_workload(w, lanes)
            ctrl = plan["controller"]
            self.assertEqual(ctrl["total_state_bytes"], expected)
            self.assertEqual(ctrl["dma_credit_capacity"] * ctrl["dma_credit_unit_bytes"], 8192)
            self.assertEqual(ctrl["next_prefetch_tiles_per_core"], 1)
            self.assertEqual(ctrl["next_experts_per_core"], 1)
            self.assertTrue(all(n >= 0 for n in ctrl["headroom_bytes_per_core"]))
            self.assertEqual(sum(ctrl["state_bytes_per_core"]), expected)
            self.assertEqual(plan["runtime_protocol"]["current_next_limit_per_core"], [1, 1])


if __name__ == "__main__":
    unittest.main()
