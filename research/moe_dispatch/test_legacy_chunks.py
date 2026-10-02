"""Capacity, conservation, and physical-alias tests for legacy token chunks."""
import copy
import hashlib
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

import compiler
import legacy_chunks


def repeated_capture(batch=96):
    """Synthetic repeated captured routes; not a new numerical measurement."""
    source = compiler.load_workloads()["workloads"][-1]
    if batch % source["batch"]:
        raise ValueError("test population must repeat complete capture batches")
    workload = copy.deepcopy(source)
    workload["id"] = "synthetic_repeated_capture_%d" % batch
    workload["batch"] = batch
    workload["tokens"] = []
    repeats = batch // source["batch"]
    for repetition in range(repeats):
        for original in source["tokens"]:
            token = copy.deepcopy(original)
            token["token_index"] += repetition * source["batch"]
            token["sample_id"] = "%s_repeat%d" % (token["sample_id"], repetition)
            workload["tokens"].append(token)
    for expert in workload["experts"]:
        original = next(e for e in source["experts"] if e["id"] == expert["id"])
        expert["token_indices"] = [t + r * source["batch"] for r in range(repeats)
                                   for t in original["token_indices"]]
        expert["route_slots"] = original["route_slots"] * repeats
        expert["route_scores"] = original["route_scores"] * repeats
        expert["Me"] = len(expert["token_indices"])
        expert["dag"] = compiler.expert_dag(expert)
    workload["input"]["shape"] = [batch, workload["hidden"]]
    return workload


class LegacyTokenChunksTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workload = repeated_capture()
        cls.prepared = legacy_chunks.prepare_legacy_workload(cls.workload, [4, 2])
        cls.wrapper = cls.prepared["legacy_batch_execution"]

    def test_supported_workloads_are_exact_original_layouts_and_deep_copies(self):
        for workload in compiler.load_workloads()["workloads"]:
            for lanes in compiler.ORGANIZATIONS:
                expected = {**copy.deepcopy(workload), "engine_layout": compiler.engine_layout(workload, lanes)}
                actual = legacy_chunks.prepare_legacy_workload(workload, lanes)
                self.assertEqual(actual, expected)
                self.assertNotIn("legacy_batch_execution", actual)
                supplied = legacy_chunks.prepare_legacy_workload(expected, lanes)
                self.assertEqual(supplied, expected)
                self.assertIsNot(supplied["engine_layout"], expected["engine_layout"])
                actual["experts"][0]["weights"]["gate"]["source_sha256"] = "changed copy"
                self.assertNotEqual(actual["experts"][0]["weights"], workload["experts"][0]["weights"])

    def test_sibling_compiler_import_survives_cached_simulator_name_collision(self):
        spec = importlib.util.spec_from_file_location("isolated_legacy_chunks", compiler.HERE / "legacy_chunks.py")
        isolated = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"compiler": types.ModuleType("simulator_compiler_without_engine_layout")}):
            spec.loader.exec_module(isolated)
            workload = compiler.load_workloads()["workloads"][0]
            actual = isolated.prepare_legacy_workload(workload, [4, 2])
        self.assertEqual(isolated.compiler.__file__, str(compiler.HERE / "compiler.py"))
        self.assertEqual(actual["engine_layout"], compiler.engine_layout(workload, [4, 2]))

    def test_unsupported_original_is_preserved_and_no_source_is_edited(self):
        original = compiler.engine_layout(self.workload, [4, 2])
        self.assertFalse(legacy_chunks._all_whole_fit(original))
        self.assertEqual(self.prepared["engine_layout"], original)
        self.assertEqual({k: v for k, v in self.prepared.items()
                          if k not in ("engine_layout", "legacy_batch_execution")}, self.workload)
        self.assertEqual(self.wrapper["schema"], legacy_chunks.SCHEMA)
        self.assertGreater(len(self.wrapper["chunks"]), 1)
        self.assertEqual(hashlib.sha256((compiler.HERE / "compiler.py").read_bytes()).hexdigest(),
                         "34faf30584318ca4eff0aeed0779ecf37637777934d804099821c335412b0a77")

    def test_all_tokens_routes_scores_weights_and_expert_order_are_conserved(self):
        reconstructed = {e["id"]: [] for e in self.workload["experts"]}
        original_by_id = {e["id"]: e for e in self.workload["experts"]}
        tokens = []
        for chunk in self.wrapper["chunks"]:
            start, end = chunk["token_range"]
            child = chunk["workload"]
            self.assertEqual(child["batch"], end - start)
            self.assertEqual(child["original_token_indices"], list(range(start, end)))
            wanted_ids = [e["id"] for e in self.workload["experts"]
                          if any(start <= t < end for t in e["token_indices"])]
            self.assertEqual([e["id"] for e in child["experts"]], wanted_ids)
            for token in child["tokens"]:
                self.assertEqual(token["original_token_index"], start + token["token_index"])
                tokens.append(token["original_token_index"])
                self.assertEqual(token["routes"], self.workload["tokens"][token["original_token_index"]]["routes"])
            for expert in child["experts"]:
                self.assertEqual(expert["weights"], original_by_id[expert["id"]]["weights"])
                self.assertEqual(expert["original_token_indices"], [start + t for t in expert["token_indices"]])
                reconstructed[expert["id"]].extend(zip(expert["original_token_indices"],
                                                       expert["route_slots"], expert["route_scores"]))
                if expert["is_shared"]:
                    self.assertEqual(expert["token_indices"], list(range(end - start)))
        self.assertEqual(tokens, list(range(self.workload["batch"])))
        for expert in self.workload["experts"]:
            self.assertEqual(reconstructed[expert["id"]], list(zip(expert["token_indices"],
                                                                  expert["route_slots"], expert["route_scores"])))
        actual_macs = sum(3 * e["Me"] * e["H"] * e["F"]
                          for c in self.wrapper["chunks"] for e in c["workload"]["experts"])
        self.assertEqual(actual_macs, self.wrapper["expected_useful_macs"])
        self.assertEqual(self.wrapper["unique_weight_bytes"], legacy_chunks._weight_bytes(self.workload))
        self.assertEqual(self.wrapper["weight_read_bytes"], sum(legacy_chunks._weight_bytes(c["workload"])
                                                               for c in self.wrapper["chunks"]))
        self.assertGreater(self.wrapper["weight_read_bytes"], self.wrapper["unique_weight_bytes"])

    def test_global_row_aliases_and_all_workspace_views_fit_exact_physical_bounds(self):
        original = self.prepared["engine_layout"]
        for parent in self.wrapper["persistent_cores"]:
            core = original["cores"][parent["core"]]
            self.assertEqual(parent["capacity"], core["capacity"])
            self.assertEqual(parent["route_state"], core["result_layout"]["route_state"])
            self.assertEqual(parent["original_x"]["shape"][0], self.workload["batch"])
            self.assertEqual(parent["combined_output"]["shape"][0], self.workload["batch"])
            self.assertEqual(parent["combined_output"]["address_bytes"],
                             parent["original_x"]["address_bytes"] + parent["original_x"]["bytes"])
            self.assertLessEqual(parent["reserved"], parent["capacity"])
        for chunk in self.wrapper["chunks"]:
            child = chunk["workload"]
            start, end = chunk["token_range"]
            layout = child["engine_layout"]
            self.assertTrue(legacy_chunks._all_whole_fit(layout))
            prior = compiler.engine_layout(child, [4, 2])
            for core, parent in zip(layout["cores"], self.wrapper["persistent_cores"]):
                result = core["result_layout"]
                for name in ("original_x", "combined_output"):
                    alias = result[name]
                    self.assertEqual(alias["address_bytes"], parent[name]["address_bytes"] + start * alias["row_stride_bytes"])
                    self.assertEqual(alias["bytes"], (end - start) * alias["row_stride_bytes"])
                    self.assertLessEqual(alias["address_bytes"] + alias["bytes"],
                                         parent[name]["address_bytes"] + parent[name]["bytes"])
                at = parent["reserved"]
                for inbox in result["inboxes"]:
                    self.assertEqual(inbox["address_bytes"], at)
                    self.assertEqual(inbox["offset_bytes"], at - parent["arena_base_bytes"])
                    at += inbox["bytes"]
                self.assertEqual(core["reserved"], compiler.align(at))
                delta = core["reserved"] - prior["cores"][core["core"]]["reserved"]
                for mode in ("whole", "paired_n"):
                    for expert_index, expert in enumerate(layout[mode]):
                        local = expert[core["core"]]
                        old = prior[mode][expert_index][core["core"]]
                        self.assertEqual(local["peak"], core["reserved"] + local["workspace_bytes"])
                        self.assertEqual(local["feasible"], local["peak"] <= core["capacity"])
                        for name, allocation in local["allocations"].items():
                            self.assertEqual(allocation["base"], old["allocations"][name]["base"] + delta)
                        alloc = local["allocations"]
                        physical = [alloc[name] for name in ("x", "gate_z", "up", "producer_y", "scratch0", "scratch1")]
                        self.assertEqual(physical[0]["base"], core["reserved"])
                        for first, second in zip(physical, physical[1:]):
                            self.assertLessEqual(first["base"] + first["bytes"], second["base"])
                        self.assertLessEqual(physical[-1]["base"] + physical[-1]["bytes"], local["peak"])
                        self.assertEqual(alloc["z"]["base"], alloc["gate_z"]["base"])
                        self.assertEqual(alloc["gate"]["base"], alloc["gate_z"]["base"] + 2 * local["f_columns"][0])
                        self.assertEqual(alloc["y"]["base"], alloc["producer_y"]["base"])
                self.assertEqual(result["route_state"]["bytes"], parent["route_state"]["bytes"])
            self.assertEqual(layout["runtime_protocol"]["legacy_token_chunks"]["wrapper_payload_copy_bytes"], 0)

    def test_largest_chunk_checks_every_actual_route_distribution(self):
        size = self.wrapper["chunk_size"]
        def fits(candidate_size):
            # Independent capacity check from the original expert rows and the
            # unchanged compiler's workspace sizes, rather than wrapper flags.
            for start in range(0, self.workload["batch"], candidate_size):
                end = min(start + candidate_size, self.workload["batch"])
                child = copy.deepcopy(self.workload)
                child["batch"] = end - start
                child["experts"] = []
                for expert in self.workload["experts"]:
                    rows = [t for t in expert["token_indices"] if start <= t < end]
                    if rows:
                        local = copy.deepcopy(expert)
                        local["Me"] = len(rows)
                        child["experts"].append(local)
                plan = compiler.engine_layout(child, [4, 2])
                inbox_rows = sum(e["Me"] for e in child["experts"])
                reserved = [compiler.align(p["reserved"] + 4 * inbox_rows * (p["columns"][1] - p["columns"][0]))
                            for p in self.wrapper["persistent_cores"]]
                if any(not any(reserved[c] + local["workspace_bytes"] <= plan["cores"][c]["capacity"]
                               for c, local in enumerate(expert)) for expert in plan["whole"]):
                    return False
            return True
        self.assertTrue(fits(size))
        for larger_size in range(size + 1, self.workload["batch"] + 1):
            self.assertFalse(fits(larger_size), "larger candidate %d fits" % larger_size)

    def test_control_record_uses_existing_headroom_with_and_without_joint_state(self):
        for joint_state in (0, 256):
            resources = compiler.hardware_budget([4, 2])
            resources["joint_state_bytes"] = joint_state
            resources["control_bytes"] = compiler.partition_bytes(compiler.CONTROL_BYTES + joint_state, [4, 2])
            prepared = legacy_chunks.prepare_legacy_workload(self.workload, [4, 2], resources=resources)
            outer = prepared["legacy_batch_execution"]
            record = outer["control_record"]
            original = prepared["engine_layout"]["control_accounting"]
            c = record["core"]
            self.assertEqual(record["bytes"], 32)
            self.assertEqual(record["base"], original["state_bytes_per_core"][c])
            self.assertLessEqual(record["base"] + record["bytes"], prepared["engine_layout"]["cores"][c]["control"])
            for chunk in outer["chunks"]:
                accounting = chunk["workload"]["engine_layout"]["control_accounting"]
                self.assertEqual(accounting["total_state_bytes"], original["total_state_bytes"] + 32)
                self.assertEqual(accounting["headroom_bytes_per_core"][c], original["headroom_bytes_per_core"][c] - 32)
                self.assertEqual(accounting["added_state"]["legacy_token_chunk_control"], 32)

    def test_outer_persistent_arena_must_fit_even_on_inactive_cores(self):
        workload = repeated_capture(192)
        with self.assertRaisesRegex(ValueError, "full-layer persistent arena exceeds"):
            legacy_chunks.prepare_legacy_workload(workload, [4, 2])
        layout = copy.deepcopy(self.prepared["engine_layout"])
        # No expert can use core 1; its persistent input/output still cannot be ignored.
        for expert in layout["whole"]:
            expert[1]["feasible"] = False
        layout["cores"][1]["capacity"] = 64
        with self.assertRaisesRegex(ValueError, "core 1 capacity"):
            legacy_chunks._persistent_cores(self.workload, layout)

    def test_bad_route_metadata_and_no_wrapper_control_headroom_are_actionable(self):
        workload = copy.deepcopy(self.workload)
        workload["experts"][0]["route_scores"][0] += 0.25
        with self.assertRaisesRegex(ValueError, "does not conserve"):
            legacy_chunks.prepare_legacy_workload(workload, [4, 2])
        layout = copy.deepcopy(self.prepared["engine_layout"])
        layout["control_accounting"]["headroom_bytes_per_core"] = [0, 0]
        with self.assertRaisesRegex(ValueError, "controller headroom"):
            legacy_chunks._control_record(layout)


if __name__ == "__main__":
    unittest.main()
