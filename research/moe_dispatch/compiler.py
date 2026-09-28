#!/usr/bin/env python3
"""Compile real captured MoE routes into finite private-memory execution plans.

This independent candidate front-end does not emit the main PLENA ISA and does
not predict latency. All addresses are byte addresses in a declared proposed
layout; they do not claim to reproduce the older phase-isolated HBM mapping.
Only the captured routes and original model tensor dimensions are inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_WORKLOADS = HERE / "fixtures/workloads.json"
SCHEMA = "plena_moe_dispatch_workloads_v1"
PLAN_SCHEMA = "plena_moe_dispatch_plan_v1"
N_TILE, K_TILE = 4, 512
ACC_BYTES, CONTROL_BYTES = 2 * 1024**2, 4096
WEIGHT_BYTES, INGRESS_BYTES, X_BYTES = 48 * 1024, 8 * 1024, 12 * 1024
EXPERT_PHASE_STRIDE = 16 * 1024**2
ORGANIZATIONS = ((6,), (3, 3), (4, 2))


def align(n: int, granule: int = 32) -> int:
    return (n + granule - 1) // granule * granule


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def column_ranges(n: int, m_lanes: list[int] | tuple[int, ...]) -> list[list[int]]:
    """Partition whole N tiles by cumulative resource ratio, without overlap."""
    if n <= 0 or not m_lanes or min(m_lanes) <= 0:
        raise ValueError("positive N and physical M lanes required")
    tiles = (n + N_TILE - 1) // N_TILE
    if tiles < len(m_lanes):
        raise ValueError("not enough N tiles for the requested paired split")
    total = sum(m_lanes)
    counts, used, previous = [], 0, 0
    for c, m in enumerate(m_lanes):
        used += m
        end = n if c == len(m_lanes) - 1 else min(n, (tiles * used // total) * N_TILE)
        if end <= previous:
            raise ValueError("paired split produced an empty core partition")
        counts.append([previous, end])
        previous = end
    return counts


def partition_bytes(total: int, m_lanes: list[int], granule: int = 32) -> list[int]:
    """Exact aggregate capacity with independently aligned private boundaries."""
    if total % granule:
        raise ValueError("budget must be divisible by its allocation granule")
    previous, used, sizes = 0, 0, []
    for i, m in enumerate(m_lanes):
        used += m
        end = total if i == len(m_lanes) - 1 else (total // granule * used // sum(m_lanes)) * granule
        sizes.append(end - previous)
        previous = end
    return sizes


def _check_routes(manifest: dict[str, Any]) -> None:
    routes, weights = manifest["routes"], manifest["route_weights"]
    if not routes or len(routes) != len(weights) or len(routes) != len(manifest["examples"]):
        raise ValueError("capture route/score/sample lengths disagree")
    top_k = len(routes[0])
    for row, score in zip(routes, weights):
        if len(row) != top_k or len(score) != top_k or len(set(row)) != top_k:
            raise ValueError("inconsistent or duplicate expert selection")
        if any(not isinstance(e, int) or e < 0 for e in row):
            raise ValueError("invalid captured expert ID")
        if any(not math.isfinite(s) or s < 0 for s in score):
            raise ValueError("invalid route score")


def _expert_weights(manifest: dict[str, Any], expert_id: int, expert_slots: int) -> dict[str, Any]:
    shared = expert_id == -1
    prefix = "model.layers.%s.mlp.%s" % (
        manifest["model_layer"], "shared_experts" if shared else "experts.%d" % expert_id
    )
    slot = expert_slots if shared else expert_id
    result = {}
    for phase_index, phase in enumerate(("gate", "up", "down")):
        tensor_name = prefix + "." + phase + "_proj.weight"
        v = manifest["tensor_provenance"][tensor_name]
        n, k = v["shape"]
        if v["dtype"] != "BF16" or n <= 0 or k <= 0:
            raise ValueError("unsupported captured weight representation")
        row_stride = align(k * 2)
        if n * row_stride > EXPERT_PHASE_STRIDE:
            raise ValueError("weight does not fit fixed expert/phase address slot")
        result[phase] = {
            "tensor_name": tensor_name, "shape_nk": [n, k], "dtype": "BF16",
            "source_sha256": v["sha256"], "source_shard": v["shard"],
            "hbm_base": (slot * 3 + phase_index) * EXPERT_PHASE_STRIDE,
            "row_stride_bytes": row_stride, "payload_bytes": n * k * 2,
            "physical_bytes": n * row_stride, "source_hash_verified_by": "captured tensor manifest",
        }
    gate, up, down = [result[p]["shape_nk"] for p in ("gate", "up", "down")]
    if gate != up or down != [gate[1], gate[0]]:
        raise ValueError("gate/up/down dimensions do not form a SwiGLU expert")
    return result


def expert_dag(expert: dict[str, Any]) -> dict[str, Any]:
    """Complete dependency graph; local scheduling may overlap independent nodes."""
    m, h, f = expert["Me"], expert["H"], expert["F"]
    nodes = [
        {"id": "input", "op": "gather_input", "output_shape": [m, h], "dtype": "BF16"},
        {"id": "gate", "op": "gemm", "M": m, "N": f, "K": h,
         "weight": "gate", "macs": m * h * f, "depends_on": ["input"], "round_output": "BF16"},
        {"id": "up", "op": "gemm", "M": m, "N": f, "K": h,
         "weight": "up", "macs": m * h * f, "depends_on": ["input"], "round_output": "BF16"},
        {"id": "silu", "op": "silu", "elements": m * f, "depends_on": ["gate"], "dtype": "BF16"},
        {"id": "z", "op": "multiply", "elements": m * f,
         "depends_on": ["silu", "up"], "dtype": "BF16", "shape": [m, f]},
        {"id": "down", "op": "gemm", "M": m, "N": h, "K": f,
         "weight": "down", "macs": m * f * h, "depends_on": ["z"], "accumulator": "FP32"},
        {"id": "retire", "op": "copy_to_result_inboxes", "depends_on": ["down"],
         "payload_bytes": 4 * m * h, "round_before_route_combine": "BF16"},
    ]
    return {"nodes": nodes, "edges": [[dep, n["id"]] for n in nodes for dep in n.get("depends_on", [])]}


def load_workloads(path: Path = DEFAULT_WORKLOADS) -> dict[str, Any]:
    """Load portable captured metadata, not the original numerical payloads."""
    path = Path(path)
    if path.resolve() == DEFAULT_WORKLOADS.resolve():
        expected = dict(line.split()[::-1] for line in
                        (HERE / "fixtures/SHA256SUMS").read_text().splitlines())
        if sha256(path) != expected[path.name]:
            raise ValueError("bundled workload checksum mismatch")
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA or not data.get("workloads"):
        raise ValueError("expected nonempty dispatch v1 workload bundle")
    return data


def extract_workloads(manifest_path: Path, config_path: Path,
                      batches: tuple[int, ...] = (2, 4, 8, 16)) -> dict[str, Any]:
    """Import an original capture and verify its locally supplied input payload.

    Relative input/dataset paths resolve beside the manifest. Model weights
    are described by the capture's hashes; they are not read by this importer.
    """
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    _check_routes(manifest)
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    expert_slots = config["n_routed_experts"]
    if expert_slots <= max(max(row) for row in manifest["routes"]):
        raise ValueError("capture expert ID exceeds model configuration")
    input_path = Path(manifest["x"]["path"])
    if not input_path.is_absolute():
        input_path = manifest_path.parent / input_path
    dataset_path = Path(manifest["source_dataset"])
    if not dataset_path.is_absolute():
        dataset_path = manifest_path.parent / dataset_path
    if sha256(input_path) != manifest["x"]["sha256"]:
        raise ValueError("captured X payload hash mismatch")
    workloads = []
    for batch in batches:
        if batch <= 0 or batch > len(manifest["routes"]):
            raise ValueError("batch exceeds captured token population")
        tokens = []
        positions: dict[int, list[tuple[int, int, float]]] = {}
        for token, (ids, scores) in enumerate(zip(manifest["routes"][:batch], manifest["route_weights"][:batch])):
            routes = []
            for rank, (expert_id, score) in enumerate(zip(ids, scores)):
                positions.setdefault(expert_id, []).append((token, rank, score))
                routes.append({"expert_id": expert_id, "slot": rank, "score": score})
            tokens.append({"token_index": token, "sample_id": manifest["examples"][token]["sample_id"],
                           "routes": routes})
        experts = []
        for expert_id in sorted(positions) + [-1]:
            weights = _expert_weights(manifest, expert_id, expert_slots)
            f, h = weights["gate"]["shape_nk"]
            p = positions[expert_id] if expert_id != -1 else [(t, -1, 1.0) for t in range(batch)]
            e = {"id": expert_id, "is_shared": expert_id == -1, "Me": len(p), "H": h, "F": f,
                 "token_indices": [v[0] for v in p], "route_slots": [v[1] for v in p],
                 "route_scores": [v[2] for v in p], "weights": weights}
            e["dag"] = expert_dag(e)
            experts.append(e)
        hidden = experts[0]["H"]
        if hidden != config["hidden_size"] or any(e["H"] != hidden for e in experts):
            raise ValueError("hidden dimension disagrees with captured weight/config")
        workloads.append({"id": "real_b%d" % batch, "batch": batch, "hidden": hidden,
                          "top_k": len(tokens[0]["routes"]), "tokens": tokens, "experts": experts,
                          "input": {**manifest["x"], "shape": [batch, hidden], "dtype": "BF16"},
                          "scope": "real first-MoE-layer last-token prefill capture; not decode rollout",
                          "routing_is_input_not_timed": True, "route_scores_renormalized": False})
    return {
        "schema": SCHEMA,
        "provenance": {
            "manifest": _source(manifest_path), "model_config": _source(config_path),
            "dataset": _source(dataset_path), "input": _source(input_path),
            "model": manifest["model"], "layer": manifest["model_layer"], "capture_phase": manifest["phase"],
            "capture_method": manifest["prefix"], "captured_tokens": len(manifest["routes"]),
            "complete_model_inference": False, "weights_payloads_loaded_by_this_compiler": False,
        },
        "address_contract": {"units": "bytes", "weight_layout": "W[N,K], BF16, 32B-aligned rows",
                             "hbm_layout": "proposed fixed expert-slot/phase slots; not old isolated-phase addresses",
                             "expert_phase_slot_bytes": EXPERT_PHASE_STRIDE,
                             "shared_expert_logical_id": -1, "shared_hbm_slot": expert_slots},
        "workloads": workloads,
    }


def _result_layout(workload: dict[str, Any], m_lanes: list[int]) -> list[dict[str, Any]]:
    """Reserve every eventual output before execution; completed experts can drain."""
    result = []
    for core, (lo, hi) in enumerate(column_ranges(workload["hidden"], m_lanes)):
        control = partition_bytes(CONTROL_BYTES, m_lanes)[core]
        route_bytes = workload["batch"] * workload["top_k"] * 16 + len(workload["experts"]) * 64
        route_state = {"address_bytes": control, "bytes": route_bytes,
                       "route_entry_bytes": 16, "expert_entry_bytes": 64,
                       "copied_per_core": True, "initial_state": "already supplied by upstream router"}
        arena_base = align(control + route_bytes)
        original_x = {"offset_bytes": 0, "bytes": 2 * workload["batch"] * (hi - lo),
                      "address_bytes": arena_base,
                      "shape": [workload["batch"], hi - lo], "dtype": "BF16",
                      "row_stride_bytes": 2 * (hi - lo),
                      "lifetime": "layer entry through last expert input gather",
                      "initial_state": "already produced upstream; upstream and router latency excluded"}
        at, inboxes = original_x["bytes"], []
        for e in workload["experts"]:
            size = 4 * e["Me"] * (hi - lo)
            inboxes.append({"expert_id": e["id"], "offset_bytes": at, "bytes": size,
                            "address_bytes": arena_base + at,
                            "shape": [e["Me"], hi - lo], "row_stride_bytes": 4 * (hi - lo),
                            "dtype": "FP32", "lifetime": "expert retirement to ordered combine"})
            at += size
        combined = {"offset_bytes": at, "address_bytes": arena_base + at,
                    "bytes": 4 * workload["batch"] * (hi - lo),
                    "shape": [workload["batch"], hi - lo], "row_stride_bytes": 4 * (hi - lo), "dtype": "FP32"}
        at += combined["bytes"]
        result.append({"core": core, "columns": [lo, hi], "space": "private_accumulator_arena",
                       "arena_base_bytes": arena_base, "route_state": route_state,
                       "original_x": original_x, "inboxes": inboxes,
                       "combined_output": combined, "reserve_bytes": at})
    return result


def _arena(expert: dict[str, Any], core: int, m_lanes: list[int], f_range: list[int],
           h_range: list[int], result: dict[str, Any], resident_bands: int) -> dict[str, Any]:
    m, h, f = expert["Me"], expert["H"], expert["F"]
    nf, nh = f_range[1] - f_range[0], h_range[1] - h_range[0]
    share = partition_bytes(ACC_BYTES, m_lanes)[core]
    control = partition_bytes(CONTROL_BYTES, m_lanes)[core]
    at = align(result["arena_base_bytes"] + result["reserve_bytes"])
    base = at
    allocations = []

    def allocate(name: str, shape: list[int], item_bytes: int, dtype: str, lifetime: str) -> None:
        nonlocal at
        at = align(at)
        size = math.prod(shape) * item_bytes
        allocations.append({"name": name, "space": "private_accumulator_arena", "core": core,
                            "offset_bytes": at, "bytes": size, "shape": shape, "dtype": dtype,
                            "row_stride_bytes": shape[-1] * item_bytes, "lifetime": lifetime})
        at += size

    allocate("x", [m, h], 2, "BF16", "admission through last gate/up operand read")
    # Gate occupies its own F slice of this full-Z allocation. The vector unit
    # reads gate/up before overwriting the gate slice with Z; peer slices arrive
    # explicitly before any down consumer. No additional Z allocation is hidden.
    allocate("gate_z", [m, f], 2, "BF16", "gate output, then Z in place, through last down operand read")
    allocate("up", [m, nf], 2, "BF16", "up output through local product")
    allocate("producer_y", [m, nh], 4, "FP32", "down partials through output inbox acknowledgement")
    for index in (0, 1):
        allocate("scratch%d" % index, [resident_bands, m, 32], 1, "FP32_data_and_metadata",
                 "bounded active groups; metadata released at group retirement")
        allocations[-1].update({"record_stride_bytes": 32, "data_bytes_per_record": 16,
                               "metadata_bytes_per_record": 16, "metadata_offset_bytes": 16})
    peak = align(at)
    gate_z = next(v for v in allocations if v["name"] == "gate_z")
    # Persistent output metadata is intentionally absent from result inboxes.
    return {"core": core, "m_lanes": m_lanes[core], "budget_bytes": share,
            "control_reserve_bytes": control, "result_reserve_bytes": result["reserve_bytes"],
            "route_state_bytes": result["route_state"]["bytes"],
            "result_arena_base_bytes": result["arena_base_bytes"],
            "workspace_base_bytes": base, "workspace_bytes": peak - base,
            "peak_private_bytes": peak, "headroom_bytes": share - peak, "feasible": peak <= share,
            "allocations": allocations,
            "aliases": [{"allocation": "gate_z", "producer": "gate", "consumer_then_writer": "silu_product",
                         "gate_view": {"address_bytes": gate_z["offset_bytes"] + 2 * f_range[0],
                                       "shape": [m, nf], "row_stride_bytes": 2 * f},
                         "z_view": {"address_bytes": gate_z["offset_bytes"],
                                    "shape": [m, f], "row_stride_bytes": 2 * f},
                         "overwrite_rule": "each gate/up chunk must finish source reads before writing its Z chunk",
                         "paired_rule": "peer Z fragments fill nonlocal F columns; down waits for all Z fragments",
                         "vector_reads_and_writes_are_charged": True}],
            "metadata_release": "retired group metadata is released; completed FP32 results do not retain old 16B/row metadata",
            "bytes_not_in_this_budget": "private weight SRAM, X operand SRAM, separately billed pipeline registers"}


def _stages(expert: dict[str, Any], f_range: list[int], h_range: list[int], core_m: int) -> list[dict[str, Any]]:
    stages = []
    for phase, (lo, hi) in (("gate", f_range), ("up", f_range), ("down", h_range)):
        w = expert["weights"][phase]
        n, k = hi - lo, w["shape_nk"][1]
        m_tiles = (expert["Me"] + core_m - 1) // core_m
        n_tiles, k_tiles = (n + N_TILE - 1) // N_TILE, (k + K_TILE - 1) // K_TILE
        stages.append({"phase": phase, "M": expert["Me"], "N": n, "K": k, "columns": [lo, hi],
                       "n_tiles": n_tiles, "k_tiles": k_tiles, "m_tiles": m_tiles,
                       "invocations": m_tiles * n_tiles * k_tiles,
                       "useful_macs": expert["Me"] * n * k,
                       "padded_macs": m_tiles * core_m * n_tiles * N_TILE * k_tiles * K_TILE,
                       "hbm_base": w["hbm_base"] + lo * w["row_stride_bytes"],
                       "weight_row_stride_bytes": w["row_stride_bytes"], "weight_bytes": n * w["row_stride_bytes"],
                       "weight_view": {"layout": "W[N,K]", "shape": [n, k], "element_bytes": 2},
                       "output_order": "increasing K segments per output; owner retained until retirement"})
    return stages


def _retire_copies(expert: dict[str, Any], source_core: int, source_columns: list[int],
                   result_layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copies = []
    for dst in result_layout:
        lo = max(source_columns[0], dst["columns"][0])
        hi = min(source_columns[1], dst["columns"][1])
        if hi <= lo:
            continue
        inbox = next(i for i in dst["inboxes"] if i["expert_id"] == expert["id"])
        payload = expert["Me"] * (hi - lo) * 4
        copies.append({"source_core": source_core, "destination_core": dst["core"], "columns": [lo, hi],
                       "payload_bytes": payload, "source_read_bytes": payload, "destination_write_bytes": payload,
                       "shared_bus_payload_bytes": payload,
                       "remote_payload_bytes": payload if source_core != dst["core"] else 0,
                       "destination_inbox_offset_bytes": inbox["offset_bytes"] + (lo - dst["columns"][0]) * 4,
                       "destination_address_bytes": inbox["address_bytes"] + (lo - dst["columns"][0]) * 4,
                       "destination_row_stride_bytes": inbox["row_stride_bytes"],
                       "local_copies_are_charged": True})
    return copies


def _input_copies(expert: dict[str, Any], destination_core: int,
                  result_layout: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copies = []
    for src in result_layout:
        lo, hi = src["columns"]
        payload = 2 * expert["Me"] * (hi - lo)
        copies.append({"source_core": src["core"], "destination_core": destination_core,
                       "columns": [lo, hi], "source_token_indices": expert["token_indices"],
                       "source_offset_bytes": src["original_x"]["offset_bytes"],
                       "source_address_bytes": src["original_x"]["address_bytes"],
                       "source_row_stride_bytes": src["original_x"]["row_stride_bytes"],
                       "destination_column_offset_bytes": lo * 2,
                       "destination_row_stride_bytes": expert["H"] * 2,
                       "payload_bytes": payload, "source_read_bytes": payload,
                       "destination_write_bytes": payload, "shared_bus_payload_bytes": payload,
                       "remote_payload_bytes": payload if src["core"] != destination_core else 0,
                       "local_copies_are_charged": True})
    return copies


def compile_workload(workload: dict[str, Any], m_lanes: list[int] | tuple[int, ...],
                     mode: str = "whole", resident_bands: int = 4) -> dict[str, Any]:
    """Emit candidates, never use future latency or secretly assign extra memory."""
    m_lanes = list(m_lanes)
    if tuple(m_lanes) not in ORGANIZATIONS or mode not in ("whole", "paired_n"):
        raise ValueError("supported organizations are 6, 3+3, 4+2; modes whole/paired_n")
    nc = len(m_lanes)
    w_slots = (WEIGHT_BYTES - INGRESS_BYTES) // nc // 4096
    if not 1 <= resident_bands <= w_slots - 1:
        raise ValueError("current group must leave one W slot for bounded lookahead")
    results = _result_layout(workload, m_lanes)
    total_rows = sum(e["Me"] for e in workload["experts"])
    expected_results = (4 * workload["hidden"] * (total_rows + workload["batch"])
                        + 2 * workload["batch"] * workload["hidden"])
    assert sum(r["reserve_bytes"] for r in results) == expected_results
    experts = []
    for e in workload["experts"]:
        variants = []
        if mode == "whole":
            choices = [[(c, [0, e["F"]], [0, e["H"]])] for c in range(nc)]
        else:
            choices = [[(c, fr, hr) for c, (fr, hr) in enumerate(zip(
                column_ranges(e["F"], m_lanes), column_ranges(e["H"], m_lanes)))]]
        for choice in choices:
            cores = []
            for c, fr, hr in choice:
                arena = _arena(e, c, m_lanes, fr, hr, results[c], resident_bands)
                stages = _stages(e, fr, hr, m_lanes[c])
                cores.append({"core": c, "f_columns": fr, "h_columns": hr,
                              "storage": arena, "stages": stages,
                              "input_bytes": 2 * e["Me"] * e["H"],
                              "input_copies": _input_copies(e, c, results),
                              "retire_copies": _retire_copies(e, c, hr, results)})
            z_copies = []
            if len(choice) > 1:
                for c, fr, _ in choice:
                    for dst, _, _ in choice:
                        if dst == c:
                            continue
                        payload = 2 * e["Me"] * (fr[1] - fr[0])
                        z_copies.append({"source_core": c, "destination_core": dst,
                                         "f_columns": fr, "payload_bytes": payload,
                                         "source_read_bytes": payload, "shared_bus_payload_bytes": payload,
                                         "destination_write_bytes": payload, "must_complete_before": "down"})
            variants.append({"owner_cores": [v[0] for v in choice], "feasible": all(c["storage"]["feasible"] for c in cores),
                             "cores": cores, "z_copies": z_copies,
                             "total_weight_bytes": sum(s["weight_bytes"] for c in cores for s in c["stages"]),
                             "total_useful_macs": sum(s["useful_macs"] for c in cores for s in c["stages"]),
                             "input_distribution_bytes": sum(c["input_bytes"] for c in cores),
                             "z_cross_core_bytes": sum(v["payload_bytes"] for v in z_copies)})
        experts.append({"expert_id": e["id"], "Me": e["Me"], "H": e["H"], "F": e["F"],
                        "is_shared": e["is_shared"], "candidates": variants})
    existing = 1200 if nc == 1 else 1872
    added = {"pending_descriptors": 8 * 64, "active_contexts": nc * 128,
             "cost_lut": 8 * 3 * nc * 8, "global_credit_age_cursors": 128}
    control_used = existing + sum(added.values())
    if control_used > CONTROL_BYTES:
        raise ValueError("finite controller state exceeds old 4KiB reservation")
    # Locate the shared pending queue/global counters in core 0's reserved
    # control arena. Local context and per-core LUT storage are charged locally.
    local_control = [existing // nc + 128 + 8 * 3 * 8 for _ in m_lanes]
    local_control[0] += added["pending_descriptors"] + added["global_credit_age_cursors"]
    local_reserves = partition_bytes(CONTROL_BYTES, m_lanes)
    if any(used > reserve for used, reserve in zip(local_control, local_reserves)):
        raise ValueError("controller state exceeds an individual private reservation")
    assert sum(local_control) == control_used
    return {
        "schema": PLAN_SCHEMA, "workload_id": workload["id"], "mode": mode, "m_lanes": m_lanes,
        "dimensions_order": "M,N,K", "n_tile": N_TILE, "k_tile": K_TILE,
        "physical_multiplier_count": sum(m_lanes) * N_TILE * K_TILE,
        "budget": {"private_accumulator_total_bytes": ACC_BYTES, "control_reserve_inside_acc_bytes": CONTROL_BYTES,
                   "private_weight_slots_per_core": w_slots, "weight_total_bytes": WEIGHT_BYTES,
                   "weight_ingress_bytes": INGRESS_BYTES, "x_total_bytes": X_BYTES, "x_slots_per_core": 2,
                   "x_bytes_per_core": [m * 2 * K_TILE * 2 for m in m_lanes],
                   "private_accumulator_bytes": partition_bytes(ACC_BYTES, m_lanes),
                   "control_bytes_per_core": partition_bytes(CONTROL_BYTES, m_lanes),
                   "residue_bytes_unallocated": 0},
        "controller": {"existing_state_bytes": existing, "added_state": added,
                       "total_state_bytes": control_used, "reserve_headroom_bytes": CONTROL_BYTES - control_used,
                       "state_bytes_per_core": local_control,
                       "headroom_bytes_per_core": [r - u for r, u in zip(local_reserves, local_control)],
                       "global_queue_storage_owner": 0,
                       "pending_window": 8, "active_experts_per_core": 1,
                       "selection_service": "finite scan and descriptor updates must be charged by simulator"},
        "result_layout": results, "result_total_bytes": expected_results,
        "result_policy": "fixed output-column owners; original X, component inboxes and final output reserved before dispatch",
        "combine_policy": "ascending captured route slot; BF16-round down results, FP32 weighted sum, BF16-round then add shared",
        "ownership_policy": "unbound until atomic workspace/result/W-group admission; immutable after first DMA",
        "weight_policy": "one expert owner does not imply all weights resident; finite streaming slots",
        "x_policy": "two operand slots; resident group uses same X across its N bands",
        "resident_bands": resident_bands,
        "metadata_policy": "only active groups retain 16B/row metadata; finished inbox holds FP32 payload only",
        "experts": experts,
        "claims": {"latency_estimated_here": False, "main_compiler_isa": False,
                   "full_model_end_to_end": False, "actual_sram_payload_execution": False},
    }


def engine_layout(workload: dict[str, Any], m_lanes: list[int] | tuple[int, ...],
                  group: int = 4) -> dict[str, Any]:
    """Compact absolute bank addresses for the timing engine, indexed by expert.

    `whole[e][c]` and `paired_n[e][c]` have the same expert/core order as the
    workload and organization. Every base is local to the named core's private
    accumulator SRAM. Alias views do not allocate additional payload.
    """
    whole = compile_workload(workload, m_lanes, "whole", group)
    paired = compile_workload(workload, m_lanes, "paired_n", group)

    def compact(c: dict[str, Any]) -> dict[str, Any]:
        storage = c["storage"]
        allocations = {
            a["name"]: {"base": a["offset_bytes"], "bytes": a["bytes"],
                        "stride": a.get("record_stride_bytes", a["row_stride_bytes"]),
                        "shape": a["shape"]}
            for a in storage["allocations"]
        }
        for name, view in (("gate", storage["aliases"][0]["gate_view"]),
                           ("z", storage["aliases"][0]["z_view"])):
            allocations[name] = {"base": view["address_bytes"], "bytes": math.prod(view["shape"]) * 2,
                                 "stride": view["row_stride_bytes"], "shape": view["shape"],
                                 "aliases": "gate_z"}
        allocations["y"] = {**allocations["producer_y"], "aliases": "producer_y"}
        return {"core": c["core"], "workspace_bytes": storage["workspace_bytes"],
                "peak": storage["peak_private_bytes"], "feasible": storage["feasible"],
                "allocations": allocations, "f_columns": c["f_columns"], "h_columns": c["h_columns"]}

    return {
        "schema": "plena_moe_dispatch_engine_layout_v1", "workload_id": workload["id"],
        "m_lanes": list(m_lanes), "group": group, "address_units": "local private SRAM bytes",
        "expert_ids": [e["id"] for e in workload["experts"]],
        "cores": [{"core": r["core"],
                   "capacity": whole["budget"]["private_accumulator_bytes"][r["core"]],
                   "control": whole["budget"]["control_bytes_per_core"][r["core"]],
                   "reserved": align(r["arena_base_bytes"] + r["reserve_bytes"]),
                   "result_layout": r}
                  for r in whole["result_layout"]],
        "whole": [[compact(candidate["cores"][0]) for candidate in e["candidates"]]
                  for e in whole["experts"]],
        "paired_n": [[compact(c) for c in e["candidates"][0]["cores"]]
                     for e in paired["experts"]],
        "control_accounting": whole["controller"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="import an original capture; requires --config and source payloads")
    source.add_argument("--workloads", type=Path, default=DEFAULT_WORKLOADS, help="portable route/dimension bundle")
    parser.add_argument("--config", type=Path, help="model config for --manifest import")
    parser.add_argument("--output", type=Path, default=Path("outputs/moe_dispatch/workloads.json"))
    parser.add_argument("--plans-dir", type=Path, help="optional: write 24 explicit plans, preferably under /tmp")
    parser.add_argument("--engine-layouts-dir", type=Path, help="optional: write 12 compact bank-address plans")
    args = parser.parse_args()
    if bool(args.manifest) != bool(args.config):
        parser.error("--manifest and --config must be supplied together")
    data = extract_workloads(args.manifest, args.config) if args.manifest else load_workloads(args.workloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.plans_dir:
        args.plans_dir.mkdir(parents=True, exist_ok=True)
        for workload in data["workloads"]:
            for shape in ORGANIZATIONS:
                for mode in ("whole", "paired_n"):
                    plan = compile_workload(workload, shape, mode)
                    name = "%s__%s__%s.json" % (workload["id"], "+".join(map(str, shape)), mode)
                    (args.plans_dir / name).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    if args.engine_layouts_dir:
        args.engine_layouts_dir.mkdir(parents=True, exist_ok=True)
        for workload in data["workloads"]:
            for shape in ORGANIZATIONS:
                name = "%s__%s.json" % (workload["id"], "+".join(map(str, shape)))
                (args.engine_layouts_dir / name).write_text(json.dumps(
                    engine_layout(workload, shape), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "workloads": len(data["workloads"]), "source_manifest_sha256": data["provenance"]["manifest"]["sha256"]}))


if __name__ == "__main__":
    main()
