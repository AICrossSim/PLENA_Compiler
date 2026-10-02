"""Bounded-token execution for otherwise unsupported legacy accumulator plans.

The original joint-v1 planner and its supported layouts remain unchanged. This
outer protocol keeps the complete layer input, final output, and original route
metadata resident, and reuses the legacy gather/retire/combine protocol for each
sequential token chunk. It allocates no additional hardware or payload copies.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any


# The study harness also imports a simulator module named ``compiler``. Resolve
# the unchanged legacy planner beside this helper, independent of sys.path and
# that cached module name, so its physical layouts come from the pinned source.
_compiler_path = Path(__file__).resolve().with_name("compiler.py")
_compiler_spec = importlib.util.spec_from_file_location(__name__ + "_original_compiler", _compiler_path)
if _compiler_spec is None or _compiler_spec.loader is None:
    raise ImportError("cannot load the original legacy planner: " + str(_compiler_path))
compiler = importlib.util.module_from_spec(_compiler_spec)
_compiler_spec.loader.exec_module(compiler)


SCHEMA = "plena_legacy_token_chunks_v1"
CONTROL_BYTES = 32
CONTROL_FIELDS = ["current_chunk_row_start", "current_chunk_row_end",
                  "current_chunk_index", "route_cursor", "expert_cursor",
                  "mode_flags", "next_chunk_bound", "reserved"]


def _all_whole_fit(layout: dict[str, Any]) -> bool:
    return bool(layout["whole"]) and all(any(c["feasible"] for c in expert)
                                         for expert in layout["whole"])


def _validate_routes(workload: dict[str, Any]) -> None:
    """Refuse ambiguous rows before slicing; never renumber or normalize routes."""
    batch = workload["batch"]
    if not isinstance(batch, int) or batch <= 0:
        raise ValueError("legacy token chunks require a positive token population")
    if [t["token_index"] for t in workload["tokens"]] != list(range(batch)):
        raise ValueError("legacy token chunks require complete ordered token indices")
    experts = workload["experts"]
    if not experts or len({e["id"] for e in experts}) != len(experts):
        raise ValueError("legacy token chunks require distinct expert IDs")
    expected = {}
    for t in workload["tokens"]:
        if len(t["routes"]) != workload["top_k"]:
            raise ValueError("legacy token chunk route coverage disagrees with top_k")
        for route in t["routes"]:
            key = (route["expert_id"], t["token_index"], route["slot"])
            if key in expected:
                raise ValueError("legacy token chunks contain a duplicate route")
            expected[key] = route["score"]
    seen = {}
    shared = 0
    for e in experts:
        indices, slots, scores = (e[k] for k in ("token_indices", "route_slots", "route_scores"))
        if e["Me"] != len(indices) or len(indices) != len(slots) or len(indices) != len(scores):
            raise ValueError("legacy expert row/route/score lengths disagree")
        if any(not isinstance(t, int) or not 0 <= t < batch for t in indices):
            raise ValueError("legacy expert has an out-of-range token index")
        if e["is_shared"]:
            shared += 1
            if indices != list(range(batch)):
                raise ValueError("legacy shared expert must retain every original token")
        else:
            for t, slot, score in zip(indices, slots, scores):
                key = (e["id"], t, slot)
                if key in seen:
                    raise ValueError("legacy expert contains a duplicate routed row")
                seen[key] = score
    if shared != 1 or seen != expected:
        raise ValueError("legacy expert metadata does not conserve every captured route")


def _control_record(layout: dict[str, Any]) -> dict[str, Any]:
    accounting = layout["control_accounting"]
    for core, headroom in enumerate(accounting["headroom_bytes_per_core"]):
        if headroom >= CONTROL_BYTES:
            base = accounting["state_bytes_per_core"][core]
            if base + CONTROL_BYTES <= layout["cores"][core]["control"]:
                return {"core": core, "base": base, "bytes": CONTROL_BYTES,
                        "field_bytes": 4, "fields": list(CONTROL_FIELDS)}
    raise ValueError("legacy token chunk wrapper requires 32B of declared controller headroom")


def _persistent_cores(workload: dict[str, Any], layout: dict[str, Any]) -> list[dict[str, Any]]:
    persistent = []
    for core in layout["cores"]:
        result = core["result_layout"]
        original_x = copy.deepcopy(result["original_x"])
        original_x["lifetime"] = "layer entry through the final chunk input gather"
        combined = copy.deepcopy(result["combined_output"])
        combined["offset_bytes"] = original_x["offset_bytes"] + original_x["bytes"]
        combined["address_bytes"] = result["arena_base_bytes"] + combined["offset_bytes"]
        combined["lifetime"] = "layer entry through all token chunk combines"
        reserved = compiler.align(combined["address_bytes"] + combined["bytes"])
        if reserved > core["capacity"]:
            raise ValueError("legacy full-layer persistent arena exceeds core %d capacity: %d > %d" %
                             (core["core"], reserved, core["capacity"]))
        persistent.append({"core": core["core"], "columns": list(result["columns"]),
                           "capacity": core["capacity"], "control": core["control"],
                           "arena_base_bytes": result["arena_base_bytes"],
                           "route_state": copy.deepcopy(result["route_state"]),
                           "original_x": original_x, "combined_output": combined,
                           "reserved": reserved, "headroom_bytes": core["capacity"] - reserved})
    return persistent


def _slice(workload: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    child = copy.deepcopy(workload)
    child.pop("engine_layout", None)
    child.pop("legacy_batch_execution", None)
    child["id"] = "%s_legacy_tokens_%d_%d" % (workload["id"], start, end)
    child["batch"] = end - start
    child["original_token_indices"] = list(range(start, end))
    child["tokens"] = copy.deepcopy(workload["tokens"][start:end])
    for token in child["tokens"]:
        token["original_token_index"] = token["token_index"]
        token["token_index"] -= start
    child["experts"] = []
    for original in workload["experts"]:
        rows = [i for i, t in enumerate(original["token_indices"]) if start <= t < end]
        if not rows:
            continue
        expert = copy.deepcopy(original)
        expert["Me"] = len(rows)
        expert["original_token_indices"] = [original["token_indices"][i] for i in rows]
        expert["token_indices"] = [t - start for t in expert["original_token_indices"]]
        expert["route_slots"] = [original["route_slots"][i] for i in rows]
        expert["route_scores"] = [original["route_scores"][i] for i in rows]
        expert["dag"] = compiler.expert_dag(expert)
        child["experts"].append(expert)
    if "input" in child:
        child["input"]["shape"] = [end - start, workload["hidden"]]
        child["input"]["original_token_range"] = [start, end]
    return child


def _row_alias(region: dict[str, Any], start: int, end: int, arena_base: int,
               alias: str) -> dict[str, Any]:
    view = copy.deepcopy(region)
    view["address_bytes"] += start * view["row_stride_bytes"]
    view["offset_bytes"] = view["address_bytes"] - arena_base
    view["bytes"] = (end - start) * view["row_stride_bytes"]
    view["shape"][0] = end - start
    view["aliases"] = alias
    view["original_token_range"] = [start, end]
    return view


def _rebase(child: dict[str, Any], layout: dict[str, Any], persistent: list[dict[str, Any]],
            start: int, end: int, control: dict[str, Any]) -> None:
    """Shift every workspace view together; keep global row aliases untouched."""
    for core, parent in zip(layout["cores"], persistent):
        old_reserved = core["reserved"]
        result = core["result_layout"]
        arena_base = parent["arena_base_bytes"]
        result["arena_base_bytes"] = arena_base
        result["route_state"] = copy.deepcopy(parent["route_state"])
        result["original_x"] = _row_alias(parent["original_x"], start, end, arena_base,
                                          "legacy_batch_execution.persistent_cores.original_x")
        result["combined_output"] = _row_alias(parent["combined_output"], start, end, arena_base,
                                               "legacy_batch_execution.persistent_cores.combined_output")
        at = parent["reserved"]
        for inbox in result["inboxes"]:
            inbox["address_bytes"] = at
            inbox["offset_bytes"] = at - arena_base
            at += inbox["bytes"]
        core["reserved"] = compiler.align(at)
        result["reserve_bytes"] = core["reserved"] - arena_base
        result["persistent_reserve_bytes"] = parent["reserved"]
        delta = core["reserved"] - old_reserved
        for mode in ("whole", "paired_n"):
            for expert in layout[mode]:
                local = expert[core["core"]]
                for allocation in local["allocations"].values():
                    allocation["base"] += delta
                local["peak"] = core["reserved"] + local["workspace_bytes"]
                local["feasible"] = local["peak"] <= core["capacity"]
    accounting = layout["control_accounting"]
    accounting["added_state"]["legacy_token_chunk_control"] = CONTROL_BYTES
    accounting["total_state_bytes"] += CONTROL_BYTES
    accounting["reserve_headroom_bytes"] -= CONTROL_BYTES
    accounting["state_bytes_per_core"][control["core"]] += CONTROL_BYTES
    accounting["headroom_bytes_per_core"][control["core"]] -= CONTROL_BYTES
    accounting["legacy_token_chunk_control"] = copy.deepcopy(control)
    layout["runtime_protocol"]["legacy_token_chunks"] = {
        "execution": "sequential; complete legacy retire and ordered combine before next chunk",
        "gather_retire_combine": "unchanged legacy moves and charges within each chunk",
        "original_x_and_final_y": "direct aliases of persistent global row ranges",
        "wrapper_payload_copy_bytes": 0,
        "control_record": copy.deepcopy(control),
    }
    child["engine_layout"] = layout


def _chunks(workload: dict[str, Any], size: int, lanes: list[int], group: int, resources,
            persistent: list[dict[str, Any]], control: dict[str, Any]) -> list[dict[str, Any]] | None:
    chunks = []
    for start in range(0, workload["batch"], size):
        end = min(start + size, workload["batch"])
        child = _slice(workload, start, end)
        layout = compiler.engine_layout(child, lanes, group, resources)
        _rebase(child, layout, persistent, start, end, control)
        if not _all_whole_fit(layout):
            return None
        chunks.append({"token_range": [start, end], "workload": child})
    return chunks


def _weight_bytes(workload: dict[str, Any]) -> int:
    return sum(w["physical_bytes"] for e in workload["experts"] for w in e["weights"].values())


def prepare_legacy_workload(workload: dict[str, Any], lanes, group: int = 4,
                            resources=None) -> dict[str, Any]:
    """Keep supported legacy plans exact, otherwise choose the largest fitting chunk.

    A candidate size is accepted only if every actual contiguous chunk, including
    its shared expert, has a feasible whole placement for every active expert.
    Persistent storage is checked on *every* physical core, including cores that
    receive no expert. The original workload and original engine layout remain
    available unchanged alongside the distinct large-batch execution protocol.
    """
    lanes = list(lanes)
    prepared = copy.deepcopy(workload)
    original = compiler.engine_layout(workload, lanes, group, resources)
    if "engine_layout" in prepared and prepared["engine_layout"] != original:
        raise ValueError("supplied legacy engine layout differs from the original compiler plan")
    prepared["engine_layout"] = original
    if _all_whole_fit(original):
        return prepared
    if "legacy_batch_execution" in prepared:
        raise ValueError("legacy workload already carries an outer token chunk plan")
    _validate_routes(workload)
    persistent = _persistent_cores(workload, original)
    control = _control_record(original)
    for size in range(workload["batch"] - 1, 0, -1):
        chunks = _chunks(workload, size, lanes, group, resources, persistent, control)
        if chunks is not None:
            expected_macs = sum(3 * e["Me"] * e["H"] * e["F"] for e in workload["experts"])
            prepared["legacy_batch_execution"] = {
                "schema": SCHEMA, "chunk_size": size, "persistent_cores": persistent,
                "control_record": control, "chunks": chunks,
                "expected_useful_macs": expected_macs,
                "unique_weight_bytes": _weight_bytes(workload),
                "weight_read_bytes": sum(_weight_bytes(c["workload"]) for c in chunks),
                "wrapper_payload_copy_bytes": 0,
                "timing_scope": "legacy large-batch token chunks; full persistent layer arena",
            }
            return prepared
    raise ValueError("no legacy token chunk fits after the full persistent layer reservation")
