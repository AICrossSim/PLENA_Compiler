"""Static-index Mixture-of-Experts routing metadata.

The first PLENA MoE lowering specializes the generated program to one compiler
input.  Only expert *indices* are selected on the host; router projection,
softmax, route-weight extraction, normalization, expert execution and combine
remain in the emitted ISA.  This module deliberately contains no ISA logic so
the routing contract is shared by the scheduled reference, compiler and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections import defaultdict
from typing import Any, Sequence

import torch


MOE_LOWERING_SCHEDULE_COMPACT_ROUTE_V2 = "compact-route-v2"
MOE_LOWERING_SCHEDULE_LEGACY_STATIC_V1 = "legacy-static-v1"
MOE_LOWERING_SCHEDULES = frozenset(
    {
        MOE_LOWERING_SCHEDULE_COMPACT_ROUTE_V2,
        MOE_LOWERING_SCHEDULE_LEGACY_STATIC_V1,
    }
)


@dataclass(frozen=True, order=True)
class StaticRoute:
    """One token-to-expert edge in deterministic token/rank order."""

    token_index: int
    physical_row: int
    rank: int
    expert_id: int

    def as_dict(self) -> dict[str, int]:
        return {
            "token_index": self.token_index,
            "physical_row": self.physical_row,
            "rank": self.rank,
            "expert_id": self.expert_id,
        }


@dataclass(frozen=True)
class MoeRoutingPlan:
    """Stable, serializable set of host-selected expert indices."""

    num_experts: int
    experts_per_token: int
    routes: tuple[StaticRoute, ...]
    topk_margin_min: float | None = None

    @property
    def route_count(self) -> int:
        return len(self.routes)

    @property
    def active_expert_ids(self) -> tuple[int, ...]:
        return tuple(sorted({route.expert_id for route in self.routes}))

    @property
    def routes_per_expert(self) -> dict[int, int]:
        counts = {expert_id: 0 for expert_id in self.active_expert_ids}
        for route in self.routes:
            counts[route.expert_id] += 1
        return counts

    @property
    def routing_plan_hash(self) -> str:
        payload = json.dumps(self.as_dict(include_hash=False), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def as_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "mode": "static-indices",
            "num_experts": self.num_experts,
            "experts_per_token": self.experts_per_token,
            "routes": [route.as_dict() for route in self.routes],
            "topk_margin_min": self.topk_margin_min,
        }
        if include_hash:
            result["routing_plan_hash"] = self.routing_plan_hash
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MoeRoutingPlan":
        routes = tuple(
            StaticRoute(
                token_index=int(route["token_index"]),
                physical_row=int(route.get("physical_row", route["token_index"])),
                rank=int(route["rank"]),
                expert_id=int(route["expert_id"]),
            )
            for route in value["routes"]
        )
        plan = cls(
            num_experts=int(value["num_experts"]),
            experts_per_token=int(value["experts_per_token"]),
            routes=routes,
            topk_margin_min=(None if value.get("topk_margin_min") is None else float(value["topk_margin_min"])),
        )
        expected_hash = value.get("routing_plan_hash")
        if expected_hash is not None and expected_hash != plan.routing_plan_hash:
            raise ValueError("routing_plan_hash does not match the canonical static routing plan")
        return plan

    def validate(self, *, active_physical_rows: Sequence[int], max_routes: int) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.experts_per_token <= 0:
            raise ValueError("experts_per_token must be positive")
        active_rows = tuple(int(row) for row in active_physical_rows)
        if len(set(active_rows)) != len(active_rows):
            raise ValueError("active_physical_rows must be unique")
        expected_route_count = len(active_rows) * self.experts_per_token
        if self.route_count != expected_route_count:
            raise ValueError(
                f"Static route count {self.route_count} does not match "
                f"active_tokens({len(active_rows)}) * top_k({self.experts_per_token}) "
                f"= {expected_route_count}"
            )
        if self.route_count > max_routes:
            raise ValueError(f"Static route count {self.route_count} exceeds max_static_routes={max_routes}")
        seen: set[tuple[int, int]] = set()
        token_ranks: dict[int, set[int]] = {}
        for route in self.routes:
            if not 0 <= route.token_index < len(active_rows):
                raise ValueError(f"token_index={route.token_index} outside [0, {len(active_rows)})")
            expected_row = active_rows[route.token_index]
            if route.physical_row != expected_row:
                raise ValueError(
                    f"Route token {route.token_index} uses physical_row={route.physical_row}, expected {expected_row}"
                )
            if not 0 <= route.expert_id < self.num_experts:
                raise ValueError(f"expert_id={route.expert_id} outside [0, {self.num_experts})")
            if not 0 <= route.rank < self.experts_per_token:
                raise ValueError(f"route rank={route.rank} outside [0, {self.experts_per_token})")
            key = (route.token_index, route.rank)
            if key in seen:
                raise ValueError(f"Duplicate static route for token/rank {key}")
            seen.add(key)
            token_ranks.setdefault(route.token_index, set()).add(route.rank)
        expected_ranks = set(range(self.experts_per_token))
        if set(token_ranks) != set(range(len(active_rows))):
            raise ValueError(
                f"Routing plan covers token indices {sorted(token_ranks)}, expected {list(range(len(active_rows)))}"
            )
        for token_index, ranks in token_ranks.items():
            if ranks != expected_ranks:
                raise ValueError(
                    f"token {token_index} has route ranks {sorted(ranks)}, expected {sorted(expected_ranks)}"
                )


@dataclass(frozen=True)
class MoeRouteRun:
    """One maximal constant-rank arithmetic run inside an expert bucket."""

    expert_id: int
    rank: int
    bucket_row_start: int
    physical_row_start: int
    count: int
    physical_row_stride: int


@dataclass(frozen=True)
class MoeExpertRoutePlan:
    """Compiler-oriented indexing of a static routing plan.

    Token order is retained for route-weight normalization. Expert routes are
    sorted by physical row because each expert FFN row is independent; this
    exposes deterministic affine runs for dispatch/combine lowering.
    """

    routes_by_token: tuple[tuple[StaticRoute, ...], ...]
    routes_by_expert: tuple[tuple[int, tuple[StaticRoute, ...]], ...]
    runs_by_expert: tuple[tuple[int, tuple[MoeRouteRun, ...]], ...]

    @classmethod
    def build(cls, plan: MoeRoutingPlan) -> "MoeExpertRoutePlan":
        token_map: dict[int, list[StaticRoute]] = defaultdict(list)
        expert_map: dict[int, list[StaticRoute]] = defaultdict(list)
        for route in plan.routes:
            token_map[route.token_index].append(route)
            expert_map[route.expert_id].append(route)

        routes_by_token = tuple(
            tuple(sorted(token_map[token], key=lambda route: route.rank)) for token in sorted(token_map)
        )
        routes_by_expert_list: list[tuple[int, tuple[StaticRoute, ...]]] = []
        runs_by_expert_list: list[tuple[int, tuple[MoeRouteRun, ...]]] = []
        for expert_id in sorted(expert_map):
            routes = tuple(
                sorted(
                    expert_map[expert_id],
                    key=lambda route: (route.rank, route.physical_row),
                )
            )
            routes_by_expert_list.append((expert_id, routes))
            runs: list[MoeRouteRun] = []
            start = 0
            while start < len(routes):
                rank = routes[start].rank
                end = start + 1
                stride = 0
                if end < len(routes) and routes[end].rank == rank:
                    stride = routes[end].physical_row - routes[start].physical_row
                    end += 1
                    while (
                        end < len(routes)
                        and routes[end].rank == rank
                        and routes[end].physical_row - routes[end - 1].physical_row == stride
                    ):
                        end += 1
                runs.append(
                    MoeRouteRun(
                        expert_id=expert_id,
                        rank=rank,
                        bucket_row_start=start,
                        physical_row_start=routes[start].physical_row,
                        count=end - start,
                        physical_row_stride=stride,
                    )
                )
                start = end
            runs_by_expert_list.append((expert_id, tuple(runs)))
        return cls(
            routes_by_token=routes_by_token,
            routes_by_expert=tuple(routes_by_expert_list),
            runs_by_expert=tuple(runs_by_expert_list),
        )

    @property
    def route_count(self) -> int:
        return sum(len(routes) for routes in self.routes_by_token)

    @property
    def run_count(self) -> int:
        return sum(len(runs) for _, runs in self.runs_by_expert)

    @property
    def affine_route_count(self) -> int:
        return sum(run.count for _, runs in self.runs_by_expert for run in runs if run.count > 1)

    @property
    def irregular_route_count(self) -> int:
        return self.route_count - self.affine_route_count

    def expert_routes(self, expert_id: int) -> tuple[StaticRoute, ...]:
        for candidate, routes in self.routes_by_expert:
            if candidate == expert_id:
                return routes
        return ()


@dataclass(frozen=True)
class FixedBalancedRoutingSummary:
    """Aggregate round-robin routing for latency-only cost evaluation.

    Unlike :class:`MoeRoutingPlan`, this object intentionally does not retain
    token-to-expert edges.  It is therefore suitable for large prefill cost
    traces where materializing every route would defeat symbolic lowering.
    """

    num_tokens: int
    num_experts: int
    experts_per_token: int
    routes_per_expert_tuple: tuple[int, ...]
    algorithm_version: str = "round_robin_token_rank_v1"

    @classmethod
    def build(cls, *, num_tokens: int, num_experts: int, experts_per_token: int) -> "FixedBalancedRoutingSummary":
        if num_tokens <= 0 or num_experts <= 0 or experts_per_token <= 0:
            raise ValueError("fixed-balanced routing requires positive tokens, experts, and top-k")
        if experts_per_token > num_experts:
            raise ValueError(f"experts_per_token={experts_per_token} exceeds num_experts={num_experts}")
        route_count = num_tokens * experts_per_token
        base, remainder = divmod(route_count, num_experts)
        counts = tuple(base + (1 if expert_id < remainder else 0) for expert_id in range(num_experts))
        return cls(
            num_tokens=num_tokens,
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            routes_per_expert_tuple=counts,
        )

    @property
    def route_count(self) -> int:
        return self.num_tokens * self.experts_per_token

    @property
    def active_expert_ids(self) -> tuple[int, ...]:
        return tuple(expert_id for expert_id, count in enumerate(self.routes_per_expert_tuple) if count)

    @property
    def routes_per_expert(self) -> dict[int, int]:
        return {expert_id: count for expert_id, count in enumerate(self.routes_per_expert_tuple) if count}

    @property
    def routing_summary_hash(self) -> str:
        payload = {
            "algorithm_version": self.algorithm_version,
            "experts_per_token": self.experts_per_token,
            "num_experts": self.num_experts,
            "num_tokens": self.num_tokens,
            "routes_per_expert": self.routes_per_expert_tuple,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def padded_bucket_rows(self, blen: int) -> dict[int, int]:
        if blen <= 0:
            raise ValueError(f"BLEN must be positive, got {blen}")
        return {expert_id: ((count + blen - 1) // blen) * blen for expert_id, count in self.routes_per_expert.items()}


def derive_static_routing_plan(
    router_probabilities: torch.Tensor,
    *,
    active_physical_rows: list[int],
    num_experts: int,
    experts_per_token: int,
    max_routes: int,
) -> MoeRoutingPlan:
    """Select indices from hardware-scheduled router probabilities."""
    if router_probabilities.ndim != 2:
        raise ValueError(f"router_probabilities must be rank 2, got {router_probabilities.shape}")
    if len(active_physical_rows) * experts_per_token > max_routes:
        raise ValueError(
            f"Static route count {len(active_physical_rows) * experts_per_token} exceeds max_static_routes={max_routes}"
        )
    if num_experts > router_probabilities.shape[1]:
        raise ValueError(f"num_experts={num_experts} exceeds router width {router_probabilities.shape[1]}")
    active = router_probabilities[active_physical_rows, :num_experts]
    values, indices = torch.topk(active, k=experts_per_token, dim=-1, sorted=True)
    routes = tuple(
        StaticRoute(
            token_index=token_idx,
            physical_row=physical_row,
            rank=rank,
            expert_id=int(indices[token_idx, rank]),
        )
        for token_idx, physical_row in enumerate(active_physical_rows)
        for rank in range(experts_per_token)
    )
    margin = None
    if num_experts > experts_per_token:
        boundary = torch.topk(active, k=experts_per_token + 1, dim=-1, sorted=True).values
        margin = float((boundary[:, experts_per_token - 1] - boundary[:, experts_per_token]).min())
    plan = MoeRoutingPlan(
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        routes=routes,
        topk_margin_min=margin,
    )
    plan.validate(active_physical_rows=active_physical_rows, max_routes=max_routes)
    return plan


def coerce_routing_plan(value: MoeRoutingPlan | dict[str, Any]) -> MoeRoutingPlan:
    if isinstance(value, MoeRoutingPlan):
        return value
    if isinstance(value, dict):
        return MoeRoutingPlan.from_dict(value)
    raise TypeError(f"Expected MoeRoutingPlan or dict, got {type(value)}")


__all__ = [
    "FixedBalancedRoutingSummary",
    "MOE_LOWERING_SCHEDULE_COMPACT_ROUTE_V2",
    "MOE_LOWERING_SCHEDULE_LEGACY_STATIC_V1",
    "MOE_LOWERING_SCHEDULES",
    "MoeExpertRoutePlan",
    "MoeRouteRun",
    "MoeRoutingPlan",
    "StaticRoute",
    "coerce_routing_plan",
    "derive_static_routing_plan",
]
