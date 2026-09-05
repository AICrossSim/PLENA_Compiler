"""Export ready-input MoE fixtures for the dual normal-buffer execution harness.

This is a manifest exporter, not a router or an ISA lowering. Weights use the
repository's PLENA E4M3/E8M0 block-8 codec (including its non-IEEE subnormals),
with output-major [N,K] rows. The independent scalar reference decodes the
exported bytes, accumulates GEMMs in ascending K with FP32 multiply/add, and
rounds each GEMM and SwiGLU output to BF16. Combine follows token/slot order.

Run this file with PLENA_Tools on PYTHONPATH and an interpreter with torch.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

BLOCK_SIZE = 8
FORMAT = "plena_e4m3_e8m0_block8"


def f32(value: float) -> float:
    """One explicit FP32 rounding; no fused multiply/add in the reference."""
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bf16_bits(value: float) -> int:
    """Round FP32 to BF16, round-to-nearest-even (finite fixture values only)."""
    value = f32(value)
    if not math.isfinite(value):
        raise ValueError("non-finite BF16 value or arithmetic overflow")
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    if (rounded & 0x7F80) == 0x7F80:
        raise ValueError("BF16 rounding overflow")
    return rounded


def bf16_value(bits: int) -> float:
    if not isinstance(bits, int) or not 0 <= bits <= 0xFFFF:
        raise ValueError("BF16 bits must be uint16")
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def _bf16(value: float) -> float:
    return bf16_value(bf16_bits(value))


def _matrix(values: Any, label: str) -> list[list[float]]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    elif hasattr(values, "tolist"):
        values = values.tolist()
    rows = [[f32(x) for x in row] for row in values]
    if not rows or not rows[0] or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError(f"{label} must be a nonempty rectangular matrix")
    if not all(math.isfinite(x) for r in rows for x in r):
        raise ValueError(f"{label} contains non-finite or FP32-overflow values")
    return rows


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def encode_matrix(values: Any) -> tuple[bytes, bytes, dict[str, int]]:
    """Use the actual PLENA_Tools quantizer; pad each reduction row to block 8.

    Element and scale payloads are separate. Padding belongs to its own row and
    never shares a block with the next output channel. No transpose/requantize
    is performed by the runtime.
    """
    import torch
    try:
        from plena_quant.mxfp.quantizer import _mx_fp_quantize_hardware
        from plena_quant.mxfp.utils import pack_fp_to_bin
    except ImportError as exc:
        raise ImportError("Place PLENA_Tools on PYTHONPATH for the actual PLENA codec") from exc

    rows = _matrix(values, "weight")
    n, k = len(rows), len(rows[0])
    stride = ((k + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    padded = torch.tensor([r + [0.0] * (stride - k) for r in rows], dtype=torch.float32)
    with torch.no_grad():
        _, exponents, mantissas, scales = _mx_fp_quantize_hardware(
            padded, width=8, exponent_width=4, exponent_bias_width=8, block_size=[BLOCK_SIZE]
        )
        elements = pack_fp_to_bin(exponents, mantissas, exp_width=4, man_width=3)
    element_list = [int(x) for x in elements.flatten().tolist()]
    scale_list = [int(x) for x in scales.flatten().tolist()]
    if any(not 0 <= x <= 254 for x in scale_list):
        raise ValueError("weight scale is outside finite PLENA E8M0 range")
    if any((x & 0x78) == 0x78 for x in element_list):
        raise ValueError("weight contains a non-finite encoded E4M3 element")
    return bytes(element_list), bytes(scale_list), {
        "rows": n, "cols": k, "element_row_stride": stride,
        "scale_row_stride": stride // BLOCK_SIZE,
    }


def decode_element(element: int, scale: int) -> float:
    """Independent scalar interpretation of quantize::FpType's local format."""
    exponent, mantissa = (element >> 3) & 15, element & 7
    if exponent == 15 or scale == 255:
        raise ValueError("non-finite PLENA weight encoding")
    if scale == 0:
        return 0.0  # existing Rust E8M0 cast treats exponent-zero/mantissa-zero as zero
    magnitude = (mantissa / 8.0 if exponent == 0 else 1.0 + mantissa / 8.0)
    value = math.ldexp(magnitude, exponent - 7 + scale - 127)
    return _bf16(-value if element & 128 else value)


def decode_matrix(hbm: bytes, region: Mapping[str, int]) -> list[list[float]]:
    rows, cols = region["rows"], region["cols"]
    return [[decode_element(
        hbm[region["element_base"] + n * region["element_row_stride"] + k],
        hbm[region["scale_base"] + n * region["scale_row_stride"] + k // BLOCK_SIZE],
    ) for k in range(cols)] for n in range(rows)]


def group_routes(routes: Sequence[Mapping[str, Any]], num_tokens: int,
                 expert_ids: set[int]) -> list[dict[str, Any]]:
    """Canonical grouping preserves every (token, slot, expert, weight) tuple."""
    seen = set()
    by_expert: dict[int, list[dict[str, Any]]] = {}
    for route in routes:
        for key in ("token", "slot", "expert"):
            if isinstance(route[key], bool) or not isinstance(route[key], int):
                raise ValueError(f"route {key} must be an integer")
        token, slot, expert = route["token"], route["slot"], route["expert"]
        weight = f32(route["weight"])
        if not 0 <= token < num_tokens or slot < 0 or expert not in expert_ids:
            raise ValueError("route has out-of-range token/slot or unknown expert")
        if not math.isfinite(weight):
            raise ValueError("route weight must be finite FP32")
        if (token, slot) in seen:
            raise ValueError("duplicate token/slot route")
        seen.add((token, slot))
        by_expert.setdefault(expert, []).append({
            "token": token, "slot": slot, "expert": expert, "weight": weight,
        })
    return [{"expert": expert, "routes": sorted(group, key=lambda r: (r["token"], r["slot"]))}
            for expert, group in sorted(by_expert.items())]


def _gemm_row(x: Sequence[float], weights: Sequence[Sequence[float]]) -> list[float]:
    out = []
    for row in weights:
        acc = 0.0
        for a, b in zip(x, row, strict=True):
            acc = f32(acc + f32(a * b))
        out.append(_bf16(acc))
    return out


def _swiglu(gate: float, up: float) -> float:
    try:
        exponential = f32(math.exp(-gate))
    except OverflowError:
        exponential = math.inf
    return _bf16(f32(f32(gate / f32(1.0 + exponential)) * up))


def numerical_reference(workload: Mapping[str, Any], hbm: bytes) -> dict[str, Any]:
    """Execute exported data in a scalar oracle independent of Rust scheduling."""
    inputs = [[bf16_value(x) for x in row] for row in workload["inputs_bf16"]]
    weights = {e["id"]: {stage: decode_matrix(hbm, e[stage]) for stage in ("gate", "up", "down")}
               for e in workload["experts"]}
    group_routes(workload["routes"], len(inputs), set(weights))
    cache: dict[tuple[int, int], list[float]] = {}

    def expert_output(token: int, expert: int) -> list[float]:
        key = token, expert
        if key not in cache:
            matrices = weights[expert]
            gate = _gemm_row(inputs[token], matrices["gate"])
            up = _gemm_row(inputs[token], matrices["up"])
            z = [_swiglu(g, u) for g, u in zip(gate, up, strict=True)]
            cache[key] = _gemm_row(z, matrices["down"])
        return cache[key]

    accum = [[0.0] * workload["input_dim"] for _ in inputs]
    contributions = []
    for route in sorted(workload["routes"], key=lambda r: (r["token"], r["slot"])):
        token = route["token"]
        values = expert_output(token, route["expert"])
        for d, value in enumerate(values):
            accum[token][d] = f32(accum[token][d] + f32(f32(route["weight"]) * value))
        contributions.append({**route, "output_bf16": [bf16_bits(x) for x in values]})
    shared = workload.get("shared_expert")
    if shared is not None:
        for token in range(len(inputs)):
            for d, value in enumerate(expert_output(token, shared["expert"])):
                accum[token][d] = f32(accum[token][d] + f32(f32(shared["weight"]) * value))
    output_bits = [[bf16_bits(x) for x in row] for row in accum]
    return {
        "schema_version": 1, "name": workload["name"],
        "semantics": "ascending-K FP32 mul/add; BF16 RNE GEMM and SwiGLU; stable token/slot FP32 combine; shared last; BF16 output",
        "output_bf16": output_bits,
        "output_f32": [[bf16_value(x) for x in row] for row in output_bits],
        "pre_round_output_f32": accum, "route_contributions": contributions,
    }


def export_workload(output_dir: str | Path, *, inputs: Any,
                    experts: Sequence[Mapping[str, Any]], routes: Sequence[Mapping[str, Any]],
                    name: str = "moe_normal", shared_expert: Mapping[str, Any] | None = None,
                    provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Export caller-supplied arrays and fixed route decisions, without executing a router.

    ``experts`` contains ``id, gate[E,D], up[E,D], down[D,E]``. An optional
    ``shared_expert={expert: id, weight: float}`` references an entry in experts.
    No biases, shared gating network, residual, attention or model loading is
    implied. Runtime grouping and the numerical reference retain duplicate
    token/expert assignments when their route slots differ.
    """
    import torch
    import plena_quant.mxfp.quantizer as quantizer
    import plena_quant.mxfp.utils as packer
    import plena_quant.common.minifloat as minifloat
    import plena_quant.common.hardware_utils as hardware_utils
    import plena_quant.common.utils as common_utils

    x = _matrix(inputs, "inputs")
    d = len(x[0])
    normalized = []
    ids: set[int] = set()
    hidden = None
    for expert in experts:
        expert_id = expert["id"]
        if isinstance(expert_id, bool) or not isinstance(expert_id, int) or expert_id < 0 or expert_id in ids:
            raise ValueError("expert IDs must be distinct nonnegative integers")
        ids.add(expert_id)
        matrices = {stage: _matrix(expert[stage], f"expert {expert_id} {stage}")
                    for stage in ("gate", "up", "down")}
        e = len(matrices["gate"])
        if hidden is None:
            hidden = e
        if (e != hidden or len(matrices["gate"][0]) != d or len(matrices["up"]) != e
                or len(matrices["up"][0]) != d or len(matrices["down"]) != d
                or len(matrices["down"][0]) != e):
            raise ValueError("all experts must have gate/up[E,D] and down[D,E] with a common E")
        normalized.append({"id": expert_id, **matrices})
    if hidden is None:
        raise ValueError("at least one expert is required")
    grouped = group_routes(routes, len(x), ids)
    normalized_routes = sorted([r for group in grouped for r in group["routes"]],
                               key=lambda r: (r["token"], r["slot"]))
    shared = None
    if shared_expert is not None:
        shared = {"expert": shared_expert["expert"], "weight": f32(shared_expert["weight"])}
        if (isinstance(shared["expert"], bool) or not isinstance(shared["expert"], int)
                or shared["expert"] not in ids or not math.isfinite(shared["weight"])):
            raise ValueError("invalid shared expert or weight")
    hbm = bytearray()

    def append_aligned(payload: bytes) -> int:
        hbm.extend(bytes((-len(hbm)) % 64))
        base = len(hbm)
        hbm.extend(payload)
        return base

    regions = []
    for expert in sorted(normalized, key=lambda e: e["id"]):
        views: dict[str, Any] = {"id": expert["id"]}
        for stage in ("gate", "up", "down"):
            elements, scales, shape = encode_matrix(expert[stage])
            views[stage] = {**shape, "element_base": append_aligned(elements),
                           "scale_base": append_aligned(scales)}
        regions.append(views)
    # Pad the final request granule so a real 64B load never leaves the image.
    hbm.extend(bytes((-len(hbm)) % 64))
    source_arrays = {"inputs_fp32": x, "experts_fp32": normalized,
                     "routes": normalized_routes, "shared_expert": shared}
    sources = [Path(__file__), *(Path(inspect.getfile(module)) for module in
                                  (quantizer, packer, minifloat, hardware_utils, common_utils))]
    metadata = {
        "evidence_scope": "caller_supplied_ready_inputs_and_routes; no runtime_router_or_full_model_claim",
        "weight_format": FORMAT, "weight_layout": "output_major_N_K",
        "quantizer_torch_version": torch.__version__,
        "scale_axis": "per_row_reduction_K", "block_size": BLOCK_SIZE,
        "hbm_sha256": _sha256(hbm), "source_arrays_sha256": _sha256(_json_bytes(source_arrays)),
        "sources": [{"path": str(p.resolve()), "sha256": _sha256(p.read_bytes())} for p in sources],
        "provenance": dict(provenance or {"scope": "caller_supplied_arrays"}),
    }
    workload = {
        "schema_version": 1, "name": name, "input_dim": d, "expert_hidden_dim": hidden,
        "hbm_file": "weights.bin",
        "inputs_bf16": [[bf16_bits(v) for v in row] for row in x],
        "routes": normalized_routes, "experts": regions, "shared_expert": shared,
        "grouped_routes": grouped, "metadata": metadata,
    }
    golden = numerical_reference(workload, bytes(hbm))
    golden["workload_sha256"] = _sha256(_json_bytes(workload))
    golden["hbm_sha256"] = metadata["hbm_sha256"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"workload": output / "workload.json", "hbm": output / "weights.bin",
             "golden": output / "golden.json", "source": output / "source_arrays.json"}
    paths["workload"].write_bytes(_json_bytes(workload))
    paths["hbm"].write_bytes(hbm)
    paths["golden"].write_bytes(_json_bytes(golden))
    paths["source"].write_bytes(_json_bytes(source_arrays))
    return {"workload": workload, "golden": golden, "paths": {k: str(v.resolve()) for k, v in paths.items()}}


def demo_arrays() -> dict[str, Any]:
    """Deterministic nonzero fixture: M=6,2,1,0; D=11/E=13 K and N tails."""
    d, e, t = 11, 13, 9
    inputs = [[((token * 7 + k * 3) % 19 - 9) / 16 for k in range(d)] for token in range(t)]
    experts = []
    for expert in range(4):
        matrices = {}
        for stage, rows, cols, offset in (("gate", e, d, 1), ("up", e, d, 3), ("down", d, e, 5)):
            matrices[stage] = [[((n * 7 + k * 11 + expert * 3 + offset) % 23 - 11)
                                * (2.0 ** (((n + k // 8 + expert) % 3) - 6))
                                for k in range(cols)] for n in range(rows)]
        experts.append({"id": expert, **matrices})
    routes = [{"token": token, "slot": 0, "expert": 0 if token < 6 else 1 if token < 8 else 2,
               "weight": 0.625 + (token % 3) * 0.125} for token in range(t)]
    return {"inputs": inputs, "experts": experts, "routes": routes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shared", action="store_true", help="also execute expert 3 for every token, weight 0.25")
    args = parser.parse_args()
    result = export_workload(args.output_dir, **demo_arrays(), name="synthetic_moe_normal_tails",
                             shared_expert={"expert": 3, "weight": 0.25} if args.shared else None,
                             provenance={"scope": "synthetic_shapes_inputs_routes_weights", "generator": "demo_arrays"})
    print(json.dumps(result["paths"], indent=2))


if __name__ == "__main__":
    main()
