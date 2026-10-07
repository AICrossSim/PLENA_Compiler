from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).parents[2]
FREEZE = json.loads((ROOT / "spec" / "hybrid_isa_freeze_v1.json").read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiler_opcodes() -> dict[str, int]:
    text = (ROOT / "doc" / "operation.svh").read_text()
    match = re.search(
        r"typedef\s+enum\s+logic\s*\[[^]]*OPCODE_WIDTH[^]]*\]\s*\{"
        r"(?P<body>.*?)\}\s*CUSTOM_ISA_OPCODE\s*;",
        text,
        flags=re.DOTALL,
    )
    assert match is not None
    return {
        name: int(value, 16)
        for name, _width, value in re.findall(
            r"(\w+)\s*=\s*(\d+)\s*'h\s*([0-9A-Fa-f]+)", match.group("body")
        )
    }


def test_frozen_contract_hashes_and_descriptor_geometry() -> None:
    for contract in FREEZE["contracts"].values():
        spec_path = ROOT / contract["spec"]
        golden_path = ROOT / contract["golden"]
        assert _sha256(spec_path) == contract["sha256"]
        assert _sha256(golden_path) == contract["golden_sha256"]
        spec = json.loads(spec_path.read_text())
        assert spec["descriptor"]["size_bytes"] == contract["descriptor_size_bytes"]
        assert (
            spec["descriptor"]["alignment_bytes"]
            == contract["descriptor_alignment_bytes"]
        )


def test_implemented_reserved_and_free_opcodes_do_not_overlap() -> None:
    implemented = FREEZE["implemented_opcodes"]
    reserved = FREEZE["reserved_not_implemented"]
    free = set(FREEZE["unallocated_opcodes"])
    assert not (set(implemented.values()) & set(reserved.values()))
    assert not (set(implemented.values()) & free)
    assert not (set(reserved.values()) & free)

    compiler = _compiler_opcodes()
    assert {name: compiler[name] for name in implemented} == implemented
    assert all(name not in compiler for name in reserved)
    occupied = set(compiler.values())
    assert not (set(reserved.values()) & occupied)
    assert not (free & occupied)


def test_subcontracts_match_the_frozen_opcode_and_status_map() -> None:
    x_state = json.loads((ROOT / FREEZE["contracts"]["x_state"]["spec"]).read_text())
    layout = json.loads((ROOT / FREEZE["contracts"]["l_scatter_m"]["spec"]).read_text())
    assert x_state["instruction"]["opcode"] == FREEZE["implemented_opcodes"]["X_STATE"]
    assert (
        layout["instruction"]["opcode"] == FREEZE["implemented_opcodes"]["L_SCATTER_M"]
    )
    assert x_state["opcode_reservations"] == FREEZE["reserved_not_implemented"] | {
        "X_STATE": FREEZE["implemented_opcodes"]["X_STATE"]
    }
    assert (
        x_state["completion_record"]["status_values"]
        == FREEZE["completion_status_values"]
    )


def test_freeze_does_not_claim_unwritten_rtl_or_route_execution() -> None:
    assert FREEZE["scope"]["rtl"] == "not_started"
    assert FREEZE["scope"]["route_extension"] == "opcode_space_reserved_only"
