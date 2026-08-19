#!/usr/bin/env python3
"""Validate X_STATE v2 and generate the Python wire-format constants."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "spec" / "x_state_v2.json"
DEFAULT_GOLDEN = ROOT / "spec" / "x_state_v2_golden.json"
DEFAULT_OUTPUT = ROOT / "aten" / "state" / "generated_contract.py"
TYPE_BYTES = {
    "u8": 1,
    "u16": 2,
    "u32": 4,
    "u64": 8,
    "bytes2": 2,
    "bytes3": 3,
    "bytes16": 16,
    "bytes56": 56,
}


def load_contract(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _validate_fields(fields: list[list[Any]], size: int, *, occupied: list[str | None] | None = None) -> None:
    slots = occupied if occupied is not None else [None] * size
    names: set[str] = set()
    for name, offset, field_type in fields:
        if name in names:
            raise ValueError(f"duplicate field {name}")
        names.add(name)
        width = TYPE_BYTES[field_type]
        if offset < 0 or offset + width > size:
            raise ValueError(f"field {name} lies outside the {size}-byte descriptor")
        for byte in range(offset, offset + width):
            if slots[byte] is not None:
                raise ValueError(f"fields {slots[byte]} and {name} overlap at byte {byte}")
            slots[byte] = name


def validate_contract(contract: dict[str, Any]) -> None:
    instruction = contract["instruction"]
    if instruction["name"] != "X_STATE" or instruction["format"] != "R_FUNCT":
        raise ValueError("X_STATE must use the PLENA R_FUNCT format")
    if not 0 <= instruction["opcode"] < 64:
        raise ValueError("X_STATE opcode does not fit six bits")
    instruction_bits = [None] * 32
    for name, offset, width in instruction["fields"]:
        for bit in range(offset, offset + width):
            if not 0 <= bit < 32 or instruction_bits[bit] is not None:
                raise ValueError(f"invalid or overlapping instruction field {name}")
            instruction_bits[bit] = name
    if any(value >= 16 for value in contract["subops"].values()):
        raise ValueError("X_STATE subop does not fit funct1")
    for enum_name in ("algorithms", "subops"):
        values = list(contract[enum_name].values())
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate value in {enum_name}")

    descriptor = contract["descriptor"]
    size = descriptor["size_bytes"]
    if size != 256 or size % descriptor["alignment_bytes"]:
        raise ValueError("X_STATE v2 descriptor must be 256 bytes and naturally aligned")
    common_occupied: list[str | None] = [None] * size
    _validate_fields(descriptor["common_fields"], size, occupied=common_occupied)
    if any(common_occupied[128:]):
        raise ValueError("common descriptor fields must fit in the first 128 bytes")
    for algorithm in contract["algorithms"]:
        payload = descriptor["payloads"].get(algorithm)
        if payload is None:
            raise ValueError(f"missing descriptor payload for {algorithm}")
        occupied = common_occupied.copy()
        _validate_fields(payload, size, occupied=occupied)
        if any(item is None for item in occupied):
            raise ValueError(f"{algorithm} descriptor leaves bytes unspecified")

    precision_values = [entry["value"] for entry in contract["precisions"].values()]
    if len(precision_values) != len(set(precision_values)):
        raise ValueError("duplicate precision value")
    completion = contract["completion_record"]
    _validate_fields(completion["fields"], completion["size_bytes"])


def validate_compiler_opcodes(contract: dict[str, Any], operation_path: Path) -> None:
    text = operation_path.read_text(encoding="utf-8")
    match = re.search(
        r"typedef\s+enum\s+logic\s*\[[^]]*OPCODE_WIDTH[^]]*\]\s*\{"
        r"(?P<body>.*?)\}\s*CUSTOM_ISA_OPCODE\s*;",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("CUSTOM_ISA_OPCODE enum not found")
    entries = [
        (name, int(value, 16))
        for name, _width, value in re.findall(
            r"(\w+)\s*=\s*(\d+)\s*'h\s*([0-9A-Fa-f]+)",
            match.group("body"),
        )
    ]
    by_name = dict(entries)
    by_value: dict[int, str] = {}
    for name, value in entries:
        if value in by_value:
            raise ValueError(
                f"compiler opcodes {by_value[value]} and {name} both use 0x{value:02X}"
            )
        by_value[value] = name
    for name, value in contract["opcode_reservations"].items():
        if name in by_name and by_name[name] != value:
            raise ValueError(
                f"compiler {name}=0x{by_name[name]:02X}, contract requires 0x{value:02X}"
            )
        occupant = by_value.get(value)
        if occupant is not None and occupant != name:
            raise ValueError(
                f"reserved opcode 0x{value:02X} for {name} is occupied by {occupant}"
            )
    if by_name.get("X_STATE") != contract["instruction"]["opcode"]:
        raise ValueError("doc/operation.svh does not declare the contract X_STATE opcode")


def validate_simulator_opcodes(
    contract: dict[str, Any], operation_path: Path, simulator_root: Path
) -> None:
    compiler_text = operation_path.read_text(encoding="utf-8")
    compiler_match = re.search(
        r"typedef\s+enum\s+logic\s*\[[^]]*OPCODE_WIDTH[^]]*\]\s*\{"
        r"(?P<body>.*?)\}\s*CUSTOM_ISA_OPCODE\s*;",
        compiler_text,
        flags=re.DOTALL,
    )
    if compiler_match is None:
        raise ValueError("CUSTOM_ISA_OPCODE enum not found")
    compiler = {
        name: int(value, 16)
        for name, _width, value in re.findall(
            r"(\w+)\s*=\s*(\d+)\s*'h\s*([0-9A-Fa-f]+)",
            compiler_match.group("body"),
        )
    }
    rust = (simulator_root / "transactional_emulator" / "src" / "op.rs").read_text(
        encoding="utf-8"
    )
    decode_match = re.search(
        r"pub\s+fn\s+decode\s*\([^)]*\)\s*->\s*Self\s*\{(?P<body>.*?)\n\s*\}\n\}",
        rust,
        flags=re.DOTALL,
    )
    if decode_match is None:
        raise ValueError("Simulator Opcode::decode body not found")
    simulator: dict[str, int] = {}
    arm_pattern = re.compile(
        r"^\s*(0x[0-9A-Fa-f]+)\s*=>\s*(?P<arm>.*?)"
        r"(?=^\s*(?:0x[0-9A-Fa-f]+|_)\s*=>)",
        flags=re.MULTILINE | re.DOTALL,
    )
    for arm in arm_pattern.finditer(decode_match.group("body")):
        variant = re.search(r"Self::([A-Za-z0-9_]+)", arm.group("arm"))
        if variant is None:
            continue
        name = variant.group(1)
        if name == "Invalid":
            name = "INVALID_OPCODE"
        simulator[name] = int(arm.group(1), 16)
    allowed_missing = set(contract["component_gaps"]["simulator_missing"])
    errors: list[str] = []
    for name, value in simulator.items():
        if name not in compiler:
            errors.append(f"Simulator-only opcode {name}=0x{value:02X}")
        elif compiler[name] != value:
            errors.append(
                f"{name}: Compiler=0x{compiler[name]:02X}, Simulator=0x{value:02X}"
            )
    missing = set(compiler) - set(simulator)
    unexpected_missing = missing - allowed_missing
    stale_gaps = allowed_missing - missing
    if unexpected_missing:
        errors.append(f"undeclared Simulator gaps: {sorted(unexpected_missing)}")
    if stale_gaps:
        errors.append(f"stale Simulator gap declarations: {sorted(stale_gaps)}")
    if errors:
        raise ValueError("Compiler/Simulator opcode maps disagree:\n  " + "\n  ".join(errors))


def _field_map(fields: list[list[Any]]) -> dict[str, tuple[int, str]]:
    return {name: (offset, field_type) for name, offset, field_type in fields}


def render_python(contract: dict[str, Any], digest: str) -> str:
    descriptor = contract["descriptor"]
    precision_values = {name: entry["value"] for name, entry in contract["precisions"].items()}
    precision_bytes = {entry["value"]: entry["element_bytes"] for entry in contract["precisions"].values()}
    payloads = {name: _field_map(fields) for name, fields in descriptor["payloads"].items()}
    completion = contract["completion_record"]
    return (
        '"""Generated from spec/x_state_v2.json; do not edit by hand."""\n\n'
        f'CONTRACT_SHA256 = "{digest}"\n'
        f"X_STATE_OPCODE = {contract['instruction']['opcode']}\n"
        f"STATE_ALGORITHMS = {contract['algorithms']!r}\n"
        f"STATE_SUBOPS = {contract['subops']!r}\n"
        f"STATE_PRECISIONS = {precision_values!r}\n"
        f"STATE_PRECISION_BYTES = {precision_bytes!r}\n"
        f"STATE_FLAG_BITS = {contract['flags']!r}\n"
        f"STATE_DESCRIPTOR_MAGIC = {descriptor['magic']}\n"
        f"STATE_DESCRIPTOR_VERSION = {descriptor['version']}\n"
        f"STATE_DESCRIPTOR_SIZE = {descriptor['size_bytes']}\n"
        f"STATE_DESCRIPTOR_ALIGNMENT = {descriptor['alignment_bytes']}\n"
        f"STATE_STREAMING_SRAM_OFFSET = {descriptor['streaming_sram_offset']}\n"
        f"STATE_COMMON_FIELDS = {_field_map(descriptor['common_fields'])!r}\n"
        f"STATE_PAYLOAD_FIELDS = {payloads!r}\n"
        f"STATE_COMPLETION_SIZE = {completion['size_bytes']}\n"
        f"STATE_COMPLETION_ALIGNMENT = {completion['alignment_bytes']}\n"
        f"STATE_COMPLETION_FIELDS = {_field_map(completion['fields'])!r}\n"
        f"STATE_STATUS = {completion['status_values']!r}\n"
        f"STATE_NO_EVENT = {contract['events']['no_event']}\n"
    )


def _rust_variant(name: str) -> str:
    return "".join(part.capitalize() for part in name.lower().split("_"))


def _render_rust_enum(
    enum_name: str,
    values: dict[str, int],
    *,
    repr_type: str,
) -> str:
    variants = "\n".join(
        f"    {_rust_variant(name)} = {value}," for name, value in values.items()
    )
    conversions = "\n".join(
        f"            {value} => Ok(Self::{_rust_variant(name)}),"
        for name, value in values.items()
    )
    return f"""#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr({repr_type})]
pub enum {enum_name} {{
{variants}
}}

impl TryFrom<{repr_type}> for {enum_name} {{
    type Error = ();

    fn try_from(value: {repr_type}) -> Result<Self, Self::Error> {{
        match value {{
{conversions}
            _ => Err(()),
        }}
    }}
}}
"""


def _render_rust_offsets(module: str, fields: list[list[Any]]) -> str:
    constants = "\n".join(
        f"    pub const {name.upper()}: usize = {offset};" for name, offset, _ in fields
    )
    return f"pub mod {module} {{\n{constants}\n}}\n"


def render_rust(contract: dict[str, Any], digest: str) -> str:
    descriptor = contract["descriptor"]
    completion = contract["completion_record"]
    precision_values = {
        name: entry["value"] for name, entry in contract["precisions"].items()
    }
    status_values = completion["status_values"]
    flag_constants = "\n".join(
        f"pub const FLAG_{name}: u32 = 1 << {bit};"
        for name, bit in contract["flags"].items()
    )
    precision_bytes = "\n".join(
        f"            Self::{_rust_variant(name)} => {entry['element_bytes']},"
        for name, entry in contract["precisions"].items()
    )
    instruction_fields = "\n".join(
        f"    pub const {name.upper()}_LSB: u32 = {offset};\n"
        f"    pub const {name.upper()}_WIDTH: u32 = {width};"
        for name, offset, width in contract["instruction"]["fields"]
    )
    payload_offsets = "\n".join(
        _render_rust_offsets(name.lower(), fields).rstrip()
        for name, fields in descriptor["payloads"].items()
    )
    rendered = f"""// Generated from spec/x_state_v2.json; do not edit by hand.
#![allow(dead_code)]

pub const CONTRACT_SHA256: &str =
    \"{digest}\";
pub const X_STATE_OPCODE: u8 = {contract['instruction']['opcode']};
pub const DESCRIPTOR_MAGIC: u32 = {descriptor['magic']};
pub const DESCRIPTOR_VERSION: u16 = {descriptor['version']};
pub const DESCRIPTOR_SIZE: usize = {descriptor['size_bytes']};
pub const DESCRIPTOR_ALIGNMENT: u64 = {descriptor['alignment_bytes']};
pub const STREAMING_SRAM_OFFSET: u32 = {descriptor['streaming_sram_offset']};
pub const COMPLETION_SIZE: usize = {completion['size_bytes']};
pub const COMPLETION_ALIGNMENT: u64 = {completion['alignment_bytes']};
pub const NO_EVENT: u32 = {contract['events']['no_event']};

{flag_constants}

pub mod instruction {{
{instruction_fields}
}}

{_render_rust_offsets('common', descriptor['common_fields'])}
{payload_offsets}

{_render_rust_offsets('completion', completion['fields'])}
{_render_rust_enum('StateAlgorithm', contract['algorithms'], repr_type='u8')}
{_render_rust_enum('StateSubop', contract['subops'], repr_type='u8')}
{_render_rust_enum('StatePrecision', precision_values, repr_type='u8')}
impl StatePrecision {{
    pub const fn element_bytes(self) -> usize {{
        match self {{
{precision_bytes}
        }}
    }}
}}

{_render_rust_enum('StateStatus', status_values, repr_type='u32')}
"""
    return rendered.rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--simulator-root", type=Path)
    parser.add_argument(
        "--sync-simulator",
        action="store_true",
        help="write the generated Rust contract into --simulator-root",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.sync_simulator and args.simulator_root is None:
        parser.error("--sync-simulator requires --simulator-root")
    if args.sync_simulator and args.check:
        parser.error("--sync-simulator and --check are mutually exclusive")
    contract = load_contract(args.spec)
    validate_contract(contract)
    validate_compiler_opcodes(contract, ROOT / "doc" / "operation.svh")
    if args.simulator_root is not None:
        validate_simulator_opcodes(
            contract,
            ROOT / "doc" / "operation.svh",
            args.simulator_root.resolve(),
        )
    digest = hashlib.sha256(args.spec.read_bytes()).hexdigest()
    rendered_python = render_python(contract, digest)
    rendered_rust = render_rust(contract, digest)
    rust_output = None
    simulator_golden = None
    if args.simulator_root is not None:
        rust_output = (
            args.simulator_root.resolve()
            / "transactional_emulator"
            / "src"
            / "state_engine"
            / "generated_contract.rs"
        )
        simulator_golden = (
            args.simulator_root.resolve()
            / "transactional_emulator"
            / "testdata"
            / "x_state_v2_golden.json"
        )
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != rendered_python
        ):
            raise SystemExit(f"generated contract is stale: run {Path(__file__).name}")
        if rust_output is not None and (
            not rust_output.exists()
            or rust_output.read_text(encoding="utf-8") != rendered_rust
        ):
            raise SystemExit(
                "generated Simulator contract is stale: run "
                f"{Path(__file__).name} --simulator-root {args.simulator_root} "
                "--sync-simulator"
            )
        if simulator_golden is not None and (
            not simulator_golden.exists()
            or simulator_golden.read_bytes() != DEFAULT_GOLDEN.read_bytes()
        ):
            raise SystemExit(
                "Simulator descriptor golden bytes are stale: run "
                f"{Path(__file__).name} --simulator-root {args.simulator_root} "
                "--sync-simulator"
            )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered_python, encoding="utf-8")
    if args.sync_simulator and rust_output is not None:
        rust_output.parent.mkdir(parents=True, exist_ok=True)
        rust_output.write_text(rendered_rust, encoding="utf-8")
        assert simulator_golden is not None
        simulator_golden.parent.mkdir(parents=True, exist_ok=True)
        simulator_golden.write_bytes(DEFAULT_GOLDEN.read_bytes())


if __name__ == "__main__":
    main()
