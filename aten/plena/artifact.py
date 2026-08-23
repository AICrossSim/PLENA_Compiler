"""Write a bounded full-model decode program with unresolved HBM parameters."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from assembler.assembly_to_binary import AssemblyToBinary
from compiler.aten.plena.full_model import (
    FullModelProgram,
    validate_symbolic_hbm_bindings,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    digest = hashlib.sha256()
    for line in io.StringIO(text):
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _binding_counts(program: FullModelProgram) -> dict[str, int]:
    counts: dict[str, int] = {}
    for binding in program.symbolic_hbm_bindings:
        key = f"{binding.source}:{binding.layout}:{binding.storage_format}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_symbolic_decode_artifact(
    program: FullModelProgram,
    output_dir: Path,
    *,
    stem: str,
    artifact_contract: str,
    scope: str,
    max_machine_code_mib: float,
    claims: dict[str, bool],
    extra_summary: dict[str, object] | None = None,
    write_assembly: bool = False,
) -> dict[str, object]:
    """Assemble one program and write every non-checkpoint execution image."""
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_symbolic_hbm_bindings(program)

    raw_machine_code_bytes = program.instruction_count * 4
    budget_bytes = int(max_machine_code_mib * 1024 * 1024)
    if raw_machine_code_bytes > budget_bytes:
        raise ValueError(
            f"{program.model} machine code needs "
            f"{raw_machine_code_bytes / (1024 * 1024):.2f} MiB, above the "
            f"{max_machine_code_mib:.2f} MiB budget"
        )

    machine_code_path = output_dir / f"{stem}.mem"
    assembler = AssemblyToBinary(
        str(REPO_ROOT / "doc" / "operation.svh"),
        str(REPO_ROOT / "doc" / "configuration.svh"),
    )
    encoded_count = assembler.generate_binary_streaming_lines(
        io.StringIO(program.assembly),
        str(machine_code_path),
    )
    if encoded_count != program.instruction_count:
        machine_code_path.unlink(missing_ok=True)
        raise AssertionError(
            f"assembler encoded {encoded_count} instructions, builder counted "
            f"{program.instruction_count}"
        )

    descriptor_path = output_dir / f"{stem}.descriptors.bin"
    layout_path = output_dir / f"{stem}.layouts.bin"
    fpram_path = output_dir / f"{stem}.fpram.json"
    weights_path = output_dir / f"{stem}.symbolic_hbm.json"
    summary_path = output_dir / f"{stem}.summary.json"
    descriptor_path.write_bytes(program.descriptor_image)
    layout_path.write_bytes(program.layout_descriptor_image)
    fpram_path.write_text(json.dumps(list(program.fpram_preload), indent=2) + "\n")
    weights_path.write_text(
        json.dumps(program.symbolic_hbm_manifest(), indent=2) + "\n"
    )
    if write_assembly:
        (output_dir / f"{stem}.asm").write_text(program.assembly)

    summary: dict[str, object] = {
        **program.to_dict(),
        "artifact_contract": artifact_contract,
        "scope": scope,
        "machine_code_budget_mib": max_machine_code_mib,
        "machine_code_raw_bytes": raw_machine_code_bytes,
        "machine_code_raw_mib": raw_machine_code_bytes / (1024 * 1024),
        "machine_code_mem_file_bytes": machine_code_path.stat().st_size,
        "symbolic_binding_counts": _binding_counts(program),
        "files": {
            "machine_code": machine_code_path.name,
            "descriptors": descriptor_path.name,
            "layouts": layout_path.name,
            "fpram": fpram_path.name,
            "symbolic_hbm": weights_path.name,
        },
        "sha256": {
            "assembly": _sha256_text(program.assembly),
            "machine_code": _sha256_file(machine_code_path),
            "descriptors": _sha256_file(descriptor_path),
            "layouts": _sha256_file(layout_path),
            "fpram": _sha256_file(fpram_path),
            "symbolic_hbm": _sha256_file(weights_path),
        },
        "claims": claims,
    }
    if extra_summary:
        summary.update(extra_summary)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


__all__ = ["write_symbolic_decode_artifact"]
