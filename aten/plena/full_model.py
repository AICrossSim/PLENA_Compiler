"""Shared contracts for whole-model physical programs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolicHbmBinding:
    """One unresolved model parameter range referenced by machine code."""

    name: str
    hbm_addr: int
    byte_size: int
    logical_shape: tuple[int, ...]
    physical_shape: tuple[int, ...]
    storage_format: str
    layout: str = "row_major"
    source: str = "checkpoint_parameter"
    layer_id: int | None = None
    metadata: tuple[tuple[str, int | float | str], ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("symbolic HBM binding needs a name")
        if self.hbm_addr < 0 or self.byte_size <= 0:
            raise ValueError(
                f"invalid HBM range for {self.name}: address={self.hbm_addr}, size={self.byte_size}"
            )
        if not self.logical_shape or any(dim <= 0 for dim in self.logical_shape):
            raise ValueError(
                f"invalid logical shape for {self.name}: {self.logical_shape}"
            )
        if not self.physical_shape or any(dim <= 0 for dim in self.physical_shape):
            raise ValueError(
                f"invalid physical shape for {self.name}: {self.physical_shape}"
            )

    @property
    def hbm_end(self) -> int:
        return self.hbm_addr + self.byte_size

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "layer_id": self.layer_id,
            "hbm_addr": self.hbm_addr,
            "hbm_end": self.hbm_end,
            "byte_size": self.byte_size,
            "logical_shape": list(self.logical_shape),
            "physical_shape": list(self.physical_shape),
            "storage_format": self.storage_format,
            "layout": self.layout,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FullModelProgram:
    """A whole-model physical program plus the images needed to execute it."""

    model: str
    phase: str
    layer_counts: dict[str, int]
    assembly: str
    instruction_count: int
    descriptor_base: int
    descriptor_image: bytes
    stage_instruction_counts: dict[str, int]
    layout_descriptor_base: int = 0
    layout_descriptor_image: bytes = b""
    fpram_preload: tuple[float, ...] = ()
    output_vram_addr: int | None = None
    hbm_size: int | None = None
    symbolic_hbm_bindings: tuple[SymbolicHbmBinding, ...] = ()

    def to_dict(self) -> dict[str, object]:
        machine_code_bytes = self.instruction_count * 4
        return {
            "contract": "plena.full_model_program/v1",
            "model": self.model,
            "phase": self.phase,
            "layer_counts": self.layer_counts,
            "instruction_count": self.instruction_count,
            "machine_code_bytes": machine_code_bytes,
            "machine_code_mib": machine_code_bytes / (1024 * 1024),
            "assembly_text_bytes": len(self.assembly.encode("utf-8")),
            "descriptor_base": self.descriptor_base,
            "descriptor_count": len(self.descriptor_image) // 256,
            "layout_descriptor_base": self.layout_descriptor_base,
            "layout_descriptor_count": len(self.layout_descriptor_image) // 256,
            "stage_instruction_counts": self.stage_instruction_counts,
            "output_vram_addr": self.output_vram_addr,
            "hbm_size": self.hbm_size,
            "symbolic_hbm_binding_count": len(self.symbolic_hbm_bindings),
            "weights_bound": False,
        }

    def symbolic_hbm_manifest(self) -> dict[str, object]:
        """Return the address contract a checkpoint packer must satisfy."""
        return {
            "contract": "plena.symbolic_hbm_manifest/v1",
            "model": self.model,
            "phase": self.phase,
            "address_unit": "byte",
            "weights_bound": False,
            "binding_count": len(self.symbolic_hbm_bindings),
            "bindings": [binding.to_dict() for binding in self.symbolic_hbm_bindings],
        }


def validate_symbolic_hbm_bindings(program: FullModelProgram) -> None:
    """Reject an ambiguous or out-of-range checkpoint address contract."""
    by_name: dict[str, int] = {}
    for binding in program.symbolic_hbm_bindings:
        by_name[binding.name] = by_name.get(binding.name, 0) + 1
    duplicates = sorted(name for name, count in by_name.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate symbolic HBM bindings: {duplicates[:8]}")
    if program.hbm_size is None:
        raise ValueError("full model program does not declare its HBM span")
    for binding in program.symbolic_hbm_bindings:
        if binding.hbm_end > program.hbm_size:
            raise ValueError(
                f"binding {binding.name} ends at {binding.hbm_end}, "
                f"beyond HBM span {program.hbm_size}"
            )

    ordered = sorted(
        program.symbolic_hbm_bindings,
        key=lambda binding: (binding.hbm_addr, binding.hbm_end),
    )
    for current, following in zip(ordered, ordered[1:]):
        if current.hbm_end > following.hbm_addr:
            raise ValueError(
                "symbolic HBM ranges overlap: "
                f"{current.name}[{current.hbm_addr},{current.hbm_end}) and "
                f"{following.name}[{following.hbm_addr},{following.hbm_end})"
            )


def assert_registers_are_free(compiler: object, where: str) -> None:
    """Reject a layer boundary that leaves compiler-owned registers live."""
    allocator = compiler.register_allocator  # type: ignore[attr-defined]
    leaked = {
        "gp": list(allocator.used_gp),
        "addr": list(allocator.used_addr),
        "fp": list(allocator.used_fp),
    }
    if any(leaked.values()):
        raise AssertionError(
            f"{where} left registers allocated: {leaked}. The next layer treats the "
            "whole register file as scratch, so a leak here silently corrupts it."
        )
