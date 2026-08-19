"""Shared contracts for whole-model physical programs."""

from __future__ import annotations

from dataclasses import dataclass


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
        }


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
