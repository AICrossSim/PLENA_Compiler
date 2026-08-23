"""Build the flat memory preload images the transactional emulator expects.

The emulator takes ``--hbm``/``--fpsram`` as raw byte files and copies them into
HBM/FP-SRAM starting at offset 0, so an address the program touches has to be
inside the allocation. The default KDA descriptor bases span 24.5 GiB, which is
why the physical program has never been executed; ``KdaScheduleConfig
.hbm_arena_base`` packs the same regions into a bounded arena instead.

Only the descriptor image has to be *written*: everything else the program reads
-- conv weights, A_log, dt_bias, the initial recurrent and convolution state --
is read back as whatever HBM already holds, and zero is a legal value for all of
them. Timing depends on the bytes moved, not their values, so a zero-filled
arena measures the same cycles as real weights. That keeps the file at tens of
kilobytes instead of hundreds of megabytes; ``hbm_size_bytes`` reports how large
the emulator's allocation still has to be.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path


#: `ScalarSram::new` allocates exactly this many entries for both scalar files
#: and `load_*` copies into `self.x[..vals.len()]`, so an oversized preload
#: panics with a slice-range error rather than being truncated. The emulator does
#: not expose the size, so it is mirrored here and enforced below.
SCALAR_SRAM_ENTRIES = 1024
FPSRAM_BYTES_PER_ENTRY = 2  # f16/bf16
INTSRAM_BYTES_PER_ENTRY = 4  # u32


_REPO_ROOT = Path(__file__).resolve().parents[2]


def assemble_words(assembly: str) -> list[int]:
    """Assemble emitted text into machine words.

    The emulator's ``--opcode`` reads hex tokens, not assembly -- feeding it the
    ``.asm`` file fails on the first comment. Lowering only produces text, so the
    assembler runs here rather than being reimplemented by every caller.
    """
    # Imported lazily: the assembler pulls in the ISA tables, and this module is
    # also used by callers that only need the byte images.
    from assembler.assembly_to_binary import AssemblyToBinary
    from assembler.parser import parse_asm_file

    with tempfile.NamedTemporaryFile("w", suffix=".asm", delete=False) as handle:
        handle.write(assembly)
        path = Path(handle.name)
    try:
        converter = AssemblyToBinary(
            str(_REPO_ROOT / "doc" / "operation.svh"),
            str(_REPO_ROOT / "doc" / "configuration.svh"),
        )
        return [
            converter._convert_to_binary(instruction)
            for instruction in parse_asm_file(str(path))
        ]
    finally:
        path.unlink(missing_ok=True)


def render_mem_image(words: list[int]) -> str:
    for index, word in enumerate(words):
        if not 0 <= word <= 0xFFFF_FFFF:
            raise ValueError(f"instruction {index} does not fit a 32-bit word: {word}")
    return "".join(f"0x{word:08X}\n" for word in words)


@dataclass(frozen=True)
class MemoryImages:
    hbm: bytes
    fpsram: bytes
    intsram: bytes
    hbm_size_bytes: int
    descriptor_base: int
    descriptor_bytes: int
    layout_descriptor_base: int | None
    layout_descriptor_bytes: int
    arena_base: int | None
    arena_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "contract": "plena.state_memory_image/v2",
            "hbm_file_bytes": len(self.hbm),
            "hbm_size_bytes": self.hbm_size_bytes,
            "hbm_size_mib": self.hbm_size_bytes / (1024 * 1024),
            "descriptor_base": self.descriptor_base,
            "descriptor_bytes": self.descriptor_bytes,
            "layout_descriptor_base": self.layout_descriptor_base,
            "layout_descriptor_bytes": self.layout_descriptor_bytes,
            "arena_base": self.arena_base,
            "arena_bytes": self.arena_bytes,
            "fpsram_file_bytes": len(self.fpsram),
            "intsram_file_bytes": len(self.intsram),
            "zero_filled": (
                "conv weights, A_log, dt_bias, D, and the initial state are read "
                "as zero; byte counts, and therefore cycles, are unaffected"
            ),
        }

    def write(self, directory: Path, *, stem: str) -> dict[str, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "hbm": directory / f"{stem}_hbm.bin",
            "fpsram": directory / f"{stem}_fpsram.bin",
            "intsram": directory / f"{stem}_intsram.bin",
            "manifest": directory / f"{stem}_memory.json",
        }
        paths["hbm"].write_bytes(self.hbm)
        paths["fpsram"].write_bytes(self.fpsram)
        paths["intsram"].write_bytes(self.intsram)
        paths["manifest"].write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return paths


def build_memory_images(
    *,
    descriptor_image: bytes,
    descriptor_base: int,
    layout_descriptor_image: bytes = b"",
    layout_descriptor_base: int | None = None,
    arena_base: int | None,
    arena_bytes: int,
    fpsram_entries: int = SCALAR_SRAM_ENTRIES,
    intsram_entries: int = SCALAR_SRAM_ENTRIES,
) -> MemoryImages:
    """Place the descriptor image and size the allocation around the arena."""
    if descriptor_base < 0 or descriptor_base % 64:
        raise ValueError("descriptor_base must be non-negative and 64-byte aligned")
    if not descriptor_image:
        raise ValueError("descriptor image is empty; nothing to execute")
    if arena_bytes <= 0:
        raise ValueError("arena_bytes must be positive")
    descriptor_end = descriptor_base + len(descriptor_image)
    if layout_descriptor_image:
        if layout_descriptor_base is None:
            raise ValueError(
                "layout_descriptor_base is required when the layout image is present"
            )
        if layout_descriptor_base < 0 or layout_descriptor_base % 64:
            raise ValueError(
                "layout_descriptor_base must be non-negative and 64-byte aligned"
            )
        layout_end = layout_descriptor_base + len(layout_descriptor_image)
        if not (
            layout_end <= descriptor_base
            or descriptor_end <= layout_descriptor_base
        ):
            raise ValueError("state and layout descriptor images overlap")
    else:
        if layout_descriptor_base is not None:
            raise ValueError(
                "layout_descriptor_base must be omitted when the layout image is empty"
            )
        layout_end = 0
    image_end = max(descriptor_end, layout_end)
    if arena_base is not None and image_end > arena_base:
        raise ValueError(
            f"descriptor images end at {image_end}, which overlaps the "
            f"HBM arena starting at {arena_base}; lower --descriptor-base or raise "
            "hbm_arena_base"
        )
    for name, entries in (("fpsram", fpsram_entries), ("intsram", intsram_entries)):
        if not 0 < entries <= SCALAR_SRAM_ENTRIES:
            raise ValueError(
                f"{name}_entries must be in (0, {SCALAR_SRAM_ENTRIES}]; the emulator "
                f"copies the preload into a fixed {SCALAR_SRAM_ENTRIES}-entry file and "
                "panics on anything longer"
            )
    hbm = bytearray(image_end)
    hbm[descriptor_base:descriptor_end] = descriptor_image
    if layout_descriptor_image:
        assert layout_descriptor_base is not None
        hbm[layout_descriptor_base:layout_end] = layout_descriptor_image
    # Round the allocation up to a 64-byte line, matching the emulator's
    # `Vec<[u8; 64]>` backing store.
    hbm_size = ((max(image_end, arena_bytes) + 63) // 64) * 64
    return MemoryImages(
        hbm=bytes(hbm),
        fpsram=bytes(fpsram_entries * FPSRAM_BYTES_PER_ENTRY),
        intsram=bytes(intsram_entries * INTSRAM_BYTES_PER_ENTRY),
        hbm_size_bytes=hbm_size,
        descriptor_base=descriptor_base,
        descriptor_bytes=len(descriptor_image),
        layout_descriptor_base=layout_descriptor_base,
        layout_descriptor_bytes=len(layout_descriptor_image),
        arena_base=arena_base,
        arena_bytes=arena_bytes,
    )
