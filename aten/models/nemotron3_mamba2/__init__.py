"""Nemotron 3 Mamba-2 executable reference model and command lowering."""

from .extract import (
    ExtractedNemotronMamba2Layer,
    extract_nemotron3_mamba2_config,
    extract_nemotron3_mamba2_layer,
)
from .lowering import (
    MambaCommandCompiler,
    MambaCommandSpec,
    MambaProgram,
    MambaRegisterAssignment,
    MambaSubop,
    MambaTensorBindings,
    allocate_mamba_tensor_bindings,
    materialize_mamba_io_images,
    materialize_mamba_weight_images,
)
from .memory import (
    ByteAddressArena,
    MambaPersistentStateAllocator,
    MambaStateAllocation,
    MemoryRegion,
)
from .reference import (
    Mamba2Config,
    Mamba2Result,
    Mamba2State,
    Mamba2Weights,
    PrecisionPolicy,
    allocate_state,
    mamba2_prefill,
    mamba2_step,
    selective_scan_chunked,
    selective_scan_sequential,
)

__all__ = [
    "ByteAddressArena",
    "ExtractedNemotronMamba2Layer",
    "Mamba2Config",
    "Mamba2Result",
    "Mamba2State",
    "Mamba2Weights",
    "MambaCommandCompiler",
    "MambaCommandSpec",
    "MambaPersistentStateAllocator",
    "MambaProgram",
    "MambaRegisterAssignment",
    "MambaStateAllocation",
    "MambaSubop",
    "MambaTensorBindings",
    "MemoryRegion",
    "PrecisionPolicy",
    "allocate_state",
    "allocate_mamba_tensor_bindings",
    "extract_nemotron3_mamba2_config",
    "extract_nemotron3_mamba2_layer",
    "mamba2_prefill",
    "mamba2_step",
    "materialize_mamba_io_images",
    "materialize_mamba_weight_images",
    "selective_scan_chunked",
    "selective_scan_sequential",
]
