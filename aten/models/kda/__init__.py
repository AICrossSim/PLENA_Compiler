"""Kimi K3 KDA (Kimi Delta Attention) CPU reference.

Lives here, not in the Simulator, for the same reason
``aten/models/mamba2/reference.py`` does: the dependency direction across the
two repositories is Simulator -> Compiler and never the reverse, so a reference
placed here is importable from both the compiler's lowering tests and the
emulator's testbench, while one placed in the Simulator is not.

**Re-exports are lazy on purpose.** ``aten/plena/compiler.py`` imports the KDA
lowering at module scope, and the lowering needs ``KdaShape``. Importing a
submodule runs this ``__init__`` first, so an eager ``from .reference import ...``
here drags ``torch`` into every ``compiler.aten.plena.*`` import -- which breaks
the ``moe-stage-guard`` CI job, whose whole point is running the source-only
guards with just pytest and pyyaml installed. ``KdaShape`` therefore lives in the
torch-free ``shape`` module, and everything that genuinely needs torch resolves
through ``__getattr__`` only when someone asks for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Torch-free, so it can be imported eagerly and by the lowering.
from compiler.aten.models.kda.shape import KdaShape

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from compiler.aten.models.kda.reference import (
        KdaConvWeights,
        KdaState,
        KdaRecurrentState,
        activate_log_decay,
        kda_recurrent_sequence,
        kda_state_engine_prefill,
        kda_state_engine_step,
        kda_step,
    )
    from compiler.aten.models.kda.state_precision import (
        StateStorage,
        quantize_state,
        storage_bytes,
    )

_REFERENCE_EXPORTS = frozenset(
    {
        "KdaConvWeights",
        "KdaState",
        "KdaRecurrentState",
        "activate_log_decay",
        "kda_recurrent_sequence",
        "kda_state_engine_prefill",
        "kda_state_engine_step",
        "kda_step",
    }
)
_PRECISION_EXPORTS = frozenset({"StateStorage", "quantize_state", "storage_bytes"})

__all__ = ["KdaShape", *sorted(_REFERENCE_EXPORTS | _PRECISION_EXPORTS)]


def __getattr__(name: str):
    """Resolve the torch-backed exports on first use (PEP 562)."""
    if name in _REFERENCE_EXPORTS:
        from compiler.aten.models.kda import reference

        return getattr(reference, name)
    if name in _PRECISION_EXPORTS:
        from compiler.aten.models.kda import state_precision

        return getattr(state_precision, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
