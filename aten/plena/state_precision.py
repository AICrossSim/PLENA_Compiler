"""Architectural BF16 contract for recurrent state transfers.

The profiled GPU implementations retain recurrent state in FP32.  PLENA uses
BF16 deliberately as a mixed-precision design point; callers must opt into this
contract explicitly instead of inheriting the attention KV format.
"""

from __future__ import annotations


STATE_PRECISION_SELECTOR = 2
STATE_ELEMENT_BYTES = 2


def require_bf16_state(settings: dict | None) -> None:
    """Validate the PLENA BF16 state contract against a PRECISION table."""

    if settings is None:
        raise ValueError("BF16 state validation needs the active PRECISION table")
    node = settings.get("HBM_STATE_TYPE", {})
    data = node.get("DATA_TYPE", {}) if isinstance(node, dict) else {}
    actual = (
        node.get("format") if isinstance(node, dict) else None,
        data.get("type") if isinstance(data, dict) else None,
        data.get("sign") if isinstance(data, dict) else None,
        data.get("exponent") if isinstance(data, dict) else None,
        data.get("mantissa") if isinstance(data, dict) else None,
    )
    if actual != ("Plain", "Fp", True, 8, 7):
        raise ValueError(
            "PLENA recurrent state requires HBM_STATE_TYPE=Plain BF16, "
            f"got {node}"
        )


__all__ = [
    "STATE_ELEMENT_BYTES",
    "STATE_PRECISION_SELECTOR",
    "require_bf16_state",
]
