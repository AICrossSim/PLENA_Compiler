"""Single source of truth for hardware sizes — reads ``plena_settings.toml``.

The simulator config (``PLENA_Simulator/plena_settings.toml``) already
holds the hardware geometry: MLEN / HLEN / BLEN / VLEN, separately for
the ``analytic`` and ``behavior`` modes, with ``[MODE].active`` picking
one. Previously the compiler hard-coded its own copies (PlenaTarget
defaults, CLI defaults, ``cluster_guard.MLEN``, ``split._DEFAULT_LANE``,
per-kernel ``MLEN`` / ``hlen`` args) — so changing target geometry meant
editing several files by hand and keeping them in sync.

This module reads the toml once and exposes the active mode's sizes, so
every compiler component can derive geometry from one place.

Override the toml path with the ``PLENA_SETTINGS`` environment variable
(useful for tests / non-default targets).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class PlenaSettingsError(RuntimeError):
    pass


def _default_settings_path() -> Path:
    """Locate ``plena_settings.toml``.

    Honours ``$PLENA_SETTINGS`` first; otherwise walks up from this file
    to ``PLENA_Simulator/`` (this module lives at
    ``PLENA_Simulator/compiler/tilelang_tvm_compiler/plena_settings.py``,
    so the toml is three parents up)."""
    env = os.environ.get("PLENA_SETTINGS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "plena_settings.toml"


@dataclass(frozen=True)
class HardwareSizes:
    """The active mode's hardware geometry, read from plena_settings.toml.

    ``mlen`` — matrix/vector lane width (full HW vector).
    ``hlen`` — narrow head dim per BTMM lane.
    ``blen`` — block tile width.
    ``vlen`` — vector SRAM row width.
    ``v_prefetch_amount`` — number of VLEN-wide rows one H_PREFETCH_V
        instruction transfers (HBM_V_Prefetch_Amount).
    ``v_writeback_amount`` — number of VLEN-wide rows one H_STORE_V
        instruction transfers (HBM_V_Writeback_Amount).
    """
    mode: str
    mlen: int
    hlen: int
    blen: int
    vlen: int
    v_prefetch_amount: int
    v_writeback_amount: int

    @property
    def hardware_lane_count(self) -> int:
        """Number of BTMM head lanes packed into one MLEN vector."""
        return self.mlen // self.hlen


@lru_cache(maxsize=None)
def load_sizes(path: str | None = None) -> HardwareSizes:
    """Parse ``plena_settings.toml`` and return the active mode's sizes.

    ``path`` defaults to :func:`_default_settings_path`. Cached — the
    toml is read once per process.
    """
    toml_path = Path(path) if path is not None else _default_settings_path()
    if not toml_path.is_file():
        raise PlenaSettingsError(
            f"plena_settings.toml not found at {toml_path}. Set "
            f"$PLENA_SETTINGS to point at it."
        )
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    # The transactional (behavior) emulator hard-codes reading the
    # [BEHAVIOR] section (see transactional_emulator load_config.rs —
    # ``#[serde(rename = "BEHAVIOR")]``). The compiler must read the
    # SAME section, or its geometry drifts from the emulator's: an
    # analytic-mode MLEN=512 against a behavior-mode MLEN=64 emulator
    # produces addresses too large for the 32-bit instruction immediate
    # field. So we ignore [MODE].active and always use [BEHAVIOR].
    active = "behavior"
    section = "BEHAVIOR"
    if section not in cfg:
        raise PlenaSettingsError(
            f"{toml_path}: no [{section}] section exists"
        )
    config = cfg[section].get("CONFIG", {})

    def _val(key: str) -> int:
        try:
            return int(config[key]["value"])
        except KeyError as e:
            raise PlenaSettingsError(
                f"{toml_path}: [{section}.CONFIG.{key}] missing or has no "
                f"'value' field"
            ) from e

    return HardwareSizes(
        mode=active,
        mlen=_val("MLEN"),
        hlen=_val("HLEN"),
        blen=_val("BLEN"),
        vlen=_val("VLEN"),
        v_prefetch_amount=_val("HBM_V_Prefetch_Amount"),
        v_writeback_amount=_val("HBM_V_Writeback_Amount"),
    )


# Convenience accessors — every call routes through the cached loader.
def mlen() -> int:
    return load_sizes().mlen


def hlen() -> int:
    return load_sizes().hlen


def blen() -> int:
    return load_sizes().blen


def vlen() -> int:
    return load_sizes().vlen


def v_prefetch_amount() -> int:
    return load_sizes().v_prefetch_amount


def v_writeback_amount() -> int:
    return load_sizes().v_writeback_amount


__all__ = [
    "HardwareSizes",
    "PlenaSettingsError",
    "load_sizes",
    "mlen",
    "hlen",
    "blen",
    "vlen",
]
