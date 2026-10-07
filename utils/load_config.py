from __future__ import annotations

import re
from pathlib import Path

try:  # stdlib since 3.11, and the reason this file used to read nothing
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    tomllib = None

try:
    import toml
except ImportError:
    toml = None


_PARAM_PATTERN = re.compile(r"\s*(?:localparam|parameter)\s+(?:\w+\s+)?(\w+)\s*=\s*([^;]+);")


def load_svh_settings(file_path: str | Path) -> dict[str, int]:
    """Parse integer `parameter`/`localparam` definitions from a SystemVerilog .svh/.sv file.

    This is intentionally tiny and self-contained so the `compiler/` repo can run
    its assembler/generator without depending on the simulator monorepo's `tools/`.
    """
    hardware_settings: dict[str, int] = {}
    path = Path(file_path)

    with path.open() as f:
        for line in f:
            match = _PARAM_PATTERN.match(line)
            if not match:
                continue

            name, value_str = match.groups()
            value_str = value_str.strip()

            # Keep behavior minimal: only accept plain integers.
            try:
                value = int(value_str)
            except ValueError:
                continue

            hardware_settings[name] = value

    return hardware_settings


def load_toml_config(file_path, section_to_load=None, mode="BEHAVIOR"):
    """``full_toml[mode][section_to_load]``, or ``{}`` if either is absent.

    Reads through `tomllib` when it is available, which since Python 3.11 is
    always. It used to require the third-party `toml` package and raise
    `ImportError` without it -- and `toml` is not a declared dependency and is
    not installed in CI, so every caller that wrapped this in a try/except and
    fell back to a default was taking the fallback unconditionally. That was
    silent: the compiler's `plena_settings.toml` values had never once been
    read.
    """
    if tomllib is not None:
        with open(file_path, "rb") as f:
            full_toml = tomllib.load(f)
    elif toml is not None:
        with open(file_path) as f:
            full_toml = toml.load(f)
    else:  # pragma: no cover - Python < 3.11 without `toml`
        raise ImportError(
            "load_toml_config needs `tomllib` (Python 3.11+) or the `toml` package"
        )
    mode_section = full_toml.get(mode, {})
    return mode_section.get(section_to_load, {})
