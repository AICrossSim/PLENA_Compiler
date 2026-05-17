import re
from pathlib import Path

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
    if toml is None:
        raise ImportError("'toml' package required for load_toml_config. Install with: pip install toml")
    with open(file_path) as f:
        full_toml = toml.load(f)
    mode_section = full_toml.get(mode, {})
    return mode_section.get(section_to_load, {})
