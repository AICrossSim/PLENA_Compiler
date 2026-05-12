"""Deprecated alias for the ATen e2e runner.

Use ``python -m compiler.aten.e2e_runner`` or import
``compiler.aten.e2e_runner.run_aten_e2e`` directly.
"""

from compiler.aten.e2e_runner import main, run_aten_e2e

__all__ = ["main", "run_aten_e2e"]


if __name__ == "__main__":
    main()
