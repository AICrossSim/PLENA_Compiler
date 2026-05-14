"""Compatibility wrapper for the old ATen e2e runner module name.

Use ``compiler.aten.sliced_emulator_runner`` for new code.
"""

from compiler.aten import sliced_emulator_runner as _impl

for _name, _value in vars(_impl).items():
    if _name not in {
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__name__",
        "__package__",
        "__spec__",
    }:
        globals()[_name] = _value

for _compat_local in ("_impl", "_name", "_value"):
    globals().pop(_compat_local, None)

__all__ = [name for name in globals() if not name.startswith("__")]


if __name__ == "__main__":
    main()
