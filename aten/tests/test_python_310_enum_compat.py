"""Exercise the 3.10 fallback even when CI itself runs Python 3.11+."""

import importlib.util
import json
from enum import auto
from pathlib import Path

import pytest


def test_strenum_fallback_preserves_contract_strings(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def without_stdlib_strenum(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "enum" and "StrEnum" in (fromlist or ()):
            raise ImportError("Python 3.10 has no enum.StrEnum")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_stdlib_strenum)
    path = Path(__file__).resolve().parents[2] / "utils" / "enum_compat.py"
    spec = importlib.util.spec_from_file_location("enum_compat_without_strenum", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class WireFormat(module.StrEnum):
        AFFINE = "affine"
        BF16 = auto()

    assert WireFormat("affine") is WireFormat.AFFINE
    assert str(WireFormat.AFFINE) == "affine"
    assert f"@layout={WireFormat.AFFINE}" == "@layout=affine"
    assert format(WireFormat.BF16, ">6") == "  bf16"
    assert json.dumps({"format": WireFormat.BF16}) == '{"format": "bf16"}'
    assert {WireFormat.BF16: 2}["bf16"] == 2
    with pytest.raises(TypeError, match="not a string"):
        class InvalidFormat(module.StrEnum):
            INTEGER = 1
