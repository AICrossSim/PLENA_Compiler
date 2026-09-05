"""The spilled-activation path must refuse an MX KV configuration.

`SPILLED_ACTIVATION` selects the `keyvalue` precision class with
`set_scale=False`. Under the shipped `[TRANSACTIONAL.PRECISION]`, that class is
Mx/e4m3 with a separate scale stream, so the read walks into the scale stream
whose `0x7f` bytes decode to e4m3 NaN. On the emulator every output of KDA's
chunked prefill came back `nan`.

`require_bf16_kv_precision` was written for exactly this and had two problems:
it read `kind` where the TOML spells `format`, so it rejected *every*
configuration including the Plain BF16 one it exists to require; and it put the
burden on a caller, with the result that nothing in the repo ever called it --
not the four SSD emitters, not the KDA one, not a test.

Both are fixed, and both directions are pinned here. The emitters now check the
active build themselves, because the emitter is what chooses the precision class.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compiler.aten.plena import PlenaCompiler  # noqa: E402

_PLAIN_BF16 = {
    "format": "Plain",
    "DATA_TYPE": {"type": "Fp", "sign": True, "exponent": 8, "mantissa": 7},
}
_MX_E4M3 = {
    "format": "Mx",
    "block": 8,
    "ELEM": {"type": "Fp", "sign": True, "exponent": 4, "mantissa": 3},
    "SCALE": {"type": "Fp", "sign": False, "exponent": 8, "mantissa": 0},
}


def _settings(node: dict) -> dict:
    return {"HBM_M_KV_TYPE": node, "HBM_V_KV_TYPE": node}


def test_accepts_plain_bf16():
    """The check has to be able to pass. It could not before: reading `kind`
    where the TOML says `format` returned None for every configuration, so a
    correctly configured build was rejected too -- which is why calling it
    anywhere would have failed the build."""
    PlenaCompiler(mlen=64, blen=4).require_bf16_kv_precision(_settings(_PLAIN_BF16))


def test_rejects_the_shipped_mx_kv_types():
    with pytest.raises(ValueError, match="Plain BF16"):
        PlenaCompiler(mlen=64, blen=4).require_bf16_kv_precision(_settings(_MX_E4M3))


def test_rejects_a_half_configured_build():
    """Only one of the two classes converted is still wrong, and is the easier
    mistake to make by hand."""
    for node_m, node_v in ((_PLAIN_BF16, _MX_E4M3), (_MX_E4M3, _PLAIN_BF16)):
        with pytest.raises(ValueError, match="Plain BF16"):
            PlenaCompiler(mlen=64, blen=4).require_bf16_kv_precision(
                {"HBM_M_KV_TYPE": node_m, "HBM_V_KV_TYPE": node_v}
            )


def test_refuses_to_report_success_without_the_table():
    """A check that cannot run must not look like a check that passed."""
    with pytest.raises(ValueError, match="needs the active PRECISION table"):
        PlenaCompiler(mlen=64, blen=4).require_bf16_kv_precision(None)


def test_the_active_build_check_is_cached():
    """One spill-using emitter per layer would otherwise re-read and re-parse
    the TOML per chunk per head."""
    prog = PlenaCompiler(mlen=64, blen=4)
    prog._bf16_kv_checked = True
    # With the flag set it must not touch the filesystem at all -- if it did,
    # the shipped MX types would raise.
    prog.require_bf16_kv_precision_from_active_build()


def test_the_shipped_settings_would_be_refused():
    """The repository's own plena_settings.toml configures MX KV types, so any
    build that spills without overriding them is refused rather than silently
    producing NaN. If this ever starts passing, the shipped defaults changed and
    the workaround in kda_stage_test.py can go."""
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    toml = root / "plena_settings.toml"
    if not toml.exists():  # pragma: no cover - depends on checkout layout
        pytest.skip("plena_settings.toml not found from this checkout")
    precision = tomllib.loads(toml.read_text())["TRANSACTIONAL"]["PRECISION"]
    with pytest.raises(ValueError, match="Plain BF16"):
        PlenaCompiler(mlen=64, blen=4).require_bf16_kv_precision(precision)
