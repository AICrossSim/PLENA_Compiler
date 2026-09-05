"""Compiler geometry and transfer sizes follow the selected machine section."""

import pytest

from compiler.aten.plena import PlenaCompiler


@pytest.fixture
def machine_settings(tmp_path, monkeypatch):
    path = tmp_path / "plena_settings.toml"
    path.write_text('''
[MODE]
active = "analytic"
[TRANSACTIONAL.CONFIG]
MLEN = 64
MATRIX_SRAM_SIZE = 4096
HLEN = 16
BROADCAST_AMOUNT = 4
HBM_V_Prefetch_Amount = 2
HBM_V_Writeback_Amount = 3
[ANALYTIC.CONFIG]
MLEN = 2048
MATRIX_SRAM_SIZE = 256
''')
    monkeypatch.setenv("PLENA_SETTINGS_TOML", str(path))


def test_machine_geometry_is_selected_by_compiled_mlen(machine_settings):
    compiler = PlenaCompiler(mlen=64, blen=4)
    assert compiler.mram_tile_capacity == 64
    assert (compiler.hlen, compiler.broadcast_amount) == (16, 4)
    assert (compiler.hbm_v_prefetch_amount, compiler.hbm_v_writeback_amount) == (2, 3)


def test_unknown_geometry_and_explicit_capacity_keep_their_defaults(machine_settings):
    assert PlenaCompiler(mlen=8, blen=2).mram_tile_capacity == 4
    assert PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=7).mram_tile_capacity == 7


def test_configured_memory_must_fit_one_matrix_tile(machine_settings):
    with pytest.raises(ValueError, match="MATRIX_SRAM_SIZE 256"):
        PlenaCompiler(mlen=2048, blen=4)
    assert PlenaCompiler(mlen=2048, blen=4, mram_tile_capacity=1).mram_tile_capacity == 1
