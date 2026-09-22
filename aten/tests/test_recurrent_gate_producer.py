"""Producer ownership checks: a future token's input must survive earlier writes."""

from dataclasses import replace

import pytest

from compiler.aten.plena.recurrent_coefficients import MambaGateRow, lower_mamba_gate_rows


def row(index=0):
    base = 65536 + index * 32768
    return MambaGateRow(*(base + i * 4096 for i in range(5)))


def test_producer_cannot_overwrite_a_later_tokens_input():
    first, second = row(), row(1)
    with pytest.raises(ValueError, match="aliases"):
        lower_mamba_gate_rows([replace(first, delta=second.raw_dt), second], 0)


@pytest.mark.parametrize("change", [dict(dt=64), dict(delta=65536), dict(dt=65536 + 3 * 4096, delta=65536 + 3 * 4096)])
def test_constants_inputs_outputs_have_distinct_ownership(change):
    with pytest.raises(ValueError, match="aliases"):
        lower_mamba_gate_rows([replace(row(), **change)], 0)


def test_capacity_address_and_empty_program_rejected():
    with pytest.raises(ValueError):
        lower_mamba_gate_rows([], 0)
    with pytest.raises(ValueError):
        lower_mamba_gate_rows([row()], 0, vector_sram_rows=25)
    with pytest.raises(ValueError):
        lower_mamba_gate_rows([replace(row(), delta=2**32 - 64)], 0)


def test_inputs_can_be_shared_but_not_modified():
    first, second = row(), row(1)
    second = replace(second, dt_bias=first.dt_bias, negative_a=first.negative_a)
    asm = lower_mamba_gate_rows([first, second], 0)
    assert asm.count("V_SOFTPLUS_V") == 2
    assert asm.count("H_STORE_V") == 4
