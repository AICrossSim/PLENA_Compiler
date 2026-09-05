import pytest
from compiler.aten.plena.matrix_recurrence_lowering import NEMOTRON_MAMBA
from compiler.aten.plena.prepared_vector_recurrence import PreparedVectorGroup, lower_prepared_vector_recurrence


def test_static_baseline_preserves_ordinary_arithmetic_and_dma_stream():
    fields = {
        name: (index + 2) * 1024 * 1024 for index, name in enumerate(("x", "a", "b", "c", "d", "dt", "zero", "output"))
    }
    groups = tuple(PreparedVectorGroup(group * 524288, fields) for group in range(2))
    a = lower_prepared_vector_recurrence(NEMOTRON_MAMBA, groups)
    b = lower_prepared_vector_recurrence(NEMOTRON_MAMBA, groups, static_address_reuse=True)
    def consumers(asm):
        return [line for line in asm.splitlines() if line.startswith(("V_", "H_"))]
    assert consumers(a) == consumers(b)
    assert len(b.splitlines()) < len(a.splitlines())
    assert "L_TILE" not in a + b
    assert all(line.endswith(", 2") for line in consumers(a) if line.startswith("H_"))
    with pytest.raises(ValueError, match="group count"):
        lower_prepared_vector_recurrence(NEMOTRON_MAMBA, groups[:1])


def test_pairwise_kda_uses_only_ordinary_isa_and_fits_existing_sram():
    from compiler.aten.plena.matrix_recurrence_lowering import KIMI_KDA
    fields = {name: (i + 4) * 1048576 for i, name in enumerate(("value", "decay", "key", "query", "beta", "zero", "output"))}
    groups = tuple(PreparedVectorGroup(i * 524288, fields) for i in range(6))
    kwargs = dict(pairwise_bf16_dot=True, static_address_reuse=True)
    asm = lower_prepared_vector_recurrence(KIMI_KDA, groups, **kwargs)
    ops = [line.split()[0] for line in asm.splitlines() if line and not line.startswith(";")]
    assert set(ops) <= {"S_LUI_INT", "S_ADDI_INT", "H_PREFETCH_V", "H_STORE_V", "V_MUL_VV", "V_ADD_VV", "V_SUB_VV"}
    # Per group: 128 state-update adds plus two 127-add trees. No copy adds.
    assert ops.count("V_ADD_VV") == 6 * (128 + 2 * 127)
    assert ops.count("V_MUL_VV") == 6 * (4 * 128 + 1)
    with pytest.raises(ValueError, match="15 existing"):
        lower_prepared_vector_recurrence(KIMI_KDA, groups, vector_sram_rows=14, **kwargs)
    with pytest.raises(ValueError, match="exclusive"):
        lower_prepared_vector_recurrence(KIMI_KDA, groups, experimental_fp32_dot=True, **kwargs)
