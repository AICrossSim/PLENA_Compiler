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
