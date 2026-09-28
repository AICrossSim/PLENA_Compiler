"""V2 placement and common-control regression gates."""
from itertools import product
import pytest
from compiler.aten.plena.ltile_v2 import Options,views,lower_group

@pytest.mark.parametrize('kind,phase,broadcast',product(('mamba','kda'),(True,False),(True,False)))
def test_all_ablation_placements_fit_without_aliases(kind,phase,broadcast):
    o=Options(kind,phased=phase,broadcast=broadcast)
    placed=views(o) # Performs exact physical capacity and collision checks.
    assert placed['state'].descriptor.shape.rows == o.chunk_rows

@pytest.mark.parametrize('kind',('mamba','kda'))
def test_row_control_uses_scalar_loop_and_identical_memory_transfers(kind):
    memory=dict(states=[0],update=[1<<20],dot=[2<<20],input=3<<20,scalar=4<<20,skip=5<<20,output=6<<20)
    arms={control:lower_group(Options(kind,control),memory).lines for control in ('row','fsm')}
    for instructions in arms.values():
        assert all('V_DOT_' not in x for x in instructions)
    assert sum(x.startswith('C_LOOP_START') for x in arms['row']) == (2 if kind=='mamba' else 3)
    assert not any(x.startswith('C_LOOP_START') for x in arms['fsm'])
    # No diagnostic store, no additional state buffer in the row control.
    assert [x for x in arms['row'] if x.startswith('H_')] == [x for x in arms['fsm'] if x.startswith('H_')]


def test_native_descriptor_encoding_and_bounds():
    from compiler.aten.plena.ltile_native import CoefficientView
    from compiler.aten.plena.mview import encode_l_tile_ccfg
    d = CoefficientView(300032, 128, 1, 3, 32, 512)
    assert d.address(127, 31) == 300543
    assert d.address(17, 0) == d.address(17, 7)
    assert d.pack() == 300032 | 128 << 19 | 1 << 32 | 3 << 41 | 31 << 44 | 511 << 50
    assert encode_l_tile_ccfg(slot=2, low_register=12, high_register=13) == 0x808000 | 12 << 6 | 13 << 10 | 0x3f
    with pytest.raises(ValueError):
        d.address(128, 31)
    with pytest.raises(ValueError):
        CoefficientView(524287, 1, 0, 0, 32, 32)
    with pytest.raises(ValueError):
        encode_l_tile_ccfg(slot=3, low_register=1, high_register=2)


@pytest.mark.parametrize('kind', ('mamba', 'kda'))
def test_native_requires_explicit_dominating_configuration(kind):
    from compiler.aten.plena.mview import validate_matrix_view_dominance
    native = ([(0x800000, 0, 1, 0, 0), (0x801000, 0, 128, 1, 3),
               (0x801000, 1024, 128, 1, 3)] if kind == 'mamba' else
              [(0x800000+i*4096, 0, 128, 1, 0) for i in range(3)])
    mem = dict(native=native, states=[0], input=0x900000, scalar=0x901000,
               skip=0x902000, output=0x903000)
    text = '\n'.join(lower_group(Options(kind), mem).lines)
    validate_matrix_view_dominance(text)
    with pytest.raises(ValueError, match='coefficient'):
        validate_matrix_view_dominance('\n'.join(x for x in text.splitlines() if not x.startswith('L_TILE_CCFG')))
    with pytest.raises(ValueError, match='FSM'):
        lower_group(Options(kind, 'row'), mem)
