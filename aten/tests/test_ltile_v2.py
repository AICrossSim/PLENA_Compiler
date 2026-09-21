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
