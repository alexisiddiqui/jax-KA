from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from interface_exposure import interface_rows
from jaxpropka.topology import load_topology


def test_separated_partners_have_zero_burial():
    topology=load_topology(Path(__file__).parent/'data/two_chains_far.pdb')
    sites,summary=interface_rows(topology,[(0,'NTERM')])
    assert abs(summary['sum_buried_area_both_partners'])<1e-5
    assert summary['interface_residues']==0
    assert sites[0]['functional_delta_sasa']==0
    assert sites[0]['nearest_partner_heavy_atom_distance']>5


def test_nearby_partners_have_nonnegative_burial_and_conserve_sum():
    topology=load_topology(Path(__file__).parent/'data/two_chains.pdb')
    _,summary=interface_rows(topology,[(0,'NTERM')])
    assert summary['sum_buried_area_both_partners']>0
    assert summary['interface_residues']>0
    values=[r['residue_delta_sasa'] for r in summary['residues']]
    assert min(values)>=-1e-3
    np.testing.assert_allclose(sum(values),summary['sum_buried_area_both_partners'],atol=1e-6)
