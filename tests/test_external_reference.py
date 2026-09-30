"""Actual third-party regression entry points, never fabricated reference values."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from jaxpropka.parameters import atom_volume
from jaxpropka.precompute import radial_summary, hbond_strength
from jaxpropka.geometry import PolarGeometry

pytestmark=pytest.mark.reference
DATA=Path(__file__).parent/'data'

@pytest.fixture
def propka_energy():
    return pytest.importorskip('propka.energy',reason='pinned PROPKA package unavailable')


def test_pip_propka_distance_scaled_coulomb_primitive(propka_energy):
    parameters=SimpleNamespace(coulomb_cutoff1=4.,coulomb_cutoff2=10.)
    for distance in (1.,4.,5.,9.,10.,12.):
        for burial in (0.,.2,1.):
            eps=160.-130.*burial;r=max(4.,distance)
            actual=244.12/r*np.clip((10.-r)/6.,0,1)/eps
            expected=propka_energy.coulomb_energy(distance,burial,parameters)
            np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_pip_propka_radial_volume_and_chain_identity_primitive(propka_energy):
    xyz=np.array([[1.,0,0],[4.,0,0],[14.9,0,0],[15.,0,0],[19.9,0,0],[20.,0,0]])
    names=['CA','CG','OD1','NZ','SG','O']
    elements=['C','C','O','N','S','O']
    volumes=np.array([atom_volume(a,b) for a,b in zip(names,elements)])
    atoms=[SimpleNamespace(x=x[0],y=x[1],z=x[2],name=name,element=element,res_num=1,chain_id='B')
           for x,name,element in zip(xyz,names,elements)]
    # Same residue number on a different chain MUST contribute; own atom must not.
    own=SimpleNamespace(x=.1,y=0,z=0,name='CA',element='C',res_num=1,chain_id='A')
    container=SimpleNamespace(get_non_hydrogen_atoms=lambda:[own,*atoms])
    target_atom=SimpleNamespace(res_num=1,chain_id='A',conformation_container=container)
    parameters=SimpleNamespace(desolv_cutoff_squared=400.,buried_cutoff_squared=225.,
        VanDerWaalsVolume={'C':1.4,'C4':2.64,'N':1.06,'O':1.,'S':1.66},
        Nmin=280.,Nmax=560.,
        desolvationSurfaceScalingFactor=.25,desolvationPrefactor=-13.,desolvationAllowance=0.)
    target=SimpleNamespace(x=0.,y=0.,z=0.,atom=target_atom,charge=-1.)
    propka_energy.radial_volume_desolvation(parameters,target)
    v,m=radial_summary(np.zeros((1,3)),xyz,volumes)
    np.testing.assert_allclose(target.energy_volume,13.*v[0]*.25,atol=1e-12)
    assert target.num_volume==m[0]


def test_pip_propka_hbond_distance_cosine_primitive(propka_energy):
    for distance in (1.,2.5,3.,4.):
        donor=PolarGeometry(np.array([[0.,0,0]]),np.array([[1.,0,0]]),np.array([[1.,0,0]]),np.ones(1,bool),np.zeros((0,3)))
        accept=PolarGeometry(np.zeros((0,3)),np.zeros((0,3)),np.zeros((0,3)),np.zeros(0,bool),np.array([[1+distance,0,0]]))
        actual=hbond_strength(donor,accept,(2.5,3.5))
        expected=propka_energy.hydrogen_bond_energy(distance,-.85,[2.5,3.5],1.)
        np.testing.assert_allclose(actual,expected,atol=1e-12)


def _full_report(tmp_path):
    pytest.importorskip('biotite',reason='Biotite absent; live structure/reference comparison NOT validated')
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache
    from jaxpropka import TitrationModel,ModelConfig
    from jaxpropka.reference import write_reference_structure,run_reference,compare_reference,assert_baseline
    top=load_topology(DATA/'two_chains.pdb');lib=build_candidates(top)
    model=TitrationModel(build_cache(top,lib),ModelConfig(steps=128))
    path=tmp_path/'reference.pdb';mapping=write_reference_structure(top,lib,path)
    reference=run_reference(path,mapping=mapping)
    report=compare_reference(model,model.native_probabilities,reference,ph=[2.,5.,7.,10.,12.])
    assert report['n_sites']==12
    assert len([s for s in report['sites'] if s['group']=='NTERM'])==2
    assert len([s for s in report['sites'] if s['group']=='CTERM'])==2
    assert all(np.isfinite(v) for v in report['metrics'].values())
    output=Path(os.environ.get('JAXPROPKA_TEST_REPORT_DIR',tmp_path))
    output.mkdir(parents=True,exist_ok=True)
    (output/'modern-two-chains.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'modern-two-chains.pka').write_text(reference.output_text)
    (output/'modern-two-chains.stdout.txt').write_text(reference.stdout)
    # Check machinery against the very same report, but DO NOT call this an
    # independent approved scientific baseline. External reviewed baselines below.
    assert_baseline(report,report)
    baseline=os.environ.get('JAXPROPKA_MODERN_BASELINE')
    if baseline:
        assert_baseline(report,json.loads(Path(baseline).read_text()))
    return report


def test_live_pip_propka_multichain_discrepancy_report(tmp_path):
    pytest.importorskip('propka',reason='modern PROPKA unavailable; external regression NOT validated')
    report=_full_report(tmp_path)
    assert report['reference']['backend']=='modern'
    assert report['reference']['version']=='3.5.1'
