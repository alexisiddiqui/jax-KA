"""Actual third-party regression entry points, never fabricated reference values."""
import importlib.util
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
def legacy_root():
    root=Path(os.environ.get('JAXPROPKA_LEGACY_ROOT','.reference/propka-3.0')).resolve()
    if not (root/'Source/calculator.py').is_file():
        if os.environ.get('JAXPROPKA_REQUIRE_LEGACY')=='1':
            pytest.fail('required real PROPKA 3.0 checkout missing')
        pytest.skip('original PROPKA 3.0 checkout unavailable; legacy parity NOT validated')
    return root

@pytest.fixture
def legacy_calculator(legacy_root):
    spec=importlib.util.spec_from_file_location('original_propka30_calculator',legacy_root/'Source/calculator.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_original30_distance_scaled_coulomb_primitive(legacy_calculator):
    version=SimpleNamespace(coulomb_diel=[30.,160.],coulomb_cutoff=[4.,10.])
    for distance in (1.,4.,5.,9.,10.,12.):
        for burial in (0.,.2,1.):
            eps=160.-130.*burial;r=max(4.,distance)
            actual=244.12/r*np.clip((10.-r)/6.,0,1)/eps
            expected=legacy_calculator.distanceScaledCoulombEnergy(distance,burial,version)
            np.testing.assert_allclose(actual,expected,atol=1e-12)


def test_original30_radial_volume_and_chain_identity_primitive(legacy_calculator):
    xyz=np.array([[1.,0,0],[4.,0,0],[14.9,0,0],[15.,0,0],[19.9,0,0],[20.,0,0]])
    names=['CA','CG','OD1','NZ','SG','O']
    elements=['C','C','O','N','S','O']
    volumes=np.array([atom_volume(a,b) for a,b in zip(names,elements)])
    atoms=[SimpleNamespace(x=x[0],y=x[1],z=x[2],name=name,element=element,resNumb=1,chainID='B')
           for x,name,element in zip(xyz,names,elements)]
    # Same residue number on a different chain MUST contribute; own atom must not.
    own=SimpleNamespace(x=.1,y=0,z=0,name='CA',element='C',resNumb=1,chainID='A')
    source={'A':{'keys':['a'],'a':[own]},'B':{'keys':['b'],'b':atoms}}
    version=SimpleNamespace(desolv_cutoff_sqr=400.,buried_cutoff_sqr=225.,
        desolvationVolume={'C':1.4,'C4':2.64,'N':1.06,'O':1.,'S':1.66},
        calculateWeight=lambda n:np.clip((n-280.)/280.,0,1),
        desolvationSurfaceScalingFactor=.25,desolvationPrefactor=-13.,desolvationAllowance=0.)
    target=SimpleNamespace(x=0.,y=0.,z=0.,resNumb=1,chainID='A',label='TEST',Q=-1.,Vmass=0.)
    legacy_calculator.radialVolumeDesolvation(target,source,version)
    v,m=radial_summary(np.zeros((1,3)),xyz,volumes)
    np.testing.assert_allclose(target.Emass,13.*v[0]*.25,atol=1e-12)
    assert target.Nmass==m[0]


def test_original30_hbond_distance_cosine_primitive(legacy_calculator):
    for distance in (1.,2.5,3.,4.):
        donor=PolarGeometry(np.array([[0.,0,0]]),np.array([[1.,0,0]]),np.array([[1.,0,0]]),np.ones(1,bool),np.zeros((0,3)))
        accept=PolarGeometry(np.zeros((0,3)),np.zeros((0,3)),np.zeros((0,3)),np.zeros(0,bool),np.array([[1+distance,0,0]]))
        actual=hbond_strength(donor,accept,(2.5,3.5))
        expected=legacy_calculator.HydrogenBondEnergy(distance,-.85,[2.5,3.5],1.)
        np.testing.assert_allclose(actual,expected,atol=1e-12)


def _full_report(tmp_path,backend,legacy_root=None):
    pytest.importorskip('biotite',reason='Biotite absent; live structure/reference comparison NOT validated')
    from jaxpropka.topology import load_topology
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache
    from jaxpropka import TitrationModel,ModelConfig
    from jaxpropka.reference import write_reference_structure,run_reference,compare_reference,assert_baseline
    top=load_topology(DATA/'two_chains.pdb');lib=build_candidates(top)
    model=TitrationModel(build_cache(top,lib),ModelConfig(steps=128))
    path=tmp_path/'reference.pdb';mapping=write_reference_structure(top,lib,path)
    reference=run_reference(path,backend=backend,legacy_root=legacy_root,mapping=mapping,
                            expected_commit=os.environ.get('JAXPROPKA_LEGACY_COMMIT'))
    report=compare_reference(model,model.native_probabilities,reference,ph=[2.,5.,7.,10.,12.])
    assert report['n_sites']==12
    assert len([s for s in report['sites'] if s['group']=='NTERM'])==2
    assert len([s for s in report['sites'] if s['group']=='CTERM'])==2
    assert all(np.isfinite(v) for v in report['metrics'].values())
    output=Path(os.environ.get('JAXPROPKA_TEST_REPORT_DIR',tmp_path))
    output.mkdir(parents=True,exist_ok=True)
    (output/f'{backend}-two-chains.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/f'{backend}-two-chains.pka').write_text(reference.output_text)
    (output/f'{backend}-two-chains.stdout.txt').write_text(reference.stdout)
    # Check machinery against the very same report, but DO NOT call this an
    # independent approved scientific baseline. External reviewed baselines below.
    assert_baseline(report,report)
    baseline=os.environ.get(f'JAXPROPKA_{backend.upper()}_BASELINE')
    if baseline:
        assert_baseline(report,json.loads(Path(baseline).read_text()))
    return report


def test_live_original30_multichain_discrepancy_report(legacy_root,tmp_path):
    _full_report(tmp_path,'legacy30',legacy_root)


def test_live_modern_multichain_discrepancy_report(tmp_path):
    pytest.importorskip('propka',reason='modern PROPKA unavailable; external regression NOT validated')
    report=_full_report(tmp_path,'modern')
    assert report['reference']['backend']=='modern'
