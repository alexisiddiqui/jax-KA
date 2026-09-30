"""Molecular integration tests: never replace Biotite with a mock implementation."""
from pathlib import Path
import numpy as np
import pytest
from jaxpropka import prepare, TitrationModel, ResidueKey, GROUPS, one_hot
from jaxpropka.topology import load_topology
from jaxpropka.geometry import build_candidates
from jaxpropka.precompute import build_cache
from jaxpropka.parameters import THREE
from jaxpropka.reference import write_reference_structure

pytestmark=pytest.mark.integration
DATA=Path(__file__).parent/'data'

@pytest.fixture(autouse=True)
def real_biotite():
    return pytest.importorskip('biotite',reason='real Biotite dependency unavailable; molecular path NOT validated')

@pytest.fixture
def peptide():
    return load_topology(DATA/'peptide.pdb')


def test_multiple_chains_repeated_numbers_and_independent_termini():
    top=load_topology(DATA/'two_chains.pdb')
    assert top.chain_ids==('A','B')
    assert top.keys[0]==ResidueKey('A',1) and top.keys[5]==ResidueKey('B',1)
    assert top.nterm.sum()==top.cterm.sum()==2
    assert top.following[4]==top.previous[5]==-1
    atom_res=np.repeat(np.arange(10),np.diff(top.starts))
    for a,b,_ in top.atoms.bonds.as_array():
        assert top.chain_index[atom_res[a]]==top.chain_index[atom_res[b]]


def test_all_twenty_candidates_and_native_heavy_atoms(peptide):
    lib=build_candidates(peptide)
    assert len(lib.residues)==5 and all(len(row)==20 for row in lib.residues)
    for i,row in enumerate(lib.residues):
        native=row[peptide.native_index[i]]
        for atom in peptide.residue(i):
            np.testing.assert_allclose(native.coordinates[str(atom.atom_name)],atom.coord,atol=0,rtol=0)
        for candidate in row:
            assert np.isfinite(candidate.side_xyz).all()
    assert lib.group_mask.sum()==5*7+2


def test_rigid_rotation_preserves_geometric_kernels(peptide):
    cache=build_cache(peptide,build_candidates(peptide),dtype=np.float64)
    atoms=peptide.atoms.copy()
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    atoms.coord=atoms.coord@rotation+np.array([9.,2.,-3.])
    top=load_topology(atoms)
    moved=build_cache(top,build_candidates(top),dtype=np.float64)
    for field in ('volume','mass','hbond','coulomb_geometry','hb_donor','hb_reverse'):
        np.testing.assert_allclose(getattr(cache,field),getattr(moved,field),rtol=3e-5,atol=3e-5)


def test_far_chains_equal_separate_single_chain_and_selection():
    import jax.numpy as jnp
    single=TitrationModel(prepare(DATA/'peptide.pdb'))
    dimer=TitrationModel(prepare(DATA/'two_chains_far.pdb'))
    ph=np.array([2.,7.,12.])
    one=single.curves(ph)(single.native_probabilities)
    two=dimer.curves(ph)(dimer.native_probabilities)
    np.testing.assert_allclose(two.chain_charge[:,0],one.total_charge,atol=2e-5)
    np.testing.assert_allclose(two.chain_charge[:,1],one.total_charge,atol=2e-5)
    np.testing.assert_allclose(two.total_charge,2*one.total_charge,atol=3e-5)
    selected=dimer.curves(ph,residues=[('B',2),('A',3)])(dimer.native_probabilities)
    np.testing.assert_allclose(selected.residue_charge,two.residue_charge[:,[6,2]],atol=0)
    np.testing.assert_allclose(selected.total_charge,two.total_charge,atol=0)
    assert selected.protonated.shape==(3,2,9)


def test_near_chain_edges_are_present():
    cache=prepare(DATA/'two_chains.pdb')
    cross=cache.chain_index[:,None]!=cache.chain_index[cache.neighbors]
    assert np.any(cache.pair_mask & cross[:,:,None,None])


def test_mmcif_author_chain_and_insertion_ids_roundtrip(peptide,tmp_path):
    from biotite.structure.io import pdbx
    atoms=peptide.atoms.copy()
    atoms.chain_id=np.full(len(atoms),'AA',dtype='U4')
    atoms.ins_code=np.where(atoms.res_id==2,'B','')
    file=pdbx.CIFFile();pdbx.set_structure(file,atoms)
    path=tmp_path/'insertion.cif';file.write(path)
    top=load_topology(path)
    assert top.chain_ids==('AA',) and top.keys[1]==ResidueKey('AA',2,'B')
    cache=build_cache(top,build_candidates(top))
    assert cache.select([('AA',2,'B')]).tolist()==[1]
    with pytest.raises(KeyError): cache.select([('AA',2)])


def test_missing_internal_segment_does_not_silently_create_charge(peptide):
    atoms=peptide.atoms[peptide.atoms.res_id!=3]
    with pytest.raises(ValueError,match='unexplained chain break'): load_topology(atoms)
    capped=load_topology(atoms,gap_policy='cap')
    free=load_topology(atoms,gap_policy='free')
    assert capped.nterm.sum()==capped.cterm.sum()==1
    assert free.nterm.sum()==free.cterm.sum()==2
    assert capped.previous[2]==-1 and len(capped.metadata['gaps'])==1


def test_pdb_ter_is_respected_even_when_chain_id_reused(tmp_path):
    lines=[]
    for line in (DATA/'two_chains_far.pdb').read_text().splitlines():
        if line.startswith(('ATOM  ','HETATM')):
            if line[21]=='B':
                number=int(line[22:26])+5
                if number==6 and line[12:16].strip()=='N': lines.append('TER')
                line=line[:21]+'A'+f'{number:4d}'+line[26:]
            lines.append(line)
    path=tmp_path/'ter.pdb';path.write_text('\n'.join(lines+['END'])+'\n')
    top=load_topology(path)
    assert top.chain_ids==('A',) and top.nterm.sum()==top.cterm.sum()==2
    assert top.previous[5]==-1


def test_missing_oxt_is_explicit_and_finite(peptide):
    atoms=peptide.atoms[peptide.atoms.atom_name!='OXT']
    top=load_topology(atoms);lib=build_candidates(top)
    assert lib.metadata['virtual_oxt']==[str(top.keys[-1])]
    assert np.isfinite(lib.centers).all()


def test_missing_native_sidechain_requires_opt_in(peptide):
    atoms=peptide.atoms[peptide.atoms.atom_name!='NZ']
    top=load_topology(atoms)
    with pytest.raises(ValueError,match='missing native sidechain'): build_candidates(top)
    lib=build_candidates(top,missing_sidechain='template')
    assert lib.metadata['rebuilt_native']


def test_unsupported_chemistry_is_not_silently_discarded(peptide):
    atoms=peptide.atoms.copy();atoms.res_name[atoms.res_id==1]='MSE'
    with pytest.raises(ValueError,match='unsupported noncanonical'): load_topology(atoms)


def test_disulfide_requires_frozen_cysteines_and_suppresses_thiols():
    from biotite.structure.info import residue
    import biotite.structure as struc
    import jax
    import jax.numpy as jnp
    a=residue('CYS');a=a[~np.isin(a.element,['H','D'])]
    a.chain_id[:]='A';a.res_id[:]=1
    b=a.copy();b.chain_id[:]='B';b.coord=b.coord+np.array([2.05,0,0])
    atoms=struc.concatenate([a,b])
    with pytest.raises(ValueError,match='disulfide detected'): load_topology(atoms)
    top=load_topology(atoms,freeze_disulfides=True)
    lib=build_candidates(top)
    assert top.disulfide.all() and not lib.group_mask[:,:7].any()
    model=TitrationModel(build_cache(top,lib))
    charge=model.charge()
    grad=jax.grad(lambda z:charge(model.probabilities_from_logits(z)).sum())(jnp.zeros((2,20)))
    np.testing.assert_allclose(grad,0,atol=0)


def test_reference_export_renumbers_without_losing_original_identity(peptide,tmp_path):
    atoms=peptide.atoms.copy();atoms.chain_id[:]='XY'
    atoms.ins_code[atoms.res_id==2]='C'
    top=load_topology(atoms);lib=build_candidates(top)
    destination=tmp_path/'normalized.pdb'
    mapping=write_reference_structure(top,lib,destination)
    assert mapping[('A',2)]==ResidueKey('XY',2,'C')
    read=load_topology(destination)
    assert len(read.keys)==len(top.keys)


def test_molecular_charge_curves_and_midpoints_have_finite_gradients(peptide):
    import jax
    import jax.numpy as jnp
    model=TitrationModel(build_cache(peptide,build_candidates(peptide)))
    charge=model.charge(7.,residues=[('A',2)])
    curves=model.curves([5.,7.,9.])
    pka=model.pka_sites([(('A',4),'HIS')])
    logits=jnp.log(.9*model.native_probabilities+.1/20)
    def loss(z):
        p=model.probabilities_from_logits(z)
        return charge(p).sum()+curves(p).total_charge.sum()+pka(p).value.sum()
    value,grad=jax.value_and_grad(loss)(logits)
    assert np.isfinite(value) and np.isfinite(grad).all()
    assert bool(pka(model.probabilities_from_logits(logits)).valid.all())


def test_reference_export_rejects_capped_terminal_mismatch(peptide,tmp_path):
    top=load_topology(peptide.atoms,capped_n=[('A',1)])
    lib=build_candidates(top)
    with pytest.raises(ValueError,match='free peptide-segment termini'):
        write_reference_structure(top,lib,tmp_path/'capped.pdb')
