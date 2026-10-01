"""Regression for PROPKA's number-only terminal detection after a singleton."""
from pathlib import Path

import numpy as np

from jaxpropka.topology import load_topology
from jaxpropka.geometry import build_candidates
from jaxpropka.reference import write_reference_structure, run_reference


def test_singleton_followed_by_segment_preserves_nterm_identity(tmp_path):
    from propka.run import single
    top=load_topology(Path(__file__).parent/'data/peptide.pdb')
    atoms=top.atoms[top.atoms.res_id!=2]
    top=load_topology(atoms,gap_policy='free')
    candidates=build_candidates(top)
    path=tmp_path/'reference.pdb'
    mapping=write_reference_structure(top,candidates,path)
    assert len({number for chain,number in mapping})==top.n_residues
    mol=single(path,write_pka=False)
    actual={mapping[g.atom.chain_id,g.atom.res_num] for g in mol.conformations['AVR'].groups
            if g.residue_type=='N+'}
    expected={top.keys[i] for i in np.flatnonzero(top.nterm)}
    assert actual==expected
    ref=run_reference(path,mapping=mapping)
    # Single-residue termini can legitimately be suppressed by PROPKA.
    assert all(s['reason']=='propka_covalent_coupling_suppression' for s in ref.excluded_sites)
    assert any(s['group']=='CTERM' for s in ref.excluded_sites)
