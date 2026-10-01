"""Pair-specific burial at fixed coordinates; no unbound relaxation implied."""
from dataclasses import asdict
import numpy as np
from biotite.structure import sasa
from scipy.spatial import cKDTree
from analyze_foldbench_context import SITE_ATOMS

POLICY={'geometry':'selected two-chain pair only; other chains and all nonprotein omitted from both scorers',
        'sasa':'observed canonical protein heavy atoms; ProtOr; 1.4 A probe; 1000 Fibonacci points',
        'delta_sasa':'isolated-chain SASA minus pair-complex SASA, identical coordinates and atoms',
        'interface':'residue deltaSASA >= 1 A^2; raw deltaSASA retained for threshold sensitivity',
        'contact':'nearest opposing-chain heavy atom; contact flag <= 5 A; separate from SASA definition',
        'missing_sidechains':'exposure is observed-atom geometry, not template-imputed; completeness flag retained',
        'interpretation':'geometric association burial, not a binding free energy or relaxed unbound structure'}


def interface_rows(topology,expected):
    atoms=topology.atoms
    chains=np.asarray(atoms.chain_id)
    partner_ids=sorted(set(chains))
    if len(partner_ids)!=2:raise ValueError('interface requires exactly two chains')
    heavy=~np.isin(np.char.upper(atoms.element),['H','D'])
    bound=np.asarray(sasa(atoms,probe_radius=1.4,point_number=1000),dtype=float)
    isolated=np.full(len(atoms),np.nan)
    for chain in partner_ids:
        select=chains==chain
        isolated[select]=sasa(atoms[select],probe_radius=1.4,point_number=1000)
    if not np.isfinite(bound[heavy]).all() or not np.isfinite(isolated[heavy]).all():
        raise ValueError('nonfinite heavy-atom SASA')
    delta=isolated-bound
    if np.min(delta[heavy]) < -1e-3:raise ValueError('negative burial with fixed-coordinate matched atoms')
    trees={chain:cKDTree(atoms.coord[(chains!=chain)&heavy]) for chain in partner_ids}
    residues=[]
    for i,key in enumerate(topology.keys):
        start,stop=topology.starts[i:i+2];hm=heavy[start:stop]
        dist=float(np.min(trees[key.chain].query(atoms.coord[start:stop][hm])[0]))
        residue_delta=float(delta[start:stop][hm].sum())
        residues.append({'residue':asdict(key),'residue_isolated_sasa':float(isolated[start:stop][hm].sum()),
            'residue_complex_sasa':float(bound[start:stop][hm].sum()),'residue_delta_sasa':residue_delta,
            'nearest_partner_heavy_atom_distance':dist,'contact_le5A':dist<=5.,
            'interface_delta_sasa_ge1':residue_delta>=1.})
    sites=[]
    for i,group in expected:
        start,stop=topology.starts[i:i+2];residue=topology.residue(i)
        required=SITE_ATOMS[group];mask=np.isin(residue.atom_name,required)
        complete=set(required)<=set(map(str,residue.atom_name))
        sites.append({**residues[i],
            'functional_isolated_sasa':float(isolated[start:stop][mask].sum()) if complete else None,
            'functional_complex_sasa':float(bound[start:stop][mask].sum()) if complete else None,
            'functional_delta_sasa':float(delta[start:stop][mask].sum()) if complete else None,
            'functional_atoms_complete':complete})
    keychain={str(key):key.chain for key in topology.keys}
    links=[pair for pair in topology.metadata.get('disulfide_pairs',[]) if keychain[pair[0]]!=keychain[pair[1]]]
    return sites,{'policy':POLICY,'chains':partner_ids,'residues':residues,
        'sum_buried_area_both_partners':float(delta[heavy].sum()),
        'half_sum_buried_area':float(delta[heavy].sum()/2),
        'interchain_disulfides':links,'interface_residues':sum(r['interface_delta_sasa_ge1'] for r in residues)}
