"""Biotite-only structure I/O and covalent topology; nothing here is traced."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import importlib.metadata
import numpy as np
from scipy.spatial import cKDTree
from .cache import ResidueKey
from .parameters import THREE_TO_INDEX


@dataclass(frozen=True)
class Topology:
    atoms: Any                 # biotite.structure.AtomArray, heavy protein atoms
    keys: tuple[ResidueKey,...]
    starts: np.ndarray         # includes exclusive stop
    native_index: np.ndarray
    chain_ids: tuple[str,...]
    chain_index: np.ndarray
    previous: np.ndarray       # actual peptide neighbors, -1 across any break
    following: np.ndarray
    nterm: np.ndarray
    cterm: np.ndarray
    disulfide: np.ndarray      # [N] endpoints, frozen when explicitly enabled
    backbone: np.ndarray       # [N,4,3], N CA C O
    metadata: dict

    @property
    def n_residues(self):
        return len(self.keys)

    def residue(self,i):
        return self.atoms[self.starts[i]:self.starts[i+1]]


def _pdb_ter_keys(path, model):
    """Biotite does not retain TER; preserve this small piece of file topology.

    Coordinate parsing and atom/residue representation remain entirely Biotite.
    MODEL selection is by ordinal position, as in Biotite's model argument.
    """
    result=set(); current=0; has_models=False; previous=None
    for line in Path(path).read_text().splitlines():
        record=line[:6].strip()
        if record=="MODEL":
            has_models=True;current+=1;previous=None
        selected=(current==model if has_models else model==1)
        if not selected:
            continue
        if record in ("ATOM","HETATM"):
            try:
                previous=ResidueKey(line[21:22].strip(),int(line[22:26]),line[26:27].strip())
            except ValueError:
                previous=None
        elif record=="TER" and previous is not None:
            result.add(previous)
        elif record=="ENDMDL":
            previous=None
    return result


def load_topology(path_or_atoms, *, model=1, altloc="occupancy", chains=None,
                  gap_policy="error", include_termini=True, capped_n=(), capped_c=(),
                  break_after=(), freeze_disulfides=False, ignore_nonprotein=False,
                  peptide_cutoff=1.9):
    """Read PDB/mmCIF/BCIF or an AtomArray, preserving chain and insertion identity.

    gap_policy='error': reject unexplained intrachain backbone breaks (default).
    'cap': do not invent titratable termini at missing-residue gaps.
    'free': treat each unexplained gap as a genuine chain end.
    PDB TER / explicit break_after always mark actual segment ends.
    Water/hydrogens are ignored. Other noncanonical material is rejected unless
    ignore_nonprotein=True, in which case omission is recorded. Covalent
    disulfides require explicit freeze_disulfides=True; mutations at those sites
    are then clamped to native Cys by TitrationModel.
    """
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io import pdbx
    if gap_policy not in ("error","cap","free"):
        raise ValueError("gap_policy must be error, cap, or free")
    if not isinstance(model,int) or model<1 or peptide_cutoff<=0:
        raise ValueError("model must be a positive ordinal and peptide_cutoff positive")
    if altloc not in ("occupancy","first"):
        raise ValueError("choose a single altloc: occupancy or first")
    breaks={ResidueKey.from_value(k) for k in break_after}
    metadata={"model":model,"altloc":altloc,"gap_policy":gap_policy,
              "biotite_version":importlib.metadata.version("biotite")}
    if isinstance(path_or_atoms,(str,Path)):
        path=Path(path_or_atoms)
        metadata.update(source_name=path.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        suffix=path.suffix.lower()
        if suffix in (".pdb",".ent"):
            atoms=PDBFile.read(path).get_structure(model=model,altloc=altloc,include_bonds=True)
            breaks |= _pdb_ter_keys(path,model)
        elif suffix in (".cif",".mmcif",".bcif"):
            file=(pdbx.BinaryCIFFile if suffix==".bcif" else pdbx.CIFFile).read(path)
            atoms=pdbx.get_structure(file,model=model,altloc=altloc,use_author_fields=True,include_bonds=True)
        else:
            raise ValueError("expected .pdb, .ent, .cif, .mmcif, or .bcif")
    elif isinstance(path_or_atoms,struc.AtomArray):
        atoms=path_or_atoms.copy()
        metadata.update(source_name="AtomArray",source_sha256=hashlib.sha256(atoms.coord.tobytes()).hexdigest())
        if "altloc_id" in atoms.get_annotation_categories():
            raise ValueError("resolve alternate locations before passing an AtomArray")
    else:
        raise TypeError("expected a path or Biotite AtomArray (not AtomArrayStack)")
    if chains is not None:
        requested=set(chains)
        missing=requested-set(atoms.chain_id)
        if missing:
            raise KeyError(f"chains absent from structure: {sorted(missing)}")
        atoms=atoms[np.isin(atoms.chain_id,list(requested))]
    canonical=np.isin(atoms.res_name,list(THREE_TO_INDEX))
    water=np.isin(atoms.res_name,["HOH","WAT","H2O","DOD"])
    omitted=sorted(set(atoms.res_name[~canonical & ~water]))
    if omitted and not ignore_nonprotein:
        raise ValueError(f"unsupported noncanonical residues/ligands: {omitted}; do not silently omit their charge")
    metadata["omitted_nonprotein"]=omitted
    metadata["water_atoms_omitted"]=int(water.sum())
    atoms=atoms[canonical & ~np.isin(atoms.element,["H","D"])]
    if len(atoms)==0 or not np.isfinite(atoms.coord).all():
        raise ValueError("expected a nonempty structure with finite coordinates")
    starts=struc.get_residue_starts(atoms,add_exclusive_stop=True)
    keys=tuple(ResidueKey(str(atoms.chain_id[s]),int(atoms.res_id[s]),str(atoms.ins_code[s]).strip())
               for s in starts[:-1])
    if len(set(keys))!=len(keys):
        raise ValueError("ambiguous repeated (chain, residue number, insertion code); rename chain segments")
    n=len(keys)
    native=np.asarray([THREE_TO_INDEX[str(atoms.res_name[s])] for s in starts[:-1]],np.int32)
    chain_ids=tuple(dict.fromkeys(k.chain for k in keys))
    ci=np.asarray([chain_ids.index(k.chain) for k in keys],np.int32)
    lookup=[];bb=np.empty((n,4,3),float)
    for i,(s,e) in enumerate(zip(starts[:-1],starts[1:])):
        names=list(atoms.atom_name[s:e])
        if len(set(names))!=len(names):
            raise ValueError(f"duplicate atom names at {keys[i]}; resolve alternate locations")
        index={name:s+j for j,name in enumerate(names)};lookup.append(index)
        missing={"N","CA","C","O"}-set(index)
        if missing:
            raise ValueError(f"missing backbone at {keys[i]}: {sorted(missing)}")
        bb[i]=[atoms.coord[index[name]] for name in ("N","CA","C","O")]
    previous=np.full(n,-1,np.int32);following=np.full(n,-1,np.int32)
    nterm=np.zeros(n,bool);cterm=np.zeros(n,bool)
    nterm[0]=True;cterm[-1]=True
    bonds=struc.connect_via_residue_names(atoms,inter_residue=False)
    gap_records=[]
    for i in range(n-1):
        j=i+1
        explicit=keys[i].chain!=keys[j].chain or keys[i] in breaks
        distance=float(np.linalg.norm(bb[i,2]-bb[j,0]))
        if explicit:
            cterm[i]=True;nterm[j]=True
        elif distance<=peptide_cutoff:
            following[i]=j;previous[j]=i
            bonds.add_bond(lookup[i]["C"],lookup[j]["N"],struc.BondType.SINGLE)
        else:
            gap_records.append({"after":str(keys[i]),"before":str(keys[j]),"distance":distance})
            if gap_policy=="error":
                raise ValueError(f"unexplained chain break {keys[i]} -> {keys[j]} ({distance:.2f} A); choose gap_policy explicitly")
            if gap_policy=="free":
                cterm[i]=True;nterm[j]=True
    # Reject possible head-to-tail cyclic segments instead of assigning false termini.
    for i in np.flatnonzero(nterm):
        j=i
        while following[j]>=0:
            j=int(following[j])
        if j!=i and np.linalg.norm(bb[j,2]-bb[i,0])<peptide_cutoff:
            raise ValueError("possible cyclic peptide: cyclic terminal chemistry is not supported")
    for keys_to_cap,mask in ((capped_n,nterm),(capped_c,cterm)):
        for value in keys_to_cap:
            key=ResidueKey.from_value(value)
            if key not in keys:
                raise KeyError(f"unknown capped terminus {key}")
            index=keys.index(key)
            if not mask[index]:
                raise ValueError(f"{key} is not a free terminus of the requested kind")
            mask[index]=False
    if not include_termini:
        nterm[:]=False;cterm[:]=False
    ss=np.zeros(n,bool)
    sg=[(i,l["SG"]) for i,l in enumerate(lookup) if "SG" in l and str(atoms.res_name[l["SG"]])=="CYS"]
    ss_pairs=[]
    if len(sg)>1:
        for a,b in cKDTree(atoms.coord[[x[1] for x in sg]]).query_pairs(2.3):
            i,ai=sg[a];j,aj=sg[b]
            ss_pairs.append((i,j));ss[i]=True;ss[j]=True
            bonds.add_bond(ai,aj,struc.BondType.SINGLE)
    if ss_pairs:
        degree=np.bincount(np.asarray(ss_pairs).reshape(-1),minlength=n)
        if np.any(degree>1):
            raise ValueError("ambiguous disulfide geometry: a cysteine has multiple nearby sulfur partners")
    if ss.any() and not freeze_disulfides:
        raise ValueError("disulfide detected: set freeze_disulfides=True to fix cysteine identities and suppress thiol titration")
    # Other explicit crosslinks cannot be represented by this fixed standard topology.
    if atoms.bonds is not None:
        atom_owner=np.repeat(np.arange(n),np.diff(starts))
        for a,b,_ in atoms.bonds.as_array():
            a,b=int(a),int(b);i,j=int(atom_owner[a]),int(atom_owner[b])
            if i==j:
                continue
            names={str(atoms.atom_name[a]),str(atoms.atom_name[b])}
            expected_peptide=(following[i]==j and atoms.atom_name[a]=="C" and atoms.atom_name[b]=="N") or \
                             (following[j]==i and atoms.atom_name[b]=="C" and atoms.atom_name[a]=="N")
            if expected_peptide:
                continue
            if names=={"SG"} and tuple(sorted((i,j))) in {tuple(sorted(pair)) for pair in ss_pairs}:
                continue
            # The reader can infer adjacent same-chain C--N links across a
            # missing segment or TER. Rebuild those from the explicit policy;
            # nonlocal or cross-chain C--N crosslinks remain unsupported.
            if names=={"C","N"} and abs(i-j)==1 and keys[i].chain==keys[j].chain:
                continue
            raise ValueError(f"unsupported covalent crosslink between {keys[i]} and {keys[j]}")
    atoms.bonds=bonds
    metadata.update(gaps=gap_records,disulfide_pairs=[[str(keys[i]),str(keys[j])] for i,j in ss_pairs],
                    free_n_termini=[str(keys[i]) for i in np.flatnonzero(nterm)],
                    free_c_termini=[str(keys[i]) for i in np.flatnonzero(cterm)],
                    explicit_break_after=[str(k) for k in sorted(breaks)])
    return Topology(atoms,keys,starts,native,chain_ids,ci,previous,following,nterm,cterm,ss,bb,metadata)
