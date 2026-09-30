"""Frozen per-identity candidates from Biotite CCD; no sequence-dependent packing."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .parameters import (THREE,GROUPS,GROUP_AA,CENTERS,DONORS,ACCEPTORS,atom_volume)


@dataclass(frozen=True)
class PolarGeometry:
    donor: np.ndarray        # [D,3], heavy atoms
    point: np.ndarray        # [D,3], H position or heavy donor for PROPKA distances
    axis: np.ndarray         # [D,3], D->H unit vector; zero for nonangular donor
    angular: np.ndarray      # [D], apply directional H angle
    acceptor: np.ndarray     # [A,3], heavy acceptor atoms


@dataclass(frozen=True)
class Candidate:
    coordinates: dict[str,np.ndarray]  # all heavy coordinates, includes fixed backbone
    elements: dict[str,str]
    side_xyz: np.ndarray
    side_volume: np.ndarray
    polar: PolarGeometry


@dataclass(frozen=True)
class CandidateLibrary:
    residues: tuple[tuple[Candidate,...],...]
    centers: np.ndarray          # [N,9,3]
    polar: tuple[tuple[PolarGeometry,...],...]
    anchors: np.ndarray          # [N,3], native or virtual CB
    backbone_xyz: tuple[np.ndarray,...]
    backbone_volume: tuple[np.ndarray,...]
    backbone_nh: tuple[PolarGeometry,...]
    backbone_co: tuple[PolarGeometry,...]
    group_mask: np.ndarray       # [N,9]
    metadata: dict


def unit(vector):
    norm=np.linalg.norm(vector)
    if norm<1e-8 or not np.isfinite(norm):
        raise ValueError("degenerate geometry prevents a local frame / virtual hydrogen")
    return vector/norm


def frame(n,ca,c):
    x=unit(n-ca);z=unit(np.cross(x,c-ca));y=np.cross(z,x)
    return np.stack((x,y,z),axis=1)


def empty_polar():
    x=np.zeros((0,3),float)
    return PolarGeometry(x.copy(),x.copy(),x.copy(),np.zeros(0,bool),x.copy())


def polar_from_atoms(coordinates,donor_names,acceptor_names,adjacency,*,angular=False):
    donors=[];points=[];axes=[];flags=[]
    for name in donor_names:
        if name not in coordinates:
            continue
        d=coordinates[name]
        connected=[coordinates[x] for x in adjacency.get(name,()) if x in coordinates]
        if not connected:
            raise ValueError(f"no bonded heavy neighbor for donor {name}")
        direction=unit(sum((unit(d-x) for x in connected),start=np.zeros(3)))
        donors.append(d);axes.append(direction);flags.append(angular)
        points.append(d+direction if angular else d)
    acc=[coordinates[x] for x in acceptor_names if x in coordinates]
    return PolarGeometry(np.asarray(donors).reshape(-1,3),np.asarray(points).reshape(-1,3),
                         np.asarray(axes).reshape(-1,3),np.asarray(flags,bool),np.asarray(acc).reshape(-1,3))


def _template(name):
    from biotite.structure.info import residue
    atoms=residue(name)
    atoms=atoms[~np.isin(atoms.element,["H","D"])]
    coordinates={str(a.atom_name):np.asarray(a.coord,float) for a in atoms}
    elements={str(a.atom_name):str(a.element) for a in atoms}
    adjacency={x:[] for x in coordinates}
    for a,b,_ in atoms.bonds.as_array():
        an,bn=str(atoms.atom_name[int(a)]),str(atoms.atom_name[int(b)])
        adjacency[an].append(bn);adjacency[bn].append(an)
    return coordinates,elements,adjacency


def build_candidates(topology, *, preserve_native=True, missing_sidechain="error", overrides=None):
    """Build 20 fixed candidates/position from CCD heavy atoms and local frames.

    Native heavy atoms are preserved exactly. Non-native identities use ONE
    unoptimized CCD conformation, not a rotamer library. Supply `overrides` as
    {(residue_index, three_letter_identity): {atom_name: xyz, ...}} to replace
    complete sidechain coordinates with externally generated frozen candidates.
    Missing native atoms are rejected unless missing_sidechain='template'.
    Virtual N-H directions use bonded heavy atoms; this differs from PROPKA's
    protonator, especially for multiply hydrogenated donors and His tautomers.
    """
    if missing_sidechain not in ("error","template"):
        raise ValueError("missing_sidechain must be error or template")
    templates=[_template(name) for name in THREE]
    overrides=overrides or {}
    n=topology.n_residues
    for key in overrides:
        if not (isinstance(key,tuple) and len(key)==2 and isinstance(key[0],int)
                and 0<=key[0]<n and key[1] in THREE):
            raise ValueError("override keys must be (position_index, canonical_three_letter_name)")
    all_candidates=[];all_polar=[];centers=np.zeros((n,9,3));anchors=np.zeros((n,3))
    bbxyz=[];bbvol=[];nh=[];co=[];rebuilt=[];virtual_oxt=[]
    mask=np.ones((n,9),bool);mask[:,7]=topology.nterm;mask[:,8]=topology.cterm
    mask[topology.disulfide,:7]=False
    for i in range(n):
        atoms=topology.residue(i)
        native={str(a.atom_name):np.asarray(a.coord,float) for a in atoms}
        n_xyz,ca_xyz,c_xyz,o_xyz=topology.backbone[i]
        basis=frame(n_xyz,ca_xyz,c_xyz)
        backbone={name:native[name] for name in ("N","CA","C","O")}
        if topology.cterm[i]:
            if "OXT" in native:
                backbone["OXT"]=native["OXT"]
            else:
                direction=unit(ca_xyz-c_xyz);v=o_xyz-c_xyz
                backbone["OXT"]=c_xyz+2*np.dot(v,direction)*direction-v
                virtual_oxt.append(str(topology.keys[i]))
        coords_bb=np.asarray(list(backbone.values()))
        bbxyz.append(coords_bb)
        bbvol.append(np.asarray([atom_volume(name,"N" if name=="N" else "O" if name in ("O","OXT") else "C") for name in backbone]))
        row=[]
        for a,name in enumerate(THREE):
            template,elements,adjacency=templates[a]
            source_frame=frame(template["N"],template["CA"],template["C"])
            rotation=source_frame@basis.T
            coords={key:(xyz-template["CA"])@rotation+ca_xyz for key,xyz in template.items() if key!="OXT"}
            coords.update(backbone)
            side_names=[key for key in template if key not in ("N","CA","C","O","OXT")]
            if preserve_native and a==topology.native_index[i]:
                absent=[key for key in side_names if key not in native]
                if absent and missing_sidechain=="error":
                    raise ValueError(f"missing native sidechain atoms at {topology.keys[i]}: {absent}")
                if absent:
                    rebuilt.append({"residue":str(topology.keys[i]),"atoms":absent})
                coords.update({key:native[key] for key in side_names if key in native})
            if (i,name) in overrides:
                supplied=overrides[i,name]
                if set(supplied)!=set(side_names):
                    raise ValueError(f"override {(i,name)} must contain exactly sidechain atoms {side_names}")
                for key,xyz in supplied.items():
                    xyz=np.asarray(xyz,float)
                    if xyz.shape!=(3,) or not np.isfinite(xyz).all():
                        raise ValueError("override coordinates must be finite xyz vectors")
                    coords[key]=xyz
            side_xyz=np.asarray([coords[key] for key in side_names]).reshape(-1,3)
            volumes=np.asarray([atom_volume(key,elements[key]) for key in side_names])
            polar=polar_from_atoms(coords,DONORS.get(name,()),ACCEPTORS.get(name,()),adjacency,
                                   angular=name in ("HIS","ARG","ASN","GLN","TRP"))
            if topology.disulfide[i] and name=="CYS":
                polar=empty_polar()
            row.append(Candidate(coords,elements,side_xyz,volumes,polar))
        all_candidates.append(tuple(row))
        anchors[i]=native.get("CB",row[0].coordinates["CB"])
        site_polar=[row[a].polar for a in GROUP_AA]
        for g,a in enumerate(GROUP_AA):
            centers[i,g]=np.mean([row[a].coordinates[key] for key in CENTERS[THREE[a]]],axis=0)
        centers[i,7]=n_xyz
        centers[i,8]=(o_xyz+backbone.get("OXT",o_xyz))/2
        n_polar=polar_from_atoms(backbone,("N",),(),{"N":["CA"]},angular=False) if topology.nterm[i] else empty_polar()
        c_polar=polar_from_atoms(backbone,("O","OXT"),("O","OXT"),{"O":["C"],"OXT":["C"]}) if topology.cterm[i] else empty_polar()
        site_polar.extend((n_polar,c_polar));all_polar.append(tuple(site_polar))
        if topology.previous[i]>=0:
            prev_c=topology.backbone[topology.previous[i],2]
            direction=unit(unit(n_xyz-ca_xyz)+unit(n_xyz-prev_c))
            nh.append(PolarGeometry(n_xyz[None],(n_xyz+direction)[None],direction[None],np.ones(1,bool),np.zeros((0,3))))
        else:
            nh.append(empty_polar())
        # A terminal carboxylate is represented as an explicit site, not as an
        # additional neutral backbone acceptor (which would double count it).
        co.append(PolarGeometry(np.zeros((0,3)),np.zeros((0,3)),np.zeros((0,3)),np.zeros(0,bool),
                                o_xyz[None] if not topology.cterm[i] else np.zeros((0,3))))
    metadata={"candidate_geometry":"native heavy atoms + one local-frame CCD conformer per alternate identity",
              "preserve_native":preserve_native,"rebuilt_native":rebuilt,
              "virtual_oxt":virtual_oxt,"overridden_candidates":len(overrides),
              "hydrogens":"single virtual direction from bonded heavy neighbors"}
    return CandidateLibrary(tuple(all_candidates),centers,tuple(all_polar),anchors,
                            tuple(bbxyz),tuple(bbvol),tuple(nh),tuple(co),mask,metadata)
