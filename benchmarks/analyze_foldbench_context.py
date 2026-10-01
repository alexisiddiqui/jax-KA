#!/usr/bin/env python3
"""Measure omitted structural context around experimental FoldBench sites."""
import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from biotite.structure.io import pdbx
from scipy.spatial import cKDTree


CANONICAL = frozenset({"ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS",
    "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
    "THR", "VAL", "TRP", "TYR"})
WATER = frozenset({"HOH", "WAT", "H2O", "DOD"})
SITE_ATOMS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"),
    "HIS": ("ND1", "NE2"), "CYS": ("SG",), "TYR": ("OH",),
    "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"), "NTERM": ("N",),
    "CTERM": ("O", "OXT")}
NONMETALS = frozenset({"H", "D", "C", "N", "O", "P", "S", "SE", "F",
                       "CL", "BR", "I"})


def nearest(atoms, target_xyz, mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    distance, local = cKDTree(atoms.coord[indices]).query(target_xyz, k=1)
    q = int(np.argmin(distance))
    atom = atoms[indices[int(local[q])]]
    return {"distance": float(distance[q]), "chain": str(atom.chain_id),
            "residue_number": int(atom.res_id), "residue_name": str(atom.res_name),
            "atom_name": str(atom.atom_name), "element": str(atom.element)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=10)
    args = parser.parse_args()
    reports = []
    for path in args.reports.glob("report-*.json"):
        index = int(path.stem.split("-")[-1])
        if index % args.shards == args.shard:
            case = json.loads(path.read_text())["cases"][0]
            if case["status"] == "passed":
                reports.append(case)
    rows = []
    with tarfile.open(args.archive) as archive:
        members = {Path(member.name).name: member for member in archive.getmembers()}
        for case in reports:
            raw = archive.extractfile(members[case["pdb_id"] + ".cif"]).read()
            atoms = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(raw.decode())),
                model=1, altloc="occupancy", use_author_fields=False, include_bonds=True)
            names = np.asarray(atoms.res_name, str)
            chains = np.asarray(atoms.chain_id, str)
            elements = np.char.upper(np.asarray(atoms.element, str))
            metal = ~np.isin(elements, list(NONMETALS))
            nonprotein = ~np.isin(names, list(CANONICAL | WATER))
            other_protein = np.isin(names, list(CANONICAL)) & (chains != case["chain_id"])
            for site in case["sites"]:
                residue = site["residue"]
                mask = ((chains == residue["chain"]) &
                        (atoms.res_id == residue["number"]) &
                        (np.asarray(atoms.ins_code, str) == residue["insertion"]) &
                        np.isin(atoms.atom_name, SITE_ATOMS[site["group"]]))
                target = atoms.coord[mask]
                if not len(target):
                    continue
                rows.append({"pdb_id": case["pdb_id"], "residue": residue,
                    "group": site["group"], "delta": site["delta"],
                    "nearest_metal": nearest(atoms, target, metal),
                    "nearest_nonprotein": nearest(atoms, target, nonprotein),
                    "nearest_other_protein_chain": nearest(atoms, target, other_protein)})
            print(f"Context {case['pdb_id']}: {len(case['sites'])} sites", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sites": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
