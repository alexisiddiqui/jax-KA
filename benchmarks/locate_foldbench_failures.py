#!/usr/bin/env python3
"""Measure observed-structure SASA around recorded titration sites; no solver rerun."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from biotite.structure import sasa
from biotite.structure.io import pdbx
from jaxpropka.parameters import THREE
from jaxpropka.topology import load_topology


SITE_ATOMS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"),
              "HIS": ("ND1", "NE2"), "CYS": ("SG",), "TYR": ("OH",),
              "LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2"),
              "NTERM": ("N",), "CTERM": ("O", "OXT")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=10)
    args = parser.parse_args()
    baseline = json.loads(args.report.read_text())
    cases = baseline["cases"][args.shard::args.shards]
    results = []
    errors = []
    with tarfile.open(args.archive) as archive:
        members = {Path(member.name).name: member for member in archive.getmembers()}
        for case in cases:
            if case.get("stage") == "topology":
                continue
            try:
                with archive.extractfile(members[case["pdb_id"] + ".cif"]) as handle:
                    source = handle.read()
                if hashlib.sha256(source).hexdigest() != case["source_sha256"]:
                    raise ValueError("structure hash differs from benchmark")
                atoms = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(source.decode())),
                    model=1, altloc="occupancy", use_author_fields=False, include_bonds=True)
                atoms = atoms[np.asarray(atoms.chain_id) == case["chain_id"]]
                topology = load_topology(atoms, gap_policy="free", freeze_disulfides=True,
                                         ignore_nonprotein=True)
                area = sasa(topology.atoms, probe_radius=1.4, point_number=1000)
                bad = set()
                complete = True
                if case.get("stage") == "grid_pka":
                    bad = set(map(tuple, ast.literal_eval(case["error"].split(": ", 1)[1])))
                    complete = len(bad) < 20
                sites = []
                matched = set()
                for i, key in enumerate(topology.keys):
                    start, stop = topology.starts[i:i+2]
                    residue = topology.residue(i)
                    group = THREE[topology.native_index[i]]
                    groups = [group] if group in SITE_ATOMS and not topology.disulfide[i] else []
                    if topology.nterm[i]:
                        groups.append("NTERM")
                    if topology.cterm[i]:
                        groups.append("CTERM")
                    for group in groups:
                        names = SITE_ATOMS[group]
                        present = set(map(str, residue.atom_name))
                        mask = np.isin(residue.atom_name, names)
                        invalid = (str(key), group) in bad
                        if invalid:
                            matched.add((str(key), group))
                        sites.append({
                            "residue": str(key), "group": group,
                            "invalid": invalid if invalid or complete else None,
                            "functional_atoms_complete": set(names) <= present,
                            "functional_sasa": float(area[start:stop][mask].sum()),
                            "residue_sasa": float(area[start:stop].sum()),
                            "on_terminal_residue": bool(topology.nterm[i] or topology.cterm[i]),
                            "internal_gap_terminus": bool(
                                group == "NTERM" and i > 0 and key.chain == topology.keys[i-1].chain
                                or group == "CTERM" and i+1 < topology.n_residues and key.chain == topology.keys[i+1].chain),
                        })
                if matched != bad:
                    raise ValueError(f"unmapped invalid sites: {bad - matched}")
                results.append({"pdb_id": case["pdb_id"], "failure_list_complete": complete,
                                "baseline_status": case["status"], "sites": sites})
                print(f"Measured {case['pdb_id']}: {len(sites)} sites", flush=True)
            except Exception as error:
                errors.append({"pdb_id": case["pdb_id"], "error": repr(error)})
                print(f"ERROR {case['pdb_id']}: {error!r}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"method": "Observed heavy-atom coordinates; ProtOr SASA, 1.4 A probe, 1000 points/atom; isolated benchmark chain", "cases": results, "errors": errors}, indent=2) + "\n")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
