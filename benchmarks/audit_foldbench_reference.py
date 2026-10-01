"""Diagnostic only: preserve PROPKA output and explain absent benchmark sites."""
import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import io
import json
import logging
from pathlib import Path
import tarfile
import traceback

import numpy as np
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile
from jaxpropka.cache import ResidueKey
from jaxpropka.geometry import build_candidates
from jaxpropka.parameters import THREE
from jaxpropka.reference import run_reference, write_reference_structure
from jaxpropka.topology import load_topology
from propka.run import single


def bond_path(start, end):
    pending = deque([(start, [start])])
    seen = {id(start)}
    while pending:
        atom, path = pending.popleft()
        if atom is end:
            return [{"residue": a.res_name, "chain": a.chain_id, "number": a.res_num,
                     "atom": a.name, "xyz": [a.x, a.y, a.z]} for a in path]
        if len(path) > 3:
            continue
        for neighbor in atom.bonded_atoms:
            if id(neighbor) not in seen:
                seen.add(id(neighbor)); pending.append((neighbor, path + [neighbor]))
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-numbering", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root.parent / "_runtime/jax-Ka/cuda12/benchmarks/foldbench-cpu/723644/summary.json").read_text())
    case = summary["cases"][args.case_index]
    directory = args.output / f"case-{args.case_index}"
    directory.mkdir(parents=True, exist_ok=True)
    result = {"pdb_id": case["pdb_id"], "case_index": args.case_index, "baseline_status": case["status"]}
    logger = logging.getLogger("propka")
    logs = io.StringIO()
    handler = logging.StreamHandler(logs)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with tarfile.open(root / "ground_truth_1522.tar") as archive:
            member = next(m for m in archive.getmembers() if Path(m.name).name == case["pdb_id"] + ".cif")
            source = archive.extractfile(member).read()
        assert hashlib.sha256(source).hexdigest() == case["source_sha256"]
        atoms = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(source.decode())), model=1,
                                  altloc="occupancy", use_author_fields=False, include_bonds=True)
        atoms = atoms[atoms.chain_id == case["chain_id"]]
        path = directory / "reference.pdb"
        if case.get("stage") == "topology":
            # No reconstruction or repair: JAX cannot construct these candidates.
            atoms = atoms[np.isin(atoms.res_name, THREE) & ~np.isin(atoms.element, ["H", "D"])]
            mapping = {(str(a.chain_id), int(a.res_id)): ResidueKey(str(a.chain_id), int(a.res_id), str(a.ins_code)) for a in atoms}
            pdb = PDBFile(); pdb.set_structure(atoms); pdb.write(path)
            result["input_kind"] = "observed canonical protein only, unrepaired backbone; not a JAX candidate export"
        else:
            topology = load_topology(atoms, gap_policy="free", freeze_disulfides=True, ignore_nonprotein=True)
            candidates = build_candidates(topology, missing_sidechain="template")
            mapping = write_reference_structure(topology, candidates, path)
            result["input_kind"] = "same native candidate export as benchmark"
            result["input_hash_matches_benchmark"] = hashlib.sha256(path.read_bytes()).hexdigest() == case["reference"]["input_sha256"]
            assert result["input_hash_matches_benchmark"]
        (directory / "mapping.json").write_text(json.dumps([
            {"export_chain": chain, "export_number": number, "original": asdict(key)}
            for (chain, number), key in mapping.items()], indent=2))
        reference = run_reference(path, backend="modern", mapping=mapping)
        result["propka_succeeded"] = True
        result["parsed_sites"] = len(reference.sites)
        result["reference"] = reference.provenance
        (directory / "reference.pka").write_text(reference.output_text)
        (directory / "reference.log").write_text(reference.stdout)
        # Introspect a separate run without changing parameters or group selection.
        molecule = single(path, write_pka=False)
        conf = molecule.conformations["AVR"]
        groups = []
        for group in conf.groups:
            key = mapping.get((group.atom.chain_id, group.atom.res_num))
            if key is None:
                continue
            kind = {"N+": "NTERM", "C-": "CTERM"}.get(group.residue_type, group.residue_type)
            if kind not in {"ASP", "GLU", "HIS", "CYS", "TYR", "LYS", "ARG", "NTERM", "CTERM"}:
                continue
            coupled = group.coupled_titrating_group
            groups.append({"residue": asdict(key), "group": kind, "label": group.label,
                           "type": group.type, "atom_name": group.atom.name,
                           "titratable": group.titratable, "pka": group.pka_value,
                           "cysteine_bridge": group.atom.cysteine_bridge,
                           "coupled_titrating_group": coupled.label if coupled else None,
                           "covalent_partners": [g.label for g in group.covalently_coupled_groups],
                           "inferred_covalent_paths": [bond_path(group.atom, g.atom) for g in group.covalently_coupled_groups],
                           "noncovalent_partners": [g.label for g in group.non_covalently_coupled_groups],
                           "summary_line": group.get_summary_string(molecule.version.parameters.remove_penalised_group)})
        result["remove_penalised_group"] = molecule.version.parameters.remove_penalised_group
        result["groups"] = groups
        parsed = {(s.key, s.group) for s in reference.sites}
        result["baseline_missing"] = []
        result["jax_invalid_reference"] = []
        for site in case.get("sites", []):
            key = ResidueKey(**site["residue"])
            matched = [g for g in groups if g["residue"] == site["residue"] and g["group"] == site["group"]]
            row = {"residue": site["residue"], "group": site["group"],
                   "present_in_repeated_parse": (key, site["group"]) in parsed,
                   "internal_groups": matched, "benchmark_reference_pka": site["reference_pka"]}
            if site["reference_missing"]:
                result["baseline_missing"].append(row)
            if site["tier"] == "invalid":
                result["jax_invalid_reference"].append(row)
        if args.probe_numbering:
            renumbered = directory / "globally_numbered.pdb"
            numbers = {pair: n for n, pair in enumerate(mapping, 1)}
            new_mapping = {(chain, numbers[chain, number]): key for (chain, number), key in mapping.items()}
            lines = []
            for line in path.read_text().splitlines(keepends=True):
                if line.startswith(("ATOM  ", "HETATM")):
                    number = numbers[line[21], int(line[22:26])]
                    line = line[:22] + f"{number:4d}" + line[26:]
                lines.append(line)
            renumbered.write_text("".join(lines))
            probe = run_reference(renumbered, backend="modern", mapping=new_mapping)
            (directory / "globally_numbered.pka").write_text(probe.output_text)
            probe_sites = {(s.key, s.group): s.pka for s in probe.sites}
            result["numbering_probe"] = [
                {"residue": s["residue"], "group": s["group"],
                 "pka": probe_sites.get((ResidueKey(**s["residue"]), s["group"]))}
                for s in result["baseline_missing"]]
        result["status"] = "complete"
    except Exception as error:
        result.update(status="error", error=repr(error), traceback=traceback.format_exc())
    finally:
        logger.removeHandler(handler)
        (directory / "introspection.log").write_text(logs.getvalue())
        result["iteration_limit_messages"] = [line for line in logs.getvalue().splitlines() if "did not converge" in line]
        (directory / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result["pdb_id"], result["status"], result.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
