#!/usr/bin/env python3
"""Checkpointed PROPKA regression sweep over FoldBench protein monomers."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import traceback

import numpy as np


CAP_RESIDUES = frozenset({"ACE", "NME", "NH2", "FOR"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_case(item: tuple[str, str, str, int, int, tuple[int, int, int]]) -> dict:
    pdb_id, chain_id, cif_name, steps, grid_points, bucket_multiple = item
    stage = "load"
    try:
        import jax
        from biotite.structure.io import pdbx
        from jaxpropka import GROUPS, ModelConfig
        from jaxpropka.batching import pack_inputs
        from jaxpropka.geometry import build_candidates
        from jaxpropka.model import curve_kernel, grid_pka_kernel, one_hot
        from jaxpropka.parameters import GROUP_AA
        from jaxpropka.precompute import build_cache
        from jaxpropka.reference import reference_charge, run_reference, write_reference_structure
        from jaxpropka.topology import load_topology

        cif_path = Path(cif_name)
        source_sha256 = sha256(cif_path)
        cif = pdbx.CIFFile.read(cif_path)
        atoms = pdbx.get_structure(
            cif,
            model=1,
            altloc="occupancy",
            use_author_fields=False,
            include_bonds=True,
        )
        atoms = atoms[np.asarray(atoms.chain_id) == chain_id]
        if len(atoms) == 0:
            raise KeyError(f"label-asym chain {chain_id!r} is absent")

        stage = "topology"
        topology = load_topology(
            atoms,
            gap_policy="free",
            freeze_disulfides=True,
            ignore_nonprotein=True,
        )
        omitted = {str(name) for name in topology.metadata["omitted_nonprotein"]}
        caps = sorted(omitted & CAP_RESIDUES)
        if caps:
            raise ValueError(f"unsupported terminal cap chemistry: {caps}")
        topology.metadata.update(
            source_name=cif_path.name,
            source_sha256=source_sha256,
            foldbench_pdb_id=pdb_id,
            foldbench_label_asym_id=chain_id,
        )

        stage = "candidate_geometry"
        candidates = build_candidates(topology, missing_sidechain="template")
        stage = "cache"
        cache = build_cache(topology, candidates)
        config = ModelConfig(steps=steps)
        arrays, probabilities, n_residues = pack_inputs(
            cache, one_hot(cache.native_index), bucket_multiple=bucket_multiple
        )
        bucket_capacity = (probabilities.shape[0], arrays["env_neighbors"].shape[1],
                           arrays["neighbors"].shape[1])
        arrays, probabilities = jax.device_put((arrays, probabilities))

        stage = "reference"
        with tempfile.TemporaryDirectory(prefix="jaxpropka-foldbench-reference-") as directory:
            reference_path = Path(directory) / "structure.pdb"
            mapping = write_reference_structure(topology, candidates, reference_path)
            reference = run_reference(reference_path, backend="modern", mapping=mapping)

        expected: list[tuple[int, str]] = []
        for index, aa_index in enumerate(cache.native_index):
            for group_index in np.flatnonzero(GROUP_AA == aa_index):
                if cache.group_mask[index, group_index]:
                    expected.append((index, GROUPS[int(group_index)]))
            for group_index in (7, 8):
                if cache.group_mask[index, group_index]:
                    expected.append((index, GROUPS[group_index]))
        if not expected:
            raise ValueError("no active physical titratable sites")

        stage = "grid_pka"
        grid = np.linspace(config.ph_min, config.ph_max, grid_points, dtype=np.float32)
        grid_result = jax.device_get(grid_pka_kernel(arrays, probabilities, grid, config=config))
        grid_values = np.asarray(grid_result.value)[:n_residues]
        grid_valid = np.asarray(grid_result.valid)[:n_residues]
        invalid = [
            (str(cache.keys[index]), group)
            for index, group in expected
            if not grid_valid[index, GROUPS.index(group)]
        ]
        if invalid:
            raise ValueError(f"invalid grid midpoint sites: {invalid[:20]}")

        reference_lookup = {(site.key, site.group): site for site in reference.sites}
        missing = [
            (str(cache.keys[index]), group)
            for index, group in expected
            if (cache.keys[index], group) not in reference_lookup
        ]
        if missing:
            raise ValueError(f"reference is missing required sites: {missing[:20]}")

        stage = "charge_curves"
        ph = np.arange(0.0, 15.0, dtype=np.float32)
        curves = jax.device_get(curve_kernel(arrays, probabilities, ph, config=config))
        if not np.all(curves.converged):
            raise ValueError("occupancy solver did not converge over reference pH grid")

        rows = []
        reference_sites = []
        for index, group in expected:
            group_index = GROUPS.index(group)
            reference_site = reference_lookup[cache.keys[index], group]
            value = float(grid_values[index, group_index])
            rows.append(
                {
                    "residue": asdict(cache.keys[index]),
                    "group": group,
                    "surrogate_pka": value,
                    "reference_pka": reference_site.pka,
                    "delta": value - reference_site.pka,
                }
            )
            reference_sites.append(reference_site)

        actual_charge = np.stack(
            [
                curves.site_charge[:, index, GROUPS.index(group)]
                for index, group in expected
            ],
            axis=-1,
        )
        expected_charge = reference_charge(reference_sites, ph)
        delta = np.asarray([row["delta"] for row in rows])
        site_charge_delta = actual_charge - expected_charge
        total_charge_delta = actual_charge.sum(-1) - expected_charge.sum(-1)
        expected_keys = {(cache.keys[index], group) for index, group in expected}
        extras = [
            {"residue": asdict(site.key), "group": site.group}
            for site in reference.sites
            if (site.key, site.group) not in expected_keys
        ]
        return {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "status": "passed",
            "source_sha256": source_sha256,
            "n_residues": cache.n_residues,
            "bucket_capacity": dict(zip(("N", "Ke", "Kc"), bucket_capacity)),
            "n_sites": len(rows),
            "omitted_nonprotein": sorted(omitted),
            "gap_count": len(topology.metadata["gaps"]),
            "disulfide_pair_count": len(topology.metadata["disulfide_pairs"]),
            "cache_fingerprint": cache.fingerprint(),
            "reference": reference.provenance,
            "midpoint_method": {
                "kind": "shared_grid_linear_interpolation",
                "minimum": float(grid[0]),
                "maximum": float(grid[-1]),
                "points": grid_points,
            },
            "sites": rows,
            "excluded_reference_sites": extras,
            "metrics": {
                "pka_mae": float(np.abs(delta).mean()),
                "pka_rmse": float(np.sqrt(np.mean(delta**2))),
                "pka_max_abs": float(np.abs(delta).max()),
                "site_charge_rmse": float(np.sqrt(np.mean(site_charge_delta**2))),
                "total_charge_rmse": float(np.sqrt(np.mean(total_charge_delta**2))),
            },
        }
    except Exception as error:
        return {
            "pdb_id": pdb_id,
            "chain_id": chain_id,
            "status": "failed",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=5),
        }


def _summary(cases: list[dict], total: int) -> dict:
    passed = [case for case in cases if case["status"] == "passed"]
    failed = [case for case in cases if case["status"] == "failed"]
    deltas = np.asarray(
        [site["delta"] for case in passed for site in case["sites"]], dtype=float
    )
    by_stage: dict[str, int] = {}
    for case in failed:
        by_stage[case["stage"]] = by_stage.get(case["stage"], 0) + 1
    result = {
        "total_manifest_cases": total,
        "completed_cases": len(cases),
        "passed_cases": len(passed),
        "failed_cases": len(failed),
        "failures_by_stage": dict(sorted(by_stage.items())),
        "compared_sites": int(deltas.size),
    }
    if deltas.size:
        result["pooled_pka_mae"] = float(np.abs(deltas).mean())
        result["pooled_pka_rmse"] = float(np.sqrt(np.mean(deltas**2)))
        result["pooled_pka_max_abs"] = float(np.abs(deltas).max())
    return result


def _write_checkpoint(path: Path, document: dict) -> None:
    document["summary"] = _summary(document["cases"], document["dataset"]["case_count"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/foldbench_protein_regression.json"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--grid-points", type=int, default=145)
    parser.add_argument("--bucket-multiple", type=int, nargs=3, default=(64, 16, 16),
                        metavar=("N", "Ke", "Kc"),
                        help="round residue/environment/pair capacities up to these multiples")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.steps < 1 or args.grid_points < 2:
        parser.error("workers, steps and grid-points must be positive")
    if any(value < 1 for value in args.bucket_multiple):
        parser.error("bucket multiples must be positive")

    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    if not rows or set(rows[0]) != {"pdb_id", "chain_id"}:
        raise ValueError("expected FoldBench monomer manifest columns: pdb_id,chain_id")
    if len({row["pdb_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate pdb_id in FoldBench monomer manifest")

    document = {
        "schema": 1,
        "dataset": {
            "name": "FoldBench protein monomers",
            "manifest_sha256": sha256(args.manifest),
            "archive_sha256": sha256(args.archive),
            "case_count": len(rows),
        },
        "reference": {"backend": "modern", "required_version": "3.5.1"},
        "policy": {
            "chain_identifier": "FoldBench label_asym_id",
            "altloc": "occupancy",
            "gap_policy": "free",
            "freeze_disulfides": True,
            "ignore_nonprotein": True,
            "missing_sidechain": "template",
            "terminal_caps": "fail",
            "steps": args.steps,
            "grid_points": args.grid_points,
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous.get("dataset") != document["dataset"]:
            raise ValueError("existing checkpoint belongs to a different FoldBench dataset")
        if previous.get("policy") != document["policy"]:
            raise ValueError("existing checkpoint uses a different regression policy")
        document["cases"] = previous.get("cases", [])
    completed = {case["pdb_id"] for case in document["cases"]}
    pending_rows = [row for row in rows if row["pdb_id"] not in completed]

    with tempfile.TemporaryDirectory(prefix="jaxpropka-foldbench-") as directory:
        root = Path(directory)
        with tarfile.open(args.archive) as archive:
            member_by_name = {Path(member.name).name: member for member in archive.getmembers()}
            for row in pending_rows:
                filename = f"{row['pdb_id']}.cif"
                member = member_by_name.get(filename)
                if member is None:
                    raise FileNotFoundError(f"{filename} is absent from {args.archive}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member {member.name}")
                (root / filename).write_bytes(source.read())

        work = [
            (row["pdb_id"], row["chain_id"], str(root / f"{row['pdb_id']}.cif"),
             args.steps, args.grid_points, tuple(args.bucket_multiple))
            for row in pending_rows
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_case, item): item[0] for item in work}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                document["cases"].append(result)
                document["cases"].sort(key=lambda case: next(
                    index for index, row in enumerate(rows) if row["pdb_id"] == case["pdb_id"]
                ))
                _write_checkpoint(args.output, document)
                summary = document["summary"]
                print(
                    f"[{summary['completed_cases']}/{len(rows)}] {result['pdb_id']}: "
                    f"{result['status']} (passed={summary['passed_cases']}, failed={summary['failed_cases']})",
                    flush=True,
                )

    _write_checkpoint(args.output, document)


if __name__ == "__main__":
    main()
