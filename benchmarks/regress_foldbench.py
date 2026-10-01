#!/usr/bin/env python3
"""Checkpointed PROPKA regression sweep over FoldBench protein monomers."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
import traceback

import numpy as np


CAP_RESIDUES = frozenset({"ACE", "NME", "NH2", "FOR"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_case(item: tuple[str, str, str], steps: int, grid_points: int,
              bucket_multiple: tuple, *, site_report: bool = False) -> dict:
    pdb_id, chain_id, cif_name = item
    result = {"pdb_id": pdb_id, "chain_id": chain_id}
    case_started = time.perf_counter()
    stage_started = case_started
    stage = "imports"
    stage_seconds: dict[str, float] = {}

    def enter(next_stage: str) -> None:
        nonlocal stage, stage_started
        now = time.perf_counter()
        stage_seconds[stage] = now - stage_started
        stage = next_stage
        stage_started = now

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

        enter("load")
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
        full_atoms = atoms
        result["source_sha256"] = source_sha256
        if site_report:
            from foldbench_site_report import POLICY
            result["site_policy"] = POLICY
            from analyze_foldbench_context import CANONICAL, WATER
            result["full_model_nonprotein_residue_names"] = sorted(
                set(map(str, atoms.res_name)) - CANONICAL - WATER)
        atoms = atoms[np.asarray(atoms.chain_id) == chain_id]
        if len(atoms) == 0:
            raise KeyError(f"label-asym chain {chain_id!r} is absent")

        enter("topology")
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

        enter("candidate_geometry")
        candidates = build_candidates(topology, missing_sidechain="template")
        enter("cache")
        cache = build_cache(topology, candidates)
        result.update({
            "source_sha256": source_sha256,
            "n_residues": cache.n_residues,
            "required_capacity": dict(zip(("N", "Ke", "Kc"),
                (cache.n_residues, cache.env_neighbors.shape[1], cache.neighbors.shape[1]))),
            "omitted_nonprotein": sorted(omitted),
            "gap_count": len(topology.metadata["gaps"]),
            "disulfide_pair_count": len(topology.metadata["disulfide_pairs"]),
        })
        enter("pack_transfer")
        config = ModelConfig(steps=steps)
        arrays, probabilities, n_residues = pack_inputs(
            cache, one_hot(cache.native_index), bucket_multiple=bucket_multiple
        )
        result["bucket_capacity"] = dict(zip(("N", "Ke", "Kc"),
            (probabilities.shape[0], arrays["env_neighbors"].shape[1], arrays["neighbors"].shape[1])))
        arrays, probabilities = jax.device_put((arrays, probabilities))
        jax.block_until_ready((arrays, probabilities))

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

        if site_report:
            from foldbench_site_report import evaluate
            enter("site_report")
            result["numerical_backend"] = {"jax_version": jax.__version__,
                                            "backend": jax.default_backend()}
            result.update(evaluate(arrays, probabilities, cache, topology, candidates,
                                   full_atoms, expected, config, grid_points))
            enter("complete")
            result["timing_seconds"] = {"stages": stage_seconds,
                                        "total": time.perf_counter() - case_started}
            return result

        enter("grid_pka")
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

        # Do not launch the external reference process for a case whose JAX
        # result is already unusable. This preserves successful-case results
        # while making invalid cases substantially cheaper.
        enter("reference")
        with tempfile.TemporaryDirectory(prefix="jaxpropka-foldbench-reference-") as directory:
            reference_path = Path(directory) / "structure.pdb"
            mapping = write_reference_structure(topology, candidates, reference_path)
            reference = run_reference(reference_path, backend="modern", mapping=mapping)

        reference_lookup = {(site.key, site.group): site for site in reference.sites}
        missing = [
            (str(cache.keys[index]), group)
            for index, group in expected
            if (cache.keys[index], group) not in reference_lookup
        ]
        if missing:
            raise ValueError(f"reference is missing required sites: {missing[:20]}")

        enter("charge_curves")
        ph = np.arange(0.0, 15.0, dtype=np.float32)
        curves = jax.device_get(curve_kernel(arrays, probabilities, ph, config=config))
        if not np.all(curves.converged):
            raise ValueError("occupancy solver did not converge over reference pH grid")

        enter("metrics")
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
        enter("complete")
        return {
            **result,
            "status": "passed",
            "n_sites": len(rows),
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
            "timing_seconds": {
                "stages": stage_seconds,
                "total": time.perf_counter() - case_started,
            },
        }
    except Exception as error:
        now = time.perf_counter()
        stage_seconds[stage] = now - stage_started
        return {
            **result,
            "status": "failed",
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=5),
            "timing_seconds": {
                "stages": stage_seconds,
                "total": now - case_started,
            },
        }


def _summary(cases: list[dict], total: int) -> dict:
    passed = [case for case in cases if case["status"] == "passed"]
    failed = [case for case in cases if case["status"] == "failed"]
    deltas = np.asarray(
        [site["delta"] for case in cases for site in case.get("sites", [])
         if site.get("delta") is not None], dtype=float
    )
    by_stage: dict[str, int] = {}
    for case in failed:
        by_stage[case["stage"]] = by_stage.get(case["stage"], 0) + 1
    result = {
        "total_manifest_cases": total,
        "completed_cases": len(cases),
        "passed_cases": len(passed),
        "failed_cases": len(failed),
        "partial_cases": sum(case["status"] == "partial" for case in cases),
        "failures_by_stage": dict(sorted(by_stage.items())),
        "compared_sites": int(deltas.size),
    }
    if deltas.size:
        result["pooled_pka_mae"] = float(np.abs(deltas).mean())
        result["pooled_pka_rmse"] = float(np.sqrt(np.mean(deltas**2)))
        result["pooled_pka_max_abs"] = float(np.abs(deltas).max())
    if any("site_policy" in case for case in cases):
        from foldbench_site_report import summarize
        result["site_report"] = summarize(cases)
    return result


def _write_checkpoint(path: Path, document: dict) -> None:
    document["summary"] = _summary(document["cases"], document["dataset"]["case_count"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    os.replace(temporary, path)


def _select_case(rows: list[dict], index: int | None) -> list[dict]:
    if index is None:
        return rows
    if not 0 <= index < len(rows):
        raise ValueError(f"case index {index} outside manifest of {len(rows)} cases")
    return [rows[index]]


def main() -> None:
    run_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/foldbench_protein_regression.json"))
    parser.add_argument("--case-index", type=int,
                        help="run one zero-based manifest row (for a Slurm array)")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--grid-points", type=int, default=145)
    parser.add_argument("--site-report", action="store_true",
                        help="report partial site coverage, adaptive convergence and refined flagged crossings")
    parser.add_argument("--bucket-multiple", type=int, nargs=3, default=(1, 1, 1),
                        metavar=("N", "Ke", "Kc"),
                        help="round residue/environment/pair capacities up to these multiples")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.grid_points < 2:
        parser.error("steps must be positive and grid-points must be at least two")
    if any(value < 1 for value in args.bucket_multiple):
        parser.error("bucket multiples must be positive")

    rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))
    if args.max_cases is not None:
        rows = rows[: args.max_cases]
    if not rows or set(rows[0]) != {"pdb_id", "chain_id"}:
        raise ValueError("expected FoldBench monomer manifest columns: pdb_id,chain_id")
    if len({row["pdb_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate pdb_id in FoldBench monomer manifest")
    manifest_case_count = len(rows)
    rows = _select_case(rows, args.case_index)

    archive_hash_started = time.perf_counter()
    archive_sha256 = sha256(args.archive)
    archive_sha256_seconds = time.perf_counter() - archive_hash_started
    document = {
        "schema": 2 if args.site_report else 1,
        "dataset": {
            "name": "FoldBench protein monomers",
            "manifest_sha256": sha256(args.manifest),
            "archive_sha256": archive_sha256,
            "case_count": len(rows),
            "manifest_case_count": manifest_case_count,
            "case_index": args.case_index,
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
            "bucket_multiple": list(args.bucket_multiple),
        },
        "timing_seconds": {
            "archive_sha256": archive_sha256_seconds,
            "archive_extract": 0.0,
            "case_execution": 0.0,
            "total": time.perf_counter() - run_started,
        },
        "cases": [],
        "execution": {"processes": 1, "hostname": os.uname().nodename,
                      "cpu_affinity_count": len(os.sched_getaffinity(0)),
                      "jax_platforms": os.environ.get("JAX_PLATFORMS", "default"),
                      "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                      "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")},
    }
    if args.site_report:
        from foldbench_site_report import POLICY
        document["policy"]["site_report"] = POLICY
        document["policy"]["implementation_sha256"] = {
            name: sha256(Path(__file__).with_name(name))
            for name in ("regress_foldbench.py", "foldbench_site_report.py")}
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
        extract_started = time.perf_counter()
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
        document["timing_seconds"]["archive_extract"] = time.perf_counter() - extract_started

        work = [
            (row["pdb_id"], row["chain_id"], str(root / f"{row['pdb_id']}.cif"))
            for row in pending_rows
        ]
        case_execution_started = time.perf_counter()
        order = {row["pdb_id"]: index for index, row in enumerate(rows)}

        def record(result: dict) -> None:
            document["cases"].append(result)
            document["cases"].sort(key=lambda case: order[case["pdb_id"]])
            document["timing_seconds"]["case_execution"] = time.perf_counter() - case_execution_started
            document["timing_seconds"]["total"] = time.perf_counter() - run_started
            _write_checkpoint(args.output, document)
            summary = document["summary"]
            print(
                f"[{summary['completed_cases']}/{len(rows)}] {result['pdb_id']}: "
                f"{result['status']} (passed={summary['passed_cases']}, failed={summary['failed_cases']})",
                flush=True,
            )

        for item in work:
            options = {"site_report": True} if args.site_report else {}
            record(_run_case(item, args.steps, args.grid_points, tuple(args.bucket_multiple), **options))

    document["timing_seconds"]["case_execution"] = time.perf_counter() - case_execution_started
    document["timing_seconds"]["total"] = time.perf_counter() - run_started
    _write_checkpoint(args.output, document)


if __name__ == "__main__":
    main()
