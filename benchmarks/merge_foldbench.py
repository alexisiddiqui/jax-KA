#!/usr/bin/env python3
"""Merge per-case array reports, rejecting mismatched inputs and missing cases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from regress_foldbench import _write_checkpoint, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-cases", type=int)
    selection.add_argument("--case-indices", type=int, nargs="+",
                           help="explicit manifest indices for a targeted validation array")
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open()))
    indices = (sorted(set(args.case_indices)) if args.case_indices is not None else
               list(range(len(rows[:args.max_cases] if args.max_cases is not None else rows))))
    if not indices or any(index < 0 or index >= len(rows) for index in indices):
        parser.error("case selection must contain valid manifest indices")
    manifest_hash = sha256(args.manifest)
    merged = None
    missing = []
    for index in indices:
        row = rows[index]
        path = args.reports / f"report-{index}.json"
        if not path.exists():
            missing.append(index)
            continue
        report = json.loads(path.read_text())
        dataset = report["dataset"]
        if (dataset["manifest_sha256"] != manifest_hash or dataset["case_index"] != index
                or dataset["manifest_case_count"] != len(rows) or len(report["cases"]) != 1):
            raise ValueError(f"incompatible shard: {path}")
        case = report["cases"][0]
        if (case["pdb_id"], case["chain_id"]) != (row["pdb_id"], row["chain_id"]):
            raise ValueError(f"wrong case in {path}")
        if merged is None:
            merged = {"schema": report["schema"], "dataset": dict(dataset),
                      "reference": report["reference"], "policy": report["policy"],
                      "cases": [], "execution": {"kind": "per-case array", "tasks": []}}
            merged["dataset"].update(case_count=len(indices), case_index=None, case_indices=indices)
        if (dataset["archive_sha256"] != merged["dataset"]["archive_sha256"]
                or report["schema"] != merged["schema"] or report["policy"] != merged["policy"]
                or report["reference"] != merged["reference"]):
            raise ValueError(f"inconsistent provenance in {path}")
        merged["cases"].append(case)
        merged["execution"]["tasks"].append({"case_index": index, **report["execution"],
                                              "timing_seconds": report["timing_seconds"]})
    if merged is None:
        raise ValueError("no case reports found")
    merged["execution"]["missing_case_indices"] = missing
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_checkpoint(args.output, merged)
    print(json.dumps(merged["summary"], indent=2))
    if missing:
        raise SystemExit(f"Incomplete array: missing case indices {missing}")


if __name__ == "__main__":
    main()
