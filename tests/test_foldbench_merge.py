"""Sparse pilot merges must preserve selection and reject mixed provenance."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import merge_foldbench as merge


@pytest.mark.parametrize("mismatch", [None, "schema", "policy"])
def test_targeted_merge(tmp_path, monkeypatch, mismatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("pdb_id,chain_id\na,A\nb,A\nc,A\n")
    output = tmp_path / "summary.json"
    for index, pdb_id in [(0, "a"), (2, "c")]:
        report = {"schema": 2, "dataset": {"manifest_sha256": merge.sha256(manifest),
                   "archive_sha256": "archive", "case_index": index, "case_count": 1,
                   "manifest_case_count": 3}, "reference": {"version": "test"},
                  "policy": {"version": "test"}, "execution": {}, "timing_seconds": {},
                  "cases": [{"pdb_id": pdb_id, "chain_id": "A", "status": "partial",
                    "site_policy": {}, "sites": [{"delta": 1., "tier": "strict"},
                                                   {"delta": None, "tier": "invalid"}]}]}
        if index == 2 and mismatch:
            report[mismatch] = 1 if mismatch == "schema" else {"version": "different"}
        (tmp_path / f"report-{index}.json").write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["merge", "--manifest", str(manifest), "--reports", str(tmp_path),
                                     "--output", str(output), "--case-indices", "0", "2"])
    if mismatch:
        with pytest.raises(ValueError, match="inconsistent provenance"):
            merge.main()
    else:
        merge.main()
        merged = json.loads(output.read_text())
        assert merged["dataset"]["case_indices"] == [0, 2]
        assert merged["dataset"]["case_count"] == 2
        assert merged["summary"]["partial_cases"] == 2
        assert merged["summary"]["compared_sites"] == 2
        assert merged["summary"]["site_report"]["overall"]["coverage"] == .5
