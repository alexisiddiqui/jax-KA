"""Array task selection must cover the manifest exactly without aliasing indices."""
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "foldbench", Path(__file__).resolve().parents[1] / "benchmarks/regress_foldbench.py"
)
foldbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foldbench)


def test_array_rows_are_disjoint_and_cover_manifest():
    rows = [{"pdb_id": str(i), "chain_id": "A"} for i in range(334)]
    selected = [foldbench._select_case(rows, i) for i in range(len(rows))]
    assert all(len(task) == 1 for task in selected)
    assert [task[0] for task in selected] == rows
    assert foldbench._select_case(rows, None) == rows


@pytest.mark.parametrize("index", [-1, 2, 334])
def test_invalid_index_is_rejected(index):
    with pytest.raises(ValueError, match="outside manifest"):
        foldbench._select_case([{"pdb_id": "a"}, {"pdb_id": "b"}], index)


def test_unsupported_inputs_are_separate_from_solver_failures():
    error=ValueError("missing backbone at A:1: ['CA']")
    assert foldbench._failure_status('topology',error,True)=='excluded_input'
    assert foldbench._failure_status('topology',error,False)=='failed'
    assert foldbench._failure_status('site_report',error,True)=='failed'
    summary=foldbench._summary([{'status':'excluded_input','stage':'topology'}],1)
    assert summary['excluded_input_cases']==1
    assert summary['failed_cases']==0
