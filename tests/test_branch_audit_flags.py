import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('branch_flags',Path(__file__).resolve().parents[1]/'benchmarks/branch_audit_flags.py')
flags=importlib.util.module_from_spec(spec);spec.loader.exec_module(flags)


def test_missing_evidence_does_not_imply_uniqueness():
    case=flags.annotate({'pdb_id':'x','sites':[{}]}, {})
    assert case['branch_audit']['status']=='not_assessed'
    assert case['sites'][0]['structure_branch_status']=='not_assessed'


def test_flag_preserves_tier_and_checks_provenance():
    case={'pdb_id':'x','source_sha256':'s','cache_fingerprint':'c','sites':[{'tier':'strict'}]}
    audit={**case,'points':[{'ph':7.,'weighted_occupancy_distances':[[0,.2],[.2,0]]}]}
    result=flags.annotate(case,{'x':audit})
    assert result['branch_audit']['status']=='known_competing_branches'
    assert result['sites'][0]['tier']=='strict'
    with pytest.raises(ValueError,match='mismatch'):
        flags.annotate({**case,'cache_fingerprint':'other'},{'x':audit})
