import numpy as np
import pytest
from jaxpropka.reference import parse_pka,reference_charge,assert_baseline
from jaxpropka import ResidueKey

SAMPLE='''Synthetic parser fixture, NOT a measured reference run
SUMMARY OF THIS PREDICTION
   Group      pKa   model-pKa
 ASP   1 A     3.80    3.80
 LYS   1 B    10.50   10.50
 N+    1 A     8.00    8.00
 C-    1 B     3.20    3.20
----------------------------------------------------
'''


def test_summary_parser_preserves_mapped_chains_and_insertions():
    mapping={('A',1):ResidueKey('heavy_chain',42,'B'),('B',1):ResidueKey('light_chain',42,'B')}
    sites=parse_pka(SAMPLE,mapping)
    assert len(sites)==4
    assert sites[0].key!=sites[1].key
    assert sites[0].key.insertion=='B'
    assert sites[2].group=='NTERM';assert sites[3].group=='CTERM'


def test_parser_fails_on_empty_duplicate_or_unmapped_reference():
    with pytest.raises(ValueError):parse_pka('nothing')
    with pytest.raises(ValueError):parse_pka(SAMPLE,{('A',1):ResidueKey('A',1)})
    duplicate=SAMPLE.replace(' LYS   1 B    10.50   10.50',' ASP   1 A     3.80    3.80')
    with pytest.raises(ValueError):parse_pka(duplicate)


def test_reference_hh_charges():
    sites=parse_pka(SAMPLE)
    q=reference_charge(sites,[3.8,10.5])
    np.testing.assert_allclose(q[0,0],-.5)
    np.testing.assert_allclose(q[1,1],.5)
    assert np.all(np.diff(q,axis=0)<=0)


def test_baseline_rejects_drift_and_reference_changes():
    report={'reference':{'version':'parser-test'},'cache_fingerprint':'test','model_config':{},
            'sites':[{'residue':{'chain':'A','number':1,'insertion':''},'group':'ASP','surrogate_pka':4.,'reference_pka':3.8}],
            'surrogate_total_charge':[-1.,-2.],'reference_total_charge':[-1.1,-2.1],'ph':[7.,8.]}
    assert_baseline(report,report)
    bad={**report,'surrogate_total_charge':[-1.,-3.]}
    with pytest.raises(AssertionError):assert_baseline(bad,report)
    with pytest.raises(AssertionError):assert_baseline({**report,'reference':{'version':'other'}},report)
