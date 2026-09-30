"""Report pipeline tests against analytic independent-site values, NOT PROPKA runs."""
import numpy as np
import pytest
from jaxpropka import TitrationModel,ModelConfig,GROUPS
from jaxpropka.synthetic import synthetic_cache
from jaxpropka.parameters import GROUP_AA,MODEL_PKA
from jaxpropka.reference import ReferenceRun,ReferenceSite,compare_reference,assert_baseline


def independent_case():
    model=TitrationModel(synthetic_cache(n=3,chains=2),
        ModelConfig(desolv_scale=0,coulomb_scale=0,hbond_scale=0))
    sites=[]
    for i,a in enumerate(model.cache.native_index):
        types=list(np.flatnonzero(GROUP_AA==a))
        types.extend(g for g in (7,8) if model.cache.group_mask[i,g])
        sites.extend(ReferenceSite(model.cache.keys[i],GROUPS[g],float(MODEL_PKA[g])) for g in types)
    return model,ReferenceRun(tuple(sites),{'backend':'analytic-independent-site-test'},'NOT external PROPKA','',0.)


def test_complete_report_matches_analytic_independent_site_baseline():
    model,reference=independent_case()
    report=compare_reference(model,model.native_probabilities,reference,ph=[2.,7.,12.])
    assert report['n_sites']==len(reference.sites)
    assert report['metrics']['pka_mae']<2e-5
    assert report['metrics']['total_charge_rmse']<2e-5
    assert_baseline(report,report)
    assert {r['residue']['chain'] for r in report['sites']}==set(model.cache.chain_ids)


def test_reference_report_rejects_missing_sites_and_soft_sequence():
    model,reference=independent_case()
    missing=ReferenceRun(reference.sites[1:],reference.provenance,'','',0.)
    with pytest.raises(ValueError,match='missing required sites'):
        compare_reference(model,model.native_probabilities,missing)
    with pytest.raises(ValueError,match='same HARD sequence'):
        compare_reference(model,.9*model.native_probabilities+.1/20,reference)
