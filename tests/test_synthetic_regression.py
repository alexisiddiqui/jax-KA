"""Pinned implementation snapshot, separate from optional external PROPKA baselines."""
from dataclasses import asdict
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpropka import TitrationModel
from jaxpropka.synthetic import synthetic_cache

@pytest.mark.parametrize('case',['hard','soft'])
def test_recorded_synthetic_output_regression(case):
    baseline=json.loads((Path(__file__).parent/'data/synthetic_baseline.json').read_text())
    model=TitrationModel(synthetic_cache(n=6,chains=2,neighbors=4,seed=7))
    assert asdict(model.config)==baseline['model_config']
    p=model.native_probabilities
    if case=='soft':p=.85*p+.15/20
    p=jnp.asarray(p,jnp.float32)
    curves=model.curves(baseline['ph'])(p)
    midpoint=model.pka_sites([(0,'ASP'),(2,'HIS'),(5,'CTERM')])(p)
    assert curves.converged.all() and midpoint.valid.all()
    for field in ('residue_charge','chain_charge','total_charge'):
        np.testing.assert_allclose(getattr(curves,field),baseline['cases'][case][field],rtol=0,atol=3e-5)
    np.testing.assert_allclose(midpoint.value,baseline['cases'][case]['selected_midpoint_pka'],rtol=0,atol=3e-5)
