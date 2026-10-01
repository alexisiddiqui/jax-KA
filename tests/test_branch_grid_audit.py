import importlib.util
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jaxpropka.model import LocalTerms, _field

spec=importlib.util.spec_from_file_location('branch_audit',Path(__file__).resolve().parents[1]/'benchmarks/audit_branch_grid.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_energy_stationarity_with_soft_weights():
    d={'neighbors':jnp.array([[1],[0]])}
    weights=jnp.array([[.2],[.7]])
    coupling=2.*weights[d['neighbors']][:,:,None,:]
    terms=LocalTerms(jnp.array([[6.],[7.]]),jnp.array([[.2],[-.1]]),coupling,weights,jnp.zeros((2,1)))
    h=jnp.array([[.3],[.8]])
    actual=jax.grad(lambda x:audit.free_energy(d,terms,x,7.))(h)
    expected=weights*(jnp.log(h/(1-h))+jnp.log(10.)*(7.-terms.intrinsic+_field(d,terms,h)))
    np.testing.assert_allclose(actual,expected,atol=1e-6)
    assert np.isfinite(audit.free_energy(d,terms,jnp.array([[0.],[1.]]),7.))


def test_refinement_keeps_anchors_and_refines_disagreement():
    ph=np.array([0.,1.,2.]); h=np.zeros((3,1,1)); r=h.copy();r[1]=.2
    out=audit.refinement_points(ph,h,r,np.ones((1,1)),np.zeros(3))
    np.testing.assert_array_equal(out,np.linspace(0,2,9))
    np.testing.assert_array_equal(audit.refinement_points(ph,h,h,np.ones((1,1)),np.zeros(3)),ph)
