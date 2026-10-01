import importlib.util
from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpropka.model import LocalTerms

for name in ['audit_branch_grid','audit_branch_stability']:
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/f'benchmarks/{name}.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
grid=sys.modules['audit_branch_grid']; audit=sys.modules['audit_branch_stability']


def pair(strength,weights):
    d={'neighbors':jnp.array([[1],[0]])}
    w=jnp.asarray(weights).reshape(2,1)
    terms=LocalTerms(jnp.zeros((2,1)),jnp.zeros((2,1)),
                     strength*w[d['neighbors']][:,:,None,:],w,jnp.zeros((2,1)))
    return d,terms


@pytest.mark.parametrize('strength,classification',[(.5,'positive'),(2.,'negative')])
def test_stable_and_unstable_pair(strength,classification):
    d,t=pair(strength,[1.,1.])
    active,w,a,error=audit.active_interaction(d,t)
    result=audit.curvature(a,w,np.array([.5,.5]))
    assert result['curvature_class']==classification
    np.testing.assert_allclose(result['minimum_preconditioned_hessian_eigenvalue'],1-np.log(10)*strength/4,atol=1e-7)
    assert error==0


def test_preconditioned_hessian_matches_autodiff_soft_weights():
    # x64 avoids float32 reciprocity rounding in this strict diagnostic.
    previous=jax.config.x64_enabled
    jax.config.update('jax_enable_x64',True)
    try:
        d,t=pair(2.,[.2,.7]);h=jnp.array([[.3],[.8]])
        active,w,a,_=audit.active_interaction(d,t)
        actual=np.asarray(jax.hessian(lambda x:grid.free_energy(d,t,x,7.))(h)).reshape(2,2)
        scale=np.sqrt(np.asarray(h).ravel()*(1-np.asarray(h).ravel())/w)
        expected=np.eye(2)+scale[:,None]*a.toarray()*scale[None,:]
        np.testing.assert_allclose(scale[:,None]*actual*scale[None,:],expected,atol=1e-12)
    finally:
        jax.config.update('jax_enable_x64',previous)


def test_nonreciprocal_interaction_rejected():
    d,t=pair(2.,[1.,1.]);t=t._replace(coupling=t.coupling.at[0].multiply(2))
    with pytest.raises(ValueError,match='not reciprocal'):
        audit.active_interaction(d,t)
