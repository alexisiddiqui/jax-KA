from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpropka import TitrationModel,ModelConfig
from jaxpropka.synthetic import synthetic_cache
from jaxpropka.parameters import MODEL_PKA


def test_grid_midpoint_converges_to_direct_root_with_refinement():
    model=TitrationModel(synthetic_cache(n=4,chains=2))
    p=model.probabilities_from_logits(jnp.zeros((4,20)))
    direct=model.pka(groups=['ASP','HIS'])(p)
    coarse=model.pka_from_grid(np.linspace(-2,16,37),groups=['ASP','HIS'])(p)
    fine=model.pka_from_grid(np.linspace(-2,16,181),groups=['ASP','HIS'])(p)
    assert np.all(direct.valid) and np.all(fine.valid)
    assert np.max(abs(fine.value-direct.value))<np.max(abs(coarse.value-direct.value))
    assert np.max(abs(fine.value-direct.value))<.002
    np.testing.assert_allclose(fine.bracket_width,.1,atol=1e-12)


def test_grid_midpoint_gradient_matches_finite_difference():
    model=TitrationModel(synthetic_cache(n=3,chains=2))
    fn=model.pka_from_grid(np.linspace(-2,16,73),groups=['ASP','HIS'])
    rng=np.random.default_rng(13)
    z=jnp.asarray(rng.normal(size=(3,20))*.1)
    direction=jnp.asarray(rng.normal(size=z.shape));direction/=jnp.linalg.norm(direction)
    loss=lambda v:fn(jax.nn.softmax(v,-1)).value.sum()
    grad=jax.grad(loss)(z)
    delta=2e-4
    fd=(loss(z+delta*direction)-loss(z-delta*direction))/(2*delta)
    np.testing.assert_allclose(jnp.sum(grad*direction),fd,rtol=2e-4,atol=2e-6)
    assert np.isfinite(grad).all()


def test_grid_midpoint_inactive_slots_and_outside_grid_are_safe():
    model=TitrationModel(synthetic_cache(n=4,chains=2))
    fn=model.pka_from_grid([6.,7.],groups=['ARG','NTERM'])
    z=jnp.zeros((4,20))
    out=fn(jax.nn.softmax(z,-1))
    assert not out.bracketed[:,0].any()
    assert not out.valid[:,0].any()
    np.testing.assert_allclose(out.value[:,0],0)
    grad=jax.grad(lambda x:fn(jax.nn.softmax(x,-1)).value.sum())(z)
    assert np.isfinite(grad).all()
    with pytest.raises(ValueError):model.pka_from_grid([7.,6.])
    with pytest.raises(ValueError):model.pka_from_grid([7.])


def test_grid_selection_matches_full_result():
    model=TitrationModel(synthetic_cache(n=4,chains=2))
    ph=np.linspace(-2,16,73);p=model.native_probabilities
    full=model.pka_from_grid(ph)(p)
    selected=model.pka_from_grid(ph,residues=[3,0],groups=['HIS','ASP'])(p)
    np.testing.assert_allclose(selected.value,np.asarray(full.value)[[3,0]][:,[2,0]],atol=0)
