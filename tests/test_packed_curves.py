"""Parity of experimental edge packing, including finite-iteration gradients."""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpropka import ModelConfig, TitrationModel
from jaxpropka.batching import pack_inputs
from jaxpropka.model import (curve_kernel, one_hot, pack_interaction_edges,
                             packed_curve_kernel)
from jaxpropka.synthetic import synthetic_cache

@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('dtype', [jnp.float32, jnp.float64])
def test_packed_values_and_gradients(empty, dtype):
    cache = synthetic_cache(n=6, neighbors=2, chains=2)
    frozen = cache.frozen.copy(); frozen[0] = True
    cache = replace(cache, frozen=frozen)
    if empty:
        cache = replace(cache, pair_mask=np.zeros_like(cache.pair_mask),
            coulomb_geometry=np.zeros_like(cache.coulomb_geometry))
    arrays, native, _ = pack_inputs(cache, one_hot(cache.native_index, dtype=np.dtype(dtype)),
        bucket_multiple=(1,1,1))
    edges = pack_interaction_edges(arrays['pair_mask'],arrays['neighbors'])
    logits = jnp.log(.95*jnp.asarray(native)+.05/20)
    ph = jnp.asarray([6., 6.7, 7.4], dtype=dtype)
    cfg = ModelConfig(steps=128)
    def evaluate(z, compact):
        p = jax.nn.softmax(z, axis=-1)
        return (packed_curve_kernel(arrays, p, ph, edges, config=cfg) if compact
            else curve_kernel(arrays, p, ph, config=cfg))
    a, b = evaluate(logits, False), evaluate(logits, True)
    tol = 2e-5 if dtype == jnp.float32 else 1e-11
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=tol, rtol=tol)
    gradients = []
    for compact in (False, True):
        loss = lambda z: jnp.trapezoid(evaluate(z, compact).total_charge, ph)
        gradient = jax.grad(loss)(logits)
        gradients.append(gradient)
        np.testing.assert_allclose(gradient[0], 0, atol=tol)
    np.testing.assert_allclose(*gradients, atol=tol, rtol=tol)
    if dtype == jnp.float64:
        direction = jnp.asarray(np.random.default_rng(5).normal(size=logits.shape))
        direction /= jnp.linalg.norm(direction)
        eps = 1e-4
        finite = (loss(logits+eps*direction)-loss(logits-eps*direction))/(2*eps)
        np.testing.assert_allclose(jnp.vdot(gradients[1], direction), finite, atol=1e-8, rtol=1e-6)


def test_opt_in_model_backend_covers_curves_grid_and_direct_roots():
    cache = synthetic_cache(n=6,neighbors=4,chains=2)
    config = ModelConfig(steps=96,root_steps=24)
    dense = TitrationModel(cache,config)
    packed = TitrationModel(cache,config,backend='packed')
    p = .95*dense.native_probabilities.astype(jnp.float64)+.05/20
    ph = np.linspace(-2,16,73)
    for make in (lambda m:m.curves(ph),lambda m:m.pka_from_grid(ph),
                 lambda m:m.pka_sites([(0,'ASP'),(1,'GLU'),(2,'HIS')])):
        a,b = make(dense)(p),make(packed)(p)
        for x,y in zip(a,b):
            np.testing.assert_allclose(x,y,atol=2e-10,rtol=2e-10)
    loss = lambda m,z:jnp.trapezoid(m.curves([6.,6.7,7.4])(
        jax.nn.softmax(z,-1)).total_charge,jnp.asarray([6.,6.7,7.4]))
    logits = jnp.log(p)
    np.testing.assert_allclose(jax.grad(lambda z:loss(dense,z))(logits),
                               jax.grad(lambda z:loss(packed,z))(logits),
                               atol=2e-10,rtol=2e-10)


def test_packed_backend_validation():
    cache = synthetic_cache(n=2,neighbors=2,chains=1)
    assert TitrationModel(cache).backend == 'dense'
    with pytest.raises(ValueError,match='backend'):
        TitrationModel(cache,backend='unknown')
