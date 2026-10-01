"""Curve-first invariants and feasible soft-sequence derivatives."""
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpropka import ModelConfig, TitrationModel, prepare
from jaxpropka.parameters import AA_TO_INDEX, MODEL_PKA
from jaxpropka.synthetic import synthetic_cache


def pair_model(coupling=.5, steps=512):
    cache=synthetic_cache(n=2,neighbors=2,chains=2)
    gm=np.zeros((2,9),bool);gm[:,2]=True
    pm=np.zeros_like(cache.pair_mask)
    cg=np.zeros_like(cache.coulomb_geometry)
    for i in range(2):
        for k,j in enumerate(cache.neighbors[i]):
            if i!=j:
                pm[i,k,2,2]=True;cg[i,k,2,2]=80*coupling
    cache=replace(cache,group_mask=gm,pair_mask=pm,coulomb_geometry=cg,
                  native_index=np.full(2,AA_TO_INDEX['H'],dtype=np.int32))
    config=ModelConfig(steps=steps,desolv_scale=0,hbond_scale=0,gate_width=0,
                       nmin=0,nmax=1,dielectric_surface=80,dielectric_buried=80)
    return TitrationModel(cache,config)


def test_identical_weakly_coupled_pair_has_known_half_occupancy():
    model=pair_model()
    result=model.curves([MODEL_PKA[2]-.25])(model.native_probabilities.astype(jnp.float64))
    np.testing.assert_allclose(result.protonated[0,:,2],.5,atol=1e-12)
    assert np.all(result.converged)


@pytest.mark.parametrize('mixture',[.01,.5,1.])
def test_soft_sequence_curve_derivatives_on_simplex(mixture):
    model=pair_model();native=model.native_probabilities.astype(jnp.float64)
    p=(1-mixture)*native+mixture/20
    logits=jnp.log(p)
    ph=np.array([5.,6.,7.,8.])
    curves=model.curves(ph)
    def loss(x):
        q=curves(jax.nn.softmax(x,axis=-1)).residue_charge
        return jnp.mean((q-jnp.array([.2,.4]))**2)
    direction=jnp.asarray(np.random.default_rng(4).normal(size=p.shape))
    direction-=direction.mean(-1,keepdims=True);direction/=jnp.linalg.norm(direction)
    gradient=jax.grad(loss)(logits)
    np.testing.assert_allclose(gradient.sum(-1),0,atol=1e-12)
    auto=jnp.vdot(gradient,direction)
    _,forward=jax.jvp(loss,(logits,),(direction,))
    np.testing.assert_allclose(auto,forward,atol=1e-12)
    for eps in [1e-2,1e-3,1e-4]:
        finite=(loss(logits+eps*direction)-loss(logits-eps*direction))/(2*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-4,atol=1e-9)
    out=curves(p)
    assert np.all(out.converged)
    assert np.all(np.isfinite(out.protonated))
    assert np.all((out.protonated>=0)&(out.protonated<=1))
    np.testing.assert_allclose(out.site_charge.sum((1,2)),out.total_charge,atol=1e-12)
    assert np.all(np.diff(out.total_charge)<=1e-12)


def test_gradient_and_curve_stable_after_convergence():
    models=[pair_model(steps=s) for s in [128,512]]
    x=jnp.zeros((2,20),dtype=jnp.float64)
    functions=[lambda z,m=m:jnp.square(m.curves([5.,7.,9.])(jax.nn.softmax(z,-1)).total_charge).sum() for m in models]
    np.testing.assert_allclose(functions[0](x),functions[1](x),atol=1e-12)
    np.testing.assert_allclose(jax.grad(functions[0])(x),jax.grad(functions[1])(x),atol=1e-12)


@pytest.mark.parametrize('mixture',[.01,.5])
def test_direct_probability_tangent_derivative(mixture):
    model=pair_model()
    p=(1-mixture)*model.native_probabilities.astype(jnp.float64)+mixture/20
    curves=model.curves([5.,7.,9.])
    loss=lambda probabilities:jnp.mean(curves(probabilities).residue_charge**2)
    v=np.random.default_rng(71).normal(size=p.shape)
    v-=v.mean(-1,keepdims=True);v/=np.linalg.norm(v);v=jnp.asarray(v)
    auto=jnp.vdot(jax.grad(loss)(p),v)
    for eps in [1e-4,1e-5,1e-6]:
        assert np.min(p-eps*v)>0 and np.min(p+eps*v)>0
        np.testing.assert_allclose((p+eps*v).sum(-1),1,atol=1e-12)
        finite=(loss(p+eps*v)-loss(p-eps*v))/(2*eps)
        np.testing.assert_allclose(auto,finite,rtol=1e-5,atol=1e-8)


def test_one_hot_boundary_has_feasible_one_sided_derivative():
    model=pair_model();p=model.native_probabilities.astype(jnp.float64)
    v=jnp.zeros_like(p).at[0,AA_TO_INDEX['H']].set(-1).at[0,AA_TO_INDEX['A']].set(1)
    loss=lambda probabilities:jnp.mean(model.curves([5.,7.,9.])(probabilities).residue_charge**2)
    auto=jnp.vdot(jax.grad(loss)(p),v)
    eps=1e-6
    assert np.min(p+eps*v)>=0
    np.testing.assert_allclose(auto,(loss(p+eps*v)-loss(p))/eps,rtol=1e-5,atol=1e-7)


@pytest.mark.parametrize('fixture,near',[('two_chains.pdb',True),('two_chains_far.pdb',False)])
def test_real_interface_cross_chain_gradient_and_separation(fixture,near):
    data=Path(__file__).parent/'data'
    model=TitrationModel(prepare(data/fixture),ModelConfig(steps=512))
    x=jnp.zeros((model.cache.n_residues,20),dtype=jnp.float64)
    def loss(z):
        out=model.curves([5.,7.,9.])(jax.nn.softmax(z,-1))
        return jnp.mean(out.chain_charge[:,0]**2)
    grad=jax.grad(loss)(x)
    cross=np.linalg.norm(np.asarray(grad)[model.cache.chain_index==1])
    if near:
        assert cross>1e-8
    else:
        assert cross<1e-12
        from jaxpropka.topology import load_topology
        topology=load_topology(data/fixture)
        out=model.curves([5.,7.,9.])(jax.nn.softmax(x,-1))
        # Use each chain's exact coordinates: the translated fixture has
        # float32 rounding relative to peptide.pdb, unrelated to interactions.
        for c,chain in enumerate(topology.chain_ids):
            atoms=topology.atoms[topology.atoms.chain_id==chain]
            single=TitrationModel(prepare(atoms),ModelConfig(steps=512))
            one=single.curves([5.,7.,9.])(jnp.full((single.cache.n_residues,20),.05,dtype=jnp.float64))
            np.testing.assert_allclose(out.chain_charge[:,c],one.total_charge,atol=1e-10)
