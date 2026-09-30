from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpropka import TitrationModel, ModelConfig, GROUPS, one_hot
from jaxpropka.parameters import MODEL_PKA, Q_DEPROT, AA_TO_INDEX
from jaxpropka.synthetic import synthetic_cache


def uncoupled(cache):
    return TitrationModel(cache,ModelConfig(coulomb_scale=0,hbond_scale=0,desolv_scale=0,steps=8))


def test_uncoupled_curves(cache):
    model=uncoupled(cache); p=model.native_probabilities.astype(jnp.float64)
    ph=np.array([2.,7.,12.])
    out=model.curves(ph)(p)
    expected=1/(1+10**(ph[:,None,None]-MODEL_PKA[None,None,:]))
    expected=np.broadcast_to(expected,(3,3,9))*cache.group_mask
    np.testing.assert_allclose(out.protonated,expected,atol=1e-12)
    assert np.all(out.converged)
    np.testing.assert_allclose(out.site_charge,np.asarray(out.probability)[None]*(Q_DEPROT+expected))


def test_uncoupled_midpoint_and_masks(cache):
    model=uncoupled(cache); out=model.pka()(model.native_probabilities.astype(jnp.float64))
    expected=np.broadcast_to(MODEL_PKA,(3,9))*cache.group_mask
    np.testing.assert_allclose(out.value,expected,atol=2e-7)
    np.testing.assert_array_equal(out.valid,cache.group_mask)
    np.testing.assert_allclose(np.asarray(out.slope)[cache.group_mask],-np.log(10)/4,atol=2e-7)
    assert np.isfinite(out.value).all()


def test_acid_base_limits_and_two_terminals_per_chain(cache):
    model=uncoupled(cache); p=model.native_probabilities
    out=model.curves([-30,40])(p)
    assert int(cache.group_mask[:,7].sum())==2
    assert int(cache.group_mask[:,8].sum())==2
    assert float(out.total_charge[0]) > float(out.total_charge[1])
    np.testing.assert_allclose(np.asarray(out.site_charge)[0,:,:7].sum(),1,atol=1e-6)
    np.testing.assert_allclose(np.asarray(out.site_charge)[1,:,:7].sum(),-2,atol=1e-6)


def test_selected_outputs_keep_full_multichain_environment(cache):
    model=TitrationModel(cache);p=model.native_probabilities
    full=model.curves([6.,7.,8.])(p)
    selected=model.curves([6.,7.,8.],residues=[cache.keys[2]])(p)
    np.testing.assert_allclose(selected.residue_charge,full.residue_charge[:,2:3],atol=1e-6)
    np.testing.assert_allclose(selected.total_charge,full.total_charge,atol=1e-6)
    np.testing.assert_allclose(selected.chain_charge.sum(-1),full.total_charge,atol=1e-6)
    for c in range(2):
        np.testing.assert_allclose(np.asarray(full.residue_charge)[:,cache.chain_index==c].sum(-1),
                                   full.chain_charge[:,c],atol=1e-6)


def test_same_residue_alternatives_do_not_interact(cache):
    for i in range(3):
        for k,j in enumerate(cache.neighbors[i]):
            if i==j:
                assert not cache.pair_mask[i,k,:7,:7].any()
                assert not np.diag(cache.pair_mask[i,k]).any()


def test_sequence_is_not_argmaxed(cache):
    model=TitrationModel(cache)
    p=jnp.full((3,20),.05,dtype=jnp.float64)
    out=model.curves([7.])(p)
    np.testing.assert_allclose(out.probability[:,:7],.05)
    grads=jax.grad(lambda x:model.charge(7)(x).sum())(p)
    assert np.isfinite(grads).all()
    assert np.linalg.norm(grads)>1e-5


def directional_check(fn,x,eps=1e-4,rtol=2e-3):
    rng=np.random.default_rng(42)
    v=jnp.asarray(rng.normal(size=x.shape),dtype=x.dtype)
    v=v/jnp.linalg.norm(v)
    auto=jnp.vdot(jax.grad(fn)(x),v)
    finite=(fn(x+eps*v)-fn(x-eps*v))/(2*eps)
    np.testing.assert_allclose(auto,finite,atol=2e-5,rtol=rtol)


def test_charge_logits_gradient_finite_difference(cache):
    model=TitrationModel(cache);charge=model.charge(7.)
    logits=jnp.zeros((3,20),dtype=jnp.float64)
    directional_check(lambda x:jnp.square(charge(model.probabilities_from_logits(x))).sum(),logits)


def test_curve_gradient_finite_difference(cache):
    model=TitrationModel(cache);curves=model.curves([5.5,6.5,7.5])
    logits=jnp.zeros((3,20),dtype=jnp.float64)
    directional_check(lambda x:curves(jax.nn.softmax(x,-1)).protonated[:,0,2].sum(),logits)


def test_pka_implicit_gradient_finite_difference(cache):
    model=TitrationModel(cache,ModelConfig(root_steps=36,steps=80))
    pka=model.pka_sites([(0,"HIS"),(2,"ASP")])
    logits=jnp.zeros((3,20),dtype=jnp.float64)
    result=pka(jax.nn.softmax(logits,-1))
    assert np.all(result.valid)
    fn=lambda x:jnp.sum(pka(jax.nn.softmax(x,-1)).value)
    directional_check(fn,logits,eps=1e-3,rtol=3e-3)
    assert np.linalg.norm(jax.grad(fn)(logits))>1e-5


def test_pka_forward_jvp_matches_reverse(cache):
    model=TitrationModel(cache,ModelConfig(root_steps=32,steps=64))
    pka=model.pka_sites([(0,"HIS")])
    x=jnp.zeros((3,20),dtype=jnp.float64);v=jnp.ones_like(x).at[1,2].set(2.)
    fn=lambda a:pka(jax.nn.softmax(a,-1)).value.sum()
    _,tangent=jax.jvp(fn,(x,),(v,))
    np.testing.assert_allclose(tangent,jnp.vdot(jax.grad(fn)(x),v),atol=1e-8)


def test_float32_gradients_are_finite(cache):
    model=TitrationModel(cache);fn=model.pka_sites([(0,"HIS")])
    x=jnp.zeros((3,20),dtype=jnp.float32)
    out=fn(jax.nn.softmax(x,-1));assert out.value.dtype==jnp.float32
    assert np.isfinite(jax.grad(lambda y:fn(jax.nn.softmax(y,-1)).value.sum())(x)).all()


def test_unbracketed_pka_is_reported_and_safe(cache):
    model=TitrationModel(cache,ModelConfig(ph_min=0,ph_max=1))
    fn=model.pka_sites([(0,"LYS")]);x=jnp.zeros((3,20),dtype=jnp.float64)
    out=fn(jax.nn.softmax(x,-1));assert not out.bracketed.any();assert not out.valid.any()
    assert np.isfinite(jax.grad(lambda y:fn(jax.nn.softmax(y,-1)).value.sum())(x)).all()


def test_vmap_sequence_batch(cache):
    model=TitrationModel(cache);fn=model.charge(7.)
    p=model.native_probabilities
    output=jax.jit(jax.vmap(fn))(jnp.stack([p,p]))
    np.testing.assert_allclose(output[0],output[1],atol=1e-7)


def test_graph_padding_does_not_change_answer(cache):
    d={}
    for name in ("env_neighbors","env_mask","volume","mass","hbond"):
        a=getattr(cache,name)
        d[name]=np.pad(a,[(0,0),(0,2)]+[(0,0)]*(a.ndim-2))
    for name in ("neighbors","pair_mask","coulomb_geometry","hb_donor","hb_reverse"):
        a=getattr(cache,name)
        d[name]=np.pad(a,[(0,0),(0,2)]+[(0,0)]*(a.ndim-2))
    padded=replace(cache,**d)
    m=TitrationModel(cache); mp=TitrationModel(padded);p=m.native_probabilities
    np.testing.assert_allclose(m.charge(7)(p),mp.charge(7)(p),atol=1e-7)


def test_frozen_topology_has_zero_sequence_gradient(cache):
    c=replace(cache,frozen=np.array([True,False,False]))
    model=TitrationModel(c);fn=model.charge(7)
    x=jnp.zeros((3,20),dtype=jnp.float64)
    grad=jax.grad(lambda a:fn(jax.nn.softmax(a,-1)).sum())(x)
    np.testing.assert_array_equal(grad[0],0)


def test_probabilities_and_selections_validation(cache):
    model=TitrationModel(cache)
    for p in (np.ones((3,20)),np.full((3,20),np.nan),np.ones((2,20))):
        with pytest.raises(ValueError): model.validate_probabilities(p)
    with pytest.raises(KeyError): cache.select([("nochain",1)])
    with pytest.raises(ValueError): cache.select([0,0])
    with pytest.raises(ValueError): model.probabilities_from_logits(jnp.zeros((3,20)),np.zeros((3,20),bool))


def test_cache_round_trip_and_bad_graph(cache,tmp_path):
    from jaxpropka import StructureCache
    path=tmp_path/"cache.npz";cache.save(path);loaded=StructureCache.load(path)
    assert loaded.fingerprint()==cache.fingerprint()
    bad=cache.neighbors.copy();bad[0,0]=999
    with pytest.raises(ValueError): replace(cache,neighbors=bad).validate()


def test_cross_chain_messages_are_present():
    c=synthetic_cache(4,neighbors=4,chains=2)
    m=TitrationModel(c);p=m.native_probabilities.astype(jnp.float64)
    fn=m.charge(6.,residues=[0])
    before=fn(p)
    p2=p.at[3].set(jnp.asarray(one_hot("D",dtype=np.float64)[0]))
    after=fn(p2)
    assert float(jnp.max(jnp.abs(before-after)))>1e-5


def test_more_iterations_stabilize_output(cache):
    p=jnp.full((3,20),.05,dtype=jnp.float64)
    m1=TitrationModel(cache,ModelConfig(steps=64));m2=TitrationModel(cache,ModelConfig(steps=96))
    np.testing.assert_allclose(m1.charge(7)(p),m2.charge(7)(p),atol=1e-8)
