from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpropka import DifferentiationConfig, ModelConfig, SelectivityObjective, TitrationModel
from jaxpropka.cache import StructureCache
from jaxpropka.synthetic import synthetic_cache
from jaxpropka.experimental import EndpointSelectivityObjective, mean_field_potential


def subset(cache, chain):
    take=np.flatnonzero(cache.chain_index==chain)
    remap=np.zeros(cache.n_residues,np.int32);remap[take]=np.arange(len(take))
    values={k:(v[take].copy() if isinstance(v,np.ndarray) else v) for k,v in vars(cache).items()}
    values.update(keys=tuple(cache.keys[i] for i in take),chain_ids=(cache.chain_ids[chain],),
                  chain_index=np.zeros(len(take),np.int32))
    for neighbors,mask in (('env_neighbors','env_mask'),('neighbors','pair_mask')):
        original=getattr(cache,neighbors)[take]
        keep=np.isin(original,take)
        values[mask]&=keep if mask=='env_mask' else keep[:,:,None,None]
        values[neighbors]=remap[original]
    return StructureCache(**values).validate()


def models(mode='checkpointed'):
    cache=synthetic_cache(n=6,neighbors=4)
    diff=DifferentiationConfig(mode=mode,implicit_residual_tolerance=1e-10,
                               linear_rtol=1e-10,linear_atol=1e-12)
    return tuple(TitrationModel(c,ModelConfig(steps=128),backend='packed_v2',differentiation=diff)
                 for c in (cache,subset(cache,0),subset(cache,1)))


@pytest.mark.parametrize('mode',['checkpointed','implicit'])
@pytest.mark.parametrize('chunk',[1,2,5])
def test_streamed_matches_monolithic_and_composes(mode,chunk):
    bound,binder,target=models(mode)
    grid=np.linspace(6,7.4,5)
    obj=SelectivityObjective(bound,binder,target,grid,ph_chunk_size=chunk)
    p=.95*binder.native_probabilities.astype(jnp.float64)+.05/20
    def reference(x):
        b,f=obj._probabilities(x)
        q=(bound.curves(grid)(b).total_charge-binder.curves(grid)(f).total_charge
           -target.curves(grid)(jnp.asarray(obj._target_p,x.dtype)).total_charge)
        s=jnp.trapezoid(q,jnp.asarray(grid))
        return obj.tau*jax.nn.softplus((obj.required_log10_ratio-s)/obj.tau)
    expected,g=jax.value_and_grad(reference)(p)
    out=obj.value_and_grad(p)
    assert out.valid and obj.evaluate(p).valid
    np.testing.assert_allclose(out.loss,expected,atol=1e-11)
    np.testing.assert_allclose(out.gradient,g,atol=1e-9,rtol=1e-7)
    np.testing.assert_allclose(jax.grad(obj.loss)(p),g,atol=1e-9,rtol=1e-7)
    z=jnp.log(p);v=jnp.asarray(np.random.default_rng(2).normal(size=p.shape))
    composed=lambda x:obj.loss(jax.nn.softmax(x,-1))
    np.testing.assert_allclose(jax.jvp(composed,(z,),(v,))[1],jnp.vdot(jax.grad(composed)(z),v),atol=1e-9)
    assert bool(out.diagnostics.linear_checked[0,0]) == (mode=='implicit')
    assert not out.diagnostics.linear_checked[2].any()


def test_residue_key_order_and_invalid_inputs():
    bound,binder,target=models()
    a=SelectivityObjective(bound,binder,target,[6.,7.4])
    b=SelectivityObjective(bound,binder,target,[6.,7.4],binder_keys=binder.cache.keys[::-1])
    p=.95*binder.native_probabilities.astype(jnp.float64)+.05/20
    oa,ob=a.value_and_grad(p),b.value_and_grad(p[::-1])
    np.testing.assert_allclose(oa.loss,ob.loss,atol=1e-12)
    np.testing.assert_allclose(oa.gradient,ob.gradient[::-1],atol=1e-12)
    for kwargs in (dict(tau=0),dict(ph_chunk_size=0),dict(binder_keys=[binder.cache.keys[0]])):
        with pytest.raises(ValueError):SelectivityObjective(bound,binder,target,[6.,7.4],**kwargs)
    with pytest.raises(ValueError):SelectivityObjective(bound,binder,target,[7.4,6.])
    changed=replace(binder.cache,frozen=binder.cache.frozen.copy())
    changed.frozen[0]=True
    with pytest.raises(ValueError,match='frozen'):
        SelectivityObjective(bound,TitrationModel(changed),target,[6.,7.4])


def test_failed_forward_is_invalid_without_fallback():
    bound,binder,target=models('implicit')
    bound=TitrationModel(bound.cache,ModelConfig(steps=1),backend='packed_v2',differentiation=bound.differentiation)
    obj=SelectivityObjective(bound,binder,target,[6.,7.4])
    p=jnp.full((binder.cache.n_residues,20),.05)
    out=obj.value_and_grad(p)
    assert not out.valid and not out.diagnostics.forward_valid[0].all()
    assert np.isnan(out.gradient).all()


def test_endpoint_energy_identity_envelope_and_quadrature():
    bound,binder,target=models()
    obj=SelectivityObjective(bound,binder,target,np.linspace(6.,7.4,129))
    endpoint=EndpointSelectivityObjective(obj)
    p=.95*binder.native_probabilities.astype(jnp.float64)+.05/20
    pb,_=obj._probabilities(p)
    terms=bound._local_terms(pb);h,residual,_=bound._solve(terms,6.7)
    assert residual<1e-10
    charge=(terms.weights*(jnp.asarray([-1,-1,0,-1,-1,0,0,0,-1])+h)).sum()
    np.testing.assert_allclose(jax.grad(lambda hp:mean_field_potential(bound,terms,h,hp))(6.7),charge,atol=1e-11)
    dh=jax.grad(lambda hh:mean_field_potential(bound,terms,hh,6.7))(h)
    np.testing.assert_allclose(jnp.where(bound._gm,dh,0),0,atol=1e-8)
    a,b=obj.value_and_grad(p),endpoint.value_and_grad(p)
    assert a.valid and b.valid
    np.testing.assert_allclose(a.selectivity,b.selectivity,atol=1e-5)
    np.testing.assert_allclose(a.gradient,b.gradient,atol=1e-5,rtol=1e-4)
    direction=jnp.asarray(np.random.default_rng(1).normal(size=p.shape));direction-=direction.mean(-1,keepdims=True)
    eps=1e-5
    fd=(endpoint.evaluate(p+eps*direction).loss-endpoint.evaluate(p-eps*direction).loss)/(2*eps)
    np.testing.assert_allclose(fd,jnp.vdot(b.gradient,direction),atol=1e-7,rtol=1e-5)
    assert endpoint.audit(p,np.linspace(6.,7.4,9)).consistent.all()


def test_failed_adjoint_is_invalid_without_fallback():
    bound,binder,target=models('implicit')
    diff=replace(bound.differentiation,linear_restart=1,linear_maxiter=1,
                 linear_rtol=1e-14,linear_atol=1e-16)
    bound=TitrationModel(bound.cache,bound.config,backend='packed_v2',differentiation=diff)
    obj=SelectivityObjective(bound,binder,target,[6.,7.4])
    p=jnp.full((binder.cache.n_residues,20),.05)
    out=obj.value_and_grad(p)
    assert out.diagnostics.forward_valid.all()
    assert not out.valid and not out.diagnostics.linear_valid[0].all()
    assert np.isnan(out.gradient).all()


def test_separated_partners_have_zero_selectivity_and_gradient():
    cache=synthetic_cache(n=6,neighbors=4,separated=True)
    model=[TitrationModel(c,ModelConfig(steps=128),backend='packed_v2',
                          differentiation=DifferentiationConfig(mode='checkpointed'))
           for c in (cache,subset(cache,0),subset(cache,1))]
    obj=SelectivityObjective(*model,[6.,6.7,7.4])
    p=jnp.full((model[1].cache.n_residues,20),.05)
    out=obj.value_and_grad(p)
    assert out.valid
    np.testing.assert_allclose(out.selectivity,0,atol=1e-12)
    np.testing.assert_allclose(out.gradient,0,atol=1e-12)
