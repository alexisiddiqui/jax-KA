from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpropka import DifferentiationConfig, ModelConfig, TitrationModel
from jaxpropka.differentiation import checked_linear_solve
from jaxpropka.model import pack_runtime
from jaxpropka.synthetic import synthetic_cache


@pytest.mark.parametrize('steps', [1, 17, 64])
@pytest.mark.parametrize('backend', ['dense', 'packed', 'packed_v2'])
def test_checkpoint_remainder_and_gradients(steps, backend):
    cache = synthetic_cache(n=4,neighbors=2)
    cfg = ModelConfig(steps=steps)
    a = TitrationModel(cache,cfg,backend=backend)
    b = TitrationModel(cache,cfg,backend=backend,
                       differentiation=DifferentiationConfig(mode='checkpointed'))
    z = jnp.log(.95*a.native_probabilities.astype(jnp.float64)+.05/20)
    def objective(model):
        read = model.curves([6.,6.7,7.4])
        return lambda zz:jnp.sum(read(jax.nn.softmax(zz,-1)).total_charge)
    va,ga = jax.value_and_grad(objective(a))(z)
    vb,gb = jax.value_and_grad(objective(b))(z)
    np.testing.assert_allclose(va,vb,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(ga,gb,atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('empty',[False,True])
def test_packed_terms_and_gradient_parity(empty):
    cache = synthetic_cache(n=4,neighbors=2)
    if empty:
        cache=replace(cache,pair_mask=np.zeros_like(cache.pair_mask))
    runtime=pack_runtime(cache)
    assert 'pair_mask' not in runtime and 'neighbors' not in runtime
    assert 'coulomb_geometry' not in runtime
    a,b=[TitrationModel(cache,backend=backend) for backend in ('packed','packed_v2')]
    p=.95*a.native_probabilities.astype(jnp.float64)+.05/20
    for x,y in zip(a.curves([6.,7.4])(p),b.curves([6.,7.4])(p)):
        np.testing.assert_allclose(x,y,atol=1e-11,rtol=1e-11)
    ga=jax.grad(lambda x:a.charge(6.7)(x).sum())(p)
    gb=jax.grad(lambda x:b.charge(6.7)(x).sum())(p)
    np.testing.assert_allclose(ga,gb,atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize('backend',['dense','packed','packed_v2'])
def test_implicit_forward_reverse_and_finite_difference(backend):
    cache=synthetic_cache(n=4,neighbors=2)
    diff=DifferentiationConfig(mode='implicit',implicit_residual_tolerance=1e-10,
                               linear_rtol=1e-10,linear_atol=1e-12)
    implicit=TitrationModel(cache,ModelConfig(steps=128),backend=backend,differentiation=diff)
    finite=TitrationModel(cache,ModelConfig(steps=128),backend=backend)
    p=.95*finite.native_probabilities.astype(jnp.float64)+.05/20
    z=jnp.log(p)
    read=implicit.charge([6.,6.7,7.4]); ref=finite.charge([6.,6.7,7.4])
    loss=lambda x:read(jax.nn.softmax(x,-1)).sum()
    g=jax.grad(loss)(z)
    gf=jax.grad(lambda x:ref(jax.nn.softmax(x,-1)).sum())(z)
    np.testing.assert_allclose(g,gf,atol=1e-9,rtol=1e-7)
    v=jnp.asarray(np.random.default_rng(4).normal(size=z.shape));v/=jnp.linalg.norm(v)
    tangent=jax.jvp(loss,(z,),(v,))[1]
    np.testing.assert_allclose(tangent,jnp.vdot(g,v),atol=1e-10,rtol=1e-8)
    fd=(loss(z+1e-4*v)-loss(z-1e-4*v))/(2e-4)
    np.testing.assert_allclose(tangent,fd,atol=1e-8,rtol=1e-5)


def test_implicit_invalid_and_zero_rhs():
    cfg=DifferentiationConfig(mode='implicit')
    x,info=checked_linear_solve(lambda x:2*x,jnp.zeros(3),cfg)
    np.testing.assert_array_equal(x,0);assert info.converged
    x,info=checked_linear_solve(lambda x:jnp.zeros_like(x),jnp.ones(3),cfg)
    assert not info.converged and np.isnan(x).all()
    model=TitrationModel(synthetic_cache(n=4),ModelConfig(steps=1),
                         backend='packed_v2',differentiation=cfg)
    p=jnp.full((4,20),.05)
    g=jax.grad(lambda x:model.charge(6.5)(x).sum())(p)
    assert not np.isfinite(g).all()


def test_implicit_midpoint_jvp():
    model=TitrationModel(synthetic_cache(n=3,neighbors=2),ModelConfig(steps=128),
        backend='packed_v2',differentiation=DifferentiationConfig(mode='implicit',
        implicit_residual_tolerance=1e-10,linear_rtol=1e-10,linear_atol=1e-12))
    p=jnp.full((3,20),.05)
    read=model.pka_sites([(1,'HIS')])
    f=lambda x:read(jax.nn.softmax(x,-1)).value.sum()
    z=jnp.log(p);v=jnp.asarray(np.random.default_rng(3).normal(size=z.shape))
    np.testing.assert_allclose(jax.jvp(f,(z,),(v,))[1],jnp.vdot(jax.grad(f)(z),v),atol=1e-8)


@pytest.mark.parametrize('kwargs',[dict(mode='bad'),dict(checkpoint_block_size=0),
                                  dict(linear_restart=True),dict(linear_atol=0)])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):DifferentiationConfig(**kwargs)


@pytest.mark.parametrize('mode',['checkpointed','implicit'])
@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
def test_frozen_identity_and_padding(mode,dtype):
    from jaxpropka.batching import pack_inputs
    from jaxpropka.model import packed_v2_curve_kernel,one_hot
    cache=synthetic_cache(n=4,neighbors=2)
    frozen=cache.frozen.copy();frozen[0]=True
    cache=replace(cache,frozen=frozen)
    diff=DifferentiationConfig(mode=mode)
    model=TitrationModel(cache,ModelConfig(steps=128),backend='packed_v2',differentiation=diff)
    p=.95*model.native_probabilities.astype(dtype)+.05/20
    gradient=jax.grad(lambda x:model.charge(6.7)(x).sum())(p)
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient[0],0)
    arrays,pp,_=pack_inputs(cache,np.asarray(p),bucket_multiple=(8,4,4))
    runtime=jax.device_put(pack_runtime(arrays))
    out=packed_v2_curve_kernel(runtime,jnp.asarray(pp),jnp.asarray([6.7],dtype),
                               config=model.config,differentiation=diff)
    np.testing.assert_allclose(out.total_charge,model.curves([6.7])(p).total_charge,atol=2e-6)


def test_checkpoint_removes_edge_history():
    from jaxpropka.batching import pack_inputs
    from jaxpropka.model import packed_curve_kernel,pack_interaction_edges,one_hot
    cache=synthetic_cache(n=6,neighbors=4)
    arrays,p,_=pack_inputs(cache,one_hot(cache.native_index),bucket_multiple=(1,1,1))
    edges=pack_interaction_edges(arrays['pair_mask'],arrays['neighbors'])
    ph=jnp.asarray([6.,6.7,7.4]);cfg=ModelConfig(steps=17)
    def shapes(jp):
        if hasattr(jp,'jaxpr'):jp=jp.jaxpr
        for equation in jp.eqns:
            if equation.primitive.name=='scan':
                for v in equation.outvars:
                    if hasattr(v.aval,'shape'):yield v.aval.shape
            for value in equation.params.values():
                if hasattr(value,'jaxpr') or hasattr(value,'eqns'):yield from shapes(value)
    for mode in ('unrolled','checkpointed','implicit'):
        def loss(x):
            return packed_curve_kernel(arrays,x,ph,edges,config=cfg,
                differentiation=DifferentiationConfig(mode=mode)).total_charge.sum()
        saved=set(shapes(jax.make_jaxpr(jax.value_and_grad(loss))(jnp.asarray(p))))
        assert ((17,3,len(edges[0])) in saved) == (mode=='unrolled')


def test_runtime_masks_environment_once_without_mutating_cache():
    from jaxpropka.batching import pack_inputs
    from jaxpropka.model import one_hot,curve_kernel,packed_v2_curve_kernel
    cache=synthetic_cache(n=4,neighbors=2)
    arrays,p,_=pack_inputs(cache,one_hot(cache.native_index),bucket_multiple=(8,4,4))
    for name in ('volume','mass','hbond'):
        arrays[name][~arrays['env_mask']]=123.
    runtime=pack_runtime(arrays)
    for name in ('volume','mass','hbond'):
        assert np.all(arrays[name][~arrays['env_mask']]==123.)
        assert np.all(runtime[name][~arrays['env_mask']]==0.)
    ph=jnp.asarray([6.,7.4]);cfg=ModelConfig()
    a=curve_kernel(arrays,jnp.asarray(p),ph,config=cfg)
    b=packed_v2_curve_kernel(runtime,jnp.asarray(p),ph,config=cfg)
    np.testing.assert_allclose(a.total_charge,b.total_charge,atol=2e-6)
