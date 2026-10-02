from dataclasses import replace
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxpropka import (TitrationModel,ModelConfig,DifferentiationConfig,EquilibriumConfig,
                       SelectivityObjective,audit_equilibrium)
from jaxpropka.differentiation import forward_state
from jaxpropka.model import LocalTerms
from jaxpropka.synthetic import synthetic_cache
from test_selectivity import subset


def design(cache=None,steps=256):
    return TitrationModel.for_design(cache or synthetic_cache(n=4,neighbors=2),ModelConfig(steps=steps),
        equilibrium=EquilibriumConfig(min_steps=16,check_interval=8),
        differentiation=DifferentiationConfig(mode='implicit',implicit_residual_tolerance=1e-10,
                                               linear_rtol=1e-10,linear_atol=1e-12))


def test_preset_and_legacy_defaults():
    cache=synthetic_cache(n=3)
    old=TitrationModel(cache);new=TitrationModel.for_design(cache)
    assert (old.backend,old.config.steps,old.differentiation.mode,old.equilibrium)==('dense',64,'unrolled',None)
    assert (new.backend,new.config.steps,new.differentiation.mode)==('packed_v2',1024,'implicit')
    with pytest.raises(ValueError):TitrationModel.for_design(cache,ModelConfig(steps=64))
    with pytest.raises(ValueError):DifferentiationConfig(equilibrium=EquilibriumConfig())
    with pytest.raises(ValueError):EquilibriumConfig(check_interval=0)


def test_adaptive_jvp_vjp_reference_and_finite_difference():
    m=design();p=.95*m.native_probabilities+.05/20
    p=p.astype(jnp.float64);z=jnp.log(p)
    read=m.curves_with_solver_diagnostics([6.,6.7,7.4])
    out,diagnostic=read(p)
    assert diagnostic.converged.all() and (diagnostic.iterations<256).all()
    assert not diagnostic.budget_exhausted.any() and not diagnostic.nonfinite.any()
    legacy=TitrationModel(m.cache,ModelConfig(steps=1024),backend='packed_v2')
    reference=lambda zz:legacy.curves([6.,6.7,7.4])(jax.nn.softmax(zz,-1)).total_charge.sum()
    f=lambda zz:read(jax.nn.softmax(zz,-1))[0].total_charge.sum()
    g=jax.grad(f)(z)
    np.testing.assert_allclose(g,jax.grad(reference)(z),atol=1e-8,rtol=1e-6)
    v=jnp.asarray(np.random.default_rng(4).normal(size=z.shape));v/=jnp.linalg.norm(v)
    tangent=jax.jvp(f,(z,),(v,))[1]
    np.testing.assert_allclose(tangent,jnp.vdot(g,v),atol=1e-10)
    np.testing.assert_allclose(tangent,(f(z+1e-4*v)-f(z-1e-4*v))/2e-4,atol=1e-8,rtol=1e-5)
    assert len(out)==len(legacy.curves([6.,6.7,7.4])(p))


def test_exact_budget_and_consecutive_check_failure():
    m=design(steps=17)
    p=jnp.full((4,20),.05)
    out,d=m.curves_with_solver_diagnostics([6.,7.4])(p)
    assert (d.iterations==17).all() and d.budget_exhausted.all()
    assert not out.converged.any()
    assert not np.isfinite(jax.grad(lambda x:m.charge(6.7)(x).sum())(p)).all()
    # Even an initially exact equilibrium needs both scheduled checks.
    terms=LocalTerms(jnp.zeros((1,1)),jnp.zeros((1,1)),jnp.zeros((1,)),jnp.ones((1,1)),jnp.zeros((1,1)))
    diff=DifferentiationConfig(mode='implicit',equilibrium=EquilibriumConfig(min_steps=16,check_interval=8))
    _,d=forward_state(terms,0.,jnp.ones((1,1),bool),lambda t,h:jnp.zeros_like(h),ModelConfig(steps=16),diff)
    assert d.residual==0 and not d.converged and d.budget_exhausted


def test_nonfinite_and_vmap_lane_freezing():
    m=design();p=jnp.full((4,20),.05);terms=m._local_terms(p)
    solve=lambda ph:forward_state(terms,ph,m._gm,m._field,m.config,m.differentiation)
    ph=jnp.asarray([6.,10.,jnp.nan])
    states,ds=jax.jit(jax.vmap(solve))(ph)
    for i in range(2):
        state,d=jax.jit(solve)(ph[i])
        np.testing.assert_allclose(states[i],state,atol=1e-14,rtol=1e-14)
        assert ds.iterations[i]==d.iterations
    assert ds.nonfinite[2] and not ds.converged[2] and ds.iterations[2]==0


@pytest.mark.parametrize('chunk',[1,2,5])
def test_streamed_diagnostics_and_endpoint(chunk):
    cache=synthetic_cache(n=6,neighbors=4)
    models=[design(c) for c in (cache,subset(cache,0),subset(cache,1))]
    objective=SelectivityObjective(*models,np.linspace(6.,7.4,5),ph_chunk_size=chunk)
    p=.95*models[1].native_probabilities.astype(jnp.float64)+.05/20
    out,ds=objective.value_and_grad_with_solver_diagnostics(p)
    plain=objective.value_and_grad(p)
    assert out.valid and ds.converged.all() and ds.iterations.shape==(3,5)
    np.testing.assert_allclose(out.gradient,plain.gradient,atol=1e-12)
    np.testing.assert_allclose(jax.grad(objective.loss)(p),out.gradient,atol=1e-12)
    _,ds2=objective.evaluate_with_solver_diagnostics(p)
    np.testing.assert_array_equal(ds.iterations,ds2.iterations)
    if chunk==1:
        assert objective.audit(p).consistent.all()
        from jaxpropka.experimental import EndpointSelectivityObjective
        endpoint=EndpointSelectivityObjective(objective)
        assert endpoint.value_and_grad(p).valid and endpoint.audit(p,np.linspace(6.,7.4,5)).consistent.all()


def test_multistart_audit_reports_disagreement_without_selecting_branch():
    mask=jnp.asarray([[True]])
    terms=LocalTerms(jnp.ones((1,1)),jnp.zeros((1,1)),jnp.zeros(1),jnp.ones((1,1)),jnp.zeros((1,1)))
    model=SimpleNamespace(validate_probabilities=lambda p:np.asarray(p),_local_terms=lambda p:terms,
        _gm=mask,_field=lambda t,h:-10*h,config=ModelConfig(steps=512),
        differentiation=DifferentiationConfig(mode='implicit',equilibrium=EquilibriumConfig()))
    result=audit_equilibrium(model,np.ones((1,20))/20,[6.])
    assert result.converged.all() and result.max_occupancy_gap[0]>.9
    assert not result.consistent.any()


def test_adaptive_midpoint_and_audit_validation():
    m=design();p=jnp.full((4,20),.05)
    f=lambda z:m.pka_sites([(1,'HIS')])(jax.nn.softmax(z,-1)).value.sum()
    z=jnp.log(p);v=jnp.ones_like(z)
    np.testing.assert_allclose(jax.jvp(f,(z,),(v,))[1],jnp.vdot(jax.grad(f)(z),v),atol=1e-8)
    with pytest.raises(ValueError):m.audit(p,[7.,6.])


def test_failed_adaptive_midpoint_cannot_return_a_usable_gradient():
    m=design(steps=16);p=jnp.full((4,20),.05)
    read=m.pka_sites([(1,'HIS')])
    assert not read(p).valid.any()
    assert not np.isfinite(jax.grad(lambda x:read(x).value.sum())(p)).all()


def test_batched_sequence_gradients():
    m=design();read=m.charge([6.,7.4])
    p=jnp.stack([.95*m.native_probabilities+.05/20,.999*m.native_probabilities+.001/20]).astype(jnp.float64)
    loss=lambda x:read(x).sum()
    batch=jax.jit(jax.vmap(jax.value_and_grad(loss)))(p)
    for i in range(2):
        single=jax.jit(jax.value_and_grad(loss))(p[i])
        for actual,expected in zip(batch,single):
            np.testing.assert_allclose(actual[i],expected,atol=1e-9,rtol=1e-7)
