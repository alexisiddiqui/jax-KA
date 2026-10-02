"""Experimental equilibrium endpoint potential; not a branch-selection method.

``valid`` certifies numerical convergence only. Run ``audit`` and compare with
refined charge quadrature before interpreting endpoint differences as integrals.
Agreement of continuation paths does not establish a global free-energy minimum.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import xlogy

from .differentiation import forward_state, occupancy_target
from .parameters import Q_DEPROT
from .selectivity import SelectivityDiagnostics, SelectivityResult, SelectivityGradient, trapezoid_weights


def mean_field_potential(model, terms, occupancy, ph):
    """Potential in kBT ln(10) units, including the charge baseline term."""
    u = jnp.where(model._gm,occupancy,.5)
    entropy = (xlogy(u,u)+xlogy(1-u,1-u))/jnp.log(jnp.asarray(10,u.dtype))
    interaction = model._field(terms,occupancy)-terms.field0
    return jnp.sum(terms.weights*(entropy+(ph-terms.intrinsic+terms.field0)*occupancy
        +ph*jnp.asarray(Q_DEPROT,u.dtype)+.5*occupancy*interaction))


def _endpoints(model,p,endpoints,with_gradient):
    if with_gradient:
        terms,pullback = jax.vjp(model._local_terms,p)
    else:
        terms = model._local_terms(p)
    def one(ph):
        u,solver = jax.tree.map(jax.lax.stop_gradient,forward_state(terms,ph,model._gm,model._field,
                                              model.config,model.differentiation))
        residual = jnp.max(jnp.abs(occupancy_target(terms,u,ph,model._gm,model._field)-u))
        if with_gradient:
            value,g = jax.value_and_grad(lambda t:mean_field_potential(model,t,u,ph))(terms)
            return value,g,residual,solver.converged
        return mean_field_potential(model,terms,u,ph),residual,solver.converged
    hp = jnp.asarray(endpoints,p.dtype)
    if with_gradient:
        values,g,residual,converged = jax.vmap(one)(hp)
        gradient = pullback(jax.tree.map(lambda x:x[1]-x[0],g))[0]
    else:
        values,residual,converged = jax.vmap(one)(hp)
        gradient = None
    tol = min(model.config.residual_tolerance,model.differentiation.implicit_residual_tolerance)
    diagnostic = SelectivityDiagnostics(residual,converged & (residual<tol),jnp.full((2,),jnp.nan,p.dtype),
        jnp.full((2,),jnp.nan,p.dtype),jnp.zeros(2,bool),jnp.ones(2,bool))
    return values[1]-values[0],gradient,diagnostic


class BranchAudit(NamedTuple):
    max_occupancy_gap: jax.Array
    max_residual: jax.Array
    consistent: jax.Array


def audit_branches(model,p,ph_grid,atol=2e-5):
    """Compare independent, increasing-pH and decreasing-pH initialization paths."""
    terms = model._local_terms(p)
    grid = jnp.asarray(ph_grid,p.dtype)
    def state(ph,initial=None):
        h,solver = forward_state(terms,ph,model._gm,model._field,model.config,model.differentiation,initial)
        return jnp.where(solver.converged,h,jnp.nan)
    independent = jax.vmap(state)(grid)
    def continuation(hps):
        def body(old,hp):
            new = state(hp,old)
            return new,new
        return jax.lax.scan(body,state(hps[0]),hps)[1]
    up,down = continuation(grid),continuation(grid[::-1])[::-1]
    gap = jnp.maximum(jnp.max(jnp.abs(up-independent),axis=(1,2)),
                      jnp.max(jnp.abs(down-independent),axis=(1,2)))
    gap = jnp.maximum(gap,jnp.max(jnp.abs(up-down),axis=(1,2)))
    def residual(states):
        return jax.vmap(lambda h,hp:jnp.max(jnp.abs(
            occupancy_target(terms,h,hp,model._gm,model._field)-h)))(states,grid)
    error = jnp.maximum(jnp.maximum(residual(independent),residual(up)),residual(down))
    tol = min(model.config.residual_tolerance,model.differentiation.implicit_residual_tolerance)
    return BranchAudit(gap,error,(gap<=atol)&(error<tol))


class EndpointSelectivityObjective:
    """Experimental endpoint/envelope counterpart of a SelectivityObjective.

    First derivatives only. Its valid flag does not certify branch consistency;
    audit() is a separate, more expensive diagnostic, never an automatic fallback.
    """
    def __init__(self,objective):
        self.objective = objective
        self.endpoints = objective.ph_grid[[0,-1]]
        self._target = {}
        for name in objective._target_cache:
            result = jax.jit(lambda p:_endpoints(objective.free_target,p,self.endpoints,False))(
                jnp.asarray(objective._target_p,dtype=np.dtype(name)))
            self._target[name] = result[0],result[2]
        self.evaluate = jax.jit(lambda p:self._calculate(p,False))
        self.value_and_grad = jax.jit(lambda p:self._calculate(p,True))

        @jax.custom_jvp
        def loss(p):
            return self.evaluate(p).loss
        @loss.defjvp
        def derivative(primals,tangents):
            out = self.value_and_grad(primals[0])
            return out.loss,jnp.vdot(jax.lax.stop_gradient(out.gradient),tangents[0])
        self.loss = jax.jit(loss)

    def _calculate(self,p,with_gradient):
        obj = self.objective
        bp,fp = obj._probabilities(p)
        b,gb,db = _endpoints(obj.bound,bp,self.endpoints,with_gradient)
        f,gf,df = _endpoints(obj.free_binder,fp,self.endpoints,with_gradient)
        t,dt = self._target[np.dtype(p.dtype).name]
        s = b-f-t
        loss = obj.tau*jax.nn.softplus((obj.required_log10_ratio-s)/obj.tau)
        diagnostic = SelectivityDiagnostics(*[jnp.stack(x) for x in zip(db,df,dt)])
        valid = jnp.all(diagnostic.forward_valid)&jnp.isfinite(s)&jnp.isfinite(loss)
        if not with_gradient:
            return SelectivityResult(loss,s,valid,diagnostic)
        g = -jax.nn.sigmoid((obj.required_log10_ratio-s)/obj.tau)*(gb[obj._bound_binder]-gf[obj._free_binder])
        valid = valid&jnp.all(jnp.isfinite(g))
        return SelectivityGradient(loss,s,jnp.where(valid,g,jnp.nan),valid,diagnostic)

    def audit(self,p,ph_grid=None):
        obj = self.objective
        grid = np.linspace(*self.endpoints,129) if ph_grid is None else np.asarray(ph_grid,dtype=float)
        trapezoid_weights(grid)
        if not np.array_equal(grid[[0,-1]],self.endpoints):
            raise ValueError('audit grid must span the endpoint interval')
        if any(model.equilibrium is not None for model in (obj.bound,obj.free_binder,obj.free_target)):
            result = obj.audit(p,grid)
            return BranchAudit(result.max_occupancy_gap,jnp.max(result.residual,axis=1),result.consistent)
        bp,fp = obj._probabilities(p)
        results = [jax.jit(lambda x:audit_branches(model,x,grid))(prob) for model,prob in
            ((obj.bound,bp),(obj.free_binder,fp),(obj.free_target,jnp.asarray(obj._target_p,p.dtype)))]
        return BranchAudit(*(jnp.stack(x) for x in zip(*results)))
