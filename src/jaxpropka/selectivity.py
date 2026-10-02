"""Streamed, fixed-geometry bound/free charge objectives.

Diagnostics have environment order (bound, binder, target), then pH order.
The loss wrapper supports first-order JVPs/VJPs; higher derivatives are not
supported. Reconstruct the objective when geometry or fixed target P changes.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .cache import ResidueKey
from .differentiation import charge_value_and_grad_terms, SolverDiagnostics
from .parameters import Q_DEPROT


class SelectivityDiagnostics(NamedTuple):
    forward_residual: jax.Array
    forward_valid: jax.Array
    linear_residual: jax.Array
    linear_threshold: jax.Array
    linear_checked: jax.Array
    linear_valid: jax.Array


class SelectivityResult(NamedTuple):
    loss: jax.Array
    selectivity: jax.Array
    valid: jax.Array
    diagnostics: SelectivityDiagnostics


class SelectivityGradient(NamedTuple):
    loss: jax.Array
    selectivity: jax.Array
    gradient: jax.Array
    valid: jax.Array
    diagnostics: SelectivityDiagnostics


def trapezoid_weights(grid):
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all() or not np.all(np.diff(grid) > 0):
        raise ValueError('ph_grid must be finite, strictly increasing, and contain at least two points')
    delta = np.diff(grid)
    return np.concatenate((delta[:1]/2, (delta[:-1]+delta[1:])/2, delta[-1:]/2))


def _forward_tolerance(model):
    return (min(model.config.residual_tolerance, model.differentiation.implicit_residual_tolerance)
            if model.differentiation.mode == 'implicit' else model.config.residual_tolerance)


def _integral(model, p, grid, weights, chunk_size, with_gradient, with_solver_diagnostics=False):
    """One environment: accumulate term cotangents, then pull back to P once."""
    if with_gradient:
        terms, pullback = jax.vjp(model._local_terms, p)
    else:
        terms = model._local_terms(p)
    count = len(grid)
    padding = (-count) % chunk_size
    ph = jnp.asarray(np.pad(grid, (0,padding), mode='edge'), dtype=p.dtype).reshape(-1,chunk_size)
    qw = jnp.asarray(np.pad(weights, (0,padding)), dtype=p.dtype).reshape(-1,chunk_size)
    q0 = jnp.asarray(Q_DEPROT, dtype=p.dtype)
    implicit = model.differentiation.mode == 'implicit'

    if with_gradient:
        def one(hp):
            return charge_value_and_grad_terms(terms,hp,model._gm,model._field,
                                               model.config,model.differentiation,q0,with_diagnostics=True)
        def body(carry, inputs):
            integral, gradient = carry
            hp, w = inputs
            charge, g, residual, linear, valid, solver = jax.vmap(one)(hp)
            gradient = jax.tree.map(lambda old, new: old+jnp.tensordot(w,new,axes=1),gradient,g)
            diagnostic = (residual,solver.converged,linear.residual,
                          linear.threshold,jnp.full(hp.shape,implicit),linear.converged,
                          solver.iterations,solver.budget_exhausted,solver.nonfinite)
            return (integral+jnp.vdot(w,charge),gradient), diagnostic
        zero_gradient = jax.tree.map(jnp.zeros_like,terms)
        (integral,gradient), diagnostic = jax.lax.scan(body,(jnp.asarray(0,p.dtype),zero_gradient),(ph,qw))
        gradient = pullback(gradient)[0]
    else:
        def one(hp):
            h,residual,_,solver = model._solve_with_diagnostics(terms,hp)
            return jnp.sum(terms.weights*(q0+h)),residual,solver
        def body(integral,inputs):
            hp,w = inputs
            charge,residual,solver = jax.vmap(one)(hp)
            diagnostic = (residual,solver.converged,
                          jnp.full(hp.shape,jnp.nan,p.dtype),jnp.full(hp.shape,jnp.nan,p.dtype),
                          jnp.zeros(hp.shape,bool),jnp.ones(hp.shape,bool),
                          solver.iterations,solver.budget_exhausted,solver.nonfinite)
            return integral+jnp.vdot(w,charge),diagnostic
        integral,diagnostic = jax.lax.scan(body,jnp.asarray(0,p.dtype),(ph,qw))
        gradient = None
    values = tuple(x.reshape(-1)[:count] for x in diagnostic)
    diagnostic = SelectivityDiagnostics(*values[:6])
    result = integral,gradient,diagnostic
    solver = SolverDiagnostics(values[6],values[0],values[1],values[7],values[8])
    return (*result,solver) if with_solver_diagnostics else result


class SelectivityObjective:
    """Only binder probabilities are variable; environments are immutable.

    ``binder_keys`` determines the row order of P_binder. All free-binder sites
    must be represented. Fixed target probabilities use free-target row order.
    Input P must be floating, finite and row-stochastic; use validate_probabilities
    outside JIT. Structural geometry must be prepared consistently by the caller.
    """
    def __init__(self, bound, free_binder, free_target, ph_grid, *,
                 binder_keys=None, target_probabilities=None,
                 required_log10_ratio=1., tau=.1, ph_chunk_size=1):
        self.bound,self.free_binder,self.free_target = bound,free_binder,free_target
        self.ph_grid = np.array(ph_grid,dtype=float,copy=True)
        self.weights = trapezoid_weights(self.ph_grid)
        if not np.isfinite(required_log10_ratio) or not np.isfinite(tau) or tau <= 0:
            raise ValueError('required_log10_ratio must be finite and tau must be finite and positive')
        if isinstance(ph_chunk_size,bool) or not isinstance(ph_chunk_size,int) or ph_chunk_size < 1:
            raise ValueError('ph_chunk_size must be a positive Python int')
        self.required_log10_ratio,self.tau,self.ph_chunk_size = required_log10_ratio,tau,ph_chunk_size
        keys = (free_binder.cache.keys if binder_keys is None else
                tuple(x if isinstance(x,ResidueKey) else ResidueKey.from_value(x) for x in binder_keys))
        if len(set(keys)) != len(keys) or set(keys) != set(free_binder.cache.keys):
            raise ValueError('binder_keys must contain each free-binder residue exactly once')
        target_keys = free_target.cache.keys
        if set(keys)&set(target_keys) or set(bound.cache.keys) != set(keys)|set(target_keys):
            raise ValueError('bound residue keys must equal the disjoint union of binder and target keys')
        self.binder_keys = tuple(keys)
        self._bound_binder = jnp.asarray(bound.cache.select(keys))
        self._free_binder = jnp.asarray(free_binder.cache.select(keys))
        self._bound_target = jnp.asarray(bound.cache.select(target_keys))
        for model in (free_binder,free_target):
            indices = bound.cache.select(model.cache.keys)
            for name in ('native_index','frozen','group_mask'):
                if not np.array_equal(getattr(bound.cache,name)[indices],getattr(model.cache,name)):
                    raise ValueError(f'bound/free {name} mismatch; check identities, termini and covalent topology')
        target = free_target.native_probabilities if target_probabilities is None else target_probabilities
        self._target_p = np.array(free_target.validate_probabilities(target),copy=True)
        self._target_cache = {}
        self._target_solver = {}
        dtypes = [np.float32,np.float64] if jax.config.x64_enabled else [np.float32]
        # Evaluate outside any enclosing trace. Only scalar integrals and small
        # diagnostic arrays survive; target states/compiled executables are not retained here.
        for dtype in dtypes:
            evaluate = jax.jit(lambda p:_integral(free_target,p,self.ph_grid,self.weights,
                                                  self.ph_chunk_size,False,True))
            integral,_,diagnostic,solver = evaluate(jnp.asarray(self._target_p,dtype=dtype))
            self._target_cache[np.dtype(dtype).name] = (integral,diagnostic)
            self._target_solver[np.dtype(dtype).name] = solver
        self.evaluate = jax.jit(self._evaluate)
        self.value_and_grad = jax.jit(self._value_and_grad)
        self.evaluate_with_solver_diagnostics = jax.jit(lambda p:self._calculate(p,False,True))
        self.value_and_grad_with_solver_diagnostics = jax.jit(lambda p:self._calculate(p,True,True))

        @jax.custom_jvp
        def loss(p):
            return self.evaluate(p).loss

        @loss.defjvp
        def loss_jvp(primals,tangents):
            result = self.value_and_grad(primals[0])
            return result.loss,jnp.vdot(jax.lax.stop_gradient(result.gradient),tangents[0])

        self.loss = jax.jit(loss)

    def validate_probabilities(self,p):
        p = np.asarray(p)
        if (p.shape != (len(self.binder_keys),20) or p.dtype not in (np.float32,np.float64)
                or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1,atol=1e-5)):
            raise ValueError('expected finite row-stochastic binder P[N,20] in float32 or float64')
        return p

    def _probabilities(self,p):
        if p.shape != (len(self.binder_keys),20) or p.dtype not in (jnp.float32,jnp.float64):
            raise ValueError('expected floating binder P[N,20]')
        bound = self.bound.native_probabilities.astype(p.dtype)
        bound = bound.at[self._bound_target].set(jnp.asarray(self._target_p,p.dtype))
        bound = bound.at[self._bound_binder].set(p)
        free = self.free_binder.native_probabilities.astype(p.dtype).at[self._free_binder].set(p)
        return bound,free

    def _calculate(self,p,with_gradient,with_solver_diagnostics=False):
        bound,free = self._probabilities(p)
        ib,gb,db,sb = _integral(self.bound,bound,self.ph_grid,self.weights,self.ph_chunk_size,with_gradient,True)
        iff,gf,df,sf = _integral(self.free_binder,free,self.ph_grid,self.weights,self.ph_chunk_size,with_gradient,True)
        it,dt = self._target_cache[np.dtype(p.dtype).name]
        st = self._target_solver[np.dtype(p.dtype).name]
        solver = jax.tree.map(lambda *x:jnp.stack(x),sb,sf,st)
        finish = lambda result:(result,solver) if with_solver_diagnostics else result
        s = ib-iff-it
        loss = self.tau*jax.nn.softplus((self.required_log10_ratio-s)/self.tau)
        diagnostics = SelectivityDiagnostics(*[jnp.stack(x) for x in zip(db,df,dt)])
        valid = (jnp.all(diagnostics.forward_valid) & jnp.all(diagnostics.linear_valid)
                 & jnp.isfinite(s) & jnp.isfinite(loss))
        if not with_gradient:
            return finish(SelectivityResult(loss,s,valid,diagnostics))
        gradient = -jax.nn.sigmoid((self.required_log10_ratio-s)/self.tau)*(gb[self._bound_binder]-gf[self._free_binder])
        valid = valid & jnp.all(jnp.isfinite(gradient))
        return finish(SelectivityGradient(loss,s,jnp.where(valid,gradient,jnp.nan),valid,diagnostics))

    def audit(self,p,ph_grid=None):
        """Explicit five-path audit; never reused as certification of future P."""
        from .audit import audit_equilibrium
        grid = self.ph_grid if ph_grid is None else ph_grid
        bp,fp = self._probabilities(p)
        results = [audit_equilibrium(model,prob,grid) for model,prob in
            ((self.bound,bp),(self.free_binder,fp),(self.free_target,jnp.asarray(self._target_p,p.dtype)))]
        return jax.tree.map(lambda *x:jnp.stack(x),*results)

    def _evaluate(self,p):
        return self._calculate(p,False)

    def _value_and_grad(self,p):
        return self._calculate(p,True)
