"""Memory policies and checked, matrix-free equilibrium derivatives.

Geometry is constant. Implicit derivatives describe the locally selected fixed
point, not the finite iteration algorithm, and do not select an equilibrium branch.
"""
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import gmres


@dataclass(frozen=True)
class EquilibriumConfig:
    """Adaptive stopping; ModelConfig.steps remains the hard iteration cap."""
    min_steps: int = 128
    check_interval: int = 16
    consecutive_checks: int = 2

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive Python int')


@dataclass(frozen=True)
class DifferentiationConfig:
    mode: str = "unrolled"
    checkpoint_block_size: int = 16
    implicit_residual_tolerance: float = 1e-6
    linear_rtol: float = 1e-6
    linear_atol: float = 1e-8
    linear_restart: int = 32
    linear_maxiter: int = 20
    equilibrium: EquilibriumConfig | None = None

    def __post_init__(self):
        if self.mode not in ("unrolled", "checkpointed", "implicit"):
            raise ValueError("mode must be unrolled, checkpointed or implicit")
        if self.equilibrium is not None:
            if not isinstance(self.equilibrium, EquilibriumConfig) or self.mode != 'implicit':
                raise ValueError('equilibrium stopping requires EquilibriumConfig and implicit differentiation')
        for name in ("checkpoint_block_size", "linear_restart", "linear_maxiter"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive Python int")
        for name in ("implicit_residual_tolerance", "linear_rtol", "linear_atol"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class LinearDiagnostics(NamedTuple):
    residual: jax.Array
    threshold: jax.Array
    converged: jax.Array


class SolverDiagnostics(NamedTuple):
    iterations: jax.Array
    residual: jax.Array
    converged: jax.Array
    budget_exhausted: jax.Array
    nonfinite: jax.Array


def forward_tolerance(config, differentiation):
    return (min(config.residual_tolerance, differentiation.implicit_residual_tolerance)
            if differentiation.mode == 'implicit' else config.residual_tolerance)


def checked_linear_solve(matvec, rhs, config):
    """Solve and transpose-solve without differentiating Krylov iterations.

Failure poisoning is INSIDE the solve callbacks, keeping the custom JVP that
uses this operation linear in its tangent argument. Check actual residuals:
GMRES's info return alone is not a convergence certificate.
"""
    def solve(operator, b):
        def nonzero(b):
            x, _ = gmres(operator, b, tol=config.linear_rtol,
                         atol=config.linear_atol, restart=config.linear_restart,
                         maxiter=config.linear_maxiter, solve_method="incremental")
            residual = jnp.linalg.norm(operator(x) - b)
            threshold = jnp.maximum(config.linear_atol,
                                    config.linear_rtol * jnp.linalg.norm(b))
            valid = jnp.all(jnp.isfinite(x)) & jnp.isfinite(residual) & (residual <= threshold)
            return jnp.where(valid, x, jnp.nan), LinearDiagnostics(residual, threshold, valid)

        def zero(b):
            return jnp.zeros_like(b), LinearDiagnostics(
                jnp.asarray(0, b.dtype), jnp.asarray(config.linear_atol, b.dtype), jnp.asarray(True))

        return jax.lax.cond(jnp.all(b == 0), zero, nonzero, b)

    return jax.lax.custom_linear_solve(matvec, rhs, solve=solve,
                                      transpose_solve=solve, has_aux=True)


def occupancy_target(terms, h, ph, mask, field):
    ln10 = jnp.log(jnp.asarray(10, dtype=terms.intrinsic.dtype))
    return jnp.where(mask, jax.nn.sigmoid(ln10 * (terms.intrinsic - ph - field(terms, h))), 0)


def finite_state(terms, ph, mask, field, config, differentiation, initial=None):
    ln10 = jnp.log(jnp.asarray(10, dtype=terms.intrinsic.dtype))
    h = (jnp.where(mask, jax.nn.sigmoid(ln10 * (terms.intrinsic - ph - terms.field0)), 0)
         if initial is None else initial)

    def step(_, old):
        return old + config.damping * (occupancy_target(terms, old, ph, mask, field) - old)

    if differentiation.mode != "checkpointed":
        return jax.lax.fori_loop(0, config.steps, step, h)
    step = jax.checkpoint(step, prevent_cse=False)
    size = differentiation.checkpoint_block_size
    blocks, remainder = divmod(config.steps, size)

    def block(_, state):
        return jax.lax.fori_loop(0, size, step, state)

    h = jax.lax.fori_loop(0, blocks, jax.checkpoint(block, prevent_cse=False), h)
    return jax.lax.fori_loop(0, remainder, step, h)


def forward_state(terms, ph, mask, field, config, differentiation, initial=None):
    """Shared primal solve. Diagnostics and stopping decisions are nondifferentiable."""
    policy = differentiation.equilibrium
    tolerance = forward_tolerance(config, differentiation)
    target = lambda h: occupancy_target(terms, h, ph, mask, field)
    if policy is None:
        h = finite_state(terms, ph, mask, field, config, differentiation, initial)
        error = target(h)-h
        residual = jnp.max(jnp.abs(error))
        nonfinite = ~jnp.all(jnp.isfinite(h)) | ~jnp.all(jnp.isfinite(error))
        valid = ~nonfinite & (residual < tolerance)
        return h, SolverDiagnostics(jnp.asarray(config.steps), residual, valid, ~valid & ~nonfinite, nonfinite)
    if config.steps < policy.min_steps:
        raise ValueError('iteration cap must be at least equilibrium.min_steps')
    h = (jnp.where(mask, jax.nn.sigmoid(jnp.log(jnp.asarray(10., terms.intrinsic.dtype)) *
         (terms.intrinsic-ph-terms.field0)), 0) if initial is None else initial)
    def inspect(h):
        t = target(h)
        bad = ~jnp.all(jnp.isfinite(h)) | ~jnp.all(jnp.isfinite(t))
        return t, jnp.max(jnp.abs(t-h)), bad
    t, residual, bad = inspect(h)
    def condition(state):
        i, _, _, _, streak, bad = state
        return (i < config.steps) & (streak < policy.consecutive_checks) & ~bad
    def body(state):
        i, h, t, residual, streak, bad = state
        active = condition(state)
        new = h+config.damping*(t-h)
        nt, nr, nb = inspect(new)
        ni = i+1
        check = (ni >= policy.min_steps) & (((ni-policy.min_steps) % policy.check_interval == 0)
                                             | (ni == config.steps))
        ns = jnp.where(check, jnp.where((nr < tolerance) & ~nb, streak+1, 0), streak)
        # vmap(while_loop) runs to the slowest lane; freeze completed lanes.
        return tuple(jnp.where(active, a, b) for a,b in zip((ni,new,nt,nr,ns,nb),state))
    i,h,_,residual,streak,bad = jax.lax.while_loop(condition, body,
        (jnp.asarray(0),h,t,residual,jnp.asarray(0),bad))
    valid = ~bad & (streak >= policy.consecutive_checks)
    return h, SolverDiagnostics(i,residual,valid,(i >= config.steps) & ~valid & ~bad,bad)


@partial(jax.custom_jvp, nondiff_argnums=(4,5,6))
def _implicit_state(terms, ph, mask, context, field_function, config, differentiation):
    field = lambda t,h:field_function(context,t,h)
    return forward_state(terms,ph,mask,field,config,differentiation)


@_implicit_state.defjvp
def _implicit_state_jvp(field_function, config, differentiation, primals, tangents):
    terms,ph,mask,context = primals
    dt,dhp,_,_ = tangents  # Geometry differentiation is outside the model contract.
    field = lambda t,h:field_function(context,t,h)
    h,diagnostic = _implicit_state(terms,ph,mask,context,field_function,config,differentiation)
    target = lambda x:occupancy_target(terms,x,ph,mask,field)
    operator = lambda v:v-jax.jvp(target,(h,),(v,))[1]
    rhs = jax.jvp(lambda t,hp:occupancy_target(t,h,hp,mask,field),
                  (terms,ph),(dt,dhp))[1]
    tangent,_ = checked_linear_solve(operator,rhs,differentiation)
    zero = lambda x: jnp.zeros(x.shape, x.dtype if jnp.issubdtype(x.dtype,jnp.inexact) else jax.dtypes.float0)
    return (h,diagnostic),(tangent*jnp.where(diagnostic.converged,1.,jnp.nan),jax.tree.map(zero,diagnostic))


def solve_occupancy(terms, ph, mask, field, config, differentiation, *, context=None, field_function=None):
    """State plus unchanged full/weighted residual diagnostics.

    Explicit geometry arguments prevent traced arrays escaping through closures
    when the implicit rule is nested inside JIT, vmap and midpoint custom JVPs.
    """
    return solve_occupancy_with_diagnostics(terms,ph,mask,field,config,differentiation,
        context=context,field_function=field_function)[:3]


def solve_occupancy_with_diagnostics(terms, ph, mask, field, config, differentiation, *, context=None, field_function=None):
    if differentiation.mode == "implicit":
        h,diagnostic = _implicit_state(terms,ph,mask,context,field_function,config,differentiation)
    else:
        h,diagnostic = forward_state(terms,ph,mask,field,config,differentiation)
    error = jnp.abs(occupancy_target(terms, h, ph, mask, field) - h)
    return h, jnp.max(error), jnp.max(error * terms.weights), diagnostic


def charge_value_and_grad_terms(terms, ph, mask, field, config, differentiation, q0, *, with_diagnostics=False):
    """A single-pH charge VJP with explicit adjoint diagnostics."""
    if differentiation.mode != "implicit":
        def value(t):
            h, residual, _ = solve_occupancy(t, ph, mask, field, config, differentiation)
            return jnp.sum(t.weights * (q0 + h)), residual
        (charge, residual), gradient = jax.value_and_grad(value, has_aux=True)(terms)
        valid = jnp.isfinite(residual) & (residual < config.residual_tolerance)
        linear = LinearDiagnostics(jnp.asarray(0, charge.dtype), jnp.asarray(0, charge.dtype), jnp.asarray(True))
        result = charge, gradient, residual, linear, valid
        diagnostic = SolverDiagnostics(jnp.asarray(config.steps),residual,valid,~valid & jnp.isfinite(residual),~jnp.isfinite(residual))
        return (*result,diagnostic) if with_diagnostics else result

    h,diagnostic = jax.tree.map(jax.lax.stop_gradient, forward_state(terms, ph, mask, field, config, differentiation))
    target = lambda u: occupancy_target(terms, u, ph, mask, field)
    _, transpose = jax.vjp(target, h)
    operator = lambda v: v - transpose(v)[0]
    adjoint, linear = checked_linear_solve(operator, terms.weights, differentiation)
    _, terms_pullback = jax.vjp(lambda t: occupancy_target(t, h, ph, mask, field), terms)
    indirect = terms_pullback(adjoint)[0]
    direct = jax.grad(lambda t: jnp.sum(t.weights * (q0 + h)))(terms)
    residual = jnp.max(jnp.abs(target(h) - h))
    valid = diagnostic.converged & linear.converged
    gradient = jax.tree.map(lambda a, b: jnp.where(valid, a + b, jnp.nan), direct, indirect)
    result = jnp.sum(terms.weights * (q0 + h)), gradient, residual, linear, valid
    return (*result,diagnostic) if with_diagnostics else result
