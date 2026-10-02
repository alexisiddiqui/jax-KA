"""Explicit sampled-branch audits. Agreement is not a global minimum certificate."""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from .differentiation import forward_state

AUDIT_PATHS = ('standard','unprotonated','protonated','increasing','decreasing')


class EquilibriumAudit(NamedTuple):
    residual: jax.Array          # [5,H], AUDIT_PATHS order
    iterations: jax.Array
    converged: jax.Array
    budget_exhausted: jax.Array
    nonfinite: jax.Array
    max_occupancy_gap: jax.Array # [H], across every pair of paths
    consistent: jax.Array       # [H], all paths converged and agree


def audit_equilibrium(model, probabilities, ph_grid, *, atol=2e-5):
    """Audit the exact supplied P/geometry, without mutating model or warm starts.

    This is an explicit host entrypoint. No gradient is defined for the audit.
    SelectivityObjective.audit stacks these fields with environment axis first.
    """
    p = model.validate_probabilities(probabilities)
    grid = np.asarray(ph_grid,dtype=float)
    if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all() or not np.all(np.diff(grid)>0):
        raise ValueError('audit grid must be finite, nonempty and strictly increasing')
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError('audit tolerance must be positive and finite')
    key = (tuple(grid),float(atol))
    readouts = getattr(model,'_audit_readouts',None)
    if readouts is None:
        readouts = model._audit_readouts = {}
    if key in readouts:
        return readouts[key](jnp.asarray(p))
    @jax.jit
    def evaluate(p):
        terms = model._local_terms(p)
        ph = jnp.asarray(grid,p.dtype)
        solve = lambda hp,initial=None:forward_state(terms,hp,model._gm,model._field,
                                                      model.config,model.differentiation,initial)
        standard = jax.vmap(solve)(ph)
        low = jax.vmap(lambda hp:solve(hp,jnp.zeros_like(terms.intrinsic)))(ph)
        high = jax.vmap(lambda hp:solve(hp,model._gm.astype(p.dtype)))(ph)
        def continuation(xs):
            first = solve(xs[0])
            def step(old,hp):
                result = solve(hp,old)
                return result[0],result
            _,rest = jax.lax.scan(step,first[0],xs[1:])
            return jax.tree.map(lambda a,b:jnp.concatenate((a[None],b),axis=0),first,rest)
        up = continuation(ph)
        down = jax.tree.map(lambda x:x[::-1],continuation(ph[::-1]))
        states,diagnostic = jax.tree.map(lambda *x:jnp.stack(x),standard,low,high,up,down)
        # max-min equals the largest pairwise absolute gap, without a [5,5,H,N,G] tensor.
        gap = jnp.max(jnp.where(model._gm,states.max(0)-states.min(0),0),axis=(1,2))
        consistent = jnp.all(diagnostic.converged,axis=0) & jnp.isfinite(gap) & (gap <= atol)
        return EquilibriumAudit(diagnostic.residual,diagnostic.iterations,diagnostic.converged,
                                diagnostic.budget_exhausted,diagnostic.nonfinite,gap,consistent)
    # Cache executable readouts only, never audit outcomes for a previous P.
    readouts[key] = evaluate
    return evaluate(jnp.asarray(p))
