"""Diagnostic only: adaptive continuation and candidate mean-field free energies.

At fixed sequence probabilities, in kBT units (up to h-independent terms):
 F = sum_i w_i [h_i log h_i + (1-h_i) log(1-h_i)
               + ln(10)*(pH-intrinsic_i+field0_i)*h_i]
     + ln(10)/2 sum_ij w_i coupling_ij h_i h_j.
Its stationarity equation is the production fixed point IF weighted coupling
is reciprocal. We check the energy gradient against that equation at interior
occupancies, including a soft sequence. Energy ranks sampled states only; it
does not certify global minima, local stability, or smooth branch selection.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jaxpropka import ModelConfig, GROUPS
from jaxpropka.model import _local_terms, _field, curve_kernel


def free_energy(d, terms, h, ph):
    entropy = jax.scipy.special.xlogy(h,h)+jax.scipy.special.xlogy(1-h,1-h)
    pair_field = _field(d,terms,h)-terms.field0
    return jnp.sum(terms.weights*(entropy+jnp.log(10.)*(
        (ph-terms.intrinsic+terms.field0)*h+.5*h*pair_field)))


def refinement_points(ph, forward, reverse, weights, residuals):
    disagreement=np.max(np.abs(forward-reverse)*weights,axis=(1,2))>1e-3
    jumps=np.maximum(np.max(np.abs(np.diff(forward,axis=0))*weights,axis=(1,2)),
                     np.max(np.abs(np.diff(reverse,axis=0))*weights,axis=(1,2)))>.05
    bad=disagreement | (residuals>=2e-5)
    intervals=np.flatnonzero(jumps | bad[:-1] | bad[1:])
    return np.unique(np.r_[ph,*[np.linspace(ph[i],ph[i+1],5)[1:-1] for i in intervals]])


def audit_branches(d,p,cache,result,checkpoint):
    cfg=ModelConfig(steps=8192,damping=.15)
    terms=_local_terms(d,p,cfg); mask=d['group_mask']; w=np.asarray(terms.weights)
    result['branch_grid_policy']={'max_iterations':8192,'damping':.15,
        'stopping_residual':1e-9,'refinement_rounds':2,
        'scope':'native sequence; sampled branches, not global equilibrium certification'}
    checks=[]
    for mixture in [0.,.05]:
        t=_local_terms(d,(1-mixture)*p+mixture/20,cfg)
        h=jnp.asarray(np.random.default_rng(2026).uniform(.1,.9,p.shape[:1]+(9,)))
        actual=jax.grad(lambda x:free_energy(d,t,x,7.))(h)
        expected=t.weights*(jnp.log(h/(1-h))+jnp.log(10.)*(7.-t.intrinsic+_field(d,t,h)))
        error=float(jnp.max(jnp.abs(actual-expected)))
        checks.append({'mixture':mixture,'max_energy_gradient_identity_error':error})
        if error>2e-5:
            raise ValueError(f'free-energy stationarity check failed: {error}')
    result['free_energy_checks']=checks
    def target(h,x):
        return jnp.where(mask,jax.nn.sigmoid(jnp.log(10.)*(terms.intrinsic-x-_field(d,terms,h))),0.)
    def initial(x):
        return jnp.where(mask,jax.nn.sigmoid(jnp.log(10.)*(terms.intrinsic-x-terms.field0)),0.)
    @jax.jit
    def solve(h,x):
        def body(state):
            i,h,_=state
            h=h+.15*(target(h,x)-h)
            return i+1,h,jnp.max(jnp.abs(target(h,x)-h))
        i,h,r=jax.lax.while_loop(lambda s:(s[0]<8192)&(s[2]>=1e-9),body,
                                (0,h,jnp.asarray(jnp.inf)))
        return h,r,i,free_energy(d,terms,h,x)
    @jax.jit
    def sweep(xs):
        def step(h,x):
            out=solve(h,x)
            return out[0],out
        return jax.lax.scan(step,initial(xs[0]),xs)[1]
    baseline_ph=np.linspace(-10,24,577)
    baseline=curve_kernel(d,p,baseline_ph,config=ModelConfig(steps=512))
    critical=baseline_ph[np.argsort(np.asarray(baseline.weighted_residual))[-3:]]
    result['critical_ph']=critical.tolist()
    ph=np.unique(np.r_[np.linspace(-10,24,69),critical])
    # Widen only when the existing endpoints have unsaturated weighted occupancy.
    low=jax.device_get(solve(initial(ph[0]),ph[0])); high=jax.device_get(solve(initial(ph[-1]),ph[-1]))
    saturation=[float(np.max(w*(1-low[0]))),float(np.max(w*high[0]))]
    result['original_endpoint_unsaturation']=saturation
    if saturation[0]>1e-4: ph=np.unique(np.r_[np.arange(-20,-10,.5),ph])
    if saturation[1]>1e-4: ph=np.unique(np.r_[ph,np.arange(24.5,36.5,.5)])
    result['levels']=[]
    for level in range(3):
        f=jax.device_get(sweep(jnp.asarray(ph)))
        rev=jax.device_get(sweep(jnp.asarray(ph[::-1])))
        r=tuple(a[::-1] for a in rev)
        diff=np.max(np.abs(f[0]-r[0])*w,axis=(1,2))
        both=(f[1]<2e-5)&(r[1]<2e-5)
        ip=int(np.argmax(np.where(both,diff,-1)))
        record={'level':level,'points':len(ph),'ph':ph.tolist(),
                'forward_residual':f[1].tolist(),'reverse_residual':r[1].tolist(),
                'forward_iterations':f[2].tolist(),'reverse_iterations':r[2].tolist(),
                'weighted_sweep_difference':diff.tolist(),
                'forward_free_energy':f[3].tolist(),'reverse_free_energy':r[3].tolist(),
                'max_converged_sweep_difference':float(diff[both].max()) if both.any() else None,
                'endpoint_unsaturation':[float(np.max(w*(1-f[0][0]))),float(np.max(w*r[0][-1]))]}
        # Independent seeds at the strongest discrepancy and original trouble pHs.
        points=np.unique(np.r_[critical,ph[ip],ph[np.argsort(np.maximum(f[1],r[1]))[-3:]]])
        record['seed_checks']=[]
        for x in points:
            k=int(np.searchsorted(ph,x)); states=[('forward',tuple(a[k] for a in f)),('reverse',tuple(a[k] for a in r))]
            for name,h in [('default',initial(x)),('zero',jnp.zeros_like(p[:,:9])),('one',mask.astype(p.dtype))]:
                states.append((name,jax.device_get(solve(h,x))))
            converged=[s[0] for _,s in states if s[1]<2e-5]
            maxdiff=max((float(np.max(np.abs(a-b)*w)) for a in converged for b in converged),default=0.)
            record['seed_checks'].append({'ph':float(x),'max_converged_seed_difference':maxdiff,
                'states':[{'seed':name,'residual':float(s[1]),'iterations':int(s[2]),'free_energy':float(s[3])} for name,s in states]})
        i,g=np.unravel_index(np.argmax(np.abs(f[0][ip]-r[0][ip])*w),w.shape)
        record['largest_converged_difference_site']={'ph':float(ph[ip]),'residue':str(cache.keys[i]),'group':GROUPS[g]}
        result['levels'].append(record);checkpoint(f'branch_grid_level_{level}')
        if level<2:
            ph=refinement_points(ph,f[0],r[0],w,np.maximum(f[1],r[1]))
