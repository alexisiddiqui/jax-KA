"""Native fixed-sequence local stability; no production branch selection.

The entropy-preconditioned Hessian B=D H D, D=sqrt(h(1-h)/w),
has the same inertia as H at interior active sites. B=I+D A D where
A_ij=ln(10)*w_i*coupling_ij. Zero-weight sites are excluded.
Numerically saturated occupancies use the limiting B (variance zero).
Positive curvature certifies only a local minimum at a stationary solution,
not a global minimum or smooth gradients through branch switches.
"""
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from jaxpropka import ModelConfig
from jaxpropka.model import _field, _local_terms
from audit_branch_grid import free_energy


def active_interaction(d,terms):
    w=np.asarray(terms.weights).ravel(); active=np.flatnonzero(w>0)
    lookup=np.full(w.size,-1,dtype=int);lookup[active]=np.arange(len(active))
    neighbors=np.asarray(d['neighbors']); coupling=np.asarray(terms.coupling)
    groups=terms.weights.shape[1];rows=[];cols=[];values=[]
    for row,flat in enumerate(active):
        i,g=divmod(int(flat),groups)
        dest=(neighbors[i,:,None]*groups+np.arange(groups)).ravel()
        val=np.log(10.)*w[flat]*coupling[i,:,g,:].ravel()
        keep=(lookup[dest]>=0)&(val!=0)
        rows.extend([row]*int(keep.sum()));cols.extend(lookup[dest[keep]]);values.extend(val[keep])
    a=sparse.csr_matrix((values,(rows,cols)),shape=(len(active),len(active)))
    delta=a-a.T;error=float(np.max(np.abs(delta.data),initial=0))
    if error>1e-10:
        raise ValueError(f'weighted coupling not reciprocal: {error}')
    return active,w[active],(a+a.T)*.5,error


def curvature(a,weights,h):
    scale=np.sqrt(np.maximum(h*(1-h),0)/weights)
    b=sparse.eye(len(h),format='csr')+sparse.diags(scale)@a@sparse.diags(scale)
    if len(h)<=2:
        vals,vecs=np.linalg.eigh(b.toarray());value=vals[0];vector=vecs[:,0]
    else:
        vals,vecs=eigsh(b,k=1,which='SA',tol=1e-10,maxiter=10000,
                       v0=np.random.default_rng(2026).normal(size=len(h)))
        value=vals[0];vector=vecs[:,0]
    residual=float(np.linalg.norm(b@vector-value*vector))
    return {'minimum_preconditioned_hessian_eigenvalue':float(value),
            'eigenpair_residual':residual,
            'curvature_class':('unresolved' if residual>1e-7 else
                               'positive' if value>1e-6 else 'negative' if value < -1e-6 else 'marginal'),
            'numerically_saturated_active_sites':int(np.sum((h<=0)|(h>=1)))}


def audit_stability(d,p,cache,result,checkpoint,report_path):
    source=report_path.read_bytes();previous=json.loads(source)
    assert previous['status']=='complete'
    for key in ['case_index','cache_fingerprint','model_sha256']:
        assert previous[key]==result[key],key
    result['branch_report']={'path':str(report_path.resolve()),'sha256':hashlib.sha256(source).hexdigest()}
    result['diagnostic_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    level=previous['levels'][-1];xs=np.asarray(level['ph'])
    difference=np.asarray(level['weighted_sweep_difference'])
    eligible=(np.asarray(level['forward_residual'])<2e-5)&(np.asarray(level['reverse_residual'])<2e-5)&(difference>1e-3)
    assert eligible.any()
    gap=np.asarray(level['forward_free_energy'])-np.asarray(level['reverse_free_energy'])
    picks=np.unique([int(np.argmax(np.where(eligible,difference,-1))),
                     int(np.argmin(np.where(eligible,np.abs(gap),np.inf))),
                     int(np.argmin(np.abs(xs-previous['critical_ph'][-1])))])
    result['selected_ph']=xs[picks].tolist()
    result['scope']='native sequence; selected pH; local curvature and sampled energy ranks only'
    cfg=ModelConfig();terms=_local_terms(d,p,cfg);mask=d['group_mask'];w=np.asarray(terms.weights)
    active,weights,a,error=active_interaction(d,terms)
    result['weighted_reciprocity_error']=error
    result['active_sites']=len(active)
    def target(h,x):
        return jnp.where(mask,jax.nn.sigmoid(jnp.log(10.)*(terms.intrinsic-x-_field(d,terms,h))),0.)
    def initial(x):
        return jnp.where(mask,jax.nn.sigmoid(jnp.log(10.)*(terms.intrinsic-x-terms.field0)),0.)
    def solve(h,x,limit,tolerance):
        def step(s):
            i,h,_=s;h=h+.15*(target(h,x)-h)
            return i+1,h,jnp.max(jnp.abs(target(h,x)-h))
        i,h,r=jax.lax.while_loop(lambda s:(s[0]<limit)&(s[2]>=tolerance),step,(0,h,jnp.asarray(jnp.inf)))
        return h,r,i
    @jax.jit
    def sweep(ph):
        def step(h,x):
            new,r,_=solve(h,x,8192,1e-9)
            return new,(new,r)
        return jax.lax.scan(step,initial(ph[0]),ph)[1]
    refine=jax.jit(lambda h,x:solve(h,x,16384,1e-11))
    forward=jax.device_get(sweep(jnp.asarray(xs)))
    checkpoint('forward_reproduced')
    reverse=jax.device_get(sweep(jnp.asarray(xs[::-1])))
    reverse=tuple(x[::-1] for x in reverse)
    reproduced=np.max(np.abs(forward[0]-reverse[0])*w,axis=(1,2))
    result['sweep_difference_reproduction_error']=float(np.max(np.abs(reproduced-difference)))
    assert result['sweep_difference_reproduction_error']<1e-7
    checkpoint('reverse_reproduced')
    result['points']=[]
    for k in picks:
        x=float(xs[k]);states=[];occupancies=[]
        seeds=[('forward',forward[0][k]),('reverse',reverse[0][k]),
               ('default',initial(x)),('zero',jnp.zeros_like(mask,dtype=p.dtype)),('one',mask.astype(p.dtype))]
        for name,seed in seeds:
            h,r,it=jax.device_get(refine(jnp.asarray(seed),x))
            c=curvature(a,weights,h.ravel()[active])
            states.append({'seed':name,'residual':float(r),'iterations':int(it),
                'free_energy_kbt':float(free_energy(d,terms,jnp.asarray(h),x)),**c,
                'local_minimum_supported':bool(r<1e-9 and c['curvature_class']=='positive')})
            occupancies.append(h)
        energies=[s['free_energy_kbt'] for s in states if s['local_minimum_supported']]
        for s in states:
            s['energy_above_lowest_sampled_local_minimum_kbt']=s['free_energy_kbt']-min(energies) if energies else None
        result['points'].append({'ph':x,'states':states,
            'weighted_occupancy_distances':[[float(np.max(np.abs(hi-hj)*w)) for hj in occupancies] for hi in occupancies]})
        checkpoint(f'stability_ph_{x:.6f}')
