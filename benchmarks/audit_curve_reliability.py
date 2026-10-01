"""Curve-first diagnostic: residuals, seeds, sweeps and soft-logit gradients.

No model parameters are fitted and no production solver settings are changed.
Gradient checks use a representative charge-curve loss, not a binder objective.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxpropka import ModelConfig, GROUPS
from jaxpropka.batching import pack_inputs
from jaxpropka.geometry import build_candidates
from jaxpropka.model import _field, _local_terms, curve_kernel, one_hot
from jaxpropka.precompute import build_cache
from jaxpropka.topology import load_topology

jax.config.update('jax_enable_x64',True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mixtures',type=float,nargs='+',default=[.05,.5])
    parser.add_argument('--branch-grid',action='store_true',help='adaptive native branch study, without gradient audit')
    parser.add_argument('--branch-stability',type=Path,help='completed branch-grid report to reproduce and assess')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    baseline=json.loads((root.parent/'_runtime/jax-Ka/cuda12/benchmarks/foldbench-cpu/723644/summary.json').read_text())
    case=baseline['cases'][args.case_index]
    result={'pdb_id':case['pdb_id'],'case_index':args.case_index,'status':'running',
            'scope':'sampled numerical checks, not proof of unique equilibrium or physical accuracy',
            'hostname':__import__('os').uname().nodename,
            'jax_version':jax.__version__, 'source_sha256':case['source_sha256'],
            'model_sha256':hashlib.sha256((root/'src/jaxpropka/model.py').read_bytes()).hexdigest()}
    started=time.perf_counter()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint(stage):
        result['stage']=stage;result['elapsed_seconds']=time.perf_counter()-started
        temporary=args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(args.output)
        print(case['pdb_id'],stage,round(result['elapsed_seconds'],1),flush=True)
    from biotite.structure.io import pdbx
    with tarfile.open(root/'ground_truth_1522.tar') as archive:
        member=next(m for m in archive.getmembers() if Path(m.name).name==case['pdb_id']+'.cif')
        source=archive.extractfile(member).read()
    assert hashlib.sha256(source).hexdigest()==case['source_sha256']
    atoms=pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(source.decode())),model=1,
                              altloc='occupancy',use_author_fields=False,include_bonds=True)
    topology=load_topology(atoms[atoms.chain_id==case['chain_id']],gap_policy='free',
                           freeze_disulfides=True,ignore_nonprotein=True)
    topology.metadata.update(source_name=case['pdb_id']+'.cif',source_sha256=case['source_sha256'],
                             foldbench_pdb_id=case['pdb_id'],foldbench_label_asym_id=case['chain_id'])
    cache=build_cache(topology,build_candidates(topology,missing_sidechain='template'))
    arrays,p,n=pack_inputs(cache,one_hot(cache.native_index),bucket_multiple=(1,1,1))
    d=jax.tree.map(jnp.asarray,arrays);p=jnp.asarray(p)
    result['n_residues']=n;result['cache_fingerprint']=cache.fingerprint()
    assert result['cache_fingerprint']==case['cache_fingerprint']
    checkpoint('cache_ready')
    if args.branch_stability:
        if args.branch_grid:
            raise ValueError('choose branch-grid or branch-stability, not both')
        from audit_branch_stability import audit_stability
        audit_stability(d,p.astype(jnp.float64),cache,result,checkpoint,args.branch_stability)
        result['status']='complete';checkpoint('complete')
        return
    if args.branch_grid:
        from audit_branch_grid import audit_branches
        audit_branches(d,p.astype(jnp.float64),cache,result,checkpoint)
        result['status']='complete';checkpoint('complete')
        return
    cfg=ModelConfig(steps=512)
    full_ph=np.linspace(-10,24,577,dtype=np.float32)
    out=jax.device_get(curve_kernel(d,p,full_ph,config=cfg))
    result['native_full_grid_512']={
        'points':len(full_ph),'max_residual':float(np.max(out.residual)),
        'max_weighted_residual':float(np.max(out.weighted_residual)),
        'unconverged_ph':full_ph[~out.converged].astype(float).tolist(),
        'finite':bool(np.isfinite(out.protonated).all()),
        'occupancy_bounds':[float(out.protonated.min()),float(out.protonated.max())],
        'max_total_charge_rise':float(np.max(np.diff(out.total_charge))),
        'charge_accounting_error':float(np.max(np.abs(out.site_charge.sum((1,2))-out.total_charge)))}
    worst=np.argsort(out.weighted_residual)[-3:]
    critical=full_ph[worst].astype(float)
    ph=np.unique(np.r_[5.,7.,9.,critical,critical-.03,critical+.03])
    result['targeted_ph']=ph.tolist()
    checkpoint('native_full_grid')
    p64=p.astype(jnp.float64);terms=_local_terms(d,p64,cfg)
    mask=d['group_mask'];weights=np.asarray(terms.weights)
    def target(h,x):
        return jnp.where(mask,jax.nn.sigmoid(jnp.log(10.)*(terms.intrinsic-x-_field(d,terms,h))),0)
    def initial(x):
        return jnp.where(mask,jax.nn.sigmoid(jnp.log(10.)*(terms.intrinsic-x-terms.field0)),0)
    @partial(jax.jit,static_argnames=('steps','damping'))
    def from_seed(h,x,*,steps,damping):
        h=jax.lax.fori_loop(0,steps,lambda _,old:old+damping*(target(old,x)-old),h)
        error=jnp.abs(target(h,x)-h)
        return h,jnp.max(error),jnp.max(error*terms.weights)
    def evaluate_seed(seed,steps,damping):
        init=jax.vmap(initial)(jnp.asarray(ph)) if seed=='default' else jnp.broadcast_to(
            jnp.asarray(mask,dtype=jnp.float64)*(1 if seed=='one' else 0),(len(ph),n,9))
        return jax.device_get(jax.vmap(lambda h,x:from_seed(h,x,steps=steps,damping=damping))(init,jnp.asarray(ph)))
    native=[];solutions={}
    for steps,damping,seed in [(512,.35,'default'),(2048,.35,'default'),(4096,.35,'default'),
                               (4096,.15,'default'),(4096,.15,'zero'),(4096,.15,'one')]:
        h,r,wr=evaluate_seed(seed,steps,damping)
        name=f'{steps}/{damping}/{seed}';solutions[name]=h
        native.append({'steps':steps,'damping':damping,'seed':seed,'dtype':'float64',
                       'max_residual':float(r.max()),'max_weighted_residual':float(wr.max()),
                       'unconverged_ph':ph[r>=cfg.residual_tolerance].tolist()})
    result['native_targeted_solves']=native
    ref=solutions['4096/0.15/default']
    result['native_seed_or_setting_max_weighted_differences']={
        name:float(np.max(np.abs(h-ref)*weights)) for name,h in solutions.items()}
    # Verify this diagnostic's update is exactly the production update.
    production=jax.device_get(curve_kernel(d,p64,ph,config=cfg))
    result['diagnostic_update_parity']=float(np.max(np.abs(production.protonated-solutions['512/0.35/default'])))
    @jax.jit
    def continuation(xs):
        def step(old,x):
            h,r,wr=from_seed(old,x,steps=4096,damping=.15)
            return h,(h,r,wr)
        return jax.lax.scan(step,initial(xs[0]),xs)[1]
    forward=jax.device_get(continuation(jnp.asarray(ph)))
    reverse=jax.device_get(continuation(jnp.asarray(ph[::-1])))
    difference=np.abs(forward[0]-reverse[0][::-1])*weights
    ip,i,g=np.unravel_index(np.argmax(difference),difference.shape)
    result['targeted_continuation']={
        'steps':4096,'damping':.15,'max_forward_residual':float(forward[1].max()),
        'max_reverse_residual':float(reverse[1].max()),
        'max_weighted_sweep_difference':float(np.max(np.abs(forward[0]-reverse[0][::-1])*weights)),
        'max_weighted_difference_from_independent':float(np.max(np.abs(forward[0]-ref)*weights))}
    result['targeted_continuation']['largest_difference_site']={
        'ph':float(ph[ip]),'residue':str(cache.keys[i]),'group':GROUPS[g],
        'forward_occupancy':float(forward[0][ip,i,g]),
        'reverse_occupancy':float(reverse[0][::-1][ip,i,g])}
    checkpoint('native_seeds_and_sweeps')
    gradient_ph=np.unique(np.r_[5.,7.,9.,critical[-1]])
    result['gradient_ph']=gradient_ph.tolist()
    result['gradient_objective']='mean squared residue charge over all residues and selected pH values; softmax logits'
    result['gradients']=[]
    rng=np.random.default_rng(2026)
    direction=rng.normal(size=p.shape);direction-=direction.mean(-1,keepdims=True)
    direction[np.asarray(cache.frozen)]=0;direction/=np.linalg.norm(direction)
    for mixture in args.mixtures:
        if not 0 < mixture <= 1:
            raise ValueError('mixtures must be in (0,1]')
        probability=(1-mixture)*np.asarray(p64)+mixture/20
        x=jnp.asarray(np.log(probability));v=jnp.asarray(direction)
        gradients={}
        for steps in [512,2048]:
            config=replace(cfg,steps=steps)
            def objective(z):
                o=curve_kernel(d,jax.nn.softmax(z,-1),jnp.asarray(gradient_ph,dtype=z.dtype),config=config)
                return jnp.mean(o.residue_charge**2)
            fn=jax.jit(objective);derivative=jax.jit(jax.value_and_grad(objective))
            value,gradient=jax.device_get(derivative(x));gradients[steps]=gradient
            auto=float(np.vdot(gradient,direction))
            _,jvp=jax.jvp(fn,(x,),(v,))
            finite=[]
            for eps in [1e-2,1e-3,1e-4]:
                a=fn(x+eps*v);b=fn(x-eps*v)
                fd=float((a-b)/(2*eps))
                finite.append({'epsilon':eps,'finite_difference':fd,'absolute_error':abs(fd-auto),
                               'relative_error':abs(fd-auto)/max(abs(fd),abs(auto),1e-9)})
            o=jax.device_get(curve_kernel(d,jax.nn.softmax(x,-1),gradient_ph,config=config))
            record={'uniform_mixture':mixture,'steps':steps,'dtype':'float64',
                    'loss':float(value),'gradient_norm':float(np.linalg.norm(gradient)),
                    'finite_gradient':bool(np.isfinite(gradient).all()),
                    'logit_gauge_error':float(np.max(np.abs(gradient.sum(-1)))),
                    'directional_autodiff':auto,'jvp_vjp_error':abs(float(jvp)-auto),
                    'finite_differences':finite,'max_residual':float(o.residual.max())}
            result['gradients'].append(record)
            if steps==2048:
                val32,grad32=jax.device_get(derivative(x.astype(jnp.float32)))
                record['float32_loss_difference']=abs(float(val32)-float(value))
                record['float32_gradient_relative_difference']=float(np.linalg.norm(grad32-gradient)/max(np.linalg.norm(gradient),1e-12))
            checkpoint(f'gradient_{mixture}_{steps}')
        result.setdefault('gradient_step_stability',[]).append({
            'uniform_mixture':mixture,'relative_l2_difference':float(np.linalg.norm(gradients[512]-gradients[2048])/max(np.linalg.norm(gradients[2048]),1e-12))})
    result['status']='complete';checkpoint('complete')


if __name__=='__main__':
    main()
