"""Prepare protein-protein caches and measure gradient modes in fresh processes."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
import types


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / '_runtime/jax-Ka/cuda12/benchmarks'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def prepare_case(args):
    import numpy as np
    from biotite.structure.io import pdbx
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache
    from jaxpropka.topology import load_topology
    report = RUNTIME / 'interfaces/724720' / f'report-{args.case}.json'
    case = json.loads(report.read_text())['cases'][0]
    destination = args.output_dir / f'case-{args.case}'
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ROOT / 'ground_truth_1522.tar') as archive:
        member = next(m for m in archive.getmembers() if Path(m.name).name == case['pdb_id']+'.cif')
        raw = archive.extractfile(member).read()
    if hashlib.sha256(raw).hexdigest() != case['source_sha256']:
        raise ValueError('source fingerprint mismatch')
    atoms = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(raw.decode())), model=1,
                              altloc='occupancy', use_author_fields=False, include_bonds=True)
    chains = case['partner_chains']
    result = dict(case=args.case, source_report=str(report), source_sha256=case['source_sha256'],
                  interface_id=case['interface_id'], chains=chains, environments={})
    environments = [('bound',chains)] if args.bound_only else [('bound',chains), ('binder',chains[:1]), ('target',chains[1:])]
    for name, selected in environments:
        started = time.perf_counter()
        topology = load_topology(atoms[np.isin(atoms.chain_id,selected)], gap_policy='free',
                                 freeze_disulfides=True, ignore_nonprotein=True)
        topology.metadata.update(source_name=case['pdb_id']+'.cif',source_sha256=case['source_sha256'],
                                 foldbench_pdb_id=case['pdb_id'],foldbench_label_asym_id=case['chain_id'])
        cache = build_cache(topology,build_candidates(topology,missing_sidechain='template'))
        if name == 'bound' and cache.fingerprint() != case['cache_fingerprint']:
            raise ValueError('bound cache fingerprint mismatch')
        cache.save(destination / f'{name}.npz')
        result['environments'][name] = dict(n=cache.n_residues, bytes=cache.nbytes,
            edges=int(np.count_nonzero(cache.pair_mask)), fingerprint=cache.fingerprint(),
            preparation_seconds=time.perf_counter()-started)
        write_json(destination/'preparation.json',result)
        print(args.case,name,result['environments'][name],flush=True)
        del cache


def baseline_module():
    source = subprocess.check_output(['git','show',
        'df5abd608cbb3c92a7e1a0097ed19f38cf2d19fb:src/jaxpropka/model.py'],cwd=ROOT,text=True)
    module = types.ModuleType('jaxpropka._gradient_baseline')
    module.__package__ = 'jaxpropka'
    sys.modules[module.__name__] = module
    exec(compile(source,'<baseline df5abd6 model.py>','exec'),module.__dict__)
    return module


def benchmark(args):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaxpropka import ModelConfig,DifferentiationConfig,TitrationModel,SelectivityObjective
    from jaxpropka.cache import StructureCache
    from jaxpropka.batching import pack_inputs
    from jaxpropka.model import packed_curve_kernel,packed_v2_curve_kernel,pack_runtime,pack_interaction_edges,one_hot
    from jaxpropka.synthetic import synthetic_cache
    jax.config.update('jax_enable_compilation_cache',False)
    started = time.perf_counter()
    name = f'{args.workload}-{args.variant}-t{args.steps}-h{args.points}'
    destination = args.output_dir/f'case-{args.case}'
    destination.mkdir(parents=True,exist_ok=True)
    cfg = ModelConfig(steps=args.steps)
    mode = 'implicit' if args.variant == 'implicit' else 'checkpointed'
    diff = DifferentiationConfig(mode=mode)
    result = dict(case=args.case,workload=args.workload,variant=args.variant,steps=args.steps,
        points=args.points,hostname=os.uname().nodename,affinity_cpus=len(os.sched_getaffinity(0)),
        jax_version=jax.__version__,backend=jax.default_backend(),dtype='float32',records=[],
        model_config=vars(cfg),differentiation_config=(None if args.variant=='baseline' else vars(diff)),
        ph_chunk_size=args.chunk_size,benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interface=('dynamic_curve_kernel' if args.workload=='complex' else
                   'original_model_curves' if args.variant=='baseline' else 'selectivity_objective'),
        baseline_commit='df5abd608cbb3c92a7e1a0097ed19f38cf2d19fb',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'src/jaxpropka').glob('*.py')})
    def checkpoint(stage):
        result.update(stage=stage,elapsed_seconds=time.perf_counter()-started)
        write_json(destination/f'{name}.json',result)
        print(args.case,name,stage,round(result['elapsed_seconds'],2),flush=True)
    cache = (synthetic_cache(n=args.synthetic_n,neighbors=8) if args.synthetic_n
             else StructureCache.load(destination/'bound.npz'))
    result.update(n_residues=cache.n_residues,cache_bytes=cache.nbytes,
                  active_edges=int(np.count_nonzero(cache.pair_mask)),synthetic=bool(args.synthetic_n))
    ph = jnp.linspace(6.,7.4,args.points,dtype=jnp.float32)
    if args.workload == 'complex':
        arrays,native,_ = pack_inputs(cache,one_hot(cache.native_index),bucket_multiple=(1,1,1))
        edges = pack_interaction_edges(arrays['pair_mask'],arrays['neighbors'])
        if args.variant in ('packed_v2','implicit'):
            arrays,edges = pack_runtime(arrays),None
        kernel = baseline_module().packed_curve_kernel if args.variant == 'baseline' else packed_curve_kernel
        def objective(d,z,h,e):
            p = jax.nn.softmax(z,-1)
            if args.variant in ('packed_v2','implicit'):
                out = packed_v2_curve_kernel(d,p,h,config=cfg,differentiation=diff)
            elif args.variant == 'baseline':
                out = kernel(d,p,h,e,config=cfg)
            else:
                out = kernel(d,p,h,e,config=cfg,differentiation=diff)
            return jnp.trapezoid(out.total_charge,h),(out.residual,out.total_charge)
        fn = jax.jit(jax.value_and_grad(objective,argnums=1,has_aux=True))
        arrays,edges = jax.device_put((arrays,edges))
        make_args = lambda z:(arrays,z,ph,edges)
    else:
        binder = StructureCache.load(destination/'binder.npz')
        target = StructureCache.load(destination/'target.npz')
        native = one_hot(binder.native_index)
        if args.variant == 'baseline':
            baseline = baseline_module()
            original_models=[baseline.TitrationModel(c,cfg,backend='packed') for c in (cache,binder,target)]
            bound_read,free_read,target_read=[m.curves(np.asarray(ph)) for m in original_models]
            pa=original_models[0].native_probabilities
            target_out = target_read(original_models[2].native_probabilities)
            target_integral = jnp.trapezoid(target_out.total_charge,ph)
            indices = jnp.asarray(cache.select(binder.keys))
            def objective(z):
                p = jax.nn.softmax(z,-1)
                a = bound_read(pa.at[indices].set(p))
                b = free_read(p)
                s = jnp.trapezoid(a.total_charge-b.total_charge,ph)-target_integral
                return .1*jax.nn.softplus((1.-s)/.1),(s,jnp.stack((a.residual,b.residual,target_out.residual)))
            fn = jax.jit(jax.value_and_grad(objective,has_aux=True))
            make_args = lambda z:(z,)
        else:
            backend = 'packed' if args.variant == 'checkpointed' else 'packed_v2'
            models = [TitrationModel(c,cfg,backend=backend,differentiation=diff) for c in (cache,binder,target)]
            obj = SelectivityObjective(*models,np.asarray(ph),ph_chunk_size=args.chunk_size)
            if args.variant == 'endpoint':
                from jaxpropka.experimental import EndpointSelectivityObjective
                obj = EndpointSelectivityObjective(obj)
            def objective(z):
                p = jax.nn.softmax(z,-1)
                out = obj.value_and_grad(p)
                gradient = p*(out.gradient-jnp.sum(p*out.gradient,axis=-1,keepdims=True))
                return out,gradient
            fn = jax.jit(objective)
            make_args = lambda z:(z,)
    checkpoint('ready')
    executable = None
    for mixture in (.05,.001):
        z = jnp.log((1-mixture)*jnp.asarray(native)+mixture/20)
        arguments = make_args(z)
        jax.block_until_ready(arguments)
        if executable is None:
            t = time.perf_counter();executable = fn.lower(*arguments).compile()
            result['compile_seconds'] = time.perf_counter()-t
            memory = executable.memory_analysis()
            result['memory'] = {k:getattr(memory,k) for k in
                ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
        jax.block_until_ready(executable(*arguments))
        times = []
        for _ in range(args.repeats):
            t = time.perf_counter();out = jax.block_until_ready(executable(*arguments))
            times.append(time.perf_counter()-t)
        out = jax.device_get(out)
        if args.workload == 'complex':
            (value,(residual,charge)),gradient = out
            s = value
            valid = np.all(residual<min(cfg.residual_tolerance,diff.implicit_residual_tolerance)
                           if args.variant=='implicit' else residual<cfg.residual_tolerance)
            extras = dict(charge=charge,residual=residual)
        elif args.variant == 'baseline':
            (value,(s,residual)),gradient = out
            valid = np.all(residual<cfg.residual_tolerance)
            extras = dict(residual=residual)
        else:
            evaluation,gradient = out
            value,s,valid = evaluation.loss,evaluation.selectivity,evaluation.valid
            residual = evaluation.diagnostics.forward_residual
            extras = evaluation.diagnostics._asdict()
        valid = bool(valid and np.isfinite(gradient).all())
        np.savez_compressed(destination/f'{name}-mix{mixture}.npz',gradient=gradient,
                            value=value,selectivity=s,**extras)
        result['records'].append(dict(mixture=mixture,value=float(value),selectivity=float(s),
            valid=valid,max_residual=float(np.max(residual)),gradient_finite=bool(np.isfinite(gradient).all()),
            warm_seconds=times,warm_median_seconds=float(np.median(times))))
        result['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        checkpoint(f'mixture_{mixture}')
    result['status'] = 'complete';checkpoint('complete')


def regression(args):
    """Repeat existing sensitive protein-protein schedules against frozen code."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaxpropka import ModelConfig,DifferentiationConfig
    from jaxpropka.cache import StructureCache
    from jaxpropka.batching import pack_inputs
    from jaxpropka.model import packed_curve_kernel,packed_v2_curve_kernel,pack_runtime,pack_interaction_edges,one_hot
    jax.config.update('jax_enable_compilation_cache',False)
    destination = args.output_dir/f'case-{args.case}'
    cache = StructureCache.load(destination/'bound.npz')
    arrays,p,_ = pack_inputs(cache,one_hot(cache.native_index),bucket_multiple=(1,1,1))
    edges = pack_interaction_edges(arrays['pair_mask'],arrays['neighbors'])
    runtime = pack_runtime(arrays)
    arrays,p,edges,runtime = jax.device_put((arrays,p,edges,runtime))
    accepted = json.loads((RUNTIME/'packed-panel/interface-full'/f'report-{args.case}.json').read_text())
    original = json.loads((RUNTIME/'interfaces/724720'/f'report-{args.case}.json').read_text())['cases'][0]
    schedules = [c for c in accepted['comparisons'] if not c.get('passed',True)]
    if not schedules:
        schedules = [dict(steps=128,points=15,grid='design')]
    baseline = baseline_module()
    result = dict(case=args.case,interface_id=original['interface_id'],
                  prior_dense_packed_status=accepted['status'],comparisons=[])
    for schedule in schedules:
        cfg = ModelConfig(**{**original['midpoint_model_config'],'steps':schedule['steps']})
        lo,hi = (6.,7.4) if schedule['grid']=='design' else (cfg.ph_min,cfg.ph_max)
        grid = jnp.linspace(lo,hi,schedule['points'],dtype=p.dtype)
        reference = jax.device_get(baseline.packed_curve_kernel(arrays,p,grid,edges,config=cfg))
        for variant in ('checkpointed','packed_v2'):
            diff = DifferentiationConfig(mode='checkpointed')
            out = jax.device_get(packed_curve_kernel(arrays,p,grid,edges,config=cfg,differentiation=diff)
                if variant=='checkpointed' else packed_v2_curve_kernel(runtime,p,grid,config=cfg,differentiation=diff))
            flags = bool(np.array_equal(out.converged,reference.converged))
            record = dict(variant=variant,schedule=schedule,
                occupancy_max_abs=float(np.max(np.abs(out.protonated-reference.protonated))),
                charge_max_abs=float(np.max(np.abs(out.total_charge-reference.total_charge))),
                convergence_flags_equal=flags,
                passed=bool(np.allclose(out.protonated,reference.protonated,atol=2e-5,rtol=2e-5)
                    and np.allclose(out.total_charge,reference.total_charge,atol=2e-4,rtol=2e-5) and flags))
            result['comparisons'].append(record)
            write_json(destination/'regression.json',result)
            print(args.case,variant,schedule['grid'],schedule['steps'],record,flush=True)
    result['passed'] = all(c['passed'] for c in result['comparisons'])
    write_json(destination/'regression.json',result)
    if not result['passed']:raise SystemExit(1)


def endpoint_audit(args):
    """Float64 envelope, quadrature-refinement and branch checks on a real pair."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jaxpropka import ModelConfig,DifferentiationConfig,TitrationModel,SelectivityObjective
    from jaxpropka.cache import StructureCache
    from jaxpropka.experimental import EndpointSelectivityObjective
    jax.config.update('jax_enable_x64',True)
    jax.config.update('jax_enable_compilation_cache',False)
    destination=args.output_dir/f'case-{args.case}'
    cfg=ModelConfig(steps=args.steps)
    diff=DifferentiationConfig(mode='implicit',implicit_residual_tolerance=1e-10,
                               linear_rtol=1e-10,linear_atol=1e-12)
    models=[TitrationModel(StructureCache.load(destination/f'{name}.npz'),cfg,
                          backend='packed_v2',differentiation=diff)
            for name in ('bound','binder','target')]
    coarse=SelectivityObjective(*models,np.linspace(6.,7.4,15))
    fine=SelectivityObjective(*models,np.linspace(6.,7.4,129))
    endpoint=EndpointSelectivityObjective(fine)
    result=dict(case=args.case,steps=args.steps,dtype='float64',records=[],
        note='Consistent continuation paths do not certify a global minimum.')
    for mixture in (.05,.001):
        p=(1-mixture)*models[1].native_probabilities.astype(jnp.float64)+mixture/20
        a=coarse.value_and_grad(p);b=fine.value_and_grad(p);c=endpoint.value_and_grad(p)
        branch=endpoint.audit(p)
        direction=jnp.asarray(np.random.default_rng(17).normal(size=p.shape))
        direction/=jnp.linalg.norm(direction)
        z=jnp.log(p);eps=1e-4
        finite=(endpoint.evaluate(jax.nn.softmax(z+eps*direction,-1)).loss-
                endpoint.evaluate(jax.nn.softmax(z-eps*direction,-1)).loss)/(2*eps)
        analytic=jnp.vdot(p*(c.gradient-jnp.sum(p*c.gradient,-1,keepdims=True)),direction)
        data=dict(mixture=mixture,endpoint_valid=bool(c.valid),coarse_valid=bool(a.valid),fine_valid=bool(b.valid),
            branch_consistent=bool(jnp.all(branch.consistent)),
            max_branch_gap=float(jnp.max(branch.max_occupancy_gap)),max_branch_residual=float(jnp.max(branch.max_residual)),
            coarse_selectivity_error=float(jnp.abs(a.selectivity-c.selectivity)),
            fine_selectivity_error=float(jnp.abs(b.selectivity-c.selectivity)),
            coarse_gradient_relative_l2=float(jnp.linalg.norm(a.gradient-c.gradient)/jnp.maximum(jnp.linalg.norm(c.gradient),1e-20)),
            fine_gradient_relative_l2=float(jnp.linalg.norm(b.gradient-c.gradient)/jnp.maximum(jnp.linalg.norm(c.gradient),1e-20)),
            directional_fd=float(finite),directional_analytic=float(analytic),
            directional_pass=bool(jnp.isclose(finite,analytic,atol=1e-6,rtol=1e-4)))
        result['records'].append(data)
        write_json(destination/'endpoint-audit.json',result)
        print(args.case,data,flush=True)
    result['status']='complete'
    result['passed']=all(r['endpoint_valid'] and r['fine_valid'] and r['branch_consistent']
        and r['directional_pass'] and r['fine_selectivity_error']<1e-4
        and r['fine_gradient_relative_l2']<1e-3 for r in result['records'])
    write_json(destination/'endpoint-audit.json',result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','regression','endpoint-audit'])
    parser.add_argument('--case',type=int,required=True)
    parser.add_argument('--output-dir',type=Path,default=RUNTIME/'gradient-memory')
    parser.add_argument('--bound-only',action='store_true')
    parser.add_argument('--variant',choices=['baseline','checkpointed','packed_v2','implicit','endpoint'],default='packed_v2')
    parser.add_argument('--workload',choices=['complex','selectivity'],default='complex')
    parser.add_argument('--steps',type=int,default=128)
    parser.add_argument('--points',type=int,default=15)
    parser.add_argument('--chunk-size',type=int,default=1)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--synthetic-n',type=int)
    args = parser.parse_args()
    if args.variant=='endpoint' and args.workload!='selectivity':parser.error('endpoint requires selectivity workload')
    if args.synthetic_n and args.workload!='complex':parser.error('synthetic fixtures support complex workload only')
    {'prepare':prepare_case,'run':benchmark,'regression':regression,'endpoint-audit':endpoint_audit}[args.action](args)


if __name__ == '__main__':
    main()
