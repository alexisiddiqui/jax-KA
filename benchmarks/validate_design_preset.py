"""CPU promotion gates for the opt-in design preset; invalid results are not wins.

Run manifest first, then one forward/gradient/scaling case per fresh process.
All artifacts are separate from the accepted historical gradient reports.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import tarfile
import time
import numpy as np
from benchmark_gradients import ROOT, RUNTIME, baseline_module, write_json

SCALING = (111,120,6,43,171)
SENSITIVE = (52,100,117,212)
DEFAULT = RUNTIME/'design-preset'


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def manifest(output):
    eligible=[];excluded=[];counts=set();provenance=set()
    for path in sorted((RUNTIME/'interfaces/724720').glob('report-*.json')):
        index=int(path.stem.split('-')[-1]);report=json.loads(path.read_text());case=report['cases'][0]
        counts.add(report['dataset']['manifest_case_count'])
        provenance.add((report['dataset']['manifest_sha256'],report['dataset']['archive_sha256']))
        if not case.get('cache_fingerprint'):
            excluded.append(dict(case=index,status=case.get('status'),reason=case.get('reason',case.get('error'))))
            continue
        packed=RUNTIME/'packed-panel/interface-full'/path.name
        old=json.loads(packed.read_text()) if packed.exists() else {}
        edges=old.get('active_pair_entries')
        if edges is None and index==171:
            edges=json.loads((RUNTIME/'gradient-memory/case-171/preparation.json').read_text())['environments']['bound']['edges']
        if edges is None:raise ValueError(f'missing edge count for {index}')
        eligible.append(dict(case=index,n=case['n_residues'],edges=edges,
            fingerprint=case['cache_fingerprint'],prior_status=case['status'],
            source_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    if counts != {len(eligible)+len(excluded)} or len(provenance)!=1:
        raise ValueError('source panel is incomplete or has inconsistent provenance')
    required=set(SCALING+SENSITIVE)
    pool=sorted((r for r in eligible if r['case'] not in required),key=lambda r:(r['edges'],r['case']))
    required.update(pool[i]['case'] for i in np.linspace(0,len(pool)-1,15).round().astype(int))
    assert len(required)==24
    result=dict(eligible=sorted(eligible,key=lambda r:r['case']),excluded=excluded,
                gradient_cases=sorted(required),scaling_cases=list(SCALING),sensitive_cases=list(SENSITIVE))
    write_json(output/'manifest.json',result)
    print(json.dumps(dict(eligible=len(eligible),excluded=len(excluded),gradient_cases=sorted(required))))


def caches(index, output, all_environments):
    from biotite.structure.io import pdbx
    from jaxpropka.geometry import build_candidates
    from jaxpropka.precompute import build_cache
    from jaxpropka.topology import load_topology
    case=json.loads((RUNTIME/f'interfaces/724720/report-{index}.json').read_text())['cases'][0]
    names=('bound','binder','target') if all_environments else ('bound',)
    paths={}
    for name in names:
        old=RUNTIME/f'gradient-memory/case-{index}/{name}.npz'
        paths[name]=old if old.exists() else output/f'caches/case-{index}/{name}.npz'
    missing=[name for name,path in paths.items() if not path.exists()]
    if missing:
        with tarfile.open(ROOT/'ground_truth_1522.tar') as archive:
            member=next(m for m in archive.getmembers() if Path(m.name).name==case['pdb_id']+'.cif')
            raw=archive.extractfile(member).read()
        assert hashlib.sha256(raw).hexdigest()==case['source_sha256']
        atoms=pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(raw.decode())),model=1,
            altloc='occupancy',use_author_fields=False,include_bonds=True)
        chains=case['partner_chains']
        selections=dict(bound=chains,binder=chains[:1],target=chains[1:])
        for name in missing:
            started=time.perf_counter()
            print(f'preparing case {index} {name}',flush=True)
            topology=load_topology(atoms[np.isin(atoms.chain_id,selections[name])],gap_policy='free',
                                  freeze_disulfides=True,ignore_nonprotein=True)
            topology.metadata.update(source_name=case['pdb_id']+'.cif',source_sha256=case['source_sha256'],
                foldbench_pdb_id=case['pdb_id'],foldbench_label_asym_id=case['chain_id'])
            cache=build_cache(topology,build_candidates(topology,missing_sidechain='template'))
            if name=='bound':assert cache.fingerprint()==case['cache_fingerprint']
            paths[name].parent.mkdir(parents=True,exist_ok=True);cache.save(paths[name])
            print(f'prepared case {index} {name}: N={cache.n_residues}, {time.perf_counter()-started:.1f}s',flush=True)
            del cache
    return paths,case


def numerical(args):
    import jax
    import jax.numpy as jnp
    from jaxpropka import TitrationModel,ModelConfig,SelectivityObjective,DifferentiationConfig
    from jaxpropka.cache import StructureCache
    jax.config.update('jax_enable_compilation_cache',False)
    if args.action=='gradient':jax.config.update('jax_enable_x64',True)
    start=time.perf_counter();destination=args.output/f'{args.action}-{args.case}.json'
    report=dict(case=args.case,action=args.action,status='running',records=[],hostname=os.uname().nodename,
        jax=jax.__version__,dtype='float64' if args.action=='gradient' else 'float32',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/jaxpropka').glob('*.py')})
    def save():
        report.update(elapsed_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(destination,report)
    save()
    paths,case=caches(args.case,args.output,args.action!='forward')
    loaded={name:StructureCache.load(path) for name,path in paths.items()}
    assert loaded['bound'].fingerprint()==case['cache_fingerprint']
    report['fingerprints']={name:c.fingerprint() for name,c in loaded.items()}
    strict=(DifferentiationConfig(mode='implicit',implicit_residual_tolerance=1e-10,
                                   linear_rtol=1e-10,linear_atol=1e-12) if args.action=='gradient' else None)
    models={name:TitrationModel.for_design(c,differentiation=strict) for name,c in loaded.items()}
    grid=np.linspace(6.,7.4,15)
    report.update(model_config=asdict(models['bound'].config),
                  differentiation_config=asdict(models['bound'].differentiation),ph_grid=grid.tolist(),
                  reference_steps=2048 if args.action=='gradient' else 1024 if args.action=='forward' else None)
    save()
    if args.action=='forward':
        model=models['bound'];read=model.curves_with_solver_diagnostics(grid)
        original=baseline_module().TitrationModel(loaded['bound'],ModelConfig(steps=1024),backend='packed')
        ref=original.curves(grid)
        for mix in (0.,.05,.001):
            p=(1-mix)*model.native_probabilities+mix/20
            actual,solver=jax.device_get(read(p));expected=jax.device_get(ref(p))
            audit=jax.device_get(model.audit(p,grid))
            numerical_valid=bool(np.all(solver.converged))
            stable=bool(np.all(audit.consistent))
            reference_valid=bool(np.all(expected.residual<1e-6))
            parity=bool(np.allclose(actual.total_charge,expected.total_charge,atol=2e-4,rtol=2e-5)
                and np.allclose(actual.protonated,expected.protonated,atol=2e-5,rtol=2e-5))
            report['records'].append(dict(mixture=mix,numerical_valid=numerical_valid,branch_consistent=stable,
                reference_valid=reference_valid,parity=parity,max_residual=float(np.max(solver.residual)),
                iterations=np.asarray(solver.iterations).tolist(),audit_iterations=audit.iterations.tolist(),
                audit_residual=audit.residual.tolist(),audit_converged=audit.converged.tolist(),
                audit_gap=audit.max_occupancy_gap.tolist(),
                classification=('convergence_regression' if not numerical_valid and reference_valid else
                    'forward_failure' if not numerical_valid else 'branch_disagreement' if not stable
                    else 'reference_failure' if not reference_valid else 'passed' if parity else 'parity_failure')))
            save()
    else:
        ms=[models[name] for name in ('bound','binder','target')]
        # Non-saturated penalty makes gradient gates meaningful at every size.
        obj=SelectivityObjective(*ms,grid,required_log10_ratio=100.,tau=1.)
        if args.action=='gradient':
            refs=[TitrationModel(c,ModelConfig(steps=2048),backend='packed_v2',
                   differentiation=DifferentiationConfig(mode='checkpointed')) for c in loaded.values()]
            reference=SelectivityObjective(*refs,grid,required_log10_ratio=100.,tau=1.)
        def calculate(z):
            p=jax.nn.softmax(z,-1);out,solver=obj.value_and_grad_with_solver_diagnostics(p)
            grad=p*(out.gradient-jnp.sum(p*out.gradient,axis=-1,keepdims=True))
            return out,solver,grad
        calculate=jax.jit(calculate)
        dtype=jnp.float64 if args.action=='gradient' else jnp.float32
        executable=None
        for mix in ((0.,.05,.001) if args.action=='gradient' else (.05,.001)):
            p=(1-mix)*ms[1].native_probabilities.astype(dtype)+mix/20
            if mix==0:
                out,solver=jax.device_get(obj.value_and_grad_with_solver_diagnostics(p));gradient=out.gradient
            else:
                z=jnp.log(p)
                p=jax.nn.softmax(z,-1)
                if executable is None:
                    t=time.perf_counter();executable=calculate.lower(z).compile();report['compile_seconds']=time.perf_counter()-t
                    memory=executable.memory_analysis()
                    report['memory']={name:getattr(memory,name) for name in ('temp_size_in_bytes','argument_size_in_bytes','output_size_in_bytes')}
                jax.block_until_ready(executable(z));times=[]
                for _ in range(5 if args.action=='scaling' else 1):
                    t=time.perf_counter();answer=jax.block_until_ready(executable(z));times.append(time.perf_counter()-t)
                out,solver,gradient=jax.device_get(answer)
            row=dict(mixture=mix,numerical_valid=bool(out.valid),forward_valid=bool(np.all(solver.converged)),
                linear_valid=bool(np.all(out.diagnostics.linear_valid)),gradient_finite=bool(np.isfinite(gradient).all()),
                max_residual=float(np.max(solver.residual)),iterations=solver.iterations.tolist())
            if mix:row['warm_median_seconds']=float(np.median(times))
            if args.action=='gradient':
                audit=jax.device_get(obj.audit(p));stable=bool(np.all(audit.consistent));row['branch_consistent']=stable
                expected=jax.device_get(reference.value_and_grad(p))
                refgrad=(expected.gradient if mix==0 else np.asarray(p)*(expected.gradient-
                    np.sum(np.asarray(p)*expected.gradient,axis=-1,keepdims=True)))
                row['reference_valid']=bool(expected.valid)
                row['value_parity']=bool(np.isclose(out.selectivity,expected.selectivity,atol=2e-4,rtol=2e-5))
                row['gradient_parity']=bool(np.allclose(gradient,refgrad,atol=2e-5,rtol=2e-4))
                row['value_abs_error']=finite_number(abs(out.selectivity-expected.selectivity))
                row['gradient_max_abs_error']=finite_number(np.max(np.abs(gradient-refgrad)))
                row['gradient_relative_l2']=finite_number(np.linalg.norm(gradient-refgrad)/max(np.linalg.norm(refgrad),1e-20))
                row['audit_max_gap']=finite_number(np.max(audit.max_occupancy_gap))
                row['audit_all_converged']=bool(np.all(audit.converged))
                row['audit_max_residual']=finite_number(np.max(audit.residual))
                if mix and out.valid and stable and expected.valid:
                    directions=np.random.default_rng(2026+args.case).normal(size=(3,*p.shape))
                    checks=[]
                    for v in directions:
                        v=jnp.asarray(v/np.linalg.norm(v));eps=1e-4
                        plus=obj.evaluate(jax.nn.softmax(z+eps*v,-1));minus=obj.evaluate(jax.nn.softmax(z-eps*v,-1))
                        fd=float((plus.loss-minus.loss)/(2*eps));analytic=float(jnp.vdot(gradient,v))
                        checks.append(dict(fd=fd,analytic=analytic,passed=bool(plus.valid and minus.valid and
                            np.isclose(fd,analytic,atol=1e-6,rtol=1e-4))))
                    row['directional_checks']=checks
                row['classification']=('forward_failure' if not row['forward_valid'] else 'adjoint_failure' if not row['numerical_valid']
                    else 'branch_disagreement' if not stable else 'reference_failure' if not expected.valid
                    else 'passed' if row['value_parity'] and row['gradient_parity'] and
                         all(c['passed'] for c in row.get('directional_checks',[])) else 'gradient_failure')
            report['records'].append(row);save()
    report['status']='complete';save()
    print(json.dumps(dict(case=args.case,action=args.action,status=report['status'],
        classifications=[r.get('classification',r.get('numerical_valid')) for r in report['records']])))


def summarize(output):
    manifest=json.loads((output/'manifest.json').read_text());missing=[];rows={};bad=[];stale=[]
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/jaxpropka').glob('*.py')}
    fingerprints={r['case']:r['fingerprint'] for r in manifest['eligible']}
    for action,indices in (('forward',[r['case'] for r in manifest['eligible']]),
                           ('gradient',manifest['gradient_cases']),('scaling',manifest['scaling_cases'])):
        rows[action]=[]
        for index in indices:
            path=output/f'{action}-{index}.json'
            if not path.exists():missing.append(f'{action}-{index}');continue
            row=json.loads(path.read_text())
            mixtures=(.05,.001) if action=='scaling' else (0.,.05,.001)
            if (row.get('status')!='complete' or row.get('case')!=index or row.get('action')!=action
                    or sorted(r['mixture'] for r in row.get('records',[]))!=sorted(mixtures)):
                missing.append(f'{action}-{index}');continue
            if row.get('source_sha256')!=source:stale.append(f'{action}-{index}')
            if row.get('fingerprints',{}).get('bound')!=fingerprints[index]:
                bad.append(dict(action=action,case=index,reason='cache_fingerprint_mismatch'))
            rows[action].append(row)
            for record in row['records']:
                # Audit classification must not hide a separate parity regression
                # on the standard, converged trajectory.
                parity_fields=('parity',) if action=='forward' else ('value_parity','gradient_parity')
                if action!='scaling' and record.get('numerical_valid') and record.get('reference_valid'):
                    if not all(record.get(key,False) for key in parity_fields):
                        bad.append(dict(action=action,case=index,mixture=record['mixture'],reason='converged_parity_failure'))
                if action=='gradient' and record.get('classification')=='passed' and record['mixture']:
                    checks=record.get('directional_checks',[])
                    if len(checks)!=3 or not all(c['passed'] for c in checks):
                        bad.append(dict(action=action,case=index,mixture=record['mixture'],reason='missing_directional_checks'))
                if record.get('classification') in ('parity_failure','gradient_failure','reference_failure','adjoint_failure','convergence_regression'):
                    bad.append(dict(action=action,case=index,mixture=record['mixture'],reason=record['classification']))
    largest=next((r for r in rows['scaling'] if r['case']==43),None)
    baseline=json.loads((RUNTIME/'gradient-memory/case-43/selectivity-baseline-t128-h15.json').read_text())
    reduction=baseline['memory']['temp_size_in_bytes']/largest['memory']['temp_size_in_bytes'] if largest else None
    scaling_valid=all(r['numerical_valid'] for row in rows['scaling'] for r in row['records'])
    qualified=[row['case'] for row in rows['gradient'] if all(
        r.get('classification')=='passed' for r in row['records'] if r['mixture'])]
    scaling_gradients=set(manifest['scaling_cases'])<=set(qualified)
    result=dict(complete=not missing,missing=missing,stale_source=stale,regressions=bad,temporary_reduction=reduction,
        qualified_gradient_cases=qualified,scaling_gradients_passed=scaling_gradients,
        numerical_gates_passed=not missing and not stale and not bad and scaling_valid and scaling_gradients and reduction is not None and reduction>=5,
        note='Recommendation additionally requires a passing current CPU test suite and review of all classified failures.',
        excluded=manifest['excluded'],results=rows)
    write_json(output/'summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','excluded')},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('manifest','forward','gradient','scaling','summarize'))
    parser.add_argument('--case',type=int)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    if args.action=='manifest':manifest(args.output)
    elif args.action=='summarize':summarize(args.output)
    else:
        if args.case is None:parser.error('--case is required')
        numerical(args)


if __name__=='__main__':main()
