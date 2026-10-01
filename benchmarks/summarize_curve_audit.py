"""Collect curve-audit evidence without treating convergence as uniqueness."""
import argparse
import json
from pathlib import Path


def summarize(case):
    flags=[]
    native=case['native_full_grid_512']
    if not native['finite'] or native['occupancy_bounds'][0]<0 or native['occupancy_bounds'][1]>1:
        flags.append('invalid_occupancies')
    if native['max_residual']>=2e-5:
        flags.append('native_512_nonconvergence')
    sweep=case['targeted_continuation']
    seed_differences=case['native_seed_or_setting_max_weighted_differences']
    reference=next(r for r in case['native_targeted_solves'] if
                   (r['steps'],r['damping'],r['seed'])==(4096,.15,'default'))
    converged_seed_difference=0.
    if reference['max_residual']<2e-5:
        converged_seed_difference=max(seed_differences[f"{r['steps']}/{r['damping']}/{r['seed']}"]
            for r in case['native_targeted_solves'] if r['max_residual']<2e-5)
    if max(sweep['max_forward_residual'],sweep['max_reverse_residual'])>=2e-5:
        flags.append('unresolved_continuation')
    if converged_seed_difference>1e-3 or (max(sweep['max_forward_residual'],sweep['max_reverse_residual'])<2e-5 and
            (sweep['max_weighted_sweep_difference']>1e-3 or
             reference['max_residual']<2e-5 and sweep['max_weighted_difference_from_independent']>1e-3)):
        flags.append('different_converged_branches')
    gradients=case['gradients']
    if any(not g['finite_gradient'] for g in gradients):
        flags.append('nonfinite_gradient')
    if any(g['max_residual']>=2e-5 for g in gradients):
        flags.append('soft_sequence_nonconvergence')
    if any(f['absolute_error']>1e-7+1e-3*abs(g['directional_autodiff']) for g in gradients for f in g['finite_differences']):
        flags.append('finite_difference_mismatch')
    if any(g['jvp_vjp_error']>1e-7 for g in gradients):
        flags.append('forward_reverse_autodiff_mismatch')
    if any(g['relative_l2_difference']>1e-3 for g in case['gradient_step_stability']):
        flags.append('iteration_sensitive_gradient')
    if any(g.get('float32_gradient_relative_difference',0)>1e-2 for g in gradients):
        flags.append('precision_sensitive_gradient')
    return {'pdb_id':case['pdb_id'],'flags':flags,
            'max_sweep_occupancy_difference':sweep['max_weighted_sweep_difference'],
            'max_converged_seed_or_setting_difference':converged_seed_difference,
            'sweep_max_residual':max(sweep['max_forward_residual'],sweep['max_reverse_residual']),
            'max_gradient_step_relative_difference':max(g['relative_l2_difference'] for g in case['gradient_step_stability']),
            'max_finite_difference_absolute_error':max(f['absolute_error'] for g in gradients for f in g['finite_differences']),
            'max_float32_gradient_relative_difference':max(g.get('float32_gradient_relative_difference',0) for g in gradients),
            'largest_sweep_difference_site':sweep.get('largest_difference_site')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--expected-cases',type=int,required=True)
    parser.add_argument('--followup',type=Path)
    parser.add_argument('--replacement',type=Path,help='retry report replacing the same case index; originals remain untouched')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paths=sorted(args.reports.glob('case-*.json'))
    assert len(paths)==args.expected_cases,'missing audit cases'
    if args.replacement:
        replacement=json.loads(args.replacement.read_text())
        matched=[p for p in paths if json.loads(p.read_text())['case_index']==replacement['case_index']]
        assert len(matched)==1,'replacement must match exactly one original case'
        paths=[args.replacement if p==matched[0] else p for p in paths]
    if args.followup:paths.append(args.followup)
    rows=[]
    for path in paths:
        case=json.loads(path.read_text())
        assert case['status']=='complete',f'incomplete audit: {path}'
        assert case['diagnostic_update_parity']<1e-10
        rows.append({**summarize(case),'report':str(path.resolve()),
                     'mixtures':[s['uniform_mixture'] for s in case['gradient_step_stability']]})
    report={'scope':'targeted audit, not certification across all sequences or interfaces',
            'core_model_changed':False,'midpoints_used_as_acceptance_criterion':False,
            'cases':rows,'branch_policy_required':any('different_converged_branches' in r['flags'] for r in rows)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
