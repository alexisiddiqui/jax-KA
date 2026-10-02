"""Compare fresh-process gradient reports; never count invalid solves as wins."""
import argparse
import json
from pathlib import Path
import numpy as np


def selectivity_gradient(saved, workload):
    """Undo the known scalar loss derivative before testing gradient accuracy.

    A saturated softplus can make absolute loss-gradient tolerances vacuous.
    Compute the comparison in float64, and reject underflowed loss slopes rather
    than claiming a zero gradient is accurate. Timed workloads are unchanged.
    """
    gradient = np.asarray(saved['gradient'], dtype=np.float64)
    if workload == 'complex':
        return gradient
    exponent = (float(saved['selectivity']) - 1.) / .1
    slope = np.exp(-np.logaddexp(0., exponent))
    if slope < np.finfo(np.float32).tiny:
        return np.full_like(gradient, np.nan)
    return -gradient / slope


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def summarize(directory):
    results=[];regressions=[];incomplete=[];audits=[]
    for case in sorted(directory.glob('case-*')):
        for path in sorted(case.glob('*-t*-h*.json')):
            row=json.loads(path.read_text())
            if row.get('status')!='complete':incomplete.append(str(path));continue
            if row['variant']=='baseline':continue
            reference=case/f"{row['workload']}-baseline-t{row['steps']}-h{row['points']}.json"
            if not reference.exists():incomplete.append(str(reference));continue
            baseline=json.loads(reference.read_text())
            if baseline.get('status')!='complete':incomplete.append(str(reference));continue
            comparisons=[]
            for record in row['records']:
                mix=record['mixture'];base_record=next(x for x in baseline['records'] if x['mixture']==mix)
                actual=np.load(path.with_name(path.stem+f'-mix{mix}.npz'))
                expected=np.load(reference.with_name(reference.stem+f'-mix{mix}.npz'))
                loss_gradient_match=bool(np.allclose(actual['gradient'],expected['gradient'],atol=2e-5,rtol=2e-4))
                ga,gb=selectivity_gradient(actual,row['workload']),selectivity_gradient(expected,row['workload'])
                gradient_match=bool(np.allclose(ga,gb,atol=2e-5,rtol=2e-4))
                value_match=bool(np.allclose(actual['selectivity'],expected['selectivity'],atol=2e-4,rtol=2e-5))
                residual_a=actual['residual'] if 'residual' in actual else actual['forward_residual']
                residual_b=expected['residual']
                flags_match=(bool(np.array_equal(residual_a<2e-5,residual_b<2e-5))
                             if residual_a.shape==residual_b.shape else None)
                comparisons.append(dict(mixture=mix,valid=record['valid'],baseline_valid=base_record['valid'],
                    value_abs_error=finite_number(abs(actual['selectivity']-expected['selectivity'])),
                    gradient_max_abs_error=finite_number(np.max(np.abs(ga-gb))),
                    gradient_relative_l2=finite_number(np.linalg.norm(ga-gb)/max(np.linalg.norm(gb),1e-20)),
                    loss_gradient_parity_pass=loss_gradient_match,
                    convergence_flags_equal=flags_match,
                    parity_pass=gradient_match and loss_gradient_match and value_match and flags_match is True,
                    warm_seconds=record['warm_median_seconds'],baseline_warm_seconds=base_record['warm_median_seconds']))
            results.append(dict(case=row['case'],workload=row['workload'],variant=row['variant'],
                n_residues=row['n_residues'],steps=row['steps'],points=row['points'],
                temporary_bytes=row['memory']['temp_size_in_bytes'],
                baseline_temporary_bytes=baseline['memory']['temp_size_in_bytes'],
                temporary_reduction=baseline['memory']['temp_size_in_bytes']/max(row['memory']['temp_size_in_bytes'],1),
                peak_rss_kib=row['peak_rss_kib'],baseline_peak_rss_kib=baseline['peak_rss_kib'],
                comparisons=comparisons))
        if (case/'regression.json').exists():regressions.append(json.loads((case/'regression.json').read_text()))
        if (case/'endpoint-audit.json').exists():audits.append(json.loads((case/'endpoint-audit.json').read_text()))
    required={(c,w,v,128,15) for c in (111,120,6,43) for w in ('complex','selectivity')
              for v in ('checkpointed','packed_v2','implicit')}
    present={(r['case'],r['workload'],r['variant'],r['steps'],r['points']) for r in results}
    missing=sorted(required-present)
    exact=[r for r in results if r['variant'] in ('checkpointed','packed_v2')]
    largest=[r for r in exact if r['variant']=='packed_v2' and r['case']==43 and r['steps']==128 and r['points']==15]
    required_regressions={52,100,117,212}
    regression_complete=required_regressions <= {r['case'] for r in regressions}
    parity=bool(exact) and all(c['parity_pass'] for r in exact for c in r['comparisons'])
    finite_valid=parity and all(c['valid'] and c['baseline_valid'] for r in exact for c in r['comparisons'])
    memory=bool(largest) and all(r['temporary_reduction']>=5 for r in largest)
    implicit=[r for r in results if r['variant']=='implicit']
    implicit_valid=bool(implicit) and all(c['valid'] and c['baseline_valid'] and c['parity_pass'] for r in implicit for c in r['comparisons'])
    complete=not missing and not incomplete and regression_complete
    primary=[r for r in results if (r['case'],r['workload'],r['variant'],r['steps'],r['points']) in required]
    primary_pass=(not missing and regression_complete and memory
        and all(r.get('passed',False) for r in regressions)
        and all(c['valid'] and c['baseline_valid'] and c['parity_pass'] for r in primary for c in r['comparisons']))
    return dict(results=results,regressions=regressions,endpoint_audits=audits,missing_required=missing,
        incomplete_files=sorted(set(incomplete)),regression_complete=regression_complete,
        finite_iteration_parity_passed=parity,large_memory_target_passed=memory,
        finite_iteration_validation_passed=finite_valid,
        implicit_validation_passed=implicit_valid,
        required_panel_acceptance_passed=primary_pass,
        complete=complete,acceptance_passed=complete and finite_valid and memory and implicit_valid and all(r.get('passed',False) for r in regressions),
        note='Gradient errors compare d(selectivity)/d(logits), undoing the softplus scalar derivative in float64. Implicit/endpoint validity and accuracy are reported separately from finite-iteration parity. Peak RSS includes host cache, preparation and compiler allocations.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=summarize(args.directory)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('results','regressions','endpoint_audits')},indent=2))


if __name__=='__main__':main()
