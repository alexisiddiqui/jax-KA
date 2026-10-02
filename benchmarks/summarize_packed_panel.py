"""Summarize dense-versus-packed panel reports with a fail-closed gate."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',required=True,type=Path)
    parser.add_argument('--expected',required=True,type=int)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    rows=[];missing=[];invalid=[]
    for i in range(args.expected):
        path=args.reports/f'report-{i}.json'
        if not path.exists():missing.append(i);continue
        try:rows.append(json.loads(path.read_text()))
        except Exception as error:invalid.append({'index':i,'error':repr(error)})
    evaluated=[x for x in rows if x.get('status')!='excluded_input']
    completed=[x for x in evaluated if x.get('status') in ('passed','failed')]
    failed=[{'pdb_id':x.get('pdb_id'),'chain_id':x.get('chain_id'),
             'status':x.get('status'),'stage':x.get('stage')}
            for x in evaluated if x.get('status')=='failed']
    incomplete=[{'pdb_id':x.get('pdb_id'),'chain_id':x.get('chain_id'),
                 'status':x.get('status'),'stage':x.get('stage')}
                for x in evaluated if x.get('status') not in ('passed','failed')]
    comparisons=[c for x in completed for c in x.get('comparisons',[])]
    summary={'expected':args.expected,'reports':len(rows),'missing_indices':missing,
        'invalid_reports':invalid,'excluded_input':sum(x.get('status')=='excluded_input' for x in rows),
        'evaluated':len(evaluated),'completed':len(completed),
        'passed':sum(x.get('status')=='passed' for x in completed),
        'failed_cases':failed,'incomplete_cases':incomplete,
        'comparison_count':len(comparisons),
        'max_occupancy_abs':max((x['occupancy_max_abs'] for x in comparisons),default=None),
        'max_charge_abs':max((x['charge_max_abs'] for x in comparisons),default=None),
        'max_pka_abs':max((x['pka_max_abs'] for x in comparisons),default=None),
        'changed_convergence_comparisons':sum(not x['convergence_flags_equal'] for x in comparisons),
        'changed_site_category_comparisons':sum(not x['site_categories_equal'] for x in comparisons),
        'changed_tier_cases':sum(not x.get('tier_classifications_equal',False) for x in completed),
        'dense_solve_seconds':sum(x['dense_seconds'] for x in comparisons),
        'packed_solve_seconds':sum(x['packed_seconds'] for x in comparisons)}
    summary['acceptance_passed']=not missing and not invalid and not failed and not incomplete
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if not summary['acceptance_passed']:raise SystemExit(1)


if __name__=='__main__':main()
