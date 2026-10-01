"""Merge validated interface shards and plot association-burial regression."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
from regress_foldbench import sha256,_write_checkpoint
from plot_foldbench_regression import metrics,shared_edges,js_divergence,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--expected',type=int,required=True)
    args=parser.parse_args()
    manifest=list(csv.DictReader(args.manifest.open()));merged=None
    for i in range(args.expected):
        path=args.reports/f'report-{i}.json'
        report=json.loads(path.read_text());dataset=report['dataset'];case=report['cases'][0]
        assert dataset['case_index']==i and dataset['manifest_sha256']==sha256(args.manifest)
        expected=manifest[i]
        assert case['pdb_id']==expected['pdb_id']
        assert case['partner_chains']==[expected['interface_chain_id_1'],expected['interface_chain_id_2']]
        if merged is None:
            merged={**report,'cases':[],'dataset':{**dataset,'case_count':args.expected,'case_index':None},
                    'execution':{'kind':'per-interface array'},'timing_seconds':{}}
        assert report['policy']==merged['policy'] and report['reference']==merged['reference']
        assert dataset['archive_sha256']==merged['dataset']['archive_sha256']
        merged['cases'].append(case)
    _write_checkpoint(args.reports/'summary.json',merged)
    rows=[]
    for case in merged['cases']:
        for s in case.get('sites',[]):
            if s.get('delta') is None:continue
            rows.append({'protein':case['interface_id'],'group':s['group'],
                'residue':str(s['residue']),'tier':s['tier'],'reference_pka':s['reference_pka'],
                'surrogate_pka':s['surrogate_pka'],'error':s['delta'],
                'residue_delta_sasa':s.get('residue_delta_sasa'),
                'functional_delta_sasa':s.get('functional_delta_sasa'),
                'interface':s.get('interface_delta_sasa_ge1'),
                'partner_distance':s.get('nearest_partner_heavy_atom_distance'),
                'structure_branch_status':s.get('structure_branch_status','not_assessed')})
    out=args.reports/'plots';out.mkdir(exist_ok=True)
    if not rows:raise ValueError('no eligible paired sites; summary saved, no plots possible')
    write_csv(out/'paired_sites.csv',rows)
    annotated=[r for r in rows if r['interface'] is not None]
    cohorts={'Overall':annotated,'Strict':[r for r in annotated if r['tier']=='strict']}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({'font.size':10,'savefig.dpi':180,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(16,9));stats={};jsrows=[];groupstats=[]
    allvalues=np.array([[r['reference_pka'],r['surrogate_pka']] for r in rows])
    edges=shared_edges(allvalues,.5)
    for row,(name,rs) in enumerate(cohorts.items()):
        if not rs:continue
        interface=[r for r in rs if r['interface']];non=[r for r in rs if not r['interface']]
        stats[name]={'all_annotated':metrics(rs),'interface':metrics(interface) if interface else None,
                     'non_interface':metrics(non) if non else None}
        ax=axs[row,0]
        if interface:
            x=np.array([r['reference_pka'] for r in interface]);y=np.array([r['surrogate_pka'] for r in interface])
            ax.hexbin(x,y,gridsize=50,mincnt=1,norm=LogNorm(),cmap='viridis')
            bounds=[min(x.min(),y.min())-1,max(x.max(),y.max())+1];ax.plot(bounds,bounds,'--',color='gray')
            s=stats[name]['interface'];r=s['pearson_r'];rho=s['spearman_rho']
            if r is not None:ax.text(.03,.97,f"n={len(interface):,}\nr={r:.3f}; ρ={rho:.3f}\nMAE={s['mae']:.3f}",va='top',transform=ax.transAxes,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        ax.set(xlabel='PROPKA pKa',ylabel='JAX midpoint pKa',title=f'{name}: interface sites')
        x=np.array([r['residue_delta_sasa'] for r in rs]);y=np.array([abs(r['error']) for r in rs])
        ax=axs[row,1];hb=ax.hexbin(np.log1p(np.maximum(x,0)),y,gridsize=55,mincnt=1,norm=LogNorm(),cmap='viridis')
        fig.colorbar(hb,ax=ax,label='Paired sites / hexagon',fraction=.046,pad=.04)
        ticks=np.array([0,1,5,20,50,100,200]);ticks=ticks[ticks<=max(x.max(),1)]
        ax.set(xticks=np.log1p(ticks),xticklabels=ticks,xlabel='Residue ΔSASA (Å²; log1p spacing)',ylabel='Absolute pKa error',title=f'{name}: association burial')
        if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0:
            r=float(pearsonr(x,y).statistic);rho=float(spearmanr(x,y).statistic)
            stats[name]['delta_sasa_vs_abs_error']={'pearson_r':r,'spearman_rho':rho}
            ax.text(.98,.97,f'r={r:.3f}; ρ={rho:.3f}',ha='right',va='top',transform=ax.transAxes,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
        ax=axs[row,2]
        for k,(label,rr) in enumerate([('Non-interface',non),('Interface',interface)]):
            if rr:
                mean=metrics(rr)['mae'];ax.bar(k,mean,color='#2563a6' if k==0 else '#c46a14')
                ax.text(k,mean,f'n={len(rr):,}',va='bottom',ha='center')
        ax.set(xticks=[0,1],xticklabels=['Non-interface','Interface'],ylabel='Mean absolute pKa error',title=f'{name}: site strata');ax.margins(y=.2)
        for subset,rr in [('all',rs),('interface',interface)]:
            for pair in sorted({r['protein'] for r in rr}):
                selected=[r for r in rr if r['protein']==pair]
                jsrows.append({'cohort':name,'subset':subset,'interface_id':pair,'paired_sites':len(selected),
                    'jsd_bits':js_divergence([r['reference_pka'] for r in selected],[r['surrogate_pka'] for r in selected],edges)})
            for group in sorted({r['group'] for r in rr}):
                groupstats.append({'cohort':name,'subset':subset,'group':group,**metrics([r for r in rr if r['group']==group])})
    fig.suptitle(f'FoldBench protein–protein · {args.expected} target pairs',fontsize=16)
    fig.text(.5,.015,'Interface: residue ΔSASA ≥ 1 Å². Isolated partners minus selected two-chain complex, fixed coordinates; other chains/nonprotein excluded.\nStrict = sampled numerical tier, not branch uniqueness. Associations are descriptive; multiple target pairs may share proteins.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95))
    for suffix in ['png','pdf']:fig.savefig(out/f'interface_regression.{suffix}')
    plt.close(fig)
    if jsrows:write_csv(out/'per_pair_jsd.csv',jsrows)
    if groupstats:write_csv(out/'per_group_metrics.csv',groupstats)
    annotations_failed=[c['interface_id'] for c in merged['cases'] if c.get('interface_annotation_error')]
    result={'target_pairs':args.expected,'paired_sites':len(rows),'annotated_paired_sites':len(annotated),
        'annotation_failures':annotations_failed,'cohorts':stats,'jsd_bin_edges':edges.tolist(),
        'summary_sha256':sha256(args.reports/'summary.json')}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if annotations_failed:raise SystemExit('interface annotations failed; inspect saved report before scaling')


if __name__=='__main__':main()
