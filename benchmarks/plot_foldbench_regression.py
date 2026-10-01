"""Plot immutable historical paired pKas; no model evaluation or reference rerun."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon


def shared_edges(values,width):
    if width<=0 or not np.isfinite(width):
        raise ValueError('bin width must be finite and positive')
    lo=np.floor(np.min(values)/width)*width
    hi=np.ceil(np.max(values)/width)*width+width
    return np.arange(lo,hi+width*.5,width)


def js_divergence(x,y,edges):
    if len(x)!=len(y) or not len(x):
        raise ValueError('JSD requires nonempty paired sites')
    a=np.histogram(x,edges)[0];b=np.histogram(y,edges)[0]
    if a.sum()!=len(x) or b.sum()!=len(y):
        raise ValueError('histogram edges omit observations')
    return float(jensenshannon(a,b,base=2)**2)


def metrics(rows):
    x=np.array([r['reference_pka'] for r in rows]);y=np.array([r['surrogate_pka'] for r in rows])
    valid=len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0
    delta=y-x
    return {'sites':len(rows),'proteins':len({r['protein'] for r in rows}),
            'pearson_r':float(pearsonr(x,y).statistic) if valid else None,
            'spearman_rho':float(spearmanr(x,y).statistic) if valid else None,
            'mae':float(np.mean(abs(delta))),'rmse':float(np.sqrt(np.mean(delta**2))),
            'bias':float(np.mean(delta)),'max_absolute_error':float(np.max(abs(delta)))}


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--title',default='Historical FoldBench 723644')
    parser.add_argument('--caption',default='Historical adapter: before terminal-mapping fixes.')
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    source=args.summary.read_bytes();report=json.loads(source)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for case in report['cases']:
        for site in case.get('sites',[]):
            if site.get('delta') is None:continue
            x,y=site.get('reference_pka'),site.get('surrogate_pka')
            if x is None or y is None or not np.isfinite([x,y]).all():continue
            assert np.isclose(y-x,site['delta'],atol=1e-7)
            residue=site['residue']
            rows.append({'protein':case['pdb_id']+':'+case['chain_id'],
                'pdb_id':case['pdb_id'],'chain':residue['chain'],'number':residue['number'],
                'insertion':residue['insertion'],'group':site['group'],'tier':site['tier'],
                'reference_pka':x,'surrogate_pka':y,'error':y-x,
                'functional_sasa':site.get('functional_sasa'),
                'structure_branch_status':site.get('structure_branch_status','not_assessed')})
    assert len(rows)==report['summary']['compared_sites']
    cohorts={'Overall':rows,'Strict':[r for r in rows if r['tier']=='strict']}
    pooled=np.array([[r['reference_pka'],r['surrogate_pka']] for r in rows])
    edges_by_width={width:shared_edges(pooled,width) for width in [.25,.5,1.]}
    stats={name:metrics(rs) for name,rs in cohorts.items()}
    js_rows=[];group_rows=[]
    for name,rs in cohorts.items():
        for protein in sorted({r['protein'] for r in rs}):
            rr=[r for r in rs if r['protein']==protein]
            x=[r['reference_pka'] for r in rr];y=[r['surrogate_pka'] for r in rr]
            for width,edges in edges_by_width.items():
                js_rows.append({'cohort':name,'protein':protein,'paired_sites':len(rr),
                                'bin_width':width,'jsd_bits':js_divergence(x,y,edges)})
        for group in sorted({r['group'] for r in rs}):
            group_rows.append({'cohort':name,'group':group,**metrics([r for r in rs if r['group']==group])})
    write_csv(args.output/'paired_sites.csv',rows)
    write_csv(args.output/'per_protein_jsd.csv',js_rows)
    write_csv(args.output/'per_group_metrics.csv',group_rows)
    colors={'Overall':'#2563a6','Strict':'#c46a14'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    limits=[np.floor(pooled.min())-1,np.ceil(pooled.max())+1]
    error_limit=np.ceil(max(abs(r['error']) for r in rows))
    fig,axs=plt.subplots(2,3,figsize=(16,9))
    for k,(name,rs) in enumerate(cohorts.items()):
        s=stats[name];x=np.array([r['reference_pka'] for r in rs]);y=np.array([r['surrogate_pka'] for r in rs]);delta=y-x
        ax=axs[k,0]
        hb=ax.hexbin(x,y,gridsize=75,mincnt=1,norm=LogNorm(),cmap='viridis',extent=limits*2)
        fig.colorbar(hb,ax=ax,label='Paired sites / hexagon',fraction=.046,pad=.04)
        ax.plot(limits,limits,color='gray',ls='--',lw=1)
        ax.set(xlim=limits,ylim=limits,xlabel='PROPKA pKa',ylabel='JAX midpoint pKa',title=f'{name}: paired predictions')
        annotation=(f"n = {s['sites']:,} sites; {s['proteins']} proteins\n"
                    f"Pearson r = {s['pearson_r']:.3f}\nSpearman ρ = {s['spearman_rho']:.3f}\n"
                    f"MAE = {s['mae']:.3f}; RMSE = {s['rmse']:.3f}")
        ax.text(.03,.97,annotation,transform=ax.transAxes,va='top',fontsize=9,
                bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
        ax=axs[k,1]
        ax.hexbin((x+y)/2,delta,gridsize=65,mincnt=1,norm=LogNorm(),cmap='viridis',extent=limits+[-error_limit,error_limit])
        ax.axhline(0,color='gray',ls='--');ax.axhline(s['bias'],color=colors[name],lw=1)
        ax.set(xlim=limits,ylim=(-error_limit,error_limit),xlabel='Mean of paired pKas',ylabel='JAX − PROPKA (pKa units)',title=f'{name}: paired errors')
        ax=axs[k,2]
        ax.hist(delta,bins=np.linspace(-error_limit,error_limit,101),density=True,color=colors[name],alpha=.8)
        ax.axvline(0,color='gray',ls='--')
        ax.set(xlim=(-error_limit,error_limit),xlabel='JAX − PROPKA (pKa units)',ylabel='Site density',title=f'{name}: error distribution')
        ax.text(.97,.95,f"Bias = {s['bias']:.3f}\nMax |error| = {s['max_absolute_error']:.3f}",transform=ax.transAxes,ha='right',va='top')
    fig.suptitle(args.title+' · paired pKa regression against PROPKA',fontsize=16)
    fig.text(.5,.012,args.caption+' Overall = all eligible pairs; Strict = sampled strict tier, not proof of branch uniqueness.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    for suffix in ['png','pdf']:fig.savefig(args.output/f'paired_regression.{suffix}')
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(16,4.8));js_summary={}
    for name in cohorts:
        selected=[r for r in js_rows if r['cohort']==name and r['bin_width']==.5]
        values=np.array([r['jsd_bits'] for r in selected]);ordered=np.sort(values)
        js_summary[name]={'proteins':len(values),'median':float(np.median(values)),'mean':float(np.mean(values)),
                          'p90':float(np.quantile(values,.9)),'min_paired_sites':min(r['paired_sites'] for r in selected)}
        label=f'{name}: n={len(values)}, median={np.median(values):.3f}'
        axs[0].hist(values,bins=np.linspace(0,1,31),histtype='step',linewidth=2,color=colors[name],label=label)
        axs[1].step(ordered,np.arange(1,len(values)+1)/len(values),where='post',color=colors[name],label=name)
        for width in edges_by_width:
            v=[r['jsd_bits'] for r in js_rows if r['cohort']==name and r['bin_width']==width]
            axs[2].plot(width,np.median(v),'o',color=colors[name])
        axs[2].plot(list(edges_by_width),[np.median([r['jsd_bits'] for r in js_rows if r['cohort']==name and r['bin_width']==width]) for width in edges_by_width],color=colors[name],label=name)
    axs[0].set(xlim=(0,1),xlabel='Per-protein JSD (bits)',ylabel='Number of proteins',title='pKa-distribution divergence · 0.5-unit bins')
    axs[1].set(xlim=(0,1),ylim=(0,1.02),xlabel='Per-protein JSD (bits)',ylabel='Fraction of proteins ≤ JSD',title='Cumulative distribution')
    axs[2].set(xlabel='Shared pKa bin width',ylabel='Median per-protein JSD (bits)',title='Bin-width sensitivity',xticks=list(edges_by_width))
    for ax in axs:ax.legend(fontsize=8)
    fig.suptitle(args.title+' · overall vs strict pKa distributions',fontsize=15)
    fig.text(.5,.015,'Base-2 JSD divergence (not distance); 0 = identical histograms, 1 = disjoint. Same paired sites within each protein; each protein weighted equally.\nCommon bins across both methods and cohorts; no pseudocounts. Distribution agreement does not establish residue-wise agreement.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.1,1,.93))
    for suffix in ['png','pdf']:fig.savefig(args.output/f'per_protein_jsd.{suffix}')
    plt.close(fig)
    burial=plot_burial(cohorts,args.output,args.title,plt)
    metadata={'source':str(args.summary.resolve()),'source_sha256':hashlib.sha256(source).hexdigest(),
        'plot_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_policy':report['policy'],'metrics':stats,'jsd_summary_bin_width_0_5':js_summary,'burial':burial,
        'jsd_edges':{str(k):v.tolist() for k,v in edges_by_width.items()},
        'jsd_definition':'base-2 Jensen-Shannon divergence between normalized pKa histograms of the same matched sites per protein; no pseudocounts',
        'cohorts':'overall: all eligible paired sites, including flagged; strict: original site-level strict tier',
        'caveats':[args.caption,
                   'Strict sampled monotonicity/convergence is not proof of branch uniqueness.',
                   'Pooled correlations include between-group differences; per-group metrics supplied separately.',
                   'JSD depends on bin width, bin origin and sample size; all proteins with at least one pair retained.',
                   'Excluded or missing sites absent from both distributions; low JSD does not certify site-wise agreement.']}
    (args.output/'metrics.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'metrics':stats,'jsd':js_summary,'output':str(args.output)},indent=2))


def exposure_correlations(rows):
    rows=[r for r in rows if r['functional_sasa'] is not None and np.isfinite(r['functional_sasa']) and r['functional_sasa']>=0]
    x=np.array([r['functional_sasa'] for r in rows]);signed=np.array([r['error'] for r in rows])
    result={'sites':len(rows)}
    for name,y in [('signed_error',signed),('absolute_error',abs(signed))]:
        valid=len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0
        result[name+'_pearson_r']=float(pearsonr(x,y).statistic) if valid else None
        result[name+'_spearman_rho']=float(spearmanr(x,y).statistic) if valid else None
    return result


def plot_burial(cohorts,output,title,plt):
    from matplotlib.colors import LogNorm
    fig,axs=plt.subplots(2,3,figsize=(16,9));stats={};by_group=[];by_protein=[]
    limits=[0,1,5,20,np.inf];labels=['<1','1–5','5–20','≥20']
    for row,(name,rs) in enumerate(cohorts.items()):
        kept=[r for r in rs if r['functional_sasa'] is not None and np.isfinite(r['functional_sasa']) and r['functional_sasa']>=0]
        stats[name]={**exposure_correlations(rs),'missing_sasa_sites':len(rs)-len(kept)}
        if not kept:continue
        x=np.array([r['functional_sasa'] for r in kept]);delta=np.array([r['error'] for r in kept])
        for col,(kind,y) in enumerate([('signed_error',delta),('absolute_error',abs(delta))]):
            ax=axs[row,col]
            hb=ax.hexbin(np.log1p(x),y,gridsize=65,mincnt=1,norm=LogNorm(),cmap='viridis')
            fig.colorbar(hb,ax=ax,label='Paired sites / hexagon',fraction=.046,pad=.04)
            ticks=np.array([0,1,5,20,50,100,200]);ticks=ticks[ticks<=max(x.max(),1)]
            ax.set(xticks=np.log1p(ticks),xticklabels=ticks,xlabel='Functional-group SASA (Å²; log1p spacing)',
                   ylabel='JAX − PROPKA' if col==0 else '|JAX − PROPKA|',title=f'{name}: '+kind.replace('_',' '))
            r=stats[name][kind+'_pearson_r'];rho=stats[name][kind+'_spearman_rho']
            if r is not None:
                ax.text(.98,.97,f'n={len(kept):,}\nPearson r={r:.3f}\nSpearman ρ={rho:.3f}',ha='right',va='top',transform=ax.transAxes,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'},fontsize=9)
        ax=axs[row,2];means=[];counts=[]
        for lo,hi in zip(limits[:-1],limits[1:]):
            values=abs(delta[(x>=lo)&(x<hi)]);counts.append(len(values));means.append(float(values.mean()) if len(values) else np.nan)
        ax.bar(labels,means,color='#2563a6' if name=='Overall' else '#c46a14')
        for k,(mean,count) in enumerate(zip(means,counts)):
            if np.isfinite(mean):ax.text(k,mean,f'n={count:,}',ha='center',va='bottom',fontsize=9)
        ax.margins(y=.18);ax.set(xlabel='Functional-group SASA (Å²)',ylabel='Mean absolute pKa error',title=f'{name}: exposure strata')
        for group in sorted({r['group'] for r in rs}):
            by_group.append({'cohort':name,'group':group,**exposure_correlations([r for r in rs if r['group']==group])})
        for protein in sorted({r['protein'] for r in rs}):
            by_protein.append({'cohort':name,'protein':protein,**exposure_correlations([r for r in rs if r['protein']==protein])})
    fig.suptitle(title+' · pKa error versus burial / exposure',fontsize=16)
    fig.text(.5,.015,'Lower SASA = more buried. SASA is area, not depth; isolated observed protein chain, 1.4 Å probe. Missing annotations excluded.\nSite-pooled associations are descriptive, not causal; residue-type and per-protein correlations supplied separately. Strict does not establish unique branches.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.065,1,.95))
    for suffix in ['png','pdf']:fig.savefig(output/f'burial_error.{suffix}')
    plt.close(fig)
    if by_group:write_csv(output/'burial_correlations_by_group.csv',by_group)
    if by_protein:write_csv(output/'burial_correlations_by_protein.csv',by_protein)
    return stats


if __name__=='__main__':main()
