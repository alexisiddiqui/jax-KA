"""Rebuild one accepted case and compare dense/packed solver schedules."""
import argparse
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

import jax
import numpy as np
from biotite.structure.io import pdbx

from foldbench_site_report import classify, select_tier
from jaxpropka import GROUPS, ModelConfig
from jaxpropka.batching import pack_inputs
from jaxpropka.geometry import build_candidates
from jaxpropka.model import (curve_kernel, one_hot, pack_interaction_edges,
                             packed_curve_kernel)
from jaxpropka.parameters import GROUP_AA
from jaxpropka.precompute import build_cache
from jaxpropka.topology import load_topology


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();started=time.perf_counter()
    source=args.report.read_bytes();document=json.loads(source);case=document['cases'][0]
    result={'status':'running','source_report':str(args.report),
        'source_report_sha256':hashlib.sha256(source).hexdigest(),
        'pdb_id':case['pdb_id'],'chain_id':case['chain_id'],
        'accepted_status':case['status'],'hostname':os.uname().nodename,
        'jax_version':jax.__version__,'comparisons':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint(stage):
        result.update(stage=stage,elapsed_seconds=time.perf_counter()-started)
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
        print(case['pdb_id'],stage,round(result['elapsed_seconds'],2),flush=True)
    checkpoint('start')
    if case.get('cache_fingerprint') is None:
        result.update(status='excluded_input',stage='complete',reason='accepted report has no numerical cache')
        checkpoint('complete');return
    with tarfile.open(args.archive) as archive:
        member=next(m for m in archive.getmembers() if Path(m.name).name==case['pdb_id']+'.cif')
        raw=archive.extractfile(member).read()
    if hashlib.sha256(raw).hexdigest()!=case['source_sha256']:
        raise ValueError('source hash mismatch')
    atoms=pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(raw.decode())),model=1,
        altloc='occupancy',use_author_fields=False,include_bonds=True)
    chains=tuple(case.get('partner_chains') or (case['chain_id'],))
    topology=load_topology(atoms[np.isin(atoms.chain_id,chains)],gap_policy='free',
        freeze_disulfides=True,ignore_nonprotein=True)
    topology.metadata.update(source_name=case['pdb_id']+'.cif',source_sha256=case['source_sha256'],
        foldbench_pdb_id=case['pdb_id'],foldbench_label_asym_id=case['chain_id'])
    cache=build_cache(topology,build_candidates(topology,missing_sidechain='template'))
    if cache.fingerprint()!=case['cache_fingerprint']:
        raise ValueError('cache fingerprint mismatch')
    arrays,p,n=pack_inputs(cache,one_hot(cache.native_index),bucket_multiple=(1,1,1))
    edges=pack_interaction_edges(arrays['pair_mask'],arrays['neighbors'])
    arrays,p,edges=jax.device_put((arrays,p,edges));jax.block_until_ready((arrays,p,edges))
    expected=[]
    for i,aa in enumerate(cache.native_index):
        expected.extend((i,int(g)) for g in np.flatnonzero(GROUP_AA==aa) if cache.group_mask[i,g])
        expected.extend((i,g) for g in (7,8) if cache.group_mask[i,g])
    result.update(n_residues=n,n_sites=len(expected),active_pair_entries=len(edges[0]),
                  padded_pair_entries=int(arrays['pair_mask'].size))
    checkpoint('cache_ready')
    classifications={'dense':{},'packed':{}}
    unique=[]
    for attempt in case['solver_attempts']:
        key=(attempt['grid'],attempt['points'],attempt['steps'])
        if key not in unique:unique.append(key)
    for label,points,steps in unique:
        schedule_key=(label,points,steps)
        cfg=replace(ModelConfig(**case['midpoint_model_config']),steps=steps)
        ph=(np.arange(15,dtype=np.float32) if label=='charge'
            else np.linspace(cfg.ph_min,cfg.ph_max,points,dtype=np.float32))
        before=time.perf_counter();dense=jax.device_get(curve_kernel(arrays,p,ph,config=cfg));dense_seconds=time.perf_counter()-before
        before=time.perf_counter();packed=jax.device_get(packed_curve_kernel(arrays,p,ph,edges,config=cfg));packed_seconds=time.perf_counter()-before
        occupancy_error=float(np.max(np.abs(dense.protonated-packed.protonated)))
        charge_error=float(np.max(np.abs(dense.total_charge-packed.total_charge)))
        flags_equal=bool(np.array_equal(dense.converged,packed.converged))
        categorical_equal=True;pka_error=0.
        if label!='charge':
            for backend,out in [('dense',dense),('packed',packed)]:
                global_ok=bool(np.all(out.converged));native_ok=bool(np.all(out.weighted_residual<cfg.residual_tolerance))
                rows=[classify(ph,out.protonated[:,i,g],global_converged=global_ok,
                    native_converged=native_ok,slope_min=cfg.slope_min) for i,g in expected]
                classifications[backend][schedule_key]=rows
            for a,b in zip(classifications['dense'][schedule_key],classifications['packed'][schedule_key]):
                categorical=('count','ambiguous','bracketed','sampled_monotone','strict','flagged_candidate','reasons')
                categorical_equal &= all(a[x]==b[x] for x in categorical)
                if a['value'] is not None and b['value'] is not None:
                    pka_error=max(pka_error,abs(a['value']-b['value']))
                else:categorical_equal &= a['value'] is b['value']
        passed=(occupancy_error<=2e-5 and charge_error<=2e-4 and flags_equal and
                categorical_equal and pka_error<=2e-4)
        result['comparisons'].append({'grid':label,'points':points,'steps':steps,
            'dense_seconds':dense_seconds,'packed_seconds':packed_seconds,
            'occupancy_max_abs':occupancy_error,'charge_max_abs':charge_error,
            'pka_max_abs':pka_error,'convergence_flags_equal':flags_equal,
            'site_categories_equal':bool(categorical_equal),'passed':bool(passed)})
        checkpoint(f'{label}_{points}_{steps}')
    tiers_equal=True
    for backend in ('dense','packed'):
        coarse={k:v for k,v in classifications[backend].items() if k[0]=='coarse'}
        refined={k:v for k,v in classifications[backend].items() if k[0]=='refined'}
        # The accepted adaptive schedule's last evaluation at each resolution is authoritative.
        c=next(reversed(coarse.values()));f=next(reversed(refined.values())) if refined else None
        result[backend+'_tiers']=[select_tier(x,f[i] if f else None) for i,x in enumerate(c)]
    tiers_equal=result['dense_tiers']==result['packed_tiers']
    result['tier_classifications_equal']=tiers_equal
    result['status']='passed' if tiers_equal and all(x['passed'] for x in result['comparisons']) else 'failed'
    checkpoint('complete')


if __name__=='__main__':main()
