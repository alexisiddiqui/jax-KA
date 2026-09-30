#!/usr/bin/env python3
"""Same-coordinate JAX/PROPKA timing and discrepancy report; no automatic speedup claim."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import jax
import numpy as np
from bench import measure
from jaxpropka import TitrationModel,ModelConfig,one_hot,GROUPS
from jaxpropka.parameters import GROUP_AA
from jaxpropka.topology import load_topology
from jaxpropka.geometry import build_candidates
from jaxpropka.precompute import build_cache
from jaxpropka.reference import write_reference_structure,run_reference,compare_reference
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('structure');p.add_argument('--backend',choices=['modern','legacy30'],default='modern')
    p.add_argument('--legacy-root');p.add_argument('--expected-commit');p.add_argument('--sequence')
    p.add_argument('--repeats',type=int,default=10);p.add_argument('--reference-repeats',type=int,default=3)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--output',default='reports/comparison.json')
    args=p.parse_args()
    if min(args.repeats,args.reference_repeats)<1:p.error('repeat counts must be positive')
    start=time.perf_counter()
    top=load_topology(args.structure,freeze_disulfides=True)
    candidates=build_candidates(top);cache=build_cache(top,candidates)
    preparation=time.perf_counter()-start
    model=TitrationModel(cache,ModelConfig(steps=args.steps))
    probabilities=model.native_probabilities if args.sequence is None else one_hot(args.sequence)
    model.validate_probabilities(probabilities)
    sequence=np.argmax(np.asarray(probabilities),axis=-1)
    sites=[]
    for i,a in enumerate(sequence):
        for g in np.flatnonzero(GROUP_AA==a):
            if cache.group_mask[i,g]:sites.append((i,GROUPS[g]))
        for g in (7,8):
            if cache.group_mask[i,g]:sites.append((i,GROUPS[g]))
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'normalized.pdb'
        mapping=write_reference_structure(top,candidates,path,sequence)
        runs=[run_reference(path,backend=args.backend,legacy_root=args.legacy_root,
              expected_commit=args.expected_commit,mapping=mapping) for _ in range(args.reference_repeats)]
    report=compare_reference(model,probabilities,runs[0])
    report['timings']={
        'preparation_seconds':preparation,
        'jax_charge_forward':measure(model.charge(7.),probabilities,args.repeats),
        'jax_curves_29_ph':measure(model.curves(np.linspace(0,14,29)),probabilities,args.repeats),
        'jax_physical_site_direct_pkas':measure(model.pka_sites(sites),probabilities,min(3,args.repeats)),
        'jax_all_conditional_grid_pkas_73_ph':measure(model.pka_from_grid(np.linspace(-2,16,73)),probabilities,args.repeats),
        'reference_complete_process_seconds':[run.elapsed_seconds for run in runs],
        'reference_complete_process_median_seconds':float(np.median([r.elapsed_seconds for r in runs])),
    }
    report['device']=[str(d) for d in jax.devices()]
    report['timing_scope']='JAX warmed sequence-only kernels vs PROPKA full subprocess (startup, parsing, calculation, writing). Different algorithms/outputs; no automatic scientific speedup ratio.'
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    out.with_suffix('.pka').write_text(runs[0].output_text)
    out.with_suffix('.stdout.txt').write_text(runs[0].stdout)
    print(json.dumps(report['metrics'],indent=2))

if __name__=='__main__':main()
