#!/usr/bin/env python3
"""Record an implementation-only snapshot; NOT external PROPKA ground truth."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from jaxpropka import TitrationModel
from jaxpropka.synthetic import synthetic_cache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='tests/data/synthetic_baseline.json')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args();path=Path(args.output)
    if path.exists() and not args.force:
        raise FileExistsError('baseline exists; an explicit --force and review are required')
    model=TitrationModel(synthetic_cache(n=6,chains=2,neighbors=4,seed=7))
    ph=[2.,5.,7.,10.,13.]
    curves=model.curves(ph);midpoint=model.pka_sites([(0,'ASP'),(2,'HIS'),(5,'CTERM')])
    result={'kind':'self-generated implementation snapshot; NOT PROPKA reference',
            'jax_version':jax.__version__,'dtype':'float32','model_config':asdict(model.config),
            'ph':ph,'cases':{}}
    for name,p in [('hard',model.native_probabilities),('soft',.85*model.native_probabilities+.15/20)]:
        p=jnp.asarray(p,jnp.float32)
        out=jax.device_get(curves(p));pk=jax.device_get(midpoint(p))
        if not np.all(out.converged) or not np.all(pk.valid):
            raise RuntimeError('refusing to record nonconverged numerical baseline')
        result['cases'][name]={'residue_charge':out.residue_charge.tolist(),
            'chain_charge':out.chain_charge.tolist(),'total_charge':out.total_charge.tolist(),
            'selected_midpoint_pka':pk.value.tolist()}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');print(path)

if __name__=='__main__': main()
