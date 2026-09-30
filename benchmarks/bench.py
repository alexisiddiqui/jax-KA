#!/usr/bin/env python3
"""Synchronized CPU/GPU benchmark. Synthetic timings are NOT protein/PROPKA timings."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import time
import jax
import jax.numpy as jnp
import numpy as np
from jaxpropka import TitrationModel, StructureCache, ModelConfig, prepare
from jaxpropka.synthetic import synthetic_cache


def ready(tree):
    return jax.tree.map(lambda x:x.block_until_ready() if hasattr(x,"block_until_ready") else x,tree)


def measure(fn, argument, repetitions):
    argument=jax.device_put(argument)
    ready(argument)
    fn=jax.jit(fn)
    start=time.perf_counter();compiled=fn.lower(argument).compile();compile_s=time.perf_counter()-start
    for _ in range(3): ready(compiled(argument))
    samples=[]
    for _ in range(repetitions):
        start=time.perf_counter();ready(compiled(argument));samples.append(time.perf_counter()-start)
    result={"compile_seconds":compile_s,"median_ms":1000*float(np.median(samples)),
            "p10_ms":1000*float(np.percentile(samples,10)),"p90_ms":1000*float(np.percentile(samples,90)),
            "repetitions":repetitions}
    try:
        memory=compiled.memory_analysis()
        if memory is not None:
            result["compiled_memory_bytes"]={k:int(getattr(memory,k)) for k in
                ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes")}
    except (AttributeError,NotImplementedError):
        pass
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group()
    source.add_argument("--pdb");source.add_argument("--cache");source.add_argument("--synthetic-n",type=int,default=None)
    parser.add_argument("--neighbors",type=int,default=8)
    parser.add_argument("--repeats",type=int,default=20)
    parser.add_argument("--steps",type=int,default=64)
    parser.add_argument("--batch",type=int,default=8)
    parser.add_argument("--pka-sites",type=int,default=2)
    parser.add_argument("--all-pka",action="store_true",help="expensive: benchmark all 9 channels at all positions")
    parser.add_argument("--require-gpu",action="store_true")
    parser.add_argument("--output",default="reports/benchmark.json")
    args=parser.parse_args()
    if args.repeats<1 or args.batch<1 or args.pka_sites<1:
        parser.error("repeat/batch/site counts must be positive")
    if args.require_gpu and not any(d.platform=="gpu" for d in jax.devices()):
        raise RuntimeError("GPU requested, but JAX reports no GPU device")
    start=time.perf_counter()
    if args.pdb: cache=prepare(args.pdb);workload="molecular structure"
    elif args.cache: cache=StructureCache.load(args.cache);workload="loaded structural cache"
    else:
        cache=synthetic_cache(args.synthetic_n or 32,neighbors=args.neighbors,chains=2)
        workload="synthetic numerical graph; NOT a protein accuracy/speed comparison"
    prep=time.perf_counter()-start
    model=TitrationModel(cache,ModelConfig(steps=args.steps))
    logits=jnp.zeros((cache.n_residues,20),jnp.float32)
    p=jax.nn.softmax(logits,-1)
    charge=model.charge(7.);curves=model.curves(np.linspace(0,14,29))
    pka=model.pka_sites([(i,"HIS") for i in range(min(args.pka_sites,cache.n_residues))])
    grid_pka=model.pka_from_grid(np.linspace(-2,16,73))
    bench={"grid_midpoint_pka_all_forward":measure(grid_pka,p,args.repeats),
           "grid_midpoint_pka_all_value_and_gradient_logits":measure(jax.value_and_grad(lambda x:grid_pka(jax.nn.softmax(x,-1)).value.sum()),logits,args.repeats),
           "charge_forward":measure(charge,p,args.repeats),
           "charge_value_and_gradient_logits":measure(jax.value_and_grad(lambda x:jnp.square(charge(jax.nn.softmax(x,-1))).sum()),logits,args.repeats),
           "curves_29_ph_forward":measure(curves,p,args.repeats),
           "curves_value_and_gradient_logits":measure(jax.value_and_grad(lambda x:jnp.square(curves(jax.nn.softmax(x,-1)).total_charge).mean()),logits,args.repeats),
           "selected_midpoint_pka_forward":measure(pka,p,min(args.repeats,5)),
           "selected_midpoint_pka_value_and_gradient_logits":measure(jax.value_and_grad(lambda x:pka(jax.nn.softmax(x,-1)).value.sum()),logits,min(args.repeats,5)),
           "charge_sequence_batch":measure(jax.vmap(charge),jnp.broadcast_to(p,(args.batch,)+p.shape),args.repeats)}
    if args.all_pka:
        bench["all_conditional_midpoint_pka_forward"]=measure(model.pka(),p,min(args.repeats,3))
    curve_check=ready(curves(p));direct_check=ready(pka(p));grid_check=ready(grid_pka(p))
    active=np.asarray(cache.group_mask)
    validation={"curve_all_converged":bool(np.all(curve_check.converged)),
                "selected_direct_all_valid":bool(np.all(direct_check.valid)),
                "grid_active_valid_count":int(np.asarray(grid_check.valid)[active].sum()),
                "active_channel_count":int(active.sum()),
                "selected_grid_vs_direct_max_abs":float(np.max(np.abs(np.asarray(grid_check.value)[:min(args.pka_sites,cache.n_residues),2]-np.asarray(direct_check.value))))}
    packages={}
    for package in ("jax","jaxlib","numpy","scipy","biotite","propka"):
        try: packages[package]=version(package)
        except PackageNotFoundError: packages[package]=None
    out={"schema":1,"workload":workload,"platform":platform.platform(),"python":platform.python_version(),
         "devices":[str(d) for d in jax.devices()],"backend":jax.default_backend(),"packages":packages,
         "n_residues":cache.n_residues,"n_chains":len(cache.chain_ids),"env_K":cache.env_neighbors.shape[1],
         "pair_K":cache.neighbors.shape[1],"cache_bytes":cache.nbytes,"preparation_seconds":prep,
         "sequence_dtype":str(p.dtype),"sequence_batch_size":args.batch,
         "selected_pka_site_count":min(args.pka_sites,cache.n_residues),
         "model_config":asdict(model.config),"validation":validation,"benchmark":bench,
         "notes":["Compilation and preprocessing are separate from synchronized warm evaluations.",
                  "pKa midpoint cost scales with requested site count and is not the single-pH charge cost.",
                  "No reference speedup is inferred from a synthetic graph."]}
    destination=Path(args.output);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))

if __name__=="__main__": main()
