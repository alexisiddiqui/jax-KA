"""Fresh-process compilation/execute split for existing interface solve schedules."""
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
from jaxpropka import ModelConfig
from jaxpropka.batching import pack_inputs
from jaxpropka.geometry import build_candidates
from jaxpropka.model import curve_kernel,one_hot
from jaxpropka.precompute import build_cache
from jaxpropka.topology import load_topology


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();started=time.perf_counter()
    jax.config.update('jax_enable_compilation_cache',False)
    source=args.report.read_bytes();original=json.loads(source);case=original['cases'][0]
    root=Path(__file__).resolve().parents[1]
    assert hashlib.sha256((root/'src/jaxpropka/model.py').read_bytes()).hexdigest()==original['policy']['model_sha256']
    result={'status':'running','interface_id':case['interface_id'],'n_residues':case['n_residues'],
        'source_report':str(args.report),'source_report_sha256':hashlib.sha256(source).hexdigest(),
        'hostname':os.uname().nodename,'cpu_affinity':len(os.sched_getaffinity(0)),
        'jax_version':jax.__version__,'persistent_compilation_cache':False,
        'scope':'fresh CPU process; historical schedule, not retrospective exact timings; execution includes device_get',
        'attempts':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint(stage):
        result['stage']=stage;result['elapsed_seconds']=time.perf_counter()-started
        tmp=args.output.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(args.output)
        print(case['interface_id'],stage,round(result['elapsed_seconds'],2),flush=True)
    with tarfile.open(root/'ground_truth_1522.tar') as archive:
        member=next(m for m in archive.getmembers() if Path(m.name).name==case['pdb_id']+'.cif')
        raw=archive.extractfile(member).read()
    assert hashlib.sha256(raw).hexdigest()==case['source_sha256']
    atoms=pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(raw.decode())),model=1,altloc='occupancy',use_author_fields=False,include_bonds=True)
    topology=load_topology(atoms[np.isin(atoms.chain_id,case['partner_chains'])],
        gap_policy='free',freeze_disulfides=True,ignore_nonprotein=True)
    topology.metadata.update(source_name=case['pdb_id']+'.cif',source_sha256=case['source_sha256'],
        foldbench_pdb_id=case['pdb_id'],foldbench_label_asym_id=case['chain_id'])
    cache=build_cache(topology,build_candidates(topology,missing_sidechain='template'))
    assert cache.fingerprint()==case['cache_fingerprint']
    arrays,p,_=pack_inputs(cache,one_hot(cache.native_index),bucket_multiple=(1,1,1))
    arrays,p=jax.device_put((arrays,p));jax.block_until_ready((arrays,p))
    checkpoint('cache_ready');result['preparation_seconds']=result['elapsed_seconds']
    config=ModelConfig(**case['midpoint_model_config']);executables={}
    for attempt in case['solver_attempts']:
        cfg=replace(config,steps=attempt['steps'])
        ph=np.arange(15,dtype=np.float32) if attempt['grid']=='charge' else np.linspace(cfg.ph_min,cfg.ph_max,attempt['points'],dtype=np.float32)
        ph=jax.device_put(ph);ph.block_until_ready()
        key=(len(ph),cfg.steps)
        lower_seconds=compile_seconds=0.
        reused=key in executables
        if not reused:
            t=time.perf_counter();lowered=curve_kernel.lower(arrays,p,ph,config=cfg);lower_seconds=time.perf_counter()-t
            t=time.perf_counter();executables[key]=lowered.compile();compile_seconds=time.perf_counter()-t
        executable=executables[key]
        durations=[]
        for repeat in range(3):
            t=time.perf_counter();out=jax.device_get(executable(arrays,p,ph));durations.append(time.perf_counter()-t)
        residual=float(np.max(out.residual))
        assert np.isclose(residual,attempt['max_all_group_residual'],atol=2e-6,rtol=.01)
        result['attempts'].append({'grid':attempt['grid'],'points':len(ph),'steps':cfg.steps,
            'reused_executable':reused,'trace_lower_seconds':lower_seconds,'compile_seconds':compile_seconds,
            'first_execution_seconds':durations[0],'warm_execution_seconds':durations[1:],
            'warm_median_seconds':float(np.median(durations[1:])),
            'reconstructed_cold_seconds':lower_seconds+compile_seconds+durations[0],
            'original_elapsed_seconds':attempt['elapsed_seconds'],'max_residual':residual})
        checkpoint(f"profile_{attempt['grid']}_{cfg.steps}")
    records=result['attempts'];comp=sum(r['trace_lower_seconds']+r['compile_seconds'] for r in records)
    cold=sum(r['reconstructed_cold_seconds'] for r in records)
    result['summary']={'trace_and_compile_seconds':comp,'first_execution_seconds':cold-comp,
        'cold_schedule_seconds':cold,'compilation_fraction':comp/cold,
        'warm_schedule_seconds':sum(r['warm_median_seconds'] for r in records)}
    result['status']='complete';checkpoint('complete')


if __name__=='__main__':main()
