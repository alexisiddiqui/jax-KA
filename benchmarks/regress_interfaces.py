"""FoldBench official protein-protein pairs using the monomer scoring policy."""
import argparse
import csv
import json
import os
from pathlib import Path
import tarfile
import tempfile
import time
from regress_foldbench import _run_case, _write_checkpoint, sha256
from foldbench_site_report import POLICY
from interface_exposure import POLICY as INTERFACE_POLICY


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--case-index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    rows=list(csv.DictReader(args.manifest.open()))
    row=rows[args.case_index]
    assert row['interface_chain_type_1']==row['interface_chain_type_2']=='protein'
    chains=(row['interface_chain_id_1'],row['interface_chain_id_2'])
    assert chains[0]!=chains[1]
    root=Path(__file__).resolve().parents[1]
    document={'schema':3,'dataset':{'name':'FoldBench protein-protein pairs','case_index':args.case_index,
        'case_count':1,'manifest_case_count':len(rows),'manifest_sha256':sha256(args.manifest),
        'archive_sha256':sha256(args.archive),
        'upstream_manifest':'https://github.com/BEAM-Labs/FoldBench/blob/4273f6877d82bd0b2fa476d1b2f34d121cbccc70/targets/interface_protein_protein.csv'},
        'reference':{'backend':'modern','required_version':'3.5.1'},
        'policy':{'steps':128,'grid_points':145,'site_report':POLICY,'interface':INTERFACE_POLICY,
            'branch_policy':'flag only; monomer branch audits do not transfer to different complexes',
            'implementation_sha256':{name:sha256(Path(__file__).with_name(name)) for name in
                ['regress_interfaces.py','regress_foldbench.py','foldbench_site_report.py','interface_exposure.py']},
            'reference_adapter_sha256':sha256(root/'src/jaxpropka/reference.py'),
            'model_sha256':sha256(root/'src/jaxpropka/model.py')},'cases':[],
        'execution':{'hostname':os.uname().nodename,'cpu_affinity_count':len(os.sched_getaffinity(0)),
            'slurm_array_job_id':os.environ.get('SLURM_ARRAY_JOB_ID'),'slurm_array_task_id':os.environ.get('SLURM_ARRAY_TASK_ID')}}
    with tempfile.TemporaryDirectory(prefix='jaxka-interface-') as directory:
        with tarfile.open(args.archive) as archive:
            filename=row['pdb_id']+'.cif'
            member=next(m for m in archive.getmembers() if Path(m.name).name==filename)
            raw=archive.extractfile(member).read()
        path=Path(directory)/filename;path.write_bytes(raw)
        case=_run_case((row['pdb_id'],'+'.join(chains),str(path)),128,145,(1,1,1),
                       site_report=True,partner_chains=chains)
    case['interface_id']=row['pdb_id']+':'+':'.join(chains);case['partner_chains']=list(chains)
    document['cases']=[case];document['timing_seconds']={'total':time.perf_counter()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True);_write_checkpoint(args.output,document)
    print(json.dumps({'interface':case['interface_id'],'status':case['status'],
        'paired_sites':case.get('compared_sites'),'annotation_error':case.get('interface_annotation_error'),
        'elapsed_seconds':document['timing_seconds']['total']},indent=2),flush=True)


if __name__=='__main__':main()
