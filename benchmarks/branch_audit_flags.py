"""Attach prior, provenance-matched evidence; never infer uniqueness from absence."""
import hashlib
import json


def load_evidence(directory,model_sha256):
    evidence={};hashes={}
    for path in sorted(directory.glob('case-*.json')):
        raw=path.read_bytes();case=json.loads(raw)
        if case['status']!='complete' or case['model_sha256']!=model_sha256:
            raise ValueError(f'incompatible branch audit: {path}')
        if case['pdb_id'] in evidence:raise ValueError('duplicate branch audit')
        evidence[case['pdb_id']]=case
        hashes[path.name]=hashlib.sha256(raw).hexdigest()
    if not evidence:raise ValueError('no branch evidence found')
    return evidence,hashes


def annotate(case,evidence):
    audit=evidence.get(case['pdb_id'])
    flag={'status':'not_assessed','scope':'structure-level, not a claim about every site'}
    if audit is not None:
        for key in ['source_sha256','cache_fingerprint']:
            if case.get(key)!=audit.get(key):raise ValueError(f'branch audit {key} mismatch')
        differences=[max(max(row) for row in point['weighted_occupancy_distances']) for point in audit['points']]
        flag.update(status='known_competing_branches' if max(differences)>1e-3 else 'no_difference_in_sampled_checks',
                    sampled_ph=[point['ph'] for point in audit['points']],
                    max_sampled_occupancy_difference=max(differences),
                    unique_equilibrium_established=False)
    case['branch_audit']=flag
    for site in case.get('sites',[]):
        site['structure_branch_status']=flag['status']
    return case
