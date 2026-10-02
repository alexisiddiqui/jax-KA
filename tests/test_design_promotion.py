import hashlib
import importlib.util
import json
from pathlib import Path


def setup_report(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'benchmarks'))
    spec=importlib.util.spec_from_file_location('design_validation',root/'benchmarks/validate_design_preset.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'RUNTIME',tmp_path)
    baseline=tmp_path/'gradient-memory/case-43';baseline.mkdir(parents=True)
    (baseline/'selectivity-baseline-t128-h15.json').write_text(json.dumps(dict(memory=dict(temp_size_in_bytes=1000))))
    output=tmp_path/'panel';output.mkdir()
    (output/'manifest.json').write_text(json.dumps(dict(eligible=[dict(case=43,fingerprint='fixture')],gradient_cases=[43],scaling_cases=[43],excluded=[])))
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src/jaxpropka').glob('*.py')}
    for action in ('forward','gradient','scaling'):
        mixtures=(.05,.001) if action=='scaling' else (0.,.05,.001)
        row=dict(case=43,action=action,status='complete',source_sha256=source,fingerprints=dict(bound='fixture'),memory=dict(temp_size_in_bytes=100),
                 records=[dict(mixture=m,numerical_valid=True,classification='passed',
                               reference_valid=True,parity=True,value_parity=True,gradient_parity=True,
                               directional_checks=[dict(passed=True)]*3) for m in mixtures])
        (output/f'{action}-43.json').write_text(json.dumps(row))
    return module,output


def test_promotion_requires_complete_current_results(tmp_path,monkeypatch):
    module,output=setup_report(tmp_path,monkeypatch)
    module.summarize(output)
    assert json.loads((output/'summary.json').read_text())['numerical_gates_passed']
    path=output/'gradient-43.json';row=json.loads(path.read_text());row['source_sha256']={}
    path.write_text(json.dumps(row));module.summarize(output)
    result=json.loads((output/'summary.json').read_text())
    assert result['stale_source']==['gradient-43'] and not result['numerical_gates_passed']
    row['records']=[];path.write_text(json.dumps(row));module.summarize(output)
    assert 'gradient-43' in json.loads((output/'summary.json').read_text())['missing']


def test_invalid_scaling_is_not_a_memory_win(tmp_path,monkeypatch):
    module,output=setup_report(tmp_path,monkeypatch)
    path=output/'scaling-43.json';row=json.loads(path.read_text());row['records'][0]['numerical_valid']=False
    path.write_text(json.dumps(row));module.summarize(output)
    result=json.loads((output/'summary.json').read_text())
    assert result['temporary_reduction']==10 and not result['numerical_gates_passed']


def test_audit_failure_cannot_hide_converged_parity_regression(tmp_path,monkeypatch):
    module,output=setup_report(tmp_path,monkeypatch)
    path=output/'forward-43.json';row=json.loads(path.read_text())
    row['records'][0].update(classification='branch_disagreement',parity=False)
    path.write_text(json.dumps(row));module.summarize(output)
    result=json.loads((output/'summary.json').read_text())
    assert not result['numerical_gates_passed']
    assert result['regressions']==[dict(action='forward',case=43,mixture=0.,reason='converged_parity_failure')]
