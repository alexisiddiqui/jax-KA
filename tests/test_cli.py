import json
import numpy as np
from jaxpropka.cli import main
from jaxpropka.synthetic import synthetic_cache


def test_predict_cli_preserves_selection_and_chain_metadata(tmp_path,capsys):
    cache=synthetic_cache(n=4,chains=2)
    source=tmp_path/'cache.npz';out=tmp_path/'out.npz';cache.save(source)
    main(['predict',str(source),'--points','3','--residues','2,0','--output',str(out)])
    report=json.loads(capsys.readouterr().out)
    assert report['all_curve_states_converged']
    with np.load(out,allow_pickle=False) as saved:
        assert saved['curves_residue_charge'].shape==(3,2)
        assert saved['curves_chain_charge'].shape==(3,2)
        metadata=json.loads(str(saved['metadata']))
        assert metadata['selected_residues'][0]==vars(cache.keys[2])
        assert metadata['chain_ids']==list(cache.chain_ids)
