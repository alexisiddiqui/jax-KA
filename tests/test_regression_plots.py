import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('regression_plots',Path(__file__).resolve().parents[1]/'benchmarks/plot_foldbench_regression.py')
plots=importlib.util.module_from_spec(spec);spec.loader.exec_module(plots)


def test_jsd_is_divergence_not_distance():
    edges=np.array([0.,1.,2.])
    assert plots.js_divergence([.5,1.5],[.5,1.5],edges)==0
    assert plots.js_divergence([.5,.5],[1.5,1.5],edges)==1
    assert np.isclose(plots.js_divergence([.5,.5],[.5,1.5],edges),.3112781244591328)


def test_shared_edges_and_no_silent_dropping():
    values=np.array([[-10.,24.],[26.45,6.5]])
    edges=plots.shared_edges(values,.5)
    assert edges[0]<=values.min() and edges[-1]>values.max()
    with pytest.raises(ValueError,match='omit'):
        plots.js_divergence([1],[3],[0,2])
    with pytest.raises(ValueError,match='paired'):
        plots.js_divergence([1],[1,2],[0,3])
