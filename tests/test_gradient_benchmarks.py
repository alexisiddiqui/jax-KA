"""Reporting must not mistake a saturated loss for accurate selectivity gradients."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    'gradient_summary', Path(__file__).resolve().parents[1] / 'benchmarks/summarize_gradient_memory.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_saturated_loss_gradient_is_rescaled():
    s = 6.5
    slope = np.exp(-np.logaddexp(0., (s - 1.) / .1))
    saved = dict(selectivity=s, gradient=np.asarray([-slope, 2*slope], dtype=np.float32))
    np.testing.assert_allclose(summary.selectivity_gradient(saved, 'selectivity'), [1., -2.], rtol=1e-6)
    wrong = dict(selectivity=s, gradient=np.asarray([0., 0.], dtype=np.float32))
    assert np.allclose(saved['gradient'], wrong['gradient'], atol=2e-5)
    assert not np.allclose(summary.selectivity_gradient(saved, 'selectivity'),
                           summary.selectivity_gradient(wrong, 'selectivity'), atol=2e-5)


def test_underflowed_loss_slope_is_not_accepted():
    saved = dict(selectivity=100., gradient=np.zeros(2, dtype=np.float32))
    assert np.isnan(summary.selectivity_gradient(saved, 'selectivity')).all()


def test_complex_charge_gradient_is_unscaled():
    saved = dict(gradient=np.asarray([.5, -.2], dtype=np.float32))
    np.testing.assert_array_equal(summary.selectivity_gradient(saved, 'complex'), saved['gradient'])
