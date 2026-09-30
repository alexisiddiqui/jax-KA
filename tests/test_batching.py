from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpropka import ModelConfig, TitrationModel
from jaxpropka.batching import pack_inputs
from jaxpropka.model import curve_kernel, grid_pka_kernel, one_hot
from jaxpropka.synthetic import synthetic_cache


def assert_result_matches(actual, expected, n):
    grid = hasattr(actual, "value")
    ph_residue_fields = {"protonated", "site_charge", "residue_charge", "effective_pka"}
    residue_fields = {"probability", "intrinsic_pka"}
    for name in actual._fields:
        value = np.asarray(getattr(actual, name))
        reference = np.asarray(getattr(expected, name))
        if grid or name in residue_fields:
            value = value[:n]
        elif name in ph_residue_fields:
            value = value[:, :n]
        if reference.dtype == bool:
            np.testing.assert_array_equal(value, reference, err_msg=name)
        else:
            np.testing.assert_allclose(value, reference, rtol=3e-5, atol=3e-6, err_msg=name)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("gate_width", [0.0, 20.0])
def test_padding_preserves_curves_grid_flags_and_frozen_identities(dtype, gate_width):
    cache = synthetic_cache(n=5, neighbors=4, chains=2)
    cache.frozen[0] = True
    config = ModelConfig(gate_width=gate_width)
    model = TitrationModel(cache, config)
    p = np.random.default_rng(22).dirichlet(np.ones(20), size=5).astype(dtype)
    ph = np.linspace(-10, 24, 69, dtype=dtype)
    arrays, padded_p, n = pack_inputs(cache, p, capacities=(8, 8, 8))
    arrays, padded_p = jax.device_put((arrays, padded_p))
    curves = curve_kernel(arrays, padded_p, ph, config=config)
    grid = grid_pka_kernel(arrays, padded_p, ph, config=config)
    assert_result_matches(curves, model.curves(ph)(p), n)
    assert_result_matches(grid, model.pka_from_grid(ph)(p), n)
    np.testing.assert_array_equal(curves.protonated[:, n:], 0)
    np.testing.assert_array_equal(curves.site_charge[:, n:], 0)
    np.testing.assert_array_equal(grid.value[n:], 0)
    assert not np.asarray(grid.valid[n:]).any()
    assert not np.asarray(grid.bracketed[n:]).any()
    for leaf in jax.tree.leaves((curves, grid)):
        assert np.isfinite(leaf).all()


def test_pack_inputs_rounds_up_and_normalizes_without_changing_cache():
    cache = synthetic_cache(n=5, neighbors=4, chains=2)
    fingerprint = cache.fingerprint()
    arrays, p, n = pack_inputs(cache, one_hot(cache.native_index), bucket_multiple=(4, 4, 4))
    assert n == 5
    assert p.shape == (8, 20)
    assert arrays["env_neighbors"].shape == (8, 4)
    assert arrays["neighbors"].shape == (8, 8)
    assert "chain_index" not in arrays
    assert arrays["native_index"].dtype == np.int32
    assert arrays["volume"].dtype == p.dtype == np.float32
    np.testing.assert_array_equal(p.sum(-1), 1)
    assert not arrays["group_mask"][n:].any()
    assert not arrays["env_mask"][n:].any()
    assert not arrays["pair_mask"][n:].any()
    assert cache.fingerprint() == fingerprint


@pytest.mark.parametrize("capacities", [(4, 8, 8), (8, 3, 8), (8, 8, 4)])
def test_pack_inputs_rejects_overflow(capacities):
    cache = synthetic_cache(n=5, neighbors=4)
    with pytest.raises(ValueError, match="truncation is forbidden"):
        pack_inputs(cache, one_hot(cache.native_index), capacities)


@pytest.mark.parametrize("multiples", [(0, 4, 4), (4, 4), (4, 2.5, 4)])
def test_pack_inputs_rejects_invalid_buckets(multiples):
    cache = synthetic_cache(n=3)
    with pytest.raises(ValueError, match="three positive integers"):
        pack_inputs(cache, one_hot(cache.native_index), bucket_multiple=multiples)


@pytest.mark.parametrize("kernel", [curve_kernel, grid_pka_kernel])
def test_shared_kernel_reuses_trace_across_structures_and_ph_values(monkeypatch, kernel):
    import jaxpropka.model as implementation

    config = ModelConfig(steps=48)
    ph = np.linspace(-10, 24, 35, dtype=np.float32)
    cases = []
    for n, capacity, seed in [(3, 8, 11), (5, 8, 19), (9, 16, 31)]:
        cache = synthetic_cache(n=n, neighbors=4, seed=seed)
        p = one_hot(cache.native_index)
        model = TitrationModel(cache, config)
        grid = ph + np.float32(seed / 100)
        expected = (model.curves(grid)(p) if kernel is curve_kernel
                    else model.pka_from_grid(grid)(p))
        arrays, padded_p, _ = pack_inputs(cache, p, capacities=(capacity, 8, 8))
        cases.append((n, jax.device_put(arrays), jax.device_put(padded_p), grid, expected))

    curve_kernel.clear_cache()
    grid_pka_kernel.clear_cache()
    traces = []
    original = implementation._local_terms

    def track_trace(d, p, cfg):
        traces.append(p.shape)
        return original(d, p, cfg)

    monkeypatch.setattr(implementation, "_local_terms", track_trace)
    try:
        for (n, arrays, p, grid, expected), count in zip(cases, (1, 1, 2)):
            actual = jax.device_get(kernel(arrays, p, grid, config=config))
            assert len(traces) == count
            assert_result_matches(actual, expected, n)
        # Static configuration changes deliberately create another specialization.
        jax.block_until_ready(kernel(arrays, p, grid, config=replace(config, steps=49)))
        assert len(traces) == 3
    finally:
        curve_kernel.clear_cache()
        grid_pka_kernel.clear_cache()


def test_padded_grid_gradient_matches_unpadded_and_has_no_dummy_contribution():
    cache = synthetic_cache(n=3, neighbors=2)
    cache.frozen[0] = True
    config = ModelConfig()
    ph = np.linspace(-10, 24, 69)
    logits = jnp.asarray(np.random.default_rng(3).normal(size=(3, 20)))
    p = np.asarray(jax.nn.softmax(logits, -1))
    arrays, _, n = pack_inputs(cache, p, capacities=(4, 4, 4))
    arrays = jax.device_put(arrays)
    padded_logits = jnp.concatenate((logits, jnp.zeros((1, 20))), axis=0)

    def loss(z):
        return grid_pka_kernel(arrays, jax.nn.softmax(z, -1), ph, config=config).value.sum()

    readout = TitrationModel(cache, config).pka_from_grid(ph)
    expected = jax.grad(lambda z: readout(jax.nn.softmax(z, -1)).value.sum())(logits)
    actual = jax.grad(loss)(padded_logits)
    np.testing.assert_allclose(actual[:n], expected, rtol=1e-8, atol=1e-9)
    np.testing.assert_array_equal(actual[n:], 0)
    np.testing.assert_array_equal(actual[0], 0)
