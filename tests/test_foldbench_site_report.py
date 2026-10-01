"""Numerical boundary cases and partial-coverage accounting (no JAX needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from foldbench_site_report import crossing_info, classify, summarize, context_rows, select_tier


@pytest.mark.parametrize("y,count,ambiguous", [
    ([.9, .5, .1], 1, False),
    ([.9, .5, .9], 0, True),
    ([.9, .5, .5, .1], 1, True),
    ([.5, .3, .1], 0, True),
    ([.9, .3, .8, .1], 3, False),
    ([.9, .8, .7], 0, False),
    ([.9, np.nan, .1], 0, True),
])
def test_crossing_boundaries(y, count, ambiguous):
    result = crossing_info(np.arange(len(y)), y)
    assert result["count"] == count
    assert result["ambiguous"] == ambiguous


def test_exact_node_midpoint_not_double_counted():
    assert crossing_info([0, 1, 2], [.9, .5, .1])["value"] == 1


def test_native_convergence_cannot_promote_strict():
    result = classify(np.arange(4), [.9, .8, .85, .1], global_converged=False,
                      native_converged=True, slope_min=1e-5)
    assert not result["strict"]
    assert result["flagged_candidate"]


@pytest.mark.parametrize("coarse,refined,tier", [
    ((True, False), (False, True), "flagged_unique_sampled"),
    ((True, False), (False, False), "invalid"),
    ((False, True), (True, False), "flagged_unique_sampled"),
    ((False, False), (True, False), "invalid"),
    ((True, False), (True, False), "strict"),
])
def test_refined_grid_can_demote_but_not_hide_coarse_failure(coarse, refined, tier):
    def info(flags):
        return dict(zip(("strict", "flagged_candidate"), flags))
    assert select_tier(info(coarse), info(refined)) == tier


def test_partial_sites_in_metrics_and_denominator():
    result = summarize([{"status": "partial", "sites": [
        {"delta": 2., "tier": "strict", "group": "CYS"},
        {"delta": None, "tier": "invalid", "group": "CYS", "reference_missing": True}]},
        {"status": "failed", "stage": "topology"}])
    assert result["overall"]["coverage"] == .5
    assert result["overall"]["mae"] == 2.
    assert result["reference_missing_sites"] == 1
    assert result["cases_without_site_inventory"] == 1


def test_context_uses_full_model_other_label_chain():
    import biotite.structure as struc
    from types import SimpleNamespace
    atoms = struc.AtomArray(2)
    atoms.coord = np.array([[0., 0., 0.], [2.3, 0., 0.]])
    atoms.chain_id = ["A", "B"]
    atoms.res_id = [1, 1]
    atoms.res_name = ["CYS", "ZN"]
    atoms.atom_name = ["SG", "ZN"]
    atoms.element = ["S", "Zn"]
    topology = SimpleNamespace(atoms=atoms[:1], starts=[0, 1],
                               keys=[SimpleNamespace(chain="A")], residue=lambda i: atoms[:1])
    row = context_rows(atoms, topology, [(0, "CYS")])[0]
    assert row["context"]["metal"]["chain"] == "B"
    assert row["context"]["metal"]["distance"] == pytest.approx(2.3)
    assert "metal" in row["context_cohort"]


@pytest.mark.parametrize("reference_failure", [False, True])
def test_evaluate_retries_and_preserves_partial_reference(monkeypatch, reference_failure):
    from types import SimpleNamespace
    from jaxpropka import ModelConfig, model, reference
    from jaxpropka.cache import ResidueKey
    import foldbench_site_report as report

    def curve(arrays, probabilities, ph, *, config):
        y = np.broadcast_to((1 / (1 + np.exp(np.asarray(ph) - 7)))[:, None, None], (len(ph), 2, 9))
        residual = np.full(len(ph), 1e-2 if config.steps == 128 else 1e-7)
        return SimpleNamespace(protonated=y, site_charge=np.zeros_like(y), residual=residual,
                               weighted_residual=residual, converged=residual < config.residual_tolerance)

    keys = [ResidueKey("A", 1), ResidueKey("A", 2)]
    def run_reference(*args, **kwargs):
        if reference_failure:
            raise RuntimeError("mock reference failure")
        return SimpleNamespace(sites=[reference.ReferenceSite(keys[0], "ASP", 6.5)], provenance={"version": "test"})

    monkeypatch.setattr(model, "curve_kernel", curve)
    monkeypatch.setattr(reference, "write_reference_structure", lambda *args: {})
    monkeypatch.setattr(reference, "run_reference", run_reference)
    monkeypatch.setattr(report, "context_rows", lambda *args: [{}, {}])
    monkeypatch.setattr(report, "audit_reference_input", lambda *args: {"passed": True})
    cache = SimpleNamespace(keys=keys, fingerprint=lambda: "test")
    result = report.evaluate(None, None, cache, None, None, None,
                             [(0, "ASP"), (1, "GLU")], ModelConfig(steps=128), 145)
    assert [a["steps"] for a in result["solver_attempts"]][:2] == [128, 512]
    assert len(result["sites"]) == 2
    assert result["sites"][1]["reference_missing"]
    if reference_failure:
        assert result["status"] == "failed"
        assert "mock reference failure" in result["reference_error"]
        assert all(s["surrogate_pka"] is not None for s in result["sites"])
    else:
        assert result["status"] == "partial"
        assert result["compared_sites"] == 1
        assert result["sites"][0]["delta"] == pytest.approx(.5)
        assert result["charge_metrics"]["sites"] == 1
