#!/usr/bin/env python3
"""Ablate interaction terms for a FoldBench grid-failure diagnosis."""
from dataclasses import replace
import time

import jax
import numpy as np

import regress_foldbench as runner
from jaxpropka import model


def main():
    original_grid = model.grid_pka_kernel
    original_case = runner._run_case
    observations = []

    variants = (
        ("baseline_512", dict(steps=512)),
        ("no_coulomb", dict(steps=512, coulomb_scale=0.0)),
        ("no_hbond", dict(steps=512, hbond_scale=0.0)),
        ("uniform_dielectric_160", dict(
            steps=512, dielectric_buried=160.0, dielectric_surface=160.0)),
        ("no_coupling_terms", dict(
            steps=512, coulomb_scale=0.0, hbond_scale=0.0)),
    )

    def diagnostic_grid(arrays, probabilities, ph, *, config):
        baseline = None
        for label, changes in variants:
            cfg = replace(config, **changes)
            started = time.perf_counter()
            grid = jax.device_get(original_grid(arrays, probabilities, ph, config=cfg))
            curves = jax.device_get(model.curve_kernel(
                arrays, probabilities, ph, config=cfg))
            if baseline is None:
                baseline = grid
            native = np.asarray(grid.probability) > 0
            rise = np.max(np.diff(curves.protonated, axis=0), axis=0)
            invalid = native & ~grid.valid
            observations.append({
                "variant": label,
                "invalid_native_sites": int(invalid.sum()),
                "nonmonotone_native_sites": int(
                    (native & ~grid.sampled_monotone).sum()),
                "unbracketed_native_sites": int((native & ~grid.bracketed).sum()),
                "unconverged_ph_points": int((~curves.converged).sum()),
                "max_residual": float(np.max(curves.residual)),
                "max_native_occupancy_rise": float(np.max(rise[native], initial=0)),
                "seconds": time.perf_counter() - started,
            })
            print(f"Coupling diagnostic: {observations[-1]}", flush=True)
        return baseline

    def diagnostic_case(*args, **kwargs):
        observations.clear()
        result = original_case(*args, **kwargs)
        result["coupling_diagnostics"] = list(observations)
        return result

    model.grid_pka_kernel = diagnostic_grid
    runner._run_case = diagnostic_case
    runner.main()


if __name__ == "__main__":
    main()
