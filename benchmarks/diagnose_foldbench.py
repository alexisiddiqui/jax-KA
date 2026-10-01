#!/usr/bin/env python3
"""Run the normal regression plus grid-failure diagnostics on the same cache.

Accepts regress_foldbench.py arguments. The recorded benchmark outcome always
uses the original configuration; variants are diagnostic observations only.
"""
from dataclasses import replace
import time

import jax
import numpy as np

import regress_foldbench as runner
from jaxpropka import GROUPS
from jaxpropka import model


def main():
    original_grid = model.grid_pka_kernel
    original_case = runner._run_case
    observations = []

    def diagnostic_grid(arrays, probabilities, ph, *, config):
        baseline = None
        for cfg in (config, replace(config, steps=512), replace(config, steps=2048),
                    replace(config, steps=512, damping=0.15)):
            started = time.perf_counter()
            grid = jax.device_get(original_grid(arrays, probabilities, ph, config=cfg))
            curves = jax.device_get(model.curve_kernel(arrays, probabilities, ph, config=cfg))
            if baseline is None:
                baseline = grid
            native = np.asarray(grid.probability) > 0
            rises = np.max(np.diff(curves.protonated, axis=0), axis=0)
            invalid = native & ~grid.valid
            examples = []
            for i, g in np.argwhere(invalid)[:12]:
                examples.append({
                    "residue_index": int(i), "group": GROUPS[g],
                    "bracketed": bool(grid.bracketed[i, g]),
                    "sampled_monotone": bool(grid.sampled_monotone[i, g]),
                    "slope": float(grid.slope[i, g]),
                    "max_occupancy_rise": float(rises[i, g]),
                    "endpoint_occupancies": [float(curves.protonated[0, i, g]),
                                               float(curves.protonated[-1, i, g])],
                })
            observations.append({
                "steps": cfg.steps, "damping": cfg.damping,
                "native_sites": int(native.sum()),
                "invalid_native_sites": int(invalid.sum()),
                "unbracketed_native_sites": int((native & ~grid.bracketed).sum()),
                "nonmonotone_native_sites": int((native & ~grid.sampled_monotone).sum()),
                "unsafe_slope_native_sites": int((native & ~(grid.slope < -cfg.slope_min)).sum()),
                "unconverged_ph_points": int((~curves.converged).sum()),
                "unconverged_native_ph_points": int((curves.weighted_residual >= cfg.residual_tolerance).sum()),
                "max_residual": float(np.max(curves.residual)),
                "max_native_residual": float(np.max(curves.weighted_residual)),
                "max_native_occupancy_rise": float(np.max(rises[native], initial=0)),
                "invalid_examples": examples,
                "seconds": time.perf_counter() - started,
            })
            print(f"Grid diagnostic: {observations[-1]}", flush=True)
        return baseline

    def diagnostic_case(*args, **kwargs):
        observations.clear()
        result = original_case(*args, **kwargs)
        result["grid_diagnostics"] = list(observations)
        return result

    model.grid_pka_kernel = diagnostic_grid
    runner._run_case = diagnostic_case
    runner.main()


if __name__ == "__main__":
    main()
