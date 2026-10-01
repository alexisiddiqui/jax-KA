#!/usr/bin/env python3
"""Decompose grid validity for one FoldBench array case."""
from dataclasses import replace

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
        for steps in (128, 512):
            cfg = replace(config, steps=steps)
            grid = jax.device_get(original_grid(
                arrays, probabilities, ph, config=cfg))
            curves = jax.device_get(model.curve_kernel(
                arrays, probabilities, ph, config=cfg))
            if baseline is None:
                baseline = grid
            native = np.asarray(grid.probability) > 0
            y = np.asarray(curves.protonated)
            transitions = (y[:-1] - 0.5) * (y[1:] - 0.5) <= 0
            crossings = transitions.sum(axis=0)
            safe = np.asarray(grid.slope) < -cfg.slope_min
            unique = crossings == 1
            native_converged = bool(np.all(
                np.asarray(curves.weighted_residual) < cfg.residual_tolerance))
            usable = native & np.asarray(grid.bracketed) & safe & unique
            per_group = {}
            for group_index, group in enumerate(GROUPS):
                selected = native[:, group_index]
                per_group[group] = {
                    "sites": int(selected.sum()),
                    "current_invalid": int((selected & ~np.asarray(grid.valid)[:, group_index]).sum()),
                    "nonmonotone": int((selected & ~np.asarray(grid.sampled_monotone)[:, group_index]).sum()),
                    "not_unique_crossing": int((selected & ~unique[:, group_index]).sum()),
                    "usable_unique_crossing": int((selected & usable[:, group_index]).sum()),
                }
            observations.append({
                "steps": steps,
                "native_sites": int(native.sum()),
                "current_invalid_native_sites": int((native & ~np.asarray(grid.valid)).sum()),
                "unbracketed_native_sites": int((native & ~np.asarray(grid.bracketed)).sum()),
                "unsafe_slope_native_sites": int((native & ~safe).sum()),
                "nonmonotone_native_sites": int((native & ~np.asarray(grid.sampled_monotone)).sum()),
                "zero_crossing_native_sites": int((native & (crossings == 0)).sum()),
                "multiple_crossing_native_sites": int((native & (crossings > 1)).sum()),
                "unique_crossing_native_sites": int((native & unique).sum()),
                "usable_unique_crossing_sites": int(usable.sum()),
                "global_converged": bool(np.all(curves.converged)),
                "native_converged": native_converged,
                "max_global_residual": float(np.max(curves.residual)),
                "max_native_residual": float(np.max(curves.weighted_residual)),
                "per_group": per_group,
            })
            print(f"Validity diagnostic: steps={steps} "
                  f"{observations[-1]}", flush=True)
        return baseline

    def diagnostic_case(*args, **kwargs):
        observations.clear()
        result = original_case(*args, **kwargs)
        result["validity_diagnostics"] = list(observations)
        return result

    model.grid_pka_kernel = diagnostic_grid
    runner._run_case = diagnostic_case
    runner.main()


if __name__ == "__main__":
    main()
