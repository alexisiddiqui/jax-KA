#!/usr/bin/env python3
"""Experimental FoldBench policy: accept a unique native-site grid crossing."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

import regress_foldbench as runner
from jaxpropka import model


def main():
    original_grid = model.grid_pka_kernel
    original_curve = model.curve_kernel
    original_case = runner._run_case

    def retry_curve(arrays, probabilities, ph, *, config):
        return original_curve(arrays, probabilities, ph,
                              config=replace(config, steps=max(config.steps, 512)))

    def unique_grid(arrays, probabilities, ph, *, config):
        cfg = replace(config, steps=max(config.steps, 512))
        grid = original_grid(arrays, probabilities, ph, config=cfg)
        curves = original_curve(arrays, probabilities, ph, config=cfg)
        y = curves.protonated
        crossing = (((y[:-1] > 0.5) & (y[1:] <= 0.5)) |
                    ((y[:-1] < 0.5) & (y[1:] >= 0.5)))
        unique = jnp.sum(crossing, axis=0) == 1
        safe = grid.slope < -cfg.slope_min
        native_converged = jnp.all(curves.weighted_residual < cfg.residual_tolerance)
        valid = grid.bracketed & safe & unique & native_converged
        return grid._replace(valid=valid)

    def experimental_case(*args, **kwargs):
        result = original_case(*args, **kwargs)
        result["experimental_grid_policy"] = {
            "steps": 512,
            "validity": "bracketed, safe local slope, exactly one half-occupancy crossing, native-weighted convergence",
            "sampled_monotonicity_required": False,
        }
        return result

    model.curve_kernel = retry_curve
    model.grid_pka_kernel = unique_grid
    runner._run_case = experimental_case
    runner.main()


if __name__ == "__main__":
    main()
