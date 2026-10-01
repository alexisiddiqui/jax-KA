#!/usr/bin/env python3
"""Inspect coupled-site branch switching for one FoldBench structure."""
import argparse
import csv
import io
import json
from pathlib import Path
import tarfile

import jax
import jax.numpy as jnp
import numpy as np
from biotite.structure.io import pdbx

from jaxpropka import GROUPS, ModelConfig
from jaxpropka.batching import pack_inputs
from jaxpropka.geometry import build_candidates
from jaxpropka.model import _local_terms, curve_kernel, one_hot
from jaxpropka.parameters import GROUP_AA
from jaxpropka.precompute import build_cache
from jaxpropka.topology import load_topology


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = list(csv.DictReader(args.manifest.open()))[args.case_index]
    with tarfile.open(args.archive) as archive:
        member = next(m for m in archive.getmembers()
                      if Path(m.name).name == row["pdb_id"] + ".cif")
        source = archive.extractfile(member).read()
    cif = pdbx.CIFFile.read(io.StringIO(source.decode()))
    atoms = pdbx.get_structure(cif, model=1, altloc="occupancy",
                               use_author_fields=False, include_bonds=True)
    atoms = atoms[np.asarray(atoms.chain_id) == row["chain_id"]]
    topology = load_topology(atoms, gap_policy="free", freeze_disulfides=True,
                             ignore_nonprotein=True)
    candidates = build_candidates(topology, missing_sidechain="template")
    cache = build_cache(topology, candidates)
    arrays, probabilities, n = pack_inputs(
        cache, one_hot(cache.native_index), bucket_multiple=(1, 1, 1))
    cfg = ModelConfig(steps=512)
    grid = np.linspace(cfg.ph_min, cfg.ph_max, 145, dtype=np.float32)
    terms = jax.device_get(_local_terms(arrays, probabilities, cfg))
    independent = jax.device_get(curve_kernel(
        arrays, probabilities, grid, config=cfg)).protonated

    d = jax.tree.map(jnp.asarray, arrays)
    t = jax.tree.map(jnp.asarray, terms)
    log10 = jnp.log(jnp.asarray(10, dtype=probabilities.dtype))

    def target(h, ph):
        field = t.field0 + jnp.einsum(
            "nkgt,nkt->ng", t.coupling, h[d["neighbors"]])
        return jnp.where(d["group_mask"],
            jax.nn.sigmoid(log10 * (t.intrinsic - ph - field)), 0)

    def converge(h, ph):
        return jax.lax.fori_loop(0, cfg.steps, lambda _, old:
            old + cfg.damping * (target(old, ph) - old), h)

    @jax.jit
    def continuation(ph_values):
        initial = jnp.where(d["group_mask"], jax.nn.sigmoid(
            log10 * (t.intrinsic - ph_values[0] - t.field0)), 0)
        def step(old, ph):
            state = converge(old, ph)
            return state, state
        _, states = jax.lax.scan(step, initial, ph_values)
        return states

    forward = np.asarray(continuation(jnp.asarray(grid)))
    reverse = np.asarray(continuation(jnp.asarray(grid[::-1])))[::-1]
    expected = [(i, int(g)) for i, aa in enumerate(cache.native_index)
                for g in np.flatnonzero(GROUP_AA == aa) if cache.group_mask[i, g]]
    expected += [(i, g) for i in range(n) for g in (7, 8) if cache.group_mask[i, g]]
    native_mask = np.zeros((n, 9), bool)
    for site in expected:
        native_mask[site] = True
    monotone = np.all(np.diff(independent, axis=0) <= 1e-6, axis=0)
    invalid = [site for site in expected if not monotone[site]]
    details = []
    for i, g in invalid:
        occupancy = independent[:, i, g]
        changes = np.diff(independent[:, i, g])
        q = int(np.argmax(changes))
        crossing_intervals = np.flatnonzero(
            (occupancy[:-1] - 0.5) * (occupancy[1:] - 0.5) <= 0)
        partners = []
        for k, j in enumerate(cache.neighbors[i]):
            for tg in range(9):
                if not native_mask[j, tg]:
                    continue
                coupling = float(terms.coupling[i, k, g, tg])
                if coupling == 0:
                    continue
                partners.append({
                    "residue": str(cache.keys[j]), "group": GROUPS[tg],
                    "coupling": coupling,
                    "occupancy_change_at_jump": float(
                        independent[q+1, j, tg] - independent[q, j, tg]),
                })
        partners.sort(key=lambda x: abs(x["coupling"]), reverse=True)
        details.append({
            "residue": str(cache.keys[i]), "group": GROUPS[g],
            "burial": float(terms.burial[i, g]),
            "intrinsic_pka": float(terms.intrinsic[i, g]),
            "largest_upward_jump": float(changes[q]),
            "occupancy_around_jump": [float(occupancy[q]), float(occupancy[q+1])],
            "jump_interval": [float(grid[q]), float(grid[q+1])],
            "half_occupancy_crossing_intervals": [
                [float(grid[x]), float(grid[x+1])] for x in crossing_intervals],
            "forward_largest_rise": float(np.max(np.diff(forward[:, i, g]))),
            "reverse_largest_rise": float(np.max(np.diff(reverse[:, i, g]))),
            "max_forward_reverse_difference": float(
                np.max(np.abs(forward[:, i, g] - reverse[:, i, g]))),
            "top_partners": partners[:8],
        })
    native_total = lambda curve: curve[:, native_mask].sum(axis=1)
    result = {
        "pdb_id": row["pdb_id"], "n_residues": n,
        "invalid_native_sites": len(invalid),
        "max_independent_native_rise": float(
            np.max(np.diff(independent, axis=0)[:, native_mask])),
        "max_forward_native_rise": float(np.max(np.diff(forward, axis=0)[:, native_mask])),
        "max_reverse_native_rise": float(np.max(np.diff(reverse, axis=0)[:, native_mask])),
        "max_forward_reverse_difference": float(np.max(np.abs(forward[:, native_mask] - reverse[:, native_mask]))),
        "total_native_protonation_monotone": {
            "independent": bool(np.all(np.diff(native_total(independent)) <= 1e-6)),
            "forward": bool(np.all(np.diff(native_total(forward)) <= 1e-6)),
            "reverse": bool(np.all(np.diff(native_total(reverse)) <= 1e-6)),
        },
        "sites": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
