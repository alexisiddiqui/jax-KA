"""Benchmark-only site policy; does not change model kernels or chemistry."""
from dataclasses import asdict, replace
import time

import numpy as np


POLICY = {
    "version": "site-report-v1",
    "retry_steps": 512,
    "refinement_factor": 4,
    "strict": "sampled monotonicity, bracket, unique safe crossing, all-group convergence on every evaluated grid",
    "flagged": "nonmonotonic on either grid; one downward crossing on coarse and refined grids; native convergence",
    "uniqueness": "sampled only, not a mathematical uniqueness guarantee",
    "context": "observed model 1 as deposited, before chain filtering; not biological-assembly expansion",
    "context_thresholds_angstrom": {"metal": 3.0, "nonprotein": 4.0, "other_chain": 4.0},
    "exposure": "isolated observed protein chain, ProtOr SASA, probe 1.4 A, 1000 points",
    "comparison_input": "both methods use native candidates from the same canonical protein-only topology; metals, ligands, waters and unselected chains excluded from both",
    "reference_export_audit": "atom identities, elements and coordinates checked against native candidates; PDB rounding tolerance 0.001 A per coordinate",
}


def crossing_info(ph, occupancy):
    """Count sign changes once at exact grid nodes; reject touches/plateaus."""
    ph, y = np.asarray(ph), np.asarray(occupancy) - 0.5
    if not np.all(np.isfinite(y)):
        return {"count": 0, "ambiguous": True, "value": None, "slope": None}
    zero = y == 0
    ambiguous = bool(zero[0] or zero[-1] or np.any(zero[:-1] & zero[1:]))
    nonzero = np.flatnonzero(~zero)
    crossings = []
    for left, right in zip(nonzero[:-1], nonzero[1:]):
        if np.sign(y[left]) == np.sign(y[right]):
            ambiguous |= right > left + 1  # A touch, not a crossing.
            continue
        slope = (y[right] - y[left]) / (ph[right] - ph[left])
        value = (ph[left + 1] if right == left + 2 else
                 ph[left] - y[left] / slope)
        crossings.append((float(value), float(slope)))
    unique = len(crossings) == 1 and not ambiguous
    return {"count": len(crossings), "ambiguous": bool(ambiguous),
            "value": crossings[0][0] if unique else None,
            "slope": crossings[0][1] if unique else None}


def classify(ph, occupancy, *, global_converged, native_converged, slope_min):
    info = crossing_info(ph, occupancy)
    y = np.asarray(occupancy)
    bracketed = bool(y[0] >= .5 and y[-1] <= .5)
    monotone = bool(np.all(np.diff(y) <= 1e-6))
    safe = info["slope"] is not None and info["slope"] < -slope_min
    unique = info["count"] == 1 and not info["ambiguous"]
    reasons = []
    for bad, reason in [(not bracketed, "unbracketed"), (not unique, "ambiguous_or_multiple_crossings"),
                        (not safe, "unsafe_slope"), (not monotone, "nonmonotonic"),
                        (not global_converged, "all_group_nonconvergence"),
                        (not native_converged, "native_nonconvergence")]:
        if bad:
            reasons.append(reason)
    return {**info, "bracketed": bracketed, "sampled_monotone": monotone,
            "strict": bool(bracketed and safe and unique and monotone and global_converged),
            "flagged_candidate": bool(bracketed and safe and unique and not monotone and native_converged),
            "reasons": reasons}


def select_tier(coarse, refined):
    if refined is None:
        return "strict" if coarse["strict"] else "invalid"
    if coarse["strict"] and refined["strict"]:
        return "strict"
    if ((coarse["strict"] or coarse["flagged_candidate"]) and
            (refined["strict"] or refined["flagged_candidate"])):
        return "flagged_unique_sampled"
    return "invalid"


def context_rows(full_atoms, topology, expected):
    from biotite.structure import sasa
    from scipy.spatial import cKDTree
    from analyze_foldbench_context import CANONICAL, WATER, SITE_ATOMS

    # Explicit elements avoid classifying unknown/nonmetal elements as metals.
    metals = set("LI BE NA MG AL K CA SC TI V CR MN FE CO NI CU ZN GA RB SR Y ZR NB MO TC RU RH PD AG CD IN SN CS BA LA CE PR ND PM SM EU GD TB DY HO ER TM YB LU HF TA W RE OS IR PT AU HG TL PB BI FR RA AC TH PA U NP PU AM CM BK CF ES FM MD NO LR".split())
    names = np.asarray(full_atoms.res_name, str)
    elements = np.char.upper(np.asarray(full_atoms.element, str))
    chains = np.asarray(full_atoms.chain_id, str)
    selected_chains = {key.chain for key in topology.keys}
    masks = {"metal": np.isin(elements, list(metals)),
             "nonprotein": ~np.isin(names, list(CANONICAL | WATER)),
             "other_chain": np.isin(names, list(CANONICAL)) & ~np.isin(chains, list(selected_chains))}
    trees = {name: (cKDTree(full_atoms.coord[mask]), np.flatnonzero(mask))
             for name, mask in masks.items() if np.any(mask)}
    area = sasa(topology.atoms, probe_radius=1.4, point_number=1000)
    rows = []
    for index, group in expected:
        residue = topology.residue(index)
        required = SITE_ATOMS[group]
        mask = np.isin(residue.atom_name, required)
        complete = set(required) <= set(map(str, residue.atom_name))
        start, stop = topology.starts[index:index + 2]
        functional_area = float(area[start:stop][mask].sum()) if complete else None
        if functional_area is not None and not np.isfinite(functional_area):
            functional_area = None
        exposure = ("unknown" if functional_area is None else "buried_lt1" if functional_area < 1
                    else "low_1to5" if functional_area < 5 else "intermediate_5to20" if functional_area < 20
                    else "exposed_ge20")
        context = {}
        for name in masks:
            nearest = None
            if np.any(mask) and name in trees:
                tree, indices = trees[name]
                distances, neighbors = tree.query(residue.coord[mask])
                closest = int(np.argmin(distances))
                atom = full_atoms[indices[int(neighbors[closest])]]
                nearest = {"distance": float(distances[closest]), "chain": str(atom.chain_id),
                           "number": int(atom.res_id), "insertion": str(atom.ins_code),
                           "residue_name": str(atom.res_name), "atom_name": str(atom.atom_name),
                           "element": str(atom.element)}
            context[name] = nearest
        nearby = [name for name, limit in POLICY["context_thresholds_angstrom"].items()
                  if context[name] is not None and context[name]["distance"] <= limit]
        rows.append({"functional_atoms_complete": complete, "functional_sasa": functional_area,
                     "exposure": exposure, "context": context,
                     "context_cohort": "+".join(nearby) if nearby else "no_nearby_context" if complete else "unknown"})
    return rows


def audit_reference_input(path, topology, candidates, mapping):
    """Fail closed if PROPKA's exported atoms differ from JAX native candidates."""
    from biotite.structure.io.pdb import PDBFile
    from jaxpropka.parameters import THREE

    atoms = PDBFile.read(path).get_structure(model=1)
    expected = {}
    for i, key in enumerate(topology.keys):
        candidate = candidates.residues[i][int(topology.native_index[i])]
        for name, xyz in candidate.coordinates.items():
            expected[key, name] = (THREE[int(topology.native_index[i])],
                                   candidate.elements.get(name, "O" if name == "OXT" else name[0]).upper(), xyz)
    seen = set()
    max_error = 0.0
    for atom in atoms:
        key = mapping.get((str(atom.chain_id), int(atom.res_id)))
        identity = (key, str(atom.atom_name))
        if identity not in expected or identity in seen:
            raise ValueError("reference export contains unexpected or duplicate atoms")
        residue, element, xyz = expected[identity]
        if str(atom.res_name) != residue or str(atom.element).upper() != element:
            raise ValueError("reference export atom identity/element mismatch")
        if element not in {"C", "N", "O", "S"}:
            raise ValueError(f"non-protein element in native/reference candidates: {element}")
        error = float(np.max(np.abs(np.asarray(atom.coord) - xyz)))
        if not np.isfinite(error) or error > .001:
            raise ValueError("reference export coordinates differ beyond PDB rounding")
        max_error = max(max_error, error)
        seen.add(identity)
    if seen != set(expected):
        raise ValueError("reference export is missing native candidate atoms")
    return {"passed": True, "native_candidate_atoms": len(expected),
            "reference_atoms": len(atoms), "metal_atoms": 0,
            "max_coordinate_rounding_error_angstrom": max_error}


def evaluate(arrays, probabilities, cache, topology, candidates, full_atoms,
             expected, config, grid_points):
    import jax
    from pathlib import Path
    import tempfile
    from jaxpropka import GROUPS
    from jaxpropka.model import curve_kernel
    from jaxpropka.reference import run_reference, write_reference_structure, reference_charge

    attempts = []

    def solve(ph, cfg, label):
        started = time.perf_counter()
        out = jax.device_get(curve_kernel(arrays, probabilities, ph, config=cfg))
        attempts.append({"grid": label, "points": len(ph), "steps": cfg.steps,
                         "elapsed_seconds": time.perf_counter() - started,
                         "max_all_group_residual": float(np.max(out.residual)),
                         "max_native_residual": float(np.max(out.weighted_residual)),
                         "all_group_converged": bool(np.all(out.converged)),
                         "native_converged": bool(np.all(out.weighted_residual < cfg.residual_tolerance))})
        return out

    def diagnostics(ph, out):
        return [classify(ph, out.protonated[:, i, GROUPS.index(group)],
                        global_converged=bool(np.all(out.converged)),
                        native_converged=bool(np.all(out.weighted_residual < config.residual_tolerance)),
                        slope_min=config.slope_min) for i, group in expected]

    coarse_ph = np.linspace(config.ph_min, config.ph_max, grid_points, dtype=np.float32)
    coarse = solve(coarse_ph, config, "coarse")
    if not np.all(coarse.converged) and config.steps < POLICY["retry_steps"]:
        config = replace(config, steps=POLICY["retry_steps"])
        coarse = solve(coarse_ph, config, "coarse")
    coarse_info = diagnostics(coarse_ph, coarse)
    fine_info = None
    if any(row["flagged_candidate"] for row in coarse_info):
        fine_ph = np.linspace(config.ph_min, config.ph_max,
                              (grid_points - 1) * POLICY["refinement_factor"] + 1, dtype=np.float32)
        fine = solve(fine_ph, config, "refined")
        if not np.all(fine.converged) and config.steps < POLICY["retry_steps"]:
            config = replace(config, steps=POLICY["retry_steps"])
            coarse = solve(coarse_ph, config, "coarse")
            coarse_info = diagnostics(coarse_ph, coarse)
            fine = solve(fine_ph, config, "refined")
        fine_info = diagnostics(fine_ph, fine)

    rows = []
    for n, (index, group) in enumerate(expected):
        coarse_row = coarse_info[n]
        refined = fine_info[n] if fine_info is not None else None
        tier = select_tier(coarse_row, refined)
        chosen = refined if tier == "flagged_unique_sampled" else coarse_row
        rows.append({"residue": asdict(cache.keys[index]), "group": group, "tier": tier,
                     "coarse": coarse_row, "refined": refined,
                     "surrogate_pka": chosen["value"] if tier != "invalid" else None,
                     "refinement_shift": (refined["value"] - coarse_row["value"]
                         if refined and refined["value"] is not None and coarse_row["value"] is not None else None),
                     "reference_pka": None, "delta": None, "reference_missing": True})
    # Preserve solver diagnostics even if structural annotation/reference fails.
    annotation_error = None
    try:
        for row, annotation in zip(rows, context_rows(full_atoms, topology, expected)):
            row.update(annotation)
    except Exception as error:
        annotation_error = repr(error)
    reference_error = None
    reference_input_audit = None
    reference = None
    matched_sites, matched_indices = [], []
    try:
        with tempfile.TemporaryDirectory(prefix="jaxpropka-site-reference-") as directory:
            path = Path(directory) / "structure.pdb"
            mapping = write_reference_structure(topology, candidates, path)
            reference_input_audit = audit_reference_input(path, topology, candidates, mapping)
            reference = run_reference(path, backend="modern", mapping=mapping)
        lookup = {(site.key, site.group): site for site in reference.sites}
        for row, (index, group) in zip(rows, expected):
            site = lookup.get((cache.keys[index], group))
            if site is not None:
                row.update(reference_pka=float(site.pka), reference_missing=False)
                if row["tier"] != "invalid":
                    row["delta"] = row["surrogate_pka"] - float(site.pka)
                    matched_sites.append(site)
                    matched_indices.append((index, GROUPS.index(group)))
    except Exception as error:
        reference_error = repr(error)

    charge_report = {"scope": "matched eligible sites only; not whole-protein total", "sites": len(matched_sites)}
    if matched_sites:
        ph = np.arange(15, dtype=np.float32)
        curves = solve(ph, config, "charge")
        if not np.all(curves.converged) and config.steps < POLICY["retry_steps"]:
            curves = solve(ph, replace(config, steps=POLICY["retry_steps"]), "charge")
        charge_report["native_converged"] = bool(np.all(curves.weighted_residual < config.residual_tolerance))
        charge_report["all_group_converged"] = bool(np.all(curves.converged))
        if charge_report["native_converged"]:
            actual = np.stack([curves.site_charge[:, i, g] for i, g in matched_indices], axis=-1)
            difference = actual - reference_charge(matched_sites, ph)
            charge_report.update(site_charge_rmse=float(np.sqrt(np.mean(difference ** 2))),
                                 matched_sum_charge_rmse=float(np.sqrt(np.mean(difference.sum(-1) ** 2))))
    compared = sum(row["delta"] is not None for row in rows)
    return {"status": "passed" if compared == len(rows) else "partial" if compared else "failed",
            "stage": "site_report", "site_policy": POLICY, "sites": rows, "n_sites": len(rows),
            "compared_sites": compared, "solver_attempts": attempts,
            "midpoint_model_config": asdict(config),
            "midpoint_method": {"kind": "sampled_crossing_linear_interpolation", "minimum": config.ph_min,
                                "maximum": config.ph_max, "coarse_points": grid_points,
                                "refined_points": (grid_points - 1) * POLICY["refinement_factor"] + 1 if fine_info else None},
            "reference": reference.provenance if reference else None,
            "reference_error": reference_error, "annotation_error": annotation_error,
            "reference_input_audit": reference_input_audit,
            "charge_metrics": charge_report, "cache_fingerprint": cache.fingerprint()}


def summarize(cases):
    rows = [site for case in cases for site in case.get("sites", [])]

    def stats(sites):
        delta = np.asarray([s["delta"] for s in sites if s.get("delta") is not None], dtype=float)
        return {"expected_sites": len(sites), "compared_sites": len(delta),
                "coverage": len(delta) / len(sites) if sites else None,
                "mae": float(np.abs(delta).mean()) if len(delta) else None,
                "rmse": float(np.sqrt(np.mean(delta ** 2))) if len(delta) else None,
                "max_abs": float(np.abs(delta).max()) if len(delta) else None}

    strata = {}
    for field in ("tier", "group", "exposure", "context_cohort"):
        strata[field] = {value: stats([s for s in rows if s.get(field, "unknown") == value])
                         for value in sorted({s.get(field, "unknown") for s in rows})}
    return {"overall": stats(rows), "strata": strata,
            "coverage_denominator": "enumerated native sites; topology/cache failures have unknown site counts",
            "cases_without_site_inventory": sum("sites" not in case for case in cases),
            "reference_missing_sites": sum(s.get("reference_missing", False) for s in rows)}
