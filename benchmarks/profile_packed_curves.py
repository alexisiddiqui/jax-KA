"""CPU warm curve/gradient benchmark; no production solver changes."""
import argparse
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time
import jax
import jax.numpy as jnp
import numpy as np
from biotite.structure.io import pdbx
from jaxpropka import ModelConfig
from jaxpropka.batching import pack_inputs
from jaxpropka.geometry import build_candidates
from jaxpropka.model import curve_kernel, one_hot
from jaxpropka.precompute import build_cache
from jaxpropka.topology import load_topology
from packed_curve_kernel import pack_edges, packed_curve_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--wide-native-check', action='store_true',
        help='Also compare native curves across the original wide pH grid.')
    args = parser.parse_args()
    started = time.perf_counter()
    jax.config.update('jax_enable_compilation_cache', False)
    original = json.loads(args.report.read_text()); case = original['cases'][0]
    root = Path(__file__).resolve().parents[1]
    result = dict(status='running', interface_id=case['interface_id'],
        hostname=os.uname().nodename, jax_version=jax.__version__,
        cpu_affinity=len(os.sched_getaffinity(0)), records=[], comparisons=[],
        scope='single complex charge integral, not full bound/free loss; float32',
        source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), root/'benchmarks/packed_curve_kernel.py', root/'src/jaxpropka/model.py']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def checkpoint(stage):
        result.update(stage=stage, elapsed_seconds=time.perf_counter()-started)
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result, indent=2)+'\n'); tmp.replace(args.output)
        print(case['interface_id'], stage, round(result['elapsed_seconds'], 2), flush=True)
    checkpoint('building_cache')
    with tarfile.open(root/'ground_truth_1522.tar') as archive:
        member = next(m for m in archive.getmembers() if Path(m.name).name == case['pdb_id']+'.cif')
        raw = archive.extractfile(member).read()
    assert hashlib.sha256(raw).hexdigest() == case['source_sha256']
    atoms = pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(raw.decode())), model=1,
        altloc='occupancy', use_author_fields=False, include_bonds=True)
    topology = load_topology(atoms[np.isin(atoms.chain_id, case['partner_chains'])],
        gap_policy='free', freeze_disulfides=True, ignore_nonprotein=True)
    topology.metadata.update(source_name=case['pdb_id']+'.cif', source_sha256=case['source_sha256'],
        foldbench_pdb_id=case['pdb_id'], foldbench_label_asym_id=case['chain_id'])
    cache = build_cache(topology, build_candidates(topology, missing_sidechain='template'))
    assert cache.fingerprint() == case['cache_fingerprint']
    arrays, native, _ = pack_inputs(cache, one_hot(cache.native_index), bucket_multiple=(1,1,1))
    edges = pack_edges(arrays)
    result.update(n_residues=cache.n_residues, cache_bytes=cache.nbytes,
        padded_pair_entries=int(arrays['pair_mask'].size), active_pair_entries=len(edges[0]))
    arrays, native, edges = jax.device_put((arrays, native, edges))
    jax.block_until_ready((arrays, native, edges)); checkpoint('cache_ready')
    if args.wide_native_check:
        result['wide_native_comparisons'] = []
        for steps in (128, 512):
            cfg = replace(ModelConfig(**case['midpoint_model_config']), steps=steps)
            ph = jnp.linspace(cfg.ph_min, cfg.ph_max, 145, dtype=native.dtype)
            a = jax.device_get(curve_kernel(arrays, native, ph, config=cfg))
            b = jax.device_get(packed_curve_kernel(arrays, native, ph, edges, config=cfg))
            result['wide_native_comparisons'].append(dict(steps=steps, points=145,
                ph_min=cfg.ph_min, ph_max=cfg.ph_max,
                occupancy_max_abs=float(np.max(np.abs(a.protonated-b.protonated))),
                charge_max_abs=float(np.max(np.abs(a.total_charge-b.total_charge))),
                dense_max_residual=float(np.max(a.residual)),
                packed_max_residual=float(np.max(b.residual)),
                dense_max_weighted_residual=float(np.max(a.weighted_residual)),
                packed_max_weighted_residual=float(np.max(b.weighted_residual)),
                convergence_flags_equal=bool(np.array_equal(a.converged, b.converged)),
                parity_pass=bool(np.allclose(a.protonated, b.protonated, atol=2e-5, rtol=2e-5)
                    and np.allclose(a.total_charge, b.total_charge, atol=2e-4, rtol=2e-5)
                    and np.array_equal(a.converged, b.converged))))
            checkpoint(f'wide_native_{steps}')
        del a, b
    for steps in (128, 512):
        cfg = replace(ModelConfig(**case['midpoint_model_config']), steps=steps)
        for points in (2, 15):
            ph = jnp.linspace(6., 7.4, points, dtype=jnp.float32)
            def dense(a, p, h, e):
                return curve_kernel(a, p, h, config=cfg)
            def packed(a, p, h, e):
                return packed_curve_kernel(a, p, h, e, config=cfg)
            outputs = {}
            for name, kernel in [('dense', dense), ('packed', packed)]:
                def objective(a, logits, h, e):
                    out = kernel(a, jax.nn.softmax(logits, axis=-1), h, e)
                    return jnp.trapezoid(out.total_charge, h), (out.residual, out.weighted_residual)
                operations = [('curves', jax.jit(kernel)),
                    ('value_grad', jax.jit(jax.value_and_grad(objective, argnums=1, has_aux=True)))]
                for operation, function in operations:
                    executable = None
                    for mixture in (.05, .001):
                        p = (1-mixture)*native+mixture/20
                        x = p if operation == 'curves' else jnp.log(p)
                        jax.block_until_ready(x)
                        compile_seconds = 0.
                        if executable is None:
                            start = time.perf_counter()
                            executable = function.lower(arrays, x, ph, edges).compile()
                            compile_seconds = time.perf_counter()-start
                        times = []
                        for repeat in range(3):
                            start = time.perf_counter()
                            out = jax.device_get(executable(arrays, x, ph, edges))
                            times.append(time.perf_counter()-start)
                        outputs[name, operation, mixture] = out
                        residual = out.residual if operation == 'curves' else out[0][1][0]
                        result['records'].append(dict(kernel=name, operation=operation, steps=steps,
                            points=points, mixture=mixture, trace_compile_seconds=compile_seconds,
                            first_seconds=times[0], warm_seconds=times[1:],
                            warm_median_seconds=float(np.median(times[1:])),
                            max_residual=float(np.max(residual)),
                            converged=bool(np.all(residual < cfg.residual_tolerance))))
                        checkpoint(f'{name}_{operation}_{steps}_{points}_{mixture}')
            for mixture in (.05, .001):
                a, b = [outputs[name, 'curves', mixture] for name in ('dense', 'packed')]
                ga, gb = [outputs[name, 'value_grad', mixture][1] for name in ('dense', 'packed')]
                comparison = dict(steps=steps, points=points, mixture=mixture,
                    occupancy_max_abs=float(np.max(np.abs(a.protonated-b.protonated))),
                    charge_max_abs=float(np.max(np.abs(a.total_charge-b.total_charge))),
                    gradient_max_abs=float(np.max(np.abs(ga-gb))),
                    gradient_relative_l2=float(np.linalg.norm(ga-gb)/max(np.linalg.norm(ga), 1e-20)),
                    convergence_flags_equal=bool(np.array_equal(a.converged, b.converged)))
                comparison['parity_pass'] = bool(np.allclose(a.protonated, b.protonated, atol=2e-5, rtol=2e-5)
                    and np.allclose(a.total_charge, b.total_charge, atol=2e-4, rtol=2e-5)
                    and np.allclose(ga, gb, atol=2e-5, rtol=2e-4)
                    and comparison['convergence_flags_equal'])
                result['comparisons'].append(comparison)
            checkpoint('compared')
    result['status'] = 'complete'; checkpoint('complete')


if __name__ == '__main__':
    main()
