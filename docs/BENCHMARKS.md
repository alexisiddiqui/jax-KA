# Executed CPU benchmarks

## Initial synthetic microbenchmarks

**This section uses synthetic numerical graphs, not molecular structures or a PROPKA performance comparison.** Real-structure packed-kernel results are recorded below.

Environment: Python 3.13.5, JAX/JAXLIB 0.9.0.1, float32 sequence inputs, CPU only. CPU model reported by the host: AMD EPYC 9V74 80-Core Processor.

Each graph has two chains, environment K=8 and interaction K=9 (including same-residue terminal edges). Actual candidate-union graphs in a protein may have much larger K. No GPU or external PROPKA run was performed.

| Operation | N=16 median ms | N=64 median ms |
|---|---:|---:|
| Single-pH residue charge | 0.377 | 1.169 |
| Charge loss + sequence gradient | 1.015 | 3.097 |
| 29-pH titration curves | 2.159 | 8.695 |
| Curve loss + sequence gradient | 10.034 | 36.857 |
| All-channel grid pKas (73 pHs) | 5.221 | 14.930 |
| Grid-pKa loss + sequence gradient | 39.009 | 65.467 |
| Two direct midpoint pKas | 28.356 | 108.266 |
| Two direct pKas + sequence gradient | 29.898 | 111.861 |
| Single-pH charge, sequence batch of 8 | 2.673 | 8.205 |
| All N×9 direct midpoint roots | 1085.276 | Not measured |

Structural array storage: 0.443 MB at N=16 and 1.774 MB at N=64. This does not include compiler state or reverse-mode activation memory.

The occupancy solver uses 64 damped iterations. Direct roots use 28 bisection iterations, with a root mapping batch size of four. The grid-midpoint method uses 73 shared pHs from -2 to 16. Benchmarks use uniform soft sequence probabilities, not a hard native sequence.

The benchmark synchronizes device execution, warms each compiled function three times and records ten timed repetitions for ordinary/grid workloads, five for selected direct roots and three for the all-direct-root workload. Compilation is measured separately. Raw JSON includes p10/p90, package versions, graph sizes, configuration and compiler memory estimates.

Forward-plus-gradient rows are separate compiled scalar-loss functions; dead-code elimination and measurement noise mean their timing need not exceed the full diagnostic-returning forward function in every small case. These host timings are not controlled hardware-affinity benchmarks.

All sampled curves converged and both selected direct roots were valid in these runs. Grid validity passed for 116/116 active channels at N=16 and 452/452 at N=64. The maximum grid-versus-direct difference over the two selected HIS channels was 0.000171661 and 0.000675201 pKa units, respectively. This is not a global interpolation-error bound or a protein accuracy estimate.

## Interpretation

Direct all-site midpoints are expensive because each query needs repeated full-environment solves at its own pH. A shared grid amortizes those solves across all positions, but gives interpolated, piecewise-differentiable pKas. Use the direct root output as the numerical comparison for a chosen grid resolution. The grid result cannot be silently substituted for a direct midpoint without recording that approximation.

Synthetic graphs demonstrate execution and give reproducible microbenchmark inputs; they do not establish relative performance against PROPKA, realistic protein memory requirements, GPU throughput, or scientific accuracy.

## Performance tuning

The dominant cost is the repeated global occupancy solve, not Python dispatch.
For `N` residues, interaction capacity `Kc`, nine titration channels, `T`
occupancy iterations, `H` sampled pHs, `B` bisection steps, and `Q` direct
midpoint queries, the leading work is approximately:

| Operation | Leading cost |
|---|---|
| Single-pH charge | `O(T N Kc 9^2)` |
| Curves or shared-grid pKas | `O(H T N Kc 9^2)` |
| Direct midpoint queries | `O(Q B T N Kc 9^2)` |

The sequence-dependent terms also contract all 20 candidate identities over the
environment graph. This is intentional: a soft sequence retains conditional
states for identities that are absent from a particular hard sequence. It also
means that a differentiable evaluation does substantially more work than a
single-hard-sequence reference calculation.

Apply the following changes in roughly this order:

1. Keep all-site direct roots out of an optimization loop. Prefer charge or curve
   objectives when they express the design goal. When midpoint pKas are required,
   `pka_from_grid()` shares each pH solve across every site. Validate the chosen
   grid against direct roots on representative hard and soft sequences because
   its interpolation is an explicit numerical approximation.
2. Request only necessary direct roots with `pka_sites()`. `pka()` without a
   selection requests all `N x 9` conditional roots, including alternative
   identities. Selecting residues reduces `Q` for direct roots, but it does not
   shrink the global electrostatic environment solved for each query.
3. Tune `ModelConfig.steps` using residuals and output comparisons. Runtime is
   approximately linear in this fixed iteration count. Sweep lower values on the
   actual design distribution and retain the smallest value that preserves
   convergence, values, and gradients. Strongly coupled cases may require more
   than the default 64, so this is not a universally safe reduction.
4. Tune `root_steps` for direct pKas. The default 28 bisections over the default
   34-pH-unit interval give a nominal final width of about `1.3e-7`; 21 steps give
   about `1.6e-5`. A smaller count can remove a meaningful fraction of direct-root
   work, but it must be checked against crossing error, validity, and pKa/gradient
   regressions rather than chosen from interval width alone.
5. Construct each JIT readout once and reuse it. Save and reload `StructureCache`
   objects when a structure is reused, warm compiled functions before timing, and
   keep preprocessing and compilation outside an optimization loop.
6. Tune `root_batch_size` per device. Larger chunks may improve accelerator
   utilization while increasing working memory; smaller chunks reduce peak
   occupancy storage. Batch independent sequences with `jax.vmap` when throughput
   matters, especially on an accelerator where a single small protein may not
   provide enough parallel work.
7. Control padding. Real candidate-union graphs can have much larger `Ke` and
   `Kc` than the `8/9` synthetic graphs above. Padding every structure to a large
   bucket increases all contractions, so use buckets tight enough to avoid large
   amounts of inactive work while still reusing compilation.

Gradient workloads require separate memory measurements. In the `N=64`
benchmark, the compiler estimated about 114 MB of temporary storage for the
grid-pKa value-and-gradient function, compared with about 2.6 MB for its forward
function. Reverse mode differentiates the finite unrolled occupancy iterations.
If this becomes the limiting resource, `jax.remat` can exchange recomputation for
activation memory. Implicit differentiation of the converged fixed point could
reduce the dependence on the unrolled history, but it would be a larger numerical
change and would require convergence and gradient validation.

The main longer-term kernel target is the padded dense `[N,Kc,9,9]` coupling
representation. Packed active type edges, degree-based buckets, or a factored
interaction kernel could reduce work for high-degree proteins. Sparse scatter or
segment reductions are not automatically faster on every backend, so compare
compiled memory and synchronized execution before adopting such a representation.
Convergence acceleration, such as pH warm starts or Anderson/Newton-style updates,
could also lower `T`, but may change the fixed-point branch reached by the model
and therefore crosses a scientific validation boundary.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench.py --synthetic-n 16 --repeats 10 --all-pka --output reports/benchmark_cpu_n16_all.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench.py --synthetic-n 64 --repeats 10 --output reports/benchmark_cpu_n64.json
JAX_PLATFORMS=mps python benchmarks/bench.py --pdb protein.pdb --require-mps --output reports/protein_mps.json
python benchmarks/compare.py protein.pdb --output reports/reference_timing.json
```

The last two commands were not run here. The comparison script labels warmed JAX evaluation separately from the complete PROPKA subprocess, including its startup, parsing, computation and writing; it does not automatically present those unlike scopes as a speedup ratio.

Raw files: `reports/benchmark_cpu_n16_all.json` and `reports/benchmark_cpu_n64.json`.

## Reusing compilation across structures

`benchmarks/regress_foldbench.py` passes structure arrays to shared JIT kernels
and pads `(N, Ke, Kc)` to multiples of `(64, 16, 16)` by default. Override these
with `--bucket-multiple N Ke Kc`. Larger multiples reduce the number of shape
variants but increase padded memory and computation. Each successful case records
its actual `bucket_capacity` in the report. Neighbors are never truncated.

Workers retain their JIT caches between cases. For a fixed operation, pH-grid
length, dtype, and model configuration, structures with the same padded shapes
reuse compilation within a worker. Different workers warm their caches separately;
there is no persistent disk cache. Padding is numerical only: inactive groups and
edges are masked, and report metrics use real residues and physical sites.

The same path can be used directly:

```python
import jax
import numpy as np
from jaxpropka import ModelConfig
from jaxpropka.batching import pack_inputs
from jaxpropka.model import grid_pka_kernel, one_hot

config = ModelConfig(steps=128)
ph = np.linspace(config.ph_min, config.ph_max, 145, dtype=np.float32)
for cache in structure_caches:
    arrays, p, n = pack_inputs(cache, one_hot(cache.native_index))
    arrays, p = jax.device_put((arrays, p))
    result = jax.device_get(grid_pka_kernel(arrays, p, ph, config=config))
    values, valid = result.value[:n], result.valid[:n]
```

`curve_kernel` accepts the same arguments and returns full site curves without
chain labels or aggregation. Keep both kernels at module scope and pass new
structures as arguments. The existing `TitrationModel` P-only readouts retain
their fixed-structure interface and remain useful for sequence design on one
structure. Direct-root pKa readouts still use that interface.

Use `JAX_LOG_COMPILES=1 JAX_EXPLAIN_CACHE_MISSES=1` to inspect specialization.
When timing a sweep, measure initial bucket compilation separately and synchronize
device results before recording execution time. The FoldBench regression runner
reports accuracy metrics, not timings.

## Experimental packed-interaction kernel: real structures (2026-10-02)

The packed backend in `src/jaxpropka/model.py` packs the fixed-geometry
`pair_mask` into active identity-to-identity edges and uses a segment reduction
for each occupancy field update. It retains all geometrically allowed hypothetical
identities, not just the native sequence's groups. Sequence-dependent local terms
still use the original dense calculation. It is available through the opt-in
`TitrationModel(..., backend="packed")`; the dense backend remains the default.

### Measurement scope

- CPU only, four allocated CPUs per process, float32; jobs excluded `comp1400`
  and `comp0601`. Pilot jobs ran on `comp0650` and `comp0651`.
- Exact unbucketed structure shapes; fixed geometry and unchanged equations,
  damping, iteration counts, and convergence tolerance.
- Soft probabilities are `(1-mixture)*native + mixture/20`, tested at mixtures
  `0.05` and `0.001`. The table below reports `0.05` at **128 iterations**.
- pH points span 6.0–7.4. Value-plus-gradient differentiates the trapezoidal
  integral of a single complex's total charge with respect to sequence logits.
  This is **not** the complete bound/free selectivity loss or a pKa midpoint.
- Each operation is compiled separately; persistent compilation caching is
  disabled. Timings synchronize via `device_get`: one first execution followed
  by two warm executions, whose median is reported. Compilation, geometry/cache
  preparation, and edge packing are excluded from these warm timings.
- These are small timing pilots, not confidence intervals or GPU measurements.
  Two pH points do not establish quadrature accuracy for the intended loss.

### Warm CPU timings

All values are seconds; each cell is **original dense → experimental packed**.

| Operation | 8srz, 176 residues | 8jdh, 289 residues | 7zhf, 500 residues | 8r7i, 1,254 residues |
|---|---:|---:|---:|---:|
| 2-pH curves | 1.255 → 0.053 | 1.991 → 0.114 | 5.331 → 0.256 | 12.692 → 0.756 |
| 2-pH value + gradient | 2.909 → 0.124 | 4.457 → 0.234 | 12.634 → 0.531 | 28.764 → 1.885 |
| 15-pH curves | 1.406 → 0.160 | 2.194 → 0.291 | 6.125 → 0.594 | 13.943 → 1.648 |
| 15-pH value + gradient | 4.056 → 0.804 | 6.246 → 1.595 | 17.761 → 3.924 | 42.938 → 10.677 |

All table evaluations converged at the sampled pHs. For the two smaller cases,
only 116,616/2,223,936 (5.24%) and 195,784/4,143,393 (4.73%) padded pair entries
were active. Avoiding the inactive entries materially reduces repeated solve
work. The speedup persists on larger structures, but large-system 15-pH reverse
gradients remain well above one second. Cache preparation is not accelerated by
this experiment.

### Numerical parity and run status

Snapshot recorded on **2026-10-02**:

- Four synthetic tests passed (float32/float64, nonempty/empty interaction
  graphs), checking output and gradient parity, frozen identities, and float64
  directional finite differences.
- Job **725125** completed both smaller structures. All 16 combinations of
  structure, 128/512 iterations, 2/15 pHs, and the two soft mixtures passed the
  real-structure parity checks. Maximum occupancy difference was `2.18e-6` and
  maximum absolute logit-gradient difference was `1.19e-7`.
- Job **725128_6** completed 7zhf: all eight soft-sequence comparisons passed.
  Its 145-point native grid over pH −10 to 24 also passed parity at 128 and 512
  iterations. The known failure was preserved: maximum native-weighted residual
  at 512 iterations was `3.86536e-4` for both kernels, above tolerance `2e-5`.
  Faster execution does not resolve the model's convergence difficulty.
- Job **725128_43** completed all eight soft-sequence comparisons and both native
  wide-grid checks. At 512 iterations and 15 pHs, its warm dense → packed times
  were 54.16 → 4.96 seconds for curves and 169.21 → 40.14 seconds for value plus
  gradient. Its process peaked at 103,718,112 KiB while sequentially measuring
  both implementations; this is not a packed-only memory measurement.

Real-structure parity gates use occupancy `atol=rtol=2e-5`, total-charge
`atol=2e-4, rtol=2e-5`, logit-gradient `atol=2e-5, rtol=2e-4`, and identical
per-pH convergence flags. The wide-native checks compare occupancies, total
charges, and convergence flags, not gradients. Agreement with the original
finite-iteration solver does not prove a unique equilibrium branch or molecular
accuracy. Broader validation is required before promoting this experimental
implementation into production.

### Artifacts and reproduction

Scripts: `benchmarks/profile_packed_curves.py`, the compatibility wrapper
`benchmarks/packed_curve_kernel.py`, and `tests/test_packed_curves.py`.
The Slurm wrapper is workspace-relative
`_HPC/submission/jax-Ka/profile-packed-curves.sbatch` (outside the repository).
Job 725128 was submitted with `--array=6,43 --mem=128G --time=04:00:00`;
the extra memory accommodates the original dense reverse-mode comparison.

Raw JSON under workspace root `/home/coulson/oc/lina4225`:

- `_runtime/jax-Ka/cuda12/benchmarks/packed-profile/725125/case-111.json`
- `_runtime/jax-Ka/cuda12/benchmarks/packed-profile/725125/case-120.json`
- `_runtime/jax-Ka/cuda12/benchmarks/packed-profile/725128/case-6.json`
- `_runtime/jax-Ka/cuda12/benchmarks/packed-profile/725128/case-43.json`

Reports record source hashes, JAX version, host, shape counts, compilation and
execution times, convergence diagnostics, and parity differences. Inputs are
the corresponding `interfaces/724720/report-{index}.json` reports; the rebuilt
structure cache fingerprint must match the original. The extended run enables
`--wide-native-check`; the initial pilot did not include that option.
