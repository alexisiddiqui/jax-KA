# Executed CPU benchmarks

**These are synthetic numerical graphs, not molecular structures or a PROPKA performance comparison.**

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
