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

## Packed interaction protein–protein regression (2026-10-02)

This is a partial-by-request regression against the accepted dense reports from
interface panel 724720. Nineteen input exclusions had no numerical cache; 259 of
260 eligible pairs completed, and the remaining 1,708-residue `8ic7 A/B` case was
stopped during cache construction. Each completed case rebuilt and fingerprinted
the exact cache, then ran dense and packed kernels over the accepted adaptive
schedule. Jobs used eight CPU cores and 32 GiB. The comparison includes first
compilation/execution but excludes cache construction and PROPKA, which is
unchanged by the interaction representation.

| Outcome | Pairs |
|---|---:|
| Manifest | 279 |
| Input exclusions | 19 |
| Numerically completed | 259 |
| Passed every parity gate | 255 |
| Failed at least one gate | 4 |
| Stopped/incomplete | 1 |

Across the 259 completed pairs, 703/708 solver-grid comparisons passed every
gate. All **165/165** pairs previously classified `passed` retained parity.
Among accepted `partial` cases, 86/87 passed; among accepted numerical failures,
4/7 passed. Thus no previously accepted-passed structure acquired a numerical,
convergence, site-category, or final-tier regression.

The four flagged pairs separate into three categories:

| Pair | Accepted state | Observation | Interpretation |
|---|---|---|---|
| `8kbm A/A-2` | failed | Refined-grid occupancy max difference `2.074e-5`, just over the `2e-5` gate; charges, convergence flags, site categories, and all final tiers agree. | Tolerance-edge mismatch, not an outcome change. |
| `8qvc A/A-2` | partial | 128-step coarse occupancy difference `3.698e-5`; the 512-step coarse and refined grids pass, and all final tiers agree. | Transient finite-iteration mismatch that disappears after the accepted retry. |
| `9c90 A/A-2` | failed | Both schedules remain globally invalid, but convergence flags differ at some pHs; maximum occupancy difference `0.0131`. All final tiers remain invalid. | Pre-existing nonconvergent, order-sensitive solve; no usable-result change. |
| `8kck A/A-2` | failed | At 128 steps, max occupancy/charge/pKa differences are `0.0700`, `0.0174`, and `0.0229`; differences shrink at 512 steps but convergence/category flags differ. Dense marks all 88 sites invalid, while packed marks 84 strict and 4 invalid. | Substantive branch/convergence sensitivity. Packed convergence cannot be treated as evidence of a unique equilibrium. |

`8kck` is the only completed pair with changed final tiers and the only one with
a changed site category. Its change is structure-global: the dense policy rejects
every site because the coupled solve is nonconvergent, whereas a floating-point
reduction-order change lets the packed iteration reach a converged branch for
most sites. This is not evidence that packed is scientifically better; it exposes
the known fact that small residuals do not establish branch uniqueness.

### Panel timing

| Grid | Comparisons | Median dense/packed ratio | Aggregate dense/packed ratio |
|---|---:|---:|---:|
| Charge, 15 pHs | 252 | 2.92x | 3.97x |
| Coarse, 145 pHs | 301 | 1.46x | 1.63x |
| Refined, 577 pHs | 155 | 1.24x | 1.23x |
| All completed schedules | 708 | 1.52x | 1.39x |

These cold/report-run speedups are intentionally more conservative than the warm
pilot timings. They include compilation and sequence-dependent dense local-term
construction, and cover heterogeneous graph densities. They are not timings of
the final three-environment bound/free loss or its gradient.

### Decision

The packed representation is accepted as an **opt-in execution backend** for
development and design profiling: its equations are shared with dense, gradients
passed the dedicated real/synthetic checks, and all accepted-passed interface
cases retained parity. It is **not accepted as the unconditional default** under
the fail-closed gate. Before default promotion, the solver needs a
backend-independent branch-stability guard (for example, multiple initial states
or pH histories) and final discrete designs should be audited independently.
Merely falling below the residual tolerance in the packed kernel is insufficient,
as demonstrated by `8kck`.

Raw reports and the corrected partial summary are under workspace root:
`_runtime/jax-Ka/cuda12/benchmarks/packed-panel/interface-full/`. The summary's
`acceptance_passed` is deliberately false because four strict gates failed and
one requested case was stopped.

## Memory-efficient gradients: CPU protein–protein scaling

The new gradient paths are opt-in; dense/unrolled defaults and the preceding
packed-panel decision are unchanged. See [GRADIENTS.md](GRADIENTS.md) for API
usage and failure semantics. The reference is the original packed implementation
at `df5abd608cbb3c92a7e1a0097ed19f38cf2d19fb`, not a rewritten surrogate baseline.

Four representative physical interfaces were prepared as separate complex,
binder and target environments. Fresh-process CPU measurements use JAX 0.11.1,
float32, four allocated CPU threads, 128 forward iterations and 15 pHs from 6.0
to 7.4. Binder probabilities are 0.05 and 0.001 uniform/native mixtures; target
sequence probabilities are fixed. Full-objective comparisons use the original
public P-only curve readouts versus the new P-only streamed objective, with the
isolated target cached outside differentiation in both. Complex-only comparisons
use dynamic-runtime curve kernels in both. Compilation and one warm-up precede
five synchronized timings per mixture. Jobs ran on different CPU hosts, so
timings are descriptive rather than controlled hardware speedup guarantees.

### Full bound/free loss-gradient temporary storage

These are **compiler-estimated temporary bytes**, in decimal MB, not total
process/device memory. The checkpointed column uses legacy packed local terms
with streamed pHs; packed_v2 also constructs active-edge local terms and tiles
environment contractions. Both columns differentiate the finite iteration
algorithm. Implicit differentiates the converged local fixed point.

| Interface | Residues | Original packed | Checkpointed | packed_v2 + checkpointed | packed_v2 + implicit |
|---|---:|---:|---:|---:|---:|
| 8srz A/B | 176 | 1,403.2 | 80.5 | 19.6 | 20.0 |
| 8jdh A/B | 289 | 2,357.0 | 157.3 | 39.1 | 39.7 |
| 7zhf A/A-2 | 500 | 5,522.6 | 1,057.3 | 87.7 | 83.3 |
| 8r7i A/A-2 | 1,254 | 17,124.0 | 5,486.6 | 272.8 | 273.1 |

For the 1,254-residue full objective, packed_v2/checkpointing reduces temporary
storage **62.8x**. Warm median times across the two mixtures are 15.41 seconds
for original packed, 17.11–17.40 seconds for packed_v2/checkpointed, and
5.09–5.33 seconds for implicit. Checkpointing is a memory/time tradeoff, not a
universal speedup. Fresh-process peak RSS is 32.50 GiB for original packed,
27.28 GiB for packed_v2/checkpointed and 27.46 GiB for implicit: host caches,
captured geometry constants and compilation still dominate process memory.
The temporary-storage reduction must not be reported as an equal RSS reduction.

The unstreamed 1,254-residue complex curve-gradient kernel drops from 11.887 GB
to 2.301 GB of temporary storage with packed_v2/checkpointing, a **5.17x**
reduction. This isolates the kernel improvement from the larger gain obtained
by streaming the scalar bound/free objective.

Increasing the iteration count from 128 to 512 on this case leaves values and
gradients unchanged at the measured precision. The complex-gradient baseline's
temporary storage grows to 35.855 GB, versus 2.317 GB with packed_v2/checkpointing
and 2.604 GB with implicit differentiation. The streamed packed_v2 objective's
temporary storage stays at 272.8 MB (checkpointed) or 273.1 MB (implicit), versus
52.223 GB for the original full-objective baseline at 512 iterations.

### Optional 1,708-residue stress case: not converged

`8ic7 A/B` completes the complex-only memory measurement, but is **not a usable
equilibrium result** at 128 iterations. Original and new packed paths both have
maximum residuals around `3.5e-4` and `7.2e-4` for the two mixtures, above the
required tolerances. The finite-iteration outputs, gradients and convergence
flags still match. Implicit mode deliberately returns nonfinite gradients.

Original/new-checkpointed temporary storage is 17.401/3.306 GB; these numbers
describe execution capacity only, not an accepted converged performance win.
The required four-case panel passes, but the aggregate report's all-case
`acceptance_passed` remains false when this invalid optional case is included.
No automatic iteration increase or derivative fallback was used.

### Numerical gates and scope

All four scaling cases pass original-packed value, gradient and convergence-flag
comparisons for both complex-only and full-objective workloads, in both mixtures.
Comparisons also undo the known softplus scalar derivative in float64 to check
the underlying selectivity/logit gradient: a saturated loss alone is not a
meaningful absolute-gradient accuracy test. Forward and adjoint validity are
required; invalid implicit solves cannot count as performance wins.

All four archived sensitive cases (`8kbm`, `9c90`, `8kck`, `8qvc`) pass the new
finite-iteration paths against the original **packed** backend at their archived
sensitive schedules. This preserves, rather than erases, the historical
dense-versus-packed branch/convergence discrepancies above.

The experimental endpoint/envelope objective passes a separate float64 audit
on the 176-residue interface at 512 iterations. Both mixtures have consistent
independent/increasing/decreasing-pH paths, directional finite-difference
agreement, and endpoint-versus-quadrature selectivity error decreasing from
`1.66–1.69e-4` at 15 pHs to `1.99–2.02e-6` at 129 pHs. Fine-grid loss-gradient relative
L2 errors are below `6.4e-6`. This does not certify a global equilibrium branch
or validate the endpoint approximation on every scaling example.

Raw fresh-process JSON, gradient arrays, cache fingerprints and preparation
timings are under workspace root
`_runtime/jax-Ka/cuda12/benchmarks/gradient-memory/`. Baseline provenance is
tracked in `reports/gradient_memory_provenance.json`; the portable aggregate,
test counts, source hashes and iteration checks are in
[gradient_memory_cpu.json](../reports/gradient_memory_cpu.json). CPU unit/gradient coverage
is detailed in [VALIDATION.md](VALIDATION.md). CUDA/MPS memory and throughput
for these new paths have not been measured.

## Adaptive design preset: CPU protein–protein panel

The opt-in `TitrationModel.for_design` preset uses packed_v2, implicit gradients
and adaptive stopping with a 1,024-update cap. Ordinary construction remains
dense/unrolled with exactly 64 updates. This panel is separate from the historical
128-update results above; increasing the cap does not change the equations,
initialization or damping.

Scaling measurements use the full bound/free selectivity objective, 15 pHs from
6.0 to 7.4, float32, and a nonsaturated softplus penalty. Each fresh CPU process
performs one warm-up followed by five synchronized repetitions per mixture.

| Residues | Compiler temporary MB | Peak process GiB | Warm seconds, 0.05 / 0.001 mixture | Maximum updates, 0.05 / 0.001 |
| ---: | ---: | ---: | ---: | ---: |
| 176 | 19.96 | 1.87 | 0.635 / 0.633 | 144 / 144 |
| 289 | 39.71 | 3.19 | 1.104 / 1.108 | 144 / 144 |
| 500 | 83.26 | 7.13 | 2.468 / 2.430 | 144 / 144 |
| 1,254 | 273.14 | 27.50 | 6.955 / 6.543 | 144 / 144 |
| 1,708 | 387.71 | 38.79 | 8.907 / 10.248 | 160 / 464 |

All ten measured scaling evaluations have valid primal and adjoint solves and
finite gradients. At 1,254 residues, compiler temporary storage is **62.69x lower**
than the original packed full-objective 128-update baseline. This is not a
62.69x reduction in process memory: geometry caches, compiler storage and other
host allocations still dominate peak RSS. Runs used heterogeneous CPU hosts;
timings are descriptive, not a controlled cross-host speedup claim.

The former 1,708-residue nonconvergence at 128 updates is resolved by continuing
the same trajectory under the larger cap. Its strict float64 full-objective
gradient checks also pass for native and both soft sequences, against a fixed
2,048-update reference and (for soft sequences) three directional finite
differences. The native strict solve needs up to 1,008 updates. This does not
imply that every sequence converges within 1,024 updates.

The promotion protocol checks all 260 eligible protein–protein interfaces from
the 279-entry archive; the 19 previously unsupported inputs remain explicitly
excluded. Forward comparisons use the frozen original packed backend with a
fixed 1,024-update schedule. The deterministic 24-interface gradient subset
includes all five scaling cases and all four sensitive cases. Gradient checks
use float64 with primal tolerance `1e-10` and adjoint `rtol=1e-10, atol=1e-12`,
against packed_v2/checkpointed fixed-2,048-update derivatives. Every qualified
soft sample also gets three directional finite differences. These stricter
gradient checks are separate from default-float32 performance measurements.

Audit failures are retained rather than treated as parity failures or silently
resolved. For example, `9c90` (case 100) still exhausts the budget for some
sequences, whereas `8qvc` (case 212) has converged float64 paths whose occupancies
differ by about 0.82. A larger budget addresses insufficient iteration counts;
it does not resolve multiple branches. Conversely, the float32 stress-case
audit flags gaps of only `2.1–2.8e-5` for native/near-native inputs, just above
the `2e-5` threshold; its stricter float64 audit passes. Such small discrepancies
need not represent distinct branches. Never transfer an audit result across
precision, sequence or geometry changes.

The archived sensitive schedules were rerun against the original packed code:
all four interfaces pass all 12 backend/schedule comparisons, with identical
total charges and maximum occupancy difference `3.58e-7`. This regression check
preserves the previous finite-iteration behavior; it does not clear the new
equilibrium audit failures. The current CPU suite passes 182 tests, with five
external-reference/MPS tests deselected.

Across the completed forward panel's 780 samples (native plus two mixtures per
interface), **764 pass**, **13 are audit-flagged**, and **three fail forward
convergence** (all case 100). All samples pass on 252 of 260 interfaces. Every
numerically valid comparison agrees with the original packed trajectory; no
new forward parity or convergence regression was observed. The audit flags
are not discarded from these counts.

The completed gradient subset has **67 passing samples**, **three audit-flagged
samples** (case 212), and **two forward failures** (case 100). All 135 directional
finite-difference checks on qualifying soft samples pass. Both soft samples
qualify on 22 of 24 interfaces, including every scaling case. No new converged
value/gradient parity or adjoint regression was observed; failed primal solves
remain unusable, not silently replaced by finite-iteration derivatives.

The numerical promotion gates pass with no missing or stale-source records.
The preset is therefore qualified for **opt-in CPU design with explicit audits**,
not promoted to the ordinary constructor default. This is numerical validation,
not experimental calibration or a global equilibrium certificate. The portable
[design_preset_cpu.json](../reports/design_preset_cpu.json) records source/input
provenance, classifications, exclusions, gradient checks and scaling results.
Full per-pH/per-path data remain under workspace root
`_runtime/jax-Ka/cuda12/benchmarks/design-preset/`. Historical benchmark reports
are unchanged. Endpoint and accelerator promotion remain out of scope.
