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

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench.py --synthetic-n 16 --repeats 10 --all-pka --output reports/benchmark_cpu_n16_all.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench.py --synthetic-n 64 --repeats 10 --output reports/benchmark_cpu_n64.json
JAX_PLATFORMS=mps python benchmarks/bench.py --pdb protein.pdb --require-mps --output reports/protein_mps.json
python benchmarks/compare.py protein.pdb --output reports/reference_timing.json
```

The last two commands were not run here. The comparison script labels warmed JAX evaluation separately from the complete PROPKA subprocess, including its startup, parsing, computation and writing; it does not automatically present those unlike scopes as a speedup ratio.

Raw files: `reports/benchmark_cpu_n16_all.json` and `reports/benchmark_cpu_n64.json`.
