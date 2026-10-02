# Validation inventory

## Executed locally

The dependency-enforced CPU suite reports **57 passed** using Python 3.13.2, JAX/JAXLIB 0.11.1, Biotite 1.7.1 and the pinned PROPKA 3.5.1 package. The dedicated `jax-mps==0.11.0` test also passes on an Apple Silicon MPS device. Numerical finite-difference tests enable JAX float64 on CPU; MPS inputs are float32 because Metal does not support float64.

The included benchmark reports preserve the earlier benchmark environment and package versions recorded in their JSON files.

| Area | Executed coverage |
|---|---|
| Analytic response | Independent-site Henderson–Hasselbalch curves, model pKa midpoints, charge limits and acid/base signs |
| Differentiation | Charge, titration curves, direct midpoint and grid-interpolated midpoint gradients checked by directional finite differences; JVP/reverse-mode agreement |
| Conditional identity | No premature acid/base averaging, soft-sequence dependence, presence probabilities, fixed-covalent gradient clamps |
| Multiple chains, numerical layer | Separate terminal sites; full-environment charge accounting; cross-chain messages; selection leaves the environment intact |
| Sparse kernels | Radial volume/count cutoffs, H-bond direction/distance factor, conservative graph coverage, overflow rejection, reciprocal blocks |
| Numerical safety | Finite padded gathers, padding invariance, unavailable sites, unbracketed pKas, validation flags, iteration-count sensitivity |
| Cache/CLI | Serialization, fingerprints, identity selection, invalid inputs and numerical prediction-file output |
| Regression machinery | Summary parsing, mapped chain/insertion identity, report construction against analytic independent sites, missing-site and soft-sequence rejection, baseline drift checks |
| Molecular integration | Biotite topology, CCD candidates, PDB/mmCIF handling, gaps, disulfides, reference export and molecular gradient smoke coverage |
| External reference | Pinned pip PROPKA 3.5.1 live two-chain discrepancy report and default CLI regression path |
| Implementation snapshots | Recorded hard/soft sequence outputs from the synthetic numerical graph, explicitly not an external reference |
| Performance | Synchronized warm CPU timings for charge, curves, gradients, direct/grid pKas and sequence batching |

The source-distribution wheel was built, and the source tree was byte-compiled. Packaging and synthetic execution do not validate the molecular implementation.

## Memory-efficient gradient implementation (CPU)

The expanded CPU suite, including three benchmark-reporting tests, passes
**168 tests**, with five external-reference/MPS tests deselected
(`pytest -m 'not reference and not mps'`). This run uses Python 3.12 and JAX
0.11.1; it does not replace the historical external-reference or MPS results.

New coverage includes finite-iteration checkpoint/remainder parity, packed-edge
local terms and masked padding, float32/float64 sequence gradients, implicit
JVP/VJP/directional finite differences, failed primal/adjoint solves, absence of
an iteration-by-edge reverse tape, residue-key mapping, pH chunking, composition
with softmax, and bound/free scalar-gradient equivalence. Experimental endpoint
tests cover stationarity, baseline charge, quadrature refinement, envelope
gradients and continuation-path diagnostics. The reporting checks prevent
saturated softplus losses from hiding selectivity-gradient errors.

Protein–protein scaling and the four archived reduction-order-sensitive cases
are documented in [BENCHMARKS.md](BENCHMARKS.md). These are numerical and
performance checks, not experimental calibration or evidence of a unique
equilibrium branch. New accelerator execution remains unvalidated.

## Adaptive design-preset coverage (CPU)

The expanded suite passes **182 tests**, with five reference/MPS tests deselected.
The additional coverage checks the unchanged ordinary defaults, adaptive
stopping and exact caps, consecutive-check requirements, nonfinite detection,
independent batched stopping, JVP/VJP/finite-difference agreement, batched
sequence gradients, direct midpoints (including unusable gradients on failed
solves), streamed diagnostics, experimental endpoint consistency, and an
explicit multistart audit that detects a constructed multiple-solution case.
Promotion-report tests reject incomplete, stale or invalid measurements and
prevent audit classifications from hiding converged-trajectory parity failures.

The CPU protein–protein promotion panel is separate from these unit tests and
from the historical packed-backend panel. It uses 260 eligible interfaces for
forward checks, a deterministic 24-interface gradient subset, and five scaling
interfaces. Numerical convergence, sampled-path consistency and derivative
agreement are separate outcomes. A failed audit can reflect finite-solve error
or distinct branches; it does not identify the cause or select an equilibrium.
See [the design preset](GRADIENTS.md#opt-in-cpu-design-preset) for the API and
explicit-audit contract. Ordinary package defaults remain unchanged.

The completed [CPU promotion report](../reports/design_preset_cpu.json) passes
the numerical gates: 764/780 forward samples and 67/72 gradient samples qualify;
remaining samples are explicitly classified as convergence or audit failures.
All 135 qualifying directional checks, all five scaling cases and the four
archived sensitive-case regressions pass. This supports the opt-in CPU preset
with caller-controlled audits, not universal convergence or scientific calibration.

## Hardware-specific execution

**One MPS forward/reverse/midpoint test** executes on the Metal device with CPU fallback disabled. CUDA execution and accelerator performance benchmarking have not yet been established.

## External reference protocol

1. Install the pinned `propka==3.5.1` reference extra and record the installed distribution version in every report. Historical PROPKA 3.0 comparisons additionally require an explicitly selected clean checkout and exact commit.
2. Preserve candidate geometry and export the same categorical sequence to the reference. Native and alternate side-chain candidates must not be compared against a different independently repacked structure without explicitly labeling that additional difference.
3. Normalize PDB labels for the external tool and map back to original chain/number/insertion identities. PDB coordinate export rounds to its coordinate field precision, and reference summary pKas are reported at finite text precision. These are additional comparison limitations.
4. Require every expected physical site and check midpoint/titration convergence. Do not selectively omit failed sites to improve an error statistic. Reference-only excluded sites are recorded in the report.
5. Record per-site pKa differences and site/total charge-curve differences. The reference charge curves are independent HH curves reconstructed from its reported pKas, not microscopic protonation ensembles.
6. Review the observed discrepancy and only then save an approved baseline. Baseline checks require matching input/reference provenance, cache fingerprint, model configuration, sites and pH grid. Existing baselines are never overwritten implicitly.

Running a live report and checking finite numbers is an **integration/discrepancy test**, not a proof of scientific accuracy. A test report compared to itself exercises the baseline machinery, not an independent regression target. The analytic tests and self-generated synthetic snapshot are kept separately from actual reference data.

## Production acceptance remains open

Before using this model to select sequences, complete the real Biotite/pinned-PROPKA test runs; evaluate native and mutant structures representative of the target design domain; measure pKa and charge-curve errors; validate mutation-gradient direction against discrete rescoring; examine iteration and grid-resolution sensitivity; and measure actual accelerator throughput/memory with realistic graph degrees. Any acceptance threshold should be chosen for the scientific use, not inferred from this package's unit tests.

The model has one frozen candidate conformation per identity, simplified protonation-state H bonds, a fractional mean-field coupling model and a smoothed burial eligibility gate. Its predictions have not been calibrated against experimental pKas. It should not be described as a reference-faithful PROPKA 3.0 replacement, a folding-energy model, or a demonstrated speedup over PROPKA.
