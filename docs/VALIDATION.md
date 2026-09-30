# Validation inventory

## Executed in the delivery environment

The completed local suite reports **38 passed and 21 explicitly skipped**. The numerical-only command reports **38 passed, 21 deselected**. The actual logs, JUnit XML and machine-readable status are in `reports/`.

The numerical execution used Python 3.13.5, JAX/JAXLIB 0.9.0.1 and a CPU device. The precise package versions are recorded in the benchmark JSON files. Numerical finite-difference tests enable JAX float64; benchmark inputs are float32. Float32 forward/gradient tests are also present.

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
| Implementation snapshots | Recorded hard/soft sequence outputs from the synthetic numerical graph, explicitly not an external reference |
| Performance | Synchronized warm CPU timings for charge, curves, gradients, direct/grid pKas and sequence batching |

The source-distribution wheel was built, and the source tree was byte-compiled. Packaging and synthetic execution do not validate the molecular implementation.

## Not executed: 21 tests

**15 Biotite integration tests** are included for canonical CCD candidates, native coordinate preservation, rigid transformations, repeated numbering across chains, separate termini, far-chain controls, near-chain interaction edges, author chain/insertion mmCIF round-tripping, PDB TER, gap policies, missing atoms, unsupported chemistry, disulfide freezing, reference export and a molecular-gradient smoke test. These require the real Biotite package; no mock package substitutes for it.

**Five external PROPKA checks** are included: original-3.0 Coulomb, radial-volume and geometric H-bond primitive comparisons; an original-3.0 two-chain live discrepancy report; and a separately identified modern-PROPKA live report. Neither implementation was available locally. The repository contains no fabricated PROPKA output and no approved full-structure PROPKA baseline.

**One GPU forward/reverse/midpoint test** is included. No GPU is available in this environment, so GPU execution and GPU performance have not been established.

The local environment could not download missing packages. This is the reason for the dependency skips, not evidence that those paths are correct. GitHub workflows are supplied to execute these paths in an installation with the required dependencies; those workflows have not been run as part of delivery.

## External reference protocol

1. Resolve an original PROPKA 3.0 Git revision, preserve its exact commit and use a clean checkout. The reference adapter refuses a modified tracked checkout or mismatched explicitly requested commit. The default inspected parameter subversion is Nov30.
2. Preserve candidate geometry and export the same categorical sequence to the reference. Native and alternate side-chain candidates must not be compared against a different independently repacked structure without explicitly labeling that additional difference.
3. Normalize PDB labels for the external tool and map back to original chain/number/insertion identities. PDB coordinate export rounds to its coordinate field precision, and reference summary pKas are reported at finite text precision. These are additional comparison limitations.
4. Require every expected physical site and check midpoint/titration convergence. Do not selectively omit failed sites to improve an error statistic. Reference-only excluded sites are recorded in the report.
5. Record per-site pKa differences and site/total charge-curve differences. The reference charge curves are independent HH curves reconstructed from its reported pKas, not microscopic protonation ensembles.
6. Review the observed discrepancy and only then save an approved baseline. Baseline checks require matching input/reference provenance, cache fingerprint, model configuration, sites and pH grid. Existing baselines are never overwritten implicitly.

Running a live report and checking finite numbers is an **integration/discrepancy test**, not a proof of scientific accuracy. A test report compared to itself exercises the baseline machinery, not an independent regression target. The analytic tests and self-generated synthetic snapshot are kept separately from actual reference data.

## Production acceptance remains open

Before using this model to select sequences, complete the real Biotite/PROPKA test runs; evaluate native and mutant structures representative of the target design domain; measure pKa and charge-curve errors; validate mutation-gradient direction against discrete rescoring; examine iteration and grid-resolution sensitivity; and measure actual GPU throughput/memory with realistic graph degrees. Any acceptance threshold should be chosen for the scientific use, not inferred from this package's unit tests.

The model has one frozen candidate conformation per identity, simplified protonation-state H bonds, a fractional mean-field coupling model and a smoothed burial eligibility gate. Its predictions have not been calibrated against experimental pKas. It should not be described as a reference-faithful PROPKA 3.0 replacement, a folding-energy model, or a demonstrated speedup over PROPKA.
