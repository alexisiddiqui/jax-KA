# Validation inventory

## Executed locally

The dependency-enforced local suite reports **57 passed and 1 hardware-specific skip** using Python 3.11.7, JAX/JAXLIB 0.9.2, Biotite 1.6.0 and the pinned PROPKA 3.5.1 package on CPU. Numerical finite-difference tests enable JAX float64; benchmark inputs are float32. Float32 forward/gradient tests are also present.

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

## Hardware-specific test

**One GPU forward/reverse/midpoint test** is included. No GPU is available in this environment, so GPU execution and GPU performance have not been established.

## External reference protocol

1. Install the pinned `propka==3.5.1` reference extra and record the installed distribution version in every report. Historical PROPKA 3.0 comparisons additionally require an explicitly selected clean checkout and exact commit.
2. Preserve candidate geometry and export the same categorical sequence to the reference. Native and alternate side-chain candidates must not be compared against a different independently repacked structure without explicitly labeling that additional difference.
3. Normalize PDB labels for the external tool and map back to original chain/number/insertion identities. PDB coordinate export rounds to its coordinate field precision, and reference summary pKas are reported at finite text precision. These are additional comparison limitations.
4. Require every expected physical site and check midpoint/titration convergence. Do not selectively omit failed sites to improve an error statistic. Reference-only excluded sites are recorded in the report.
5. Record per-site pKa differences and site/total charge-curve differences. The reference charge curves are independent HH curves reconstructed from its reported pKas, not microscopic protonation ensembles.
6. Review the observed discrepancy and only then save an approved baseline. Baseline checks require matching input/reference provenance, cache fingerprint, model configuration, sites and pH grid. Existing baselines are never overwritten implicitly.

Running a live report and checking finite numbers is an **integration/discrepancy test**, not a proof of scientific accuracy. A test report compared to itself exercises the baseline machinery, not an independent regression target. The analytic tests and self-generated synthetic snapshot are kept separately from actual reference data.

## Production acceptance remains open

Before using this model to select sequences, complete the real Biotite/pinned-PROPKA test runs; evaluate native and mutant structures representative of the target design domain; measure pKa and charge-curve errors; validate mutation-gradient direction against discrete rescoring; examine iteration and grid-resolution sensitivity; and measure actual GPU throughput/memory with realistic graph degrees. Any acceptance threshold should be chosen for the scientific use, not inferred from this package's unit tests.

The model has one frozen candidate conformation per identity, simplified protonation-state H bonds, a fractional mean-field coupling model and a smoothed burial eligibility gate. Its predictions have not been calibrated against experimental pKas. It should not be described as a reference-faithful PROPKA 3.0 replacement, a folding-energy model, or a demonstrated speedup over PROPKA.
