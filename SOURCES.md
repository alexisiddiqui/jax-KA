# Primary sources and provenance

Inspected 2026-09-30. URLs here are source records, not a claim that a particular external Git revision was executed. The legacy reference fetcher records an exact commit at execution; **no legacy checkout or approved full-structure reference baseline was obtained in the delivery environment**.

## PROPKA 3.0

- Original repository: https://github.com/jensengroup/propka-3.0
- Numerical model pKas, donor/acceptor interaction classes and distance ranges: https://raw.githubusercontent.com/jensengroup/propka-3.0/master/Source/parameters_new.py
- Effective Nov30 overrides, burial, dielectric and eligibility conventions: https://raw.githubusercontent.com/jensengroup/propka-3.0/master/Source/version.py
- Radial-volume/desolvation, Coulomb geometry, geometric H-bond primitive and backbone reorganization: https://raw.githubusercontent.com/jensengroup/propka-3.0/master/Source/calculator.py
- Residue ionization-center definitions, including ARG CZ, and default version selection: https://raw.githubusercontent.com/jensengroup/propka-3.0/master/Source/lib.py
- Original determinant assignment: https://raw.githubusercontent.com/jensengroup/propka-3.0/master/Source/determinants.py
- Original iterative coupling rules: https://raw.githubusercontent.com/jensengroup/propka-3.0/master/Source/iterative.py
- Legacy protonator: https://raw.githubusercontent.com/jensengroup/propka-3.0/master/Source/protonator.py
- Output format: https://raw.githubusercontent.com/jensengroup/propka-3.0/master/Source/output.py
- Command-line entry point: https://raw.githubusercontent.com/jensengroup/propka-3.0/master/propka.py

The original source headers identify upstream licensing separately; this repository does not bundle that code. The sparse kernels, fractional-state energy model, topology/candidate machinery, differentiation and tests in this repository are newly written. Shared numerical parameter values and primitive formulas are attributed above. The proposed mean-field coupling model is not described as original PROPKA behavior.

## Pinned pip PROPKA reference

- Repository/API entry point: https://github.com/jensengroup/propka and https://raw.githubusercontent.com/jensengroup/propka/master/propka/run.py
- The `reference` extra pins distribution version 3.5.1 for reproducible development and CI runs.
- Adapter uses `propka.run.single(..., write_pka=True)` and records the installed distribution version.
- A modern reference run is never labeled PROPKA 3.0.

## Biotite

- PDB parsing/writing and TER representation caveat: https://www.biotite-python.org/latest/apidoc/biotite.structure.io.pdb.PDBFile.html
- mmCIF/BCIF coordinate parsing, author fields and bond options: https://www.biotite-python.org/latest/apidoc/biotite.structure.io.pdbx.get_structure.html
- mmCIF writing: https://www.biotite-python.org/latest/apidoc/biotite.structure.io.pdbx.set_structure.html
- Residue boundaries: https://www.biotite-python.org/latest/apidoc/biotite.structure.get_residue_starts.html
- CCD-based intra-residue bonds: https://www.biotite-python.org/latest/apidoc/biotite.structure.connect_via_residue_names.html
- CCD residue coordinates and topology: https://www.biotite-python.org/latest/apidoc/biotite.structure.info.residue.html

The implementation uses these APIs but could not execute the molecular path locally because Biotite was unavailable. Do not mistake API verification against documentation for an integration-test pass.

## JAX

- Static loop behavior: https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html
- Custom JVP API: https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html
- Batched sequential mapping: https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html
- Safe masking / undefined-gradient pitfalls: https://docs.jax.dev/en/latest/faq.html
- Synchronization and benchmarking: https://docs.jax.dev/en/latest/benchmarking.html
- Closed-over constant behavior: https://docs.jax.dev/en/latest/internals/constants.html

## Structure fixtures

`tests/data/peptide.pdb`, `two_chains.pdb` and `two_chains_far.pdb` are **synthetic RDKit-generated ADKHE peptides**, not deposited structures or experimental benchmarks. `tests/data/FIXTURES.json` records generation details. They were generated offline with RDKit and are read by the implementation exclusively through Biotite. RDKit is not a runtime dependency.

`scripts/fetch_structure.py` can explicitly retrieve a deposited test structure from RCSB and record its input digest. It was not run successfully in the delivery environment. No protein-accuracy claim is inferred from synthetic numerical graph tests or these topology fixtures.
