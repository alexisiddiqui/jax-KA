# 00 — Shared contract

Decisions every stage inherits. **Change them here, not in a stage doc.** A stage doc that
needs a different rule adds a row to the decision log at the bottom and says why.

Step 1 builds everything in this file. Steps 02–04 consume it.

---

## Glossary

| Term | Meaning |
|---|---|
| Partner | One side of a binary interface. May be several chains (antibody H+L is one partner). |
| States | `AB` complex; `A`, `B` each partner alone, **bound conformation retained**. No relaxation. |
| Site | One titratable group: `(complex_id, chain, resnum, icode, group)` |
| Group | jax-Ka channel name: `ASP GLU HIS CYS TYR LYS ARG NTERM CTERM` |
| ΔpKa | `pKa(AB) − pKa(free state containing the site)`. Positive = binding raises the pKa. |
| Teacher | PypKa with the locked config below |
| Tier A | pKPDB: single-structure PypKa midpoints. Output-level labels only. |
| Tier B | This project's paired teacher runs: midpoints + curves + intermediates (if exposed) |
| Interface residue | Residue heavy-atom ΔSASA > 10 Å² |
| Interface zone | Any heavy atom within 10 Å of a partner heavy atom |
| Scoring shell | Sites within 20 Å of the partner. Interface residues are the headline subset. |

---

## Selection

Two passes. Metadata first (cheap, whole universe), coordinates second (only what gets sampled).

**Metadata:** PDB biological assembly 1 + SAbDab. X-ray or cryo-EM. Total ≤1500 residues
(provisional — reset from smoke-set timings, see Jobs).

**Partner rule (v1):** exactly two partners.
- two-chain assemblies → one chain each; homomers kept and flagged `homomeric`
- SAbDab → H+L vs a single antigen chain
- anything else → reject `multi_partner`

**Coordinates:** half-sum buried area (per side) ≥ 500 Å²; ≥10 interface residues on the complex.

---

## Prep policy

Every method reads the **same heavy-atom file** per state. Hydrogen placement and protonation
are method-internal. Provenance dict stored with each structure.

| Decision | Policy |
|---|---|
| Altlocs | Highest occupancy; tie → first; record which |
| Missing backbone atoms in a residue | Reject `missing_backbone` |
| Chain gaps | Any gap whose flanking residue lies in the interface zone → reject `interface_gap`. Distal gaps allowed: record list + length; break termini are **never scored**; jax-Ka runs `gap_policy="cap"` |
| Missing side-chain atoms | Any affected residue in the interface zone → reject `interface_missing_sidechain`. Distal: complete once in prep with PDB2PQR's rebuild (already a teacher dependency), write heavy atoms only; flag `was_completed` |
| Hydrogens | Strip |
| Waters | Remove |
| Ions | Remove (PypKa default `keep_ions=False`) |
| Ligands / glycans / metals | Reject `ligand` / `glycan` / `metal` |
| Nonstandard residues | Reject `nonstandard_residue` |
| Disulfides within a partner | Keep; jax-Ka `freeze_disulfides=True` |
| Disulfides across partners | Reject `interpartner_disulfide` |
| Multiple models (NMR) | Model 1 only |

Why distal defects are tolerated: they are identical in both states, so most of their error
cancels in ΔpKa. Interface-zone defects don't cancel, so they are rejected.

jax-Ka then runs with `missing_sidechain="error"` as a check that completion worked.

### Rejection codes

Machine-readable, one row per rejected candidate: `(candidate_id, stage, code, detail)`.

```
selection:  multi_partner, size_cap, buried_area, interface_residues
prep:       missing_backbone, interface_gap, interface_missing_sidechain,
            ligand, glycan, metal, nonstandard_residue, interpartner_disulfide,
            ambiguous_disulfide, ambiguous_residue_key, cyclic_peptide, covalent_crosslink
teacher:    teacher_timeout, teacher_failed
```

`load_topology` currently raises free-text `ValueError`s; map each to a code (split the
single noncanonical error into ligand / glycan / metal / nonstandard by CCD type).

A method failing on an accepted structure is **not** a rejection. It is recorded as a
method status and shows up as coverage.

---

## States

Each state is its own file: `AB.cif`, `A.cif`, `B.cif`. Partners run in their own PB box,
never translated inside the complex box. The sites of `A` are the A-chain sites of `AB`.

---

## Teacher config (locked)

Must match pKPDB, or tier A and tier B are different teachers:

- internal dielectric 15, solvent 80, ionic strength 0.1 M
- 81-point grid, `pbc_dimensions=0`
- GROMOS 54a7, PDB2PQR with H optimisation, no ions
- temperature 298.15 K

> Verify against the pKPDB deposit manifest, not just the docs (gate G2).

**pH grid for every curve, every method:** −2 to 16, step 0.25 (73 points). Same grid as
jax-Ka's `pka_from_grid` default usage.

Store curves and, if PypKa exposes them, intrinsic pKa + site–site pair terms on **every**
teacher run from day one. Re-running the teacher later because the schema grew is the
double-back this file exists to prevent.

---

## Split

Frozen once in step 1, on the whole candidate universe after metadata selection, before
any prep or teacher run. Later stages never re-split.

1. MMseqs2 on every chain: `--min-seq-id 0.3 -c 0.8`
2. Graph: nodes are clusters, each complex is an edge between its two partners' clusters
3. Split unit = connected component. Assign components to train / val / test at 80 / 10 / 10
   by complex count
4. Forced to test: components touching any set-2 experimental system or PKAD-3 protein

**Antibody problem (check before freezing).** At 30% identity, VH and VL frameworks collapse
into a few clusters. That puts every antibody complex into one giant component, so all
antibodies would land in one split. Default fix: the antibody partner's node is the cluster
of its concatenated CDR sequence, not of its chains. Check on the real universe:
**no component may hold > 5% of complexes.** If one does, adjust and log it below.

Outputs: `split.parquet` (`complex_id, component_id, split`) plus the MMseqs2 inputs and
version.

### Pools

One generation campaign feeds every stage:

| Pool | Split | Size | Used by |
|---|---|---|---|
| Smoke set | — | ~50 FoldBench protein–protein pairs | day 1 of step 1 only |
| Set 1 | test | ~500 complexes | 01 scoring; final eval of 02/03/04 |
| Pilot | train | ~500 complexes | 02 training; satisfies 03 Part A |
| Full | train/val | ~20k | 03 Part C |

---

## Tables

Long format, Parquet. Every method writes the same `predictions` rows.

```
structures   complex_id, pdb_id, assembly, partner_A_chains, partner_B_chains,
             n_residues, homomeric, antibody, resolution, exp_method,
             crystallization_ph, provenance (json), content_sha256, split, component_id

sites        complex_id, chain, resnum, icode, group, restype, partner,
             residue_delta_sasa, functional_delta_sasa, functional_atoms_complete,
             min_partner_distance, in_interface_zone, is_break_terminus, was_completed

predictions  complex_id, state, chain, resnum, icode, group,
             method, method_version, config_sha256,
             pka, status, curve (float32[73] | null), curve_source, intrinsic_pka (null ok)

pairs        complex_id, state, site_i, site_j, w        # teacher only, sparse

rejections   candidate_id, stage, code, detail
```

`status`: `ok | out_of_range | not_titrating | not_reported | failed`.
`curve` is the protonated fraction. `curve_source`: `native | hh` (Henderson–Hasselbalch
from the midpoint, for methods that report only pKa values).

Raw ΔSASA and distances are stored. Thresholds are applied at analysis time, never baked in.

---

## Scoring (one module, used by 01–04)

- **Common site set.** Primary metrics use sites where the reference and every scored
  method return `ok` in both states. Per-method coverage is reported separately.
- **Skill** = `1 − MSE_model / MSE_null`, with `MSE_null = mean(ΔpKa_ref²)`
- **Spearman ρ** on ΔpKa
- **Sign accuracy**, only on sites with `|ΔpKa_ref| ≥ 0.5`
- **Bootstrap**: resample split components, 1000 reps, percentile 95% CI. Never resample
  sites; sites in one complex are correlated.
- **Error cancellation**: `corr(e_AB, e_free)`, with `e = pred − ref` per state
- **Structural zeros**: fraction with `|ΔpKa_pred| < 0.01` where `|ΔpKa_ref| ≥ 0.1`, per
  distance shell 0–5 / 5–10 / 10–15 / 15–20 Å

**Linkage** (Wyman):

```
ΔG_bind(pH) − ΔG_bind(pH₀) = +RT·ln10 · ∫_{pH₀}^{pH} ΔQ(pH′) dpH′
ΔQ = Q_AB − Q_A − Q_B          # total charge; equals protons taken up on binding
pH₀ = 7.0,  T = 298.15 K,  RT·ln10 = 1.364 kcal/mol
```

Trapezoid rule on the shared grid. Sign check: if binding takes up protons (ΔQ > 0), binding
must weaken as pH rises. Unit-test that before using the function.

---

## Jobs

Pattern already in `benchmarks/regress_interfaces.py` + `merge_foldbench.py`:

- one job = one complex × one method, all three states inside
- 1 core per job, many concurrent
- per-job timeout set from smoke-set timings so ≤2% time out; timeouts are logged as
  `teacher_timeout`, and accepted-vs-candidate size distributions are reported so the size
  bias is visible
- each job writes `<out>/<method>/<complex_id>.parquet` + a JSON sidecar (method version,
  config sha256, input sha256, wall time, status)
- merge validates hashes; no shared database during generation

External methods run in their own environments as subprocesses. Adapter contract:
`run(state_files: dict[str, Path], workdir: Path) -> predictions rows`.

---

## Code

New package `src/pkabench/` (provisional name), kept separate so the `jaxpropka` library
doesn't pick up PypKa / MMseqs2 / CatBoost dependencies.

| Module | Starts from |
|---|---|
| `prep.py` | `jaxpropka.topology.load_topology` (add rejection codes) |
| `annotate.py` | `benchmarks/interface_exposure.py` (generalise two chains → two partners) |
| `split.py` | new |
| `adapters/propka.py` | `jaxpropka/reference.py` |
| `adapters/{pypka,pkai,kaml,jaxka,null}.py` | new |
| `jobs.py` | `benchmarks/regress_interfaces.py`, `merge_foldbench.py` |
| `score.py`, `linkage.py` | new |

Smoke set: FoldBench protein–protein manifest (upstream URL in `regress_interfaces.py`)
over `ground_truth_1522.tar`.

---

## Day-1 gates

| Gate | Check | If it fails |
|---|---|---|
| G1 | PypKa installs (DelPhi licence), runs one smoke pair in all three states, returns curves | Stop and re-plan. There is no fallback teacher. |
| G1b | PypKa exposes intrinsic pKa + pair terms | Not fatal. Decide whether 03 calls DelPhi directly. Tier B schema keeps the columns nullable either way. |
| G2 | Teacher config matches the pKPDB deposit | Fix config before any production run |
| G3 | Each core method treats a multi-chain file as one system | Drop that method from the core tier |
| G4 | pKAI weights + training code usable for fine-tuning | 02 picks another fine-tune target |
| G5 | PDB2PQR completion keeps residue identity and heavy-atom naming | Fall back to rejecting any titratable residue with missing atoms |

---

## Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-10-03 | Reject gaps / missing side chains in the interface zone; tolerate distal ones | Distal defects are identical in both states and largely cancel in ΔpKa |
| 2026-10-03 | Noise floor (old set 3) moved to step 1b | Independent of the shared pipeline; not on the critical path |
| 2026-10-03 | Set 1 drawn from the frozen test split; one teacher campaign feeds 01/02/03 | Avoids re-running PypKa and makes 01 numbers comparable with the final model |
| 2026-10-03 | Buried-area cutoff read as half-sum (per side) ≥ 500 Å² | The original "BSA ≥ 500 Å²" was ambiguous between total and per-side |
