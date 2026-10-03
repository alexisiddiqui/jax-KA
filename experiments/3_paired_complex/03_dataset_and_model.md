# 03 — Generate the paired dataset and train the model

**Timebox:** 3 days (0.5 pilot, 1.5 generation on CPU in background, 1 training)
**Hardware:** 600+ CPU node for generation; 3090 for the main run; 4× A40 for HPO
**Depends on:** curation pipeline from 01

---

## Part A — Pilot (500 complexes, half a day)

Run the full chain end to end before spending the big compute:

```
select → prep → two states → PypKa → dump intermediates → pair → CatBoost on intrinsics
```

Every failure mode shows up in the first few hundred: interface glycans, metals, missing
CDR side chains, altloc ambiguity, assemblies that aren't really assemblies. Each one
changes what you store, which is why the pilot comes before the schema freezes.

**Exit criteria:** ≥70% accept rate, intermediates parse, CatBoost trains, schema stable.
If accept rate is under 50%, fix curation before scaling — you'll otherwise burn
1.5 days producing a biased subset.

---

## Part B — Dump the PB intermediates

**This is the highest-leverage decision in the project.**

PypKa computes per-site intrinsic pKa's and a site–site interaction matrix, then runs
Monte Carlo over them. Those intermediates (`.pkint` / `.g`-style files feeding the MC
step) are direct supervision for exactly the two tensors the network predicts.

For a 450-residue complex with ~100 titratable sites:
- midpoints only: ~100 numbers
- intrinsics + pairs: ~100 + ~5,000 numbers

**~50× the signal for the same PB solve.** And the gradient path is one layer deep
instead of backpropagating a scalar through the whole fixed-point solve.

> **Verify PypKa exposes these before committing to the full run.** If it won't, you may
> need to call the underlying DelPhi step directly. Settle this during the pilot.

---

## Part C — Full generation

### Scale and budget

| | |
|---|---|
| Complexes | ~20,000 after curation losses |
| States each | 2 (complex, rigid-separated) |
| Core-hours | ~10,000 (≈0.5 core-h per Fv-scale triple) |
| On 600 cores | ~17 h |
| Ionic-strength subset | 1/3 of complexes × 3 values (0.05 / 0.15 / 0.5 M) → +~7,000 core-h |
| **Total** | **~17,000 core-h ≈ 28 h wall** |

Comfortably inside 1.5 days. Headroom exists; don't spend it on MD.

### Scheduling

Run **1 core per job, 600 concurrent jobs**, not 16-core jobs. PypKa's internal
parallelism is for latency; you want throughput. A 450-residue complex takes ~20 min on
one core — irrelevant for throughput, but set a **per-job timeout (60 min) and let the
tail die**. A handful of pathological structures will otherwise hold the whole run.

Checkpoint per job to its own file. No shared database during generation; merge after.

### Conditions

Everything must match pKPDB's defaults or tiers A and B are different teachers:
internal dielectric 15, solvent 80, ionic strength 0.1 M, 81-point grid,
`pbc_dimensions=0`, GROMOS 54a7, PDB2PQR with H optimisation, no ions.

> Verify these against the pKPDB deposit manifest, not just the docs. Note that
> εᵢₙ = 15 is high and implicitly absorbs conformational relaxation — it compresses
> shifts toward model values, so the teacher systematically under-predicts large shifts.

### Schema

Per site, per state, per condition:

```
chain, resnum, icode, restype,
intrinsic_pka, pair_row (sparse), midpoint_pka,
charge_curve[pH −2:16 step 0.25],
dsasa, interface_flag, convergence_flag, valid_flag
```

Per structure: coordinates, prep provenance, teacher version, content hash, and
**covariates stored but unused in v1** — crystallization pH from PDB metadata,
resolution, method. These let you test later whether predictions drift with the pH the
structure was solved at.

Do **not** precompute heavy features. You have the CPUs and it's tempting, but a
coordinate model wants raw coordinates plus local frames, and frozen features will
constrain architecture choices you haven't made. Featurise on the fly; precompute only
for the CatBoost baseline.

Shard by sequence cluster so splits are file-level.

Disk: pair matrices ~2–3 GB; coordinates dominate at a few hundred GB.

### Splits

MMseqs2 at 30% identity on **both** partners, split by cluster *pair*. Holding out only
the antigen leaks badly on antibody sets where frameworks repeat.

Held out and never touched until the end:
- tier-B clusters
- PKAD-3, cluster-decontaminated against pKPDB
- experimental ΔpKa and pH-dependent affinity from 01's set 2

---

## Part D — Model

### Shape

Siamese: one encoder, two passes, subtract. Same weights. (Rationale in 02.)

Sanity invariant, as a diagnostic not a loss: ΔpKa(A→AB) must equal −ΔpKa(AB→A).
It's automatic if the architecture is right, so drift means a bug.

### Encoder

- Invariant features in local N–CA–C frames (not equivariant vector channels — simpler,
  and gradients still reach coordinates through frame construction)
- 4–6 layers, d=128, 8 heads → ~1–3M params
- Sparse radius-graph attention; **query tokens only at titratable sites** (~25% of residues)
- Guard Gram–Schmidt frame construction against degenerate triples
- Soft `P[N,20]` sequence representation end to end — no argmax, no rotamer search

### Output head — residuals on physics, bounded

```python
intrinsic = model_pka[restype] + SCALE * tanh(head_i(h))      # starts at "no shift"
W_raw     = debye_huckel_baseline(coords, I) * (1 + head_w(h_pair))
W         = 0.5 * (W_raw + W_raw.T); W = W.at[diag].set(0)
```

The DH baseline supplies the correct distance decay for free; the network learns only
the deviation. Keeps gradients small and well-scaled, and an untrained net starts in a
physically sane place.

### Solver

Keep jax-Ka's relaxation. Three changes:

1. **Mean-field + exact small clusters.** Mean-field is genuinely wrong for strongly
   coupled dyads (the classic Asp/Glu pairs). Partition sites by coupling strength,
   enumerate exactly within clusters of ≤10–12 sites via `logsumexp` over 2^k states
   (perfectly smooth), mean-field between clusters. Compute the partition from a
   **detached** coupling matrix so the assignment itself carries no gradient.

2. **Implicit differentiation of the fixed point.** `lax.custom_root` or custom VJP:
   `dx/dθ = (I − ∂F/∂x)⁻¹ ∂F/∂θ`, solved with CG. Constant memory, no pathology from
   variable iteration counts. Add small Tikhonov regularisation — the system is
   ill-conditioned near strongly coupled transitions.

3. **Midpoints by implicit root**, not grid interpolation. The current grid-midpoint
   approach has gradient kinks when the crossing moves between intervals.

### Smoothness checklist (gradients flow to coordinates)

- Cosine/sigmoid switching over a 2 Å window — no hard distance cutoffs
- Fixed-k neighbours from a **detached** index computation; smooth weights carry the gradient
- `sqrt(r² + ε)` everywhere; never `norm()` at zero
- SiLU/GELU, not ReLU, if you want second derivatives for design later
- **fp32.** FlashABB had to run fp32 because the expanded-form distance trick cancels
  catastrophically in bf16; your pair matrix is distance-dependent and inherits this.

### Losses, in priority order

1. intrinsics + pair matrix (direct, tier B)
2. charge curves through the solver — **primary solver-level loss**, since the curve is
   smooth in the parameters while the midpoint is a root and is badly conditioned where
   curves are flat (buried, weakly titrating sites)
3. midpoints
4. ΔpKa between states — **ramp this up over training**; it's the smallest-magnitude
   signal and will otherwise be ignored

Train on **shifts from model values** (3.7 Asp, 4.2 Glu, 6.5 His, 8.5 Cys, 9.5 Tyr,
10.4 Lys), not absolutes. The null model already gets RMSE ~1.36 acid / 1.04 base, so
absolute regression spends capacity on residue identity and inflates R². Normalise per
residue type.

### Schedule

- Tier A pretrain (pKPDB, output-level only — no intermediates available) → ~12 h on 3090
- Tier B fine-tune (paired, intermediate-level) → ~12 h
- Calibrate on PKAD-3
- HPO on the 4× A40s: layers, d, loss weights, cluster size cap, DH baseline screening length

### Evaluation

Everything from 01, run identically, plus the linkage readout
`ΔG_bind(pH) − ΔG_bind(7) = +RT·ln10·∫_7^pH [Q_complex − ΣQ_free]dpH′` (definition in
`00_shared.md`) against experimental pH-dependent
affinity.

**Must beat the delta-learning CatBoost from 02.** If it doesn't, report that honestly;
the differentiability is still a contribution but the accuracy claim isn't.

---

## Outputs

```
data/tierB/            # sharded by cluster
results/model/
  checkpoints/
  eval_vs_benchmark.csv
  linkage_validation.png
  ablations.csv
```

## Scope of the claim

pH-dependent titration and binding linkage **from a fixed conformation**. Not
pH-dependent conformational change. Histidine-switch and endosomal-release cases are
expected failure modes — say so in the paper rather than letting a reviewer find it.
