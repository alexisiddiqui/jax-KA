# 02 — Fine-tune an existing model on paired ΔpKa

**Timebox:** 1 day (a few hours on the 3090, rest is analysis)
**Hardware:** 1× 3090
**Runs concurrently with:** 03's dataset generation on the CPU node. No contention.

**Purpose is diagnostic, not a product.** This answers one question: *is the gap found
in 01 a training-signal problem or an architecture problem?* The answer determines how
much of 03 is actually necessary.

---

## Which model

**pKAI.** Reasons:

- Smallest and fastest; fine-tunes in minutes, so you can run the full ablation grid
- Trained on pKPDB, so its prior matches the tier-A distribution you'll use in 03
- Predicts a per-site value from a local environment — the simplest possible
  parameterisation, which makes it the cleanest test of whether paired supervision alone
  is enough
- Permissive tooling, no web-service rate limits

**KaML-CBtree as the second arm.** Not fine-tuned — retrained from scratch on paired
features (see Delta-learning below). It is cheap enough that there's no reason not to,
and given CatBoost beat the GAT on absolute pKa it may well beat everything here.

Do **not** pick DeepKa (web-server-mediated) or KaML-GAT (heavier, worse than the trees
on the task it was designed for).

---

## Setup

### Siamese wrapper

```
ΔpKa_pred = f_θ(complex, site) − f_θ(separated, site)
```

Same weights, two passes, subtract. Not a two-tower model with a difference head.
The gradient of the difference is the difference of gradients, so any error the encoder
makes identically in both states contributes nothing to the loss. Error cancellation
becomes a training-time property, not just a hope.

Partner presence enters as input content only — the free state is the identical
structure with the partner's atoms removed. No architectural flag, no extra pathway.
The only input difference between the two passes is the thing being measured.

### Data

Use the 500-complex pilot from 03 (available end of day 1 of that track) for the
initial run, then the full set when it lands. ~500 complexes × ~40 interface sites
≈ 20k paired examples is enough to answer the diagnostic question.

Split by sequence cluster pair at 30% identity. Never by PDB ID.

### Loss

Huber on ΔpKa, normalised per residue type. Upweight |ΔpKa| > 0.5 — the bulk of
near-zero surface sites will otherwise dominate and you'll learn the null model with
extra steps.

---

## Arms to run

| Arm | What it tests |
|---|---|
| Frozen pKAI, siamese, no training | Reproduces 01's result; sanity check |
| Fine-tune last layer only | Is a recalibration enough? |
| Fine-tune all layers | Full capacity of the existing parameterisation |
| Train pKAI architecture from scratch on paired data | Does pretraining help at all here? |
| CatBoost delta-learning (below) | The baseline to beat |

---

## Delta-learning baseline — build this, it may win

Do not fine-tune anything. Predict the *residual* of the physics.

```
target = ΔpKa_true − ΔpKa_PROPKA3     (or − ΔpKa_PypKa)
```

Features, all computed at the site from the two-state pair:

- ΔSASA of the site on binding
- change in burial depth / coordination number
- new salt bridges across the interface (count, distances)
- new H-bond donors/acceptors within 4 Å from the partner
- change in local net charge within 6 / 10 Å
- change in local dielectric-ish proxy (heavy-atom density)
- residue type, one-hot
- distance to interface centroid

CatBoost, separate models for acids and bases (KaML found this materially helps).

This inherits the physics' sign and magnitude behaviour, trains in minutes on CPU, needs
no GPU, and sets the bar. **If the transformer in 03 cannot beat this, the transformer
has not earned its place.** Report it as a first-class method, not an afterthought.

---

## Decision rules out of this section

| Observation | Action in 03 |
|---|---|
| Fine-tuned pKAI goes from zero skill to good skill | Gap is training signal. Proceed with 03 as planned; expect it to work. |
| Fine-tuned pKAI plateaus well below delta-learning | Single-site parameterisation is the limit → the intrinsic/pair factorisation is load-bearing. Strengthens the paper. |
| Delta-learning beats everything by a wide margin | Seriously consider shipping it as the headline method and the neural model as the differentiable alternative. Be honest about this. |
| From-scratch ≈ fine-tuned | pKPDB pretraining transfers little to ΔpKa → in 03, weight tier B over tier A. |

---

## Outputs

```
results/finetune/
  arms.csv                  # skill score, Spearman, sign acc per arm
  deltalearn_model.cbm
  feature_importance.png
  decision.md               # which branch of the table above fired
```

## Non-goals

No titration curves, no pH-dependence, no linkage integral, no coupled sites. These
models cannot produce them — that is precisely why 03 exists. Don't try to retrofit it.
