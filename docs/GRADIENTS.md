# Memory-efficient sequence gradients

The default remains the dense backend with finite-iteration (`unrolled`)
derivatives. Every mode uses frozen geometry and the existing initialization,
damping and forward iteration budget. Prepare new caches and construct new
readouts outside differentiation when geometry changes.

## Opt-in CPU design preset

```python
from jaxpropka import TitrationModel, ModelConfig, EquilibriumConfig, SelectivityObjective

bound, binder, target = [TitrationModel.for_design(cache)
                         for cache in (bound_cache, binder_cache, target_cache)]
objective = SelectivityObjective(bound, binder, target, ph_grid)
result, solver = objective.value_and_grad_with_solver_diagnostics(P_binder)
# result.valid is numerical validity only. Never use an invalid gradient.
# solver.iterations, .residual, .converged, .budget_exhausted, .nonfinite: [3,H]
audit = objective.audit(P_binder)  # explicit, separate and more expensive
```

The preset selects `packed_v2`, implicit gradients, and adaptive forward solving
with damping 0.35. `ModelConfig.steps` becomes the hard iteration cap (1,024 by
default), not a mandatory amount of work. Checks begin after 128 updates and
repeat every 16 updates; two consecutive successful checks are required. The
full conditional-state residual must be below the stricter model/implicit
tolerance (`1e-6` by default). The initial state and physical equations are
unchanged; additional iterations continue the same trajectory. There is no
automatic damping change, tolerance relaxation, branch selection or derivative
fallback. Nonfinite states and exhausted budgets yield invalid diagnostics and
nonfinite gradients. A small residual at the cap is insufficient if the required
consecutive checks have not completed.

Pass `config=ModelConfig(steps=512)` to cap a preset at 512 updates, or provide
`equilibrium=EquilibriumConfig(min_steps=128, check_interval=16,
consecutive_checks=2)` explicitly. Explicit configurations are respected: a cap
below `min_steps` is rejected rather than silently increased. Adaptive stopping
requires implicit differentiation. For low-level kernels, put the policy in
`DifferentiationConfig(mode="implicit", equilibrium=EquilibriumConfig())`.

Ordinary `TitrationModel(...)` construction is unchanged: dense, unrolled,
exactly 64 iterations by default. Existing return tuple layouts are unchanged.
`model.curves_with_solver_diagnostics(ph_grid)` returns a P-only readout producing
`(CurveResult, SolverDiagnostics)` in a single solve. The objective similarly
provides `evaluate_with_solver_diagnostics(P)` and
`value_and_grad_with_solver_diagnostics(P)`; its old methods remain available.
Solver status is auxiliary data, not a differentiable observable. Grid/direct
midpoints and experimental endpoint evaluations use the same forward policy.

### Explicit branch audits

`model.audit(P, ph_grid)` and `objective.audit(P, ph_grid=None)` compare five
paths: standard, fully unprotonated, fully protonated, increasing-pH continuation
and decreasing-pH continuation. They use the supplied solver cap/tolerances.
Results contain per-path residuals, iteration counts and solver status, plus
the largest conditional-occupancy disagreement. `consistent` requires every
path to converge and disagreement no larger than `2e-5`.

For a model, path fields have shape `[5,H]` and consistency/gap fields `[H]`;
the objective adds a leading bound/binder/target axis. Only compiled readouts
are cached, never the outcome of a previous sequence's audit. Treat models and
geometry as immutable, as for other readouts.

An audit applies only to its supplied sequence, geometry and sampled pHs; it
does not certify a unique branch or global minimum. Run it at initialization,
after geometry changes, and on final discrete candidates. Auditing during
optimization remains caller-controlled. Ordinary `valid` flags report numerical
convergence and checked adjoint solves, **not branch consistency**.
Disagreement can reflect remaining finite-solve error as well as different
branches; the audit flags it conservatively rather than deciding which cause
applies. Precision changes also require a fresh audit.

The preset has passed its CPU numerical promotion gates and is recommended as
the **opt-in CPU design workflow**, with validity checks and explicit audits as
described above. Ordinary package defaults remain unchanged. The completed
[CPU report](../reports/design_preset_cpu.json) covers 182 unit tests, 260 forward
interfaces, 24 gradient interfaces, five scaling cases and four sensitive-case
regressions. It retains all convergence and audit failures: qualification does
not mean every sequence has a usable equilibrium result. See the
[measured outcomes and limitations](BENCHMARKS.md#adaptive-design-preset-cpu-proteinprotein-panel).
CUDA/MPS performance and the endpoint objective are not promoted by CPU results;
experimental calibration also remains open.

The promotion runner requires the same archived protein–protein structures and
accepted input reports as the existing benchmark suite:

```bash
python benchmarks/validate_design_preset.py manifest
JAX_PLATFORMS=cpu python benchmarks/validate_design_preset.py forward --case 111
JAX_PLATFORMS=cpu python benchmarks/validate_design_preset.py gradient --case 111
JAX_PLATFORMS=cpu python benchmarks/validate_design_preset.py scaling --case 111
python benchmarks/validate_design_preset.py summarize
```

Those per-case commands illustrate one interface, not the full gate. Run all
260 eligible forward cases, the 24 gradient cases and five scaling cases listed
in the generated manifest, each in a fresh process. Reports live separately
under `_runtime/jax-Ka/cuda12/benchmarks/design-preset/`; historical results are
not overwritten. Summaries reject missing records, stale package-source hashes,
cache mismatches and invalid scaling results. Gradient checks use float64 and
stricter solve tolerances, so their convergence classification can differ from
the float32 forward panel. All failed and excluded inputs remain visible.

## Choosing a numerical path

```python
from jaxpropka import DifferentiationConfig, ModelConfig, TitrationModel

model = TitrationModel(
    cache,
    ModelConfig(steps=128),
    backend="packed_v2",
    differentiation=DifferentiationConfig(mode="checkpointed"),
)
curves = model.curves(ph_grid)  # construct once
```

`backend="packed"` retains its existing dense local-term calculation and packs
only the repeated field contraction. `packed_v2` also constructs Coulomb,
H-bond and probability-weighted coupling terms directly on active edges. Its
runtime omits dense pair geometry and masks fixed environment kernels once on
the host. Environment contractions use rematerialized tiles of 32 residues to
bound layout-transpose workspace. The portable `StructureCache` format is unchanged; it still occupies
host memory. Packing never removes an identity because its current probability
is small or zero.

The low-level `pack_runtime(cache_or_numpy_arrays)` and
`packed_v2_curve_kernel(runtime, P, ph, config=..., differentiation=...)` APIs
are in `jaxpropka.model`. Use dynamic runtime arguments for multi-structure
compilation. Existing dense and packed kernel signatures accept the optional
`differentiation` keyword and otherwise retain their behavior. In `packed_v2`,
the `local_terms()` inspector returns a one-dimensional edge coupling array;
the other term shapes remain `[N,9]`.

| Mode | Derivative meaning | Memory behavior |
|---|---|---|
| `unrolled` | The configured finite-iteration algorithm | Retains iteration intermediates |
| `checkpointed` | The same finite-iteration algorithm | Recomputes edge intermediates; retains block/state checkpoints |
| `implicit` | The locally selected converged fixed point | Matrix-free linear solve; no forward iteration tape |

Checkpointing defaults to 16-step blocks with each update also checkpointed.
Set `checkpoint_block_size=1` for update-only checkpointing. Nondivisible
iteration counts execute the exact requested number of updates.

## Bound/free selectivity

```python
import jax
import jax.numpy as jnp
import numpy as np
from jaxpropka import SelectivityObjective

# bound, free_binder and free_target are models prepared independently from
# the same geometry, with the appropriate physical chain selections.
objective = SelectivityObjective(
    bound, free_binder, free_target,
    np.linspace(6.0, 7.4, 15),
    binder_keys=free_binder.cache.keys,
    target_probabilities=free_target.native_probabilities,
    required_log10_ratio=1.0,
    tau=0.1,
    ph_chunk_size=1,
)

P = jax.nn.softmax(binder_logits, axis=-1)
result = objective.value_and_grad(P)
# result.gradient is d(loss)/dP, in binder_keys order.
# Check result.valid before using the update.

def combined_loss(logits):
    return objective.loss(jax.nn.softmax(logits, axis=-1))

loss, logit_gradient = jax.value_and_grad(combined_loss)(binder_logits)
```

`evaluate(P)` returns loss, selectivity, numerical validity and diagnostics
without computing a gradient. `value_and_grad(P)` additionally returns the
gradient and, for implicit mode, explicit adjoint residuals. `loss(P)` composes
with an upstream sequence model and supports first-order JVPs and VJPs. **Higher
derivatives through this wrapper are unsupported**: its custom rule treats the
computed gradient as a constant. Use the underlying curve APIs when higher
derivatives are required and validate their numerical meaning separately.

The objective checks residue-key correspondence, native identities, terminal
presence and frozen covalent constraints. Matching coordinates are the caller's
responsibility: these cannot be reconstructed from a numerical cache alone.
Use `validate_probabilities(P)` outside JIT or construct P with softmax.

The isolated target is evaluated once per supported probability dtype at
construction. Bound target protonation remains responsive to binder mutations.
Reconstruct the objective if target probabilities, geometry, model settings or
quadrature change. Treat the supplied models as immutable.

The calculation accumulates trapezoidal charge integrals and term gradients in
pH chunks, then applies the softplus derivative once to the complete bound/free
selectivity. Local terms are built once per variable environment. A single
probability pullback per environment maps accumulated term gradients back to P.
The scalar wrapper retains the compact final binder gradient for its backward
pass, rather than retaining all pH chunk tapes.

Diagnostics have shape `[3,H]`, with environment order **bound, binder, target**:
`forward_residual`, `forward_valid`, `linear_residual`, `linear_threshold`,
`linear_checked`, and `linear_valid`. Unchecked linear residuals are not evidence
of a solved adjoint; consult `linear_checked`. Target adjoints are never needed.
Invalid objective gradients are NaN, including failure of a required forward
solve. The forward scalar remains available for diagnosis.

## Implicit derivatives and failure handling

For the undamped map `u = f(u, P, ph)`, the custom JVP solves

```text
(I - df/du) du = (df/dP) dP + (df/dph) dph
```

and reverse mode uses the transposed operator. Geometry arguments are explicit
to support JIT, batching and the existing midpoint custom JVP. No dense Jacobian
is constructed. GMRES uses a restart size of 32 and at most 20 restart cycles;
these are configurable. Forward and transpose solves explicitly recompute their
linear residual, requiring

```text
norm(A(x) - rhs) <= max(linear_atol, linear_rtol * norm(rhs))
```

Defaults are `implicit_residual_tolerance=1e-6`, `linear_rtol=1e-6`, and
`linear_atol=1e-8`. The objective also respects a stricter physical-model
residual tolerance. Float64 checks can use `1e-10`, `1e-10` and `1e-12`.
Zero right-hand sides return zero directly. Failed solves yield nonfinite
derivatives; no implicit-to-finite fallback, regularization or additional forward
iterations are applied automatically. Generic JAX transformations cannot return
adjoint diagnostics as auxiliary forward results; use `value_and_grad()` on the
selectivity objective when those diagnostics are needed.

A small linear residual does not establish good conditioning, a unique fixed
point or the globally preferred equilibrium. Check directional derivatives,
iteration sensitivity and branch behavior in the actual design regime.

## Experimental endpoint/envelope objective

```python
from jaxpropka.experimental import EndpointSelectivityObjective

endpoint = EndpointSelectivityObjective(objective)
result = endpoint.value_and_grad(P)
branch_audit = endpoint.audit(P)  # 129 pHs by default; intentionally separate
```

The dimensionless potential includes weighted binary entropy, intrinsic and
pair fields, and the deprotonated baseline charge term. At stationarity,
`dF/dph = total_charge`; therefore consistent-branch endpoint differences can
replace charge quadrature. Its first derivative holds converged occupancies
fixed and differentiates every explicit sequence-dependent energy term.
This has neither an occupancy backward tape nor an adjoint solve.

This API is experimental. Its `valid` flag means numerical convergence, **not
branch consistency**. `audit()` compares independently initialized curves with
increasing- and decreasing-pH continuation, and reports occupancy gaps and full
conditional-state residuals. All three environments are audited. Agreement is
not proof of a global free-energy minimum. Compare endpoint values and gradients
against refined quadrature before using the result; a nonconverged or
inconsistently selected branch invalidates the integral interpretation.

## Reproducing CPU measurements

`benchmarks/benchmark_gradients.py` prepares the three environments from the
existing protein-protein input archive and report fingerprints. Its baseline
loads the original packed model from commit
`df5abd608cbb3c92a7e1a0097ed19f38cf2d19fb` without changing the worktree.

```bash
JAX_PLATFORMS=cpu python benchmarks/benchmark_gradients.py prepare --case 111
JAX_PLATFORMS=cpu python benchmarks/benchmark_gradients.py run --case 111 --variant baseline --workload selectivity
JAX_PLATFORMS=cpu python benchmarks/benchmark_gradients.py run --case 111 --variant packed_v2 --workload selectivity
JAX_PLATFORMS=cpu python benchmarks/benchmark_gradients.py run --case 111 --variant implicit --workload selectivity
```

Run each variant in a fresh process. Cases 111, 120, 6 and 43 correspond to
176, 289, 500 and 1,254 residues. The optional stress case is 171 (1,708 residues).
Use `--steps 512`, `--points 2`, or `--chunk-size` for the additional scaling axes.
The benchmark uses one warm-up and five synchronized repetitions for each of
the two existing soft mixtures. Compilation, preparation, compiler temporary
memory and process peak RSS are recorded separately. RSS includes host cache,
compiler and initialization memory; compiler temporary storage does not.

`regression --case N` repeats the archived sensitive schedules for cases 52,
100, 117 and 212. These compare against the original **packed** backend; historical
dense-versus-packed discrepancies remain separately identified. No PROPKA or
monomer panel rerun is needed. `summarize_gradient_memory.py` aggregates saved
results, includes missing work explicitly, and compares finite-iteration parity
separately from implicit and endpoint validity.

The measured CPU panel and its remaining memory limitations are summarized in
[BENCHMARKS.md](BENCHMARKS.md#memory-efficient-gradients-cpu-proteinprotein-scaling),
with machine-readable results in
[gradient_memory_cpu.json](../reports/gradient_memory_cpu.json). The required
four-case panel passes. The optional 1,708-residue case is nonconvergent in the
original and new code at 128 steps, so its implicit gradients are deliberately
unusable and aggregate all-case acceptance remains false.
