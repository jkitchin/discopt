# Changelog

All notable changes to discopt are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The release procedure that produces these entries is documented in
[`RELEASE.md`](RELEASE.md).

## [Unreleased]

### Fixed

- **The McCormick LP downgrade told users to pass the flag they had just passed**
  (`relaxation`, #1112). A model whose nonlinearity lives inside a
  `dm.custom`/`CustomCall` node has no *lifted* relaxation — the body is opaque to
  the DAG walker by construction — so `has_relaxable_nonlinearity` is False, the LP
  relaxer is discarded, and the issue-#120 soundness guard demotes `"nlp"` to
  `"none"`. That demotion is correct and is unchanged. Its *message* was not: it
  advised `Use mccormick_bounds='lp'` to callers who had passed exactly that, and
  it fired even when the reduced-space engine was about to supply a perfectly good
  spatial bound, where there is no fallback to announce. The warning is now
  suppressed (`debug`, with an accurate one-liner) whenever reduced-space bounding
  is forced; every other caller sees the original text verbatim.

  **Two of the issue's own claims are falsified and recorded here rather than
  quietly dropped.** (1) Section (b) states "the node bound it actually gets is
  alphaBB". Measured on an MCBox-traceable CustomCall model with
  `mccormick_bounds="lp"`: the reduced engine is active and bounds 0.5413 against
  `f(0.5, 0.5) = 0.6065` at an independently-checked feasible point — a real
  spatial bound, not an alphaBB-grade one. (2) The `discopt.nn` rows in the issue's
  table are a *different bug*: a tanh network reports
  `has_relaxable_nonlinearity is True` and never reaches the downgrade at all
  (its loose bound is the missing spanning-box S-envelope —
  `_curv_by_sign` returns `None` on an inflection-spanning box and `_emit_1d`
  emits zero rows). `test_1112_downgrade_message.py` pins that distinction so the
  two are not conflated again.

  A branch explaining the opaque cause to a *non*-reduced-space caller was written
  and then removed as dead code: a non-admissible `CustomCall` model returns early
  via `_withhold_local_optimality_certificate` and never reaches the guard, so no
  `CustomCall` model can arrive there without `_force_reduced_space`. That
  reachability fact is itself pinned by a test.

  Not addressed, and tracked separately: alphaBB is still enabled alongside the
  reduced engine because `_use_alphabb` keys on `_mc_lp_relaxer is None` and is
  computed before `_reduced_space_active`. Suppressing it is bound-changing (if the
  node bound is currently the better of the two, removing alphaBB can loosen it) and
  needs the §5 flag-and-panel treatment.

- **`sqrt`/`asin`/`acos`/`acosh` derivatives no longer divide by zero at a singular
  endpoint** (`relaxation`, #1111). `_dsqrt` and friends evaluate the derivative
  *limit* explicitly instead of computing `1/0`, so the
  `RuntimeWarning: divide by zero encountered in scalar divide` is gone at the
  source rather than suppressed; returned values are bit-identical. Separately,
  `spatial_producer` now declines a zero-touching `sqrt` box: the Rust mirror
  (`bnb/mccormick_patch.rs::univariate_rows`) guards `f` but not `f'` and would
  have written `f'(0) = inf` and a `NaN` intercept into the node LP.

- **Presolve reached its iteration cap on every model with a bound-redundant
  row** (`presolve`, #1053). `SimplifyPass` is `PassCategory::BoundsOnly` — it
  holds `&ctx.model` and cannot remove anything — yet it stamped the rows it
  proved *redundant* into `PresolveDelta::constraints_removed`, which
  `made_progress()` reads as a model change. The rows stayed in the model, the
  next sweep re-derived the identical list, and the `NoProgress` break became
  unreachable. `StructureManifest::implications` had the same defect. Detections
  now go to the new `StructureManifest::redundant_constraints`, which is
  reported but never counted as progress, and a `debug_assert` in the
  orchestrator rejects any `BoundsOnly` pass claiming a model change it cannot
  have made. Measured standalone on MINLPLib `hda` at a 30 s presolve budget:
  16 sweeps / `IterationCap` / 22.68 s before, 6 sweeps / `NoProgress` /
  9.55 s after, with an identical reduction. (Standalone, not end-to-end —
  the in-solve figure is under the next entry, which had to land first.)

- **Presolve counted last-bit numerical noise as a tightening and never reached
  its fixed point** (`presolve`, #1053). `count_tightened` — the single choke
  point all seven bound passes route `bounds_tightened` through — compared
  endpoints with no tolerance, so a bound converging asymptotically reported
  progress forever. Fixing the redundancy defect above was not enough on its
  own: `hda`'s in-solve presolve still ran to `TimeBudget` at 10 sweeps and
  15.00 s, exactly its `0.25 × time_limit` cap. With `max_iterations` pinned,
  sweeps 5..14 moved the returned bound vector by at most **4.0e-14** across
  4650 endpoints while reporting 3-9 "tightenings" each. Movement is now
  ignored below a relative `TIGHTEN_PROGRESS_TOL` of 1e-9 — three orders below
  `FEAS_TOL`, five above the observed noise — scaled by the smaller of the two
  magnitudes so an unbounded endpoint becoming finite always counts. The
  tightening itself is still applied and returned; only the progress *signal*
  is withheld, so presolve stops with a box looser by at most the tolerance,
  which is the safe direction. In-solve on `hda` (782 vars, 778 cons):
  15.00 s / 10 sweeps / `TimeBudget` → 8.87 s / 5 sweeps / `NoProgress`.

- **Convexity classification spent its entire time budget on LPs that answered
  nothing** (`_relax/convexity`, #1053). `_refine_sign` needs only the *sign* of
  an affine expression, but called `LinearContext.affine_range`, which solves a
  POUNCE LP pair whenever the model has any linear row. `affine_range`
  intersects its LP result with the free box enclosure, so a strict box sign is
  already the final answer and the LP cannot change it. The new
  `LinearContext.affine_sign` short-circuits on that case — exact, not an
  approximation. On `hda` all 12 calls were box-conclusive and cost 12.66 s,
  which overran the 12 s classification budget (`0.2 × time_limit`) and aborted
  the classification entirely: three `ConvexityBudgetExceeded` fallbacks to
  convexity-unknown. Classification now completes in 0.33 s and identifies 567
  of 718 constraints as convex.

- **The BARON head-to-head scored a false infeasibility claim as an ordinary
  miss** (`benchmarks`, #1053). A solver asserting "infeasible" on an instance
  with a published finite optimum is making a false claim; `classify` returned
  `n/a`, indistinguishable from an honest out-of-time result. It is now a
  `VIOLATION` for either solver, with its own report section. Diagnosed on
  `hda`: BARON 25.12.10 under the full CMU floating-network license returns
  `19 Infeasible - No Solution` in 0.27 s ("Problem solved during preprocessing
  / Lower bound is infinity"), while the same build with bound tightening
  disabled (`LBTTDo 0 / OBTTDo 0 / TDo 0 / MDo 0`) returns the published optimum
  -5964.5341 — so the fault is BARON's preprocessing, not the GAMS driver as
  #1053 supposed.

- **macOS and Windows wheels covered only Python 3.12** (`fix(release)`, #1056,
  closes #1055). `pyproject.toml` declares `requires-python = ">=3.10"`, but the
  release workflow's macOS and Windows jobs passed no interpreter list, so maturin
  built against the runner's `setup-python` version alone. v0.8.0 published 11
  wheels covering 4 of 12 platform/version combinations off Linux; `pip install
  discopt` on macOS or Windows under 3.10, 3.11 or 3.13 fell through to the sdist
  and failed on a missing Rust toolchain, which reads as a broken package rather
  than a missing wheel. Present since at least v0.7.0, whose file list has the
  identical shape.

  Fixed with `abi3` rather than by extending the interpreter matrix: pyo3 gains
  `abi3-py310`, so **one** wheel per platform serves every supported interpreter.
  The artifact count drops 12 → 6 and the build matrix shrinks instead of
  quadrupling. Verified by building a single `cp310-abi3` wheel and solving the
  same model on 3.10, 3.11, 3.12 and 3.13 — identical objective to the bit on all
  four.

  The stable ABI routes the Python↔Rust boundary through the limited API, and this
  solver crosses that boundary per B&B node, so the cost was measured rather than
  assumed: 5 interleaved reps over a 6-instance panel plus 3 reps on
  `clay0303hfsg`. Node counts are identical on every instance (bound-neutral, per
  `CLAUDE.md` verification regime 1); wall-clock difference is within noise. Full
  numbers in #1056.

- **The release could publish an incomplete artifact set with every job green**
  (`fix(release)`, #1056). Nothing checked *what* was built, only that building
  and uploading succeeded — which is why the gap above survived a release.
  `.github/scripts/check_wheel_coverage.py` now runs on the collected artifacts
  before the PyPI upload and fails if a platform is missing, if a wheel is not
  `abi3`, or if the `abi3` floor has drifted from `requires-python` (bumping one
  without the other would publish wheels claiming support they were never built
  against). The guard was tested against the real v0.8.0 file list, which it
  rejects, and against this release's, which it accepts.

- **`discopt install-skills` crashed on Python 3.10** (`fix(skills)`, #1056).
  `discopt/skills/__init__.py` imported `Traversable` from
  `importlib.resources.abc`, which exists only on 3.11+, so `import
  discopt.skills` raised `ModuleNotFoundError` on the 3.10 floor
  `requires-python` promises. The name is used only in annotations and the module
  already has `from __future__ import annotations`, so the import moved under
  `if TYPE_CHECKING:` with a `sys.version_info` split (3.10 reaches the same class
  as `importlib.abc.Traversable`). Found while fixing the wheel gap above: until
  the `abi3` change, no macOS or Windows user could install on 3.10 at all, so a
  3.10-only crash could not surface there.

- **The correctness tier could not be collected on Python 3.10** (`fix(tests)`,
  #1056). `python/tests/_optima.py` — the reference-optima oracle shared by every
  soundness suite — did a bare `import tomllib` (3.11+), making six test modules
  collection errors on 3.10. It now falls back to `tomli`, matching
  `discopt/profiles.py`, and raises a pointed error if neither parser is present
  rather than degrading to an empty registry, which would silently turn every
  soundness assertion into a no-op. `tomli` is now declared in the `dev` extra
  under a `python_version < "3.11"` marker.

- **Nothing checked that shipped code runs on the oldest supported Python**
  (`test`, #1056). Both 3.10 breaks above survived because all seven CI jobs run
  3.12. `python/tests/test_requires_python_floor.py` AST-scans `python/discopt/`
  and `python/tests/` for module-scope imports of stdlib modules or names newer
  than the `requires-python` floor, ignoring those guarded by `try`/`except`,
  `TYPE_CHECKING`, or a `sys.version_info` branch. It runs on any interpreter, so
  it does not need a 3.10 job to catch the next one. A companion test feeds the
  scanner a known violation plus three correctly-guarded imports, so a broken
  scanner cannot pass vacuously.

## [0.8.0] - 2026-08-16

### Changed

- **feral bumped to `0.16.0`; the sparse LU now pivots with threshold Markowitz**
  (`deps`, #1008). `0.16.0` is a breaking release, and its one breaking item is
  the reason to take it: `SparseLu::factor` chooses its column order *during*
  factorization (feral #171/#172) instead of consuming an AMD-on-AᵀA ordering and
  factoring with Gilbert–Peierls. discopt keeps the new default at both call
  sites; `LuPivoting::GilbertPeierls` would restore the old rule.

  Why: #1008 attributed 72.6% of LP wall to `LuNumeric` and named **fill** as the
  lever. Markowitz attacks fill directly — upstream's 16-basis corpus reports
  `factor_nnz/nnz(B)` geomean **2.77x → 1.06x**, never worse on any basis, faster
  on 15 of 16, best case 1066.64 ms → 10.98 ms.

  The upgrade is safe on the release's silent-substitution hazard: Markowitz
  ignores the `symbolic` argument, so a caller that passes a deliberately chosen
  ordering has it discarded without warning. discopt passes a throwaway
  `SparseLuSymbolic::analyze` at both sites, asserts nothing on `reach_visits()`
  or `used_dense_bump()`, and pins no pivot row or permutation; `FeralLU::params()`
  builds with `..LuParams::default()`, so the new `LuParams::pivoting` field is
  not a build break. The audit is `docs/dev/performance-plan.md` §18i.

### Added

- **LP refactorization attribution, and the measurement it settled** (`perf(lp)`,
  #1008). Three counters — `DualRefactorizations`, `DualRefacCap`,
  `DualRefacFtFail` — plus `DISCOPT_LP_REFAC_INTERVAL`, which exposes the update
  cap both simplex loops previously hardcoded as a literal `48`. Unset is **48**,
  so every solve is byte-identical to the previous engine; an unparseable value is
  refused loudly rather than read as the default (an A/B arm that silently reads
  as the baseline makes a harness measure it twice).

  The dual loop's refactorizations were previously invisible: the existing
  `Refactorizations`/`Refac*` counters are incremented only by the primal, so an
  LP the dual solved outright reported 100+ `LuSparseFactorizations` against zero
  refactorization events.

  What they measured, over captured relaxation LPs: **`LuNumeric` is 72.6% of LP
  wall and pricing is 1.5%** (`PriceSweep` alone 0.1%), so #1008's cost is the LU
  and not the PRICE loop. And raising the interval — the fix that attribution most
  obviously suggests — is a **regression**: 48 → 100+ gives +19% factorizations
  and **+312% factor nonzeros** (fill 5.32x → 13.7x), because above ~100 the cap
  stops firing and the product-form update runs to its own stability limit
  instead, landing the search on denser bases. The 48-update cap is load-bearing.
  No default moved. Details and the falsification record in
  `docs/dev/performance-plan.md` §18f.

### Measured (no behaviour change)

- **Recovering the dropped vertical-tangent facet is sound but not helpful**
  (#1111, `DISCOPT_SINGULAR_TANGENT`, default **off**). Where `f` is finite at a box
  endpoint but `f'` diverges there, `_emit_1d` silently dropped the tangent facet,
  leaving the envelope one-sided. Re-anchoring it at an interior ladder point only
  ever *adds* a row, so the flag-ON polytope is a subset of the flag-OFF one and the
  node LP bound can only improve. The §5 panel is **cert-clean** and **fails
  net-positive: the eager anchor is measured harmful.** On `tspn08` — the only
  instance that terminates inside a deterministic node budget and moves at all —
  the tree grows 135 → **191 nodes (+41.5 %)** to buy a bound gain in the 11th digit
  (290.56592504129753 → 290.56599569540646). `mathopt5_6` is flat at 5 → 5 with a
  bit-identical bound and `kriging_peaks-full010` is flat to 14 digits. Soundness
  held throughout: 0 violations, no bound above its `.solu` reference, no
  certification regression, incumbents identical.

  The subset property is not contradicted — each node's ON bound *is* ≥ its OFF
  counterpart — but a different LP vertex changes the branching choice, so a tighter
  relaxation can still grow the tree. Nor does the root win survive branching:
  `kriging_peaks` gains 13–40× at the root (`full200` −74356.93 → −5596.46), yet
  after a few hundred nodes the arms agree to 14 digits. B&B was already recovering
  that bound cheaply.

  The `DISCOPT_CUT_INHERIT` outcome. Note that the two #581 precedents in
  `solver_tuning.py` *removed* such flags rather than leaving them in default-OFF
  limbo; this one is retained by owner decision as a measurement lever for **#1115** —
  whether any formulation of the singular-endpoint tangent pays for itself — with the
  failing panel written into the field's docstring. #1115 carries the same disposition
  rule: if the successor mechanism is falsified too, the flag gets removed.
  Flag-OFF is byte-identical to the pre-#1111 relaxation.

  **Double retraction (§11) — a retraction that was itself wrong.** An earlier
  revision of this entry, of the field docstring, and of PR #1113 *withdrew* the
  `tspn08` 135 → 191 result and recorded the flag as "sound, and neutral", on the
  strength of a panel showing 135 → 135 with bit-identical bounds. **That withdrawal
  is itself withdrawn and the +41 % is reinstated**, as measured above. The panel
  behind it never fired.

  The instrument defect, measured not inferred: `solve()` is wrapped by a decorator
  (`solver.py:6274-6280`) that begins `_set_tuning(kwargs.pop("tuning", None))`, and
  `solver_tuning.set_current(None)` publishes a **fresh, env-resolved**
  `SolverTuning()`. A probe that installs a tuning with `set_current(...)` and then
  calls `m.solve()` without `tuning=` therefore has that context **silently
  discarded at the solve boundary**. Counting `_interior_tangent_point` calls inside
  a full solve of `kriging_peaks-full010`: off 0 calls / 311 nodes; `set_current`
  **0 calls** / 311 nodes; `m.solve(tuning=...)` 2671 / 313; `DISCOPT_SINGULAR_TANGENT=1`
  2671 / 313. `tuning=` and the environment variable agree bit-for-bit, and
  `set_current` is inert — so "both arms bit-identical" was never a neutrality
  result, it is the signature of a probe that never fired, which is also exactly why
  the arms agreed to the last digit. That panel carried no drops counter, so nothing
  caught it (§6). The root-relaxation measurements below are unaffected: they drive
  `build_uniform_relaxation` directly, where `set_current` does reach (instrumented
  10/10).

  The reinstated number was re-measured on a corrected instrument — `tuning=`
  delivery, the ON arm asserted to have fired, a `max_nodes` budget with **no**
  `time_limit` (a wall limit changes the kernel path and makes the run
  non-reproducible; `max_nodes` alone is bit-reproducible over 3 repetitions per
  arm), both arms in one process on one binary — and reproduces at 135 → 191 on
  every run since.

  Still retracted, for reasons unrelated to that defect: the first panel's
  time-limited rows (`kriging_peaks-full100` −1.144, `tspn12` −0.520, …), taken at a
  60 s wall limit under load 37–87 on 14 cores, fail the §9 load gate outright; and
  the harness's printed tally `better 10 worse 2 unchanged 1`, whose classifier
  scored `nodes_on < nodes_off` as "better" — correct for a terminating run,
  backwards for a time-limited one, where fewer nodes means the arm did *less* work
  in the same budget. And #1111's own motivating hypothesis is falsified — on the
  adiabatic LVC MECP model the facet is recovered but the root bound and node count
  are unchanged (1 node either way).

  **Follow-up measurement: the anchor is not the problem, the mechanism is.** Two
  more of #1111's premises were tested and failed, and a candidate replacement was
  built and rejected on measurement rather than on argument:

  * *Corpus scope.* Across all 1610 MINLPLib `.nl` instances, `sqrt` appears in 63
    and `asin`/`acos`/`acosh` in **zero** — the opposite of the issue's expectation
    that the inverse-trig rows carry the weight.
  * *The first panel was ~36 % diluted* (see the retraction above). The targeted
    successor panel was rebuilt by screening every corpus instance for an
    actually-dropped facet,
    using two complementary instruments — a cheap root screen and a full-solve
    census — because the root screen alone is structurally blind to the `elec*`
    family, whose singular boxes only appear after branching.
  * *A fixed-offset anchor is strictly worse.* `delta = width/8` was implemented and
    measured against the shipped κ-capped ladder: about **half** the root-bound gain
    on every instance where the bound moves (`kriging_peaks-full010` +1513 vs +2975,
    `full200` +34967 vs +68769). It was discarded. The argument that motivated it —
    that `width/8` minimises *mean envelope slack* over the box, by 3.31× — is
    retracted: mean slack does not predict bound tightness, because only slack where
    the relaxation binds matters, and on this family the LP optimum sits at the
    singular endpoint.
  * *No constant is right.* Sweeping the offset over eight orders on the three
    `kriging_peaks-full` instances that move the root bound, the gain rises toward
    the endpoint, peaks near `delta ~ 1e-5`, then **collapses to zero** by `1e-12`;
    on an isolated `min 3x - sqrt(x)` atom the peak is orders of magnitude further
    out. Both ends of the range are degenerate and the optimum is problem-dependent,
    so a static geometric anchor cannot be right for every box.

  This does give `singular_tangent_kappa` the measured justification it previously
  lacked — the cap's stated rationale (outward-rounding slack growing with slope) is
  real in direction but ~1e-14 in magnitude against a ~0.35 cut depth, and is
  retracted; what the cap actually does is keep the ladder off the degenerate end,
  adaptively per box. A regression test now pins that: an effectively uncapped κ
  still emits the facet but buys < 1e-6 of root bound, where κ=100 buys > 1e-3.

  The direction the evidence points is to stop placing the tangent geometrically and
  place it *where the LP binds* — lazily at the incumbent LP point, as
  `MccormickLPRelaxer._separate_convex`'s Kelley loop already does for composite
  convex/concave lifts — which also stops the solver paying for an eager extra row at
  every node whether or not that node's LP is near the singular endpoint. That cost
  is exactly what the reinstated `tspn08` regression charges, so it is evidence, not
  merely an argument. Built and measured as #1115, below.

- **Placing the vertical-tangent facet where the LP binds removes the eager
  anchor's regression and tightens the bound on part of the corpus — but costs
  25-54 % where it tightens nothing, so it stays default-OFF** (#1115,
  `DISCOPT_SINGULAR_TANGENT_LAZY`, default **on**, consulted only when
  `singular_tangent` is on — so the shipped default path is unchanged).
  `MccormickLPRelaxer._separate_singular_tangent` adds the supporting tangent at the
  current LP point, and only when that point violates it, instead of emitting one row
  per node at a fixed geometric anchor. Where the LP vertex sits *on* the singularity
  — the case the facet exists for, and where no finite-slope tangent is available —
  the touch point falls back to #1111's conditioning-capped ladder while the
  violation test still decides whether the row goes in, so the separator does not
  degrade to a no-op precisely where it is needed.

  On isolated atoms the mechanism does what it was designed to do: on
  `min 2x - sqrt(x)` over `[0,4]` the root LP bound goes −0.7071 → **−0.12503**
  against a true optimum of −0.125 (>99 % of the gap closed), where the eager anchor
  reaches only −0.6557; the same holds on boxes nine orders of magnitude apart
  (`[0,1e-3]`, `[0,1e6]`, the latter recovering −353.55 → −125 exactly).

  **On the corpus lazy never loses the bound, and that is not enough.** Three-arm
  panel (off / eager / lazy), one binary, `max_nodes` budget with **no** `time_limit`,
  each arm's firing mechanism separately instrumented and asserted. 12 instances
  produced a scorable three-arm row:

  | instance | off | eager | lazy | lazy rows |
  |---|---|---|---|---|
  | `kriging_peaks-full050` | −142.657069 | flat | **−139.917835 (BETTER, +1.9 %)** | 7860 |
  | `kriging_peaks-full100` | −348.053602 | flat | **−342.588463 (BETTER, +1.6 %)** | 13214 |
  | `tspn15` | 269.3529936 | **269.8773406 (BETTER)** | 269.3558802 (BETTER, marginal) | 85 |
  | `tspn08` | 135 nodes | **191 nodes (WORSE, +41.5 %)** | 135 nodes | 25 |
  | `tspn10` | 177.43065803 | **177.39907299 (WORSE)** | 177.43065803 | 1 |
  | `mathopt5_6`, `eq6_1`, `maxmin`, `full010/020/030`, `tspn12` | — | flat | flat | 0–35585 |

  `eager vs off: BETTER 1, WORSE 2, flat 9`. `lazy vs off: BETTER 3, flat 9, worse 0`.
  Soundness checked on every arm of every run against `minlplib.solu` — **0 violations**,
  including per-repetition asserts in the reproducibility and timing panels. Each lazy
  gain occurs at an *identical node count and identical incumbent*: a strictly better
  bound for the same tree.

  **The timing panel is what decides it, and it decides against graduation.**
  Interleaved off/lazy, 3 repetitions, per-arm standard deviation and a pooled sd,
  load gate recorded at every instance start (2.5–5.2 on 14 cores), both arms verified
  to explore identical trees so the wall delta is attributable to separation alone:

  | instance | off | lazy | Δ | lazy rows | bound gain |
  |---|---|---|---|---|---|
  | `eq6_1` | 10.82 ± 0.12 s | 13.59 ± 0.09 s | **+25.6 %** | 31 614 | none |
  | `maxmin` | 33.52 ± 0.13 s | 51.68 ± 0.04 s | **+54.2 %** | 35 585 | none |
  | `kriging_peaks-full050` | 42.21 ± 0.21 s | 44.09 ± 0.12 s | +4.5 % | 7 860 | +1.9 % of gap |
  | `kriging_peaks-full100` | 101.21 ± 0.21 s | 105.22 ± 0.22 s | +4.0 % | 12 498 | +1.6 % of gap |

  Every delta is 12–200 pooled standard deviations; none is noise. The shape is the
  disqualifying part: **lazy is cheap where it helps and expensive where it does not.**
  The two instances it charges +25.6 % and +54.2 % are exactly the two that gain
  nothing, because they are the ones that draw the most rows. A corpus-wide default-ON
  would pay the `maxmin` bill everywhere to collect the `full050` gain occasionally.
  Gate 1 (cert-clean) passes; **gate 2 (net-positive) fails**, so the flag stays
  default-OFF — the `DISCOPT_CUT_INHERIT` disposition, reached by a different route
  (there: neutral-or-harmful; here: helpful on a narrow class, unaffordable off it).

  Three corrections to earlier statements in this entry (§11):

  * **`kriging_peaks-full200` is withdrawn from the gains table.** It was reported at
    −746.566175 → −734.495407 (+1.6 %). Over three repetitions on a quiet machine the
    instance is nondeterministic *in both arms* — off returns nodes {301, 303} with two
    distinct bounds and **zero** separated rows; lazy returns {301, 303} with rows
    {32451, 32472, 32583}. Two single runs of a nondeterministic instance are not a
    comparison. This is pre-existing solver behaviour at that size, unrelated to this
    flag, and is filed separately.
  * **An intermediate claim that lazy *caused* that nondeterminism is also withdrawn.**
    It rested on two repetitions in which the off arm agreed; the third falsified it.
  * **The quoted bounds are load-dependent to ~3 significant figures.** Quiet, `full100`
    gives off −350.768042 → lazy −345.634363 over 12 498 rows; under the contention of
    the original panel, −348.053602 → −342.588463 over 13 214. Separation is wall-bounded
    (`_separate_*` all break on a shared `_deadline`; the node LP re-solve takes
    `time_limit=_remaining()`), so row counts — and the bounds they produce — shift with
    machine load. The *direction* (lazy better by ~1.5 % of gap) reproduces; the digits
    do not, and should not be read as properties of the instance alone.

  Remaining gaps, recorded rather than glossed:

  * **Lazy has a coverage hole eager does not.** The separation chain is gated on
    `if separate:` (`mccormick_lp.py:1641`), forced off on yield rounds and pool-free
    re-solves. On all three `elec*` instances the lazy arm registered thousands of
    specs (19 236 on `elec100`) and the separator ran **zero** times, while eager
    fired at build time throughout. Lazy trades "emit a geometric guess wherever a
    relaxation is built" for "emit an exact tangent only where a separation round
    runs" — it is not a coverage superset.
  * **5 of 17 screened instances produced no scorable row:** `elec*` and `full500`
    return no dual bound at all (1 node, or 0 for `full500`), and `full200` is
    nondeterministic in both arms (above). `full030` and `tspn15` were initially lost
    to a per-instance wall kill set too low — a biased loss, since lazy runs third and
    every truncation cut it preferentially — and were recovered by a per-arm re-run.

  Consequently the #1111 finding "the root win does not survive branching" is scoped
  to the **eager** anchor. Where the dropped facet dominates the root relaxation,
  lazy placement does retain part of it; elsewhere it is inert, and on `elec*` it
  cannot act at all.

### Removed

- **The inert `superposition` knob** (`fix(relax)`, #1046, closes #1035).
  `build_milp_relaxation` accepted a `superposition` parameter and never read
  it — the #632 uniform-factorable cutover stopped consuming it — so
  `relaxation_arithmetic="superposition"` quietly returned the plain McCormick
  relaxation. The generator behind it, `discopt._relax.superposition`, was
  reachable only from its own tests. Per `CLAUDE.md` §3 (*no dead flags*) the
  switch is deleted rather than re-wired: an accepted-and-ignored knob is worse
  than no knob, because a measurement taken with it set reads as a measurement
  of the feature. `relaxation_arithmetic` itself is unaffected — its live values
  (`mccormick`, `chebyshev`, `taylor`, `ellipsoidal`, `alphabb`) never included
  `superposition`. One consequence recorded in `docs/dev/performance-plan.md`
  §6: the `superposition cuts` row of the ex1252 lever table measured the
  baseline and is retracted.

### Fixed

- **A GDP disjunction could be declared `infeasible` when it was not**
  (`fix(gdp)`, #1044, closes part of #1043). The hull reformulation's perspective
  function was not exact at both integer faces, so a disjunct that admitted a
  feasible point could have it cut away and the root node returned
  `infeasible` in a single node. A false `infeasible` is a hard-gate violation
  under `CLAUDE.md` §1 — the certificate is wrong, not merely loose — and this
  was the most serious defect fixed in this release.

- **The solver reported multipliers of the presolved model, not the declared
  one** (`fix(duals)`, #1042, closes #1037). Bound and row multipliers were
  handed back in the presolved space, so a user reading `result` got duals that
  did not correspond to the model they wrote. Row activity is now judged
  relative to row scale as well. This took `python/tests/test_minlptests.py`
  from **79 failed / 46 passed to 4 failed / 121 passed**.

- **The reported incumbent could be a barrier-interior point rather than a
  stationary one** (`fix(correctness)`, #1045, closes #1043). Two causes, both
  in the terminal incumbent polish: the polish ran in the *reduced*
  (FBBT/OBBT/root) box, where a reduction can place a bound exactly on the
  optimum; and an interior-point method stops a distance `~ mu/lambda` inside a
  *weakly* active bound, which the periodicity reduction creates systematically
  (it maps a doubly-infinite periodic-only variable to exactly `[-pi, pi]`, and
  a periodic function attains its extrema at the period boundary). The polish
  now runs over the declared box for free columns and crosses over onto weakly
  active bounds, guarded by feasibility, objective-not-worse, and a dual-bound
  floor so the `bound <= incumbent` invariant binds the polish too. Measured:
  stationarity 1.331e-06 -> 7.84e-10 on `nlp_008_010`, and 2.037e-04 -> 1.22e-16
  on `nlp_001_010`. With #1042 and #1044 this brings `test_minlptests.py` to
  **125/125**.

- **The primal simplex could return a false `Unbounded` on a bounded LP**
  (`fix(lp)`, #1008). The ratio test now certifies the primal ray before the
  verdict is issued: `A d = 0`, `cᵀd < 0` under the cost the phase is actually
  minimizing, and box recession on every coordinate `d` moves. A ray that fails
  any of the three is a numerical breakdown, not an unbounded LP, so the status
  becomes `Numerical` — an honest refusal, and the one that routes into
  `dense_retry`'s dense-LU path, which a false `Unbounded` bypassed entirely.

  `Infeasible` has never been taken on faith (`farkas_ray_certifies_cols`);
  `Unbounded` was, and the asymmetry was not justified. "No basic variable
  blocks" is a statement *about* `α = B⁻¹A_q`, so a silently broken ftran
  produces it verbatim. On the captured QPLIB_2170 root relaxation it did: every
  basic `α` came back zero for a column that is not zero, giving a ray with a
  single nonzero, `|A d| = 1.0` and `cᵀd = 0`, against an LP HiGHS certifies
  optimal at 0 in 81 pivots.

  What that cost depends on the driver. `spatial_tree::verdict_for` already
  lumps `Unbounded` with `Numerical` into `Undecided`, so there the damage was
  the bypassed `dense_retry`, not a bad bound. `milp_driver` is the serious one:
  a node LP returning `Unbounded` sets `out.unbounded`, which breaks the search
  (`hit_unbounded`) and short-circuits `decide_status` ahead of every other
  branch — so a single false node verdict returns `MilpStatus::Unbounded` for a
  bounded MILP, discarding any incumbent. That is a false certificate, which is
  the one output §1 gives no slack at all.

  The stall-bail flip below does not subsume this. End to end on QPLIB_2170's
  root relaxation with `DISCOPT_LP_DUAL_STALL_BAIL=0` — the shipped default —
  and `time_limit=None`, `UnboundedRejectRowResidual` is 2: the false verdict was
  reachable without the bail ever firing. The externally visible outcome on that
  cell is unchanged (no solution either way); what changed is that the engine no
  longer *claims* a certificate it cannot support. The remaining loss on that
  cell is a separate defect (the unstable-pivot recovery is gated on
  `bank_deadline_duals`, so a caller passing no `time_limit` loses it — the same
  LP solves to `optimal 0` with `time_limit=40.0`); it is not addressed here.

  The margins are built from accumulation magnitudes, not from the result
  (#1017); the relative slack is `1e-7` rather than `gamma(nnz)` because the
  residual carries the LU solve's error, not just summation rounding. Regression
  test `cold_primal_does_not_claim_unbounded_on_a_bounded_lp` returns
  `Unbounded` (obj 351) without the certifier and `Numerical` with it, and
  asserts a rejection counter fired so the fixture is proven to reach the guard.

- **The dual simplex's unstable-pivot recovery was gated on the caller passing a
  `time_limit`** (`fix(lp)`, #1008). `SimplexOptions::bank_deadline_duals` is set
  as `deadline.is_some()` and was doing double duty: besides banking the dual
  loop's anytime floor (#928, its actual job), it also switched on the recovery
  that re-tries a near-zero pivot instead of abandoning the re-solve. The two have
  nothing to do with each other, so a caller who passed no `time_limit` silently
  lost a numerical safeguard.

  Measured on the captured QPLIB_2170 root relaxation, by counter rather than by
  reading control flow:

  | call | `DualUnstablePivotRecoveries` | `DualUnstablePivotBails` | result |
  |---|---|---|---|
  | `time_limit=None` | 0 | **1** | no solution |
  | `time_limit=40.0` | **1** | 0 | **optimal 0** |

  Exactly one unstable pivot separates a certified optimum from nothing, decided
  solely by whether the caller bounded the LP in time. The recovery now has its
  own gate, `SimplexOptions::recover_unstable_pivot`
  (`DISCOPT_LP_UNSTABLE_PIVOT_RECOVERY`), and `bank_deadline_duals` is back to
  banking duals only.

  **Default-OFF for deadline-free callers, per §5**: the recovery changes which
  pivot the dual loop takes, so it is bound-changing and stays behind the flag
  until a corpus differential panel clears both the cert-clean and net-positive
  bars. The deadline path keeps the recovery unconditionally
  (`deadline.is_some() || …`), so it is bit-identical to what its own panel already
  judged. Two counters (`DualUnstablePivotRecoveries` / `DualUnstablePivotBails`)
  make the mechanism measurable instead of inferred, and
  `unstable_pivot_recovery_is_not_gated_on_a_deadline` pins the split (bails 1 /
  not optimal with the flag off, recoveries 1 / optimal 0 with it on, and asserts
  no deadline is set in either arm).

- **A NaN variable bound reached the simplex and was read two contradictory
  ways** (`fix(lp)`, #1008). The modeling layer spells "no bound" as **NaN**
  (`Model.continuous(ub=None)` stores `array(nan)`); the LP layer spells it as
  the sentinel `±1e20`. Nothing translated between them, and an untranslated NaN
  does not fail loudly — every comparison against it is false, so each guard
  reads it as whichever answer its comparison happens to be written for. The
  ratio test asks `ub < INF` ("does this bound block?") and calls a NaN bound
  **open**, stepping to `t = INF`; the ray certifier above asks `ub >= INF` ("is
  this side open?") and calls the same bound **closed**. This is the
  `INF`-is-`1e20` hazard already documented in `CLAUDE.md`, in its other guise:
  there the sentinel silently survives a multiplication, here it is silently
  absent.

  Found by the certifier: a Benders recourse LP (`min -w` over `w ∈ [0, NaN]`,
  `-w ≤ 0`) had been reported `unbounded` on the strength of a box no guard could
  certify as recessive. That verdict was correct — the LP *is* unbounded below —
  but it was correct by luck, resting on which of the two readings the ratio test
  happened to use, over a box the engine could not derive it from.

  Both halves are fixed: `lp_simplex._finite_box` translates the box (NaN and
  `±inf`, and any magnitude past the sentinel, onto `±1e20`) so the simplex sees
  one convention, and the PyO3 LP/MILP entry points refuse a NaN bound with a
  `ValueError` naming the index, so a caller that skips the translation gets a
  loud error instead of a verdict derived from two incompatible readings of the
  same number. `±inf` is deliberately *not* refused: it satisfies `>= INF` and
  fails `< INF`, so both readings already agree it is open. Regression tests in
  `python/tests/test_1008_nan_lp_bound.py` fail without the translation
  (`ValueError: ub[0] is NaN`) and pass with it.
  The pre-existing `unbounded_detected` / `unbounded_emits_a_valid_primal_ray`
  tests confirm a genuine unbounded ray still certifies untouched.
  `cargo test -p discopt-core` → 610 passed.

- **The #1013 degeneracy-stall bail could turn a certified optimum into no bound,
  and shipped default-ON** (`fix(lp)`, #1008). `DISCOPT_LP_DUAL_STALL_BAIL` is now
  **default-OFF**; `=1`/`true`/`on` opts in, and the mechanism itself is unchanged.

  The entry below justified the default-ON graduation with "it can change only
  *which* engine finishes a solve, never the value". That holds only if the cold
  two-phase primal the bail hands off to actually finishes, which all three
  bailing cells of the 100-LP panel happened to do. On an LP where it does not,
  the bail abandons a warm loop that was converging and the caller gets nothing.
  Measured on two QPLIB root relaxations outside that panel, at
  `time_limit=None`: `QPLIB_2738` goes from `optimal −5.0587686` (9.6 s) to no
  solution, and `QPLIB_2170` from `optimal 0` (1.7 s, with a deadline) to no
  solution — in the latter case the cold path does not refuse but returns
  `Unbounded`, against an optimum HiGHS certifies as 0 in 81 pivots. The
  detector cannot separate "stalled" from "converging slowly": QPLIB_2170 reaches
  its optimum after ~22 800 degenerate pivots driven by Bland's rule, well past
  the 2048-pivot patience.

  Regressed against the panel's own measured benefit — inert on 97 of 100 LPs,
  1.34x and 1.13x on two cells — this is not a trade §1 permits. #1013's PR body
  had already withdrawn the "broadly net-positive" claim and named flipping the
  default to OFF as the alternative; this does that. Vendored fixture
  `qplib2170_cold_fail_lp.json` (1755×3193) and the regression test
  `dual_stall_bail_can_cost_a_bound_when_the_cold_solve_fails` pin it: the test
  returns `Unbounded` on the previous default and `optimal 0` on this one.
  `cargo test -p discopt-core` → 609 passed.

- **A stalled warm dual re-solve had no escape, and the two that were supposed to
  cover it were unreachable** (`fix(lp)`, #1013). The warm dual simplex escalates
  to Bland's rule after `2·(n+1)` consecutive degenerate pivots, and the F2 stall
  guard trips at the size-derived pivot cap `20·(m+n)+500`. On a lifted
  relaxation those are 58 194 and ~10⁶ pivots: measured over a 100-LP panel of
  in-repo root relaxations (all 9 vendored QPLIB instances and all 68 vendored
  MINLPLib `.nl` instances, `rlt_lineq` off and on), `DualBlandActivations` and
  `DualStallTrips` are **0 on every single LP** — including cells where 98.7 % of
  pivots are degenerate and the solve exhausts its budget. A degenerate stall
  therefore ran to the iteration cap, and one LP (`QPLIB_3814_rlt1`) ended it by
  returning `infeasible` on an LP that SciPy/HiGHS, every perturbed arm of our own
  engine, and an elastic `min t` feasibility LP all agree is feasible.

  `SimplexOptions::dual_stall_patience` (default 2048 as shipped here; **flipped
  to OFF by #1008 — see the entry above**) now hands such a solve to the caller's
  cold two-phase primal — the action every other difficulty in the dual loop
  already takes, and one that self-verifies its own verdict, so it can change only
  *which* engine finishes a solve, never the value. (**That last clause is
  withdrawn**: it assumes the cold solve finishes.) The threshold
  separates two non-overlapping measured populations: every warm loop that
  converges peaks at a 902-pivot degenerate run, every one that does not runs
  ≥ 1274. `DISCOPT_LP_DUAL_STALL_BAIL=<n≥2>` overrides it; `=0` restores the
  previous loop (`1`/`true`/`on` mean "enabled at the default patience", not a
  patience of one pivot).

  Graduation panel (2 reps, default vs off), re-run on the post-#1017 base:
  **99 of 100 LPs identical in status *and* iteration count**, 0 status
  regressions, **0 status improvements**, objective drift 0.00e+00 across 97
  optimal/optimal pairs, and 1.34x / 1.13x / neutral on the three cells where it
  fires. (An earlier revision credited this change with an `infeasible` →
  `optimal` improvement on `QPLIB_3814_rlt1`; #1019's Farkas margin fix for
  #1017 landed on `main` in the meantime and that LP is now `optimal` with the
  bail off as well as on, so the improvement is #1017's, not this one's. This is
  a **tail guard** — inert on 97 of 100 — not a throughput change.) A
  tree-level panel over every vendored `.nl` with a recorded optimum (16
  instances) is bit-identical in objective, bound and node count on all 16, with
  96 soundness assertions and 0 issues.
  Measurement, falsified alternatives (including #1008's dual Harris pass
  re-tested scoped to the stall, and Bland at a reachable threshold — both
  regress a status elsewhere) and the reproduction harnesses:
  `docs/dev/performance-plan.md` §18, `scratchpad/i1013/FINDINGS.md`.

  The `QPLIB_3814_rlt1` certificate itself is a **separate** defect — its
  Neumaier–Shcherbina margin is built from result magnitudes rather than
  accumulation magnitudes, so a `bᵀy` of 3.0e-8 against a term magnitude of 600
  passes as a proof of infeasibility — and is filed separately; this change only
  stops a stalled warm loop from being the thing that decides it.

### Added

- **Degeneracy-stall instrumentation for the warm dual simplex** (#1013).
  `DualDegenerateRunMax` (longest run of consecutive degenerate pivots — a
  maximum, not a sum), `DualDegenerateRunArms` (episodes crossing the 32-pivot
  arming threshold), `DualDegenerateStallBails` (warm solves handed to the cold
  path by the stall bail), plus `profile::record_max`. `DISCOPT_LP_DUAL_TRACE=1`
  emits one `DUALTRACE` line per dual pivot (chosen pivot magnitude, primal and
  dual step lengths, degeneracy, ratio-test state) — the instrument that showed
  the stall on this corpus is *not* the hypothesized tiny-pivot mechanism: on the
  worst cell the chosen pivot magnitude is exactly 1.0 at the median with none
  below 1e-4.

- **`solver="direct"` — derivative-free global search (DIRECT).** A new backend
  in `discopt.solvers.direct` for models whose objective or constraints contain
  an opaque `dm.custom` (`CustomCall`) body outside the reduced-space `MCBox`
  scope. Such a model previously degraded to a single local NLP with no global
  search at all, or raised outright when integer variables were also present
  (sound-or-refuse); it now has a systematic search over the box.

  Implements Jones/Perttunen/Stuckman (1993) with the modifications
  Jones & Martins (JOGO 2021) conclude are generally beneficial, all default-on:
  trisect one long side (Jones 2001, choosing the dimension split fewest times so
  far), select one rectangle among ties, an `epsilon` floor, and hybridization
  with a local solve. Also implemented: Jones 2001 integer centres, DIRECT-GLce
  constraint handling, and an evaluation cache. DIRECT-GL's two-step selection is
  available as `direct_variant="gl"` but is **not** the default — measured,
  evaluations to 1% accuracy: `hartman_6` classic 105 / gl 277, `shubert` classic
  2269 / gl 181, reproducing both directions of the survey's trade-off.

  `local_refine_method` selects the gradient NLP, Powell, or `"auto"` (NLP with a
  Powell fallback). The fallback matters because a `dm.custom` body is
  JAX-*traceable* by construction but not necessarily usefully *differentiable*:
  on a staircase objective the NLP stalls at 0.0312790 while Powell reaches
  0.0312500. Refinement launches from the best of {caller's start, DIRECT's
  incumbent}, so the backend cannot lose to the local-only path.

  `n_jobs` evaluates each iteration's independent sample points concurrently
  (threads; a `dm.custom` model is not picklable so a process pool cannot carry
  the evaluator). Measured: a 200-evaluation run on a 50 ms objective drops from
  10.0 s to 1.9 s on 8 threads, and a 900-evaluation JAX objective from 1.55 s to
  0.51 s on 4. The result is **identical**, not merely equivalent — pinned by
  differential tests against the pre-batching implementation across 44
  function/variant configurations.

  **Returns no certificate**, and the contract is enforced at the single
  `SolveResult` construction site: `bound` and `gap` are `None`, `gap_certified`
  is `False`, the status is never `"optimal"`, and an exhausted budget is a limit
  status — never `"infeasible"`, since DIRECT cannot prove infeasibility. A
  non-finite box and a missing objective raise rather than being approximated.

  Entry experiment and its falsification record:
  `docs/dev/direct-entry-2026-08-12.md`; reproduction
  `scripts/direct_entry_experiment.py --self-check`. Notebook:
  `docs/notebooks/direct_global.md`, which states plainly that this is a baseline
  rather than the state of the art and names the alternatives.

- **`solver="surrogate"` — model-based search for expensive black boxes.** A new
  backend in `discopt.solvers.surrogate` serving the same class as
  `solver="direct"` (an opaque `dm.custom` body with no algebraic relaxation) but
  the other cost regime: it spends real computation between evaluations — a
  linear solve and a global optimization of the acquisition — so that each
  evaluation counts. Use it when an evaluation costs minutes; use `"direct"` when
  it costs milliseconds.

  Two families behind one interface. **RBF is the default** (`surrogate="rbf"`):
  a cubic / thin-plate / linear kernel with a linear tail, fitted by one symmetric
  solve, chosen over a GP because integer variables work natively (discopt is a
  MINLP solver), fitting has no failure mode of its own, and it degrades better
  with dimension. **Kriging + expected improvement** (`surrogate="kriging"`) is
  the alternative for smooth, low-dimensional, very expensive objectives, with a
  nugget so a noisy objective is not forced through its own measurement error.
  Trust-region restriction and batch proposals are left as seams, not built.

  The acquisition subproblem is an ordinary algebraic model and is solved by
  discopt's own spatial branch-and-bound. **This is the one place a certificate
  appears, and it certifies where to sample next — not the answer.** Two
  measurements bound the claim, and both contradict what was originally planned:

  - *Certified EI does not work.* Built with the division lifted away via
    `EI(x) = max_u [d(x)Φ(u) + s(x)φ(u)]` and the dense `rᵀR⁻¹r` whitened to
    `Σvᵢ²`, B&B finds the true acquisition maximum to 5 significant figures, but
    the dual bound never closes: on branin the bound runs 3.82 / 23.1 / 597 /
    4871 against optima 0.60 / 0.26 / 0.42 / 1.51 at m = 8/12/20/30. discopt is
    an excellent *primal* acquisition optimizer here, not a certifying one.
  - *Certified CORS does work, and the kernel decides it.* Relative gap on branin
    at m = 6/8/12/20: linear 6.5e-5 / 4.2e-5 / 6.7e-6 / 2.6e-8 (all certified);
    cubic 6.6e-8 / 0.19 / 6.0 / 27.7, as `max|λ|` grows 10 → 164 faster than a
    McCormick relaxation can bound. `rbf_kernel` stays `"cubic"` for optimization
    quality; `"linear"` is documented as the choice when the certificate matters.

  Sample efficiency is real but **not an order of magnitude**. Evaluations to
  1e-2 relative accuracy versus DIRECT: six_hump_camel 32 vs 137, branin 38 vs
  69, hartman_3 46 vs 79, ackley_2 48 vs 67 — and goldstein_price 96 vs 75, a
  **loss**, caused by objective dynamic range (3 → ~1e6); RBFOpt's monotone
  objective transformation is the named remedy and is documented as a follow-up
  rather than implemented. DIRECT is a stronger baseline than the surrogate
  literature's framing suggests.

  Cost model, worth knowing before choosing this backend: nearly all the wall
  clock is the acquisition solve, not the objective. On branin with a free
  objective and `max_evals=30`, the 15-point initial design costs 0.8 s total and
  every subsequent evaluation costs almost exactly `acquisition_time_limit`
  (20 s). That is the intended trade when one evaluation dwarfs 20 s of solver
  time, and the wrong one otherwise — on a cheap objective `solver="direct"` is
  far faster for a better answer. Shortening `acquisition_time_limit` is a trap:
  with the default cubic kernel the acquisition never certifies, so the budget
  *looks* wasted, but it is buying primal quality — relative error at 20 s vs 2 s
  is branin 0.2156/0.2029, six_hump_camel 0.0164/**0.8063**, hartman_3
  0.0098/0.0103. The default stays at 20 s.

  **Returns no certificate** for the original problem: `bound` and `gap` are
  `None`, `gap_certified` is `False`, the status is never `"optimal"`, and an
  exhausted budget is a limit — never `"infeasible"`. `on_evaluation` is a
  progress hook, present because a run whose objective takes minutes is otherwise
  indistinguishable from a hung one, and because evaluations-to-target is not
  recoverable from a `SolveResult`.

- **DIRECT as a governed root primal heuristic** (`DISCOPT_DIRECT_HEURISTIC`,
  **default-OFF**). A bounded-budget DIRECT run (300 evaluations, `epsilon=1e-2`)
  over the root box, registered as a governed source in `heuristic_governor.py`
  alongside `rens`, so the existing hit-rate throttle can disable it if it spends
  without improving. Every point it proposes goes through the same
  feasibility verification and incumbent path as the other heuristics; the dual
  bound is never touched, so this is CLAUDE.md §5's heuristic-policy regime and
  can only ever cost nodes.

  Soundness spot-check across 16 in-repo `.nl` instances at 10 s: 12 probe
  invocations, 0 violations — no bound rose, no objective degraded, no
  `gap_certified` lost. Note the dual bound is *not* bit-identical in every case:
  on `st_e13` it moved 1.4e-15 **tighter**, because an earlier incumbent prunes a
  different node set and the surviving frontier minimum shifts in the last ulps.
  The bound is never computed differently.

  **Known limitation:** the #764 native Rust spatial kernel (default-ON) bypasses
  this wiring for its covered subset (scalar variables; bilinear / monomial /
  affine-square / sqrt), because `solve_model` hands the tree to `discopt-core`
  before the Python spatial loop and that path seeds itself. Everything outside
  that subset — trig, exp, log, division, i.e. the multimodal class DIRECT exists
  for — still routes through the heuristic.

### Measured (no behaviour change)

- **#966 coupled graduation panel, re-run on top of #990: the three deadline
  flags do NOT graduate. They stay default-OFF.** 19 instances x 3 arms x 3 reps
  x 3 budgets (10/15/20 s) = 9 artifacts, 513 solves
  (`discopt_benchmarks/results/issue966_postfix{10,15,20}_rep{1,2,3}.json`).

  *Bar 1 (cert-clean): PASS.* 9/9 artifacts with 0 unsound, 0 cert_regressions,
  0 lost_incumbents, 0 lost_bound, 0 incumbent_verification_failed, each over 48
  executed oracle comparisons (432 total, non-vacuous). The pre-#990 panel failed
  this on an `nvs05` cert regression, which did not recur.

  *Bar 2 (net-positive): FAIL*, for three reasons, the first decisive:

  1. The gain is confined to two named instances, which CLAUDE.md §2 rejects. At
     15 s, `hda` + `contvar` account for **107.5%** of the total cand-vs-base
     overrun reduction — the other 17 instances are net negative (`bchoco07`
     −0.46 s). At 10 s `contvar` alone is 55.9%.
  2. #966's own two flags (`DISCOPT_NODE_ROUND_BUDGET`,
     `DISCOPT_HESS_COMPILE_GATE`) are neutral at 10 s and *worse than base* at
     15 s (9.43 vs 8.56 s). All of the candidate arm's benefit arrives with
     #928's `DISCOPT_LP_WARM_DEADLINE`.
  3. At 20 s the flags are unmeasurable (−0.47 ± 0.96 s, mean inside its own rep
     spread, sign flip across reps) — consistent with 0/19 instances exhibiting
     the deficiency at that budget once #990 landed.

  This is the `DISCOPT_CUT_INHERIT` outcome: sound but not helpful.

- **#966 deficiency (2) — coarse global-deadline compliance — is fixed in
  default behaviour by #990.** Worst single cell as a multiple of its budget,
  across panel generations: 10.50x (`issue966_yield_binding20`, `heatexch_gen3`
  ~210 s against 20 s) → 1.37x → 1.23x (pre-#990) → **1.06x** (post-#990), with
  zero cells above 1.5x since the second generation. The multi-hundred-second
  blowup class the issue was opened about no longer occurs.

- **The #966 graduation scorer's `cert_gains > 0` conjunct is not a measurement**
  and is recorded here rather than silently dropped. Across 190 cells in 10
  artifacts, certification differed between arms in exactly 2 cells — both
  `nvs05`, both times the flags *losing* it. `nvs05` is the panel's only
  certification-boundary instance (it certifies only when it closes the gap a
  hair under budget: walls 19.98 / 20.05 / 19.89 s); the other four certifying
  instances certify in all three arms every time. The conjunct is therefore a
  coin flip on one instance, and is stricter than §5, whose net-positive metrics
  are "node count / wall / bound". It was left in place: relaxing a criterion
  after seeing it fail is what §5 exists to prevent. Both readings are reported
  on #966. **The flags fail §5's literal reading too**, on reason 1 above.

### Changed

- **`feral` 0.14 → 0.15.1 and `pounce-solver` `>=0.9` → `>=0.10`**
  (`chore(deps)`). Two pins that move together: POUNCE 0.10.0's own workspace
  pins feral 0.15.1, so holding both here keeps a single LU implementation
  behind the simplex node engine and the NLP evaluator rather than two.

  feral 0.15.0 is a **breaking** release, but only in its diagnostic API
  (`SupernodeTiming` `us` → `ns`, `BucketStats::sum_us`/`avg_us` → `*_ns`,
  `ProfileReport::loop_us` → `loop_ns`). discopt consumes only the LU layer —
  `SparseLu`/`DenseLu`/`SparseColMatrix`/`SparseLuSymbolic`/`LuParams`/
  `RefactorCause`/`FeralError` in `linsolve.rs` — and none of the multifrontal
  or profiler surface, so no discopt source changes. What it does buy the node
  engine: the packed BLAS-3 trailing update is now an explicit pulp SIMD kernel
  (upstream reports byte-exact output at every SIMD width, pinned by golden
  bit-digests), and four tuning-knob `env::var` lookups leave the per-panel
  dispatch path for a `OnceLock`. That second one is per-front *fixed* cost, so
  it lands hardest on many-small-front bases — the per-node simplex regime.

  POUNCE 0.10.0 carries #477/#478 (`NlProblem` is sendable), so on an
  in-contract install the tape evaluator shares one problem and the per-thread
  fallback no longer runs. The fallback is **retained**, not deleted: the
  capability is probed rather than inferred from a version string, and a build
  whose probe says "not sendable" must still solve.
  `test_75_tape_nlp_evaluator.py` keeps it covered by forcing the probe false.

  **Verified bound-neutral (CLAUDE.md §5).** 49-instance cert panel, both arms
  built on one machine from the same source and told apart by the `feral-<ver>`
  string cargo bakes into the `_rust` extension: **49/49 rows bit-identical in
  status, `node_count` and `objective`**. A same-build repeat ran first as the
  determinism control — also 49/49 bit-identical — so node equality is a real
  gate here and not an artifact of a panel that never varies. Also green:
  `cargo test -p discopt-core` (597+4+1), `pytest -m smoke` (943 passed, 1
  skipped, 2 xpassed), the adversarial suite (10 passed), and
  `pytest -m requires_pounce` (127 passed, 1 skipped) apart from one failure
  reproduced identically on the pre-bump control (see below).

  Note on method: the arms were diffed against *each other*, not against
  `docs/dev/data/cert-baseline.jsonl`, because at the time that reference was
  stale (18 violations on an *unmodified* tree). It has since been regenerated
  in this same series — see the next entry — so a future re-run of this check
  can use the committed reference directly.

- **`docs/dev/data/cert-baseline.jsonl` regenerated, 49 → 52 rows**
  (`chore(cert)`). The §0.2.5 bound-neutrality reference had drifted out of
  reproducibility against its own tree: `check_cert_neutrality.py` on an
  *unmodified* checkout reported **18 violations** — objective drift past
  `_OBJ_TOL` (`cvxnonsep_nsig30` 1.5e-6, `cvxnonsep_psig40r` 2.9e-6, `m3`
  5.9e-7) and node swings (`nvs13` 23 → 637, `dispatch` 3 → 23, `tls2`
  343 → 255, `tspn05` 51 → 39). All 49 still certified `optimal`, so this was a
  drifted reference and not a soundness bug — but it left the neutrality guard
  unable to detect a real regression while still reading as a guard, which is
  worse than having none.

  Regenerated with `gen_cert_baseline.py --time-limit 60`; the generator's
  determinism filter still gates admission (3 solves per instance, bit-identical
  `node_count` and `objective`, certifying within 0.6 of budget), so the
  reference is reproducible by construction. 56 panel rows, `incorrect_count`
  **0** against 58 oracles, **52 admitted** (`nvs05`, `nvs17`, `tanksize` newly
  qualify; nothing previously present was dropped), 4 excluded as non-optimal
  (`carton7`/`hda` time-limited 3/3, `casctanks`/`ex1252` feasible-not-optimal
  3/3). **Acceptance:** `check_cert_neutrality.py` against the fresh file
  returns NEUTRAL, 52/52, exit 0. Nothing in `.github/workflows/` consumes this
  file — it is a local guard, also used by `graduation_gate.py`.

  The `_KNOWN_PERF_GATED` exemption for `nvs17` is left in place though its
  rationale ("~45s/60s wall") is now stale — it certifies in 15.7s. The
  exemption only relaxes node-count strictness while soundness is still checked;
  tightening it is a separate change needing its own measurement.

- **#966's three coupled flags stay default-OFF** — graduation panel scored and
  **failed both §5 bars**. `DISCOPT_LP_WARM_DEADLINE`,
  `DISCOPT_NODE_ROUND_BUDGET` and `DISCOPT_HESS_COMPILE_GATE` were measured over
  2 budgets × 3 reps × 19 instances (artifacts and scorer committed under
  `discopt_benchmarks/`). *Cert-clean* fails on one certification regression
  (`nvs05`, bench20 rep2): at a 20s budget that instance sits on its own
  certification deadline — the fresh baseline puts it at 20.5s — and base
  certifies it 1/3 while the candidate certifies 0/3. *Net-positive* fails
  because the benefit is budget-dependent: overrun drops −5.00 ± 0.61 s at a 15s
  budget but only −0.17 ± 0.72 s at 20s, where the mean is inside its own rep
  spread and the sign flips between reps; node totals rise slightly at both
  budgets and `cert_gains` is 0 in all six reps. Bounds are net favorable (28
  tighter vs 12 looser) and objectives 7 better / 0 worse, so this is *not
  broadly helpful* rather than harmful — the `DISCOPT_CUT_INHERIT` shape, and
  per §5 such a flag stays OFF with the measurement recorded.

### Fixed

- **Opaque `dm.custom` models reported a fabricated (invalid) dual bound**
  (`fix(correctness)`, C-42, closes #998). A `CustomCall` body that is not
  MCBox-reducible falls back to a single local NLP; that caller cleared
  `gap_certified` but left the *local* objective in `bound` / `gap` /
  `root_bound` / `root_gap`. On a nonconvex body the local optimum is routinely
  above the true global minimum, so the reported "dual bound" was not a bound at
  all — 2-D Ackley behind `dm.custom` on `[-25.768, 39.768]` returned
  `bound=15.06, gap=0.0` against a true global minimum of `0.0`, while the
  algebraic twin of the same mathematics correctly returned `bound=None`. This is
  the C-33/SC-1 defect, whose strip was never applied to this caller; an opaque
  body cannot be inspected at all, so convexity can never be established for it
  and the strip applies a fortiori. The fix keeps the feasible incumbent and
  strips the bound/gap on any non-`infeasible` result — it only ever removes a
  claim, so it can neither introduce a false optimum nor loosen a valid bound.
  Rigorous `status="infeasible"` and the MCBox-reducible (global reduced-space)
  `CustomCall` path are unaffected. Regression tests:
  `python/tests/test_customcall_local_bound.py`.

  **Behaviour change (both uncertified local-NLP paths).** `_solve_continuous`
  reported `status="optimal"` on the same basis it reported the bound — the NLP
  converged — which on an unproven-convex model is the same unearned claim. Both
  the opaque-`CustomCall` path and the pure-continuous convexity-unknown path
  (`skip_convex_check`, or the convexity classifier abstained) now report
  `status="feasible"`: a feasible point was found, global optimality was not
  proved. This is already the convention the spatial path uses when its gap does
  not close. `"unknown"` is reported when the incumbent itself was withheld. Code
  that gated on `status == "optimal"` from these paths should check for
  `"feasible"` too — or, better, read `gap_certified`. The convex fast path,
  MCBox-relaxable `CustomCall` models, and every certified path are unchanged.

- **`solve(time_limit=T)` overran `T` because heuristic sub-NLPs were polled but
  never capped** (`fix(heuristics)`, contributes to #966). Every deadline guard in
  `_relax/primal_heuristics.py` gated whether a sub-NLP *starts*; none bounded how
  long the started solve *runs*. Polling therefore caps the **number** of
  overshooting solves at one and says nothing about its duration — and
  `feasibility_pump`'s round 0 was not polled at all. `_HEURISTIC_NLP_MAX_ITER`
  does not cover this: an iteration cap is not a time cap, and on the
  no-relaxation flowsheet class a single IPM iteration carrying an exact Hessian
  is itself seconds long.

  **Attribution (measured, not assumed).** A post-deadline stack sampler put
  **100 % of its 73 post-deadline samples** inside one `nlp_pounce.solve_nlp`
  call — 91.8 % reached from `feasibility_pump`, 8.2 % from
  `integer_local_search`. The overrun reproduced with all three coupled #966 flags
  ON and OFF alike, within ~0.5 s of each other, which is what makes it a
  **default-configuration** defect rather than a flag effect. (It is also what was
  masking the flags' benefit in the #966 graduation panel; the flags themselves
  remain default-OFF, and #966 stays open.)

  **The fix.** `_deadline_wall_cap()` derives a per-solve `max_wall_time` as
  `max(0.1, min(3.0, deadline - now))` — the same clamp `solver.py` already
  applies to the root relaxation — and it is now applied in `feasibility_pump`,
  `integer_local_search` (both `subnlp` repairs and its relaxation seed) and
  `diving`, whose own comment already recorded `heatexch_gen3` running "tens of
  seconds past the deadline" after it gained an entry poll. A caller that passes
  no deadline gets `None` and is unchanged bit-for-bit; an explicit caller
  `max_wall_time` still wins.

  `feasibility_pump` now also accepts a `TIME_LIMIT` result: a cap that discards
  its own truncated points merely trades an overrun for a lost incumbent, which is
  a different regression, not a fix. The status was **verified, not assumed** —
  `nlp_ipopt._IPOPT_STATUS_MAP` maps Ipopt −5 `Maximum_WallTime_Exceeded` to
  exactly this status. This widening is deliberately **not** folded into the
  shared `_is_nlp_feasible` (used at sites with no re-verification behind it): the
  pump re-verifies the point independently of its status via
  `_is_integer_feasible` + `_check_constraint_feasibility`, with `inject_incumbent`
  enforcing strict improvement — the same footing that already licenses `subnlp`'s
  `ITERATION_LIMIT`. Two tests pin that soundness: a time-limited but *infeasible*
  point is still rejected, and `_is_nlp_feasible` itself still refuses
  `TIME_LIMIT`.

  **Measured.** Interleaved A/B (never blocked), 3 reps, arm asserted inside each
  child process by a version marker — present for fixed, **absent** for base — so
  a stale `.pyc` crashes instead of reporting the wrong arm's number:
  `bchoco07` **+1.65 ± 0.18 s → +0.22 ± 0.04 s**, `bchoco08`
  **+1.05 ± 0.12 s → +0.25 ± 0.00 s** on a 20 s budget. `heatexch_gen3` overran
  neither arm in this run (+0.16 s both), so it is neither confirmed nor refuted
  here. **Bound-neutral (CLAUDE.md §5):** the 52-instance cert panel is
  **52/52 identical in `node_count`, all 52 `optimal`, max |Δobj| = 0.00e+00**,
  with a real objective comparison on every row (zero `nan`s). Note the panel
  shows *no regression* rather than stressing the cap — those instances certify
  well inside their budgets, so the cap never binds; the binding behaviour is
  covered by `test_966_heuristic_deadline_wall_cap.py` (7 of its 11 tests fail
  before the change).

  **Not a hard guarantee**, and recorded as such in the code: POUNCE tests
  `max_wall_time` between IPM iterations, so one expensive iteration still
  overruns it (`test_f4_root_budget_gate` records the same observation). This
  converts an *unbounded* overrun into roughly a one-iteration one; F4's entry
  gating and this duration cap are complementary, not alternatives.

- **Vectorized models silently got no relaxation at all** (#981). A `Constraint`
  body may be array-valued — `k * x - b == 0` on a 3-vector is *one* `Constraint`
  object standing for *three* scalar rows, the same one-object-many-rows
  distinction that #908 fixed in the incumbent verifiers. The relaxation engine
  assumed one scalar row per `Constraint`, so its interval walk hit a numpy array
  where it expected a float, raised `TypeError: only 0-dimensional arrays can be
  converted to Python scalars`, and the caller degraded to building *no*
  relaxation. The dual bound then fell back to the trivial objective floor —
  exactly `0.0` for a sum of squares — and the relative gap against zero can
  never close, so no time limit could certify. Sound, but useless: every DAE
  collocation model, every `nn` layer formulation, and every numpy-style user
  model was affected.

  `_relax/scalarize.py` is new: `static_shape()` recomputes an expression's shape
  bottom-up (the cached `_known_shape` is `None` above a matmul, so it cannot be
  trusted) and `scalar_elements()` rewrites an array-valued body into one scalar
  expression per flat element. `canonicalize()` now emits one canonical node per
  scalar row, with a `constraint_index` row map back to the originating
  `Constraint`; `uniform_relax` and `term_classifier` consume the expansion. An
  expression the rewrite cannot expand returns `None` and the caller keeps its
  previous behaviour — never `[]`, which would drop rows silently.

  Measured on `tutorial_dae`'s parameter-estimation cell: `feasible`,
  `bound=0.0`, 493 s → `optimal`, `bound=0.00155799`, gap 2.1e-07, certified,
  9.7 s, with the same estimate `k=0.4836`. A vectorized bilinear model now
  produces the same objective, bound, bilinear terms, and partition candidates as
  its scalar transcription. Bound-neutral for scalar models by construction: over
  all 66 in-repo `.nl` instances the canonical DAG is identical object-for-object
  (`scalar_elements` returns the body itself), pinned by a regression test.

### Changed

- **`tutorial_oa` documented an API that no longer selects OA** (`docs`). Every
  solve cell used `gdp_method="oa"`, which was deprecated in favour of
  `solver="mip-nlp", mip_nlp_method="oa"`: it now emits a `DeprecationWarning`
  and is reinterpreted as `gdp_method="big-m"`, a *GDP reformulation*. So the
  notebook ran spatial branch & bound throughout — its "Comparing OA with Branch
  & Bound" cell was comparing B&B against B&B, and its ECP cell never entered ECP
  mode. All five solve cells now use the current spelling, with a note explaining
  that `gdp_method=` and `solver=` are orthogonal axes.

  Two further corrections. The primary example (`example_simple_minlp`) closes
  the gap on its first master/subproblem pair, so it illustrates nothing about
  the alternation; the notebook now also builds `synthes1` (Duran & Grossmann
  1986, test problem 1), which gives a genuine four-iteration loop, and prints
  the per-iteration `mip_nlp_trace` — LB 1.411895 → 6.009759 against UB 7.092732
  → 6.009759 — plus the OA/ECP counter contrast (OA: 4 masters, 4 NLPs, 35 cuts;
  ECP: 8 masters, 0 NLPs, 20 cuts). The Limitations section claimed single-tree
  LP/NLP B&B and level-method regularization were "not yet implemented"; both
  exist (`mip_nlp_method="lp_nlp_bb"`, which needs `milp_solver="gurobi"`, and
  `add_regularization`/`level_coef`), and the section now says so.

- **The manuscript's "hybrid Rust/JAX/Python" framing retracted and rewritten**
  (`docs`, #1049). `manuscript/discopt.org` reported that JAX numerical
  computation was 92–99.9% of solve wall time. Re-measured on this tree — ten
  `.nl` corpus instances, three repeats, subprocess-isolated, `discopt.__file__`
  and `__version__` asserted in every child, interleaved arms, spreads reported:

  | | JAX share of wall | Rust | Python |
  |---|---|---|---|
  | default path | **0.00% on all ten** | 1.5–98.5% | 1.0–84.0% |
  | `DISCOPT_NLP_EVAL=jax` | 3.4–12.9% | — | — |

  Even with JAX deliberately switched on it never exceeds 12.9%, so the retired
  figure was not merely stale — it was wrong about where the work happens. The
  Rust/Python split tracks how much search an instance needs, not a language
  boundary: `alan` (21 nodes) is 84.0% Python because import, `.nl` parse and
  model build dominate a 0.2 s solve; `tanksize` (17,095 nodes) is 98.5% Rust.
  Node counts matched between arms on 9 of 10.

  The paper is retitled to "hybrid Rust/Python", §5 is rewritten around the
  uniform factorable pass and the AD tape that actually run, contributions 2 and
  3 are recast (batched node evaluation is POUNCE's Rust `solve_nlp_batch`, not
  `jax.vmap`; the pure-JAX IPM is reported as retired, with the measurement that
  retired it), and the αBB section now leads with the *rigorous* interval-Hessian
  Gershgorin bound rather than the sampled `jax.hessian` estimator — a soundness
  distinction, since sampling cannot certify a bound over a box.

  The same stale claim was corrected in `README.md`, `docs/intro.md` and
  `CLAUDE.md`. README additionally asserted that the relaxation layer "still uses
  JAX for envelope evaluation and cut separation"; measured over eight nonlinear
  instances, both of those run (`uniform_relax`, `mccormick_lp`,
  `incremental_mccormick`; `cutting_planes`, `multilinear_separation`,
  `psd_cuts`) and zero `jax` modules are imported.

  `ROADMAP.md` gets a note rather than an edit. Its phase tables record each task
  as completed at the time — "T8 NLP evaluator … via JAX", "T17 GPU-batched IPM:
  Pure-JAX IPM solver", "T7 HiGHS LP wrapper", "T5 … 19 relaxation functions" —
  and four of those no longer describe the tree (the tape evaluator, the retired
  IPM, the in-house Rust simplex, and 28 relaxation functions). Back-editing a
  historical record would erase what actually happened, so the rows stand and a
  header block states which ones are superseded and that the tree wins.

  A re-audit of `manuscript/fact_check_report.md` (headed "Status: ALL ERRORS
  FIXED", 2026-02-15) found four of its six items still unfixed, including a
  McCormick relaxation count that was wrong in both the report and the
  manuscript (`mccormick.py` defines 28 `relax_*` functions). All are fixed and
  the report's header now records what it actually asserted versus verified.

- **All 62 documentation notebooks re-executed** (`docs`). Their committed
  outputs dated from 2026-04-25 and `docs/_config.yml` has
  `execute_notebooks: "off"`, so nothing had re-run them in three and a half
  months. All 62 execute cleanly. Across 372 paired objective/bound comparisons
  no objective moved by more than 1e-4 relative; the diff is dominated by
  wall-clock numbers. Three visible improvements: the "POUNCE LP returned an
  infeasible point labeled optimal" fallback no longer fires anywhere (it
  appeared six times across three notebooks), `tutorial_milp` now reports a real
  gap where it printed `Gap: None`, and the debugger's `iter_start` checkpoint
  now carries a root bound instead of `-inf`.

  The sweep also surfaced #981: `tutorial_dae`'s parameter-estimation cell had
  been running 493 s to report `feasible`, because collocation models got a dual
  bound of exactly 0.0 and the relative gap can never close. That cell was
  capped with an explicit `time_limit` as a stopgap; the cap and the note
  explaining it are gone now that #981 is fixed and the cell certifies. Its
  prose was also corrected —
  it still described `least_squares` as matching measurements to the nearest
  node, which stopped being true in #101.

- **`discopt._jax` is now `discopt._relax`** (`refactor`). The package was named
  for JAX when JAX built the relaxations. It no longer does: the 128 modules
  under it are numpy, and JAX does not enter `sys.modules` at any point during a
  solve. A package whose name describes none of its contents is a standing
  source of wrong conclusions — anyone grepping for JAX on the solve path found
  1818 import lines and could reasonably believe it was still there.

  `_relax` names what the modules actually are: McCormick/alphaBB envelopes,
  cutting planes, factorable reformulation, convexity detection, the DAG
  compiler and the relaxation compiler.

  The rewrite was restricted to the exact tokens `discopt._jax` and
  `discopt/_jax`, so identifiers that merely look similar (`t_jax_start`,
  `self._eval_jax`, `eval_jaxpr`, and the many `test_*_jax_free_*` names) are
  untouched. All 126 submodules import under the new name; `discopt._jax` now
  raises `ModuleNotFoundError` rather than resolving to a shim, so a stale
  reference fails loudly instead of silently working. A new
  `python/tests/test_relax_package_name.py` guards against reintroduction —
  every other branch still spells it `discopt._jax`, and a merge that brings one
  of those lines back would otherwise only surface on whichever test happened to
  touch that module.

### Fixed

- **`docs/notebooks/benchmark_dashboard.ipynb` crashed on a clean clone**
  (`docs`, #1049). `results/` is gitignored, so a fresh checkout has none, and
  the dashboard's third code cell built `pd.DataFrame([])` — a frame with *no
  columns* — then immediately indexed `df["status"]`, raising `KeyError`. Every
  other cell already degraded gracefully; this one line sat upstream of all of
  them, so `make notebooks` failed at 2 of 9 cells. Passing `columns=`
  explicitly yields an empty-but-typed frame: 9 of 9 cells now execute on an
  empty `results/`, and 9 of 9 on a populated one, still loading the same 5
  categories / 107 rows / four solvers as before.

- **The benchmark layer profile could not report a measured zero** (`fix`). Both
  producers of the Rust/JAX/Python time fractions —
  `benchmarks/runner.py` and `benchmarks/_subprocess_worker.py` — computed the
  share as `x / wall if x else None`, so a layer measured at exactly `0.0`
  became `None`. `layer_profiling_summary` filters `None` out, so
  `mean_jax_fraction` then came back `nan`, which reads as *"this was never
  instrumented"*.

  This inverted the instrument on the one case it exists for. On the `global50`
  panel `jax_time_fraction` was `None` on **50 of 50** instances — not because
  the profile failed, but because JAX genuinely no longer runs on the solve path
  — and `rust_time_fraction` on 7 of 50 fast instances for the same reason. The
  field would have read identically if JAX had been running throughout.

  Fractions are now computed by a shared `metrics.time_fraction()` that branches
  on `is None`, so a measured `0.0` records as `0.0` and `None` is reserved for a
  solver that does not report the field at all. Verified end-to-end: on `main`
  the smoke suite now reports a JAX fraction up to 0.93, on the JAX-removal
  branch it reports 0.0 — two states the old encoding could not tell apart.

  Also adds the missing **`pounce_time_fraction`**. POUNCE is where the NLP cost
  went once it left JAX (78% of a `tspn08` solve per `discopt._timing`), and the
  benchmark record had no field for it, so the dominant layer was invisible. It
  reads `None` until the `_timing`/`pounce_time` work merges, then populates.
  `layer_profiling_summary` gains `n_profiled` so callers can distinguish
  "0.0 across 50 runs" from "0 runs reported it", and the generated report
  renders an unreported layer as `not reported` rather than `nan%`.

- **The POUNCE-native NLP path now requires — and verifies — a cross-thread-safe
  `NlProblem`** (`fix`, #932). #932 reported the pyo3 panic
  `_pounce::nl_problem::PyNlProblem is unsendable, but sent to another thread`
  and asked for one of two outcomes: pin the object to one thread, or establish
  that it is safe to send and drop the marker. **POUNCE took the second route** —
  pounce#477 ("Let one NlProblem serve a whole worker pool") removed the
  `unsendable` marker after establishing that `NlTnlp` is `Send` and that every
  `&self` method evaluates under the GIL, citing the same reasoning #932 did:
  `PanicException` derives from `BaseException`, so a cross-thread access slips
  past every `except Exception` in a host and surfaces as a wrong answer rather
  than an error, and the *drop* path tripped even for code that never used an
  instance cross-thread. There is therefore no discopt-side thread-affinity bug,
  and the model-level base cache — one `NlProblem` shared by every thread that
  solves a node — is correct as written.

  What changed here is the requirement being made explicit instead of assumed.
  POUNCE's version string cannot express it (pounce#477 landed inside an
  unreleased `0.9.0`, so a pre- and post-fix build are both "0.9.0"), so
  `nlp_native` now *measures* the guarantee once per process on the first base
  built — a real second thread touching the problem, catching `BaseException`
  because a pyo3 panic is one — and disables the native path with a warning
  naming pounce#477 when it does not hold, falling back to the JAX callback
  bridge rather than letting an uncatchable panic abort a solve. The stale
  "enable explicitly once PyNlProblem is made Send-safe" note in `solver.py` is
  corrected; `DISCOPT_NLP_NATIVE` stays default-OFF on its remaining grounds
  (neutral-to-modest speedup, MIQP-batch certification perturbation), which is a
  §5 graduation-panel decision rather than a bug fix.
  Tests: `test_932_native_base_thread_sharing.py`.

### Added

- **`docs/notebooks/tutorial_gbd.ipynb`** — a tutorial for Generalized Benders
  Decomposition (`docs`). `solve_gbd` had shipped with no notebook of its own;
  `tutorial_benders` covers only the classical linear-recourse case. The new
  notebook derives the Lagrangian cut, then runs the master/subproblem loop by
  hand on a model whose value function and multiplier are closed form
  (`v(y) = (3 - 2y₁ - y₂)₊²`, `μ = 2(3 - 2y₁ - y₂)₊`), so the mechanics are
  visible without a solver in the loop — it converges in two iterations, LB
  4 → 5 against UB 9 → 5, and `solve_gbd` reproduces the same optimum. A
  capacity-expansion model with convex-quadratic congestion cost then shows the
  three entry points (`solve_gbd`, `solve_benders` dispatch, and
  `Model.solve(decomposition="benders")`) and a GBD/OA/spatial-B&B comparison, all
  three agreeing on 20.8000. Covers the dominance hierarchy `z_OA ≥ z_GBD ≥ z_LP`
  (Grossmann 2002), the convexity requirement for a rigorous bound, and the #946
  degenerate-recourse failure mode with its integer L-shaped fallback (Laporte &
  Louveaux 1993). New BibTeX entries: `Grossmann2002`, `Laporte1993`.

- **Anytime dual bound: root seeding + one unified report** (`fix`, #933).
  27% of a 200-instance MINLPLib sweep reported **no dual bound at all** at
  `time_limit=8`, for two architectural reasons #933 measured: (a) the root
  relaxation bound proved during setup was never installed into the tree, so
  `global_lower_bound` stayed `-inf` until the first fully-processed batch; and
  (b) the reported bound was re-derived independently in five solve exit paths,
  four of which had no recovery when the tree bound was unusable — including
  silently discarding a perfectly *valid* finite tree bound on every
  no-incumbent limit exit by conflating "the exit is not certified" with "the
  tree bound is invalid".

  Part (a): `TreeManager::seed_root_bound` (new PyO3 entry point) floors node
  0's `local_lower_bound` — and permanently the reported global bound — at a
  root-box bound proved by the *same trusted engine* that bounds the path's
  nodes (the spatial path's McCormick root probe behind the #930 box-equality
  gate; the MILP path's structured-node root LP; the #781 HiGHS root-cut bound
  is deliberately not seeded). Children inherit the floor, so the dual bound is
  anytime and monotone by construction. The native spatial kernel gets the
  kernel-side mirror of the #844 policy: its deadline is shortened by the
  root-fallback reserve, reclaimed once (`bound_time_extension`) the moment its
  own bound is finite, so only a bound-less kernel forfeits the slice — which
  the caller then spends on the rigorous root-relaxation fallback. Seeding and
  the kernel reserve are **bound-changing** and gated by
  `DISCOPT_ROOT_BOUND_SEED` — graduated **default ON** through the Regime-2
  panel (`discopt_benchmarks/scripts/issue933_seed_graduation_panel.py`,
  66 in-repo instances at TL=8, paired OFF/ON: 31 oracle checks with 0
  crossings, 0 certification regressions, 0 objective drift; bound coverage
  61→62, ON tighter on 6 / looser on 0 paired bounds, one
  feasible→certified-optimal upgrade, wall mean −0.55 s; raw log in
  `discopt_benchmarks/results/issue933/`). `=0` opts out and restores the
  legacy unseeded tree.

  Part (b) (unconditional, reporting-only): `_finalize_reported_bound` is the
  one chokepoint for an uncertified exit's reported bound — taint rule applied
  once (valid+finite+sub-sentinel at the 1e19 threshold, closing the #930
  `np.isfinite(1e20)` hole for every path), independent-root-bound fallback
  applied once (root LP / untainted root-batch snapshot; tightest wins), the
  `bound <= incumbent` certificate invariant enforced, sense mapped once. The
  NLP-BB/MILP-BB/MIQP-BB paths now capture the taint flag *before* the
  exit-status logic overwrites it, so a valid tree bound survives no-incumbent
  time/node-limit exits, and the MILP path reports its live frontier bound
  instead of the stale root LP value on budget-limited exits.

- **Deterministic work budgets for the root primal heuristics** (`fix`, #912).
  The search tree was a function of *machine speed*: the root
  `integer_local_search` bounded its own extent with a wall clock
  (`time_budget = min(5.0, 0.15·time_limit)`), its descent routinely never
  converges, so the incumbent it handed the tree — and therefore `node_count` —
  depended on how fast the box was. #912 measured `gear2` closing in 3 nodes at a
  5 s budget and 91 at 3 s, with the default sitting exactly on that cliff. This
  invalidates the repo's bound-neutral verification regime at the root: "node
  counts exactly unchanged" is only meaningful for a function of the input.

  New `discopt._work_budget.WorkBudget` counts *operations* instead of seconds,
  per kind: constraint/objective evaluations and continuous-repair sub-NLP
  solves, each with its own cap (`SolverTuning.ils_eval_budget` /
  `ils_solve_budget`, `DISCOPT_ILS_EVAL_BUDGET` / `DISCOPT_ILS_SOLVE_BUDGET`,
  defaults 20 000 / 128; both `0` restores the legacy wall gate). Two counters
  rather than one converted currency because the conversion was tried first and
  measurably failed — a sub-NLP costs 2 933-77 964 evaluations depending on the
  instance, and at the geomean `nvs09` regressed 5 -> 29 nodes while `syn05hfsg`
  got 3x its legacy wall time. The clock stays only as a **backstop**: the
  caller's solve deadline is passed down so `time_limit` is still honoured, and
  `WorkBudget.stopped_on` records whether a search was decided by work
  (reproducible) or by the clock (not).

  The other three extent gates in that layer follow the same conversion:
  `integer_box_search` (cell enumeration), `one_hot_swap_search` (swap descent)
  and `local_branching` — the last being the purest form of the bug in the repo,
  since it predicted a round's cost as `C(n, r) x measured_mean_subnlp_seconds`
  against `deadline - now`, making the enumeration radius a ratio of two machine
  measurements. Each takes a share of the ILS budget matching the wall slice it
  used to get (0.8 / 0.2 / 0.4); handing all four the ILS number was measured
  interleaved at 1.38x wall on syn05hfsg and 1.18x on fac2 for identical node
  counts, and did not ship.

  A **global deterministic work clock** for the remaining ~20 wall budgets
  (OBBT, nonlinear bound tightening, root cuts, PSD separation, convexity
  classification, …) was built twice and **falsified twice**, and the second
  result is the load-bearing one. Charging the four primitives at their proper
  boundaries — the NLP *backend* rather than one call site, which profiling fac2
  showed was seeing 5 of its 80 solves — lifted coverage only from 0.13 to 0.19
  of wall (fac2: 1.59 deterministic seconds against 20 s). The arithmetic says
  why: fac2's NLP solves cost ~86 ms each against a 15 ms nominal price, drawn
  from a 1.9-104 ms distribution, so **no fixed pricing can track wall time when
  the per-operation cost varies 55x across instances** — even with perfect
  coverage. A seconds-valued budget therefore cannot be re-denominated in
  deterministic units without being re-tuned, and each remaining gate needs its
  own natural unit, its own calibration and (for the bound-changing ones) its own
  differential-bound panel. Both attempts reverted; measurements in the
  calibration doc §9. The gates are enumerated and ratcheted by
  `python/tests/test_912_wall_budget_inventory.py`, which fails on any new
  unrecorded one.

  #912 is closed **not planned** for those 20: after the conversion the
  clock-scale panel is 18 in-scope comparisons with 0 mismatches, and no residual
  gate was ever observed moving a tree. That evidence is bounded by corpus
  coverage — these budgets bind mainly on large models and the in-repo corpus is
  66 small ones — so the trigger to revisit is a large-instance panel showing one
  of them move a tree inside its `time_limit`. See the calibration doc §11.

  Measured on the in-repo corpus (`item912_clock_determinism_probe.py`): under
  the old gate 7 of the 22 ILS-firing instances had their extent cut by the clock
  (nvs09, ex1224, st_e29, ex1225, tspn05, syn05hfsg, fac2) — the gear2 mechanism,
  in-repo — and two returned different node counts across two identical sweeps on
  the same machine. With the fix, clock-scale panel at `time_limit=60`:
  **18 in-scope comparisons, 0 mismatches** at 1x vs 2x (the 4 out-of-scope rows
  are whole-solve budget starvation, reported with their numbers). Flag ON vs OFF:
  **88 field comparisons over 22 instances, 0 differences** in status, node count,
  objective and bound, at +1.6 % total wall. See
  `docs/dev/work-budget-calibration-2026-08-01.md`.

- **Perspective terms in the convex-kernel producer** (`feat`, #865, inside the
  default-off `DISCOPT_CONVEX_KERNEL` path). Hull-reformulated (`*hfsg`) models
  write their disjunctive nonlinearities as the perspective `s·f(a/s)` with the
  smoothed indicator `s = 0.001 + 0.999·y` — e.g. `syn05hfsg`'s
  `(x2/ε − log(x0/ε + 1))·ε ≤ 0`. Syntactically that is a product of two
  non-constant subexpressions, so the gate rejected the whole family as a
  "bilinear product"; mathematically the perspective of a convex `f` is *jointly
  convex* in `(a, s)` on `s > 0`, so admitting it **recognises convexity the
  syntactic gate missed rather than loosening the gate**. `_decompose` now
  recognises the shape, lifts `s·h(·/s)` to affine terms plus perspective terms
  (an algebraic identity), and `CompositeTerm` carries an optional affine `scale`
  whose value/gradient feed the existing OA-tangent machinery unchanged
  (`∂/∂a = coeff·f'(t)`, `∂/∂s = coeff·(f(t) − t·f'(t))`, `t = a/s`).

  Three hard gates gate admission, each a refusal rather than an approximation:
  the existing `sign(coeff)·curvature(func) ≥ 0` curvature rule; `s > 0` **proven
  by interval arithmetic over the variable box** (the convexity precondition —
  the smoothing floor is exactly what makes it hold); and `a/s` inside `func`'s
  domain over the box. A genuine bilinear product, a mismatched denominator, and
  a scale that touches 0 all still fall back to the (always-correct) default path.

  Measured: `syn05hfsg` goes from *declined* to **certified optimal 837.7324009
  in 2 nodes / 0.00 s**, matching the BARON reference and the value the
  already-trusted path gets on its sibling `syn05m`. Verified over the box —
  marshaled rows equal the pristine model's pointwise (worst relative error
  1.0e-15, so the lift is exact) and every routed row satisfies the midpoint
  convexity inequality (worst violation 0.0). Bound-neutral for everything routed
  before: `syn05m` and `cvxnonsep_psig40r` marshal and solve identically
  (node counts 3→3, 1→1; objectives bit-identical). Corpus-wide over the 147
  in-repo `.nl` instances the only routing change is `syn05hfsg` being admitted;
  nothing previously routed was lost.

- **Quadratic inner function (`** 2` → `sqr`) for the convex kernel** (`feat`,
  #879, same default-off `DISCOPT_CONVEX_KERNEL` path). `clay*hfsg`'s hull rows are
  `ε·((x/ε)² − c·x/ε + …)`, i.e. the quadratic perspective `x²/ε`
  (quadratic-over-linear, jointly convex on `ε > 0`), so they declined only because
  `_FUNC`/`ConvexFunc` had no `sqr` entry. Adds `ConvexFunc::Sqr` and `_pow_as_sqr`,
  which composes with the perspective machinery unchanged — a plain `x² ≤ c` row is
  routable too. **Only the exponent 2 is admitted**; every other power (odd,
  fractional, negative, or variable) is nonconvex, domain-restricted, or signomial,
  and keeps falling back, as does a non-affine base such as `(log x)²`.

  This class was withdrawn once (evidence recorded in #879) on the reading that its
  `a²/s` tangents produced an invalid (too-tight) relaxation. **That reading is falsified** — see
  the *Fixed* entry below. Re-admitted here together with the three guards that let
  it certify: the dominated-column bound, a per-node relaxation size cap, and a cold
  retry of a numerically broken warm re-optimize.

  Measured: `clay0303hfsg` goes from *declined* to **certified optimal
  `26669.1096` in 211 nodes** end-to-end through `Model.solve()` (39.2 s on an idle
  machine, single run — the node count is the deterministic figure), matching
  its MINLPLib reference (`26669.10955143`, now recorded in
  `python/tests/data/known_optima.toml`) and incumbent-verified against the pristine
  model. Bound-neutral for everything routed before: `syn05m` (3 nodes), `syn05hfsg`
  (2 nodes) and `cvxnonsep_psig40r` (1 node) are bit-identical in status, objective
  and node count. Over the 149 in-repo `.nl` instances `clay0303hfsg` is the only
  routing change.

- **Dominated-cost-column upper bound for the convex kernel** (`feat`, #871/#879,
  default-ON inside the default-off kernel, opt out with
  `DISCOPT_CVX_DOMINATED_COLS=0`). FBBT cannot close a column the constraints only
  bound from *below*; `clay0303hfsg` has six (its fixed-charge cost variables
  `x81..x86`, objective coefficients 300/240/100/…), and an infinite `ub` is what
  makes the node LP break down and the Neumaier–Shcherbina safe bound decline.
  `tighten_dominated_columns` gives such a column the finite bound
  `U_j = max(lo_j, max_i (maxact_rest_i − rhs_i)/(−a_ij))`.

  This is an **optimality** argument, not FBBT: it qualifies a column only when its
  minimized-objective coefficient is `> 0`, it appears in no equality and no
  nonlinear row, and every `≤` row containing it has `a_ij < 0`. Lowering `x_j` to
  `U_j` from any feasible point then keeps every row satisfied and strictly lowers
  the objective, so the node's optimal *value* is unchanged and the dual bound stays
  valid — but unlike FBBT the box no longer contains every feasible point, which is
  why it keeps its own switch. Measured: turning it off takes `clay0303hfsg` from
  `optimal` back to `exhausted`, and it is a bit-identical no-op on every other
  routed instance.

### Changed

- **CI no longer runs a nightly, and the three failure reporters can now reach
  GitHub** (`ci`). Two coupled corrections to `23fff2a6`:
  1. **The nightly schedule is removed.** A daily ~2 h runner job against a SHA
     that has usually not moved is not worth the CI minutes — the same judgement
     `143bd1e9` made on 2026-05-28 when it deleted the previous nightly
     ("duplicate work and the source of false-alarm emails"). The lane survives
     as `python-correctness-slow`, `workflow_dispatch`-only. **Stated cost:** its
     153 `slow`+`correctness` tests now have no continuous watch; running them is
     a deliberate act before a release or after a change to the certificate path.
     Two tests in `test_ci_changes_gate.py` pin the decision so a third re-add of
     a `schedule:` trigger is a red check rather than a silent line.
  2. **All three `report-failure` jobs died on their first `gh` call.** They skip
     `actions/checkout` by design, so `gh` — which resolves the repo from the git
     remote — exited `fatal: not a git repository`. The three reporters titled
     "repair three signals that had stopped signalling" had therefore never
     signalled: the `ci-signal` label does not exist and no issue was ever filed.
     A real solver defect (#977) sat unrecorded for two days behind a red nightly
     because of it. Fixed with `GH_REPO: ${{ github.repository }}`, which needs no
     checkout. The `ci.yml` reporter also loses its now-unreachable
     `github.event_name == 'schedule'` gate.

- **The #843 QUBO/Ising local-search primal is now default-ON** (`perf`, #843,
  graduating the #846 seed; opt out with `DISCOPT_QUBO_PRIMAL=0`). An
  unconstrained all-binary quadratic model (the `chimera_k64ising` /Max-Cut
  structure) used to return **no incumbent**; the greedy-1opt + tabu seed now
  fires by default and lands a sound one. Graduated on the §5 differential
  panel (`docs/dev/data/issue843-qubo-primal-graduation.md`): structural
  no-fire proven on all 67 vendored instances, byte-identical ON-vs-OFF on the
  fast corpus subset, brute-force-exact on small QUBOs, and strictly better on
  every structure-carrying case (generated chimera-topology Ising: C8 512-var
  none→846, C12 1152-var none→1770; 0 soundness violations). At graduation the
  heuristic moved to the JAX-free `discopt/qubo_primal.py` (structural gate +
  MIQP classification gate + the `extract_qp_data` ladder instead of the JAX
  Hessian), so the default-ON gate keeps the pure LP/MILP cold-start path
  JAX-free; a linear objective is now explicitly left to the exact simplex
  MILP path and a degree>2 objective is refused (the #846 code did not enforce
  the degree cap its docstring claimed). The in-pass deadline poll (every 256
  tabu iterations) bounds the seed's cost on any budget.

- **`gap_certified` now requires BOTH ends of a gap** (`fix`, #875). A limit exit
  with a valid dual bound but *no incumbent* used to report
  `status="time_limit", objective=None, gap=None, gap_certified=True` — a certified
  gap where no gap was ever formed. `SolveResult.__post_init__` already required a
  finite bound; it now also requires an objective, for every status except
  `infeasible` (whose certificate is not a gap). The downgrade is True → False only,
  so it cannot manufacture a certificate, and the dual `bound` is left in place
  because a bound with no incumbent is still a valid bound.

### Fixed

- **Two CI signals that reported without measuring** (`ci`). Both were found while
  driving #960 to green, and both share a shape: a check that answers confidently
  from something other than the thing it claims to be testing.

  1. **The `changes` gate answered from a guessed diff on all three event
     arms.** It resolves a base commit and diffs against it; every arm had a
     path where the base is unavailable and the code substituted
     `HEAD~1..HEAD` — the tip commit's file list — instead of recognising that
     it did not know. When that guess lands on a docs-only commit the gate says
     `code=false` and every solver job below it reports `skipped`, which renders
     as *not red*. That is the #953 failure mode: a wrong `false` here is not a
     missing test, it is a green-looking wall of nothing.

     * `workflow_dispatch`: `github.event.before` exists only on `push`, so it
       expands to the empty string, the first `git diff` fails, and the fallback
       fires. Dispatching CI on #960, whose tip touched `CHANGELOG.md` and
       `docs/dev/data/claim-baseline.jsonl`, produced `code=false` and left
       Rust, `Python fast`, `Python claim-boundary`, `Python correctness` and
       AMP all `skipped` — nine checks that executed nothing. Now `code=true`
       unconditionally: pressing the button means "run everything", and there is
       no base to diff against anyway.
     * `pull_request` — the trigger that fires on nearly every change here —
       used `${base:-HEAD~1}`, reachable two ways: the `git fetch` above it is
       `|| true`, so a fetch failure is silent, and the `--depth=200` fetch
       installed a shallow boundary that hides the merge base of a branch forked
       further back than that. Either way a PR that changes the solver behind a
       docs-only tip commit skipped every solver job. The depth cap is gone (the
       checkout is already `fetch-depth: 0`) and an unresolvable base is now
       treated as uncertain.
     * `push`: an all-zero `before` (a branch's first push), a non-existent
       object (force-push, shallow clone), or an empty one is likewise the
       *uncertainty* case the job's own comment already promised to resolve as
       `true`.

     New `python/tests/test_ci_changes_gate.py` extracts the gate's real shell
     script out of `ci.yml` — not a copy, so it cannot drift — renders the
     `${{ github.* }}` expressions, and runs it against throwaway git repos, one
     case per event type. 12 tests; 6 fail on the pre-change workflow (one per
     defective arm plus the four `push` base shapes) and the 6 pinning behaviour
     that must not change (docs-only PRs and pushes still skip, a code file
     runs, `schedule` still skips the PR matrix, a resolvable merge base is still
     used in preference to the tip) pass both ways. `pyyaml` moves into the `dev`
     extra so that file is a skip rather than a collection error where it is
     absent.

  2. **Two `ex1252` quality bars measured the runner, not the relaxation.**
     `compute_disjunctive_config_bound` stops on whichever of `max_leaf_solves`
     and `deadline` comes first, and reported no indication of which. The leaf
     budget is reproducible; the clock is not. The nightly lane read 33498
     (42/120 leaves) and 31567 (32/48) on a runner ~3x slower than the reference
     machine and failed both bars as if the bound had regressed. It had not: at
     the full budget the bound is **bit-identical** across the #957 boundary —
     37945.427865923564 at 48 leaves and 63080.286987442756 at 120, matching the
     values the tests' own docstrings record, with the same leaf and prune counts
     in both arms.

     `DisjunctiveConfigResult` gains `stopped_on` (`"budget"` / `"deadline"` /
     `"exhausted"`), mirroring `WorkBudget.stopped_on` from #912 — the same
     distinction between a search decided by work and one decided by the clock.
     The two bars now assert `stopped_on != "deadline"` *before* the bound
     comparison, so a slow machine and a bound regression produce different
     failures with different instructions, and the clock-bound message says
     explicitly not to lower the bar. Deadlines are re-sized as backstops at ~2x
     the measured runner rate (0.175 leaf solves/s → 600 s for 48 leaves, 1500 s
     for 120), and the nightly lane's `--timeout` goes 900 s → 1800 s so the
     harness timeout is never what decides a quality bar. The 120-minute job
     timeout remains the backstop against a hang.

- **Outward rounding no longer turns an exact zero into a subnormal** (`fix`, #957).
  `interval.py`'s `_round_down`/`_round_up` are `np.nextafter(x, ∓inf)`, and
  `nextafter(0.0, ±inf)` is `±5e-324`. At every other magnitude "one ULP" is a
  negligible *relative* widening; at zero it crosses into the subnormal range and
  manufactures a coefficient ~300 orders of magnitude below everything else in
  the model. Measured on the in-repo corpus, **384 of 730 relaxation LPs carried
  such a value**, and in 67 of them the coefficient-spread guard at
  `milp_relaxation.py:548` — `nz.max() / nz.min() > TRIGGER` — *overflowed to
  `inf`* while making its decision (with a `RuntimeWarning` as the tell), so
  equilibration fired unconditionally regardless of how benign the matrix was.

  **The obvious fix — leave every exact zero alone — is unsound, and was not
  taken.** The nudge at zero is load-bearing wherever the producing operation can
  underflow: `fl(exp(-800))` is `0.0` while the exact value is `3.6e-348`, so a
  blanket special-case would give `exp` an upper endpoint *below* its own image.
  What ships instead is an opt-in pair, `_round_down_exact0` / `_round_up_exact0`,
  applied only where `fl(result) == 0 ⟹ result == 0` is provable: `+`, `-` and
  `width` (every double is a multiple of `2**-1074`, so a float sum is zero only
  when the exact sum is — fuzz-checked against `longdouble` over 14 268 cancelling
  pairs), `abs`, `sqrt`, `log`, the reciprocal, and `*` / `**2` behind an explicit
  no-underflow guard on the corner factors. The generic monotone wrapper keeps the
  unconditional helpers, so a future atom inherits the safe behaviour by default.

  Second, independent half: `_coefficient_spread_exceeds()` replaces the open-coded
  spread test at both conditioning seams. It uses the overflow-free cross-multiplied
  form (already applied at the false-infeasible seam by #732 Stage 1-B) *and* drops
  entries below the smallest normal double, so the guard measures the conditioning
  of the model rather than of an underflow artifact. Both halves were needed: the
  interval fix clears `A_ub` on 14 of the 15 affected instances, and `st_e11`'s
  subnormals — which come from a different producer — are neutralised by the floor
  (19 overflowing LPs → 0).

  Corpus panel (66 instances, 10 s budget, arms interleaved): **0 soundness
  violations** against the 16 oracle-checked instances, total node count identical
  (3298 → 3298), subnormal-carrying LPs 384/730 → 28/782, spread-test overflows
  67 → 0. Every status/bound/node difference the panel reported was on an instance
  sitting within ~1 s of the deadline; re-measured unloaded with the arms
  interleaved, the two decisive ones (`cvxnonsep_nsig30`, `st_e36`) are bit-identical
  between arms.

  Found while fixing the above: `psd_2x2_sufficient` had the same bug pointing the
  other way. It skipped its upward round when `fl(off**2) == 0.0`, which is *not*
  an exact zero when `off` is around `1e-200` — the exact `off**2` is ~1e-400, so
  a matrix with a (barely) negative determinant could pass the PSD test. The zero
  test now reads the *factors*, not the flushed product. `gershgorin_lambda_min`
  /`_max` gain the same treatment, so a linear function's all-zero Hessian bounds
  come back as exactly `0.0` instead of `∓5e-324` and no longer miss a boundary
  `λ_min ≥ 0` test.

  **Claim-boundary baseline regenerated, attributed against a `main` control.**
  This is a bound-changing change, so `docs/dev/data/claim-baseline.jsonl` moves —
  but `claim_differential`'s contract is that every changed instance be
  independently attributed, not merely re-recorded. A naive regeneration is
  *not* attributable here: the committed baseline was stamped at `605d29b` and
  its `root_lp_bound` column had gone stale, so regenerating on this branch
  showed 56 bound moves and looked like this PR had done all of them.

  Three-way measurement, with the control arm asserted to lack the marker
  (`_round_down_exact0` absent in the `main` tree, checked before generating):

  - **committed `605d29b` → `main` `d5f6eff6`** (62 instances): `SHAPE_MOVED=0`,
    `ROOT_LP_MOVED=52`. All 52 predate this branch, including every one that
    looked alarming — `nvs13` −4843.01 → −751.12, and `beuster`/`casctanks`/
    `st_miqp2`/`st_miqp3`/`st_miqp4` dropping to no recorded root bound. The
    *shape* column was never stale, which is why the gate — shape-only — has been
    passing on `main` throughout.
  - **`main` → this branch** (66 instances): `SHAPE_MOVED=7`, `ROOT_LP_MOVED=18`.
    That is this PR's actual footprint.

  The 7 shape movers are rows-only — column and integer-column counts unchanged
  everywhere, the signature of extra cuts rather than a different formulation —
  and they are the claim-boundary job's list instance for instance: `bchoco06`
  832→848, `bchoco07` 1081→1163, `bchoco08` 1604→1898, `fac2` 54→66, `tspn08`
  604→609, `tspn10` 935→940, `tspn12` 1340→1345. An independent two-arm probe in
  a single tree (OFF = the opt-in helpers rebound to the unconditional
  `_round_down`/`_round_up` in both modules plus the pre-#960 `psd_2x2_sufficient`
  at both its import sites) reproduces exactly that set, and its OFF arm
  reproduces the committed baseline shape for all 62 rows
  (`CONTROL_MATCHED_BASELINE=62 MISMATCHED=[]`).

  Of the 18 bound moves, 12 are last-digit (≤1e-10 relative: `bchoco06/07/08`,
  `hda`, `heatexch_gen1/2`, `nvs12`, `nvs13`, `nvs21`, `st_e36`, `st_miqp1`,
  `tspn05`). The 6 substantive ones are all tightenings that stay below their
  reference optimum — every one of these instances minimizes (`O0 0`), so a valid
  dual bound may rise but never past the optimum:

  | instance | `main` | this branch | reference optimum |
  | --- | --- | --- | --- |
  | `fac2` | none | 303 398.521 | 331 837 498.2 |
  | `tspn08` | none | 231.983 | 290.567 |
  | `tspn10` | none | 161.161 | 225.126 |
  | `tspn12` | none | 183.326 | 262.647 |
  | `chance` | 26.350 | 28.943 | 29.894 |
  | `nvs04` | −5e-322 | 0.0 | 0.72 |

  `nvs04` is the bug itself in the baseline: the recorded root bound *was* the
  subnormal artifact, and it is now an exact zero. Solving the relaxation MILP
  rather than the root LP tells the same story — `fac2` no usable bound →
  2 132 258.914, and `tspn08/10/12` go from **unbounded** to a finite valid bound
  (246.211 / 183.117 / 203.126), all below their optima, `UNSOUND=0`.

  The 52 pre-existing `root_lp_bound` drifts are carried in by this regeneration
  because the file is generated wholesale; they are recorded here as measured on
  `main`, not claimed as this PR's doing.
- **The MILP/MIQP B&B incumbents are verified against every declared row and
  bound before they are returned** (`fix`, #952). `_solve_miqp_bb`'s only
  feasibility gate was `_check_lp_solution_feasibility(A_eq_full, b_eq, x_full)`
  — an equality residual at `tol=1e-4`, 100x the declared `abs=1e-6`, with no
  inequality-row check, no bound check, and no verification of the point whose
  objective is returned as `objective` **and**, on `optimal`, as the dual `bound`.

  That gate was a **tautology** on the path that runs.
  `_pounce_qp_relaxation_nodes` solves only the structural columns and then
  *reconstructs* the slacks as `z = S⁺(b_eq − A_struct x_s)`, so
  `A_eq_full [x_s, z] == b_eq` holds to machine precision for **any** `x_s`: a
  violated inequality row comes back as a negative slack, and the gate never
  looked at slack bounds. Measured over the issue's 40-seed family: 212 gate
  invocations, worst equality residual **8.9e-16**, while every returned
  incumbent sat ~9e-9 outside a declared inequality row. Those excursions are
  inside `abs=1e-6`, so this was never a live false certificate — the defect is
  that *nothing bounded them*; their size was set by whatever the QP IPM
  converged to.

  Node points are now gated on the decomposed `A_ub`/`A_eq` rows plus the node's
  own box (`_matrix_solution_feasible`, the repo's single arbiter, at
  `abs=1e-6`), matching what the MILP path's `_solve_node_lp_pounce` already did;
  a snapped incumbent is verified before `tree.inject_incumbent`; and every
  incumbent leaving the function is verified against every declared row and the
  declared box (captured before FBBT), raising rather than silently repairing or
  downgrading the status — the same loud refusal as the Gurobi QCP path. The gate
  covers `feasible` as well as `optimal`: `objective` is a primal claim in both.
  `_check_lp_solution_feasibility` had no callers left and is **deleted**, which
  closes its unexplained 1e-4 rather than restating it.

  A survey of the five solve exit paths found the same omission on
  `_solve_milp_bb`, whose incumbent exit was structurally identical (round,
  unpack, return); it gets the same gate, since fixing only the reported path
  would be a single-instance fix of a per-path defect. `solve_model` (#779) and
  `_try_native_spatial_kernel` (#789) already verify. `_solve_nlp_bb` is left
  open in #954: its rows are nonlinear, so the check needs the evaluator rather
  than the matrix arbiter, with different tolerance semantics that #952 never
  measured. This is the primal-side twin of #933, which tracks the same per-path
  duplication on the dual bound.

  The MILP gate immediately refused
  `test_nn_equivalence::test_tree_ensemble_fixed_input`, and the producer was not
  the solve: the incumbent satisfies its equalities to **4.4e-16** and the
  *snapped* point misses one by **1.55e-6**, from per-coordinate snaps of at most
  3.9e-7 over a 5-term row. The C-3 call site's comment claimed the snap "cannot
  move a linear row by more than the integrality tol" — wrong, a row takes one
  snap per term — and `_round_incumbent_integers`'s `feasible` flag, documented
  as the caller's guard against exactly this, is vacuous at both B&B call sites
  because neither passes it an evaluator. Both now use the matrix arbiter as that
  checker and report the unrounded point when snapping would leave the rows; the
  unrounded point satisfies *both* declared tolerances, so this is a correction,
  not a tolerance concession.

  The MILP gate then refused the `vub=1e3`/`vub=1e6` arms of
  `test_benders_soundness`, and again the producer was upstream of the exit.
  `_pounce_snap_incumbent` fixes the integers and takes the POUNCE IPM's
  continuous completion, which honours the slack bounds standing in for
  inequality rows only to ~1e-8 *relative*; on `v <= 1e6*y` that is a **1e-2**
  absolute excursion. The point comes back exactly integral, so nothing
  downstream re-examines it — the tree took it as the incumbent and reported it
  `optimal`. Traced with a spy on the injection funnel: all 4 node-relaxation
  points were inside the rows, and the offending `[y=1.0, eta=-1.00000001e+06]`
  arrived through this funnel, not a node LP. `_solve_milp_bb`'s
  `_maybe_inject_snapped` now verifies the candidate against the same rows and
  node box the node solves are held to, which is the check the MIQP twin of that
  funnel already ran (`_node_point_feasible`). Declining costs nothing:
  injection is a heuristic accelerator, the subtree stays open, and a genuinely
  feasible point is re-found from a node relaxation.

  A side effect worth naming, and attributed by measurement rather than by
  assumption: `test_946_gbd_degenerate_multiplier`'s non-binary arm used to exit
  `iteration_limit` with bound -2.0, because the master promoted an incumbent
  whose integer snap had left its own rows and the optimality cut added there
  carried no usable `eta` information. It is the C-3 correction above — report
  the unrounded point when snapping leaves the rows — and not the funnel check,
  that changes this: on `origin/main`'s `solver.py` the arm still stalls, and on
  this branch with the funnel check reverted it already certifies. It now
  reaches `optimal` at bound -1.0000000149881925 against an optimum of -1, in 4
  recourse solves. That test pins both arms: the explanation on the uncertified
  exit (reached deterministically with `max_iterations=1`) and the newly
  certified full-budget outcome.

  Verified bound-neutral per CLAUDE.md §5 over a 106-instance panel (the 40-seed
  family plus every in-repo `minlplib_nl` instance), 49 of them through
  `_solve_miqp_bb`: **0 drifts** in status/objective/bound/node_count/
  gap_certified across the 88 self-terminating instances, and byte-identical
  worst violation (9.973171e-09) on both arms. The single apparent drift
  (`tls2`) was falsified as a wall-clock artifact — it dispatches to
  `_solve_nlp_bb`, which this change does not touch, and an interleaved A/B on a
  quiet machine returned bit-identical results (389 nodes, 3/3 rounds per arm).

- **The NLP-BB incumbent is verified against every declared row and bound before
  it is returned** (`fix`, #954). `_solve_nlp_bb` was the last of the five solve
  exit paths with nothing checking the point it returns (#779/#789 closed
  `solve_model` and `_try_native_spatial_kernel`; #952 closed `_solve_milp_bb` and
  `_solve_miqp_bb`). Its exit was the same three steps those two had — snap the
  integers, unpack, return — with no verification of the point whose objective is
  reported as `objective` and, on `optimal`, as the dual `bound` too.

  Two things kept this from being "apply the #952 gate here". Its rows are
  nonlinear, so the arbiter cannot be `_matrix_solution_feasible`; the new
  `_nonlinear_point_excess` applies the repo's nonlinear convention instead
  (`viol <= tol + rtol*term_scale` at `abs=1e-6`, `rtol=1e-9` — the same test
  `_jax.primal_heuristics._check_constraint_feasibility` applies whenever a primal
  heuristic accepts an incumbent), and returns the excess it measured so the
  refusal names a magnitude and the reported number cannot drift from the decision.
  And the terminal refine re-solve can *replace* the reported point after every
  gate the search ran, so the check is the last thing before the return. The rows
  judged are the ones the model declares — root cuts (#781) are appended to the
  model before the evaluator is compiled and are excluded — and the box is the
  declared one, captured before FBBT overwrites it.

  The rounding guard at this call site (#954 item 3) ran at
  `_round_incumbent_integers`'s default `feas_tol=1e-4`, 100x the declared
  `abs=1e-6`. It now takes the exit's own arbiter, so a snap can no longer be
  admitted at 1e-4 and then refused by the exit at 1e-6; a rejected snap keeps the
  unrounded incumbent, which is C-3's documented fallback.

  Entry experiment before the gate was written (CLAUDE.md §4), over a 106-item
  panel — a 40-seed convex-MINLP family plus every in-repo `minlplib_nl` instance
  driven through this path with `nlp_bb=True`: 100 entered `_solve_nlp_bb`, 69
  returned a point, 2638 comparisons; worst raw violation 9.36e-11 (`tls2`), worst
  tolerance that would be needed to admit any returned point 3.93e-14, and **0**
  instances a 1e-6 gate would newly refuse. As with #952 this was never a live
  false certificate — the defect is that nothing bounded the excursion, whose size
  was set by whatever the NLP backend converged to, and the exit would equally
  have returned 1e-2. Bound-neutral per CLAUDE.md §5: the gated arm reproduces the
  baseline's `node_count`, `objective`, status and `gap_certified` on every
  self-terminating instance; the 7 apparent drifts in the contended panel run were
  all on wall-truncated (`gap_certified=False`) searches and vanished on
  uncontended reruns of both arms.

- **Three CI signals that had stopped signalling** (`ci`). An audit of why #927
  (an unsound `ex1252` dual bound, asserted by a test in the tree) could sit open
  found that the lanes meant to catch it were all dark:

  1. **No lane ran `slow` + `correctness` — 153 tests, including every
     end-to-end certificate assertion.** `python-correctness` runs `not slow`;
     its comment said the rest was "the nightly `python-coverage` job's
     responsibility", but that job runs `-m "not slow and not correctness ..."`
     and `ci.yml` had no `schedule:` trigger at all, so there was no nightly. New
     `python-correctness-nightly` job (serial, `--timeout=900`, 06:20 UTC) runs
     exactly that set. `changes` now reports `code=false` on a schedule event so
     the nightly trigger does not re-run the whole PR matrix.
  2. **The weekly flag-graduation gate had been exiting 2 before gating
     anything since 2026-07-13.** The workflow passed a hardcoded
     `--flags root_fixpoint,node_reduce,...`; `node_reduce` was deprecated out of
     `GRADUATION_ARMS` in #581, so every scheduled run died on
     `unknown flag(s): ['node_reduce']` — four consecutive Mondays where the
     cert-neutrality guard proved nothing. The `--flags` argument is now omitted
     entirely so the gated set tracks the registry (7 arms, not the copy's 5),
     and the workflow gained a `pull_request` trigger on its own path so an edit
     to it is validated by the PR that makes it, not a week later.
  3. **The AMP integration suite had failed on every push to `main` since
     2026-07-27** — 40+ consecutive red runs, no tracking issue — on
     `test_amp_time_limit_with_incumbent_returns_feasible`. The solver contract
     is intact; the test was the broken part. It drove the timeout by *read
     ordinal* (`chain([0.0, 0.0], repeat(1.5))`, "the first two reads are before
     the deadline"), and `monkeypatch.setattr(amp_mod.time, ...)` patches the
     **global** `time` module, so a burst of ~26 reads inside
     `nonlinear_bound_tightening`'s budget polling consumed the pre-deadline
     values. The deadline then blew before any incumbent was stored and
     `solve_amp` returned through the "no feasible solution found" arm — so the
     assertion described a path the test no longer executed, and reported it as
     `assert 'time_limit' == 'feasible'`, which reads as a solver bug. The
     timeout is now driven by *state* (pre-deadline until an incumbent exists),
     which no read count can perturb, plus a callback counter so the test cannot
     pass while measuring nothing (CLAUDE.md §6).

  All three scheduled/main lanes now file-or-refresh a single `ci-signal`-labelled
  issue on failure. Each of these was red or absent for weeks; none of them
  reached anyone.

- **A clock seam for the primal-heuristic deadlines, ending a CI flake**
  (`test`, #950). `TestLocalBranchingBudget::test_deadline_expiring_mid_round_truncates`
  failed intermittently on the parallel lane as `assert 0 >= 2`. The mechanism is
  scheduling, not solver behaviour: `local_branching` stamps
  `slice_deadline = perf_counter() + submip_time_limit` *before* its first budget
  poll, and since #912 no sub-NLP may start past that deadline — so any stall
  longer than the test's 250 ms slice (an xdist worker descheduled on a loaded
  runner) correctly retires the budget before sub-NLP #1, and the test's
  wall-clock-dependent assertion could not tell that apart from broken truncation
  logic. Reproduced deterministically by injecting a 0.30 s stall and changing
  nothing else: `calls` 2 → 0.

  `primal_heuristics._now()` is now the module's single clock read — every
  heuristic's deadline stamp and poll goes through it, and every `WorkBudget`
  built there is handed it via the new `WorkBudget(clock=...)` parameter (default
  `time.perf_counter`; production passes nothing else). The test drives the
  deadline from a clock the backend advances, so it pins the truncation
  *schedule* — radius 0 whole, radius 1 cut after its first sub-NLP — as an exact
  count rather than racing the machine. Verified by re-running the old and new
  test bodies under a 0.5 s injected stall: old `calls=0` (the reported failure),
  new `calls=2` with and without the stall; with the injected clock frozen the
  enumeration runs all `C(3,1)=3` flips (`calls=4`), which is what proves the
  seam — and not something else — is what fires the gate. `submip_time_limit`
  was **not** widened: that would have made the race rarer without removing it,
  and the budget edge is the test's subject. Production behaviour is unchanged;
  the #912 wall-budget inventory scanner was taught to see through the seam so a
  new gate cannot leave the ratchet by routing through `_now()`.

- **A non-zero `Constraint.rhs` was a silent wrong answer through the public API**
  (`fix(correctness)`, #909). `Constraint` is exported in
  `discopt.modeling.__all__`, and `m.subject_to(Constraint(w, ">=", 5.0))` solved
  to `w = 0` while the equivalent `m.subject_to(w >= 5.0)` solved to `w = 5`. The
  cause is a split in the tree over what an unnormalized row means: **26 modules
  read `Constraint.body` and never read `.rhs`** — the entire relaxation/branching
  stack, whose seam is `_relax/dag_compiler.compile_constraint` — while
  `validation/feasibility` computes `signed = body - rhs`. The row was therefore
  *solved* as `body sense 0` but *verified* as `body - rhs sense 0`, so the solver
  returned a point its own verifier called feasible and the user called wrong.

  The fix **refuses loudly** rather than teaching the solve path to honour `rhs`
  (CLAUDE.md §3). Threading it through would mean correcting all 26 modules, each
  of which encodes the `body sense 0` form structurally rather than
  arithmetically, and a partial job is strictly *worse* than the status quo: today
  the relaxation stack is uniformly rhs-blind, so its McCormick envelope relaxes
  the same row the verifier checks; half-honoured, the envelope would be built for
  a *different* row than the one verified — a soundness hazard replacing a
  wrong-answer hazard. The refusal fires at `subject_to` (both the scalar and the
  list arm, so the error lands on the offending line) and at `Model.validate` (so
  a direct `_constraints.append` cannot slip past, since `solve` always
  validates), and the message names both rewrites. Array `rhs` is tested with
  `np.any`, not `float()`, which would raise a confusing `TypeError` on a vector
  row.

  Entry experiment (CLAUDE.md §4) established that refusing breaks nothing:
  **66 models / 3,705 executed rhs comparisons / 0 violations**, with a planted
  control arm proving the probe can see a non-zero `rhs`; a dynamic arm over the
  suite found 58 non-zero constructions, **all from test fixtures — zero
  production producers**.

  `Model.validate` gains `for_solve=True`. Writers that represent a row faithfully
  rather than solving it pass `False`: **`.nl`** (folds the constant into the
  r-section bound) and **GAMS** (emits `rhs` verbatim) export such a row correctly
  and must not be refused. **`.lp` and `.mps` keep the refusal** — and this
  corrects issue #909's own text, which listed them as honouring `rhs`: measured,
  both rebuild the right-hand side from the *body* alone, so `Constraint(w, ">=",
  5.0)` emitted `c0: w >= 0` (LP) and an empty `RHS` section (MPS). For those two
  the refusal converts a silently wrong export into a loud error.

- **The convex kernel's `clay0303hfsg` "false optimum" was a false CERTIFICATE, not
  an invalid relaxation** (`fix(correctness)`, #879). #879 recorded three
  mutually-inconsistent `optimal` results on `clay0303hfsg` (28351.42 / 36397.83 /
  55092.52), each strictly worse than a point the default path attains, and read
  them as a dual bound sitting *above* the true optimum — an invalid, too-tight
  `a²/s` relaxation. Re-measured with the #871 certificate fix in place, **that
  hypothesis is falsified and is retracted here**:

  * every one of those numbers is an **incumbent**, published as certified because
    the tree had silently discarded a `numerical` node and then let `bound` fall
    back to the incumbent's own objective. `28351.41943619983` is reproduced exactly
    as the incumbent of an *uncertified* run;
  * the relaxation is not too tight anywhere. `clay0303hfsg`'s root safe bound is
    `0.0` at every separation setting (0/1/2/4/12) against an optimum of `26669.11`
    — valid, and in fact trivially weak, the opposite failure. Pinned by
    `test_clay0303hfsg_root_relaxation_is_sound_not_too_tight`.

  What actually blocked certification was the node LP breaking down (`numerical`) on
  12 of 453 nodes, correctly poisoning the certificate. Three causes, each fixed and
  each measured to be necessary — removing any one alone takes the instance back to
  `exhausted`: the six infinite structural upper bounds (dominated-column bound,
  above); an unbounded per-node OA tangent pool (`appended_row_cap`, `max(8·n, 500)`
  — one node reached 307 OA rounds / 2042 tangents on 99 columns before the
  factorization broke); and a warm re-optimize whose carried basis, extended once per
  appended row across the whole OA chain, is exactly what is in doubt after a
  breakdown (now re-solved once **cold**, mirroring the LP layer's own
  `dense_retry`). All three are refusals or robust re-solves — a capped node returns
  a valid, merely weaker bound, and a vertex reached without OA convergence still
  violates a nonlinear row so it cannot be mistaken for an incumbent.

  The methodological lesson from #879 stands and is now enforced: exactness and
  convexity of the marshaled rows both **passed** while the false certificate was
  live, so they are not sufficient to admit a term class. A routed instance's
  certified objective must be checked against a known optimum —
  `test_clay0303hfsg_certifies_against_its_known_optimum` does exactly that, against
  the shared `known_optima.toml` registry.

- **Root setup no longer scales with `n_vars × n_constraints`, and no longer runs
  unbudgeted** (`fix(performance)`, #875, successor to #863/#868). After #868,
  `watercontamination0202` (106,711 vars / 107,209 rows) returned instead of hanging
  but still took 579 s against a 30 s `time_limit` with `nodes=0`. Two distinct
  defects, both entirely before the first branch-and-bound node:
  - *Cost.* `_fix_single_var_equalities` was `O(n_constraints × n_vars)` — the affine
    linearizer it calls allocated and zeroed a dense `n_vars` array per constraint,
    and the caller then walked all `n_vars` entries in Python to find the single
    nonzero — for bodies with **one leaf each**. That was ~460 s of the 579 s (23 of
    29 stack samples). Adds `_linearize_affine_expr_sparse` as the core, with the
    dense `_linearize_affine_expr` as a view of it, and routes every per-constraint
    scan through the sparse form (`_fix_single_var_equalities`, the one-hot group
    scan, the monomial-factor fallback, the affine bound helpers). A new
    `_any_linear_constraint_form` short-circuits the RLT-applicability probe, which
    previously built and kept one dense row vector per constraint — 91 GB on that
    instance — to evaluate a boolean. Measured on a synthetic probe at a fixed
    constraint count: 0.85 s → 0.001 s at `n_vars=32,000`, and flat in `n_vars`
    (18.8× → 1.0× across a 16× variable increase) instead of exactly linear.
  - *Budget.* `tighten_nonlinear_bounds` (80.9 s over three root calls) had no
    `deadline` parameter at all; it now takes one and polls at three granularities —
    between rounds, between rules, and inside a rule's constraint scan — because a
    single rule's sweep over 107k rows already exceeds a tight budget (#868's
    `probing` lesson: a poll at the wrong granularity is not a poll). Rules declare
    `row_scan_is_anytime`; the default is `False`, so a rule whose conclusion rests
    on a row being *absent* (`PeriodicVariableBoundRule`) is skipped whole rather
    than truncated. The convexity classifier's budget is a fraction of `time_limit`
    recomputed per *model object*, so a reformulation restarted it and the fractions
    added up (14.7 s over two runs); it is now additionally clamped to the absolute
    `model._solve_deadline`. Every early exit does strictly less work — a looser box,
    fewer rules, `convexity-unknown` routing to sound spatial branch-and-bound —
    never a wrong one.
- **The QCP probe extractor's Hessian may be sparse** (`fix`, #875).
  `_extract_quadratic_coefficients_from_values` still swept all `O(n²)` variable
  pairs into a dense `(n, n)` array, per constraint — the one extractor #863/#868
  left dense. It now restricts off-diagonal probing to the evaluator's support (free:
  the `O(n)` diagonal probes already identify it) and materialises through
  `_materialise_Q`, matching the QP path. `_quadratic_row_has_terms` and the Gurobi
  QCP entry point are made sparse-aware in step, so a sparse `Q` cannot reach a
  consumer as the 0-d object array `np.asarray` silently produces.
- **Two false-certificate paths in the convex kernel tree** (`fix(correctness)`,
  #871). Both are latent on today's default path (the kernel is default-off and
  `try_convex_solve` seeds no incumbent), and both are removed.

  1. **A silently discarded subtree could still certify.** `solve_tree` skips any
     node whose LP does not return `Optimal`. For a *proven-infeasible* node that
     is a legitimate fathom; for `numerical` / `unbounded` / iteration-limit the
     subtree is simply never explored — yet the tree upgraded `Exhausted →
     Optimal` on the drained frontier, or exited `Optimal` on a closed frontier
     gap. The reported `bound` then fell back to the **incumbent's own objective**
     (reading as a closed gap) and the incumbent was clamped against an infinite
     dual to `±inf`. An uncertified drop now poisons the certificate: neither
     `Optimal` exit can fire, the bound reports `±inf` ("no bound"), and the
     incumbent is clamped only against a *finite* dual.

  2. **A seeded incumbent was certified as the optimum on a MINIMIZATION.** The
     root node's trivial dual bound was written `sense * INFINITY`, which is `−∞`
     for a min — the *best* value in the loop's "maximize `sense·bound`"
     convention — so the root looked already fathomed against any seeded incumbent
     and the gap check fired on node 0. Measured: seeding `9.0` on a spec whose
     true optimum is `2.5` returned `Optimal, bound=9, nodes=0`. Maximization was
     correct by accident. The trivial bound is `+∞` in the convention for both
     senses.

  The guards are exact: `syn05hfsg` (2 nodes), `syn05m` (3 nodes) and
  `cvxnonsep_psig40r` (1 node) are bit-identical, statuses and objectives
  unchanged.


## [0.7.0] - 2026-07-24

### Added

- **SGO integer signomial MINLPs + exact single-negative-monomial transform**
  (`feat`, #741 Task 2, inside the default-off `DISCOPT_SGO` path). The signomial
  global optimizer now admits the `cvxnonsep_nsig*` family — positive-bounded
  **integer** variables — via integer spatial branching that wraps the same
  continuous node relaxation (which relaxes integers to the continuous log-box,
  so every node bound stays a valid lower bound on the integer optimum). Adds:
  certified integer bound rounding (an empty enclosed-integer range prunes a node
  as integer-infeasible), most-fractional integer branch selection with an
  integer-domain-split fallback, and integer-feasible incumbent recovery (fix
  each integer to an enclosed integer, solve the continuous remainder, and
  verify every true constraint — so a reported incumbent is genuinely integer-
  feasible). A fully pruned integer tree yields a rigorous `status="infeasible"`
  certificate. Enabling this required the **exact convex reformulation of
  single-negative-monomial constraint rows** (issue #741 Task 1 lever 2 / the
  Lundell–Westerlund single-sign power transform): a row `Pplus(u) − c·exp(a·u) ≤
  0` with one negative monomial is exactly convex-representable as the posynomial
  `Σ exp(log_cₖ − log c + (bₖ − a)·u) ≤ 1` (identical feasible set — the divisor
  is positive), so that row's node relaxation becomes *exact* instead of the
  loose DC secant. Measured: `cvxnonsep_nsig30` (15 continuous + 15 integer vars)
  is admitted and its integer-feasible incumbent reaches the known optimum
  130.6287 with a sound dual bound (full certification of the 30-var box is not
  in-budget — the same wide-box certification frontier as ex7_2_3); small
  integer MINLPs and box-only integer programs certify to their brute-force
  optima in a handful of nodes. The exact transform also makes the Task-1 4-var
  probe certify at the root (0 branch nodes). Classifier still abstains on binary
  / zero-lower-bound variables (the log lift needs `x > 0`). The pre-#741
  reference path (`obbt=False`) keeps the all-DC-secant relaxation; `DISCOPT_SGO`
  stays default-OFF (graduation is Task 3, pending the corpus panel). Falsified
  in passing (recorded in `docs/dev/performance-plan.md` §6): using the corner
  certificate to tighten single-monomial secant *argument* ranges is a dead end.
  Reproduction: `discopt_benchmarks/scripts/sgo_741_tightening_probe.py nsig`.

- **SGO constrained node tightening** (`feat`, #741, inside the default-off
  `DISCOPT_SGO` path). The signomial global optimizer's constrained node
  relaxation — previously sound but too loose to certify anything beyond 2-var
  probes (#736's measured blocker) — gains a stack of certified devices:
  iterated log-domain OBBT with the incumbent objective cut (every coordinate
  bound proven by the Lagrangian corner mechanism, never by trusting the convex
  subsolver), rigorous interval floors on the objective and the fitted
  Lagrangian, monotone parent-bound inheritance, certified
  infeasibility/objective-cut pruning (a fully pruned tree now returns a
  rigorous `status="infeasible"` certificate), DC-secant-gap-guided branching,
  phase-1 feasibility restoration for incumbent recovery, and per-node
  frozen-pack evaluation (~13× node rate). Measured: a 4-var wide-box class
  instance certifies (562 nodes at 1e-2) where the old relaxation's bound sits
  at −327 vs opt 10.90 at the same node budget; ex3_1_2's tree bound moves
  −1.1e10 → −30701 (opt −30665.5) and ex7_2_3's −3e5 → +3233 with the optimal
  incumbent found (was: none). The pre-#741 node relaxation is preserved behind
  `solve_signomial_global(..., obbt=False)` as the differential-test reference;
  box-only solves are untouched. `DISCOPT_SGO` itself stays default-OFF
  (graduation is #741 Task 3, pending the corpus panel). Falsified in passing
  (recorded in `docs/dev/performance-plan.md` §6): certified ξ-argument-range
  tightening via the same corner machinery returns ranges wider than
  box-implied — not the lever. Reproduction:
  `discopt_benchmarks/scripts/sgo_741_tightening_probe.py`.

- **Flow-aware integer-multilinear envelope** (`feat`, #707, default-off behind
  `DISCOPT_INTEGER_MULTILINEAR_REFORM`). Generalizes the integer-*bilinear* exact
  reformulation to products of ≥3 variable factors where every factor but at most
  one is integer- or binary-valued (declared or *implied*) — e.g. ex1252's
  objective `(6329.03 + 1800·x15)·x0·x3·x18` with integer flow factors
  `x0,x3 ∈ {0..3}` and a 0/1 indicator `x18`. Each integer factor is
  binary-expanded (`x = lo + Σ 2ᵏ eₖ`), the product distributed into binary
  monomials (`e² = e`), and each monomial lifted to its **exact** hull: an n-ary
  AND (`z ≤ eᵢ`, `z ≥ Σ eᵢ − (n−1)`, `z` binary) for the pure-integer part plus one
  big-M product for the lone continuous factor. The rewrite is a value-preserving
  algebraic identity (verified to ~1e-9 over sampled points), so it can only
  tighten — never invalidate — the dual bound; it replaces the loose term-wise
  trilinear McCormick envelope over the continuous box with the per-integer-level
  exact envelope. Unlike the bilinear pass (adopted only when it yields a pure
  MILP), this pass is **kept on the spatial B&B path** when residual *continuous*
  nonlinearity remains, since the integer-term tightening is a strict gain there.
  On ex1252 this lifts the global dual bound off its structural **5134** floor
  (the plateau the term-wise envelope and SOS1-selector branching both stall at)
  to ~32k in 45 s and climbing; the structure is general (present in 8 of the 81
  in-repo MINLPLib instances: ex1252/ex1252a, nvs01/05/16/22, st_e36/e40). A
  spatial-path **blowup guard** keeps a non-pure-MILP reform only when the column
  count stays within a modest factor of the original, so an already-tractable
  instance the spatial solve certifies fast (nvs01: 3 → ~200 columns) is left on
  its original path rather than regressed. Differential panel over the eight
  in-repo integer-multilinear instances (flag off vs on): **cert-clean** (every
  bound ≤ its reference optimum, every incumbent feasible, both regimes) and
  **net non-negative** (large gain on ex1252/ex1252a, neutral elsewhere). The flag
  is default-off pending a corpus-wide §5 graduation; the default (flag-off) path
  is byte-identical (the bilinear reform and its blowup guard are untouched — the
  new aux-sharing cache is scoped to the multilinear path). Full end-to-end
  certification of ex1252 additionally requires closing a *separate* residual
  barrier — the continuous cubic cost rows `x15 = f(x6, x12)` — which spatial
  branching, not this envelope, must tighten; recorded in the issue.

- **GP-structured MINLP: y-space node relaxations + integer branch-and-bound**
  (`feat`, #116). A MINLP whose *continuous relaxation* is a geometric program
  (posynomial objective, `posynomial ≤ monomial` / `monomial == monomial`
  constraints) but which also carries integer variables is now solved by integer
  branch-and-bound in which every node relaxation is the **exact convex
  log-space NLP**: per node the box `xᵢ ∈ [lᵢ, uᵢ]` maps to
  `yᵢ ∈ [log lᵢ, log uᵢ]` (`lᵢ > 0` required), branching is on the discrete
  variables in `x`-space, and incumbents recover via `x = exp(y)`. Because each
  node bound is a rigorous convex-GP bound and integer branching is exhaustive, a
  closed tree returns a **certified** global optimum (`gap_certified=True`,
  `gap == 0`); a certifiably-infeasible model is reported as such. New surface:
  `discopt.gp.classify_gp_minlp` / `is_gp_minlp` / `solve_gp_minlp`, and the
  `solver="gp-minlp"` selector. The continuation of #113 (which shipped the pure
  continuous GP auto-route); it does **not** auto-fire from a plain `solve()`
  unless `DISCOPT_GP_MINLP` is set (default off, pending a corpus-wide
  differential-panel graduation), so default behaviour is unchanged. Verified
  against the independent classic spatial B&B over a differential panel.
- **Continuous stratified multistart at the root** (`feat`, #188). Pure-continuous
  nonconvex models had zero basin diversification on the spatial McCormick-LP
  path: the integer-centric primal heuristics (pump/ILS/diving/RINS/RENS) all
  no-op without integers, the root multistart NLP is skipped there, and the
  strided node NLP warm-starts from the parent point — so the incumbent stayed
  locked in the first LP-vertex basin (kall_congruentcircles_c51: parked at the
  1.5371 two-row packing vs the 1.0730 global). `solve_model` now runs a
  budgeted, deadline-gated stratified multistart
  (`primal_heuristics.continuous_multistart`) once at the root for nonconvex
  models with no integer variables; the c51-class reconstruction reaches the
  1.07301 global on the default path (siblings and the C-38 `kall_circles_c8a`
  soundness lock unregressed). Primal-only (heuristic-policy regime): every
  point is constraint-re-verified and `inject_incumbent` enforces strict
  improvement, so dual bounds and certificates are untouched.
  `DISCOPT_CONTINUOUS_MULTISTART=0` / `SolverTuning.continuous_multistart=False`
  restores the prior behavior.

- **Reduced-space global optimization of hidden-function (`CustomCall`) models**
  (`feat`, #713). A model term written as opaque JAX code and wrapped with
  `dm.custom(...)` is now solved to a **certified global optimum** by relaxing the
  opaque body through the reduced-space McCormick type (`MCBox`) and branching
  **only on the true degrees of freedom** — the hidden internal intermediates never
  become optimization or branching variables (the signature capability of MAiNGO,
  {cite:t}`Bongartz2018`). Continuous *and* integer-DOF models are supported (P3.1,
  P3.2); the `MCBox` intrinsic namespace covers arithmetic plus `exp`/`log`/`sqrt`
  and **even, odd, and fractional/signomial powers** `x**a`, each with a tight
  monomial hull (P1.3, P1.4). The contract is **sound-or-refuse**: a body that does
  not trace soundly through `MCBox` (raw `jnp` intrinsic, non-affine hidden
  division, unbounded box) falls back to the local NLP path — a valid solution but
  no global certificate — never a partial or invalid global bound. Non-affine
  reciprocal lifting was investigated and **falsified** (the refusal stands). New
  worked example: `docs/notebooks/reduced_space_customcall.md` (a reactor cascade of
  nested `CustomCall` units), complementing the `m.implicit(...)` recycle-loop
  notebook.

- **Root disjunctive configuration bound** (`feat`, #732, default-off behind
  `DISCOPT_DISJUNCTIVE_CONFIG_BOUND`). For reforms carrying configuration structure
  (range-{0,1} indicator and span≥2 count factors of exact-linearized products,
  e.g. ex1252's pump counts), a root pass **partitions** on the configuration
  variables instead of relaxing across them: it enumerates the 2^k indicator
  patterns (a valid partition — every feasible point has integral indicators),
  bounds each configuration box by per-box interval FBBT → budgeted OBBT → node LP,
  and takes the min across boxes as a rigorous root dual bound. On ex1252 this lifts
  the root dual from 0 → 42725. Default-off pending a corpus-wide graduation panel.

- **Convex LP-OA branch-and-cut kernel** (`feat`, #798/#799, default-off behind
  `DISCOPT_CONVEX_KERNEL`). An in-house LP-relaxation-per-node branch-and-cut path
  for the convex MINLP family (Quesada–Grossmann OA + GMI/cover/MIR separation,
  best-bound tree, pseudocost branching, Neumaier–Shcherbina safe bounds, #779
  incumbent verification). Certifies the convex `rsyn*`/`syn*` family cert-clean
  and decisively faster than the NLP-BB path (panel certifies in ~24 s vs NLP-BB
  timing out uncertified at 482 s). Larger-instance SCIP-parity and default-ON
  graduation remain open (#800).

- **Native spatial branch-and-bound kernel** (`feat`, #764, default-off). An
  in-house spatial B&B with FBBT propagation, finite-slack certification, and
  driver-side incumbent seeding; certifies `tanksize` natively with a verified
  incumbent. (A false-certificate bug found and fixed as part of the FBBT pass.)

- **Native-warm-LP node solve** (`feat`, #807, default-off behind
  `DISCOPT_CVX_NATIVELP`). A shared persistent-LP warm-restart path (bounds-in-place
  dual reoptimize + node-local cuts). Built and **sound** (cert-clean, flag-OFF
  bit-identical), but a durable measurement shows it does not net out on the full
  cut-driven certifying panel; it stays default-off with the finding recorded.

- **Big-M coefficient tightening, re-implemented** (`feat`, #774/#780, default-off
  behind `DISCOPT_COEF_TIGHTEN`).

- **NLP-BB root cutting-plane stage** (`feat`, #781/#784, default-off behind
  `DISCOPT_NLPBB_ROOT_CUTS`): GMI cuts + efficacy×orthogonality selection at the
  root, with a root-cut quality gate and model-row safety margin (#781/#785).

### Changed

- **`pounce-solver` minimum raised to `>=0.9`** (`chore(deps)`). Bumps the core
  dependency (and the back-compat `[pounce]` alias extra) from `>=0.8`; discopt's
  usage is unaffected.

- **CI: fast-correctness PR lane** (`ci`, #813). Every PR now runs a dedicated
  `python-correctness` lane so the `incorrect_count == 0` gate is enforced before
  merge, not only nightly.

- **OBBT-on-auxiliaries reverse-FBBT cascade graduated default-ON** (`perf`, #208).
  The root branch-and-reduce fixpoint now propagates OBBT-tightened auxiliary
  (product/ratio) column bounds back onto the original variables through the
  nonlinear term definitions — the hyperbolic/root bounds the linear McCormick rows
  cannot express (`w=a·b ⟹ a∈[w]/[b]`, `w=aᵖ ⟹` p-th-root box, plus the
  trilinear/multilinear and ratio-of-products generalizations). The extra aux
  min/max LPs are budgeted to the reverse-FBBT-*reachable* columns
  (`obbt.cascade_reachable_aux`), which is bound-neutral vs a blanket cascade but
  drops ~87% of the aux probes, and the cascade runs root-only (no per-node cost).
  Graduation gate (`design/ab_cascade_aux.py`, 65-instance corpus, fair 30 s
  budget): cert-clean — 0 differential soundness violations, 0 optimum mismatches,
  0 cert regressions, +1 cert gain (`tls2` F→T) — and net-positive with 0
  regression: node-neutral on the convergent integer-heavy majority (converged-only
  2228 vs 2228) and helpful on the continuous spatial-branch class (`tspn08/10/12`
  prune to 1 node, `heatexch_gen3` 208→31 s wall; all-instances node_count −2.5%).
  An earlier too-tight 8 s A/B had read as net-negative; that was time-limited noise
  (`fac2`/`cvxnonsep_nsig30` converge bit-identically at a fair budget).
  `DISCOPT_OBBT_CASCADE_AUX=0` restores the prior default-OFF behavior.

### Removed

- **Untrained GNN branching scaffold** (`chore`, #236, closed not-planned).
  `branching_policy="gnn"` had always been inert — `solver.py` called
  `select_branch_variable_gnn(graph, params=None)`, which falls back to
  most-fractional, and no trained weights were ever shipped — so the option
  silently behaved identically to the default. The Stage-4 entry experiment
  (`docs/dev/performance-plan.md` §6, 2026-06-24) showed branching — including
  real strong branching — is not the lever for the slow tail (bound/
  range-reduction limited), so a learned imitation of strong branching cannot
  pay off there and would fail the net-positive graduation gate. Removed
  `_relax/gnn_policy.py`, `_relax/gnn_branching.py`, `_relax/problem_graph.py`
  (`INTEGRALITY_TOL` moved into `_relax/strong_branching.py`), the
  `branching_policy` parameter from `Model.solve()`/`solve_model()`/the CLI
  (it had exactly one functional value), and the `gnn_branching` notebook.
  The `gnn`/`learned` extras remain — equinox/optax still power the ICNN
  learned-relaxations path. Default branching is unchanged: reliability
  (pseudocost + priority + strong branching) in the Rust `TreeManager`.

### Fixed

- **Convex single-NLP certificate soundness hardening** (`fix(correctness)`,
  #849 / #850 / #853). Three related false-certificate classes on the convex /
  pure-LP fast paths, each of which certified a dual bound that crossed the true
  optimum, are closed. **#849**: the convexity-certified single-NLP path trusted the
  backend's *scaled* `optimal`; under a large objective/constraint coefficient the
  returned point satisfied the scaled stopping test yet grossly violated the
  *unscaled* KKT conditions (`min -x s.t. x²≤1e18` certified ~20× short of the
  optimum). It now recomputes the unscaled KKT residuals and withholds the
  certificate when they are not met. **#850**: the pure-LP fast path returned
  contradictory certificates depending only on the LP engine — POUNCE deferred to
  `unbounded` on the default box where the exact simplex returns `optimal` at the
  corner, and accepted a super-optimal constraint-infeasible incumbent on a general
  inequality row; a per-row term-scaled feasibility tolerance and a relaxed-bound
  guard restore cross-backend consistency. **#853**: a convex objective whose
  gradient asymptotically vanishes (`min -log(x)`) was certified `optimal` at an
  interior stall whose stationarity residual is small only because the Hessian
  flattens, while the true optimum sits at a distant box bound (including the
  default ~9.999e19 box). The fast path now runs a Frank–Wolfe better-point
  refutation over the box and withholds the certificate when it exhibits a
  strictly-better feasible point. All three only ever *downgrade* a certificate, so
  they cannot introduce a false optimum or a wrong bound.

- **nvs16 garbage dual-bound guard restored** (`fix(correctness)`, #812 / #248).
  The uniform relaxer again rejects a non-finite / garbage bound, fixing an unsound
  dual bound on nvs16.

- **`solve()` no longer crashes with `RecursionError` on deeply-nested
  expressions** (`fix`, #810/#811). `_expression_degree` (the incumbent-verification
  fast-family guard, on the default solve path) recursed on the expression tree
  with no depth guard, so a long left-associated sum overflowed the Python stack.
  Rewritten as an iterative post-order over the expression DAG with memoization by
  node identity. Surfaced by the MINLPLib benchmark (`autocorr_bern*`); those
  instances now solve. Not a correctness bug — a hard crash on a valid model.

- **Native spatial kernel robustness** (`fix`, #764/#788/#789/#790): honor the
  outer wall-clock budget (#788/#795); feature-safe routing + final-incumbent
  verification (#789/#794); a full warm-startable simplex basis on slackless
  equality LPs, P1.0 (#790/#793, default-off gated).

## [0.6.0] - 2026-07-12

### Removed

- **AC-OPF and pooling application builders extracted to the `discopt-apps`
  plugin** (`refactor`, #431). `python/discopt/opf.py` (rectangular AC-OPF:
  `build_ac_opf_rectangular`, `ACOPF`, `Bus`, `Line`, `admittance_matrix`,
  `two_bus_example`) and `python/discopt/pooling.py` (pq-formulation:
  `build_pq_formulation`, `PoolingProblem`, `Input`, `Pool`, `Output`,
  `haverly_hpp1`), their tests, and the `ac_opf`/`pooling_pq` doc notebooks now
  live in the standalone
  [discopt-apps](https://github.com/jkitchin/discopt-apps) package, mirroring
  the course extraction (#430). Because discopt is a namespace package,
  `pip install discopt-apps` restores `discopt.opf` and `discopt.pooling`
  imports unchanged. These are pure builders over `discopt.modeling.core`; no
  core solver behavior changes. (The in-core
  `discopt.modeling.examples.example_pooling_haverly` example stays.)
- **DOE module extracted to the `discopt-doe` plugin** (`refactor!`, #389).
  `python/discopt/doe/` (design, screening, FIM, identifiability,
  discrimination, workbook CLI, Streamlit GUI), its 19 test files, 11 doc
  notebooks, notebook build scripts, and the doe skill/agents now live in the
  standalone [discopt-doe](https://github.com/jkitchin/discopt-doe) package.
  Because discopt is a namespace package, `pip install discopt-doe` restores
  `discopt.doe` imports and the `discopt doe` subcommand unchanged. The
  `doe`/`doe-gui` extras are gone — use `discopt-doe[gui,ml]`. discopt ≤0.5.x
  is the last line with DOE built in.

### Added

- **Public parametric-compilation API** (`feat(api)`, #389). New
  `discopt.parametric` module — the stable contract for external plugins
  (discopt-doe, discopt-mkm, ...) that compile model expressions into
  JAX-differentiable functions of `(x_flat, p_flat)`: `compile_expression`,
  `compile_response_function`, `extract_x_flat`, `flatten_params`,
  `param_total_size`, `variable_total_size`, `variable_slices`. The in-tree
  DOE module and `discopt.estimate` now consume this API instead of reaching
  into `discopt._relax` internals.
- **Generic CLI plugin subcommands** (`feat(cli)`, #389). External packages
  can register subcommands via the `"discopt.cli"` entry-point group (name =
  subcommand; value = module exposing `add_subparser(subparsers)` and
  `run(args)`). Plugin modules load lazily — only for their own subcommand or
  full help — and built-in names cannot be shadowed. The in-tree `discopt doe`
  subcommand now registers through this mechanism, ahead of its extraction to
  the standalone `discopt-doe` package (#389).
- **Reduced-space McCormick relaxation (MAiNGO-parity, opt-in)** (`feat(mcbox)`,
  #574, #575, #576, #577, #579, #580, #583). A propagating McCormick type with
  rule-based subgradients that relaxes in the original variable space (no
  auxiliary lifting), plus a jitted Kelley-cut LP on the in-house simplex,
  S-shaped intrinsics with per-regime subgradients, and sign-definite reciprocal
  division. Selected with `DISCOPT_RELAX_SPACE=reduced` (**default-OFF**); sound
  where it engages, pursued for generality/robustness parity with MAiNGO rather
  than as a speed lever. See `docs/dev/maingo-parity-plan.md`.
- **Hybrid physics + ML: trainable surrogates and simultaneous neural-DAE
  training** (`feat(nn)`, #595). Surrogate weights become decision variables so a
  neural rate law can be trained jointly with a physics model (e.g. inside a
  collocation DAE); see `discopt.nn.trainable` / `discopt.nn.surrogate` and the
  multi-experiment fitting glue in `discopt.dae.fit`.
- **Entropy-family relaxation: centropy tangent-plane underestimator**
  (`feat(relax)`, #597). Linearizes the `x·log(x)` / entropy atom so entropy-style
  objectives become certifiable rather than local-NLP-only.
- **Exact 1-D univariate-composite and positive-product envelopes (opt-in)**
  (`cert:LR-2`, #627). `DISCOPT_UNIVARIATE_ENVELOPE` (H-UNI) and
  `DISCOPT_LOG_MONOMIAL` (H-LOG) add exact convex/concave hulls for
  single-variable composites and log-space positive products (root-certifies
  nvs09). Both **default-OFF**; graduation to default-ON is tracked in #632
  (canonical factorable normal form). OFF is byte-identical to prior main.
- **Binary (`b`) `.nl` parsing** (`cert:TAIL-1b`, #593). Binary-format AMPL `.nl`
  files are transcoded to text and read (e.g. `st_miqp5`).

### Changed

- **`pounce-solver` minimum raised to `>=0.8`** (`chore(deps)`, #633). pounce 0.8
  makes `Problem` a compiled object; discopt's usage is unaffected (method
  dispatch behind `hasattr`).
- **`feral` LP engine bumped to crates.io `0.14.0`** (`chore(deps)`, #628),
  retiring the git-rev pin from #112.
- **Default-ON graduation of three per-node levers** (`graduate`, #616): the
  density-aware LU route (`lu_density_route`), objective-branch-priority
  (`obj_branch_priority`), and loose-product lifting (`lift_loose_products`) are
  now on by default, validated soundness-neutral (`incorrect_count = 0`).
- **Density-aware dense/sparse LU route for wide-McCormick node bases**
  (`perf(lp)`, #573, #591) with a failure-triggered dense retry that cures the
  nvs21 certificate loss; plus zero-pivot / refactorization-breakdown fixes in
  the LP engine (#570, #571).
- **Documentation build is now zero-warning** (`docs`, #605; 132 → 0).

### Fixed

- **AMP false-feasible from per-iteration MILP-budget starvation** (`fix(amp)`,
  #621). The per-iteration MILP time budget was sized off the `max_iter` cap, so
  a large `max_iter` starved each MILP on cold runs; it is now sized by an
  expected iteration horizon.
- **Gap-closed solves are now certified soundly** (`fix(cert)`, #604, #603,
  #613). A node-LP failure no longer poisons certification when the bound
  accounting stays rigorous (per-node sound accounting, #604); tainted trees now
  report the strongest rigorous dual bound instead of the taint-dropped one
  (#603); spatial gap-closed solves certify over soundly-floored sentinel
  removals (#613).
- **`nvs22` false-optimal on the reduced-space and cut-inheritance paths**
  (`fix`, #582, #568). The reduced-space evaluator now refuses non-finite boxes;
  column-identity-safe cut inheritance fixes the cut-inherit false-optimal.
- **H-UNI no longer builds a hull over effectively-unbounded boxes** (`cert:LR-3`,
  #631) — was a false-infeasible on the opt-in envelope path; guarded by a
  solver-sense finiteness check.
- **AMP integration suite de-flaked** (`ci(amp)`, #607, #608) via pytest-xdist
  process distribution and isolation of the slow trig/abs certification tests.

## [0.5.0] - 2026-07-01

### Added

- **Implicit-function expression node** (`feat(modeling)`, #379). `m.implicit(
  residual, u_inputs, n_unknowns, x0=)` defines a vector `v` by a square system
  `g(u, v) = 0`, compiled to a differentiable JAX inner solve (Newton forward;
  implicit-function-theorem derivatives via `jax.lax.custom_root`, which supports
  the higher-order AD the NLP Hessian needs). Rides on `CustomCall`, so it is
  **local-NLP-only** (no global certificate) and rejects integers. The core-side
  primitive for implicit variable-aggregation of irreducible cyclic blocks;
  documented in `docs/notebooks/implicit_function_node.ipynb`.
- **Hardened pure-Rust LP engine** (`feat(lp)`, #368). Numeric-focus LU with
  in-engine iterative refinement and condition/growth signals (via `feral`
  0.12.0), primal + dual refined recovery on a drifted-Optimal, dual-simplex
  anti-cycling (Bland + stall counter), and **EXPAND anti-degeneracy** (Gill et
  al.) in the Harris ratio test — ~15× fewer degenerate pivots on the
  lifted-relaxation corpus, validated soundness-neutral against the gauntlet and
  a BARON head-to-head.
- **Namespace-package support** (`feat(packaging)`). `discopt` now extends its
  `__path__` via `pkgutil.extend_path`, so external distributions (e.g. a
  `discopt-aggregation` plugin) can contribute submodules under the `discopt.*`
  namespace from a separate location on `sys.path` without modifying the core.
- **Set & index abstractions** (`feat(modeling)`). A Pyomo/JuMP-style named-set
  layer for sparse models, implemented as a pure-Python desugaring over the
  existing flat model (no solver/backend changes). Completes the Phase 7
  roadmap item. New public API:
  - `discopt.Set` / `discopt.RangeSet` (+ `ProductSet`) and `Model.set(...)`:
    arbitrary hashable members with inferred/declared `dimen`, set algebra
    (`|`, `&`, `-`, `*`), and filtering (`Set.where`, `with_first`,
    `with_last`). Product sets are lazy and accept flat or nested keys.
  - `Model.continuous/binary/integer/parameter(..., over=SET)` returning
    `IndexedVar` / `IndexedParam` backed by a single flat variable/parameter;
    per-key bounds/values via scalar, `dict`, or callable.
  - `Model.constraint(SET, rule, name=)` generating one constraint per member
    (named `name[key]`), with a `Skip` sentinel; `subject_to` now accepts
    generators of constraints. `dm.sum`/`dm.prod` aggregate over sets.
  - A transparent **linear fast path**: single-variable-affine, uniform-sense
    families are emitted as one sparse-matrix builder call (`fast=True`
    default) with identical results, falling back automatically otherwise.
  Documented in `docs/notebooks/sets_and_indexing.ipynb`; design in
  `docs/design/sets-and-indexing.md`; examples
  `example_transportation` / `example_assignment` /
  `example_multicommodity_flow`.
- **Decomposition benchmark instances** (`test(decomposition)`). Block-structured
  / two-stage MILP instances with known optima (`decomposition_problems.py`,
  registered in the `milp` suite) plus a consolidated correctness gate
  (`test_decomposition_benchmarks.py`) that checks Benders and Lagrangian
  against the known optima and the monolithic solver.
- **Generalized Benders Decomposition** (`feat(decomposition)`, Geoffrion 1972).
  `solve_benders` now handles a **convex nonlinear recourse** subproblem, not
  just a linear LP: when the model has a nonlinear objective or constraints it
  dispatches to `solve_gbd` (`discopt.solve_gbd`). Each optimality cut is the
  **Lagrangian dual value** as an affine function of the first-stage variables —
  `eta >= [L(x̂,ŷ) + m_y] + ∇_x L^T (x − x̂)` with sign-projected (dual-feasible)
  multipliers and the closed-form recourse-box correction `m_y` — which is a
  valid lower bound for *any* recourse point by the joint-subgradient inequality,
  so it stays sound even if the recourse NLP returns an inexact primal (the
  analogue of classical Benders' complete-dual cut). Recourse infeasibility at a
  0/1 first-stage point is excluded with a no-good cut. The reported lower bound
  is rigorous when the model is convex (gated on `classify_oa_cut_convexity`); on
  a nonconvex model GBD runs heuristically and reports `bound=None` so the
  `incorrect_count <= 0` gate is never threatened. POUNCE-only (no HiGHS).
- **Lagrangian B&B node-bound hook** (`feat(decomposition)`). `model.solve(
  lagrangian_bound=True)` fixes Lagrangian multipliers at the root and, at each
  MILP branch-and-bound node, combines a valid Lagrangian dual lower bound with
  the node's LP relaxation bound (`max()`), tightening pruning when the block
  subproblems lack the integrality property. Opt-in (default off), applies to
  linear minimization models with coupling structure, and no-ops cleanly
  otherwise; bounds are verified sound against brute-force enumeration.
- **Lagrangian relaxation solver** (`feat(decomposition)`). Dualizes coupling
  constraints (annotate with `model.mark_coupling(...)`) to produce a rigorous
  dual lower bound via `model.solve(decomposition="lagrangian")` /
  `discopt.solve_lagrangian`. The dual is maximized by a subgradient method
  (Polyak step) or a bundle / cutting-plane method, and a Lagrangian heuristic
  recovers a feasible primal incumbent. Documented in
  `docs/notebooks/tutorial_lagrangian.ipynb`.
- **Benders decomposition solver** (`feat(decomposition)`). Classical Benders
  for two-stage / block-angular (mixed-integer) linear programs via
  `model.solve(decomposition="benders")` / `discopt.solve_benders`. The master
  holds the complicating variables; the recourse-LP duals generate optimality
  cuts and a slack-penalized feasibility LP generates feasibility cuts. Cuts are
  **anchored at the primal recourse value** with a row-dual slope, so they stay
  sound even when the recourse optimum is set by variable bounds and with
  POUNCE's interior-point duals — **no HiGHS dependency** (runs on the POUNCE
  LP/MILP stack). Every cut is a global under-estimator, so the master objective
  is a rigorous lower bound (gap certified on convergence). Documented in
  `docs/notebooks/tutorial_benders.ipynb`.
- **Decomposition structure layer** (`feat(decomposition)`). Foundation for the
  upcoming Benders / Lagrangian solvers: a `Model` annotation API
  (`first_stage`/`second_stage`/`set_stage`/`set_block`/`mark_coupling`) and
  `discopt.detect_decomposition(model)`, which resolves annotations and
  auto-detects block structure (complicating variables default to integers;
  coupling constraints via a bridge heuristic reusing the separability scan).
  Exposed as `discopt.detect_decomposition` / `DecompositionStructure`.
- **Irreducible Infeasible Subsystem (IIS)** (`feat(infeasibility)`, #227). New
  `compute_iis(model)` returns a minimal infeasible subset of constraints/bounds
  via deletion filtering — exact for LP/MILP/convex, best-effort for nonconvex.
  Exposed as `discopt.compute_iis` / `IISResult`; documented in
  `docs/notebooks/infeasibility_iis.ipynb`.
- **Complementarity constraints via GDP disjunction** (`feat(modeling)`, #231).
  `Model.complementarity(x, y)` now reformulates through a GDP disjunction by
  default (`method="gdp"`), alongside the existing Scholtes regularization and
  SOS1/disjunctive paths, all unified behind one front-end and the
  `discopt.mpec` reformulation module. Documented in
  `docs/notebooks/complementarity_mpec.ipynb`.
- **RLT as a first-class solve option** (`feat(rlt)`, #223, #212). Level-1
  Reformulation-Linearization-Technique cuts are now a first-class solver choice
  (`rlt_cuts=True`) with per-node targeted RLT cut separation.
- **PSD / SOC cuts for QCQP + AC optimal power flow** (`feat(cuts)`, #203, #209).
  Dense-moment (PSD) eigenvalue cut separator and second-order-cone cuts for
  QCQP structure, plus a rectangular AC-OPF builder (`discopt.opf`:
  `build_ac_opf_rectangular`, `Bus`, `Line`, `ACOPF`) as the capstone target.
- **Differentiable MILP / MIQP** (`feat(diff)`, #221). Fix-and-differentiate
  framework propagates parameter sensitivities through integer programs via
  implicit (KKT) differentiation of the fixed continuous subproblem.
- **Conflict analysis / no-good cuts** (`feat(conflict)`). FBBT-driven conflict
  analysis derives no-good cuts from infeasible nodes.
- **Standard pooling problem + pq-formulation** (`feat(pooling)`). Bilinear
  quality-blending model builder with pq-cuts that tighten the McCormick
  relaxation; documented in `docs/notebooks/pooling_pq.ipynb`.
- **Geometric programming detection + log-space reformulation** (`feat(gp)`).
  `discopt.gp` recognizes posynomial structure (`classify_gp`) and solves via the
  convex log-space transformation (`as_geometric_program`, `solve_gp`).
- **Primal improvement heuristics** (`feat(heuristics)`). Diving, RINS, and
  local-branching heuristics added to the B&B primal-side search.
- **Public FBBT bound-tightening API** (`feat(tightening)`, #198). `discopt.tightening`
  exposes feasibility-based bound tightening for manual use; documented in
  `docs/notebooks/bound_tightening.ipynb`.
- **Integrality-aware FBBT bound snapping** (`feat(fbbt)`). Binary-indicator
  propagation snaps tightened bounds to integer values for sharper inference.
- **Periodic-variable bound reduction** (`feat(presolve)`, #215). Presolve pass
  reduces bounds of variables that only enter through periodic functions
  (`sin`/`cos`), unblocking otherwise-free angular variables.
- **`cuts='auto'` is the solver default** (`feat(cuts)`, #217). Auto cut
  selection balances bound tightening against node-count reduction.
- **Best-estimate node selection** (`feat(bnb)`) and **objective-gating priority
  branching** (`feat(bnb)`, #184) B&B search strategies.
- **Transcendental relaxation coverage** (`feat(relax)`, #216, #218). LP relaxer
  engaged for general transcendental nonlinearity; `asin`/`acos`/`acosh` gaps
  closed; non-smooth `abs`/`min`/`max` fixed; relaxation coverage audit added.
- **Run discopt as a GAMS solver** (`feat(gams)`, #119). GMO/GEV control-file
  link lets GAMS call discopt as an external solver; see `docs/gams_solver_link.md`.
- **Batched / multiple-RHS LP solving** (`feat(simplex)`). Shared-matrix batched
  ftran/btran for solving many RHS over one factorization, used by node-relaxation
  and DoE batches.
- **Per-node lifted-LP FBBT** (`feat(relaxation)`, #184). Opt-in
  (`DISCOPT_LIFTED_FBBT=1`) feasibility-based bound tightening that propagates
  the McCormick relaxation's *own* rows (`A_ub·z ≤ b_ub`, spanning the lifted
  product/monomial columns), recovering the bilinear-implied factor bounds that
  purely linear FBBT misses, then rebuilds the relaxation on the tightened box.
  This lifts `ex1252`'s structurally-zero node bound off 0 (a branched node goes
  `bound 0 → ~18987`, sound) so the B&B can certify optimality. Implemented in
  `discopt._relax.mccormick_lp` (vectorised over the sparse matrix); pinned
  multilinear factors are un-pinned by a hair so the build keeps the term at
  full arity and never drops `objective_bound_valid`. Sound by construction —
  only valid rows tighten, and the un-pin only enlarges the box. Regression
  locks in `test_bucket2_sound_bounds.py`.

- **Lifted-relaxation LP equilibration** (`feat(relaxation)`, #184). The lifted
  McCormick rows of a product over a wide variable box mix tiny constants (~1e-9)
  with large bound-derived coefficients (~1e7), giving a >1e15 coefficient spread
  on ex1252's boundary sub-boxes. HiGHS stalls on it (a 452×96 LP hits its time
  limit; the per-node soundness re-verifications then dominate the solve) while
  the pure-Rust simplex, which equilibrates internally, solves it in ~0.03s.
  `equilibrate_relaxation_lp` (`discopt._relax.milp_relaxation`) applies
  geometric-mean (Ruiz) row/column scaling, snapped to powers of two, before the
  external (HiGHS/POUNCE) backend solve when the spread exceeds 1e6 — turning
  previously timing-out boundary boxes into ~3-4s converged solves (e.g. the
  `x36=1,x37=1` box: time-out → 4.5s). The rescaling is exact (bound/feasibility
  unchanged, integer columns never scaled, solution mapped back through the
  column scale), so it only ever conditions — never alters — the result.
  Regression-locked in `test_bucket2_sound_bounds.py`.

- **Objective-gating priority branching** (`feat(bnb)`, #184). Opt-in
  (`DISCOPT_OBJ_BRANCH_PRIORITY=1`) branching-order heuristic that branches the
  integer variables gating the objective's nonlinear terms (those appearing in,
  or equality-linked to, a lifted product/monomial — e.g. ex1252's line-selection
  binaries `x36/x37/x38`) before other integers. The global dual bound is the
  minimum over the open frontier, so on problems whose bound is structurally 0
  until a *set* of binaries is jointly fixed it stays pinned at 0 under
  most-fractional branching (no single-variable score sees the joint jump);
  branching the gating binaries first reaches the depth where the per-node
  relaxation lifts each leaf off 0. Implemented via the existing `set_branch_hints`
  path in `solve_model` (`discopt.solver`) — pure search reordering over already
  fractional integer candidates, so it can never affect a bound or feasibility
  verdict. Detector locked by `test_bucket2_sound_bounds.py`.

- **POUNCE NLP backend declared as a dependency** (`feat(solvers)`). New
  `pounce` optional extra (`pip install discopt[pounce]`, dist `pounce-solver`)
  and a `requires_pounce` test marker. POUNCE is a standalone pure-Rust port of
  Ipopt (https://github.com/jkitchin/pounce). Added `solve_nlp_from_model` to
  `discopt.solvers.nlp_pounce` for parity with the cyipopt wrapper, plus
  `python/tests/test_nlp_pounce.py`.

### Changed

- **`feral` pinned to the crates.io 0.12.0 release** (`chore(deps)`, #375),
  carrying the LU-hardening APIs (element-growth getters, unsymmetric-LU
  condition estimate, richer `update()` instability signal) the numeric-focus
  simplex consumes; replaces the temporary git-rev pin.
- **Minimum `pounce-solver` bumped to 0.7** (`chore(deps)`). The interior-point
  KKT solve (`solve_lp_kkt`) the differentiable LP/QP layers and crossover use
  after the JAX LP-IPM retirement requires POUNCE ≥ 0.7.
- **POUNCE is now the default single-solve NLP backend** (`feat(solvers)`).
  For single continuous solves the `ipm` default is promoted to a KKT-valid
  backend via `_default_nlp_solver()`, resolving to POUNCE when installed and
  falling back to cyipopt. B&B convex-polish / dual-recovery passes likewise
  prefer POUNCE through the new `_solve_node_nlp_kkt` wrapper.
- **LP / MILP / QP / MIQP solves stay JAX-free** (`perf(solver)`, #224, #225).
  Linear and quadratic solve paths no longer import JAX, removing the cold-start
  compile tax on fresh solves; node QP relaxations now route through POUNCE.
- **Faster simplex** (`perf(simplex)`, #178, #180). Dual Devex pricing and a
  bound-flipping ratio test; one LU factorization is reused across a node's
  strong-branch probes, and one equilibration is shared across each batch of
  node solves.

### Removed

- **BREAKING: the JAX LP interior-point method is retired** (`refactor(lp)!`,
  #368, #371, #373). The MILP/MINLP node LP relaxations and the standalone LP
  path now use the pure-Rust simplex (degrading to POUNCE); `nlp_solver` governs
  only the NLP subproblem solver. `discopt._relax.lp_ipm` was deleted. This
  completes retirement of the LP fallback chain (`Rust simplex → HiGHS → POUNCE
  → JAX-IPM`).
- **HiGHS removed from the LP/MILP path** (`feat(solvers)`, #356) and the QP path
  (`qp_highs`, #359) — the pure-Rust core is the sole LP/MILP engine.
- **BREAKING: removed the deprecated `ripopt` aliases** (`feat(solvers)!`). The
  old in-repo Rust IPM crate `ripopt` was already superseded by POUNCE; the
  remaining compatibility shims are gone: the `discopt.solvers.nlp_ripopt`
  module, `nlp_solver="ripopt"` (now raises `ValueError`),
  `DISCOPT_MCCORMICK_BACKEND=ripopt`, `sipopt.ripopt_sensitivity`, and the
  `discopt_ripopt` benchmark key. Use `pounce` / `pounce_sensitivity` instead.

### Fixed

- **Sound lower bounds for `log²`/`exp²` objectives** (`fix(relax)`, #372, closes
  #369). The objective linearizer now registers squares of *any* lifted
  univariate call (not just trig), so a mixed objective like nvs09's
  `Σ log(·)² − (∏x)^0.2` produces a sound lower bound instead of falling back to
  a feasibility objective with no bound.
- **Pytest virtual-address cap raised 16 → 32 GB** (`ci`, #360) so JAX/XLA
  compilation no longer aborts with `std::bad_alloc` / exit 134 in CI.
- **Decomposition stage annotation on indexed variables** (`fix(decomposition)`).
  `model.first_stage(y[i])` / `set_stage` / `set_block` on an indexed element
  (`y[i]`) stringified to a stray key (`"y[3][0]"`) that never matched the
  variable name, so the annotated variable silently fell into the recourse
  subproblem and tripped the "integer in recourse" guard. The annotation now
  resolves an indexed reference (or single-variable expression) to its base
  variable name. Surfaced by the new curated adversarial example suite
  (`test_decomposition_adversarial.py`), which carries hand-crafted Benders/GBD
  instances with analytically known optima for each correctness hazard.
- **`.nl` export of builder constraints and objectives** (`fix(export)`). Linear
  constraints built directly into the Rust builder — via the fast-construction
  `add_linear_constraints` API and the indexed-constraint fast path — were
  silently omitted from `to_nl`, which reads `model._constraints`; likewise an
  objective set via `add_linear_objective` / `add_quadratic_objective` was
  exported as a zero placeholder. The model now records each emitted block
  (constraints and the `0.5 x'Qx + c'x + constant` objective) and the `.nl`
  writer reconstructs them — the quadratic part as an n-ary `SUMLIST` nonlinear
  objective term — so a fast-construction model round-trips through `.nl` with
  all constraints and the correct linear/quadratic objective (including a
  constant offset and `maximize` sense) intact.
- **MILP B&B node bound soundness** (`fix(solver)`). The per-node LP soundness
  gate now also rejects a relaxation point that violates the node's variable
  bounds, not only its constraint rows. The pure-Rust simplex adapter (and the
  POUNCE IPM) could return a basic point that violated the variable box on mixed
  equality/inequality nodes; such a point can be integral but off-bound (e.g. a
  binary at -1), pass the row check, and be accepted by the tree as a spurious
  integer incumbent — returning a wrong (too-low) optimum on some
  generalized-assignment-style MILPs. Regression covered in
  `python/tests/test_milp_node_bound_soundness.py`.
- **Clean errors for unsupported decomposition models** (`fix(decomposition)`).
  `solve_benders` / `solve_lagrangian` (and the B&B hook) now raise a clear
  `NotImplementedError` on models the linear extractor cannot handle — e.g.
  multi-dimensional indexed variables — instead of a stray internal `TypeError`.
- **Simplex equilibration over-scaling on noise entries** (`fix(simplex)`). The
  root cause of the MILP wrong-optimum bug below: the geometric-mean
  equilibration treated a numerically-negligible matrix entry (e.g. a ~1e-16 cut
  coefficient that is float noise, not structure) as a genuine nonzero, so a
  column's scale factor blew up to ~1e8. That pinned the variable's *scaled*
  bounds to ~0; the scaled simplex returned a within-tolerance value that
  unscaled into a gross original-space bound violation (a `[0,1]` variable at
  -1), accepted as a spurious integer incumbent. The equilibration now ignores
  entries more than ten orders of magnitude below a line's maximum when forming
  the factor, bounding the per-line dynamic range. The simplex now returns the
  correct vertex (verified against brute-force enumeration and HiGHS). Rust
  regression in `scaling::tests::noise_entry_does_not_overscale_column`.
- **MILP B&B node bound soundness** (`fix(solver)`). Defense in depth for the
  above: the per-node LP soundness gate now also rejects a relaxation point that
  violates the node's variable bounds, not only its constraint rows, so a
  bound-violating point can never seed a spurious integer incumbent. Regression
  covered in `python/tests/test_milp_node_bound_soundness.py`.
- **Relaxation soundness hardening** across the global-opt loop: reject a
  fabricated finite bound on an unbounded McCormick relaxation (`himmel16`,
  `fix(soundness)`); never trust an unconverged simplex objective as an LP lower
  bound (`gear4`, `fix(soundness)`); tangent-separate lifted univariate squares
  (#199); pre-reform interval bound + even-power FBBT (`rbrock` 43s→1.3s, #204);
  fold variable-free product factors and emit sound feasible-exit certificates
  (#179); reject denominator clearing that fabricates a false infeasibility;
  certify `du-opt` globally via epigraph relaxation + rank-1 Hessian (#182).
- **Bound convexity classification can no longer blow the time limit**
  (`fix(solver)`, #228).
- **Preserve integrality for discrete vars in nonlinear `.nl` export**
  (`fix(export)`, #214).
- **`from_gams` correctness on real GAMS files** (`fix(gams)`, #176): 1-D
  parameters and embedded objective variables now translate correctly.
- **Keep wrongly-omitted constraints in the AMP MILP relaxation** (`fix(amp)`, #200).
- **Corrected 9 wrong known-optima** in the benchmark set against MINLPLib
  `minlplib.solu` (`fix(benchmarks)`).

## [0.4.0] - 2026-05-17

### Added

- **AMP global MINLP solver, hardened end-to-end** (`feat(amp)`, #86, #15, #71). Adaptive Multivariate Partitioning gets the contributor build from #44 promoted to first-class status: lifted fractional powers to MILP aux variables (`d8ebffa`); piecewise secants + cover for every nonlinear term (`cc8f741`); piecewise secants for concave fractional powers (`9248fa1`); β-driven piecewise McCormick on bilinear-with-fp (`6cd81e3`); opt-in OBBT-on-relaxation (`e595a11`); cutoff-OBBT now honors `obbt_with_cutoff` and uses live `disc_state` (#71). New README section + worked tutorial at `docs/notebooks/amp_global_minlp.ipynb`.
- **Structural presolve pipeline (#53)** (`feat(presolve)`, #77). Orchestrator wiring 22 structural passes; M4+M5, M9, M10 wired into the root presolve pipeline; presolve roadmap grounded in the literature with B4/D6 prioritization (`fc268a1`, `cfe5b4f`, `22c6298`, `b23c0e7`).
- **Convexification roadmap M1–M11** (`feat(relaxation)`, #51, #75, #79). Permutation-symmetric trilinear McCormick (`70008ef`); M2/M3 relaxation arithmetics + M6 eigenvalue bound (#79); rank-1 certificate path for `x^2/y` on wide boxes (#74).
- **Examiner / KKT validator + solver-dual plumbing** (`feat(examiner)`, `feat(validation)`, #55, #65, #83). New `Model.solve(validate=True)`; `SolveResult` now carries solver duals; Examiner-style KKT validator with independent dual recovery; `minlptests` validator re-validates the primal at the returned `x`.
- **30-lesson optimization course + `discopt tutor` CLI** (`feat(course)`, #85). Full tutorial curriculum and interactive tutor CLI.
- **Deadline-aware JAX IPM** (`feat(deadline)`, #80). Wall-clock `time_limit` honored inside JAX-compiled IPM `while_loop`s.
- **Slice indexing on `IndexExpression`** (`feat(modeling)`, #61). `IndexExpression` now supports Python slice syntax.
- **Tiered Python test suite + ripopt 0.8** (`test`, #69). Fast PR-tier markers separated from full and integration tiers; ripopt bumped to 0.8.
- `discopt-dev` script splits developer commands out of the main `discopt` CLI (`a003ac3`).

### Changed

- **CI Python tests parallelized; coverage moved off PR path** (`ci`, #68, #72). `pytest-xdist` parallel execution by default; coverage job runs nightly + on push-to-main + on `coverage`-labeled PRs to keep PR turnaround fast.
- **Coverage floor temporarily lowered 85% → 70%** (`ci`, #88, tracking #87). AMP merge added ~7k statements without proportional smoke-test coverage; 85% target restored once the AMP test surface is expanded.
- `make test` now matches CI's parallel xdist invocation (`chore(test)`, #68, #84).

### Fixed

- **LOA/OA gap computation near-zero objective** (`fix(loa,oa)`, `9838fdb`). Relative gap was undefined when the objective was near zero; now uses a safe denominator.
- **Serial Ipopt B&B incumbent injection + NaN guards** (`fix(solver)`, #34, #73). `inject_incumbent` now wired into the serial Ipopt B&B path; starting points are clipped before evaluation to suppress NaNs.
- **Convexity certificate for `x^2/y` on wide boxes** (`fix(convexity)`, #42, #74). Rank-1 certificate path correctly identifies convexity over wide variable boxes.
- **LP-data extraction for vector-valued constraint bodies** (`fix(classifier)`, #67). `extract_lp_data` no longer drops vector-valued constraint bodies.
- **Latent mypy + clippy after #53** (`fix(ci)`, `32eb334`). Cleared lint failures introduced by the presolve merge.
- **Large-bound conservatism** (carried forward from `[Unreleased]`). Large-bound warnings remain conservative when nonlinear tightening can infer a smaller box but that tightened box is not applied to every solve path.

## [0.3.0] - 2026-04-24

This release skips the never-tagged `0.2.6` and folds its draft entries into `0.3.0` along with the post-`0.2.6` feature and infrastructure work.

### Added

- **`discopt.mo` -- multi-objective optimization** (`feat(mo): multi-objective optimization via scalarization`). Weighted-sum, AUGMECON2 ε-constraint, weighted-Tchebycheff, NBI, and NNC scalarizations; ideal/nadir payoff-table utilities; `ParetoFront` container; hypervolume / IGD / spread / ε-indicator quality metrics under `discopt.mo.indicators`.
- **`discopt.doe` -- model-based design of experiments**. Identifiability + estimability + profile-likelihood analysis (`feat(doe): identifiability + estimability + profile likelihood`, #48); model discrimination criteria + selection + sequential-design loop (`feat(doe): model discrimination`, #49, #50); batch / parallel experimental design (`feat(doe): batch / parallel experimental design`).
- **AMP -- Adaptive Multivariate Partitioning global MINLP solver** (`feat(amp)`, #44). Iterates MILP relaxation -> NLP subproblem -> partition refinement with the soundness guarantee `LB_k <= global_opt <= UB_k` at every iteration.
- **SUSPECT-style convexity detector** with sound certificates (#46). Structural convexity / concavity / monotonicity proofs for use by the convex NLP fast path and `discopt.mo` reformulations.
- **Claude Code skills + CLI installer** (`feat(cli): ship Claude Code skills in package + discopt install-skills`, `feat(skills): 20 discopt feature / algorithm expert agents`). 20 expert agents shipped in the package and installable into a user's `~/.claude/skills/` via `discopt install-skills`.
- **Crucible knowledge base** tracked in git (`feat(crucible): track wiki, bib, and 3 new articles in git`).
- **Zenodo metadata** and refined manuscript sections (#47).
- `RELEASE.md` -- authoritative release checklist documenting the procedure for cutting a discopt release.
- `CHANGELOG.md` -- this file, in Keep a Changelog format.
- Local `cargo-fmt` pre-commit hook so Rust formatting is enforced alongside `ruff` and `mypy`.

### Changed

- **`ripopt` workspace dependency `0.6.1` -> `0.7.0`** (via `0.6.2`; `Cargo.toml`, `Cargo.lock`). The `0.6.2` step transitively updated `rmumps` `0.1.0` -> `0.1.1`; the `0.7.0` step adapted `crates/discopt-python/src/ripopt_bindings.rs` to the new `NlpProblem` trait signatures: evaluation methods (`objective`, `gradient`, `constraints`, `jacobian_values`, `hessian_values`) now take an explicit `new_x: bool` flag and return `bool` (success / evaluation-failure), matching Ipopt's TNLP contract. Added match arms for the new `SolveStatus::Acceptable`, `SolveStatus::EvaluationError`, and `SolveStatus::UserRequestedStop` variants, surfaced as `"acceptable"` / `"evaluation_error"` / `"user_requested_stop"` on the Python side; `acceptable` maps to `SolveStatus.OPTIMAL` (KKT residuals within Ipopt's relaxed-acceptable-level tolerances).
- `_solve_continuous` (pure-continuous NLP fast path) now promotes the default `nlp_solver="ipm"` to `"ipopt"` for single-problem solves. The pure-JAX IPM's acceptable-tolerance check only covers variable-bound complementarity, so on problems with unbounded variables plus inequality constraints it could terminate at a non-KKT point and report OPTIMAL. Ipopt is more reliable for single solves; the JAX IPM remains the default for B&B subproblems.
- `differentiable_solve` and `differentiable_solve_l3` default backend changed from `"ipm"` to `"ipopt"` for the same reason.
- `solver` now routes pure-MILP problems through HiGHS MIP with a B&B fallback (`fix(solver): route MILP through HiGHS MIP with B&B fallback`).
- **DAE collocation perf** (`perf(dae): vectorize collocation and fix sparse Jacobian for NMPC warm solves`). Vectorized collocation residuals; sparse Jacobian assembly fixed so NMPC warm-start solves don't densify.
- `manuscript/discopt.tex` is no longer tracked -- it is generated from `manuscript/discopt.org`.

### Fixed

- **Jupyter Book docs build with zero warnings**. Cleaned up RST-formatting issues in module docstrings for `benchmarks/problems/gas_network_minlp.py`, `modeling/core.py`, `ro/formulations/box.py`, `solvers/qp_highs.py`, `solvers/sipopt.py`, `doe/discrimination.py`, `doe/discrimination_sequential.py`, `doe/selection.py`, `mo/indicators.py`, `mo/scalarization.py`, `mo/utils.py`, and `solvers/amp.py`; suppressed autoapi import-resolution warnings for the compiled `discopt._rust` extension; escaped `**kwargs` parameter entries to keep Sphinx from parsing the leading `**` as inline strong.
- **HiGHS LP/QP false optimality on wide bounds**: `solvers/qp_highs.py` and `solvers/lp_highs.py` now clip any bound with magnitude `>= 1e15` to `highspy.kHighsInf` before passing to HiGHS. Bounds like discopt's default `+/-9.999e19` fall just below HiGHS's internal infinity threshold (`1e20`) and caused HiGHS to return false-optimal solutions on convex QPs with unbounded variables.
- **Single-solve starting point**: `_solve_continuous` now clips the default starting point to `+/-10` (respecting actual bounds) instead of the previous `+/-100`, preventing ipopt from exploding on exp/log NLPs with one-sided large bounds.
- **Stationary-point starting point**: Fully unbounded variables (`|lb| > 1e15` and `|ub| > 1e15`) now start at `0.5` instead of the midpoint of `0`. Zero is a stationary point of periodic functions (sin, cos) and even functions generally; starting at `0.5` lets first-order NLP methods pick a descent direction and escape local maxima of the objective. Same fix applied in `_relax/differentiable.py::_safe_x0`.
- `_solve_qp_highs` and `_solve_qp_jax` now set `SolveResult.convex_fast_path = True` when solving a detected convex QP directly, matching the semantics of the convex NLP fast path.
- **Cutting planes with bilinear terms (#35)**: `_relax/cutting_planes.py::generate_rlt_cuts` no longer emits unsound inequalities when a bilinear term has no auxiliary `w_index`. The old no-auxiliary branch produced cuts purely in the original variable space that were not valid relaxations of the product (e.g. with `x, y in [0.1, 5]` it emitted `0.1*x + 0.1*y <= 0.01`, excluding every feasible point). Since `detect_bilinear_terms` always returns `w_index=None`, every RLT cut fed into `_AugmentedEvaluator` made the NLP infeasible at every B&B node, so no incumbent could be accepted on mixed convex/nonconvex MINLPs when `cutting_planes=True`. The function now returns `[]` in that case. Fixed in `383985e`.
- `fix(estimate)`: `discopt.estimate` now uses all array observations in residuals and Fisher information, instead of dropping all but the first row.
- `fix(ci)`: cleared a clippy `collapsible_match` and repaired the T24 `vmap` path after the MILP rerouting change.

## [0.2.5] and earlier

Historical releases (`v0.2.0` through `v0.2.5`) are not backfilled in this file.
For commit-level history of those releases, see:

```bash
git log v0.2.4..v0.2.5
git log v0.2.3..v0.2.4
git log v0.2.2..v0.2.3
git log v0.2.1..v0.2.2
git log v0.2.0..v0.2.1
```

Going forward, every release will have a section above with curated entries.

[Unreleased]: https://github.com/jkitchin/discopt/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/jkitchin/discopt/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/jkitchin/discopt/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/jkitchin/discopt/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/jkitchin/discopt/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/jkitchin/discopt/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/jkitchin/discopt/compare/v0.2.5...v0.3.0
