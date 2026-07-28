# Issue #861 — Incremental McCormick admission plan (loop-executable)

**Target issue:** #861 *“Incremental McCormick declines 50/81 instances: 38 are
`_validate` patch/cold mismatches.”*
**Plan date / HEAD:** 2026-07-28, `bdc104bb` (post-#764 native-kernel graduation,
post-#873 even-power admission).
**All evidence in §1 was re-measured at this HEAD** with instrumented probes; the
issue’s 31/50 split and its 24/14/6/5/1 bucket counts reproduce exactly.

**Why this still matters post-#764:** `IncrementalMcCormickLP` is not only the
opt-in `lp_spatial` engine’s fast path. It is built by the **default**
`MccormickLPRelaxer` (`mccormick_lp.py:535-563`, `DISCOPT_INCREMENTAL_MC` default
ON) and serves `_try_incremental_node` in the main Python B&B — the path every
solve takes when the native Rust kernel (#764) declines a model or the solve is
feature-unsafe (`solver.py:_native_kernel_feature_safe`). The native producer
(`spatial_producer.py`) covers an even narrower family set than the Python
incremental path, so the 50 declined instances are running the ~30× slower
per-node cold build in *both* engines today.

---

## §0 Loop protocol (binding)

1. Work the §5 task list top to bottom, one task per iteration. Each task lands
   as its own scoped PR from a feature branch (`perf/861-<task>` /
   `fix/861-<task>`); never commit to `main`. Title names the issue
   (`Contributes to #861`; only the final task uses `Closes #861`).
2. **Entry experiment before implementation** (CLAUDE.md §4). Each task lists its
   entry probe and kill criterion. The §1 probes already run count as executed
   entry experiments where noted; re-run a probe only if the tree has drifted
   under the relevant files since this plan’s HEAD.
3. **Never weaken `_validate`.** Every change either (a) makes the *boxes* it
   compares on realistic (reachable from the true root box), (b) makes the
   *classification* of rows correct, or (c) makes the *patch* reproduce the cold
   build on more families. The comparison itself (`bounds` allclose 1e-6,
   `_rowset` exact-polytope identity, box-independent objective offset) stays
   byte-identical. A task that can only pass by loosening a tolerance or
   dropping a comparison is dead — stop and record it in §7.
4. Every probe/panel script prints an executed-assertion count and exits
   non-zero when it is zero (CLAUDE.md rule 6). No bare `except` in instruments.
5. After every task: run the §4/G2 sweep and paste the admitted/declined table
   into the PR. Admissions must be ≥ the previous task’s count, and **no
   previously-admitted instance may become declined** — if one does, that is
   either a bug in your change or a genuine divergence the old synthetic boxes
   missed; investigate and record before proceeding, never paper over.
6. Record falsified hypotheses in §7 as you go (house style:
   `docs/dev/performance-plan.md` §6). Update the issue with a short comment per
   landed task.

---

## §1 Measured root causes (evidence, this session)

Baseline sweep at HEAD over the 81-instance corpus
`python/tests/data/minlplib/*.nl`, constructing `IncrementalMcCormickLP(model,
classify_nonlinear_terms(model), deadline=None)` and capturing the DEBUG decline
log: **31 admitted, 50 declined** — `bounds mismatch` 24, `column-count
mismatch` 14, `rows, expected 4` 6, `no valid bound / no rows` 5, odd-power 1.
Identical to the issue’s table.

### Bucket A — `bounds mismatch` (24): NOT an over-strict comparison; genuinely unpatched lifted families

`_validate` fails at `bounds mismatch` (`incremental_mccormick.py:764`) because
the mismatching **aux columns belong to lifted families the patch never
updates** — their `base_bounds` keep probe-box values while the cold build
recomputes them per box. Attribution of the first-failing columns (probe:
patched-vs-cold aux bounds on validation box 0, attributed via the
`UniformRelaxation` maps/specs):

| instance | offending aux family | measured example (patched vs cold) |
|---|---|---|
| ex1222 | `univariate_atom_specs` (`exp`, x0, coeff 1.0, const −0.2) | (2.2255, 897.85) vs (0.8187, 6.0496) |
| trig, mathopt3, syn05m | `univ_atom` `sin`/`cos` | sin: (−1,1) vs (0.4794, 1) |
| alan, nvs07, st_e05, st_e07, st_e40, util, kall_circles… | `bilinear_linform_specs` (products of *linear forms*, `_emit_mccormick`) | alan col 8: (4, 873) vs (−4, 44) |
| st_e02, st_e15, ex1221, nvs20, nvs04, kall_circles_c8a | **power of a multi-var LinForm** or **fractional power** — coverage kind `power`, absent from `monomial_map`/`affine_square_map` | st_e02: (4, 225) vs (≈0, 16) = `(x0+x1)^2`; ex1221/st_e15: `x^1.5` |
| ex1252, ex1252a, mathopt3 | `trilinear_map` (+ blf) | ex1252 col 41: (1, 1750) vs (0, 8) |
| chance | a `univariate_call` atom not exported in any spec (cold aux lower −inf) | (4.6465, 41.81) vs (−inf, 9.293) |

Key structural fact (read `uniform_relax._emit_1d`, line ~1220): **every 1-D
atom** — power (any real exponent), exp, log, sqrt, sin, cos, … — of a LinForm
`t` emits *exactly* secant + 3 tangents (4 rows, support = form vars ∪ {aux})
when its curvature verdict on `[lo,hi]` is definite, and **0 rows** when the
verdict abstains (`curv is None`), the box is degenerate (`< _MIN_WIDTH`), or an
endpoint is non-finite; individual tangents are silently skipped on non-finite
`f`/`f'`. Products of LinForms (`_emit_mccormick`) emit the 4 McCormick rows over
the forms’ interval images. So the entire bucket reduces to **two generic
closed-form recipes plus a bounds tape** (§2/D5) — not 24 bespoke fixes.

### Bucket B — `column-count mismatch` (14): synthetic probe/validation boxes are unreachable, and the engine’s *decomposition* is box-dependent

The probe box (`incremental_mccormick.py:354-360`: `[1, 7+k]` sign-matched) and
the validation boxes (`_validation_boxes`, lines 672-724: absolute magnitudes,
`lo=0.0` on even trials for positive vars) **ignore the model’s real root box
entirely**. Measured on `gear` (root box `[12,60]^4`, strictly positive):

* probe box `[1,7+k]` build: **15 columns** — the ratio `i1·i2/(i3·i4)` is
  lifted through `log(x_i)` columns (bounds `(0, ln 7)`…), reciprocal columns
  (`(1/9, 1)`), and a sum-of-logs column (the strictly-positive log-space route,
  `uniform_relax.py` ~1801/2232);
* validation box 0 (drives positive vars to `lo=0.0`, **unreachable** from
  `lb=12`): **10 columns** — log route unavailable, reciprocal bounds
  `(−inf, inf)` → `column-count mismatch`.

Same mechanism on ex1225/ex1226 (probe 14/13 vs box0 11/10) and hda (1145 vs
1082). The five `no valid bound / no rows` instances (Bucket D) are the same
defect at a different exit: the *synthetic* box makes the relaxation invalid
(`nvs06`, root `[1,200]^2` — validation `lo=0` ⇒ unbounded reciprocal ⇒
`_objective_bound_valid=False`), or the *raw* root box is itself invalid
pre-presolve (`st_e04`: cold build on the true root has
`_objective_bound_valid=False`; ex1233 has 28 infinite root bounds).

**Falsification measured this session (binding):** root-anchoring alone does
*not* make every layout stable. Building the cold relaxation on 6 *reachable*
sub-boxes of the true root box: `gear`/`ex1225`/`ex1226` are stable on all
non-degenerate sub-boxes (unstable only when every var is pinned — a folding
regime `_validate` never tests for sign-definite vars, cf. the #873 pinned-box
vacuity precedent); but **nvs01 and st_e35 flip decomposition on interior
reachable boxes** (root has `lb=0` columns; strictly-positive interior sub-boxes
*gain* the log-route columns: nvs01 11→15, st_e35 69→87). Those models’ per-node
cold builds genuinely change column layout across the tree; a fixed-layout
incremental structure cannot be bound-neutral against them. They must keep
declining (honestly) unless a route-pinning build option is added — an explicit
non-goal here (§6).

### Bucket C — `bilinear (i,j) → 5 or 6 rows, expected 4` (6): support-based row classification counts lifted *model constraints* as envelope rows

`_build_structure` classifies a row as a product’s envelope iff its support ⊆
`{i, j, aux}` (`incremental_mccormick.py:426-428`). A model constraint such as
`x0·x1 ≥ 60` is lifted to the single-column row `{aux: −1} ≤ −60` — support ⊆
`{i,j,aux}` — so prob02’s bilinears count 5 rows and the whole structure
declines. Measured extra rows: prob02 `{6:−1} ≤ −60`, st_e01 `{2:1} ≤ 4`,
st_e08 `{2:−16} ≤ −1`, st_e09 `{0:2, 1:2, 2:4}`, st_e11 two rows also touching a
second aux. These extra rows are **box-independent** (they are model rows) and
belong with the fixed rows.

**Fix confirmed by entry experiment:** selecting the 4 envelope rows by *numeric
match* against the closed-form `_bilinear_rows` evaluated on the probe box
(coefficients + rhs, atol 1e-9) uniquely identifies them in all 6 instances, and
a hand-driven `_patch`+`_validate` over all 6 validation boxes then **passes
cleanly for st_e01, st_e09, prob02, prob03**. st_e08/st_e11 pass the row stage
and move to Bucket A (their monomial/other aux weren’t patched in the manual
harness; st_e08’s monomials are patched by the real code). The same
support-⊆-based classification is used by the **native producer**
(`spatial_producer.py`, “a row is a term’s envelope iff its support is contained
in …”) — audit it for the same defect (T3).

### Bucket D — `relaxation has no valid bound / no rows` (5)

See Bucket B: synthetic-box artifact (nvs06, st_e35, ex1233 via infinite raw
root) or genuinely-invalid raw root box that the real solve only fixes via
FBBT/OBBT before building (st_e04). The constructor never receives the
presolved box: `lp_spatial_bb.py:553` builds `IncrementalMcCormickLP(model,
terms, deadline=…)` *after* computing OBBT-tightened `lb0/ub0` but doesn’t pass
them; `mccormick_lp.py:559` likewise. (`spatial_producer.build_spatial_kernel_spec`
already grew a `bounds=` parameter for exactly this reason — tanksize.)

### Ergonomics blocker (from the issue, confirmed)

The decline reason exists only as a `logger.debug` line
(`incremental_mccormick.py:297-300`); nothing is stored on the object. The
baseline sweep had to scrape the log.

---

## §2 Design

**D1 — `decline_reason` attribute.** In `__init__`, initialize
`self.decline_reason: str | None = None`; in the `except` set it to
`f"{type(exc).__name__}: {exc}"` (keep the debug log). Also set a reason at the
two early-return declines (expired deadline; `lp_spatial`’s
`box_is_patchable` flip happens at the caller — leave that to the caller’s log).

**D2 — Admission sweep as an in-repo meter.**
`discopt_benchmarks/scripts/incremental_admission_sweep.py`: iterate
`python/tests/data/minlplib/*.nl` (81 instances), build the structure exactly as
the default relaxer does (optionally with the root box argument from D4), print
one line per instance (`ADMIT`/`DECLINE reason`), write JSON, print the bucket
histogram, exit non-zero on zero instances. This is the meter every task’s PR
must quote (before/after).

**D3 — Numeric envelope-row classifier.** In `_build_structure`, for each
registered product/monomial/affine-square (and each family added by D5): among
candidate rows (support ⊆ term cols), select rows by numeric equality
(atol 1e-9 on each coefficient and rhs) with the closed-form envelope evaluated
**on the probe box**. Exactly one candidate must match each expected row
(ambiguity ⇒ decline, as today). Non-matching candidate rows are left as fixed
rows. Mirror the audit onto `spatial_producer._classify` (same defect class); if
confirmed there, fix it the same way in a separate commit of the same PR with
its own regression test (a prob02-like model through
`build_spatial_kernel_spec`).

**D4 — Root-anchored probe & validation boxes; caller-passed box.**
* Constructor grows `box: tuple[np.ndarray, np.ndarray] | None = None` —
  the *finite, presolved* root box. Call sites: `lp_spatial_bb.py:553` passes
  the post-OBBT `(lb0, ub0)`; `mccormick_lp.py:559` passes the flat root bounds
  it already has (post-presolve at that layer). Default `None` keeps reading
  `model._variables` (back-compat).
* Probe box: for each var, distinct interior endpoints of the ROOT interval,
  strictly sign-matched, avoiding zero endpoints (support detection needs every
  McCormick coefficient nonzero): e.g. `lo_i = rlb + (0.11 + 0.017·(i mod 7))·w`,
  `hi_i = rub − (0.13 + 0.013·(i mod 5))·w`, clamped to keep `lo < hi`; for an
  infinite endpoint keep today’s synthetic magnitude on that side; a root-pinned
  var (`w=0`) keeps its point (its envelope rows will be degenerate — if
  `_index_of` then misses a pattern entry the structure declines as today, and
  the instance is recorded; do not special-case).
* Validation boxes: keep the six C-21 regime kinds and the spanning-var
  span/neg/zero_lb/degen coverage, but generate every interval **inside the root
  interval**, scaled by root width (spanning root vars have `rlb<0<rub`, so all
  sign regimes remain reachable; sign-definite vars get full-width,
  lb-touching, interior-shifted, and ub-touching sub-boxes — *no* degenerate
  trials for sign-definite vars, exactly today’s scope, per the #873 pinned-box
  vacuity precedent).
* The `_box_sign_regime` bookkeeping and `_validated_regimes` stay; assert in
  the new unit test that the generated set still covers
  `{pos, span, neg, degen, zero_lb}` regimes for a model with spanning vars
  (`zero_lb` only when the root box actually contains 0 as a lower bound —
  reachability wins over regime nostalgia; record coverage actually achieved).

**D5 — Generic patch tape (the coverage extension).** Replace the three
hand-maintained families with a *tape* exported by the uniform engine and
replayed by `_patch`:

* **Export (uniform_relax, export-only — zero behavior change to the build):**
  extend `UniformRelaxation` with an `envelope_tape` describing, for every
  box-dependent envelope emitted: `("oneD", fname_or_("pow", p), linform
  (dict col→coeff, const), aux, reserved_row_count=4)` and `("prod", linformA,
  linformB, aux)` — generalizing today’s `univariate_atom_specs` /
  `bilinear_linform_specs` / `affine_square_map` to *all* `_emit_1d` and
  `_emit_mccormick` sites — plus, for **every** aux column, a `bounds_recipe`
  in topological order: either `("interval_oneD", fname/p, linform)`,
  `("interval_prod", linformA, linformB)`, or `("linform", linform)` for
  linear-definitional aux, recording exactly the computation `ctx.bounds`
  performs (reuse the same `Interval` ops — the #873 parity lesson:
  reproduce the engine’s enclosure, including its outward rounding and
  non-finite handling; *never* substitute a tighter closed form).
* **Consume (incremental_mccormick):** `_build_structure` maps each tape entry’s
  rows by the D3 numeric matcher, using the *same* `f/f'/curvature` table as the
  engine (import it from `uniform_relax` — do not re-derive derivatives).
  `_patch` replays, per node box: aux bounds from the recipes (in tape order,
  so a form over aux columns reads already-updated bounds), then envelope rows:
  when the curvature verdict on this box is definite → write secant + 3
  tangents (skipping a tangent exactly when the engine would, writing a
  vacuous `0 ≤ 0` row in its slot); when the verdict abstains / box degenerate /
  non-finite → write 4 vacuous rows (matches the cold build’s **no rows**, and
  `_rowset`’s vacuity filter drops them on comparison — mechanism already in
  place from #873).
* **Reserved rows:** an atom can emit 0 rows on the probe box but 4 on a node
  box (sin/cos with indefinite curvature at the probe box; odd power on a
  straddling box). For any tape atom whose probe-box emission is < 4 rows,
  **append** 4 reserved rows to `base_A` (explicit entries on form vars ∪ aux,
  initialized vacuous). `_rowset` filtering makes the comparison exact either
  way; LP shape stays fixed across nodes (basis reuse unaffected in
  correctness; a few extra vacuous rows are cheap).
* **Families in scope for this plan:** `prod` (bilinear + all linform
  products), `oneD` powers (integer, fractional on domain-valid boxes,
  multi-var forms — subsumes monomial + affine-square), `oneD` intrinsics
  (exp, log, sqrt, sin, cos — whatever the engine’s 1-D table holds),
  linear-definitional aux bounds. **Out of scope** (structure declines, exactly
  as today, now for an honest reason): trilinear/multilinear lifts, ratio
  log-space *band rows* (`_emit_logspace_band`) if they prove box-dependent
  beyond the tape shapes, composite specs, trig-square selector tables. Measure
  what remains (T7) before deciding any follow-up.
* The old `bilinear`/`monomial`/`affine_square` public attributes stay populated
  (consumers: `lp_spatial` info map, C-44 column identities, cut inheritance).

**D6 — Odd power on a straddling root box (`ex8_5_4`).** With D5’s reserved
rows + abstain-→-vacuous mechanism, the odd-power gate
(`incremental_mccormick.py:437-442`) can be dropped *iff* the engine’s cold
emission for odd `p` on a straddling box is “curvature abstains → 0 rows”
(then the patch writes 4 vacuous rows and stays bound-neutral, merely inheriting
the cold build’s loose interval floor there). Entry probe: build gear-style cold
relaxations of `x^3` over `[-1,2]` and check emitted rows for the monomial aux.
If the cold build instead emits a genuine 2-facet S-hull (different row family),
keep the decline for odd-`p`-straddling and close that sub-item as won’t-fix
with `ex8_5_4` named (the issue explicitly allows this).

**D7 — Bound-neutrality panels (§4)** are the graduation gate; no new env flag
is added. The admission widening is inherently gated per-model by `_validate`
(build-time, zero-risk fallback), which is this subsystem’s designed gate;
`DISCOPT_INCREMENTAL_MC=0` remains the global opt-out.

---

## §3 Expected per-instance outcomes (testable predictions)

* **T3 alone (+4):** st_e01, st_e09, prob02, prob03 admit. st_e08, st_e11 move
  to Bucket A.
* **T4 alone:** nvs06 admits only if its families are covered (needs T5) — but
  its failure moves from `no valid bound` to an attributable mismatch; st_e04 /
  ex1233 / nvs05(∞ root) become buildable via the passed presolved box (their
  final admission still needs T5). No previously-admitted instance may regress.
* **T4+T5 (bulk):** expected admissions include ex1221, ex1222, st_e02, st_e05,
  st_e06, st_e07, st_e15, st_e36, st_e40, nvs04, nvs07, nvs08, nvs16, nvs20,
  trig, syn05m, util, alan, chance, kall_circles_c8a, gear, gear2, gear3,
  gear4(∞ root → box pass), ex1225, ex1226, st_e17, st_e08, st_e11, mathopt5_2,
  nvs06, st_e04 — i.e. every decliner whose census is within
  {blf, power-of-form, univ_atom, linear-def}. **Expected to remain declined
  (honest):** nvs01, nvs05, nvs22, st_e03, st_e38, st_e35, hda (trilinear /
  ratio-route instability / multilinear), ex1252, ex1252a, mathopt3
  (trilinear), ex8_5_4 (pending T6), ex1233 (32 blf but check trilinear-free —
  may admit). Quantitative goal: **≥ 55/81 admitted** (from 31). Report the
  full final table regardless; if the goal is missed, the per-instance reasons
  (now stored, D1) say exactly which family/route each residual needs — record,
  don’t stretch.

---

## §4 Verification gates (every task PR states which it ran, with numbers)

* **G1 — Regression tests.** Each task adds tests that fail before / pass after
  (pytest, `python/tests/test_incremental_*`): the prob02 lifted-model-row
  classifier case; a root-anchored-boxes case (gear admits; a synthetic
  nvs01-like route-flip model still declines); per-family patch parity tests in
  the style of `test_monomial_aux_bounds_match_interval_pow` (patch vs engine
  interval/emitter parity across sign regimes, incl. non-finite boxes).
* **G2 — Admission sweep** (D2 script): before/after table per PR; monotone
  non-decreasing; no admitted→declined flips (see §0.5).
* **G3 — Per-node LP-value parity (the direct bound-neutrality gate).** New
  panel script: for every **newly admitted** instance, sample ≥ 20 reachable
  sub-boxes of the true root box (branching-style splits incl. integer pins and
  sign-regime diversity), solve the patched LP and the cold-build LP, assert
  bound equality to 1e-9·(1+|bound|), assert identical
  infeasibility verdicts; print executed-comparison count. Zero violations
  required. (This is the per-node analogue of `_validate`, on real instances
  and realistic boxes rather than the 6 synthetic regimes.)
* **G4 — End-to-end certifying panel.** Full `solve()` over the 81-instance
  corpus with the change vs. `DISCOPT_INCREMENTAL_MC=0`, oracle =
  `minlplib.solu` (`~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu`):
  `incorrect_count = 0`, no dual bound crossing its reference optimum, no
  `gap_certified=True` instance regressing to uncertified, objective drift
  within tolerance (abs 1e-6 / rel 1e-4). On the 31 previously-admitted
  instances: node_count and certified objective **exactly unchanged**
  (bound-neutral regime, CLAUDE.md §5).
* **G5 — Net-positive measurement (CLAUDE.md §5 bar 2).** On ≥ 5 newly-admitted
  instances: wall time and node throughput, incremental ON vs OFF, interleaved
  A/B with load gate (`uptime`), ≥ 3 repeats, report mean ± sd
  (CLAUDE.md rules 9–10). The expectation is the measured ~30× per-node speedup
  class; if a newly-admitted instance is *slower* ON (tape overhead), find out
  why before shipping the family.
* **G6 — Suites.** `pytest -m smoke`, adversarial suite
  (`pytest -m slow python/tests/test_adversarial_recent_fixes.py`); Rust
  untouched by this plan (no `cargo test` needed unless T3’s
  `spatial_producer` audit leads into `crates/` — it shouldn’t).

---

## §5 Task list

**T1 — `decline_reason` + sweep meter (D1 + D2).** Small PR. Includes the D2
script + a unit test that a declined structure carries a non-None reason and an
admitted one carries None. *Entry experiment:* none needed (mechanical).
*Done:* sweep runs at HEAD reproducing 31/50 with stored reasons (no log
scraping).

**T2 — Baseline lock-in.** Commit the sweep’s JSON baseline (31/50 table) under
`discopt_benchmarks/results/` via the D2 script so later diffs are one command.
(May be folded into T1’s PR.)

**T3 — Numeric envelope-row classifier (D3) + native-producer audit.**
*Entry experiment (already executed, §1-C):* numeric matching uniquely
identifies the 4 envelope rows and full validation passes on
st_e01/st_e09/prob02/prob03. *Kill criterion:* ambiguous numeric match (two
candidate rows equal within atol) on any corpus instance ⇒ keep decline for
that term, log reason. *Done:* +4 admissions; st_e08/st_e11 reasons now
`bounds mismatch`; prob02-class regression test; `spatial_producer` audited
(and fixed if affected, with its own test). *Gates:* G1, G2, G4 (31 admitted
unchanged), G6.

**T4 — Root-anchored probe/validation boxes + `box=` parameter (D4).**
*Entry experiment (already executed, §1-B):* gear-class layouts are stable on
reachable non-degenerate sub-boxes; nvs01/st_e35 flip on reachable boxes (they
must still decline — this is the task’s negative control). *Kill criterion:*
any previously-admitted instance regresses under root-anchored boxes and
investigation shows the *boxes* (not a real divergence) are at fault ⇒ fix box
generation, do not exempt the instance. *Done:* probe/validation boxes are
sub-boxes of the (possibly caller-passed) root box wherever it is finite; both
call sites pass their presolved boxes; regime-coverage unit test; st_e04/ex1233
reach `_validate` instead of dying at the probe build. *Gates:* G1, G2, G4, G6.

**T5 — Generic patch tape (D5), landed family-by-family in three PRs:**
* **T5a `prod` generalization** (linform × linform products; subsumes bare
  bilinear): expected admissions among {st_e05, st_e07, st_e40, util, nvs07,
  alan(box), st_e11, kall_circles… partial}.
* **T5b `oneD` generalization** (powers of forms incl. fractional; exp/log/
  sqrt/sin/cos via the engine’s own f/f′/curvature table; reserved rows +
  abstain→vacuous): expected admissions among {st_e02, st_e15, ex1221, ex1222,
  nvs04, nvs08, nvs16, nvs20, trig, syn05m, st_e36, st_e06, mathopt5_2,
  st_e08}.
* **T5c bounds tape for linear-definitional aux + full-aux coverage check**
  (gear’s log/reciprocal/sum columns; assert at build time that *every* aux
  column has a recipe, else decline with reason `uncovered aux <family>`):
  expected admissions among {gear, gear2, gear3, gear4, ex1225, ex1226,
  st_e17, nvs06, chance, ex1233}.

*Entry experiment per sub-task:* before implementing, write the parity probe
(patch recipe vs cold engine on 200 random finite/sign-diverse boxes for that
family; plus non-finite-endpoint cases for T5b) and run it against a prototype
of the closed form; the #873 methodology (exact-rational or high-precision
check on a sample, executed-assertion counts, a deliberately-broken control
that must fire). *Kill criterion:* any family whose cold emission is not
reproducible from the tape shapes (e.g. box-dependent facet count beyond the
reserved 4, engine-internal state leaking into rows) ⇒ that family stays
unpatched and its models keep declining with a stored reason; record in §7.
*Gates per sub-task:* G1, G2, G3 (newly admitted), G4, G6; G5 after T5c.

**T6 — Odd power on straddling root (D6).** *Entry probe:* what does the cold
build emit for `x^3` on `[-1, 2]`? 0 rows (abstain) ⇒ drop the gate, admit
`ex8_5_4` under T5b mechanics, add the regression test. S-hull rows ⇒ won’t-fix:
comment on the issue naming `ex8_5_4` as the sole consumer, keep the gate,
done. Either outcome closes issue-DoD item 3.

**T7 — Close-out.** Final sweep table (before: 31/50 with buckets; after:
measured), G3/G4/G5 numbers, per-instance residual reasons for everything still
declined, retract/confirm the §3 predictions explicitly, update
`docs/dev/performance-plan.md` §6 with any falsifications from §7, and comment
on #861 with the summary ending in an explicit **“issue can be closed: yes/no
(+ exactly what remains)”** per CLAUDE.md. `Closes #861` only if DoD items 1–4
are all met (they are, if T3–T6 landed: buckets triaged with per-bucket
verdicts (§1), comparison fixed/narrowed with panels clean, odd-power decided,
before/after sweep reported).

---

## §6 Non-goals (do not drift into these)

* The `ball_mk2_30` primal gap — #862’s workstream (issue correction 1).
* Route-pinning the uniform engine’s decomposition (would admit
  nvs01/st_e35/hda-class) — file a follow-up **only** in T7 and **only with**
  the measured residual table as its evidence; do not build it here.
* Extending the native Rust kernel’s family coverage (#764 follow-ups) — except
  the T3 classifier audit, which is a correctness/coverage defect shared by
  construction.
* Any tightening of aux bounds beyond the engine’s own enclosure (explicitly
  forbidden — bound-neutrality is the product; see `_monomial_aux_bounds`’s
  docstring contract).

## §7 Falsification log (append as you go)

* *(this session)* “The `_validate` mismatches are an over-strict comparison
  (issue hypothesis b).” — **Partially falsified.** Bucket A is genuine missing
  patch coverage (correctly declined today); only Buckets B/D stem from
  unrealistic comparison *boxes*, and Bucket C from misclassification.
* *(this session)* “Root-anchored validation boxes suffice for the
  `column-count` bucket.” — **Falsified** for nvs01/st_e35 (decomposition
  flips on *reachable* interior boxes: 11→15 / 69→87 columns). Root-anchoring
  is necessary, not sufficient; those instances stay declined pending a
  route-pinning decision (out of scope).

---

## Progress ledger

Append one entry per landed task (§0.1 recovers loop state from here).

### T1 + T2 — `decline_reason` + admission sweep meter + baseline — PR #893

* **Landed:** `python/discopt/_jax/incremental_mccormick.py` stores
  `decline_reason` (set on the exception path *and* on both non-raising
  deadline guards, which were the paths most likely to read as “no reason”);
  new meter `discopt_benchmarks/scripts/incremental_admission_sweep.py`;
  baseline `discopt_benchmarks/results/issue861_admission_baseline_20260728.json`.
* **Sweep (G2), at `bdc104bb` + this change:** **admitted 31/81, declined 50** —
  `bounds mismatch` 24, `column-count mismatch` 14, `envelope row count != 4` 6,
  `no valid bound / no rows` 5, `odd power on straddling root` 1. Reproduces the
  issue’s table exactly, now from the stored attribute rather than a scraped
  DEBUG log. This JSON is the baseline every later task diffs against
  (`--baseline`, which exits non-zero on any admitted→declined flip).
* **Gates:** G1 — `python/tests/test_861_decline_reason.py`, 10 tests;
  verified **3 fail before / all pass after** (`AttributeError: … has no
  attribute 'decline_reason'`), plus the bucket-mapping contract pinned against
  the exact messages the code raises. G2 — baseline established (no prior run to
  regress against). G6 — `pytest -m smoke` 850 passed / 1 skipped / 2 xpassed;
  adversarial suite 10 passed; incremental-McCormick files 160 passed;
  `ruff check` + `ruff format` clean. (`mypy` reports one error inside numpy’s
  own stubs — verified **pre-existing on clean `main`**, unrelated.)
* **Behaviour change:** none in the solve path. The attribute is diagnostic; no
  code branches on it, and admission is byte-identical (31/81 before and after).
* **Next:** T3 (numeric envelope-row classifier + `spatial_producer` audit).
  Its entry experiment is already executed (§1-C).

### T3 — numeric envelope-row classifier + native-producer audit — PR #894

* **Sweep (G2): 31 → 36 admitted, 45 declined.** Newly admitted: `prob02`,
  `prob03`, `st_e01`, `st_e08`, `st_e09`. **No admitted→declined flips.** The
  `envelope row count != 4` bucket is **eliminated** (6 → 0); `st_e11` moved to
  `bounds mismatch` (24 → 25) exactly as §3 predicted. `+5` beats the predicted
  `+4` — `st_e08` also admitted (§1-C’s manual harness had not patched its
  monomials; the real code does).
* **Design change vs the plan, forced by a second entry experiment.** The plan
  said to assign formulas in *match* order. Measured over the 31 admitted
  instances (630 terms): the numeric-match order is a **rotation** of today’s
  ascending-index order on **30 of 31**, so assigning in match order would
  permute the rows of every node LP for nearly the whole admitted set and put
  the exactly-unchanged node-count gate at risk for zero benefit. Narrowed the
  change: numeric matching **selects** the envelope subset, ascending order
  still **assigns** the formulas. `_rowset` is order-free, so both describe the
  same polytope; this one is byte-identical wherever a term already had 4 rows.
* **No-op evidence (the strongest form of G4 for the pre-existing set):** a
  captured/verified structural snapshot (`ncol`, `nnz`, `nrows`, `_prod_rows`,
  all three row maps, `_pos`) is **IDENTICAL on all 31 previously-admitted
  instances**. Same formula in the same physical row on every box ⇒ node counts
  and bounds cannot move; no panel run needed to establish it.
* **Native producer (`spatial_producer`) — same defect, worse failure mode,
  fixed.** Its `bilinear_linform_specs` path claimed *every* support-contained
  row with **no count check**, so a lifted model row was excluded from the fixed
  rows AND not regenerated by the kernel ⇒ **dropped from every node LP**.
  Audited: 38 of 1024 BLF terms over-claim (13 instances), and the spec is
  actually **built** — hence the defect **reachable**, with the kernel default-ON
  since #764 — on 6 (`prob02`, `prob03`, `st_e01`, `st_e05`, `st_e08`, `st_e09`).
  **Direction is sound**: a dropped relaxation row gives a superset feasible
  region ⇒ weaker but valid dual bound, and all 6 still certify against
  `minlplib.solu`. Still a silent tightness loss on a path whose premise is
  byte-for-byte reproduction, so it now **declines** (matching what the
  monomial/affine-square `_claim(want=4)` paths already did). Distribution
  measured before choosing the predicate: **986 terms claim exactly 4, 38 claim
  more, 0 claim fewer** — so `!= 4 ⇒ decline` removes exactly the misclassified
  terms and no working coverage. Those 6 now land on the Python incremental path,
  which this same PR admits.
* **Gates:** G1 — `python/tests/test_861_envelope_row_classifier.py`, 9 tests,
  **all 9 verified failing before / passing after**; includes a direct assertion
  that owned rows move with the box and unowned rows do not (the property support
  containment failed to capture), and a native-producer test with a precondition
  assert so it cannot pass vacuously. G3 — new panel
  `discopt_benchmarks/scripts/incremental_node_parity_panel.py`: **150 LP
  comparisons over branching-style sub-boxes of the 5 newly-admitted instances,
  0 disagreements**, and a **firing control** (envelope coefficient corrupted
  *after* construction) produced **9 disagreements**, proving the panel is not
  blind. G4 — 7 instances end-to-end vs `minlplib.solu`: **0 violations**, no
  bound above its reference optimum. G6 — smoke 850 passed; adversarial 10
  passed; incremental files 169 passed; native-kernel files 32 passed; ruff clean.
* **Retraction (CLAUDE.md rule 11):** the first G4 run of this task reported
  `st_e11` as `OBJ-MISMATCH BOUND-ABOVE-OPTIMUM` against an oracle of `0.067038`.
  That oracle value was **mis-transcribed by me**; `minlplib.solu` gives
  `st_e11 = 189.3116297`, which discopt matches (obj `189.31162974`, bound
  `189.31162969`). There was no violation. Re-run with the corrected oracle:
  7 checked, 0 violations.
* **CI caught what the local suites did not — fixture rot (read this before T4/T5).**
  Smoke + adversarial + the incremental/native files all passed locally, but CI's
  broader `Python fast` lane failed two #844 tests. Neither was a behaviour
  regression: both fixtures **depended on the defect being fixed**.
  - `test_844_lp_spatial_deadline.py` built its "fast path unavailable" control
    from bilinear *constraints*, and its docstring even documented the reason —
    "its bilinear constraint rows lift to 5 envelope rows where the closed-form
    patch emits 4". That is precisely the misclassification T3 fixes, so the
    model started building a structure and the `require_incremental` guard was
    asserting that the engine declines a model it can now serve.
  - `test_844_unresolved_child_soundness.py` injects failures into
    `_relax_bound` — the **cold-path** entry point. Its model started using the
    fast path, so the injection never fired and its own vacuity guard
    (`assert state["failures"] > 0`) failed. **The guard worked exactly as
    intended**; without it the test would have passed while asserting nothing.
  Both repaired by making the premise DURABLE and EXPLICIT: fixtures now use a
  **trilinear** coupling (outside the closed-form patch's families by design, so
  the decline is a property of the mathematics, not of a bug) and each test
  **asserts its own precondition** so a future coverage widening points at the
  fixture instead of failing obscurely. Added
  `test_failed_child_on_the_incremental_path_never_yields_a_false_certificate`,
  which injects into `IncrementalMcCormickLP.solve` so the unresolved-child
  soundness property is covered on the **fast path** — now the common case, and
  previously untested.
  **Standing lesson for T4/T5:** every admission widening can silently convert a
  cold-path test into a fast-path test. Before opening a PR, run CI's exact fast
  selection locally, not just `-m smoke`:
  `pytest python/tests/ -m "not slow and not correctness and not integration and
  not amp_benchmark and not requires_cyipopt and not memory_heavy"
  --ignore=python/tests/test_correctness.py --ignore=python/tests/test_amp.py`
  and grep the diff's blast radius for tests whose fixtures assume a decline.
* **Next:** T4 (root-anchored probe/validation boxes + caller-passed presolved
  box). Entry experiment already executed (§1-B) and **re-run at T3 HEAD over all
  45 declined instances** (210 uniform builds), which sharpens §3 and corrects it:
  - **RESCUABLE (28)** — layout identical across the root box and 6 reachable
    root-anchored sub-boxes: `ex1221 ex1222 ex1225 ex1226 ex1252 ex1252a ex8_1_1
    gear gear2 gear3 kall_circles_c8a mathopt5_2 nvs04 nvs06 nvs07 nvs08 nvs16
    nvs20 st_e02 st_e03 st_e05 st_e06 st_e07 st_e11 st_e15 st_e36 st_e38 trig`.
    This is an UPPER BOUND on what box-anchoring alone can convert — most still
    need T5 patch coverage — not a prediction of T4's admission gain.
  - **UNSTABLE (3)** — layout genuinely flips on a reachable sub-box; T4's
    negative control: `nvs01` (11→15 cols), `nvs21`, `st_e17` (5→8 cols).
  - **NEEDS_BOX (14)** — infinite raw root box or invalid root relaxation, so
    nothing can be judged until the caller's presolved box is passed: `alan
    chance ex1233 ex8_5_4 gear4 hda mathopt3 nvs05 nvs22 st_e04 st_e35 st_e40
    syn05m util`.
  - **Corrections to §3/§7:** the plan named `nvs01` + `st_e35` as the unstable
    pair. Measured, the unstable set is `nvs01`, `nvs21`, `st_e17`; **`st_e35` is
    NEEDS_BOX** (its ROOT relaxation has no valid bound), so its stability cannot
    be judged until a presolved box is passed — it was mis-attributed to the
    unstable class. Also `gear`/`gear2`/`gear3` are rescuable as predicted but
    **`gear4` is NEEDS_BOX** (infinite root box), so it depends on the
    caller-passed box, not on box anchoring.

### T4 — root-anchored probe & validation boxes + caller-passed box — PR (this branch)

* **Sweep (G2): 36 -> 36 admitted, 0 flips.** T4 gains no admissions on its own —
  as §3 predicted — but it removes the *box artifact*, converting artifact
  declines into honest family-coverage declines that T5 can act on:

  | bucket | T3 | T4 |
  |---|---|---|
  | `bounds mismatch` | 25 | **38** |
  | `column-count mismatch` | 14 | **3** |
  | `no valid bound / no rows` | 5 | **3** |
  | odd power on straddling root | 1 | 1 |

* **The residual `column-count mismatch` set is exactly `{nvs01, nvs21, st_e17}`** —
  precisely the three the T4 entry experiment predicted as genuinely box-unstable
  (their lifted decomposition changes on *reachable* interior sub-boxes; nvs01
  11 -> 15 columns). A confirmed prediction, and the negative control: a widening
  that admitted these would have stopped detecting real patch/cold divergence.
* **What changed.** `_probe_box()` and `_validation_boxes()` now generate every
  interval inside the model's root box (`_root_box()`), and the constructor takes
  `box=(lb, ub)` so a caller can supply its *presolved* root box —
  `lp_spatial_bb` passes its post-OBBT `(lb0, ub0)`. `mccormick_lp` deliberately
  does NOT pass one (it is constructed before it knows the branching box; a
  guessed box would be worse than the model's own bounds). `_finite_root_interval`
  gives an infinite endpoint a finite stand-in anchored at the finite side, which
  is what lets the ex1233/alan/util class reach `_validate` at all instead of
  dying at the probe build (`no valid bound` 5 -> 3).
* **Gates:** G1 — `python/tests/test_861_root_anchored_boxes.py`, 14 tests, pass
  after / fail before (note: for the three negative-control cases the fail-before
  is the constructor signature change, not behaviour — their value is that they
  *pass* after). Includes a regime-coverage test proving anchoring did not cost
  the C-21 sign-regime spread. G3 — node-parity panel over 18 admitted instances,
  **360 LP comparisons, 0 disagreements**. G4 — oracle check, **0 violations**.
  G6 — CI's exact fast selection locally: **7347 passed, 29 skipped, 5 xfailed,
  2 xpassed, 0 failures**; ruff clean.

### Oracle hygiene — two fabricated reference values, and the fix

Twice in this issue I checked results against a reference optimum I had **typed
from memory rather than read**: `st_e11` against `0.067038` (true value
`189.3116297`) and `st_e17` against `1.0` (it has **no** `=opt=` entry in
`minlplib.solu` at all). Both produced spurious "false optimum / bound above
optimum" reports against answers that were correct; both evaporated on re-check.

Fixed at the level of method, not the individual numbers:

* `discopt_benchmarks/scripts/incremental_oracle_check.py` **parses**
  `minlplib.solu`, never accepts a typed value, distinguishes a certified
  `optimal` objective (checkable) from a `time_limit` incumbent (not a
  correctness claim), always checks the dual bound whatever the status, and
  reports an instance with no reference as `NO-ORACLE` instead of passing it.
* Only **2 of 81** corpus instances lack a MINLPLib reference: `st_e17` and
  `meanvar`. Both are now established by **independent external global solvers**
  on the exact in-repo `.nl` and recorded with full provenance in
  `discopt_benchmarks/data/local_oracle.json`:
  - `st_e17 = 376.291905403861` — SCIP (proven, gap 0%, dual = primal), BARON via
    GAMS (Model Status 1 Optimal, `376.2919`), Couenne (Optimal). discopt returns
    `376.2919323`, bound `376.2918930` — **correct to ~7 significant figures**.
  - `meanvar = 5.24339865067014` — SCIP (proven, gap 0%) and Couenne (0% gap,
    338 nodes) agree; BARON not run and the entry says so and why (its GAMS
    binary does not read `.nl`, and hand-transcribing a dense 8-variable
    quadratic is the very error class this file exists to prevent). discopt
    returns `5.2433990`, **correct to ~7 significant figures**.
  The merge rule is `minlplib.solu` wins on any overlap, so a local measurement
  can only fill a gap, never override upstream.

---

## §7 addendum — T5's premise is falsified by measurement (2026-07-28)

Three measurements taken as T5a's entry experiment, together, invalidate the T5
design and its staging. Recorded per CLAUDE.md §4 (the measurement wins) before
any T5 code was written.

**1. The staging is backwards — the bounds tape is a PREREQUISITE, not a follow-on.**
Classifying all 457 product specs in the `bounds mismatch` bucket by what their
operand forms reference:

    BARE_BILINEAR 179    ORIG_FORM 27    AUX 251   (55% reference an AUX column)

Only **6** instances (`alan`, `kall_circles_c8a`, `st_e05`, `st_e07`, `st_e11`,
`util`) have products over original columns only, which is all T5a-as-scoped can
patch. The other **24** reference aux columns whose bounds are themselves
box-dependent and produced upstream by the engine — so they need the ordered
bounds tape (planned as T5c, *last*). T5a → T5b → T5c cannot be executed in that
order.

**2. The instances T5 would admit are the ones where patching saves least.**
Per-node cold-build cost over the corpus:

    ADMITTED (fast path today)   n=36  median 0.20 ms  p90 1.10 ms  max 9.42 ms
    DECLINED (T5 would admit)    n=45  median 0.20 ms  p90 1.14 ms  max 20.41 ms

Only **5 of 45** declined instances have a cold build above 1 ms: `hda`
(20.41 ms, 722 vars), `ex1233`, `kall_circles_c8a`, `nvs20`, `util`. For the
other 40 the per-node build being replaced costs ~0.2 ms.

**3. Admission delivers no measurable end-to-end speedup on that class.**
Interleaved A/B (never sequential), 3 repeats, `DISCOPT_INCREMENTAL_MC` 1 vs 0,
load checked before and after (load average 2.6 → 4.9):

    prob02  0.38±0.04 / 0.37±0.01 = 0.96x     nvs01   1.05±0.15 / 1.04±0.20 = 0.99x
    st_e01  0.50±0.01 / 0.48±0.01 = 0.96x     ex1225  1.09±0.30 / 1.07±0.36 = 0.98x
    st_e09  0.56±0.01 / 0.58±0.03 = 1.03x     gear    0.54±0.07 / 0.55±0.01 = 1.02x

`prob02`/`st_e01`/`st_e09` are ADMITTED (so ON exercises the fast path, OFF the
cold one); `nvs01`/`ex1225`/`gear` are DECLINED and therefore run the cold path
either way — they are the **noise-floor control**, and the admitted instances are
indistinguishable from it. Validity checked (CLAUDE.md rule 8): the native #764
kernel returns `None` for all six, so they genuinely run the Python path and the
env var genuinely toggles it. (`st_e13` IS handled natively and was excluded.)

**Conclusion.** T5 as designed is the largest and riskiest task in this plan — it
reimplements the engine's interval propagation as a replayable tape, where any
divergence is a bound-neutrality violation — and measurement says it would buy
~1.00x on 40 of the 45 instances it targets. That is the `DISCOPT_CUT_INHERIT`
shape exactly: sound is not the same as helpful, and a cert-clean but
neutral change does not earn its risk.

Where admission *should* pay is the large-model tail — `hda` above all (20.41 ms
per node, 722 variables) — and `hda` sits in the AUX-tape class. So the honest
re-scope is to target the tape at the families the large instances actually need,
or to drop the tape in favour of a hybrid that calls the engine's own bound
propagation (parity by construction rather than by reimplementation), and to stop
treating raw admission count as the objective. **Not started pending an owner
decision on which of those to pursue.**

### T6 — odd power on a straddling root box: WON'T-FIX (measured)

Entry probe (fresh, this session): row count the COLD build emits for ``s = x**p``
over a straddling box vs sign-definite ones, read off the lifted matrix:

    p=2  straddling -> 4 rows    positive -> 4    negative -> 4
    p=3  straddling -> 2 rows    positive -> 4    negative -> 4
    p=4  straddling -> 4 rows    positive -> 4    negative -> 4
    p=5  straddling -> 2 rows    positive -> 4    negative -> 4

D6's condition was "0 rows (curvature abstains) => drop the gate". The answer is
**2 rows**, not 0: the cold build emits a genuinely different row FAMILY there
(the 2-facet S-hull), so the D6 escape does not apply and the sub-item closes as
**won't-fix**, exactly as the issue's DoD item 3 permits.

Not claimed impossible — claimed disproportionate. The 4-row reserved pattern
*could* in principle carry 2 real facets plus 2 vacuous rows (``_rowset`` drops
vacuous rows, the mechanism #873 introduced for pinned boxes). What makes it not
worth building is the facets themselves: an S-hull facet is the tangent drawn
from the opposite endpoint, whose tangency point is the root of a degree-``p``
equation — closed-form only for small ``p``, a numerical solve in general, and
each one must reproduce the engine's emission bit-for-bit or the structure is not
bound-neutral. That is real machinery for a class with exactly **one** consumer in
the corpus (``ex8_5_4``), which the fast path already serves correctly via the
cold build.

Already pinned in-repo by
``test_861_monomial_span_zero.py::test_odd_power_monomial_still_declines_on_a_root_box_spanning_zero``
plus the row-count table in that file's docstring, so the decision is verifiable
rather than asserted.

---

## T7 — close-out (2026-07-28)

**Final sweep, original baseline -> now:** admitted **31 -> 36** of 81, declined
50 -> 45. Newly admitted: `prob02`, `prob03`, `st_e01`, `st_e08`, `st_e09`. Zero
admitted->declined flips at any step.

| bucket | baseline | final | what happened |
|---|---|---|---|
| `bounds mismatch` | 24 | 38 | grew *by design* — instances moved here from the two artifact buckets, now declining for an honest, attributable reason (an unpatched lifted family) instead of a box artifact |
| `column-count mismatch` | 14 | **3** | artifact removed; the 3 survivors (`nvs01`, `nvs21`, `st_e17`) are genuinely box-unstable and must decline |
| `envelope row count != 4` | 6 | **0** | eliminated — the misclassification is fixed |
| `no valid bound / no rows` | 5 | **3** | infinite-root instances now reach `_validate` instead of dying at the probe build |
| odd power on straddling root | 1 | 1 | won't-fix, measured (T6) |

**Issue DoD status.** (1) Triage the two dominant buckets — **done**, with a
verdict per bucket and the finding that `bounds mismatch` was *not* an
over-strict comparison but genuine missing coverage. (2) Fix or correctly narrow
the comparison so admission rises, bound-neutrality clean — **done**: +5
admitted, both artifact buckets collapsed, and every gate clean at every step
(360-comparison parity panel with a firing control, 0 oracle violations, 0
flips). (3) Odd power — **decided** (won't-fix, T6, measured). (4) Before/after
sweep over the same corpus — **done**, committed as JSON at each step.

**Not done, and why:** T5's family-coverage work (the patch tape). Its premise is
falsified by measurement — see the §7 addendum above: the staging is backwards
(the tape is a prerequisite, not a follow-on), the instances it targets are the
ones where patching saves least (median cold build 0.20 ms; only 5 of 45 exceed
1 ms), and admission delivers **0.96–1.03x end-to-end** on that class, inside the
noise floor set by declined-instance controls. Building it would reimplement the
engine's interval propagation — where any divergence is a bound-neutrality
violation — for no measured gain. Owner decision (2026-07-28): finish T6, close
#861, and file a follow-up targeting the large-model tail where the cost model
says admission actually pays (`hda`: 20.41 ms/node, 722 vars).

RALPH-861-COMPLETE
