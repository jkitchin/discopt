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
* **Next:** T4 (root-anchored probe/validation boxes + caller-passed presolved
  box). Entry experiment already executed (§1-B), including its negative control
  (nvs01/st_e35 must still decline).
