# discopt ⟷ SUSPECT head-to-head

A direct, agree/disagree comparison of discopt's structure detectors against
[SUSPECT](https://github.com/cog-imperial/suspect) (`cog-suspect`), the MINLP
special-structure detector from Misener's group referenced in issue #38.

Both tools run over **one shared corpus** of curated instances, compared on the
three axes SUSPECT reports:

| Axis | discopt API | Live test |
|------|-------------|-----------|
| **Convexity** | `discopt._jax.convexity.classify_expr` | [`test_convexity_suspect_parity.py`](../../python/tests/test_convexity_suspect_parity.py) |
| **Monotonicity** | `discopt._jax.monotonicity.classify_monotonicity` | [`test_monotonicity_suspect_parity.py`](../../python/tests/test_monotonicity_suspect_parity.py) |
| **Interval bounds (FBBT)** | `discopt._jax.convexity.interval_eval.evaluate_interval` | [`test_fbbt_bounds_suspect_parity.py`](../../python/tests/test_fbbt_bounds_suspect_parity.py) |

A key asymmetry, established when the monotonicity / bounds axes were added,
shapes what each test asserts:

* **discopt's verdicts are rigorous.** A convex/concave, nondecreasing/
  nonincreasing, or interval-bound verdict is a *proof* (interval Hessian /
  gradient / value enclosure). Each monotonicity and bounds verdict is validated
  directly against a dense numeric sampling of the real body — independent of
  SUSPECT.
* **SUSPECT's monotonicity and bounds are heuristic and occasionally unsound.**
  Its interval cosine ignores interior critical points, so it e.g. declares
  `sin` monotone — and mis-bounds it — on `[-3, 3]`, where it is neither. discopt
  correctly abstains / encloses there. (Its *convexity* axis, by contrast, is
  sound, so that test asserts mutual no-contradiction symmetrically.)

### Convexity invariants

1. **No contradictions** — neither tool ever proves *convex* where the other
   proves *concave* on the same `≤`-normalised body. Both are sound here, so a
   contradiction would mean one is buggy.
2. **No undetected SUSPECT-stronger cases** — every instance where SUSPECT
   proves curvature discopt misses is a tracked detector gap (currently five:
   SUSPECT's bound-aware trig and inverse-trig rules on a sign-restricted
   branch, see below).

### Monotonicity invariants

1. **discopt monotonicity is sound** — every discopt nondecreasing /
   nonincreasing / constant verdict matches a coordinate-wise numeric sampling.
2. **No proven-direction contradictions** — the two never prove opposite
   directions on the same raw body (pinned empty; discopt's independently
   validated soundness means any conflict would be a SUSPECT defect).
3. **Detector-gap sets pinned** — three SUSPECT-stronger items
   (`sqrt_concave`, `cubic`, `sine`) are tracked; the first two are sound
   discopt conservatism (domain-edge / sub-ULP), the third a SUSPECT unsoundness
   discopt rightly avoids.

### FBBT interval-bounds invariants

discopt's `evaluate_interval` is a *forward* natural-range enclosure (it does
not use a constraint's RHS); SUSPECT's `bounds.get(...)` is full FBBT (forward ∩
backward-from-RHS), so on constraint bodies SUSPECT is typically tighter, while
on quadratics / powers / `tan` discopt is tighter (SUSPECT leaves those
unbounded). The test asserts:

1. **discopt bounds are sound** — the forward enclosure contains the body's
   dense numeric sampling on every item.
2. **No disjoint enclosures** — discopt's and SUSPECT's enclosures of the same
   body always overlap (a disjoint pair would prove one tool unsound).
3. **Unbounded-abstention set pinned** — the five items where discopt's forward
   pass abstains to an unbounded endpoint (sqrt-of-sum-of-squares, inverse trig)
   are tracked so the set cannot silently grow.

## Why SUSPECT runs out of process

`cog-suspect` 2.1.3 is unmaintained and incompatible with discopt's runtime:
it imports `np.float` (removed in numpy ≥ 1.24) and Pyomo's
`ReciprocalExpression` (removed in Pyomo ≥ 6.2), neither of which can coexist
with discopt's numpy 2.x / Pyomo 6.10. So SUSPECT cannot be a live dependency.

Instead its verdicts are recorded **once**, in an isolated environment, into
the committed golden file [`suspect_verdicts.json`](suspect_verdicts.json). The
parity test imports only the neutral corpus and the discopt renderer — never
SUSPECT — and compares discopt's live verdicts against that golden file.

## Files

| File | Environment | Purpose |
|------|-------------|---------|
| `corpus.py` | none (stdlib only) | Single source of truth: a neutral expression AST + 40 curated instances, plus a numeric `eval_ast` used to validate verdicts against the real body. Imported by *both* sides. |
| `render_pyomo.py` | SUSPECT env | Renders the neutral AST into Pyomo models. |
| `run_suspect.py` | SUSPECT env | Runs SUSPECT over the corpus (convexity + monotonicity + FBBT bounds), writes `suspect_verdicts.json`. |
| `render_discopt.py` | discopt env | Renders the *same* AST into discopt models; `build_discopt_items` also exposes each item's *raw* (pre-canonicalisation) body for the monotonicity / bounds cross-checks. |
| `suspect_verdicts.json` | — | Committed golden verdicts (all three axes per item). |

Because both renderers walk the same neutral AST, the comparison is a genuine
head-to-head on identical mathematics, not two independently hand-written
models that might silently differ.

## Regenerating the golden file

Only needed when the corpus changes. Build the pinned, isolated SUSPECT
environment and run the oracle:

```bash
# 1. Create an isolated env (SUSPECT needs an old, narrow dependency window).
uv venv /tmp/suspect_oracle_env -p 3.10
source /tmp/suspect_oracle_env/bin/activate

# 2. Pin the exact compatible versions:
#    - pyomo 6.0–6.1 is the only window with BOTH ScalarExpression (≥6.0) and
#      ReciprocalExpression (removed in 6.2);
#    - numpy 1.23.x still provides the np.float alias SUSPECT imports;
#    - setuptools<81 still ships pkg_resources.
uv pip install "cog-suspect==2.1.3" "pyomo==6.1.2" "numpy==1.23.5" "setuptools<81"

# 3. Generate (writes suspect_verdicts.json next to run_suspect.py):
cd scripts/suspect_oracle
python run_suspect.py            # or: python run_suspect.py --check  (print only)
```

`run_suspect.py` installs small `np.float`/`np.int`/… shims at import time and
isolates each instance in its own `try/except` (SUSPECT's interval FBBT can
raise `EmptyIntervalError` on degenerate boxes), so one failing instance never
sinks the whole run. The instance count and any SUSPECT errors are recorded in
the file's `_meta` block.

After regenerating, re-run the parity test and update the pinned
`EXPECTED_DISCOPT_STRONGER` / `KNOWN_SUSPECT_STRONGER` sets in the test if the
agreement profile changed:

```bash
pytest python/tests/test_convexity_suspect_parity.py \
       python/tests/test_monotonicity_suspect_parity.py \
       python/tests/test_fbbt_bounds_suspect_parity.py -v
```

## Current result (40 instances, 43 compared items)

### Convexity

| Category | Count | Meaning |
|----------|------:|---------|
| agree | 33 | identical verdicts (modulo affine ⊆ convex∩concave) |
| discopt-stronger | 5 | discopt proves curvature where SUSPECT abstains |
| SUSPECT-stronger | 5 | SUSPECT proves curvature discopt leaves UNKNOWN |
| contradiction | 0 | — |

The five **discopt-stronger** items are exactly the cone primitives SUSPECT
2.1.3 does not recognise: the Euclidean norm (`sqrt` of a PSD quadratic),
quadratic-over-affine (`x²/y`), the perspective of `exp` (`y·exp(x/y)`), the
`nlp_cvx_108`-style fractional epigraph, and a second-order-cone constraint.

The five **SUSPECT-stronger** items are SUSPECT's bound-aware trig / inverse-trig
rules: on a sign-restricted branch it proves `sin` convex (`[π, 2π]`), `cos`
concave (`[−π/2, π/2]`), `asin` convex (`(0,1)`), `acos` concave (`(0,1)`), and
`atan` concave (`x > 0`) — curvature discopt's detector currently leaves
UNKNOWN. They are pinned in `KNOWN_SUSPECT_STRONGER` as tracked detector gaps.

### Monotonicity

| Category | Count | Meaning |
|----------|------:|---------|
| agree | 40 | identical direction (modulo constant ⊆ nondec∩noninc) |
| discopt-stronger | 0 | — |
| SUSPECT-stronger | 3 | `sqrt_concave`, `cubic`, `sine` (see invariants above) |
| contradiction | 0 | — |

Every discopt nondecreasing / nonincreasing / constant verdict is also checked
against a coordinate-wise numeric sampling of the body, so the comparison rests
on discopt's own validated soundness rather than on trusting SUSPECT.

### FBBT interval bounds

discopt's forward enclosure is **sound on all 43 items** (contains the sampled
range), and the two enclosures are **never disjoint**. The enclosures rarely
coincide exactly because they compute different things (forward natural range vs
full FBBT) — for constraints SUSPECT is tighter (it uses the RHS); for
quadratics / powers / `tan` discopt is tighter (SUSPECT leaves them unbounded).
discopt abstains to an unbounded endpoint on five items (`euclidean_norm`,
`norm_le`, and the three inverse-trig objectives), pinned in
`DISCOPT_UNBOUNDED`.

## Atom-coverage gaps

The corpus walks SUSPECT's *buildable* atom × regime matrix (abs, tan, log2,
the full power-exponent set, bound-restricted convex trig, and inverse trig).
discopt's expression layer is a generic `FunctionCall` node whose JAX backend
already maps every atom SUSPECT has a rule for (`asin → jnp.arcsin`, etc.), so
**discopt can represent everything SUSPECT can** — `asin`/`acos`/`atan` simply
lack a `discopt.modeling` convenience wrapper and are built directly via
`FunctionCall` in `render_discopt.py`. Hence `UNBUILDABLE_SUSPECT_ATOMS` is
**empty**.

The only one-directional gap is the reverse: atoms discopt expresses but
SUSPECT 2.1.3 cannot, recorded as a named constant in `corpus.py`:

| Constant | Atoms | Reason |
|----------|-------|--------|
| `UNBUILDABLE_SUSPECT_ATOMS` | *(none)* | discopt's `FunctionCall` backend represents every SUSPECT atom. |
| `SUSPECT_UNSUPPORTED_ATOMS` | `tanh`, `sinh`, `cosh`, `log10` | discopt expresses them, but SUSPECT 2.1.3 raises `ValueError: Unknown function type` (no rule in its registry). |

(`log2` *is* buildable: it is rendered as a positively-scaled native `log`, so
SUSPECT sees an ordinary concave `log` rather than a dedicated atom.)
