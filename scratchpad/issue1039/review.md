## Re-measured against current `main` (267 commits after triage)

Every one of the 18 was re-run on `main` at `0c561b45`, with the Rust extension
rebuilt first (the checked-in `.so` was 13 days stale and produced 17 bogus smoke
failures until rebuilt — CLAUDE.md §8). Net: **17 of 18 still fail, 1 now passes,
and 1 new failure joined the same test as an existing item.** No test file in the
list has been touched since the triage.

The headline is bucket A: the retraction comment above is wrong, and it is wrong
in a way that inverts the disposition.

---

### 1. Bucket A — the retraction exonerated the one flag that works, and cleared two it never measured

The three bucket-A tests toggle **three different flags**, not one:

| test | flag |
|---|---|
| `test_relax_row_filter.py:119` | `DISCOPT_RELAX_ROW_FILTER` |
| `test_issue_517_numerical_dual_bound.py:59` | `DISCOPT_NODE_NUMERICAL_DUAL_BOUND` |
| `test_issue_671_lp_iterative_refinement.py:114` | `DISCOPT_LP_ITERATIVE_REFINEMENT` |

The retraction measured `DISCOPT_RELAX_ROW_FILTER` only (`mccormick_lp.py:1615`,
`filter_invocations=0` in both arms), concluded "the mechanism it gates is simply
dormant on `hda`", and then applied that finding to all three under "Revised
disposition for bucket A".

Measured on `hda.nl` at `time_limit=60`, all three flags, both arms, one flag set
per run (`scratchpad/issue1039/probe_hda.py`, 6 executed checks, non-zero exit at zero):

```
DISCOPT_RELAX_ROW_FILTER=0        bound=-13992288065.862448   <- loose candidate-A floor
DISCOPT_RELAX_ROW_FILTER=1        bound=-64509.84568260102    <- tight
DISCOPT_NODE_NUMERICAL_DUAL_BOUND=0  bound=-64509.84568260102
DISCOPT_NODE_NUMERICAL_DUAL_BOUND=1  bound=-64509.84568260102
DISCOPT_LP_ITERATIVE_REFINEMENT=0    bound=-64509.84568260102
DISCOPT_LP_ITERATIVE_REFINEMENT=1    bound=-64509.84568260102
```

* `DISCOPT_RELAX_ROW_FILTER` is **live on `hda`** and does precisely what its test
  asserts: OFF gives the loose floor (−1.40e10), ON gives the tight bound. The two
  arms differ by six orders of magnitude. The retraction's claim that "the two arms
  are identical because the mechanism is dormant" does not hold on today's tree.
  Its test now passes, and it passes **genuinely** — I checked specifically that the
  asserts are not being skipped by the `if r.bound is not None and math.isfinite(...)`
  guard at `test_relax_row_filter.py:117`, which would otherwise make the test pass
  vacuously (§6). The bound is finite; the asserts run.
* The other two flags **are** inert on `hda` — byte-identical in both arms — and
  their two tests still fail. These are the two the retraction never probed.

So the original bucket-A observation was right for two of three flags, the
retraction cleared the wrong one, and the §5 "keep the `=0` opt-out intact" question
is still open for `DISCOPT_NODE_NUMERICAL_DUAL_BOUND` and
`DISCOPT_LP_ITERATIVE_REFINEMENT`.

**Caveat, applying the same standard to myself:** inert *on `hda`* is not inert
everywhere. Both remaining flags gate rescues that fire only when a node LP breaks
down, and with the row filter at its default `hda`'s root LP solves cleanly — so
there may be nothing to rescue rather than a broken opt-out. Deciding that needs an
invocation count for those two flags, on an instance that still provokes an
uncertified node LP. That is the same measurement the retraction ran for the row
filter, just pointed at the right flags.

The retraction's *recommended action* survives intact and now applies to two tests
instead of three: re-point them at an instance that still breaks down, rather than
retire them.

(Minor: the tight value has moved from the triage's −64473.44 to −64509.85.)

### 2. Bucket E — the `bchoco07` drift is real and flag-caused; `casctanks` is a broken assertion

The issue read both as "last-bit nondeterminism". They are two different things.
Holding the flag fixed at `0` and re-running (`probe_withinarm.py`, 4 comparisons):

```
bchoco07  flag=0 x3 : 1.0000000000002582, 1.0000000000002582, 1.0000000000002582   (nodes=3,3,3)
casctanks flag=0 x3 : 6.040896844275638,  6.040896844275636,  6.040896844275639    (nodes=3,5,3)
```

* **`bchoco07` is deterministic within an arm.** Interleaved ON/OFF/ON/OFF/ON
  (`probe_bchoco.py`) gives `1.0000000000002498` on all three ON runs and
  `1.0000000000002582` on both OFF runs. The flag genuinely changes the bound, which
  contradicts the test's stated premise ("the filter never fires ... byte-identical").
  Either the filter fires on `bchoco07` or the flag perturbs a path it should not.
  Worth root-causing — it is a small but reproducible ON/OFF divergence on a path
  documented as untouched.
* Also from that run: the ON arm consistently takes **21.5–21.9 s against a 20 s
  `time_limit`**, the OFF arm 20.1 s. A reproducible ~1.5 s overrun that appears only
  with the flag ON. That connects to bucket B.
* **`casctanks` is a NEW failure**, not in the original 18 — same test, different
  parameter — and its bound varies *within a single arm*, along with the node count
  (3, 5, 3). Both arms hit `time_limit=20`, so the assertion is comparing two
  truncated searches: how far the tree got depends on the clock. `off.bound ==
  on.bound` cannot hold for `casctanks` regardless of the flag. That assertion is
  invalid as written for any instance that times out; it is not evidence about the
  flag. This one is a test defect, and it will keep spreading to more parameters as
  timings shift.

The other three bucket-E items reproduce with values matching the issue exactly
(`gp_corpus` 1.998683979470214; gauss-newton `feasible` vs `optimal`; `gdpopt_loa`
`unknown`).

### 3. `test_gdpopt_loa` is not a hard problem — that path does nothing at all

The issue calls this "the most concerning of these". It is worse than a status miss.
The model is `min x`, `x ∈ [0,10]`, `(x ≤ 3) ∨ (x ≥ 7)`:

```
gdp_method="loa"    status=unknown  obj=None  bound=0.0  nodes=0  wall=0.20s
                    mip_count=0  subnlp_calls=0  algorithm_route=None
gdp_method="big-m"  status=optimal  obj=0.0   nodes=1              <- control, same model
```

Zero MIP solves, zero NLP subproblems, zero nodes, returns in 0.2 s. LOA is not
struggling; it never runs an iteration.

The test file contains its own controlled comparison: `test_simple_disjunction`
builds **identical** math via `m.either_or(...)` and passes. The only failing LOA
test is the only one using the `make_disjunct`/`add_disjunction` block API, which
lowers eagerly at build time into two `_IndicatorConstraint`s plus an exactly-one
row — leaving no disjunction for the LOA driver to find. Given a model with nothing
it recognises, LOA returns `unknown` instead of falling back to ordinary B&B or
refusing loudly (§3). Nothing unsound (`obj=None`, no certificate), but a silent
no-op return is the wrong behavior either way.

### 4. Bucket B — the overrun is a constant ~30 s, not a ratio

```
budget=30s : wall=60.6s  (issue: 61.1s)   excess 30.6s
budget=60s : wall=89.4s  (issue: 90.0s)   excess 29.4s
```

The issue reports this as "1.5–2.0x". Both points are the same **fixed ~30 s** of
un-deadlined root setup; the ratio only shrinks because the denominator grows. That
predicts ~a 30 s excess at any budget, which is a sharper and more testable
statement of what #875 left unbounded. Reproduced at load 3.95→6.06 with an OS
`mediaanalysisd` at 119% CPU — but a 30 s absolute excess with a constant-offset
structure across two budgets is far outside anything load explains (§9).

### 5. Buckets C, D, F — unchanged, all 8 reproduce

C (3/3) and D (3/3) fail on the exact named tests. D's "one instance measured twice"
inference is correct: `test_auto_cut_policy.py:92` and `test_psd_node_reduction.py:56`
both build a 6-variable unconstrained dense indefinite box-QP
(`_qcqp(6, 0, constrained=False)` / `_dense_indefinite_qcqp(6, 0)`) and compare
`cuts="manual"` against PSD. Worth noting that
`test_auto_matches_best_family_and_preserves_optimum` has a second half covering a
constrained QCQP/RLT case that **never executes**, because the box-QP assert fails
first — so that coverage is dark, not merely failing.

F (2/2) reproduces. `test_issue654` deserves promoting out of "completeness misses":

```
sonet23v4: status=time_limit  bound=None  nodes=0  wall=4.7s
```

Zero nodes and **no dual bound at all** in 4.7 s of a much larger budget — the §8
truncation regression the test names, not a search that ran out of budget. That is
a different and more serious shape than its bucket-mate `nvs19`, which does search
(7688 nodes, 54.4 s) and simply does not reach the optimum. `nvs19` is a genuine
completeness miss as the issue says; `sonet23v4` is a dropped bound.

### 6. Bucket G — decidable, and a live consumer is already miscounting

This reproduces and is not merely a contract ambiguity. On `nvs17`:

```
time_limit=6s : status=time_limit  obj=-1100.4  bound=-1514.706902950524
                gap=37.65%  gap_certified=True  nodes=17329
time_limit=20s: status=optimal     obj=-1100.4  bound=-1100.4
                gap=9.5e-14  gap_certified=True
```

The issue is right that nothing is unsound — the bound is below the true optimum and
the status is honest. But the tie-breaker it asks for already exists in the tree:
`discopt_benchmarks/scripts/gp_minlp_graduation_panel.py:185-188` reads the flag as
"closed its tree":

```python
# (c) ON must certify (a recognised GP-MINLP closes its tree).
if not on.gap_certified:
    res.cert_clean = False
    res.violations.append("ON did not certify (gap_certified=False)")
```

So a 37.7 %-gap time-limited run is currently counted as certified by a graduation
gate. That is the `__post_init__` reading, not the docstring reading, and it is the
one with a consumer behind it. The docstring reading has no consumer I could find.
That is enough to settle it in favour of the test: `gap_certified` should be `False`
here, and the docstring at `core.py:2240` is what needs correcting. This is a
gate-correctness question, not a cosmetic one.

### 7. The stated prerequisite was settled in the opposite direction

> Prerequisite for all of it: #1034. Until these tiers run somewhere on a schedule,
> any fix here can regress the same way.

#1034 is closed, but not by adding a schedule. `.github/workflows/ci.yml:15-21` now
records the opposite decision:

```
# NO `schedule:` trigger. A nightly re-run of the slow correctness lane is a
# ~2 h runner job every day against a SHA that has usually not moved, and the
# CI-minute cost is not worth it ... `python-correctness-slow` below is
# `workflow_dispatch`-only: run it deliberately, before a release or when
# touching the certificate path.
```

That is a reasonable call on its own terms, but it means this issue's premise no
longer holds: nothing will catch a regression in these 18 between deliberate runs,
and the accumulation that produced them can recur. Worth either accepting that
explicitly in the issue or revisiting the cadence question separately — it should
not sit here as an unmet blocker.

---

### Suggested re-prioritisation

1. **`gdpopt_loa`** — a documented method silently does nothing on one of two
   modelling APIs. Cheapest to confirm, worst behaviour.
2. **`sonet23v4` dual bound dropped at 0 nodes** — a bound that disappears is worth
   more than a bound that is loose.
3. **Bucket G** — decide `gap_certified`, since a graduation gate is miscounting
   today. My read: the test is right.
4. **Bucket A, the two unmeasured flags** — run the retraction's invocation-count
   probe against `DISCOPT_NODE_NUMERICAL_DUAL_BOUND` and
   `DISCOPT_LP_ITERATIVE_REFINEMENT`, on an instance that still breaks down.
5. **`casctanks`/`bchoco07`** — split them: fix the assertion for timed-out
   instances, root-cause the reproducible `bchoco07` ON/OFF divergence.
6. **Bucket B** — re-state as a constant ~30 s of un-deadlined root setup.
7. **C and D** — probe health, as triaged. D's dark second half is worth a line.

Probes: `scratchpad/issue1039/{probe_hda,probe_withinarm,probe_bchoco}.py` — each
prints an executed-check count and exits non-zero at zero; none swallows an
exception.
