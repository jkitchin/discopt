## Review addressed — the blocker was real, and it was mine

Thank you for the catch. I verified the 🔴 finding on both arms before acting on
it, and it reproduces exactly as described. §8 marker counts confirm which code
each arm ran (`grep -c SumOverExpression gdp_reformulate.py`):

|  | main (marker=2) | this branch @ bf079fc1 (marker=4) |
|---|---|---|
| `auto` | RAISED `ValueError` | `status=optimal obj=-3.0 bound=-3.0` |
| `hull` | RAISED `HullPerspectiveOriginError` | `status=optimal obj=-3.0 bound=-3.0` |
| `big-m` | RAISED `ValueError` | RAISED `ValueError` |

`bound = -3.0` is **above** the true minimum `-30.0` of a minimization — an
invalid dual bound. My change converted three loud refusals into two silent
false certificates. That is a §1 violation, and §3 besides: a silent wrong
answer replacing a refusal.

**I am retracting the "one real code fix" framing in the PR body (§11).** That
commit was not a fix; it introduced a correctness regression that the 354
passing GDP tests did not cover.

### The fix I took, and why it is not the one you proposed

Your verified patch (teach `_collect_variables`, `_substitute_vars`,
`_hull_linear_substitute`, `_body_at_zero`) is correct and I confirmed it gives
`-30.0`. I did **not** ship it, because it only becomes reachable by keeping the
`_is_linear` widening — and that is exactly the bound-changing change your 🟠
finding says must not ride along in a test-repair PR. It is filed as **#1154**
with your patch recorded, the five functions listed, and the §5 differential
panel as its gate.

What I shipped instead is that `_is_linear` was never the right question at the
LOA call site. Both OA and LOA gated their linear row block behind **two**
predicates — `_is_linear(c.body)` first, then `_extract_body_coeffs` — and the
conservative one wins. The extractor answers with the row itself, so it cannot
promise more capability than it delivers; it is the correct and only gate.

- `_is_linear`: `SumOverExpression` case **reverted**, with a comment recording
  why admitting it is unsound for the hull consumers.
- `_extract_body_coeffs`: keeps the fold — this is the actual bucket-E fix.
- `gdpopt_loa.py` **and** `oa.py`: redundant pre-gate dropped. `oa.py:1348` had
  the byte-identical shape, so both are fixed (§2).

This makes your 🟠 §5 finding moot rather than answered: the FBBT structural mask
(`solver.py:3068`), the aux-lift gate (`factorable_reform.py:684`) and the
Benders route selection are now byte-identical to `main`. No panel is owed.

### 🟡 Missing regression test — added, and it fails before on *both* defects

`python/tests/test_1039_gdp_sumover_rows.py`. Fail-before/pass-after run in the
installed tree (the worktree attempt was void — §8: it imported the main tree's
`discopt`, which is why I re-ran it this way):

| sources at | marker | result |
|---|---|---|
| `bf079fc1` (the widening) | 4 | **2 failed** — `[auto]`, `[hull]`: bound `-3.0` above the optimum |
| `main` | 2 | **2 failed** — extractor returns `None`; LOA returns `unknown` |
| `HEAD` | 3 | **5 passed** |

The blocker arm **asserts** the refusal rather than skipping on it — a skip on
all three routes would have made it a no-op (§6) — and is written as "no route
may report a bound above the true optimum", so #1154 will *strengthen* it rather
than break it.

### 🟡 `row_filter/*` emitted only when > 0 — you were right, and it was hiding one

Fixed to emit zeros, and the assertion now requires the key to be **present**.
That immediately caught a genuine vacuous pass: **`alan` is routed to the convex
`mip-nlp/oa` algorithm and emits no `solver_stats` at all**, so it had been
"passing" against an empty dict. Dormancy is now a positive observation with two
arms — counter present and zero, or a *named non-default* route with no
McCormick LP relaxer. (`algorithm_route is None` means the default path and is
the first arm, not an excuse for a missing counter.) 18 passed.

### 🟡 `ruff format` on the panel script — confirmed and fixed

Also worth noting *why* it got through: the lint job runs `ruff format --check
python/`, so `discopt_benchmarks/` is unchecked.

### 🟠 Issues filed for all four strict xfails, each linked from its reason

- **#1151** — reported objective BELOW the global minimum on quotient
  expressions. False certificate; error is the absolute tolerance amplified by
  `1/denominator`, so it is unbounded as the denominator shrinks. Filed before
  merge as you asked.
- **#1152** — `time_limit` role-1 contract: root-setup overrun (#875) and
  deadline-gated bound loss (#654) are contradictory *as tested*; covers both
  budget xfails.
- **#1153** — incumbent quality non-monotone in `time_limit` (nvs19: 2x budget,
  5x fewer nodes, worse answer).
- **#1154** — complete the hull substitution family, then admit the node.

### Also found, and NOT mine

`test_oa.py::TestOAEdgeCases::test_infeasible_model` asserts `result.x == {}` and
gets `None`. Confirmed pre-existing by running it against `main`'s sources in
this tree (marker=2). It is outside the 18 this issue triaged, so I have left it
alone rather than widen the PR — say the word if you want it pulled in.
