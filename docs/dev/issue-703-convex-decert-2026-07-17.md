# Issue #703 — "convex NLP-BB gap_certified=False": working-as-intended, hypothesis falsified

**Date:** 2026-07-17 · **Branch:** `docs(#703)` · **Verdict:** no solver change. The
conservative behavior is correct, and the issue's hypothesized *mechanism* is falsified by
measurement. The real lever is root-relaxation quality (already #282 R2-2), not node-solve
robustness.

## TL;DR

Issue #703 asks *"why are the convex node NLPs untrusted?"* — assuming `gap_certified=False`
on the four convex instances (`rsyn0805m`, `rsyn0810m`, `rsyn0815m`, `syn40m`) is produced by
the node-level decertification guards at `solver.py:~10206` (`_batch_trusted`) and the serial
`ITERATION_LIMIT` path.

**Measured on current `main` (24f9b29a): the node NLPs are NOT untrusted. Zero of them.**
Every convex node NLP converges to `SolveStatus.OPTIMAL` (a clean KKT point) and is trusted.
`gap_certified` stays `True` through the entire search. The `False` is produced solely by the
**feasible-exit reset** (`_solve_nlp_bb`, `solver.py:10919-10921`): the search does not close
the gap within the budget, so optimality is not proven, so `gap_certified=False`. That is
sound-correct — an open gap cannot be certified optimal.

The premise "certification is being dropped at the node level" is therefore **incorrect** for
this family. Nothing is dropped at the node level; the guards the issue names never fire.

## Evidence (instrumented repro, `rsyn0805m`, 60 s, `JAX_ENABLE_X64=1`, cpu)

Instrumented three points on the running solver: (a) `pounce.solve_nlp_batch` status/iter
distribution, (b) `_solve_batch_pounce`'s `trusted` vector, (c) `_gap_certified` at the
`_solve_nlp_bb` status decision. Result:

```
status: feasible   obj: 1116.458   bound: 1690.717   gap_certified: False   nlp_bb: True
POUNCE batch node NLP status:  SolveStatus.OPTIMAL: 206   (min iter 19, max 37, mean 26)
serial node NLP status:        SolveStatus.OPTIMAL: 2, SolveStatus.TIME_LIMIT: 5
trust decisions:  batches=15  nodes=206  untrusted_nodes=0  batches_with_untrusted=0
[trace] _gap_certified(pre-status)=True  gap_converged=False  is_finished=False
        _unconverged_fathom=False        (every batch iteration)
```

- **`untrusted_nodes = 0`.** The `_batch_trusted` guard at `solver.py:~10206` never trips.
  Every batched node NLP returns `OPTIMAL` (KKT), so `np.all(_batch_trusted)` is `True`.
- **No `ITERATION_LIMIT`.** The serial convex `ITERATION_LIMIT` decert path
  (`_serial_nlp_trusted → False`, `solver.py:~7317`/`~7566`) never fires — no node returns
  `ITERATION_LIMIT`. (The polish-retry exists for it, but there is nothing to polish.)
- **`_gap_certified` is `True` right up to the status decision**, and
  `_unconverged_fathom=False`. The `False` appears only at `10919-10921`:

  ```python
  _tree_bound_valid = _gap_certified          # True here
  if status == "feasible":                    # search did not close (open ~46% gap, TL)
      _gap_certified = False                  # <-- the operative site
  ```

- **`gap_converged=False`, `is_finished=False`.** At 60 s the incumbent is 1116.5, the bound
  1690.7 (a ~46 % open gap) and the time limit is hit; `status="feasible"`. Optimality is
  simply not proven.

The five serial `TIME_LIMIT` nodes set an infeasibility sentinel but do **not** decertify:
`TIME_LIMIT` is not in the accepted-status set, so the `ITERATION_LIMIT` trust-lowering branch
never runs, `_serial_nlp_trusted` stays `True`, and `_unconverged_fathom` stays `False`
(confirmed in the trace). They are not the issue's mechanism either.

### Generalizes across the convex family (all 60 s)

| instance   | opt    | obj    | bound  | status   | untrusted_nodes | node NLP statuses          | gap_certified |
|------------|--------|--------|--------|----------|-----------------|----------------------------|---------------|
| `rsyn0805m`| 1296.1 | 1116.5 | 1690.7 | feasible | **0**           | 206 OPTIMAL (+5 serial TL) | False         |
| `rsyn0810m`| 1721.4 | 1548.6 | 2476.4 | feasible | **0**           | 110 OPTIMAL (+5 serial TL) | False         |
| `syn40m`   |   67.7 |   33.2 | 1228.7 | feasible | **0**           | 254 OPTIMAL (+7 serial TL) | False         |

Every reported `bound` is a valid upper fence for its maximize problem (`obj < opt < bound` on
all three) — the dual bound is rigorous and sound; it is simply **loose and the gap is open**.
No false certificate is possible: all exits are uncertified feasible.

## Why the conservative behavior is correct

`gap_certified` in this codebase means **"optimality is proven / the gap is closed"**, not
"the reported dual bound is valid". Those are two different things, and the code separates them
explicitly (`solver.py:10913-10921`, mirrored at `9353-9368`):

- `_tree_bound_valid` — the tree bound is *untainted*: no node was fathomed without a soundness
  proof, so the frontier minimum is a valid global dual bound. **This is `True` here.** It is
  surfaced as the `bound`/`gap` fields on the feasible exit (kept, not dropped — #138).
- `gap_certified` — the gap is *closed* (global optimality). **This is `False` here, correctly**,
  because the search hit the time limit with a wide-open gap.

Making `gap_certified=True` on these runs would require *closing the gap* (proving optimality),
which discopt cannot do on this family in 60 s — SCIP proves them optimal in 0.5–1.6 s while
discopt's root bound is 62.9 %–2609 % loose (#282 R2-2). It would be a **false certificate** to
flip the flag without closing the gap; CLAUDE.md §1 forbids that (a false `gap_certified=True`
is the worst-class regression). So the guard stays as-is.

## What the real lever is (and is not)

- **Not node-solve robustness.** The node NLPs are already fully robust on this family: 0/206
  untrusted, every one a clean KKT `OPTIMAL`. The issue's suggested first-steps (dump NLP
  status/KKT residual, tune tolerances/scaling/warm-start/bounds because nodes are hitting
  `ITERATION_LIMIT`) target a failure mode that **does not occur here**. Even if node trust were
  somehow improved, `gap_certified` would still be `False`, because the operative cause is the
  open gap, not node trust.
- **The lever is root/relaxation bound quality**, exactly the #282 R2-2 finding: the dual bound
  is far too loose to close the gap in budget. Only when the gap actually closes does
  `gap_certified` legitimately become `True`. This is a performance (dual-bound-strengthening)
  problem, tracked by #282, not a certification bug.

## Note on the "graduation baseline" concern

The issue notes that a future graduation panel requiring "no `gap_certified=True` instance
regresses to uncertified" is scoring against an already-uncertified baseline here. That is true
and *honest*: these instances are uncertified because discopt does not prove them optimal in
budget, not because a guard is over-triggering. They should be treated as uncertified baselines
until the #282 dual-bound work closes their gaps. No action needed on the certification layer.

## Recommendation

**Close #703** (or fold it into #282 as a note). There is no node-trust bug: the convex node
NLPs are all trusted, and `gap_certified=False` is the correct consequence of an unclosed gap.
The productive work — tighten the root/dual bound so these convex instances actually close — is
already scoped by #282 R2-2. Weakening the certification guard is explicitly rejected (it would
manufacture false certificates on unproven gaps).

## Repro

```bash
# Instance from the MINLPLib snapshot; oracle in minlplib.solu (=opt=).
JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python - <<'PY'
from discopt.modeling.core import from_nl
m = from_nl(".../minlplib/nl/rsyn0805m.nl")
r = m.solve(time_limit=60, gap_tolerance=1e-4)
print(r.status, r.objective, r.bound, r.gap_certified)   # feasible 1116.5 1690.7 False
PY
```

Node-trust instrumentation (wrap `pounce.solve_nlp_batch` for the status distribution and
`discopt.solver._solve_batch_pounce` for the `trusted` vector) reproduces
`untrusted_nodes=0`. Full instrumented script used for this diagnosis is transient; the
numbers above are the record.
