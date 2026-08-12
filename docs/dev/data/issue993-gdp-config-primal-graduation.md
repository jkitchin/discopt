# #993 — DISCOPT_GDP_CONFIG_PRIMAL graduation record

**Flag:** `DISCOPT_GDP_CONFIG_PRIMAL` — the disjunct-configuration primal for
one-hot (`sum_k y_k == 1`) structures: the #823 Hamming-**wave** constructor
(`one_hot_config_subnlp`) plus the #993 **dive** constructor
(`one_hot_config_dive`) it chains to. Default-OFF at merge in #823.
**Decision:** _(filled from the panel below)_
**Regime:** heuristic-policy / purely primal. Both constructors only ever
*propose* a point; every proposal is re-verified integer- and
constraint-feasible by `subnlp` / `_check_constraint_feasibility` before it can
become an incumbent, and neither touches a relaxation, a dual bound, or a
certificate. A local "infeasible" verdict on a nonconvex partial fixing
redirects the heuristic's own search and is never a pruning decision.

## What the flag gates, at graduation

`_gdp_config_primal_enabled()` in `python/discopt/solver.py` guards three root
call sites (≈1038, ≈11624, ≈14204), all of which enter through
`one_hot_config_subnlp`. Since #993 that entry point is a two-stage ladder:

1. **Wave (#823)** — per-group argmax of the relaxation, then Hamming-distance
   waves of demotions crossed with `_residual_assignments`, capped at
   `max_configs=256`. It never re-solves the relaxation, so every plan is scored
   against the *same* root point.
2. **Dive (#993)** — reached only when the wave returns nothing. It re-solves the
   relaxation between disjunct choices, so each successive choice is made at a
   point that already respects the previous ones. Adds a learned decision order
   carried across restarts, GRASP randomization that ramps with restart count,
   and local repair (release the last *k* decisions, `k` doubling 3→6→12) before
   a full restart.

The two stages **split the budget by a solve count**: the wave hands over after
`_WAVE_SOLVE_CAP = 48` sub-NLP solves and the dive spends what the caller's
deadline leaves. This is not cosmetic — see "Falsification during construction"
below.

The split was originally a wall-clock one (half of what remained), which #912
rejects and CI caught on PR #1000: a clock may decide *when to stop*, never *how
much work*, so a wall-sized wave made the number of configurations tried — and
from there the incumbent and the whole subtree — a function of machine speed (28
plans on a slow clock against 256 on a fast one). `_WAVE_SOLVE_CAP` is calibrated
from entry experiment E1/E2: cstr's feasible plans are 8, 42, 47, 57 and its
*best* is plan 42, while batch_processing and syngas produce nothing in all 256,
so the cap must sit just above 42. At the production grant the conversion is
answer-for-answer identical (cstr best 3.0620146, batch_processing 822533.66,
syngas nothing) and 5.7× cheaper on cstr.

Budget: `_gdp_config_deadline` grants
`min(_GDP_CONFIG_BUDGET_FRACTION × remaining, _GDP_CONFIG_BUDGET_CAP_S)` =
`min(0.15 × remaining, 15 s)`, never past the caller's own deadline. A root
constructor must be cheap when it fails: handed the *whole* remaining budget it
cost `batch_processing` 71 % of its nodes (307 → 89) while finding nothing on it
(pinned by `test_constructor_gets_a_bounded_share_of_the_budget`).

## Falsification during construction (CLAUDE.md §4/§11)

Two things this session measured that contradicted the plan, recorded before the
panel ran:

* **Both of #993's stated premises were wrong.** The issue framed
  `batch_processing` as a *start-point* problem and `syngas` as a *continuous
  subproblem* problem. Neither survived: the root cause in both is configuration
  **reachability** — the wave scores every plan at one fixed relaxation point, so
  a configuration many demotions away from the argmax is not in its reach at any
  budget. Retracted on the issue (comment 5268543715) before implementation.
* **The dive got zero relaxation solves in its first production wiring.** At the
  9 s budget a 60 s time limit grants, the wave consumed the *entire* budget
  (`batch_processing` reached 66 of 256 plans and stopped on "deadline";
  `syngas` 53 of 256), leaving the dive nothing. Measured, not assumed — the
  chain would have been a dead flag. Fixed by reserving half the budget for the
  dive, justified by `cstr` exhausting all 256 wave plans in 4.6 s of its 15 s.

## Panel design

Instrument: `scratchpad/issue993/panel.py` (OFF/ON arms, one subprocess per
(model, arm) because the flag is read from the environment at solve time);
scored by `scratchpad/issue993/score_panel.py`. Both print an
executed-comparison / executed-check count and exit non-zero when it is zero
(§6). Arm order alternates by model index so a warm-cache or load-drift
advantage cannot accrue systematically to one arm (§9). Machine load is
snapshotted into `panel.log` before the run.

Corpus: the whole in-repo GDP corpus at ≤500 variables — the 12-model
`gdplib_small` suite — big-M reformulated, 120 s per solve, oracle enabled.
`disease_model` is discovered and under the ceiling but is *not* in the panel: it
has no verified reference optimum, so the runner exits 3 (vacuous sweep) on both
arms and the pair yields no comparison. This is the right corpus rather than a subset of it: the flag
only fires where `_scan_one_hot_rows` finds one-hot groups, which is exactly the
GDP class. On a model without that structure both constructors return `[]` after
one scan, pinned by `test_dive_is_a_noop_without_one_hot_structure`.

Note on reading node counts: at a *fixed wall-clock limit*, fewer nodes ON is a
**cost**, not a benefit — it means the constructor spent time the tree did not
get. Only an incumbent gained or improved counts on the credit side.

### Bar 1 — cert-clean

* runner `INCORRECT=0` and `bound-crossings=0` on every arm (the runner exits 1
  on a soundness flag, 3 on a vacuous sweep — both hard stops);
* no incumbent better than its reference optimum in either arm (a point below
  the true optimum *is* an infeasible point);
* no dual bound past its reference optimum;
* no certification regression: a model certified `optimal` OFF must not lose it.

### Bar 2 — net-positive

For a primal **constructor** the metric is incumbent presence and quality:
incumbents gained (none OFF → one ON), improved, worsened, or lost, with node
deltas reported alongside per the caveat above. Sound-but-unhelpful is not
enough — the `DISCOPT_CUT_INHERIT` lesson.

## Results

Run 2026-08-12 12:09–12:59 EDT, `scratchpad/issue993/panel_cap.log`, arms under
`scratchpad/issue993/panel/`, scored by `score_panel.py` (72 executed checks, 12
executed comparisons). Load before launch: 1-minute average 2.36, nothing above
40 % CPU. Every arm ran against the shipping code: `panel.py` asserts
`discopt.__file__` and the `_WAVE_SOLVE_CAP` marker before it starts (§8).

**The panel was re-run.** A first attempt on the same corpus measured the
wall-clock wave/dive split that #912 rejected; it was killed rather than scored,
because a panel that characterizes a mechanism which will not ship is worse than
no panel. It is kept at `scratchpad/issue993/panel_wallsplit_stale/` with a
`WHY_STALE.txt`. These results are entirely from the second run.

| model | arm | status | objective | bound | nodes | wall |
|---|---|---|---|---|---|---|
| batch_processing | off | time_limit | — | 583983.04 | 593 | 123.0 |
| batch_processing | **on** | **feasible** | **822533.66** | 589989.66 | 459 | 123.2 |
| cstr | off | time_limit | — | 0.11262613 | 19996 | 120.0 |
| cstr | **on** | time_limit | **3.0620146** | 0.050690227 | 22760 | 120.0 |
| ex1_linan_2023 | off | optimal | -0.9996 | -0.9996 | 199 | 2.4 |
| ex1_linan_2023 | on | optimal | -0.9996 | -0.9996 | 199 | 2.8 |
| gdp_col | off | feasible | 20100.296 | 8000 | 37 | 126.8 |
| gdp_col | **on** | feasible | **20916.392** | 8000 | 23 | 121.1 |
| jobshop | off | optimal | 11 | 11 | 7 | 0.1 |
| jobshop | on | optimal | 11 | 11 | 7 | 0.1 |
| methanol | off | feasible | -1793.4292 | -4649.5731 | 305 | 123.3 |
| methanol | on | feasible | -1793.4292 | -4649.5731 | 305 | 123.3 |
| modprodnet | off | optimal | 3592.9243 | 3592.9253 | 5 | 7.3 |
| modprodnet | on | optimal | 3592.9243 | 3592.9253 | 5 | 7.3 |
| positioning | off | optimal | -8.0641361 | -8.0641364 | 409 | 3.1 |
| positioning | **on** | optimal | -8.0641361 | -8.0641364 | **307** | 3.1 |
| small_batch | off | feasible | 167427.66 | 160860.75 | 181 | 7.2 |
| small_batch | **on** | **optimal** | 167427.66 | **167427.65** | 181 | 14.9 |
| spectralog | off | optimal | 12.089262 | 12.08906 | 459 | 40.4 |
| spectralog | on | optimal | 12.089262 | 12.08906 | 459 | 40.4 |
| syngas | off | time_limit | — | 2.059438 | 95 | 129.1 |
| syngas | on | time_limit | — | 2.059438 | 63 | 121.0 |
| water_network | off | feasible | 348337.04 | 295402.94 | 285 | 121.8 |
| water_network | on | feasible | 348337.02 | 295402.94 | 263 | 120.7 |

### Bar 1 — cert-clean: **PASS**

72 executed checks, zero violations. No incumbent beats its reference optimum in
either arm; no dual bound sits past one; no model certified `optimal` OFF lost
that ON. Every arm's runner summary reads `INCORRECT=0 | bound-crossings=0`.

Scope, stated rather than glossed: **10 of the 12 models were compared against an
oracle**. `gdp_col` and `methanol` have none available, so the runner exits 3 on
both of their arms and soundness there is *unestablished*, not clean —
`INCORRECT=0` with `oracle-checked=0` is no result at all. The verdict below
rests on the 10 that were verified.

### Bar 2 — net-positive: **PASS**

Credits:

* **`batch_processing`** — no incumbent at all after 120 s OFF; ON returns
  822533.66, and its dual bound moves 583983.04 → 589989.66 (still under the
  679365.33 reference). This is #993 item 1 landing on the panel.
* **`cstr`** — no incumbent OFF; ON returns **3.0620146**, the reference optimum
  3.0620073 to +0.0002 %. It also runs *more* nodes ON (19996 → 22760) despite
  paying the constructor's grant, so the incumbent is making nodes cheaper.
* **`small_batch`** — a **certificate gained**: `feasible` OFF (bound 160860.75)
  becomes `optimal` ON (bound 167427.65 against oracle 167428) at the *same* 181
  nodes, for ~7.7 s of extra wall. The bound moved toward the optimum and never
  past it.
* **`positioning`** — same certified optimum in 25 % fewer nodes (409 → 307) at
  equal wall (3.1 s). This model terminates far inside the limit, so its node
  drop is a real efficiency gain rather than the grant being charged.
* **`water_network`** — 348337.04 → 348337.02, marginal.

Debit:

* **`gdp_col`** — 20100.296 → 20916.392, +4.1 % worse. Both arms hit the limit;
  this model's nodes are so expensive that 120 s buys ~30 of them, so a 15 s
  grant is a large fraction of the whole search. It is one of the two models with
  no oracle. A time-race loss, not an unsound point.

Node deltas on the six models that *hit* the limit (`batch_processing` −23 %,
`syngas` −34 %, `gdp_col` −38 %, `water_network` −8 %, `methanol` 0 %, `cstr`
+14 %) are read as cost accounting per the caveat above, not as credit.

Four material gains — two of them an answer where 120 s of search produced none,
one a certificate, one a genuine node reduction — against a single time-race
regression on an unverified model. That is "measurably helpful broadly", not the
cert-clean-but-neutral shape the `DISCOPT_CUT_INHERIT` rule exists to reject.

### Engagement (the check that makes the above mean anything)

The two arms differ only by an environment variable in a subprocess, and the
runner prints nothing about the constructor. Had the variable failed to plumb
through, the ON arm would have been a second OFF arm, every model would have
scored "unchanged", and the honest reading of *that* panel would have been "not
net-positive" — a confident negative from a measurement that never happened.
`scratchpad/issue993/probe_engagement.py` closes it, with 3 executed checks:
`_gdp_config_primal_enabled()` is False in a subprocess built exactly the way
`panel.py` builds the OFF arm's environment and True for the ON arm's, and a real
`Model.solve()` on `cstr` with the flag ON enters `one_hot_config_subnlp` once.

## Verdict

**Graduate `DISCOPT_GDP_CONFIG_PRIMAL` to default ON**, keeping the `=0` opt-out
and the legacy path intact (CLAUDE.md §5). Both bars pass on one panel run, which
is what the 2026-07-17 policy requires; the nightly panel remains the ongoing
regression watch rather than a further gate.

Pinned by `test_flag_is_default_on_with_an_opt_out`, which asserts the new default
*and* that `0`/`off`/`false`/`no` still opt out. An **empty** value is deliberately
not among them: `export DISCOPT_GDP_CONFIG_PRIMAL="$UNSET"` exports an empty
string, and a graduated default-ON path must not be switched back to the legacy
one by an accident of shell quoting while reading, in every log and `env` dump, as
"not set, so the default applies". This matches `_qubo_primal_enabled` and every
other graduated flag in `solver.py`.

What this verdict does **not** claim: nothing here establishes soundness on
`gdp_col` or `methanol`, and `gdp_col` is measurably worse ON. If an oracle
becomes available for either, they should be re-scored before the next flag
decision that leans on this corpus.
