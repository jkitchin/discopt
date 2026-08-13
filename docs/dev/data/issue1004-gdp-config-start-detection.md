# #1004 — the GDP constructor's per-candidate feasibility test: measurement record

**Issue:** #1004, "GDP primal: the constructor's per-candidate feasibility test is
single-start and rejects ~80 % of genuinely feasible configurations".
**Verdict:** _(filled from the panel below)_
**Regime:** heuristic-policy / purely primal. Nothing here touches a relaxation, a
dual bound, or a certificate; every point involved is one `subnlp` re-verified
integer- and constraint-feasible before it can become an incumbent.

## What the issue claimed, and what it asked for

With the integers pinned to a configuration known to be feasible on `syngas`,
two probes measured that most starts failed to produce a feasible point — 12 of
67 (B1) and 2 of 6 (#993 C2) — and the issue inferred that
`one_hot_config_subnlp`, which spends **one** sub-NLP solve per candidate
configuration, "discards a genuinely feasible candidate most of the time".

The issue then argued against the obvious fix on its own: under a fixed sub-NLP
budget `B`, per-start detection `p` and feasible fraction `f`, `B` configurations
× 1 start expects `B·f·p` finds while `B/k` configurations × `k` starts expects
`(B/k)·f·(1−(1−p)^k)`; the ratio `(1−(1−p)^k)/(k·p)` is `≤ 1` for every `k ≥ 1`,
so multistart per candidate is **dominated** by testing more candidates. It named
one escape hatch — if restarts on an already-built sub-problem are materially
cheaper than the first solve, the arithmetic changes sign — and two entry
experiments to run *before* building anything: (E1) repeat the known-feasible pin
across GDPlib models and report per-model detection, (E2) time the 1st vs 2..k-th
start on the same fixed-integer sub-NLP.

Both were run. Neither supports building anything.

## The structural fact the issue's inference missed

`one_hot_config_subnlp` does not draw its start from a family. For a candidate
configuration it builds

```
zero_start = x_relax.copy()
zero_start[continuous] = clip(0, lb, ub)[continuous]     # every continuous slot
zero_start[group members] = the plan's disjunct selection # every one-hot binary
zero_start[residual binaries] = the plan's residual assignment
```

so on a big-M GDP the start is a function of **the model and the candidate
configuration alone**. `x_relax` survives only in *general* integer slots outside
every one-hot row and outside the 0/1 residual set — none of the eleven measured
models has one except `modprodnet`.

A detection rate measured by sampling starts therefore measures the *sampler*, not
the constructor. That is the whole of the issue's arithmetic error: B1's 12/67 and
C2's 2/6 are the detection rate of the start families those probes drew from, and
neither family contains the start the constructor actually uses.

Pinned by `python/tests/test_issue1004_gdp_config_start.py`:
`test_per_candidate_start_is_independent_of_the_relaxation_point` drives the wave
with two relaxation points that disagree on every slot and asserts the continuous
half of every seed is `clip(0, lb, ub)`; two further tests assert each candidate
configuration is tested exactly once, so the budget stays denominated in
candidates rather than in starts. Both were mutation-checked: removing the
zero-continuous assignment fails the first, adding a second start per candidate
fails the other two.

## E1 — per-model detection, `gdplib_small`

Instrument: `scratchpad/issue1004/E1_detection_rate.py` (loader:
`gdp_loader.py`, the same Pyomo → `gdp.bigm` → `.nl` → `from_nl` pipeline
`benchmarks.gdplib_runner` uses). Three arms per configuration:

* **Z** — the constructor's exact zero-continuous start (1 solve);
* **X** — the relaxation point's continuous slots, i.e. the second base seed
  `enumerate_binary_seeds_subnlp` uses (1 solve);
* **R** — 8 stratified random starts over the box (`_generate_starts`, the
  sampler `continuous_multistart` draws from).

A configuration counts as **genuinely feasible** when *any* arm returns a point
that `subnlp` verified integer- and constraint-feasible. The oracle is the union
of the arms, so it is symmetric and no arm is scored on a set it selected.

**The bias trap this probe had to be rebuilt around.** The first version sourced
its known-feasible configurations from `one_hot_config_dive`. The dive accepts a
completed configuration by trying **the zero-continuous start first** and only
then its own point, so a dive-derived witness set is enriched with exactly the
configurations arm Z can already solve — the arm under test would have been
scored on a set it had selected. The pool is therefore split: `random` (uniformly
sampled valid configurations) and `neighbour` (one group re-pointed to a different
disjunct) are the **unbiased** pool and carry the verdict; the dive-derived
configurations are reported separately and never pooled with them.

_(results below)_

## E2 — the escape hatch: is a restart cheaper than the first solve?

Instrument: `scratchpad/issue1004/E2_restart_cost.py`. On a configuration the
dive proved feasible, the integers are pinned and `--starts` sub-NLP solves run
back to back, each timed individually; `t1` is compared against
`mean(t2..tk)` with a standard deviation, over `--reps` interleaved repetitions
so machine drift lands on both arms equally (§9). A control run
(`--same-start`) repeats the *same* start, which separates caching in the
evaluator or the backend from genuine start-dependence.

What the code says before the clock does: `subnlp` takes the evaluator from its
caller (`cached_evaluator(model)`, shared across every plan and every
configuration), saves and restores the integer bounds around one backend call,
and keeps nothing between calls. There is no retained factorization and no
per-configuration structure to reuse, so a restart on the same fixed-integer
sub-problem is expected to cost what the first solve cost. The measurement is
what decides it.

_(results below)_
