# #1004 — the GDP constructor's per-candidate feasibility test: measurement record

**Issue:** #1004, "GDP primal: the constructor's per-candidate feasibility test is
single-start and rejects ~80 % of genuinely feasible configurations".
**Verdict:** the issue's premise is **falsified** and its escape hatch is
**closed**. The constructor's per-candidate feasibility test detects 192 of 193
feasible configurations (99 %) across the `gdplib_small` corpus — not ~20 % — and
a restart on an already-built fixed-integer sub-NLP costs what the first solve
costs (ratio 0.95–1.03), where breaking even at the measured detection rate would
require a restart to cost under 0.52 % of it. **No multistart is built.** The
single-start design stands, now pinned by regression tests and with the
measurement recorded so the question is not reopened from the same reasoning.
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

### E1 results (run 2026-08-13, `scratchpad/issue1004/E1.log`, `E1_results.json`)

All twelve `gdplib_small` models, 45 s dive per model, 8 random starts per
configuration, **6,690 executed feasibility tests**. Every per-model row
reproduced an earlier identical-seed pass exactly.

**Unbiased pool** — configurations sampled or one-flip-perturbed, never sourced
from the dive:

| model | cfgs | feasible | zero | relax | random |
|---|---:|---:|---:|---:|---:|
| jobshop | 2 | 0 | — | — | — |
| ex1_linan_2023 | 63 | 2 | 2/2 100 % | 2/2 100 % | 16/16 100 % |
| positioning | 63 | 63 | 63/63 100 % | 63/63 100 % | 504/504 100 % |
| small_batch | 58 | 0 | — | — | — |
| cstr | 61 | 0 | — | — | — |
| spectralog | 52 | 52 | 52/52 100 % | 52/52 100 % | 416/416 100 % |
| methanol | 4 | 4 | 4/4 100 % | 4/4 100 % | 31/32 97 % |
| batch_processing | 69 | 64 | **64/64 100 %** | 64/64 100 % | **0/512 0 %** |
| syngas | 60 | 0 | — | — | — |
| water_network | 26 | 7 | 6/7 86 % | 0/7 0 % | 6/56 11 % |
| gdp_col | 58 | 0 | — | — | — |
| modprodnet | 1 | 1 | 1/1 100 % | 1/1 100 % | 8/8 100 % |
| **TOTAL** | **517** | **193** | **192/193 99 %** | 186/193 96 % | 981/1544 64 % |

**The number that answers the issue.** Restricted to the 189 configurations
proven feasible by a start *other than* the constructor's — so the constructor is
scored on a set it played no part in selecting — the constructor's start detects
**188 of 189 (99 %)**. Four further configurations were found *only* by the
constructor's start; one was missed by it.

The dive-derived pool (152 configurations, reported separately because the dive's
acceptance tries the zero start first) agrees: zero 150/152, relax 149/152, random
1086/1216.

**So the issue's ~20 % is a property of the sampler, not the constructor.**
`batch_processing` is the cleanest demonstration: 64 configurations that the
constructor's start solves every time, and 512 stratified random starts on those
same 64 configurations that solve **none**. A probe that samples starts on this
model would report 0 % detection and conclude the constructor rejects
*everything*; the constructor in fact rejects nothing.

**What this does not claim.** Neither start dominates. `water_network` is the one
model where the constructor's start misses a feasible configuration (6/7), and
`gdp_col`'s dive-derived pool is the reverse case — the constructor's start gets
3/5 there while the relaxation-point start gets 5/5. Nothing here says
`zero_start` is optimal; it says it is ~99 % on this corpus, which is the number
the issue's arithmetic needs and is nowhere near 20 %.

**Coverage stated rather than glossed.** Five models (`jobshop`, `small_batch`,
`cstr`, `syngas`, `gdp_col`) contributed **no** arm comparison: their unbiased
pools held zero feasible configurations, which is the rare-event regime #993
measured directly (1 feasible configuration in 2600 at distance 3 on `syngas`).
`syngas` — the issue's own model — therefore has no measurement here: its 45 s
dive returned nothing and none of its 60 sampled configurations was feasible. The
one in-tree datum on it remains #993's, and it points the same way: with the
integers pinned to BARON's proven-optimal configuration the sub-NLP solves in
0.36 s **from the constructor's own zero start** (and `batch_processing` in
0.07 s), which is arm Z detecting on the first try the very configuration B1
pinned.

### E1 deep pass — the models whose first pool came up empty

`jobshop`, `small_batch` and `cstr` contributed no comparison above because their
60-odd sampled configurations held no feasible one. Re-run with the dive disabled
entirely and a 600-configuration request, so **every** configuration in the pool
comes from a channel that never consults the zero start
(`E1_deep.log`, **5,490 executed tests**):

| model | cfgs | feasible | zero | relax | random |
|---|---:|---:|---:|---:|---:|
| jobshop | 8 | 6 | 6/6 100 % | 6/6 100 % | 24/24 100 % |
| small_batch | 460 | 2 | 2/2 100 % | 2/2 100 % | 8/8 100 % |
| cstr | 447 | 1 | 1/1 100 % | 1/1 100 % | 1/4 25 % |
| **TOTAL** | **915** | **9** | **9/9 100 %** | 9/9 100 % | 33/36 92 % |

Nine more feasible configurations, none of them discovered with the constructor's
start, and the constructor's start solves **all nine**. (`jobshop` has only 3
two-way disjunctions, so its pool saturates at 8 distinct configurations however
many are requested.) It also quantifies how rare feasibility is on this class:
2 in 460 on `small_batch`, 1 in 447 on `cstr` — the rare-event regime the issue
described, and the reason `syngas` and `gdp_col` stay unmeasured here.

Across both passes: **12,180 executed feasibility tests**, constructor-start
detection **201/202**.

## E2 — the escape hatch: is a restart cheaper than the first solve?

Instrument: `scratchpad/issue1004/E2_restart_cost.py`. On a configuration the
dive proved feasible, the integers are pinned and `--starts` sub-NLP solves run
back to back, each timed individually; `t1` is compared against
`mean(t2..tk)` with a standard deviation, over `--reps` interleaved repetitions
so machine drift lands on both arms equally (§9). A control run
(`--same-start`) repeats the *same* start, which separates caching in the
evaluator or the backend from genuine start-dependence.

What the code says before the clock does: `subnlp` takes the evaluator from its
caller (`cached_evaluator(model)`, keyed on the model's structural fingerprint and
shared across every plan and every configuration; bounds are read live), saves and
restores the integer bounds around one backend call, and keeps nothing between
calls. There is no retained factorization and no per-configuration structure to
reuse, so a restart on the same fixed-integer sub-problem is expected to cost what
the first solve cost. The measurement is what decides it.

### E2 results (2026-08-13, 3 configurations × 5 starts × 3 interleaved reps)

**Control — the same start repeated** (`E2_samestart.log`, 210 solves). This is
the arm that answers the question, because it varies nothing but *repetition*:

| model | start 1 (ms) | starts 2..5 (ms) | ratio |
|---|---:|---:|---:|
| small_batch | 20.7 ± 1.3 | 20.0 ± 1.0 | 0.967 |
| cstr | 41.1 ± 7.3 | 39.3 ± 4.1 | 0.954 |
| spectralog | 29.8 ± 3.1 | 30.6 ± 2.3 | 1.026 |
| batch_processing | 260.2 ± 13.0 | 262.8 ± 13.7 | 1.010 |
| gdp_col | 1354.1 ± 404.0 | 1360.6 ± 404.1 | 1.005 |

**A restart costs what the first solve costs** — every ratio within ±5 % of 1.0,
well inside the per-model spread. There is no warm-start discount to spend.

**Different starts** (`E2.log`, 210 solves), for completeness and because one row
of it is a trap:

| model | start 1 (ms) | starts 2..5 (ms) | ratio |
|---|---:|---:|---:|
| small_batch | 21.3 ± 2.2 | 18.1 ± 2.3 | 0.848 |
| cstr | 47.2 ± 16.0 | 156.1 ± 83.2 | 3.308 |
| spectralog | 29.4 ± 2.0 | 52.0 ± 3.4 | 1.767 |
| batch_processing | 271.2 ± 15.6 | 4.9 ± 0.3 | **0.018** |
| gdp_col | 1588.1 ± 913.4 | 1789.2 ± 322.8 | 1.127 |

`batch_processing`'s 0.018 looks like the escape hatch and is its opposite. Those
random starts are the same ones E1 measured at **0/512** detection on this model:
a 4.9 ms solve there is an *immediate rejection*, not a cheap restart. The control
row for the same model is 1.010. A ratio below 1 in this arm tracks how fast a
start fails, and the reason to run the control was precisely that a fast failure
and a cheap restart are indistinguishable without it. (`syngas` is absent from
both arms: the dive found no feasible configuration in 45 s, so there was nothing
to pin — reported rather than silently dropped.)

### What the escape hatch would have needed

Re-run the issue's budget arithmetic with the *measured* per-start detection
`p = 192/193 = 0.995` and a restart cost ratio `ρ`. Under budget `B`, testing `k`
starts per candidate covers `B/(1+(k−1)ρ)` candidates, so the yield ratio against
single-start is `(1−(1−p)^k) / [(1+(k−1)ρ)·p]`. Break-even requires

| k | maximum restart cost that breaks even |
|---|---|
| 2 | **0.52 %** of the first solve |
| 3 | 0.26 % |
| 5 | 0.13 % |

At the measured `ρ ≈ 1.0`, `k = 2` yields **0.50×** of single-start — it halves
the answer rate. Even a hypothetical restart costing 10 % of the first solve still
loses (0.91×). The escape hatch is not narrowly missed; at `p ≈ 0.995` there is
almost no headroom for a second start to buy, because the first one already
succeeds.

## Verdict and what ships

**Nothing is built.** Both entry experiments came back negative, in the direction
that says the current design is already right:

1. The detection rate the issue attributes to the constructor belongs to the
   *start samplers* its two probes used. The constructor's own start is
   deterministic, relaxation-point independent, and detects **99 %** of the
   feasible configurations on this corpus — including 188 of the 189 that some
   other start proved feasible, so the number is not an artifact of the
   constructor grading its own homework.
2. The escape hatch is closed twice over: restarts are **not** cheaper (ratio
   0.95–1.03 on the same-start control), and at `p ≈ 0.995` they would have had
   to be nearly free (< 0.52 % of a first solve) to break even even if they were.

What ships instead is the thing that keeps this from being re-litigated from the
same reasoning: `python/tests/test_issue1004_gdp_config_start.py` pins the two
properties the argument rests on — the per-candidate start does not depend on the
relaxation point, and each candidate configuration is tested exactly once, so the
wave's `_WAVE_SOLVE_CAP` stays a *candidate* cap rather than a start cap. Both
were mutation-checked against the specific regressions they exist to catch.

### Retraction (CLAUDE.md §11)

#1004's headline — "rejects ~80 % of genuinely feasible configurations" — is
withdrawn. The underlying B1 and C2 measurements stand as measurements; the
inference from them to the constructor does not, because neither start family
contains the constructor's start. The issue's own budget argument was correct and
is now stronger: at the measured `p`, multistart-per-candidate is worse than the
2/3-of-single-start the issue estimated at `p = 0.2, k = 5` — it is about half.

### Two things noticed in passing, neither chased (CLAUDE.md §3 scope)

* **Neither start dominates.** `water_network` costs the zero start one
  configuration of seven; `gdp_col`'s dive-derived pool costs it two of five while
  the relaxation-point start gets all five. The union would detect more, at
  exactly the cost the budget argument rejects. Recorded, not acted on.
* **`cached_evaluator` on `modprodnet` overflows the default recursion limit.**
  `_nl_expr_compiler._lower` recurses per expression node, and this model's DAG
  exceeds 1000 frames; raising the limit alone segfaults, so the probe runs on a
  thread with a 512 MB stack. The solver reaches `modprodnet` through a different
  path and solves it (`optimal`, 5 nodes — #993's panel), so this is a property of
  building the evaluator directly, not a live solve failure. Out of scope here.

## Artifacts

| file | what |
|---|---|
| `scratchpad/issue1004/E1_detection_rate.py` | E1 probe (arms Z / X / R, arm-symmetric oracle) |
| `scratchpad/issue1004/E2_restart_cost.py` | E2 probe (interleaved timing, `--same-start` control) |
| `scratchpad/issue1004/gdp_loader.py` | GDPlib → `gdp.bigm` → `.nl` → `from_nl` loader |
| `scratchpad/issue1004/RESULTS.md` | every summary table as printed (the `.log` files are gitignored by `scratchpad/**/*.log`) |
| `scratchpad/issue1004/E1_results.json` | E1 panel, per-configuration, 6,690 executed tests |
| `scratchpad/issue1004/E1_deep_results.json` | E1 deep pool on the models whose first pool was empty, 5,490 tests |
| `scratchpad/issue1004/E2_results.json` | E2 different-starts arm, 210 solves |
| `scratchpad/issue1004/E2_samestart_results.json` | E2 same-start control, 210 solves |
| `scratchpad/issue1004/run_rest.sh` | serial driver for the E2 arms and the deep pool |
| `python/tests/test_issue1004_gdp_config_start.py` | the three regression pins |

Every probe prints an executed-test count and exits non-zero at zero (§6); no
probe swallows an exception (§7); the timing arms record `os.getloadavg()` before
and after, interleave repetitions, and report a standard deviation (§9).
