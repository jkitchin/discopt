# #2167 — complementarity formulation for ReLU networks: measurements

This is the record of the entry experiments and panels that decide each slice
of [#2167](https://github.com/jkitchin/discopt/issues/2167), per CLAUDE.md §4–§5.
Raw rows are in `docs/dev/data/2167/`; the scripts are
`discopt_benchmarks/scripts/relu_local_h1.py`, `relu_native_h2.py`,
`relu_h3_entry.py`, `mpec_native_panel.py`.

Nets are He-initialised random ReLU MLPs (`relu_local_h1.make_net`) on the
input box `[-1, 1]^n`, scalar output, objective `min output`. "physics" adds a
nonconvex ring constraint on the inputs and a quadratic cost.

## Status by slice

| slice | what | verdict |
|---|---|---|
| 1 | `relu_complementarity` strategy, `FROM_ACTIVATION` role | shipped; exact lowerings reproduce `relu_bigm` |
| 2 | multistart Scholtes local mode + active-set polish | **kept** — H1 passed both bars |
| 3 | native lowering + `DISCOPT_COMPL_BRANCH` | lowering **shipped**; flag **kept as documented opt-in** (H2, route isolation) |
| 4 | continuous-q Anderson cuts | **RETIRED** (H3 failed its kill criterion) |

## H3 — continuous-q Anderson cuts (slice 4): RETIRED

**Hypothesis.** Anderson et al.'s ideal inequalities, separated on a continuous
`q in [0, 1]` per unstable neuron, tighten the native relaxation beyond the
triangle. **Kill criterion** (from the issue): drop the option if it closes
less than 10% of the remaining gap on median.

**LP proxy (upper bound on what the hull can add).** `relu_extended`'s LP (the
per-neuron hull, which every Anderson inequality is valid for) against
`relu_bigm`'s LP (the triangle), 20 instances with a certified optimum: the hull
closes a median **32.7%** of the big-M LP gap (19.3%–69.9%). But the solver's
native root bound — which includes its root bound tightening — is already
tighter than the hull LP on **16 of 20** instances (median −23.5% measured
against the native root), and tighter than the big-M LP on 20 of 20. The proxy
could not decide H3, so the option was built and measured directly.

**Direct measurement.** Global dual bound after 100 nodes (`max_nodes=100`,
`deterministic=True`), cuts vs none, both arms on the Python tree
(`DISCOPT_COMPL_BRANCH=1`), same 20 instances. Gap closed =
`(bound_cuts − bound_plain) / (opt − bound_plain)`:

* **median −0.0%**, mean 1.7%, over 17 informative instances (3 close in both
  arms); 10 of 17 slightly negative; 1 of 17 ≥ 10% (4-16-16-1 seed 1, 28%);
* wall, cuts / plain: median **1.26×** (0.60×–1.60×);
* no bound above the optimum in either arm.

`root_bound` was not usable as the instrument: callback cuts reach the
relaxation only after the root LP is solved, so the root bound is identical
with and without them (4 instances, 4–19 cuts each). Recorded so the next
measurement does not repeat it.

**Verdict.** Fails the kill criterion with margin, and costs wall. Retired per
CLAUDE.md §5: the `continuous_q` option, its tie rows and the
`NNFormulation(..., strategy="relu_complementarity", ideal_cuts=True)` path are
deleted; `ideal_cuts=True` on that strategy is refused with a pointer here. Kept:
`relu_h3_entry.py` (the LP proxy) and both data files.

**Soundness note recorded with the retirement.** The first version used a FREE
`q`. That is unsound in the lifted space: an Anderson inequality is valid for
the projection onto `(x, y)` (some `q` exists for every feasible point), but a
lifted feasible point with an arbitrary `q` — `q = 0.3` on an active neuron — can
violate it. `callbacks.validate_cut_against_witnesses` refused exactly such a
cut. Tying `q` with the exact rows `q·s <= 0`, `(1−q)·y <= 0` made every cut
valid on every lifted feasible point (a test checked every cut emitted at 300
random points against 200 sampled feasible points); it is the tied version that
was measured above.

## Slice 3 — non-ReLU complementarity panel

The native branch applies to every pair lowered with `method="native"`, so the
panel must cover non-ReLU complementarity too. The in-repo `.nl` corpus has
none (all 76 files report zero complementarity constraints), so
`mpec_native_panel.py` builds the class: `distance`, `balanced`, MacMPEC
`bard1`, and random LCP-constrained LPs/QPs (`n = 3, 5, 8`, 3 seeds each) — 21
instances × {gdp, sos1, native flag OFF, native flag ON}, each arm in its own
subprocess, 60 s.

**255 executed checks, 0 failures**: no dual bound crosses any arm's
incumbent, certified objectives and certified infeasibilities agree across all
four arms, every incumbent's source complementarity residual ≤ 1e-6.

Where they differ, native is the stronger route: `lcp_qp` n=8 seed 0 — gdp and
sos1 hit the 60 s limit (14 495 / 14 911 nodes), both native arms prove
infeasibility in 3 nodes; `lcp_qp` n=5 seeds 0, 1 — gdp/sos1 2 047 nodes, native
1; `lcp_qp` n=8 seed 2 — native ON 3 nodes, native OFF 71, gdp 65, sos1 207;
`bard1` — native ON 5, gdp 11, sos1 29. Cost: on trivial instances native pays
~0.5–1 s of setup (0.4 s → 0.8–1.5 s), and on the infeasible ones 4–15 s vs
0.4 s.

**Known limitation.** `balanced` (`max x + y`, `x == y`, `x ⊥ y`, optimum 0):
both native arms return `feasible`, incumbent 7.8e-7, bound 2.0e-4 — sound but
not certified at the default absolute gap. Presolve on `x·y <= 0` with `x = y`
(`x² <= 0`) tightens to the tolerance-scale box `x <= sqrt(1e-8)`; with
`abs_gap_tolerance=1e-3` the same run certifies. gdp and sos1 certify 0.

## H1 — multistart Scholtes local mode (slice 2): KEPT

**Hypothesis.** Scholtes on POUNCE reaches near-optimal points on large ReLU nets
much faster than `relu_bigm` closes them. **Kill criterion** (from the issue):
retire if, on nets `relu_bigm` closes, the median verified objective is more than
1% worse; or if, on larger nets, it is not at least 5× faster to a point within
1% of the best known.

Per instance: `relu_bigm` with a 120 s limit, then `nn.solve_local(n_starts=20,
time_limit=120)`, interleaved. "first" = wall time to the first verified point;
"within 1%" = to a point within 1% of the best known (better of the two arms).
Measured after the residual-report fix (below); `docs/dev/data/2167/h1.jsonl`.

| net | relations | relu_bigm (120 s) | local best | first | within 1% |
|---|---|---|---|---|---|
| 4-16×2 s0 | 32 | optimal -1.0586 (3 s) | -1.0586 | 1.1 s | 3.0 s |
| 4-16×2 s1 | 32 | optimal -0.7928 (2 s) | -0.7928 | 1.0 s | 1.0 s |
| 4-32×2 s0 | 64 | feasible -0.3620 | -0.3620 | 2.0 s | 10.4 s |
| 4-32×2 s1 | 64 | optimal -1.0336 (112 s) | -1.0336 | 2.5 s | 2.5 s |
| 4-64×2 s0 | 128 | feasible -0.0043 | -2.2101 | 8.0 s | 14.7 s |
| 4-64×2 s1 | 128 | feasible 0.0000 | -0.9247 | 5.6 s | 5.6 s |
| 4-128×2 s0 | 256 | feasible -0.1965 | -3.9068 | 38.4 s | 106.2 s |
| 4-128×2 s1 | 256 | feasible -0.1531 | -1.0000 | 23.7 s | 90.4 s |
| 4-256×2 s0, s1 | 512 | no point | no point | — | — |
| 4-512×2 s0, s1 | 1024 | no point | no point | — | — |
| 4-1024×2 s0, s1 | 2048 | no point | no point | — | — |
| 4-32×3 s0 | 96 | feasible -0.1067 | -1.0010 | 2.9 s | 22.1 s |
| 4-32×3 s1 | 96 | feasible 0.0060 | -0.7881 | 4.0 s | 4.0 s |
| 4-64×4 s0 | 256 | feasible -0.0038 | -0.7946 | 22.3 s | 35.1 s |
| 4-64×4 s1 | 256 | feasible -0.1698 | -1.6179 | 18.5 s | 58.2 s |
| 4-128×4 s0 | 512 | no point | -0.6202 | 96.8 s | 96.8 s |
| 4-128×4 s1 | 512 | feasible 0.1031 | -0.2666 | 32.1 s | 102.3 s |
| 4-64×6 s0 | 384 | no point | -1.8651 | 30.7 s | 73.3 s |
| 4-64×6 s1 | 384 | no point | -1.0595 | 36.8 s | 66.8 s |
| 8-256×3 s0, s1 | 768 | no point | no point | — | — |
| 4-32×2 phys s0 | 64 | no point | -0.3275 | 1.6 s | 1.6 s |
| 4-32×2 phys s1 | 64 | no point | -0.8977 | 1.3 s | 1.3 s |
| 4-64×3 phys s0 | 192 | no point | 0.1528 | 3.6 s | 12.3 s |
| 4-64×3 phys s1 | 192 | no point | -2.0713 | 5.9 s | 17.4 s |
| 8-128×2 phys s0 | 256 | no point | -0.9475 | 19.1 s | 19.1 s |
| 8-128×2 phys s1 | 256 | no point | -0.9955 | 12.5 s | 35.5 s |

**Bar 1** (nets `relu_bigm` closes, 3): median gap of the local objective to the
certified optimum **6e-9** (kill at 1%). **Pass.**

**Bar 2** (larger nets where either arm found a point, 19): on 18 of 19
`relu_bigm`'s 120 s incumbent is not within 1% of the best known. The local mode
is **≥ 5× faster** on 9 by the 120 s budget alone (within-1% time ≤ 24 s; all 6
physics instances, where `relu_bigm` finds no point at all, are among them). For
the remaining ones `relu_bigm` was re-run with a budget of 5× the local mode's
within-1% time (`h1_bigm_5x.jsonl`): on all 4 that ran — 4-32×2 s0 (52 s),
4-128×2 s0 (531 s), 4-128×2 s1 (452 s), 4-64×4 s0 (176 s) — it still did not
reach within 1% (on the 128-wide nets it stayed at -0.197 / -0.153 against
-3.907 / -1.000). That makes **13 of 18** proven ≥ 5×, so the median is; the
other 5 re-runs were stopped as unable to change the verdict. **Pass.**

**Stated limits.** On the 8 instances with layers ≥ 256 wide, one Scholtes start
does not finish inside 120 s, so the local mode returns `local_infeasible` (no
verified point) — as does `relu_bigm`. On the 1024-wide nets the run still
overshoots its budget (≈ 200 s on 120 s): the remaining time is one-off setup
(lowering, evaluator construction) outside the NLP solves the cap applies to.

Three measurements taken while building the mode, recorded because they changed
the code:

* `t0 = 1` (the single-start default) collapses every start onto one local point
  (10 forward-pass starts on 4-32-32-1 s0 → all -0.1746); `t0 = 1e-2` / `1e-4`
  keep them apart and reach -0.3620. Multistart defaults to `1e-4`.
* `time_limit` was checked only between starts; one start on 4-256-256-1 took
  367 s against a 120 s budget. Each stage and polish NLP is now capped.
* The source-residual report rebuilt the point's interval box for every operand
  (quadratic in model size): 42 s of a 68 s solve on 4-256-256-1, and the cause
  of 572-638 s runs on the 1024-wide nets. Building it once per call is
  bit-identical (1536 residuals, 3 points) and 31× faster (8.83 s → 0.28 s).
  The first H1 run (widths 16-1024) predates this fix and was discarded.

## H2 — native lowering and `DISCOPT_COMPL_BRANCH` (slice 3)

**Hypothesis.** The native branch matches `relu_bigm` node counts within 1.5×
and wins where big-M is weak or infinite. **Kill criterion**: retire if median
nodes or wall are worse by more than 1.5× and no instance class improves.

Panel: 12 nets × 2 seeds in three classes — `box` (input box [-1, 1]), `loose`
([-L, L], L = 10 or 30, weak big-M), `physics` — × {`relu_bigm`, native flag
OFF, native flag ON}, each arm in its own subprocess, 120 s
(`docs/dev/data/2167/h2.jsonl`). **216 executed checks, 0 failures; all 72
solves certified optimal**, the three arms agreeing on every instance.

| class | n | nodes, median (bigm / OFF / ON) | wall, median (bigm / OFF / ON) |
|---|---|---|---|
| box | 10 | 13 / 244 / 3 | 0.6 s / 14.4 s / 18.9 s |
| loose | 8 | 14 / 314 / 4 | 0.6 s / 12.7 s / 14.5 s |
| physics | 6 | 34 / 228 / 5 | 12.0 s / 16.1 s / 19.2 s |

Native ON against `relu_bigm`: nodes **0.53×** median (passes the 1.5× bar), wall
**22.9×** median on box/loose, 1.6× on physics (fails it). Classes that improve:
networks with no declared input box, which `relu_bigm` refuses outright and the
native lowering certifies (`test_compl_branch_certifies_without_input_bounds`),
and the non-ReLU LCP-constrained programs above, where native proves results in
1-3 nodes that gdp/sos1 need 2 047 nodes or more than 60 s for. **Verdict: not
retired.** The native lowering stays as an explicit `solve_mpec(method=
"native")`; `relu_bigm` stays the ReLU default (the issue's non-goal, now
answered by measurement: on box-bounded ReLU nets native is ~23× slower).

### Route isolation, and a retraction

The ON/OFF columns above differ in two ways: the branching rule, and the tree —
ON bypasses the native Rust spatial kernel, which has no such branch, so OFF ran
on the kernel and ON on the Python tree. The first reading of H2 ("the flag cuts
nodes 30-45×") attributed the whole node reduction to the branch. **That is
retracted** (CLAUDE.md §11): rerun with both arms on the Python tree
(`DISCOPT_NATIVE_SPATIAL_KERNEL=0` for OFF;
`discopt_benchmarks/scripts/compl_branch_isolation.py`,
`compl_branch_isolation.jsonl`), over the 24 H2 and 21 non-ReLU instances:

* 38 paired certified results, **0 disagreements**; both arms certify 44 of 45;
* nodes **equal on 34**, **fewer with ON on 10** (2-7× on the deeper trees:
  93 → 13, 111 → 45, 101 → 15, 77 → 39), **more on 1** (3 → 5).

So the 30-45× is the route: on natively lowered models the Python tree needs a
handful of nodes where the kernel needs hundreds, while still being faster in
wall per solve. The branch itself is sound and modestly helpful on deep trees.

**Flag verdict: kept as a documented opt-in** (CLAUDE.md §5, third outcome).
Cert-clean and node-positive on the same tree, but turning it on changes the
route, and the route costs +27-34% median wall on ReLU nets (0.70× on the
non-ReLU panel). What would change that: the dichotomy implemented in
`bnb/spatial_tree.rs`, so the flag no longer moves the solve; then re-run the
isolation script and H2. The flag's docstring and the audit row say so.
