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
| 2 | multistart Scholtes local mode + active-set polish | H1 — see below |
| 3 | native lowering + `DISCOPT_COMPL_BRANCH` | H2 — see below; non-ReLU panel cert-clean |
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

## H1 — multistart Scholtes local mode (slice 2)

*In progress; widths 16–128 below, 256 and up (with the in-start time cap) and
the physics variant pending.*

Per instance: `relu_bigm` with a 120 s limit, then `nn.solve_local(n_starts=20,
time_limit=120)`. "first" = wall time to the first verified point; "within 1%"
= to a point within 1% of the best known (the better of the two arms).

| net | relations | relu_bigm (120 s) | local best | first | within 1% |
|---|---|---|---|---|---|
| 4-16-16-1 s0 | 32 | optimal −1.0586 (3 s) | −1.0586 | 1.2 s | 3.2 s |
| 4-16-16-1 s1 | 32 | optimal −0.7928 (2 s) | −0.7928 | 1.1 s | 1.1 s |
| 4-32-32-1 s0 | 64 | feasible −0.3620, bound −0.946 | −0.3620 | 2.8 s | 14.2 s |
| 4-32-32-1 s1 | 64 | optimal −1.0336 (106 s) | −1.0336 | 3.0 s | 3.0 s |
| 4-64-64-1 s0 | 128 | feasible −0.0043, bound −14.08 | −2.2101 | 9.7 s | 17.8 s |
| 4-64-64-1 s1 | 128 | feasible 0.0000, bound −7.83 | −0.9247 | 8.0 s | 8.0 s |
| 4-128-128-1 s0 | 256 | feasible −0.1965, bound −35.0 | −3.9068 | 47.8 s | 130.8 s |
| 4-128-128-1 s1 | 256 | feasible −0.1531, bound −26.8 | −1.0000 | 30.2 s | 107.3 s |

On the three nets `relu_bigm` closes, the local mode's verified objective
matches the certified optimum to 1e-6 (kill bar: more than 1% worse). On the
64-wide nets the local mode's *first* verified point (8–10 s) is far better than
`relu_bigm`'s 120 s incumbent, ≥ 12× faster to a point within 1%. On the
128-wide nets `relu_bigm` does not get within 1% in 120 s, so the 5× bar needs a
longer `relu_bigm` run (pending).

Two measurements taken while building the mode, recorded because they changed
the code:

* `t0 = 1` (the single-start default) collapses every start onto one local point
  (10 forward-pass starts on 4-32-32-1 s0 → all −0.1746); `t0 = 1e-2` / `1e-4`
  keep them apart and reach −0.3620. Multistart defaults to `1e-4`.
* `time_limit` was checked only between starts, and one start on 4-256-256-1
  took 367 s against a 120 s budget. Each stage and polish NLP is now capped by
  the remaining time.

## H2 — native branch vs relu_bigm (slice 3)

*Pending.*
