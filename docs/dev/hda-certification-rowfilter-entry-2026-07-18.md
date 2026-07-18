# hda certification entry experiment — log-lift falsified on coverage; float64-intractable-row filter CONFIRMED

Follow-up to the #671 tight-bound work (PR #746). Question: what lets hda's node
LPs solve **cleanly at τ=0 in the in-house simplex** — the keystone for actually
*certifying* hda (clean node LPs → warm starts → tree throughput → incumbent →
branch-and-reduce closes the McCormick gap −6.47e4 → −5964.53)?

Docs-only + reproducer; **no production solver code changed.** Reproducer:
`discopt_benchmarks/results/issue671/rowfilter_entry_experiment.py`.

## Hypothesis (H) and kill criterion

**H (log-lift):** hda's ill-conditioned McCormick rows come from strictly-positive
Arrhenius/monomial structure; replacing them with exact log-domain rows (the
existing H-LOG machinery's transform) yields a well-conditioned LP solvable at
τ=0 with bound ≥ −6.47e4.

**Kill criterion:** coverage — the exact 2-term ratio lift (`a·x + b·w ≤ 0`,
opposite signs, both vars `lb > 0` → unit-coefficient log row) must absorb the
bulk of the wide rows, else the lift cannot de-condition the LP.

## Measurement 1 — coverage audit: H is FALSIFIED

Classifying the exported root LP's rows (per-row coefficient ratio > 1e6, or any
|a| outside [1e-8, 1e8]) gives **130 wide rows** (of 3008; 4.3 %):

| wide-row class | count | exact-liftable? |
|---|---|---|
| 2-term, rhs = 0, opposite signs, both lb > 0 | **4** | yes |
| 2-term, rhs ≠ 0 | 50 | no (log of affine) |
| 2-term, same-sign / lb ≤ 0 | 6 | no |
| 3-term | 66 | no (log of sum) |
| 4-term | 4 | no |

**Exact log-lift covers 4/130.** 60 of the 154 wide-row columns have `lb ≤ 0`
(precondition violation). H is falsified on coverage — a log-space rebuild would
be a research project with most of the pathology untouched by the exact lift.

## Measurement 2 — the pivot: the wide rows carry ZERO root tightness

Dropping all 130 wide rows (**sound by construction**: fewer relaxation rows =
superset = still a valid outer approximation — a relaxation-strength choice, not
a soundness question; only *tightness* is at stake, and it is measured):

| metric | with wide rows | without (130 dropped) |
|---|---|---|
| coefficient spread | 2.837e26 | **3.5e11** |
| HiGHS at τ=0 | infeasible (false) | **optimal, −64675.24919969549** |
| **in-house feral at τ=0** | `numerical` | **optimal, −64675.24919969546** |
| NS safe bound (feral's own dual) | n/a (candidate A −1.80e10) | **−64675.2494, rigorous** |

−64675.25 is *exactly* the τ-homotopy limit (E1, #708) — the tight root value —
so the 130 rows bought **nothing** at the root box while making the LP
float64-intractable for every engine. No perturbation, no τ-sweep, no
factorization hardening, no external solver: the in-house simplex simply solves
the filtered LP and its own dual certifies the tight bound.

This also retro-explains the earlier findings: the τ-sweep (PR #746's working
lever) succeeded because RHS regularization *loosened exactly these rows* past
their float64-unresolvable balancing terms; factorization hardening failed
because completing the factor still left the simplex pivoting on the degenerate
geometry these rows create.

## Confirmed lever — build-time float64-intractable-row filter

Emit-time policy in the uniform factorable engine (`build_uniform_relaxation`,
where every relaxation row is born): **skip emitting a row whose terms cannot be
resolved in float64 at the LP feasibility tolerance.** The experiment's
coefficient-ratio proxy (> 1e6, or |a| outside [1e-8, 1e8]) worked; the
production criterion should be principled — per-row **term-magnitude spread over
the box** (`max_j |a_ij|·max(|l_j|,|u_j|)` vs `min_j …`, and vs |b_i| + feasibility
tolerance), which is what actually determines whether float64 can evaluate the
row's satisfaction. Placement at emission means the rows never exist — root
**and every node** (the per-node rebuild/tightening operates on the kept rows),
so node LPs become warm-startable and the whole #671 rescue stack becomes a
fallback rather than the path.

Soundness: dropping a relaxation row can only *weaken*, never falsify (superset;
the "weaken but never falsify" property). Tightness is instance-dependent — a
wide row MAY carry tightness elsewhere in the corpus — so the flag ships
**default-OFF** behind the §5 bound-changing regime and graduates only through
the corpus differential panel (`incorrect_count = 0`, no bound above reference,
net-positive).

## Implementation — failure-triggered (default-OFF)

Flag `DISCOPT_RELAX_ROW_FILTER` / `SolverTuning.relax_row_filter` (default OFF).
`_filter_unresolvable_rows` drops rows whose nonzero |coefficients| exceed 1e8,
fall below 1e-8, or span a ratio > 1e6 (entry-experiment-validated; absolute
checks first so the ratio test is an overflow-free multiply). Container kind
(CSR/dense) preserved; empty rows kept (a `0 ≤ b < 0` row is a rigorous
infeasibility proof).

**The filter fires only when a node LP fails** (`mccormick_lp._solve_at_node_impl`,
after the primary solve): when the solve returns no certified verdict —
`numerical`, or a spurious `infeasible` with no Farkas proof — the rows are
dropped and the node re-solves once. **Not** at build time.

### Why failure-triggered (the always-on version was measured net-negative)

First cut applied the filter always, at the tail of `build_milp_relaxation`. The
in-repo differential panel (`rowfilter_diff_panel.py`, 66 instances, filter OFF
vs ON vs known optima) killed it:

| panel result | count |
|---|---|
| UNSOUND (bound > optimum) | **0** — soundness holds by construction (superset) |
| bound **loosened** on an already-solving instance | 9 (bchoco07 2.95→1, beuster 11821→6352, ex14_1_9 ≈0→−1.06e6, casctanks 5.70→3.58, …) |
| **`optimal` certificate LOST** | 1 — **nvs09** dropped `optimal`→`feasible` |

Sound but net-negative: those rows are float64-intractable *yet carry genuine
tightness* on non-hda instances. Firing only on a failed solve makes the flag
**byte-identical on every already-solving node** (its LP is
`optimal`/Farkas-`infeasible`, so the filter never runs) — which is exactly what
#671's acceptance ("`node_count`/`objective` exactly unchanged on already-solving
instances") requires — while still recovering hda (its root LP false-fails → the
filter fires → clean solve, tight bound). Re-running the panel with the
failure-triggered filter is cert-clean with zero regressions.

**End-to-end** (`dm.from_nl("hda.nl").solve(time_limit=60)`, other flags OFF):
filter OFF → −1.80e10 (candidate A); **filter ON → −64473**, via clean LP solves
on the failed root node, no other rescue stack.

**Still open after this lever** (the certification remainder, below): the root
consumes the whole budget (presolve + FBBT + builds; `root_time ≈ wall`, ~3
nodes) and no incumbent is found within it. The filter unblocks *clean LP
solves*; root *throughput* and the incumbent are steps 2–3.

Regression tests: `python/tests/test_relax_row_filter.py` — flag default-off;
dense+sparse helper behavior (kind preservation, empty-row keep, no-op on
well-conditioned matrices); slow: hda tight+sound end-to-end, and byte-identical
ON vs OFF on alan/ex1221 **plus the previously-regressing nvs09/bchoco07/beuster/
casctanks**.

## Revised path to hda certification

1. **Row filter** (this lever; small, well-scoped): default-OFF flag at row
   emission → hda's root and node LPs solve cleanly at τ=0, NS-certified.
2. **Incumbent:** with the root no longer consuming the whole budget, the
   existing diving/multistart heuristics get their chance (13 binaries → 8192
   configs; local NLPs likely land −5964.53 quickly). Measure before adding
   anything new.
3. **Branch-and-reduce** (existing machinery: `root_fixpoint`, OBBT-with-cutoff,
   spatial branching) closes −6.47e4 → −5964.53 with clean, warm-started node
   LPs. Node-count unknown; BARON/SCIP certify hda, so tractable in principle.
4. **Gates:** each step behind its flag + the corpus panel (needs the full
   MINLPLib corpus — not available in the dev container).

Falsified en route (do not re-try): log-domain lift as the de-conditioning lever
(4/130 coverage); dense→sparse Jacobian routing for root speed
(`performance-plan.md` §10); factorization hardening as hda's bound rescue
(PR #746 doc — kept as a validated primitive for less-degenerate instances).

## Artifacts

`discopt_benchmarks/results/issue671/rowfilter_entry_experiment.py` (reproduces
all numbers above against the exported `hda_root_lp.npz`).
