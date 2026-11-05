# #2142 exact hull — entry experiment, pre-registered (2026-11-05)

Written before any arm ran past 1 node. The 1-node smoke runs (`arm_*_1.log`) only verified
that the prototype lowers and the root solves.

## Hypothesis

The cstr hull bound is pinned at 0.094 (§114) because the eps-perspective puts `1/yhat` (up
to 1e8) inside every product of the rate law, so the lifted spec carries columns to 1e24 and
bounds to 1e27, above the LP's 1e20 sentinel, and every node LP is near-vacuous or numerical.
A hull lowering of the same rows that is exact at `y in {0, 1}`, is a relaxation of the
disjunction at fractional `y`, and keeps every product between the disaggregated copies (box
<= 10) and `y` (box [0, 1]) removes the magnitude problem: the spec's largest |global bound|
drops from 1e27 to ~1e3, the numerical-LP-verdict rate drops to big-M's order, and the tree
bound moves off 0.094.

## Arms (deterministic, `PYTHONHASHSEED=0`, `max_nodes`, `time_limit=7200` backstop)

- `hull`: the shipped eps-perspective (reference: 0.0940 at 10k, §111/§114).
- `bigm`: the shipped default, dual proofs ON (§112; seed-0 20k reference 2.7875).
- `ratio`: the shipped #1917 eps-free lifted perspective applied to every nonlinear row
  (`y g(t)`, `v = y t`).
- `homog`: homogenized polynomial perspective, no aux (candidate (a)).
- `lifted`: lift-then-hull with homogenized aux definitions and the perspective of each
  aux's McCormick envelope (candidate (b)+(a)).

## Measurements

Per arm: root bound (1 node), bound at 10k nodes, bound at 20k nodes for the arm(s) that
pass 10k, the spec's largest finite |global bound|, `LpVerdictNumerical` and the node-status
histogram, wall. Walls of concurrent runs are recorded as such and not compared.

## Kill criterion (fixed before running)

The candidate is killed unless at least one holds:

1. its deterministic 20k-node bound exceeds `bigm`'s 20k-node bound measured by the same
   driver in the same build (dual proofs ON for both), or
2. it certifies cstr on the default path within the 300 s GDP parity budget.

A secondary bar that is necessary but not sufficient: the 10k-node bound must exceed the
shipped hull's 0.0940. A candidate that clears only the secondary bar is recorded as "better
hull, still not the lever", and the proposal says so.

## Exactness checks (`check.py`, run on every candidate before the 10k arms are believed)

- the big-M certified optimum, mapped to the lowered columns, satisfies every lowered row;
- for 6 fixed selector configurations, the candidate's optimum equals big-M's (both solved to
  certification or to the same incumbent within 1e-5), and the candidate's solution projects
  to a point feasible for the original model's rows and the selected disjuncts' rows;
- convex combinations of lowered feasible points from different configurations satisfy every
  LINEAR lowered row (the hull rows must contain the hull).

Each check prints an executed-assertion count and exits non-zero at zero.
