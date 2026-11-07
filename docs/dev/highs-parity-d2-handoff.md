# D2 handoff — where the next session starts (2026-11-07)

Branch `claude/happy-hamilton-0vvo3b`, draft PR jkitchin/discopt#2196 (contributes to
#1671). The owner's goal is to eliminate HiGHS. D2, the pure-MILP route default, is the
last big default that still needs HiGHS. The detailed record is
`highs-parity-d2-plan.md` §0-§2, one section per workstream (W1-W6). This page is
the short version: what is on, what is off, and what to do first.

## State

- **Panel:** MIPLIB-38 + M (62 instances), 20 s.
- **Score:** Rust certifies ~42 per rep (125/186 on the last 3-rep panel); HiGHS ~59.

**Default ON since this session (graduated on §5 panel + corpus differential):**
- `DISCOPT_MILP_CUT_LP`: HiGHS's persistent-LP cut policy for the node-GMI pool.
- `DISCOPT_MILP_SB_ARMED`: strong-branch probes on the degeneracy-armed warm path.

**Fixed this session (correctness):**
- the lazy separator judges the integral realisation, and branches on a repeated
  near-integral veto (#1561 witness);
- the HiGHS MILP route's LP checks run on HiGHS's LP;
- the #1153 root-cut budget is saturated;
- the cut-row deletion that dropped a slack column other cut rows used (a false
  certificate on `p0548` while the flag was default-OFF).

**Default OFF, measured, still in the code (each has a reason to stay):**

| flag | state |
|---|---|
| `DISCOPT_MILP_RAND_ROUND` | sound, primal-positive, fails 2b alone; graduate with the next primal lever or retire at the end of D2 (audit row) |
| `DISCOPT_MILP_D2_CMIR`, `DISCOPT_MILP_CMIR_HIGHS`, `DISCOPT_MILP_CMIR_FLIP`, `DISCOPT_MILP_LIFTED_MIC` | W4 c-MIR port; root bound on `mik-1` -52,544 -> -51,493. Route panel failed (LP numerics), see the plan's W4 update. Re-panel after the engine work |
| `DISCOPT_LP_PRIMAL_INCR_PRICING` | engine step 1: incremental reduced costs in the primal simplex. Pricing 2-6x cheaper on the primal-heavy rows. §5 panel (`d2-analysis/eng_incr_pricing_panel_r3.json.gz`): cert-clean, nodes -16 % (2b passes), 1 dual loss, median primal gap level. It **narrowly fails**: certified 125 -> 124 and `p0548`'s incumbent lost in one rep. `p0548` also lost an incumbent in the RAND_ROUND and EXPEL panels, so check it as a trajectory-sensitive row first. Re-panel together with step 2 (row-wise pricing) |

**Retired this session (measurement recorded in the plan):**
- `DISCOPT_MILP_SUBMIP_SCHED`;
- `DISCOPT_MILP_NODE_EXPEL_ARTIFICIALS`;
- `DISCOPT_MILP_SUBTOL_INF_SKIP`;
- the root-basis and recent-basis fallbacks of the cut LP;
- the nnz-ratio fold cap.

## Full-suite result and the one branch-caused failure (2026-11-07, late)

`pytest python/tests -m "not slow" -n 4` on `1dccb8e3`: 7 failed, 20,680 passed.
- 6 are environmental: 3 need `cyipopt` (no Ipopt C library in the container), 3 need
  `sympy` (pass once it is installed).
- 1 is branch-caused: `test_nlpbb_unscaled_refine::test_large_coefficient_rows_leave_through_the_gate`.
  It passes on `origin/main` (4.8 s) and fails here (61 s, the refine never runs).

**Cause, bisected (Python file, then hunk).** The D3 default flip of
`_convex_route_oa_master` to the in-house master. With `=highs` the test passes.
- The route's `lp_nlp_bb` master returns `infeasible` in 0.2 s. HiGHS returns a point
  at 0.0282905 on the same master. NLP-BB then runs alone and finds no incumbent in 60 s.
- On the captured master MILP (17 columns), the in-house driver says INFEASIBLE with the
  D3 lazy root cuts on (`DISCOPT_MILP_LAZY_ROOT_CUTS`, the c-MIR part). It says OPTIMAL
  without them, and main's driver says OPTIMAL too.

**It is not a false certificate.**
- Checked in exact rationals, none of the 256 integer points in that master's box
  satisfies its rows. The closest misses the budget row `sum x = 1` by 3.35e-7.
- That 3.35e-7 is exactly the residual of HiGHS's point and of main's "optimal" point.
- The route gets the same box on main (`x9 in [87078, 87079]`, ...), so the box is
  exactly infeasible upstream of this branch.
- Main and HiGHS accept a tolerance-feasible point. This branch's trajectory proves
  exact emptiness first: Farkas fathoms at margin 1e-9, plus FBBT through
  `x15 = 55000 x6` and integer rounding.

**What it leaves open (D3).**
- The in-house master's infeasibility proofs are exact. Its incumbent acceptance is
  tolerance-based (`ROW_VERIFY_TOL` 1e-6). A master that is feasible only within
  tolerance therefore gets "infeasible" or "feasible" depending on the trajectory.
- HiGHS's propagation accepts a continuous tightening only above `1000 * feastol` and
  30 % of the domain (`HighsDomain.cpp` 1349-1410). Ours accepts anything above the
  pivot tolerance.
- Porting that rule was tried and did not fix this master (the Farkas path still
  fathoms), so it was not kept.
- The test stays red until D3 settles how an exact-proof master meets a tolerance-only
  feasible box. The options are tolerance-robust Farkas and FBBT margins, or
  confirmation by a second engine.

## First things to do

1. **Run the full `-m "not slow"` suite.** `origin/main` was merged at `8c86eab9`; the
   audit test, `cargo test -p discopt-core --lib` (1032) and smoke (4610) pass. The full
   suite found 11 branch-caused regressions last time that smoke did not; 6 failures in
   this container need `cyipopt`/`sympy` and are environmental. The audit now has rows
   for every default-OFF gate below (marker 62).
2. **Engine step 2 (written, not built):**
   `docs/dev/data/highs-parity/d2-analysis/eng_rowprice.patch`.
   - It is a row-wise (hyper-sparse) pivot row for the incremental pricing: a CSR copy
     built once per primal loop, and α_r accumulated over ρ's support when its rows
     hold under half the matrix.
   - Apply it, run `cargo test -p discopt-core --lib` with and without
     `DISCOPT_LP_PRIMAL_INCR_PRICING=1`, then measure `PriceSweep` on the rows in
     `d2-analysis/eng_incr_pricing_entry.jsonl.gz`.
3. **Then, in order (plan W5 list):**
   - (a) cold LP solves through the dual simplex (dual phase 1 with artificial bounds)
     where `dual_slack_basis` gives up;
   - (d) physical removal of lazily deleted cut rows;
   - re-panel the W4 c-MIR bundle on the faster engine.

## Tools that worked

- `docs/dev/data/highs-parity/d2-analysis/d2_probe.py`: per-instance probe with
  profile counters. Run it on the 4-12 target rows of a lever as its entry
  experiment.
- `.../w4_root_bound.py`: root bound per cut arm.
- `discopt_benchmarks/scripts/d2_flag_panel.py --flag F[,G] --fired CTR --reps 3`: the
  §5 graduation panel, about 1 h.
- `discopt_benchmarks/scripts/flag_onoff_corpus_panel.py --flag F --solu
  ref/minlplib/minlplib.solu --extra-dir python/tests/data/minlplib`: the corpus
  differential, about 20 min.
- **The HiGHS expert.** `python/discopt/skills/agents/highs-expert.md` as a persona for
  a general-purpose agent, with the source in `ref/HiGHS`. Every principled fix this
  session came from asking it before building.
- **Pitfalls:**
  - an oracle check must respect the objective sense (`2122` maximizes);
  - never run builds or a second panel during a panel (§9);
  - containers restart; `--resume` the panel.
