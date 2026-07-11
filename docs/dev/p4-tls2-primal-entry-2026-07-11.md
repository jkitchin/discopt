# SOTA-P4 (task #100) — tls2 primal: entry experiment + KILL

**Date:** 2026-07-11 · **Base:** `origin/main` @ `c2493ba` · **Scope:** PRIMAL half of
the tls2 miss (find a feasible incumbent). **Measurement-only** — no solver change
shipped (see §5 for why shipping a heuristic here would be a dead flag).

## 0. TL;DR

tls2 is the only PRIMAL miss on global50 (DECOMP-1 §3: dual bound 91 % closed,
**no incumbent ever found** → `time_limit` with no feasible point). The task
hypothesis was that this is a *wiring gap* — a primal heuristic that would work
isn't invoked on tls2's path — fixable by a small, sound addition (e.g. a
Fischetti objective feasibility pump, or wiring the spatial-path integer-lattice
search into the NLP-BB path tls2 routes through).

**Verdict: KILL.** tls2 feasibility is a genuinely hard combinatorial
MIP-feasibility subproblem. Every finder in discopt's arsenal — and a
from-scratch Fischetti objective pump prototype, and a 651-way random-restart
subNLP — fails to find a single feasible point in-budget, on **both** the NLP-BB
and the spatial B&B path. SCIP solves tls2 to optimality in **0.21 s** and BARON
at the **root (0.0 s)**, both via their MIP-feasibility machinery over a strong
LP relaxation + cuts + presolve. The tls2 primal miss is **downstream of
relaxation weakness (Lever A)**, not a heuristic wiring gap: discopt's weak
McCormick/NLP relaxation parks the binaries at ~0.37, so no rounding lands in a
feasible basin. This matches the kill criterion exactly ("BARON/SCIP also lean
on their MIP feasibility machinery → the lever is a real MIP-feasibility build").

## 1. Reproduction (TL = 60 s, defaults)

`from_nl(tls2.nl).solve(time_limit=60)`: status `time_limit`, objective `None`,
bound `None` reported (dual 91 % closed internally per DECOMP-1), **1631 nodes**,
0 incumbent injections.

## 2. Structure (why it is hard)

37 vars = **31 binary + 2 general-integer** (x4,x5 ∈ [1,100]) + 4 continuous;
24 constraints. Linearity probe (Jacobian constant across two random points):

- **Objective: LINEAR.**
- **22 of 24 constraints LINEAR**, including **6 equalities** (con12,13,16-19).
- Only **con0/con1 are NONLINEAR** — bilinear trim-loss coupling of the form
  `f(x4·x0) + g(x5·x1)` (integer count × continuous width), the classic
  trim-loss "patterns × pattern-width = demand" structure.

So tls2 is essentially a MILP with two bilinear coupling rows. Feasibility is a
combinatorial pattern-selection problem: pick which of the 31 patterns are active
and the two integer multiplicities so the bilinear demand equalities close. This
is a measure-zero discrete feasible set; continuous completion of a *wrong*
integer assignment is infeasible.

Evidence it is a discrete-feasibility wall, not a continuous one: HiGHS solves the
**linear-only** relaxed MILP (drop con0/con1) trivially (status 0), but that
integer point violates con0/con1 by 8.0 and 7.0 — the nonlinear coupling is the
binding difficulty, and it is not separable from the integer choice.

## 3. What fires today, and why it fails (NLP-BB path)

tls2 is detected **convex** → auto-routes to `_solve_nlp_bb`. Its root finders:

| heuristic | fired? | result | why |
|---|---|---|---|
| RENS | yes | None (declined) | 10/33 integers fractional (max 0.45) at the relaxation; neighbourhood too large / sub-solve infeasible |
| feasibility_pump (×2) | yes | None | all-at-once rounding of 10 fractional binaries is constraint-infeasible; fixed-integer continuous re-solve has no feasible completion |
| fractional_diving (×2) | yes | None | dive hits infeasibility fixing integers one at a time |

The NLP-BB node loop has **no node-level *finder*** — RINS/local-branching are
guarded by `if _lns_inc is not None` (they *improve* an incumbent), so with no
first incumbent they never fire. This is a real asymmetry vs the spatial path,
but §4 shows wiring the spatial finders in does **not** rescue tls2.

## 4. The kill: the whole arsenal fails, on both paths

All runs from `origin/main`, tls2.nl, on the shared macOS arm64 laptop.

1. **Spatial path** (`nlp_bb=False`, the richer node suite with
   `integer_local_search` / node `subnlp` / `enumerate` / diving): `time_limit`,
   **no incumbent**, bound 4.3, 1375 nodes. Wiring is not the lever.
2. **`integer_local_search` standalone** (violation-guided 1-opt/2-opt +
   subNLP repair — the heuristic explicitly built for "relaxation integers
   satisfy the relaxed but not the TRUE constraints"): **None in 1.8 s** (stalls
   at violation local minima on every restart; subNLP repair infeasible each
   time). Used only 1.8 s of a 25 s budget → it is *stuck*, not *starved*.
3. **Fischetti objective feasibility pump** (from-scratch prototype: swap the
   objective for squared L1 distance to the rounded integers, solve the
   continuous relaxation, iterate): **cycles on round 1** — parks at a fixed
   fractional vertex (12 fractional, max frac 0.372, rounded-point violation
   37.6) that the distance projection cannot escape. None in 30 s / 40 rounds.
   The relaxation vertex is a fixed point of the distance projection.
4. **RENS with `max_free=40`** (fix the ~21 integral binaries, solve the entire
   fractional neighbourhood as a 15 s nested sub-MINLP — the strongest
   MIP-feasibility move available in the arsenal): **None in 39.5 s** (the nested
   solve itself times out without feasibility).
5. **651-way random integer multistart + full continuous subNLP completion**
   (uniform random integer draws, each completed by fixing integers and solving
   the continuous NLP): **0 feasible in 30 s / 651 tries.**

Nothing finds a feasible point. Injecting nothing is the only sound behaviour;
adding a heuristic that is inert here would be a dead flag (CLAUDE.md §3).

## 5. External oracle — confirms MIP-feasibility machinery is the difference

- **SCIP: `optimal` in 0.21 s** (`~/Dropbox/projects/discopt-minlp-benchmark/scip_join.csv`,
  row `tls2,0.21,optimal,True,37,24,31,2,192`).
- **BARON: root, 0.0 s** (DECOMP-1 §1 / gap-closing §1 baseline).
- **Oracle optimum: 5.3** (`minlplib.solu`).

Neither wins with a magic pump; both win because a **strong LP relaxation +
cutting planes + full MIP presolve** makes the rounding neighbourhood
feasible-reachable, and their LP-based diving/RENS then complete it in
milliseconds. discopt's per-node relaxation on this class is the weak
McCormick/NLP one (binaries at ~0.37, no cut loop), so its rounding neighbourhood
contains no feasible integer point to find.

## 6. Re-scope: the real follow-on (a MIP-feasibility build, not a heuristic patch)

tls2's primal miss is a **Lever-A consequence** (DECOMP-1 §5: relaxation strength
dominates) surfacing on the primal side. The precise follow-on:

1. **A cut-strengthened LP relaxation on the (near-)MILP class** (tls2 is a MILP
   + 2 bilinear rows): a real LP root with gomory/MIR/knapsack cuts + MIP presolve
   (probing, coefficient tightening on the set-partitioning rows), so the
   fractional binaries move off 0.37 toward a feasible vertex. This is the
   c-MIR / cut-strength front (CUTS-1/#81) applied to the integer-linear core,
   plus a genuine MIP presolve — *not* a primal-heuristic addition.
2. **On top of that stronger relaxation, an LP-based feasibility pump / diving**
   (Fischetti–Glover–Lodi objective pump solving the distance projection as an
   LP that respects integrality of the fixed part, with SCIP-style restart /
   randomized-rounding perturbation to break the cycling seen in §4.3). The pump
   is only worth building *after* the relaxation is strong enough that its
   projection is not a fixed fractional vertex.
3. **Node-level finder in NLP-BB** (wire a bounded `subnlp`/rounding finder at
   nodes without an incumbent, mirroring the spatial path) is a cheap
   *correctness-of-coverage* fix worth doing regardless, but §4.1 proves it is
   **not sufficient** for tls2 alone — it only helps once branching has fixed
   enough binaries that a node relaxation is near-feasible, which needs (1).

Sequencing: (1) is the binding lever and belongs on the CUTS / relaxation-strength
front; (2) and (3) are the primal layer that pays off *only after* (1). Track as a
follow-on to CUTS-1, not as a standalone primal task.

## 7. Soundness note

No incumbent was ever injected in any experiment; the certificate invariant is
untouched. Every candidate a heuristic *would* have injected was independently
re-verified (integer + all-constraint feasibility via
`_check_constraint_feasibility`) and none passed — which is exactly why the run
correctly reports no incumbent rather than a wrong one.

## 8. Reproduction

```bash
# baseline (no incumbent, 60 s)
PYTHONPATH=python python -c "from discopt.modeling.core import from_nl; \
  print(from_nl('.../minlplib/nl/tls2.nl').solve(time_limit=60).objective)"
# spatial path (also no incumbent): solve_model(m, time_limit=60, nlp_bb=False)
```

Scratch prototypes for §4 (objective pump, RENS max_free, random multistart,
HiGHS linear-feasibility probe) are not committed — they are one-off diagnostics,
their numbers recorded above.
