# TAIL-1a — certification-stall root-gap attribution (nvs05, nvs09, tanksize)

**Task:** gap-closing execution plan §3 TAIL-1(a). For each stall instance (optimum
found instantly, dual bound flat for 60 s while BARON closes at 0.0–2.8 s),
determine *which constraint/operator class keeps the relaxation loose*, whether
root OBBT moves the box, and whether the branch-and-reduce loop would close it.
Fix the **operator class** if a clean, non-instance-keyed fix emerges; otherwise
record the dependency and do **not** duplicate the TD-A / BR / CUTS agents' work.

**Verdict: no new operator-class code. All three roots trace to already-identified,
already-built-or-planned levers.** This is a diagnostic record, not a build.

## Method

Instrumented `Model.solve(time_limit=12, gap_tolerance=1e-4)` per instance with
`discopt` INFO logging, reading `SolveResult.root_bound/root_gap/root_time` (the
TAIL-1c producer) and the OBBT/FBBT/linearizer log lines. BARON root bounds are
from the TAIL-1c iteration-1 parse (`global_opt_baron_vs_discopt.py`, GAMS
`lo=3`). Operator classes are the `.nl` opcode histograms
(`python/tests/data/minlplib_nl/<inst>.nl`). Env: `JAX_PLATFORMS=cpu`,
`JAX_ENABLE_X64=1`, isolated venv.

## Per-instance attribution

| instance | dominant nonlinearity (.nl opcodes) | discopt root_bound | BARON root_bound | discopt root_gap | BARON root_gap | loose because | lever (owner) |
|---|---|---|---|---|---|---|---|
| **nvs05** | products o2×29, pow o5×11, div o3×5, sqrt o39×3 | 0.674 | 4.283 | **0.877** | 0.217 | root_bound == the box-minimum of an *already-exactly-relaxed* objective monomial (0.674); the loose bound is the **box**, not the envelope | **box reduction — BR-1/BR-2 (branch-and-reduce)**; TD-A ON gives byte-identical 0.674 (not a lifting problem) |
| **nvs09** | **log o43×20**, pow o5×21, products o2×9 | −72.90 | −43.13 | **0.690** | **0.000** | objective has **log(·)·log(·)** (squared univariate logs); `_decompose_product` cannot split two transcendental factors → term dropped → **feasibility objective, no linearized bound** | **operator-class fix EXISTS: TD-A `DISCOPT_LIFT_LOOSE_PRODUCTS`** (default OFF, pending graduation) |
| **tanksize** | **products o2×84**, sumlist o54×11, sqrt o39×3 | 0.847 | 0.955 | **0.332** | 0.247 | genuine (no linearizer refusal) but loose **bilinear/multilinear McCormick** envelope; root OBBT tightens 65 bounds and the reduce↔relax OBBT/DBBT loop fires 5× (57–65 bounds/sweep) yet the bound stalls | **box moves but envelope stays loose — BR + bilinear cuts (BR-2 / CUTS-1)** |

BARON closes nvs09 **at the root** (root_gap 0.000, root_bound == optimum), and is
near-tight on nvs05/tanksize at t ≤ 0.11 s — the campaign's target profile.

## Findings (dependencies, not builds)

1. **nvs09 — operator-class fix already implemented (TD-A), do not rebuild.**
   `DISCOPT_LIFT_LOOSE_PRODUCTS` (`_jax/factorable_reform.py`) lifts `t == g(x)`
   for an integer power of a transcendental call and rewrites `g(x)**n` as the
   monomial `t**n` the pipeline relaxes exactly. Reproduced this run:

   | | root_bound | root_gap | linearizer |
   |---|---|---|---|
   | flag OFF | −72.90 | 69.0 % | "Cannot decompose product: log·log" → feasibility objective |
   | flag ON  | −54.83 | 27.1 % | linearized (no refusal), optimum still −43.134, nodes 95→63 |

   Matches the documented TD-A entry experiment exactly (−72.90 → −54.83). The
   flag is **blocked from default-ON only by the graduation pipeline** (last gate
   run 2026-07-07 was INELIGIBLE for a *shared* reason — the OFF control's tls2
   correctness gate, since fixed by C-38, not a TD-A defect;
   `docs/dev/flag-graduation-run-2026-07-07.md` line 41/87, `g4-hard-tail-bounding-2026-07-07.md`).
   **Dependency: re-run `graduation_gate.py --flags lift_loose_products` now that
   C-38 is fixed** — owned by the flag-graduation (G1) track, not TAIL. No new
   TAIL code.

2. **nvs05 — not a lifting problem; box reduction.** TD-A ON reproduces the
   identical root_bound 0.674 (== the objective monomial's box minimum), so the
   envelope is already exact; the loose bound is the *box*. The lever is
   branch-and-reduce / cutoff-OBBT range reduction (**BR-1/BR-2**), previously
   recorded in `uncertified-tail-plan-results-2026-07-06.md` §TD-A. No TAIL code.

3. **tanksize — bilinear envelope + box, not a refusal.** No linearizer refusal;
   the multilinear (o2×84) McCormick envelope is genuine but loose and the OBBT
   loop already moves the box without closing it. Levers are branch-and-reduce
   (**BR-2**) and bilinear/aggregation cuts (**CUTS-1**). No TAIL code.

## Bottom line

TAIL-1a surfaces **zero new operator-class work**. The one genuinely
envelope-loose root (nvs09) is closed by shipped-but-flagged **TD-A**, whose only
blocker is a graduation re-run (G1 track, unblocked by C-38). nvs05 and tanksize
are **box-reduction / cut** problems owned by BR-1/BR-2 and CUTS-1. Recorded per
the §0.3 falsification-is-progress rule; no build, no gate weakened.
