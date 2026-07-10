# A-UNBOUNDED (task #94): is deriving finite bounds on unbounded continuous vars the certification lever?

**Date:** 2026-07-10 · **Base:** `main` @ `85a6663a` (worktree `a-unbounded/entry`) ·
**Measurement-first entry experiment** — no solver changes shipped in this PR.

## 0. TL;DR — VERDICT: **KILL** (with one narrow, sound engine-robustness sub-finding)

The A-RESCUE #91 / F9 finding was: the nvs05/tanksize/casctanks certification stall is
driven by **unbounded continuous variables** — over an unbounded box alphaBB abstains,
the interval enclosure is loose, and branching the integers never tightens the
continuous dims, so the dual bound freezes. The hypothesized lever was: derive finite,
tight bounds on those unbounded vars.

**The measurements falsify the "finitize the vars" lever three ways, and the honest
finding is Lever A (relaxation strength), not bound-inference:**

1. **The constraints DO imply finite bounds — and discopt's FBBT already derives them.**
   `discopt.tightening.fbbt_box` finitizes **100 %** of the unbounded vars on all three
   instances (nvs05 4/4, tanksize 26/26, casctanks 296/296). So the KILL-branch of the
   entry plan ("variables genuinely free, no finite implied bound") is itself **false**:
   these MINLPLib instances are *not* under-specified; each free/half-bounded var is
   pinned by a defining equality or one-sided constraint, and the Rust FBBT engine
   recovers a finite box.

2. **The solver already derives those finite bounds internally (root OBBT), so
   pre-conditioning the model with `fbbt_box` changes NOTHING end-to-end.** Applying the
   FBBT box to the model before `solve()` reproduces the *identical* `root_bound`,
   final bound, and node count on nvs05 (0.6740 → 1.3521, 315–331 nodes) and tanksize
   (root 0.8473, final 0.8680, 61 nodes). The finite bounds are already in hand; the
   bound is loose *anyway*.

3. **The bound is loose because the RELAXATION is loose, not because vars are
   unbounded.** The clinching measurement: **nvs05's objective does not even contain
   any unbounded variable.** It is `1.10471·x0²·x1 + 0.04811·x2·x3·(14+x1)` over the
   *already-bounded* x0–x3 (x0,x1 ∈ [0.01,200] continuous; x2,x3 ∈ [1,200] integer).
   The four free vars x4–x7 appear only in **constraints**. The `root_bound` 0.674 is the
   interval enclosure of `x0²·x1` over the large declared box — the constraints (not the
   box) are what actually restrict x0–x3 near the optimum, and the relaxation does not
   propagate that. Finitizing x4–x7 cannot move an objective bound that never saw them.

**nvs05's final bound 1.3520892806701879 is exactly the F8 taint floor** (§6 F8, task
#89): a non-rigorous NLP-failure sentinel fathom pins the reported bound at the pop-time
bound 1.3521 of the earliest tainted node. So nvs05 is capped by the *already-banked* F8
mechanism, not by unbounded vars.

**Lever 2 (`DISCOPT_ROOT_FIXPOINT=1` + `DISCOPT_NODE_REDUCE=1`) is INERT** on
nvs05/tanksize — identical root/final bound and node count — because both hang off
`_mc_lp_relaxer is not None`, and this class routes to alphaBB/interval per-node.

**Gate analysis contradicts the task hypothesis:** the entry plan guessed nvs05's
unbounded vars fall *outside* the `n<=100` per-node-OBBT gate. They do **not** — nvs05
(8 vars) and tanksize (47) both PASS `n<=100` *and* `n<=50`; only casctanks (500) is
size-gated. The real per-instance disabler is the **McCormick-LP root probe running on
the raw unbounded box**: it returns `numerical` (tanksize) or `optimal`-with-no-safe-bound
(casctanks), which nulls the LP relaxer (`solver.py:5176`) and with it the whole
OBBT/node-reduce path.

**The one narrow, sound sub-finding (recorded, not shipped):** running FBBT *before* that
root probe flips tanksize's probe `numerical`(useless) → `optimal`/**0.840**(useful) and
casctanks' `optimal`/None → `optimal`/**−90.2**(useful). This is a real robustness gap —
the probe stumbles on ±inf bounds it did not need to see. **But end-to-end it does not
close the gap** (tanksize stays 0.847/0.868 identical; casctanks stays −90.2 root / 90 %
gap), because the relaxation is loose regardless. So it is at most engine-robustness, not
a certification lever, and it is **not** worth shipping against this class. **KILL.**

## 1. Method

- Instances: nvs05, tanksize, casctanks from
  `~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`; oracle `minlplib.solu`
  (nvs05 `=opt=` 5.4709341, tanksize `=opt=` 1.2686438, casctanks `=best=` 9.1634794).
- Rust ext built in this worktree (`maturin develop --release`, exit 0); import verified
  to resolve to the worktree (`python/discopt/__init__.py`, native `_rust`).
- `Model.solve(time_limit=30)` per instance, defaults, one process each. FBBT via the
  public `discopt.tightening.fbbt_box(model, max_iter=50)`. Node-engine attribution by
  monkeypatch-counting `_compute_alphabb_bound` / `_compute_interval_bound`; per-node
  OBBT engagement by counting `discopt._jax.obbt.obbt_tighten_root`; probe outcomes by
  calling `MccormickLPRelaxer.solve_at_node` directly on the raw and FBBT boxes.
- Deterministic metrics (bounds, node counts, probe status, var ranges) drive the
  verdict; wall times are indicative (shared machine, flag-graduation agent co-resident).

## 2. Per-instance unbounded-var + constraint-implied-bound characterization

| instance | vars | unbounded (declared) | kind | in objective? | constraints imply finite? | FBBT finitizes |
|---|---:|---:|---|---|---|---|
| nvs05 | 8 | 4 (x4,x5,x6,x7) | FREE ±inf, continuous | **no** — obj is over x0–x3 | **yes** — each pinned by a defining equality | 4/4 |
| tanksize | 47 | 26 | 25× `no-ub` (lb=0), 1× FREE (x17) | **x17 IS the objective** | **yes** | 26/26 |
| casctanks | 500 | 296 | `no-ub` (lb=0) | no (80 obj vars, all bounded) | **yes** | 296/296 |

**nvs05 defining equalities (the crux — the region is bounded, the declared box is not):**

- `x4 = 4243.28/(x0·x1)` (c0) — bounded once x0,x1 ∈ [0.01,200]
- `x6 = sqrt(0.25·x1² + (0.5·x2+0.5·x0)²)` (c1) — bounded
- `x7 = 0.5·x1/x6` (c3) — bounded once x6 > 0 bounded
- `x5 = ((59405.9+2121.64·x1)·x6)/((x0·x1)·(0.0833·x1²+(0.5·x2+0.5·x0)²))` (c2) — bounded
- c4 is then a genuine inequality among already-bounded vars.

FBBT before/after (nvs05): x4 [−∞,∞]→[0.106, 1.36e4]; x5 →[4.4e-5, 1.36e4];
x6 →[0.505, 154.3]; x7 →[3.2e-5, 198]. (tanksize x17 →[0.838, 1840]; casctanks
296 vars → all finite, e.g. the concentration block → [0.004, 1.4].)

## 3. FBBT / OBBT before-after and bound trajectory

### Lever 1 — FBBT / constraint bound inference

| instance | interval bound (raw box) | interval bound (FBBT box) | note |
|---|---|---|---|
| nvs05 | **0.6740** | **0.6740** (identical) | obj over x0–x3 only; free vars irrelevant to it |
| tanksize | **−inf** | **0.8382** | obj = free var x17; finitization gives the finite obj bound |
| casctanks | (loose) | (loose) | obj over bounded x280+; unaffected |

alphaBB on nvs05: `−inf` (raw) and `−3.8e6` (FBBT box) — *worse* than interval; the
supporting-hyperplane underestimator of `x0²·x1` over a 1.36e4-wide box is useless. The
0.674 root bound is the **interval** value, and it is loose because the box is loose in
x0–x3, which only the *constraints* tighten.

**End-to-end FBBT-preconditioned solve (apply `fbbt_box` to the model, then `solve`):**

| instance | default root_bound | default final / nodes / cert | FBBT-precond root_bound | FBBT-precond final / nodes / cert |
|---|---|---|---|---|
| nvs05 | 0.6740 | 1.3521 / ~325 / no | 0.6740 | 1.3521 / ~323 / no |
| tanksize | 0.8473 | 0.8680 / 61 / no | 0.8473 | 0.8680 / 61 / no |
| casctanks | −90.18 | 0.9023 / 7 / no | −90.18 | 0.9023 / 7 / no |

**No change on any instance** — the solver's internal root OBBT already recovers the
finite box; FBBT up-front is redundant.

### Lever 2 — root_fixpoint + node_reduce (branch-and-reduce already in main)

| instance | default | `ROOT_FIXPOINT=1 NODE_REDUCE=1` |
|---|---|---|
| nvs05 | root 0.6740, final 1.3521, ~321 nodes | root 0.6740, final 1.3521, ~321 nodes (**inert**) |
| tanksize | root 0.8473, final 0.8680, 61 nodes | root 0.8473, final 0.8680, 61 nodes (**inert**) |

Both flags gate on `_mc_lp_relaxer is not None` (`solver.py:5680`, `:5657`); this class
routes per-node to alphaBB/interval (nvs05: measured 240 alphaBB + 480 interval calls,
0 McCormick-LP), so the reduce machinery never sees a node-LP marginal to work with.

## 4. The n<=100 gate analysis

| instance | n_vars | dependent vars | `n<=100` (per-node OBBT) | `n<=50` (auto-RLT) | binding disabler |
|---|---:|---:|:--:|:--:|---|
| nvs05 | 8 | 4 (x4–x7) | **PASS** | **PASS** | none by size — see below |
| tanksize | 47 | 3 | **PASS** | **PASS** | McCormick-LP probe → `numerical` → LP relaxer nulled |
| casctanks | 500 | 220 | **FAIL** | **FAIL** | size gate **and** probe → no-safe-bound |

- The task hypothesis ("nvs05's unbounded vars fall OUTSIDE the `n<=100` gate, explaining
  the ~neutral root_fixpoint") is **FALSE**. nvs05 passes both size gates; per-node OBBT
  *does* engage (measured: `obbt_tighten_root` called 4× during the nvs05 solve). It
  simply does not lift the bound past the F8 taint floor 1.3521 — 4 OBBT rounds on an
  alphaBB-declining node close nothing.
- The genuine per-instance disabler is the **McCormick-LP root probe** on the raw
  unbounded box (`solver.py:5146–5177`): `_probe_useful = infeasible OR lower_bound is not
  None`; when false, `_mc_lp_relaxer = None` and the entire OBBT/node-reduce/LP path is
  dropped. Measured raw-box probes: nvs05 `optimal`/0.674 (**useful**, survives),
  tanksize `numerical`/None (**useless → nulled**), casctanks `optimal`/None
  (**useless → nulled**).

## 5. The one sound sub-finding (recorded; NOT a certification lever)

Running FBBT *before* the root probe, the probe succeeds where the raw box failed:

| instance | raw-box probe | FBBT-box probe |
|---|---|---|
| tanksize | `numerical`, lb=None → **useless** | `optimal`, lb=**0.8401** → **useful** |
| casctanks | `optimal`, lb=None → **useless** | `optimal`, lb=**−90.18** → **useful** |

The ±inf bounds make the McCormick LP ill-posed/unbounded on the raw box; a finite box
fixes that. **But** the full-solve numbers (§3, FBBT-precond) show enabling the LP path
this way closes **zero** additional gap — the relaxation is loose regardless. So this is
engine robustness (a probe seeing bounds it did not need to), not certification. Not
worth shipping against this class; a general "FBBT the box before the LP probe" change
would be a broad-corpus bound-neutral robustness task, separately motivated, not an
A-UNBOUNDED certification win.

## 6. Verdict and falsification record

**KILL.** The A-UNBOUNDED hypothesis ("derive finite tight bounds on the unbounded vars
and the certified bound climbs toward the incumbent") is falsified:

- The unbounded vars are **not genuinely free** — constraints imply finite bounds and
  FBBT already recovers them (so the plan's KILL-branch reasoning, "instances are
  under-specified / need user bounds," is *also* wrong; discopt need not warn/require
  finite bounds here).
- The finite bounds are **already derived internally**; pre-conditioning is redundant.
- The stall is **relaxation looseness (Lever A)**, confirmed by nvs05's objective not
  containing any unbounded var and by FBBT-preconditioning closing no gap on any of the
  three — consistent with DECOMP-1 (nvs05/tanksize/casctanks all attributed to Lever A)
  and F8 (nvs05 capped at the taint floor 1.3521).

**Next fix (per DECOMP-1 §5, unchanged by this experiment):** attack Lever A on the
alphaBB/interval-routed sub-class (nvs05, tanksize) — the per-node relaxation, not the
tree policy or bound-inference, is binding. This entry experiment removes
"finitize the unbounded vars" from the candidate-lever list for this class.

### Falsification to record in `gap-closing-execution-plan.md` §6 and `performance-plan.md` §6

> **F9** *(re-scoped by task #94)* — "Finitizing the unbounded continuous vars on the
> nvs05/tanksize/casctanks class is the certification lever." **Falsified.** FBBT already
> finitizes 100 % of them (nvs05 4/4, tanksize 26/26, casctanks 296/296) and the solver's
> root OBBT already applies equivalent bounds; FBBT-preconditioning closes 0 additional
> gap end-to-end. nvs05's objective contains no unbounded var (bound 0.674 is a loose
> interval enclosure of `x0²·x1`); its final bound 1.3521 is the F8 taint floor. Lever 2
> (`ROOT_FIXPOINT`+`NODE_REDUCE`) is inert (both gate on `_mc_lp_relaxer`, which this
> class lacks per-node). The class is a Lever-A (relaxation-strength) problem, not a
> bound-inference one. Sub-finding banked separately: FBBT before the McCormick-LP root
> probe rescues the probe from `numerical`/no-safe-bound on tanksize/casctanks — engine
> robustness, not a certification lever (no end-to-end gap closed). *(2026-07-10, #94)*

## 7. Reproduction

```bash
# characterization + FBBT finitization
python scripts/a_unbounded_characterize.py         # unbounded vars + FBBT before/after
# per-instance solve + engine/probe attribution
python scripts/a_unbounded_probe.py                # raw vs FBBT-box McCormick-LP probe
DISCOPT_ROOT_FIXPOINT=1 DISCOPT_NODE_REDUCE=1 \
  python scripts/a_unbounded_solve.py nvs05 30      # Lever 2 inertness
```

Numbers in this report were produced on macOS arm64, base `main` @ `85a6663a`,
2026-07-10, in an isolated worktree.
