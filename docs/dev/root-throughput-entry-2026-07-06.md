# Root-relaxation throughput — THRU-1 entry experiment (where the BARON gap actually is)

**Date:** 2026-07-06
**Status:** measured (measurement + prototype only; no production relaxation
change shipped — entry experiment per cert-gap-plan §0.1 clause 2).
**Scope:** the re-scoped certification-gap lever. CUT-1 established that discopt's
default spatial root relaxation is *already* as tight as SCIP's fully-cut root
(nvs17 −1105.89 vs SCIP −1105.10; nvs19 parity) but expensive — nvs24's root
fixpoint does not converge at node 1 within 280 s. This experiment asks the crux
question: **is discopt over-computing a tight bound (a policy problem), or is the
dominant stage genuinely bound-necessary (an engineering problem)?**

> **Method.** Branch `thru1-root-relaxation-throughput` from `origin/main`
> (a25cc3cb). Apple M-series arm64, Python 3.12.11, JAX 0.10.2
> (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`), **release** pounce 0.7.0
> (`_pounce.abi3.so` = 4.73 MB — the debug-build engine artifact of pounce#182
> does not apply), `maturin develop --release`. Instances from the full MINLPLib
> snapshot (`~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`), oracle
> `minlplib.solu`. Per-stage wall from the T0.3 `SolveResult.solver_stats`
> families (`reduce/{fbbt,obbt}`, `separate/{multilinear,edge_concave,
> univariate_square,convex,psd,rlt}`) + `root_time`/`root_bound`; stage shares
> cross-checked with `profile_instance.py --mode cprofile` (Rust/pounce C entries
> timed directly). Reusable harness:
> `discopt_benchmarks/scripts/root_stage_ablation.py` (this PR). Slow set: nvs17,
> nvs19, nvs24 (the PSD/QCQP class CUT-1 named), plus gear4, ex1252a, casctanks
> (certified-but-slow vs BARON, different structure). 60 s budget unless noted.

---

## 1. Per-stage ROOT cost — which stage dominates? (all stages ON)

| instance | wall | root_time | nodes | status | root_bound | bound | oracle | dominant stage(s) (cumulative s) |
|---|---:|---:|---:|---|---:|---:|---:|---|
| **nvs24** | 99.0 | **56.7** | 7 | TL, no incumbent | **none** | −8376* | −1033.2 | **psd 21.2 + univariate_square 42.6**; node MILP solve 30 s / 3 nodes |
| nvs17 | 37.8 | 1.9 | 93 | optimal | −1105.9 | −1100.4 | −1100.4 | **psd 22.3** + univariate_square 7.1 |
| nvs19 | 60.1 | 3.5 | 73 | feasible | −1104.2 | −1103.5 | −1098.4 | **psd 35.8** + univariate_square 15.8 |
| gear4 | 37.9 | 1.8 | 6045 | optimal | ~0 | 1.64 | 1.643 | **obbt 14.8** + fbbt 1.3 (PSD/sq ≈ 0) |
| ex1252a | 21.4 | 15.5 | 19 | feasible | ~0 | 0.0 | 128893.7 | root NLP/heuristic (~10 s untimed) + univariate_square 2.9 + obbt 2.2 |
| casctanks | 64.0 | 20.0 | 127 | TL | −90.2 | −90.2 | 9.163 | **fbbt 11.3** + univariate_square 3.1 |

*nvs24 `bound=−8376` is the loose sentinel from the fallback path (`node MILP` never
finishes a converged relaxation solve at the root within budget).

**The dominant stage is class-specific, not universal:**

- **PSD/QCQP class (nvs17, nvs19, nvs24):** `separate/psd` dominates (22–36 s,
  ~60 % of wall), with `separate/univariate_square` second (7–43 s). On nvs24
  both run at the root/first nodes and the root fixpoint never converges.
- **gear4:** `reduce/obbt` (14.8 s) dominates — the known big-M / branch-and-reduce
  pinned-root class (root bound ~0), *not* a cut stage.
- **casctanks:** `reduce/fbbt` (11.3 s) dominates.
- **ex1252a:** neither a cut nor a reduction stage — ~10 s of the 15.5 s root is
  root NLP/heuristics (untimed by the separation families).

**cProfile attribution on nvs24 (the pathology), 40 s run (52.96 s profiled):**

| bucket | tottime | ncalls | note |
|---|---:|---:|---|
| `discopt._rust.solve_milp_py` | **30.06 s** | 3 | integer-MILP node relaxation solve (~10 s/node) |
| `discopt._rust.solve_lp_warm_csc_py` | **19.60 s** | 78 | warm LP re-solves: **60 from OBBT**, 18 from the cut-separation loop |
| `build_milp_relaxation` (JAX envelope) | **1.85 s** cum | 12 | **NOT the sink** — falsifies the Phase-4/5 "JAX envelope rebuild" hypothesis for this instance |

So on nvs24 the wall is **~94 % in the Rust LP/MILP solver**, driven by (a) the
per-node relaxation being solved as an *integer MILP* (the McCormick relaxation
retains integrality on this dense integer-QP, so the pure-LP warm fast path in
`milp_relaxation.py:326` is bypassed — `_integrality is None` is false), and
(b) the cut-separation re-solve loop feeding it. The **JAX autodiff envelope
rebuild is 1.9 s — not a lever here.** The multilinear hull is `0.0006 s`, so the
"multilinear hull exponential in a high-arity product" hypothesis for nvs24's
non-convergence is **falsified** — nvs24 has no high-arity multilinear cost;
the pathology is PSD/square separation + the integer-MILP node solve.

---

## 2. Value-vs-cost — the crux (does the dominant stage buy any bound?)

For the PSD/QCQP class, turn PSD off (`cuts="manual"` skips the auto-policy that
turns PSD on for box-QP) and univariate-square off (`DISCOPT_SQUARE_SEPARATE=0`),
and measure the **root-bound loss** vs the **wall saved**.

| instance | config | wall | root_time | nodes | status | **root_bound** | bound | valid vs oracle |
|---|---|---:|---:|---:|---|---:|---:|---|
| **nvs17** | base (PSD+sq on) | 37.8 | 1.9 | 93 | optimal | **−1105.89** | −1100.40 | ✓ |
| nvs17 | nopsd | 22.5 | 1.7 | 175 | optimal | **−1105.89** | −1100.40 | ✓ |
| **nvs17** | **nopsd_nosq** | **17.8** | 1.5 | 277 | **optimal** | **−1105.89** | −1100.40 | ✓ **2.1× faster, identical bound** |
| nvs17 | nosq (PSD on) | 60.2 | 1.9 | 159 | feasible (worse) | −1105.89 | −1104.19 | ✓ |
| **nvs19** | base | 60.1 | 3.5 | 73 | feasible | **−1104.24** | −1103.46 | ✓ |
| nvs19 | nopsd | 60.2 | 2.3 | 187 | feasible | **−1104.24** | −1102.60 | ✓ |
| nvs19 | nopsd_nosq | 60.1 | 2.0 | 537 | feasible | **−1104.24** | −1099.98 | ✓ (**7.4× more nodes**) |
| **nvs24** | base | 99.0 | 56.7 | 7 | TL, no incumbent | **none (root ⊘)** | — | — |
| **nvs24** | **nopsd_nosq** | 60.2 | **6.7** | **335** | **feasible** | **−1035.66** | −1035.38 | ✓ **root now converges** |

**The finding: PSD moment cuts are bound-inert on this class.** The reported
`root_bound` is **bit-identical** with and without PSD (nvs17 −1105.8900022127812
in every config; nvs19 −1104.2400022094807). Routing PSD to a 150-round root cut
pool (`DISCOPT_ROOT_CUT_ROUNDS=150`) *also* leaves the root bound identical — even
150 spectral rounds move it 0. So the ~60 %-of-wall `separate/psd` stage
contributes **0 %** of the certified root bound on nvs17/nvs19/nvs24. The tight
−1105.89 root bound (the CUT-1 win vs SCIP) comes from the base
**McCormick + RLT** relaxation, *not* from PSD.

`univariate_square` separation is a smaller but real per-node cost (7–43 s
cumulative); turning it off with PSD costs nothing on this class (same root bound,
same certified objective) and is the difference between nvs17 22.5 s (nopsd) and
17.8 s (nopsd_nosq). `nosq` alone (PSD *on*) is worse — it lets PSD dominate.

**value-vs-cost verdict (nvs17/19/24):**

| stage | cumulative cost (share of wall) | root-bound contribution | verdict |
|---|---|---|---|
| **PSD moment cuts** | 22–36 s (≈ 60 %) | **0** (bit-identical) | **expensive, bound-inert → the lever** |
| univariate-square sep | 7–43 s | 0 on this class | expensive, bound-inert here |
| McCormick + RLT (base) | small | **all of −1105.89** | necessary (the CUT-1 win) |
| OBBT (gear4), FBBT (casctanks) | 15 s / 11 s | class-necessary (branch-and-reduce / feasibility) | different lever (not PSD) |

---

## 3. Does a cost-aware gate make nvs24's root converge? (prototype)

Yes. Disabling the bound-inert PSD + square stages:

- **nvs24 root_time 56.7 s → 6.7 s (8.5×), root now CONVERGES** with a valid
  `root_bound = −1035.66` (≤ oracle −1033.2, no crossing), and the tree makes real
  progress: **7 → 335 nodes**, finding an incumbent (feasible) inside 60 s where
  the baseline finds none.
- **nvs17 37.8 s → 17.8 s (2.1×)**, still `optimal` at the identical certified
  objective −1100.40 and identical root bound −1105.89.
- **nvs19** stays `feasible` at 60 s but explores **7.4× more nodes** (73 → 537)
  and reaches the same root bound — the wall previously burned on inert PSD now
  goes to search.

**Prototype (`root_stage_ablation.py --gate-psd`, default-OFF, findings-only).**
A cheap probe (short base-vs-manual solve, compare root bounds) correctly gates
nvs17/nvs19 to `manual` (PSD off) and preserves correctness. It is **unreliable on
nvs24**: within a short probe the base root sometimes fails to converge at all
(no root bound to compare) and sometimes converges right at the boundary, so the
probe's decision is non-deterministic there. **The robust form is a per-round
in-loop bound-delta gate**, not a probe solve: run one PSD separation round, and
if the LP bound delta is below a relative threshold (it is exactly 0 here),
abandon PSD for that node. That is the recommended build (§4).

---

## 4. Recommendation — GO: cost-aware relaxation policy (per-round bound-delta gate)

**The kill criterion did NOT fire.** The kill criterion was: *if every expensive
stage is bound-necessary (turning any off loses > 5 % of the root bound on ≥ half
the set), then it's a genuine compute cost and the lever is a stage-speedup.* On
the PSD/QCQP class the opposite holds — PSD loses **0 %** of the root bound on
**3/3** instances (nvs17, nvs19, nvs24). This is an over-computation / policy
problem, exactly as CUT-1 hypothesized ("discopt over-computes a tight bound").

**GO — build a cost-aware PSD (moment-cut) gate.** Mechanism (general, not
instance-keyed):

1. In the separation loop, after the **first** PSD round on a node, measure the LP
   lower-bound delta `Δ = lb_after − lb_before`. If `Δ ≤ τ · (1 + |lb_before|)`
   (τ ≈ 1e-4), **abandon PSD for the remaining rounds on that node** and mark the
   relaxer so subsequent nodes skip PSD's first round too after K consecutive
   inert nodes. This keeps PSD where it *does* pay (a real box-QP whose Shor gap
   is open) and drops it where it is inert (nvs17/19/24), with no problem-name
   special-casing.
2. Pair with `univariate_square` under the same gate (measure its per-round
   bound delta), since it is the second inert stage on this class.

**Measured expected wall win (this panel):** nvs17 37.8 → ~18 s (2.1×); nvs19
gets 7.4× the node throughput at the same bound; **nvs24's root converges
(56.7 → 6.7 s root) and it goes from no-incumbent to 335 nodes / feasible** — the
CUT-1 pathology is resolved. No root-bound loss (the bound is set by McCormick+RLT,
which the gate never touches).

**Verification regime (BOUND-CHANGING — a cut policy that skips valid cuts changes
the dual bound; per CLAUDE.md §5 / cert-gap-plan §0.2.6):** feature flag,
default-OFF. Differential bound test (gated bound ≥ oracle-safe AND the *skip*
never *raises* the bound above what PSD-on would prove — skipping only-valid cuts
can only *loosen*, so the invariant is `bound_gated ≤ bound_baseline` and both
≤ incumbent) + feasible-point sampling (skipping cuts cannot cut a feasible point,
so this is automatically safe — the gate only *removes* rows). `incorrect_count ≤ 0`
on the cert panel + 3 green nightlies before any default flip. Note: because the
gate only *drops* cuts, it can never produce an unsound bound — the worst case is a
looser bound, caught by the differential test — which makes this a low-risk
bound-changing change.

### Secondary engineering target (independent, also GO-worthy)

The nvs24 cProfile exposes a **second, orthogonal** sink the PSD gate does not
address: the per-node relaxation is solved as an **integer MILP**
(`solve_milp_py`, ~10 s/node) because the McCormick relaxation retains integrality
and the pure-LP warm fast path (`milp_relaxation.py:326`,
`self._integrality is None`) is bypassed on this dense-integer-QP class. Dropping
integrality at the node relaxation (the LP bound is still valid — the outer tree
enforces integrality; this is exactly the `_lp_node_bound` design that already
applies elsewhere) would replace three 10 s MILP solves with sub-second LP solves.
This is a **bound-neutral** engineering task (the LP relaxation of the MILP is a
valid lower bound; node_count/objective must stay exactly unchanged where the MILP
already returned the LP-optimal vertex) and is the right follow-on once the PSD
gate lands. It is *not* the JAX envelope rebuild (measured 1.9 s, inert) and *not*
the OBBT loop (that's gear4's separate lever).

---

## 5. Falsifications recorded (measurement beats plan, cert-gap-plan §0.4)

1. **"nvs24's non-convergence is a high-arity multilinear hull cost"** (THRU-1
   brief's alternative hypothesis). *Falsified:* `separate/multilinear = 0.0006 s`
   on nvs24. The sinks are `separate/psd` (21 s), `separate/univariate_square`
   (43 s), and the integer-MILP node solve (30 s).
2. **"the JAX `build_milp_relaxation` envelope rebuild (Phase-4/5) is the root
   throughput lever"** (bottleneck-profile-2026-07-05 Phase-4/5 note, as applied
   to this class). *Falsified for nvs24:* `build_milp_relaxation` cumtime = 1.85 s
   (~3.5 % of profiled wall). The envelope rebuild is not the sink on the
   PSD/QCQP class; the Rust LP/MILP solver is (94 %).
3. **"PSD moment cuts tighten the root bound near the Shor SDP bound on the nvs17
   class"** (the `_separate_psd` docstring's nvs17 −1221 example / the P3 root cut
   pool rationale). *Falsified as a bound lever here:* on nvs17/nvs19/nvs24 the
   reported `root_bound` is bit-identical with PSD on, off, per-node, or at a
   150-round root pool. Whatever the standalone spectral prototype tightened, it
   does **not** move the certified root bound the solver actually reports on these
   three instances — while costing ~60 % of wall. (The base McCormick+RLT root
   bound is already the CUT-1 win.)

---

## 6. Artifacts

- `discopt_benchmarks/scripts/root_stage_ablation.py` — reusable per-stage cost +
  value-vs-cost ablation harness, with the default-OFF `--gate-psd` prototype
  (findings-only; not wired into any default path). Configs: `base`, `nopsd`,
  `nosq`, `nopsd_nosq`, `nomultilinear`.
- `discopt_benchmarks/scripts/profile_instance.py` (pre-existing, T0.3) — used for
  the cProfile Rust/pounce bucket attribution on nvs24.
- Raw per-instance JSON was produced in the session scratchpad (regenerate with
  the harness above; every table names its instance + config).

---

## 7. THRU-2b follow-up — the "integer-MILP node solve" was a mis-attribution (2026-07-06)

**Verdict: RE-SCOPE.** THRU-1's "Secondary engineering target" (§4) claimed the
nvs24 per-node relaxation is solved as an **integer MILP** (`solve_milp_py`,
~10 s/node) because the McCormick relaxation "retains integrality" and the pure-LP
warm fast path (`milp_relaxation.py:326`) is bypassed. Re-measuring on
`origin/main` (post R2/#524) **falsifies the mechanism**: the node relaxation is
already a **pure LP** — there is no integer node B&B on any default (or even
`node_bound_mode=milp`) configuration.

**What actually happens on nvs24 (instrumented, release build):**

1. Integrality is dropped at every node. `mccormick_lp.py:738` routes on
   `self._rlt_applicable or self._lp_node_bound`; on nvs24 **both** are true
   (`node_bound_mode="lp"` is the default since #257, *and* RLT applies to its
   quadratic constraints), so `milp._integrality = None` unconditionally. Patch
   counters over a 40 s solve: **13/13 node solves have `integrality is None`; 0
   integer-MILP node solves.** Forcing `DISCOPT_NODE_BOUND_MODE=milp` does not
   change this (RLT still forces the LP path). *There is no config under which
   nvs24's node relaxation is an integer MILP.*
2. The `solve_milp_py` calls THRU-1's cProfile attributed to an "integer-MILP node
   solve" are **pure LPs** (`int_cols` empty, `nint=0`), reached through the
   *fallback* in `MilpRelaxationModel.solve` (`milp_relaxation.py:375`): the warm
   sparse simplex (`solve_lp_warm_csc_py`) breaks down with
   `status=numerical` **at `iters=0`** (a factorization/setup failure, *not* an
   iteration-limit — raising `max_iter` to 1e6 does not help) on 2–3 hard,
   ill-conditioned lifted node LPs; the equilibrated sparse retry
   (`_solve_lp_warm_equilibrated`) also returns `numerical`; the code then falls
   through to the robust dense `solve_milp` (LP presolve → decides them), at
   2–11 s each. So the 30 s / 3-call bucket = **the dense-cold LP fallback for LPs
   the warm simplex cannot factorize**, not integer branching.
3. Across the class: nvs17 **0** dense fallbacks (warm + equilibrated retry rescue
   all 9 hard LPs of 1129), nvs21 **0**, nvs19 **2**, nvs24 **2–3**. The fallback
   is rare and only bites the nvs19/nvs24 tail.

**Falsification recorded:** the §4 "Secondary engineering target" premise ("swap
the integer-MILP node solve for an LP → 10 s → sub-second, bound-neutral") is
**void** — the swap it proposes is already in place. The residual sink is a
*numerical robustness* gap in the warm sparse simplex (no LP presolve on that
path), which the dense `solve_milp` covers correctly but slowly. Making those 2–3
LPs sub-second requires **LP presolve on the warm sparse simplex** (a Rust change
to `lp/simplex`, not in THRU-2b scope) — filed as a re-scoped follow-on. It is
*not* the nvs24 wall lever regardless: nvs24 is dominated by PSD + univariate-square
separation (§2), addressed by the THRU-1 PSD gate, not by these fallback LPs.

**Shipped (the one sound, in-scope, bound-neutral sub-fix):** when the fallback LP
has **no integer columns** (a genuine LP), `milp_simplex.solve_milp` now runs
`solve_milp_py` with the integer-search machinery **off** (`root_cuts=0`,
`cut_rounds=0`, `gmi_cuts=False`, `heuristics=False`, `strong_branch=False`). With
no integers none of that machinery can fire (GMI needs a fractional integer;
heuristics round nothing; nothing to branch on), so it is pure overhead on the
root LP that is the entire answer. Bound-neutral by construction. Measured on the
two nvs24 fallback LPs: 10.9 s → 5.5 s and 2.4 s → 2.2 s. Cert-baseline
**NEUTRAL** (41/41 instances, node_count exactly unchanged, |Δobj| = 0). It does
**not** hit the THRU-2b "sub-second node solve" acceptance — that target rested on
the false integer-MILP premise; the LP itself is genuinely hard for the warm path.

**Artifacts:** `python/discopt/solvers/milp_simplex.py` (the pure-LP short-circuit).
