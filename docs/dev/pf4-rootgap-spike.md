# PF4 SPIKE — residual root-gap relaxation classes

Status: **FULL ITEM → KILL on SOUNDNESS (2026-07-14).** The spike's GO verdict below
is **superseded** — see §6. The proposed `GM ≤ LMTD ≤ AM` envelope is UNSOUND for the
term the heatexch model actually contains, because the spike sampled the ε-FREE mean
`(a−b)/log(a/b)` while the model term is `(a−b)/log(a/(ε+b))` with a **pole inside the
box**. No relaxation change was landed; the finding + a guard test
(`python/tests/test_pf4_lmtd_epsilon_pole.py`) were.

Status (original): **DONE → GO (scoped: LMTD / logarithmic-mean envelope class)** (2026-07-14).
Measurement + isolated prototype only; lives in an **isolated worktree**, NOT pushed.
Companion: `docs/dev/sota-proof-plan.md` §2 PF4; `docs/dev/pf3-branching-spike.md`
(redirect that pointed here); `docs/dev/nvs05-division-lever-entry-2026-07-11.md` (F17,
nvs05 root-lever KILL); `docs/dev/avm-canonicalization-plan.md` §10 (residual classes).

Env: `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, `PYTHONPATH=<worktree>/python`, worktree
reset to branch tip `834743f` (PF1 ON), `_rust.so` copied in from the shared build
(same commit → identical). Root bound = discopt's own engine,
`MccormickLPRelaxer(model).solve_at_node(root_box)` — a single deterministic
relaxation solve (contention-robust). Probes: `docs/dev/lr0_probe/pf4_rootgap.py`,
`/tmp/lmtd_iso.py`, `/tmp/lmtd_env.py` (soundness). Oracle optima from
`reports/v_baron_remeasure_2026-07-07T11-06-06.json` (`known`).

## 1. Root-gap table (measured)

min sense; `rel_gap = (known_opt − root_bound)/|known_opt|`. Root bound is the
McCormick LP over the RAW model box (the deterministic root-tightness signal;
FBBT/presolve run before the real B&B root, so the solver's post-FBBT root can be
tighter — but this isolates the *envelope* strength, not FBBT).

| instance | root_bound | known_opt | abs_gap | rel_gap | panel status (PF1-ON) | blocked-by |
|---|---:|---:|---:|---:|---|---|
| cvxnonsep_nsig30 | 30.47 | 130.63 | 100.16 | **76.7%** | **OPTIMAL** 165 n / 15.6 s | node B&B closes it → NOT root-gap |
| heatexch_gen1 | 38 184 | 154 896 | 116 712 | **75.3%** | TIMEOUT 3 n, **no incumbent** | ROOT (LMTD un-enveloped) + per-node cost |
| cvxnonsep_psig40r | 40.0 | 86.55 | 46.55 | 53.8% | **OPTIMAL** 95 n | NOT root-gap |
| cvxnonsep_psig30 | 51.87 | 79.00 | 27.13 | 34.3% | **OPTIMAL** 89 n | NOT root-gap |
| heatexch_gen3 | 43 888 | 64 844 | 20 956 | 32.3% | TIMEOUT 1 n, **no incumbent** | ROOT (LMTD) + per-node cost |
| heatexch_gen2 | 543 500 | 635 839 | 92 343 | 14.5% | TIMEOUT feasible 676 411 | per-node cost |
| tspn05 | 167.79 | 191.26 | 23.47 | 12.3% | feasible (HAS opt incumbent) | bound-climb / per-node cost |
| nvs05 | **unbounded\*** | 5.471 | — | — | feasible 215 n | FBBT + spatial B&B (F17 KILL) |
| tspn08 / tspn10 / tspn12 | **unbounded\*** | ? | — | — | — | larger tspn |

\* raw-box root LP is **unbounded**: nvs05's ratio aux vars x4..x7 and tspn's vars
have (−∞,+∞) / semi-infinite bounds; a finite bound exists only after FBBT.

## 2. Central finding — root-gap magnitude does NOT predict blocking

The PF4 premise ("unproved instances time out because the ROOT bound is frozen far
from the optimum") is **only partially true**, and the naive reading is falsified:

- **The two LARGEST root gaps (cvxnonsep_nsig30 76.7%, psig40r 53.8%) are on
  ALREADY-PROVED instances.** These are separable sum-of-signomials; the B&B closes
  even a 77% root gap in 89–165 nodes / <16 s. Root-gap size is not the blocker.
- **nvs05** (the PF3 redirect target): raw-box root LP is *unbounded*, and F17
  already exhaustively KILLED a root division/ratio + OBBT lever (best sound root
  0.78 of 5.47; needs thousands of spatial-B&B nodes). Not a root-envelope instance.
- **tspn05** already holds the optimal incumbent (191.26); its 12.3% root gap and
  37 nodes/30 s make it a bound-climb / per-node-cost instance, not a frozen root.

**The one class where a large root gap AND a blocked proof coincide is heatexch**
(gen1 75.3%, gen3 32.3%, gen2 14.5%) — and the mechanism there is not "loose", it
is a **relaxation hole** (§3).

## 3. Structural diagnosis — heatexch `general_nl` = LMTD, and it is UN-enveloped

heatexch_gen1/2/3 term classification: `bilinear` (32/40/200) + `general_nl`
(24/53/205). The `general_nl` division terms are, verbatim:

```
(x20 - x21) / log( x20 / (1e-06 + x21) )
```

i.e. the **Log-Mean Temperature Difference** `LMTD(a,b) = (a−b)/log(a/b)` — the
classic nonconvex driving-force term of heat-exchanger-network synthesis, and
exactly PF4's named "general linear-fractional A(x)/B(x) (heatexch class)".
(gen2/gen3 add `**` area-cost powers.) The LMTD inputs are bounded **[10, +∞)** —
semi-infinite, FBBT-only.

**The exact envelope weakness (measured, `/tmp/lmtd_iso.py`):** maximizing the LMTD
output over its box in the current relaxer returns **`status = unbounded`** on every
box tried. `build_milp_relaxation` **never references `terms.general_nl`** (grep:
one docstring mention, zero code) — so a general_nl LMTD term gets **no cut at
all**; its lifted output is free to +∞. Because heatexch's objective (minimise area
cost, area ∝ Q/(U·LMTD)) is *decreasing* in LMTD, an unbounded-above LMTD lets the
LP drive area/cost arbitrarily low → the dual bound is capped far below the optimum.
That is the 75% root gap, and it is a *hole*, not mere looseness (contrast nvs05,
whose terms ARE enveloped, just loosely — F17).

## 4. Prototype — sound logarithmic-mean envelope (`/tmp/lmtd_env.py`)

The logarithmic mean obeys `GM ≤ LMTD ≤ AM`, i.e. `sqrt(ab) ≤ (a−b)/log(a/b) ≤
(a+b)/2`. Prototyped as valid cuts on a lifted output `w`:

- **Over-estimator (the decisive one):** `w ≤ (a+b)/2` — **linear**, exact on the
  diagonal a=b. Turns the unbounded output into a bounded one:

  | box | current relaxer max | with AM cut | true LMTD max | over-est sound? |
  |---|---|---|---|---|
  | a[10,500] b[10,500] | **unbounded** | 500.0 | 500.0 | yes (exact) |
  | a[10,100] b[10,100] | **unbounded** | 100.0 | 100.0 | yes (exact) |
  | a[50,400] b[20,120] | **unbounded** | 260.0 | 232.6 | yes (12% loose) |
  | a[10,300] b[10,80]  | **unbounded** | 190.0 | 166.4 | yes (14% loose) |

- **Under-estimator:** `w ≥ secant_sqrt( McCormick_lower(a·b) )` — McCormick lower
  envelope of `p=a·b`, then the concave secant of `sqrt` over `[a_lo·b_lo, a_hi·b_hi]`.
  Sound (concave secant lies below).

**Soundness (zero-tolerance gate — PASSED):**
- AM over-estimator: **0 / 200 000** feasible samples had `LMTD > AM` (worst margin
  `AM−LMTD = +3.7e-9`, i.e. never cut).
- GM-secant under-estimator: **0 / 500 000** had `under > LMTD` (worst margin < 0 on
  every box).
- **A soundness bug was caught and rejected:** the GM *tangent* (linearising the
  concave `sqrt(ab)` by a tangent plane) is UNSOUND — on box a[10,300] b[10,80] it
  forced the LP min to 14.3 > the true min 10.0, **cutting a feasible point**. The
  tangent of a concave under-shape lies ABOVE it; only the **secant/chord** is a
  valid under-estimator. The shipped envelope must use the secant. (This is exactly
  the "a tighter envelope must never cut a feasible point" hard gate doing its job.)

## 5. Verdict — **GO (scoped)**: build the LMTD / logarithmic-mean envelope class

Per the plan's GO criterion ("a sound class envelope closes a gap enough to
plausibly gain a proof **or materially tighten a class**"):

- It **materially tightens a class**: the current relaxation places **no upper bound
  whatsoever** on every LMTD term; the AM cut is *necessary and sufficient* to give
  the whole heat-exchanger LMTD class any finite dual contribution. This is closing a
  genuine relaxation hole, sound over 700 k feasible samples.
- It is a **class fix, not an instance fix**: LMTD `(a−b)/log(a/b)` recurs across the
  MINLPLib heat-exchanger-network family (heatexch_gen1/2/3 here; the broader
  hxn/gtm/process-synthesis instances in the full corpus), matching CLAUDE.md rule 2.

**Honest caveats (binding for the full item):**
1. This will **not by itself flip a heatexch PROOF on the vendored panel**: gen1/gen3
   are co-blocked by **per-node cost** (1–3 nodes explored in 30–60 s ≈ 17 s/node —
   PF2 territory; corroborates PF1's "bchoco/heatexch stuck *before* reduction can
   bite"). The envelope removes the root hole; PF2 must make nodes cheap enough to
   cash it in. The two levers are complementary, on disjoint layers.
2. It does **nothing** for nvs05 (F17 KILL — needs spatial-B&B throughput) or tspn05
   (already has the optimum; bound-climb-limited). Do NOT scope those into PF4.
3. The isolated before/after (unbounded → tight & sound) is proven; the **aggregate
   heatexch_gen1 root-bound jump is not yet measured** because it requires the lift
   infrastructure that IS the build (the builder drops general_nl today). Measure it
   as the item's entry experiment before flipping default-ON.

### What the full PF4 item would build
1. **`python/discopt/_jax/term_classifier.py`** — recognise the logarithmic-mean /
   LMTD pattern `(a−b)/log(a/(ε+b))` (and its `log`-of-ratio siblings) into a new
   `log_mean` term category (alongside `ratio_of_products`), so the builder can see
   it instead of dumping it in the un-relaxed `general_nl` bucket.
2. **`python/discopt/_jax/milp_relaxation.py`** — for each `log_mean` term, lift the
   output column `w` and emit: the AM over-estimator `w ≤ (a+b)/2`; the GM **secant**
   under-estimator (McCormick(a·b) → concave-secant sqrt); optionally tangent-plane
   over-estimators of the concave LMTD for extra tightness. All rows outward-rounded.
3. Gate: PF0 differential (fixed-box bound never LOWER, never CROSSED) +
   feasible-point sampler (0 cuts — the GM-tangent trap above must stay rejected) +
   panel outcome; default-ON in-branch per §0.3 once green; ledger row.

## Reproduce
```bash
cd <worktree>; source /home/user/discopt/.venv/bin/activate
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$PWD/python
python docs/dev/lr0_probe/pf4_rootgap.py          # root-gap table
python /tmp/lmtd_iso.py                           # LMTD output unbounded in current relaxer
python /tmp/lmtd_env.py                           # AM/GM envelope + feasible-point soundness
```

## 6. FULL-ITEM FALSIFICATION — the ε-pole makes AM/GM UNSOUND (2026-07-14, KILL)

The full item ran the entry experiment against the term the model **actually
contains** and the envelope collapsed on the hard soundness gate. Recorded here in
`performance-plan.md` §6 house style.

**Hypothesis (spike):** `GM(a,b) ≤ LMTD(a,b) ≤ AM(a,b)`, so `w ≤ (a+b)/2` (AM) and
`w ≥ chord(√(ab))` (GM secant) are sound outer rows for the lifted LMTD aux `w`.

**The measurement that killed it.** The spike sampled the ε-FREE logarithmic mean
`LMTD₀(a,b) = (a−b)/log(a/b)` (see `lmtd_iso.py`/`lmtd_env.py`: `lmtd()` uses
`log(a/max(b,1e-12))`, no ε). `LMTD₀` genuinely obeys `GM ≤ LMTD₀ ≤ AM`. But the
heatexch atoms are, verbatim (canonical reconstruct of `heatexch_gen1`):
`(a−b)/log(a/(ε+b))` with **ε = 1e-6** and `a,b ∈ [10, +∞)`. The `+ε` does **not
remove** the LMTD singularity — it **moves** it from the diagonal `a=b` to the line
`a = ε+b`, which lies strictly **inside** the box (e.g. `a=10.000001, b=10`). There
the denominator `log(a/(ε+b))` crosses **zero** with a non-zero numerator, so the
exact aux value `w = (a−b)/log(a/(ε+b))` is **genuinely unbounded (±∞)**. Sampling
the exact term over `a,b ∈ [10,650]²` (2 M points):

| row | violated feasible points | worst violation |
|---|---|---|
| AM `w ≤ (a+b)/2` | **3 211 / 2 000 000** | `w−AM = +5.10` (near `a=b=494.5`, `a−b≈1e-4`) |
| GM `w ≥ √(ab)` | **3 971 / 2 000 000** | `GM−w = +3.78` (other side of the pole) |

Both rows **cut feasible points** — the worst class of bug (false-infeasible / lost
optimum). A "just require the denominator sign-definite" gate is **insufficient**:
on a box with `a > ε+b` throughout (margin 0.0057) AM still had 201 violations, and
on a box hugging the pole (margin ≥ 2e-6, still `log>0`) AM was violated at
**417 707 / 3 000 000** points (worst `+12.4`). The margin AM needs is quantitative
and box-dependent, and it fails exactly at the small-approach (pinch) region where
good heat-exchanger-network solutions live.

**Why the "hole" is not a hole.** Because `w` is truly unbounded on any
pole-straddling box, the current relaxation's "maximise `w` → unbounded" is the
**SOUND** answer, not a fixable weakness. The AM over-estimator (the only one that
could tighten heatexch's dual bound, since area-cost minimisation drives `w` large)
is precisely the unsound one; the sound half (GM under-estimator) does not bind the
dual in the relevant direction, so even shipping only GM yields **zero** root
improvement. Entry-experiment root bounds are unchanged from §1 (gen1 38 184 /
75.3%, gen2 543 496 / 14.5%, gen3 43 888 / 32.3%).

**Landed:** nothing in the relaxation. A guard test
`python/tests/test_pf4_lmtd_epsilon_pole.py` pins the falsification (in-box pole;
AM/GM unsound; sign-gate insufficient; current relaxation cuts no near-pole feasible
point) so an ungated LMTD envelope cannot be re-introduced silently.

**If PF4 is ever revisited:** the only sound route is to relax the term as written
over pole-EXCLUDED sub-boxes with a *quantitative* AM-validity margin (not a sign
gate), which is inert at the root and of unmeasured deep-node value — a new spike
with its own entry experiment, not this envelope. Do **not** relax it as the ε-free
mean: that relaxes a different function and reintroduces the cut.

### Reproduce the falsification
```bash
cd /home/user/discopt; source .venv/bin/activate
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
python -m pytest python/tests/test_pf4_lmtd_epsilon_pole.py -q
```
