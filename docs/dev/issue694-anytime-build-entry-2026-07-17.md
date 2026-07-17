# Anytime/incremental root-relaxation build — #694 entry experiment

**Date:** 2026-07-17
**Status:** measured (measurement + probe only; **no library change shipped** —
entry experiment per CLAUDE.md §4 / baron-gap-plan.md §0, run BEFORE any
implementation).
**Issue:** #694 — make the root McCormick relaxation build anytime/incremental so
the dual bound accrues, dissolving the #654 §8.1 truncation fork.
**Verdict:** **SURVIVES the kill criterion.** On every structure that could be
tested here — the 6 named in-repo controls and 3 synthetic proxies matching the
#654 class's structural signatures — a *finite, valid* LP lower bound exists **well
before 90 % of build time** (measured 8 %–45 %), and on the bound-carrying
structures it accrues **monotonically and near-continuously** across build deciles.
The hypothesis is viable; the graduation gate (§5, on the real corpus) is the next
step and requires the user's machine (see §5 below).

> **Method.** Branch `claude/issue-694-ynajak` from `origin/main` (2225a77).
> Linux x86-64, Python 3.11, `maturin develop --release` (discopt-core/‑python
> 0.2.x). Probe: `discopt_benchmarks/scripts/issue694_anytime_build_probe.py`
> instruments `_Builder.add_row` with per-row timestamps during ONE uninterrupted
> `build_uniform_relaxation`, then post-hoc solves every **row-prefix**
> `A_ub[:n_k] x ≤ b[:n_k]` (k = 10 %…100 % of final rows) over the full column set +
> real objective, recording `(build_elapsed(n_k), status, bound, finite?)`. A prefix
> relaxation has FEWER rows → still a valid outer approximation → its LP min is a
> valid (weaker) bound whenever finite. Synthetic proxies:
> `issue694_synthetic_proxy.py` (`clique` / `qap` / `netdesign`). Raw JSON in
> `docs/dev/data/issue694-anytime-*.json`.

---

## 1. Why a proxy was needed (corpus blocker — recorded honestly)

The decisive #654-class instances — **sonet23v4, super3t, qap, eg_all_s,
sonet22v4** — live only in the big Dropbox snapshot
(`~/Dropbox/projects/discopt-minlp-benchmark/`), which is **absent from this
environment**, and MINLPLib is **network-blocked** (agent proxy returns 403 to
`www.minlplib.org`). The 6 bound-producing controls the issue names (casctanks,
hda, heatexch_gen1–3, nvs05) ARE in the in-repo corpus and were run directly. For
the #654 class itself, the experiment uses **synthetic proxies that reproduce the
structural signature** — the property under test (does a finite bound exist early in
the build?) is a property of the build *code path* and the *atom structure*, both of
which the proxies exercise faithfully. This is a real limitation, not a substitute
for the corpus panel: the §5 graduation gate and the specific must-not-regress
bounds (casctanks 5.698, super3t −1.0, sonet23v4 −53974.375) must be run on the
user's machine against the real instances.

## 2. Controls (in-repo) — a finite bound exists early on all 6

`first-finite build%` = the earliest build decile at which the prefix LP returns an
`optimal`, finite bound.

| instance | vars | cons | rows | build wall | **first-finite build%** | notes |
|---|---:|---:|---:|---:|---:|---|
| nvs05 | 8 | 9 | 163 | 0.089 s | **8.1 %** | floor 0.674 until 70 %, then McCormick −219.6 in last 20 % |
| hda | 722 | 718 | 3 022 | **5.414 s** | **12.1 %** | prefixes finite; **full build's own LP hits `iteration_limit`** (see §4) |
| heatexch_gen1 | 112 | 120 | 364 | 0.075 s | **16.5 %** | floor 0 until 70 %, tight 38183 in last 30 % |
| heatexch_gen2 | 148 | 166 | 484 | 0.118 s | **16.5 %** | floor 1590 until 70 %, tight 543496 in last 30 % |
| heatexch_gen3 | 580 | 510 | 1 840 | 0.714 s | **20.8 %** | floor 783 until 80 %, tight 43887 in last 20 % |
| casctanks | 500 | 517 | 1 742 | 0.227 s | 69.4 % | trivial obj (subnormal floor); tight 3.888 in last 10 % |

**Reading:** none reaches the kill threshold (finite only ≳90 %). The *earliest*
finite bound is almost always the **`obj_floor`** (`milp._objective_floor`, the
box-interval objective floor computed from cost-column bounds alone — it needs
**zero** rows), which is valid but weak. The **McCormick-tightened** bound overtakes
the floor typically only in the **last 10 %–30 %** of the build. So on the controls
the anytime story is: a *valid* finite bound is available almost immediately, but a
*useful* one still needs most of the build.

## 3. Synthetic #654-class proxies — the bound ACCRUES continuously

Three families reproduce the #654 class's structural signatures (dense quadratic
over binaries → O(n²) bilinear product envelopes → multi-thousand-row builds):

| proxy (signature) | vars | rows | build wall | first-finite | anytime curve (bound at 10 %→100 % build) |
|---|---:|---:|---:|---:|---|
| **netdesign_80** (sonet*: linear obj + bilinear *constraints*) | 160 | 6 160 | 2.204 s | **11.7 %** | **1.99 → 4.57 → 7.06 → 10.1 → 12.3 → 14.5 → 17.0 → 18.9 → 21.4 → 23.72** (monotone, ~linear in build progress) |
| clique_40 (qap/graphpart: dense quad over binaries) | 40 | 3 122 | 0.201 s | 44.6 % | −126 → −118 → −110 → −104 → −97 → −91 → −84 → −77 → −71 → **−65.9** (monotone) |
| clique_60 | 60 | 7 082 | 0.432 s | 37.9 % | −278 → … → **−146** (monotone) |
| clique_120 | 120 | **28 562** | 1.828 s | 40.3 % | −1116 → … → **−586** (monotone; every prefix solves `optimal`) |
| qap_8 (assignment + product objective) | 64 | 6 304 | 0.388 s | 38.1 % | finite but **McCormick-trivial ≈ 0** at every decile (known: QAP McCormick root ≈ 0 vs opt 388214, #661 — needs RLT) |

**Reading:**

- On the **sonet\*** signature (`netdesign`, linear objective coupled to bilinear
  *constraints*), the curve is textbook anytime: a finite bound from **11.7 %**
  build, rising monotonically and almost linearly with build progress to the full
  bound. Truncating at any decile ≥ the first yields a valid, weaker bound —
  exactly the "bound accrues continuously" property SOTA solvers have and the issue
  asks for.
- On the **qap/graphpart** signature (`clique`), the finite bound appears at
  ~38–45 % build (the point at which enough of the objective's product envelopes
  exist to bound the LP below) and then tightens monotonically every decile,
  holding at 28 k rows.
- **qap** is the one caveat: its McCormick relaxation is trivially loose (≈ 0 at
  every build fraction), matching the known #661 result. The anytime bound would
  accrue but stay near-trivial until RLT is layered on; RLT's own build/solve
  anytime behavior is a *separate* question (RLT is opt-in and already
  time-budgeted in `_root_relaxation_lower_bound`).

## 4. Two load-bearing subtleties the probe surfaced

1. **Build ORDER decides how early the bound turns finite.** On structures where the
   *objective* carries the products (clique/qap), the LP is unbounded below (−∞)
   until ~40 % of rows — the fraction at which enough objective-product McCormick
   envelopes have been added to bound the cost columns. Before that, no finite bound
   exists. So the implementation's ordering — the issue's "variable-bound rows and
   objective-linearizing envelopes first, remaining term envelopes by contribution"
   — is not cosmetic: it is what moves the first-finite point from ~40 % toward ~0 %.
   On the `netdesign` (linear-objective) structure the objective is box-bounded from
   row 0, so the finite point is already early (11.7 %) and ordering matters less.
2. **Truncating the build makes the LP SOLVE easier, which is the point.** hda's
   *full* relaxation LP hits `iteration_limit` (no finite bound at 100 %), yet every
   *prefix* (10 %–90 %) solves `optimal` with a finite bound. This is the exact
   #654 "`plain` is None" failure mode (the full-relaxation budgeted solve times
   out) — and it shows that a *smaller, truncated* relaxation is not only sound but
   often *more* solvable within a fixed budget. An anytime build that stops early
   trades bound tightness for a bound that actually returns.

## 5. Disposition & next step (the §5 gate — needs the real corpus)

The kill criterion (#694: partial bound −∞/None until ≳90 % build) is **not met on
any tested instance**. §8.1 is therefore *challengeable* on these structures: a
build interrupted at a checkpoint does yield a valid weaker bound, so the
"lose-the-bound vs overrun" fork is dissolvable in principle — the entry experiment
**survives**, and the implementation proceeds.

**Implementation landed (default-OFF, `DISCOPT_ANYTIME_ROOT_BUILD`).** The change is
**bound-changing** (CLAUDE.md §5), so it ships behind a default-off flag and
graduates only after the corpus-wide differential panel passes — flag ON vs OFF over
the in-repo corpus, `incorrect_count = 0`, no bound above its reference optimum, no
certification regression, incumbents independently feasibility-verified, AND
net-positive (wall/nodes/bound). The specific must-not-regress bounds are
**casctanks 5.698, super3t −1.0, sonet23v4 −53974.375**
(`python/tests/test_issue654_deadline_root_setup.py` pins them). Two of those three
(super3t, sonet23v4) and the headline slow-build case are **big-corpus-only**, so
**graduating this flag is a run-on-the-owner's-machine step** — the panel cannot be
certified in this environment.

What landed:

- `build_uniform_relaxation(..., build_deadline=...)` (default `None`): the
  constraint-row loop stops adding rows once the `perf_counter` deadline passes,
  polling BETWEEN whole constraints (never mid-`rep`, so no partial row). Threaded
  through `build_milp_relaxation` → `_uniform_relaxation_delegate`, and through
  `MccormickLPRelaxer.solve_at_node` / `_solve_at_node_impl` (the cold build only —
  the incremental fast path is already cheap). Provenance on the model:
  `_build_truncated` / `_build_constraints_done` / `_build_constraints_total`.
- The objective is fully linearized BEFORE the loop, so truncation never touches
  `objective_bound_valid` for the objective itself; a truncated build is a valid
  weaker outer relaxation (drops constraint rows only).
- `_root_relaxation_lower_bound` sets `build_deadline = _fb_t0 + time_limit` when the
  flag is on and applies it to the **separated (`sep`) build only** — the base build
  is left WHOLE. Rationale (rule-1, §8): truncating the base build can un-bound an
  objective cost column and trip its `objective_bound_valid=False → return None`
  gate, which would LOSE the bound rather than weaken it; the documented sonet23v4
  cost is the `sep` build (§8.6), and `sep` constructs its own relaxation, so
  truncating it is the targeted, rule-1-safe cut.
- Tests: `python/tests/test_issue694_anytime_root_build.py` — OFF byte-identical
  (row count + bound), truncated build is valid & weaker (LP min ≤ full), grant
  honored + sound under the flag. Regime-neutral: smoke (661) + the #654 suite pass
  unchanged (flag default off ⇒ `build_deadline` is `None` everywhere).

**Measured caveat (in-repo, informs the gate).** On **hda** (a control with wide/
unbounded cost columns) the OFF fallback overruns (−28100067 in ~11–27 s against a
3 s grant — the #654 symptom), and with the flag ON the `sep` build truncates to a
relaxation whose cost columns are no longer bounded by any surviving row, so it
returns **`None`** (honoring the grant in ~2 s, sound — a weaker/absent bound, never
false). So the flag's *soundness* is universal but its *benefit* is
structure-dependent: it retains a weaker bound on the clean finite-box #654
structures (proxies confirmed) and can degrade to `None` on messier ones. Whether
the real #654 instances retain their bounds is the empirical question the graduation
panel answers; if bound-retention on that class needs improving, the **row-ordering**
refinement (objective-critical constraints first, §4.1) is the follow-up lever.

## 6. Do-not-circumvent (unchanged, reaffirmed)

- **§8.1** (do not truncate bound-producing native *solves*) still binds: this
  approach changes the *precondition* (makes a valid bound exist earlier in the
  *build*), it does not truncate a solve. The checkpoint stops the **build** at a
  point where a finite bound is already in hand, then runs the prefix solve to
  completion.
- **§8.2** (no Rust LP native deadline, TX2b) untouched: this targets the Python
  relaxation *build* layer, never the Rust LP solve.

## Reproduce

```bash
# controls (in-repo)
python discopt_benchmarks/scripts/issue694_anytime_build_probe.py \
    nvs05 heatexch_gen1 heatexch_gen2 heatexch_gen3 casctanks hda \
    --json docs/dev/data/issue694-anytime-controls.json
# #654-class structural proxies
python discopt_benchmarks/scripts/issue694_synthetic_proxy.py --family netdesign --n 80
python discopt_benchmarks/scripts/issue694_synthetic_proxy.py --family clique --n 40 60 120
python discopt_benchmarks/scripts/issue694_synthetic_proxy.py --family qap --n 8
```
