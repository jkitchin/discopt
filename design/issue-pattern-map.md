# Issue → relaxation-pattern map

A triage of open issues against the relaxation/cut **pattern catalog**
(`design/relaxation-patterns.md`). For each pattern-addressable issue: the
nonconvex structure, the pattern(s) that apply, current coverage, the residual
gap, and a comment-ready recommendation. Issues that are pure search /
LP-throughput / infrastructure are listed at the end as *not* pattern-addressable.

Status key: **covered** (a done pattern applies), **partial** (atom covered, gap
elsewhere), **new-pattern** (needs a pattern not yet implemented), **search/infra**
(not a relaxation-pattern problem).

---

## Pattern-addressable issues

### #15 — Gas-network Weymouth chain — **covered**
- Structure: `f^2 = C(p_in^2 − p_out^2)` chain + compressor ratio + power obj.
- Pattern: **P5 square-difference network** (+ the auto recognizer).
- Status: done — `recognize_and_inject` closes 66.7%→0% automatically.

### #189 — fractional-power / general nonconvex (cvxnonsep_nsig30, clay0303hfsg, chakra, demo7) — **partial**
- Structure: signomial / fractional-power terms `x^p` and products `∏ x_j^{a_j}`.
- Findings (measured): the **univariate `x^p` envelopes are already tight** in the
  engine across regimes — `x^0.5, x^0.7` (concave), `x^1.5, x^2.5, x^-1, x^-0.5`
  (convex), all sound, tight convex hull. So the issue's "tighter `x^p` envelope"
  lever is **largely already in place** for univariate powers.
- Residual gap: (a) **multivariate signomial** products `∏ x_j^{a_j}` relaxed
  compositionally are loose → need the **G-convex / log-domain joint hull**
  (P11/P14, new); (b) **OBBT** on the variables feeding power terms (→ #208) to
  shrink the boxes the tight envelopes act on.
- Recommendation: the univariate part is closed; route remaining looseness to the
  signomial-hull pattern (P11) and OBBT (#208). Add `cvxnonsep_nsig30` /
  `clay0303hfsg` as `gap_certified` regression tests (both vendored).

### #188 — kall_congruentcircles_c51: Euclidean separation — **partial / search**
- Structure: bilinear objective `x_i·x_j` (**P3**) + separation constraints
  `(x_i−x_j)^2 + (y_i−y_j)^2 >= 1` (convex-LHS `>=`, nonconvex feasible region).
- Pattern lever: expand `(x_i−x_j)^2 = x_i^2 − 2x_ix_j + x_j^2` and apply the
  **P3 bilinear hull** + RLT on the squared differences (a *squared-difference /
  Euclidean-separation* pattern, new). Tighter bilinear relaxation is one of the
  owner's named levers.
- Honest caveat: the dominant gap is **combinatorial multimodality** (global
  search / feasibility pump), not a single missing envelope. Pattern work helps
  the bound but won't alone reach the global optimum.

### #114 — signomial (mixed-sign) global solver (SGO) — **new-pattern**
- Structure: signomial terms `±c·∏ x_j^{a_j}` with **mixed signs**.
- Pattern: **P7 posynomial** covers only the positive (posynomial) terms. Signed
  signomials are difference-of-convex in the log domain (posynomial − posynomial);
  the negative terms need a concave-overestimator treatment. → new **signed
  signomial** pattern (P11).
- Recommendation: implement signed-signomial decomposition on top of P7's
  log-convex monomial value; cite Lundell & Westerlund (SGO).

### #181 — G-convexity / convex-transformable relaxations — **new-pattern**
- Structure: terms convex after a variable transform (log, power, exp).
- Pattern: **P7** is the canonical special case (log transform of monomials);
  the issue asks for the *general* framework (Khajavirad-Michalek-Sahinidis 2012).
  → general **G-convexity detection + transform** pattern (P11/P12).
- Recommendation: generalize the log-domain pattern to a transform registry; the
  log-curvature lattice (#115) is the detection layer for it.

### #115 — per-expression log-curvature lattice — **new-pattern (detection)**
- Structure: detect log-convex / log-concave subexpressions.
- Pattern: the **detection layer** that powers P7/P11 (which terms are GP-convex).
  Mirrors the existing convexity lattice but in log-space.

### #116 — y-space (log) branching/bound-tightening for GP — **search/infra (GP)**
- The branching/OBBT counterpart of the GP patterns; pairs with P7/P11 but is a
  presolve/branching feature, not an envelope.

### #231 — complementarity `x·y = 0`, `x,y >= 0` — **new-pattern**
- Structure: complementarity / disjunctive `x·y = 0`.
- Pattern: new **complementarity** pattern — `xy = 0` with `x,y>=0` is the
  disjunction `x=0 ∨ y=0`; the bilinear `w=xy` McCormick hull plus the equality
  `w = 0` gives a valid relaxation, and the disjunction drives branching.
- Recommendation: add an IR node (per the issue) + the `w=xy, w=0` cut; provable.

### #187 — autocorr_bern: products of binaries — **new-pattern (exact)**
- Structure: products of **binary** variables `∏ b_i` (Fortet/Glover).
- Pattern: new **Fortet/Glover linearization** — *exact* (not a relaxation):
  `z = ∏_i b_i ⟺ z <= b_i ∀i, z >= Σ b_i − (n−1), z >= 0`.
- Recommendation: highest-value clean win — exact, provable, and avoids the
  full-DAG Jacobian XLA blowup the issue flags. Implement first.

### #219 — transcendental over unbounded/implied-bound domains — **search/infra (bounds)**
- Structure: `exp/log/...` whose argument has no finite bound → dropped from LP.
- Pattern lever: **FBBT / implied-bound propagation** to supply a finite argument
  box so the existing envelopes (P1/P2) can fire. A bound-propagation pass, not a
  new envelope.

### #208 — OBBT-on-auxiliaries — **search/infra (presolve, amplifies all envelopes)**
- Rebuild McCormick from OBBT-tightened aux bounds. Not a pattern itself, but it
  *tightens every envelope pattern* (P3/P4/P5/P7…) by shrinking boxes. Directly
  amplifies #189, #188.

---

## Not pattern-addressable (search / LP throughput / infrastructure)

- **#236** GNN branching (ML training/weights).
- **#229** spatial-B&B node LP ~750× slower than HiGHS (FT-update / factorization).
- **#196** ex1252 gap closure (node throughput, SOS1 selectors).
- **#194** spatial branching on nonlinear-term integers; unbounded-var policy.
- **#186** ex1263a/trim-loss: per-node McCormick MILP solve speed.
- **#163** benchmark harness setup.
- **#87** restore test coverage to 85%.
- **#27** user-defined functions (JuMP.register equivalent).

---

## New patterns this review surfaces (added to the catalog)

| Pattern | Issues | Kind | Status |
|---|---|---|---|
| P10 Fortet/Glover binary-product linearization | #187 | exact | implement first |
| P11 Signed signomial / G-convex transform | #114, #181, #189 | relaxation | partial (posynomial done) |
| P12 Complementarity `x·y=0` | #231 | relaxation + disjunction | new |
| P13 Squared-difference / Euclidean separation | #188 | relaxation (RLT) | new |
| P14 Multivariate signomial joint hull | #189 | relaxation | roadmap |
| (detection) log-curvature lattice | #115 | analysis | roadmap |
| (presolve) OBBT-on-aux, transcendental FBBT | #208, #219 | presolve | amplifier |

## Fix-and-comment workflow

As each pattern lands: (1) implement + prove + certify in `symbolic/patterns.py`
and `test_patterns.py`; (2) wire its detector into `cut_recognizer` where it
auto-fires; (3) comment on the mapped issue(s) with the measured before/after and
a soundness note. This document is the index to drive that loop.
