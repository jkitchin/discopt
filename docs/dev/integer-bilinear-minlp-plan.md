# Plan: SCIP-grade support for integer-bilinear MINLPs in discopt

**Goal.** Solve the integer-bilinear MINLP class (the `ex126x` trim-loss family and
similar problems where bilinear terms have integer or implied-integer factors) to
**proven optimality**, matching the reference solver (SCIP: ex1263 → 19.6, 0.11 s).

**Proven diagnosis (this is the foundation, established experimentally).**
discopt relaxes `x_i·x_j` with one continuous McCormick envelope over a wide box;
when the factors are integers this envelope is too loose, so the relaxation's
*integer optimum* sits strictly below the true optimum (ex1263: 19.1 vs 19.6 —
confirmed by solving discopt's own relaxation with SCIP). Exact handling of the
integer products closes it: a binary-expansion linearization through discopt's
own pipeline already moves the bound 19.1 → 19.6.

**Correctness invariant (non-negotiable, applies to every step).**
Every reformulation must be *value-preserving* (identical optimum and feasible
projection); `incorrect_count == 0` on the full suite must hold after each step;
no variable is ever marked integer unless **provably** implied-integer (marking a
free variable integer can cut off the optimum — the cardinal violation).

---

## Architecture decision: reuse the MILP path via exact reformulation

SCIP does not binary-expand; it keeps the products and tightens via cuts. But
replicating SCIP's integrated nonlinear+cut+presolve loop is the largest possible
build. The **reuse-first** architecture gets the same result with far less new,
correctness-critical code:

1. Detect implied-integer factor variables (**P1**).
2. Reformulate `integer·x_j` products to their **exact linear** form — binary
   expansion of the integer factor + big-M linearization of the resulting
   `binary·x_j` products — turning the bilinear MINLP into a **pure, equivalent
   MILP** (**P2**).
3. Route that MILP through discopt's **existing** MILP solver (`_solve_milp_bb`),
   whose cover/clique/Gomory/MIR cuts and primal heuristics are already wired
   (**P3**).

This reuses discopt's MILP machinery instead of building new bilinear cuts. The
binary×var big-M linearization is exact (standard, textbook-sound), so the MILP
is a true equivalent — its optimum *is* the MINLP optimum.

Trade-off: more variables than SCIP's cut-based approach. Mitigated by expanding
the *shared* factor (fewer variables) and gated by **P0** below, which proves the
MILP is small enough for discopt's solver before any further investment.

---

## Cross-cutting verification harness (build first, used by every step)

`python/tests/test_integer_bilinear.py` plus a reusable `utils` oracle:

1. **Reference-solver oracle.** A helper `reference_optimum(nl_path) -> float`
   that shells SCIP (and BARON as a second witness) on the original `.nl` and
   returns the certified optimum. SCIP/BARON are installed locally
   (`/opt/homebrew/bin/scip`, GAMS BARON). Used as ground truth.
2. **Value-preservation checker.** `assert_equivalent(model_a, model_b)`: solve
   both (discopt and/or SCIP-on-export) and assert identical optima within
   tolerance, AND that a known optimal point of one is feasible in the other.
3. **Instance basket.** ex1263–1266 (ground truth 19.6 / 10.3 / 10.3 / 16.3 from
   SCIP), plus 3–4 *declared*-integer bilinear instances (constructed) for testing
   P2 independently of P1.
4. **Regression gate.** `pytest -m "correctness and regression"` with
   `incorrect_count == 0` is run after **every** step; a step is not "done" until
   it is green.

---

## P0 — Architecture validation (de-risk before building anything)

**Why first.** The whole plan rests on "exact reformulation → discopt's MILP
solver closes it fast enough." Prove that before investing in P1/P2 hardening.

**Do.** Extend the existing `integer_product_reform.py` prototype to emit a
*pure MILP*: after binary-expanding the integer factor, lift each `e_k·x_j`
product to an aux variable `v` with the exact big-M rows
(`v ≤ U·e_k`, `v ≤ x_j`, `v ≥ x_j − U·(1−e_k)`, `v ≥ 0`) so **no bilinear term
remains**. On ex1263 (factor vars manually marked integer), confirm discopt
classifies it as a MILP and `_solve_milp_bb` solves it to 19.6.

**Verify correct.**
- The reformulated model has zero nonlinear terms (`classify_nonlinear_terms`
  returns empty bilinear/general lists).
- SCIP on the exported reformulated `.lp` returns 19.6 (exactness of the big-M).
- discopt's `_solve_milp_bb` returns status `optimal`, objective `19.6`,
  `incorrect_count == 0`.

**Test.** ex1263 + the 3 declared-integer constructed instances solve to their
reference optima; record node count and wall time (the go/no-go for the
var-count trade-off).

**Decision gate.** If discopt's MILP solver closes ex1263 in reasonable time →
proceed. If it is far too slow even on the exact MILP → the bottleneck is
discopt's MILP solver itself (cuts/primal), and P3 becomes the priority; revisit
before P1.

---

## P1 — Provably-sound implied-integer detection

**Spec.** New presolve pass `python/discopt/_jax/implied_integer.py`:
`detect_implied_integers(model) -> set[(var_index, elem)]` returning only
variables that are **provably** integer at every feasible point.

**Sound sufficient conditions (only these; conservative by design).**
1. *Integer-defining equality.* `a·x = Σ cᵢ zᵢ + d` with `a, cᵢ, d ∈ ℤ`, every
   `zᵢ` integer/binary, and `a | cᵢ` for all i and `a | d` ⇒ `x` integer.
2. *Unit-coefficient bound pair against integer expression with integer gap* only
   when it provably forces integrality (it generally does **not** — e.g.
   `b ≤ x ≤ b+4` does not; such cases are **rejected**, matching the finding that
   ex1263's range links alone are insufficient).
3. *Column-integrality (TU-style).* `x` appears only in rows with integer data and
   the relevant submatrix is provably such that every basic value of `x` is
   integer. Implement only the safe, checkable special cases first.

Anything not provably integral is **left continuous**. Under-detection is safe
(just misses a tightening); over-detection is a correctness bug.

**Verify correct.**
- *Soundness (the critical check):* for every variable the detector marks, an
  automated test re-solves the instance with that variable *constrained* integer
  (via SCIP on export) and asserts the **optimum is unchanged**. Any change ⇒ the
  detector is unsound ⇒ fail. This empirically backstops the proof for every
  basket instance.
- *Coverage:* compare the count to SCIP's `implints` line (ex1263: 20). Fewer is
  acceptable (conservative); more than SCIP, or any var SCIP did not mark, is a
  red flag to investigate.
- *No-op safety:* on models with no implied integers, returns `∅` and changes
  nothing.

**Test.** Unit tests for each sufficient condition (positive + negative cases,
including the `b ≤ x ≤ b+4` *negative* case that must NOT be detected). End-to-end:
detector finds ex1263's structure; the soundness re-solve passes.

---

## P2 — Exact integer-bilinear → MILP reformulation (harden the P0 prototype)

**Spec.** Promote `integer_product_reform.py` to a production pass:
`expand_integer_products(model) -> model` (already drafted), extended with the
big-M lifting from P0 and the following hardening.

**Hardening items.**
- *Shared-factor selection.* Expand the factor appearing in the **most** products
  (e.g. the 4 knives, 20 bits) rather than the smaller-range factor (16 content
  vars, 48 bits) — pre-count product participation, minimize added variables.
- *Bit-count correctness.* `nbits = ⌈log₂(hi−lo+1)⌉`; the linking equality plus the
  variable's own upper bound exclude over-range bit combinations (validated: the
  ex1263 knives reach 11, which the earlier 3-bit bug truncated — regression test
  this exact case).
- *Trigger.* Run only when `detect_implied_integers` (P1) ∪ declared-integer
  factors yield an integer-factor bilinear term; gate behind the convexity check
  (nonconvex path only), mirroring `factorable_reform`.
- *Idempotence / no-op.* Returns the model unchanged when nothing applies.

**Verify correct.**
- *Value preservation:* `assert_equivalent(original, reformulated)` on every
  basket instance — identical optimum, and the original's optimum point maps to a
  feasible reformulated point.
- *Exactness:* the reformulated model's LP/relaxation optimum equals the true
  optimum on ex126x (19.6, …) — no McCormick gap remains.
- *Bit-expansion regression:* a unit test with an integer factor whose range
  needs ≥ 4 bits (the "knife=11" class) confirms the product value is exact.

**Test.** ex126x + declared-integer instances reach reference optima through
discopt; full regression suite green (`incorrect_count == 0`).

---

## P3 — Routing + MILP-solver closure

**Spec.** Wire P1+P2 into `solver.solve_model` so an integer-bilinear MINLP is
detected, reformulated to MILP, and routed to `_solve_milp_bb` (the MILP path
with cover/clique/Gomory/MIR cuts + primal). No new cut code if the existing MILP
machinery suffices on the reformulated model.

**Verify correct.**
- The reformulated ex1263 routes to `_solve_milp_bb` (assert `nlp_bb`/path flag);
  the root cut loop fires ("root cuts added N"); the bound reaches 19.6 and an
  incumbent at 19.6 is found and certified (`gap ≈ 0`, `gap_certified`).
- Differential vs SCIP: same optimum, `incorrect_count == 0`.

**Contingency (only if P0's gate flagged the MILP solver as too weak).**
- *P3a:* ensure the spatial-path/relaxation MIP cuts are reachable for any residual
  bilinear (should be none after P2's full linearization).
- *P3b:* strengthen the MILP primal on the reformulated model (the validated
  finding that a tight relaxation + rounding the bits should yield the incumbent —
  add a "solve LP, round expansion bits, verify" heuristic if discopt's generic
  primal misses it).

**Test.** ex126x solve to optimum with certified gap; node count + wall time
recorded vs SCIP; regression suite green.

---

## P4 — Integration, breadth, and final validation

- Run the **full** correctness + regression + smoke suites; `incorrect_count == 0`.
- Broaden the basket: run a sweep of MINLPLib integer-bilinear instances; report
  how many newly reach proven optimality vs the pre-change baseline, with SCIP as
  oracle (no regressions on previously-solved instances — the gate).
- Performance report: ex126x wall time vs SCIP; honest accounting of the gap.
- Docs: a notebook + CHANGELOG entry; default-on only after the full suite is
  green (mirroring the project's phase-gate discipline).

---

## Sequencing & dependencies

```
verification harness ──► P0 (de-risk) ──► P1 (implied-integer) ──► P2 (reform) ──► P3 (route/close) ──► P4 (integrate)
                                   │                                   ▲
                                   └── P2 testable on declared-integer ┘ (independent of P1)
```
P0 gates the whole approach. P2 can be developed/tested on declared-integer
instances before P1 lands. Each step ends only when its verification + the
regression gate are green.

## Risk register

| Risk | Mitigation |
|---|---|
| Unsound implied-integer marking cuts off optimum | P1 soundness re-solve (SCIP oracle) on every marked var; conservative conditions; reject the `b≤x≤b+4` non-case |
| Variable blow-up makes discopt's MILP slow | P0 gate; shared-factor expansion; P3b primal |
| Big-M linearization numerically loose (large M) | use tight `U` per variable; equilibration (existing in milp_relaxation); test conditioning |
| Regression on currently-solved models | reformulation gated to nonconvex + integer-bilinear; no-op otherwise; full regression gate after each step |
| discopt MILP solver weaker than SCIP | acceptable if it still *closes* ex126x; speed-parity with SCIP is a stretch goal, not a correctness gate |

## Success criteria
- ex1263–1266 solved to **proven** optimality (matching SCIP) through discopt's
  own pipeline, automatically (P1 detection, no manual marking).
- `incorrect_count == 0` across the full suite; every reformulation verified
  value-preserving against SCIP.
- A measured, honest performance comparison vs SCIP.
