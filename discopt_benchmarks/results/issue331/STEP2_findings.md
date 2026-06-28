# Issue #331 — Step 2: deepening bounds (cuts / presolve / probing)

Step 1 attributed the node gap to **weak bounds** (discopt closes 2–3× less root
integrality gap than SCIP). Step 2 set out to deepen the weakest lever. This
documents what was tried, measured, and concluded. **Net result: on the current
dense LP layer, deepening cuts/presolve/probing cannot meet the acceptance
criterion (≤3× SCIP nodes *and* wall-time parity) on these instances — the
payoff is gated on the sparse LP data layer (Regime C).** No solver-core change
ships from Step 2; the evidence redirects the work.

All numbers are from `discopt_benchmarks/perf/milp_node_efficiency.py` driving
`solve_milp_py` directly (SCIP 10.0 via pyscipopt 6.2.1, feral 0.11, single host).

## 1. The cut-separation ceiling — discopt plateaus below SCIP

Root integrality gap closed (cap-independent), discopt at increasing cut budgets
vs SCIP:

| instance | 16 cuts / 1 round (prod) | 100 / 10 | 500 / 50 | SCIP |
|---|---|---|---|---|
| mdk60x8 | 24.6% | 32.5% | 32.7% | **48.6%** |
| mdk90x12 | 14.0% | 26.5% | 30.6% | **45.9%** |
| mdk120x15 | 9.2% | 12.7% | 13.4% | **22.4%** |
| mdk200x25 | 4.9% | 8.7% | 8.8% | **13.9%** |

Two facts: (a) the production default (1 round) leaves roughly half the
*reachable* gap unclaimed; (b) even maxed, discopt plateaus at ~60–70% of SCIP's
closed gap — its cover separation is genuinely weaker than SCIP's.

## 2. …but closing more gap costs more than it returns on the dense layer

* **The per-round yield, not the cut cap, binds.** Raising `root_cuts` 16→64 at
  one round gives **identical** node counts on every instance — one separation
  round simply does not find more cuts. So the cap was never the limit.
* **More gap needs more *rounds*, and rounds are expensive.** Each round
  re-solves a growing **dense** LP. On `mdk30x5`, going from prod (1 round,
  0.013 s, 87 nodes) to 20 rounds (2.6 s, 51 nodes) is a **200× wall-time blow-up
  for a 41% node cut** — a catastrophic trade against the issue's hard
  "net wall time" constraint.
* **Stronger single-round separation doesn't rescue it.** A reworked cover
  separator that emits *multiple* covers per row (forcing each fractional
  variable into a cover) was implemented and validated (exhaustive 0/1
  feasibility tests pass). At the **root** it closes *no* extra gap in one round
  — an LP vertex has too few fractional variables to seed distinct covers — and
  on the large instances it **regresses wall time** (`mdk200x25`: 3.8 s → 4.9 s
  for +0% nodes). It was therefore reverted, not shipped. (It would pay off once
  cut re-solves are cheap; see §4.)

## 3. Presolve / probing are structurally inert on these instances

The driver's root presolve is FBBT only, which Step 1 measured at **0%** node
reduction. The natural deepening — probing + clique/implied-bound cuts — was
checked *before* implementing, and it cannot fire here:

```
instance     singleton fixings   conflict pairs (x_j + x_k > cap_i)
mdk30x5             0                    0   (of 2 175 possible)
mdk90x12            0                    0   (of 48 060)
mdk120x15           0                    0   (of 107 100)
mdk200x25           0                    0   (of 497 500)
```

With capacities at half the weight sum and small uniform items (~3% of capacity
each), no single item is forced out and no two items conflict — so probing
discovers **no** fixings, implied bounds, or cliques. On *these* instances
probing is provably a no-op. (It is genuinely valuable on structured MILPs —
set-packing/covering with cliques, dominated columns — which this bench
deliberately lacks. Adding such an instance + a sound matrix-level probing pass
is the "probing next" follow-up.)

## 4. Conclusion & redirection

* **The Step 1 attribution holds and is sharpened:** the node gap is a
  weak-*bounds* gap, and the bound deficit is real (discopt closes 2–3× less root
  gap than SCIP).
* **But every bound lever is gated on the dense LP layer.** Presolve/probing are
  structurally inert on these instances; cuts can close more gap *only* through
  multiple rounds whose dense re-solve cost exceeds the node savings. discopt
  currently wins wall time precisely *because* it cuts shallowly (1 round) and
  relies on cheap nodes + the primal heuristic.
* **Therefore the ordering matters: land the sparse LP data layer (Regime C)
  first.** Once a cut-augmented LP re-solves cheaply, multi-round cutting (and
  the stronger multi-cover separator) becomes affordable and the bound work pays
  off — exactly the complementarity issue #331 itself notes ("node savings only
  pay off if each node is also cheap"). Until then, expanding the cut portfolio
  (MIR, flow-cover, clique, more covers) trades wall time for nodes and regresses
  the instances discopt already wins.
* **The one lever independent of Regime C is branching/heuristic quality** — the
  residual after bounds. That is the place to look for node wins that don't first
  require the sparse layer.

**Recommended sequencing:** Regime C (sparse LP) → re-run this committed bench to
confirm cuts now pay → then expand the cut portfolio and wire probing (with a
structured instance added to the bench). The bench and ablation harness from
Step 1 are in place to validate each step.

### Reproducing the measurements here

```bash
# cut-separation ceiling and per-round economics
python -m discopt_benchmarks.perf.milp_node_efficiency --out discopt_benchmarks/results/issue331
```

The probing-yield and cut-budget sweeps above are one-off diagnostics over the
same `gen_mdk` instances (weights `W`, capacities `cap` from
`milp_node_efficiency.gen_mdk`); they are described here rather than committed as
scripts because they are throwaway confirmations, not part of the regression
bench.
