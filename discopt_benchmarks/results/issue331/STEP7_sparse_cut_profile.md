# Issue #331 — Step 7: the sparse cut profile is difficulty-adaptive, not sparsity-adaptive

Step 6 identified the sparse-frontier lever (cuts) and built the management
primitive (`lp/cut_select.rs`). Step 7 asks the concrete follow-on: **can a
"sparse cut profile" — disable the dense GMI tableau cuts, keep the row-local
cover cuts, select aggressively, run several rounds — be wired as an adaptive
default that wins the sparse regime without regressing dense?**

The answer, grounded on every instance: **no — the win is keyed to instance
*difficulty*, not to row sparsity, so a sparsity gate would regress the easy
majority. Ship the levers (`gmi_cuts`, `cut_select`), not a default-changing
gate.**

## What was built — the `gmi_cuts` lever

GMI cuts are separated off the simplex tableau (`B⁻¹A`), which mixes all columns,
so they are **dense** even on a sparse-row model (measured Step 6: smdk60x20 GMI
rows 45/60 nnz, smdk80x25 61/80 nnz, vs ~25 % structural row density). Carrying
them densifies the cut-augmented matrix and defeats the #334 sparse-LP fast path.
`gmi_cuts` (default `true`, behavior-preserving) lets a caller drop GMI and keep
only the sparse cover cuts. Together with `cut_select` (Step 6) this is the full
sparse-cut **management** surface; both are sound (they only choose among valid
cuts, never modify one) and tested.

## Grounding — the full sparse profile across the whole sparse family

Profile = `gmi_cuts=False, cut_select=True, cut_rounds=3, root_cuts=2m,
max_pool_cuts=4m`, every other lever at `prod`. Measured vs `prod`, objectives
match SCIP on all (sound):

| instance | prod nodes | prod wall | profile nodes | profile wall | SCIP nodes / wall | verdict |
|---|---|---|---|---|---|---|
| smdk50x15 | 703 | 0.12s | 1097 | 0.27s | 21 / 0.18s | **regresses both** |
| smdk60x20 | 1025 | 0.32s | 1365 | 0.42s | 43 / 0.42s | **regresses both** |
| smdk70x20 | 7643 | 0.95s | 4265 | 1.13s | 145 / 0.95s | nodes ↓44 %, **wall ↑19 %** |
| smdk80x25 | 29147 | 3.81s | 15695 | 3.28s | 753 / 1.85s | **wins both** (−46 % n, −14 % s) |

The profile is a clean win on exactly **one** instance — the hardest
(`smdk80x25`, ~29 k nodes). On the next one down it trades nodes for wall; on the
two easy ones it loses on both. The crossover tracks node count (≈ difficulty),
**not** the row density, which is fixed at 25 % across all four.

This is the same finding that killed the Step-6 density gate, now measured on the
profile itself: GMI's (and aggressive cut management's) value tracks how hard the
tree is, not how sparse the rows are. A gate keyed on "is this model sparse?"
would apply the profile to all four and **net-regress three of them**. There is
no static, root-visible signal here that separates `smdk80x25` from `smdk50x15`
except solving far enough to see the tree explode — i.e. the gate would have to
be a *dynamic* (restart/aging) decision, which Step 4's objective-cutoff probe
already showed grounds out on this engine.

## Dense regime — untouched (confirmed)

The profile is sparse-only and changes no defaults, so the committed dense bench
is unchanged. Re-measured at current HEAD for the record:

| instance | prod nodes | prod wall |
|---|---|---|
| mdk120x15 | 4713 | 0.30s |
| mdk150x20 | 1797 | 0.34s |
| mdk200x25 | 17935 | 2.12s |

(The Step-5 reduced-cost-fixing win still holds: dense node ratios stay ≤3× SCIP
with faster wall — `mdk200x25` 1.4×.)

## Decision

- **Shipped:** the `gmi_cuts` lever (default `true`), completing the cut-profile
  surface (`gmi_cuts` + `cut_select` + `cut_rounds` + `root_cuts`). A caller
  targeting a known-hard sparse instance can now dial in the winning profile
  explicitly; it is measured to win `smdk80x25` on both nodes and wall.
- **Not shipped:** an adaptive "apply on sparse models" gate. The data does not
  support it — it would regress the easy-sparse majority. Shipping it would
  violate the issue's net-wall-time-on-*every*-instance constraint and the
  project's "no inert/debatable defaults" rule.
- **Unchanged blocker for a *uniform* sparse win:** the real fix remains the
  data-layer item from Step 6 — sparse cut representation in the LP (keep
  cut-augmented matrices sparse, age/remove cuts), "Regime C part 2" — which
  would make the profile cheap enough to pay off on the easy instances too. That
  is a #334-class data-layer project, not a driver tuning knob.

### Reproducing

```bash
# the profile vs prod, all sparse + dense (objectives cross-checked vs SCIP)
python -m discopt_benchmarks.perf.milp_node_efficiency --out discopt_benchmarks/results/issue331
```

Profile via `solve_milp_py(..., gmi_cuts=False, cut_select=True, cut_rounds=3,
root_cuts=2*m, max_pool_cuts=4*m)` over `gen_sparse_mdk` instances.
