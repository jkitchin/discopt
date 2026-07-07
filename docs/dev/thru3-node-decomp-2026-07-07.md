# THRU-3 — per-node cost decomposition on nvs19/nvs24 (2026-07-07)

**Task.** G4 flagged nvs19/24 as near-tight-root but low-throughput and guessed
the lever was "add incremental warm LP" — already present (THRU-2b #525). THRU-3's
grounded hypothesis: the dominant per-node cost is **per-node PSD (moment)
separation** (`mccormick_lp.py::_separate_psd`, now default-on since the
`psd_cost_gate` graduated, #537). Decompose the per-node wall, confirm or refute,
and — only if PSD dominates — prototype a per-node PSD budget.

**Verdict: the hypothesis is REFUTED. PSD is NOT the dominant per-node cost.**
The dominant per-node cost is the **univariate-square tangent-separation loop**
(`_separate_univariate_square`) — the *ungated twin* of the PSD loop. Both run up
to 8 full MILP re-solves per node; THRU-2a gave PSD a cost gate but left the
square loop unbudgeted, so it is now the bigger drag. And even that is not the
whole story: on nvs24 the true floor is the **base per-node MILP simplex solve
itself**, which no round-budget can touch.

Build: release (`maturin develop --release`), pounce `_pounce.abi3.so` = 4.51 MB.
Env: `PYTHONPATH=<wt>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`. TL 60 s,
`gap_tolerance=1e-4`. Oracle (`minlplib.solu`): nvs24 = −1033.2, nvs19 = −1098.4.
Both instances have **no linear constraints**, so `_apply_auto_cut_policy` selects
**PSD** per node (`_psd_cuts=True`, `_rlt_cuts=False`) — PSD *does* fire here.

## Part 1 — per-node bucket breakdown (defaults, 60 s)

Per-family separation timers (`solver_stats["separate/*"]`, accumulated across all
`solve_at_node` calls incl. root) plus the 200 Hz leaf-frame sampler.

### nvs24 (defaults): 9 nodes in 63 s → 0.14 nodes/s

| bucket | wall (s) | % of solve wall |
|---|---:|---:|
| **univariate_square separation** | **46.5** | **73%** |
| PSD (moment) separation | 7.63 | 12% |
| root processing (of which square/PSD above) | 23.6 root_time | — |
| edge_concave | 0.05 | <1% |
| multilinear / rlt / convex | ~0.001 | ~0 |

Sampler (whole-stack containing-frame): `solve_at_node` 57%, `obbt` 22%,
`build_milp_relaxation` 12%, `_separate_univariate_square` 4%, `_separate_psd`
0.5%. Top leaf frames: `milp_simplex.py:solve_milp` **48%**,
`obbt.py:solve` 8%, `nlp_pounce:solve_nlp` 5%. → the node wall is dominated by the
**MILP simplex solve**, re-invoked by every square/PSD separation round.

### nvs19 (defaults): 203 nodes in 60 s → 3.4 nodes/s

| bucket | wall (s) | % of solve wall |
|---|---:|---:|
| **univariate_square separation** | **25.0** | **42%** |
| PSD (moment) separation | 12.3 | 20% |
| edge_concave | 0.31 | <1% |
| root processing | 2.7 root_time | — |

Sampler: `solve_at_node` 87%, **`_separate_univariate_square` 48%**, `_separate_psd`
9%. Top leaf frames: `solve_lp_warm_std` 34%, `_safe_lp_lower_bound_std` 12%,
`sparse._matmul_vector` 11%, `_fbbt_eq_bounds` 6%. → nearly all node wall is the
**per-node LP re-solve loop of the square separator**.

**Both agree: `univariate_square` ≫ `psd` per node.** The hypothesis (PSD
dominates) is refuted on the sampler AND the timers.

## Part 1 — control matrix (60 s)

`psd_off` = force `_psd_cuts=False` (keep square). `square_off` =
`DISCOPT_SQUARE_SEPARATE=0` (keep PSD). `both_off` = both.

| instance | arm | nodes | nodes/s | bound@TL | incumbent |
|---|---|---:|---:|---:|---:|
| nvs24 | defaults   | 9   | 0.14 | −1035.66 | −1031.80 |
| nvs24 | psd_off    | 9   | 0.13 | −1035.66 | −1031.80 |
| nvs24 | square_off | 11  | 0.18 | −1035.66 | −1031.80 |
| nvs24 | **both_off** | **309** | **5.05** | −1035.38 | −1031.80 |
| nvs19 | defaults   | 203 | 3.4  | −1102.10 | −1098.20 |
| nvs19 | psd_off    | 187 | 3.1  | −1102.60 | −1097.60 |
| nvs19 | square_off | 355 | 5.9  | −1100.14 | −1097.60 |
| nvs19 | **both_off** | **521** | **8.67** | −1099.98 | **−1098.40** |

**The PSD-off control does NOT jump nodes/s** (nvs24 0.14→0.13, nvs19 3.4→3.1):
turning off PSD just hands its budget to the ungated square loop (nvs24
`univariate_square` 46.5→61.8 s). This is the direct refutation the task asked
for. Symmetrically, `square_off` hands time to PSD (nvs24 `psd` 7.6→45.7 s).

**Throughput only unlocks when BOTH per-node separators are off** — nvs24 9→309
nodes (**36×**), nvs19 203→521 nodes (**2.6×**) — and the dual bound at TL barely
moves (nvs24 −1035.66→−1035.38; nvs19 −1102.1→−1100.0, both still valid
underestimators ≤ =opt=) while the incumbent is same-or-better (nvs19 `both_off`
reaches the optimum −1098.4). Every bound is ≤ its incumbent and ≤ =opt= — no arm
cut a feasible point.

**Conclusion.** The two per-node separators (square + PSD) are *fungible twins*
competing for the same node-wall budget through the same MILP-re-solve loop. PSD
was blamed because THRU-1 profiled it at the root; per-node, the *ungated square
loop* is the larger of the two, and neither alone is the throughput lever — the
node relaxation itself (base MILP solve + the lifted square/PSD aux columns) is.

## Part 2 — prototype: per-node univariate-square cost gate (default-OFF)

Since PSD is refuted, the task's Part-2 PSD-budget prototype is not built. The
data instead points at the *general* class fix: the square loop is the one
per-node separator with **no cost budget** (§0.2 — this keys on cost, not on any
instance name). I mirrored the already-graduated PSD gate as a **default-OFF**
prototype: `DISCOPT_SQUARE_COST_GATE` / `SolverTuning(square_cost_gate=…)`, with
`square_cost_gate_budget` (× base LP-solve wall) and `square_cost_gate_tau`
(diminishing-returns). SOUND by construction: it only ever *shortens* the loop
(drops valid tangent cuts → looser relaxation, never cuts a feasible point);
default-OFF path is bit-identical.

### Prototype ON vs OFF (60 s)

| instance | gate | nodes | nodes/s | bound@TL | incumbent | sq_fires | square_s | psd_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| nvs24 | OFF | 9   | 0.14  | −1035.66 | −1031.80 | 0   | 47.2 | 7.8 |
| nvs24 | ON  | 9   | 0.15  | −1035.66 | −1031.80 | **5**   | 27.7 | 20.3 |
| nvs19 | OFF | 197 | 3.28  | −1102.10 | −1098.20 | 0   | 24.9 | 12.2 |
| nvs19 | ON  | 201 | 3.34  | −1102.55 | −1097.60 | **144** | 15.0 | 18.4 |

**The gate FIRES** (`gate/square_fires` = 5 / 144, surfaced in `solver_stats`) and
cuts square wall (nvs24 47→28 s, nvs19 25→15 s) — but **nodes/s barely moves**,
because the *freed time is absorbed by the PSD loop* (nvs24 psd 7.8→20.3 s), the
exact swap the control matrix predicted. Capping *both* loops (both gates at
budget 0.25, or tau 10 to abandon after the first weak round) also leaves nvs24 at
**9 nodes** — proving the round-budget is not the nvs24 lever at all: nvs24's floor
is the **base per-node MILP simplex solve** (`solve_milp` 48% of samples), which a
round-budget cannot reduce. The `both_off` 309-node result comes from a
*structurally cheaper* relaxation (no square/PSD aux columns lifted → a smaller
LP), not merely fewer separation rounds.

**Honest status of the prototype:** correct, sound, fires, cert-neutral OFF — but
**throughput-inert** as a single-separator gate, for the measured reason above. It
is shipped default-OFF as the instrumented, reusable budget knob (and the fire
counter), not as a throughput win. Killing it as "the lever" per the build-time
kill criterion; kept as the sound, measured budget primitive.

## The real lever (finding)

The throughput lever on this instance class is **not** a per-round separation
budget. It is one of:

1. **Structure-gate the per-node square/PSD separation off below the root** (or
   after the first slack-free round) — i.e. inherit the root cut pool and skip
   per-node re-derivation, the direction `both_off` measures (36× on nvs24) at a
   ≤0.3-unit bound cost that branching recovers. This is the RLT root-pool
   inheritance pattern (P1) extended to the square/PSD families. Needs a
   differential-bound gate before it can default-on (bound-changing regime).
2. **Cheapen the base per-node MILP solve** on dense integer-product QPs — the
   `solve_milp` 48% floor on nvs24 — which is a simplex/warm-start question, not a
   separation-budget one.

Both are larger than a one-PR flag; recorded here so the next THRU task starts
from the measurement, not the (now-falsified) "PSD is the drag" premise.

## Gates

`pytest -m smoke` (617 passed, 14 skipped); adversarial
(`test_adversarial_recent_fixes.py`, 10 passed); `ruff check` + `ruff
format --check` clean; pre-commit `mypy` Passed. No Rust touched → `cargo test`
n/a. Flag fires: `gate/square_fires` = 5 (nvs24) / 144 (nvs19).

**`check_cert_neutrality.py` (41-panel):** 40/41 byte-identical (`nodes n->n`,
`|Δobj|=0`). The one flagged row — `nvs13 node_count 19 -> 49` — is **pre-existing
baseline drift on `main`, NOT this change**: stashing this PR's 3-file diff and
re-solving nvs13 on the branch point (`d1b09d3e`) yields **49 nodes**
deterministically (2/2 runs), identical to this branch (3/3 runs). The committed
`cert-baseline.jsonl` (node_count 19) predates a since-merged main commit that
moved nvs13; the default-OFF square-gate path is verifiably byte-identical
(49 → 49). Refreshing the baseline is out of scope for this PR (a separate,
unrelated main-drift fix).
