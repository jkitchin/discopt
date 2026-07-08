# C-42 — cut-inherit cold-path probe must not truncate the B&B loop (2026-07-07)

**Task.** THRU-4's graduation validation (#552,
`docs/dev/thru4-graduate-2026-07-07.md` on the `thru4-graduate` branch) found
two deterministic flag-ON certificate losses that keep `DISCOPT_CUT_INHERIT`
(#551) default-OFF: **nvs06** exits after 1 node at 1.5 s with 8.5 s of budget
unused (`feasible 231.70` instead of the certified `1.7703125`), and **tspn05**
loses its certificate because per-node re-separation is load-bearing there.
This change root-causes and fixes the nvs06-class truncation (Part 1) and ships
the lazy re-separation trigger for the tspn05 class (Part 2). The flag default
stays **OFF** — no default-path change in this PR.

Build: release (`maturin develop --release`, pounce `_pounce.abi3.so` 4.73 MB).
Env: `PYTHONPATH=<wt>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, corpus
`~/Dropbox/projects/discopt-minlp-benchmark/`, oracle `minlplib.solu`
(nvs06 = 1.7703125, tspn05 = 191.25521, nvs19 = −1098.4, nvs24 = −1033.2).
Single runs per arm, same machine, fresh interpreter per solve.

## Part 1 — root cause of the nvs06 truncation (file:line)

Reproduced deterministically: flag-ON nvs06 → `feasible 231.70004, 1 node,
1.5 s`; flag-OFF → `optimal 1.7703125, 5 nodes, 1.4 s`. The failure chain has
four links:

1. **The cold-path pool probe captures a VALID cut.** With the flag on, the
   `elif ... or _cut_inherit` branch (`python/discopt/solver.py:5012`, pre-fix
   numbering) runs the general root-pool capture on nvs06 (where the
   incremental engine declines). The captured pool is ONE row — verified by
   hand to be a legitimate PSD eigencut `vᵀMv ≥ 0` over the lifted clique
   `(aux0, aux1)` with `v = (0.2332, 0.7392, −0.6318)`, `b = v₀²`; it holds at
   every exactly-lifted feasible point (checked at integer feasible points).
   The cut is NOT the bug.
2. **Appending that one row makes the cold node solve fail numerically.**
   At the root node solve, `solve_at_node`
   (`python/discopt/_jax/mccormick_lp.py:877`, pre-fix) appends the row and the
   warm-simplex integer-aware relaxation solve flips from `optimal 1.10` to an
   **uncertified `infeasible`** (or `iteration_limit`, depending on warm
   state) — the C-38 numerical-false-infeasible class, triggered here by the
   extra row instead of a stale basis. Isolated A/B on the identical box:
   `inherit-only → numerical`, `skip-only → optimal 1.10`,
   `inherit+skip → numerical`, `neither → optimal 1.10` — the inherited row
   alone is necessary and sufficient. The existing guard
   (`mccormick_lp.py:967–993`) correctly refuses to fathom on an uncertified
   infeasible and returns `status="numerical"` **with no bound**.
3. **The driver turns "no bound" into a failure sentinel at the root.** On the
   LP-relaxer path the root's node NLP is deliberately skipped, so
   `result_lbs[i]` is pre-set to `_INFEASIBILITY_SENTINEL`
   (`solver.py:6107`, pre-fix) on the expectation that the LP block overwrites
   it. With the LP solve failed, the sentinel survives;
   `_nonrigorous_sentinel_fathom` (`solver.py:6235`) decertifies the gap and
   the root is imported as pruned.
4. **Loop truncation + pre-tree incumbent-search reroute.** With the root
   pruned, `tree.export_batch` returns 0 nodes and the `while True` loop breaks
   (`solver.py:5493`) after ONE node with 8.5 s unused. The pre-tree pump chain
   (feasibility pump → NLP-relaxation pump → integer local search,
   `solver.py:6486` gate `best_root_idx is not None`) never runs because no
   root relaxation bound landed in `result_lbs` — this is the "reroute": only
   the SubNLP/box-search heuristics (different gates) fire, seeding 251.3 →
   231.7. Exit: `feasible 231.70`, dual bound 1.10 (valid — not a false
   certificate, but a lost certificate + 130× degraded incumbent).

### The fix (general, not keyed to nvs06)

**The inherited pool is an accelerator, never a dependency.** In
`solve_at_node` (`mccormick_lp.py`, after the initial cold solve): when pool
rows were appended and the solve produced no certified verdict — not
`optimal` and not a Farkas-certified `infeasible` — the pool rows are stripped
and the node re-solved. The retry is byte-identical to the no-pool solve the
default path performs, so a destabilizing pool can perturb neither the
incumbent search nor loop termination at any node. The per-node square/PSD
skip is lifted for that node too (its justification — "the pool already
supplies the family" — no longer holds once the pool is dropped).
Soundness: dropping valid rows only loosens the relaxation; a Farkas-certified
infeasible on the pool-augmented system is still a rigorous fathom (valid cuts
preserve all feasible points) and is kept. Instrumented as
`pool/dropped_nodes`.

The driver-layer hazard behind link 3 — a *deliberately skipped* node NLP
sharing the failure sentinel with a *failed* one, so any node whose LP yields
no bound is pruned non-rigorously — is pre-existing, flag-independent, and
NOT changed here (changing it would alter default-path node accounting and
break cert-neutrality in this PR). Recorded as a follow-up for the
correctness backlog (#396): `solver.py` sentinel-on-skip at the line noted
above.

### nvs06 before/after (flag-ON, TL 10 s)

| arm | status | objective | bound | nodes | wall | pool |
|---|---|---|---:|---:|---:|---|
| before (main @ cd6e199d) | feasible | 231.70004 | 1.1000150 | 1 | 1.5 s | size 1, inherited 1, skipped 1 |
| **after** | **optimal** | **1.7703125** | 1.7703125 | 5 | 2.0 s | size 1, inherited 3, skipped 2, **dropped 1** |
| flag-OFF (reference, unchanged) | optimal | 1.7703125 | 1.7703125 | 5 | 1.4 s | — |

Flag-ON now reproduces the flag-OFF certificate exactly (same 5 nodes, same
objective/bound to the last digit); `pool/dropped_nodes = 1` fires at the
root, confirming the mechanism.

## Part 2 — lazy re-separation trigger (the tspn05 blocker)

Design shipped: a **driver-side global-bound-stall governor** plus a
**relaxer-side stride safety net** (all constants global, never
per-instance):

* Governor (`solver.py`, `_LAZY_RESEP_STALL_WINDOW = 24`,
  `_LAZY_RESEP_PROBE_BUDGET = 8`, `_LAZY_RESEP_GLB_EPS = 1e-9`): watch the
  tree's global lower bound each iteration (only under active pool
  inheritance — the default path reads no extra state). The governor is
  ARMED by the first genuine in-tree bound improvement — a bound that has
  never moved since the root is the pool-at-fixed-point signature (nvs24:
  root pool separated to convergence; per-node re-separation measured
  bound-inert), where a probe only burns the most expensive separation wall
  in the corpus. Once armed: a stall of a full window of node solves
  re-enables the full square/PSD separation pass (`probing`); an improvement
  while probing REFRESHES the probe (separation is demonstrably productive —
  this is what lets tspn05 lock separation in and close instead of rationing
  it to an 8-in-32 duty cycle); a probe that spends its whole budget with no
  improvement mutes until the bound next moves. Stats:
  `pool/stall_reseparations`.
* Stride net (`mccormick_lp.py`, `_LAZY_RESEP_STRIDE = 64`): every 64th
  skip-eligible node solve separates regardless, so inheritance can never
  fully starve a class the governor misjudges (including the never-armed
  shape). Stats: `pool/lazy_reseparations`.

### Falsified designs (recorded per CLAUDE.md §4; measurements this branch)

The task sketch's per-NODE stall test — "re-separate when the node's LP bound
improvement vs its parent is below a small threshold" — was implemented first
(via a Rust `node_lower_bounds` getter; children inherit
`parent.local_lower_bound` at branch time) and **falsified**: on the dense
integer-QP class the node LP sitting at the parent's bound is the NORMAL state
(closure comes from branching depth), so the arm fired on **185/191 nvs19
nodes** and destroyed #551's win (nvs19: optimal 52.8 s → feasible at TL with
a worse incumbent; nvs24: 49 → 13 nodes). A per-fire LP-GAIN productivity
mute (BARON-style) was tried next and also **falsified**: the per-fire gain
does not discriminate — nvs19's fires gain a LOT of LP value (median relative
gain 0.117) yet are worthless for the 60 s certificate, while tspn05's
load-bearing fires gain little (median relative gain 1.3e-3); any threshold
keeps the wrong class firing. The discriminating signal is the GLOBAL bound's
progress, which only the driver sees — hence the shipped governor. Three
supporting measurements shaped the final knobs: (i) a 1-in-16 stride net costs
nvs24 49 → 29 nodes (one separation pass there is seconds — THRU-3's 73%+12%
wall share), 1-in-64 is within noise; (ii) without probe-refresh-on-improvement
the governor rations tspn05's load-bearing separation to an 8-in-32 duty cycle
and the certificate lands at 60.15 s — at the budget's edge (with refresh:
48.8 s); (iii) without the arming rule the governor's very first probe fires
on nvs24 (whose bound never moves from the root) and costs 49 → 31 nodes.

### Results (TL 60 s except nvs06 at 10 s; flag-ON, single runs, this build)

| instance | flag-ON on main @ cd6e199d (the #552 blockers) | flag-ON, this change | flag-OFF reference (this build) | verdict |
|---|---|---|---|---|
| tspn05 | feasible 191.2552, bound STALLS at 190.2786, 199 nodes @ TL — **cert lost** | **optimal 191.25521**, bound 191.2478 (gap 3.9e-5 ≤ 1e-4), 119 nodes, **48.8 s**; 26 stall + 1 stride fires | optimal 191.25521, 39 nodes, 19.2 s | **cert regained** |
| nvs19 | optimal −1098.4, 367 nodes, 52.8 s (the #551 win) | **optimal −1098.4, 295 nodes, 49.3 s**; 48 stall + 3 stride fires | feasible @ TL, 197 nodes, 0.35 % gap | **win retained** (certifies, slightly faster) |
| nvs24 | feasible, 49 nodes @ TL, bound −1035.6600 (5.3× nodes/s vs OFF's 9 nodes) | feasible, **49 nodes @ TL, bound −1035.6600** — governor never arms (bound never moves from the root), 38 skips, 0 fires: byte-matches the #551 flag-ON baseline | feasible, 9 nodes @ TL, same bound | **win retained** (identical) |
| nvs06 | **feasible 231.70, 1 node, 1.5 s — cert lost** (Part 1) | **optimal 1.7703125, 5 nodes, 2.0 s**, `pool/dropped_nodes` = 1 | optimal 1.7703125, 5 nodes, 1.4 s | **cert regained** |

Every dual bound above stays ≤ its incumbent and ≤ the oracle optimum (min
sense) — the certificate invariant holds on every arm.

## Gates

| gate | result |
|---|---|
| regression tests (`python/tests/test_c42_cut_inherit_coldpath.py`) | 2 smoke (pool-drop retry mechanism — verified to FAIL on the pre-fix code and pass after; stride net) + 2 slow (nvs06 flag-ON certifies 1.7703125; tspn05 flag-ON certifies 191.25521) — all pass |
| `check_cert_neutrality.py` (default OFF, 41-instance panel) | **40/41 byte-identical** (`|Δobj| = 0.00e+00` on every row, `nodes n->n`). The one flagged row — `nvs13 node_count 19 -> 49` — is the SAME pre-existing stale-baseline drift #550/#551/#552 already recorded: re-verified here by solving nvs13 default-OFF on the BARE branch point (this diff stashed) → **49 nodes, −585.2, bound −585.2000011714 — byte-identical to this branch's default-OFF run**. No rebaseline (same disposition as #550/#551/#552: main-drift, not this branch's). |
| `pytest -m smoke` (python/tests) | 622 passed, 14 skipped (includes the 2 new smoke tests) |
| `pytest -m slow python/tests/test_adversarial_recent_fixes.py` | 10 passed |
| `cargo test -p discopt-core` | 424 passed (run during development; the final diff touches NO Rust — the `node_lower_bounds` getter added for the falsified per-node arm was removed again) |
| `ruff check` / `ruff format --check` (v0.14.6) | clean |
| pre-commit `mypy` (the actual hook, whole package) | Passed |

Flag default stays **OFF** (`SolverTuning.cut_inherit=False`); no default-path
behaviour change in this PR.

## Follow-ups (recorded, not shipped)

* Driver sentinel-on-deliberate-skip (Part 1 link 3): a node whose NLP was
  deliberately skipped shares the failure sentinel with a genuinely failed
  one, so ANY node whose LP yields no bound is pruned non-rigorously
  (decertified, but still pruned). Flag-independent and pre-existing; file on
  the #396 backlog.
* Re-attempt THRU-4 graduation (#552's protocol) now that both panel losses
  are fixed: the cert-panel arm should be re-run flag-ON against
  `cert-baseline.jsonl` plus a fresh held-out draw before any flip discussion.
