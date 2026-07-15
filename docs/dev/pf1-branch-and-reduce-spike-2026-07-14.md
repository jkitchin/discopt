# PF1 SPIKE — branch-and-reduce payoff sweep (2026-07-14)

Measurement-only spike per `docs/dev/sota-proof-plan.md` §2 PF1. No library code
changed. All solves `time_limit=30` (plus one 60 s clay probe), subprocess per
run, `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, vendored corpus
`python/tests/data/minlplib_nl/`.

## Headline findings

1. **The PF1 premise is untestable via the existing flags on the instances that
   matter.** `in_tree_presolve_stride` / `DISCOPT_NODE_PROBING` /
   `DISCOPT_NODE_PROBE_MAX_VARS` are wired ONLY into `_solve_nlp_bb`
   (`python/discopt/solver.py:9300`, passed at :4279/:4580). The global spatial
   B&B path — where every unproved spike instance lives (tspn*, bchoco*,
   heatexch_gen*, contvar, nvs05) — never calls the kernel. Verified two ways:
   monkeypatch call-counter shows **0 `in_tree_presolve` invocations** on
   nvs14/st_miqp4/tspn05 with probing on (`nlp_bb=False`), and full config
   sweeps are node-identical on all 12 global-path instances.
2. **Stride is on/off only.** The single call site hardcodes `node_depth=0`
   (`solver.py:9324`), so `0 % stride == 0` always fires: stride 1 ≡ stride 4
   (confirmed empirically: identical nodes on fac2/m3/clay0303hfsg).
3. **On the NLP-BB path the machinery works and pays.** With
   `in_tree_presolve_stride=1` (FBBT only, no probing): **m3 gains a proof**
   (baseline: feasible, 61 nodes, no bound → optimal, 47 nodes, 13.0 s) and
   **fac2 cuts nodes 43 %** (69→39, 15.8→13.8 s). No proof lost anywhere.
4. **Probing is a mixed bag even where reachable.** m3 improves further
   (47→25 nodes, 11.3 s); but cvxnonsep_psig40r pays +40 % wall (13.6→19.9 s,
   same 95 nodes — pure per-node probing overhead, 0 tightenings), and on
   clay0303hfsg at 60 s probing processes 63 nodes with **no incumbent and no
   bound** where baseline processes 251 nodes with incumbent 26669.1 / bound
   4997.3 — probing quadruples node cost there and delays the first incumbent
   past the budget.
5. **`DISCOPT_OBBT_TOPK` is inert on the spike set** (statuses/nodes/bounds
   unchanged on all 12 global-path instances — its size(>100-var)/structural
   gates never trip here). OBBT rounds (`_PER_NODE_OBBT_ROUNDS=3`,
   `solver.py:237`) is a module constant, not env/kwarg-tunable — that grid
   axis needs a code edit and was NOT run (noted per spike rules).

## Soundness

Zero violations over all 87 runs: `bound ≤ objective (+tol)` on every record
with both present, and all proved objectives agree across configs per instance.

## Config × instance (status / nodes / wall s)

Global path (probing flags proven inert — baseline shown; other configs
node-identical, wall within noise): tspn05 feasible 31/30 s (gap 6.8 %→6.2 %
under any stride≥1 config — root-presolve side effect, see note below),
tspn08 feasible 1, tspn12 feasible 1, bchoco06 TL 3, bchoco07 TL 1–3,
bchoco08 TL 1, contvar TL 1 (bound 171244.81), heatexch_gen1 TL 3–7 (bound
38183.53), heatexch_gen2 TL 3 (bound 555767.79), nvs05 feasible 167–301
(gap 75 %), nvs14 optimal 223/2.6 s, nvs21 optimal 183/13.6 s, st_miqp4
optimal 3, st_e38 optimal 3. hda: hung past tl+90 s, dropped after baseline.

NLP-BB path:

| instance | baseline | fbbt_s1 | probe_s1_v32 | probe_s1_v64 / s4 |
|---|---|---|---|---|
| fac2 | opt 69 n 15.8 s | opt 39 n 13.8 s | opt 39 n 14.0 s | same |
| flay02m | opt 7 n 1.4 s | opt 7 n | opt 7 n | same |
| m3 | **feasible** 61 n 19.7 s | **opt** 47 n 13.0 s | **opt** 25 n 11.3 s | same |
| clay0303hfsg | TL 127 n | TL 95 n | TL 15 n (no inc/bound) | same |
| cvxnonsep_psig30 | opt 89 n 4.1 s | opt 89 n | opt 89 n | same |
| cvxnonsep_nsig30 | opt 165 n 9.7 s | opt 165 n | opt 165 n | same |
| cvxnonsep_psig40r | opt 95 n 13.6 s | opt 95 n 14.2 s | opt 95 n **19.9 s** | 20.2 s |

Node-ratio (config/baseline, meaningful cells): fbbt_s1 — fac2 0.57, m3 0.77
(+proof), clay 0.75, others 1.0. probe_s1_v32 — m3 0.41, fac2 0.57, clay 0.12
(confounded: node cost ×~4, worse bound state at equal budget).

## Verdict: **GO (scoped) + wiring gap is the real PF1 item**

Per the kill criterion (GO = new proof or ≥20 % node cut without losing
proofs): fbbt_s1 gains the m3 proof and cuts fac2 nodes 43 % with no
regression — GO. But the win is confined to the NLP-BB (convex) path; on the
global spatial path (where the BARON gap lives) the premise was neither
confirmed nor killed — the flags do not reach it.

Full PF1 item should:
1. **Wire the in-tree FBBT(+probing) kernel into the global spatial B&B node
   loop** (the batch loop in `_solve_bnb`, alongside the per-node OBBT block at
   `solver.py:~6155` / the `node_reduce` path), passing real `node_depth`;
   then re-spike the global set. This is the actual branch-and-reduce cash-in.
2. Flip `in_tree_presolve_stride` default 0→1 **on the NLP-BB path** (FBBT
   only) — supported by this data (m3 proof, fac2 −43 % nodes, no overhead).
3. Keep `DISCOPT_NODE_PROBING` default OFF pending a scored/budgeted probing
   policy (psig40r +40 % wall and clay incumbent starvation are the
   counterexamples; the P3 ledger's "tuned scoring policy" caveat is confirmed).
4. Fix the hardcoded `node_depth=0` so stride gating is real.
5. If an OBBT-rounds axis is wanted, make `_PER_NODE_OBBT_ROUNDS` tunable
   (env/kwarg) first — currently requires a code edit.

Note: on tspn05 any stride≥1 config improved the final bound
(178.272→179.443, gap 6.79 %→6.18 %) *without* any kernel firing — same effect
in fbbt_s1/probe/obbt_topk runs; likely run-to-run variation in node order at
the 30 s cutoff rather than a flag effect (node counts differ 31/35/45), so
not claimed as a win.

Raw records: 87-run JSONL in the session scratchpad (`results.jsonl`); harness
`run_one.py`/`drive.py` (subprocess-per-solve, env read at import).
