# #1116 — `kriging_peaks-full200` does not reproduce run-to-run

Probes for the root-cause investigation. Every one prints an executed-comparison
count and exits non-zero when it made none (CLAUDE.md §6), swallows no exception
(§7), and prints `discopt.__file__` (§8). `.log` outputs are gitignored; the
harnesses are kept so any verdict here can be re-run.

## Verdict

The cause is the **wall clock**, in #912's role 2 ("how much work do we do?"), not
role 1 ("when do we stop?"). Fixed by `Model.solve(deterministic=True)` /
`DISCOPT_DETERMINISTIC` — see `solver_tuning.SolverTuning.deterministic` and
`solver._role2_deadline` / `_role2_horizon`.

| probe | question | verdict |
|---|---|---|
| `repro_probe.py` | does it reproduce at all? | **no** — 3 bounds, nodes 301↔303 at `max_nodes=300` |
| `min_budget_scan.py` | cheapest reproducing budget | `max_nodes=1`: 3 bounds spanning **14 %** (−25371.8/−28852.0/−28072.6), incumbent bit-identical |
| `rust_hash_iteration_scan.py` | the issue's "suggested first step" | **falsified** — 79 files / 61 404 lines / 18 hash-container decls / **3** iteration sites, all order-insensitive; the five named `bnb` sites are only `insert`/`entry`/`remove`/`get`/`len`ed |
| `run_e3_e2.sh` (E3) | thread scheduling? | **falsified** — rayon+BLAS pinned to 1, still 301↔303 |
| `e5_stable_hash.py` | `Variable.__hash__ = id(self)`? | **falsified** — patched to `self._index` (26 377 430 firings), bound still moves |
| `lp_seam_bisect.py` | is the Rust simplex a faithful function of its input? | **yes** — `first_differing_OUTPUT_at_equal_input = None`; divergence is upstream, at LP call 0 |
| `build_component_bisect.py` | how does LP call 0 differ? | **structurally** — `c` (1532,) vs (1469,), `A_ub` (4860,1532) vs (4734,1469). Not float noise |
| `build_truncation_probe.py` | is the #694 anytime build truncating? | **falsified** — `build_truncated=False`, `cons_done == cons_total` (404/404) |
| `root_fixpoint_probe.py` | does the deadline-driven root fixpoint return a different box? | **yes** — `n_tightened` 894 vs 898, out_box hashes differ, OBBT 237.7 s vs 247.55 s, *and the input box already differs* (an upstream role-2 clock too) |
| `no_wall_probe.py` (E4) | wall clocks at all? | **yes, and confounded** — clock+LP+NLP patched together reproduces bit-exactly |
| `wall_arm_probe.py` (E6) | *which* wall seam? | **the clock alone**: `clock` arm −1044.8190276107248 twice; `nlp` arm still varies (−32525.9 / −34452.9); the LP seam never binds (`timelimited=0`) |
| `stage_trace.py` | where does divergence first appear? | at the **root LP**, not in a later stage — every structural quantity before it agrees |
| `build_hash_probe.py` | is the built relaxation itself bit-identical? | **yes, within one process** — the divergence is in the box handed to the builder, not the builder |
| `clock_trace.py` | which wall-gated call site differs? | tallies `perf_counter` calls per `(file, line)`; the differing sites are the tightening budgets |
| `nlp_trace.py` | is the first divergence in a local NLP? | **no** — the incumbent's last digits move but the dual bound diverges first |
| `deterministic_verify.py` | does the shipped fix reproduce? | **yes** — acceptance test for `deterministic=True`; see below |

Note the no-clock bound (−1044.819) is **three orders of magnitude tighter** than
every wall-bounded run: an early-truncated tightening stage is not merely
nondeterministic, it leaves bound on the table.

## What the fix covers, and what it deliberately does not

The flag neutralizes role-2 clocks at their **origin** — 27 call sites in
`solver.py` (via `_role2_budget` / `_role2_deadline` / `_role2_horizon`), 3 direct
`_tuning().deterministic` guards where the budget is a loop condition rather than
an argument, and the integer-ratio dive in `_relax/mccormick_lp.py` — rather than at the seam, so the
class is fixed and not the instance (CLAUDE.md §2). The wrapped sites span both
sides of the search: the dual side (root presolve, declared-box tightening, OBBT,
per-node OBBT, the root fixpoint, root cuts, the box/probe/pool LP slices, the
MILP stall slice) and the primal side (native-kernel seeds, root sub-NLP seeds,
ILS, continuous multistart, one-hot swap, dual recovery at the incumbent) — a
machine-speed-dependent *incumbent* moves the cutoff, which moves the tree, which
moves the bound, so the primal half is not optional.

Two role-1 mechanisms are left live on purpose:

* the phase-entry gates (`_deadline_exhausted()` / `_remaining_budget() > x`) that
  decide whether an optional preprocessing phase *starts*;
* the two POUNCE funnels' `max_wall_time = min(30.0, caller_limit)` stall backstop.

Neutralizing either lets preprocessing overrun the user's `time_limit` without
bound — trading a reproducibility bug for a broken role-1 promise, which §1 does
not permit. So the guarantee is scoped: **a solve reproduces when the role-1
budget never binds.** Both residuals were measured not to bind here — the 30 s
POUNCE cap was live and real during the `clock` arm that reproduced bit-exactly,
and the solve finishes in ~7 min against the default 3600 s limit.

## Re-running

    python -u repro_probe.py <instance-stem> <max_nodes> <reps>
    python -u wall_arm_probe.py <stem> <max_nodes> <reps> <none|clock|lp|nlp[,...]>
    python -u deterministic_verify.py <stem> <max_nodes> <reps> <0|1>

`<instance-stem>` is resolved under
`~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/`. `full200` costs ~5-6
min per repetition at `max_nodes=1` and ~7 min at `max_nodes=300`.
