# CUTOFF-SOUND-1 — R2 cutoff-reduction soundness triage (2026-07-07)

**Verdict: R2 (root_fixpoint + node_reduce) is SOUND. The two flagged signals split
into two distinct causes, and NEITHER is an R2 bug:**

1. **Held-out arm `n_soundness_violations` (2 / 4)** — a **single genuinely
   false-optimal instance, `kall_circles_c8a`, on the DEFAULT (all-flags-OFF)
   path**, inherited by every arm through the shared OFF-control baseline. This is a
   **core relaxation/branching soundness bug (C-38, P0)**, NOT caused by the R2
   flags (flag-ON reproduces the identical wrong certificate). Filed/handed off as
   C-38; it is out of CUTOFF-SOUND-1's scope (R2 cutoff-reduction) and belongs in a
   separate, focused correctness PR.
2. **Cert-panel drift (st_e38 / ex1225)** — a benign, tolerance-accurate certified
   objective drift *toward the true optimum*, flagged only because the gate's
   cert-neutrality objective check used a **byte-reproducibility** tolerance (~1e-8)
   on a **bound-changing** flag. **This IS a gate false-positive (bucket B)**; this
   PR fixes it with a regime-aware objective check (below).

**The gate's held-out bracket check is NOT weakened** — it correctly caught the real
C-38 false-optimal. Only the cert-panel *byte-reproducibility* objective tolerance
is corrected, and only for the bound-changing regime, and it still catches a genuine
false certificate (a cross of the true optimum beyond correctness tolerance).

Branch `cutoff-sound-1` from `origin/main`. Build: `maturin develop --release`
(pounce present). Run env `PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu
JAX_ENABLE_X64=1`.

## Part 1 — Triage (the entry experiment)

### The two evidence streams the gate produces

The graduation gate reports two *independent* soundness signals per flag, and the
prompt's evidence mixes them:

1. **Held-out arm `n_soundness_violations`** — `generality_sweep.check_soundness`
   over a seeded, held-out MINLPLib sample (seed 0, N=40, tl 25s). It uses the
   `[=bestdual=, =best=]` **bracket** from `minlplib.solu`: a dual bound crossing
   `=best=`, or a certified-optimal objective outside the bracket, is a violation.
   This check is already bracket-based (not byte-identity) and is correctly
   designed.
2. **Cert-panel drift** — `run_cert_neutrality` re-solves the 41-instance vendored
   cert panel with the flag ON and compares the certified objective against
   `cert-baseline.jsonl` via `utils.cert_neutrality.check_neutrality`, whose
   objective tolerance is `OBJ_TOL=1e-8 / OBJ_RTOL=1e-9` — a **byte-reproducibility**
   tolerance. The prompt's "cert drift on st_e38 / ex1225" are this signal.

The prompt's `n_soundness_violations=2` (root_fixpoint) / `=4` (node_reduce) and the
named "cert drift" instances (st_e38, ex1225) are the same underlying phenomenon
surfaced through the two signals: a legitimate, tolerance-accurate objective drift
under a bound-changing flag.

### The named cert-panel drifts (reproduced exactly on this build)

`minlplib.solu` gives the **exact** optimum for both instances as `=opt=` (NOT
`=best=` — note `load_solu` only parses `=best=`/`=bestdual=`, so these two never
even enter the held-out selection; they are cert-panel instances). `cert-optima.json`
carries the true optima for the panel.

| instance | flag | OFF obj | ON obj | true opt (`=opt=`/cert-optima) | \|ON−opt\| | correct (abs 1e-4 / rel 1e-3)? | dual bound (ON) | bound ≤ opt? | drift direction |
|---|---|---|---|---|---:|:-:|---|:-:|---|
| st_e38 | root_fixpoint | 7197.727116839705 | 7197.727148532429 | 7197.727148524341 | 8.1e-09 | **yes** | 7197.727116826533 | yes | **toward opt** (OFF was 3.2e-5 *below* opt; ON is ~1e-8 from it) |
| ex1225 | node_reduce | 30.999999951817372 | **31.0** | 31.0 | 0.0 | **yes** | 31.0 | yes | **onto opt** (exact) |
| ex1225 | root_fixpoint | 30.999999951817372 | 30.999999968717816 | 31.0 | 3.1e-08 | **yes** | 30.999999968717820 | yes | toward opt |

In every case: (a) the flag-ON objective agrees with the TRUE optimum to
**correctness** tolerance (`incorrect_count` = 0), (b) the ON objective is **closer
to** (st_e38, ex1225-root) or **exactly on** (ex1225-node) the true optimum than the
OFF baseline, and (c) the dual bound never crosses the true optimum. Node counts are
identical (st_e38 3→3, ex1225 7→7) — this is not even a search-tree divergence; the
tighter root box lands the incumbent on a slightly different, more-accurate vertex.

The cert-neutrality check flags these because the drift exceeds the **1e-8
byte-reproducibility** threshold (`|Δobj|=3.17e-5 > 8.2e-6` for st_e38;
`4.82e-8 > 4.1e-8` for ex1225) — roughly **four orders of magnitude tighter** than
the actual correctness gate (`benchmarks.metrics.incorrect_count`, abs 1e-4 / rel
1e-3). That tolerance is right for a **bound-neutral** change (a refactor must
reproduce the objective exactly) but a **category error** for a **bound-changing**
flag (a reduction that, by design, changes the search).

### Held-out arm (seed 0, N=40, tl 25s) — the `n_soundness_violations` source

The graduation gate's `n_soundness_violations` come from `generality_sweep.run_arm`
→ `check_soundness` on the held-out sample, using the `[=bestdual=, =best=]`
bracket. Critically, `run_arm` computes `check_soundness` for **both** the OFF
control and the arm's ON solve, and the OFF control is **shared (cached) across all
arms**. So an OFF-control violation is counted in *every* arm — which is exactly the
pattern the graduation gate saw: **all five parked flags reported the same base
count** (node_reduce 4, psd_cost_gate 4, lift_zero_spanning 4, lift_loose_products
4; root_fixpoint 2 — root_fixpoint's tightening changes the count on 2 of the OFF
violations).

The offending instance is **`kall_circles_c8a`** (a circle-packing QCQP). On the
DEFAULT (flags-OFF) path, discopt certifies (reproduced on this branch's build):

```
status=optimal  obj=3.6142347590019357  bound=3.6142347419851370  nodes=3  gap=0.0
```

but the MINLPLib oracle is `=best=`=**2.5409191380**, `=bestdual=`=2.5409129340.
For a **minimize** problem (`.nl` `O0 0`) the certified dual **lower** bound
3.6142 **exceeds a feasible objective 2.5409** — an impossible bound, a genuine
FALSE CERTIFICATE. It reproduces **byte-identically flag-ON** (node_reduce ON:
`obj=3.6142…, bound=3.6142…, 3 nodes`), proving it is a **core** bug, not an R2
effect.

Localization lead (for C-38): the **root McCormick LP bound is valid** (`-1e-9`,
≤ 2.5409) — so the root relaxation is sound. The bound then **climbs to 3.6142 over
3 nodes**, crossing the true optimum. The invalid bound is introduced during
**branching / per-node processing** (a presolve/FBBT or spatial reduction that
over-tightens a node box on the non-overlap `(xi−xj)²+(yi−yj)² ≥ (ri+rj)²`
structure, removing the region containing the 2.5409 optimum, so the surviving
nodes' bounds exceed the true optimum). Core layer, flags OFF — `crates/discopt-core`
presolve/FBBT or the spatial-branch node reduction, not `_jax/{root,node}_reduce.py`.

## Part 2 — Bucket classification

| flagged signal | instance(s) | bucket | why |
|---|---|---|---|
| cert-panel drift | st_e38 (root_fixpoint) | **B** (gate false-positive) | ON obj within correctness tol of, and closer to, true opt 7197.7271490; bound ≤ opt; flagged only by the 1e-8 byte-repro tolerance |
| cert-panel drift | ex1225 (node_reduce) | **B** (gate false-positive) | ON obj is **exactly** true opt 31.0; the drift *fixes* a 4.8e-8 baseline inaccuracy; flagged only by the 4.1e-8 byte-repro tolerance |
| held-out arm | `kall_circles_c8a` | **A** (GENUINE), but **core / flag-independent** | default-path dual bound 3.6142 > feasible 2.5409 (min) — a real false certificate; inherited by every arm via the shared OFF baseline; **not caused by R2** → C-38 |

**R2 itself introduces zero soundness violations.** Every R2-attributable signal is
either the shared-baseline C-38 (a core bug) or a bucket-B cert-panel drift. R2's
design invariants hold (tighten-only intersection; `objective <= cutoff` is a
non-strict inequality so the optimum is never removed; NS-safe bounds only).

## The fix (gate soundness-check correction, not solver math)

The over-strict check is `utils.cert_neutrality.check_neutrality`'s objective
tolerance applied to a bound-changing flag. The fix makes that check **regime-aware**:

- `check_neutrality(..., regime="bound_neutral")` (**default, unchanged**) keeps the
  1e-8 byte-reproducibility tolerance. `check_cert_neutrality.py` (the default-OFF
  main / CI 41-panel guard) uses this default, so the vendored-panel regression
  guard is untouched.
- `check_neutrality(..., regime="bound_changing", oracle=<cert-optima>)` judges the
  certified objective against the **true optimum** with the **correctness**
  tolerance (abs 1e-4 / rel 1e-3, matching `incorrect_count`). An objective is a
  violation only when it disagrees with `=opt=` beyond correctness tolerance (a
  genuine false certificate) — a benign toward-optimum drift is not. When no oracle
  exists for an instance, it falls back to a correctness-tolerance drift bound vs
  the baseline (still catches a gross wrong answer).

`graduation_gate.py::run_cert_neutrality` passes the flag's `regime`
(`gs.ARMS[flag]["regime"]`) and loads `cert-optima.json` as the oracle into the
worker. Node-regression stays a downgraded perf note for bound-changing flags
(pre-existing behavior). Status / missing-instance violations stay hard fails.

**The gate is NOT weakened below true correctness:** a certified objective crossing
the true optimum beyond correctness tolerance is still an `objective` hard fail
(unit test `test_bound_changing_still_catches_a_real_false_certificate`), and the
held-out arm's `[=bestdual=, =best=]` bracket check is unchanged.

### Before / after on the flagged instances

- **Pre-fix** (byte-strict default): ex1225 31.0 drift flags
  `objective |Δobj|=4.818e-08`; st_e38 flags `objective |Δobj|=3.169e-05`.
- **Post-fix** (`bound_changing` + oracle): both clear (0 objective violations) —
  the certified value agrees with `=opt=` to correctness tolerance.

Reproduced directly:
```
pre-fix (byte-strict default): flags = [('objective', '|Δobj|=4.818e-08 ...31.0')]
post-fix (bound_changing+oracle): flags = []
```

## R2's corrected verdict

With the cert-panel objective check corrected and C-38 identified as the true owner
of the held-out violations, **R2 has 0 R2-attributable soundness violations**. Its
graduation blocker is **performance** (held-out `regression_rate` 0.20 > the 0.10
threshold), not soundness — a perf concern, out of scope for this correctness task.

R2 still FIRES and helps (seed 0, N=40, tl 25s, this build): root_fixpoint benefits
`spring` (49→27 nodes), `tln12` (639→607); node_reduce benefits `sonet22v5` (7→3),
`tln12` (607→575), `spring` (49→45). The gate fix does not disable the reduction.

## C-38 (handoff) — default-path false-optimal on `kall_circles_c8a`

A genuine core P0, discovered during this triage but **separate from CUTOFF-SOUND-1's
scope** (R2 cutoff-reduction). Reproduce (flags OFF): `from_nl(kall_circles_c8a.nl)
.solve(time_limit=60)` → `optimal obj=3.6142 bound=3.6142 3 nodes`, true min 2.5409.
Root McCormick bound is valid (≈0); the invalid bound appears during branching. Fix
belongs in `crates/discopt-core` presolve/FBBT or the spatial-branch node reduction.
Handed to a fresh agent (own PR `fix(correctness): C-38 …`) per one-task-per-PR.

## Gates run

- `pytest discopt_benchmarks/tests/test_cert_neutrality_regime.py` — 6 passed
  (the two toward-optimum tests FAIL on `origin/main` / byte-strict default, pass
  after the fix; a real false certificate still flags).
- `ruff check` + `ruff format --check` on the three changed files — clean.
- `mypy --ignore-missing-imports` on the two changed modules (strict config) — clean.
- `pytest -m smoke` (benchmark + repo) — see PR body.
- `pytest -m slow python/tests/test_adversarial_recent_fixes.py` — see PR body.
- No `python/discopt/` files touched → the pre-commit mypy hook
  (scoped to `python/discopt/`) and the default-OFF 41-panel cert guard
  (`check_cert_neutrality.py`, which calls `check_neutrality` with the unchanged
  `bound_neutral` default) are unaffected. No Rust touched → no `cargo test`.
