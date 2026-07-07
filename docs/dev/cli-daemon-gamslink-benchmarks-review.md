# CLI / Daemon / GAMS-link / Benchmarks Review

**Date:** 2026-07-03
**Scope:** `cli.py` (620), `daemon.py` (219) + `_daemon_core.py` (498),
`gams/` solver-link (7 files, ~1,688 — GMO/GEV API, distinct from the already-
reviewed `modeling/gams_parser.py`), `benchmarks/` in-package hooks (5 files, ~690).
**Method:** Delegated verification; testable paths (CLI solve, daemon round-trip,
GAMS sense, benchmark metrics) exercised empirically. All 111 relevant tests pass.

**Bottom line: no high-severity bug** — none of these modules silently misreports a
solve status/objective or corrupts state between solves. Findings are minor edge
cases and one latent fail-open default worth tightening.

---

## 1. Verified correct (the load-bearing concerns)

- **CLI `solve` reports faithfully.** `alan.nl` → 2.925, `ex1221.nl` → 7.66718,
  both matching the MINLPLib oracle and the in-process solve. Exit codes correct
  (1 for missing/non-`.nl`/infeasible/unbounded, 2 for argparse/`--tuning`
  errors). Lazy subcommand imports (tutor/doe) work.
- **Daemon round-trip + state isolation.** Daemon results match in-process
  bit-for-bit; `alan → ex1221 → alan` in one warm daemon gives a stable 2.925 both
  times (`test_no_cross_problem_contamination` asserts bit-for-bit and passes).
  Each request rebuilds a fresh `Model` from the `.nl`; exceptions are isolated
  into `{"ok": False, "error": …}` so one bad model can't take the daemon down.
- **GAMS objective sense/sign.** For a maximize, `result.objective` is un-negated
  to natural sense before `gmoSetHeadnTail` writes it; `obj_nl_sign` is applied
  consistently to the nonlinear part; instruction stack/arg ordering verified
  (binary `div`, n-ary fold). Discontinuous intrinsics (`ceil/floor/round/mod/…`)
  are **rejected loudly** rather than silently mis-solved.
- **Benchmark metrics catch the dangerous direction.** `n_incorrect` flags an
  "optimal" claim whose objective is *below* the oracle (false-optimal) — the
  correctness-critical case.

## 2. Findings (all SUSPECTED / minor)

| # | Loc | Finding |
|---|-----|---------|
| CD-1 | `gams/gmo_translate.py:72`, `_finite:160-169` | An unbounded-above GAMS **integer** is capped at `ub=1e9` (`_finite(1e30,1e9)→1e9`); if the true optimum lies in `[1e9,∞)` it is cut off — a latent soundness edge for that case. Also asymmetric: the `0.0` default passed for the integer lower bound is dead for the `-inf` case. Rare (GAMS integers usually finite-bounded); continuous bounds are handled soundly (±1e20). Widen the integer cap or comment it |
| CD-2 | `benchmarks/metrics.py:82-90` | `n_incorrect` counts a legitimately non-converged `status=="feasible"` point (objective worse than oracle by > 1e-4) as "incorrect" — over-counts (never hides a real wrong answer) but conflates "wrong" with "not-yet-converged", a poor correctness signal for feasible results. (This is the *in-package* hook, not the phase gate in `discopt_benchmarks/`.) |
| CD-3 | `gams/link.py:80` | `getattr(result, "gap_certified", True)` defaults to **`True`** (certified-global) if the field were absent — the fail-safe default for a globality claim must be `False`. Latent only (the field is always present), but it is a fail-*open* default on a certificate flag — flip to `False` |
| CD-4 | `daemon.py:solve_via_daemon` | A hard-deadline SIGKILL makes the client re-send the same solve payload to a fresh daemon once before falling back in-process — wastes up to one extra full solve on the safety-wall path. Not incorrect, just inefficient in a rare path |
| CD-5 | cosmetic | `gmo_translate.py:83` drops an exactly-`0.0` GAMS level from warm-start hints (hint-only, no correctness impact); `_daemon_core.py` docstring claims a "length-prefixed JSON-line protocol" but the wire format is newline-delimited only (doc inaccuracy; `_recv_line` is correct) |

## 3. Plan (for Opus)

A small hygiene PR `fix(gams/benchmarks): CD-1..CD-5`: flip the `gap_certified`
default to `False` (CD-3 — the one with a whiff of soundness); widen or document
the integer 1e9 cap (CD-1); optionally split `n_incorrect` into "wrong-answer" vs
"not-converged" so the feasible gap doesn't read as incorrectness (CD-2); avoid the
redundant re-solve on the deadline-kill path (CD-4); fix the two docstrings. None
blocks; CD-3 is the only one touching a certificate flag.
