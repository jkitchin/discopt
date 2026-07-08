# CORR-SWEEP — broad certificate audit vs oracle (2026-07-07)

**Task:** systematically surface remaining default-path false-optima by checking
discopt's certificates against the independent MINLPLib oracle across a broad,
bug-family-weighted corpus. This is a **measurement**, not a fix: any real
violation becomes its own C-issue + dedicated fix (like C-38 #536 / C-40 #543).

Branch cut from `origin/main` (contains the C-38 and C-40 fixes). Solve at
**defaults** (all shipped capabilities default-ON), `time_limit = 30 s`,
`deterministic=True`, one subprocess per instance (crash/hang isolation, hard
kill at 90 s).

## Headline

| metric | value |
|---|---|
| Instances solved | **250** |
| Certificates produced (`optimal` \| `infeasible`) | **91** (91 optimal, 0 infeasible) |
| Certificates checked against oracle | **91** |
| **True correctness violations** | **0** |

**No false-optimal, no false-infeasible, no bound-cross on this 250-instance
sample.** 90 of 91 `optimal` certificates match the primal oracle within
tol (abs 1e-6 / rel 1e-4). The one certificate that differs from the recorded
`=best=` is a *legitimate improvement* over a stale best-known value, fully
consistent with the oracle's own dual bound (details below) — it is **not** a
violation.

This is strong evidence that the default-path certificate surface is clean on
this sample post-C-38/C-40. It does not prove the surface is globally clean;
it narrows the remaining risk to instances outside this stratification.

## Corpus and stratification

250 instances drawn from `~/Dropbox/projects/discopt-minlp-benchmark/`
(`minlplib/nl/`, oracle `minlplib.solu`), all with a known oracle value and
`vars ≤ 500`, seeded (`20260707`) and stratified to over-weight the families
where bugs have clustered:

- **Quadratic / bilinear** over-weight: QCP 54, MBQCP 43, QP 29, BQP 15,
  MIQCP 14, QCQP 6, plus IQCP/IQCQP/MIQP/BQCP/BQCQP/MBQCQP/MIQCQP/IQP.
- **Geometric / circle-packing** (kall_\*, packing/pointpack/ringpack, clay,
  nous): 44 instances.
- **Small-node** (≤ 40 vars, where a false-optimal often certifies in few
  nodes): 133 instances. Median node count among the 91 optimal certs = 19.
- Broad random fill across all remaining types for coverage
  (MBNLP 26, NLP 22, MINLP 13).

Oracle strength: 199 instances carry a proven `=opt=`, 51 a best-known `=best=`.

## Status breakdown (250)

| status | count | is it a certificate? |
|---|---|---|
| `optimal` | 91 | yes — checked vs oracle |
| `feasible` | 91 | no (honest gap: found a point, no proof) |
| `time_limit` | 44 | no (honest gap) |
| `hard_timeout` | 20 | no (killed at 90 s wall) |
| `error` | 4 | no — loud refusal (see below) |

The 44 `time_limit` + 20 `hard_timeout` + 91 `feasible` are honest gaps at a
30 s limit, not violations. The high gap rate is expected: the sample is
deliberately hard (nonconvex QCP/bilinear/geometric, small time budget).

## Violation table

**Zero true violations.** The audit applies three checks to every certificate:

1. **FALSE-INFEASIBLE** — `status=infeasible` on an instance with a finite
   oracle best. *(0 found — no instance certified infeasible.)*
2. **FALSE-OPTIMAL** — `status=optimal` with objective differing from a proven
   `=opt=` beyond tol, or an incumbent provably below a valid `=bestdual=`
   (min sense). *(0 found.)*
3. **BOUND-CROSS** — reported dual `bound` crosses the primal oracle in the
   invalid direction (an invalid under/over-estimator). *(0 found.)*

### Single soft-flag, adjudicated NOT a violation

| instance | type | certified obj | oracle `=best=` | oracle `=bestdual=` | sense | nodes | verdict |
|---|---|---|---|---|---|---|---|
| `hybriddynamic_fixedcc` | QP | 1.4735125711 | 1.4737777780 | 1.4735037640 | min | 3 | **legit improvement** |

discopt's certified optimum (1.4735125711) lies strictly *between* the oracle's
valid dual bound (`=bestdual=` 1.4735037640, a rigorous LB from reference
solvers) and the oracle's best-known primal (`=best=` 1.4737777780). i.e.

```
bestdual 1.47350376  ≤  discopt 1.47351257  ≤  best-known 1.47377778
```

The incumbent respects the valid lower bound and *beats* a stale best-known
primal by ~1.8e-4 rel. This is discopt closing a gap the MINLPLib `=best=`
left open, not a bogus certificate. Reproduce:

```
PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python -c \
"from discopt.modeling.core import from_nl; \
r=from_nl('<corpus>/minlplib/nl/hybriddynamic_fixedcc.nl').solve(time_limit=30); \
print(r.status, r.objective, r.bound)"
```

### `error` instances (loud refusals, not violations)

`du-opt`, `portfol_classical050_1`, `portfol_robust050_34`, `st_miqp5` all raise
`ValueError: binary .nl format not supported` — the `.nl` reader loudly refuses
the binary ('b') encoding rather than silently mis-parsing. This is the correct
"refuse loudly" behavior (CLAUDE.md §3), not a wrong certificate. (Follow-on: a
binary-`.nl` reader would let these ~4 instances be audited; tracked separately,
out of scope for this measurement.)

## Method notes / limitations

- Bound-cross is checked against `=bestdual=` where available — the mathematically
  correct oracle for a valid dual bound — not merely the primal `=best=`. An
  incumbent below a valid dual bound is impossible for a feasible point, so it
  would expose a bogus certificate; none was found.
- A better-than-`=best=` incumbent is treated as a *candidate improvement*, not a
  violation, unless it violates `=bestdual=` or a proven `=opt=`. Only wrong
  certificates count (per the task): timeouts and honest `feasible` gaps do not.
- 30 s is a short budget; a longer budget would convert some `feasible`/`time_limit`
  into certificates and widen the audited set. The clean result here is a lower
  bound on coverage, not the whole surface.

## Artifacts

(`/reports/` is gitignored in this repo, so the raw artifacts live in a tracked
subdirectory beside this doc.)

- Raw per-instance results: `docs/dev/corr-sweep-2026-07-07/results.jsonl` (250 lines).
- Instance selection + oracle join: `docs/dev/corr-sweep-2026-07-07/selection.csv`.

Each line of the JSONL records `status`, `objective`, `bound`, `sense`,
`node_count`, `gap`, `gap_certified`, `root_bound`, `wall`, and the oracle
metadata (`oracle_tag`, `oracle_val`, `probtype`, sizes) used for the check.
