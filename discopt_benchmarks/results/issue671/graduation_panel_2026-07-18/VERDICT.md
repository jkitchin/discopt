# `DISCOPT_RELAX_ROW_FILTER` graduation panel (issue #671) — VERDICT: **HOLD**

Date: 2026-07-18. CLAUDE.md §5 two-bar graduation gate, ON-vs-OFF differential
panel over the 66-instance in-repo certifying corpus
(`python/tests/data/minlplib_nl/*.nl`) plus the hda target.

- Harness: `discopt_benchmarks/scripts/issue671_rowfilter_graduation_panel.py`
- Raw data: `panel_data.json` (sense-aware assessment); console: `panel_console.log`
- Settings: single process (warm JAX), deterministic, flag toggled via
  `DISCOPT_RELAX_ROW_FILTER` before each build; corpus TL=60 s, hda TL=120 s.
- Per instance, both arms: dual bound, objective, status, `gap_certified`,
  `node_count`, wall, filter rows-dropped, and an INDEPENDENT incumbent
  feasibility re-verification.

## Verdict: HOLD — the flag stays default-OFF / opt-in. Issue #671 CANNOT close.

Bar 1 (cert-clean) FAILS; bar 2 (net-positive) is therefore moot.

### Bar 2 deliverable IS achieved by the flag (hda)

| instance | oracle | OFF bound | ON bound | note |
|---|---|---|---|---|
| hda | −5964.534 | **−1.802e10** | **−6.447e4** (−64473.44) | ON tight & sound (≤ opt); 731 rows dropped in 6 builds |

hda's tight finite dual bound — the #671 target — is delivered with the filter
ON. But graduation requires the flag be safe corpus-wide, and it is not.

### Bar 1 (cert-clean) FAILS — the filter is NOT inert on the already-solving corpus

`incorrect_count = 0` (no unsound bound; every ON bound is sound by superset).
BUT the flag is a **bound-changing** lever and #671's own acceptance is
**bound-neutral**: `node_count` and certified `objective` EXACTLY unchanged on
every already-solving instance, and no certification regression. The filter
fires on **30 / 66** instances, and on **4 already-solving** instances it
degrades the result:

| instance | oracle | OFF | ON | filter | regression |
|---|---|---|---|---|---|
| **nvs05** | 5.471 | `optimal`, bound 5.471, **certified** | `feasible`, bound 5.461, **UNcertified** | dropped 28756 rows / 1947 builds | **LOST CERTIFICATE** (optimal→feasible) |
| **nvs09** | −43.13 | `optimal`, −43.13, **certified** | `feasible`, −43.9, **UNcertified** | dropped 8142 rows / 639 builds | **LOST CERTIFICATE** (optimal→feasible) |
| **nvs13** | −585.2 | `optimal`, 23 nodes | `optimal`, 21 nodes | dropped 1080 rows | node_count drift (not byte-identical) |
| **nvs21** | −5.685 | `optimal`, 3 nodes, obj −5.68478263 | `optimal`, 17 nodes, obj −5.68478261 | dropped 291 rows | node_count + certified-obj drift |

Each is a §5 bound-neutral violation on an instance that already solves OFF. On
nvs05/nvs09 the dropped rows carried GENUINE tightness — losing them loosened the
node bounds enough that the search could no longer certify optimality within the
time limit.

### Additional (SOFT, sound) — severe loosening on still-unsolved instances

Sound by superset (weaker relaxation) and on instances that time out in BOTH
arms, so not a hard cert-fail, but evidence the coefficient-spread trigger fires
destructively well beyond hda:

| instance | OFF partial bound | ON partial bound |
|---|---|---|
| ex14_1_9 | −1.98e-16 (≈ opt 0) | **−1.065e6** |
| 4stufen | 2.16e4 | 1.93e4 |
| beuster | 1.99e4 | 1.78e4 |
| casctanks | 2.969 | 1.738 |
| heatexch_gen2 | 5.585e5 | 5.573e5 |

### Sound flags that are NOT regressions

- **syn05hfsg**: the panel first flagged `bound 911 > oracle 837.7` in BOTH arms;
  syn05hfsg is a **MAXIMIZE** instance, so 911 is a valid UPPER dual bound and the
  incumbent 837.732 = oracle exactly. Sound, identical OFF/ON — a min-sense
  harness artifact, corrected by the sense-aware assessment. Not a filter effect.
- ex1221 (fired 4 rows), dispatch, nvs01/06/08, st_e11/e13/e36, tspn05: the filter
  fired but node_count, objective and bound were unchanged — dropped rows carried
  no tightness there. Neutral, not regressions.

## Root-cause finding (surface loudly)

The MERGED code applies the filter **unconditionally at the tail of
`build_milp_relaxation`** (`python/discopt/_jax/milp_relaxation.py:1957`) — i.e. at
the root build AND every per-node cold rebuild — whenever a coefficient-spread row
exists. This is exactly the **"always at build time" first cut that PR #746's own
body states was KILLED** by the in-repo panel ("10/66 regressions, incl. nvs09
dropping optimal→feasible"). PR #746 describes a subsequent refactor to
**failure-triggered** firing ("fire only on a failed node solve, so already-solving
nodes are byte-identical") — but that refactor is **NOT present in the merged
code**. This panel reproduces the first-cut regressions (nvs05/nvs09 cert loss,
nvs13/nvs21 node drift), confirming the shipped flag behaves like the rejected
build-time version, not the validated failure-triggered design.

## What is required before this can graduate

The filter must be made genuinely **failure-triggered** — fire only when a node LP
breaks down numerically (`numerical` / spurious `infeasible`), so already-solving
nodes are byte-identical — OR otherwise gated so it is provably inert on
nvs05/nvs09/nvs13/nvs21 and the rest of the already-solving corpus. That is
follow-up implementation work on the mechanism, not a graduation decision. The
flag stays default-OFF and #671 remains open.
