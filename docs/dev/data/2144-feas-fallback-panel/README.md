# #2144: feasibility-first fallback panel (2026-11-05)

Graduation panel for `DISCOPT_GDP_CONFIG_FEAS_FALLBACK`, and the measurement of the
`_reduce_subnlp` non-finite-derivative fix that ships with it.

## Arms

| arm | code | flag |
|---|---|---|
| `main` | `origin/main` at `a0b3b5e1` (worktree on `PYTHONPATH`) | absent |
| `off` | branch at `21d4c282` | `=0` |
| `on` | branch at `21d4c282` | `=1` |

Every row records the `git_head` of the `discopt` it imported; the report refuses a
row whose head does not match its arm (the only mismatches are kaibel's, below,
which never imported a solve).

- **GDP panel** (`gdp_parity_3arm.jsonl`): `gdp_parity` 24 instances x {big-M, hull},
  120 s, the three arms back to back per cell in a rotating order, 3 cells at a
  time on a 4-core host. Load 1-min: 0.3-7.9 (median ~5), so walls and node counts
  at the limit are noisy; the bars below are statuses and incumbents.
- **Corpus fire phase** (`corpus_fire_30s.jsonl`): the 74 in-repo `.nl` instances,
  flag ON, 30 s, recording the fallback's firing counter.

`panel3.py` / `corpus_fire.py` are the drivers as run (their paths point at the
session scratchpad). `gdp_parity_3arm.report.txt` is `panel3.py --report`.

## Environment notes

- The host had no `ipopt` executable. gdplib's `mod_hens` builder solves a 2x2
  square system with it (Cafaro coefficients); a harness-only `sitecustomize`
  solved the same system with `scipy.optimize.brentq` (residual < 1e-8,
  k = 14.9153, b = 0.0305925). Not in the repo.
- `kaibel`'s builder runs ipopt initialisation solves of the column; it could not
  be built, and its 6 rows are `error` in every arm (excluded).

## Result

Cert-clean in all three comparisons: 0 incorrect, 0 dual bounds past an oracle,
0 solved instances lost (27 -> 27 -> 27). Every incumbent re-checked in Pyomo.

**Flag (off -> on).** It fired on exactly 2 cells, both gaining their first
incumbent: batch_processing big-M 1034233.32 and hull 1019222.46 (OFF: none in
120 s; BARON 679365.33). Each was found on the first sample. No other cell changed
objective. On mod_hens big-M (the #2150 kill cell) 106766.53 in 445 nodes in all
three arms. gdp_col hull had an incumbent OFF (21360.46) and none ON, but the
fallback did not fire there (counter 0, so ON ran OFF's code); a repeat gave none
in both arms, and main had none: run-to-run variance at 7-15 nodes. Corpus: fired
on 0 of 74.

**`_reduce_subnlp` fix (main -> off).** mod_hens hull gains 106766.53 (BARON's best
known; main none). One objective regression: **methanol hull -1793.43 -> 645.51**.
Traced (`main` vs branch, flag OFF): on `main` the LOA proposer's point is falsely
refused by the reduced pose, so no incumbent exists and the root feasibility pump
runs and finds -1793.43 at ~90 s; on the branch the proposer's point (645.51,
genuinely feasible) is accepted, and `DISCOPT_ROOT_PUMP_SKIP_WITH_INCUMBENT` then
skips the pump. The fix is correct; the loss comes from the pump-skip rule
acting on a poor first incumbent.
Tracked in #2170. batch_processing's remaining dual gap (no certificate at 120 s)
is #2172.
