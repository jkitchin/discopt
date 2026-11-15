# Pseudocost sentinel-child guard: measurement

`TreeManager::process_evaluated` feeds every evaluated child's `local_lower_bound` into
`Pseudocosts::update` before pruning. A child the Rust MILP driver found LP-infeasible
carries the `1e30` exclusion sentinel (`sentinel_is_exclusion`), which `import_results`
keeps as its bound, so the update recorded `(1e30 - parent_lb) / frac` as a bound gain.
The guard drops observations whose parent or child bound is non-finite or `|b| >= 1e20`
(the LP layer's INF), count included.

Instrument: `discopt_benchmarks/scripts/pseudocost_sentinel_panel.py` (profile counters
`PseudocostUpdates`, `PseudocostSentinelSkipped`, `PseudocostUntrustedChild`).
Population: `p42_rens_panel.population()` — MIPLIB-38 + HiGHS v1.15.1 `check` panel M
(62 rows, 58 distinct names), Rust MILP route, `DISCOPT_PANEL_DETERMINISTIC=1`,
`max_nodes=5000`, `time_limit=120`, 3 jobs on a 4-core container (load 1-10).
Builds: `b4931e6b` = counter only (before), `5c9e482b` = counter + guard (after).

## Did it fire? (before build)

Yes. The sentinel reached `Pseudocosts::update` on **27 of 58** instances: 4,381 of
105,035 updates (4.2 %) over the 58 distinct instances (4,594 of 110,936 over all 62
rows; four instances sit in both panels). Largest shares: `enlight8` 1198/4042 (30 %), `neos17` 929/5309,
`beavma` 372/3786, `neos-2624317-amur` 379/2164, `enlight_hard` 266/3183, `lseu`
227/1457, `fiber` 196/4132, `neos-3611689-kaihu` 149/2873, `flugpl` 101/328.
`PseudocostUntrustedChild` equals `PseudocostSentinelSkipped` up to a handful per
instance: the sentinel children are the untrusted ones; failed-LP children (floored at
the parent bound, read as a zero gain) are rare (at most 4 per instance, on 9 instances,
5 of them instances where the sentinel never fired).

## Certificates (both builds, 216 / 212 executed oracle checks)

0 violations in both: no certified objective off the oracle, no bound past the oracle,
no incumbent better than the oracle, every returned point `max_viol <= 1e-6`.
Certified 35/58 in both; `CERT_LOST=[] CERT_GAINED=[]`.

## Node counts

Certified rows are deterministic: an after-build rerun (`after_rep.jsonl`) reproduced all
11 certified fired instances' node counts exactly. Rows that end on `time_limit` are
load-dependent and not compared; `mik-250-20-75-2` changed (5029 -> 5119) with zero
sentinel hits, so node-limited uncertified rows are not deterministic either.

Instances certified in both arms (35): **18,197 -> 17,951 nodes (-1.4 %)**; of those,
the 15 where the guard fired: 14,119 -> 13,873.
Down: `neos-3611689-kaihu` 2179 -> 1803, `p0201` 657 -> 625, `flugpl` 237 -> 207,
`supportcase14` 101 -> 91, `supportcase16` 97 -> 83, `dcmulti` 697 -> 685.
Up: `gen` 623 -> 761, `22433` 103 -> 143, `23588` 3919 -> 3949, `lseu` 1081 -> 1099,
`neos-3611447-jijia` 3545 -> 3547.

Larger budget (`*_big.jsonl`, 50k nodes / 90 s, 1 job, builds run sequentially):
`fiber` 6473 -> 5273 and `bell5` 6467 -> 5137 (both certified both arms), `blend2`
5813 -> 8705 (certified both arms).

## Primal side: a regression the guard exposes

On `enlight8` (30 % of updates were sentinels) the old build finds the optimum 27 within
5000 nodes (uncertified); the guarded build finds **no incumbent** within 5000 nodes or
within 50k nodes / 90 s. `blend2` likewise ends its 5000-node budget without an incumbent
(it certifies with the larger budget). The polluted pseudocosts were acting as an
accidental, infinitely-weighted fail-first rule: a variable with an infeasible child
scored ~1e30 and was branched on first. That infeasibility signal is real information
— HiGHS keeps it as a separate cutoff-rate term in its branching score
(`HighsPseudocost::addCutoffObservation`, `getScore`) — but as a bound gain it was a
numerical accident.

The guard alone also broke a test on a default route:
`test_1658_pounce_tol_and_oa_master.py::test_issue_portfolio_certifies_on_the_route[perspective]`
failed 3/3 with it and passed 3/3 without it, on highspy 1.16.0 and again on 1.15.1. The
Rust lazy OA master no longer closed its gap in time and the auto-route fell back
(the answer stayed certified-optimal; the test forbids the fallback). That solve made
334 updates of which 4 were sentinels before the guard: a handful of near-ties decided
the outcome.

## The signal restored as a cutoff-rate term

An infeasible child is now a **cutoff observation** (per-direction counts, as HiGHS's
`addCutoffObservation`), never a cost observation. `Pseudocosts::score` follows HiGHS's
`getScore` (`highs/mip/HighsPseudocost.h` at `04024d70`):

    score = map(cost / avg_cost^2) + 1e-4 * map(rate_down * rate_up / avg_rate^2)

with `cost` the existing product `(1e-6 + d)(1e-6 + u)`, `rate = cutoffs / (cutoffs +
cost observations)`, and `map(s) = s / (1 + s)`. Only the order of scores is used.

**Falsified on the way (recorded, CLAUDE.md §11):** HiGHS writes `map` as `1 - 1/(1+s)`.
I claimed that with the cutoff weight at 0 the new score ranks exactly as the old
product. Measured: false. `w = 0` passed `test_1658` while the guard-only build failed
it, because `1 - 1/(1+s)` cancels for small `s` (exactly 0 below ~1e-16), turning
near-ties into exact ties broken by index. With `s / (1 + s)` the claim holds:
`cutoff_w0.jsonl` has the guard-only node count on all 35 certified rows.

Arms (`cutoff_w*.jsonl`, same population and settings as above, temporary weight
override, removed before merge; `pseudocost_cutoff_arms.py`, `cutoff_arms.summary.log`):

| arm | certified | violations | nodes, 35 certified in all arms | no incumbent at 5000 nodes |
|---|---|---|---|---|
| before (polluted) | 35 | 0 | 18,197 | 4 |
| guard only | 35 | 0 | 17,951 | 6 (+`blend2`, `enlight8`) |
| w = 0 | 35 | 0 | 17,951 (identical per row) | 6 |
| **w = 1e-4 (shipped)** | 35 | 0 | 17,953 | 6 |
| w = 1e-2 | 35 | 0 | 17,937 | 6 |
| w = 1 | 36 (+`fiber`) | 0 | 18,389 (`lseu` 1081 -> 1539) | 6 |

`test_1658[perspective]`: passes at every `w > 0`, fails at `w = 0`. Larger budget at
`w = 1e-4` (`cutoff_w1e-4_big.jsonl`, 50k nodes / 90 s): `blend2` 8705, `fiber` 5273,
`bell5` 5137 (identical to the guard alone; before: 5813 / 6473 / 6467); `enlight8`
still no incumbent (before: the optimum 27).

Reading: at a sane scale the infeasibility signal is a tie-breaker. It repairs the
near-tie case (`test_1658`) at no measurable panel cost, but **no weight restores the
`enlight8`/`blend2` incumbents** -- what helped them was an effectively infinite weight,
i.e. pure fail-first branching on a feasibility puzzle. HiGHS's weight is kept rather
than one fitted to this panel (`1e-2` is within noise of it; `1` costs nodes).
