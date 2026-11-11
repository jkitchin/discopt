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
numerical accident. Restoring the signal as a properly scaled cutoff term is a separate
branching change needing its own panel.
