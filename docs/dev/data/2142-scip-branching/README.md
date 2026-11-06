# #2142 item 3: combined SCIP branching kill test (performance-plan §110)

`scip_branching.patch` is the retired implementation (commit `39c1ee3c`, reverted in
the next commit; applies to `1e8d957e`): the §107 point rule (clamp 0.2, midpull
reldomtrig 0.5, from `4ed8ea1e`), the §108 violation measure (relative gap /
`max(1, |f|)`, midness split, from `7f964225`) and the §109 pseudocost reliability /
normalisation (`spatial_pscost_reliable = 2`, `2142-pscost-norm/pscost_norm.patch`),
all behind one flag `DISCOPT_SPATIAL_BRANCH_SCIP` (default OFF).

`k_one.py FILE MAX_NODES -` runs one deterministic solve to certification (default
gap tolerance); it asserts the marker helper is importable from the worktree (§8)
and, with the flag on, that both the scored rule (`SpatialViolScipPicks`) and the
point rule (`SpatialSplitMoved`) acted; with it off, that neither did (§6).
`k_<instance>_<flag>_tail.log` are the last 40 lines of each run (profile counters
+ result line). The four runs ran concurrently on a 4-core container, so wall
times are indicative only; node counts are deterministic.
