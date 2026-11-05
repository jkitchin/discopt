# #2142 item 3: pseudocost normalisation kill test (performance-plan §109)

`pscost_norm.patch` is the uncommitted experimental implementation (applies to
`0194eb79`): SCIP-style pseudocost reliability (`spatial_pscost_reliable=2`) —
unreliable pseudocosts dropped and each candidate's score divided by the sum of
the weights actually used — optionally combined with the §108 violation measure
(`spatial_viol_scip`). `k1_one.py <instance> <mode>` runs one deterministic solve
(mode `0` base, `1` measure+normalisation, `norm` normalisation alone).
`*_tail.log` are the last 40 lines of each run (profile counters + result line;
the full logs were 2.7 MB each and are not committed).
