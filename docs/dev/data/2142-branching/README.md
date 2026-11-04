# #2142 item 3: branching-point data (performance-plan §107)

Every script asserts that `discopt` and `discopt._rust` were loaded from `DISCOPT_TREE`
(CLAUDE.md §8) and prints an executed-assertion count (§6). `DISCOPT_LP_DUAL_REBUILD`
must be unset (retired). All tree runs are deterministic with a node budget.

| file | what |
| --- | --- |
| `s1_splits.py`, `s1_{10k,20k}.log`, `s1_*_summary.json` | step 1: every branching decision of the default cstr big-M tree, recovered from the `node_observer` stream (child pre-propagation box vs parent post-propagation box), with relative position and relative width |
| `s2_arms.py`, `runarms.sh`, `arm_<arm>_<nodes>.log` | step 2: cstr big-M arms with kernel kwargs injected (`base`, `clamp`, `trig`, `scip`, `scip_sf`, `lp_clamp`, `sf`) at 10k/20k nodes; `*_150000` is the nodes-to-certify run |
| `s3_capture.py`, `s3_capture.log` | step 3: 200 (box, cutoff, our decision) samples evenly over the 9,994 branched parents of the default 20k tree; the `.npz` (spec + samples, 2.4 MB) is not committed, and `s3_capture.py 20000 200 cap20k.npz` regenerates it from the deterministic tree |
| `s3_scip.py` | SCIP (PySCIPOpt 6.3, libscip 10) on discopt's spec with the node's box and cutoff, presolve/heuristics/root separation off, `limits/nodes=1`; decision from the root's children |
| `s3_nosep.json` | SCIP defaults otherwise: 118/200 boxes closed |
| `s3_inf_{nolim,noobbt,noproproot,noroot}.json` | the 118 closed boxes re-run without the objective limit / without OBBT / without root propagation / without OBBT, probing, nlobbt and root propagation |
| `s3_noobbt.json`, `s3_analyse.py`, `s3_noobbt_analysis.txt` | the in-tree emulation (OBBT off): variable agreement 27/195 |
| `panel/` | step 4: §5 panel rows and logs (see below) |

Panel files (`panel/`):

| file | what |
| --- | --- |
| `corpus.json.partial.jsonl`, `corpus.log`, `corpus.summary.json` | `issue2115_graduation_panel.py --flag DISCOPT_SPATIAL_BRANCH_CLAMP --deterministic --max-nodes 2000 --time-limit 120 --assert-unfired-identical --jobs 2`, 74 in-repo instances (oracle: minlplib.solu + cert-optima.json). The first verdict line in `corpus.log` reads FAIL only because that version of the script required the retired `DualRebuild*` counters; `corpus.summary.json` is the `--rows` re-derivation after the check was scoped to the flag's own counters (0 violations) |
| `s4_screen.py`, `screen_{a,b}.jsonl` | ON-only fire screen (2000 nodes, 30 s) over the 58 `python/tests/data/minlplib` instances not in the corpus, plus nous1 and st_m2 |
| `extra.*`, `extra_nous1.*` | the panel over the 12 instances the screen found the rule acting on |
| `s4_one.py`, `nous1_cert_{0,1}.log` | nous1 to certification, both arms |
| `gdp*.jsonl`, `gdp*.log` | `issue1917_gdp_flag_panel.py --phase solve --all --deterministic --max-nodes 5000 --time-limit 120 --reps 1`, 24 GDP parity instances x {big-m, hull}, split over three processes by `--include` and merged into `gdp_merged.jsonl`; `gdp_report.txt` is `--report` over it. The first process was stopped after `modprodnet` (it had written no row past it), and the other two covered the rest |

Load: 1-8 one-minute average on 4 cores (at most 3 solver processes, one Rust build
overlapping the first arms). The bounds and node counts of work-terminated rows do not
depend on load; walls and wall-terminated rows do.
