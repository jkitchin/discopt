# #2142: branching-variable data (performance-plan §108)

Every script asserts that `discopt._rust` was loaded from `DISCOPT_TREE` (CLAUDE.md §8)
and prints an executed-assertion count (§6). The capture `cap20k.npz` is §107's
(`docs/dev/data/2142-branching/s3_capture.py 20000 200 cap20k.npz` regenerates it from the
deterministic tree; not committed, 2.4 MB). The measured build is commit `feat(kernel):
#2142 — SCIP's violation measure` (`DISCOPT_SPATIAL_VIOL_SCIP`), retired in the next commit.

| file | what |
| --- | --- |
| `v2_scip_lp.py`, `v2_scip_lp.json`, `v2_scip_lp.tail.log` | §107's `s3_scip.py` replay re-run with `propagating/obbt/freq=-1` (the in-tree emulation), also recording SCIP's full last root LP vector (`scip_x`). Reproduces all 195 of §107's SCIP decisions |
| `v1_rules.py`, `v1_ourlp.log`, `v1_scipx.log` | candidate violation rules scored offline on the 192 boxes whose kernel node LP re-solves `optimal`, at the kernel's own LP point (`v1_ourlp.log`) and at SCIP's (`--scip-x`, `v1_scipx.log`). `top1`: the rule's best column is SCIP's; `inset`: SCIP's column is in the rule's `>= 0.9 x max` set (SCIP picks at random inside that set); `expect`: the expected top-1 agreement of a rule that also picks at random in it |
| `v3_arms.py`, `arm_<arm>_<nodes>.log` | cstr big-M tree arms, kwargs injected into `solve_spatial_tree_py`, deterministic: `base`, `viol` (`spatial_viol_scip`), `viol_nopc` (and pseudocost weight 0). Three concurrent processes, so walls are indicative only. `arm_viol_150000.log` is the nodes-to-certify run (one process next to the panels) |
| `panel/corpus.*` | `issue2115_graduation_panel.py --flag DISCOPT_SPATIAL_VIOL_SCIP --deterministic --max-nodes 2000 --time-limit 120 --assert-unfired-identical --jobs 3`, 74 in-repo instances (oracle `minlplib.solu` + `cert-optima.json`), load 6-10 on 4 cores |
| `panel/rerun.*` | the two unfired wall-backstop rows the corpus run flagged (`cvxnonsep_psig30`, `heatexch_gen1`), re-run at lower load |
| `panel/v4_screen.py`, `panel/screen.*` | ON-only fire screen (2000 nodes / 30 s) over §107's 60 extra instances (`python/tests/data/minlplib` not in the corpus, plus nous1 and st_m2) |
| `panel/extra.*`, `panel/extra_nous1.*` | the panel above over the 10 instances the screen found the rule acting on (`--extra-dir python/tests/data/minlplib` / `issue_1871`) |
| `panel/v5_one.py`, `panel/nous1_cert_{0,1}.log` | nous1 to certification (`gap_tolerance=1e-4`), both arms, concurrently |
