# #2142 Part A: learned dual-proof rows (performance-plan §112)

Every script asserts the loaded `discopt` / `discopt._rust` is this tree's build and carries
the #2142 kwarg/counters (CLAUDE.md §8), and prints an executed-assertion count (§6).
Build: branch `fix/2142-cstr-gdp-bound`; the flag `DISCOPT_DUAL_PROOF_PROPAGATION`
(default OFF while measured, graduated default ON in commit `4a263477`).

| file | what |
| --- | --- |
| `e1_arm.py` | cstr through `Model.solve`, deterministic, `MAX_NODES`, fixed rows permuted by `PERM_SEED` (as `2142-scip-expert/py/d3_arm.py`); prints the kernel's `dual_proof` counters |
| `pilot_{off,ray,p1}_s0.log` | design pilot, seed 0, 20k nodes: OFF / ray restricted to global rows / plus the global-rows feasibility-LP fallback (the chosen design) |
| `run_entry.sh`, `entry_s{0..3}_f{0,1}.log` | the pre-registered entry experiment (4 seeds x OFF/ON, 20k nodes, 3 concurrent) |
| `cert_bigm_on_det.log` | cstr big-M ON to certification, deterministic, cap 150k |
| `e2_parity.py`, `parity_bigm_{on,off}.log`, `parity_bigm_default_after_graduation.log` | cstr big-M on the default path at the GDP parity budget (300 s) |
| `hull_on_1k.log` | cstr hull ON, 1000 nodes (no effect) |
| `panel/corpus.*` | `issue2115_graduation_panel.py --flag DISCOPT_DUAL_PROOF_PROPAGATION --deterministic --max-nodes 2000 --time-limit 120 --assert-unfired-identical --jobs 2` over the 74-instance in-repo corpus (oracle: MINLPLib `minlplib.solu` + `cert-optima.json`) |
| `panel/rerun_tls2.*` | the corpus's one flagged row re-run at load 1.5 |
| `panel/run_extras.sh`, `panel/extra.*`, `panel/extra_nous1.*` | the same panel over section 107's 12 extras |
| `panel/v4_screen.py`, `panel/screen_files.txt`, `panel/screen.*` | ON-only fire screen (2000 nodes / 30 s) over the 58 `python/tests/data/minlplib` instances not in the corpus, plus st_m2 |
| `panel/screenfired.*`, `panel/rerun_carton7_{1,2}.*` | the panel over the screen's new acting instances; carton7's 120 s cell repeated twice at load 1 |
| `panel/v6_nodes.py`, `panel/carton7_nodes1300.log`, `panel/nous1_counters.log` | equal-node-budget runs with no binding wall (profile dump lines removed) |
| `panel/v5_one.py`, `panel/nous1_cert_{0,1}.log` | nous1 to certification, both arms (concurrent: node counts only) |
| `panel/v7_timing.py`, `panel/timing.log` | section 9 timing: tanksize and nous1, interleaved, load-gated, 3 reps |
| `panel/gdp_{bigm,hull}.*`, `panel/gdp_merged.jsonl`, `panel/gdp_report.txt` | `issue1917_gdp_flag_panel.py --phase solve --all --deterministic --max-nodes 5000 --time-limit 120 --reps 1`, one stream per method; `--report` over the merge |
