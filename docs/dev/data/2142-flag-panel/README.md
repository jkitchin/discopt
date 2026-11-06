# #2142 flag panels

## `DISCOPT_HULL_LIFTED_PERSPECTIVE` (retired)

`DISCOPT_HULL_LIFTED_PERSPECTIVE.jsonl`: `issue1917_gdp_flag_panel.py`, fire then
solve phase. Verdict and numbers: the audit row in `docs/dev/flag-retirement-audit.md`.

## `DISCOPT_LP_DUAL_REBUILD` (retired 2026-11-04)

CLAUDE.md section 5 panel, flag ON vs OFF, every solve in a fresh process with
`DISCOPT_PROFILE=1` and `PYTHONHASHSEED=0`, `deterministic=True`, arms interleaved
(order alternating per instance). The counter `DualRebuilds` is non-zero exactly
where the mechanism ran; it was 0 in every OFF arm.

| file | what |
| --- | --- |
| `*.corpus.seg1-300s.jsonl` | in-repo corpus (`python/tests/data/minlplib_nl`), first 9 instances, `max_nodes=2000`, 300 s backstop, 2 jobs |
| `*.corpus.seg2-120s.jsonl` | the other 65, `max_nodes=2000`, 120 s backstop, 2 jobs (the 300 s run was too slow for the session; restarted) |
| `*.corpus.summary.json` | `issue2115_graduation_panel.py --rows seg1,seg2 --assert-unfired-identical` over all 74 (oracle: minlplib.solu `=opt=`/`=best=` + `cert-optima.json`) |
| `*.gdp-bigm.jsonl`, `*.gdp-hull.jsonl` | `issue1917_gdp_flag_panel.py --phase solve --all --deterministic --max-nodes 5000 --time-limit 120`, 24 GDP parity instances x {big-m, hull}, 1 rep |
| `*.gdp-followup-1000n.jsonl` | spectralog and constrained_layout_clay0203 big-M, `max_nodes=1000`, 600 s backstop, 2 reps (work-terminated) |
| `*.followups.txt` | A/B/A/B sequential repeats: cvxnonsep_psig30, launch, contvar |

Load (`uptime`) during the runs: 1.4-9.6 one-minute average on 4 cores (another
session was building Rust); the deterministic node budgets make the work-terminated
rows load-independent, the wall-terminated ones are not.

Result: cert-clean, not net-positive (see the audit row).
