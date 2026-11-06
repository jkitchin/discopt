# DISCOPT_NLP_LADDER_SCOPED §5 panels (#1917, retired)

Tree: main `f43f29f7` (after #2161, before #2169), worktree-loaded (`_nlp_ladder_scope.__file__` asserted).

| file | command |
|---|---|
| `gdplib_120s.jsonl` | `issue1917_gdp_flag_panel.py --flag DISCOPT_NLP_LADDER_SCOPED --phase solve --all --reps 2 --time-limit 120 --max-load 6` (24 GDP parity instances x {hull, big-m}, arms interleaved; the flag was given a `FLAG_METHODS` entry `("hull", "big-m")` for the run) |
| `corpus_20s.jsonl` | `flag_onoff_corpus_panel.py --flag DISCOPT_NLP_LADDER_SCOPED --solu minlplib.solu --time-limit 20 --jobs 3 --reps 2` (in-repo `minlplib_nl` corpus, 74 instances) |

Results:

- **GDPlib, 96 pairs.** Certified 60 OFF, 60 ON. Wall geomean ON/OFF is 0.959 over the 60 pairs that certify in both arms.
- **hda hull, both reps identical** (maximize):

  | arm | nodes | incumbent | bound |
  |---|---|---|---|
  | OFF | 15 | 4824.32 | 112572.07 |
  | ON | 7 | 2323.22 | 121140.98 (the root bound) |

- **syngas big-m** certifies OFF in 54–55 s and 3 nodes.
- **Corpus, 148 pairs.** Certified 116 OFF, 116 ON. No infeasible incumbent in either arm. Wall geomean 0.960. The bound differences are noise: tspn08 swaps between reps.

Verdict: retired. The flag is cert-clean but not net-positive. See the audit row in `docs/dev/flag-retirement-audit.md`.
