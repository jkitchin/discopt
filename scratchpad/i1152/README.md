# scratchpad/i1152 — measuring the `time_limit` root-setup contract

Probes for issue #1152 ("root-setup overrun and deadline-gated bound loss are
contradictory as tested"). Every one prints an executed-comparison/event count and
exits non-zero at zero (CLAUDE.md §6); none swallows an exception (§7); the
long-running ones print per-record progress under `python -u` (§10).

Run from the repository root with an interpreter that has `discopt` installed.

## The instruments

| script | what it answers |
|---|---|
| `overrun_sweep.py <T[,T...]> [names]` | which in-repo instances overrun `solve(time_limit=T)`, and which return no bound |
| `phase_attrib.py <inst> <T>` / `phase_attrib2.py` | where the wall goes, phase by phase, relative to the solve start (`phase_attrib2` adds the solver's own log timeline) |
| `phase_attrib3.py <inst> <T>` | **which caller** builds a relaxation late in root setup — prints the stack of every `build_uniform_relaxation`, plus the `build_deadline` it was given |
| `fb_probe.py <inst> <T>` | did the root-relaxation fallback run at all, and with what grant |
| `log_timeline.py <inst> <T>` | timestamped `discopt` log lines for one solve |
| `panel.py --time-limits 5,20 --out <f>.jsonl` | the §5 differential panel: the in-repo corpus, flag OFF vs ON, **interleaved** per instance |
| `panel_report.py <f>.jsonl` | the panel's verdict — soundness, certification, bound quality, punctuality |

## What they found

`phase_attrib3.py casctanks 5` is the one that located the defect. It shows root
OBBT entering `build_milp_relaxation` at t=4.39 s with 0.61 s of a 5 s budget left
and spending 1.85 s in it; `fb_probe.py casctanks 5` then shows
`# fallback_calls=0` — the #654 short-circuit reached `_remaining_budget() == 0`
and skipped the only remaining bound producer. So the 1.29x overrun and the
`bound=None` are one defect, not the two contradictory contracts the issue read
them as.

The first two attribution probes are kept even though `phase_attrib3` supersedes
them, because they are the reason the search narrowed: `phase_attrib.py` wrapped
`_root_relaxation_lower_bound` and recorded **zero** calls while three relaxation
builds ran, which is what showed the fallback was never entered. A probe that
records nothing is a finding only when you can tell it fired — `# events=` is the
counter that made that readable.

## Raw data

* `sweep_T5.log` — the entry measurement: the in-repo corpus at `time_limit=5` on
  `main` (max overrun 1.38x; `casctanks` and `bchoco08` return no bound at all).
* `panel_T5_T20.jsonl` — the graduation panel, both arms, one record per
  (instance, `time_limit`, arm).
* `hda_T8.jsonl` — the 3-rep interleaved A/B behind the one counter-example the
  panel doc records (`hda` at 8 s goes 1.21x -> 1.28x with a bit-identical bound).
* `oracle_probe.py` — the 120 s incumbent hunt that supplies the soundness oracle
  for the instances whose bound moved (`casctanks` 9.163 feasible, `tanksize`
  1.26864 certified optimal; `4stufen`/`beuster`/`bchoco08` yield no feasible point
  at any budget tried, which is recorded rather than papered over).
