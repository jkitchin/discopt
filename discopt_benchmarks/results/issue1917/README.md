# #1917: does legacy (`DISCOPT_OBBT_NS_STRICT` OFF) OBBT/DBBT cut off feasible points?

Probe: `discopt_benchmarks/scripts/obbt_ns_unsafe_acceptance_probe.py` (its header
holds the decision rule, written before the runs). Run 2026-11-01 on `54c7f944`
(main after #2077/#2080), root `obbt_tighten_root` only (rounds=3, cutoff = the
reference point's internal objective + 1e-6(1+|f|), 90 s deadline per arm),
`DISCOPT_NATIVE_SPATIAL_KERNEL=0` for the GDPlib rows. Machine load 70-90: no
timing claims.

Corpus: the 74 in-repo `minlplib_nl` instances (reference = MINLPLib `.p1.sol`
mapped through `.col`; bchoco06-08 have none), 73 QPLIB instances
(`usable_oracle`, `ncons >= 1`, `2 <= nvars <= 400`, smallest first, run until 60
had OBBT probes), and GDPlib clay0203 / ex_633_trespalacios under hull and big-m
(reference = discopt's certified optimum). Every reference point is verified with
`check_feasibility(tol=1e-6)` before use (147 of 151 pass; heatexch_gen2 fails and
is not used as a reference).

| | count |
|---|---|
| instances where OBBT probed | 117 of 151 |
| OBBT probes examined (OFF) | 28108 (27913 with an NS bound `g`) |
| root DBBT passes examined | 281 |
| unsafe acceptances: raw beyond `g` (kind a) | 27771 (6084 changed a bound) |
| unsafe acceptances: raw with `g` None (kind b) | 0 |
| DBBT legacy tighter than strict | 389 (strict never abstained) |
| max excess `raw - g`, finite bounds | 0.0495 (QPLIB_3385, raw -1.69e5; relative 2.9e-7) |
| max excess, DBBT | 4.5e-6 |
| per-event reference checks | 28128 |
| reference excluded by > 1e-6 | **0** |
| reference excluded by > 0 | 177 events, all <= 4.12e-7; 11 above 1e-9 |
| final root box excludes reference by > 1e-6 (OFF) | **0** |

The largest exclusion: clay0203 hull, column 67, min probe raw 4.12e-7 vs
g -2.32e-7, applied (lb 0 -> 4.12e-7), certified-optimal reference value 0. The OFF
final box excludes the reference by 4.12e-7; the ON box excludes it by 0. Below the
pre-registered 1e-6 threshold. (QPLIB_2635 also logs 528 events of excess ~1e10 on
bounds of magnitude ~1e20, the LP infinity sentinel; none changed a bound.)

Full OFF/ON solves (`gdp_full_solve_off_on.jsonl`, 300 s): clay0203 big-m and
ex_633 big-m/hull certify the same objective and bound in both arms (clay0203 big-m
nodes 133 OFF, 123 ON). Together with the 74-instance §5 panel already recorded
for #2077 (cert-clean, no certified result differs), no certified result shows OFF
wrong.

**Verdict under the rule: keep OFF.** Files: `obbt_ns_unsafe_acceptance.jsonl`
(per instance, event lists reduced to counts and top-3 by excess / by reference
violation) and `obbt_ns_unsafe_acceptance_summary.txt` (`--summarize` output).
