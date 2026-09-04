# #1151 — reported objective below the global minimum on quotient expressions

Probes and measurement records for the fix in
`python/discopt/validation/feasibility.py` (`_row_scales`: the row-scale term
becomes `max_j |J_ij| * |x_j|`, dropping the `max(1, |x_j|)` floor).

Every script prints an executed-assertion / comparison count and exits non-zero
when it is zero (CLAUDE.md §6); none swallows an exception (§7); the arms are
interleaved in one process and the timing run reports a spread (§9); the panels
assert which sources they loaded (§8).

| file | what it is |
|---|---|
| `probe_family.py` | The issue's own table, re-derived: reported objective vs a plain-Python oracle at the solver's own returned point, per expression family x box floor. |
| `panel.py`, `run_panel.sh`, `panel.json`, `panelA.txt` | **Panel A** — 119 vendored `.nl` instances, arms interleaved per instance, 20 s. Records `_row_scales` invocations: **0 in both arms**, so the changed path never ran and the arms executed identical code. |
| `compare.py` | Panel A analysis: result differences, the #1151 oracle per arm, and every dual bound against `known_optima.toml`. |
| `null_control.py`, `null.txt` | **Null control** for Panel A's 10 result differences: the same instances re-run with NO code difference between arms. 8 of 10 still disagree, which is what `divergent_rows = 0` implies. |
| `panel_api.py`, `panel_api.json`, `panel_api.txt` | **Panel B** — the class where the path *does* fire (quotient objectives, 10 models x 2 arms). Positive control plus the ON/OFF differential. |
| `cert_cost.py`, `cert.txt` | What an honest certificate costs on the two models where the defect fired, 3 interleaved reps with a standard deviation. |

## Headline numbers

Panel B, every case where `_row_scales` fired: **8 super-optimal / mismatched
reports, all in the OFF arm, 0 in the ON arm.**

```
case                    arm      status              reported              oracle@x        delta    opt  rows  div
balance_floor0.001       on    feasible                     2                     2    0.000e+00      2     1    2
balance_floor0.001      off     optimal      1.99868397947021      2.00000224782965   -1.318e-03      2     1    0
balance_floor0.01        on    feasible                     2                     2    0.000e+00      2     1    2
balance_floor0.01       off     optimal      1.99985334212268       2.0000384197609   -1.851e-04      2     1    0
ratio3_floor0.001        on    feasible                     3                     3    0.000e+00      3     1    3
ratio3_floor0.001       off     optimal      2.99800969965084      3.00010933723672   -2.100e-03      3     1    0
ratio3_floor0.01         on    feasible                     3                     3    0.000e+00      3     1    3
ratio3_floor0.01        off     optimal      2.99976909822596      3.00002939785867   -2.603e-04      3     1    0
```

`ratio3` (`x/y + y/z + z/x`, optimum 3) is not in the issue: it was found by
this panel, and it confirms the defect is a class rather than the one model.

Certification cost, 3 interleaved reps at a 300 s budget (load 0.73 at start):

```
floor 1e-03 off: optimal 1.99868397947   wall mean   0.60s sd 0.02   <- false
floor 1e-03  on: optimal 2.0             wall mean  37.95s sd 0.27
floor 1e-02 off: optimal 1.999853342123  wall mean   0.75s sd 0.02   <- false
floor 1e-02  on: optimal 2.0             wall mean 228.16s sd 0.34
```

Both arms certify; OFF's speed is manufactured by its own too-low incumbent (the
tree stops at `bound >= incumbent - gap_tol`). The trade is a slow true
certificate for a fast false one, which is the only direction CLAUDE.md §1
allows.
