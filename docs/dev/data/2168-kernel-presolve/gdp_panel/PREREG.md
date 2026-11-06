# #2168 kernel presolve: GDP parity panel + full-solve cstr hull (pre-registration)

Written 2026-11-05 before any run below (CLAUDE.md §4/§5). Branch `claude/gracious-hawking-iywpk2`
at `ad416e98`, the extension built from that tree (rows record `git_head`). Host: macOS,
14 cores, gdplib from source (`gdplib.cstr` importable), ipopt on PATH.

## Runs

1. **GDP parity panel.** `issue1917_gdp_flag_panel.py --flag DISCOPT_KERNEL_PRESOLVE --phase solve
   --all --deterministic --max-nodes 5000 --time-limit 120 --reps 1` (the #2142 kernel-flag recipe),
   24 GDPlib instances x {big-M, hull}, OFF (`=0`) vs ON (`=1`) interleaved, split over processes by
   `--include`. `KernelPresolveApplied` marks where the main kernel call saw a reduced spec.
2. **Full-solve cstr hull** (`cstr_arm.py`): `Model.solve(gdp_method="hull")`, deterministic,
   `PYTHONHASHSEED=0`, `max_nodes` 10k and 20k, OFF vs ON; plus OFF vs ON non-deterministic at
   `time_limit=1800` with no node cap (does it certify, and when).

## Bars

* **Cert-clean** (any failure retires or blocks): `--report` exits 0 -- 0 `incorrect`, no dual bound
  past a proven oracle, no instance solved OFF and not ON, every unfired pair bit-identical (status,
  objective, bound, nodes) unless at the wall. cstr: every bound <= 3.0620073 + 1e-6, every
  incumbent >= 3.0620073 - 1e-5.
* **Net-positive** (both needed):
  - cstr hull full solve: ON 20k bound > OFF 20k bound, and ON 20k bound > 2.7875 (the entry's
    kill line, now on the full solve) or ON certifies.
  - GDP panel, over cells where it fired: (solved gained >= 1 or bound-better cells exceed
    bound-worse cells by >= 2, bound compared at the 5000-node budget, relative 1e-6), and no
    solved cell lost. If it fires on fewer than 3 cells, "broadly" is not shown.

## Outcomes (§5, three-outcome rule)

* Both bars -> **graduate** (default ON, keep `=0`).
* Not cert-clean -> **retire**, or fix the bug and re-run.
* Cert-clean but not net-positive -> **retire** (delete the flag and `spatial_presolve.py`, keep
  the tests' fixtures and the data), unless the owner keeps it as a documented opt-out.
