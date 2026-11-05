# #2168 kernel presolve: GDP parity panel + full-solve cstr hull (2026-11-05)

Pre-registration: `PREREG.md` (written before any run). Branch head `ad416e98`, extension built
from that tree; all 96 panel rows record `git_head = ad416e98` (CLAUDE.md §8). Host: macOS, 14
cores, gdplib from source, ipopt on PATH. Load 3 at launch, 6-14 during (9 processes); the panel
compares node-budgeted bounds, which do not depend on load, and walls are reported only as context.

| file | what |
|---|---|
| `gdp_{1,2,3}.jsonl`, `.log` | `issue1917_gdp_flag_panel.py --flag DISCOPT_KERNEL_PRESOLVE --phase solve --all --deterministic --max-nodes 5000 --time-limit 120 --reps 1`, split by `--include` |
| `gdp_merged.jsonl`, `gdp_report.txt` | the three shards concatenated; `--report` over them (exit 0) |
| `fired_cells.txt` | OFF/ON rows for the 4 cells where `KernelPresolveApplied > 0` |
| `cstr_arm.py`, `cstr_*.log` | full `Model.solve(gdp_method="hull")` on GDPlib cstr, OFF/ON |

## Full-solve cstr hull

| arm | 10k det | 20k det | uncapped, 1800 s (default 100k node cap) |
|---|---|---|---|
| OFF | 0.6624 | 0.9314 | node limit at 100,000, bound 2.6136, 452 s |
| **ON** | **1.6168** | **2.2395** | **certified optimal 3.0620146 at 41,949 nodes, 374 s** |

The presolve fixed 51 columns and eliminated 112, the same as at the kernel-level entry. The
full-solve 20k bound (2.2395) is lower than the kernel-harness 20k bound (2.7936, `../README.md`):
the harness's seeded arm does not track the full solve as closely as its control suggested. It
passes the pre-registered cstr bar on its "or ON certifies" branch, not on 2.7875.

## GDP parity panel

* **Cert-clean:** 96 solves; 0 incorrect, 0 bound crossings, 0 solved lost; 27 unfired pairs
  bit-identical (17 more at the wall).
* **Fired on 4 cells** (cstr and syngas, each under big-M and hull):

| cell | OFF bound @5000 | ON bound @5000 | |
|---|---|---|---|
| cstr big-M | 1.42805 | 1.47116 | better |
| cstr hull | 0.47825 | 0.21334 | **worse** (ON overtakes by 10k: 1.6168 vs 0.6624) |
| syngas big-M | certified, 3 nodes | certified, 3 nodes | same |
| syngas hull | certified, 3 nodes | certified, 3 nodes | same |

  Solved gained 0. Bound-better minus bound-worse = 1 - 1 = 0, against a pre-registered bar of >= 2.

## Verdict against the pre-registration

* Cert-clean: **pass**.
* Net-positive: the cstr bar **passes** (ON certifies a model OFF does not within 100k nodes); the
  GDP-panel bar **fails** (it fires on 4 cells, and its one bound gain is cancelled by cstr hull's
  5,000-node loss). The pre-registration required both, so by its rule the flag is **not
  graduated**. The rule's default is retire. The owner decides between retiring it and keeping
  it as a documented opt-out, since its gain is confined to one instance (CLAUDE.md §2) but that
  instance is the one #2168 targets.
