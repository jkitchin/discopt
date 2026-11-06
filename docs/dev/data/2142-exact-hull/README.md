# #2142 Direction 2: an exact (division-free) hull for nonconvex disjunct rows — entry experiment

Deliverable: `proposal.md`. Pre-registration: `PREREG.md` (written before any run past 1 node).
Every script asserts the loaded `discopt` is this tree's (CLAUDE.md §8), prints an
executed-assertion count (§6) and catches nothing (§7). Build: `c58f3258` extension
(`_rust.abi3.so` without the retired §114 marker, asserted by every script).

| file | what |
| --- | --- |
| `p0_structure.py` / `.log` | cstr's 10 hull groups, the 5 nonlinear disjunct rows as expression trees (`rate*Q*Q + 2*FA*FB == 0`), the shipped hull's lowered rows (`v/yhat` inside every product), model sizes |
| `exact_hull_proto.py` | the prototype lowerings, installed by monkeypatching the hull's #1917 origin-free hook: `ratio`, `homog`, `lifted`, `both` (see `proposal.md` §2) |
| `arm.py` | one deterministic cstr solve per arm; records the kernel calls, the spec's largest finite \|global bound\|, the node-status histogram and the LP-verdict counters |
| `arm_<arm>_<nodes>.log` | the runs (`run_10k.sh`, `run_b.sh`: up to 4 concurrent on 4 cores, so walls are not comparable between arms) |
| `check.py` / `check.log` | exactness: x* on every lowered row; six fixed selector configurations against big-M; projection of each candidate solution onto the original rows; convex combinations of lowered points against the linear lowered rows |
| `arm.py flag ...`, `flag_*.summary.log`, `proto_homog_2000.summary.log` | the shipped `DISCOPT_HULL_EXACT_PERSPECTIVE` (arm `flag`, no monkeypatch) against the prototype: 2,000 nodes (also `PYTHONHASHSEED=1`), 10,000 nodes, and the default path at 300 s; profile dumps stripped, every other line kept |
| `panel/corpus.*`, `panel/corpus_summary.txt` | §5 corpus panel (74 in-repo instances, `issue2115_graduation_panel.py`); the summary is re-derived with `--rows` (the live log lost its tail to an unlinked inode) |
| `panel/gdp_hull.jsonl`, `panel/gdp_{fire,solve}.log`, `panel/gdp_report.txt` | §5 GDP parity panel under hull (`issue1917_gdp_flag_panel.py`, fire phase, then the 6 fired instances solved deterministic, 5,000 nodes / 120 s) |

> **Raw logs removed (#2142 follow-up).** The `.log` / `.log.gz` run traces cited in this directory were
> removed to keep checkouts small; the results they support are recorded in these notes. The originals
> remain in git history: `git show e39975e1:docs/dev/data/2142-exact-hull/<name>.log.gz | gunzip` (or `.log`, no gunzip).
