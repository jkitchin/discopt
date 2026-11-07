# #2202: the nous1 stall is an undecided-node-LP cascade (performance-plan §121)

Every script asserts it loaded this tree's `discopt` with the #2202 kwarg (CLAUDE.md §8)
and prints an executed-assertion count (§6).

| file | what |
| --- | --- |
| `perm_probe.py` | nous1 through `Model.solve` with the kernel's fixed rows permuted by `PERM_SEED`; `MAX_NODES` makes it deterministic under a node budget. Prints status, bound, nodes, wall, undecided nodes and dual-proof counters |
| `sweep_retry_absent_dp_on.log` | seeds 4–12, `DISCOPT_DUAL_PROOF_PROPAGATION=1`, 120 s, on `main` before the retry existed (three at a time on 4 cores): seed 5 times out with 1,336 undecided nodes; wall tracks the undecided count |
| `sweep_retry_on.log` | seeds 1–12 × dual proof {1, 0}, 120 s, with the retry (two at a time): 24 / 24 certified |
| `fire_screen.py`, `panel/fire_screen.jsonl` | ON-only screen (2000 nodes, 60 s, deterministic) over nous1 and the 58 `python/tests/data/minlplib` instances not in the corpus: the retry acts on nous1, carton7, ex1252, ex1252a, nvs20, st_e31 |
| `panel/corpus.*` | `issue2115_graduation_panel.py --flag DISCOPT_NODE_LP_SLACK_RETRY --deterministic --max-nodes 2000 --time-limit 120 --assert-unfired-identical --jobs 2` over the 74-instance corpus (oracle `cert-optima.json`) |
| `panel/fired.*`, `panel/fired_nous1.*` | the same panel over the screen's acting instances, oracle MINLPLib `minlplib.solu` |
| `panel/default_fired.*` | the default (non-deterministic) path at 120 s over the screen's acting instances, `--jobs 1` |
| `panel/nous1_timing.log` | nous1 to certification, default path, interleaved OFF/ON, load-gated |

The captured node LPs themselves are the Rust fixtures
`crates/discopt-core/src/bnb/testdata/nous1_inherited_basis_numerical{,_infeasible}_lp.json`
(seed 5, flag OFF, from a `node_observer` capture; HiGHS: optimal 1.2060754564 /
infeasible).

The sweeps before the final wiring ran with the retry gated by a Rust env read
(`DISCOPT_NODE_LP_SLACK_RETRY`, same code path); the panels ran on the final build, where
`solver.py` reads the same variable and passes `node_lp_slack_retry` to the kernel.
