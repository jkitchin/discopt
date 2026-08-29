# OA master MILPs captured for #1066

Two outer-approximation master MILPs, marshalled exactly as
`discopt.solvers.milp_simplex.solve_milp` received them during a default
`Model.solve()` on the named MINLPLib instance, written by
`scratchpad/issue1066/capture_master.py`.

They are here because they are the two *classes* the #1066 root-cut escalation
has to tell apart, and neither is reproducible from a synthetic generator — the
distinguishing structure is the OA cuts the loop had already accumulated:

| file | rows x cols | legacy budget (16/1) | raised budget (200/10/select) |
|---|---|---|---|
| `tls2_master0.npz` | 52 x 37 | **optimal in 241 nodes, 0.0 s** | feasible only, still open at 60 s |
| `rsyn0830m_master0.npz` | 915 x 250 | optimal in 529 573 nodes, 49.2 s | **optimal in 1 197 nodes, 0.3 s** |

Neither budget dominates: that measurement is what killed the "just raise the
budget" fix and produced the probe-then-escalate policy instead. See
`docs/dev/performance-plan.md` §23.
