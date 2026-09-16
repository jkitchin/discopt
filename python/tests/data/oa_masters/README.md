# OA master MILPs captured from real solves

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


## `fac2_master0.npz` (#1236)

A third master, captured the same way (from a default `Model.solve()` on
`fac2.nl`), for a different reason: it is the instance on which the in-house
MILP driver returned a **certified false `optimal`**.

| file | rows x cols | integers | note |
|---|---|---|---|
| `fac2_master0.npz` | 418 x 67 | 12 | 27 nodes to a wrong `optimal`, pre-#1236 |

It carries two extra arrays the #1066 pair does not:

- `z_star` — a point verified feasible for this master (rows, bounds,
  integrality), and
- `z_star_objective` — its objective, `331837498.17693394`.

Before #1236 the driver reported `optimal` at `331845337.439685`, i.e. **7839
above a point feasible for the very problem it was handed**; HiGHS and POUNCE
both found `z_star`'s value. The witness is what makes the defect assertable
without an external oracle, so `test_1236_gmi_drops_no_term.py` re-verifies
`z_star`'s feasibility before using it (CLAUDE.md §6) rather than trusting the
fixture.

Tolerances against this fixture must be **absolute**. `|z_star|max` is 3.3e8, so
a scale-relative gate at 1e-9 is a slack of 0.86 — larger than the 0.017 cut
violation that caused the defect, which is exactly how it was missed once.
