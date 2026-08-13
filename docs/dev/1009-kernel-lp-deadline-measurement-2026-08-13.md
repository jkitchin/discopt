# What the #1009 kernel node-LP deadline is worth — measurement + coverage (2026-08-13)

**Status.** Measurement-and-coverage only; no behavior change. #1014 already threaded
the tree's live deadline into the node LP (`node_lp_opts`) and #1015 fixed the
incremental McCormick fast path, closing #1009. This records what the kernel-side
change actually bounds, and closes a coverage gap it left.

## 0. Correcting the attribution first

#1009 was opened against `spatial_bindings.rs` with QPLIB_1157 (240.83 s against a
20 s `time_limit`, 11.9×) as evidence. **That attribution was wrong**, and #1014
retracted it: threading the deadline did not shorten that repro (419.74 s, 21.0×,
nodes=1), and `faulthandler` sampling put the wall clock in
`_root_relaxation_lower_bound` → `MccormickLPRelaxer.solve_at_node` →
`solve_lp_warm_std` — the Python-side root relaxation computed *before* the kernel
tree is entered. #1015 fixed that. Any reading of the numbers below as "this is what
caused the QPLIB_1157 overrun" is the same error; they are about a different,
genuinely present defect that #1009 correctly identified by inspection.

## 1. The quantity

`solve_spatial_tree` checks its clock only BETWEEN nodes. Without a per-LP deadline
the wall-clock floor of any kernel solve is therefore the cost of **one node LP**,
however small the caller's budget — unbounded in the size of the relaxation, and not
a property of any corpus instance.

#1014 shipped on a soundness-by-construction argument with no measurement
("Sound by construction, not by measurement"), so this is the missing half.

## 2. Method

Both arms run against the **same fixed code**, so this reproduces on `main` with no
flag and no revert (`scripts/spatial_kernel_lp_deadline_scaling.py`):

* **uncapped** — `time_limit_s=None, max_nodes=1`: one node LP run to completion.
  This is exactly what the pre-#1014 path spent on node 1 regardless of the budget,
  i.e. the floor the fix removes.
* **capped** — `time_limit_s=BUDGET`: the same LP under the deadline.

> x86-64 Linux, 4 cores, Python 3.11, **release** build (`pip install -e .` → maturin
> release). Load gate: `uptime` load average 0.24–0.63. Arm order alternates per rep
> (CLAUDE.md §9); medians of 3 with sd. Driver is a general dense-bilinear model
> (`min Σ_{i<j} x_i x_j s.t. Σx ≥ n/2, x ∈ [0,1]^n`), not a named instance
> (CLAUDE.md §2), whose global optimum `C(n/2, 2)` is closed-form and is asserted as a
> soundness oracle on every solve.

## 3. Result — budget fixed at 1 s

| n | lifted terms | cols | uncapped (one node LP) | capped | bound ≤ optimum |
|---:|---:|---:|---:|---:|:--:|
| 50 | 1 225 | 1 275 | 0.81 s (sd 0.03) — 0.8× | 1.02 s (sd 0.02) — 1.0× | ok |
| 70 | 2 415 | 2 485 | 2.68 s (sd 0.04) — 2.7× | 1.20 s (sd 0.02) — 1.2× | ok |
| 90 | 4 005 | 4 095 | 7.69 s (sd 0.09) — 7.7× | 1.23 s (sd 0.01) — 1.2× | ok |
| 120 | 7 140 | 7 260 | 25.90 s (sd 0.39) — **25.9×** | 1.05 s (sd 0.01) — 1.0× | ok |

The uncapped column is the floor, and it grows without limit; the capped column
tracks the budget. At n = 50 the capped run is *longer* than the uncapped one — the LP
is cheaper than the budget there, so the tree keeps branching until the clock runs
out. That is the expected shape: the deadline bounds the overshoot, it does not
shorten solves that were already inside their budget.

## 4. Corpus check — the change is inert where the budget does not bind

A 20-instance differential panel (in-repo `minlplib_nl` + `qplib`, the instances the
producer accepts) at 0.05 / 0.25 / 1.0 / 3.0 s kernel budgets, run at the binding with
both arms interleaved in one process, was **cert-clean at every budget**: no dual
bound past its reference optimum (sense-aware, from `qplib.solu` and
`known_optima.toml`), no incumbent invented, and no instance losing a finite bound at
a budget ≥ 1 s. Bounds and node counts are unchanged on every instance that finishes
inside its budget; worst corpus overshoot went from +0.20 s to +0.01 s.

**The one cost, stated plainly.** At a budget *smaller than a single LP*, a bound is
traded for compliance: `QPLIB_3815` at ≤ 0.25 s goes from `-96` — obtained by
overrunning — to no bound at all. That is not avoidable; you cannot both stop on time
and finish an LP that costs more than the whole budget. It is also the case #933
exists for (the caller withholds a root-relaxation reserve precisely for a bound-less
time-limited kernel exit), and measured end-to-end through `Model.solve` on that
instance at 0.5 / 1 / 2 s both arms are identical (`bound=None`, inside budget,
sd ≤ 0.03 s), so it does not propagate to the reported result there.

Soundness is structural, not statistical: a cut LP bails as `IterLimit`,
`solve_spatial_node` certifies a Neumaier–Shcherbina safe bound only on `Optimal`, and
`verdict_for` routes `IterLimit` to `Undecided` — branched with the parent's valid
bound, never fathomed. It can only loosen a bound, never lift one.

## 5. The coverage gap this closes

`spatial_tree.rs` unit-tests `node_lp_opts` in isolation and pins the
`IterLimit → Undecided` verdict, but nothing tested that the tree's deadline **reaches
the LP at all**. Measured, not assumed: mutating the call site to
`node_lp_opts(opts, None)` — a full revert of #1014's effect, leaving the
unit-tested helper correct — leaves `cargo test -p discopt-core --lib` at
**602 passed, 0 failed**.

(The neighbouring `extension_is_taken_only_with_an_incumbent_in_hand` does catch a
*too-early* deadline: mutating the call site to freeze `config.deadline` instead of
the live value fails it. It cannot catch a *missing* one, because an LP with no
deadline solves fine and reports a finite bound.)

`python/tests/test_1009_kernel_lp_deadline.py` closes that: two of its three tests
fail on the `None` mutant and pass on `main`. They exercise a real LP cut mid-flight
across the PyO3 boundary, calibrated against the uncapped single-node cost on the
same build, and skip rather than pass vacuously if one LP is already cheaper than the
budget on the host.
