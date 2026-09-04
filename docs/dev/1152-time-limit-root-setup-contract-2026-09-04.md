# #1152 — what `solve(time_limit=T)` owes: the root-setup contract

**Status:** decided and implemented (option 3). Issue #1152 asked the owner to pick
between three readings of the role-1 `time_limit` contract; this doc records the
choice, why the two tests that appeared to contradict each other do not, the
measurement that located the defect, and the differential panel behind the
default-ON flip of `DISCOPT_ROOT_SETUP_BUILD_DEADLINE`.

## 1. The decision

> `time_limit` is a **hard wall with an anytime bound**. The solve returns by the
> deadline. The dual bound it reports is whatever it could *prove* inside the
> budget — weaker under a short budget, never unsound, and never absent merely
> because the operation that would have produced it was long.

This is issue #1152's **option 3**, and it is the only one of the three that
retires no contract. Options 1 ("hard wall, `bound=None` is acceptable") and 2
("soft target, document an overrun multiple") each keep one of the two tests by
abandoning the other; option 3 keeps both thresholds exactly as written:

* `test_875_root_setup_budget.py` — wall within **1.25x** of `time_limit`;
* `test_issue654_deadline_root_setup.py` — the deadline work must not cost the
  dual bound.

## 2. Why the two are not contradictory

The issue read the pair as a fork: *stop overrunning* ⟹ decline the long
bound-producing operation ⟹ *lose the bound*. That fork is real only while the
long operation is all-or-nothing. It is not, and #694's entry experiment already
measured that it is not — a McCormick relaxation build's dual bound turns finite at
8–21 % of build time and then accrues monotonically, because the objective is fully
linearized *before* the constraint-row loop. #694 (`anytime_root_build`) and #832
(`root_build_deadline`, default-ON since 2026-08-17) built that mechanism and gave
it to the root-relaxation **fallback**.

What was left was the layer above it: the root-setup phases that run *before* the
fallback. Each of them polls a deadline between its LP solves, and none of them
polls inside its relaxation **build** — and each clamps its own budget to *all* of
the live remainder, including the `_ROOT_FALLBACK_RESERVE_S` slice the fallback
needs to prove anything at all.

## 3. The measurement (in-repo, `casctanks`, `time_limit=5`)

`casctanks` is one of the four instances #1152 names as the overrun class
(2.7x at 120 s on the owner's machine) and the only one vendored in-repo. It
reproduces **both** halves of the issue at a 5 s budget on a quiet 4-core Linux box
(`scratchpad/i1152/phase_attrib3.py`, `fb_probe.py`, `log_timeline.py`):

| t (s) | phase | note |
|---|---|---|
| 0.0–2.1 | presolve + convexity classification | classification stops on its own budget |
| ~4.4 | root OBBT enters its round `build_milp_relaxation` | **0.61 s of budget left** |
| 4.4–6.3 | that build | **1.85 s, uninterruptible** |
| 6.3 | root OBBT's sweep, then the #654 short-circuit | `fallback grant 0.000s of the 1.500s reserve` |
| 6.4 | return | wall **1.29x**, `bound=None` |

So the 1.29x overrun and the missing bound are **one defect with two symptoms**,
not two contracts in tension. The build ran past the deadline *and* the phase's own
budget clamp (`min(..., _remaining_budget())`) let it spend the reserve, so the
last-ditch bound producer was handed a grant of exactly zero.

`sonet23v4` at `tl=2` — issue #1152's side B — is the same shape one size up:
root setup outruns the whole 2 s budget, `_rr_remaining <= 0`, the fallback is
skipped, `bound=None`. It is not a truncation bug; it is a reserve that nothing
reserved.

## 4. What ships

Behind `DISCOPT_ROOT_SETUP_BUILD_DEADLINE` (`SolverTuning.root_setup_build_deadline`),
`solve_model` gains one root-setup budget discipline with two halves:

* `_root_setup_build_deadline()` — the absolute deadline
  `_solve_t0 + time_limit - _rr_reserve_s`, threaded as the `build_deadline` of
  every pre-B&B relaxation build: root OBBT's per-round envelope build (new
  `obbt_tighten_root(build_deadline=...)`, applied to the round build and the DBBT
  rebuild), the root LP probe's `solve_at_node`, and both root cut-pool
  `solve_at_node` calls. `None` under `deterministic` (a clock that decides how many
  rows get built is role 2, #912) and when the limit is not finite.
* `_setup_remaining_budget()` — `_remaining_budget()` minus the same reserve, used
  by those phases' own budget clamps, so a phase entered near the deadline can no
  longer spend the fallback's slice.

The native-kernel spec build is **not** truncated: it is two whole relaxation
builds whose row sets must correspond (a probe box for the structure, the real box
for the column bounds), and a prefix of one need not describe the same relaxation as
a prefix of the other. It is instead declined once the root-setup deadline is spent
— falling through to the trusted Python path, the same sound outcome an
out-of-scope model already gets.

### Soundness

Every truncation reached by this change only ever **weakens**:

* **OBBT** tightens `x_i` to the optimum of `min x_i` over the relaxation polytope.
  Dropping constraint rows *enlarges* that polytope, so the LP optimum moves
  outward and the tightening is weaker — never invalid. The #208 aux cascade is the
  one place where a truncated build is not merely weaker: its carried bounds are
  keyed by column index and a truncated build has a different lifted layout, so
  applying one could tighten the *wrong* column. Both the apply and the capture are
  skipped on `milp._build_truncated`.
* **The root LP probe** decides whether to keep the relaxer and banks a bound that
  `_root_relaxation_lower_bound` re-gates on exact box equality; fewer rows can only
  make that bound weaker, and #928's cut-short objective floor keeps it finite.
* **The root cut pool** only adds cuts; separating fewer of them loosens per-node
  bounds and never invalidates one.

§8.1 ("do not truncate bound-producing native solves") and §8.2 ("do not re-add the
Rust LP native deadline") of `baron-gap-plan.md` are un-circumvented: this bounds
Python relaxation *builds*, never an LP solve.

## 5. Effect on `casctanks` (the same solve, flag OFF vs ON)

| arm | wall | ratio | bound |
|---|---|---|---|
| OFF | 6.43 s | 1.29x | **None** |
| ON | 5.35 s | 1.07x | **1.2584408140882861** |

Both halves of #1152, on one instance, with both thresholds untouched. The bound is
sound (a valid lower bound below the 5.698 `casctanks` value #654's do-not-regress
list records).

## 6. §5 differential panel

See §7 below for the numbers. Scope note, stated rather than glossed: this panel is
the **66-instance in-repo corpus** at `time_limit` 5 s and 20 s, flag OFF vs ON,
interleaved per instance. The full-MINLPLib held-out arm that
`graduation_gate.py --flags` runs, and the two big-corpus instances #1152 names
(`sonet23v4`, `watercontamination0202`), are **not** reachable from this
environment: the snapshot under `~/Dropbox/projects/discopt-minlp-benchmark/` is not
present and minlplib.org is not reachable through the egress proxy. The in-repo
corpus does contain `casctanks`, one of the four overrun instances the issue names,
and it reproduces both symptoms — so the class is exercised, at a smaller size.

**That gap is now closed** (2026-09-04, on the owner's machine, commit `fe4d69c`
during the review of PR #1155 — so the paragraph above stands as the state of the
panel, not as the state of the evidence). The two instances the issue names were run
there, and they are the two that close #1152's own tests:
`test_issue654_deadline_root_setup.py::test_sonet23v4_bound_survives_the_deadline_gating`
(`tl=2`) and `test_875_root_setup_budget.py::test_watercontamination0202_honours_its_time_limit`
(`tl=30`, `tl=60`). Back-to-back on one machine at load 5.4 — elevated, which only
strengthens the result, since load pushes the overrun test toward failing:

| arm | result | wall |
|---|---|---|
| flag ON (the default) | **3 XPASS** | 101.6 s |
| `DISCOPT_ROOT_SETUP_BUILD_DEADLINE=0` | 3 XFAIL | 158.0 s |
| xfail markers removed | **3 passed** | 100.6 s |

The OFF arm is the load-bearing one: it reproduces the defect exactly as documented,
which attributes the pass to the flag rather than to the machine, and it is an
end-to-end check that the graduated flag's §5 opt-out is genuinely live. The strict
xfails #1150 had added for #1152 came off in the same commit — every threshold and
soundness assertion is still the one #875 and #654 set.

**To re-run it.** The flag is default-ON, so the control arm has to turn it OFF —
`DISCOPT_ROOT_SETUP_BUILD_DEADLINE=0` is the baseline and the unset default is the
treatment, the reverse of `generality_sweep`'s convention for a parked flag.
(`generality_sweep`'s `root_build_deadline` arm has the same problem since #832
graduated it: it sets `=1` against a control that no longer turns it off, so that arm
is a no-op today. Fixing that registry is out of scope here; it is noted so the next
person does not read its verdict as an A/B.)

## 7. Panel result

66 instances x `time_limit` {5 s, 20 s}, OFF and ON interleaved per instance:
**132 pairs, 559 executed comparisons** (`scratchpad/i1152/panel_T5_T20.jsonl`,
verdict from `panel_report.py`).

### Cert-clean

| check | result |
|---|---|
| dual bound above its reference optimum (either arm) | **0** |
| dual bound crossing its own run's incumbent | **0** |
| `gap_certified` lost, OFF -> ON | **0** |
| certified objective changed, OFF -> ON | **0** |

The reference optimum per instance is the tightest justified upper bound: the
`known_optima.toml` value where the instance is listed, otherwise the best feasible
objective any run of either arm returned. Three of the instances whose bound moved
(`4stufen`, `beuster`, `bchoco08`) have neither, in this environment — no feasible
point exists for them at any budget tried, including a 120 s reference run
(`scratchpad/i1152/oracle_probe.py`), and `minlplib.solu` is not reachable here. For
those three the soundness argument is the structural one of §4 plus the consistency
check that each moved bound stays below the bound the 120 s run proves (`4stufen`
19055 < 20712, `beuster` 6352 < 6395, `bchoco08` 1.0 - 2e-12 < 1.0 + 5e-13). The two
that do have an oracle are checked against it directly: `nvs09` -48.99 <= the
-43.134 `=opt=`, `tanksize` 1.25304 <= the 1.26864 the solver certifies itself.

### Net-positive

| effect | count | instances |
|---|---|---|
| bound RECOVERED (`None` -> finite) | 2 | `casctanks` -> 1.2584, `bchoco08` -> 1.0 |
| bound TIGHTER | 3 | `4stufen` 18770 -> 19055, `beuster` 5942 -> 6352, `nvs09` -50.59 -> -48.99 |
| bound LOOSER | 1 | `tanksize` @20 s, 1.253535 -> 1.253040 (4e-4 relative) |
| bound LOST | **0** | — |

Punctuality, `wall / time_limit` over the same 132 pairs:

| arm | mean | max | pairs > 1.25x |
|---|---|---|---|
| OFF | 0.447 | 1.448 | 2 |
| ON | 0.440 | 1.434 | 2 |

Per-pair delta: mean **-0.006x**, and at the 5 s budget where the reserve actually
binds, **-0.012x** with 5 pairs better by >0.05x against 2 worse. The two large moves
are exactly the class: `casctanks` 1.23x -> **1.02x**, `beuster` 1.13x -> **0.98x**.

### The counter-example, measured rather than averaged away

`hda` at a budget the panel did not cover — 8 s, where the reserve is 2.4 s — goes
the *other* way, and consistently. Interleaved, 3 reps per arm, load 0.58:

| arm | wall (mean, sd) | ratio | nodes | bound |
|---|---|---|---|---|
| OFF | 9.68 s, sd 0.25 | 1.21x | 1 | -119284.66095887942 |
| ON | 10.26 s, sd 0.10 | 1.28x | 3 | -119284.66095887942 |

The bound is bit-identical; what changed is that root setup now *finishes 2.4 s
earlier*, so the search starts two more nodes and the last of them straddles the
deadline. That is the pre-existing per-node overrun (#966's `node_round_budget`
territory — a node round's non-LP cost is spent after its admission check,
unclamped), not a setup overrun; this change hands it more opportunities to fire on
this instance. `casctanks` at the same 8 s budget moves the other way in the same
sweep (8.88 s -> 8.34 s, means of the same 3 interleaved reps). It is recorded here
rather than folded into a mean because a reader deciding whether to keep the flag on
should see the shape of the trade, not only its sign.

## 8. What #1152 does not fix

The residual overrun that is **not** a root-setup relaxation build stays, and the
panel says so plainly: `tls2` (1.45x -> 1.43x), `tspn12` (1.39x -> 1.38x) and
`bchoco07` (1.21x, unchanged) at a 5 s budget are where the two arms agree, because
the phases this change bounds are not where their time goes. Those are the two pairs
still over 1.25x in both arms.

The contract above is what the solver now owes and what it delivers on the measured
class; an instance whose *mandatory* root work (Rust presolve,
convexity classification, `from_nl`) alone exceeds `time_limit` will still return
late, because none of that work is optional and none of it is a relaxation build.
That is the honest residual, and it is bounded by the reserve rather than unbounded
by it: the fallback now runs whenever any of the budget survives setup.
