# Default-OFF flag audit (CLAUDE.md §5 retirement rule)

Standing record of every `DISCOPT_*` gate over **solver math** that defaults OFF, and
which of §5's three states it is in: **graduated**, **retired**, or **kept as a
documented opt-out**. A gate in none of them is a defect.

First taken 2026-09-19 against `0ae8cad` (issue #1345). A new default-OFF gate over
solver math adds its row here in the same PR that introduces the flag.

**Re-derived 2026-09-20 against `924169b`** (v0.9.0 release audit). The methodology below
was re-run rather than trusted; the numbers hold:

```
call sites scanned                 : 97
distinct DISCOPT_* flags read      : 89   (was 86)
  default is a literal "0"         : 16   (was 19)
    numeric knobs ("0" is a value) :  2   out of scope
    gates over non-solver behaviour:  3   out of scope
    gates over solver math         : 11   <- this audit
```

The 86 -> 89 growth is **not** new default-OFF gates — the literal-`"0"` set is fully
accounted for, so no new row is owed. The 19 -> 16 drop is #1346 (graduated), #1357 and
#1358 (retired). Of the 11 live gates, two are resolved below and **nine are in none of
§5's three states**, tracked in #1388.

**Why no panel was ever run, which the first pass did not identify.** The infrastructure is
generic and healthy: `discopt_benchmarks/scripts/graduation_gate.py` drives arms from
`generality_sweep.GRADUATION_ARMS`, and the nightly `graduation-gate.yml` passes no
`--flags` so the gated set tracks the registry instead of a drifting copy. **None of the
nine flags is in that registry** — there is no arm to run. (Note `psd_cost_gate` in
`GRADUATION_ARMS` is `DISCOPT_PSD_COST_GATE`, a *different* flag from `DISCOPT_PSD_QFORM`.)
The first step for each of the nine is one `ARMS` entry, then
`graduation_gate.py --flags <arm>`.

## How the population was determined

`environ.get("DISCOPT_…")` call sites across `python/discopt` (350 files), classified by
**how the flag is consumed** and **what it gates** — not by its default being `"0"`:

```
distinct DISCOPT_* flags read      : 86
  with a literal default           : 69
  default is a literal "0"         : 19
    numeric knobs ("0" is a value) :  2   out of scope
    gates over non-solver behaviour:  3   out of scope
    gates over solver math         : 14   ← this audit (12 live after #1357, #1358)
```

**Retired since the first pass.** `DISCOPT_LP_SPATIAL_MIXED` (#1357) and
`DISCOPT_POUNCE_DECLARED_BOX` (#1358). Their rows are kept below with the outcome
recorded, because an audit that deletes its own history cannot show the rule working.

**Out of scope, and why.** `DISCOPT_HEUR_OFFSET` and `DISCOPT_ROOT_CUT_ROUNDS` are
numeric knobs — `float(os.environ.get(...))` / `int(...)`, where `0` is a value (zero
offset, zero rounds), not an off-switch. `DISCOPT_EAGER_IMPORTS`,
`DISCOPT_DISABLE_JAX_CACHE` and `DISCOPT_GAMS_NO_DAEMON` are booleans, but they gate an
import strategy, a compilation cache and a subprocess daemon — no §5 panel can apply to
them. Opt-*outs* for shipped defaults (`DISCOPT_NATIVE_SPATIAL_KERNEL=0`,
`DISCOPT_LP_MILP_BACKEND=rust`, `DISCOPT_NS_MARGIN=0`) never appear here: they exist so a
default can be A/B'd, which is the rule working.

## The 14 gates

Verdicts are grounded in what each flag's own docstring records. Where a docstring
promises a panel that was never run, that is stated as such — **"panel owed, never run"
is itself the finding**, and under the new clause it is a defect rather than a
steady state.

| flag | what its own text records | state | verdict |
|---|---|---|---|
| ~~`DISCOPT_LP_SPATIAL_MIXED`~~ | *"it ran its graduation panel and did NOT graduate, on **both**"*; *"Sound but harmful stays OFF, with the measurement recorded"* | panel ran, failed | **RETIRED (#1357).** Flag and both production call sites removed; the `mixed=` capability and its tests kept, with the killing measurement moved onto `_is_in_scope` so it survives the flag. |
| ~~`DISCOPT_POUNCE_DECLARED_BOX`~~ | *"Default-OFF pending the §5 graduation gate"*; #1327's panel: gate 1 cert-clean **PASS**, gate 2 net-positive **INCONCLUSIVE** (only 2 of 66 instances are in the affected window) | panel ran, inconclusive by construction | **RETIRED (#1358).** Flag removed; the **scoped** `declared_box_honored()` override kept — it is what #1327's retry uses, and moving the threshold for one call is a different mechanism from moving it for the process. Panel result preserved on `finite_bound_threshold`. |
| ~~`DISCOPT_CONVEX_KERNEL`~~ | default-OFF, never default-ON; 3,105 lines of Rust behind it | panel run, **passed both bars** | **GRADUATED (#1346).** Default-ON, `=0` opt-out kept per §5. The park was never a failed panel: #798 proved both bars and #800 deferred graduation to #807's *SCIP wall parity*, a bar above what §5 asks. Graduation also had to fix a latent routing defect — the gate claimed every pure LP/MILP — which the corpus panel could not see. |
| `DISCOPT_NLP_NATIVE` | *"Default stays OFF on the remaining grounds — the speedup …"*, plus (2026-09-20) the bar that would flip it | measured, reasoned | **RESOLVED — keep as a documented opt-out.** The "confirm it says what would change that" action is done: the comment now names the blocking bar explicitly — a Regime-2 panel with **zero MIQP-batch certification regressions** (the perturbation is the risk, not the speed) and a more-than-modest wall win. |
| `DISCOPT_CMIR_AGGREGATION` | *"ships dark behind this flag until proven on nightlies"* | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_COEF_TIGHTEN` | *"default-OFF until a corpus-wide differential panel graduates it"* (#282) | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_SGO` | *"a default-OFF env flag until a differential panel graduates it"* | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_GP_MINLP` | *"a corpus-wide differential panel graduates it"*; `solver="gp-minlp"` is an always-available explicit opt-in | panel owed, but has an explicit alternative entry point | **Keep as documented opt-out** if the explicit `solver=` route is the intended interface — then the env flag is redundant and should go; otherwise run the panel. |
| `DISCOPT_OA_INFEASIBLE_NOGOOD` | *"changes the master's dual bound (CLAUDE.md §5 regime 2) and ships behind a flag until a corpus panel clears"* | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_PRESOLVE_SUBSTITUTE` | *"bound-changing work ships behind a flag until a differential panel passes"* | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_PSD_QFORM` | *"can prove more constraints/objectives convex, which changes node relaxations and counts — hence it ships behind a flag"* | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_G_CONVEX_CUTS` | gate function is one line; rationale is thin | **status unrecorded** | **Record a status first.** A gate whose docstring does not say what it is waiting for cannot be triaged; that absence is the defect. |
| `DISCOPT_DIRECT_HEURISTIC` | substantial measurement, incl. `docs/dev/direct-entry-2026-08-12.md`; explicitly *heuristic-policy, not bound-changing* | measured, regime stated | **Read the recorded panel and decide.** The evidence exists; the verdict was never written down. |
| `DISCOPT_IPX_CHEAP_FIRST` | the falsification *and* the verdict are both in `_ipx_cheap_first_enabled`: node count was the wrong metric; re-measured in wall clock the gate is **11.1x slower** for a 6.9 % node saving (12 instances, interleaved, 2 reps, pooled sd <= 0.26 s) | measured, verdict recorded | **RESOLVED — keep as a documented opt-out.** The 2026-09-19 pass read this as "verdict never written down"; that was wrong. The docstring says *"Kept default-OFF rather than deleted, per the `DISCOPT_CUT_INHERIT` precedent... Re-graduating it requires a WALL-CLOCK panel"* — §5 state 3, complete. Corrected 2026-09-20. |

## Gates added after the first audit

| flag | what its own text records | state | verdict |
|---|---|---|---|
| `DISCOPT_OA_CONVEXITY_CERTIFICATE` | #1352: OA consults the interval-Hessian certificate and the exact-QP route for objective (and per-row) convexity, so a PSD objective written with `dm.sum` gets objective cuts. The first panel's "not net-positive" was OA's master-gap window and fixed-NLP tolerance, both fixed in #1360 — retracted in the flag's docstring | panel ran (performance-plan §73) | **Graduated in #1360** — default ON, opt-out `=0`. The default solve does not route a certificate-only objective to OA (`DISCOPT_CONVEX_ROUTE_SYNTACTIC_OBJECTIVE`, default ON, opt-out `=0`), so the graduation changes explicit OA-family callers only. |
| `DISCOPT_FARKAS_RAY_CLEANUP` (Rust, `lp/simplex/primal.rs`) | #1355: re-verify a rejected infeasibility ray with its rounding-noise entries (`\|y_i\| ≤ 1e-12·‖y‖∞`) zeroed; soundness rests on the unchanged rigorous verifier. Captured OA master: 278-node `ITERATION_LIMIT` → 149-node `OPTIMAL` | panel ran (performance-plan §73) | **Graduated in #1360** — default ON, opt-out `=0`. The §5 panel (990 rows, 7346 certificate checks, 198 instances × 30 s, the two arms differing only in this flag) is *cert-clean* — no bound above its reference optimum, no certification regression — and *net-positive*: 97 → 100 certified instances, total wall 3362.6 s → 3293.8 s. The one instance that loses a certificate with the cleanup ON is `gams01`, which certifies **falsely** with it OFF (bound 28274.71 against a reference optimum of 21380.20), so that loss is the fix, not a regression. |
| `DISCOPT_TREE_SENTINEL_PRUNE_GUARD` (Rust, `bnb/tree_manager.rs`) | C-47: `import_results` took `result.lower_bound.max(node.local_lower_bound)` unconditionally, so a *failure* sentinel (`1e30`) became the node's lower bound, pruned it as "dominated" and let `update_global_lower_bound` collapse the tree bound onto the incumbent. The guard demotes a sentinel that did not arrive with `sentinel_is_exclusion` to `-inf` | panel ran (this PR) | **Graduated on introduction** — default ON, opt-out `=0`. 63 instances × 20 s, interleaved, 340 executed checks: *cert-clean*, 0 violations, certified 49/63 in both arms, `CERT_GAINED=[] CERT_LOST=[]`. Cost: 5/63 changed node count, +570 nodes (11284 → 11854) for +0.6 s wall; four of the five are at the time limit (two improved their dual bound), and among instances that terminate exactly one moved — `ex14_1_9`, 5 → 11 nodes, same objective, same certificate. §5's *net-positive* bar is not the applicable gate: it exists for bound-tightening features, and this is a soundness guard, which §1 governs. |

| `DISCOPT_BLOCK_VECTOR_EVAL` | #1370 Part B: Jacobian and Lagrangian-Hessian VALUES for a model that *declares* block structure come from one colored, vectorised pass instead of the tape's per-row walk. Measured (`docs/dev/1370-block-eval-entry-2026-09-20.md`): derivatives 3.16x / 2.30x and **solve 1.26–1.29x at K=32**, identical objectives and iteration counts | measured; engaged only on a declared block partition | **Keep as a documented opt-in.** Not the default for two stated reasons: it applies only to models that declare a block partition (no in-repo corpus instance does, so the §5 corpus panel has nothing to run on), and it puts JAX on the solve path, which is deliberately tape-only. What would change that: a corpus of block-structured instances to run the differential panel over — then it graduates on the usual two bars. Until then the values are guarded per build by an entrywise admission check against the default evaluator, and a disagreement is a refusal, not a fallback. |

## What this audit does not do

It does **not** delete anything. Per #1345 the deletions are follow-on PRs; this lands the
rule and the verdicts. `DISCOPT_LP_SPATIAL_MIXED` (#1357) and `DISCOPT_POUNCE_DECLARED_BOX` (#1358) have since
been **retired** — the rule's first two applications, and the proof it terminates.
`DISCOPT_CONVEX_KERNEL` has since **graduated** (#1346) — the rule's first
graduation, after two retirements, and the evidence that it is a real three-outcome rule
rather than a deletion pipeline. No verdict above is left as "recorded and forgotten",
which is the state #1345 exists to end.

**What #1346 adds to the rule, and it is not about this flag.** Its §5 panel passed both
bars over the full in-repo corpus *while the change it was scoring hijacked the entire
LP/MILP route* — a pure LP or MILP satisfies every clause of the convexity gate trivially,
and `Model.solve()` consults the kernel before the HiGHS route. The corpus could not show
it: all 66 in-repo instances are MINLP `.nl` files, so not one is a pure LP or MILP. The
smoke suite caught it, 17 failures. A second finding the same day: the counter-case that
motivated a guard shipped alongside the graduation (`watercontamination0202`, recorded in
the parity analysis at 2001 s with no bound) turned out, when run, to be refused by the
gate's own pre-existing `nonlinear objective` clause — the guard was defending against a
route it could not reach, so it was deleted rather than shipped. **A passing corpus panel
bounds what was looked at, not what was affected**; a graduation owes the affected-class
question separately, and the flag's own docstring owes the counter-case an actual run.

**What retirement kept, and why it is not deletion-by-name.** #1357 removed the env flag
and both production call sites, so the default path uses the pre-#860 gate. It did *not*
remove the `mixed=` parameter on `_is_in_scope` / `solve_lp_spatial_bb`: that capability
is sound, separately tested (four cases in `test_lp_spatial_bb.py`), and is what §5 means
by "keep reusable pieces, regression fixtures and benchmark cases; the entry point goes".
The graduation measurement moved onto `_is_in_scope`'s docstring rather than dying with
the flag, so anyone weighing the widening again starts from the evidence.

**The nine that remain now have an owner-facing home: #1388.** #1345's "done when" required that no flag be left as "recorded and forgotten"; nine were, until the v0.9.0 release audit. #1388 carries the checklist, the per-flag reasoning, and the `GRADUATION_ARMS` step each one needs first.

It also does not re-run any panel. Every verdict above rests on what the tree already
records — which is the point: the evidence was never the missing part.
