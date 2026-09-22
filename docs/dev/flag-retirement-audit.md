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
#1358 (retired).

**Settled 2026-09-20 (#1388).** Of the 11 live solver-math gates, **six are retired** and
**five keep a recorded status**. Re-derived after the work:

```
call sites scanned                 : 90   (was 97)
distinct DISCOPT_* flags read      : 83   (was 89)
  default is a literal "0"         : 10   (was 16)
    numeric knobs / non-solver     :  5   out of scope
    gates over solver math         :  5   all with a recorded §5 state
```

**Zero gates are now in none of §5's three states** — the defect the rule names is closed.

> ### RETRACTED 2026-09-22 — see [#1421](https://github.com/jkitchin/discopt/issues/1421)
>
> The sentence immediately above is **false**, and not marginally. It is true only of the
> population this document's scan can see, and that scan recognises **one** of the four ways
> this codebase reads a `DISCOPT_*` boolean gate. Measured with `ast` over 352 files under
> `python/discopt/`:
>
> ```
> FORM 1  environ.get(F, '<lit>')       : 66 flags
>           of which default '0'        : 10  <-- the ONLY form scanned
>           of which default ''         : 11
> FORM 2  environ.get(F)  [default None]: 16 flags
> FORM 3  _env_flag(F, default=...)     : 46 flags
>           default=False (OFF)         : 23
>
> flags the scan sees       : 10
> flags the scan CANNOT see : 50   (overlap with the scanned set: none)
> ```
>
> The 23 `_env_flag(..., default=False)` gates are default-OFF stated more plainly than any
> `"0"` literal, and many are squarely bound-changing solver math — the whole `DISCOPT_RLT`
> family, `SHOR_SDP_ROOT_BOUND`, `PHASE2_DBBT`, `MULTILINEAR_COUPLING_RLT`,
> `SOS1_SELECTOR_BRANCH`. None has ever appeared in this audit.
>
> **Why the marker below could not catch this.** `test_1345_flag_retirement_audit.py` argues
> that cross-checking the scan against the document is "strictly STRONGER" than a numeric
> floor, because "a regex that silently stops matching some gates now fails here". That holds
> for a regex that *stops* matching. This one never matched these forms at all, so the scan
> and the marker are both derived from the same too-narrow definition of "default-OFF" and
> agree with each other while the real population is several times larger. Two instruments,
> one blind spot, mutual confirmation — the §6 failure mode, in the instrument built to
> enforce §5.
>
> **Polarity is not readable from the default literal.** Five of the eleven `""`-default
> flags (`CONVEX_STALL_ABSTAIN`, `GDP_CONFIG_PRIMAL`, `NLPBB_ROOT_CUTS`, `QUBO_PRIMAL`,
> `ROOT_BOUND_SEED`) are default-**ON**: their predicate is negative (`not in ("0","false",
> "no")`) or returns `True` on the empty string. A widening that treats every `""` default as
> OFF misclassifies all five. Polarity has to come from the gate's own comparison, or from
> `_env_flag`'s `default=` keyword.
>
> **Six `""`-default gates over solver math are missing rows**, and #1421 records what a
> `docs/dev` grep already turns up for them — including `DISCOPT_OBBT_ITERATE`, whose §5 panel
> **was already run** (issue-282 plan, Workstream B, "HOLD" 2026-07-18: sound, 0 violations,
> **0/82 certifications gained**) and whose result was never brought here. That is the #1388
> lesson repeating: grep `docs/dev` before triaging from a docstring.
>
> **The marker below is left at its current value on purpose.** Moving it to 11 (the
> `""`-default fix alone) would make the test agree with the document again at a new number
> while 50 flags stayed invisible — the same failure this retraction reports, one number to
> the right. A partial widening was written and deliberately reverted. The marker moves once,
> to the real population, under #1421; until then it should be read as *"gates of FORM 1 with
> a `\"0\"` default"*, which is all it has ever meant.
The five that remain: `COEF_TIGHTEN`, `G_CONVEX_CUTS`, `OA_INFEASIBLE_NOGOOD`,
`IPX_CHEAP_FIRST` and `NLP_NATIVE` — all five now **documented opt-outs**.
`COEF_TIGHTEN` was the last graduation candidate; its panel was run under #1414 and
settled it (below).

The marker below is the machine-readable count of *live* default-OFF gates over solver
math. `test_1345_flag_retirement_audit.py` asserts the source scan agrees with it exactly,
so this document and the tree cannot drift apart silently — and a retirement has to update
it, which §5 requires anyway. Three prose blocks above quote historical counts (14, 11);
this marker is the only current one.

<!-- live-solver-math-gates: 5 -->

**The finding that matters more than the count.** This audit's first pass triaged flags
from their *gate docstrings*, and for **four of the nine** that was wrong — the evidence
was in `docs/dev/` all along. `G_CONVEX_CUTS` had a panel showing −26 % nodes;
`COEF_TIGHTEN` had a measured +2608 % → +1145 % root-gap gain; `OA_INFEASIBLE_NOGOOD` had
a written decision; `DIRECT_HEURISTIC`'s cited document was about a different component.
Two of those were nearly deleted on the strength of a docstring that said nothing. An
audit that reads the docstring measures the docstring, not the flag — grep `docs/dev`
first.

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
| ~~`DISCOPT_CMIR_AGGREGATION`~~ | `docs/dev/cut-engine-entry-2026-07-06.md`: with the flag ON the root bound on `nvs17`/`nvs19` is **bit-identical** to cuts-off; the separator "finds nothing to add" and "correctly self-disables" — discopt's relaxation already dominates SCIP-with-cuts on that family | measured **INERT** | **RETIRED (#1388).** The textbook §5 cert-clean-but-neutral outcome. Deleted end to end: flag, `_separate_aggregation_mir_cuts`, both call sites, the `aggregation_mir_cuts_py` binding and `crates/discopt-core/src/lp/aggregation.rs` (484 lines). It was also the only gate of the nine with no Python test at all. |
| `DISCOPT_COEF_TIGHTEN` | #1414 graduation panel, 92 firing instances at 60 s: **node count 95.0 % neutral (±2 %), ON/OFF ratio 1.0542**; answer quality **0/92 status transitions**, no `error` rows either arm. Earlier: `docs/dev/issue-282-stage2-verdict.md` `syn40m` root gap +2608 % → +1145 % | **panel RUN 2026-09-21** | **RESOLVED — documented opt-out.** BAR 1 (cert-clean) passes, BAR 2 (net-positive) does not: cert-clean but neutral-to-slightly-worse on the corpus, so it does not graduate (the `CUT_INHERIT` lesson). Not retired either: it is the only mechanism that removes the #1380/#1414 big-M weakness at the source (`x <= M z` → `x <= 10 z`, exact relaxation, 0 extra nodes at every `M`), where the default path now certifies the class by *branching* (#1414). MINLPLib's big-M rows arrive pre-tightened by the modeller, so the corpus cannot see the win. **What would change it:** a panel over user-authored big-M models (indicator coefficient exceeding the implied bound). |
| ~~`DISCOPT_SGO`~~ | no panel; and the flag was the 1,532-line engine's **only** caller, so §5 retirement would have deleted Lundell-Westerlund/Maranas-Floudas signomial global optimisation outright, rigorous infeasibility certificate and all | no measurement either way | **RETIRED (#1388) — flag only; the engine was KEPT and given the entry point it never had.** `solver="sgo"`, the same shape `gp-minlp` already used. Its three integration tests moved from `DISCOPT_SGO=1` + `solve()` to `solve(solver="sgo")` with **every assertion unchanged** — same certified objective, same bound, same `gap_certified`. That is the evidence the retirement was entry-point-only. |
| ~~`DISCOPT_GP_MINLP`~~ | gated **auto-routing** from a plain `solve()`, never the engine; `solver="gp-minlp"` was always the real interface | n/a — nothing to measure | **RETIRED (#1388), zero capability lost.** The audit's own "then the env flag is redundant and should go" branch, taken. `test_auto_route_on_with_flag` went with it; `test_auto_route_off_by_default` became `test_never_auto_routed` and now guards the retirement. |
| `DISCOPT_OA_INFEASIBLE_NOGOOD` | `docs/dev/performance-plan.md` §25.7, titled **"REJECTED as a default"**, with §25's verdict table recording **"stays OFF"** | decision taken and recorded | **RESOLVED — keep as a documented opt-out.** Corrected 2026-09-20; the gate docstring said only "until a corpus panel clears", which read as a stalled graduation. The reasoning is written out: an OA cut excludes the *point*, not the *assignment* (7 of 172 assignments re-proposed, one six times), and the no-good exclusion is sound only when the assignment is **proven** infeasible — which is why mapping an Ipopt code-2 return to `INFEASIBLE` was rejected. The docstring now cites §25.7. |
| ~~`DISCOPT_PRESOLVE_SUBSTITUTE`~~ | #888's §5 panel, via `docs/dev/sota-parity-analysis-2026-07-27.md`: 13 of 66 vendored instances had anything to substitute and the flag gained **0 incumbents and 0 certifications** (48/48 certified, 54/54 incumbents, both arms). Plus an open blocker — `hda` bound quality OFF −64,473 → ON **−1.56e8**, sound but far looser — and a 300-instance census where it reduces nothing on 64.3 % and reaches ≥10× on **0/300** | **panel ran; gate 1 PASS, gate 2 FAIL** | **RETIRED (#1388).** The strongest retirement case of the nine, and the same shape as #1357. The doc itself pre-empts the alternative: completing the mechanism "would land a second default-OFF flag with the same measured 0-incumbent, 0-certification profile — the `DISCOPT_CUT_INHERIT` lesson (sound ≠ helpful)". Only the **solve-path entry** was deleted; the Rust `presolve/substitute.rs` pass, `substitute_to_fixpoint`/`postsolve_chain`, the `ModelRepr.substitute` binding and `propagate_bounds_to_model`'s use of its tightened bounds all stay. |
| ~~`DISCOPT_PSD_QFORM`~~ | no benefit measurement of its own — and its **successor graduated**: #936's `DISCOPT_QP_EXACT_CONVEXITY` is default-ON on a full panel (0/284 violations, 46/46 proven-optimal instances byte-identical in nodes *and* objective, repro `feasible`/5,101 nodes/60.4 s → **optimal/0 nodes/0.59 s**) and makes the identical exact-eigenvalue argument from an extractor that also handles the vectorized API | superseded | **RETIRED (#1388).** What went with it is the exact-PSD verdict on *constraint* bodies and on objectives not classified QP/MIQP; those fall back to the interval-Hessian + Gershgorin path, which abstains rather than mis-certifies. `_PSD_TOL` stays — four other call sites use it. |
| `DISCOPT_G_CONVEX_CUTS` | gate function was one line; rationale looked thin — **but a panel exists**, `docs/dev/g-convexity-cut-panel-2026-07-17.md` | panel ran TWICE: root arm cert-clean but inert (0 cuts over 46 instances); per-node arm 0 soundness/neutrality violations over 18 instances and **53 → 39 nodes (-26 %)** at the same certified optimum | **RESOLVED — keep as a documented opt-out.** Corrected 2026-09-20: the 2026-09-19 pass read the thin gate docstring and recorded "status unrecorded", which led the v0.9.0 audit to list this among the *panel owed, never run* set. It was not — only the ~4,800-instance corpus benefit arm is missing, for want of the MINLPLib snapshot. The status now sits on the gate function. **Not a retirement candidate.** |
| ~~`DISCOPT_DIRECT_HEURISTIC`~~ | **the audit cited the wrong document.** `docs/dev/direct-entry-2026-08-12.md` is the pre-registered entry experiment for the `solver="direct"` **backend**, which ships as a selector — not for this root primal *heuristic*. The heuristic's own 585-line test file states its scope outright: *"the soundness envelope, **not the search quality**"* | no benefit measurement exists | **RETIRED (#1388).** Sound, governed and well-tested, but never shown to help anything, and reachable only on models outside the #764 native spatial kernel. `_direct_root_primal`, its two constants, the gate and the B&B call site go; `direct` leaves `EXPENSIVE_SOURCES`/`GOVERNED_SOURCES`, restoring the governed set to the one source its own panel measured. `solver="direct"` is untouched. |
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
