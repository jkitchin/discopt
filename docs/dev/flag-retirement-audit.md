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
it, which §5 requires anyway. Several prose blocks above quote historical counts (14, 11,
5) taken under the old, too-narrow scan; the marker at the end of the next section is the
only current one, and it is the first that describes the whole population.

## Re-derived 2026-09-22 against the whole population (#1421)

The retraction above promised the marker would move **once**, to the real population.
This is that move. The scan was rebuilt in `python/tests/flag_audit_scan.py` — an
`ast` walk over `python/discopt/`, not a regex — and it reads **five** forms, one more
than the retraction listed:

```
FORM 1  environ.get(F, "<literal>")              : 75 sites
FORM 2  environ.get(F)              [default None]: 18 sites
FORM 3  _env_flag(F, default=...)                : 46 sites
FORM 4  environ["F"] / a non-literal default      :  1 site
FORM 5  environ.get(CONST) where CONST = "DISCOPT_…" at module level
```

FORM 5 was **not** in the retraction's list and is the reason this pass re-derived the
forms rather than trusting them. `DISCOPT_BLOCK_VECTOR_EVAL` — a documented default-OFF
gate with a row in this very document — is read only through a module-level constant, so
a scan that already handled four forms still could not see it. An enumeration of read
forms is itself a measurement and decays the same way; the scan now asserts it visited
>100 files and prints an executed-classification count (§6).

```
distinct DISCOPT_* flags read      : 133   (the old scan saw 10)
call sites                         : 140
  default ON  (shipped default)    :  65   out of scope — opt-outs, §5 says so
  selector / numeric knob          :  17   out of scope — "0" is a value, not an off-switch
  value, not a gate                :   4   see below
  default OFF                      :  47
    gates over non-solver behaviour:   3   out of scope (EAGER_IMPORTS, DISABLE_JAX_CACHE, GAMS_NO_DAEMON)
    gates over solver math         :  45   <- this audit
```

**Polarity is measured, never inferred from the default literal.** The scan classifies a
read by the *word class* of its default against the gate's own comparison vocabulary
(`1/true/yes/on` vs `0/false/no/off`), and for a zero-argument accessor it imports the
module and **calls** it under a cleared environment. The operator sign is deliberately not
consulted: `x in ("0","false")` and `x not in ("0","false")` differ in sign but the
feature's default state is fixed by whether the default word is an ON-word or an OFF-word.
An earlier version of the scan read the sign and got `MILP_ROOT_CUTS`,
`CONVEX_STALL_ABSTAIN` and `NLPBB_ROOT_CUTS` backwards; `test_polarity_is_not_read_off_the_default_literal`
pins all five `""`-default ON gates the retraction named.

**Validation of the instrument (§6, §8).** The scan's verdict was cross-checked against
the 42 flags whose state is already documented somewhere in the tree: **0 disagreements**.
Four reads it cannot classify as booleans are listed in `NOT_A_GATE` in the test with the
reason each is a value rather than a switch — `DISCOPT_DECOMP_STORE` (a filesystem path),
`DISCOPT_LLM_MODEL` (a litellm model id), `DISCOPT_PROVENANCE_AUTHOR` (an author name) and
`DISCOPT_LP_SPATIAL_PLUNGE` (a three-state override; audited as a gate anyway, below). An
`unknown` that is not accounted for **fails the test** — the pre-#1421 scan had no notion
of a read it could not classify, which is why fifty flags were not "unclassified" but
simply absent, and absence looked like a clean bill of health.

### Retraction of the settled-2026-09-20 claim, restated

"Zero gates are now in none of §5's three states" was false when written and is now
**measured** false by a factor of nine: 5 gates were audited, 45 exist. The claim is
withdrawn in full. What replaces it is not another count but the table below, which has a
row for **every** one of the 45, and a test that fails if a forty-sixth appears without one.

### What the triage found, and it is the #1388 lesson again

Of the 37 gates that had no row, **22 already have written evidence in `docs/`** — an
entry experiment, a plan section, or a measured verdict that was never brought here. That
is the same finding as the first pass ("an audit that reads the docstring measures the
docstring"), at four times the scale. Ten have nothing anywhere: `DETERMINISTIC`,
`LP_COLD_DUAL_START`, `LP_ROW_LOGICALS`, `LP_SPATIAL_PLUNGE`, `NARROW_BOX_BRANCH`,
`PRESOLVE_BOUND_PROPAGATION`, `RLT_LINEQ`, `ROOT_PROBE_SEEDS_FALLBACK`, `SINGULAR_TANGENT`,
`SOS1_SELECTOR_BRANCH`, `TRIVIAL_PRIMAL`.

**None of the 45 is in `generality_sweep.GRADUATION_ARMS`**, so for none of them is there
an arm to run — the structural reason the first pass identified for its nine holds for all
forty-five. "Panel owed, never run" is therefore the honest state for most rows below, and
this document has said since its first pass that **that is itself the finding**, not a gap
in the audit.

<!-- live-solver-math-gates: 45 -->

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

## The 38 gates #1421 made visible

Every row below is a gate that was default-OFF over solver math for as long as it has
existed and that this audit could not see until the scan was rebuilt. The verdict column
uses §5's three states; where the evidence is an entry experiment rather than a graduation
panel, the row says so, because **an entry experiment is not a panel** and treating one as
the other is how a flag graduates on a synthetic proxy (the #727 RLT lesson, which two of
these rows are literally about).

### Retraction: nine of these rows were wrong when first written (2026-09-22)

CLAUDE.md §11 requires retracting a published claim in writing when a measurement
contradicts it. The first pass of the table below was written largely from each gate's
**docstring**. A second pass grepped `docs/dev` for every flag name *and its mechanism*,
and found a decisive recorded verdict for nine gates this table had filed as "panel owed"
or "keep as an opt-out". Every citation below was re-read at the cited line before this
correction was written.

| flag | what this table said on the first pass | what `docs/dev` actually records | corrected state |
|---|---|---|---|
| `DISCOPT_OBBT_TOPK` | "**nothing recorded anywhere** … neither records a verdict" | `certification-gap-plan.md:393`: **"BUILT → KILL (2026-07-11) … bound is flat/WORSE and net wall is negative (kill-crit b) … do NOT flip default"**, `casctanks` bound 1.8023 → **1.3523** | **retirement candidate** |
| `DISCOPT_OBBT_ITERATE` | *no row at all* (only a prose mention at line 80) | `issue-282-implementation-plan.md:209-211`: **"VERDICT: HOLD (recorded 2026-07-18) — Workstream B is CLOSED, do not re-run the panel … sound (0 violations) but net-negative-to-neutral — 0/82 certifications gained"** | **retirement candidate** |
| `DISCOPT_BUDGET_SATURATION` | "Panel owed — high priority" | `budget-monotonicity-1153.md:205,212`: panel **ran** — incumbent better/worse/same **0/0/17**, dual bound weaker under ON on **3**, stronger on **0**, "**Verdict: cert-clean, NOT net-positive**"; §6.1 falsifies the premise and line 3 declares "the whole wall-budget fix family is **ruled out**" | **retirement candidate** |
| `DISCOPT_HEUR_ENTRY_SHARE` | "Panel owed — high priority" | same doc, `:460,474,477`: panel D **ran** — 3/2/63, "**Neither arm graduates**", "**This rules out the whole 'gate or bound the pump by wall budget' family**" | **retirement candidate** |
| `DISCOPT_ANALYTIC_SEPGRAD` | "the path's own benefit is unmeasured … keep as a documented opt-out" | `jax-removal-plan.md:199-204` (RESCOPE 2026-08-04): "**Graduating `DISCOPT_ANALYTIC_SEPGRAD` is dropped.** Its panel passed cert-clean … but was **neutral** on benefit — **48 of 49 instances identical, 1 better, 0 worse**" | **retirement candidate** |
| `DISCOPT_PHASE2_DBBT` | "Panel owed" | `issue-764-root-relaxation-plan.md:347,355`: "**DBBT is VACUOUS on the target class**" — 0 tightenings on `tanksize` with marginals available on 23/23 node solves; fires only where the bound is already tight | **retirement candidate** |
| `DISCOPT_ROOT_PROBE_SEEDS_FALLBACK` | "Panel owed, and it should be short" | the panel **already ran** and lives in the gate's own docstring, `solver_tuning.py:714,724`: "**Panel verdict: stays OFF.** Sec.5 bar 1 PASSES, bar 2 does not clear the `DISCOPT_CUT_INHERIT` precedent … the **flag did not move a single dual bound**", with script `issue930_root_probe_bound_panel.py` and results in `results/issue930/` | **retirement candidate** |
| `DISCOPT_NODE_PROBING` | "the cost/benefit question … is unanswered" | `pf1-branch-and-reduce-spike-2026-07-14.md:27,87`: `cvxnonsep_psig40r` **+40 % wall** (13.6 → 19.9 s, same node count, 0 tightenings) and clay incumbent starvation; recommendation "keep default OFF **pending a scored/budgeted probing policy**". `engine-performance-plan.md:94` carries that as **EP6**, an open owned item | **documented opt-out** |
| `DISCOPT_P3_FORCE_CUT_PATH` | "a measurement instrument … meets state 3 as written" | correct as far as it goes, but omitted the verdict: `certification-gap-plan.md:1050,1075` — "**Verdict — NO-GO / RE-SCOPE**", "not a shipping feature". The experiment it exists for is **finished** | **opt-out, with the NO-GO recorded** |

**How the error happened, and why it is the same error this document is about.** Eight of
the nine rows were written from the gate's docstring. #1388's lesson was "an audit that
reads only the gate's docstring measures the docstring, not the flag" — and this audit,
written to repair that, reproduced it at scale on its own first pass.

The sharpest case is `DISCOPT_HEUR_ENTRY_SHARE`, where the docstring does not merely omit
the verdict — **it documents a mechanism the tree does not ship.**
`solver_tuning.py:1347` describes requiring a heuristic "to fit a bounded SHARE of the
remaining budget", i.e. *entry refusal*; `solver.py:13934` states outright that "**this
flag CAPS a finder stage's clock … rather than refusing its entry. Refusing was measured
and rejected: it costs `nvs05` its incumbent outright (1269.7 → none)**". The first pass
transcribed the refusal mechanism into this table as if it were shipping. A row describing
the wrong mechanism is worse than a missing row, because it reads as a verdict.

**The converse trap, which a `docs/dev` grep alone would have walked into.** Three gates
carry their entire panel **in the gate docstring and nowhere in `docs/dev`** —
`ROOT_PROBE_SEEDS_FALLBACK` (a full two-bar verdict, plus its own §11 retraction of an
earlier result), `LP_COLD_DUAL_START` (an 8-row QPLIB A/B) and `SPARSE_LARGE_LP`. So
neither source is sufficient alone: #1388 says do not triage from the docstring, and these
three say do not conclude "unmeasured" from a `docs/dev` grep. **Both must be read**, and
the flag's *mechanism* must be grepped alongside its name — `SINGULAR_TANGENT`'s name
appears nowhere in `docs/dev`; its panel was found only by grepping "vertical tangent".

### Panel RAN and did not pass — retirement candidates

These are the clearest cases in the whole audit: §5's second state, reached by its own
procedure. They are listed as candidates rather than executed here because each deletes
solver math, and #1345's rule that "this audit does not delete anything; the deletions are
follow-on PRs" is the precedent.

| flag | what its own text records | state | verdict |
|---|---|---|---|
| `DISCOPT_SINGULAR_TANGENT` | **"The §5 panel ran. Gate 1 (cert-clean) PASSES; gate 2 (net-positive) FAILS: the EAGER anchor is measured HARMFUL."** `tspn08` 135 → 191 nodes (**+41.5 %**) to buy a bound gain in the 11th digit (290.56592504129753 → 290.56599569540646); `mathopt5_6` flat 5 → 5, bit-identical bound; `kriging_peaks-full010` flat to 14 digits | **panel ran; gate 1 PASS, gate 2 FAIL** | **RETIREMENT CANDIDATE** — the textbook `CUT_INHERIT` outcome, already measured. The underlying defect it found is real and separate: `_emit_1d` silently drops a tangent facet at a vertical tangent (`sqrt` at 0, `asin`/`acos` at ±1). Retiring the flag must keep `_interior_tangent_point` and the regression tests; what goes is the eager anchor. #1111. |
| `DISCOPT_NODE_ROUND_BUDGET` | #966: "the ON arm's wall goes UP — 20 s-budget sign flip: ON−OFF **+325.4/+68.5/+12.7 s over 3 reps**" (contvar @ 20 s, `scratchpad/issue966_phase_probe.py`) | **measured harmful, 3 reps, consistent sign** | **RETIREMENT CANDIDATE.** Not cert-unclean — it honors a grant, which is sound — but net-negative by a wide margin and never re-measured. The mechanism it exposed (a round's non-LP cost spent after the admission check) is a real finding worth keeping in the plan doc. |
| `DISCOPT_OBBT_TOPK` | cert:T2.5, `certification-gap-plan.md:393`: **"BUILT → KILL (2026-07-11) … OBBT now RUNS on casctanks (n=500)/ex8_3_13, but bound is flat/WORSE and net wall is negative (kill-crit b) … do NOT flip default"** — `casctanks` 1.8023 → **1.3523**, `ex8_3_13` flat, three instances inert. Independently found inert on all 12 global-path instances in the PF1 spike | **panel ran; killed 2026-07-11** | **RETIREMENT CANDIDATE.** Keep the width×\|RC\| scoring for the "targeted-budget revisit" the verdict names; the de-gate goes. **Its module comment at `solver.py:565` is actively false** — it tells the reader the flag is "Default OFF until the differential + panel gates are green on consecutive nightlies", when the panel ran and killed it 2026-07-11, and consecutive-nightly graduation was dropped 2026-07-17. Fix or delete that comment in the same PR. |
| `DISCOPT_OBBT_ITERATE` | `issue-282-implementation-plan.md:209-211`: **"VERDICT: HOLD (recorded 2026-07-18) — Workstream B is CLOSED, do not re-run the panel. The §5 graduation panel was already run as #727 Track 1.2. Outcome: sound (0 violations) but net-negative-to-neutral — 0/82 certifications gained."** Artifact: `results/issue282/obbt_iterate_workstream_b_HOLD_verdict.json` | **panel ran; gate 1 PASS, gate 2 FAIL** | **RETIREMENT CANDIDATE.** The verdict even says not to re-run it, so "run the panel" is not an available third option. This is the `CUT_INHERIT` outcome with the measurement already in hand and an artifact on disk. |
| `DISCOPT_BUDGET_SATURATION` | `budget-monotonicity-1153.md:205,212`: incumbent better/worse/same **0/0/17**; dual bound **weaker** under ON on 3 (`beuster` 10245.8 → 10221.6, `casctanks` 6.8014 → 6.5773, `contvar` 183637 → 183436), **stronger on 0**; "**Verdict: cert-clean, NOT net-positive.**" §6.1 goes further and falsifies the premise: the panel is monotone in **both** arms (34/34), and `ROLE2_SATURATION_S = 150` makes the flag **inert below a 150 s budget**, so it cannot touch the 30 s vs 60 s inversion #1153 actually reports | **panel ran, gate 2 FAIL, premise falsified** | **RETIREMENT CANDIDATE — the strongest in the audit.** The doc's header declares "the whole wall-budget fix family is **ruled out**". Keep `saturate_role2`'s unit contract, the 38-site carve inventory and the static ratchet in `test_1153_budget_monotonicity.py`; the flag and its six call-site wrappers are the entry point that goes. The `nvs19` non-monotonicity this table first cited is the **motivating defect**, not evidence for the flag — the flag was the attempted fix, and it failed. |
| `DISCOPT_HEUR_ENTRY_SHARE` | Same doc, `:460,474,477`: panel D ran — incumbent better/worse/same **3/2/63**, nodes up/down 5/1, 0 cert regressions; "**Neither arm graduates: D still loses `tanksize`@5 s outright and degrades `tspn10`@5 s**"; family verdict "**This rules out the whole 'gate or bound the pump by wall budget' family**" | **panel ran; gate 2 FAIL** | **RETIREMENT CANDIDATE**, with `BUDGET_SATURATION` — they are two halves of one ruled-out family. **And its docstring must be fixed or deleted regardless of the §5 outcome**: `solver_tuning.py:1347` documents *entry refusal*, while the shipped flag caps a stage clock (`solver.py:8124`) and leaves `_root_heur_nlp_entry_ok` on the legacy rule, because refusal was measured and rejected (`nvs05` 1269.7 → none). The docstring's "why it cannot graduate as written" paragraph reasons about a mechanism that was removed. |
| `DISCOPT_ANALYTIC_SEPGRAD` | `jax-removal-plan.md:199-204`, RESCOPE 2026-08-04: "**Graduating `DISCOPT_ANALYTIC_SEPGRAD` is dropped.** Its panel passed cert-clean … but was **neutral** on benefit — **48 of 49 instances identical, 1 better, 0 worse**." Superseded by `DISCOPT_SEPGRAD=tape`, graduated default-ON at `a2fb90d2`. Its `interval_ad` covers 6 of ~30 operators at 8.53e-14 accuracy against the tape's 2.08e-16 | **panel ran; gate 1 PASS, gate 2 neutral** | **RETIREMENT CANDIDATE** (the `PSD_QFORM` shape). Two earlier docs (`tenx-plan.md:316`, `nvs05-decline-taint-2026-07-16.md:159-167`) file it as awaiting graduation with "+1 proof" — both **predate** the RESCOPE, and that single proof is exactly what the 49-instance panel re-read as neutral. Dating resolves the conflict. Related stale text to fix: `_relax/uniform_relax.py:1188` still says the caller "falls back to JAX"; it falls through to the tape. |
| `DISCOPT_PHASE2_DBBT` | `issue-764-root-relaxation-plan.md:347,355`: "**sound; but the cheap DBBT is VACUOUS on the target class**" — 0 tightenings on `tanksize` with marginals available on **23/23** node solves; fires only where the bound is already tight (`gkocis` 2+2, `ex1221` 1); step 4 calls graduation "unlikely" | **measured on the target class; vacuous** | **RETIREMENT CANDIDATE** (the `CMIR_AGGREGATION` shape: the measurement that would have been the panel already ran and said no). Not a formal corpus panel, so if anyone wants to keep it, the burden is to name an instance class where DBBT is *not* vacuous — which is precisely what step 4 doubted. |
| `DISCOPT_ROOT_PROBE_SEEDS_FALLBACK` | The panel ran and lives in the gate's **own docstring**, `solver_tuning.py:714,724`: "**Panel verdict: stays OFF.** Sec.5 bar 1 PASSES, bar 2 does not clear the `DISCOPT_CUT_INHERIT` precedent" — 22 ON/OFF bound pairs, **0 lost, 0 looser, 0 tighter**, 0 `gap_certified` True→False; "**the flag did not move a single dual bound**". Bar 2: paired wall ON−OFF **+0.048 s/run** (sd 0.369, n=34) — fractionally *worse*. Carries its own §11 retraction of an earlier −0.230 s/run result (wrong baseline branch + bimodal `contvar`). Script `issue930_root_probe_bound_panel.py`, raw results in `results/issue930/` | **panel ran; bound-inert and marginally slower** | **RETIREMENT CANDIDATE** — the exact `CMIR_AGGREGATION` shape #1388 retired. Judgement call rather than a defect: by the letter of §5 its docstring already qualifies it as state 3, since it names what would settle it. But "measured bound-inert with a small wall cost" is the definition of not-net-positive, and the duplicate-solve finding it made is worth keeping on its own. |

| `DISCOPT_RLT` | A **redundant legacy entry point**. `mccormick_lp.py:565-567`: the first-class control is `rlt_level1=True` threaded from `Model.solve(rlt=...)`, "with the **legacy `DISCOPT_RLT=1` environment variable kept as a force-on override for benchmarking**". Where it has been measured on a real instance it is an exact no-op: `issue-282-syn-rsyn-diagnosis-2026-07-17.md:503` (syn/rsyn root gap 0.838 → 0.838, "exactly no-op") and `issue-764-root-relaxation-plan.md:74-81,159` (`tanksize` root 0.8382 unchanged by both `rlt_level1=True` and `rlt_cuts=True`; "do not ship an RLT-family relaxation for this class") | **redundant entry point; measured inert on both real classes tested** | **RETIREMENT CANDIDATE — entry point only, not the capability.** The capability stays: it is reachable through the public `rlt=` / `rlt_level1=` arguments, which is why this is a §5 retirement and not a loss of function. **Caveat for whoever deletes it: it is not an exact alias.** `solver.py:12904,12918` shows `rlt=True` forcing level-1 **and** per-node cuts (`_eff_rlt_cuts`), while the env flag only makes level-1 *applicable* via `_rlt_applicable`. A benchmarking script that switched from the flag to `rlt=True` would silently also turn on per-node cuts. |

### Panel owed, never run — the bulk

Each of these has a *sound-by-construction* argument in its own text and, for most, a
measured entry experiment on its target class. None has a corpus-wide differential panel,
and **none is in `GRADUATION_ARMS`**, so for none of them does an arm exist to run. Under
§5's three-outcome rule that is not a steady state; it is the defect the rule names, and
the count is 29, not zero.

| flag | what its own text records | state | verdict |
|---|---|---|---|
| `DISCOPT_RLT_LINEQ` | Sherali–Adams level-1 constraint-factor rows; holds with equality at every feasible point, so it never cuts one. Measured on QPLIB continuous nonconvex QPs: `QPLIB_1157` McCormick −14.8046 → −11.7716 against optimum −10.9482, **78.6 % of the gap closed**, and the bound-factor rows add nothing on top | entry experiment on the real class (QPLIB), no panel | **Panel owed.** The strongest unrun candidate in the audit: a real-corpus measurement, a no-box-data row that carries into the native spatial kernel unchanged, and a mechanism that cannot cut a feasible point. First step is one `GRADUATION_ARMS` entry. |
| `DISCOPT_RLT_SPARSE_AUTO` | Widens the RLT auto-engage gate from raw variable count (a "poor cost proxy") to a product-term envelope for sparse-bilinear models. #727 | reasoned; **no measurement of its own** | **Panel owed — and #727 is the cautionary tale attached to this exact family** (synthetic root-gain 0.68, real gain 0.0). Its entry experiment must be on real pooling/bilinear-flow instances, per CLAUDE.md's "Working on an issue" §2. |
| `DISCOPT_RLT1_ROOT_BOUND` | RLT-1 for constrained binary QPs; surfaces the **Neumaier–Shcherbina safe dual bound**, not the raw vertex objective (issue #145); joins the root candidates via `max`, so it can only raise the bound | sound by construction; motivating phenomenon measured (qap root LP ~0) | **Panel owed.** |
| `DISCOPT_RLT1_LAGRANGIAN` | Same rigorous RLT-1 bound via Lagrangian dual, avoiding the degenerate all-pairs LP; `g(mu) <= RLT-1 opt <= true opt` for every `mu` by weak duality, so every iterate is valid. Reaches **100 % of the monolithic RLT-1 bound on synthetic QAPs**, target-free; inner McCormick LP ~0.1 s against >25 min for the monolithic solve | **synthetic only, and its own docstring says so** | **Panel owed, and the entry experiment comes first** — "default off pending the qap-scale entry experiment on the **real** instance" (`rlt-lagrangian-plan.md` §3). #727 again: a 100 % result on synthetic QAPs is exactly the shape that was 0.0 on the real class. |
| `DISCOPT_MULTILINEAR_COUPLING_RLT` | Coupling rows tie the lifted product back to its continuous factor; entry experiment **12658 → ~57435 at that node** (`performance-plan.md` §6) | entry experiment, single node | **Panel owed.** |
| `DISCOPT_SHOR_SDP_ROOT_BOUND` | Strong-Shor SDP root bound; surfaces `shor_sdp_safe_dual_bound`, the SDP analogue of NS — valid for *any* multipliers, so solver convergence affects tightness only, never soundness. Entry experiment: qap root ~0 → **377098 = 97.1 % of the optimum 388214** at ~86 s; exact on brute-forced synthetic Koopmans–Beckmann QAPs. The *plain* Shor SDP was falsified on this class (unbounded on qap, `issue-661-qap-sdp-entry-experiment-2026-07-17.md`) | entry experiment on the real class; falsification recorded | **Keep as a documented opt-out, panel owed for graduation.** Its docstring already meets state 3's bar — it says why it is not the default (**an ~86 s root cost is a deliberate opt-in**) and what would change that (the §5 corpus panel). It also depends on an optional extra (`discopt[sdp]`, SCS), which is an independent reason a default cannot assume it. |
| `DISCOPT_INTEGER_MULTILINEAR_REFORM` | Value-preserving algebraic identity (binary expansion + exact n-ary AND hull); only the *relaxation* changes, so it can only tighten. ex1252: lifts the SOS1-selector-branch dual bound off its 5134 floor. #707 | sound by construction; measured on one instance | **Panel owed.** |
| `DISCOPT_DISJUNCTIVE_CONFIG_BOUND` | Valid by partition, anytime-valid. ex1252 (#732 Stage 2): standalone root pass certifies 37945 at a 48-leaf budget (tree at 400 nodes: 16304) and 63080 at 120 leaves; end-to-end global dual **0.0 → 42725** (240 s) and **0.0 → 74915** (600 s) | measured on one instance, substantial | **Panel owed.** A 0.0 → 74915 dual bound is not a marginal result; this is a graduation candidate waiting only for an arm. |
| `DISCOPT_SOS1_SELECTOR_BRANCH` | Branch-**order** metadata only, never a bound or feasibility input, so it cannot change a bound's validity; the midpoint split is a sound cover. ex1252: an ambiguous box's bound 12658 → ~67–83k once a selector is pinned. #196 | sound by construction; measured on one instance | **Panel owed.** Being order-only, §5's gate 1 is nearly free for it — the open question is purely gate 2. |
| `DISCOPT_NARROW_BOX_BRANCH` | #732 Stage 1-A. A numerically-failed nonconvex node is normally fathomed non-rigorously, **tainting the certified dual bound** (ex1252: internal bound reaches the optimum yet the report collapses to ~0). ON, a still-branchable failed node is kept open at its rigorous parent-inherited bound. Converts ONLY a branchable node, at 2× the Rust brancher's `SPATIAL_MIN_WIDTH`, because an unbranchable open node could falsely certify (the #467 hazard) | sound by construction, hazard identified and guarded | **Panel owed — and this one is a §1 certificate-quality fix, not a performance feature.** It removes a known source of dropped certificates. It should be prioritized above every bound-tightening candidate in this table. |
| `DISCOPT_NODE_PROBING` | P3 per-node probing (#632): tentatively fix each discrete variable at a bound, re-run cutoff-FBBT, contract on **proven** infeasibility. Measured in the PF1 spike (`pf1-branch-and-reduce-spike-2026-07-14.md`): `m3` improves 47 → 25 nodes, but `cvxnonsep_psig40r` pays **+40 % wall** (13.6 → 19.9 s at the *same* 95 nodes and 0 tightenings) and `clay0303hfsg` @ 60 s processes 63 nodes with **no incumbent and no bound** where the baseline gets 251 nodes, incumbent 26669.1, bound 4997.3 | **measured against it on a 12-instance spike (not a panel)**; a named condition exists | **Documented opt-out.** The spike's own recommendation is "keep `DISCOPT_NODE_PROBING` default OFF **pending a scored/budgeted probing policy**", and `engine-performance-plan.md:94` carries exactly that as **EP6**, an open owned item: EP1–EP5 cutting per-node cost is what flips the expensive-node × small-tree economics. That is a why-not-default *and* a what-would-change-it, so it is state 3 — the docstring just needs to say so, since today it names only the cost. If EP6 is dead, this converts to a retirement candidate. |
| `DISCOPT_PRESOLVE_BOUND_PROPAGATION` | Adopts FBBT-derived **feasibility** bounds (`propagate_bounds_to_model` *raises* on an optimality-derived box rather than adopting one). Measured root dual-bound ratio against the reference optimum, 1-node solves, OFF → ON: `syn40m` 27.084× → **15.643×**, `rsyn0840m` 8.534× → **6.185×**, `syn20m02m` 2.788× → 2.646×, `rsyn0805m` 1.629× unchanged; infinite-bound counts 83/169/122/99 → **0**. The OFF column reproduces the ratios #1061 published, which is what makes the ON column comparable | **measured on four real instances, large gains, refusal rather than approximation on the unsound case** | **Panel owed — the strongest graduation candidate in this table.** A 27× → 15.6× root ratio is not marginal, the soundness argument is a loud refusal (§3), and the comparability of the arms was established deliberately. |
| `DISCOPT_ELLIPSOID_BOUNDS` | Convex-quadratic ellipsoid bound rule. Refuses a row worse-conditioned than `_ELLIPSOID_MAX_COND = 1e8` rather than tightening from digits that are not there (cites CLAUDE.md §3); radius slack applied **one-sided, outward**, because a tightening error cuts off feasible points | soundness engineering explicit; benefit not measured | **Panel owed.** Listed in `docs/design/relaxation-catalog.md`. |
| `DISCOPT_TRIVIAL_PRIMAL` | #827: `ball_mk2_30`'s optimum is the origin and `chimera_k64ising-*` has zero nonlinear constraints (so any box point is feasible) — **both return NO incumbent while SCIP solves them instantly**. ON, the root evaluates a handful of trivial points | motivating failure measured on two real instances | **Panel owed.** Primal-only, so it cannot touch a bound; §5's gate 1 is trivial and gate 2 is the only question. Cheap to run. |
| `DISCOPT_ROOT_FIXPOINT_REPOOL` | Refreshing the root cut pool after fixpoint tightening "can only *strengthen* the pool", but costs a full separating solve — **measured +3.7 s on `pooling_adhya1stp`**, an irreducible ~1 s floor not bounded by the LP `time_limit`, which violates cert-plan §14 T2.4's ≤1.05 wall guard. Cuts captured on the wider box stay valid on any sub-box, so no refresh is needed *for soundness* | **measured; declined on an explicit, cited wall guard** | **Keep as a documented opt-out.** This is state 3 and the comment nearly says so already: it names why it is not the default (a measured wall-guard violation) and what it is for (a future strength A/B). The comment should be promoted to the §5 wording; nothing about the decision needs revisiting. |
| `DISCOPT_P3_FORCE_CUT_PATH` | cert:P3.1c. An **entry-experiment lever only**: skips the `nlp_solver → "simplex"` reroute so the integer-product/graphpart class stays on the cut-carrying `_solve_milp_bb` path, measuring whether making cuts *reachable* closes the 0b root gap. "math-neutral when off — bit-for-bit identical" | a deliberate experiment lever; **and the experiment concluded** | **Keep as a documented opt-out — but record the verdict.** Not a stalled graduation and never was; it is a measurement instrument. What the first pass of this row omitted is that the experiment it exists for is **finished**: `certification-gap-plan.md:1050` reads "**Verdict — NO-GO / RE-SCOPE.** Reachability was the wrong diagnosis", and `:1075` "it changes no default behavior and **is not a shipping feature**" — 4 of 5 instances fired 0 cuts and moved the root 0.000; `ex1263a` fired 8 cuts for +0.55 %. Independently re-measured at `performance-plan.md:423` (`autocorr_bern25-25`: bound unchanged at 12.0, root ~10× slower). So state 3 is defensible only as a *diagnostic lever kept on purpose*; one sentence naming the concluded experiment finishes the docstring, and if nobody wants the lever, this is a retirement candidate instead. |
| `DISCOPT_LP_SPATIAL_PLUNGE` | Measured: `nvs24` gained, none lost; quality better=7 worse=2; wall **+7.0 %**; and **exactly one certification regression, `gear2`**. The default was therefore restricted to the fallback path, keeping the primal gains and leaving `gear2` bit-identical. Node ORDER cannot change the node set, and the in-loop global lower bound was generalized to a true minimum over live nodes so the two orders share one sound accounting | **panel ran; one certification regression found and acted on** | **Keep as a documented opt-out.** The `=1`/`=0` pair is a three-state override, not a default-OFF gate — unset defers to the caller's `require_incremental`. §1 governed the outcome: a single certification regression was enough to scope the default down. Recorded here because a reader scanning for `LP_SPATIAL_PLUNGE` must find it. |
| `DISCOPT_LP_ROW_LOGICALS` | Every LP row gets its own logical column, equalities included. OFF reproduces the two pre-consolidation layouts exactly. The added columns are **fixed at zero**, so the feasible set and every valid bound are unchanged — but restoring the dual warm start, GMI separation and slack substitution changes which bounds the search proves, and with them the node counts | bound-changing *in effect*, and the docstring says exactly why | **Panel owed.** A consolidation blocked behind a panel; the two legacy layouts it replaces are a maintenance cost paid every day the panel does not run. |
| `DISCOPT_LP_COLD_DUAL_START` | Since `lp_warm_deadline` is itself default-OFF, **no deadline reaches the LP on the default path**, so every cold node LP takes the Rust cold primal loop — the loop whose own comment reads "can otherwise grind toward `max_iter` and run uninterruptibly for minutes". A/B on QPLIB relaxation LPs with one variable changed and no deadline on either arm (`scratchpad/qplib_run/coldstart_ab.py`) | **measured A/B with the LP optimum as a control** | **Panel owed.** Note the coupling: its value is partly a function of another default-OFF flag, so panelling it in isolation may understate it. That coupling should be stated in the arm. |
| `DISCOPT_LP_ITERATIVE_REFINEMENT` | #671. On the ill-conditioned hda-class McCormick relaxations, re-solve RHS-regularized neighbours and keep the tightest NS safe bound — evaluated against the **original** `b`, never `b+tau`, so it is valid for any recovered dual. Reported bound is the `max` over the sweep and candidate A's drifted-dual bound (#662), so never looser and never unsound. Fires only on the numerical-failure path, never the hot per-node engine, and uses **no external solver**. **hda root dual bound −1.80e10 → ≈ −6.47e4** (the true root McCormick value) | measured; `issue-671-gsw-iterative-refinement-2026-07-18.md` | **Panel owed — high priority.** A bound that is wrong by six orders of magnitude on a known class is a certificate-quality problem (§1), and the fix fires only where the default path has already failed. |
| `DISCOPT_SPARSE_LARGE_LP` | The `_MAX_RELAX_DENSE_CELLS` guard's "would force a multi-GB dense allocation" premise is **obsolete** — the whole per-node path is sparse now, and qap's 85756-row McCormick relaxation solves in ~0.1 s at <1 GB. Sound: only ever adds a bound, never loosens a node | mechanism measured; benefit absent by construction | **Keep as a documented opt-out.** Its own docstring names what would change that and it is not a panel: "pending a benchmark instance that **measurably benefits** — qap's indefinite-QP McCormick bound is ~0". The gate is waiting on an instance, not on an arm. That is state 3, and the row exists so the next reader does not mistake it for a stalled graduation. |
| `DISCOPT_ANYTIME_ROOT_BUILD` | #694. On large sparse network-design/QAP/graph-partition MINLPs the fallback's dual bound comes from a single **uninterruptible** McCormick-LP build (`sonet23v4`: 16.8 s, not bounded by the solve's `time_limit`), so `solve(time_limit=2)` still took **24.5 s**. Sound by construction: fewer rows is still a valid outer approximation, so truncation can only weaken, never falsify. Entry experiment: a finite bound exists by **8–45 % of build** on every tested structure | measured on the real class; `issue694-anytime-build-entry-2026-07-17.md` | **Panel owed.** A solve that overruns its own `time_limit` by 12× is a contract violation, not a tuning question. |
| `DISCOPT_HESS_COMPILE_GATE` | #966. Rare severe overruns (200–500 s past a 20 s budget, in **both** `LP_WARM_DEADLINE` arms) were caught in flight with `faulthandler.dump_traceback_later`: the phase is the uninterruptible first-time XLA compile of the colored-HVP Lagrangian-Hessian kernel — `heatexch_gen3` @ 20 s, XLA's own alarm reporting **124 s** of compile, run wall 162.5 s. An eager (`jax.disable_jit`) fallback was **falsified** as the fix: 8–10 s per call steady-state. A compile cannot be interrupted, so entry refusal is the whole mechanism | measured; alternative falsified and recorded | **Panel owed.** The §10 instrument (`dump_traceback_later` on a call that may not return) is exactly the tool CLAUDE.md prescribes, and it worked. |
| `DISCOPT_DETERMINISTIC` | #912/#1116. `kriging_peaks-full200` at `max_nodes=1`, same process, same binary, no user time pressure: root dual bounds **−25371.8 / −28852.0 / −28072.6, a 14 % swing**, with the incumbent bit-identical. Neutralizing the role-2 wall budgets made the same solve reproduce exactly (`-1044.819…` twice at `max_nodes=1`, `-754.478794470719` twice at `max_nodes=300`) and **tighter than every wall-bounded run**. `_work_budget.py` calls the underlying issue "a correctness-of-process bug, not a performance detail" | **measured; reproducibility restored AND the bound improved** | **Arguably OUT OF SCOPE, and §5's net-positive bar is the wrong gate for it.** A dual bound that swings 14 % between runs of the same binary on the same model is a §1 certificate-reproducibility problem, not a performance feature. But the decisive point is how the flag is *consumed*, which is §5's own classification rule: this is an **alias for a user-selected mode** that is already first-class public API (`Model.solve(deterministic=True)`, `modeling/core.py:7021`). "Graduating" it to default-ON would neutralize every role-2 wall budget for every user, making the default solve slower by construction — so the net-positive bar cannot be the right test. Its docstring is the best in the audit: it states the guarantee, what the guarantee deliberately does *not* cover, and why. Recommend reclassifying it out of scope alongside the other mode/selector reads rather than leaving it as a stalled graduation. The one caveat is that its *mechanism* is bound-changing, so any future change to the role-2 carve set still owes a differential check. |
| `DISCOPT_NORM_ATOM` | #632 adjacent-atom family. The factorable path relaxes the outer `sqrt` of a sum of squares as a **concave** atom — the wrong curvature — collapsing the underestimator to a loose floor. Emits the convex OA `‖t‖ >= a·t` for unit `a` (Cauchy–Schwarz) plus axis facets; sound for **any** direction. Byte-identical when off | sound by construction; prototype scope stated | **Panel owed (family).** |
| `DISCOPT_LOGSUMEXP_ATOM` | Same family. Measured motivating **root gap ~2.7**. Supporting-hyperplane (softmax-gradient) tangent cuts; any reference `t0` gives a sound global underestimator | sound by construction; motivation measured | **Panel owed (family).** |
| `DISCOPT_XEXP_ATOM` | Same family. `t·exp(t)` is convex on `t >= -2`; the factorable path shatters it into a bilinear — measured **root gap ~2.3** on an interior-min box. Falls through to the sound product path on `lo < -2` (box spans the inflection) or an overflow-prone `hi` | sound by construction; motivation measured; abstains outside its envelope | **Panel owed (family).** |
| `DISCOPT_ENTROPY_ATOM` | Same family. McCormick on the decoupled box allows `x=x_ub` with `log=log(x_lb)` simultaneously — a floor far below the true convex minimum — and the generic interval-Hessian certifier cannot recover the factored form's convexity (the dependency problem). Emits the **exact** convex hull via the shared 1-D emitter | sound by construction | **Panel owed (family).** |
| `DISCOPT_RELENT_ATOM` | Same family. `x·log(x/y)` is jointly convex on `x,y>0` (the perspective of `x log x`) but is relaxed as a bilinear against a concave `log(x/y)`. Emits joint tangent planes; sound for any `(x0,y0)` with `x0,y0>0` | sound by construction | **Panel owed (family).** |

**On the five atoms.** They are one mechanism applied to five expressions and should be
panelled as **one arm**, not five. Each is sound by construction (a tangent of a convex
function is a global underestimator), each is byte-identical when off, and each carries a
measured root-gap motivation on its target class. What none has is a corpus panel — and
the in-repo 61-file corpus may well contain no instance that exercises `relent` or
`logsumexp` at all, which is a reason to draw instances from the MINLPLib snapshot per
CLAUDE.md's corpus note rather than to conclude the atoms are inert.

## What #1421 changes about the rule, and what it does not

It does not change §5. It repairs the instrument that was supposed to enforce it, and the
repair's finding is that the enforcement had been reporting on 11 % of its subject.

Two things are worth carrying forward:

1. **An enumeration of read forms is a measurement and decays like one.** The retraction
   listed four forms after an `ast` census; a fifth (module-constant indirection) was
   hiding a flag that already had a row in this document. The scan now asserts it visited
   the tree it claims to scan and prints an executed-classification count, and the test
   pins one flag per form so a narrowing fails loudly (§6).
1a. **"Has a row" must mean a row, not a mention.** The enforcement test originally
   accepted any occurrence of the flag name anywhere in this document. Two gates passed
   it on prose alone — `DISCOPT_OBBT_ITERATE`, named in a paragraph *about* a verdict
   nobody had acted on, and `DISCOPT_RLT`, named only as "the whole `DISCOPT_RLT`
   family" — and both turned out to carry decisive recorded verdicts. Counting a mention
   is the same class of error as reading a docstring: it measures the flag being spoken
   about, not triaged. The test now requires a table row, and that tightening was checked
   against the previous commit of this file, where it fails on exactly those two.
2. **Most of these gates are not stalled graduations — they are unowned ones.** The
   common shape is a sound mechanism with a real measurement on its target class and no
   arm in `GRADUATION_ARMS`. The fix is not more triage; it is arms. Four rows above are
   certificate-quality or contract issues rather than performance features
   (`NARROW_BOX_BRANCH`, `LP_ITERATIVE_REFINEMENT`, `DETERMINISTIC`,
   `ANYTIME_ROOT_BUILD`) and belong ahead of the bound-tightening candidates under §1.
   The `GRADUATION_ARMS` gap was re-verified flag by flag across all 45 during the
   correction pass: **not one of them has an arm**, so "run the panel" is not an action
   anyone can take today for any gate in this document — it requires writing an arm
   entry first, every time. That is the single highest-leverage thing this audit points
   at, and it is a code change, not a documentation one.
3. **Nine of the rows were wrong on the first pass, and the count is the finding.** The
   retraction above is not a footnote: it is nine of thirty-eight rows, and eight of the
   nine were written from a docstring. Two correction rules fall out, and they pull in
   opposite directions, so both are needed. *Never triage from a docstring* (#1388) —
   six gates had a decisive verdict in `docs/dev` their docstring never mentions. *And
   never conclude "unmeasured" from a `docs/dev` grep* — three gates
   (`ROOT_PROBE_SEEDS_FALLBACK`, `LP_COLD_DUAL_START`, `SPARSE_LARGE_LP`) carry their
   entire panel in the docstring and nothing in `docs/dev`. Grep the **mechanism**, not
   only the flag name: `SINGULAR_TANGENT`'s name appears nowhere in `docs/dev`, and its
   panel was found only by searching for "vertical tangent".
4. **Three docstrings in the tree are actively misleading and should be fixed
   regardless of what §5 state their flag lands in.** `solver.py:565` tells the reader
   `OBBT_TOPK` awaits a panel "on consecutive nightlies" — the panel ran and killed it
   2026-07-11, and consecutive-nightly graduation was dropped 2026-07-17, so the comment
   is false twice over. `solver_tuning.py:1347` documents `HEUR_ENTRY_SHARE` as *entry
   refusal* when the shipped flag caps a stage clock and the refusal mechanism was
   measured and removed (`solver.py:13934`). `_relax/uniform_relax.py:1188` still says
   the caller "falls back to JAX"; it falls through to the tape. A docstring that
   describes a mechanism the tree does not ship is worse than a missing one — this
   audit's own `HEUR_ENTRY_SHARE` row was wrong *because* it trusted that docstring.
