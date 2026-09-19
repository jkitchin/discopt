# Default-OFF flag audit (CLAUDE.md §5 retirement rule)

Standing record of every `DISCOPT_*` gate over **solver math** that defaults OFF, and
which of §5's three states it is in: **graduated**, **retired**, or **kept as a
documented opt-out**. A gate in none of them is a defect.

First taken 2026-09-19 against `0ae8cad` (issue #1345). A new default-OFF gate over
solver math adds its row here in the same PR that introduces the flag.

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
| `DISCOPT_CONVEX_KERNEL` | default-OFF, never default-ON; 3,105 lines of Rust behind it | panel owed | **Decide via #1346** (already split out): graduate or delete. |
| `DISCOPT_NLP_NATIVE` | *"Default stays OFF on the remaining grounds — the speedup …"* | measured, reasoned | **Keep as documented opt-out** — it already states why it is not the default. Confirm the docstring also says what would change that. |
| `DISCOPT_CMIR_AGGREGATION` | *"ships dark behind this flag until proven on nightlies"* | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_COEF_TIGHTEN` | *"default-OFF until a corpus-wide differential panel graduates it"* (#282) | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_SGO` | *"a default-OFF env flag until a differential panel graduates it"* | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_GP_MINLP` | *"a corpus-wide differential panel graduates it"*; `solver="gp-minlp"` is an always-available explicit opt-in | panel owed, but has an explicit alternative entry point | **Keep as documented opt-out** if the explicit `solver=` route is the intended interface — then the env flag is redundant and should go; otherwise run the panel. |
| `DISCOPT_OA_INFEASIBLE_NOGOOD` | *"changes the master's dual bound (CLAUDE.md §5 regime 2) and ships behind a flag until a corpus panel clears"* | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_PRESOLVE_SUBSTITUTE` | *"bound-changing work ships behind a flag until a differential panel passes"* | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_PSD_QFORM` | *"can prove more constraints/objectives convex, which changes node relaxations and counts — hence it ships behind a flag"* | panel owed, never run | **Run the panel or retire.** |
| `DISCOPT_G_CONVEX_CUTS` | gate function is one line; rationale is thin | **status unrecorded** | **Record a status first.** A gate whose docstring does not say what it is waiting for cannot be triaged; that absence is the defect. |
| `DISCOPT_DIRECT_HEURISTIC` | substantial measurement, incl. `docs/dev/direct-entry-2026-08-12.md`; explicitly *heuristic-policy, not bound-changing* | measured, regime stated | **Read the recorded panel and decide.** The evidence exists; the verdict was never written down. |
| `DISCOPT_IPX_CHEAP_FIRST` | *"the gate only ever decides which of two correct routes runs"*; measurement language present | measured, soundness-neutral | **Read the recorded panel and decide.** |

## What this audit does not do

It does **not** delete anything. Per #1345 the deletions are follow-on PRs; this lands the
rule and the verdicts. `DISCOPT_LP_SPATIAL_MIXED` (#1357) and `DISCOPT_POUNCE_DECLARED_BOX` (#1358) have since
been **retired** — the rule's first two applications, and the proof it terminates.
`DISCOPT_CONVEX_KERNEL` → #1346 remains. No verdict above is left as "recorded and
forgotten", which is the state #1345 exists to end.

**What retirement kept, and why it is not deletion-by-name.** #1357 removed the env flag
and both production call sites, so the default path uses the pre-#860 gate. It did *not*
remove the `mixed=` parameter on `_is_in_scope` / `solve_lp_spatial_bb`: that capability
is sound, separately tested (four cases in `test_lp_spatial_bb.py`), and is what §5 means
by "keep reusable pieces, regression fixtures and benchmark cases; the entry point goes".
The graduation measurement moved onto `_is_in_scope`'s docstring rather than dying with
the flag, so anyone weighing the widening again starts from the evidence.

It also does not re-run any panel. Every verdict above rests on what the tree already
records — which is the point: the evidence was never the missing part.
