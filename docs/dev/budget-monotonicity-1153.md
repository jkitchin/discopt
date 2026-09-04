# #1153 — incumbent quality must be non-decreasing in `time_limit`

Status: **implemented behind `DISCOPT_BUDGET_SATURATION`, default OFF pending the
CLAUDE.md §5 graduation panel.**

## 1. The defect

Reported on `nvs19` (known optimum `-1098.4`):

| `time_limit` | incumbent | nodes |
|---|---|---|
| 30 s | -1098.2 | 38 403 |
| 60 s | **-1001.2** | **7 619** |
| ≥ 120 s | stops on a ~100 000-node cap | |

Doubling the budget made the answer *worse* and explored **5x fewer nodes**.
There is no false certificate here — the incumbents are feasible and the status
is not `optimal` — so this is a **completeness** miss plus a **monotonicity**
defect, and the monotonicity half is the general one: a user who doubles
`time_limit` and gets a worse answer has no way to reason about the solver.

## 2. Hypothesis, evidence, kill criterion (CLAUDE.md §4)

**Hypothesis.** The harm is #1116's role-1/role-2 coupling read one step
further. Role 1 (*"when do we stop?"*) is the caller's `time_limit`. Role 2
(*"how much work does this stage do?"*) is a sub-budget carved as a fraction of
it. Carving role 2 out of role 1 is not itself wrong — a stage must not outlive
the solve — but it becomes wrong when the carve never **saturates**. A root
stage whose grant keeps growing with the caller's budget goes on separating
cuts; every subsequent node LP then carries them; the per-node cost therefore
rises with `time_limit`; and the tree the remaining budget can cover shrinks.

**Evidence for the mechanism, in this repo.** The pattern is already documented
at a site that was fixed for exactly this reason: `solver.py`'s big-M/AND
exact-linearisation adoption gate records that on the configuration class "the
reform only pays for its heavier per-node LPs once … a generous budget"
is available, and that at `time_limit=60` "heavier LPs halve node throughput".
That is the same causal chain — a bigger budget buying heavier nodes — arrived
at from the other direction.

**Entry experiment.** Inventory every sub-budget in the package computed by
multiplying or dividing a role-1 budget, and classify each as *saturating*
(bounded above by a constant) or *growing*. If the growing set is empty, the
hypothesis is dead.

**Result (2026-09-04, `scratchpad/i1153/scan3.py`).** 38 carve sites; **10 do
not saturate**, of which six are genuine role-2 work allowances:

| site | carve | ceiling |
|---|---|---|
| `solver.py` root LP probe | `max(0.1·T, 2)` | none |
| `solver.py` root cut pool (PSD) | `max(0.25·T, 5)` | none |
| `solver.py` root cut pool (general) | `max(0.25·T, 5)` | none |
| `solver.py` cumulative per-node OBBT | `0.6·T` | none |
| `solver.py` root fixpoint | `max(0.10·T, 1)` | none |
| `_relax/lp_spatial_bb.py` root OBBT | `remaining / 3` | none |

The two cut-pool sites are the ones the mechanism predicts will hurt most: the
pool they separate is *inherited by every node LP*, so their grant sets the
per-node cost for the whole search. The root LP probe site is the sharpest
evidence that this is an oversight rather than a design: its own comment says it
mirrors the root OBBT grant — and it does, except for that site's 15 s ceiling.

The remaining four growing sites are not role-2 work allowances (a division of
the caller's budget between two consumers, or a product whose other factor is
already bounded); they are recorded with categories in
`python/tests/test_1153_budget_monotonicity.py::KNOWN`.

## 3. The change

`solver_tuning.saturate_role2(seconds, frac)` caps a carve at the value it takes
at a `ROLE2_SATURATION_S = 150 s` role-1 budget, so beyond that point every extra
second the caller grants goes to the *search*.

**150 s is not a new number.** It is the reference the root OBBT grant already
carries — `min(min(max(0.1·T, 2.0), 15.0), remaining)` ceilings at 15 s, i.e. at
`T = 150` — and it is the *loosest* of the ceilings the sibling root stages
carry (0.25-fraction presolve: 30 s / 120 s; 0.2-fraction convexity: 20 s /
100 s; 0.15-fraction NBT: 30 s / 200 s). Taking the loosest is the conservative
direction for a change that moves the default path: it caps the stages written
without a ceiling at the point their already-ceilinged siblings stop, and nowhere
tighter. Below 150 s the flag is inert by construction.

**Soundness.** Every capped site is optional *tightening*. Truncating it yields a
weaker relaxation (fewer cuts, fewer OBBT rounds), never an invalid one, so no
bound can rise above its true value and no certificate can be fabricated. The
flag can only change which valid answer path is walked.

## 4. What is pinned against regression

`python/tests/test_1153_budget_monotonicity.py`:

* the unit contract of `saturate_role2` (saturates on, inert off);
* a **static ratchet** in the `test_912_wall_budget_inventory` idiom — a new
  unsaturated carve fails the build unless it is recorded with a category;
* a count of the six saturated call sites, so quietly dropping a wrapper (which
  the ratchet alone would not catch, because the author would simply re-record
  the line) fails too;
* the behavioural gate: over a panel and a ladder that straddles the saturation
  reference, the incumbent never gets worse.

Every probe reports an executed-comparison count and fails at zero (CLAUDE.md
§6).

## 5. Measurements

### 5.1 Baseline monotonicity panel (flag OFF)

61 in-repo MINLPLib instances, ladder 5 / 10 / 20 / 40 s, one process, serial.
See `scratchpad/i1153/panel_base.log`.

<!-- RESULTS-BASELINE -->

### 5.2 Graduation panel (flag ON vs OFF)

Ladder 30 / 90 / 240 s — chosen to straddle `ROLE2_SATURATION_S`, because an
arm that never reaches the code under test measures nothing (CLAUDE.md §6). The
panel is the subset still OPEN at the largest phase-A rung, selected by that
measured property rather than by name (CLAUDE.md §2).

<!-- RESULTS-GRADUATION -->

## 6. Falsifications and residuals

<!-- RESIDUALS -->
