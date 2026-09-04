# #1153 — incumbent quality must be non-decreasing in `time_limit`

Status: **in progress.** Two mechanisms were investigated. The first
(`DISCOPT_BUDGET_SATURATION`) was measured and **falsified** as the cause on this
corpus and stays default OFF. The second (`DISCOPT_HEUR_ENTRY_SHARE`) is the
measured cause, is fixed at the budget where it was measured, and is **not yet
general** — see §6.

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

## 3. The changes

### 3.1 Saturating carves (`DISCOPT_BUDGET_SATURATION`, default OFF)

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

This is the hypothesis §6.1 falsifies. It ships default OFF and is not proposed
for graduation; the inventory and ratchet it came with are what survive.

### 3.2 Bounded finder entry (`DISCOPT_HEUR_ENTRY_SHARE`, default OFF)

`_root_heur_nlp_entry_ok` admits an optional root primal heuristic whenever the
time left can absorb one whole solve of the largest size seen so far — i.e. a
heuristic may consume 100 % of the remainder and still be admitted.
`solver_tuning.heuristic_entry_share()` turns that into a bounded share (25 %),
returning `1.0` — the legacy rule, byte-identical — with the flag off.

The flag additionally hands the two feasibility-pump call sites a stage-scoped
deadline (`_heur_stage_deadline`) instead of `_deadline`, the whole *solve*
deadline. That second half was tried **first, on its own, and measured inert**
(§6.4); it is kept because it is sound, free, and turned up one incumbent the
legacy arm missed, but the entry rule is what carries the effect.

**Soundness.** Every gated call is a primal heuristic, so refusing one changes
which incumbent is found and when, never the dual bound or the certificate
(§0.3 heuristic-policy). `DISCOPT_ROOT_BUDGET_GATE=0` still disables the whole
gate, flag or no flag.

This is the measured cause (§6.2) and it is **not yet a general fix** (§6.3).

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

**66 instances, 198 comparisons, 1 violation** — `clay0303hfsg`, 5 s → 10 s,
incumbent `47287.56` → none. Attributed (`scratchpad/i1153/clay_attrib.log`) to a
recursive RENS sub-solve: the convex route hands off to NLP-BB, whose RENS
heuristic calls `solve_model` again with `0.5 x remaining`; at 0.65 s that
sub-solve finds a feasible point (`UB=47287.56`), at 2.0 s it takes a different
path through elastic feasibility restoration and ends iteration 0 at `UB=1e20`.
Same class one level down, different mechanism from a carve.

The panel also shows the **throughput** half of the defect, which the incumbent
comparison alone hides: **5 of 66 instances explore FEWER nodes as the budget
grows**, all at the 5 s → 10 s step — `casctanks`, `clay0303hfsg`,
`heatexch_gen2`, `tspn10`, `tspn12`.

### 5.2 Graduation panel (flag ON vs OFF)

Ladder 30 / 90 / 240 s — chosen to straddle `ROLE2_SATURATION_S`, because an
arm that never reaches the code under test measures nothing (CLAUDE.md §6). The
panel is the subset still OPEN at the largest phase-A rung, selected by that
measured property rather than by name (CLAUDE.md §2).

17 instances (still open at the largest phase-A rung), both arms, two
concurrent processes under symmetric load, `OMP_NUM_THREADS=1`.

**Monotonicity: 34 comparisons, 0 violations in BOTH arms.** The baseline is
already monotone on this panel at these budgets, so there was nothing here for
the flag to repair.

**Differential at the 240 s rung:**

| | result |
|---|---|
| incumbent better / worse / same | 0 / 0 / **17** |
| certification regressions | none |
| node count ON higher / lower / equal | 1 / 1 / 15 |
| dual bound weaker under ON | 3 (`beuster` 10245.8 -> 10221.6, `casctanks` 6.8014 -> 6.5773, `contvar` 183637 -> 183436) |
| dual bound stronger under ON | 0 |

The ON arm genuinely frees tree time (`casctanks` 145 -> 177 nodes), so the
mechanism is real — but it buys those nodes by truncating root tightening, and
pays for them in the dual bound.

**Verdict: cert-clean, NOT net-positive.** Per CLAUDE.md §5 and the
`DISCOPT_CUT_INHERIT` precedent, a cert-clean but neutral-or-harmful flag stays
OFF with the measurement recorded. It does.

## 6. Falsifications and residuals

### 6.1 The saturation hypothesis is falsified as the cause (CLAUDE.md §4, §11)

§2's hypothesis — that the unsaturated carves are what makes a bigger budget buy
a worse answer — does **not** survive its own panel. Three findings kill it:

1. The graduation panel is monotone in *both* arms (34/34), so on this corpus
   there was no violation at 30/90/240 s for saturation to fix.
2. Capping the carves changed **no incumbent on any of 17 instances** and made
   three dual bounds *weaker*.
3. `ROLE2_SATURATION_S = 150` makes the flag inert below a 150 s budget, so it
   cannot touch the 30 s vs 60 s inversion the issue actually reports. Choosing
   the loosest existing sibling ceiling was defensible on its own terms and
   *wrong for this problem* — it put the fix out of reach of the budgets where
   the harm lives.

What survives is the inventory and the ratchet: 38 carve sites, the ten that do
not saturate recorded with categories, and a test that fails on a new uncapped
one. That is worth keeping regardless of the flag's fate. The flag itself stays
OFF and is not proposed for graduation.

### 6.2 The measured cause: an unbounded finder-heuristic entry

Reproduced with zero spread over three repetitions:

| instance | nodes @5 s | nodes @10 s |
|---|---|---|
| `tspn12` | 5, 5, 5 | 3, 3, 3 |
| `heatexch_gen2` | 7, 7, 7 | 3, 3, 3 |
| `tspn10` | 7, 7, 3 | 3, 3, 3 |

`heatexch_gen2`'s layer profile locates it: `root_time` is 3.5 s of a 5.1 s wall
at the small budget and **the entire 10.1 s** at the large one, with
`pounce_time` 1.35 s -> 5.79 s. The NLP probe names the consumer: **at 5 s the
feasibility pump never starts; at 10 s it starts and its two sub-NLPs spend 6.4 s
of the 10 s budget** (each overrunning its own 3.0 s grant) and return **no
incumbent** — identical `obj=None` and identical bound in both runs.

The admitting rule is `_root_heur_nlp_entry_ok`, which refuses a heuristic only
when the time left cannot absorb *one whole* solve of the largest size seen so
far. A heuristic costing 100 % of the remainder is therefore admitted, and
crossing that threshold costs more than the budget increment that unlocked it.
The repository already applies the right doctrine to the *improver* role
(`_improver_allowed`'s success-weighted, node-proportional contingent); the
*finder* role is exempt, and the exemption is unbounded.

### 6.3 What the entry-share flag does and does not fix

`DISCOPT_HEUR_ENTRY_SHARE` requires a finder to fit 25 % of the remainder.
Two repetitions, arms interleaved within each repetition, zero spread:

| @10 s | nodes OFF -> ON | bound OFF -> ON | incumbent OFF -> ON |
|---|---|---|---|
| `tspn10` | 3 -> **31** | 161.160 -> **165.223** (tighter) | unchanged |
| `heatexch_gen2` | 3 -> **7** | unchanged | unchanged (none) |
| `tspn12` | 3 -> 5 | unchanged | 262.647 -> **282.244 (worse)** |

It fixes the collapse where the pump is wasted and **loses a genuinely better
incumbent where the pump is productive**. And at the 20 s rung the ON arm is
identical to OFF on all three: the gate re-admits the pump, so the cliff has
**moved, not gone**.

Two causes of the residual, both identified and neither yet fixed:

* **The gate's cost estimate is wrong on first use.** `_heur_nlp_cost` seeds at a
  2.0 s default while the pump actually costs 6.4 s, so the first admission is
  made on a 3x underestimate. Self-calibration only starts after that solve.
* **Any entry threshold is a cliff.** Refusing on cost alone cannot distinguish
  the productive pump (`tspn12`) from the wasted one (`tspn10`,
  `heatexch_gen2`). The improver role already solves exactly this with success
  weighting (`3 * (found + 1) / (calls + 1)`); extending that weighting to the
  finder role — allow the first attempt, then charge it against a
  success-weighted contingent — is the next step, and would let a productive
  pump keep running while starving an unproductive one.

### 6.4 The stage-deadline form, measured and falsified on its own

Handing the pump a share-sized deadline instead of `_deadline` is the more
attractive fix on paper — it bounds the cost by construction, needs no cost
estimate, and does not deny a productive pump its win. It does not work:

| @10 s, nodes | OFF | ON (stage deadline only) |
|---|---|---|
| `heatexch_gen2` | 3, 3 | 3, 3 |
| `tspn12` | 3, 3 | 3, 3 |
| `tspn10` | 3, 3 | 3, 3 |

ON equals OFF on 15 of 16 cells across 5/10/20/40 s (the exception is a *gain*:
`heatexch_gen2` at 20 s found `825180.5` where the legacy arm found none). The
reason is visible in the NLP probe: the pump's sub-NLPs **overrun their own
`max_wall_time`** (3.27 s against a 3.0 s grant), and the pump polls its deadline
only between rounds, so a share-sized deadline does not bind on round one. A
deadline is not a bound on a stage whose unit of work already exceeds it.

The entry rule is therefore retained as the mechanism, with the clock cap kept
alongside it.

### 6.5 Status

Neither flag is proposed for default-ON on this evidence, and #1153 is **not
closed**. What is settled: the harm class reproduces here; its cause is
identified and measured to the individual heuristic call; one intervention moves
it and one does not; and the inventory, ratchet and monotonicity panel are in
place so the next attempt starts from measurement rather than from scratch. What
remains is a rule that separates the productive pump from the wasted one — the
improver role's success weighting, `3 * (found + 1) / (calls + 1)`, extended to
the finder role — and a corpus-wide panel of it.
