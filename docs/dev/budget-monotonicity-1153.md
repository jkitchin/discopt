# #1153 — incumbent quality must be non-decreasing in `time_limit`

Status: **diagnosed; the whole wall-budget fix family is ruled out. #1153 stays
open.**

Six candidate mechanisms were built and measured across four decision panels.
Five were falsified, inert, or superseded and were deleted; the survivor
(`DISCOPT_HEUR_ENTRY_SHARE`, in its **stage-cap** form) is cert-clean but does
not pay for itself, and ships default OFF.

**The headline is §6.6c's family verdict, not any individual arm.** Refusing the
feasibility pump buys node throughput (20 instance-rungs up, 0 down) and costs
incumbents; capping it keeps the incumbents (`nvs05` goes from *losing* its
incumbent to improving it) and hands the throughput back (5 up). Both effects
have the same cause — how much pump runs — so **no wall-budget policy separates
the pump's cost from its value**; every setting trades along that one line. That
retires §6.1, §6.4, §6.5 and all four panels of §6.6 as instances of one family,
and leaves exactly one direction: make the pump *cheaper for the same value* via
#912's `WorkBudget` (§6.7).

Read §6.7 first if you are picking this up. Also note two framing facts: the
issue's stated gate **already passes** on this corpus (198 comparisons, one
violation), and the reported `nvs19` reproduction **does not reproduce here** at
any budget. What does reproduce is the *throughput* half, diagnosed to a single
heuristic call.

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

### 3.2 Bounded finder STAGE (`DISCOPT_HEUR_ENTRY_SHARE`, default OFF)

**The flag caps; it does not refuse.** That is the opposite of what it did in
earlier revisions of this PR, and the change is the result of §6.6c rather than a
preference: refusing costs `nvs05` its incumbent outright, because a productive
pump never runs at all.

`_root_heur_nlp_entry_ok` — the shared gate for thirteen root-heuristic entries —
is left on the **legacy rule**. What the flag changes is the *deadline the finder
stage runs under*: `_heur_stage_deadline()` bounds it to
:data:`HEURISTIC_ENTRY_SHARE` (25 %) of what is left, returning the global
`_deadline` unchanged when the share is 1.0 (flag off, byte-identical).

Two details are load-bearing and both were bugs first (§6.6c):

* **A finder stage is two NLP consumers**, a multistart seed that produces the
  rounding point (3.27 s measured) and the pump itself (3.11 s). Bounding only
  the pump caps the cheaper half and measures inert — which is exactly what §6.4
  concluded, wrongly. Every consumer reads the stage deadline now, including the
  sibling pump on the `_solve_nlp_bb` route.
* **The deadline is taken ONCE per stage**, not recomputed per call. Recomputing
  `now + share * (deadline - now)` at each site is Zeno's bound: per-call grants
  fell 3.0 s -> 1.75 s while total stage wall did not move (7.08 s -> 7.47 s),
  because the stage simply ran more, cheaper rounds.

**Soundness.** Every affected call is a primal heuristic, so truncating one
changes which incumbent is found and when, never the dual bound or the
certificate (§0.3 heuristic-policy). `DISCOPT_ROOT_BUDGET_GATE=0` still disables
the whole entry gate independently of this flag.

This is the measured cause (§6.2); the flag is sound and incumbent-positive but
**does not pay for itself** (§6.6c), so it ships OFF.

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

**Read the earlier numbers in this section's history with care.** The container
this work ran on restarted mid-investigation and the BASELINE moved with it:
`tspn10` at 5 s was 7/7 nodes under the legacy arm before the restart and 3/3
after; `tspn12` at 40 s went 15 -> 63. Any comparison spanning that boundary
attributes an environment change to the code. Everything below is a single
three-arm run in ONE process on ONE container, arms interleaved within each
repetition, 54 runs — `scratchpad/i1153/three_arm.log`.

Node counts, two repetitions, at the rung where the pump is admitted:

| @10 s | legacy | flat share | success-weighted |
|---|---|---|---|
| `heatexch_gen2` | 3, 3 | **7, 7** | 3, 3 |
| `tspn10` | 3, 3 | **31, 31** | 3, 3 |
| `tspn12` | 3, 3 | 5, 5 | 3, 3 |

And at 5 s the flat arm also removes a bimodality: `heatexch_gen2` legacy is
`[7, 3]` (sd 2.0) and flat is `[7, 7]` (sd 0); `tspn10` legacy `[3, 3]`, flat
`[7, 7]`.

**The cost, and it is real.** `tspn12` at 10 s: legacy reaches `262.647` in both
repetitions, flat reaches `282.244` in both. Reproducible, not noise. The pump
there is productive and the flat share refuses it.

**One apparent second regression is noise, and per-rep data is what shows it.**
`heatexch_gen2` at 20 s reads `legacy 808843.77 / flat None` in the summary — but
the legacy arm returned `None` in rep 0 and `808843.77` in rep 1. Bimodal. A
summary that prints only the last repetition would have recorded a regression
that is not there; this is why the probe prints every repetition.

So the flat share buys node throughput at 5-10 s and costs one reproducible
incumbent. Whether that trade is net-positive is a corpus question, not a
three-instance one — §6.6.

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

### 6.5 Success weighting on the finder role, measured and falsified

The obvious repair for `tspn12` is to discriminate on OUTCOME rather than cost:
allow the first attempt, then shrink the admitted share once for every fruitless
one — `base ** (calls - found)`, the *improver* role's success weighting
(`_improver_allowed`'s `3 * (found + 1) / (calls + 1)`) carried across to the
finder role. It was implemented, made unit-testable as a pure function, and
measured in the same three-arm run above.

**It is inert on all nine cells** — identical to legacy everywhere, including the
two the flat share fixes. The reason is visible once stated: the harm is done by
the *first* pump, and a rule whose whole point is to admit the first attempt
cannot prevent it. Refusing only the *second* attempt saves wall but not enough
to change a node count.

The code was therefore removed rather than shipped. An inert flag is a dead flag
(CLAUDE.md §3), and keeping it would have implied a fix that measurement says is
not there. What remains of the attempt is this record and the module-level
refactor it forced, which is what made the three-arm probe possible at all.

### 6.6 Decision panels: shared gate vs finder-scoped

17 instances (the subset still OPEN at the largest phase-A rung, selected by that
measured property rather than by name), ladder 5/10/20/40 s, both arms
interleaved in ONE process, 136 solves each.

**Panel A** (`panel4.log`) applied the share inside `_root_heur_nlp_entry_ok`
unconditionally — which, as review found, is the shared gate for *thirteen* root
heuristic entries, several of them improver-role and already bounded by
`_improver_allowed`'s success-weighted contingent. **Panel B** (`panel5.log`)
applies it only where the caller passes `finder=True`: the two feasibility-pump
entries.

| | A: shared gate | B: finder-scoped |
|---|---|---|
| monotonicity violations | 0 / 0 (both arms) | 0 / 0 (both arms) |
| incumbent better / worse / same | 1 / **3** / 64 | **2 / 2** / 64 |
| node count flat higher / lower | 17 / **4** | **20 / 0** |
| certification regressions | none | none |

Scoping is a clear improvement and confirms the review's diagnosis: it removes
one regression (`nvs05` at 20 s, `1107.89 -> 1269.7`, gone), adds a gain
(`heatexch_gen2` at 40 s, none -> `808844`), and **eliminates every node-count
decrease** — throughput now improves strictly wherever it moves at all, 20 up and
0 down, across the whole `bchoco*` / `heatexch_gen*` / `beuster` / `4stufen`
family that sits at 3 nodes under the legacy rule at every budget.

Two incumbent regressions survive, both from the finder pump itself:
`nvs05` at 10 s (`1269.7` -> none, lost outright) and `tspn12` at 10 s
(`262.647` -> `282.244`).

**Verdict on B: still not graduating, but the reason narrowed.** Throughput is
now strictly better and the incumbent column is a wash (2/2) rather than a net
loss — yet losing an incumbent outright on `nvs05` is a user-visible harm, and
the estimator §6.7 names is the standing suspect for it.

### 6.6b The estimator, and panel C

Review identified the mechanism: `_mean_heur_nlp_cost()` returns a running **max**
that never decays — its own comment says "the *max* (not mean) is deliberate",
recording ~15 s overruns — so dividing it by the share makes the admission test
*"remaining > 4x the worst case ever seen"*, not "4x typical". One expensive early
NLP then refuses every later root heuristic for the rest of the solve at any
`time_limit` under 60 s, which is exactly the shape of `nvs05` losing its
incumbent at 10 s while gaining nodes.

The share now divides a running **mean** (`_typical_heur_nlp_cost`), while the
legacy overrun guard keeps the max — it answers a different question ("could
another solve of the worst size still fit?") where the max is right. Flag off is
unchanged: share 1.0 selects the max, i.e. the legacy rule byte-identically.

**Panel C measured it, and the ordering probe falsified it.**

| | B: finder-scoped, max | C: finder-scoped, mean |
|---|---|---|
| monotonicity violations | 0 / 0 | 0 / 0 |
| incumbent better / worse / same | **2 / 2** / 64 | 1 / 2 / 65 |
| node count flat higher / lower | **20 / 0** | 18 / 0 |
| certification regressions | none | none |

C is no better than B — `nvs05` at 10 s and `tspn12` at 10 s still regress, and C
loses B's `heatexch_gen2`@40 s gain. Within noise, but certainly not the recovery
the hypothesis predicted.

The *mechanism* probe (`scratchpad/i1153/ordering_probe.py`) explains why, and is
the decisive measurement rather than the panel. Review's own caution was that a
panel improving `nvs05` would not by itself clear the estimator, because the
max-never-decays cliff bites only when an expensive NLP is observed **before** the
finder entry decision. So the probe records the interleaved sequence of
(NLP observed) and (finder decision) events directly:

```
nvs05  @10 s  first finder decision at index 0, NLPs observed before it: 0
tspn12 @10 s  first finder decision at index 0, NLPs observed before it: 0
```

On **both** regressing instances, in **both** arms, the finder decision fires
before any heuristic NLP has been observed. At that moment
`max == mean == the 2.0 s default`, so the two estimators are numerically
**identical** at the only decision that matters, and the estimator cannot be the
mechanism here whatever the panel shows.

The change was therefore reverted rather than kept: it shipped on a hypothesis
this probe falsifies, and its own panel does not support it (CLAUDE.md §4). What
survives is the **rename** — `_mean_heur_nlp_cost` -> `_worst_heur_nlp_cost` —
which was always the causally interesting half: a function returning a max, named
"mean", and consumed as one at a division site is how the share came to divide a
worst-case-ever by a fraction and read as "4x typical".

**The real lever at these budgets is the 2.0 s DEFAULT SEED, not the statistic.**
With nothing observed, the gate reduces to "is more than ``2.0 / share`` seconds
left?" on any model, independent of the pump's measured 6.4 s cost. That is a
work-budget question (#912's `WorkBudget` would make the cost knowable in advance
rather than guessed), not an estimator one — and it is the single most promising
open item, now backed by a direct measurement rather than by argument.

### 6.6c Panel D — capping the stage instead of refusing it, and the family verdict

§6.4 concluded that bounding the pump's *clock* was inert. **That conclusion was
an artifact of an incomplete change**, and finding out why is what made this panel
worth running. A finder stage is TWO NLP consumers — a multistart seed producing
the rounding point (3.27 s measured) and the pump itself (3.11 s) — and §6.4
bounded only the pump, leaving the more expensive seed on the global deadline. It
capped the cheaper half.

A second defect surfaced while fixing that: the stage deadline was recomputed at
each call site as ``now + share * (deadline - now)``, which is Zeno's bound —
every call takes a share of whatever is left *at that moment* and the total is
never bounded. Measured on `heatexch_gen2` at 10 s: per-call grants fell
3.0 s -> 1.75 s while total stage wall did **not** move (7.08 s -> 7.47 s),
because the stage simply ran more, cheaper rounds. `_finder_stage_deadline` is now
taken **once** per stage and shared. Both defects were caught by checking that
the cap *bound* before panelling — the step §6.4 skipped.

With the stage genuinely bounded (panel D), against the refuse arm (panel B):

| | B: refuse entry | D: cap the stage |
|---|---|---|
| monotonicity violations | 0 / 0 | 0 / 0 |
| incumbent better / worse / same | 2 / 2 / 64 | **3 / 2** / 63 |
| node count up / down | **20 / 0** | 5 / 1 |
| certification regressions | none | none |

Capping recovers what refusing destroyed — `nvs05` goes from *losing* its
incumbent to **finding one at 5 s and improving it at 10 s** (`1269.70 ->
1107.89`) — and `tspn12`'s regression disappears. But it gives back nearly all the
throughput: 5 instance-rungs up instead of 20.

**The family verdict, and it is the useful part.** The throughput gain and the
incumbent losses have the *same* cause: not running the pump. Refusing buys nodes
and costs incumbents; capping keeps incumbents and returns the nodes. No setting
of a wall-budget policy separates the pump's cost from its value, because the
policy only ever decides *how much pump* runs, and cost and value both scale with
that. **This rules out the whole "gate or bound the pump by wall budget" family**,
which is the family §6.1, §6.4, §6.5 and §6.6 were all drawn from.

Neither arm graduates: D still loses `tanksize`@5 s outright and degrades
`tspn10`@5 s. The flag ships OFF in its **cap** form, which is the more principled
mechanism (bound, don't refuse) and the one with a net-positive incumbent column.

### 6.7 Status

Neither flag graduates, and **#1153 is not closed.**

What is settled, and is the substance of this work:

* **Incumbent monotonicity — the issue's stated gate — very nearly holds already
  on the in-repo corpus.** 198 comparisons over 66 instances at 5/10/20/40 s
  produced **one** violation (`clay0303hfsg`), and the 17-instance decision panel
  produced none in either arm. The `nvs19` failure the issue reports does not
  reproduce on this machine at all: it returns `-1098.2` stably at 15/30/60/120 s
  (`scratchpad/i1153/nvs19_base.json`). So the gate as written cannot be "made to
  pass" here — it already passes, and passing it is not evidence of a fix.
* **The throughput half DOES reproduce, and is diagnosed to the individual
  call.** A whole family (`bchoco*`, `heatexch_gen*`, `beuster`, `4stufen`,
  `contvar`, `hda`) explores exactly 3 nodes at every budget from 5 s to 240 s.
  The cause is the finder-heuristic entry rule admitting a feasibility pump that
  consumes up to 100 % of the remaining budget and frequently returns nothing.
* **The obvious fix is sound but costs more primal quality than it buys** (§6.6),
  and two other candidate mechanisms were falsified outright (§6.1, §6.4, §6.5).

What remains is now a **single** direction, because §6.6c eliminates the rest.

Budget policy cannot fix this: refusing the pump buys nodes and costs incumbents,
capping it keeps incumbents and returns the nodes, and every intermediate setting
trades along that same line, because the policy only decides *how much pump* runs
while cost and value both scale with that. The pump must become **cheaper for the
same value**, not smaller.

That is the #912 `WorkBudget` direction: bound the pump in deterministic
operations rather than seconds, so its cost is knowable in advance instead of
estimated at a 2.0 s default that is 3x wrong, and so the same work is done on
every machine. It is the one lever left that changes the cost/value ratio rather
than sliding along it.

### 6.8 #1435 — the pump's projection was the wrong projection (fix landed)

§6.7 left exactly one direction: make the pump **cheaper for the same value**
rather than smaller. #1435 is that lever, found by following the one remaining
question §6.7 does not answer — *why does the pump so often return nothing?*

**The mechanism, measured to the call.** The fix-and-solve round pins the
integers and then minimizes the **model objective** over the continuous
variables, and tests **only that solve's terminal iterate** for constraint
feasibility. Feasibility is incidental to what that solve optimizes, so whether
the pump returns anything is decided by where the solve happened to *stop* —
which `_deadline_wall_cap` derives from the caller's remaining wall. Measured on
`heatexch_gen2`'s second (NLP-seeded) root pump at `time_limit=60`, sweeping the
per-solve cap over the same subproblem:

| `max_wall_time` | 0.05 s | 0.1 s | 0.2 s | 0.5 s | 1.0 s | 3.0 s |
|---|---|---|---|---|---|---|
| terminal iterate constraint-feasible | — | **yes** | — | — | — | — |

and equivalently on the deterministic axis, `max_iter` ∈ {10, 25, 50, 75, **100**,
150, 300} is feasible at **100** alone. A feasible point is *on the projection
path* and is discarded because only the endpoint is checked. That is a primal
heuristic whose **outcome** is a function of `time_limit`, which is #1153's gate
directly — and it explains the same family §6.7 names, not one instance.

**A fixed probe iteration count is not the fix** (CLAUDE.md §2): feasibility at
exactly 100 and not at 75, 150 or 300 is a knife-edge tuned to this instance.

**Falsified on the way (CLAUDE.md §4, §11).** The first hypothesis was that the
round-2 projection's `ITERATION_LIMIT` iterate was a feasible point being
discarded unchecked, since `_PUMP_ACCEPT_STATUSES` contained `TIME_LIMIT` but not
`ITERATION_LIMIT` — making the accept set itself budget-dependent. Kill criterion
stated in advance: *if admitting it still returns no incumbent at `time_limit=10`,
the hypothesis dies.* It was implemented and measured: **no incumbent, 3 of 3**,
pump wall unchanged (1.935 s vs 1.910 s). The iterate is genuinely infeasible.
The widening was **reverted**; the comment above `_PUMP_ACCEPT_STATUSES` now
records why it is not there.

**The fix.** `_repair_to_feasible` in `_relax/primal_heuristics.py` adds a
*second* candidate per pump round, from the same pinned subproblem: a min-norm
elastic normal step (`pounce.project_to_feasible`), i.e. the projection a
feasibility pump is supposed to take (Fischetti–Glover–Lodi; Bonami et al. for
the MINLP form) — minimize distance to the rounding subject to the constraints,
not the original objective. It reads **no clock**; its cost is a bounded number
of sparse QPs, so it is the #912 shape §6.7 asks for. It runs **only** when the
objective solve produced no verified point, so nothing the pump finds today is
displaced, and every candidate still passes the same independent
`_is_integer_feasible` + `_check_constraint_feasibility` gate, so soundness is
untouched.

Measured on `heatexch_gen2`'s pump #2, per round: the objective solve is
infeasible on all 5 rounds at ~0.45 s each; the repair is **feasible on 4 of 5**
at ~0.20 s each. pounce's default of 3 outer re-linearizations is not enough
(0 of 5) — the feasible set is curved enough that a tangent step lands outside
it — so the cap is 10, with `tol` set to the verifier's own 1e-6 so it is a work
*cap* rather than a target.

**Instance gate, 3 reps interleaved, load-gated:**

| rung | before (`df2bc458`) | after |
|---|---|---|
| 5 s | none, 3/3 | none, 3/3 |
| 10 s | none, 3/3 | **834176.34**, 3/3 |
| 20 s | none, 3/3, 31 nodes, bound 555767.79 | **834176.34**, 3/3, **63 nodes**, bound **558540.17** |

The incumbent is independently feasibility-verified, and sits above the
reference optimum 635838.85 as a feasible suboptimal point should; the dual
bound stays below the reference dual 583986.91. Note the baseline finds nothing
at *any* rung here — the earlier "2 of 3 at 5 s" reading in probe 1 was
probe-perturbed timing on a bimodal instance, and is retracted (CLAUDE.md §11).
The node count rising 31 → 63 at 20 s is the incumbent finally pruning: the
pump did not get smaller, it got *useful*, which is the distinction §6.7 turns on.

`DISCOPT_PUMP_REPAIR=0` opts out. It is an opt-*out* for a shipped default in
CLAUDE.md §5's "Out of scope" sense — it exists so the default can be A/B'd,
which is how the corpus panel below was run in a single tree — not a graduation
gate, and it takes no row in the flag-retirement audit.

#### 6.8a The corpus panel, the cost it found, and the cap

The instance gate above is a *probe*, not the case for the change (CLAUDE.md §2).
The case is the corpus panel, and the first one **failed its second bar**.

**Uncapped, 118 MINLPLib-snapshot instances at `tl=20 s`, ON vs OFF, interleaved
within each instance** (120 drawn by `random.Random(1435)` from the `MB*`/`MI*`
nonlinear probtypes, excluding the 66 vendored; `johnall` and `saa_2` timed out
of the harness):

- **Bar 1 (cert-clean): PASS.** 0 bounds above a `minlplib.solu` reference
  optimum, 0 certification regressions, every ON incumbent independently
  feasibility-verified.
- **Bar 2 (net-positive): FAIL.** 0 incumbents gained, 0 lost, 0 improved, 0
  worsened — 118 unchanged — and node throughput **moved on 30 instances, 28 of
  them down, median −8.5 %**. Sorted worst first: `multiplants_stg1a` −76.2 %
  (63 → 15 nodes), `autocorr_bern45-45` / `sfacloc2_2_80` / `waterful2` −57.1 %,
  `autocorr_bern25-19` −51.6 %, … `sfacloc1_3_90` −25 %, `o9_ar4_1` −17 %.
  (An earlier draft of this section quoted `sfacloc1_3_90`'s −25 % as the *worst*
  case; it is not, it was simply the first one I looked at. Corrected per
  CLAUDE.md §11 — the true worst is three times larger, which strengthens rather
  than weakens the case for the cap.)

That is §5's *sound but not helpful* verdict, and on its own it kills the change
as written. The mechanism is plain: the repair is cheap when it **succeeds**
(the pump returns that round), and the panel is dominated by the other case — a
rounding that cannot be repaired, where an uncapped repair pays ~0.2 s on every
one of the pump's five rounds, twice per root, and returns nothing.

**The reshape.** Hypothesis: the cost is repeated *failed* repairs, and one
attempt per pump keeps the entire gain, because a rounding whose repair succeeds
succeeds on the **first** round. Kill criterion stated in advance: *if
`heatexch_gen2` loses its incumbent, or the worst node-losers do not recover,
the whole change is reverted.* Entry experiment, both halves:

| instance | uncapped | capped (`_PUMP_REPAIR_ATTEMPTS = 1`) |
|---|---|---|
| `sfacloc1_3_90` | −25 % | **0 %** (223 → 223 nodes) |
| `autocorr_bern25-06` | −10 % | **0 %** (36287 → 36297) |
| `fo9`, `no7_ar2_1` | moved | **0 %** |
| `o9_ar4_1` | −17 % | −5.6 % |
| `o7_ar3_1` | −14 % | −4.8 % |
| `heatexch_gen2` | 834176.34 | **834176.34**, 63 nodes, 2/2 |

**Capped panel, 66 vendored instances at `tl=20 s`:** bar 1 clean, bar 2
**1 incumbent gained, 0 lost, 0 worsened, node count 4 up / 0 down**.

One row flagged and is **not** a regression: `tls2` showed OFF certifying
`optimal` in 14.34 s / 153 nodes in this run while the uncapped run had OFF
*and* ON both at `feasible` / 205 nodes. Three of those four cells are
bit-identical; the cell that moved is **OFF**, which runs the same code in both
runs (with the repair disabled no candidate can come from it). `tls2`'s
completion time sits right on the 20 s limit, so it certifies on a lucky draw —
the same bimodality this document warns about throughout. The ON arm has never
been the cell that varies.

**The honest rate.** The repair can only help an instance that finds no
incumbent at all. That population is **67** across both capped panels (57
external + 10 vendored), and the repair converts **2** of them
(`heatexch_gen2` vendored, `kan_peaks_h1_n2_g24` external). It is adopted
because it is sound, never loses a point, and — capped — costs nothing
measurable, not because it is broadly transformative. Anyone revisiting this
should read it as "a free strict improvement with a low hit rate on this
corpus", which is a different claim from the one §6.8's instance table alone
would support.

**A measurement failure worth recording (CLAUDE.md §8).** The first capped
external panel produced 114 of 120 children failing with `AssertionError: tree
predates the #1435 repair` — because the working tree was switched to `main`
mid-run for unrelated work, and the panel's children import `discopt` from that
tree. The §8 marker assertion is the only reason this was caught: without it the
run would have compared `main` against `main`, returned a flawless null result
(0 gained, 0 lost, 0 node change), and read as the cleanest possible pass.
**Do not run `git checkout` in a tree a panel is measuring**, and keep the
version-marker assertion in every panel child.

**The capped external panel — the measurement the cap exists to satisfy.**
120 instances at `tl=20 s`, 2 harness timeouts (`johnall`, `saa_2`, both timing
out in the uncapped run too), 118 compared.

*Bar 2 (net-positive):* **1 incumbent found where none existed
(`kan_peaks_h1_n2_g24`), 0 lost, 0 worsened, 117 unchanged; node count 12 up /
15 down.* The cost the cap was built to remove is gone: the uncapped run moved
30 instances, 28 of them **down**, median **−8.5 %**, worst **−76.2 %**
(`multiplants_stg1a`, 63 → 15). Capped, 26 move, only **15** down, median of
the moved **−2.1 %**, worst **−57.1 %** (`var_con5`, 7 → 3 — a 4-node
difference on a tiny instance, which is what a percentage does to small
denominators). Over all 118 compared instances the median node change is
**0.0 %** in both runs; what changed is the tail.

*Bar 1 (cert-clean):* 0 bounds above the reference optimum, 0 certification
regressions, 0 lost incumbents. One row flags: `gasnet`'s ON incumbent fails
independent feasibility verification — but **so does its OFF incumbent**, and
the two arms are bit-identical (obj `6999381.553035677`, bound
`864103.864320254`, 3 nodes, `feasible`). Its objective matches the `.solu`
reference to ~1.3e-9 relative; this is a tolerance artifact in the panel's
verifier at 1e7 magnitude, present with the repair disabled, and therefore not
something this change caused. It is noted rather than fixed here.

`kan_peaks_h1_n2_g24` was checked against the false-primal failure mode before
being counted: sense is **minimize** and the oracle is `=opt= -5.1956649970`, so
an incumbent of `0.3344124554665887` is a legal feasible-suboptimal point, and
the bound `-5.195664996854543` sits below it — the certificate invariant holds.
Had the instance been a maximization this row would have been a false primal and
the change dead.

**A second measurement failure, and the re-check it forced (CLAUDE.md §8, §9).**
While that panel was running, a mutation test briefly wrote a sabotaged
`primal_heuristics.py` into the same tree the panel's children import from —
with the constraint check removed from the accept path. Progress at the time was
44/120. **The §8 marker assertion would not have caught this**: the sabotage left
`_repair_to_feasible` present, so every child would have imported it happily and
the panel would have reported whatever the sabotaged code produced. The failure
mode it could produce is precisely an ON incumbent that never passed the
feasibility gate — i.e. a fabricated bar-2 gain.

Instances 34–52 (a band wider than the ~39–44 the timing implies, and containing
both flagged rows) were therefore re-run in both arms in a verified-clean tree
and diffed field-by-field against the recorded panel: 254 field comparisons.
`kan_peaks_h1_n2_g24` and `gasnet` reproduce **bit-for-bit in all four cells**,
and **no incumbent appears or disappears anywhere in the band** — the only
signature the sabotage could have left. The 12 differing fields are bounds (and
one 15th-digit objective) on instances whose node counts also drifted, symmetric
across arms: ordinary wall-limit nondeterminism on a 20 s budget. The panel
stands.

The lesson generalizes past the §8 one above: **do not mutate *or* switch a tree
a panel is measuring** — run mutation tests in a separate worktree, or after the
panel. A version marker defends against the wrong *version*; nothing in the
child defends against the right version being edited underneath it.
