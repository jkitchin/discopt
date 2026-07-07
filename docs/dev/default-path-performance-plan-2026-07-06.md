# Default-path performance plan — closing the SCIP/BARON gap systematically (2026-07-06)

**Status:** planned (no task started; G1's first graduation — the ILS-cap
held-out validation — is in flight)
**Executor:** fresh Claude (Opus) sessions, one task per iteration, following
the entry-experiment → kill-criterion → build → verify → CI-green → merge
discipline. Everything needed is in this doc + the cited evidence docs +
CLAUDE.md.
**What this plan is:** the synthesis of the 2026-07-05/06 performance
campaign — why the shipped improvements moved the benchmark less than
expected, the three measured systemic causes, and the prioritized work that
actually closes the gap to SCIP/BARON on the default path.

---

## §0 Binding contract (additions to CLAUDE.md — read that first)

0.1 **Default-path wins or explicit graduations only.** The campaign's core
    lesson: a capability that ships default-OFF and never graduates is inert
    for every user. Each task below either (a) improves the *default* solve
    path under the bound-neutral / heuristic-policy regimes, or (b) explicitly
    graduates a parked flag to default-ON through the G1 validation gate. No
    new default-OFF flags without a named graduation plan.

0.2 **Verify it fires.** No task is done until instrumentation proves the
    change executes on the target instances AND the wall/certs move there
    (the R2-didn't-fire lesson: `run_root_fixpoint` shipped sound but
    executes 0× on the panel by default and is structurally unreachable on
    m3/fac2 even forced on — `coverage_map.py` exists to check exactly this).

0.3 **Profile with the sampler, not cProfile tottime.** cProfile inflates
    high-call-count cheap functions (the NBT walker read as the #1 sink at
    ~0% of real wall). `discopt_benchmarks/scripts/python_residual_profile.py`
    (200 Hz leaf-frame sampler) is the wall-attribution instrument of record.

0.4 **Do not relitigate the campaign's falsifications (§4).** Measurement
    beats plan; these were each killed by direct experiment.

0.5 **Regimes per CLAUDE.md §5.** Heuristic-policy changes (most of this
    plan): certified objectives unchanged, `incorrect_count = 0`,
    incumbent-quality-at-TL not degraded, no instance >10 % slower.
    Bound-neutral: byte-identical node_count + objective. The measuring
    instruments: `check_cert_neutrality.py`, `generality_sweep.py` (held-out,
    excludes panel + named probes), `global_opt_baron_vs_discopt.py`.

---

## §1 The accounting — what shipped vs what reached the user (all measured)

The 2026-07-05/06 campaign vs the 2026-07-05 baseline (61-instance BARON
head-to-head, 60 s):

| arc | measured win | reaches the default user? |
|---|---|---|
| A1–A3 (bound reporting), F1 (LNS budget), F2 (simplex stall guard), F4 (root budget/time-limit contract), THRU-2b (pure-LP short-circuit) | **23.1 → 18.8 min, 40 → 42 certs, 0 violations, budget contract restored** | **YES — default-on. The only arc that moved the benchmark.** |
| R2 branch-and-reduce (#524: root fixpoint + `reduce_node`) | ex1224 53→5 nodes; held-out benefit 31 % | NO — `DISCOPT_ROOT_FIXPOINT`/`DISCOPT_NODE_REDUCE` default-OFF; fires **0×** on the panel by default; structurally unreachable on m3/fac2 even forced ON |
| THRU-2a PSD cost gate (#526) | nvs24 root 56.7→6.7 s (converges at all); nvs17 1.6× | NO — default-OFF, and the QCQP class (nvs17/19/24) is not in the 61-panel |
| R4 zero-spanning lift (#518), TD-A squared-log lift (#521) | st_e36 certifies; nvs09 root gap 69→27 % | NO — default-OFF; structures rare in-panel |
| VOLUME-1 ILS cap (#530) | nvs06 888→31 solves, 3.54→1.39 s (2.5×); nvs08 3.1× | NOT YET — default-OFF; the G1 held-out validation to flip it is in flight |

Flags-ON panel re-measure: **42→42, 18.8→18.9 min** — the parked capabilities
are panel-invisible (wrong structures) or unreachable (path-gated). The
held-out generality pilot (N=20) showed they are class-real (31 %/30 %
benefit, 0 soundness violations), i.e. the panel's zero was an instrument
artifact — but "class-real and parked" still means **no default user sees
them**.

## §2 The three measured causes of the remaining SCIP/BARON gap

**C-A. No effort governor — uncalibrated primal artillery at the root.**
Root phase is **60–97 % of wall** on the easy class. nvs06: **911 POUNCE NLP
solves for a 5-node problem**, 888 from `integer_local_search` — a heuristic
with a **0 % incumbent hit rate on every instance tested, including its own
docstring example (nvs23)**. The incumbent always comes from multistart
start #1. m3/fac2's variant is `rens` (a nested B&B as a root heuristic).
QCQP's variant is PSD separation (60 % of wall, 0 % of the bound). SCIP
tracks per-heuristic success and throttles losers; BARON scales effort to
difficulty. discopt fires everything, unconditionally, on every instance.
Eliminated alternatives (each by direct measurement): Python orchestration
overhead is **~2 % removable** (PYFIX-1 #529 — the 24–31 % "bridge" share is
genuine JAX kernel compute under Python frames); cut strength is a **NO-GO**
(CUT-1 #522 — discopt's root bound already ≥ SCIP's fully-cut root; SCIP's
own aggregation-c-MIR cuts injected into discopt's LP close 0–1.8 %);
release POUNCE ≈ 1.1× Ipopt (pounce#182).

**C-B. No flag-graduation pipeline.** The bound-changing regime (correctly)
parks wins behind default-OFF flags "pending 3 green nightlies" — but no
nightly pipeline exists, so **nothing ever graduates**. Five validated
capabilities are inert on main. Every future win will suffer the same fate
until the pipeline exists. This is a process gap that silently caps all
engineering ROI.

**C-C. Fragmented solve paths.** convex fast path / MILP driver / spatial
`_solve_nlp_bb` (LP-relaxer) / alphaBB / AMP. Capabilities get wired into one
path: R2 lives in the LP-relaxer spatial block (m3/fac2 route around it;
nvs05 routes to alphaBB where every flag is inert). BARON has one
branch-and-reduce loop every instance flows through. Any per-path improvement
under-delivers by construction.

**And the hard tail is research, not engineering.** nvs05/nvs09/tls2/
tanksize/hda + the Class-H flowsheets are **bounding-limited**: products of
zero-spanning factors and general-expr `log` rows have **no finite
underestimator** under the McCormick machinery (F5 #509, TD-B #520 —
structural proofs, not tuning failures). No heuristic work reaches them.

---

## §3 Task list (priority = measured leverage × readiness)

### G1 — Graduate the parked flags (cheap, highest certainty; FIRST)

- **Evidence:** §1 table — five sound, class-validated capabilities inert on
  main. Held-out pilot: branch-and-reduce 31 % benefit / 8 % regression, PSD
  gate 30 % / 10 %, 0 soundness violations (N=20; `generality_sweep.py`).
- **Sub-tasks, in order:**
  1. **G1.1 ILS cap → default-ON** (in flight): the held-out validation
     (~60–80 integer instances, cap-off vs cap-on) flips
     `DISCOPT_ILS_SOLVE_CAP` default if **0 incumbents lost**; keep `=0` as
     the escape hatch. Expected: integer class 2.5–3× by default.
  2. **G1.2 stand up the nightly graduation gate** — **DONE** (see
     `docs/dev/flag-graduation-protocol.md`). `generality_sweep.py` now has
     per-flag `ARMS` (each parked flag isolated, everything else default-OFF —
     the isolation the N=20 pilot lacked) + `run_arm`/`arm_stats`;
     `graduation_gate.py` is the wrapper: per flag it runs the held-out arm +
     the cert-panel neutrality check (fresh subprocess with the flag ON) +
     `incorrect_count = 0`/no-oracle-cross, emits a machine-readable verdict
     (`{flag, eligible, benefit_fraction, regression_rate, soundness_ok,
     cert_neutral, notes}`), and appends it to
     `docs/dev/data/graduation-ledger.jsonl` so 3-consecutive-green is
     checkable. Corpus honesty (§4): the full held-out gate runs locally on a
     corpus machine (`make graduation-gate` / documented cron); GitHub CI runs
     only the cert-neutrality + `incorrect_count` subset over the vendored 61
     panel (`.github/workflows/graduation-gate.yml`, `--ci-subset`). Three
     consecutive green → a flag is *eligible*; flips happen in reviewed PRs.
     Without this, stop shipping default-OFF flags at all.
  3. **G1.3 PSD gate → default-ON** after its single-flag arm is clean
     (its 10 % regression rate must be re-measured in isolation — the pilot
     bundled it with R2; nvs13 19→49-node regression is the known cost).
  4. **G1.4 R2 → default-ON** after its arm is clean (8 % regression known);
     note G3 widens its reach first or in parallel.
  5. **G1.5 R4 + TD-A → default-ON** (cheap: structures are rare; the lift
     is inert when absent — validated sound where present).
- **Regime:** each flip is heuristic-policy/bound-changing per its original
  PR; the flip PR carries the held-out table + cert-panel proof.
- **Kill:** any lost incumbent / soundness violation in an arm → that flag
  stays OFF with the instance recorded (not a plan failure — the gate doing
  its job).

### G2 — The effort governor: hit-rate-adaptive root-heuristic scheduling (the biggest un-built lever)

- **Evidence:** C-A. Root = 60–97 % of easy-instance wall; the volume is
  demonstrably non-load-bearing (0 % hit rates; incumbent from start #1).
  VOLUME-1's source-attribution instrumentation
  (`docs/dev/nlp-solve-volume-2026-07-06.md`) already tags every NLP solve
  by call site — the governor's sensor exists.
- **Hypothesis:** an adaptive scheduler — cheap heuristics first; each
  heuristic tracked by (solves, incumbents found, wall); a source that fails
  k consecutive times is throttled/disabled for the rest of the solve;
  expensive sources (RENS nested-B&B, ILS sweeps, deep diving) run only while
  the primal-dual gap is open AND their class hit-rate is nonzero — cuts the
  easy-class root phase by most of its 60–97 % share without losing
  incumbents, because the incumbents demonstrably come from the cheap first
  stage.
- **Entry experiment:** replay the panel + a held-out slice with the VOLUME-1
  counters ON; compute per-source (hit-rate, wall-share) pooled across
  instances. If ≥2 sources beyond ILS have ~0 % pooled hit rate and material
  wall share, the governor generalizes (build). **Kill:** if every non-ILS
  source has a material hit rate, the governor reduces to G1.1 (already
  done) — record and close.
- **Build sketch:** a small `HeuristicGovernor` in the root/heuristic
  dispatch (solver.py) holding per-source stats; policy constants (k
  failures, gap threshold) documented, global, not per-instance; every
  heuristic call site routes its budget request through it. Heuristic-policy
  regime; default-ON only via the G1.2 gate.
- **Acceptance:** easy-class (m3, fac2, nvs06/08/13, ex1224…) geomean wall
  materially down (target ≥2× on the class; BARON-parity ambition is
  sub-second on the smallest); zero certified-objective changes; zero lost
  incumbents panel-wide + held-out; time-limited instances keep same-or-
  better objective-at-TL.

### G3 — Unify path coverage (makes G1/G2 apply universally)

- **Evidence:** C-C; `coverage_map.py` output (R2 fires 0× by default;
  m3/fac2 structurally unreachable; nvs05 → alphaBB inert).
- **Scope:** route every spatial path through one reduce/heuristic-governor
  seam: (a) expose node-LP marginals on the paths that lack them (the T2.4a
  pattern), (b) call `reduce_node`/root-fixpoint (flag-gated) from the
  MILP-driver and alphaBB paths, (c) the governor (G2) wraps heuristics on
  all paths, not just `_solve_nlp_bb`.
- **Entry experiment:** the coverage map over the 61-panel + a held-out
  slice: for each instance, which path, which capabilities reachable. The
  build order = the paths carrying the most wall.
- **Regime:** wiring is bound-changing where it enables reductions on new
  paths → flags + differential; the plumbing itself bound-neutral.
- **Kill (per path):** if a path's instances are all fast already or the
  reductions are inert there (nvs13 pattern: fires, tightens nothing), skip
  that path — record it.

### G4 — Bounding strength for the hard tail (research-grade; the long pole)

- **Evidence:** the ~15-instance uncertified tail is bounding-limited
  (uncertified-tail plan + results docs); F5/TD-B structural proofs (no
  finite underestimator for zero-spanning `log`/products); T2.1-revisit
  (reduction alone median 7.4 % on the tail); R3b (branching is not the
  lever); TD-A shows the *shape* of a win (lift-before-distribute,
  nvs09 69→27 %).
- **Direction (each needs its own entry experiment; kill criteria per the
  house style):** (a) extend the TD-A lift-before-distribute pattern to more
  composite families; (b) alphaBB-as-primary on structures McCormick cannot
  bound (rigorous-α exists; the question is routing + cost); (c) the
  branch-and-reduce loop *on the reformulated models the tree actually
  branches* (the R3b finding — fingerprints were measured on the wrong
  model); (d) for the no-bound flowsheet class (hda/heatexch): a bound-only
  fallback relaxation for rows the MILP relaxer drops (#517's re-scope).
- **Honest framing:** months, not weeks; this is where BARON's remaining
  certs live; schedule after G1/G2 bank the easy-class wins.

### G5 — Fixed-tax trim (small, last)

- ~0.5 s import/JIT on every first solve (profile §1.7; F7's opt-in eager
  imports + the JAX compile cache exist). Matters only once G2 makes the
  easy class sub-2 s; then flip the cache/eager path default-ON via G1.2.

### V — the measurement of record after each G-milestone

Re-run `global_opt_baron_vs_discopt.py` (60 s, 61 instances) after G1 and
after G2 vs the 2026-07-06 baselines
(`reports/global_opt_baron_vs_discopt_2026-07-06T03-25-20.json` flag-off,
`…T12-36-20.json` flags-on), and `generality_sweep.py --n 100` held-out.
Hard gates every time: 0 violations, no cert lost, time-limit contract,
bound-vs-oracle clean. Report defaults-only numbers — that is what a user
gets.

---

## §4 Do-not-relitigate (this campaign's falsifications — all by direct experiment)

- **Python orchestration overhead** — ~2 % removable on m3/fac2 (PYFIX-1
  #529; the sampler's 24–31 % "bridge" is genuine JAX kernel compute; the
  NBT walker's huge call counts are a cProfile artifact, ~0 % of wall).
- **Cut strength (aggregation/c-MIR)** — NO-GO (CUT-1 #522): discopt's root
  bound is already ≥ SCIP's fully-cut root on the integer-product family;
  SCIP's own cuts injected close 0–1.8 %. The separator already exists
  (`lp/aggregation.rs`) and self-disables.
- **POUNCE per-solve speed** — ≈1.1× Ipopt on release builds
  (pounce#182 resolved; the 10× was a debug-build artifact).
- **NLP warm-starting across the tree** — iteration-neutral-to-worse
  (F6 #510).
- **Responsiveness-aware branching** — worse on the real reformed models
  (R3b #516).
- **Composite even-power envelopes** — already exact (F5 #509).
- **`log(<expr>)`-argument lifting for the no-bound class** — arguments span
  zero; no finite underestimator; reform gated off on 722-var models
  (TD-B #520).
- **Panel-only measurement** — the 61-panel lacks the QCQP/reduce classes;
  use `generality_sweep.py` held-out for generality claims.

## §5 Effort summary

G1.1 in flight; G1.2 ~2–4 d; G1.3–G1.5 ~1–2 d each after their arms.
G2 entry ~2 d, build ~1–2 EW. G3 map ~1 d, wiring ~1–2 EW. G4 open-ended
(research; each direction gets its own entry-experiment gate). G5 ~1 d.
Expected shape after G1+G2: the easy class approaches BARON's order of
magnitude by default; the tail waits on G4.
