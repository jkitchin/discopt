# CUT-INHERIT-GRAD — cut-pool inheritance graduation: predicate found, but the
# default-ON flip is KILLED by a flag-path false-optimal (nvs22). Stays opt-in. (2026-07-08)

**Status: NOT graduated to default-ON. `DISCOPT_CUT_INHERIT` / `SolverTuning.cut_inherit`
stays OPT-IN (default force-off).** The task asked to graduate root-cut-pool
inheritance to a *structure-gated default-ON* that ships the dense-QP 2–5× where
it pays and is byte-identical elsewhere. Two findings reshaped that:

1. **The structural premise was falsified (CLAUDE.md §4 — the measurement wins).**
   THRU-4-graduate (#552) concluded the 2–5× is *specific to the dense
   integer-QP class* and a broad flip is *throughput-neutral (1.004×)*. Under
   clean serial measurement that 1.004× does **not reproduce** — it was a TL=30s /
   parallel-contention artifact. Every pool-firing instance benefits (IQCQP,
   QCP, QCQP, MBNLP alike: 1.6–9.7×). So there is **no neutral firing class to
   gate out**; a class-keyed subset gate would wrongly withhold real wins. The
   honest predicate is the simplest one: **pool-fires ⇒ ON** (inherit iff a
   non-empty root pool is separated).

2. **But the flag path has a false-optimal that blocks the flip (CLAUDE.md §1).**
   The broad validation surfaced a **new deterministic false certificate**:
   `nvs22` (MINLP, pure-integer nonconvex, an adversarial-suite soundness probe)
   certifies `optimal 33.55166` against the oracle optimum **6.0582** — flag-ON,
   both force-on and structure-gated. This is the nvs06-class incumbent-search
   reroute that C-42 (#553) only *partially* fixed: C-42's pool-drop-retry fires
   only when the pool-augmented node solve *fails numerically*; on nvs22 the pool
   solve **succeeds**, but the pre-tree incumbent pump is still rerouted, the
   region holding 6.0582 is fathomed, and a wrong optimum is certified. A false
   certificate is a hard regression — the default cannot flip while it exists.

**What ships (this PR).** The tri-state `cut_inherit` machinery + the pool-fires
structure gate, wired and instrumented, **default force-off** (byte-identical to
`origin/main`). The gate is reachable as an **opt-in** (`DISCOPT_CUT_INHERIT=gated`
/ `cut_inherit=None`), the throughput win is documented, and the nvs22 blocker is
pinned by a strict-xfail regression test so the flip can be re-attempted the
moment the reroute is fixed.

> **Method.** Apple M-series (arm64), Python 3.12, release build
> (`maturin develop --release`; pounce `_pounce.abi3.so` = 4.73 MB, not debug),
> `JAX_PLATFORMS=cpu JAX_ENABLE_X64=1`, corpus
> `~/Dropbox/projects/discopt-minlp-benchmark/`, oracle `minlplib.solu`.
> **Throughput A/B pairs were run SERIALLY** (one solve at a time) — the entry
> experiment's first pass was run 15-way parallel and its node/throughput numbers
> were corrupted by contention; the serial re-run is the reference below and is
> what falsifies #552's "neutral".

---

## Part 1 — the structural predicate (entry experiment)

### 1a. The named candidate feature does NOT separate (falsified)

The task's leading candidate was the *fraction of the root/node wall the square+PSD
separation loops consume*. Implemented as a cheap root-time feature
(`pool/root_sqpsd_frac`: the square+PSD wall of the ONE root pool separation solve
÷ that solve's wall, isolable because no node solves have run yet). Measured
flag-ON on the labeled set (WIN = the #551/#552 win probes; NEUTRAL = the #552
"neutral" firing slice):

| instance | label (#552) | root_sqpsd_frac | pool |
|---|---|---:|---:|
| nvs23 | WIN | 0.941 | 82 |
| nvs24 | WIN | 0.857 | 30 |
| nvs19 | WIN | 0.702 | 70 |
| nvs17 | WIN | 0.590 | 59 |
| kall_circles_c6b | "neutral" | 0.912 | 31 |
| kall_circles_c6c | "neutral" | 0.860 | 30 |
| kall_circlespolygons_c1p12 | "neutral" | 0.859 | 122 |
| knp3-12 | "neutral" | 0.516 | 64 |
| ringpack_10_2 | "neutral" | 0.399 | 69 |
| spring | control | 0.052 | 6 |

**No threshold separates them.** The "neutral" kall instances (0.86–0.91) sit
*above* WIN nvs17/19 (0.59/0.70). Pool size overlaps completely (kall 122 > nvs 30).
`pure_discrete` also fails (nvs17/19 pure=1 WIN; nvs13/23 pure=1 neutral; kall
pure=0). **Kill criterion for "the named feature": triggered.**

### 1b. Why nothing separates — the #552 "neutral" is a measurement artifact

Re-labeling by the *ground truth* (serial ON/OFF node-throughput ratio at TL=60s)
shows the "neutral" label itself was wrong:

| instance | probtype | OFF nodes | ON nodes | OFF n/s | ON n/s | **ratio** | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| nvs24 | IQCQP | 9 | 49 | 0.14 | 0.79 | **5.46×** | WIN |
| nvs23 | IQCQP | 69 | 125 | 1.14 | 2.08 | **1.82×** | WIN |
| nvs17 | IQCQP | 173 | 225 | 7.39 | 12.85 | **1.74×** | WIN |
| nvs19 | IQCQP | 215 | 295 | 3.57 | 6.09 | **1.70×** | WIN (+cert) |
| kall_circles_c6c | QCP | 55 | 143 | 0.91 | 2.38 | **2.63×** | WIN |
| kall_circles_c6b | QCP | 61 | 111 | 1.01 | 1.84 | **1.82×** | WIN |
| knp3-12 | QCP | 31 | 287 | 0.49 | 4.77 | **9.72×** | WIN |
| dispatch | QCQP | 3 | 15 | 6.63 | 21.51 | **3.25×** | WIN |
| tspn05 | MBNLP | 39 | 119 | 2.07 | 2.47 | 1.19× (+cert) | WIN |

Every pool-firing instance is a WIN, across IQCQP / QCP / QCQP / MBNLP. #552's
knp3-12 "127→127 identical" reproduces its OFF=ON only at TL=30s under
contention; serially it is OFF 31 → ON 287 nodes (~4–9×). Soundness of the wins
verified: objectives identical (knp3-12 to 8 digits, kall_* bit-identical) or
ON strictly better (nvs19 −1098.2 → −1098.4 = oracle); every bound ≤ incumbent
and ≤ oracle.

### 1c. The predicate that IS correct: pool-fires

The one root-time signal that is present-and-correct is **whether a non-empty
root cut pool is separated** (`_root_cut_pool is not None`). It is the cheapest
possible predicate (already computed), keys on measured structure not on
name/shape (CLAUDE.md §2), and:
- **fires ⇒ broadly beneficial** (§1b), and
- **does not fire ⇒ byte-identical** (nothing to inherit or skip; §3b).

## Part 2 — the structure gate (built, default force-off)

`cut_inherit` is now **tri-state** (`Optional[bool]`):
`True` force-on · `False` force-off · `None` structure-gated. Env
`DISCOPT_CUT_INHERIT`: unset/`0` ⇒ force-off (**shipped default**),
`gated`/`auto` ⇒ `None`, any other non-`0` (e.g. `1`) ⇒ force-on. The gate is the
pre-existing pool-population guard unified with the tri-state: capture the pool
optimistically at the root when *not* forced off, and engage the per-node
square/PSD skip iff the pool actually populated. Instrumented: `pool/gate_mode`
(1/0/−1) and `pool/gate_decision` (1 iff inheritance engaged), plus the diagnostic
`pool/root_sqpsd_frac`.

Fire-proof: `pool/gate_decision=1` on nvs24 under `=gated`; `=0` on the default
(force-off) path and on a pool-empty model.

## Part 3 — verification

### 3a. On-class win retained under the gate (gated vs OFF, serial, TL 60s)

| instance | OFF | gated | nodes/s ratio | gate | cert |
|---|---|---|---:|---:|---|
| nvs24 | feasible, 9 n | feasible, 49 n | **5.40×** | ON | bound identical −1035.66 |
| nvs19 | feasible, 217 n | **optimal, 295 n** | 1.66× | ON | **gains cert** (−1098.4 = opt) |
| nvs17 | optimal, 173 n | optimal, 225 n | 1.74× | ON | preserved |
| nvs23 | feasible, 69 n | feasible, 109 n | 1.58× | ON | obj identical |

### 3b. Off-class byte-identical (gated vs force-off, serial) — 0 wrongful fires

On 12 non-firing instances that terminate deterministically, `gated` is
**byte-identical** to force-off (node_count + objective exactly equal,
`gate_decision=0` on all): cvxnonsep_{normcon20,nsig20,psig20}, enpro56pb,
ravempb, waterund01, gabriel02, sssd18-08persp, pointpack12, alan, gbd,
st_miqp1. **12/12 IDENTICAL, 0 wrongful fires.** (The broad TL-bound held-out
slice shows node-count wobble ON-vs-OFF on time-limited rows — that is wall-clock
nondeterminism at the deadline, not a code path difference: those rows do not
fire the pool, so gated≡force-off by construction.)

### 3c. Bound-changing soundness on the on-class

Differential: on every on-class arm the dual bound stays ≤ its incumbent and ≤
the oracle (min sense); the root bound is identical on/off (nvs19/24
−1104.24/−1035.66). Feasible-point sampling (existing
`test_root_pool_cuts_valid_on_every_child_feasible_point`, exhaustive over the
dense-QP box): every pooled row holds at every feasible lifted point → no
inherited cut removes a feasible point. **0 soundness violations.**

### 3d. THE BLOCKER — nvs22 flag-path false-optimal

`nvs22` (MINLP), TL 25s, oracle 6.0582:

| arm | status | objective | bound | nodes |
|---|---|---:|---:|---:|
| default (force-off) | optimal | **6.0582** (correct) | 6.0582 | 35 |
| force-on | optimal | **33.55166** (FALSE) | 33.55166 | 11 |
| structure-gated | optimal | **33.55166** (FALSE) | 33.55166 | 11 |

Deterministic (persists at TL=60s). `pool/dropped_nodes` is *not* set — C-42's
retry never triggers because the pool solve **succeeds**; the failure is the
pre-tree incumbent-search reroute (nvs06-class, flagged as an unfixed follow-up
in `c42-cut-inherit-fix-2026-07-07.md` §"Follow-ups": *driver
sentinel-on-deliberate-skip*). This is a **false certificate** → CLAUDE.md §1
KILL for default-ON. It was invisible to #551's probes and #552's held-out draw
(neither contained nvs22); the adversarial soundness suite catches it.

### 3e. cert-neutrality (default force-off vs committed baseline)

`check_cert_neutrality.py`, flag unset (force-off), 41-instance panel:
**41/41 byte-identical, `|Δobj|=0.00e+00` on every row, exit 0.** dispatch 3→3,
nvs13 49→49, tspn05 39→39 — the default path is byte-identical to `origin/main`,
so **no rebaseline** (the gate does not fire on the shipped default; the
"justified rebaseline" clause is moot). Cleaner than #551/#552, which carried the
nvs13 19→49 main-drift flag; that drift is absent here (the committed baseline was
since refreshed to 49).

## Part 4 — flip decision

**Not flipped.** Parts 1–3 are not clean (Part 3d). Per the graduation protocol's
kill criterion and CLAUDE.md §1, a flag with a deterministic false-optimal stays
opt-in. `solver_tuning.py` default is force-off; `=1` / `=0` / `=gated` overrides
all work and are tested.

## Follow-ups (recorded, not shipped — the path back to default-ON)

1. **Fix the nvs22 / nvs06-class reroute (the blocker).** Root-cause why the
   flag-ON pre-tree incumbent pump is rerouted when the cold-path pool solve
   *succeeds* (the feasibility pump → NLP-relaxation pump → integer local search
   chain is bypassed, seeding a worse region). This is the same driver-layer
   sentinel-on-deliberate-skip hazard `c42-cut-inherit-fix` deferred to the #396
   backlog; it is now a *soundness* bug (false certificate), not merely a
   degraded incumbent. File on #396.
2. Once (1) is fixed, re-run this CUT-INHERIT-GRAD validation. The pool-fires
   gate is already built and validated broadly beneficial; only the soundness
   blocker stands between it and the flip.

## Gates

| gate | result |
|---|---|
| `pytest -m smoke` (python/tests) | **627 passed, 14 skipped** (incl. 6 CUT-INHERIT-GRAD tests: gate-fires, off-class byte-identical, tri-state env precedence, force-off no-skip) |
| `pytest -m slow test_adversarial_recent_fixes.py` | **12 passed** (nvs22 sound on the default path) |
| `pytest -m slow test_c42_cut_inherit_coldpath.py` | **2 passed + 1 xfailed** (the nvs22 blocker, strict-xfail — flips the suite red the moment the reroute is fixed) |
| `check_cert_neutrality.py` (default force-off, 41-panel) | **41/41 byte-identical**, `|Δobj|=0`, exit 0 — no rebaseline |
| `ruff check` / `ruff format --check` (v0.14.6) | clean |
| pre-commit `mypy` (whole `python/discopt/`, v2.1.0) | clean on the changed files (`solver_tuning.py`, `solver.py`) |
| `cargo test -p discopt-core` | n/a — no Rust touched |
