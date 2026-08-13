# #1013 — the dual pivot path's marginal stability, measured on the in-repo corpus

Companion to `docs/dev/performance-plan.md` §17. Everything here is reproducible
from this directory; the harnesses are the ones that produced the numbers.

## Environment caveat, stated first

The instance the issue is written around (`QPLIB_0911`) is **not reachable from
this environment**: the MINLPLib/QPLIB snapshot under `~/Dropbox` is absent and
`qplib.zib.de` is blocked by the network policy (`CONNECT` → 403). The captured
LPs `scratchpad/i1008/lps/*.npz` the issue points at were never committed either.
So the issue's headline cell (1279 pivots → `iter_limit`, and the 10.5x recovery)
could **not** be re-measured, and nothing below claims to have reproduced it.

What is measured instead is the *class*, on a panel built from the vendored
corpora: 102 root-relaxation LPs captured from all 9 in-repo QPLIB instances and
all 68 in-repo MINLPLib `.nl` instances, each with `rlt_lineq` off and on
(`capture.py`); 100 of them have a sign-matched dual-feasible slack start and are
solvable through the warm dual path (`panel.py`, `lprun.py`).

## Harnesses

| script | what it does |
|---|---|
| `capture.py` | captures the 102 root-relaxation LPs to `lps/*.npz` |
| `lprun.py` | solves one captured LP through `solve_lp_warm_csc_py`, prints one `RES` JSON line with status/objective/iterations + LP-stats counters |
| `panel.py` | runs the panel under one or more env arms, interleaved within each rep, in child processes |
| `trace_report.py` | summarizes the per-pivot `DUALTRACE` stream (pivot magnitude, step lengths, degenerate-run histogram) |
| `pinf_report.py` | the same stream read for *progress* — **superseded, and wrong**: it reconstructed the progress test in Python without the engine's tolerance filter (§4a) |
| `noprog_report.py` | the corrected version: reads the engine's own run/no-progress counters out of the trace |
| `report_modes.py` | arm-vs-arm comparison: status changes, objective drift, wall/iteration medians |
| `perturb.py` | the marginal-stability probe: random column permutations (a rounding-level perturbation needing no flag) |
| `oracle.py` | independent verdict on a captured LP from SciPy's HiGHS |
| `verify_3814.py` | feasibility check of a returned point against the LP data |
| `farkas_exact.py` | re-evaluates a returned Farkas ray in exact rational arithmetic |
| `make_fixture.py` | writes a captured LP as a Rust test fixture in the `parse_stall_fixture` format |

Every script prints an executed-count and exits non-zero at zero (CLAUDE.md §6);
none catches exceptions around the thing under test (§7).

## 1. The two existing escapes are inert on the whole corpus

`DISCOPT_PROFILE=1` over all 100 LPs (`runstats.jsonl`):

- `DualStallTrips` = **0 on every LP** — including `QPLIB_3815_rlt1`, which is
  98.7 % degenerate and exhausts a 20 s budget at 10 752 pivots. The counter can
  only fire when the F2 *size-derived cap* is reached, and a cap of
  `20·(m+n)+500` is ~10⁶ pivots on these LPs.
- `DualBlandActivations` = **0 on every LP**. Bland engages at `2·(n+1)`
  consecutive degenerate pivots — 58 194 on that same LP.

So the issue's point 3 generalizes past `QPLIB_0911`: *neither* anti-degeneracy
mechanism is reachable on this problem class, and the instrument that was
supposed to report the stall reports 0.

## 2. What the stall actually is — not the hypothesized tiny pivots

The issue hypothesizes that the dual ratio test picks numerically tiny pivot
elements under degeneracy, lengthening the eta chain and giving zero-length
steps. The per-pivot trace (`DUALTRACE`, `trace_report.py`) on the corpus's
worst cell (`QPLIB_3815_rlt1`, 8192 pivots) says otherwise:

| quantity | value |
|---|---|
| degenerate pivots | 98.6 % |
| chosen \|pivot\| | min 1.1e-2, **median exactly 1.0**, max 4.6e2 |
| pivots with \|pivot\| < 1e-4 | **0** |
| primal step \|t\| | min 6.0e-3, median 1.0, **never 0** |

The degeneracy is *dual*: `d_q ≈ 0`, so the dual objective is flat, while the
primal step is nonzero every time. That is the structural consequence of a
sparse objective on a lifted relaxation, not a numerical accident — and it is
why a stability tie-break on `|α_rj|` has nothing to discriminate on this corpus.

## 3. The dual Harris pass, re-tested scoped to the stall — falsified again

Implemented exactly as #1008 describes (among breakpoints at or before the
bound-flipping walk's own stopping index, within a 1e-7 ratio slack, take the
largest `|α_rj|`), but **armed only inside a degeneracy stall** (after 32
consecutive degenerate pivots, disarmed after 32 productive ones) — the scoping
the issue asks for. Panel of 100 LPs, arm vs `off`:

| metric | value |
|---|---|
| fires on | 22 / 100 LPs |
| status changes | 2 — `QPLIB_3814_rlt1` `infeasible`→`optimal`, **`tspn10_rlt1` `optimal`→`iter_limit`** |
| wall speedup | median **0.995x** (0.416x – 1.651x) |
| iteration delta | median **0**, changed on 17 / 95 |
| objective drift (optimal/optimal) | ≤ 5.8e-14 |

Same verdict as #1008 on a different corpus, and the same *shape* of failure: a
status regression somewhere else. Scoping it to the stall did not rescue it.
Removed rather than shipped default-off (CLAUDE.md §3).

Bland-at-a-reachable-threshold was measured in the same panel as a third arm and
is also falsified: it regresses `tspn10_rlt1` `optimal`→`iter_limit` too, fires
on 8 LPs, median 1.007x.

## 4. What does work: hand a stalled warm loop to the cold solve

The shipped mechanism is: **`STALL_PATIENCE` consecutive degenerate pivots →
return `None`**, i.e. the caller's cold two-phase primal solves the LP. That is
the action every other difficulty in this loop already takes, and the cold path
self-verifies its own verdict, so the bail cannot make a solve wrong — only
slower or faster.

### 4a. A second progress measure was tried, and it discriminates nothing (retraction)

The first design tested progress in *both* of the loop's measures — the dual
objective (flat on a degenerate pivot) *and* the total primal infeasibility —
bailing only when neither moved for 512 pivots. `pinf_report.py` appeared to show
that this separates the classes where run length alone does not:
`QPLIB_3871_rlt0` runs 663 consecutive degenerate pivots but (said the probe)
improved `pinf` every ≤24 pivots, so it would be left alone.

**That probe was wrong, and the claim is withdrawn (CLAUDE.md §11).** It summed
raw bound violations, while the engine's `select_leaving` only counts violations
above `tol`; sub-tolerance noise therefore read as progress. Measured with the
engine's own counter instead (`noprog_report.py`, reading the `DUALTRACE`
stream), the primal-infeasibility term **never once** reset the counter that the
degeneracy test had not already reset — on any of the LPs traced,
`max_noprog == max_run` exactly. It added no discriminating power for its cost
(a changed `select_leaving` signature and an extra branch), so it was removed
rather than shipped as an untested branch (§3).

The panel run with the two-measure test at patience 512 confirms it from the
other end: it still bailed on `QPLIB_3871_rlt0`/`rlt1` and still cost 2.7x there
(0.04 s → 0.12 s) — the very cell it was introduced to protect.

### 4b. The threshold, and the two populations it separates

Longest degenerate run per LP, measured by the engine (`noprog_report.py`,
`runstats.jsonl`):

| LP | longest run | outcome of the warm loop |
|---|---:|---|
| `QPLIB_3852_rlt0` | 19 | converges, 0.05 s |
| `st_testgr3_rlt0` | 119 | converges |
| `QPLIB_3815_rlt0` | 122 | converges |
| `tspn05_rlt1` | 190 | converges |
| `cvxnonsep_psig30_rlt0` | 248 | converges |
| `tspn10_rlt0` | 341 | converges |
| `tspn12_rlt0` | 620 | converges, 0.15 s |
| `QPLIB_3871_rlt0/1` | 663 | converges, 0.04 s |
| `tspn08_rlt1` | **902** | converges, 1.4 s — the highest converging run |
| `tspn12_rlt1` | **1274** | `iter_limit` |
| `QPLIB_3814_rlt1` | 3352 | `infeasible` (contradicted by HiGHS — §5) |
| `tspn10_rlt1` | 5512 | cold fallback anyway, 15–17 s |
| `QPLIB_3815_rlt1` | 6864 | `iter_limit` |

The two populations do not overlap: converging ≤ 902, non-converging ≥ 1274.
`STALL_PATIENCE = 2048` sits 2.3x above the highest converging run and below
every non-converging one. It is a **corpus-derived constant** and is documented
as one; `DISCOPT_LP_DUAL_STALL_BAIL=<n>` overrides it (and `=0` disables the
bail) so a future panel can re-derive it without a rebuild.

### 4c. The tree-level panel, and the flag-parsing footgun it caught

`tree_panel.py` solves every vendored `.nl` with a recorded optimum in
`known_optima.toml` (16 instances, 60 s each) under both arms and asserts the
CLAUDE.md §5 cert-clean bar: no bound past its reference optimum, no incumbent
beyond it, no certification or status regression. Result: **96 assertions, 0
issues**, and objective, bound and node count **bit-identical on all 16** — the
bail never fires in these trees (longest degenerate runs 47–87 pivots).

The first run of that panel reported `nvs12` at 2.8 s → 47.8 s and `nvs11` at
0.7 s → 6.6 s, i.e. a 17x tree regression. It was **the harness, not the
mechanism**: the "on" arm set `DISCOPT_LP_DUAL_STALL_BAIL=1`, and the flag then
parsed that literally as *a patience of one pivot* — bailing on the first
degenerate pivot of every node LP. `DualDegenerateStallBails` = 16 with
`DualDegenerateRunMax` = 1 is what gave it away: a bail with no run behind it is
impossible under the intended semantics (CLAUDE.md §8 — verify what you actually
loaded/ran, and read the counters back).

The flag now reads `1`/`true`/`on` as "enabled at the default patience" and only
takes integers ≥ 2 as an explicit patience. With that fixed, `nvs11` and `nvs12`
are **identical in nodes and in every LP counter** across arms (the bail never
fires in either tree: longest degenerate runs are 47 and 87).

## 5. The `QPLIB_3814_rlt1` verdict, which is why this is not only a perf issue

On the default path this LP returns **`infeasible`** after 5543 pivots (3352 of
them a single degenerate run). Independent checks:

- SciPy/HiGHS on the same matrix: `optimal`, obj 0.238394628159.
- The perturbed arms of our own engine: `optimal`, obj 0.238394628195.
- The returned point satisfies every row and bound to 2e-8 (`verify_3814.py`).
- An elastic LP (`min t` s.t. `Ax − t ≤ b`) returns `t* = 0.0`.
- The engine's own certificate (`FarkasRejectMargin` = 1, the other sign
  certifying) turns on `bᵀy = 3.0e-8` against a term-magnitude
  `Σ|b_i y_i| = 600` — a relative margin of 5e-11 — while the
  Neumaier–Shcherbina margin it is tested against, `1e-9·(1+|bᵀy|+Σ|boxmax|)`,
  is built from the *result* magnitudes and is 1e-9.

So a terminal verdict on this LP flips with the pivot path, and the deciding
quantity is 11 orders of magnitude below the data it is computed from. The
no-progress bail replaces that verdict with the cold path's `optimal` (0.39 s vs
1.51 s). **It does not fix the certificate.** Whether the NS margin should
account for accumulation magnitude (`Σ|b_i y_i|`) rather than result magnitude is
a separate defect, filed separately — this issue's mechanism only stops the
stalled warm loop from being the thing that decides it.
