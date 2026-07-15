#!/usr/bin/env python3
"""Out-of-panel generality sweep (GEN-1) — does a flagged capability help a *class*?

discopt ships three bound-changing capabilities behind default-OFF env flags:

- **branch-and-reduce** — the root branch-and-reduce fixpoint
  (``DISCOPT_ROOT_FIXPOINT``) + the per-node cheap reduction
  (``DISCOPT_NODE_REDUCE``); and
- **PSD cost gate** — the cost-aware gate on the per-node PSD (moment) cut
  separation loop (``DISCOPT_PSD_COST_GATE``).

They were tuned on a handful of named probes (nvs17/19/23/24, st_e36, …). The
V-remeasure showed they are *invisible* on the 61-instance vendored panel
(42→42 proved-optimal, 0 net change) — because that panel does not contain
their target structures. That is not evidence they are inert; it is evidence
the panel is the wrong instrument. Per CLAUDE.md §0.2 ("fix the *class*, not the
instance") the honest question is: **on a held-out sample that DOES contain the
target structures, does each capability help a class, or only the probes it was
tuned on?**

This harness is that instrument. It:

1. Draws a **held-out** stratified sample from the MINLPLib corpus, EXCLUDING
   (a) the 61 vendored panel instances and (b) the named tuning probes, keeping
   only instances discopt can parse and that carry a ``.solu`` oracle.
2. Solves each instance **flags-OFF** and **flags-ON** in isolated subprocesses
   (reusing the ``global_opt_baron_vs_discopt.py`` worker pattern; flags
   propagated via ``env``), recording status/objective/bound/nodes/wall.
3. Reports, per capability and overall:
   - **benefit-fraction** — % of instances flags-ON materially improves
     (nodes ↓, or wall ↓ > 5 %, or a status upgrade feasible→optimal) + geomean
     node/wall ratios;
   - **regression-rate** — % where flags-ON is > 5 % slower or loses a status;
   - **SOUNDNESS (hard gate)** — 0 oracle crossings, 0 false-optimal in EITHER
     config (any violation is a P0, flagged loudly);
   - **structural-prevalence** — how many sampled instances even *have* each
     capability's target structure (the honest reach number, K/N).

It writes a JSON (with the exact instance list, for reproducibility) and a
markdown summary with the per-capability verdict (class win vs probe win vs
inert on the held-out sample).

The harness sets **only env flags** per subprocess; it changes no solver math.

Usage:
    PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      python discopt_benchmarks/scripts/generality_sweep.py \
        --n 30 --seed 0 --time-limit 30 [--corpus <dir>] [--out-dir reports]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[2]
PANEL_DIR = REPO / "python" / "tests" / "data" / "minlplib_nl"
DEFAULT_CORPUS = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark"))

# Named probes the three capabilities were tuned on (root-throughput-entry /
# uncertified-tail-plan-results, 2026-07-06). Held out of the sample so the
# sweep measures reach beyond the tuning set, not the tuning set itself.
TUNING_PROBES = frozenset(
    {
        "nvs17",
        "nvs19",
        "nvs24",
        "nvs23",
        "nvs13",
        "st_e36",
        "nvs09",
        "nvs05",
        "ex1224",
        "wastewater04m2",
    }
)

# The three capabilities' flags. Grouped so per-capability attribution is honest:
# the ON config sets ALL of them (that is the shipped bundle), and each solve is
# tagged with which capabilities' target structure it carries (structural
# prevalence) so a benefit can be attributed to the capability whose structure
# is present.
FLAGS_ON = {
    "DISCOPT_ROOT_FIXPOINT": "1",
    "DISCOPT_NODE_REDUCE": "1",
    "DISCOPT_PSD_COST_GATE": "1",
}
CAPABILITIES = ("branch_reduce", "psd_gate")

# --------------------------------------------------------------------------- #
# Per-flag ARMS (G1.2) — the isolation the N=20 pilot lacked.
#
# The pilot bundled ALL flags together (FLAGS_ON), so a benefit could not be
# attributed to a single capability and its regression rate was contaminated by
# its neighbours (e.g. the nvs13 19->49-node regression is PSD's, but the bundle
# hid it inside branch-and-reduce's numbers). The flag-graduation gate needs each
# capability's benefit-fraction / regression-rate / soundness measured in
# ISOLATION: one arm per parked flag, each setting exactly that capability's env
# flags (everything else at its default OFF), plus an ``off`` control and the
# ``all`` bundle for a bundle-vs-isolated cross-check.
#
# Each arm names:
#   - ``env``          the exact env flags it sets (empty for the OFF control);
#   - ``struct_attr``  which ``Row`` structural-prevalence attribute gates it
#                      (only structure-carrying instances score the arm), or None
#                      for the whole-sample arms (off / all);
#   - ``regime``       ``bound_neutral`` (must be cert byte-identical) or
#                      ``bound_changing`` (may legitimately change nodes/bound; the
#                      differential + oracle-bracket soundness check applies). This
#                      selects which neutrality check the gate runs for the flag.
ARMS: dict[str, dict] = {
    "off": {"env": {}, "struct_attr": None, "regime": "control"},
    "root_fixpoint": {
        "env": {"DISCOPT_ROOT_FIXPOINT": "1"},
        "struct_attr": "reduce_struct",
        "regime": "bound_changing",
    },
    "node_reduce": {
        "env": {"DISCOPT_NODE_REDUCE": "1"},
        "struct_attr": "reduce_struct",
        "regime": "bound_changing",
    },
    "psd_cost_gate": {
        "env": {"DISCOPT_PSD_COST_GATE": "1"},
        "struct_attr": "psd_struct",
        "regime": "bound_changing",
    },
    "lift_zero_spanning": {
        "env": {"DISCOPT_LIFT_ZERO_SPANNING_FACTORS": "1"},
        "struct_attr": None,  # structure is a runtime property; no cheap static proxy
        "regime": "bound_changing",
    },
    "lift_loose_products": {
        "env": {"DISCOPT_LIFT_LOOSE_PRODUCTS": "1"},
        "struct_attr": None,
        "regime": "bound_changing",
    },
    # --- #581 flags wired in 2026-07-15 (previously untracked by the gate) ------ #
    # LU density route (#557): a factorization *route*. A ``bound_neutral`` guess was
    # FALSIFIED by the gate — the alternate route reorders pivots, so it is NOT
    # byte-identical: node_count drifts (nvs02 101->337) and the objective drifts by
    # ~1e-7 (oaer 1.96e-7, st_e13 3.46e-7), both from floating-point roundoff, well
    # within the 1e-6 abs / 1e-4 rel tolerance. So it is ``bound_changing`` (certified
    # objective enforced TO TOLERANCE + oracle bracket; node drift a perf note the
    # graduation PR must weigh — the nvs02 +234% node case is a real trade-off). No
    # cheap static structure proxy (density is a runtime property).
    "lu_density_route": {
        "env": {"DISCOPT_LU_DENSITY_ROUTE": "1"},
        "struct_attr": None,
        "regime": "bound_changing",
    },
    # Square-cost gate (THRU-3): when ON it *shortens* the per-node x**2 tangent
    # loop, dropping cuts → a legitimately looser relaxation (more nodes, same valid
    # bound) → ``bound_changing`` (objective + oracle-bracket enforced; node drift a
    # perf note).
    "square_cost_gate": {
        "env": {"DISCOPT_SQUARE_COST_GATE": "1"},
        "struct_attr": None,
        "regime": "bound_changing",
    },
    # Lifted-FBBT: adds an FBBT sweep + conditional relaxation rebuild per node →
    # tighter boxes / stronger bound (wins ex1252) → ``bound_changing``.
    "lifted_fbbt": {
        "env": {"DISCOPT_LIFTED_FBBT": "1"},
        "struct_attr": None,
        "regime": "bound_changing",
    },
    # alpha-BB alongside the LP: adds an extra (valid) dual bound → ``bound_changing``.
    "alphabb_with_lp": {
        "env": {"DISCOPT_ALPHABB_WITH_LP": "1"},
        "struct_attr": None,
        "regime": "bound_changing",
    },
    "all": {"env": dict(FLAGS_ON), "struct_attr": None, "regime": "bound_changing"},
}

# The parked flags that graduation targets (excludes the ``off`` control and the
# ``all`` bundle cross-check). These are the arms a graduation verdict is emitted
# for.
GRADUATION_ARMS = (
    "root_fixpoint",
    "node_reduce",
    "psd_cost_gate",
    "lift_zero_spanning",
    "lift_loose_products",
    # #581 flags wired in 2026-07-15
    "lu_density_route",
    "square_cost_gate",
    "lifted_fbbt",
    "alphabb_with_lp",
)

# correctness tolerance (matches conftest abs=1e-6, rel=1e-4)
ATOL, RTOL = 1e-6, 1e-4

# "material" node reduction: flags-ON must cut node_count by at least this
# relative amount to count as a node benefit (small jitter is not a class win).
NODE_MATERIAL = 0.05
WALL_MATERIAL = 0.05  # > 5 % wall change is "material" (benefit or regression)


# --------------------------------------------------------------------------- #
# oracle + metadata (corpus)
# --------------------------------------------------------------------------- #
def load_solu(corpus: Path) -> tuple[dict[str, float], dict[str, float]]:
    """MINLPLib ``.solu`` oracle. Returns ``(best, bestdual)``:

    - ``best``     = ``=best=``     — the best *known primal* (an upper fence on the
      true optimum for a min problem; lower fence for max). This is the best value
      anyone has *found*, and can be **loose** — a solver may legitimately find a
      strictly better feasible point.
    - ``bestdual`` = ``=bestdual=`` — the best *known dual bound* (a lower fence on
      the true optimum for min; upper fence for max). No valid dual bound and no
      feasible objective can be on the far side of it.

    The true optimum lies in the bracket ``[bestdual, best]`` (min) /
    ``[best, bestdual]`` (max). Soundness is judged against this **bracket**, not
    against ``=best=`` alone, so a discopt incumbent that merely beats a loose
    ``=best=`` (while respecting ``=bestdual=``) is NOT a false certificate."""
    best: dict[str, float] = {}
    bestdual: dict[str, float] = {}
    path = corpus / "minlplib.solu"
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            with contextlib.suppress(ValueError):
                if parts[0] == "=best=":
                    best[parts[1]] = float(parts[2])
                elif parts[0] == "=bestdual=":
                    bestdual[parts[1]] = float(parts[2])
    return best, bestdual


def load_types(corpus: Path) -> dict[str, str]:
    types: dict[str, str] = {}
    with open(corpus / "minlplib_types.csv") as fh:
        for row in csv.DictReader(fh):
            types[row["name"]] = row["probtype"]
    return types


def load_sizes(corpus: Path) -> dict[str, dict[str, int]]:
    """name -> {vars, cons, nl_cons, binary, integer, discrete, nnz}."""
    sizes: dict[str, dict[str, int]] = {}
    with open(corpus / "problem_sizes.csv") as fh:
        for row in csv.DictReader(fh):
            rec: dict[str, int] = {}
            for k, v in row.items():
                if k == "name":
                    continue
                with contextlib.suppress(ValueError, TypeError):
                    rec[k] = int(v)
            sizes[row["name"]] = rec
    return sizes


def load_curated(corpus: Path, stem: str) -> set[str]:
    out: set[str] = set()
    path = corpus / f"problems_{stem}.txt"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def panel_instances() -> set[str]:
    return {p.name[:-3] for p in PANEL_DIR.glob("*.nl")}


# --------------------------------------------------------------------------- #
# structural prevalence — the honest reach number (R4-style corpus scan)
# --------------------------------------------------------------------------- #
_QCQP_TYPES = {
    "QP",
    "QCP",
    "QCQP",
    "BQP",
    "BQCP",
    "BQCQP",
    "IQP",
    "IQCP",
    "IQCQP",
    "MIQP",
    "MIQCP",
    "MIQCQP",
    "MBQP",
    "MBQCP",
    "MBQCQP",
}


def has_psd_structure(probtype: str) -> bool:
    """PSD (moment) cuts can only bind on a quadratically-constrained/quadratic
    program — the McCormick+RLT gap the moment cut targets requires a quadratic
    form. Approximate the target class by MINLPLib probtype (any Q[C]P family).
    This is a *reach* upper bound: not every QCQP is genuinely-nonconvex, but a
    non-QCQP cannot exercise the PSD separator at all, so a non-QCQP is a true
    negative. Reported honestly as such in the summary."""
    return probtype in _QCQP_TYPES


def has_reduce_structure(size: dict[str, int]) -> bool:
    """The root/node branch-and-reduce fixpoint (OBBT/FBBT-with-cutoff, RC-fixing)
    is only responsive when there is a non-trivial box to tighten: at least one
    variable and at least one constraint to propagate through. A pure unconstrained
    or 0-variable model has nothing to reduce. This is the cheap static proxy for
    "reduce-responsive box"; the *actual* responsiveness is the OFF↔ON benefit the
    solve measures (a runtime signal), which the report pairs with this proxy."""
    nv = size.get("vars", 0)
    nc = size.get("cons", 0)
    return nv >= 1 and nc >= 1


# --------------------------------------------------------------------------- #
# instance selection (held-out, stratified, seeded)
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    name: str
    probtype: str
    nvars: int
    oracle: float  # =best= primal (loose upper fence for min)
    dual: float | None  # =bestdual= (tight lower fence for min); None if absent
    psd_struct: bool
    reduce_struct: bool


def select_instances(
    corpus: Path, n: int, seed: int, max_vars: int
) -> tuple[list[Instance], dict[str, str]]:
    """Draw a held-out, stratified, reproducible sample.

    Exclusions: the 61 vendored panel instances + the named tuning probes.
    Keep only instances with a ``.nl`` in the corpus AND a ``.solu`` oracle, and
    (to keep each pilot solve cheap) with ``vars <= max_vars``. Prefer curated
    small/short (cheap solves). Stratify by probtype so the sample spans the
    structures (QCQP for PSD, mixed-integer for branch-and-reduce), then within a
    stratum prefer smaller instances (fewer vars) so each solve is cheap.
    """
    nl_dir = corpus / "minlplib" / "nl"
    have_nl = {p.name[:-3] for p in nl_dir.glob("*.nl")}
    oracle, dual = load_solu(corpus)
    types = load_types(corpus)
    sizes = load_sizes(corpus)
    excluded = panel_instances() | TUNING_PROBES
    curated = load_curated(corpus, "small") | load_curated(corpus, "short")

    eligible = [
        name
        for name in sorted(have_nl)
        if name not in excluded
        and name in oracle
        and not math.isnan(oracle[name])
        and name in types
        and sizes.get(name, {}).get("vars", 10**9) <= max_vars
    ]

    # Stratify by probtype. Within a stratum, order by (not curated, nvars, name):
    # curated-cheap first, then by size, then name for determinism.
    def cheapness(name: str) -> tuple[int, int, str]:
        nv = sizes.get(name, {}).get("vars", 10**9)
        return (0 if name in curated else 1, nv, name)

    by_type: dict[str, list[str]] = {}
    for name in eligible:
        by_type.setdefault(types[name], []).append(name)
    for names in by_type.values():
        names.sort(key=cheapness)

    # Round-robin across strata (shuffled deterministically by seed) so the sample
    # spans types even at small N, drawing the cheapest unused instance from each
    # stratum in turn.
    rng = random.Random(seed)
    strata = sorted(by_type)
    rng.shuffle(strata)
    cursors = dict.fromkeys(strata, 0)
    picked: list[str] = []
    while len(picked) < n:
        progressed = False
        for t in strata:
            if len(picked) >= n:
                break
            c = cursors[t]
            if c < len(by_type[t]):
                picked.append(by_type[t][c])
                cursors[t] = c + 1
                progressed = True
        if not progressed:
            break  # exhausted the eligible pool

    instances = [
        Instance(
            name=name,
            probtype=types[name],
            nvars=sizes.get(name, {}).get("vars", 0),
            oracle=oracle[name],
            dual=dual.get(name),
            psd_struct=has_psd_structure(types[name]),
            reduce_struct=has_reduce_structure(sizes.get(name, {})),
        )
        for name in picked
    ]
    selection_meta = {
        "eligible_pool": str(len(eligible)),
        "n_strata": str(len(strata)),
        "excluded_panel": str(len(panel_instances())),
        "excluded_probes": str(len(TUNING_PROBES)),
        "max_vars": str(max_vars),
    }
    return instances, selection_meta


# --------------------------------------------------------------------------- #
# discopt solve (isolated subprocess; flags via env) — worker pattern reused
# from global_opt_baron_vs_discopt.py
# --------------------------------------------------------------------------- #
DISCOPT_WORKER = r"""
import json, sys, time
from discopt.modeling.core import from_nl
nl, tl = sys.argv[1], float(sys.argv[2])
t0 = time.perf_counter()
try:
    model = from_nl(nl)
    res = model.solve(time_limit=tl, gap_tolerance=1e-4)
    dt = time.perf_counter() - t0
    lb = getattr(res, "bound", None)
    print(json.dumps({
        "ok": True,
        "status": str(getattr(res, "status", "")),
        "objective": (None if res.objective is None else float(res.objective)),
        "bound": (None if lb is None else float(lb)),
        "gap": (None if res.gap is None else float(res.gap)),
        "node_count": int(getattr(res, "node_count", 0) or 0),
        "wall_time": dt,
    }))
except Exception as e:
    dt = time.perf_counter() - t0
    print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}", "wall_time": dt}))
"""


@dataclass
class SolveRun:
    status: str = "ERROR"
    objective: float | None = None
    bound: float | None = None
    gap: float | None = None
    node_count: int = 0
    wall_time: float = 0.0
    error: str | None = None


def run_discopt(nl_path: Path, tl: float, extra_env: dict[str, str] | None) -> SolveRun:
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    if extra_env:
        env.update(extra_env)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", DISCOPT_WORKER, str(nl_path), str(tl)],
            capture_output=True,
            text=True,
            env=env,
            timeout=tl + 90,
        )
    except subprocess.TimeoutExpired:
        return SolveRun(
            status="TIME_LIMIT",
            wall_time=tl + 90,
            error="outer-timeout (solver hung past budget)",
        )
    dt = time.perf_counter() - t0
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        d = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return SolveRun(
            status="ERROR", wall_time=dt, error=(proc.stderr.strip()[-200:] or "no-json")
        )
    if not d.get("ok"):
        return SolveRun(status="ERROR", wall_time=d.get("wall_time", dt), error=d.get("error"))
    return SolveRun(
        status=d["status"],
        objective=d["objective"],
        bound=d["bound"],
        gap=d["gap"],
        node_count=d["node_count"],
        wall_time=d["wall_time"],
    )


# --------------------------------------------------------------------------- #
# soundness + benefit classification
# --------------------------------------------------------------------------- #
def nl_is_maximize(nl_path: Path) -> bool:
    """Objective sense from the .nl ``O`` segment (``O<k> 1`` == maximize)."""
    try:
        with open(nl_path, errors="replace") as fh:
            for line in fh:
                if line.startswith("O"):
                    parts = line.split()
                    return len(parts) >= 2 and parts[1].strip() == "1"
    except OSError:
        pass
    return False


def matches(obj: float | None, known: float) -> bool:
    if obj is None or math.isnan(known):
        return False
    return abs(obj - known) <= RTOL * max(1.0, abs(known)) + ATOL


def is_optimal(status: str) -> bool:
    return (status or "").strip().lower() == "optimal"


def _tol(x: float) -> float:
    return RTOL * max(1.0, abs(x)) + ATOL


def bound_crosses_primal(bound: float | None, best: float, maximize: bool) -> bool:
    """A valid *dual bound* can never cross the best-known *primal* ``=best=``,
    because ``=best=`` is an *achieved* feasible value: for min the dual lower
    bound must satisfy ``bound <= best`` (a lower bound cannot exceed an achievable
    value); for max the dual upper bound must satisfy ``bound >= best``. A dual bound
    on the far side of an achievable objective is an impossible bound — a false
    certificate seed (CLAUDE.md certificate invariant).
    (Note the asymmetry vs the incumbent test below: a *bound* crossing ``=best=``
    is always invalid, whereas an *incumbent* beating ``=best=`` may just mean
    ``=best=`` is a loose primal — see :func:`false_optimal`.)"""
    if bound is None or math.isnan(best):
        return False
    t = _tol(best)
    # min: dual LOWER bound must be <= best (an achievable value); > best is invalid.
    # max: dual UPPER bound must be >= best; < best is invalid.
    return (bound < best - t) if maximize else (bound > best + t)


def false_optimal(run: SolveRun, best: float, dual: float | None, maximize: bool) -> bool:
    """A certified ``optimal`` that is inconsistent with the ``[dual, best]`` bracket.

    The true optimum lies in ``[bestdual, best]`` (min) / ``[best, bestdual]`` (max).
    A certified-optimal incumbent is a VIOLATION when it is on the *impossible* side
    of the bracket:

    - **below the dual fence** (min) / **above it** (max): better than a valid dual
      bound — impossible, a genuine false certificate; or
    - **beyond the primal fence** in the *worsening* direction while claiming
      ``optimal`` (min: obj > best+tol; max: obj < best-tol): a wrong certified
      value.

    An incumbent strictly *inside* the bracket (e.g. below a loose ``=best=`` for
    min but at/above ``=bestdual=``) is **feasible-and-better**, NOT a violation —
    it means ``=best=`` was a loose primal, a MINLPLib data-quality fact, not a
    discopt bug. When no dual fence is available we fall back to the ``=best=``
    tolerance match (the conservative pre-bracket behavior)."""
    if not is_optimal(run.status) or run.objective is None:
        return False
    obj = run.objective
    if math.isnan(best):
        return False
    tb = _tol(best)
    # worsening side of the primal fence while claiming optimal
    worse_than_best = (obj < best - tb) if maximize else (obj > best + tb)
    if worse_than_best:
        return True
    if dual is not None and not math.isnan(dual):
        td = _tol(dual)
        # impossible side of the dual fence (better than any valid dual bound)
        beyond_dual = (obj > dual + td) if maximize else (obj < dual - td)
        return beyond_dual
    # no dual fence: conservative — must match =best= within tolerance
    return not matches(obj, best)


@dataclass
class Violation:
    instance: str
    config: str  # "off" | "on"
    kind: str  # "bound_crosses_primal" | "false_optimal"
    detail: str


def check_soundness(inst: Instance, run: SolveRun, config: str, maximize: bool) -> list[Violation]:
    v: list[Violation] = []
    if bound_crosses_primal(run.bound, inst.oracle, maximize):
        v.append(
            Violation(
                inst.name,
                config,
                "bound_crosses_primal",
                f"dual bound={run.bound!r} crosses best-known primal={inst.oracle!r} "
                f"({'max' if maximize else 'min'})",
            )
        )
    if false_optimal(run, inst.oracle, inst.dual, maximize):
        v.append(
            Violation(
                inst.name,
                config,
                "false_optimal",
                f"claimed optimal obj={run.objective!r} outside bracket "
                f"[dual={inst.dual!r}, best={inst.oracle!r}] ({'max' if maximize else 'min'})",
            )
        )
    return v


# benefit / regression verdicts
BENEFIT, REGRESS, NEUTRAL = "benefit", "regression", "neutral"


def status_rank(status: str) -> int:
    """Order solve outcomes so a status *upgrade*/*downgrade* is detectable.
    optimal (2) > feasible (1) > everything else (0)."""
    s = (status or "").strip().lower()
    if s == "optimal":
        return 2
    if s == "feasible":
        return 1
    return 0


@dataclass
class Compare:
    instance: str
    verdict: str = NEUTRAL
    node_ratio: float | None = None  # on / off
    wall_ratio: float | None = None  # on / off
    status_off: str = ""
    status_on: str = ""
    reasons: list[str] = field(default_factory=list)


def compare(off: SolveRun, on: SolveRun, name: str) -> Compare:
    """Is flags-ON a benefit, a regression, or neutral vs flags-OFF?

    benefit    : status upgrade (feasible→optimal), OR material node ↓, OR wall ↓ > 5%
    regression : status downgrade, OR wall ↑ > 5% with no compensating node/status win
    neutral    : otherwise
    A solve that ERRORed in either config is excluded from ratios (marked errored).
    """
    c = Compare(instance=name, status_off=off.status, status_on=on.status)
    if off.error or on.error:
        c.verdict = NEUTRAL
        c.reasons.append(f"errored (off={off.error!r}, on={on.error!r})")
        return c

    r_off, r_on = status_rank(off.status), status_rank(on.status)
    if off.node_count > 0:
        c.node_ratio = on.node_count / off.node_count
    if off.wall_time > 0:
        c.wall_ratio = on.wall_time / off.wall_time

    upgrade = r_on > r_off
    downgrade = r_on < r_off
    node_win = c.node_ratio is not None and c.node_ratio <= 1.0 - NODE_MATERIAL
    node_loss = c.node_ratio is not None and c.node_ratio >= 1.0 + NODE_MATERIAL
    wall_win = c.wall_ratio is not None and c.wall_ratio <= 1.0 - WALL_MATERIAL
    wall_loss = c.wall_ratio is not None and c.wall_ratio >= 1.0 + WALL_MATERIAL

    if downgrade:
        c.verdict = REGRESS
        c.reasons.append(f"status {off.status}→{on.status}")
        return c
    if upgrade:
        c.verdict = BENEFIT
        c.reasons.append(f"status {off.status}→{on.status}")
        return c
    # same status: node / wall
    if node_win:
        c.verdict = BENEFIT
        c.reasons.append(f"nodes {off.node_count}→{on.node_count} ({c.node_ratio:.2f}×)")
        return c
    if wall_win and not node_loss:
        c.verdict = BENEFIT
        c.reasons.append(f"wall {off.wall_time:.1f}→{on.wall_time:.1f}s ({c.wall_ratio:.2f}×)")
        return c
    if node_loss or wall_loss:
        c.verdict = REGRESS
        if node_loss:
            c.reasons.append(f"nodes {off.node_count}→{on.node_count} ({c.node_ratio:.2f}×)")
        if wall_loss:
            c.reasons.append(f"wall {off.wall_time:.1f}→{on.wall_time:.1f}s ({c.wall_ratio:.2f}×)")
        return c
    c.verdict = NEUTRAL
    return c


def geomean(xs: Sequence[float | None]) -> float | None:
    vals = [x for x in xs if x is not None and x > 0 and math.isfinite(x)]
    if not vals:
        return None
    return math.exp(statistics.fmean(math.log(x) for x in vals))


def pct_or_dash(x: object) -> str:
    return f"{x * 100:.0f}%" if isinstance(x, (int, float)) else "—"


# --------------------------------------------------------------------------- #
# orchestration + reporting
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    instance: str
    probtype: str
    nvars: int
    oracle: float  # =best= primal
    dual: float | None  # =bestdual=
    maximize: bool
    psd_struct: bool
    reduce_struct: bool
    off: SolveRun
    on: SolveRun
    cmp: Compare
    violations: list[Violation]


def beats_loose_primal(run: SolveRun, best: float, dual: float | None, maximize: bool) -> bool:
    """discopt found a feasible objective strictly *better* than the best-known
    ``=best=`` primal while still respecting the ``=bestdual=`` fence — i.e.
    ``=best=`` is a loose MINLPLib primal, not a discopt error. A data-quality note,
    not a violation. Requires a dual fence to assert the point is inside the bracket."""
    if run.objective is None or math.isnan(best) or dual is None or math.isnan(dual):
        return False
    obj = run.objective
    t = _tol(best)
    strictly_better = (obj > best + t) if maximize else (obj < best - t)
    td = _tol(dual)
    inside_dual = (obj <= dual + td) if maximize else (obj >= dual - td)
    return strictly_better and inside_dual


@dataclass
class CapStats:
    """Per-capability tally over the subset of rows carrying its target structure."""

    structural_prevalence: int  # K in K/N — rows with this capability's structure
    scored: int  # of those, how many ran (non-errored) in both configs
    benefit_count: int
    regression_count: int
    benefit_fraction: float | None
    regression_rate: float | None
    geomean_node_ratio: float | None
    geomean_wall_ratio: float | None
    benefit_instances: list[str]
    regression_instances: list[str]


def per_capability_stats(rows: list[Row], structattr: str) -> CapStats:
    """Restrict to rows carrying this capability's structure, then tally
    benefit-fraction / regression-rate / geomean ratios over that subset."""
    sub = [r for r in rows if getattr(r, structattr)]
    scored = [r for r in sub if not (r.off.error or r.on.error)]
    n = len(scored)
    benefit = sum(r.cmp.verdict == BENEFIT for r in scored)
    regress = sum(r.cmp.verdict == REGRESS for r in scored)
    node_ratios = [r.cmp.node_ratio for r in scored if r.cmp.node_ratio is not None]
    wall_ratios = [r.cmp.wall_ratio for r in scored if r.cmp.wall_ratio is not None]
    return CapStats(
        structural_prevalence=len(sub),
        scored=n,
        benefit_count=benefit,
        regression_count=regress,
        benefit_fraction=(benefit / n) if n else None,
        regression_rate=(regress / n) if n else None,
        geomean_node_ratio=geomean(node_ratios),
        geomean_wall_ratio=geomean(wall_ratios),
        benefit_instances=[r.instance for r in scored if r.cmp.verdict == BENEFIT],
        regression_instances=[r.instance for r in scored if r.cmp.verdict == REGRESS],
    )


# --------------------------------------------------------------------------- #
# Per-flag ARMS runner (G1.2) — reusable by graduation_gate.py.
#
# Solves each selected instance once for the OFF control and once per requested
# arm, then tallies per-arm CapStats (restricted to the arm's structure-carrying
# instances when it has a static structural proxy; whole-sample otherwise). Also
# collects soundness violations across ALL arms (any oracle crossing / false
# optimal, in any config, is a P0 — the arm's env flags cannot excuse it).
# --------------------------------------------------------------------------- #
@dataclass
class ArmRow:
    """One instance solved under the OFF control + one arm's flags."""

    instance: str
    probtype: str
    nvars: int
    oracle: float
    dual: float | None
    maximize: bool
    psd_struct: bool
    reduce_struct: bool
    off: SolveRun
    on: SolveRun  # the arm's ON solve
    cmp: Compare
    violations: list[Violation]


def _arm_struct_ok(arm: str, inst: Instance) -> bool:
    """Does this instance carry the arm's target structure? Arms with no cheap
    static proxy (lifts, all, off) score every instance."""
    attr = ARMS[arm]["struct_attr"]
    if attr is None:
        return True
    return bool(getattr(inst, attr))


def arm_stats(rows: list[ArmRow], arm: str) -> CapStats:
    """Benefit/regression/soundness tally for one arm, restricted to the arm's
    structure-carrying instances (whole-sample when the arm has no static proxy).
    Mirrors :func:`per_capability_stats` but over ArmRows."""
    attr = ARMS[arm]["struct_attr"]
    sub = [r for r in rows if (attr is None or getattr(r, attr))]
    scored = [r for r in sub if not (r.off.error or r.on.error)]
    n = len(scored)
    benefit = sum(r.cmp.verdict == BENEFIT for r in scored)
    regress = sum(r.cmp.verdict == REGRESS for r in scored)
    node_ratios = [r.cmp.node_ratio for r in scored if r.cmp.node_ratio is not None]
    wall_ratios = [r.cmp.wall_ratio for r in scored if r.cmp.wall_ratio is not None]
    return CapStats(
        structural_prevalence=len(sub),
        scored=n,
        benefit_count=benefit,
        regression_count=regress,
        benefit_fraction=(benefit / n) if n else None,
        regression_rate=(regress / n) if n else None,
        geomean_node_ratio=geomean(node_ratios),
        geomean_wall_ratio=geomean(wall_ratios),
        benefit_instances=[r.instance for r in scored if r.cmp.verdict == BENEFIT],
        regression_instances=[r.instance for r in scored if r.cmp.verdict == REGRESS],
    )


def run_arm(
    instances: list[Instance],
    nl_dir: Path,
    arm: str,
    tl: float,
    off_cache: dict[str, SolveRun] | None = None,
    progress: bool = True,
) -> list[ArmRow]:
    """Solve every instance OFF (control) and ON (this arm's env flags).

    ``off_cache`` lets callers reuse one OFF control solve across multiple arms
    (the OFF solve is identical for every arm, so a per-flag sweep pays for it
    once). Mutated in place with each computed control run.
    """
    env_on = ARMS[arm]["env"]
    rows: list[ArmRow] = []
    for idx, inst in enumerate(instances, 1):
        nl_path = nl_dir / f"{inst.name}.nl"
        maximize = nl_is_maximize(nl_path)
        if off_cache is not None and inst.name in off_cache:
            off = off_cache[inst.name]
        else:
            off = run_discopt(nl_path, tl, extra_env=None)
            if off_cache is not None:
                off_cache[inst.name] = off
        on = run_discopt(nl_path, tl, extra_env=(env_on or None))
        viol = check_soundness(inst, off, "off", maximize) + check_soundness(
            inst, on, arm, maximize
        )
        cmp = compare(off, on, inst.name)
        rows.append(
            ArmRow(
                instance=inst.name,
                probtype=inst.probtype,
                nvars=inst.nvars,
                oracle=inst.oracle,
                dual=inst.dual,
                maximize=maximize,
                psd_struct=inst.psd_struct,
                reduce_struct=inst.reduce_struct,
                off=off,
                on=on,
                cmp=cmp,
                violations=viol,
            )
        )
        if progress:
            vtag = "  !!VIOL" if viol else ""
            struct = "★" if _arm_struct_ok(arm, inst) else "·"
            print(
                f"[{arm}] [{idx:2}/{len(instances)}] {inst.name:22} {struct} "
                f"OFF {off.status[:8]:8} {off.node_count:5}n {off.wall_time:5.1f}s | "
                f"ON {on.status[:8]:8} {on.node_count:5}n {on.wall_time:5.1f}s "
                f"-> {cmp.verdict}{vtag}",
                flush=True,
            )
    return rows


def verdict_line(name: str, stats: CapStats, total_n: int) -> str:
    if stats.structural_prevalence == 0:
        return (
            f"**{name}: NO REACH on this sample.** Its target structure appears in "
            f"0/{total_n} held-out instances — the sample cannot test it. Draw a "
            f"structure-targeted sample (larger N / a stratum filter) to measure it."
        )
    bf = stats.benefit_fraction or 0.0
    rr = stats.regression_rate or 0.0
    benefits = stats.benefit_instances
    if bf >= 0.30 and rr <= 0.10:
        tag = "CLASS WIN"
        note = (
            f"helps {int(bf * 100)}% of the {stats.scored} structure-carrying "
            f"instances with regression-rate {int(rr * 100)}% — a class effect, not a probe effect"
        )
    elif bf > 0 and len(benefits) <= 1:
        tag = "PROBE-LIKE"
        note = (
            f"only {len(benefits)} instance improves ({benefits}); on the held-out "
            f"sample this looks confined, not a class — needs larger N to distinguish"
        )
    elif bf == 0:
        tag = "INERT (on sample)"
        note = (
            f"0/{stats.scored} structure-carrying instances improve; either the "
            f"capability is confined to its tuning probes or this N is too small to surface it"
        )
    else:
        tag = "MIXED"
        note = (
            f"{int(bf * 100)}% benefit / {int(rr * 100)}% regression on "
            f"{stats.scored} instances — inconclusive at this N"
        )
    return f"**{name}: {tag}** — {note}."


def write_reports(
    rows: list[Row],
    args: argparse.Namespace,
    selection_meta: dict[str, str],
    ts: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    all_viol = [v for r in rows for v in r.violations]
    scored = [r for r in rows if not (r.off.error or r.on.error)]
    errored = [r for r in rows if r.off.error or r.on.error]

    overall_benefit = sum(r.cmp.verdict == BENEFIT for r in scored)
    overall_regress = sum(r.cmp.verdict == REGRESS for r in scored)
    n_scored = len(scored)

    caps = {
        "branch_reduce": per_capability_stats(rows, "reduce_struct"),
        "psd_gate": per_capability_stats(rows, "psd_struct"),
    }

    overall = {
        "scored": n_scored,
        "errored": len(errored),
        "benefit_count": overall_benefit,
        "regression_count": overall_regress,
        "benefit_fraction": (overall_benefit / n_scored) if n_scored else None,
        "regression_rate": (overall_regress / n_scored) if n_scored else None,
        "geomean_node_ratio": geomean([r.cmp.node_ratio for r in scored]),
        "geomean_wall_ratio": geomean([r.cmp.wall_ratio for r in scored]),
    }

    summary = {
        "timestamp": ts,
        "n_requested": args.n,
        "n_selected": len(rows),
        "seed": args.seed,
        "time_limit": args.time_limit,
        "max_vars": args.max_vars,
        "selection_meta": selection_meta,
        "flags_on": FLAGS_ON,
        "instances": [r.instance for r in rows],
        "soundness": {
            "violations": [asdict(v) for v in all_viol],
            "n_violations": len(all_viol),
            "zero_crossings": len(all_viol) == 0,
        },
        "overall": overall,
        "capabilities": {k: asdict(v) for k, v in caps.items()},
        "rows": [
            {
                "instance": r.instance,
                "probtype": r.probtype,
                "nvars": r.nvars,
                "oracle": r.oracle,
                "dual": r.dual,
                "maximize": r.maximize,
                "psd_struct": r.psd_struct,
                "reduce_struct": r.reduce_struct,
                "off": asdict(r.off),
                "on": asdict(r.on),
                "verdict": r.cmp.verdict,
                "node_ratio": r.cmp.node_ratio,
                "wall_ratio": r.cmp.wall_ratio,
                "reasons": r.cmp.reasons,
            }
            for r in rows
        ],
    }

    js = out_dir / f"generality_sweep_{ts}.json"
    js.write_text(json.dumps(summary, indent=2))

    def pct(x: object) -> str:
        return f"{x * 100:.0f}%" if isinstance(x, (int, float)) else "—"

    def num(x: object, p: int = 2) -> str:
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "—"

    lines = [
        "# Generality sweep (GEN-1) — out-of-panel pilot",
        "",
        f"- Generated: {ts}",
        f"- N requested **{args.n}**, selected **{len(rows)}** "
        f"(eligible pool {selection_meta['eligible_pool']}, seed {args.seed})",
        f"- Time limit **{int(args.time_limit)} s**/solve, gap 1e-4, "
        f"max-vars {args.max_vars}; "
        "correctness tol abs=1e-6 rel=1e-4 vs MINLPLib `.solu` `=best=`",
        "- **Held out:** the 61 vendored panel instances + the "
        f"{len(TUNING_PROBES)} named tuning probes "
        f"({', '.join(sorted(TUNING_PROBES))})",
        "- flags-OFF: stock defaults. flags-ON: "
        f"`{' '.join(f'{k}={v}' for k, v in FLAGS_ON.items())}` (per subprocess env)",
        "",
        "## Soundness (hard gate)",
        "",
    ]
    if all_viol:
        lines.append(
            f"> **P0 — {len(all_viol)} SOUNDNESS VIOLATION(S). "
            "A dual bound crossed the best-known primal, or a certified-optimal "
            "landed outside the `[=bestdual=, =best=]` bracket. INVESTIGATE.**"
        )
        lines.append("")
        lines.append("| instance | config | kind | detail |")
        lines.append("|---|---|---|---|")
        for v in all_viol:
            lines.append(f"| {v.instance} | {v.config} | {v.kind} | {v.detail} |")
    else:
        lines.append(
            "> **0 dual-bound crossings, 0 false-optimal in EITHER config.** "
            f"Checked on all {len(rows)} selected instances (both OFF and ON), "
            "against the `[=bestdual=, =best=]` bracket."
        )

    # loose-primal notes: discopt beat a loose =best= while respecting =bestdual=
    loose = []
    for r in rows:
        for cfg, run in (("off", r.off), ("on", r.on)):
            if beats_loose_primal(run, r.oracle, r.dual, r.maximize):
                loose.append((r.instance, cfg, run.objective, r.oracle, r.dual))
    if loose:
        lines += [
            "",
            "### Data-quality note (NOT violations): discopt beat a loose `=best=`",
            "",
            "These runs found a feasible objective **better than** MINLPLib's "
            "best-known primal `=best=` while staying on the valid side of the dual "
            "fence `=bestdual=` — i.e. `=best=` is a loose primal, not a discopt "
            "error. The certificate invariant holds (objective inside the "
            "`[=bestdual=, =best=]` bracket).",
            "",
            "| instance | config | discopt obj | =best= | =bestdual= |",
            "|---|---|---|---|---|",
        ]
        for name, cfg, obj, best, dl in loose:
            lines.append(
                f"| {name} | {cfg} | {obj:.6g} | {best:.6g} | "
                f"{'—' if dl is None else f'{dl:.6g}'} |"
            )

    lines += [
        "",
        "## Per-capability verdict",
        "",
        "| capability | structural prevalence (K/N) | scored | benefit-fraction | "
        "regression-rate | geomean node ratio (on/off) | geomean wall ratio | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    cap_labels = {
        "branch_reduce": "branch-and-reduce (ROOT_FIXPOINT+NODE_REDUCE)",
        "psd_gate": "PSD cost gate (PSD_COST_GATE)",
    }
    verdicts: list[str] = []
    for key in CAPABILITIES:
        s = caps[key]
        lines.append(
            f"| {cap_labels[key]} | {s.structural_prevalence}/{len(rows)} | "
            f"{s.scored} | {pct(s.benefit_fraction)} | {pct(s.regression_rate)} | "
            f"{num(s.geomean_node_ratio)} | {num(s.geomean_wall_ratio)} | "
            f"{_short_verdict(s)} |"
        )
        verdicts.append(verdict_line(cap_labels[key], s, len(rows)))

    ov = overall
    lines += [
        "",
        "## Overall (whole sample, flags bundle ON vs OFF)",
        "",
        f"- Scored (non-errored both configs): **{ov['scored']}** (errored {ov['errored']})",
        f"- benefit-fraction **{pct(ov['benefit_fraction'])}** "
        f"({ov['benefit_count']}/{ov['scored']})  ·  "
        f"regression-rate **{pct(ov['regression_rate'])}** "
        f"({ov['regression_count']}/{ov['scored']})",
        f"- geomean node ratio (on/off) **{num(ov['geomean_node_ratio'])}**  ·  "
        f"geomean wall ratio **{num(ov['geomean_wall_ratio'])}**",
        "",
        "## Honest verdict per capability (§0.2 — class win vs probe win)",
        "",
    ]
    for line in verdicts:
        lines.append(f"- {line}")

    lines += [
        "",
        "## Per-instance detail",
        "",
        "| instance | type | vars | psd? | red? | OFF status | ON status | "
        "OFF nodes | ON nodes | OFF wall | ON wall | verdict | why |",
        "|---|---|---:|:-:|:-:|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in sorted(rows, key=lambda x: x.instance):
        lines.append(
            f"| {r.instance} | {r.probtype} | {r.nvars} | "
            f"{'Y' if r.psd_struct else '·'} | {'Y' if r.reduce_struct else '·'} | "
            f"{r.off.status} | {r.on.status} | {r.off.node_count} | {r.on.node_count} | "
            f"{r.off.wall_time:.1f} | {r.on.wall_time:.1f} | {r.cmp.verdict} | "
            f"{'; '.join(r.cmp.reasons) if r.cmp.reasons else ''} |"
        )

    if errored:
        lines += ["", "## Errored solves (excluded from ratios)", ""]
        for r in errored:
            lines.append(f"- **{r.instance}**: off={r.off.error!r} on={r.on.error!r}")

    lines += [
        "",
        "## Reproduce",
        "",
        "```",
        "PYTHONPATH=<worktree>/python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\",
        "  python discopt_benchmarks/scripts/generality_sweep.py \\",
        f"    --n {args.n} --seed {args.seed} --time-limit {int(args.time_limit)} "
        f"--max-vars {args.max_vars}",
        "```",
        "",
        f"Exact instance list (order = solve order): {', '.join(r.instance for r in rows)}",
    ]

    md = out_dir / f"generality_sweep_{ts}.md"
    md.write_text("\n".join(lines) + "\n")
    return js, md


def _short_verdict(stats: CapStats) -> str:
    if stats.structural_prevalence == 0:
        return "no reach"
    bf = stats.benefit_fraction or 0.0
    rr = stats.regression_rate or 0.0
    if bf >= 0.30 and rr <= 0.10:
        return "CLASS WIN"
    if bf > 0 and len(stats.benefit_instances) <= 1:
        return "probe-like"
    if bf == 0:
        return "inert"
    return "mixed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20, help="held-out sample size (pilot default 20)")
    ap.add_argument("--seed", type=int, default=0, help="selection seed (reproducibility)")
    ap.add_argument("--time-limit", type=float, default=30.0, help="seconds per solve")
    ap.add_argument(
        "--max-vars",
        type=int,
        default=500,
        help="cap on variable count (keeps each pilot solve cheap; default 500)",
    )
    ap.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS))
    ap.add_argument("--out-dir", type=str, default=str(REPO / "reports"))
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selected instance list and structural prevalence, then exit (no solves)",
    )
    ap.add_argument(
        "--arms",
        type=str,
        default=None,
        help=(
            "comma-separated per-flag arms to run in ISOLATION instead of the "
            f"bundled OFF-vs-ON sweep (choices: {','.join(GRADUATION_ARMS)},all). "
            "Each arm sets only its capability's env flags; the OFF control is "
            "shared across arms. Emits a per-arm CapStats JSON."
        ),
    )
    args = ap.parse_args()

    corpus = Path(os.path.expanduser(args.corpus))
    if not (corpus / "minlplib" / "nl").is_dir():
        print(f"# ERROR: corpus not found at {corpus}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")

    instances, selection_meta = select_instances(corpus, args.n, args.seed, args.max_vars)
    nl_dir = corpus / "minlplib" / "nl"

    n_psd = sum(i.psd_struct for i in instances)
    n_red = sum(i.reduce_struct for i in instances)
    print(
        f"# GEN-1 generality sweep: selected {len(instances)}/{args.n} "
        f"(eligible {selection_meta['eligible_pool']}, seed {args.seed}, "
        f"tl {int(args.time_limit)}s)",
        flush=True,
    )
    print(
        f"# structural prevalence: PSD-target {n_psd}/{len(instances)}, "
        f"reduce-target {n_red}/{len(instances)}",
        flush=True,
    )
    if args.dry_run:
        for i in instances:
            print(
                f"  {i.name:24} {i.probtype:8} vars={i.nvars:5} "
                f"psd={'Y' if i.psd_struct else '·'} red={'Y' if i.reduce_struct else '·'} "
                f"oracle={i.oracle:.6g}",
                flush=True,
            )
        return 0

    # --- Per-flag arms mode (G1.2) --------------------------------------- #
    if args.arms:
        requested = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in requested if a not in ARMS or a == "off"]
        if unknown:
            print(f"# ERROR: unknown/invalid arm(s): {unknown}", file=sys.stderr)
            return 2
        off_cache: dict[str, SolveRun] = {}
        arms_out: dict[str, dict] = {}
        total_viol = 0
        for arm in requested:
            arm_rows = run_arm(instances, nl_dir, arm, args.time_limit, off_cache=off_cache)
            stats = arm_stats(arm_rows, arm)
            viol = [asdict(v) for r in arm_rows for v in r.violations]
            total_viol += len(viol)
            arms_out[arm] = {
                "env": ARMS[arm]["env"],
                "regime": ARMS[arm]["regime"],
                "struct_attr": ARMS[arm]["struct_attr"],
                "stats": asdict(stats),
                "violations": viol,
                "rows": [
                    {
                        "instance": r.instance,
                        "probtype": r.probtype,
                        "nvars": r.nvars,
                        "off": asdict(r.off),
                        "on": asdict(r.on),
                        "verdict": r.cmp.verdict,
                        "node_ratio": r.cmp.node_ratio,
                        "wall_ratio": r.cmp.wall_ratio,
                        "reasons": r.cmp.reasons,
                    }
                    for r in arm_rows
                ],
            }
        payload = {
            "timestamp": ts,
            "mode": "arms",
            "n_requested": args.n,
            "n_selected": len(instances),
            "seed": args.seed,
            "time_limit": args.time_limit,
            "max_vars": args.max_vars,
            "selection_meta": selection_meta,
            "instances": [i.name for i in instances],
            "arms": arms_out,
            "n_violations": total_viol,
        }
        js = out_dir / f"generality_arms_{ts}.json"
        js.write_text(json.dumps(payload, indent=2))
        print("", flush=True)
        for arm in requested:
            s = arms_out[arm]["stats"]
            print(
                f"# arm {arm:20} scored {s['scored']:3} "
                f"benefit {pct_or_dash(s['benefit_fraction'])} "
                f"regression {pct_or_dash(s['regression_rate'])} "
                f"viol {len(arms_out[arm]['violations'])}",
                flush=True,
            )
        print(f"# ARMS JSON: {js}", flush=True)
        return 1 if total_viol else 0

    rows: list[Row] = []
    for idx, inst in enumerate(instances, 1):
        nl_path = nl_dir / f"{inst.name}.nl"
        maximize = nl_is_maximize(nl_path)
        off = run_discopt(nl_path, args.time_limit, extra_env=None)
        on = run_discopt(nl_path, args.time_limit, extra_env=FLAGS_ON)
        viol = check_soundness(inst, off, "off", maximize) + check_soundness(
            inst, on, "on", maximize
        )
        cmp = compare(off, on, inst.name)
        rows.append(
            Row(
                instance=inst.name,
                probtype=inst.probtype,
                nvars=inst.nvars,
                oracle=inst.oracle,
                dual=inst.dual,
                maximize=maximize,
                psd_struct=inst.psd_struct,
                reduce_struct=inst.reduce_struct,
                off=off,
                on=on,
                cmp=cmp,
                violations=viol,
            )
        )
        vtag = "  ⚠️VIOL" if viol else ""
        print(
            f"[{idx:2}/{len(instances)}] {inst.name:22} {inst.probtype:7} "
            f"OFF {off.status[:10]:10} {off.node_count:5}n {off.wall_time:5.1f}s | "
            f"ON {on.status[:10]:10} {on.node_count:5}n {on.wall_time:5.1f}s "
            f"→ {cmp.verdict}{vtag}",
            flush=True,
        )

    js, md = write_reports(rows, args, selection_meta, ts, out_dir)
    nviol = sum(len(r.violations) for r in rows)
    scored = [r for r in rows if not (r.off.error or r.on.error)]
    n_benefit = sum(r.cmp.verdict == BENEFIT for r in scored)
    n_regress = sum(r.cmp.verdict == REGRESS for r in scored)
    print("", flush=True)
    print(
        f"# DONE. scored {len(scored)}, benefit {n_benefit}, "
        f"regression {n_regress}, SOUNDNESS violations {nviol}.",
        flush=True,
    )
    print(f"# JSON: {js}\n# MD:   {md}", flush=True)
    # Non-zero exit on a soundness violation (a P0) so CI/loops catch it.
    return 1 if nviol else 0


if __name__ == "__main__":
    raise SystemExit(main())
