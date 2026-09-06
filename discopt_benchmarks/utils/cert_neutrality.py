"""Differential bound-neutrality check for Phase 1 (cert:T1.2, direction (a)).

Phase 1's incremental engine reproduces the *result*, not the exact search tree
(it solves each node with the Neumaier–Shcherbina safe bound + a warm-started
basis — §0.3 mechanisms — so the tree differs *safely*). The neutrality check is
therefore differential, not exact:

  * **certified objective** unchanged to tolerance (a certified optimum jitters
    ~1e-10 across runs; bit-exact equality is not meaningful);
  * **status** still ``optimal`` (the instance still certifies);
  * **node_count** is a *one-directional* performance guard — it may improve, but
    must not get materially worse than the baseline.

The stronger "identical relaxation math" guarantee is enforced separately and
directly by ``IncrementalMcCormickLP._validate`` (row-set equality per box) and
the T0.4 ``assert_bound_sound`` differential-bound harness; this module checks the
end-to-end solve against the committed ``cert-baseline.jsonl``.
"""

from __future__ import annotations

import json
from collections.abc import Collection  # noqa: TC003
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003

# Objective reproducibility tolerance (matches gen_cert_baseline). This is a
# *byte-reproducibility* tolerance: the baseline was produced by the same solver
# math, so a bound-*neutral* change (refactor/cache/marshaling) must reproduce the
# certified objective to ~1e-10. It is DELIBERATELY ~4 orders tighter than the
# correctness tolerance below — for a bound-neutral change, any drift beyond it is
# evidence the change altered the search, i.e. a bug (CLAUDE.md §5, bound-neutral
# regime).
OBJ_TOL = 1e-8
OBJ_RTOL = 1e-9
# Correctness tolerance (matches benchmarks.metrics.incorrect_count / conftest):
# the objective disagrees with the *true optimum* only beyond this. A bound-CHANGING
# flag (a relaxation/reduction/cut behind a default-OFF env flag) legitimately
# changes the search tree, so its certified objective may drift beyond OBJ_TOL while
# staying well within CORRECTNESS tolerance and (crucially) not crossing the true
# optimum on the worsening side. Judging such a flag against OBJ_TOL is a category
# error — it flags a *sound, more-accurate* result as a violation (the ex1225 /
# st_e38 shape). For the bound-changing regime the objective check therefore uses
# this correctness tolerance + an oracle-bracket guard (never cross =opt=).
CORRECTNESS_ATOL = 1e-4
CORRECTNESS_RTOL = 1e-3
# Allowed node_count regression before it's a violation (one-directional guard).
NODE_REGRESSION_FRAC = 0.05

#: Statuses that mean "the run stopped because a budget ran out", not "the run
#: finished". A row that ended this way did an amount of work set by the budget,
#: so two such rows are not two measurements of the same thing.
LIMIT_STATUSES = frozenset({"time_limit", "node_limit", "iteration_limit"})

#: The subset of :data:`LIMIT_STATUSES` whose budget is the **wall clock**. These
#: are the rows ``deterministic=True`` says nothing about (#1187): the flag makes
#: the search a function of the model by neutralizing the role-2 budgets, but it
#: cannot equalise work on a run whose terminating condition *is* the clock, since
#: that condition is role 1 and is deliberately left live. ``node_limit`` and
#: ``iteration_limit`` are excluded — those budgets are deterministic counts, so
#: two runs that hit them did the same amount of work and remain comparable.
WALL_LIMIT_STATUSES = frozenset({"time_limit"})

#: Statuses that mean the run reached a verdict of its own, so a wall-clock
#: coincidence cannot explain a difference between two of them.
_SETTLED_STATUSES = frozenset({"optimal", "infeasible", "unbounded"})

#: Fraction of the budget at which a non-settled row is treated as wall-limited
#: even though its status does not say so. Status alone is not enough: a run that
#: is cut off by ``time_limit`` while holding an incumbent reports **feasible**,
#: not ``time_limit``, and that is the common case rather than the exception.
#: Measured on ``tls2`` at a 60 s budget (30 s in the smaller panel): every run
#: ends ``feasible`` at the wall, and three *baseline* runs returned 245 / 217 /
#: 179 nodes with three different dual bounds — an instance that does not
#: reproduce against itself, which a status-only test would have compared anyway.
WALL_LIMIT_WALL_FRACTION = 0.98


@dataclass
class NeutralityViolation:
    instance: str
    kind: str  # "objective" | "status" | "node_regression" | "missing"
    detail: str


def load_baseline(path: str | Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                out[d["instance"]] = d
    return out


def _objective_violation(
    inst: str,
    nb: float,
    no: float,
    *,
    regime: str,
    oracle: dict[str, float] | None,
    obj_tol: float,
    obj_rtol: float,
) -> NeutralityViolation | None:
    """Decide whether the certified objective drift ``nb -> no`` is a violation.

    - ``bound_neutral`` regime (default): byte-reproducibility. Any drift beyond
      ``obj_tol + obj_rtol*|nb|`` is a violation — a bound-neutral change must
      reproduce the certified objective exactly.
    - ``bound_changing`` regime: the flag legitimately alters the search, so a drift
      within *correctness* tolerance is expected and NOT a violation. It is a
      violation only if the new objective is a **false certificate**: it disagrees
      with the true optimum (``oracle``) by more than the correctness tolerance, OR
      (when no oracle is available for the instance) it drifts from the baseline by
      more than the correctness tolerance. This never masks a wrong answer — it only
      stops flagging a sound, tolerance-accurate drift (e.g. one that lands closer to
      or exactly on the true optimum) as a soundness fault.
    """
    if regime != "bound_changing":
        if abs(no - nb) > obj_tol + obj_rtol * abs(nb):
            return NeutralityViolation(
                inst, "objective", f"|Δobj|={abs(no - nb):.3e} (obj {nb} -> {no})"
            )
        return None
    ctol = CORRECTNESS_ATOL + CORRECTNESS_RTOL * abs(nb)
    opt = (oracle or {}).get(inst)
    if opt is not None:
        # Genuine soundness: the certified value must agree with the TRUE optimum to
        # correctness tolerance. A drift that crosses =opt= beyond tolerance is a
        # real false certificate and is still flagged.
        if abs(no - opt) > CORRECTNESS_ATOL + CORRECTNESS_RTOL * abs(opt):
            return NeutralityViolation(
                inst,
                "objective",
                f"certified obj {no} disagrees with true optimum {opt} "
                f"(|Δ|={abs(no - opt):.3e} > correctness tol) — FALSE CERTIFICATE",
            )
        return None
    # No oracle for this instance: fall back to a correctness-tolerance drift guard
    # vs the baseline (still catches a gross wrong answer; tolerant of benign jitter).
    if abs(no - nb) > ctol:
        return NeutralityViolation(
            inst,
            "objective",
            f"|Δobj|={abs(no - nb):.3e} exceeds correctness tol {ctol:.3e} "
            f"(obj {nb} -> {no}; no oracle to bracket against)",
        )
    return None


def _is_wall_limited(row: dict, budget: float | None) -> bool:
    """Whether ``row`` stopped because the wall clock ran out.

    Two ways to tell, and both are needed. The status says so outright
    (``time_limit``), or the run did not settle and spent essentially its whole
    budget — which is how a wall-cut run that *has* an incumbent presents, since
    it reports ``feasible``.
    """
    status = row.get("status")
    if status in WALL_LIMIT_STATUSES:
        return True
    if status in _SETTLED_STATUSES or budget is None:
        return False
    wall = row.get("wall_time")
    if wall is None:
        return False
    try:
        return float(wall) >= WALL_LIMIT_WALL_FRACTION * float(budget)
    except (TypeError, ValueError):
        return False


def wall_limited_rows(
    new_rows: dict[str, dict],
    baseline: dict[str, dict],
    *,
    budgets: dict[str, float] | None = None,
) -> dict[str, str]:
    """Instances whose neutrality verdict would be read off the wall clock (#1187).

    A row that ended on ``time_limit`` in **both** arms did whatever amount of work
    the clock allowed on the day, so the two arms are not two measurements of the
    same search: an objective or ``node_count`` difference between them is a
    difference in *work*, not in behaviour. Comparing them and calling the result a
    behavior change is how #1180's corpus sweep manufactured a reproducible
    "0.516x regression" that re-measured as a 5x-more-nodes, 30 %-tighter-bound
    improvement — on 13 of 66 rows.

    ``deterministic=True`` does not fix this and does not claim to. The flag
    neutralizes the **role-2** budgets — the sub-budgets that decide *how much
    work* a stage does — while ``time_limit`` itself is role 1 and stays live by
    design (neutralizing it would let a solve run without bound). So on a
    wall-limited run the terminating condition *is* the clock, and no amount of
    role-2 suppression makes two such runs comparable.

    ``budgets`` maps instance -> the wall budget the run was given. Pass it: without
    it only an explicit ``time_limit`` status is detected, and a wall-cut run that
    found an incumbent reports ``feasible`` instead (see
    :data:`WALL_LIMIT_WALL_FRACTION`).

    Returns ``instance -> reason``. Only rows limited on the **wall** in both arms
    are returned: a row that lost a certification (baseline ``optimal`` -> new
    ``time_limit``) is a genuine regression and stays a violation, and a row that
    stopped on ``max_nodes`` stopped on a deterministic count and stays comparable.
    """
    budgets = budgets or {}
    out: dict[str, str] = {}
    for inst, base in baseline.items():
        new = new_rows.get(inst)
        if new is None:
            continue
        budget = budgets.get(inst)
        if _is_wall_limited(base, budget) and _is_wall_limited(new, budget):
            out[inst] = (
                f"both arms ended on the wall clock (status "
                f"{base.get('status')!r} -> {new.get('status')!r}); the work done is "
                f"set by the budget, not by the change under test (#1187)"
            )
    return out


def check_neutrality(
    new_rows: dict[str, dict],
    baseline: dict[str, dict],
    *,
    obj_tol: float = OBJ_TOL,
    obj_rtol: float = OBJ_RTOL,
    node_regression_frac: float = NODE_REGRESSION_FRAC,
    known_perf_gated: dict[str, str] | None = None,
    regime: str = "bound_neutral",
    oracle: dict[str, float] | None = None,
    exclude: Collection[str] = (),
) -> list[NeutralityViolation]:
    """Return the list of neutrality violations of ``new_rows`` vs ``baseline``.

    ``new_rows`` / ``baseline`` map instance -> a dict with at least ``status``,
    ``objective``, ``node_count`` (as produced by ``SolveResult.to_dict``). Every
    baseline instance must be present in ``new_rows``; extras in ``new_rows`` are
    ignored. An empty list means neutral.

    ``known_perf_gated`` maps instance -> reason for instances with a *documented*
    performance-only regression (a slower per-node / near-budget certification that
    a later task fixes — e.g. T1.4 warm-starts). For those, the **perf-class**
    checks (``status`` completeness, ``node_regression``) are downgraded to
    non-violations; the **soundness-class** checks (``objective``, ``missing``) are
    *always* enforced. This keeps a known, tracked perf issue from blocking a
    sound, node-improving change while never masking a wrong answer.

    ``exclude`` names instances that are not evidence either way and must not be
    compared at all — pass :func:`wall_limited_rows` here (#1187). This is not a
    softening of the check: an excluded row yields no verdict, so the caller has to
    report it as unmeasured rather than count it as a pass. Excluding a row the
    caller has not established to be indeterminate would hide a real regression, so
    the set is computed, never hardcoded.

    ``regime`` selects the objective check (see :func:`_objective_violation`):
    ``bound_neutral`` (default) demands byte-reproducibility; ``bound_changing``
    demands agreement with the true optimum ``oracle`` (or a correctness-tolerance
    drift bound when no oracle is present). Node-regression stays a one-directional
    perf guard in both regimes; the caller decides whether to treat it as fatal.
    """
    perf_gated = known_perf_gated or {}
    excluded = set(exclude)
    violations: list[NeutralityViolation] = []
    for inst, base in baseline.items():
        new = new_rows.get(inst)
        if new is None:
            violations.append(NeutralityViolation(inst, "missing", "absent from new run"))
            continue
        # Not evidence either way (#1187). ``missing`` above is still reported for an
        # excluded instance that never ran — "we chose not to read this row" and
        # "the row is not there" are different facts.
        if inst in excluded:
            continue
        gated = inst in perf_gated
        # status: a REGRESSION relative to the reference, not an absolute demand.
        #
        # This used to read `new.status != "optimal"`, which is equivalent whenever
        # the reference is the committed cert-baseline (all 52 of its rows are
        # `optimal` by construction — gen_cert_baseline writes only the
        # deterministically-certifying subset). It is wrong for any *live*
        # reference, such as the flag-OFF panel the graduation gate's CI subset now
        # compares each arm against: an instance that fails to certify inside its
        # wall-clock budget in BOTH arms is a property of the budget and the
        # machine, not something the flag did, and charging it to the flag makes the
        # guard measure the runner. Phrased against the reference it still catches
        # the case that matters — the flag LOST a certification the reference had.
        if base.get("status") == "optimal" and new.get("status") != "optimal" and not gated:
            violations.append(
                NeutralityViolation(
                    inst, "status", f"status={new.get('status')} (baseline optimal)"
                )
            )
        # certified objective (soundness-class — ALWAYS enforced, regime-aware tol).
        nb, no = base.get("objective"), new.get("objective")
        if no is None and gated:
            pass  # a perf-gated instance that didn't certify has no objective to check
        elif nb is None:
            # The REFERENCE certified nothing, so there is no certificate for this
            # run to disagree with. This used to be `nb is None or no is None` ->
            # an "objective" violation, which is soundness-class and hard-fails.
            # Against the committed baseline it could never fire that way (no row
            # has a null objective), so it only ever meant "lost a certificate".
            # Against a live flag-OFF reference it also fires for None -> None
            # (neither side certified — nothing to be wrong about) and for
            # None -> value (the flag certified what the reference could not, an
            # improvement). Neither is a false certificate. A genuinely lost
            # certificate is `nb is not None and no is None`, still flagged below.
            pass
        elif no is None:
            violations.append(NeutralityViolation(inst, "objective", f"objective {nb!r} -> {no!r}"))
        else:
            ov = _objective_violation(
                inst, nb, no, regime=regime, oracle=oracle, obj_tol=obj_tol, obj_rtol=obj_rtol
            )
            if ov is not None:
                violations.append(ov)
        # node_count one-directional guard (perf-class — suppressed if perf-gated).
        base_nc, new_nc = base.get("node_count", 0), new.get("node_count", 0)
        if base_nc > 0 and new_nc > base_nc * (1.0 + node_regression_frac) and not gated:
            violations.append(
                NeutralityViolation(
                    inst,
                    "node_regression",
                    f"node_count {base_nc} -> {new_nc} (+{100 * (new_nc / base_nc - 1):.0f}%)",
                )
            )
    return violations
