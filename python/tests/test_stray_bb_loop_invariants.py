"""Standing certificate invariants for the three stray branch-and-bound loops.

Card 4c's premise was that porting ``_jax/lp_spatial_bb``, ``gp/solve_gp_minlp``
and ``_jax/convexity/signomial_global`` onto ``PyTreeManager`` would make "every
certificate-critical pruning decision flow through one audited tree manager". Two
entry experiments killed the port (plan §6, 2026-07-30): the Regime-N panel cannot
invoke two of the three loops at all, the only budget-independent comparable is a
five-node tree, and a faithful port needs five policy switches plus two
``export_batch``/``import_results`` contract extensions inside the audited
component — making it harder to audit, not easier. The port is **RETIRED**.

The *goal* survives the port, and this file is how it is served instead: what makes
a pruning decision auditable is that it is **observable**, not that it is
centralised. Each loop now reports every bound-fathom decision through
``discopt.validation.fathom_audit`` (default-inactive: one global read per
decision), and this suite is the standing regression watch none of the three had.

Three invariants, per loop, on real corpus instances with the loops' flags forced
ON — the whole point of the entry experiments' finding that the panel never
executes them on defaults:

**I1 — no wrongful fathom.** A node is never discarded while its bound is better
than the incumbent by more than the optimality tolerance. Fathoming such a node
throws away a subtree that may hold the optimum: a false ``optimal`` certificate.
The admissible slack is re-derived **here**, from the ``gap_tolerance`` the caller
declared — never from the ``slack`` the loop reports. Asserting ``node_bound >=
incumbent - loop_slack`` is a tautology (the loop only fathoms when that holds) and
would read as a pass while measuring nothing (CLAUDE.md §6).

**I2 — the dual bound never crosses the oracle.** For a minimisation the reported
bound must satisfy ``bound <= optimum + tol``. This is the literal soundness
statement and it catches a wrongful fathom end-to-end even where I1 cannot see the
decision. Oracles come from ``discopt_benchmarks/utils/reference_optima``
(``minlplib.solu`` when a snapshot exists, else the vendored
``known_optima.toml`` / ``cert-optima.json``), so the suite scores in CI instead of
degrading to a no-op.

**I3 — the incumbent is feasible.** The returned point passes
``validation/feasibility.verify_point``, the row-scale-aware verifier every
certificate-gating consumer uses. A loop that returns an infeasible incumbent has
reported a primal certificate it cannot honour.

Non-vacuity (CLAUDE.md §6). Every arm bumps a counter; a module-scoped finalizer
fails when any counter is zero, and per-loop guards require that the audit hook
actually observed decisions. A loop that silently stops being reachable — a flag
rename, a classifier tightening, a route reorder — turns this file red instead of
leaving it quietly measuring nothing, which is exactly how Card 4c's own gate
failed.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: E402
from discopt.validation.fathom_audit import fathom_audit  # noqa: E402
from discopt.validation.feasibility import verify_point  # noqa: E402

pytestmark = pytest.mark.slow

_REPO = Path(__file__).resolve().parents[2]
_CORPUS_DIRS = (
    _REPO / "python" / "tests" / "data" / "minlplib_nl",
    _REPO / "python" / "tests" / "data" / "minlplib",
)

# ``discopt_benchmarks`` is not an installed package for the ``python/tests`` suite;
# its ``utils`` package is what owns the oracle chain, so put it on the path rather
# than growing an 18th copy of the ``.solu`` parse (see reference_optima's docstring).
_BENCH = _REPO / "discopt_benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from utils.reference_optima import reference_oracle  # noqa: E402

# The instances each loop's classifier actually accepts, measured over all 119
# corpus instances by ``discopt_benchmarks/scripts/card4c_reachability.py``
# (357 executed classifications, 0 errors) and recorded in
# ``reports/card4c_reachability.json``. Hard-coding the measured list rather than
# re-classifying keeps the suite fast; the per-loop observation guards below fail
# if the list ever stops being reachable.
GP_MINLP_INSTANCES = ("prob03", "cvxnonsep_psig30")
SIGNOMIAL_INSTANCES = ("st_e38", "prob03")
# ``lp_spatial_bb`` is opt-in via ``lp_spatial=True`` and declines (falls through)
# on models its relaxation cannot serve, so this list is instances measured to
# actually enter the loop.
LP_SPATIAL_INSTANCES = ("nvs17",)

# Both budget-independent on this hardware at the limits used below; the
# time-limited instances are still audited for I1/I3 (a fathom is unsound whether
# or not the tree closed) but are excluded from the I2 certification check, which
# only means anything on a proven bound.
BUDGET_INDEPENDENT = {"prob03", "st_e38"}

GAP_TOLERANCE = 1e-4
TIME_LIMIT = 90.0

COUNTS: dict[str, int] = {
    "i1_fathom_decisions": 0,
    "i1_fathomed": 0,
    "i2_bound_vs_oracle": 0,
    "i3_incumbent_verified": 0,
    "loops_observed_gp_minlp": 0,
    "loops_observed_signomial_global": 0,
    "loops_observed_lp_spatial_bb": 0,
}


def _corpus_path(name: str) -> Path:
    for d in _CORPUS_DIRS:
        p = d / f"{name}.nl"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{name}.nl not in {[str(d) for d in _CORPUS_DIRS]} — the suite would "
        f"silently skip an instance it claims to cover (CLAUDE.md §6)."
    )


def _admissible_slack(incumbent: float) -> float:
    """The largest slack the declared optimality tolerance can justify.

    Re-derived from ``GAP_TOLERANCE`` alone. Deliberately the *loosest* defensible
    form — relative term plus an absolute term — so a failure means the loop fathomed
    beyond anything the tolerance could excuse, not that this suite picked a stricter
    convention than the loop did.
    """
    if not math.isfinite(incumbent):
        # No incumbent: nothing may be fathomed by bound at all.
        return 0.0
    return GAP_TOLERANCE * max(1.0, abs(incumbent)) + GAP_TOLERANCE


def _check_i1(log, loop: str) -> tuple[int, int]:
    """I1 over one audit log. Returns (decisions_examined, fathoms_examined)."""
    records = log.for_loop(loop)
    n_fathom = 0
    for d in records:
        assert math.isfinite(d.node_bound) or d.node_bound == -math.inf, (
            f"{loop}/{d.site}: non-finite node bound {d.node_bound!r}"
        )
        if not d.fathomed:
            continue
        n_fathom += 1
        admissible = _admissible_slack(d.incumbent)
        # Internal minimisation sense throughout: improvement > 0 means the node
        # claims to beat the incumbent.
        improvement = d.improvement()
        assert improvement <= admissible + 1e-12, (
            f"I1 VIOLATED — {loop}/{d.site} fathomed a node that could still improve "
            f"the incumbent by {improvement:.6g}, which exceeds the {admissible:.6g} "
            f"admissible at gap_tolerance={GAP_TOLERANCE:g}. "
            f"node_bound={d.node_bound!r} incumbent={d.incumbent!r} "
            f"loop_slack={d.slack!r} extra={d.extra!r}"
        )
    return len(records), n_fathom


def _check_i2(model, result, name: str) -> bool:
    """I2 for one solve. Returns True when an oracle comparison was executed."""
    oracle = reference_oracle(name)
    if oracle is None or not oracle.proven:
        return False
    bound = getattr(result, "bound", None)
    if bound is None or not math.isfinite(float(bound)):
        return False
    bound = float(bound)
    opt = float(oracle.value)
    # ``reference_optima`` stores every value in the instance's own sense and does
    # not normalize, so the sense must come from the model (its docstring is
    # explicit that assuming "min" makes a sound bound look like a violation).
    maximize = model._objective.sense == ObjectiveSense.MAXIMIZE
    tol = 1e-6 + 1e-4 * max(1.0, abs(opt))
    if maximize:
        assert bound >= opt - tol, (
            f"I2 VIOLATED — {name}: reported dual bound {bound!r} is BELOW the known "
            f"optimum {opt!r} for a MAXIMIZE model (source {oracle.source})."
        )
    else:
        assert bound <= opt + tol, (
            f"I2 VIOLATED — {name}: reported dual bound {bound!r} CROSSES the known "
            f"optimum {opt!r} (source {oracle.source}). A dual bound above the true "
            f"optimum is a false certificate."
        )
    return True


def _check_i3(model, result, name: str) -> bool:
    """I3 for one solve. Returns True when a point was actually verified."""
    x = getattr(result, "x", None)
    if not isinstance(x, dict) or not x:
        return False
    try:
        parts = [np.asarray(x[v.name], dtype=np.float64).ravel() for v in model._variables]
    except KeyError as exc:  # reported, never swallowed (CLAUDE.md §7)
        raise AssertionError(
            f"{name}: SolveResult.x is missing variable {exc} — the incumbent cannot "
            f"be verified, so I3 would silently pass on nothing."
        ) from exc
    x_flat = np.concatenate(parts) if parts else np.zeros(0)
    v = verify_point(model, x_flat)
    assert v.ok, (
        f"I3 VIOLATED — {name}: the returned incumbent fails the shared feasibility "
        f"verifier. refusal={getattr(v, 'refusal', None)!r} "
        f"violations={getattr(v, 'violations', None)!r}"
    )
    return True


def _run(name: str, *, env: dict, solve_kwargs: dict):
    """Solve one instance under an audit, with *env* forced for the duration."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        model = from_nl(str(_corpus_path(name)))
        with fathom_audit() as log:
            result = model.solve(time_limit=TIME_LIMIT, gap_tolerance=GAP_TOLERANCE, **solve_kwargs)
        return model, result, log
    finally:
        for k, old in saved.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def _assert_invariants(loop: str, name: str, model, result, log) -> None:
    n_dec, n_fath = _check_i1(log, loop)
    COUNTS["i1_fathom_decisions"] += n_dec
    COUNTS["i1_fathomed"] += n_fath
    if n_dec:
        COUNTS[f"loops_observed_{loop}"] += 1
    if name in BUDGET_INDEPENDENT and _check_i2(model, result, name):
        COUNTS["i2_bound_vs_oracle"] += 1
    if _check_i3(model, result, name):
        COUNTS["i3_incumbent_verified"] += 1
    print(
        f"  {loop:18s} {name:18s} status={result.status:12s} "
        f"obj={result.objective!r} bound={getattr(result, 'bound', None)!r} "
        f"nodes={result.node_count} decisions={n_dec} fathomed={n_fath}"
    )


# ──────────────────────────────────────────────────────────────────────────
# gp/solve_gp_minlp — reached by the explicit ``solver="gp-minlp"`` route AND
# by ``DISCOPT_GP_MINLP=1``; both are forced so a route-table change cannot
# quietly stop exercising the loop.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", GP_MINLP_INSTANCES)
def test_gp_minlp_loop_certificate_invariants(name):
    model, result, log = _run(
        name, env={"DISCOPT_GP_MINLP": "1"}, solve_kwargs={"solver": "gp-minlp"}
    )
    _assert_invariants("gp_minlp", name, model, result, log)


# ──────────────────────────────────────────────────────────────────────────
# _jax/convexity/signomial_global — only reachable through DISCOPT_SGO.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", SIGNOMIAL_INSTANCES)
def test_signomial_global_loop_certificate_invariants(name):
    model, result, log = _run(name, env={"DISCOPT_SGO": "1"}, solve_kwargs={})
    _assert_invariants("signomial_global", name, model, result, log)


# ──────────────────────────────────────────────────────────────────────────
# _jax/lp_spatial_bb — opt-in via ``lp_spatial=True``.
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", LP_SPATIAL_INSTANCES)
def test_lp_spatial_bb_loop_certificate_invariants(name):
    model, result, log = _run(name, env={}, solve_kwargs={"lp_spatial": True})
    _assert_invariants("lp_spatial_bb", name, model, result, log)


# ──────────────────────────────────────────────────────────────────────────
# The audit instrument itself must be able to see a violation (CLAUDE.md §6).
# ──────────────────────────────────────────────────────────────────────────


def test_i1_checker_rejects_a_planted_wrongful_fathom():
    """The discriminator discriminates: a synthetic bad record must fail I1.

    Without this, ``_check_i1`` returning quietly would be indistinguishable from
    ``_check_i1`` having nothing to look at — the "0 violations = pass" failure.
    """
    from discopt.validation.fathom_audit import FathomLog, record_fathom, set_fathom_hook

    log = FathomLog()
    prev = set_fathom_hook(log.append)
    try:
        # A node whose bound is 1.0 better than the incumbent, fathomed anyway.
        record_fathom(
            "gp_minlp",
            "planted",
            node_bound=4.0,
            incumbent=5.0,
            fathomed=True,
            slack=1.5,
        )
    finally:
        set_fathom_hook(prev)

    assert len(log) == 1, "the hook did not record the planted decision"
    with pytest.raises(AssertionError, match="I1 VIOLATED"):
        _check_i1(log, "gp_minlp")

    # And the same record, NOT fathomed, must pass — the check keys on the fathom,
    # not merely on the arithmetic.
    log[0] = type(log[0])(**{**log[0].__dict__, "fathomed": False})
    n_dec, n_fath = _check_i1(log, "gp_minlp")
    assert (n_dec, n_fath) == (1, 0)


def test_audit_hook_is_inactive_by_default():
    """No hook installed means no observation and no cost — the default path."""
    from discopt.validation.fathom_audit import get_fathom_hook, record_fathom

    assert get_fathom_hook() is None
    record_fathom("gp_minlp", "noop", node_bound=0.0, incumbent=0.0, fathomed=True, slack=0.0)


@pytest.fixture(scope="module", autouse=True)
def _executed_assertion_counts():
    """Fail at module teardown if any invariant arm executed zero times."""
    yield
    print("\nCard 4c stray-loop invariant executed counts:")
    for key, n in COUNTS.items():
        print(f"  {key:28s} {n}")
    zero = [k for k, n in COUNTS.items() if n == 0]
    assert not zero, (
        f"vacuous arms (zero executions): {zero}. A loop that stopped being reachable "
        f"must fail this suite, not pass it silently."
    )
