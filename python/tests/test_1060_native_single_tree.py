"""#1060 — LP/NLP branch-and-bound without a Gurobi license.

Before this change ``mip_nlp_method="lp_nlp_bb"`` raised for every MILP backend
but ``"gurobi"``, because the single tree needs a persistent lazy-constraint
callback. The Rust MILP driver now exposes one (``solve_milp_lazy_csc_py``), so
``"auto"``/``"simplex"`` run the same algorithm in-house.

Every end-to-end test here asserts the separator actually fired
(``mipsol_calls``/``lazy_requeues``) — a single-tree run that never calls the
callback would still report ``optimal`` on these models by solving the initial
linearization, and would read as a pass while testing nothing (CLAUDE.md §6).
"""

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import Model
from discopt.solvers import SolveStatus
from discopt.solvers.milp_simplex import solve_milp, solve_milp_with_lazy_cuts
from discopt.solvers.oa import _resolve_lp_nlp_bb_backend


def _build_convex_minlp(n: int = 6) -> Model:
    """A convex MINLP whose first integer incumbent is *not* optimal.

    The knapsack-style cardinality limit plus the big-M links make the initial
    linearization loose enough that the separator has to reject several integer
    points, so the requeue path is exercised rather than just the accept path.
    """
    m = Model("convex_minlp_1060")
    x = [m.continuous(f"x{i}", lb=0.0, ub=5.0) for i in range(n)]
    b = [m.binary(f"b{i}") for i in range(n)]
    for i in range(n):
        m.subject_to(x[i] <= 5.0 * b[i])
    m.subject_to(sum(b) <= 3)
    m.subject_to(sum(x) >= 6.0)
    obj = 0
    for i in range(n):
        obj = obj + dm.exp(0.4 * x[i]) + (x[i] - float(i) * 0.5) ** 2 - 6.0 * b[i]
    m.minimize(obj)
    return m


def _callback_stats(result) -> dict:
    return dict(result.mip_nlp_trace["summary"]["callback_stats"])


# --------------------------------------------------------------------------
# backend resolution
# --------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "requested,expected",
    [("auto", "simplex"), ("simplex", "simplex"), ("SIMPLEX", "simplex"), ("gurobi", "gurobi")],
)
def test_lazy_backend_resolution(requested, expected):
    assert _resolve_lp_nlp_bb_backend(requested, shot_profile=False) == expected


@pytest.mark.smoke
def test_pounce_backend_is_refused_not_silently_substituted():
    # POUNCE's matrix MILP has no separator hook. Substituting a different
    # backend would hide that the caller's choice was ignored.
    with pytest.raises(RuntimeError, match="no separator hook"):
        _resolve_lp_nlp_bb_backend("pounce", shot_profile=False)


@pytest.mark.smoke
def test_shot_profile_runs_on_any_backend_with_a_fractional_node_hook():
    # SHOT separates hyperplanes at *fractional* node relaxations. Until #1141 only
    # Gurobi's MIPNODE could do that and this resolution refused everything else;
    # the native driver now has the hook too, so 'simplex'/'auto' resolve as well.
    assert _resolve_lp_nlp_bb_backend("simplex", shot_profile=True) == "simplex"
    assert _resolve_lp_nlp_bb_backend("auto", shot_profile=True) == "simplex"
    assert _resolve_lp_nlp_bb_backend("gurobi", shot_profile=True) == "gurobi"


@pytest.mark.smoke
def test_shot_profile_still_refuses_the_highs_master():
    # The HiGHS master separates only at integer-feasible incumbents, so running
    # SHOT there would report a SHOT run that never ran SHOT's cut generation.
    with pytest.raises(RuntimeError, match="fractional-node cut hook"):
        _resolve_lp_nlp_bb_backend("highs", shot_profile=True)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


@pytest.mark.smoke
def test_lp_nlp_bb_runs_on_the_native_backend():
    """The regression: this raised RuntimeError before #1060."""
    result = _build_convex_minlp().solve(
        solver="mip-nlp",
        mip_nlp_method="lp_nlp_bb",
        milp_solver="simplex",
        time_limit=120,
    )
    assert result.status == "optimal"
    assert result.objective is not None
    assert result.mip_nlp_trace["milp_backend"] == "simplex"

    stats = _callback_stats(result)
    # Anti-vacuity: the separator ran, rejected points, and those rejections
    # were served by re-queueing rather than fathoming.
    assert stats["mipsol_calls"] > 0
    assert stats["driver_lazy_calls"] >= stats["mipsol_calls"]
    assert stats["lazy_cuts"] > 0
    assert stats["lazy_requeues"] > 0


@pytest.mark.smoke
def test_native_single_tree_agrees_with_multi_tree_oa_and_global_bb():
    single = _build_convex_minlp().solve(
        solver="mip-nlp", mip_nlp_method="lp_nlp_bb", milp_solver="simplex", time_limit=120
    )
    multi = _build_convex_minlp().solve(
        solver="mip-nlp", mip_nlp_method="oa", milp_solver="simplex", time_limit=120
    )
    spatial = _build_convex_minlp().solve(time_limit=180)

    assert _callback_stats(single)["lazy_requeues"] > 0, "separator never vetoed; test is vacuous"
    for name, other in (("oa", multi), ("bb", spatial)):
        assert other.objective is not None, name
        assert single.objective == pytest.approx(other.objective, rel=1e-6, abs=1e-6), name


@pytest.mark.smoke
def test_auto_backend_reaches_the_native_single_tree():
    result = _build_convex_minlp(4).solve(
        solver="mip-nlp", mip_nlp_method="lp_nlp_bb", time_limit=120
    )
    assert result.mip_nlp_trace["milp_backend"] == "simplex"
    assert _callback_stats(result)["mipsol_calls"] > 0


# --------------------------------------------------------------------------
# the native lazy wrapper itself
# --------------------------------------------------------------------------


def _knapsack():
    # min -10 x0 - 9 x1 - 8 x2 - x3  s.t.  5(x0+x1+x2+x3) <= 9, x binary
    c = np.array([-10.0, -9.0, -8.0, -1.0])
    A_ub = sp.csr_matrix(np.full((1, 4), 5.0))
    b_ub = np.array([9.0])
    bounds = [(0.0, 1.0)] * 4
    integrality = np.ones(4, dtype=np.int64)
    return c, A_ub, b_ub, bounds, integrality


@pytest.mark.smoke
def test_accept_all_separator_is_bound_neutral():
    c, A_ub, b_ub, bounds, integrality = _knapsack()
    calls = {"n": 0}

    def accept_everything(_x):
        calls["n"] += 1
        return []

    baseline = solve_milp(c, A_ub, b_ub, None, None, bounds, integrality)
    lazy = solve_milp_with_lazy_cuts(
        c, A_ub, b_ub, None, None, bounds, integrality, lazy_callback=accept_everything
    )
    assert calls["n"] > 0, "separator never fired; neutrality claim is vacuous"
    assert lazy.status == baseline.status
    assert lazy.objective == pytest.approx(baseline.objective, rel=0, abs=0)
    assert lazy.node_count == baseline.node_count


@pytest.mark.smoke
def test_veto_of_the_optimum_returns_the_next_best_point():
    c, A_ub, b_ub, bounds, integrality = _knapsack()

    def veto_x0(x):
        if x[0] > 0.5:
            return [(np.array([1.0, 0.0, 0.0, 0.0]), 0.0)]  # x0 <= 0
        return []

    result = solve_milp_with_lazy_cuts(
        c, A_ub, b_ub, None, None, bounds, integrality, lazy_callback=veto_x0
    )
    assert result.status == SolveStatus.OPTIMAL
    assert result.objective == pytest.approx(-9.0)
    assert result.x[0] < 0.5
    assert result.callback_stats["lazy_requeues"] > 0


@pytest.mark.smoke
def test_separator_exception_propagates_rather_than_being_swallowed():
    c, A_ub, b_ub, bounds, integrality = _knapsack()

    def boom(_x):
        raise ZeroDivisionError("separator blew up")

    with pytest.raises(Exception, match="separator blew up"):
        solve_milp_with_lazy_cuts(
            c, A_ub, b_ub, None, None, bounds, integrality, lazy_callback=boom
        )


@pytest.mark.smoke
def test_unsupported_callbacks_are_refused_loudly():
    c, A_ub, b_ub, bounds, integrality = _knapsack()
    kw = dict(
        c=c, A_ub=A_ub, b_ub=b_ub, A_eq=None, b_eq=None, bounds=bounds, integrality=integrality
    )

    with pytest.raises(ValueError, match="lazy_callback"):
        solve_milp_with_lazy_cuts(**kw, lazy_callback=None)
    # #1141 gave the driver a fractional-node hook, so `node_callback` is honoured
    # now -- but only with a budget it can actually spend. A zero budget makes the
    # separator unfireable, and `mipnode_calls == 0` would then be indistinguishable
    # from "it ran and found nothing" (CLAUDE.md §6), so it is refused.
    with pytest.raises(ValueError, match="node_hook_rounds"):
        solve_milp_with_lazy_cuts(
            **kw,
            lazy_callback=lambda _x: [],
            node_callback=lambda _x: [],
            node_hook_rounds=0,
        )
    with pytest.raises(ValueError, match="node_hook_cut_cap"):
        solve_milp_with_lazy_cuts(
            **kw,
            lazy_callback=lambda _x: [],
            node_callback=lambda _x: [],
            node_hook_cut_cap=0,
        )
    # Accepting this while ignoring it would report a termination that never fired.
    with pytest.raises(NotImplementedError, match="terminate_callback"):
        solve_milp_with_lazy_cuts(
            **kw, lazy_callback=lambda _x: [], terminate_callback=lambda _s: False
        )
