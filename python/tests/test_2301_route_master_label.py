"""#2301: the convex-MINLP route's ``master=`` label names the engine that ran.

``_convex_minlp_auto_route`` labels its reason with the master it picked from
``DISCOPT_CONVEX_ROUTE_OA_MASTER``, but ``solve_model`` merges the route's options
with ``setdefault``, so an explicit ``milp_solver=`` outranks it. Before the fix the
label was never updated: on 0.13.0 ``milp_solver="highs"`` ran the HiGHS master
(``mip_nlp_trace["milp_backend"] == "highs"``, 3 nodes on ``synthes1``) under the
label ``master=simplex``, and the reverse with the variable set to ``highs``.
Reporting only -- bounds and answers were unaffected.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.solver import _relabel_convex_route_master

MASTER_ENV = "DISCOPT_CONVEX_ROUTE_OA_MASTER"
METHOD_ENV = "DISCOPT_CONVEX_ROUTE_METHOD"
SYNTHES1_OPT = 6.009759


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        MASTER_ENV,
        METHOD_ENV,
        "DISCOPT_CONVEX_MINLP_ROUTE",
        "DISCOPT_LP_ENGINE",
        "DISCOPT_LP_MILP_BACKEND",
        "DISCOPT_LP_BACKEND",
    ):
        monkeypatch.delenv(var, raising=False)


def _synthes1() -> dm.Model:
    """MINLPLib ``synthes1`` (convex MINLP; optimum 6.009759)."""
    m = dm.Model("synthes1")
    x1, x2 = m.continuous("x1", lb=0, ub=2), m.continuous("x2", lb=0, ub=2)
    x3, y = m.continuous("x3", lb=0, ub=1), m.binary("y", shape=(3,))
    m.minimize(
        5 * y[0]
        + 6 * y[1]
        + 8 * y[2]
        + 10 * x1
        - 7 * x3
        + 10
        - 18 * dm.log(x2 + 1)
        - 19.2 * dm.log(x1 - x2 + 1)
    )
    m.subject_to(0.8 * dm.log(x2 + 1) + 0.96 * dm.log(x1 - x2 + 1) - 0.8 * x3 >= 0)
    m.subject_to(dm.log(x2 + 1) + 1.2 * dm.log(x1 - x2 + 1) - x3 - 2 * y[2] >= -2)
    m.subject_to(x2 <= x1)
    m.subject_to(x2 <= 2 * y[0])
    m.subject_to(x1 - x2 <= 2 * y[1])
    m.subject_to(y[0] + y[1] <= 1)
    return m


def _label(route: str) -> str:
    assert "master=" in route, route
    return route.rsplit("master=", 1)[1].rstrip(")")


@pytest.mark.requires_highs
@pytest.mark.parametrize("env_master", [None, "simplex", "highs"])
@pytest.mark.parametrize("milp_solver", ["highs", "simplex", "auto"])
def test_label_matches_the_master_that_ran(monkeypatch, env_master, milp_solver):
    """``lp_nlp_bb`` (the route default) traces the master it resolved; the label
    must equal it whatever the variable says, and ``"auto"`` names the resolved
    engine, not the request."""
    if env_master is not None:
        monkeypatch.setenv(MASTER_ENV, env_master)
    r = _synthes1().solve(milp_solver=milp_solver, time_limit=60)
    route = r.algorithm_route or ""
    assert route.startswith("mip-nlp/lp_nlp_bb:"), route
    ran = (r.mip_nlp_trace or {}).get("milp_backend")
    assert ran == ("simplex" if milp_solver == "auto" else milp_solver), r.mip_nlp_trace
    assert _label(route) == ran, route
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(SYNTHES1_OPT, abs=1e-5)


@pytest.mark.requires_highs
@pytest.mark.parametrize("env_master", [None, "highs"])
def test_label_without_keyword_follows_the_variable(monkeypatch, env_master):
    """No ``milp_solver=``: the route's own pick runs and is the label."""
    if env_master is not None:
        monkeypatch.setenv(MASTER_ENV, env_master)
    r = _synthes1().solve(time_limit=60)
    ran = (r.mip_nlp_trace or {}).get("milp_backend")
    assert ran == (env_master or "simplex")
    assert _label(r.algorithm_route or "") == ran


@pytest.mark.requires_highs
@pytest.mark.parametrize("milp_solver", ["highs", "simplex", "auto"])
def test_label_on_the_oa_method(monkeypatch, milp_solver):
    """``DISCOPT_CONVEX_ROUTE_METHOD=oa``: multi-tree OA takes its master from
    ``get_milp_solver(backend=milp_solver)`` and traces no ``milp_backend``, so the
    label is that selector's pick (``"auto"`` -> ``"simplex"``), whatever the
    variable says (set here to the other engine)."""
    monkeypatch.setenv(METHOD_ENV, "oa")
    monkeypatch.setenv(MASTER_ENV, "simplex" if milp_solver == "highs" else "highs")
    r = _synthes1().solve(milp_solver=milp_solver, time_limit=60)
    route = r.algorithm_route or ""
    assert route.startswith("mip-nlp/oa:"), route
    assert _label(route) == ("simplex" if milp_solver == "auto" else milp_solver), route
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(SYNTHES1_OPT, abs=1e-5)


@pytest.mark.parametrize(
    "backend, name",
    [("auto", "simplex"), ("simplex", "simplex"), ("pounce", "simplex"), ("highs", "highs")],
)
def test_resolve_milp_backend_matches_the_selector(backend, name):
    """The name comes from the selector ``get_milp_solver`` itself."""
    import warnings

    from discopt.solvers.lp_backend import get_milp_solver, resolve_milp_backend

    assert resolve_milp_backend(backend) == name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        picked = get_milp_solver(backend=backend)
    assert (
        resolve_milp_backend(backend)
        == {
            "discopt.solvers.milp_simplex": "simplex",
            "discopt.solvers.milp_pounce": "simplex",
            "discopt.solvers.milp_highs": "highs",
        }[picked.__module__]
    )


def test_relabel_helper():
    reason = "mip-nlp/oa: minlp certified convex at the root (X; master=simplex)"
    assert _relabel_convex_route_master(reason, "HiGHS ") == reason.replace(
        "master=simplex", "master=highs"
    )
    with pytest.raises(ValueError, match="master="):
        _relabel_convex_route_master("not routed: whatever", "highs")
