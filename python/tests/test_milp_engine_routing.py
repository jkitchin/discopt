"""The default MILP route, and the dual bound a deferral used to throw away.

Two behaviours, both new and both bound-changing, so both are pinned here.

**Routing.** The monolithic Rust MILP engine (``_solve_milp_simplex`` →
``crates/discopt-core/src/bnb/milp_driver.rs``) used to be reachable only via
``nlp_solver="simplex"``; the universal default ``"pounce"`` drove the per-node
Python B&B. On a 10-instance MILP panel (60 s, ``max_nodes`` un-starved, 2 reps
with rotated arm order, every ``optimal`` claim oracle-checked) the engine solved
7/10 against 6/10 and cut total wall 275 s → 174 s — ``mk_cflp`` 39x,
``mk_knapsack50`` 28x, ``mk_uflp`` 20x, and ``mk_pmedian`` flipping ``time_limit``
→ ``optimal`` in 9.89 s.

**The deferral.** The engine returns ``None`` when it runs out of node budget with
no incumbent, so the robust Python path takes over. It had, by then, *proven a
dual bound on the same problem*, and that bound went out with the result. Keeping
the tighter of the two is sound — they bound the same feasible set — but it must
not upgrade the status, and it must not cross the incumbent.
"""

import pytest
from discopt.modeling.core import Model, SolveResult
from discopt.solver import _merge_engine_bound, _milp_engine_default_on


def _milp(maximize: bool = False) -> Model:
    """A two-variable pure MILP. Small enough that routing is the only variable."""
    m = Model("route")
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    m.subject_to(x + y <= 7)
    m.subject_to(2 * x + y <= 10)
    if maximize:
        m.maximize(3 * x + 2 * y)
    else:
        m.minimize(-3 * x - 2 * y)
    return m


def _result(**kw) -> SolveResult:
    base = dict(status="feasible", objective=10.0, bound=2.0, gap=0.8)
    base.update(kw)
    return SolveResult(**base)


# --------------------------------------------------------------------------
# the default-on gate
# --------------------------------------------------------------------------


def test_engine_is_the_default_route(monkeypatch):
    monkeypatch.delenv("DISCOPT_MILP_ENGINE", raising=False)
    assert _milp_engine_default_on() is True


@pytest.mark.parametrize("off", ["0", "false", "no", "FALSE", "No"])
def test_opt_out_is_honored(monkeypatch, off):
    """CLAUDE.md §5 keeps the legacy path reachable; that is what the panel A/Bs."""
    monkeypatch.setenv("DISCOPT_MILP_ENGINE", off)
    assert _milp_engine_default_on() is False


def test_gate_is_not_cached(monkeypatch):
    """Read per call, so a test or a panel can flip arms inside one process."""
    monkeypatch.setenv("DISCOPT_MILP_ENGINE", "0")
    assert _milp_engine_default_on() is False
    monkeypatch.setenv("DISCOPT_MILP_ENGINE", "1")
    assert _milp_engine_default_on() is True


def test_default_solve_reaches_the_engine(monkeypatch):
    """The routing change itself: with no ``nlp_solver`` argument at all, the
    engine is consulted. Before the change it was not."""
    import discopt.solver as sv

    seen: list[dict] = []
    real = sv._solve_milp_simplex

    def spy(*args, **kwargs):
        seen.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(sv, "_solve_milp_simplex", spy)
    monkeypatch.delenv("DISCOPT_MILP_ENGINE", raising=False)
    res = _milp().solve(time_limit=30.0)
    assert seen, "the default MILP route never consulted the Rust engine"
    # And it is still right: x=3, y=4 -> -9-8 = -17.
    assert res.status == "optimal"
    assert res.objective == pytest.approx(-17.0, abs=1e-6)


def test_opt_out_keeps_the_python_path(monkeypatch):
    import discopt.solver as sv

    seen: list[int] = []
    real = sv._solve_milp_simplex

    def spy(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(sv, "_solve_milp_simplex", spy)
    monkeypatch.setenv("DISCOPT_MILP_ENGINE", "0")
    res = _milp().solve(time_limit=30.0)
    assert not seen, "opt-out still routed to the engine"
    assert res.status == "optimal"
    assert res.objective == pytest.approx(-17.0, abs=1e-6)


def test_lagrangian_bound_is_not_silently_rerouted(monkeypatch):
    """``lagrangian_bound`` needs a per-node hook the monolithic engine does not
    have. An explicit ``nlp_solver="simplex"`` warns and proceeds (unchanged), but
    the *default* route must decline rather than quietly ignore the request."""
    import discopt.solver as sv

    seen: list[int] = []
    real = sv._solve_milp_simplex

    def spy(*args, **kwargs):
        seen.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(sv, "_solve_milp_simplex", spy)
    monkeypatch.delenv("DISCOPT_MILP_ENGINE", raising=False)
    _milp().solve(time_limit=30.0, lagrangian_bound=True)
    assert not seen, "lagrangian_bound was silently rerouted to the hookless engine"


# --------------------------------------------------------------------------
# the deferred bound
# --------------------------------------------------------------------------


def test_tighter_engine_bound_is_kept_for_a_minimization():
    res = _merge_engine_bound(_result(bound=2.0), 5.0, _milp())
    assert res.bound == pytest.approx(5.0)


def test_looser_engine_bound_is_ignored():
    res = _merge_engine_bound(_result(bound=7.0), 5.0, _milp())
    assert res.bound == pytest.approx(7.0)


def test_engine_bound_fills_an_absent_bound():
    res = _merge_engine_bound(_result(bound=None), 5.0, _milp())
    assert res.bound == pytest.approx(5.0)


def test_maximization_keeps_the_smaller_bound():
    """For a maximization the valid bound is an UPPER one, so tighter means
    smaller. Getting this backwards would report a bound below the optimum — a
    silent false certificate, which is why the sense is read from the model."""
    m = _milp(maximize=True)
    assert _merge_engine_bound(_result(objective=15.0, bound=40.0), 20.0, m).bound == pytest.approx(
        20.0
    )
    assert _merge_engine_bound(_result(objective=15.0, bound=18.0), 20.0, m).bound == pytest.approx(
        18.0
    )


def test_merged_bound_never_crosses_the_incumbent():
    """Two searches, two independent rounding errors: the engine's bound can land
    a hair past an objective the fallback proved optimal. ``bound > objective``
    reads as a broken certificate, so it is clamped."""
    res = _merge_engine_bound(
        _result(status="optimal", objective=10.0, bound=10.0), 10.0 + 1e-7, _milp()
    )
    assert res.bound <= 10.0 + 1e-12


def test_status_is_never_upgraded():
    """A tightened bound can close the gap arithmetically. ``optimal`` is a
    certificate claim and nothing here re-ran the search that would justify it
    (CLAUDE.md §1)."""
    res = _merge_engine_bound(
        _result(status="node_limit", objective=10.0, bound=2.0), 10.0, _milp()
    )
    assert res.status == "node_limit"
    assert res.bound == pytest.approx(10.0)


def test_gap_is_recomputed_to_match_the_new_bound():
    res = _merge_engine_bound(_result(objective=10.0, bound=2.0, gap=0.8), 6.0, _milp())
    assert res.gap == pytest.approx(abs(10.0 - 6.0) / 10.0)


@pytest.mark.parametrize("bad", [None, float("inf"), float("nan")])
def test_an_unusable_engine_bound_changes_nothing(bad):
    original = _result()
    assert _merge_engine_bound(original, bad, _milp()) is original


def test_deferred_dict_is_optional():
    """Every other caller of ``_solve_milp_simplex`` passes no ``deferred``; the
    parameter must stay entirely opt-in."""
    import inspect

    import discopt.solver as sv

    sig = inspect.signature(sv._solve_milp_simplex)
    assert sig.parameters["deferred"].default is None
