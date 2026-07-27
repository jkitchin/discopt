"""Regression tests for the batch substitution aggregator (#844, P2(a′)).

Covers the three properties the Rust pass has to guarantee end to end:

1. **substitute-everywhere** — a variable defined by a linear equality is
   rewritten out of every row *and* the objective, which is exactly what the
   pre-existing ``aggregate``/``eliminate`` passes cannot do (they require the
   eliminated variable to appear in a single expression). ``test_legacy_*``
   pins that pre-existing limitation so the difference is measured, not
   asserted;
2. **postsolve inverts exactly** — including a chain of ≥3 substitutions and a
   rejected cycle;
3. **the solve path agrees** — ``DISCOPT_PRESOLVE_SUBSTITUTE=1`` returns the
   same optimum, in the ORIGINAL variables, as the default path.
"""

import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._rust import model_to_repr


def _repr_of(model):
    return model_to_repr(model, getattr(model, "_builder", None))


def _chain_model():
    """x0 = 2·x1 + 1, x1 = 3·x2 + 2, x2 = 4·x3 + 3 (a depth-3 chain), plus an
    inequality and an objective in which every chained variable appears."""
    m = dm.Model("chain")
    x = [m.continuous(f"x{i}", lb=-1e3, ub=1e3) for i in range(5)]
    m.subject_to(x[0] - 2 * x[1] == 1)
    m.subject_to(x[1] - 3 * x[2] == 2)
    m.subject_to(x[2] - 4 * x[3] == 3)
    m.subject_to(x[0] + x[1] + x[2] + x[3] <= 100)
    m.minimize(x[0] + x[4])
    return m


@pytest.mark.smoke
def test_legacy_aggregate_pass_cannot_reduce_the_chain_model():
    """Fails-before evidence: the pre-existing passes eliminate nothing here."""
    rep = _repr_of(_chain_model())
    reduced, _ = rep.presolve(
        passes=["eliminate", "aggregate", "factorable_elim"],
        max_iterations=8,
        time_limit_ms=5000,
    )
    assert reduced.n_vars == rep.n_vars == 5


@pytest.mark.smoke
def test_chain_of_three_substitutions_is_eliminated_and_inverted():
    m = _chain_model()
    rep = _repr_of(m)
    reduced, chain = rep.substitute(4)

    assert chain.variables_eliminated == 3
    assert reduced.n_vars == 2, "x3 and the unused x4 survive"
    assert reduced.n_constraints == 1, "only the inequality remains"

    # Every definition names the SAME surviving representative: the chain
    # collapsed, which is why postsolve needs no topological ordering.
    recs = chain.records(0)
    sources = {r["source_block"] for r in recs if r["kind"] == "affine"}
    assert len(sources) == 1, f"expected one representative, got {sources}"

    checked = 0
    for t in (-3.5, 0.0, 2.25, 7.0):
        x_red = [t + 0.5 * i for i in range(reduced.n_vars)]
        x_full = np.asarray(chain.postsolve(x_red), float)
        assert x_full.shape == (rep.n_vars,)
        obj_r, _, _ = reduced.evaluate_point(x_red)
        obj_p, con_p, _ = rep.evaluate_point(list(x_full))
        # The three dropped equalities are the ONLY equalities in the model, so
        # a zero constraint violation on the pristine model at the lifted point
        # is exactly the statement that postsolve inverted them.
        assert con_p == pytest.approx(0.0, abs=1e-9), f"dropped rows violated at t={t}"
        assert obj_p == pytest.approx(obj_r, abs=1e-9)
        checked += 1
    assert checked == 4


@pytest.mark.smoke
def test_cycle_is_rejected_and_its_row_is_kept():
    m = dm.Model("cycle")
    x = [m.continuous(f"x{i}", lb=-10, ub=10) for i in range(3)]
    m.subject_to(x[0] - x[1] == 0)
    m.subject_to(x[1] - x[2] == 0)
    m.subject_to(x[2] - x[0] == 0)  # closes the cycle: defines nothing new
    m.minimize(x[0])

    rep = _repr_of(m)
    reduced, chain = rep.substitute(4)
    stats = chain.sweep_stats()[0]

    assert chain.variables_eliminated == 2
    assert stats["cycles_rejected"] == 1
    assert reduced.n_vars == 1
    assert reduced.n_constraints == 1, "the cycle row stays in the model"
    assert chain.postsolve([4.25]) == pytest.approx([4.25, 4.25, 4.25])


@pytest.mark.smoke
def test_infeasible_transfer_is_reported_and_the_model_is_unchanged():
    m = dm.Model("infeas")
    a = m.continuous("a", lb=0.0, ub=1.0)
    b = m.continuous("b", lb=5.0, ub=6.0)
    m.subject_to(a - b == 0)
    m.minimize(a)

    rep = _repr_of(m)
    reduced, chain = rep.substitute(4)
    assert chain.variables_eliminated == 0
    assert reduced.n_vars == rep.n_vars
    assert reduced.n_constraints == rep.n_constraints


@pytest.mark.smoke
def test_integer_source_is_kept_and_continuous_target_is_eliminated():
    m = dm.Model("intsrc")
    x = m.continuous("x", lb=-100, ub=100)
    y = m.integer("y", lb=0, ub=10)
    m.subject_to(x - 2 * y == 1)
    m.minimize(x)

    rep = _repr_of(m)
    reduced, chain = rep.substitute(4)
    assert chain.variables_eliminated == 1
    assert reduced.var_types() == ["integer"]
    assert chain.postsolve([3.0]) == pytest.approx([7.0, 3.0])


@pytest.mark.smoke
def test_solve_path_flag_agrees_with_the_default_path():
    """The reduced solve must return the same optimum, in original variables."""
    prev = os.environ.get("DISCOPT_PRESOLVE_SUBSTITUTE")
    try:
        os.environ["DISCOPT_PRESOLVE_SUBSTITUTE"] = "0"
        base = _chain_model().solve(time_limit=30)
        os.environ["DISCOPT_PRESOLVE_SUBSTITUTE"] = "1"
        sub = _chain_model().solve(time_limit=30)
    finally:
        if prev is None:
            os.environ.pop("DISCOPT_PRESOLVE_SUBSTITUTE", None)
        else:
            os.environ["DISCOPT_PRESOLVE_SUBSTITUTE"] = prev

    assert sub.status == base.status
    assert sub.objective == pytest.approx(base.objective, rel=1e-6, abs=1e-6)
    # Reported in the ORIGINAL variables, and feasible for the pristine model.
    assert set(sub.x) == {"x0", "x1", "x2", "x3", "x4"}
    rep = _repr_of(_chain_model())
    flat = [float(np.asarray(sub.x[f"x{i}"]).reshape(-1)[0]) for i in range(5)]
    _, con_viol, bnd_viol = rep.evaluate_point(flat)
    assert max(con_viol, bnd_viol) < 1e-6
