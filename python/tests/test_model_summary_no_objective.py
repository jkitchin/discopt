"""``Model.summary()`` / ``repr(model)`` on a model whose objective is not set yet.

Building the variables and rows first and choosing the objective afterwards is a
supported order -- ``validate()`` is what refuses a still-objectiveless model at
solve time, and the same model object is deliberately re-purposed across jobs
(see ``docs/notebooks/cstr_fit_doe_optimize.ipynb``, where one reactor model is
simulated, regressed, designed on, and optimized, with a different objective each
time). ``summary()`` read ``self._objective.sense`` unguarded, so echoing such a
model -- the reflex in a notebook or a debugger, and what ``__repr__`` does --
raised ``AttributeError: 'NoneType' object has no attribute 'sense'``.
"""

from __future__ import annotations

import discopt.modeling as dm


def _half_built():
    m = dm.Model("half_built")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.parameter("c", 2.0)
    m.subject_to(x + y <= 3.0)
    return m, x, y


def test_summary_without_an_objective_reports_none_set():
    m, _, _ = _half_built()
    assert m._objective is None
    text = m.summary()
    assert "Objective: <none set>" in text
    # the rest of the summary still reports the model that is there
    assert "Model: half_built" in text
    assert "Variables: 2" in text
    assert "Constraints: 1" in text
    assert "Parameters: 1" in text


def test_repr_without_an_objective_does_not_raise():
    """``__repr__`` delegates to ``summary()``; echoing the model must not raise."""
    m, _, _ = _half_built()
    assert "Objective: <none set>" in repr(m)


def test_summary_reports_the_objective_once_one_is_set():
    m, x, y = _half_built()
    m.maximize(x + y)
    text = m.summary()
    assert "<none set>" not in text
    assert "Objective: maximize" in text
