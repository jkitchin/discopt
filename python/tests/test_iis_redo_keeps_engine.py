"""PR #2196 review: the IIS filter's redo path keeps the caller's engine.

When a presolve-on verdict the filter acted on is refuted, ``redo_each`` reruns the
filter with per-probe confirmation. It called ``_milp_iis_arrays`` without
``engine=``, so a default ``iis_engine="rust"`` run silently switched to HiGHS when
highspy was installed, and raised ``MissingHighsError`` from deep inside the call when
it was not. The internal defaults now say ``"rust"`` too.

The refutation is forced through ``_Engine.confirm_infeasible`` (checkpoint 0 answers
"feasible" once), with highspy blocked, so the test is deterministic and proves the
redo ran without HiGHS.
"""

from __future__ import annotations

import sys

import discopt.modeling as dm
from discopt import _iis_backend as _bk
from discopt import _iis_milp


def _parity_model(n=8):
    m = dm.Model("parity")
    x = m.binary("x", shape=(n,))
    m.subject_to(sum(x[k] for k in range(n)) == n / 2 + 0.5, name="half")
    m.minimize(0 * x[0])
    return m


def test_a_refuted_verdict_redoes_the_filter_on_the_callers_engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "highspy", None)  # HiGHS unavailable
    monkeypatch.delenv("DISCOPT_IIS_ENGINE", raising=False)
    engines: list[str] = []
    real_make = _bk.make_backend

    def make(engine, *a, **k):
        engines.append(engine)
        return real_make(engine, *a, **k)

    monkeypatch.setattr(_bk, "make_backend", make)
    refuted = {"n": 0}
    real_confirm = _iis_milp._Engine.confirm_infeasible

    def confirm(self, *a, **k):
        if refuted["n"] == 0 and k.get("presolve") == ("off",):
            refuted["n"] += 1
            return False  # checkpoint 0 refutes the presolve-on base proof
        return real_confirm(self, *a, **k)

    monkeypatch.setattr(_iis_milp._Engine, "confirm_infeasible", confirm)
    res = _iis_milp.milp_iis(_parity_model(), confirm="final", total_time_limit=60.0)
    assert refuted["n"] == 1, "the refutation was never forced; the redo path did not run"
    assert res.fallback, "the result did not come from the redo"
    assert engines and set(engines) == {"rust"}, engines
