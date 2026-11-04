"""#2143: the root presolve's marginal stop (``DISCOPT_PRESOLVE_MARGINAL_STOP``, default OFF).

GDPlib ``syngas`` under hull spent 30 s of a 120 s solve in the Rust root presolve
(to ``TimeBudget``, 4 sweeps). Two things kept it sweeping:

* probing re-ran on every sweep, about 7 s each;
* a propagation cycle tightened 15-18 bounds by about 3.4 % per sweep without end.

By default the solve keeps none of the presolve box, only the rewritten model
(``DISCOPT_PRESOLVE_BOUND_PROPAGATION`` is OFF). Under the flag, probing runs in the
first sweep only. Later sweeps go on only for progress the solve keeps: rewrites
and new fixings, plus 5 % box moves when the box is adopted.

``bchoco06`` is the cheap witness of the cycle: legacy runs all 16 sweeps to
``IterationCap``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import discopt.solver as S
import numpy as np
import pytest
from discopt._relax.presolve_pipeline import run_root_presolve
from discopt._rust import model_to_repr
from discopt.modeling import from_nl

FLAG = "DISCOPT_PRESOLVE_MARGINAL_STOP"
NL = Path(__file__).parent / "data" / "minlplib_nl" / "bchoco06.nl"


def _fingerprint(r) -> str:
    """The part of the presolve result a default solve keeps: the returned repr."""
    h = hashlib.sha256()
    h.update(repr((r.n_constraints, r.n_var_blocks, r.n_vars, r.arena_len())).encode())
    for b in range(r.n_var_blocks):
        h.update(np.asarray(r.var_lb(b), float).tobytes())
        h.update(np.asarray(r.var_ub(b), float).tobytes())
    ids = r.constraint_ids()
    for c in range(r.n_constraints):
        h.update(repr((r.constraint_rhs(c), r.constraint_sense(c), ids[c])).encode())
    return h.hexdigest()


def _presolve(**kw):
    m = from_nl(str(NL))
    rep = model_to_repr(m, getattr(m, "_builder", None))
    return run_root_presolve(rep, eliminate=True, polynomial=False, fbbt=True, **kw)


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert S._presolve_marginal_stop_enabled() is False
    assert S._presolve_marginal_stop_kwargs() == {}


def test_flag_threshold_follows_whether_the_box_is_kept(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.delenv("DISCOPT_PRESOLVE_BOUND_PROPAGATION", raising=False)
    kw = S._presolve_marginal_stop_kwargs()
    assert kw["exhaustive_once"] is True and kw["significant_rel"] == float("inf")
    monkeypatch.setenv("DISCOPT_PRESOLVE_BOUND_PROPAGATION", "1")
    kw = S._presolve_marginal_stop_kwargs()
    assert kw["significant_rel"] == S._PRESOLVE_SIGNIFICANT_REL


def test_legacy_presolve_sweeps_to_the_cap_on_the_cycle():
    _, st = _presolve()
    assert st["terminated_by"] == "IterationCap"
    assert st["iterations"] == 16


@pytest.mark.parametrize("rel", [float("inf"), 0.05])
def test_marginal_stop_ends_early_with_the_same_model(rel):
    """Fails before #2143: the binding takes no ``significant_rel``/``exhaustive_once``."""
    legacy_repr, legacy = _presolve()
    new_repr, st = _presolve(significant_rel=rel, exhaustive_once=True)
    assert st["terminated_by"] == "NoProgress"
    assert st["iterations"] < legacy["iterations"]
    assert _fingerprint(new_repr) == _fingerprint(legacy_repr)
    # Probing ran in the first sweep only.
    probing = [d for d in st["deltas"] if d["pass_name"] == "probing"]
    assert len(probing) == 1


def test_default_call_is_the_legacy_call():
    a_repr, a = _presolve()
    b_repr, b = _presolve(significant_rel=0.0, exhaustive_once=False)
    assert a["iterations"] == b["iterations"]
    assert _fingerprint(a_repr) == _fingerprint(b_repr)
    np.testing.assert_array_equal(a["bounds_lo"], b["bounds_lo"])
    np.testing.assert_array_equal(a["bounds_hi"], b["bounds_hi"])
