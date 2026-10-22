"""Issue #1869 (a): the spatial tree dead-ended on unbounded variables.

The spatial selectors skip a dimension of infinite width (#467), so a node whose only
open direction was unbounded -- or whose relaxation could not be certified because of
one (a Farkas ray cannot clear an infinite bound) -- was fathomed unresolved or
sentinel-fathomed, and the tree "exhausted" with the gap open after a handful of
nodes. ``DISCOPT_UNBOUNDED_BRANCH`` splits such a dimension at a finite point.

Measured: ``haverly`` 69 nodes, bound -475 vs -400 -> optimal in 57 nodes;
``ex7_3_1`` 1 node at bound 0.0 -> optimal in 5 nodes. ``prolog``/``ex4_1_5`` keep an
unprunable outer piece (no relaxation bounds it yet) and stay open honestly.
"""

from __future__ import annotations

import os
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _bounds_moved, _unbounded_branching_enabled

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_UNBOUNDED_BRANCH"
DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")
# MINLPLib ``minlplib.solu`` optima.
OPTIMA = {"haverly": -400.0, "ex7_3_1": 0.341739562}


def _solve(name: str, tl: float = 30.0):
    return dm.from_nl(os.path.join(DATA, f"{name}.nl")).solve(time_limit=tl)


@pytest.mark.parametrize("name", sorted(OPTIMA))
def test_flag_on_certifies_the_unbounded_instance(name, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    r = _solve(name)
    opt = OPTIMA[name]
    assert r.status == "optimal" and r.gap_certified, (r.status, r.bound, r.node_count)
    assert r.objective == pytest.approx(opt, rel=1e-4, abs=1e-6)
    # The certificate never crosses the oracle.
    assert r.bound <= opt + 1e-4 * max(1.0, abs(opt))


@pytest.mark.parametrize("name", sorted(OPTIMA))
def test_flag_off_drains_unresolved(name, monkeypatch):
    """OFF is the #467 fathom: uncertified, and labelled ``unresolved`` -- not
    ``exhausted``, which #1585 reserves for a complete search."""
    monkeypatch.setenv(FLAG, "0")
    r = _solve(name)
    assert not r.gap_certified
    assert r.termination == "unresolved", r.termination


def test_unprunable_outer_piece_stays_open_and_honest(monkeypatch):
    """Without a bound for ``prolog``'s outer pieces, unbounded branching keeps
    searching and never claims a bound or a certificate it does not have.

    Part (b)'s verified free-column LP bound and the endgame polish together now
    certify prolog (``test_issue_1869_endgame_polish.py``), so both are pinned off
    here to keep this about part (a) alone."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("DISCOPT_FREE_COLUMN_LP_CERT", "0")
    monkeypatch.setenv("DISCOPT_ENDGAME_POLISH", "0")
    r = _solve("prolog", tl=5.0)
    assert not r.gap_certified and r.status == "feasible"
    assert r.bound is None or r.bound <= 0.0 + 1e-6  # optimum 0
    assert r.node_count > 1, "the unbounded root was not branched"


def test_bounds_moved_has_no_inf_minus_inf_warning():
    lb = np.array([0.0, -np.inf, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not _bounds_moved(lb.copy(), lb)
        assert _bounds_moved(np.array([0.0, -np.inf, 1.5]), lb)
        assert _bounds_moved(np.array([0.0, -5.0, 1.0]), lb)  # inf -> finite
        assert not _bounds_moved(np.array([0.0, -np.inf, 1.0 + 1e-14]), lb)


def test_flag_defaults_on(monkeypatch):
    """Graduated under CLAUDE.md section 5 (panel in the gate's docstring)."""
    monkeypatch.delenv(FLAG, raising=False)
    assert _unbounded_branching_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not _unbounded_branching_enabled()
