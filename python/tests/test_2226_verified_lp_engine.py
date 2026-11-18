"""#2226: the verified free-column LP bound follows ``DISCOPT_LP_ENGINE``.

``mccormick_lp._verified_free_column_bound`` certifies a node LP every in-house
certificate declined (#1869 (b), #1941). It was written as "the verified HiGHS LP
route", and #2196 moved it to the Rust engine with ``solve_lp_std``'s default. #2233
returned the LP seams to ``auto`` (HiGHS where ``highspy`` imports, Rust on
``error``) but missed this one.

The fixture is the ex8_5_3 node LP that exposed it (bundle ON, ``max_nodes=300``,
node 34; ``min c'x, A_ub x <= b_ub``, 155 rows x 37 columns). The Rust simplex
reports ``numerical`` on it (its final audit finds a basic column outside its bound
after EXPAND) and HiGHS solves it to ``optimal`` -0.0675604775. Before the fix the
route returned no bound, the node was fathomed without proof, and ex8_5_3 stopped
uncertified at bound -0.5665 instead of certifying at 219 nodes
(``test_1979_ex8_5_unbounded_root.py::test_ex8_5_3_gets_a_sound_bound_with_the_bundle``).
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax import mccormick_lp as M

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "lp_2226_ex8_5_3_node.npz")
#: HiGHS's certified optimum of the fixture LP (``solve_lp_std(engine="highs")``).
NODE_OPT = -0.06756047745692995


def _milp():
    d = np.load(FIXTURE)
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=tuple(d["A_shape"]))
    return types.SimpleNamespace(
        _A_ub=A,
        _b_ub=d["b_ub"],
        _bounds=list(zip(d["lb"].tolist(), d["ub"].tolist())),
        _c=d["c"],
        _obj_offset=float(d["obj_offset"][0]),
    )


def _clear(monkeypatch):
    for v in ("DISCOPT_LP_STD_ENGINE", "DISCOPT_LP_ENGINE"):
        monkeypatch.delenv(v, raising=False)


@pytest.mark.requires_highs
def test_default_route_certifies_the_node_lp(monkeypatch):
    """Fails before #2226: the route ran only the Rust engine and returned None."""
    _clear(monkeypatch)
    bound, x = M._verified_free_column_bound(_milp(), 60.0)
    assert bound is not None and x is not None
    assert bound == pytest.approx(NODE_OPT, abs=1e-7)
    assert M._verified_lp_engines() == ("highs", "rust")


@pytest.mark.requires_highs
def test_highs_error_falls_back_to_rust(monkeypatch):
    """Under ``auto`` an ``error`` from the first engine is retried on the next one,
    and the fallback is counted; nothing else is."""
    _clear(monkeypatch)
    from discopt.solvers import lp_milp_highs as H

    real = H.solve_lp_std
    seen = []

    def fake(sf, *, time_limit=None, engine=None):
        seen.append(engine)
        if engine == "highs":
            return H.HighsOutcome("error", message="forced (#2226 test)")
        return real(sf, time_limit=time_limit, engine=engine)

    monkeypatch.setattr(H, "solve_lp_std", fake)
    before = M.BOUND_PROVENANCE["verified_engine_fallback"]
    M._verified_free_column_bound(_milp(), 60.0)
    assert seen == ["highs", "rust"]
    assert M.BOUND_PROVENANCE["verified_engine_fallback"] == before + 1


def test_explicit_engine_choices_have_no_fallback(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    assert M._verified_lp_engines() == ("rust",)
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "highs")
    assert M._verified_lp_engines() == ("highs",)
    # The seam's own variable wins and is left to ``solve_lp_std`` to read.
    monkeypatch.setenv("DISCOPT_LP_STD_ENGINE", "rust")
    assert M._verified_lp_engines() == (None,)


def test_without_highspy_auto_is_rust(monkeypatch):
    _clear(monkeypatch)
    import discopt._optional_highs as OH

    monkeypatch.setattr(OH, "highs_available", lambda: False)
    assert M._verified_lp_engines() == ("rust",)
