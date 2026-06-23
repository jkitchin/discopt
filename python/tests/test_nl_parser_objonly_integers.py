"""Regression: .nl parser must type objective-only integer variables when
``nlvo > nlvc`` (AMPL variable-ordering edge case).

ex1252a's header has ``nlvc=15, nlvo=21, nlvb=6, nlvoi=3``: its 3 binaries
(b22,b23,b24) are integer variables that appear nonlinearly only in the
*objective*. The old parser sized the objective-only group as ``nlvo - nlvb`` and
placed its integer tail at index 27 — out of range for a 24-variable model — so
the binaries were silently left **continuous**. That relaxed them from {0,1} to
[0,1], enlarging the feasible set, and the solver returned a *false-feasible*
incumbent (obj ~92117, below the proven optimum 128893.74).

Correct AMPL layout when ``nlvo > nlvc``: the nonlinear block is ``[0, nlvo)`` and
the objective-only group is ``[nlvc, nlvo)``, with its integer tail at
``[nlvo-nlvoi, nlvo)``. Both ex1252a and ghg_1veh are the only vendored instances
whose types change under the fix.
"""

from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt.modeling.core import VarType  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")
_EX1252A_OPT = 128893.741


def _flat_types(model):
    out = []
    for v in model._variables:
        out.extend([v.var_type] * v.size)
    return out


def test_ex1252a_objective_only_binaries_typed_integer():
    """The 3 objective-only binaries must parse as discrete, not continuous."""
    path = os.path.join(_DATA, "ex1252a.nl")
    if not os.path.exists(path):
        pytest.skip("ex1252a instance unavailable")
    m = dm.from_nl(path)
    types = _flat_types(m)
    counts = Counter(str(t).split(".")[-1] for t in types)
    # 6 nonlinear-in-both integers (i16..i21) + 3 objective-only binaries (b22..b24).
    assert counts.get("INTEGER", 0) + counts.get("BINARY", 0) == 9, (
        f"expected 9 discrete vars, got {dict(counts)} (objective-only binaries "
        "were dropped to continuous)"
    )
    # The 3 binaries are discrete with a [0,1] box.
    lbs, ubs = [], []
    for v in m._variables:
        lbs.extend(np.atleast_1d(np.asarray(v.lb, float)).ravel().tolist())
        ubs.extend(np.atleast_1d(np.asarray(v.ub, float)).ravel().tolist())
    n_binary_box = sum(
        1
        for t, lo, hi in zip(types, lbs, ubs)
        if t in (VarType.BINARY, VarType.INTEGER) and lo == 0.0 and hi == 1.0
    )
    assert n_binary_box >= 3, f"expected >=3 discrete [0,1] vars, got {n_binary_box}"


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_ex1252a_not_false_feasible():
    """With the binaries correctly discrete, no incumbent may sit below the proven
    optimum (the pre-fix relaxed-binary feasible set yielded obj ~92117 < opt)."""
    path = os.path.join(_DATA, "ex1252a.nl")
    if not os.path.exists(path):
        pytest.skip("ex1252a instance unavailable")
    r = dm.from_nl(path).solve(time_limit=40, gap_tolerance=1e-4)
    if r.objective is not None:
        assert r.objective >= _EX1252A_OPT - 1e-2, (
            f"false-feasible: incumbent {r.objective} is below the proven optimum {_EX1252A_OPT}"
        )
