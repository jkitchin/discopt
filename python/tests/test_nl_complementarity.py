"""`.nl` complementarity (type-5) preservation and GDP lowering (#658).

An AMPL `.nl` complementarity row is `5 k i` in the `r` (constraint bounds)
segment: `k` is the AMPL MP ``ComplInfo`` bound flag and `i` is the *1-based
index of the complementary variable* — neither is a numeric bound. The old
parser misread the two operands as `lb`/`ub` and fabricated a bogus
`k <= body <= i` range constraint, silently corrupting every `.nl` MPEC.

These tests lock:

1. the parser recovers the complementarity pair and emits **no** fabricated
   range constraint (the correctness regression), and
2. `from_nl` lowers the pair through the exact GDP disjunction, so a `.nl` MPCC
   solves to the same global optimum as the equivalent Python
   ``Model.complementarity`` build (the capability slice).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import discopt.modeling.core as dm
import numpy as np
import pytest
from discopt._rust import parse_nl_string

# The #658 fixture. It is parsed here only at the raw-row level: its flag (2, on a
# variable with both bounds finite) was written for a swapped reading of AMPL's
# ComplInfo bits, so ``from_nl`` now refuses it as malformed (#2056). It stays
# byte-for-byte because the serialization golden documents pin its digest.
_FIXTURE = Path(__file__).parent / "data" / "mpcc_complementarity.nl"
_NL = _FIXTURE.read_text()

# AMPL output for   min (x-1)^2 + (y-1)^2   s.t.   0 <= y <= Infinity complements x,
# x <= 10, cap: y <= 10   (``data/nl_complementarity/ampl_base.mod``). Global
# optimum 1.0 at (0,1) or (1,0). AMPL omits x's ``>= 0`` (implied by the relation)
# and leaves y's ``<= 10`` as a row, so ``_load_base`` restores both declared
# bounds for a finite big-M; the feasible set is unchanged.
_BASE = Path(__file__).parent / "data" / "nl_complementarity" / "ampl_base.nl"


def _load_base():
    m = dm.from_nl(str(_BASE))
    x0, x1 = m._variables
    x0.lb, x0.ub = 0.0, 10.0
    x1.lb, x1.ub = 0.0, 10.0
    return m


def test_parser_records_pair_and_no_fabricated_constraint():
    """Type-5 row → one complementarity pair, zero ordinary constraints."""
    repr_ = parse_nl_string(_NL)

    assert repr_.n_vars == 2
    # Regression for the fabricated-range bug: the old parser emitted two bogus
    # range constraints (`2 <= x <= 2`) from the flag/index operands.
    assert repr_.n_constraints == 0
    assert repr_.n_complementarities == 1

    body_id, var_index, flag = repr_.complementarity_info(0)
    assert var_index == 1  # complementary variable is y (0-based)
    assert flag == 2  # raw value; AMPL MP ComplInfo INF_LB (finite variable upper bound)
    # body is the linear expression `x` (arena variable node, index 0)
    node = repr_.get_node(body_id)
    assert node["type"] == "variable"
    assert node["index"] == 0


def test_from_nl_lowers_to_gdp_no_native_path():
    """`from_nl` records the pair, lowers it to a disjunction, drops the native
    `.nl` fast path (the reformulated model no longer matches the raw file)."""
    m = dm.from_nl(str(_BASE))

    assert len(m._variables) == 2
    assert len(m._complementarities) == 1
    # GDP lowering added the f>=0, g>=0 constraints (plus the either_or selector),
    # and crucially the source-.nl native path is suppressed for a reformulated
    # model (would otherwise solve a structurally different problem).
    assert not hasattr(m, "_source_nl_path")


def test_from_nl_mpcc_matches_python_build():
    """A `.nl` MPCC solves to the same global optimum as the Python build."""
    m_nl = _load_base()
    res_nl = m_nl.solve()

    m_py = dm.Model("equiv")
    x = m_py.continuous("x", lb=0, ub=10)
    y = m_py.continuous("y", lb=0, ub=10)
    m_py.minimize((x - 1) ** 2 + (y - 1) ** 2)
    m_py.complementarity(x, y)
    res_py = m_py.solve()

    assert res_nl.status == "optimal"
    assert res_py.status == "optimal"
    assert res_nl.objective == pytest.approx(1.0, abs=1e-5)
    assert res_nl.objective == pytest.approx(res_py.objective, abs=1e-5)

    # Complementarity holds at the solution: x * y == 0.
    xv, yv = float(res_nl.x["x0"]), float(res_nl.x["x1"])
    assert xv * yv == pytest.approx(0.0, abs=1e-5)


def _from_text(text: str) -> dm.Model:
    """``from_nl`` on ``text`` written to a temporary ``.nl`` file, after locking
    the #658 regression: one complementarity, no fabricated range row."""
    repr_ = parse_nl_string(text)
    assert repr_.n_complementarities == 1
    assert repr_.n_constraints == 0
    fd, path = tempfile.mkstemp(suffix=".nl")
    os.write(fd, text.encode())
    os.close(fd)
    try:
        return dm.from_nl(path)
    finally:
        os.unlink(path)


def _legacy_variant(*, flag: int, x_bounds: str, y_bounds: str, x_target: str, y_target: str):
    """The #658 fixture -- ``min (x - a)^2 + (y - b)^2`` with ``x ⊥ y`` -- with the
    flag, both variables' ``b``-segment lines and the targets ``a, b`` replaced.
    Only the replaced fields differ from the committed file."""
    text = _NL.replace("5 2 2", f"5 {flag} 2")
    head, rest = text.split("\nb\n")
    bounds, tail = rest.split("\nk1\n")
    assert bounds == "0 0 10\n0 0 10"
    head = head.replace("v0\nn1\n", f"v0\nn{x_target}\n", 1)
    head = head.replace("v1\nn1\n", f"v1\nn{y_target}\n", 1)
    return head + "\nb\n" + f"{x_bounds}\n{y_bounds}" + "\nk1\n" + tail


def test_doubly_bounded_row_lowers_as_a_box():
    """``x ⊥ y in [0, 10]`` (AMPL flag 3) with ``x in [-10, 10]`` free in sign and
    ``min (x + 1)^2 + (y - 10)^2``. The box admits ``y = 10, x <= 0``, so the
    optimum is 0 at ``(-1, 10)``; an NCP reading (``x >= 0``) gives 1. Was
    refused as an unsupported bound configuration before #2058."""
    from discopt.mpec import ComplementarityRole

    m = _from_text(
        _legacy_variant(
            flag=3, x_bounds="0 -10 10", y_bounds="0 0 10", x_target="-1", y_target="10"
        )
    )
    rel = m._complementarities[0]
    assert rel.role is ComplementarityRole.BOX_MCP
    assert rel.g_bounds == (0.0, 10.0)
    assert rel.lowering_in(m) == "gdp"
    assert not hasattr(m, "_source_nl_path")
    res = m.solve(time_limit=60.0)
    assert res.status == "optimal", res.status
    assert res.objective == pytest.approx(0.0, abs=1e-6)
    assert res.objective < 0.5, "an NCP reading certifies 1.0"
    assert float(res.x["x0"]) == pytest.approx(-1.0, abs=1e-5)
    assert float(res.x["x1"]) == pytest.approx(10.0, abs=1e-5)


def test_free_variable_row_lowers_as_an_equation():
    """``x ⊥ y`` with ``y`` free (AMPL flag 0) is ``x == 0``. With
    ``min (x - 1)^2 + (y - 1)^2`` the optimum is 1 at ``(0, 1)`` (0 if the row
    were dropped). Was refused before #2058."""
    m = _from_text(
        _legacy_variant(flag=0, x_bounds="0 -10 10", y_bounds="3", x_target="1", y_target="1")
    )
    rel = m._complementarities[0]
    assert rel.g_bounds == (-np.inf, np.inf)
    assert rel.lowering_in(m) == "gdp"
    assert [c.name for c in m._constraints] == ["nl_compl0_eq"]
    res = m.solve(time_limit=60.0)
    assert res.status == "optimal", res.status
    assert res.objective == pytest.approx(1.0, abs=1e-6)
    assert float(res.x["x0"]) == pytest.approx(0.0, abs=1e-6)
    assert float(res.x["x1"]) == pytest.approx(1.0, abs=1e-5)
