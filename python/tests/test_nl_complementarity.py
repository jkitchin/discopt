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
import pytest
from discopt._rust import parse_nl_string

_FIXTURE = Path(__file__).parent / "data" / "mpcc_complementarity.nl"

# The committed fixture, inlined so the parser tests do not depend on file IO.
#   min (x-1)^2 + (y-1)^2   s.t.   0 <= x  ⊥  y >= 0,   x,y in [0,10]
# Global optimum: 1.0 at (0,1) or (1,0).
_NL = _FIXTURE.read_text()


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
    assert flag == 2  # AMPL MP ComplInfo: body >= 0
    # body is the linear expression `x` (arena variable node, index 0)
    node = repr_.get_node(body_id)
    assert node["type"] == "variable"
    assert node["index"] == 0


def test_from_nl_lowers_to_gdp_no_native_path():
    """`from_nl` records the pair, lowers it to a disjunction, drops the native
    `.nl` fast path (the reformulated model no longer matches the raw file)."""
    m = dm.from_nl(str(_FIXTURE))

    assert len(m._variables) == 2
    assert len(m._complementarities) == 1
    # GDP lowering added the f>=0, g>=0 constraints (plus the either_or selector),
    # and crucially the source-.nl native path is suppressed for a reformulated
    # model (would otherwise solve a structurally different problem).
    assert not hasattr(m, "_source_nl_path")


def test_from_nl_mpcc_matches_python_build():
    """A `.nl` MPCC solves to the same global optimum as the Python build."""
    m_nl = dm.from_nl(str(_FIXTURE))
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


def test_unsupported_complementarity_config_refuses_loudly():
    """A free/double-bounded body (flag=3) that cannot map onto the nonnegative
    primitive is refused loudly rather than silently mis-modeled (#658 scope)."""
    # Same fixture but with the complementarity flag set to 3 (body free): the
    # `.nl → GDP` slice cannot express this soundly, so `from_nl` must raise.
    nl_free = _NL.replace("5 2 2", "5 3 2")
    fd, path = tempfile.mkstemp(suffix=".nl")
    os.write(fd, nl_free.encode())
    os.close(fd)
    try:
        with pytest.raises(ValueError, match="unsupported bound configuration"):
            dm.from_nl(path)
    finally:
        os.unlink(path)
